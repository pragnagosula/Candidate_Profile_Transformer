"""Pipeline Orchestrator.

Wires all pipeline layers together in sequence:

  Parse → Extract → Normalize → Validate
  → Entity-Resolve
  → (per group) Merge → Confidence → Provenance → Project → Schema-Validate
  → Write JSON

Entry point::

    pipeline = Pipeline()
    result = pipeline.run(
        inputs=[(DataSource.CSV, "data/candidates.csv"),
                (DataSource.RESUME_PDF, "data/alice.pdf")],
        output_path="output/profiles.json",
    )
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from app.confidence.engine import ConfidenceEngine
from app.config.loader import get_config
from app.config.models import PipelineConfig
from app.extractors import extractor_registry
from app.mergers.candidate_group import CandidateGroup
from app.mergers.entity_resolver import EntityResolver
from app.mergers.merge_engine import MergeEngine
from app.models.candidate import (
    CandidateProfile,
    DataSource,
    NormalizedCandidate,
    ValidationIssue,
    ValidationResult,
)  # noqa: F401 — ValidationIssue used in _combine_validations
from app.normalizers.engine import NormalizationDiff, NormalizationEngine
from app.parsers import parser_registry
from app.projection.engine import ProjectionEngine
from app.provenance.engine import ProvenanceEngine
from app.schema.validator import SchemaValidator
from app.validators.engine import ValidationEngine
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    """Summary of a complete pipeline run."""

    profiles: list[CandidateProfile] = field(default_factory=list)
    total_inputs: int = 0
    total_groups: int = 0
    errors: list[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.utcnow)
    finished_at: datetime | None = None

    @property
    def success(self) -> bool:
        """True when no pipeline-level errors occurred."""
        return len(self.errors) == 0


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class Pipeline:
    """End-to-end candidate data transformation pipeline.

    Stateless between calls — ``run()`` can be called multiple times.
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        cfg = config or get_config()
        self._config = cfg

        self._norm_engine = NormalizationEngine(cfg.normalization)
        self._val_engine = ValidationEngine(cfg.confidence.field_categories)
        self._resolver = EntityResolver(cfg.entity_resolution)
        self._merge_engine = MergeEngine(cfg.merge_rules)
        self._confidence_engine = ConfidenceEngine(cfg.source_reliability)
        self._provenance_engine = ProvenanceEngine(cfg.source_reliability)
        self._projection_engine = ProjectionEngine(cfg.projection)
        self._schema_validator = SchemaValidator(cfg.projection)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        inputs: list[tuple[DataSource, str | Path]],
        output_path: str | Path | None = None,
    ) -> PipelineResult:
        """Run the full pipeline and return a :class:`PipelineResult`.

        Args:
            inputs: List of ``(DataSource, file_path)`` pairs.  A single
                source file (e.g. a CSV with many rows) may produce multiple
                candidate records.
            output_path: Destination for the output JSON file.  When *None*
                the value from ``config.output_dir / config.output_filename``
                is used.

        Returns:
            A :class:`PipelineResult`.  Never raises — errors are recorded in
            ``result.errors``.
        """
        result = PipelineResult(started_at=datetime.utcnow())

        try:
            normalized, diffs, validations = self._ingest(inputs)
            result.total_inputs = len(normalized)

            groups = self._resolver.resolve(normalized)
            result.total_groups = len(groups)

            for group in groups:
                try:
                    profile = self._process_group(group, diffs, validations)
                    result.profiles.append(profile)
                except Exception as exc:  # noqa: BLE001
                    msg = f"Group processing failed for '{group.primary_name}': {exc}"
                    logger.error(msg)
                    result.errors.append(msg)

        except Exception as exc:  # noqa: BLE001
            msg = f"Pipeline ingestion failed: {exc}"
            logger.error(msg)
            result.errors.append(msg)

        try:
            dest = self._resolve_output_path(output_path)
            self._write_output(result.profiles, dest)
        except Exception as exc:  # noqa: BLE001
            msg = f"Failed to write output: {exc}"
            logger.error(msg)
            result.errors.append(msg)

        result.finished_at = datetime.utcnow()
        logger.info(
            "Pipeline complete: %d input(s) -> %d group(s) -> %d profile(s), %d error(s)",
            result.total_inputs,
            result.total_groups,
            len(result.profiles),
            len(result.errors),
        )
        return result

    # ------------------------------------------------------------------
    # Ingestion (parse → extract → normalize → validate)
    # ------------------------------------------------------------------

    def _ingest(
        self,
        inputs: list[tuple[DataSource, str | Path]],
    ) -> tuple[
        list[NormalizedCandidate],
        dict[int, NormalizationDiff],
        dict[int, ValidationResult],
    ]:
        """Process each input file through parse → extract → normalize → validate.

        Returns:
            Tuple of (candidates, diffs keyed by id(candidate),
            validation_results keyed by id(candidate)).
        """
        normalized: list[NormalizedCandidate] = []
        diffs: dict[int, NormalizationDiff] = {}
        validations: dict[int, ValidationResult] = {}

        for source, path in inputs:
            try:
                raw_records = parser_registry.parse(source, path)
            except Exception as exc:  # noqa: BLE001
                logger.error("Parse failed for %s/%s: %s", source, path, exc)
                continue

            for raw in raw_records:
                if raw.parse_errors:
                    logger.warning(
                        "Parse errors in %s: %s", path, raw.parse_errors
                    )

                try:
                    extracted = extractor_registry.extract(raw)
                    norm, diff = self._norm_engine.normalize(extracted)
                    validation = self._val_engine.validate(norm)

                    normalized.append(norm)
                    diffs[id(norm)] = diff
                    validations[id(norm)] = validation

                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "Ingest failed for record from %s/%s: %s",
                        source, path, exc,
                    )

        return normalized, diffs, validations

    # ------------------------------------------------------------------
    # Per-group processing
    # ------------------------------------------------------------------

    def _process_group(
        self,
        group: CandidateGroup,
        diffs: dict[int, NormalizationDiff],
        validations: dict[int, ValidationResult],
    ) -> CandidateProfile:
        """Merge one :class:`CandidateGroup` into a final :class:`CandidateProfile`."""
        candidates = group.candidates

        source_diffs = [
            (c, diffs.get(id(c), NormalizationDiff())) for c in candidates
        ]

        group_validation = self._combine_validations(
            [validations[id(c)] for c in candidates if id(c) in validations]
        )

        merged = self._merge_engine.merge(group)
        confidence = self._confidence_engine.score(merged, group_validation)
        provenance = self._provenance_engine.build(merged, source_diffs, confidence)

        profile = self._projection_engine.project(
            merged,
            confidence=confidence,
            provenance=provenance,
            validation=group_validation,
        )

        schema_result = self._schema_validator.validate(profile)
        if not schema_result.is_valid:
            logger.warning(
                "Schema validation failed for '%s': %d error(s)",
                group.primary_name,
                len(schema_result.errors),
            )

        return profile

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _combine_validations(
        results: list[ValidationResult],
    ) -> ValidationResult | None:
        """Merge per-source :class:`ValidationResult` objects into one.

        Deduplicates on (field, severity, message) so the same completeness
        warning (e.g. "Skills list is missing") is not repeated once per
        source record when multiple sources contribute to the same group.
        """
        if not results:
            return None
        seen: set[tuple[str, str, str]] = set()
        all_issues: list[ValidationIssue] = []
        for r in results:
            for issue in r.issues:
                key = (issue.field, issue.severity, issue.message)
                if key not in seen:
                    seen.add(key)
                    all_issues.append(issue)
        has_errors = any(i.severity == "error" for i in all_issues)
        return ValidationResult(is_valid=not has_errors, issues=all_issues)

    def _resolve_output_path(self, output_path: str | Path | None) -> Path:
        if output_path is not None:
            return Path(output_path)
        return Path(self._config.output_dir) / self._config.output_filename

    def _write_output(self, profiles: list[CandidateProfile], dest: Path) -> None:
        """Serialise *profiles* to a JSON array and write to *dest*."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        payload = [p.model_dump(mode="json") for p in profiles]
        with dest.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
        logger.info("Wrote %d profile(s) to %s", len(profiles), dest)
