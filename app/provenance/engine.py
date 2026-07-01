"""Provenance Engine.

Builds the list[ProvenanceEntry] that is attached to a MergedCandidate.
Each entry records the complete lineage for one field from one source:
  - original_value  — the raw string/value before normalization
  - normalized_value — the canonical value after normalization
  - extraction_method — how the value was obtained (DIRECT / REGEX / INFERRED)
  - confidence — the reliability weight of the contributing source
  - notes — human-readable summary of what changed (if anything)

Input contract
──────────────
The engine is called with:
  merged       — the MergedCandidate whose provenance we are building
  source_diffs — list of (NormalizedCandidate, NormalizationDiff) pairs,
                 one pair per source record, produced by NormalizationEngine
  confidence_report — optional; not used for per-entry confidence (we use
                       source reliability weights for that) but available
                       for future extension

Only fields that are non-empty in the NormalizedCandidate are tracked.
Fields absent from a source are silently skipped — the merged record may
still have a value from another source.
"""

from __future__ import annotations

from app.config.loader import get_config
from app.config.models import SourceReliabilityConfig
from app.models.candidate import (
    ConfidenceReport,
    DataSource,
    ExtractionMethod,
    MergedCandidate,
    NormalizedCandidate,
    ProvenanceEntry,
)
from app.normalizers.engine import NormalizationDiff
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

# Fields tracked in provenance (mirrors the MergedCandidate structure).
_TRACKED_FIELDS = (
    "name", "email", "phone", "location", "summary",
    "skills", "experience", "education", "links",
)

# ExtractionMethod assigned by source type for each field.
_PDF_REGEX_FIELDS = {"email", "phone", "skills", "links", "location", "summary",
                     "experience", "education"}


class ProvenanceEngine:
    """Build provenance entries for a merged candidate record.

    Stateless between calls; safe to instantiate once and reuse.
    """

    def __init__(self, reliability_config: SourceReliabilityConfig | None = None) -> None:
        self._reliability: SourceReliabilityConfig = (
            reliability_config
            if reliability_config is not None
            else get_config().source_reliability
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        merged: MergedCandidate,  # noqa: ARG002 — reserved for future use
        source_diffs: list[tuple[NormalizedCandidate, NormalizationDiff]],
        confidence_report: ConfidenceReport | None = None,  # noqa: ARG002
    ) -> list[ProvenanceEntry]:
        """Build one ProvenanceEntry per (field, source) pair.

        Args:
            merged:          The merged record (currently used for logging).
            source_diffs:    Paired (NormalizedCandidate, NormalizationDiff)
                             tuples from the normalization pass.
            confidence_report: Optional ConfidenceReport (reserved for future
                             per-field confidence refinement).

        Returns:
            A list of :class:`ProvenanceEntry` objects.  Never raises.
        """
        entries: list[ProvenanceEntry] = []

        for candidate, diff in source_diffs:
            for field in _TRACKED_FIELDS:
                value = getattr(candidate, field, None)
                if not _non_empty(value):
                    continue

                original, normalized = _original_normalized(field, value, diff)
                method = _extraction_method(field, candidate.source)
                confidence = self._reliability.get(candidate.source)
                notes = _change_notes(field, diff)

                entries.append(
                    ProvenanceEntry(
                        field_name=field,
                        source=candidate.source,
                        original_value=original,
                        normalized_value=normalized,
                        extraction_method=method,
                        confidence=confidence,
                        notes=notes,
                    )
                )

        logger.info(
            "ProvenanceEngine: %d entries from %d source(s)",
            len(entries),
            len(source_diffs),
        )
        return entries


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def _non_empty(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict)):
        return bool(value)
    return True


def _original_normalized(
    field: str, current_value: object, diff: NormalizationDiff
) -> tuple[object, object]:
    """Return (original_value, normalized_value) for a field.

    If the field is recorded in the diff it was changed during normalization;
    we return the pre-normalization original and the post-normalization value.
    Otherwise the value was not changed and both sides are the same.
    """
    if field in diff.changes:
        original, normalized = diff.changes[field]
        return original, normalized
    return current_value, current_value


def _extraction_method(field: str, source: DataSource) -> ExtractionMethod:
    """Infer how a field was extracted based on source type and field name.

    PDF extractors use regex patterns for everything except the candidate's
    name, which is heuristically detected (INFERRED).  Structured sources
    (CSV, JSON, ATS, LinkedIn, GitHub) provide data directly (DIRECT).
    """
    if source == DataSource.RESUME_PDF:
        if field == "name":
            return ExtractionMethod.INFERRED
        return ExtractionMethod.REGEX
    return ExtractionMethod.DIRECT


def _change_notes(field: str, diff: NormalizationDiff) -> str | None:
    """Return a human-readable note when a field was changed by normalization."""
    if field in diff.changes:
        original, _ = diff.changes[field]
        return f"normalized from {original!r}"
    return None
