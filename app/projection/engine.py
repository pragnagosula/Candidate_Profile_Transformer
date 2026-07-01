"""Projection Engine.

Applies the projection.yaml rules to a MergedCandidate to produce the final
CandidateProfile that is written to the output JSON file.

What this layer does
────────────────────
1.  Reads the ProjectionConfig (from projection.yaml or injected).
2.  For each field with include=true:
      - reads the value from MergedCandidate under the source field name
      - falls back to pf.default when the value is None / not present
      - renames the key to pf.output_name (e.g. "name" → "full_name")
      - serialises Pydantic sub-models to plain dicts so the output is
        JSON-serialisable without a second pass
3.  Attaches confidence / provenance / validation blocks when their
    include_* flags are True.

Only included fields appear in CandidateProfile.fields — excluded fields are
not present in the output at all (they are not nulled out).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from app.config.loader import get_config
from app.config.models import ProjectionConfig
from app.models.candidate import (
    CandidateProfile,
    ConfidenceReport,
    MergedCandidate,
    ProvenanceEntry,
    ValidationResult,
)
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


class ProjectionEngine:
    """Project a MergedCandidate into a CandidateProfile.

    Stateless between calls; safe to instantiate once and reuse.
    """

    def __init__(self, config: ProjectionConfig | None = None) -> None:
        self._config: ProjectionConfig = (
            config if config is not None else get_config().projection
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def project(
        self,
        merged: MergedCandidate,
        confidence: ConfidenceReport | None = None,
        provenance: list[ProvenanceEntry] | None = None,
        validation: ValidationResult | None = None,
    ) -> CandidateProfile:
        """Project *merged* into a :class:`CandidateProfile`.

        Args:
            merged:     The merged candidate record.
            confidence: Optional ConfidenceReport to attach.
            provenance: Optional list of ProvenanceEntry to attach.
            validation: Optional ValidationResult to attach.

        Returns:
            A :class:`CandidateProfile` with ``fields`` populated according
            to the projection config.  Never raises.
        """
        fields: dict[str, Any] = {}

        for pf in self._config.fields:
            if not pf.include:
                continue
            raw = getattr(merged, pf.source, None)
            value = raw if raw is not None else pf.default
            fields[pf.output_name] = _serialize(value)

        profile = CandidateProfile(
            fields=fields,
            confidence=confidence if self._config.include_confidence else None,
            provenance=(provenance or []) if self._config.include_provenance else [],
            validation=validation if self._config.include_validation else None,
        )

        logger.debug(
            "ProjectionEngine: projected %d field(s); confidence=%s provenance=%s validation=%s",
            len(fields),
            self._config.include_confidence,
            self._config.include_provenance,
            self._config.include_validation,
        )
        return profile


# ---------------------------------------------------------------------------
# Serialization helper
# ---------------------------------------------------------------------------


def _serialize(value: Any) -> Any:
    """Recursively convert Pydantic models to plain dicts for JSON output."""
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    return value
