"""Completeness validator — categorises missing fields by importance.

Fields are classified into three tiers via :class:`~app.config.models.FieldCategoryConfig`:

Required
    Core candidate attributes (name, email, phone, skills, experience, education).
    Missing → ``warning`` severity; reduces completeness and confidence.

Recommended
    Fields that improve profile quality (links: LinkedIn, GitHub, Portfolio).
    Missing → ``warning`` severity; small confidence impact.

Optional
    Supplementary information (summary, location).
    Missing → ``info`` severity; no confidence penalty whatsoever.
"""

from __future__ import annotations

from app.config.models import FieldCategoryConfig
from app.models.candidate import NormalizedCandidate, ValidationIssue
from app.validators.base import BaseValidator

_FIELD_LABELS: dict[str, str] = {
    "name":       "Candidate name",
    "email":      "Email address",
    "phone":      "Phone number",
    "skills":     "Skills list",
    "experience": "Work experience",
    "education":  "Education history",
    "links":      "Professional links (LinkedIn/GitHub/Portfolio)",
    "summary":    "Professional summary",
    "location":   "Location",
}


def _label(field_name: str) -> str:
    return _FIELD_LABELS.get(field_name, field_name.replace("_", " ").title())


def _is_empty(value: object) -> bool:
    return (
        value is None
        or (isinstance(value, str) and not value.strip())
        or (isinstance(value, list) and not value)
    )


class CompletenessValidator(BaseValidator):
    """Emit completeness issues classified by field importance.

    Args:
        categories: Field category configuration.  Defaults to the standard
                    required/recommended/optional split when omitted.
    """

    def __init__(self, categories: FieldCategoryConfig | None = None) -> None:
        self._categories = categories or FieldCategoryConfig()

    def validate(self, candidate: NormalizedCandidate) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        for field_name in self._categories.required_fields:
            if _is_empty(getattr(candidate, field_name, None)):
                issues.append(
                    self._warning(field_name, f"{_label(field_name)} is missing")
                )

        for field_name in self._categories.recommended_fields:
            if _is_empty(getattr(candidate, field_name, None)):
                issues.append(
                    self._warning(
                        field_name,
                        f"{_label(field_name)} not provided"
                        " — recommended for a complete profile",
                    )
                )

        for field_name in self._categories.optional_fields:
            if _is_empty(getattr(candidate, field_name, None)):
                issues.append(
                    self._info(field_name, f"{_label(field_name)} not provided (optional)")
                )

        return issues
