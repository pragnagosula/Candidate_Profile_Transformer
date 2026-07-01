"""Validates that mandatory fields are present and non-empty."""

from __future__ import annotations

from app.models.candidate import NormalizedCandidate, ValidationIssue
from app.validators.base import BaseValidator

# Fields that must be present for a candidate record to be usable.
_REQUIRED_FIELDS: list[tuple[str, str]] = [
    ("name",  "Candidate name is required"),
    ("email", "Candidate email is required"),
]


class RequiredFieldsValidator(BaseValidator):
    """Raise an error for each mandatory field that is absent or blank."""

    def validate(self, candidate: NormalizedCandidate) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        for field_name, message in _REQUIRED_FIELDS:
            value = getattr(candidate, field_name, None)
            if not value or (isinstance(value, str) and not value.strip()):
                issues.append(self._error(field_name, message, value=value))
        return issues
