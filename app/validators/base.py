"""Abstract base class for all validators."""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.models.candidate import NormalizedCandidate, ValidationIssue


class BaseValidator(ABC):
    """Single-responsibility validator that inspects one aspect of a candidate.

    Each subclass must:
    - Never raise an exception.
    - Return an empty list when the candidate passes validation.
    - Return one ValidationIssue per distinct problem found.
    """

    @abstractmethod
    def validate(self, candidate: NormalizedCandidate) -> list[ValidationIssue]:
        """Inspect the candidate and return any issues found.

        Args:
            candidate: A normalised candidate ready for validation.

        Returns:
            List of :class:`~app.models.candidate.ValidationIssue` objects.
            Empty list means the candidate passed this check.
        """

    @staticmethod
    def _error(field: str, message: str, value=None) -> ValidationIssue:
        return ValidationIssue(field=field, severity="error", message=message, value=value)

    @staticmethod
    def _warning(field: str, message: str, value=None) -> ValidationIssue:
        return ValidationIssue(field=field, severity="warning", message=message, value=value)

    @staticmethod
    def _info(field: str, message: str, value=None) -> ValidationIssue:
        return ValidationIssue(field=field, severity="info", message=message, value=value)
