"""Validates phone number format.

Uses the phonenumbers library when available for authoritative validation;
falls back to a digit-count heuristic (7–15 digits) so the pipeline never
fails just because the optional library is missing.
"""

from __future__ import annotations

from app.models.candidate import NormalizedCandidate, ValidationIssue
from app.validators.base import BaseValidator
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


def _digit_count(value: str) -> int:
    return sum(c.isdigit() for c in value)


def _is_valid_phone(phone: str) -> bool:
    """Return True if *phone* looks like a plausible phone number."""
    try:
        import phonenumbers
        parsed = phonenumbers.parse(phone, None)
        return phonenumbers.is_valid_number(parsed)
    except Exception:  # noqa: BLE001 — library unavailable or parse failure
        pass

    # Fallback: ITU-T E.164 allows 7–15 digits
    digits = _digit_count(phone)
    return 7 <= digits <= 15


class PhoneValidator(BaseValidator):
    """Emit a warning when the phone field fails plausibility checks."""

    def validate(self, candidate: NormalizedCandidate) -> list[ValidationIssue]:
        if not candidate.phone:
            return []   # absence is completeness validator's concern

        if not _is_valid_phone(candidate.phone):
            return [
                self._warning(
                    "phone",
                    f"'{candidate.phone}' does not appear to be a valid phone number",
                    value=candidate.phone,
                )
            ]
        return []
