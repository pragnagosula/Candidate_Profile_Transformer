"""Validates email address format against a strict RFC 5322 pattern."""

from __future__ import annotations

import re

from app.models.candidate import NormalizedCandidate, ValidationIssue
from app.validators.base import BaseValidator

# Deliberately strict: rejects edge cases that are technically valid but
# almost never seen in real candidate data (e.g. quoted local parts).
_EMAIL_RE = re.compile(
    r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
)


class EmailValidator(BaseValidator):
    """Emit an error when the email field fails basic RFC 5322 validation."""

    def validate(self, candidate: NormalizedCandidate) -> list[ValidationIssue]:
        if not candidate.email:
            return []   # absence is RequiredFieldsValidator's concern

        if not _EMAIL_RE.match(candidate.email):
            return [
                self._error(
                    "email",
                    f"'{candidate.email}' is not a valid email address",
                    value=candidate.email,
                )
            ]
        return []
