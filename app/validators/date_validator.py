"""Validates that all date strings in experience / education are parseable.

Accepted formats after normalization:
    YYYY-MM   2022-01
    YYYY      2022

Any other value that survived normalization is reported as a warning
(not an error — dates are optional and their absence is not fatal).
"""

from __future__ import annotations

import re

from app.models.candidate import NormalizedCandidate, ValidationIssue
from app.validators.base import BaseValidator

_CANONICAL_DATE_RE = re.compile(r"^\d{4}(?:-(?:0[1-9]|1[0-2]))?$")


def _is_valid_date(value: str | None) -> bool:
    if not value:
        return True  # None / empty means "present" or unknown — that's fine
    return bool(_CANONICAL_DATE_RE.match(value))


class DateValidator(BaseValidator):
    """Warn when a date string in experience or education is not canonical."""

    def validate(self, candidate: NormalizedCandidate) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        for i, exp in enumerate(candidate.experience):
            if exp.duration:
                for date_field in ("start", "end"):
                    v = getattr(exp.duration, date_field)
                    if v and not _is_valid_date(v):
                        issues.append(
                            self._warning(
                                f"experience[{i}].duration.{date_field}",
                                f"Non-canonical date '{v}' in experience record",
                                value=v,
                            )
                        )

        for i, edu in enumerate(candidate.education):
            if edu.duration:
                for date_field in ("start", "end"):
                    v = getattr(edu.duration, date_field)
                    if v and not _is_valid_date(v):
                        issues.append(
                            self._warning(
                                f"education[{i}].duration.{date_field}",
                                f"Non-canonical date '{v}' in education record",
                                value=v,
                            )
                        )

        return issues
