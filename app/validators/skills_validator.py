"""Validates the skills list for case-insensitive duplicates.

The normalization engine should have eliminated duplicates already.
This validator acts as a post-normalization safety net and surfaces any
that slipped through as warnings (not errors — a duplicate skill is
inconvenient, not a data integrity failure).
"""

from __future__ import annotations

from app.models.candidate import NormalizedCandidate, ValidationIssue
from app.validators.base import BaseValidator


class SkillsValidator(BaseValidator):
    """Warn on duplicate skill entries after normalization."""

    def validate(self, candidate: NormalizedCandidate) -> list[ValidationIssue]:
        if not candidate.skills:
            return []

        seen: dict[str, str] = {}  # lowercase → original
        issues: list[ValidationIssue] = []

        for skill in candidate.skills:
            key = skill.lower().strip()
            if key in seen:
                issues.append(
                    self._warning(
                        "skills",
                        f"Duplicate skill detected: '{skill}' (already present as '{seen[key]}')",
                        value=skill,
                    )
                )
            else:
                seen[key] = skill

        return issues
