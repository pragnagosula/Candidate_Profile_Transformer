"""Validates URLs on all Link objects attached to a candidate."""

from __future__ import annotations

from urllib.parse import urlparse

from app.models.candidate import NormalizedCandidate, ValidationIssue
from app.validators.base import BaseValidator


def _is_valid_url(url: str) -> bool:
    """Return True when *url* has a scheme and a non-empty netloc."""
    try:
        parsed = urlparse(url)
        return bool(parsed.scheme in ("http", "https") and parsed.netloc)
    except Exception:  # noqa: BLE001
        return False


class URLValidator(BaseValidator):
    """Warn when any link URL is malformed or missing a scheme."""

    def validate(self, candidate: NormalizedCandidate) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        for i, link in enumerate(candidate.links):
            if not _is_valid_url(link.url):
                issues.append(
                    self._warning(
                        f"links[{i}].url",
                        f"'{link.url}' is not a valid URL (label: {link.label!r})",
                        value=link.url,
                    )
                )
        return issues
