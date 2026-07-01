"""Email normalizer — lowercase, strip whitespace, basic sanity check."""

from __future__ import annotations

import re

from app.config.models import NormalizationConfig
from app.normalizers.base import BaseFieldNormalizer
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

_EMAIL_RE = re.compile(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$")


class EmailNormalizer(BaseFieldNormalizer[str | None]):
    """Normalise an email address to lowercase with stripped whitespace."""

    def normalize(self, value: str | None, config: NormalizationConfig) -> str | None:
        if not value or not isinstance(value, str):
            return value

        cleaned = value.strip().lower()

        if not _EMAIL_RE.match(cleaned):
            logger.debug("EmailNormalizer: '%s' does not look like a valid email", cleaned)

        return cleaned
