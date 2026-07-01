"""Phone normalizer — uses the `phonenumbers` library for reliable parsing.

Supports:
    +91 98765-43210    →  +919876543210
    (800) 555-0199     →  +18005550199
    9876543210         →  +919876543210  (default country from config)
    +1-800-555-0199    →  +18005550199

Falls back gracefully: if the library can't parse a number it returns the
original value unchanged rather than crashing or returning None.
"""

from __future__ import annotations

import re

from app.config.models import NormalizationConfig
from app.normalizers.base import BaseFieldNormalizer
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


def _digit_count(value: str) -> int:
    return sum(c.isdigit() for c in value)


class PhoneNormalizer(BaseFieldNormalizer[str | None]):
    """Normalise a phone number string to E.164 format."""

    def normalize(self, value: str | None, config: NormalizationConfig) -> str | None:
        if not value or not isinstance(value, str):
            return value

        stripped = value.strip()
        if not stripped or _digit_count(stripped) < 7:
            logger.debug("PhoneNormalizer: too few digits in '%s', skipping", stripped)
            return stripped

        try:
            import phonenumbers  # deferred — optional dependency
            from phonenumbers import PhoneNumberFormat, format_number, parse

            default_region = config.phone.default_country_code
            parsed = parse(stripped, default_region)

            fmt_map = {
                "E164": PhoneNumberFormat.E164,
                "NATIONAL": PhoneNumberFormat.NATIONAL,
                "INTERNATIONAL": PhoneNumberFormat.INTERNATIONAL,
            }
            fmt = fmt_map.get(config.phone.output_format, PhoneNumberFormat.E164)
            result = format_number(parsed, fmt)
            logger.debug("PhoneNormalizer: '%s' -> '%s'", stripped, result)
            return result

        except Exception as exc:  # noqa: BLE001
            logger.debug("PhoneNormalizer: could not parse '%s': %s", stripped, exc)
            return stripped
