"""Date normalizer — converts many date formats to a canonical YYYY-MM string.

Handles:
    Jan 2022          →  2022-01
    January 2022      →  2022-01
    01/2022           →  2022-01
    2022-01           →  2022-01  (already canonical)
    2022              →  2022     (year-only, kept as-is)
    01-2022           →  2022-01
    present / current →  None     (caller interprets as ongoing)

Uses python-dateutil for robust parsing with a configurable default day.
"""

from __future__ import annotations

import re

from app.config.models import NormalizationConfig
from app.normalizers.base import BaseFieldNormalizer
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

_YEAR_ONLY_RE = re.compile(r"^\d{4}$")
_PRESENT_RE = re.compile(r"^(?:present|current|now|ongoing)$", re.IGNORECASE)


def _try_parse(value: str, default_day: int) -> str | None:
    """Attempt to parse *value* as a date using dateutil."""
    try:
        from dateutil import parser as du_parser
        from dateutil.parser import ParserError

        # dateutil needs a day to anchor month-only strings like "Jan 2022"
        parsed = du_parser.parse(value, default=__import__("datetime").datetime(2000, 1, default_day))
        return parsed.strftime("%Y-%m")
    except Exception:  # noqa: BLE001
        return None


class DateNormalizer(BaseFieldNormalizer[str | None]):
    """Normalise a date string to YYYY-MM (or YYYY for year-only values)."""

    def normalize(self, value: str | None, config: NormalizationConfig) -> str | None:
        if not value or not isinstance(value, str):
            return value

        stripped = value.strip()
        if not stripped:
            return value

        # "present", "current", etc. → None (means ongoing)
        if _PRESENT_RE.match(stripped):
            return None

        # Pure 4-digit year → keep as-is for now
        if _YEAR_ONLY_RE.match(stripped):
            return stripped

        default_day = config.dates.assume_day
        result = _try_parse(stripped, default_day)
        if result:
            logger.debug("DateNormalizer: '%s' -> '%s'", stripped, result)
            return result

        logger.debug("DateNormalizer: could not parse '%s', returning as-is", stripped)
        return stripped
