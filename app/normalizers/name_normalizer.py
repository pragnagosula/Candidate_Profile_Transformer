"""Name normalizer — title case, unicode normalization, whitespace cleanup.

Handles:
    ALICE JOHNSON       →  Alice Johnson
    alice johnson       →  Alice Johnson
    Ãlicé Jøhnson       →  preserved (unicode kept, just NFKC normalised)
    "  John   Smith  "  →  "John Smith"
    John A. Smith       →  John A. Smith  (initials preserved)
"""

from __future__ import annotations

import re
import unicodedata

from app.config.models import NormalizationConfig
from app.normalizers.base import BaseFieldNormalizer
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

# Particles that should remain lowercase in a name (van, de, etc.)
_LOWERCASE_PARTICLES = frozenset(
    ["van", "de", "der", "den", "von", "del", "della", "di", "da", "la", "le", "al"]
)


def _title_case_name(name: str) -> str:
    """Apply title casing that respects particles and initials."""
    words = name.split()
    result = []
    for i, word in enumerate(words):
        lower = word.lower()
        # Always capitalise the first word regardless of particle rules
        if i == 0 or lower not in _LOWERCASE_PARTICLES:
            # Preserve single-letter initials (A. → A.)
            if re.match(r"^[a-zA-Z]\.$", word):
                result.append(word.upper()[0] + ".")
            else:
                result.append(word.capitalize())
        else:
            result.append(lower)
    return " ".join(result)


class NameNormalizer(BaseFieldNormalizer[str | None]):
    """Normalise a candidate's name string."""

    def normalize(self, value: str | None, config: NormalizationConfig) -> str | None:
        if not value or not isinstance(value, str):
            return value

        # NFKC unicode normalisation (decomposes ligatures, normalises combining chars)
        normalised = unicodedata.normalize("NFKC", value)

        # Collapse internal whitespace
        collapsed = re.sub(r"\s+", " ", normalised).strip()

        if not collapsed:
            return value

        titled = _title_case_name(collapsed)
        if titled != value:
            logger.debug("NameNormalizer: '%s' -> '%s'", value, titled)
        return titled
