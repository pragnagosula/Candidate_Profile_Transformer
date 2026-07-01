"""URL normalizer — ensures https scheme, lowercases host, strips trailing slash."""

from __future__ import annotations

from urllib.parse import urlparse, urlunparse

from app.config.models import NormalizationConfig
from app.normalizers.base import BaseFieldNormalizer
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


class URLNormalizer(BaseFieldNormalizer[str | None]):
    """Normalise a URL string to a consistent canonical form."""

    def normalize(self, value: str | None, config: NormalizationConfig) -> str | None:
        if not value or not isinstance(value, str):
            return value

        stripped = value.strip()
        if not stripped:
            return value

        # Add scheme if missing
        if not stripped.startswith(("http://", "https://")):
            stripped = "https://" + stripped

        try:
            parsed = urlparse(stripped)
            normalised = urlunparse(
                parsed._replace(
                    scheme=parsed.scheme.lower(),
                    netloc=parsed.netloc.lower(),
                    path=parsed.path.rstrip("/") or "/",
                )
            )
            if normalised != value:
                logger.debug("URLNormalizer: '%s' -> '%s'", value, normalised)
            return normalised
        except Exception as exc:  # noqa: BLE001
            logger.debug("URLNormalizer: could not parse '%s': %s", stripped, exc)
            return stripped
