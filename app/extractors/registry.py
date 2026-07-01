"""Extractor registry — maps DataSource → BaseExtractor subclass.

Usage:
    @extractor_registry.register(DataSource.CSV)
    class CSVExtractor(BaseExtractor): ...

    candidate = extractor_registry.extract(DataSource.CSV, raw_record)
"""

from __future__ import annotations

from typing import Type

from app.extractors.base import BaseExtractor
from app.models.candidate import DataSource, ExtractedCandidate, RawCandidateData
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


class ExtractorRegistry:
    """Singleton registry that maps DataSource → BaseExtractor subclass."""

    def __init__(self) -> None:
        self._registry: dict[DataSource, Type[BaseExtractor]] = {}

    def register(self, source: DataSource):
        """Class decorator that registers an extractor for a given source."""
        def decorator(cls: Type[BaseExtractor]) -> Type[BaseExtractor]:
            cls.source = source
            self._registry[source] = cls
            logger.debug("Registered extractor %s for source '%s'", cls.__name__, source)
            return cls
        return decorator

    def get(self, source: DataSource) -> Type[BaseExtractor]:
        """Return the extractor class for a source.

        Raises:
            KeyError: If no extractor is registered for the source.
        """
        if source not in self._registry:
            available = [s.value for s in self._registry]
            raise KeyError(
                f"No extractor registered for source '{source}'. Available: {available}"
            )
        return self._registry[source]

    def extract(self, raw: RawCandidateData) -> ExtractedCandidate:
        """Instantiate the correct extractor and run extraction.

        Returns a blank ExtractedCandidate if no extractor is registered.
        """
        try:
            extractor_cls = self.get(raw.source)
        except KeyError as exc:
            logger.warning(str(exc))
            return ExtractedCandidate(source=raw.source, source_file=raw.source_file)

        logger.debug("Extracting %s record with %s", raw.source, extractor_cls.__name__)
        return extractor_cls().extract(raw)

    @property
    def registered_sources(self) -> list[DataSource]:
        return list(self._registry.keys())


extractor_registry = ExtractorRegistry()
