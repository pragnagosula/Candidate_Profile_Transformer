"""Parser registry — maps DataSource values to parser classes.

Parsers self-register at module import time using the decorator:

    @parser_registry.register(DataSource.CSV)
    class CSVParser(BaseParser): ...

Calling code never references concrete parser classes:

    records = parser_registry.parse(DataSource.CSV, "input/candidates.csv")
"""

from __future__ import annotations

from pathlib import Path
from typing import Type

from app.models.candidate import DataSource, RawCandidateData
from app.parsers.base import BaseParser
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


class ParserRegistry:
    """Singleton registry that maps DataSource → BaseParser subclass."""

    def __init__(self) -> None:
        self._registry: dict[DataSource, Type[BaseParser]] = {}

    def register(self, source: DataSource):
        """Class decorator that registers a parser for a given source.

        Args:
            source: The :class:`~app.models.candidate.DataSource` this
                parser handles.

        Returns:
            The original class, unmodified (decorator pattern).
        """
        def decorator(cls: Type[BaseParser]) -> Type[BaseParser]:
            if source in self._registry:
                logger.warning(
                    "Parser for source '%s' is being overridden by %s",
                    source,
                    cls.__name__,
                )
            cls.source = source
            self._registry[source] = cls
            logger.debug("Registered parser %s for source '%s'", cls.__name__, source)
            return cls

        return decorator

    def get(self, source: DataSource) -> Type[BaseParser]:
        """Return the parser class for a given source.

        Args:
            source: The data source to look up.

        Returns:
            A :class:`BaseParser` subclass.

        Raises:
            KeyError: If no parser is registered for the given source.
        """
        if source not in self._registry:
            available = [s.value for s in self._registry]
            raise KeyError(
                f"No parser registered for source '{source}'. "
                f"Available: {available}"
            )
        return self._registry[source]

    def parse(self, source: DataSource, file_path: str | Path) -> list[RawCandidateData]:
        """Instantiate the correct parser and parse the given file.

        Args:
            source: Which parser to use.
            file_path: Path to the source file.

        Returns:
            List of raw candidate records.  Returns a single error record
            if no parser is registered for the source.
        """
        try:
            parser_cls = self.get(source)
        except KeyError as exc:
            logger.error(str(exc))
            return [
                RawCandidateData(
                    source=source,
                    source_file=str(file_path),
                    parse_errors=[str(exc)],
                )
            ]

        logger.info("Parsing %s with %s", file_path, parser_cls.__name__)
        parser = parser_cls()
        return parser.parse(file_path)

    @property
    def registered_sources(self) -> list[DataSource]:
        return list(self._registry.keys())


# Module-level singleton — import this everywhere
parser_registry = ParserRegistry()
