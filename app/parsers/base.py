"""Abstract base class that every parser must implement.

The contract is deliberately minimal: accept a file path, return a list
of RawCandidateData records.  Error handling is the parser's responsibility
— parsers must never raise; they capture errors into parse_errors instead.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from app.models.candidate import DataSource, RawCandidateData


class BaseParser(ABC):
    """Common interface for all data source parsers.

    Subclasses implement :meth:`parse` and register themselves with the
    :class:`~app.parsers.registry.ParserRegistry` via the
    ``@parser_registry.register`` decorator.
    """

    #: Declare which source this parser handles — used by the registry.
    source: DataSource

    @abstractmethod
    def parse(self, file_path: str | Path) -> list[RawCandidateData]:
        """Parse the given file and return one record per candidate.

        Args:
            file_path: Path to the source file.

        Returns:
            List of :class:`~app.models.candidate.RawCandidateData` records.
            Returns an empty list (not raises) on unrecoverable failure.
        """

    def _error_record(self, file_path: str | Path, reason: str) -> RawCandidateData:
        """Build a RawCandidateData that records a parse failure."""
        return RawCandidateData(
            source=self.source,
            source_file=str(file_path),
            parse_errors=[reason],
        )
