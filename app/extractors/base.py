"""Abstract base class for all source extractors."""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.models.candidate import DataSource, ExtractedCandidate, RawCandidateData


class BaseExtractor(ABC):
    """Convert a :class:`~app.models.candidate.RawCandidateData` record into
    a typed :class:`~app.models.candidate.ExtractedCandidate`.

    Each extractor handles exactly one :class:`~app.models.candidate.DataSource`.
    Errors must be captured and surfaced via the returned model — never raised.
    """

    source: DataSource

    @abstractmethod
    def extract(self, raw: RawCandidateData) -> ExtractedCandidate:
        """Extract structured fields from raw candidate data.

        Args:
            raw: Output of the corresponding parser.

        Returns:
            :class:`~app.models.candidate.ExtractedCandidate` with as many
            fields populated as the source allows.
        """

    def _empty(self, raw: RawCandidateData) -> ExtractedCandidate:
        """Return a blank ExtractedCandidate preserving source metadata."""
        return ExtractedCandidate(
            source=raw.source,
            source_file=raw.source_file,
        )
