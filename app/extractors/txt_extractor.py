"""TXT resume extractor — delegates all parsing to TextResumeParser.

Responsibilities (this file only):
  1. Accept a :class:`~app.models.candidate.RawCandidateData` record whose
     raw text is stored in ``raw_fields["raw_text"]``.
  2. Pass that text directly to :class:`~app.extractors.text_resume_parser.TextResumeParser`.
  3. Return the resulting :class:`~app.models.candidate.ExtractedCandidate`.

No parsing logic lives here.  The public API is identical to
:class:`~app.extractors.pdf_extractor.PDFExtractor`.
"""

from __future__ import annotations

from app.extractors.base import BaseExtractor
from app.extractors.registry import extractor_registry
from app.extractors.text_resume_parser import TextResumeParser
from app.models.candidate import DataSource, ExtractedCandidate, RawCandidateData
from app.utils.logging_config import get_logger

logger = get_logger(__name__)
print("TXTExtractor initialized. Delegating parsing to TextResumeParser.")

@extractor_registry.register(DataSource.RESUME_TXT)
class TXTExtractor(BaseExtractor):
    """Extract structured candidate fields from a plain-text resume.

    Reads raw text from ``raw.raw_fields["raw_text"]`` and delegates all
    field extraction to :class:`~app.extractors.text_resume_parser.TextResumeParser`,
    the single source of truth for resume parsing logic.
    """

    def extract(self, raw: RawCandidateData) -> ExtractedCandidate:
        """Parse a plain-text record into a structured :class:`ExtractedCandidate`.

        Args:
            raw: Record with resume text in ``raw_fields["raw_text"]``.

        Returns:
            Populated :class:`~app.models.candidate.ExtractedCandidate`.
        """
        text = raw.raw_fields.get("raw_text", "")

        if not text:
            logger.warning("TXT record has no text to extract from: %s", raw.source_file)
            return self._empty(raw)

        candidate = TextResumeParser().parse(
            text,
            source=DataSource.RESUME_TXT,
            source_file=raw.source_file,
            # No PDF annotation links for plain-text files.
        )

        logger.debug(
            "TXTExtractor: name=%r email=%r skills=%d experience=%d",
            candidate.name,
            candidate.email,
            len(candidate.skills),
            len(candidate.experience),
        )
        return candidate
