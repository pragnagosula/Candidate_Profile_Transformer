"""PDF extractor — thin wrapper around TextResumeParser.

Responsibilities (this file only):
  1. Extract raw text from PDF binary content (via ``raw_fields["raw_text"]``).
  2. Extract hyperlink annotations embedded in the PDF (PDF-specific; not
     accessible from plain text).
  3. Delegate *all* resume parsing to :class:`~app.extractors.text_resume_parser.TextResumeParser`.

All parsing logic lives in :mod:`app.extractors.text_resume_parser`.
"""

from __future__ import annotations

import re
from typing import Optional

from app.extractors.base import BaseExtractor
from app.extractors.registry import extractor_registry

# Re-export every helper that external code currently imports from this module
# so that existing call-sites and tests require no changes.
from app.extractors.text_resume_parser import (  # noqa: F401 — public re-exports
    TextResumeParser,
    _classify_url,
    _detect_sections,
    _extract_education,
    _extract_email,
    _extract_experience,
    _extract_links_from_text,
    _extract_name,
    _extract_phone,
    _extract_skills,
    _merge_links,
    _normalize_link_url,
    _normalize_skill,
    _scan_tech_dictionary,
)
from app.models.candidate import DataSource, ExtractedCandidate, Link, RawCandidateData
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

try:
    import fitz  # PyMuPDF — PDF hyperlink annotation reader
except ImportError:  # pragma: no cover
    fitz = None
    logger.warning(
        "PyMuPDF (fitz) is not installed; PDF hyperlink-annotation extraction "
        "will be skipped and only plain-text URLs will be found."
    )

# ---------------------------------------------------------------------------
# PDF-specific link extraction (not available from plain text)
# ---------------------------------------------------------------------------


def _resolve_pdf_path(raw: RawCandidateData) -> Optional[str]:
    """Best-effort resolution of the on-disk PDF path from a raw record."""
    candidate_path = raw.raw_fields.get("file_path") or raw.raw_fields.get("pdf_path")
    if candidate_path:
        return candidate_path
    if raw.source_file and raw.source_file.lower().endswith(".pdf"):
        return raw.source_file
    return None


def _extract_links_from_pdf_annotations(raw: RawCandidateData) -> list[Link]:
    """Extract hyperlink annotations embedded in the PDF binary.

    Catches links whose *visible text* is just "LinkedIn" or "GitHub" — the
    real destination URL only lives in the ``/Annots`` object, not as plain
    text, so a text-only scan can never find it.
    """
    if fitz is None:
        return []

    pdf_path = _resolve_pdf_path(raw)
    if not pdf_path:
        return []

    found: list[Link] = []
    try:
        with fitz.open(pdf_path) as document:
            for page in document:
                for link in page.get_links():
                    uri = link.get("uri")
                    if not uri:
                        continue
                    if not re.match(r"^https?://", uri, re.IGNORECASE):
                        if uri.lower().startswith("mailto:"):
                            continue
                        uri = _normalize_link_url(uri)
                    found.append(Link(url=uri, label=_classify_url(uri)))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to read PDF hyperlink annotations from %s: %s", pdf_path, exc)
        return []

    return found


# ---------------------------------------------------------------------------
# Extractor class
# ---------------------------------------------------------------------------


@extractor_registry.register(DataSource.RESUME_PDF)
class PDFExtractor(BaseExtractor):
    """Extract structured candidate fields from a PDF resume.

    Reads raw text from ``raw.raw_fields["raw_text"]``, extracts PDF
    hyperlink annotations (if available), then delegates all parsing to
    :class:`~app.extractors.text_resume_parser.TextResumeParser`.
    """

    def extract(self, raw: RawCandidateData) -> ExtractedCandidate:
        """Parse a PDF record into a structured :class:`ExtractedCandidate`.

        Args:
            raw: Record produced by ResumePDFParser.  The full extracted text
                 must be in ``raw.raw_fields["raw_text"]``.

        Returns:
            Populated :class:`~app.models.candidate.ExtractedCandidate`.
        """
        text = raw.raw_fields.get("raw_text", "")

        if not text:
            logger.warning("PDF record has no text to extract from: %s", raw.source_file)
            return self._empty(raw)

        # PDF-only step: pull hyperlink annotations before delegating to the parser.
        annotation_links = _extract_links_from_pdf_annotations(raw)

        candidate = TextResumeParser().parse(
            text,
            source=DataSource.RESUME_PDF,
            source_file=raw.source_file,
            extra_links=annotation_links,
        )

        logger.debug(
            "PDFExtractor: name=%r email=%r skills=%d experience=%d",
            candidate.name,
            candidate.email,
            len(candidate.skills),
            len(candidate.experience),
        )
        return candidate
