"""PDF resume parser — extracts raw text from PDF files using pdfplumber.

Design note:
    PDF parsing produces *unstructured text* — a wall of characters from
    the resume.  Structured field extraction (name, email, experience, etc.)
    is the responsibility of the Extraction Layer, not this parser.

    This parser's only job is:
        PDF file  →  RawCandidateData(raw_text=<full text>)

Robustness:
- Corrupted / password-protected PDF → error record, no crash
- Empty PDF (0 pages or blank pages) → warning + empty raw_text
- Very large PDFs → all pages concatenated with newline separators
- Fallback to pypdf if pdfplumber fails (rare edge cases)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from app.models.candidate import DataSource, RawCandidateData
from app.parsers.base import BaseParser
from app.parsers.registry import parser_registry
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


def _extract_with_pdfplumber(path: Path) -> tuple[Optional[str], Optional[str]]:
    """Extract text using pdfplumber.

    Returns:
        (text, error_message).  text is None on failure.
    """
    try:
        import pdfplumber  # deferred import — optional dependency
    except ImportError:
        return None, "pdfplumber is not installed; run: pip install pdfplumber"

    try:
        with pdfplumber.open(str(path)) as pdf:
            pages = []
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                pages.append(page_text)
            full_text = "\n".join(pages).strip()
        return full_text or None, None
    except Exception as exc:  # noqa: BLE001 — intentional broad catch
        return None, f"pdfplumber failed: {exc}"


def _extract_with_pypdf(path: Path) -> tuple[Optional[str], Optional[str]]:
    """Fallback extractor using pypdf.

    Returns:
        (text, error_message).  text is None on failure.
    """
    try:
        from pypdf import PdfReader  # deferred import — optional dependency
    except ImportError:
        return None, "pypdf is not installed; run: pip install pypdf"

    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            return None, "PDF is password-protected"
        pages = [page.extract_text() or "" for page in reader.pages]
        full_text = "\n".join(pages).strip()
        return full_text or None, None
    except Exception as exc:  # noqa: BLE001
        return None, f"pypdf failed: {exc}"


@parser_registry.register(DataSource.RESUME_PDF)
class ResumePDFParser(BaseParser):
    """Parse a PDF resume into a single RawCandidateData with raw_text set."""

    def parse(self, file_path: str | Path) -> list[RawCandidateData]:
        """Parse a PDF file and return one record containing its full text.

        Args:
            file_path: Path to the .pdf file.

        Returns:
            A single-element list with raw_text populated, or an error record.
        """
        path = Path(file_path)

        if not path.exists():
            logger.error("PDF file not found: %s", path)
            return [self._error_record(path, f"File not found: {path}")]

        if path.stat().st_size == 0:
            logger.warning("PDF file is empty (0 bytes): %s", path)
            return [self._error_record(path, "PDF file is empty")]

        # Primary extractor
        text, primary_error = _extract_with_pdfplumber(path)

        # Fallback if primary failed
        if text is None:
            logger.warning(
                "pdfplumber extraction failed for %s (%s); trying pypdf fallback",
                path.name,
                primary_error,
            )
            text, fallback_error = _extract_with_pypdf(path)
            if text is None:
                combined_error = f"{primary_error} | {fallback_error}"
                logger.error("All PDF extractors failed for %s: %s", path.name, combined_error)
                return [self._error_record(path, combined_error)]

        if not text:
            logger.warning("PDF parsed but no text extracted from %s", path.name)

        record = RawCandidateData(
            source=DataSource.RESUME_PDF,
            source_file=str(path),
            raw_fields={"raw_text": text or ""},
        )

        logger.info(
            "ResumePDFParser extracted %d characters from %s",
            len(text or ""),
            path.name,
        )
        return [record]
