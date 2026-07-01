"""Plain-text resume parser — reads .txt files into RawCandidateData.

Design note:
    Like ResumePDFParser, this parser only performs I/O.  It reads the file
    and stores the complete text in ``raw_fields["raw_text"]``.  Structured
    field extraction (name, email, experience, etc.) is the responsibility of
    TXTExtractor in the Extraction Layer.

        .txt file  →  RawCandidateData(raw_text=<full text>)

Robustness:
- Missing / empty file → error record, no crash
- UTF-8 (with or without BOM) → primary encoding
- latin-1 fallback → handles files saved by Windows Notepad
"""

from __future__ import annotations

from pathlib import Path

from app.models.candidate import DataSource, RawCandidateData
from app.parsers.base import BaseParser
from app.parsers.registry import parser_registry
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

# Encodings tried in order; latin-1 never raises a UnicodeDecodeError.
_ENCODINGS = ("utf-8-sig", "utf-8", "latin-1")


def _read_text(path: Path) -> tuple[str | None, str | None]:
    """Read *path* trying multiple encodings.

    Returns:
        ``(text, None)`` on success or ``(None, error_message)`` on failure.
    """
    for encoding in _ENCODINGS:
        try:
            return path.read_text(encoding=encoding), None
        except UnicodeDecodeError:
            continue
        except OSError as exc:
            return None, f"OS error reading file: {exc}"
    return None, f"Could not decode {path.name} with any of {_ENCODINGS}"


@parser_registry.register(DataSource.RESUME_TXT)
class ResumeTXTParser(BaseParser):
    """Parse a plain-text resume into a single RawCandidateData with raw_text set."""

    def parse(self, file_path: str | Path) -> list[RawCandidateData]:
        """Read a .txt file and return one record containing its full text.

        Args:
            file_path: Path to the .txt file.

        Returns:
            A single-element list with raw_text populated, or an error record.
        """
        path = Path(file_path)

        if not path.exists():
            logger.error("TXT file not found: %s", path)
            return [self._error_record(path, f"File not found: {path}")]

        if path.stat().st_size == 0:
            logger.warning("TXT file is empty (0 bytes): %s", path)
            return [self._error_record(path, "TXT file is empty")]

        text, error = _read_text(path)
        if text is None:
            logger.error("Failed to read %s: %s", path.name, error)
            return [self._error_record(path, error)]

        text = text.strip()
        if not text:
            logger.warning("TXT file contains only whitespace: %s", path.name)

        record = RawCandidateData(
            source=DataSource.RESUME_TXT,
            source_file=str(path),
            raw_fields={"raw_text": text},
        )

        logger.info(
            "ResumeTXTParser read %d characters from %s",
            len(text),
            path.name,
        )
        return [record]
