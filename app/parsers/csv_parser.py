"""CSV parser — handles structured candidate data in comma-separated format.

Each row in the CSV becomes one RawCandidateData record.  Column names
are preserved as-is in raw_fields; normalisation happens downstream.

Robustness:
- Empty file → returns empty list, no crash
- Encoding fallback: tries UTF-8 then latin-1
- Missing/extra columns → silently tolerated
- Rows with all-empty values → skipped
"""

from __future__ import annotations

import csv
from pathlib import Path

from app.models.candidate import DataSource, RawCandidateData
from app.parsers.base import BaseParser
from app.parsers.registry import parser_registry
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

_ENCODINGS = ("utf-8-sig", "utf-8", "latin-1")


def _read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    """Read a CSV with encoding fallback.

    Returns:
        Tuple of (rows as list-of-dicts, list of parse warning strings).
    """
    errors: list[str] = []
    for encoding in _ENCODINGS:
        try:
            with path.open(newline="", encoding=encoding) as fh:
                reader = csv.DictReader(fh)
                rows = [dict(row) for row in reader]
                logger.debug("Read %d row(s) from %s (encoding=%s)", len(rows), path, encoding)
                return rows, errors
        except UnicodeDecodeError:
            errors.append(f"Encoding '{encoding}' failed for {path.name}")
            continue
        except csv.Error as exc:
            errors.append(f"CSV parse error: {exc}")
            return [], errors

    errors.append(f"All encodings failed for {path.name}")
    return [], errors


@parser_registry.register(DataSource.CSV)
class CSVParser(BaseParser):
    """Parse a CSV file into one RawCandidateData record per non-empty row."""

    def parse(self, file_path: str | Path) -> list[RawCandidateData]:
        """Parse a CSV file.

        Args:
            file_path: Path to the .csv file.

        Returns:
            One :class:`~app.models.candidate.RawCandidateData` per data row.
            Returns an error record if the file cannot be read.
        """
        path = Path(file_path)

        if not path.exists():
            logger.error("CSV file not found: %s", path)
            return [self._error_record(path, f"File not found: {path}")]

        if path.stat().st_size == 0:
            logger.warning("CSV file is empty: %s", path)
            return []

        rows, read_errors = _read_csv(path)

        if not rows and read_errors:
            return [self._error_record(path, "; ".join(read_errors))]

        records: list[RawCandidateData] = []
        for i, row in enumerate(rows, start=2):  # row 1 = header
            # Strip whitespace from all values
            cleaned = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in row.items() if k}

            # Skip rows where every value is empty
            if not any(cleaned.values()):
                logger.debug("Skipping empty row %d in %s", i, path.name)
                continue

            records.append(
                RawCandidateData(
                    source=DataSource.CSV,
                    source_file=str(path),
                    raw_fields=cleaned,
                    parse_errors=read_errors if i == 2 else [],
                )
            )

        logger.info("CSVParser produced %d record(s) from %s", len(records), path.name)
        return records
