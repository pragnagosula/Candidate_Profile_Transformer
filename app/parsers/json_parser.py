"""JSON parser — handles structured candidate data in JSON format.

Supports both:
  - Single candidate object:  { "name": "Alice", ... }
  - Array of candidates:      [ { "name": "Alice" }, { "name": "Bob" } ]

Robustness:
- Empty file → returns empty list
- Malformed JSON → returns error record
- Deeply nested values are flattened one level (top-level keys only)
- Non-dict top-level values → error record
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.models.candidate import DataSource, RawCandidateData
from app.parsers.base import BaseParser
from app.parsers.registry import parser_registry
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


def _flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow copy with all string values stripped of whitespace.

    Non-string values (lists, dicts, numbers) are kept as-is so downstream
    extractors can work with them directly.
    """
    return {
        k: (v.strip() if isinstance(v, str) else v)
        for k, v in record.items()
        if k is not None
    }


@parser_registry.register(DataSource.JSON)
class JSONParser(BaseParser):
    """Parse a JSON file into one RawCandidateData record per candidate object."""

    def parse(self, file_path: str | Path) -> list[RawCandidateData]:
        """Parse a JSON file.

        Args:
            file_path: Path to the .json file.

        Returns:
            One :class:`~app.models.candidate.RawCandidateData` per candidate
            entry found in the JSON.
        """
        path = Path(file_path)

        if not path.exists():
            logger.error("JSON file not found: %s", path)
            return [self._error_record(path, f"File not found: {path}")]

        if path.stat().st_size == 0:
            logger.warning("JSON file is empty: %s", path)
            return []

        try:
            with path.open(encoding="utf-8") as fh:
                content = json.load(fh)
        except json.JSONDecodeError as exc:
            msg = f"JSON decode error at line {exc.lineno}: {exc.msg}"
            logger.error("%s in %s", msg, path)
            return [self._error_record(path, msg)]
        except UnicodeDecodeError as exc:
            msg = f"Encoding error: {exc}"
            logger.error("%s in %s", msg, path)
            return [self._error_record(path, msg)]

        # Normalise to a list of dicts
        if isinstance(content, dict):
            candidates_raw = [content]
        elif isinstance(content, list):
            candidates_raw = content
        else:
            msg = f"Unexpected JSON root type: {type(content).__name__}"
            logger.error("%s in %s", msg, path)
            return [self._error_record(path, msg)]

        records: list[RawCandidateData] = []
        for i, item in enumerate(candidates_raw):
            if not isinstance(item, dict):
                logger.warning(
                    "Skipping non-dict item at index %d in %s (type=%s)",
                    i,
                    path.name,
                    type(item).__name__,
                )
                continue

            records.append(
                RawCandidateData(
                    source=DataSource.JSON,
                    source_file=str(path),
                    raw_fields=_flatten_record(item),
                )
            )

        logger.info("JSONParser produced %d record(s) from %s", len(records), path.name)
        return records
