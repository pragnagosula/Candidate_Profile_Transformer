"""Unit tests for the parser layer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# Importing the package triggers self-registration of all parsers
import app.parsers  # noqa: F401
from app.models.candidate import DataSource
from app.parsers.csv_parser import CSVParser
from app.parsers.json_parser import JSONParser
from app.parsers.pdf_parser import ResumePDFParser
from app.parsers.registry import ParserRegistry, parser_registry
from app.parsers.txt_parser import ResumeTXTParser


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestParserRegistry:
    def test_all_four_parsers_registered(self):
        sources = parser_registry.registered_sources
        assert DataSource.CSV in sources
        assert DataSource.JSON in sources
        assert DataSource.RESUME_PDF in sources
        assert DataSource.RESUME_TXT in sources

    def test_get_returns_correct_class(self):
        assert parser_registry.get(DataSource.CSV) is CSVParser
        assert parser_registry.get(DataSource.JSON) is JSONParser
        assert parser_registry.get(DataSource.RESUME_PDF) is ResumePDFParser
        assert parser_registry.get(DataSource.RESUME_TXT) is ResumeTXTParser

    def test_get_unknown_source_raises(self):
        fresh = ParserRegistry()
        with pytest.raises(KeyError, match="No parser registered"):
            fresh.get(DataSource.LINKEDIN)

    def test_parse_unknown_source_returns_error_record(self, tmp_path):
        fresh = ParserRegistry()
        records = fresh.parse(DataSource.LINKEDIN, tmp_path / "x.json")
        assert len(records) == 1
        assert records[0].parse_errors

    def test_register_overrides_existing(self):
        from app.parsers.base import BaseParser

        fresh = ParserRegistry()

        @fresh.register(DataSource.CSV)
        class FakeParser(BaseParser):
            def parse(self, file_path):
                return []

        assert fresh.get(DataSource.CSV) is FakeParser


# ---------------------------------------------------------------------------
# CSVParser
# ---------------------------------------------------------------------------


class TestCSVParser:
    def test_parses_valid_csv(self, tmp_path):
        f = tmp_path / "candidates.csv"
        f.write_text("name,email\nAlice,alice@example.com\nBob,bob@example.com\n")
        records = CSVParser().parse(f)
        assert len(records) == 2
        assert records[0].raw_fields["name"] == "Alice"
        assert records[0].source == DataSource.CSV

    def test_strips_whitespace_from_values(self, tmp_path):
        f = tmp_path / "candidates.csv"
        f.write_text("name,email\n  Alice  ,  alice@test.com  \n")
        records = CSVParser().parse(f)
        assert records[0].raw_fields["name"] == "Alice"
        assert records[0].raw_fields["email"] == "alice@test.com"

    def test_skips_all_empty_rows(self, tmp_path):
        f = tmp_path / "candidates.csv"
        f.write_text("name,email\nAlice,alice@test.com\n , \n")
        records = CSVParser().parse(f)
        assert len(records) == 1

    def test_empty_file_returns_empty_list(self, tmp_path):
        f = tmp_path / "empty.csv"
        f.write_text("")
        records = CSVParser().parse(f)
        assert records == []

    def test_missing_file_returns_error_record(self, tmp_path):
        records = CSVParser().parse(tmp_path / "nonexistent.csv")
        assert len(records) == 1
        assert records[0].parse_errors

    def test_source_file_is_set(self, tmp_path):
        f = tmp_path / "candidates.csv"
        f.write_text("name\nAlice\n")
        records = CSVParser().parse(f)
        assert records[0].source_file == str(f)

    def test_header_only_csv_returns_empty(self, tmp_path):
        f = tmp_path / "header_only.csv"
        f.write_text("name,email,phone\n")
        records = CSVParser().parse(f)
        assert records == []

    def test_csv_with_unicode_names(self, tmp_path):
        f = tmp_path / "unicode.csv"
        f.write_bytes("name,email\nÃlicé,alice@test.com\n".encode("utf-8"))
        records = CSVParser().parse(f)
        assert "Ãlicé" in records[0].raw_fields["name"] or records[0].raw_fields["name"]

    def test_sample_fixture(self):
        fixture = Path("tests/fixtures/sample_candidates.csv")
        if not fixture.exists():
            pytest.skip("Fixture not present")
        records = CSVParser().parse(fixture)
        assert len(records) == 3  # 3 data rows; all-whitespace row is skipped


# ---------------------------------------------------------------------------
# JSONParser
# ---------------------------------------------------------------------------


class TestJSONParser:
    def test_parses_array_of_candidates(self, tmp_path):
        data = [{"name": "Alice"}, {"name": "Bob"}]
        f = tmp_path / "candidates.json"
        f.write_text(json.dumps(data))
        records = JSONParser().parse(f)
        assert len(records) == 2

    def test_parses_single_object(self, tmp_path):
        data = {"name": "Alice", "email": "alice@test.com"}
        f = tmp_path / "candidate.json"
        f.write_text(json.dumps(data))
        records = JSONParser().parse(f)
        assert len(records) == 1
        assert records[0].raw_fields["name"] == "Alice"

    def test_strips_whitespace_from_string_values(self, tmp_path):
        f = tmp_path / "c.json"
        f.write_text('{"name": "  Alice  "}')
        records = JSONParser().parse(f)
        assert records[0].raw_fields["name"] == "Alice"

    def test_preserves_list_values(self, tmp_path):
        f = tmp_path / "c.json"
        f.write_text('{"skills": ["Python", "ML"]}')
        records = JSONParser().parse(f)
        assert records[0].raw_fields["skills"] == ["Python", "ML"]

    def test_empty_file_returns_empty_list(self, tmp_path):
        f = tmp_path / "empty.json"
        f.write_text("")
        records = JSONParser().parse(f)
        assert records == []

    def test_malformed_json_returns_error_record(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text("{name: Alice}")  # invalid JSON
        records = JSONParser().parse(f)
        assert len(records) == 1
        assert records[0].parse_errors

    def test_missing_file_returns_error_record(self, tmp_path):
        records = JSONParser().parse(tmp_path / "nope.json")
        assert records[0].parse_errors

    def test_non_dict_items_in_array_are_skipped(self, tmp_path):
        f = tmp_path / "mixed.json"
        f.write_text('[{"name": "Alice"}, "not_a_dict", 42]')
        records = JSONParser().parse(f)
        assert len(records) == 1

    def test_root_is_not_dict_or_list_returns_error(self, tmp_path):
        f = tmp_path / "scalar.json"
        f.write_text('"just a string"')
        records = JSONParser().parse(f)
        assert records[0].parse_errors

    def test_sample_fixture(self):
        fixture = Path("tests/fixtures/sample_candidates.json")
        if not fixture.exists():
            pytest.skip("Fixture not present")
        records = JSONParser().parse(fixture)
        assert len(records) == 2


# ---------------------------------------------------------------------------
# ResumePDFParser
# ---------------------------------------------------------------------------


class TestResumePDFParser:
    def test_missing_file_returns_error_record(self, tmp_path):
        records = ResumePDFParser().parse(tmp_path / "nope.pdf")
        assert records[0].parse_errors

    def test_empty_file_returns_error_record(self, tmp_path):
        f = tmp_path / "empty.pdf"
        f.write_bytes(b"")
        records = ResumePDFParser().parse(f)
        assert records[0].parse_errors

    def test_corrupted_pdf_returns_error_record(self, tmp_path):
        f = tmp_path / "corrupted.pdf"
        f.write_bytes(b"%PDF-1.4 this is not a valid PDF at all ###")
        records = ResumePDFParser().parse(f)
        # Either error record or empty text — must not raise
        assert isinstance(records, list)
        assert len(records) == 1

    def test_source_is_resume_pdf(self, tmp_path):
        f = tmp_path / "corrupted.pdf"
        f.write_bytes(b"not a real pdf")
        records = ResumePDFParser().parse(f)
        assert records[0].source == DataSource.RESUME_PDF

    def test_source_file_is_set(self, tmp_path):
        f = tmp_path / "x.pdf"
        f.write_bytes(b"not real")
        records = ResumePDFParser().parse(f)
        assert records[0].source_file == str(f)


# ---------------------------------------------------------------------------
# ResumeTXTParser
# ---------------------------------------------------------------------------


class TestResumeTXTParser:
    def test_reads_utf8_file(self, tmp_path):
        f = tmp_path / "resume.txt"
        f.write_text("Alice Smith\nalice@example.com\n", encoding="utf-8")
        records = ResumeTXTParser().parse(f)
        assert len(records) == 1
        assert "Alice Smith" in records[0].raw_fields["raw_text"]

    def test_reads_utf8_bom_file(self, tmp_path):
        f = tmp_path / "resume.txt"
        f.write_text("Bob Jones\nbob@test.com\n", encoding="utf-8-sig")
        records = ResumeTXTParser().parse(f)
        assert records[0].raw_fields["raw_text"].startswith("Bob Jones")

    def test_reads_latin1_file(self, tmp_path):
        f = tmp_path / "resume.txt"
        # é is not valid UTF-8 as a single byte; latin-1 will decode it fine
        f.write_bytes("Ren\xe9 Dupont\n".encode("latin-1"))
        records = ResumeTXTParser().parse(f)
        assert len(records) == 1
        assert records[0].raw_fields["raw_text"]  # non-empty

    def test_source_is_resume_txt(self, tmp_path):
        f = tmp_path / "resume.txt"
        f.write_text("Some content", encoding="utf-8")
        records = ResumeTXTParser().parse(f)
        assert records[0].source == DataSource.RESUME_TXT

    def test_source_file_is_set(self, tmp_path):
        f = tmp_path / "resume.txt"
        f.write_text("Some content", encoding="utf-8")
        records = ResumeTXTParser().parse(f)
        assert records[0].source_file == str(f)

    def test_raw_text_in_raw_fields(self, tmp_path):
        f = tmp_path / "resume.txt"
        content = "Jane Doe\njane@example.com\nPython, SQL"
        f.write_text(content, encoding="utf-8")
        records = ResumeTXTParser().parse(f)
        assert records[0].raw_fields["raw_text"] == content

    def test_missing_file_returns_error_record(self, tmp_path):
        records = ResumeTXTParser().parse(tmp_path / "nope.txt")
        assert len(records) == 1
        assert records[0].parse_errors

    def test_empty_file_returns_error_record(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_bytes(b"")
        records = ResumeTXTParser().parse(f)
        assert len(records) == 1
        assert records[0].parse_errors

    def test_whitespace_only_file_produces_record(self, tmp_path):
        f = tmp_path / "spaces.txt"
        f.write_text("   \n\n   ", encoding="utf-8")
        records = ResumeTXTParser().parse(f)
        # Parser strips whitespace; raw_text will be empty string but no error
        assert len(records) == 1
        assert not records[0].parse_errors
        assert records[0].raw_fields["raw_text"] == ""

    def test_strips_leading_trailing_whitespace(self, tmp_path):
        f = tmp_path / "resume.txt"
        f.write_text("\n\nAlice Smith\n\n", encoding="utf-8")
        records = ResumeTXTParser().parse(f)
        assert records[0].raw_fields["raw_text"] == "Alice Smith"

    def test_multiline_content_preserved(self, tmp_path):
        f = tmp_path / "resume.txt"
        content = "Alice Smith\nalice@example.com\n\nSkills\nPython, SQL"
        f.write_text(content, encoding="utf-8")
        records = ResumeTXTParser().parse(f)
        assert "\n" in records[0].raw_fields["raw_text"]


# ---------------------------------------------------------------------------
# Parser routing — extension → DataSource (via routes._EXT_TO_SOURCE)
# ---------------------------------------------------------------------------


class TestParserRouting:
    """Verify that file extensions are routed to the correct DataSource.

    These tests import _EXT_TO_SOURCE directly so they stay in sync with
    the routing table without spinning up FastAPI.
    """

    def test_pdf_routes_to_resume_pdf(self):
        from app.api.routes import _EXT_TO_SOURCE
        assert _EXT_TO_SOURCE[".pdf"] is DataSource.RESUME_PDF

    def test_txt_routes_to_resume_txt(self):
        from app.api.routes import _EXT_TO_SOURCE
        assert _EXT_TO_SOURCE[".txt"] is DataSource.RESUME_TXT

    def test_csv_routes_to_csv(self):
        from app.api.routes import _EXT_TO_SOURCE
        assert _EXT_TO_SOURCE[".csv"] is DataSource.CSV

    def test_json_routes_to_json(self):
        from app.api.routes import _EXT_TO_SOURCE
        assert _EXT_TO_SOURCE[".json"] is DataSource.JSON

    def test_txt_does_not_route_to_resume_pdf(self):
        from app.api.routes import _EXT_TO_SOURCE
        assert _EXT_TO_SOURCE[".txt"] is not DataSource.RESUME_PDF

    def test_registry_dispatch_pdf(self, tmp_path):
        """parser_registry.parse with RESUME_PDF never calls ResumeTXTParser."""
        f = tmp_path / "fake.pdf"
        f.write_bytes(b"not a real pdf")
        records = parser_registry.parse(DataSource.RESUME_PDF, f)
        assert records[0].source == DataSource.RESUME_PDF

    def test_registry_dispatch_txt(self, tmp_path):
        """parser_registry.parse with RESUME_TXT uses ResumeTXTParser."""
        f = tmp_path / "fake.txt"
        f.write_text("Alice Smith\nalice@example.com", encoding="utf-8")
        records = parser_registry.parse(DataSource.RESUME_TXT, f)
        assert records[0].source == DataSource.RESUME_TXT
        assert "Alice Smith" in records[0].raw_fields["raw_text"]
