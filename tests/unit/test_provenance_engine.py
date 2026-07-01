"""Unit tests for the Provenance Engine.

Coverage targets:
  - Empty source_diffs produces empty list
  - Single source with a single present field → one entry
  - Absent fields are silently skipped
  - original_value / normalized_value come from NormalizationDiff when field changed
  - original_value == normalized_value when field not in diff
  - ExtractionMethod: PDF name → INFERRED; PDF other → REGEX; CSV/JSON → DIRECT
  - confidence == reliability weight of the source
  - notes mention normalization when field was changed; None otherwise
  - Multiple sources produce entries for each source independently
  - All nine tracked fields can be present
  - timestamp is populated on every entry
"""

from __future__ import annotations

import pytest

from app.config.models import SourceReliabilityConfig
from app.models.candidate import (
    DataSource,
    ExtractionMethod,
    MergedCandidate,
    NormalizedCandidate,
)
from app.normalizers.engine import NormalizationDiff
from app.provenance.engine import (
    ProvenanceEngine,
    _change_notes,
    _extraction_method,
    _non_empty,
    _original_normalized,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _nc(
    source: DataSource = DataSource.CSV,
    name: str | None = None,
    email: str | None = None,
    phone: str | None = None,
    skills: list[str] | None = None,
    **kwargs,
) -> NormalizedCandidate:
    return NormalizedCandidate(
        source=source,
        name=name,
        email=email,
        phone=phone,
        skills=skills or [],
        **kwargs,
    )


def _diff(**changes) -> NormalizationDiff:
    d = NormalizationDiff()
    for field, (original, normalized) in changes.items():
        d.changes[field] = (original, normalized)
    return d


def _empty_diff() -> NormalizationDiff:
    return NormalizationDiff()


def _merged(sources: list[NormalizedCandidate] | None = None) -> MergedCandidate:
    return MergedCandidate(source_records=sources or [])


def _engine(weights: dict[str, float] | None = None) -> ProvenanceEngine:
    if weights:
        return ProvenanceEngine(SourceReliabilityConfig(weights=weights))
    return ProvenanceEngine()


# ---------------------------------------------------------------------------
# Empty / trivial cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_source_diffs_returns_empty(self):
        entries = _engine().build(_merged(), [])
        assert entries == []

    def test_source_with_all_null_fields_returns_empty(self):
        c = _nc(DataSource.CSV)   # name=None, email=None, skills=[]
        entries = _engine().build(_merged([c]), [(c, _empty_diff())])
        assert entries == []

    def test_single_present_field_creates_one_entry(self):
        c = _nc(DataSource.CSV, name="Alice")
        entries = _engine().build(_merged([c]), [(c, _empty_diff())])
        name_entries = [e for e in entries if e.field_name == "name"]
        assert len(name_entries) == 1

    def test_entry_source_matches_candidate(self):
        c = _nc(DataSource.JSON, name="Alice")
        entries = _engine().build(_merged([c]), [(c, _empty_diff())])
        assert entries[0].source == DataSource.JSON

    def test_timestamp_populated_on_every_entry(self):
        c = _nc(DataSource.CSV, name="Alice", email="a@ex.com")
        entries = _engine().build(_merged([c]), [(c, _empty_diff())])
        for e in entries:
            assert e.timestamp is not None


# ---------------------------------------------------------------------------
# original_value / normalized_value from diff
# ---------------------------------------------------------------------------


class TestOriginalNormalized:
    def test_field_in_diff_returns_diff_values(self):
        c = _nc(DataSource.CSV, email="Alice@EX.COM")
        d = _diff(email=("Alice@EX.COM", "alice@ex.com"))
        entries = _engine().build(_merged([c]), [(c, d)])
        e = next(e for e in entries if e.field_name == "email")
        assert e.original_value == "Alice@EX.COM"
        assert e.normalized_value == "alice@ex.com"

    def test_field_not_in_diff_original_equals_normalized(self):
        c = _nc(DataSource.CSV, name="Alice Smith")
        entries = _engine().build(_merged([c]), [(c, _empty_diff())])
        e = next(e for e in entries if e.field_name == "name")
        assert e.original_value == "Alice Smith"
        assert e.normalized_value == "Alice Smith"

    def test_skills_diff_tracked(self):
        c = _nc(DataSource.CSV, skills=["ML"])
        d = _diff(skills=(("ML",), ("Machine Learning",)))
        entries = _engine().build(_merged([c]), [(c, d)])
        e = next(e for e in entries if e.field_name == "skills")
        assert e.original_value == ("ML",)
        assert e.normalized_value == ("Machine Learning",)


# ---------------------------------------------------------------------------
# ExtractionMethod
# ---------------------------------------------------------------------------


class TestExtractionMethod:
    def test_pdf_name_is_inferred(self):
        c = _nc(DataSource.RESUME_PDF, name="Alice")
        entries = _engine().build(_merged([c]), [(c, _empty_diff())])
        e = next(e for e in entries if e.field_name == "name")
        assert e.extraction_method == ExtractionMethod.INFERRED

    def test_pdf_email_is_regex(self):
        c = _nc(DataSource.RESUME_PDF, email="alice@ex.com")
        entries = _engine().build(_merged([c]), [(c, _empty_diff())])
        e = next(e for e in entries if e.field_name == "email")
        assert e.extraction_method == ExtractionMethod.REGEX

    def test_pdf_phone_is_regex(self):
        c = _nc(DataSource.RESUME_PDF, phone="+1234567890")
        entries = _engine().build(_merged([c]), [(c, _empty_diff())])
        e = next(e for e in entries if e.field_name == "phone")
        assert e.extraction_method == ExtractionMethod.REGEX

    def test_csv_name_is_direct(self):
        c = _nc(DataSource.CSV, name="Alice")
        entries = _engine().build(_merged([c]), [(c, _empty_diff())])
        e = next(e for e in entries if e.field_name == "name")
        assert e.extraction_method == ExtractionMethod.DIRECT

    def test_json_email_is_direct(self):
        c = _nc(DataSource.JSON, email="alice@ex.com")
        entries = _engine().build(_merged([c]), [(c, _empty_diff())])
        e = next(e for e in entries if e.field_name == "email")
        assert e.extraction_method == ExtractionMethod.DIRECT

    def test_ats_is_direct(self):
        c = _nc(DataSource.ATS, name="Alice")
        entries = _engine().build(_merged([c]), [(c, _empty_diff())])
        e = next(e for e in entries if e.field_name == "name")
        assert e.extraction_method == ExtractionMethod.DIRECT


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------


class TestConfidence:
    def test_confidence_equals_reliability_weight(self):
        # Default CSV weight is 0.70
        c = _nc(DataSource.CSV, name="Alice")
        entries = _engine().build(_merged([c]), [(c, _empty_diff())])
        e = next(e for e in entries if e.field_name == "name")
        assert e.confidence == pytest.approx(0.70, abs=0.01)

    def test_custom_weight_reflected(self):
        e = _engine(weights={"csv": 0.42})
        c = _nc(DataSource.CSV, name="Alice")
        entries = e.build(_merged([c]), [(c, _empty_diff())])
        entry = next(en for en in entries if en.field_name == "name")
        assert entry.confidence == pytest.approx(0.42, abs=0.01)

    def test_higher_reliability_source_has_higher_confidence(self):
        csv_c = _nc(DataSource.CSV, name="Alice")
        ats_c = _nc(DataSource.ATS, name="Alice")
        e = _engine()
        csv_entry = next(
            en for en in e.build(_merged([csv_c]), [(csv_c, _empty_diff())])
            if en.field_name == "name"
        )
        ats_entry = next(
            en for en in e.build(_merged([ats_c]), [(ats_c, _empty_diff())])
            if en.field_name == "name"
        )
        assert ats_entry.confidence > csv_entry.confidence


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------


class TestNotes:
    def test_notes_mention_original_when_changed(self):
        c = _nc(DataSource.CSV, email="ALICE@EX.COM")
        d = _diff(email=("ALICE@EX.COM", "alice@ex.com"))
        entries = _engine().build(_merged([c]), [(c, d)])
        e = next(en for en in entries if en.field_name == "email")
        assert e.notes is not None
        assert "ALICE@EX.COM" in e.notes

    def test_notes_none_when_unchanged(self):
        c = _nc(DataSource.CSV, name="Alice Smith")
        entries = _engine().build(_merged([c]), [(c, _empty_diff())])
        e = next(en for en in entries if en.field_name == "name")
        assert e.notes is None


# ---------------------------------------------------------------------------
# Multiple sources
# ---------------------------------------------------------------------------


class TestMultipleSources:
    def test_two_sources_two_name_entries(self):
        csv_c = _nc(DataSource.CSV, name="Alice")
        json_c = _nc(DataSource.JSON, name="Alice Smith")
        entries = _engine().build(
            _merged([csv_c, json_c]),
            [(csv_c, _empty_diff()), (json_c, _empty_diff())],
        )
        name_entries = [e for e in entries if e.field_name == "name"]
        assert len(name_entries) == 2
        sources = {e.source for e in name_entries}
        assert DataSource.CSV in sources
        assert DataSource.JSON in sources

    def test_field_absent_in_one_source_not_tracked_for_that_source(self):
        csv_c = _nc(DataSource.CSV, name="Alice")
        json_c = _nc(DataSource.JSON, name=None)  # no name
        entries = _engine().build(
            _merged([csv_c, json_c]),
            [(csv_c, _empty_diff()), (json_c, _empty_diff())],
        )
        name_entries = [e for e in entries if e.field_name == "name"]
        assert len(name_entries) == 1
        assert name_entries[0].source == DataSource.CSV

    def test_total_entry_count_matches_present_fields(self):
        # 3 fields present across 2 sources: name+email in csv, phone in json
        csv_c = _nc(DataSource.CSV, name="Alice", email="a@ex.com")
        json_c = _nc(DataSource.JSON, phone="+1234567890")
        entries = _engine().build(
            _merged([csv_c, json_c]),
            [(csv_c, _empty_diff()), (json_c, _empty_diff())],
        )
        assert len(entries) == 3


# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_non_empty_none_false(self):
        assert _non_empty(None) is False

    def test_non_empty_blank_false(self):
        assert _non_empty("  ") is False

    def test_non_empty_string_true(self):
        assert _non_empty("x") is True

    def test_non_empty_empty_list_false(self):
        assert _non_empty([]) is False

    def test_non_empty_nonempty_list_true(self):
        assert _non_empty(["a"]) is True

    def test_original_normalized_from_diff(self):
        d = _diff(phone=("+91 98765", "+919876543210"))
        orig, norm = _original_normalized("phone", "+919876543210", d)
        assert orig == "+91 98765"
        assert norm == "+919876543210"

    def test_original_normalized_not_in_diff(self):
        orig, norm = _original_normalized("name", "Alice", _empty_diff())
        assert orig == norm == "Alice"

    def test_extraction_method_pdf_name_inferred(self):
        assert _extraction_method("name", DataSource.RESUME_PDF) == ExtractionMethod.INFERRED

    def test_extraction_method_pdf_email_regex(self):
        assert _extraction_method("email", DataSource.RESUME_PDF) == ExtractionMethod.REGEX

    def test_extraction_method_csv_is_direct(self):
        assert _extraction_method("name", DataSource.CSV) == ExtractionMethod.DIRECT

    def test_change_notes_when_changed(self):
        d = _diff(email=("OLD", "new"))
        note = _change_notes("email", d)
        assert note is not None
        assert "OLD" in note

    def test_change_notes_none_when_not_changed(self):
        assert _change_notes("name", _empty_diff()) is None
