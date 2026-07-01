"""Unit tests for core data models."""

import pytest
from pydantic import ValidationError

from app.models.candidate import (
    CandidateProfile,
    ConfidenceReport,
    DataSource,
    DateRange,
    Education,
    Experience,
    ExtractedCandidate,
    FieldConfidence,
    Link,
    MergedCandidate,
    NormalizedCandidate,
    ProvenanceEntry,
    RawCandidateData,
    ValidationIssue,
    ValidationResult,
    ExtractionMethod,
)


class TestRawCandidateData:
    def test_defaults_are_empty_collections(self):
        raw = RawCandidateData(source=DataSource.CSV)
        assert raw.raw_fields == {}
        assert raw.parse_errors == []

    def test_stores_arbitrary_fields(self):
        raw = RawCandidateData(source=DataSource.JSON, raw_fields={"name": "Alice", "age": "30"})
        assert raw.raw_fields["name"] == "Alice"

    def test_parse_timestamp_is_set(self):
        raw = RawCandidateData(source=DataSource.CSV)
        assert raw.parse_timestamp is not None


class TestNormalizedCandidate:
    def test_email_is_lowercased(self):
        c = NormalizedCandidate(source=DataSource.CSV, email="Alice@Example.COM")
        assert c.email == "alice@example.com"

    def test_email_is_stripped(self):
        c = NormalizedCandidate(source=DataSource.CSV, email="  bob@test.com  ")
        assert c.email == "bob@test.com"

    def test_email_none_passes(self):
        c = NormalizedCandidate(source=DataSource.CSV, email=None)
        assert c.email is None

    def test_skills_default_to_empty(self):
        c = NormalizedCandidate(source=DataSource.CSV)
        assert c.skills == []


class TestValidationResult:
    def test_errors_filtered(self):
        result = ValidationResult(
            is_valid=False,
            issues=[
                ValidationIssue(field="email", severity="error", message="invalid"),
                ValidationIssue(field="phone", severity="warning", message="unformatted"),
            ],
        )
        assert len(result.errors) == 1
        assert result.errors[0].field == "email"

    def test_warnings_filtered(self):
        result = ValidationResult(
            is_valid=True,
            issues=[
                ValidationIssue(field="phone", severity="warning", message="unformatted"),
            ],
        )
        assert len(result.warnings) == 1
        assert result.warnings[0].field == "phone"

    def test_empty_issues(self):
        result = ValidationResult(is_valid=True)
        assert result.errors == []
        assert result.warnings == []


class TestConfidenceReport:
    def test_score_bounds(self):
        with pytest.raises(ValidationError):
            ConfidenceReport(overall_score=1.5, completeness=0.9, source_agreement=0.8)

    def test_valid_report(self):
        report = ConfidenceReport(
            overall_score=0.85,
            completeness=0.9,
            source_agreement=0.8,
            field_scores=[
                FieldConfidence(
                    field_name="email",
                    score=1.0,
                    contributing_sources=[DataSource.CSV],
                )
            ],
        )
        assert report.overall_score == 0.85
        assert len(report.field_scores) == 1


class TestMergedCandidate:
    def test_empty_merged_candidate(self):
        merged = MergedCandidate()
        assert merged.skills == []
        assert merged.experience == []
        assert merged.provenance == []

    def test_holds_source_records(self):
        src = NormalizedCandidate(source=DataSource.CSV, name="Alice")
        merged = MergedCandidate(source_records=[src])
        assert merged.source_records[0].name == "Alice"


class TestDateRange:
    def test_is_current_defaults_false(self):
        dr = DateRange(start="2020-01")
        assert dr.is_current is False

    def test_end_none_means_present(self):
        dr = DateRange(start="2022-06", end=None, is_current=True)
        assert dr.end is None
        assert dr.is_current is True


class TestExperience:
    def test_all_optional(self):
        exp = Experience()
        assert exp.company is None
        assert exp.title is None

    def test_with_duration(self):
        exp = Experience(
            company="Acme",
            title="Engineer",
            duration=DateRange(start="2021-01", end="2023-06"),
        )
        assert exp.duration.start == "2021-01"


class TestProvenanceEntry:
    def test_timestamp_auto_set(self):
        entry = ProvenanceEntry(
            field_name="email",
            source=DataSource.CSV,
            original_value="TEST@EXAMPLE.COM",
            normalized_value="test@example.com",
            extraction_method=ExtractionMethod.DIRECT,
            confidence=0.95,
        )
        assert entry.timestamp is not None

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            ProvenanceEntry(
                field_name="email",
                source=DataSource.CSV,
                original_value="x",
                normalized_value="x",
                extraction_method=ExtractionMethod.DIRECT,
                confidence=1.5,
            )
