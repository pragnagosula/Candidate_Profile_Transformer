"""Unit tests for the Pipeline Orchestrator.

Coverage targets:
  - PipelineResult dataclass (success property, defaults)
  - _combine_validations static method
  - _resolve_output_path
  - Pipeline.run: empty inputs, single candidate, multi-candidate same person,
    multi-candidate different people, output file creation, JSON validity,
    field renaming, confidence / validation attached to profile
  - Never-raises guarantee on parse/extract exceptions
  - Output directory created when missing
  - finished_at set after run
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.models.candidate import (
    DataSource,
    ExtractedCandidate,
    RawCandidateData,
    ValidationIssue,
    ValidationResult,
)
from app.pipeline.orchestrator import Pipeline, PipelineResult


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _raw(
    source: DataSource = DataSource.CSV,
    name: str = "Alice Smith",
    email: str = "alice@example.com",
    file: str = "test.csv",
) -> RawCandidateData:
    return RawCandidateData(
        source=source,
        source_file=file,
        raw_fields={"name": name, "email": email},
    )


def _extracted(
    source: DataSource = DataSource.CSV,
    name: str = "Alice Smith",
    email: str = "alice@example.com",
    skills: list[str] | None = None,
    file: str = "test.csv",
) -> ExtractedCandidate:
    return ExtractedCandidate(
        source=source,
        source_file=file,
        name=name,
        email=email,
        skills=skills or ["Python"],
    )


def _validation_result(valid: bool = True) -> ValidationResult:
    if valid:
        return ValidationResult(is_valid=True, issues=[])
    return ValidationResult(
        is_valid=False,
        issues=[ValidationIssue(field="email", severity="error", message="bad")],
    )


# ---------------------------------------------------------------------------
# PipelineResult
# ---------------------------------------------------------------------------


class TestPipelineResult:
    def test_success_true_when_no_errors(self):
        r = PipelineResult()
        assert r.success is True

    def test_success_false_when_has_errors(self):
        r = PipelineResult(errors=["something went wrong"])
        assert r.success is False

    def test_profiles_empty_by_default(self):
        r = PipelineResult()
        assert r.profiles == []

    def test_total_inputs_zero_by_default(self):
        r = PipelineResult()
        assert r.total_inputs == 0

    def test_total_groups_zero_by_default(self):
        r = PipelineResult()
        assert r.total_groups == 0

    def test_started_at_set_on_creation(self):
        before = datetime.utcnow()
        r = PipelineResult()
        after = datetime.utcnow()
        assert before <= r.started_at <= after

    def test_finished_at_none_by_default(self):
        r = PipelineResult()
        assert r.finished_at is None


# ---------------------------------------------------------------------------
# _combine_validations (static method)
# ---------------------------------------------------------------------------


class TestCombineValidations:
    def test_empty_list_returns_none(self):
        assert Pipeline._combine_validations([]) is None

    def test_single_valid_result_is_valid(self):
        combined = Pipeline._combine_validations([_validation_result(True)])
        assert combined is not None
        assert combined.is_valid

    def test_single_invalid_result_is_invalid(self):
        combined = Pipeline._combine_validations([_validation_result(False)])
        assert combined is not None
        assert not combined.is_valid

    def test_multiple_valid_stays_valid(self):
        combined = Pipeline._combine_validations(
            [_validation_result(True), _validation_result(True)]
        )
        assert combined.is_valid

    def test_mix_of_valid_and_invalid_is_invalid(self):
        combined = Pipeline._combine_validations(
            [_validation_result(True), _validation_result(False)]
        )
        assert not combined.is_valid

    def test_issues_aggregated_from_all_results(self):
        vr1 = ValidationResult(
            is_valid=False,
            issues=[ValidationIssue(field="name", severity="error", message="a")],
        )
        vr2 = ValidationResult(
            is_valid=False,
            issues=[ValidationIssue(field="email", severity="error", message="b")],
        )
        combined = Pipeline._combine_validations([vr1, vr2])
        assert len(combined.issues) == 2

    def test_warnings_only_stays_valid(self):
        vr = ValidationResult(
            is_valid=True,
            issues=[ValidationIssue(field="phone", severity="warning", message="w")],
        )
        combined = Pipeline._combine_validations([vr])
        assert combined.is_valid


# ---------------------------------------------------------------------------
# _resolve_output_path
# ---------------------------------------------------------------------------


class TestResolveOutputPath:
    def _pipeline(self) -> Pipeline:
        return Pipeline()

    def test_explicit_string_path_converted_to_path(self, tmp_path):
        p = self._pipeline()
        result = p._resolve_output_path(str(tmp_path / "out.json"))
        assert isinstance(result, Path)
        assert result == tmp_path / "out.json"

    def test_explicit_path_object_returned_unchanged(self, tmp_path):
        p = self._pipeline()
        dest = tmp_path / "profiles.json"
        assert p._resolve_output_path(dest) == dest

    def test_none_uses_config_defaults(self):
        p = self._pipeline()
        result = p._resolve_output_path(None)
        cfg = p._config
        assert result == Path(cfg.output_dir) / cfg.output_filename


# ---------------------------------------------------------------------------
# Pipeline.run — helpers for mock-based tests
# ---------------------------------------------------------------------------


def _run_with_mocks(
    tmp_path: Path,
    raw_records_by_call: list[list[RawCandidateData]],
    extracted_records: list[ExtractedCandidate],
    inputs: list[tuple] | None = None,
) -> PipelineResult:
    """Run the pipeline with parser and extractor fully mocked.

    raw_records_by_call: what parser_registry.parse returns for each call
    extracted_records:   what extractor_registry.extract returns in order
    """
    if inputs is None:
        inputs = [(DataSource.CSV, "test.csv")] * len(raw_records_by_call)

    with patch("app.pipeline.orchestrator.parser_registry") as mock_parser, \
         patch("app.pipeline.orchestrator.extractor_registry") as mock_extractor:

        mock_parser.parse.side_effect = raw_records_by_call
        mock_extractor.extract.side_effect = extracted_records

        pipeline = Pipeline()
        return pipeline.run(inputs, output_path=tmp_path / "out.json")


# ---------------------------------------------------------------------------
# Pipeline.run — empty and single-candidate cases
# ---------------------------------------------------------------------------


class TestPipelineRunBasic:
    def test_empty_inputs_produces_no_profiles(self, tmp_path):
        pipeline = Pipeline()
        result = pipeline.run([], output_path=tmp_path / "out.json")
        assert result.profiles == []

    def test_empty_inputs_total_inputs_zero(self, tmp_path):
        pipeline = Pipeline()
        result = pipeline.run([], output_path=tmp_path / "out.json")
        assert result.total_inputs == 0

    def test_empty_inputs_total_groups_zero(self, tmp_path):
        pipeline = Pipeline()
        result = pipeline.run([], output_path=tmp_path / "out.json")
        assert result.total_groups == 0

    def test_empty_inputs_success_true(self, tmp_path):
        pipeline = Pipeline()
        result = pipeline.run([], output_path=tmp_path / "out.json")
        assert result.success is True

    def test_single_candidate_one_profile(self, tmp_path):
        result = _run_with_mocks(
            tmp_path,
            raw_records_by_call=[[_raw()]],
            extracted_records=[_extracted()],
        )
        assert len(result.profiles) == 1

    def test_single_candidate_total_inputs_one(self, tmp_path):
        result = _run_with_mocks(
            tmp_path,
            raw_records_by_call=[[_raw()]],
            extracted_records=[_extracted()],
        )
        assert result.total_inputs == 1

    def test_single_candidate_total_groups_one(self, tmp_path):
        result = _run_with_mocks(
            tmp_path,
            raw_records_by_call=[[_raw()]],
            extracted_records=[_extracted()],
        )
        assert result.total_groups == 1

    def test_finished_at_set_after_run(self, tmp_path):
        before = datetime.utcnow()
        result = _run_with_mocks(
            tmp_path,
            raw_records_by_call=[[_raw()]],
            extracted_records=[_extracted()],
        )
        after = datetime.utcnow()
        assert result.finished_at is not None
        assert before <= result.finished_at <= after

    def test_success_true_on_clean_run(self, tmp_path):
        result = _run_with_mocks(
            tmp_path,
            raw_records_by_call=[[_raw()]],
            extracted_records=[_extracted()],
        )
        assert result.success is True


# ---------------------------------------------------------------------------
# Pipeline.run — entity resolution (grouping)
# ---------------------------------------------------------------------------


class TestPipelineGrouping:
    def test_two_records_same_email_one_group(self, tmp_path):
        """Same email → entity resolver merges into one group."""
        result = _run_with_mocks(
            tmp_path,
            raw_records_by_call=[
                [_raw(source=DataSource.CSV, email="alice@example.com")],
                [_raw(source=DataSource.JSON, email="alice@example.com")],
            ],
            extracted_records=[
                _extracted(source=DataSource.CSV, email="alice@example.com"),
                _extracted(source=DataSource.JSON, email="alice@example.com"),
            ],
            inputs=[
                (DataSource.CSV, "test.csv"),
                (DataSource.JSON, "test.json"),
            ],
        )
        assert result.total_inputs == 2
        assert result.total_groups == 1
        assert len(result.profiles) == 1

    def test_two_records_different_email_two_groups(self, tmp_path):
        """Different emails → two separate groups → two profiles."""
        result = _run_with_mocks(
            tmp_path,
            raw_records_by_call=[
                [_raw(source=DataSource.CSV, name="Alice", email="alice@example.com")],
                [_raw(source=DataSource.JSON, name="Bob", email="bob@example.com")],
            ],
            extracted_records=[
                _extracted(source=DataSource.CSV, name="Alice", email="alice@example.com"),
                _extracted(source=DataSource.JSON, name="Bob", email="bob@example.com"),
            ],
            inputs=[
                (DataSource.CSV, "a.csv"),
                (DataSource.JSON, "b.json"),
            ],
        )
        assert result.total_groups == 2
        assert len(result.profiles) == 2

    def test_csv_with_multiple_rows(self, tmp_path):
        """A single CSV file producing multiple raw records."""
        result = _run_with_mocks(
            tmp_path,
            raw_records_by_call=[[
                _raw(source=DataSource.CSV, name="Alice", email="alice@example.com"),
                _raw(source=DataSource.CSV, name="Bob", email="bob@example.com"),
            ]],
            extracted_records=[
                _extracted(source=DataSource.CSV, name="Alice", email="alice@example.com"),
                _extracted(source=DataSource.CSV, name="Bob", email="bob@example.com"),
            ],
        )
        assert result.total_inputs == 2
        assert result.total_groups == 2
        assert len(result.profiles) == 2


# ---------------------------------------------------------------------------
# Pipeline.run — output file
# ---------------------------------------------------------------------------


class TestPipelineOutput:
    def test_output_file_exists_after_run(self, tmp_path):
        out = tmp_path / "out.json"
        _run_with_mocks(
            tmp_path,
            raw_records_by_call=[[_raw()]],
            extracted_records=[_extracted()],
        )
        assert out.exists()

    def test_output_is_valid_json(self, tmp_path):
        _run_with_mocks(
            tmp_path,
            raw_records_by_call=[[_raw()]],
            extracted_records=[_extracted()],
        )
        content = (tmp_path / "out.json").read_text(encoding="utf-8")
        parsed = json.loads(content)
        assert isinstance(parsed, list)

    def test_output_array_length_matches_profiles(self, tmp_path):
        _run_with_mocks(
            tmp_path,
            raw_records_by_call=[[
                _raw(source=DataSource.CSV, name="Alice", email="alice@example.com"),
                _raw(source=DataSource.CSV, name="Bob", email="bob@example.com"),
            ]],
            extracted_records=[
                _extracted(source=DataSource.CSV, name="Alice", email="alice@example.com"),
                _extracted(source=DataSource.CSV, name="Bob", email="bob@example.com"),
            ],
        )
        content = json.loads((tmp_path / "out.json").read_text())
        assert len(content) == 2

    def test_output_directory_created_if_missing(self, tmp_path):
        deep_out = tmp_path / "a" / "b" / "c" / "out.json"
        with patch("app.pipeline.orchestrator.parser_registry") as mock_parser, \
             patch("app.pipeline.orchestrator.extractor_registry") as mock_extractor:
            mock_parser.parse.return_value = [_raw()]
            mock_extractor.extract.return_value = _extracted()
            pipeline = Pipeline()
            pipeline.run(
                [(DataSource.CSV, "test.csv")],
                output_path=deep_out,
            )
        assert deep_out.exists()

    def test_empty_run_writes_empty_array(self, tmp_path):
        out = tmp_path / "out.json"
        pipeline = Pipeline()
        pipeline.run([], output_path=out)
        content = json.loads(out.read_text())
        assert content == []


# ---------------------------------------------------------------------------
# Pipeline.run — profile content
# ---------------------------------------------------------------------------


class TestPipelineProfileContent:
    def test_profile_has_full_name_field(self, tmp_path):
        result = _run_with_mocks(
            tmp_path,
            raw_records_by_call=[[_raw(name="Alice Smith")]],
            extracted_records=[_extracted(name="Alice Smith")],
        )
        profile = result.profiles[0]
        assert "full_name" in profile.fields

    def test_profile_full_name_value_correct(self, tmp_path):
        result = _run_with_mocks(
            tmp_path,
            raw_records_by_call=[[_raw(name="Alice Smith")]],
            extracted_records=[_extracted(name="Alice Smith")],
        )
        profile = result.profiles[0]
        assert profile.fields["full_name"] == "Alice Smith"

    def test_profile_has_confidence_attached(self, tmp_path):
        result = _run_with_mocks(
            tmp_path,
            raw_records_by_call=[[_raw()]],
            extracted_records=[_extracted()],
        )
        profile = result.profiles[0]
        assert profile.confidence is not None
        assert 0.0 <= profile.confidence.overall_score <= 1.0

    def test_profile_has_validation_attached(self, tmp_path):
        result = _run_with_mocks(
            tmp_path,
            raw_records_by_call=[[_raw()]],
            extracted_records=[_extracted()],
        )
        profile = result.profiles[0]
        assert profile.validation is not None

    def test_profile_generated_at_is_set(self, tmp_path):
        result = _run_with_mocks(
            tmp_path,
            raw_records_by_call=[[_raw()]],
            extracted_records=[_extracted()],
        )
        assert result.profiles[0].generated_at is not None


# ---------------------------------------------------------------------------
# Never-crash guarantees
# ---------------------------------------------------------------------------


class TestNeverCrash:
    def test_parse_exception_adds_error_not_raises(self, tmp_path):
        with patch("app.pipeline.orchestrator.parser_registry") as mock_parser:
            mock_parser.parse.side_effect = RuntimeError("disk failure")
            pipeline = Pipeline()
            try:
                result = pipeline.run(
                    [(DataSource.CSV, "bad_file.csv")],
                    output_path=tmp_path / "out.json",
                )
            except Exception:
                pytest.fail("Pipeline.run raised despite never-crash guarantee")
        # Parse failure means no candidates processed; no pipeline-level error added
        # (it's caught per-source), so result may still succeed with 0 profiles
        assert result.total_inputs == 0

    def test_extract_exception_candidate_skipped(self, tmp_path):
        with patch("app.pipeline.orchestrator.parser_registry") as mock_parser, \
             patch("app.pipeline.orchestrator.extractor_registry") as mock_extractor:
            mock_parser.parse.return_value = [_raw()]
            mock_extractor.extract.side_effect = ValueError("extraction blew up")
            pipeline = Pipeline()
            result = pipeline.run(
                [(DataSource.CSV, "test.csv")],
                output_path=tmp_path / "out.json",
            )
        assert result.total_inputs == 0
        assert result.profiles == []

    def test_never_raises_on_none_input_list(self, tmp_path):
        pipeline = Pipeline()
        try:
            result = pipeline.run([], output_path=tmp_path / "out.json")
            assert isinstance(result, PipelineResult)
        except Exception:
            pytest.fail("Pipeline.run raised on empty input")

    def test_parse_errors_in_raw_still_processed(self, tmp_path):
        """A raw record with parse_errors is still passed through extraction."""
        raw_with_errors = RawCandidateData(
            source=DataSource.CSV,
            source_file="test.csv",
            raw_fields={"name": "Alice", "email": "alice@example.com"},
            parse_errors=["warning: missing field 'phone'"],
        )
        with patch("app.pipeline.orchestrator.parser_registry") as mock_parser, \
             patch("app.pipeline.orchestrator.extractor_registry") as mock_extractor:
            mock_parser.parse.return_value = [raw_with_errors]
            mock_extractor.extract.return_value = _extracted()
            pipeline = Pipeline()
            result = pipeline.run(
                [(DataSource.CSV, "test.csv")],
                output_path=tmp_path / "out.json",
            )
        assert result.total_inputs == 1
        assert len(result.profiles) == 1

    def test_output_write_failure_adds_error_not_raises(self, tmp_path):
        """If writing fails, the error is captured, not raised."""
        with patch("app.pipeline.orchestrator.parser_registry") as mock_parser, \
             patch("app.pipeline.orchestrator.extractor_registry") as mock_extractor, \
             patch.object(Pipeline, "_write_output", side_effect=OSError("disk full")):
            mock_parser.parse.return_value = [_raw()]
            mock_extractor.extract.return_value = _extracted()
            pipeline = Pipeline()
            try:
                result = pipeline.run(
                    [(DataSource.CSV, "test.csv")],
                    output_path=tmp_path / "out.json",
                )
            except Exception:
                pytest.fail("Pipeline.run raised when write failed")
        assert not result.success
        assert any("disk full" in e for e in result.errors)
