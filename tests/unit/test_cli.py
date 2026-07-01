"""Unit tests for the Click CLI (app/cli.py).

Coverage targets:
  - No inputs → exit 1, helpful error message
  - --csv / --json / --pdf flags route to correct DataSource
  - Multiple flags of same type collected
  - Mixed flags all passed to pipeline
  - --output flag forwarded to pipeline.run
  - --output absent → pipeline called with output_path=None
  - Successful run → exit 0, "Done." printed
  - Pipeline errors → exit 1, errors printed to stderr
  - Summary lines printed (inputs, groups, profiles)
  - --output path echoed when provided
  - --verbose accepted without crash
  - --config triggers custom config load
  - --help exits 0 and shows usage
  - --version exits 0
  - main.py contains correct entry-point call
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from app.cli import main_cli
from app.pipeline.orchestrator import PipelineResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _result(
    total_inputs: int = 1,
    total_groups: int = 1,
    n_profiles: int = 1,
    errors: list[str] | None = None,
) -> PipelineResult:
    """Build a minimal PipelineResult for use in mocked tests."""
    from app.models.candidate import CandidateProfile
    profiles = [
        CandidateProfile(fields={"full_name": f"Candidate {i}"})
        for i in range(n_profiles)
    ]
    return PipelineResult(
        profiles=profiles,
        total_inputs=total_inputs,
        total_groups=total_groups,
        errors=errors or [],
    )


def _invoke(*args: str, pipeline_result: PipelineResult | None = None) -> object:
    """Invoke the CLI via CliRunner with a mocked Pipeline."""
    if pipeline_result is None:
        pipeline_result = _result()

    runner = CliRunner()
    with patch("app.cli.Pipeline") as mock_cls:
        mock_cls.return_value.run.return_value = pipeline_result
        return runner.invoke(main_cli, list(args)), mock_cls


# ---------------------------------------------------------------------------
# No inputs
# ---------------------------------------------------------------------------


class TestNoInputs:
    def test_no_args_exits_nonzero(self):
        runner = CliRunner()
        result = runner.invoke(main_cli, [])
        assert result.exit_code != 0

    def test_no_args_error_mentions_csv_flag(self):
        runner = CliRunner()
        result = runner.invoke(main_cli, [])
        assert "--csv" in (result.output or "")

    def test_no_args_error_mentions_json_flag(self):
        runner = CliRunner()
        result = runner.invoke(main_cli, [])
        combined = result.output or ""
        assert "--json" in combined or "--pdf" in combined


# ---------------------------------------------------------------------------
# Source routing — correct DataSource per flag
# ---------------------------------------------------------------------------


class TestSourceRouting:
    def test_csv_flag_creates_csv_source(self):
        result, mock_cls = _invoke("--csv", "data.csv")
        call_args = mock_cls.return_value.run.call_args
        inputs = call_args.kwargs.get("inputs") or call_args[1].get("inputs") or call_args[0][0]
        assert any(src.value == "csv" for src, _ in inputs)

    def test_json_flag_creates_json_source(self):
        result, mock_cls = _invoke("--json", "data.json")
        inputs = mock_cls.return_value.run.call_args[1]["inputs"]
        assert any(src.value == "json" for src, _ in inputs)

    def test_pdf_flag_creates_resume_pdf_source(self):
        result, mock_cls = _invoke("--pdf", "resume.pdf")
        inputs = mock_cls.return_value.run.call_args[1]["inputs"]
        assert any(src.value == "resume_pdf" for src, _ in inputs)

    def test_csv_file_path_preserved(self):
        result, mock_cls = _invoke("--csv", "path/to/candidates.csv")
        inputs = mock_cls.return_value.run.call_args[1]["inputs"]
        assert any(path == "path/to/candidates.csv" for _, path in inputs)

    def test_multiple_csv_flags_all_collected(self):
        result, mock_cls = _invoke("--csv", "a.csv", "--csv", "b.csv")
        inputs = mock_cls.return_value.run.call_args[1]["inputs"]
        csv_paths = [path for src, path in inputs if src.value == "csv"]
        assert sorted(csv_paths) == ["a.csv", "b.csv"]

    def test_mixed_flags_all_three_sources_present(self):
        result, mock_cls = _invoke(
            "--csv", "a.csv", "--json", "b.json", "--pdf", "c.pdf"
        )
        inputs = mock_cls.return_value.run.call_args[1]["inputs"]
        sources = {src.value for src, _ in inputs}
        assert sources == {"csv", "json", "resume_pdf"}

    def test_mixed_flags_total_input_count(self):
        result, mock_cls = _invoke(
            "--csv", "a.csv", "--json", "b.json", "--pdf", "c.pdf"
        )
        inputs = mock_cls.return_value.run.call_args[1]["inputs"]
        assert len(inputs) == 3


# ---------------------------------------------------------------------------
# --output flag
# ---------------------------------------------------------------------------


class TestOutputFlag:
    def test_output_flag_forwarded_to_pipeline_run(self):
        result, mock_cls = _invoke("--csv", "data.csv", "--output", "out/profiles.json")
        call_kwargs = mock_cls.return_value.run.call_args[1]
        assert call_kwargs["output_path"] == "out/profiles.json"

    def test_output_shorthand_accepted(self):
        result, mock_cls = _invoke("--csv", "data.csv", "-o", "profiles.json")
        call_kwargs = mock_cls.return_value.run.call_args[1]
        assert call_kwargs["output_path"] == "profiles.json"

    def test_no_output_flag_passes_none(self):
        result, mock_cls = _invoke("--csv", "data.csv")
        call_kwargs = mock_cls.return_value.run.call_args[1]
        assert call_kwargs["output_path"] is None

    def test_output_path_echoed_in_stdout(self):
        cli_result, _ = _invoke("--csv", "data.csv", "--output", "result.json")
        assert "result.json" in cli_result.output


# ---------------------------------------------------------------------------
# Exit codes and output messages
# ---------------------------------------------------------------------------


class TestExitCodesAndMessages:
    def test_clean_run_exits_zero(self):
        cli_result, _ = _invoke("--csv", "data.csv")
        assert cli_result.exit_code == 0

    def test_done_printed_on_success(self):
        cli_result, _ = _invoke("--csv", "data.csv")
        assert "Done." in cli_result.output

    def test_processing_message_printed(self):
        cli_result, _ = _invoke("--csv", "data.csv")
        assert "Processing" in cli_result.output

    def test_total_inputs_in_output(self):
        cli_result, _ = _invoke("--csv", "data.csv", pipeline_result=_result(total_inputs=3))
        assert "3" in cli_result.output

    def test_total_groups_in_output(self):
        cli_result, _ = _invoke("--csv", "data.csv", pipeline_result=_result(total_groups=2))
        assert "2" in cli_result.output

    def test_profiles_count_in_output(self):
        cli_result, _ = _invoke(
            "--csv", "data.csv",
            pipeline_result=_result(n_profiles=4, total_inputs=4, total_groups=4),
        )
        assert "4" in cli_result.output

    def test_pipeline_errors_exit_nonzero(self):
        cli_result, _ = _invoke(
            "--csv", "data.csv",
            pipeline_result=_result(errors=["something broke"]),
        )
        assert cli_result.exit_code != 0

    def test_pipeline_errors_printed_to_stderr(self):
        cli_result, _ = _invoke(
            "--csv", "data.csv",
            pipeline_result=_result(errors=["disk failure"]),
        )
        assert "disk failure" in (cli_result.output or "")

    def test_multiple_errors_all_printed(self):
        cli_result, _ = _invoke(
            "--csv", "data.csv",
            pipeline_result=_result(errors=["err1", "err2"]),
        )
        assert "err1" in (cli_result.output or "")
        assert "err2" in (cli_result.output or "")


# ---------------------------------------------------------------------------
# --verbose and --config flags
# ---------------------------------------------------------------------------


class TestVerboseAndConfig:
    def test_verbose_flag_accepted_no_crash(self):
        cli_result, _ = _invoke("--csv", "data.csv", "--verbose")
        assert cli_result.exit_code == 0

    def test_verbose_shorthand_accepted(self):
        cli_result, _ = _invoke("--csv", "data.csv", "-v")
        assert cli_result.exit_code == 0

    def test_config_flag_triggers_custom_config(self):
        """--config should invoke Pipeline with a custom PipelineConfig."""
        runner = CliRunner()
        pipeline_result = _result()

        with patch("app.cli.Pipeline") as mock_cls, \
             patch("app.cli.get_config") as mock_get_config, \
             patch("app.cli.reset_config_cache"):
            mock_get_config.return_value = MagicMock()
            mock_cls.return_value.run.return_value = pipeline_result
            runner.invoke(main_cli, ["--csv", "data.csv", "--config", "custom.yaml"])

        mock_get_config.assert_called_once_with("custom.yaml")

    def test_config_shorthand_accepted(self):
        runner = CliRunner()
        pipeline_result = _result()

        with patch("app.cli.Pipeline") as mock_cls, \
             patch("app.cli.get_config") as mock_get_config, \
             patch("app.cli.reset_config_cache"):
            mock_get_config.return_value = MagicMock()
            mock_cls.return_value.run.return_value = pipeline_result
            result = runner.invoke(main_cli, ["--csv", "data.csv", "-c", "custom.yaml"])

        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# --help and --version
# ---------------------------------------------------------------------------


class TestHelpAndVersion:
    def test_help_exits_zero(self):
        runner = CliRunner()
        result = runner.invoke(main_cli, ["--help"])
        assert result.exit_code == 0

    def test_help_output_mentions_csv(self):
        runner = CliRunner()
        result = runner.invoke(main_cli, ["--help"])
        assert "--csv" in result.output

    def test_help_output_mentions_output(self):
        runner = CliRunner()
        result = runner.invoke(main_cli, ["--help"])
        assert "--output" in result.output or "-o" in result.output

    def test_version_flag_exits_zero(self):
        runner = CliRunner()
        result = runner.invoke(main_cli, ["--version"])
        assert result.exit_code == 0

    def test_version_output_contains_version(self):
        runner = CliRunner()
        result = runner.invoke(main_cli, ["--version"])
        assert "1.0.0" in result.output


# ---------------------------------------------------------------------------
# main.py entry point
# ---------------------------------------------------------------------------


class TestMainPy:
    def test_main_py_exists(self):
        assert Path("main.py").exists()

    def test_main_py_imports_main_cli(self):
        content = Path("main.py").read_text(encoding="utf-8")
        assert "main_cli" in content

    def test_main_py_has_name_guard(self):
        content = Path("main.py").read_text(encoding="utf-8")
        assert '__name__' in content and '__main__' in content
