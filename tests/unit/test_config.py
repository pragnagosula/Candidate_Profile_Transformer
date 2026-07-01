"""Unit tests for the configuration system."""

import textwrap
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from app.config.loader import get_config, reset_config_cache
from app.config.models import (
    EntityResolutionConfig,
    FieldMergeRule,
    MergeRulesConfig,
    NormalizationConfig,
    PipelineConfig,
    ProjectionConfig,
    SourceReliabilityConfig,
)
from app.models.candidate import DataSource


# ---------------------------------------------------------------------------
# SourceReliabilityConfig
# ---------------------------------------------------------------------------


class TestSourceReliabilityConfig:
    def test_defaults_present(self):
        cfg = SourceReliabilityConfig()
        assert "csv" in cfg.weights or DataSource.CSV in cfg.weights

    def test_weight_out_of_bounds_raises(self):
        with pytest.raises(ValidationError):
            SourceReliabilityConfig(weights={"csv": 1.5})

    def test_get_returns_default_for_unknown_source(self):
        cfg = SourceReliabilityConfig(weights={"csv": 0.7})
        assert cfg.get("nonexistent", 0.42) == 0.42

    def test_get_returns_weight_for_known_source(self):
        cfg = SourceReliabilityConfig(weights={"csv": 0.7})
        assert cfg.get("csv") == 0.7


# ---------------------------------------------------------------------------
# FieldMergeRule
# ---------------------------------------------------------------------------


class TestFieldMergeRule:
    def test_invalid_strategy_raises(self):
        with pytest.raises(ValidationError):
            FieldMergeRule(priority=["csv"], strategy="magic")

    def test_valid_strategies(self):
        for strategy in ("priority", "most_complete", "union"):
            rule = FieldMergeRule(priority=["csv"], strategy=strategy)
            assert rule.strategy == strategy


# ---------------------------------------------------------------------------
# MergeRulesConfig
# ---------------------------------------------------------------------------


class TestMergeRulesConfig:
    def test_get_rule_returns_field_rule(self):
        cfg = MergeRulesConfig(
            field_rules={
                "email": FieldMergeRule(priority=["csv", "json"], strategy="priority")
            }
        )
        rule = cfg.get_rule("email")
        assert rule.priority[0] == "csv"

    def test_get_rule_falls_back_to_default(self):
        cfg = MergeRulesConfig()
        rule = cfg.get_rule("nonexistent_field")
        assert rule.strategy == cfg.default_strategy


# ---------------------------------------------------------------------------
# PipelineConfig
# ---------------------------------------------------------------------------


class TestPipelineConfig:
    def test_default_log_level_is_info(self):
        cfg = PipelineConfig()
        assert cfg.log_level == "INFO"

    def test_log_level_is_uppercased(self):
        cfg = PipelineConfig(log_level="debug")
        assert cfg.log_level == "DEBUG"

    def test_invalid_log_level_raises(self):
        with pytest.raises(ValidationError):
            PipelineConfig(log_level="VERBOSE")


# ---------------------------------------------------------------------------
# ProjectionConfig
# ---------------------------------------------------------------------------


class TestProjectionConfig:
    def test_output_field_map_excludes_disabled_fields(self):
        from app.config.models import ProjectedField

        cfg = ProjectionConfig(
            fields=[
                ProjectedField(source="name", output_name="full_name", include=True),
                ProjectedField(source="email", output_name="email_address", include=False),
            ]
        )
        field_map = cfg.output_field_map()
        assert "name" in field_map
        assert "email" not in field_map

    def test_output_field_map_renames_correctly(self):
        from app.config.models import ProjectedField

        cfg = ProjectionConfig(
            fields=[
                ProjectedField(source="phone", output_name="phone_number", include=True),
            ]
        )
        assert cfg.output_field_map()["phone"] == "phone_number"


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


class TestConfigLoader:
    def setup_method(self):
        reset_config_cache()

    def teardown_method(self):
        reset_config_cache()

    def test_loads_default_config(self, tmp_path):
        pipeline_yaml = tmp_path / "pipeline.yaml"
        pipeline_yaml.write_text("version: '2.0.0'\nlog_level: WARNING\n")
        cfg = get_config(str(pipeline_yaml))
        assert cfg.version == "2.0.0"
        assert cfg.log_level == "WARNING"

    def test_missing_pipeline_yaml_uses_defaults(self, tmp_path):
        missing = tmp_path / "does_not_exist.yaml"
        cfg = get_config(str(missing))
        assert cfg.version == "1.0.0"

    def test_returns_same_singleton(self, tmp_path):
        pipeline_yaml = tmp_path / "pipeline.yaml"
        pipeline_yaml.write_text("")
        cfg1 = get_config(str(pipeline_yaml))
        cfg2 = get_config(str(pipeline_yaml))
        assert cfg1 is cfg2

    def test_reset_clears_singleton(self, tmp_path):
        pipeline_yaml = tmp_path / "pipeline.yaml"
        pipeline_yaml.write_text("version: '1.0.0'\n")
        cfg1 = get_config(str(pipeline_yaml))
        reset_config_cache()
        pipeline_yaml.write_text("version: '9.9.9'\n")
        cfg2 = get_config(str(pipeline_yaml))
        assert cfg2.version == "9.9.9"

    def test_bad_yaml_raises(self, tmp_path):
        pipeline_yaml = tmp_path / "pipeline.yaml"
        pipeline_yaml.write_text(": broken: yaml: [")
        with pytest.raises(Exception):
            get_config(str(pipeline_yaml))

    def test_invalid_config_value_raises_validation_error(self, tmp_path):
        pipeline_yaml = tmp_path / "pipeline.yaml"
        pipeline_yaml.write_text("log_level: NONSENSE\n")
        with pytest.raises(ValidationError):
            get_config(str(pipeline_yaml))
