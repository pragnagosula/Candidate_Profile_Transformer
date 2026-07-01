"""YAML configuration loader with Pydantic validation.

Usage:
    from app.config.loader import get_config
    cfg = get_config()          # returns singleton PipelineConfig
    cfg = get_config("custom/pipeline.yaml")   # override path
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml
from pydantic import ValidationError

from app.config.models import (
    MergeRulesConfig,
    NormalizationConfig,
    PipelineConfig,
    ProjectionConfig,
    SourceReliabilityConfig,
    EntityResolutionConfig,
)
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

_DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
_DEFAULT_PIPELINE_YAML = _DEFAULT_CONFIG_DIR / "pipeline.yaml"
_PROJECTION_KEYS = {"fields", "include_confidence", "include_provenance", "include_validation"}


def _load_yaml(path: Path) -> dict:
    """Read a YAML file and return its content as a dict.

    Returns an empty dict if the file does not exist.
    """
    if not path.exists():
        logger.warning("Config file not found, using defaults: %s", path)
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except yaml.YAMLError as exc:
        logger.error("Failed to parse YAML at %s: %s", path, exc)
        raise


def _build_config(pipeline_path: Path) -> PipelineConfig:
    """Load and merge all config YAML files into a single PipelineConfig.

    Each sub-config file can override defaults independently.  Missing
    files are silently ignored (defaults apply).

    Args:
        pipeline_path: Path to the main pipeline.yaml file.

    Returns:
        Validated :class:`PipelineConfig` instance.

    Raises:
        ValidationError: If any config value violates its schema.
    """
    config_dir = pipeline_path.parent
    pipeline_raw = _load_yaml(pipeline_path)

    # Allow a standalone projection YAML file to be passed as the config path.
    # In that case, the projection keys live at the top level instead of under
    # a nested ``projection`` block.
    projection_raw: dict = {}
    if isinstance(pipeline_raw.get("projection"), dict):
        projection_raw = dict(pipeline_raw.pop("projection"))
    else:
        projection_raw = {key: pipeline_raw[key] for key in _PROJECTION_KEYS if key in pipeline_raw}

    for key in _PROJECTION_KEYS:
        pipeline_raw.pop(key, None)

    merge_raw = _load_yaml(config_dir / "merge_rules.yaml")
    norm_raw = _load_yaml(config_dir / "normalization.yaml")
    proj_raw = projection_raw or _load_yaml(config_dir / "projection.yaml")
    reliability_raw = _load_yaml(config_dir / "source_reliability.yaml")
    entity_raw = _load_yaml(config_dir / "entity_resolution.yaml")

    try:
        source_reliability = SourceReliabilityConfig(**reliability_raw) if reliability_raw else SourceReliabilityConfig()
        merge_rules = MergeRulesConfig(**merge_raw) if merge_raw else MergeRulesConfig()
        normalization = NormalizationConfig(**norm_raw) if norm_raw else NormalizationConfig()
        projection = ProjectionConfig(**proj_raw) if proj_raw else ProjectionConfig()
        entity_resolution = EntityResolutionConfig(**entity_raw) if entity_raw else EntityResolutionConfig()
    except ValidationError as exc:
        logger.error("Configuration validation failed:\n%s", exc)
        raise

    merged = {
        **pipeline_raw,
        "source_reliability": source_reliability,
        "merge_rules": merge_rules,
        "normalization": normalization,
        "projection": projection,
        "entity_resolution": entity_resolution,
    }

    try:
        config = PipelineConfig(**merged)
    except ValidationError as exc:
        logger.error("Pipeline config validation failed:\n%s", exc)
        raise

    logger.info(
        "Configuration loaded (version=%s, log_level=%s)",
        config.version,
        config.log_level,
    )
    return config


@lru_cache(maxsize=1)
def _cached_config(pipeline_path: str) -> PipelineConfig:
    """Internal cached loader keyed by path string."""
    return _build_config(Path(pipeline_path))


def get_config(pipeline_yaml: Optional[str] = None) -> PipelineConfig:
    """Return the singleton PipelineConfig, loading it on first call.

    Args:
        pipeline_yaml: Optional path to pipeline.yaml.  Defaults to
            ``config/pipeline.yaml`` at the project root.

    Returns:
        Validated :class:`PipelineConfig`.
    """
    path = pipeline_yaml or str(_DEFAULT_PIPELINE_YAML)
    return _cached_config(path)


def reset_config_cache() -> None:
    """Clear the config singleton — useful in tests that need fresh config."""
    _cached_config.cache_clear()
