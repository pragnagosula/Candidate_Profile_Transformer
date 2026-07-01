"""Skill normalizer — synonym resolution, deduplication, fuzzy near-dedup.

Three-pass algorithm:
    Pass 1  Synonym resolution — "ML" → "Machine Learning" using config table.
    Pass 2  Exact deduplication (case-insensitive).
    Pass 3  Fuzzy near-dedup — removes skills that are >threshold similar to
            an already-accepted skill (e.g. "machine learning" vs "Machine Learning"
            that survived synonym resolution).

Config-driven: synonyms come from normalization.yaml, not hardcoded here.
"""

from __future__ import annotations

from app.config.models import NormalizationConfig, SkillNormalizationConfig
from app.normalizers.base import BaseFieldNormalizer
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

_FUZZY_THRESHOLD = 85  # RapidFuzz score 0–100


def _build_synonym_index(skill_cfg: SkillNormalizationConfig) -> dict[str, str]:
    """Build a lowercased alias → canonical name lookup from the config.

    Returns:
        Dict mapping any alias (lowercased) → canonical skill name.
    """
    index: dict[str, str] = {}
    for canonical, aliases in skill_cfg.synonyms.items():
        # The canonical name also maps to itself
        index[canonical.lower()] = canonical
        for alias in aliases:
            index[alias.lower()] = canonical
    return index


def _resolve_synonyms(skills: list[str], index: dict[str, str]) -> list[str]:
    return [index.get(s.lower(), s) for s in skills]


def _exact_dedup(skills: list[str], case_sensitive: bool) -> list[str]:
    """Remove exact duplicates, preserving order of first occurrence."""
    seen: set[str] = set()
    result: list[str] = []
    for skill in skills:
        key = skill if case_sensitive else skill.lower()
        if key not in seen:
            seen.add(key)
            result.append(skill)
    return result


def _fuzzy_dedup(skills: list[str], threshold: int) -> list[str]:
    """Remove near-duplicate skills using rapidfuzz token sort ratio."""
    try:
        from rapidfuzz import fuzz
    except ImportError:
        logger.debug("rapidfuzz not available; skipping fuzzy dedup")
        return skills

    accepted: list[str] = []
    for skill in skills:
        is_duplicate = any(
            fuzz.token_sort_ratio(skill.lower(), accepted_skill.lower()) >= threshold
            for accepted_skill in accepted
        )
        if not is_duplicate:
            accepted.append(skill)
        else:
            logger.debug("SkillNormalizer: fuzzy-dedup removed '%s'", skill)
    return accepted


class SkillNormalizer(BaseFieldNormalizer[list[str]]):
    """Normalise a list of skill strings."""

    def normalize(self, value: list[str], config: NormalizationConfig) -> list[str]:
        if not value:
            return value

        skill_cfg = config.skills
        synonym_index = _build_synonym_index(skill_cfg)

        # Pass 1: synonym resolution
        resolved = _resolve_synonyms(value, synonym_index)

        # Pass 2: exact dedup
        deduped = _exact_dedup(resolved, skill_cfg.case_sensitive)

        # Pass 3: fuzzy dedup
        final = _fuzzy_dedup(deduped, _FUZZY_THRESHOLD)

        if len(final) < len(value):
            logger.debug(
                "SkillNormalizer: %d skills -> %d after normalisation", len(value), len(final)
            )
        return final
