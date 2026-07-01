"""Pydantic models that validate the YAML configuration files.

Config is validated once at startup.  If any field is wrong the process
fails immediately with a clear error — not silently at runtime.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from app.models.candidate import DataSource


# ---------------------------------------------------------------------------
# Source reliability config
# ---------------------------------------------------------------------------


class SourceReliabilityConfig(BaseModel):
    """Reliability weight (0.0–1.0) for each data source.

    Higher weight means the source's values are preferred during merge
    and contribute more to confidence scoring.
    """

    weights: dict[str, float] = Field(
        default_factory=lambda: {
            DataSource.CSV: 0.70,
            DataSource.JSON: 0.75,
            DataSource.RESUME_PDF: 0.85,
            DataSource.LINKEDIN: 0.90,
            DataSource.GITHUB: 0.80,
            DataSource.ATS: 0.95,
        }
    )

    @field_validator("weights")
    @classmethod
    def weights_in_bounds(cls, v: dict[str, float]) -> dict[str, float]:
        for source, weight in v.items():
            if not 0.0 <= weight <= 1.0:
                raise ValueError(f"Source weight for '{source}' must be between 0 and 1, got {weight}")
        return v

    def get(self, source: DataSource | str, default: float = 0.5) -> float:
        return self.weights.get(str(source), self.weights.get(source.value if hasattr(source, "value") else source, default))


# ---------------------------------------------------------------------------
# Merge rules config
# ---------------------------------------------------------------------------


class FieldMergeRule(BaseModel):
    """Priority order and conflict strategy for a single output field."""

    priority: list[str] = Field(
        description="Source names in descending priority (first = highest)."
    )
    strategy: str = Field(
        default="priority",
        description="'priority' | 'most_complete' | 'union' (for list fields)",
    )

    @field_validator("strategy")
    @classmethod
    def valid_strategy(cls, v: str) -> str:
        allowed = {"priority", "most_complete", "union"}
        if v not in allowed:
            raise ValueError(f"strategy must be one of {allowed}, got '{v}'")
        return v


class MergeRulesConfig(BaseModel):
    """Per-field merge rules loaded from merge_rules.yaml."""

    field_rules: dict[str, FieldMergeRule] = Field(default_factory=dict)
    default_priority: list[str] = Field(
        default_factory=lambda: [
            DataSource.ATS,
            DataSource.RESUME_PDF,
            DataSource.LINKEDIN,
            DataSource.JSON,
            DataSource.CSV,
        ]
    )
    default_strategy: str = "priority"

    def get_rule(self, field_name: str) -> FieldMergeRule:
        """Return the merge rule for a field, falling back to the default."""
        if field_name in self.field_rules:
            return self.field_rules[field_name]
        return FieldMergeRule(
            priority=self.default_priority,
            strategy=self.default_strategy,
        )


# ---------------------------------------------------------------------------
# Normalization config
# ---------------------------------------------------------------------------


class PhoneNormalizationConfig(BaseModel):
    default_country_code: str = "IN"
    output_format: str = "E164"   # E164 | NATIONAL | INTERNATIONAL


class DateNormalizationConfig(BaseModel):
    output_format: str = "YYYY-MM"
    assume_day: int = 1


class SkillNormalizationConfig(BaseModel):
    """Synonym groups: canonical name → list of aliases."""

    synonyms: dict[str, list[str]] = Field(default_factory=dict)
    case_sensitive: bool = False


class NormalizationConfig(BaseModel):
    phone: PhoneNormalizationConfig = Field(default_factory=PhoneNormalizationConfig)
    dates: DateNormalizationConfig = Field(default_factory=DateNormalizationConfig)
    skills: SkillNormalizationConfig = Field(default_factory=SkillNormalizationConfig)
    strip_unicode: bool = False
    normalize_whitespace: bool = True


# ---------------------------------------------------------------------------
# Projection config
# ---------------------------------------------------------------------------


class ProjectedField(BaseModel):
    """A single field in the projected output."""

    source: str = Field(description="Internal MergedCandidate field name")
    output_name: str = Field(description="Name in the final JSON output")
    include: bool = True
    default: Any = None


class ProjectionConfig(BaseModel):
    """Controls which fields appear in the output and under what names."""

    fields: list[ProjectedField] = Field(default_factory=list)
    include_confidence: bool = True
    include_provenance: bool = False
    include_validation: bool = True

    def output_field_map(self) -> dict[str, str]:
        """Return a {source_field: output_name} mapping for included fields."""
        return {f.source: f.output_name for f in self.fields if f.include}


# ---------------------------------------------------------------------------
# Entity resolution config
# ---------------------------------------------------------------------------


class EntityResolutionConfig(BaseModel):
    """Thresholds for fuzzy matching during entity resolution."""

    name_similarity_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    email_exact_match: bool = True
    skill_similarity_threshold: float = Field(default=0.80, ge=0.0, le=1.0)
    company_similarity_threshold: float = Field(default=0.80, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Confidence engine config
# ---------------------------------------------------------------------------


class FieldSimilarityConfig(BaseModel):
    """Per-field agreement thresholds and behaviour flags.

    Improvement 3: field-specific thresholds replace the global pair so that
    email (requires exact match) and name (tolerates abbreviations) are not
    governed by the same numbers.
    """

    full_threshold: float = Field(
        default=0.95, ge=0.0, le=1.0,
        description="Similarity >= this → full agreement bonus.",
    )
    partial_threshold: float = Field(
        default=0.80, ge=0.0, le=1.0,
        description="Similarity in [partial, full) → partial bonus.",
    )
    skip_agreement: bool = Field(
        default=False,
        description=(
            "Skip agreement scoring for naturally divergent fields such as "
            "'summary' which differ between Resume and LinkedIn by design."
        ),
    )


class FreshnessDecayConfig(BaseModel):
    """Exponential freshness decay — Improvement 5.

    weight = max(min_freshness, 0.5 ** (days_old / half_life_days))

    Replaces the previous step-function (recent/mid/old buckets) with a
    continuous curve so that a source 181 days old is not penalised the
    same as one 364 days old.
    """

    half_life_days: float = Field(
        default=365.0, gt=0.0,
        description="Days after which a source's freshness multiplier halves.",
    )
    min_freshness: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Floor on the freshness multiplier (prevents extreme decay).",
    )
    staleness_fields: list[str] = Field(
        default_factory=lambda: ["phone", "location", "summary", "experience"],
        description=(
            "Only these fields are subject to freshness decay.  Stable "
            "information (name, education, skills) is never penalised."
        ),
    )


class SourceDiversityConfig(BaseModel):
    """Bonus awarded when multiple independent trusted sources confirm data.

    Improvement 8 (extended): independent of the adaptive-weights mechanism,
    this adds a configurable bonus to the overall score when several
    high-reliability sources all contribute.
    """

    enabled: bool = True
    max_bonus: float = Field(
        default=0.05, ge=0.0, le=1.0,
        description="Maximum bonus applied to the overall score.",
    )
    min_sources: int = Field(
        default=2, ge=1,
        description="Minimum qualifying sources needed for any diversity bonus.",
    )
    min_reliability: float = Field(
        default=0.75, ge=0.0, le=1.0,
        description=(
            "A source must exceed this reliability to count toward diversity.  "
            "Low-quality sources (scrapers, OCR) are excluded."
        ),
    )


class RecencyConfig(BaseModel):
    """Recency multipliers applied to source records based on timestamp age."""

    recent_months: int = Field(default=12, ge=1, description="Age threshold (months) for 'recent' sources")
    recent_weight: float = Field(default=1.0, ge=0.0, le=1.0)
    mid_months: int = Field(default=36, ge=1, description="Age threshold (months) for 'mid' sources")
    mid_weight: float = Field(default=0.9, ge=0.0, le=1.0)
    old_weight: float = Field(default=0.8, ge=0.0, le=1.0)


class AdaptiveOverallWeightsConfig(BaseModel):
    """Overall scoring weights that scale with number of contributing sources.

    Each triplet (field_avg, completeness, agreement) must sum to 1.0.
    """

    # 0–1 sources: agreement is meaningless, lean on field quality
    one_source_field: float = Field(default=0.70, ge=0.0, le=1.0)
    one_source_completeness: float = Field(default=0.30, ge=0.0, le=1.0)
    one_source_agreement: float = Field(default=0.00, ge=0.0, le=1.0)

    # 2–(many_threshold-1) sources
    two_source_field: float = Field(default=0.50, ge=0.0, le=1.0)
    two_source_completeness: float = Field(default=0.25, ge=0.0, le=1.0)
    two_source_agreement: float = Field(default=0.25, ge=0.0, le=1.0)

    # many_threshold+ sources: agreement is highly informative
    many_source_field: float = Field(default=0.45, ge=0.0, le=1.0)
    many_source_completeness: float = Field(default=0.20, ge=0.0, le=1.0)
    many_source_agreement: float = Field(default=0.35, ge=0.0, le=1.0)

    many_threshold: int = Field(default=5, ge=2, description="Number of sources at which 'many' weights kick in")

    @model_validator(mode="after")
    def _triplets_sum_to_one(self) -> "AdaptiveOverallWeightsConfig":
        pairs = [
            ("one_source",  (self.one_source_field,  self.one_source_completeness,  self.one_source_agreement)),
            ("two_source",  (self.two_source_field,  self.two_source_completeness,  self.two_source_agreement)),
            ("many_source", (self.many_source_field, self.many_source_completeness, self.many_source_agreement)),
        ]
        for label, triplet in pairs:
            total = round(sum(triplet), 6)
            if abs(total - 1.0) > 1e-4:
                raise ValueError(f"{label} weights must sum to 1.0, got {total:.6f}")
        return self


class FieldCategoryConfig(BaseModel):
    """Classifies candidate fields by importance for completeness and confidence.

    - required_fields:    Missing → warning, reduces completeness and confidence.
    - recommended_fields: Missing → warning, small confidence reduction.
    - optional_fields:    Missing → info only, no confidence penalty.
    """

    required_fields: list[str] = Field(
        default_factory=lambda: [
            "name", "email", "phone", "skills", "experience", "education",
        ],
        description="Core fields whose absence reduces completeness and confidence.",
    )
    recommended_fields: list[str] = Field(
        default_factory=lambda: ["links"],
        description=(
            "Fields that improve profile quality; missing triggers a small warning "
            "but no major confidence reduction."
        ),
    )
    optional_fields: list[str] = Field(
        default_factory=lambda: ["summary", "location"],
        description=(
            "Supplementary fields.  Missing generates an informational message only "
            "— no completeness deduction and no confidence penalty."
        ),
    )


class ConfidenceConfig(BaseModel):
    """All tunable parameters for the Confidence Engine.

    Centralises every 'magic number' so nothing is hard-coded in the engine.
    """

    # --- Field categories (required / recommended / optional) ---
    field_categories: FieldCategoryConfig = Field(
        default_factory=FieldCategoryConfig,
        description=(
            "Classifies fields by importance.  Only required/recommended fields "
            "contribute to completeness; optional fields carry zero weight."
        ),
    )

    # --- Improvement 1: per-field importance weights (must sum to 1.0) ---
    # Optional fields (summary, location) carry 0.0 weight so their absence
    # does not reduce the completeness score.
    field_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "name":       0.10,
            "email":      0.08,
            "phone":      0.07,
            "location":   0.00,
            "summary":    0.00,
            "skills":     0.20,
            "experience": 0.30,
            "education":  0.20,
            "links":      0.05,
        },
        description="Importance weight per field; values must sum to 1.0.",
    )

    # --- Improvements 2–4: fuzzy agreement thresholds and bonuses/penalties ---
    similarity_full_threshold: float = Field(
        default=0.95, ge=0.0, le=1.0,
        description="Similarity >= this → full agreement, full bonus.",
    )
    similarity_partial_threshold: float = Field(
        default=0.80, ge=0.0, le=1.0,
        description="Similarity in [partial, full) → partial agreement, scaled bonus.",
    )
    max_agreement_bonus: float = Field(
        default=0.05, ge=0.0, le=1.0,
        description="Maximum bonus added when all sources fully agree on a field.",
    )
    conflict_penalty: float = Field(
        default=0.10, ge=0.0, le=1.0,
        description="Score deducted from a field when sources strongly disagree.",
    )

    # --- Validation penalty parameters ---
    error_penalty: float = Field(default=0.05, ge=0.0, le=1.0)
    warning_penalty: float = Field(default=0.01, ge=0.0, le=1.0)
    max_error_penalty: float = Field(default=0.20, ge=0.0, le=1.0)
    max_warning_penalty: float = Field(default=0.10, ge=0.0, le=1.0)

    # --- Improvements 2 & 3: per-field similarity thresholds ---
    field_similarity: dict[str, FieldSimilarityConfig] = Field(
        default_factory=lambda: {
            "name":     FieldSimilarityConfig(full_threshold=0.90, partial_threshold=0.70),
            "email":    FieldSimilarityConfig(full_threshold=1.00, partial_threshold=0.95),
            "phone":    FieldSimilarityConfig(full_threshold=1.00, partial_threshold=0.90),
            "location": FieldSimilarityConfig(full_threshold=0.85, partial_threshold=0.65),
            "summary":  FieldSimilarityConfig(skip_agreement=True),
        },
        description="Per-field agreement thresholds; falls back to global thresholds if absent.",
    )

    # --- Improvement 5: exponential freshness decay (replaces step-function recency) ---
    recency: RecencyConfig = Field(
        default_factory=RecencyConfig,
        description="Legacy step-function recency; kept for backward compat with YAML configs.",
    )
    freshness: FreshnessDecayConfig = Field(
        default_factory=FreshnessDecayConfig,
        description="Exponential freshness decay (takes precedence over 'recency').",
    )

    # --- Improvement 8 (extended): source diversity bonus ---
    diversity: SourceDiversityConfig = Field(default_factory=SourceDiversityConfig)

    # --- Improvement 8: adaptive overall weights ---
    adaptive_weights: AdaptiveOverallWeightsConfig = Field(
        default_factory=AdaptiveOverallWeightsConfig
    )

    # --- Improvement 11: location alias dictionary ---
    location_aliases: dict[str, str] = Field(
        default_factory=lambda: {
            "bengaluru": "bangalore",
            "bombay":    "mumbai",
            "madras":    "chennai",
            "calcutta":  "kolkata",
            "nyc":       "new york city",
            "ny":        "new york",
            "la":        "los angeles",
            "sf":        "san francisco",
            "dc":        "washington dc",
            "uk":        "united kingdom",
        },
        description="Alias → canonical token for location agreement comparison.",
    )

    @field_validator("field_weights")
    @classmethod
    def _field_weights_sum_to_one(cls, v: dict[str, float]) -> dict[str, float]:
        total = round(sum(v.values()), 6)
        if abs(total - 1.0) > 1e-4:
            raise ValueError(f"field_weights must sum to 1.0, got {total:.6f}")
        return v


# ---------------------------------------------------------------------------
# Root pipeline config
# ---------------------------------------------------------------------------


class PipelineConfig(BaseModel):
    """Root configuration object.  Aggregates all sub-configs."""

    version: str = "1.0.0"
    log_level: str = "INFO"

    source_reliability: SourceReliabilityConfig = Field(
        default_factory=SourceReliabilityConfig
    )
    merge_rules: MergeRulesConfig = Field(default_factory=MergeRulesConfig)
    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)
    projection: ProjectionConfig = Field(default_factory=ProjectionConfig)
    entity_resolution: EntityResolutionConfig = Field(
        default_factory=EntityResolutionConfig
    )
    confidence: ConfidenceConfig = Field(
        default_factory=ConfidenceConfig,
        description="All tunable parameters for the Confidence Engine.",
    )

    output_dir: str = "output"
    output_filename: str = "candidate_profile.json"

    @field_validator("log_level")
    @classmethod
    def valid_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}, got '{v}'")
        return upper
