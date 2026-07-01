"""Core domain models for the candidate data pipeline.

All layers in the pipeline communicate exclusively through these types.
Pydantic enforces field contracts and provides free serialization.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, EmailStr, Field, field_validator


class DataSource(str, Enum):
    """Enumeration of all recognised input sources."""

    CSV = "csv"
    JSON = "json"
    RESUME_PDF = "resume_pdf"
    RESUME_TXT = "resume_txt"
    LINKEDIN = "linkedin"
    GITHUB = "github"
    ATS = "ats"
    UNKNOWN = "unknown"


class ExtractionMethod(str, Enum):
    """How a field value was obtained from the raw source."""

    DIRECT = "direct"        # Explicitly present in structured data
    REGEX = "regex"          # Extracted via regular expression
    NLP = "nlp"              # Extracted via NLP / text analysis
    INFERRED = "inferred"    # Derived from other fields
    DEFAULT = "default"      # System-applied default


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class DateRange(BaseModel):
    """An inclusive date range, e.g. employment tenure."""

    start: Optional[str] = None   # ISO 8601 partial date: YYYY-MM or YYYY-MM-DD
    end: Optional[str] = None     # None means "present"
    is_current: bool = False


class Experience(BaseModel):
    """A single employment record."""

    company: Optional[str] = None
    title: Optional[str] = None
    duration: Optional[DateRange] = None
    description: Optional[str] = None
    location: Optional[str] = None


class Education(BaseModel):
    """A single academic record."""

    institution: Optional[str] = None
    degree: Optional[str] = None
    field_of_study: Optional[str] = None
    duration: Optional[DateRange] = None
    gpa: Optional[float] = None


class Link(BaseModel):
    """A URL with optional label, e.g. GitHub, portfolio."""

    url: str
    label: Optional[str] = None


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


class ProvenanceEntry(BaseModel):
    """Full lineage record for a single field value."""

    field_name: str
    source: DataSource
    original_value: Any
    normalized_value: Any
    extraction_method: ExtractionMethod
    confidence: float = Field(ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    notes: Optional[str] = None


# ---------------------------------------------------------------------------
# Raw data (parser output) — loosely typed, exactly what the source provides
# ---------------------------------------------------------------------------


class RawCandidateData(BaseModel):
    """Unprocessed candidate data as delivered by a parser.

    Fields are all Optional[str] or loose types because raw sources are
    unreliable.  Downstream layers are responsible for typing and cleaning.
    """

    source: DataSource
    raw_fields: dict[str, Any] = Field(default_factory=dict)
    source_file: Optional[str] = None
    parse_timestamp: datetime = Field(default_factory=datetime.utcnow)
    parse_errors: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Extracted candidate — field-level structured data after extractor pass
# ---------------------------------------------------------------------------


class ExtractedCandidate(BaseModel):
    """Typed but un-normalised candidate fields after extraction."""

    source: DataSource
    source_file: Optional[str] = None

    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    summary: Optional[str] = None

    skills: list[str] = Field(default_factory=list)
    experience: list[Experience] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    links: list[Link] = Field(default_factory=list)

    raw_text: Optional[str] = None   # Full text from unstructured sources
    extra_fields: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Normalised candidate — clean, canonical values ready for merging
# ---------------------------------------------------------------------------


class NormalizedCandidate(BaseModel):
    """Candidate with all fields normalised to canonical form."""

    source: DataSource
    source_file: Optional[str] = None

    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    summary: Optional[str] = None

    skills: list[str] = Field(default_factory=list)
    experience: list[Experience] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    links: list[Link] = Field(default_factory=list)

    extra_fields: dict[str, Any] = Field(default_factory=dict)

    # Improvement 6: overall parser/OCR confidence for this record (0–1).
    # None means "not reported"; treated as 1.0 (no penalty).
    extraction_confidence: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Parser or OCR confidence for this entire record.",
    )
    # Improvement 5: when the source data was last updated (for recency scoring).
    source_timestamp: Optional[datetime] = Field(
        default=None,
        description="Last-updated timestamp of the source record.",
    )

    @field_validator("email", mode="before")
    @classmethod
    def lowercase_email(cls, v: Any) -> Any:
        return v.lower().strip() if isinstance(v, str) else v


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------


class FieldConfidence(BaseModel):
    """Confidence metadata for a single merged field."""

    field_name: str
    score: float = Field(ge=0.0, le=1.0)
    contributing_sources: list[DataSource] = Field(default_factory=list)
    reason: Optional[str] = None
    # Improvements 2–4: minimum pairwise fuzzy similarity across contributing
    # source values.  None when there is only one source or the field is a list.
    similarity: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class ConfidenceReport(BaseModel):
    """Aggregate confidence for a merged candidate profile."""

    overall_score: float = Field(ge=0.0, le=1.0)
    field_scores: list[FieldConfidence] = Field(default_factory=list)
    completeness: float = Field(ge=0.0, le=1.0, description="Weighted fraction of key fields present")
    source_agreement: float = Field(ge=0.0, le=1.0, description="How much sources agree")
    # Improvement 9: human-readable explanation of every bonus and deduction
    explanations: list[str] = Field(default_factory=list)
    # Improvement 10: independently inspectable breakdown components
    diversity_bonus: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Source diversity bonus applied to the overall score.",
    )
    validation_penalty: float = Field(
        default=0.0, ge=0.0,
        description="Total validation penalty deducted from the overall score.",
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class ValidationIssue(BaseModel):
    """A single validation finding (error or warning)."""

    field: str
    severity: str  # "error" | "warning"
    message: str
    value: Any = None


class ValidationResult(BaseModel):
    """Outcome of running a candidate through the validation layer."""

    is_valid: bool
    issues: list[ValidationIssue] = Field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def info(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "info"]


# ---------------------------------------------------------------------------
# Merged / final candidate
# ---------------------------------------------------------------------------


class MergedCandidate(BaseModel):
    """The result of merging one or more NormalizedCandidates.

    This is the canonical output before projection.
    """

    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    summary: Optional[str] = None

    skills: list[str] = Field(default_factory=list)
    experience: list[Experience] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    links: list[Link] = Field(default_factory=list)

    extra_fields: dict[str, Any] = Field(default_factory=dict)

    source_records: list[NormalizedCandidate] = Field(default_factory=list)
    confidence: Optional[ConfidenceReport] = None
    provenance: list[ProvenanceEntry] = Field(default_factory=list)
    validation: Optional[ValidationResult] = None

    merge_timestamp: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Final output model
# ---------------------------------------------------------------------------


class CandidateProfile(BaseModel):
    """The fully projected, validated, schema-checked final output.

    This is what gets written to the output JSON file.
    """

    model_config = {"populate_by_name": True}

    fields: dict[str, Any] = Field(
        description="Projected output fields; shape controlled by projection config."
    )
    confidence: Optional[ConfidenceReport] = None
    provenance: list[ProvenanceEntry] = Field(default_factory=list)
    validation: Optional[ValidationResult] = None
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    pipeline_version: str = "1.0.0"
