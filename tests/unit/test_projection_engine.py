"""Unit tests for the Projection Engine.

Coverage targets:
  - Output keys use output_name (e.g. "full_name") not source name ("name")
  - Excluded fields (include=false) are absent from output
  - Default applied when field value is None
  - Pydantic sub-models serialised to plain dicts
  - Skills list passed through as-is
  - include_confidence / include_provenance / include_validation flags
  - generated_at and pipeline_version set on every profile
  - Custom ProjectionConfig injection
  - Empty merged candidate → defaults applied throughout
"""

from __future__ import annotations

import pytest

from app.config.models import ProjectedField, ProjectionConfig
from app.models.candidate import (
    CandidateProfile,
    ConfidenceReport,
    DataSource,
    Education,
    Experience,
    FieldConfidence,
    Link,
    MergedCandidate,
    NormalizedCandidate,
    ProvenanceEntry,
    ValidationIssue,
    ValidationResult,
)
from app.projection.engine import ProjectionEngine, _serialize


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _merged(
    name: str | None = None,
    email: str | None = None,
    phone: str | None = None,
    location: str | None = None,
    summary: str | None = None,
    skills: list[str] | None = None,
    experience: list[Experience] | None = None,
    education: list[Education] | None = None,
    links: list[Link] | None = None,
) -> MergedCandidate:
    return MergedCandidate(
        name=name,
        email=email,
        phone=phone,
        location=location,
        summary=summary,
        skills=skills or [],
        experience=experience or [],
        education=education or [],
        links=links or [],
    )


def _confidence() -> ConfidenceReport:
    return ConfidenceReport(
        overall_score=0.85,
        completeness=0.75,
        source_agreement=1.0,
        field_scores=[FieldConfidence(field_name="name", score=0.85, reason="test")],
    )


def _provenance() -> list[ProvenanceEntry]:
    return [
        ProvenanceEntry(
            field_name="name",
            source=DataSource.CSV,
            original_value="Alice",
            normalized_value="Alice",
            extraction_method="direct",
            confidence=0.70,
        )
    ]


def _validation(valid: bool = True) -> ValidationResult:
    return ValidationResult(
        is_valid=valid,
        issues=[] if valid else [
            ValidationIssue(field="email", severity="error", message="bad email")
        ],
    )


def _default_engine() -> ProjectionEngine:
    """Engine with the default projection.yaml field mappings."""
    return ProjectionEngine()


def _engine_with(
    fields: list[dict],
    include_confidence: bool = True,
    include_provenance: bool = False,
    include_validation: bool = True,
) -> ProjectionEngine:
    cfg = ProjectionConfig(
        fields=[ProjectedField(**f) for f in fields],
        include_confidence=include_confidence,
        include_provenance=include_provenance,
        include_validation=include_validation,
    )
    return ProjectionEngine(config=cfg)


# ---------------------------------------------------------------------------
# Field renaming and inclusion
# ---------------------------------------------------------------------------


class TestFieldRenaming:
    def test_name_renamed_to_full_name(self):
        m = _merged(name="Alice Smith")
        profile = _default_engine().project(m)
        assert "full_name" in profile.fields
        assert "name" not in profile.fields

    def test_email_renamed_to_email_address(self):
        m = _merged(email="alice@ex.com")
        profile = _default_engine().project(m)
        assert "email_address" in profile.fields

    def test_phone_renamed_to_phone_number(self):
        m = _merged(phone="+919876543210")
        profile = _default_engine().project(m)
        assert "phone_number" in profile.fields

    def test_summary_renamed_to_professional_summary(self):
        m = _merged(summary="Experienced engineer")
        profile = _default_engine().project(m)
        assert "professional_summary" in profile.fields

    def test_experience_renamed_to_work_experience(self):
        m = _merged(experience=[Experience(company="Acme")])
        profile = _default_engine().project(m)
        assert "work_experience" in profile.fields

    def test_field_value_preserved_after_rename(self):
        m = _merged(name="Alice Smith")
        profile = _default_engine().project(m)
        assert profile.fields["full_name"] == "Alice Smith"


class TestFieldExclusion:
    def test_excluded_field_absent_from_output(self):
        e = _engine_with([
            {"source": "name", "output_name": "full_name", "include": True},
            {"source": "email", "output_name": "email_address", "include": False},
        ])
        m = _merged(name="Alice", email="alice@ex.com")
        profile = e.project(m)
        assert "full_name" in profile.fields
        assert "email_address" not in profile.fields

    def test_all_excluded_gives_empty_fields(self):
        e = _engine_with([
            {"source": "name", "output_name": "full_name", "include": False},
        ])
        profile = e.project(_merged(name="Alice"))
        assert profile.fields == {}

    def test_no_field_rules_gives_empty_fields(self):
        e = ProjectionEngine(config=ProjectionConfig(fields=[]))
        profile = e.project(_merged(name="Alice"))
        assert profile.fields == {}


class TestDefaults:
    def test_none_field_uses_default_null(self):
        e = _engine_with([
            {"source": "name", "output_name": "full_name", "include": True, "default": None}
        ])
        profile = e.project(_merged(name=None))
        assert profile.fields["full_name"] is None

    def test_none_list_field_uses_default_empty_list(self):
        e = _engine_with([
            {"source": "skills", "output_name": "skills", "include": True, "default": []}
        ])
        # skills is always a list (never None), but if we clear it:
        m = MergedCandidate(skills=[])
        profile = e.project(m)
        # Empty list is not None, so default is not applied; value is []
        assert profile.fields["skills"] == []

    def test_custom_default_applied(self):
        e = _engine_with([
            {"source": "phone", "output_name": "phone_number", "include": True,
             "default": "N/A"}
        ])
        profile = e.project(_merged(phone=None))
        assert profile.fields["phone_number"] == "N/A"


# ---------------------------------------------------------------------------
# Serialization of Pydantic sub-models
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_experience_serialised_to_dict(self):
        exp = Experience(company="Acme Corp", title="Engineer")
        m = _merged(experience=[exp])
        profile = _default_engine().project(m)
        work_exp = profile.fields["work_experience"]
        assert isinstance(work_exp, list)
        assert isinstance(work_exp[0], dict)
        assert work_exp[0]["company"] == "Acme Corp"

    def test_education_serialised_to_dict(self):
        edu = Education(institution="MIT", degree="BSc")
        m = _merged(education=[edu])
        profile = _default_engine().project(m)
        edu_out = profile.fields["education"]
        assert isinstance(edu_out[0], dict)
        assert edu_out[0]["institution"] == "MIT"

    def test_links_serialised_to_dict(self):
        lnk = Link(url="https://github.com/alice", label="GitHub")
        m = _merged(links=[lnk])
        profile = _default_engine().project(m)
        links_out = profile.fields["links"]
        assert isinstance(links_out[0], dict)
        assert links_out[0]["url"] == "https://github.com/alice"

    def test_skills_list_passthrough(self):
        m = _merged(skills=["Python", "SQL"])
        profile = _default_engine().project(m)
        assert profile.fields["skills"] == ["Python", "SQL"]

    def test_empty_list_serialises_to_empty_list(self):
        m = _merged(experience=[])
        profile = _default_engine().project(m)
        assert profile.fields["work_experience"] == []


# ---------------------------------------------------------------------------
# Confidence / Provenance / Validation flags
# ---------------------------------------------------------------------------


class TestFlags:
    def test_confidence_included_by_default(self):
        profile = _default_engine().project(_merged(), confidence=_confidence())
        assert profile.confidence is not None
        assert profile.confidence.overall_score == 0.85

    def test_confidence_excluded_when_flag_false(self):
        e = ProjectionEngine(config=ProjectionConfig(
            fields=[], include_confidence=False
        ))
        profile = e.project(_merged(), confidence=_confidence())
        assert profile.confidence is None

    def test_provenance_excluded_by_default(self):
        profile = _default_engine().project(_merged(), provenance=_provenance())
        assert profile.provenance == []

    def test_provenance_included_when_flag_true(self):
        e = ProjectionEngine(config=ProjectionConfig(
            fields=[], include_provenance=True
        ))
        prov = _provenance()
        profile = e.project(_merged(), provenance=prov)
        assert len(profile.provenance) == 1

    def test_validation_included_by_default(self):
        val = _validation(valid=False)
        profile = _default_engine().project(_merged(), validation=val)
        assert profile.validation is not None
        assert profile.validation.is_valid is False

    def test_validation_excluded_when_flag_false(self):
        e = ProjectionEngine(config=ProjectionConfig(
            fields=[], include_validation=False
        ))
        profile = e.project(_merged(), validation=_validation())
        assert profile.validation is None

    def test_none_confidence_stays_none(self):
        profile = _default_engine().project(_merged(), confidence=None)
        assert profile.confidence is None

    def test_none_provenance_produces_empty_list(self):
        e = ProjectionEngine(config=ProjectionConfig(
            fields=[], include_provenance=True
        ))
        profile = e.project(_merged(), provenance=None)
        assert profile.provenance == []


# ---------------------------------------------------------------------------
# Profile metadata
# ---------------------------------------------------------------------------


class TestProfileMetadata:
    def test_generated_at_set(self):
        profile = _default_engine().project(_merged())
        assert profile.generated_at is not None

    def test_pipeline_version_set(self):
        profile = _default_engine().project(_merged())
        assert profile.pipeline_version == "1.0.0"

    def test_returns_candidate_profile_instance(self):
        profile = _default_engine().project(_merged())
        assert isinstance(profile, CandidateProfile)


# ---------------------------------------------------------------------------
# _serialize helper
# ---------------------------------------------------------------------------


class TestSerializeHelper:
    def test_string_passthrough(self):
        assert _serialize("hello") == "hello"

    def test_none_passthrough(self):
        assert _serialize(None) is None

    def test_pydantic_model_to_dict(self):
        exp = Experience(company="Acme")
        result = _serialize(exp)
        assert isinstance(result, dict)
        assert result["company"] == "Acme"

    def test_list_of_models_to_list_of_dicts(self):
        items = [Experience(company="A"), Experience(company="B")]
        result = _serialize(items)
        assert all(isinstance(r, dict) for r in result)
        assert result[0]["company"] == "A"

    def test_nested_list_in_list(self):
        result = _serialize([["a", "b"], ["c"]])
        assert result == [["a", "b"], ["c"]]

    def test_int_passthrough(self):
        assert _serialize(42) == 42
