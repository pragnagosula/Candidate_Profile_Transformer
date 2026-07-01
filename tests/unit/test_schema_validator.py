"""Unit tests for the Schema Validator.

Coverage targets:
  - Valid profile produces no issues
  - Missing required field (include=True) → error
  - Excluded field (include=False) not checked
  - List source field with non-list value → error
  - Scalar source field with list/dict value → warning
  - None scalar value is valid (nullable)
  - Empty list value is valid for list fields
  - confidence.overall_score out of bounds → error
  - Field-level confidence score out of bounds → error
  - No confidence block → bounds not checked
  - Never raises on pathological input
  - Custom config injection
  - Extra keys in profile.fields (not in config) are silently ignored
"""

from __future__ import annotations

import pytest

from app.config.models import ProjectedField, ProjectionConfig
from app.models.candidate import (
    CandidateProfile,
    ConfidenceReport,
    FieldConfidence,
    ValidationResult,
)
from app.schema.validator import SchemaValidator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _profile(
    fields: dict | None = None,
    confidence: ConfidenceReport | None = None,
) -> CandidateProfile:
    return CandidateProfile(
        fields=fields or {},
        confidence=confidence,
    )


def _confidence(overall: float = 0.85, field_score: float = 0.80) -> ConfidenceReport:
    return ConfidenceReport(
        overall_score=overall,
        completeness=0.75,
        source_agreement=1.0,
        field_scores=[
            FieldConfidence(field_name="name", score=field_score, reason="test")
        ],
    )


def _config(*field_defs: dict) -> ProjectionConfig:
    return ProjectionConfig(
        fields=[ProjectedField(**f) for f in field_defs],
        include_confidence=True,
        include_provenance=False,
        include_validation=True,
    )


def _scalar(source: str, output_name: str, include: bool = True) -> dict:
    return {"source": source, "output_name": output_name, "include": include}


def _list_field(source: str, output_name: str, include: bool = True) -> dict:
    return {"source": source, "output_name": output_name, "include": include, "default": []}


def _validator(*field_defs: dict) -> SchemaValidator:
    return SchemaValidator(config=_config(*field_defs))


# ---------------------------------------------------------------------------
# Presence checks
# ---------------------------------------------------------------------------


class TestPresence:
    def test_present_field_no_issue(self):
        v = _validator(_scalar("name", "full_name"))
        result = v.validate(_profile({"full_name": "Alice"}))
        assert result.is_valid
        assert result.errors == []

    def test_missing_required_field_is_error(self):
        v = _validator(_scalar("name", "full_name"))
        result = v.validate(_profile({}))
        assert not result.is_valid
        assert any(i.field == "full_name" and i.severity == "error" for i in result.errors)

    def test_excluded_field_not_checked(self):
        v = _validator(_scalar("name", "full_name", include=False))
        # Profile has no full_name — that's fine because field is excluded
        result = v.validate(_profile({}))
        assert result.is_valid

    def test_multiple_missing_fields_multiple_errors(self):
        v = _validator(
            _scalar("name", "full_name"),
            _scalar("email", "email_address"),
        )
        result = v.validate(_profile({}))
        assert len(result.errors) == 2

    def test_extra_keys_in_profile_ignored(self):
        v = _validator(_scalar("name", "full_name"))
        # profile has extra key not in config
        result = v.validate(_profile({"full_name": "Alice", "unexpected_key": "x"}))
        assert result.is_valid

    def test_no_field_rules_always_valid(self):
        v = SchemaValidator(config=ProjectionConfig(fields=[]))
        result = v.validate(_profile({}))
        assert result.is_valid


# ---------------------------------------------------------------------------
# Type checks — list fields
# ---------------------------------------------------------------------------


class TestListFields:
    def test_list_value_is_valid(self):
        v = _validator(_list_field("skills", "skills"))
        result = v.validate(_profile({"skills": ["Python", "SQL"]}))
        assert result.is_valid

    def test_empty_list_is_valid(self):
        v = _validator(_list_field("skills", "skills"))
        result = v.validate(_profile({"skills": []}))
        assert result.is_valid

    def test_none_list_field_is_valid(self):
        v = _validator(_list_field("skills", "skills"))
        result = v.validate(_profile({"skills": None}))
        assert result.is_valid

    def test_string_instead_of_list_is_error(self):
        v = _validator(_list_field("skills", "skills"))
        result = v.validate(_profile({"skills": "Python, SQL"}))
        assert not result.is_valid
        assert any(i.field == "skills" and i.severity == "error" for i in result.errors)

    def test_dict_instead_of_list_is_error(self):
        v = _validator(_list_field("experience", "work_experience"))
        result = v.validate(_profile({"work_experience": {"company": "Acme"}}))
        assert not result.is_valid

    def test_education_list_valid(self):
        v = _validator(_list_field("education", "education"))
        result = v.validate(_profile({"education": [{"institution": "MIT"}]}))
        assert result.is_valid


# ---------------------------------------------------------------------------
# Type checks — scalar fields
# ---------------------------------------------------------------------------


class TestScalarFields:
    def test_string_scalar_is_valid(self):
        v = _validator(_scalar("name", "full_name"))
        result = v.validate(_profile({"full_name": "Alice Smith"}))
        assert result.is_valid

    def test_none_scalar_is_valid(self):
        v = _validator(_scalar("name", "full_name"))
        result = v.validate(_profile({"full_name": None}))
        assert result.is_valid

    def test_list_as_scalar_is_warning(self):
        v = _validator(_scalar("name", "full_name"))
        result = v.validate(_profile({"full_name": ["Alice", "Smith"]}))
        # Warning only — is_valid can still be True
        assert any(i.field == "full_name" and i.severity == "warning" for i in result.warnings)

    def test_dict_as_scalar_is_warning(self):
        v = _validator(_scalar("email", "email_address"))
        result = v.validate(_profile({"email_address": {"value": "alice@ex.com"}}))
        assert any(i.field == "email_address" and i.severity == "warning" for i in result.warnings)

    def test_warning_alone_does_not_make_invalid(self):
        v = _validator(_scalar("name", "full_name"))
        result = v.validate(_profile({"full_name": ["Alice"]}))
        assert result.is_valid  # warning-only


# ---------------------------------------------------------------------------
# Confidence bounds
# ---------------------------------------------------------------------------


class TestConfidenceBounds:
    def test_valid_confidence_no_issue(self):
        v = _validator(_scalar("name", "full_name"))
        result = v.validate(_profile({"full_name": "Alice"}, confidence=_confidence(0.85, 0.80)))
        assert result.errors == []

    def test_overall_score_above_one_is_error(self):
        # Pydantic guards against this for ConfidenceReport created normally,
        # but SchemaValidator still checks as a defence-in-depth measure.
        v = SchemaValidator(config=ProjectionConfig(fields=[]))
        # Bypass Pydantic validation to inject a bad value
        conf = ConfidenceReport(overall_score=0.5, completeness=0.5, source_agreement=0.5)
        object.__setattr__(conf, "overall_score", 1.5)  # bypass Pydantic
        result = v.validate(_profile(confidence=conf))
        assert any("overall_score" in i.field for i in result.errors)

    def test_field_score_above_one_is_error(self):
        v = SchemaValidator(config=ProjectionConfig(fields=[]))
        conf = ConfidenceReport(
            overall_score=0.5,
            completeness=0.5,
            source_agreement=0.5,
            field_scores=[FieldConfidence(field_name="name", score=0.8)],
        )
        # Inject bad field score
        object.__setattr__(conf.field_scores[0], "score", 1.2)
        result = v.validate(_profile(confidence=conf))
        assert any("field_scores" in i.field for i in result.errors)

    def test_no_confidence_block_skips_bounds_check(self):
        v = _validator(_scalar("name", "full_name"))
        result = v.validate(_profile({"full_name": "Alice"}, confidence=None))
        assert result.errors == []


# ---------------------------------------------------------------------------
# Never-crash guarantee
# ---------------------------------------------------------------------------


class TestNeverCrash:
    def test_none_fields_dict_does_not_raise(self):
        # CandidateProfile.fields is always a dict, but profile object might be
        # malformed — validator must not raise
        v = _validator(_scalar("name", "full_name"))
        profile = CandidateProfile(fields={})
        # Inject None to fields to simulate corruption
        object.__setattr__(profile, "fields", None)
        # Should return a result, not raise
        try:
            result = v.validate(profile)
            assert isinstance(result, ValidationResult)
        except Exception:
            pytest.fail("SchemaValidator raised on malformed profile")

    def test_completely_empty_profile_no_rules_valid(self):
        v = SchemaValidator(config=ProjectionConfig(fields=[]))
        result = v.validate(_profile())
        assert result.is_valid


# ---------------------------------------------------------------------------
# Custom config
# ---------------------------------------------------------------------------


class TestCustomConfig:
    def test_custom_output_name_checked(self):
        v = _validator({"source": "name", "output_name": "candidate_name", "include": True})
        # Profile uses "candidate_name" not "full_name"
        result = v.validate(_profile({"candidate_name": "Alice"}))
        assert result.is_valid

    def test_custom_config_missing_custom_field_is_error(self):
        v = _validator({"source": "name", "output_name": "candidate_name", "include": True})
        result = v.validate(_profile({"full_name": "Alice"}))  # wrong key
        assert not result.is_valid
