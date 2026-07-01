"""Unit tests for the validation layer."""

from __future__ import annotations

import pytest

from app.models.candidate import (
    DataSource,
    DateRange,
    Education,
    Experience,
    Link,
    NormalizedCandidate,
    ValidationResult,
)
from app.config.models import FieldCategoryConfig
from app.validators.completeness_validator import CompletenessValidator
from app.validators.date_validator import DateValidator, _is_valid_date
from app.validators.email_validator import EmailValidator
from app.validators.engine import ValidationEngine
from app.validators.phone_validator import PhoneValidator
from app.validators.required_fields import RequiredFieldsValidator
from app.validators.skills_validator import SkillsValidator
from app.validators.url_validator import URLValidator, _is_valid_url


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _candidate(**kwargs) -> NormalizedCandidate:
    defaults = dict(source=DataSource.CSV, name="Alice Johnson", email="alice@example.com")
    defaults.update(kwargs)
    return NormalizedCandidate(**defaults)


# ---------------------------------------------------------------------------
# RequiredFieldsValidator
# ---------------------------------------------------------------------------

class TestRequiredFieldsValidator:
    V = RequiredFieldsValidator()

    def test_valid_passes(self):
        assert self.V.validate(_candidate()) == []

    def test_missing_name_is_error(self):
        issues = self.V.validate(_candidate(name=None))
        assert any(i.field == "name" and i.severity == "error" for i in issues)

    def test_missing_email_is_error(self):
        issues = self.V.validate(_candidate(email=None))
        assert any(i.field == "email" and i.severity == "error" for i in issues)

    def test_empty_name_is_error(self):
        issues = self.V.validate(_candidate(name="   "))
        assert any(i.field == "name" and i.severity == "error" for i in issues)

    def test_both_missing_gives_two_errors(self):
        issues = self.V.validate(_candidate(name=None, email=None))
        assert len(issues) == 2

    def test_does_not_flag_phone_or_skills(self):
        issues = self.V.validate(_candidate(phone=None, skills=[]))
        assert issues == []


# ---------------------------------------------------------------------------
# EmailValidator
# ---------------------------------------------------------------------------

class TestEmailValidator:
    V = EmailValidator()

    def test_valid_email_passes(self):
        assert self.V.validate(_candidate(email="alice@example.com")) == []

    def test_missing_email_skipped(self):
        # Absence is RequiredFieldsValidator's concern; this one skips None
        assert self.V.validate(_candidate(email=None)) == []

    def test_no_at_sign_is_error(self):
        issues = self.V.validate(_candidate(email="notanemail"))
        assert issues and issues[0].severity == "error"

    def test_no_domain_is_error(self):
        issues = self.V.validate(_candidate(email="alice@"))
        assert issues and issues[0].severity == "error"

    def test_no_tld_is_error(self):
        issues = self.V.validate(_candidate(email="alice@example"))
        assert issues and issues[0].severity == "error"

    def test_valid_with_plus_addressing(self):
        assert self.V.validate(_candidate(email="alice+tag@example.com")) == []

    def test_valid_with_subdomain(self):
        assert self.V.validate(_candidate(email="alice@mail.example.co.uk")) == []

    def test_invalid_double_at(self):
        issues = self.V.validate(_candidate(email="alice@@example.com"))
        assert issues


# ---------------------------------------------------------------------------
# PhoneValidator
# ---------------------------------------------------------------------------

class TestPhoneValidator:
    V = PhoneValidator()

    def test_valid_e164_passes(self):
        assert self.V.validate(_candidate(phone="+919876543210")) == []

    def test_valid_us_number_passes(self):
        assert self.V.validate(_candidate(phone="+18005550199")) == []

    def test_missing_phone_skipped(self):
        assert self.V.validate(_candidate(phone=None)) == []

    def test_too_few_digits_is_warning(self):
        issues = self.V.validate(_candidate(phone="123"))
        assert issues and issues[0].severity == "warning"

    def test_garbage_string_is_warning(self):
        issues = self.V.validate(_candidate(phone="not-a-phone"))
        assert issues and issues[0].severity == "warning"


# ---------------------------------------------------------------------------
# DateValidator
# ---------------------------------------------------------------------------

class TestIsValidDate:
    def test_yyyy_mm_valid(self):
        assert _is_valid_date("2022-01") is True

    def test_yyyy_valid(self):
        assert _is_valid_date("2022") is True

    def test_none_valid(self):
        assert _is_valid_date(None) is True

    def test_empty_string_valid(self):
        assert _is_valid_date("") is True

    def test_jan_2022_invalid(self):
        assert _is_valid_date("Jan 2022") is False

    def test_slash_format_invalid(self):
        assert _is_valid_date("01/2022") is False

    def test_invalid_month_13(self):
        assert _is_valid_date("2022-13") is False


class TestDateValidator:
    V = DateValidator()

    def test_valid_experience_dates(self):
        exp = Experience(duration=DateRange(start="2021-01", end="2023-06"))
        issues = self.V.validate(_candidate(experience=[exp]))
        assert issues == []

    def test_non_canonical_start_date_is_warning(self):
        exp = Experience(duration=DateRange(start="Jan 2021", end="2023-06"))
        issues = self.V.validate(_candidate(experience=[exp]))
        assert any(i.severity == "warning" for i in issues)

    def test_present_end_date_is_valid(self):
        exp = Experience(duration=DateRange(start="2021-01", end=None, is_current=True))
        assert self.V.validate(_candidate(experience=[exp])) == []

    def test_non_canonical_education_date_is_warning(self):
        edu = Education(duration=DateRange(start="2018", end="01/2022"))
        issues = self.V.validate(_candidate(education=[edu]))
        assert any("education" in i.field for i in issues)

    def test_no_experience_no_issues(self):
        assert self.V.validate(_candidate()) == []


# ---------------------------------------------------------------------------
# SkillsValidator
# ---------------------------------------------------------------------------

class TestSkillsValidator:
    V = SkillsValidator()

    def test_unique_skills_pass(self):
        c = _candidate(skills=["Python", "Machine Learning", "SQL"])
        assert self.V.validate(c) == []

    def test_exact_duplicate_is_warning(self):
        c = _candidate(skills=["Python", "Python"])
        issues = self.V.validate(c)
        assert issues and issues[0].severity == "warning"

    def test_case_insensitive_duplicate_is_warning(self):
        c = _candidate(skills=["Python", "python"])
        issues = self.V.validate(c)
        assert issues

    def test_empty_skills_passes(self):
        assert self.V.validate(_candidate(skills=[])) == []

    def test_three_duplicates_gives_two_warnings(self):
        c = _candidate(skills=["Python", "Python", "Python"])
        issues = self.V.validate(c)
        assert len(issues) == 2  # second and third occurrences flagged


# ---------------------------------------------------------------------------
# URLValidator
# ---------------------------------------------------------------------------

class TestIsValidURL:
    def test_https_url_valid(self):
        assert _is_valid_url("https://linkedin.com/in/alice") is True

    def test_http_url_valid(self):
        assert _is_valid_url("http://example.com") is True

    def test_no_scheme_invalid(self):
        assert _is_valid_url("linkedin.com/in/alice") is False

    def test_empty_string_invalid(self):
        assert _is_valid_url("") is False


class TestURLValidator:
    V = URLValidator()

    def test_valid_https_link_passes(self):
        c = _candidate(links=[Link(url="https://linkedin.com/in/alice", label="LinkedIn")])
        assert self.V.validate(c) == []

    def test_missing_scheme_is_warning(self):
        c = _candidate(links=[Link(url="linkedin.com/in/alice", label="LinkedIn")])
        issues = self.V.validate(c)
        assert issues and issues[0].severity == "warning"

    def test_empty_links_passes(self):
        assert self.V.validate(_candidate(links=[])) == []

    def test_multiple_links_each_checked(self):
        c = _candidate(links=[
            Link(url="https://github.com/alice", label="GitHub"),
            Link(url="not_a_url", label="Portfolio"),
        ])
        issues = self.V.validate(c)
        assert len(issues) == 1
        assert "Portfolio" in issues[0].message


# ---------------------------------------------------------------------------
# CompletenessValidator
# ---------------------------------------------------------------------------

class TestCompletenessValidator:
    V = CompletenessValidator()

    def test_fully_required_fields_present_no_warnings(self):
        # Provide all required fields; location/summary are optional → no warnings.
        c = _candidate(
            phone="+919876543210",
            skills=["Python"],
            experience=[Experience(company="Acme", title="Engineer")],
            education=[Education(institution="MIT", degree="BSc")],
            links=[Link(url="https://github.com/alice", label="GitHub")],
        )
        issues = self.V.validate(c)
        assert not any(i.severity == "warning" for i in issues)
        # Optional fields (location, summary) generate info messages — not warnings
        assert all(i.severity == "info" for i in issues)

    def test_missing_phone_is_warning(self):
        issues = self.V.validate(_candidate(phone=None))
        assert any(i.field == "phone" and i.severity == "warning" for i in issues)

    def test_missing_skills_is_warning(self):
        issues = self.V.validate(_candidate(skills=[]))
        assert any(i.field == "skills" and i.severity == "warning" for i in issues)

    def test_missing_experience_is_warning(self):
        issues = self.V.validate(_candidate(experience=[]))
        assert any(i.field == "experience" and i.severity == "warning" for i in issues)

    def test_missing_education_is_warning(self):
        issues = self.V.validate(_candidate(education=[]))
        assert any(i.field == "education" and i.severity == "warning" for i in issues)

    def test_missing_links_is_warning(self):
        # links is recommended → warning
        issues = self.V.validate(_candidate(links=[]))
        assert any(i.field == "links" and i.severity == "warning" for i in issues)

    def test_missing_location_is_info_not_warning(self):
        issues = self.V.validate(_candidate(location=None))
        location_issues = [i for i in issues if i.field == "location"]
        assert location_issues
        assert all(i.severity == "info" for i in location_issues)

    def test_missing_summary_is_info_not_warning(self):
        issues = self.V.validate(_candidate(summary=None))
        summary_issues = [i for i in issues if i.field == "summary"]
        assert summary_issues
        assert all(i.severity == "info" for i in summary_issues)

    def test_optional_fields_produce_no_warnings(self):
        # Candidate with all required/recommended fields but no summary/location
        c = _candidate(
            phone="+919876543210",
            skills=["Python"],
            experience=[Experience(company="Acme", title="Engineer")],
            education=[Education(institution="MIT", degree="BSc")],
            links=[Link(url="https://github.com/alice", label="GitHub")],
            location=None,
            summary=None,
        )
        issues = self.V.validate(c)
        assert not any(i.severity == "warning" for i in issues)
        info_issues = [i for i in issues if i.severity == "info"]
        info_fields = {i.field for i in info_issues}
        assert "location" in info_fields
        assert "summary" in info_fields

    def test_required_and_optional_issue_severities(self):
        # Missing phone (required) and missing summary (optional) in same candidate
        c = _candidate(phone=None, summary=None)
        issues = self.V.validate(c)
        phone_issue = next(i for i in issues if i.field == "phone")
        summary_issue = next(i for i in issues if i.field == "summary")
        assert phone_issue.severity == "warning"
        assert summary_issue.severity == "info"

    def test_custom_categories_respected(self):
        # Make 'location' required via custom config
        cats = FieldCategoryConfig(
            required_fields=["location"],
            recommended_fields=[],
            optional_fields=[],
        )
        v = CompletenessValidator(cats)
        issues = v.validate(_candidate(location=None))
        assert any(i.field == "location" and i.severity == "warning" for i in issues)

    def test_no_issues_when_all_required_and_recommended_present(self):
        c = _candidate(
            phone="+919876543210",
            skills=["Python"],
            experience=[Experience(company="Acme", title="Engineer")],
            education=[Education(institution="MIT", degree="BSc")],
            links=[Link(url="https://github.com/alice", label="GitHub")],
            location="Bangalore",
            summary="Experienced engineer",
        )
        issues = self.V.validate(c)
        assert issues == []


# ---------------------------------------------------------------------------
# ValidationEngine (integration)
# ---------------------------------------------------------------------------

class TestValidationEngine:
    E = ValidationEngine()

    def test_fully_valid_candidate(self):
        c = _candidate(
            phone="+919876543210",
            location="Bangalore",
            summary="Experienced engineer.",
            skills=["Python", "SQL"],
            links=[Link(url="https://linkedin.com/in/alice", label="LinkedIn")],
            experience=[Experience(duration=DateRange(start="2021-01", end=None, is_current=True))],
            education=[Education(institution="MIT", degree="BSc")],
        )
        result = self.E.validate(c)
        assert result.is_valid is True
        assert result.errors == []

    def test_missing_required_fields_invalid(self):
        c = _candidate(name=None, email=None)
        result = self.E.validate(c)
        assert result.is_valid is False
        assert len(result.errors) == 2

    def test_missing_optional_fields_do_not_make_invalid(self):
        # Missing summary and location → info only, never warnings or errors
        c = _candidate(
            phone="+919876543210",
            skills=["Python"],
            experience=[Experience(company="Acme", title="Eng")],
            education=[Education(institution="MIT")],
            links=[Link(url="https://github.com/alice", label="GitHub")],
            summary=None,
            location=None,
        )
        result = self.E.validate(c)
        assert result.is_valid is True
        assert result.warnings == []
        assert result.info  # two info messages (summary, location)

    def test_missing_required_field_gives_warning(self):
        # Missing phone (required) → warning, not error
        c = _candidate(phone=None)
        result = self.E.validate(c)
        assert result.is_valid is True  # warnings don't invalidate
        assert any(i.field == "phone" and i.severity == "warning" for i in result.issues)

    def test_buggy_validator_isolated(self):
        from app.validators.base import BaseValidator

        class BuggyValidator(BaseValidator):
            def validate(self, candidate):
                raise RuntimeError("I always crash")

        engine = ValidationEngine(extra_validators=[BuggyValidator()])
        # Should not raise; should return whatever the other validators found
        result = engine.validate(_candidate())
        assert isinstance(result, ValidationResult)

    def test_result_carries_all_issue_types(self):
        c = _candidate(
            email="not-valid",            # → error
            phone=None,                   # → warning (completeness)
            skills=["Python", "python"],  # → warning (duplicate)
        )
        result = self.E.validate(c)
        assert result.errors
        assert result.warnings

    def test_duplicate_email_domain_warning(self):
        c = _candidate(skills=["Docker", "Docker"])
        result = self.E.validate(c)
        dup_warnings = [i for i in result.warnings if "Duplicate" in i.message]
        assert dup_warnings

    def test_info_issues_not_in_warnings(self):
        # Missing summary → info severity; must not appear in result.warnings
        c = _candidate(summary=None)
        result = self.E.validate(c)
        summary_warnings = [i for i in result.warnings if i.field == "summary"]
        summary_info    = [i for i in result.info    if i.field == "summary"]
        assert not summary_warnings
        assert summary_info

    def test_deduplication_across_validators(self):
        # The engine should not emit the same issue twice even if two validator
        # instances somehow produce identical output (e.g. via extra_validators).
        from app.validators.completeness_validator import CompletenessValidator

        extra = CompletenessValidator()  # second completeness check
        engine = ValidationEngine(extra_validators=[extra])
        c = _candidate(phone=None)
        result = engine.validate(c)
        phone_issues = [i for i in result.issues if i.field == "phone" and i.severity == "warning"]
        assert len(phone_issues) == 1  # deduplicated to exactly one

    def test_validation_result_info_property(self):
        from app.models.candidate import ValidationIssue, ValidationResult
        result = ValidationResult(
            is_valid=True,
            issues=[
                ValidationIssue(field="summary", severity="info", message="optional"),
                ValidationIssue(field="email", severity="warning", message="missing"),
            ],
        )
        assert len(result.info) == 1
        assert result.info[0].field == "summary"
        assert len(result.warnings) == 1
