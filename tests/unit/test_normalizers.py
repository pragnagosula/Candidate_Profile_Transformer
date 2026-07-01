"""Unit tests for the normalization engine and individual normalizers."""

from __future__ import annotations

import pytest

from app.config.models import NormalizationConfig, PhoneNormalizationConfig, SkillNormalizationConfig
from app.models.candidate import DataSource, DateRange, Education, Experience, ExtractedCandidate, Link
from app.normalizers.date_normalizer import DateNormalizer
from app.normalizers.email_normalizer import EmailNormalizer
from app.normalizers.engine import NormalizationEngine
from app.normalizers.name_normalizer import NameNormalizer, _title_case_name
from app.normalizers.phone_normalizer import PhoneNormalizer
from app.normalizers.skill_normalizer import SkillNormalizer, _exact_dedup, _fuzzy_dedup, _resolve_synonyms
from app.normalizers.url_normalizer import URLNormalizer


# ---------------------------------------------------------------------------
# EmailNormalizer
# ---------------------------------------------------------------------------

class TestEmailNormalizer:
    def _norm(self, value):
        return EmailNormalizer().normalize(value, NormalizationConfig())

    def test_lowercases(self):
        assert self._norm("ALICE@EXAMPLE.COM") == "alice@example.com"

    def test_strips_whitespace(self):
        assert self._norm("  bob@test.com  ") == "bob@test.com"

    def test_none_passthrough(self):
        assert self._norm(None) is None

    def test_empty_passthrough(self):
        assert self._norm("") == ""

    def test_already_lowercase_unchanged(self):
        assert self._norm("alice@example.com") == "alice@example.com"

    def test_mixed_case_domain(self):
        assert self._norm("User@Example.COM") == "user@example.com"

    def test_idempotent(self):
        n = EmailNormalizer()
        cfg = NormalizationConfig()
        v = "ALICE@TEST.COM"
        assert n.normalize(n.normalize(v, cfg), cfg) == n.normalize(v, cfg)


# ---------------------------------------------------------------------------
# PhoneNormalizer
# ---------------------------------------------------------------------------

class TestPhoneNormalizer:
    def _norm(self, value, country="IN"):
        cfg = NormalizationConfig(phone=PhoneNormalizationConfig(default_country_code=country))
        return PhoneNormalizer().normalize(value, cfg)

    def test_plain_ten_digits_india(self):
        result = self._norm("9876543210")
        assert result == "+919876543210"

    def test_spaced_international(self):
        result = self._norm("+91 98765-43210")
        assert result == "+919876543210"

    def test_parenthesis_format(self):
        result = self._norm("(+91)9876543210")
        assert result == "+919876543210"

    def test_us_number(self):
        result = self._norm("+1-800-555-0199", country="US")
        assert result == "+18005550199"

    def test_none_passthrough(self):
        assert self._norm(None) is None

    def test_too_few_digits_unchanged(self):
        result = self._norm("123")
        assert result == "123"

    def test_garbage_unchanged(self):
        result = self._norm("not-a-phone")
        assert result == "not-a-phone"

    def test_idempotent(self):
        cfg = NormalizationConfig(phone=PhoneNormalizationConfig(default_country_code="IN"))
        n = PhoneNormalizer()
        v = "9876543210"
        assert n.normalize(n.normalize(v, cfg), cfg) == n.normalize(v, cfg)


# ---------------------------------------------------------------------------
# DateNormalizer
# ---------------------------------------------------------------------------

class TestDateNormalizer:
    def _norm(self, value):
        return DateNormalizer().normalize(value, NormalizationConfig())

    def test_jan_2022(self):
        assert self._norm("Jan 2022") == "2022-01"

    def test_january_2022(self):
        assert self._norm("January 2022") == "2022-01"

    def test_slash_format(self):
        assert self._norm("01/2022") == "2022-01"

    def test_already_canonical(self):
        assert self._norm("2022-01") == "2022-01"

    def test_year_only(self):
        assert self._norm("2022") == "2022"

    def test_present_returns_none(self):
        assert self._norm("present") is None

    def test_current_returns_none(self):
        assert self._norm("current") is None

    def test_now_returns_none(self):
        assert self._norm("now") is None

    def test_none_passthrough(self):
        assert self._norm(None) is None

    def test_empty_passthrough(self):
        assert self._norm("") == ""

    def test_idempotent(self):
        n = DateNormalizer()
        cfg = NormalizationConfig()
        v = "Jan 2022"
        assert n.normalize(n.normalize(v, cfg), cfg) == n.normalize(v, cfg)


# ---------------------------------------------------------------------------
# NameNormalizer
# ---------------------------------------------------------------------------

class TestNameNormalizer:
    def _norm(self, value):
        return NameNormalizer().normalize(value, NormalizationConfig())

    def test_lowercase_to_title(self):
        assert self._norm("alice johnson") == "Alice Johnson"

    def test_uppercase_to_title(self):
        assert self._norm("ALICE JOHNSON") == "Alice Johnson"

    def test_already_titled_unchanged(self):
        assert self._norm("Alice Johnson") == "Alice Johnson"

    def test_strips_extra_whitespace(self):
        assert self._norm("  Alice   Johnson  ") == "Alice Johnson"

    def test_preserves_particles(self):
        result = self._norm("jan van der berg")
        assert result == "Jan van der Berg"

    def test_initial_preserved(self):
        result = self._norm("john a. smith")
        assert "A." in result or "a." in result  # initial handled

    def test_unicode_name_nfkc(self):
        result = self._norm("Ãlicé Jøhnson")
        assert result is not None and len(result) > 0

    def test_none_passthrough(self):
        assert self._norm(None) is None

    def test_empty_passthrough(self):
        assert self._norm("") == ""

    def test_idempotent(self):
        n = NameNormalizer()
        cfg = NormalizationConfig()
        v = "alice johnson"
        assert n.normalize(n.normalize(v, cfg), cfg) == n.normalize(v, cfg)


# ---------------------------------------------------------------------------
# SkillNormalizer
# ---------------------------------------------------------------------------

SKILL_CFG = SkillNormalizationConfig(
    synonyms={
        "Machine Learning": ["ML", "machine learning", "machine-learning"],
        "Python": ["python3", "py"],
    },
    case_sensitive=False,
)


class TestSkillNormalizer:
    def _norm(self, skills):
        cfg = NormalizationConfig(skills=SKILL_CFG)
        return SkillNormalizer().normalize(skills, cfg)

    def test_synonym_resolved(self):
        result = self._norm(["ML"])
        assert "Machine Learning" in result

    def test_multiple_synonyms(self):
        result = self._norm(["ML", "machine learning"])
        assert result.count("Machine Learning") == 1

    def test_exact_dedup_case_insensitive(self):
        result = self._norm(["Python", "python", "PYTHON"])
        assert len(result) == 1

    def test_non_synonym_skill_preserved(self):
        result = self._norm(["Docker", "SQL"])
        assert "Docker" in result
        assert "SQL" in result

    def test_empty_list(self):
        assert self._norm([]) == []

    def test_synonym_and_canonical_deduped(self):
        result = self._norm(["python3", "Python"])
        # Both resolve to "Python" — should be one entry
        assert result.count("Python") == 1

    def test_order_preserved(self):
        result = self._norm(["Docker", "SQL", "ML"])
        # After dedup and synonym resolution, relative order preserved
        assert result.index("Docker") < result.index("SQL")


class TestExactDedup:
    def test_case_insensitive(self):
        assert _exact_dedup(["Python", "python", "PYTHON"], False) == ["Python"]

    def test_case_sensitive(self):
        result = _exact_dedup(["Python", "python"], True)
        assert len(result) == 2

    def test_order_preserved(self):
        assert _exact_dedup(["A", "B", "A", "C"], False) == ["A", "B", "C"]


class TestFuzzyDedup:
    def test_removes_near_duplicate(self):
        result = _fuzzy_dedup(["Machine Learning", "machine learning"], 85)
        assert len(result) == 1

    def test_keeps_distinct_skills(self):
        result = _fuzzy_dedup(["Python", "Java"], 85)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# URLNormalizer
# ---------------------------------------------------------------------------

class TestURLNormalizer:
    def _norm(self, value):
        return URLNormalizer().normalize(value, NormalizationConfig())

    def test_adds_https(self):
        result = self._norm("linkedin.com/in/alice")
        assert result.startswith("https://")

    def test_lowercases_host(self):
        result = self._norm("https://LinkedIn.COM/in/alice")
        assert "linkedin.com" in result

    def test_strips_trailing_slash(self):
        result = self._norm("https://github.com/alice/")
        assert not result.endswith("/")

    def test_preserves_path(self):
        result = self._norm("https://github.com/alice/repo")
        assert "/alice/repo" in result

    def test_none_passthrough(self):
        assert self._norm(None) is None

    def test_idempotent(self):
        n = URLNormalizer()
        cfg = NormalizationConfig()
        v = "linkedin.com/in/alice"
        assert n.normalize(n.normalize(v, cfg), cfg) == n.normalize(v, cfg)


# ---------------------------------------------------------------------------
# NormalizationEngine (integration)
# ---------------------------------------------------------------------------

class TestNormalizationEngine:
    def _engine(self):
        return NormalizationEngine(NormalizationConfig(skills=SKILL_CFG))

    def _extracted(self, **kwargs) -> ExtractedCandidate:
        defaults = dict(source=DataSource.CSV)
        defaults.update(kwargs)
        return ExtractedCandidate(**defaults)

    def test_normalizes_email(self):
        e = self._extracted(email="ALICE@EXAMPLE.COM")
        result, diff = self._engine().normalize(e)
        assert result.email == "alice@example.com"
        assert "email" in diff.changes

    def test_normalizes_name(self):
        e = self._extracted(name="alice johnson")
        result, diff = self._engine().normalize(e)
        assert result.name == "Alice Johnson"

    def test_normalizes_phone(self):
        e = self._extracted(phone="9876543210")
        result, diff = self._engine().normalize(e)
        assert result.phone == "+919876543210"

    def test_normalizes_skills_with_synonym(self):
        e = self._extracted(skills=["ML", "Python"])
        result, diff = self._engine().normalize(e)
        assert "Machine Learning" in result.skills
        assert "Python" in result.skills

    def test_normalizes_date_in_experience(self):
        exp = Experience(duration=DateRange(start="Jan 2021", end="Dec 2022"))
        e = self._extracted(experience=[exp])
        result, _ = self._engine().normalize(e)
        assert result.experience[0].duration.start == "2021-01"
        assert result.experience[0].duration.end == "2022-12"

    def test_normalizes_link_url(self):
        link = Link(url="linkedin.com/in/alice", label="LinkedIn")
        e = self._extracted(links=[link])
        result, _ = self._engine().normalize(e)
        assert result.links[0].url.startswith("https://")

    def test_unchanged_fields_not_in_diff(self):
        e = self._extracted(email="alice@example.com")  # already normalized
        _, diff = self._engine().normalize(e)
        assert "email" not in diff.changes

    def test_source_preserved(self):
        e = self._extracted(source=DataSource.RESUME_PDF)
        result, _ = self._engine().normalize(e)
        assert result.source == DataSource.RESUME_PDF

    def test_empty_candidate_no_crash(self):
        e = self._extracted()
        result, diff = self._engine().normalize(e)
        assert result is not None
        assert diff.changes == {}
