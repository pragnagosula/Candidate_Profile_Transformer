"""Unit tests for the Merge Engine.

Coverage targets:
  - Singleton group passthrough (no data loss)
  - Priority strategy: highest-priority source wins
  - Priority strategy: fallback when highest-priority source is missing the field
  - Priority strategy: global fallback when no priority source has the value
  - Most-complete strategy: longest string wins
  - Union strategy: skills combined from all sources, case-insensitive dedup
  - Union strategy: links deduplicated by URL
  - Union strategy: experience/education combined (structural equality dedup)
  - Extra fields merged (higher priority overwrites lower)
  - Source records contain all input candidates
  - Empty group returns empty MergedCandidate
  - Custom merge rules config injection
  - Helper functions: _non_empty, _sort_by_priority, _already_seen
"""

from __future__ import annotations

import pytest

from app.config.models import FieldMergeRule, MergeRulesConfig
from app.mergers.candidate_group import CandidateGroup
from app.mergers.merge_engine import (
    MergeEngine,
    _already_seen,
    _education_similar,
    _experience_similar,
    _find_similar_index,
    _non_empty,
    _normalize_degree,
    _sort_by_priority,
)
from app.models.candidate import (
    DataSource,
    DateRange,
    Education,
    Experience,
    Link,
    NormalizedCandidate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _c(
    source: DataSource = DataSource.CSV,
    name: str | None = None,
    email: str | None = None,
    phone: str | None = None,
    location: str | None = None,
    summary: str | None = None,
    skills: list[str] | None = None,
    experience: list[Experience] | None = None,
    education: list[Education] | None = None,
    links: list[Link] | None = None,
    extra_fields: dict | None = None,
) -> NormalizedCandidate:
    return NormalizedCandidate(
        source=source,
        name=name,
        email=email,
        phone=phone,
        location=location,
        summary=summary,
        skills=skills or [],
        experience=experience or [],
        education=education or [],
        links=links or [],
        extra_fields=extra_fields or {},
    )


def _group(*candidates: NormalizedCandidate) -> CandidateGroup:
    return CandidateGroup(candidates=list(candidates))


def _engine_with_rules(**field_overrides) -> MergeEngine:
    """Build a MergeEngine whose rules for each named field are overridden."""
    rules = {
        field: FieldMergeRule(**spec)
        for field, spec in field_overrides.items()
    }
    cfg = MergeRulesConfig(field_rules=rules)
    return MergeEngine(config=cfg)


# ---------------------------------------------------------------------------
# Empty / singleton
# ---------------------------------------------------------------------------


class TestEdgeCases:
    E = MergeEngine()

    def test_empty_group_returns_empty_merged(self):
        result = self.E.merge(_group())
        assert result.name is None
        assert result.skills == []
        assert result.source_records == []

    def test_singleton_copies_all_scalar_fields(self):
        c = _c(
            source=DataSource.CSV,
            name="Alice Smith",
            email="alice@ex.com",
            phone="+919876543210",
            location="Bangalore",
            summary="Engineer",
        )
        result = self.E.merge(_group(c))
        assert result.name == "Alice Smith"
        assert result.email == "alice@ex.com"
        assert result.phone == "+919876543210"
        assert result.location == "Bangalore"
        assert result.summary == "Engineer"

    def test_singleton_copies_list_fields(self):
        exp = Experience(company="Acme")
        edu = Education(institution="MIT")
        lnk = Link(url="https://github.com/alice", label="GitHub")
        c = _c(
            source=DataSource.CSV,
            skills=["Python", "SQL"],
            experience=[exp],
            education=[edu],
            links=[lnk],
        )
        result = self.E.merge(_group(c))
        assert result.skills == ["Python", "SQL"]
        assert result.experience == [exp]
        assert result.education == [edu]
        assert result.links == [lnk]

    def test_source_records_populated(self):
        a = _c(DataSource.CSV, name="Alice")
        b = _c(DataSource.JSON, name="Alice")
        result = self.E.merge(_group(a, b))
        assert len(result.source_records) == 2
        assert a in result.source_records
        assert b in result.source_records

    def test_merge_timestamp_is_set(self):
        result = self.E.merge(_group(_c(DataSource.CSV, name="Alice")))
        assert result.merge_timestamp is not None


# ---------------------------------------------------------------------------
# Priority strategy
# ---------------------------------------------------------------------------


class TestPriorityStrategy:
    def test_highest_priority_source_wins(self):
        # resume_pdf > csv for "name" per default config
        pdf = _c(DataSource.RESUME_PDF, name="Alice Smith (PDF)")
        csv = _c(DataSource.CSV, name="Alice Smith (CSV)")
        e = MergeEngine()
        result = e.merge(_group(pdf, csv))
        assert result.name == "Alice Smith (PDF)"

    def test_fallback_when_top_source_missing(self):
        # resume_pdf missing name → should fall back to csv
        pdf = _c(DataSource.RESUME_PDF, name=None)
        csv = _c(DataSource.CSV, name="Alice Smith (CSV)")
        e = MergeEngine()
        result = e.merge(_group(pdf, csv))
        assert result.name == "Alice Smith (CSV)"

    def test_global_fallback_when_no_priority_source_present(self):
        # Neither resume_pdf nor linkedin in group; any non-null value is used
        csv = _c(DataSource.CSV, email="alice@csv.com")
        result = MergeEngine().merge(_group(csv))
        assert result.email == "alice@csv.com"

    def test_all_sources_missing_returns_none(self):
        a = _c(DataSource.CSV, name=None)
        b = _c(DataSource.JSON, name=None)
        result = MergeEngine().merge(_group(a, b))
        assert result.name is None

    def test_blank_string_treated_as_missing(self):
        # Whitespace-only strings should be skipped
        pdf = _c(DataSource.RESUME_PDF, name="   ")
        csv = _c(DataSource.CSV, name="Alice")
        result = MergeEngine().merge(_group(pdf, csv))
        assert result.name == "Alice"

    def test_custom_priority_order_respected(self):
        e = _engine_with_rules(
            name={"priority": ["csv", "resume_pdf"], "strategy": "priority"}
        )
        pdf = _c(DataSource.RESUME_PDF, name="Name from PDF")
        csv = _c(DataSource.CSV, name="Name from CSV")
        result = e.merge(_group(pdf, csv))
        assert result.name == "Name from CSV"


# ---------------------------------------------------------------------------
# Most-complete strategy
# ---------------------------------------------------------------------------


class TestMostCompleteStrategy:
    def test_longer_summary_wins(self):
        e = _engine_with_rules(
            summary={"priority": ["csv"], "strategy": "most_complete"}
        )
        short = _c(DataSource.CSV, summary="Short summary.")
        long_ = _c(DataSource.JSON, summary="A much longer and more detailed summary about the candidate.")
        result = e.merge(_group(short, long_))
        assert result.summary == long_.summary

    def test_equal_length_returns_a_value(self):
        e = _engine_with_rules(
            summary={"priority": ["csv"], "strategy": "most_complete"}
        )
        a = _c(DataSource.CSV, summary="ABC")
        b = _c(DataSource.JSON, summary="XYZ")
        result = e.merge(_group(a, b))
        assert result.summary in ("ABC", "XYZ")

    def test_all_null_returns_none(self):
        e = _engine_with_rules(
            summary={"priority": ["csv"], "strategy": "most_complete"}
        )
        result = e.merge(_group(
            _c(DataSource.CSV, summary=None),
            _c(DataSource.JSON, summary=None),
        ))
        assert result.summary is None


# ---------------------------------------------------------------------------
# Union strategy — skills
# ---------------------------------------------------------------------------


class TestUnionSkills:
    E = MergeEngine()

    def test_skills_from_all_sources_combined(self):
        a = _c(DataSource.CSV, skills=["Python", "SQL"])
        b = _c(DataSource.JSON, skills=["Java", "Kubernetes"])
        result = self.E.merge(_group(a, b))
        assert set(result.skills) == {"Python", "SQL", "Java", "Kubernetes"}

    def test_exact_duplicate_skill_deduplicated(self):
        a = _c(DataSource.CSV, skills=["Python"])
        b = _c(DataSource.JSON, skills=["Python"])
        result = self.E.merge(_group(a, b))
        assert result.skills.count("Python") == 1

    def test_case_insensitive_skill_dedup(self):
        a = _c(DataSource.CSV, skills=["Python"])
        b = _c(DataSource.JSON, skills=["python", "PYTHON"])
        result = self.E.merge(_group(a, b))
        assert len([s for s in result.skills if s.lower() == "python"]) == 1

    def test_higher_priority_skills_come_first(self):
        # resume_pdf > csv in default skill priority
        pdf = _c(DataSource.RESUME_PDF, skills=["Python"])
        csv = _c(DataSource.CSV, skills=["Java"])
        result = self.E.merge(_group(pdf, csv))
        assert result.skills[0] == "Python"

    def test_empty_skills_list_ignored(self):
        a = _c(DataSource.CSV, skills=[])
        b = _c(DataSource.JSON, skills=["Go"])
        result = self.E.merge(_group(a, b))
        assert result.skills == ["Go"]


# ---------------------------------------------------------------------------
# Union strategy — links
# ---------------------------------------------------------------------------


class TestUnionLinks:
    E = MergeEngine()

    def test_links_from_all_sources_combined(self):
        lnk1 = Link(url="https://github.com/alice", label="GitHub")
        lnk2 = Link(url="https://linkedin.com/in/alice", label="LinkedIn")
        a = _c(DataSource.CSV, links=[lnk1])
        b = _c(DataSource.JSON, links=[lnk2])
        result = self.E.merge(_group(a, b))
        assert len(result.links) == 2

    def test_duplicate_url_deduplicated(self):
        lnk = Link(url="https://github.com/alice", label="GitHub")
        a = _c(DataSource.CSV, links=[lnk])
        b = _c(DataSource.JSON, links=[Link(url="https://github.com/alice", label="GH")])
        result = self.E.merge(_group(a, b))
        assert len(result.links) == 1

    def test_url_dedup_case_insensitive(self):
        a = _c(DataSource.CSV, links=[Link(url="https://GitHub.com/alice")])
        b = _c(DataSource.JSON, links=[Link(url="https://github.com/alice")])
        result = self.E.merge(_group(a, b))
        assert len(result.links) == 1


# ---------------------------------------------------------------------------
# Union strategy — experience / education
# ---------------------------------------------------------------------------


class TestUnionStructuredLists:
    E = MergeEngine()

    def test_experience_from_all_sources_combined(self):
        exp1 = Experience(company="Acme", title="Engineer")
        exp2 = Experience(company="BigCo", title="Lead")
        a = _c(DataSource.RESUME_PDF, experience=[exp1])
        b = _c(DataSource.JSON, experience=[exp2])
        result = self.E.merge(_group(a, b))
        assert len(result.experience) == 2

    def test_identical_experience_deduplicated(self):
        exp = Experience(company="Acme", title="Engineer")
        a = _c(DataSource.RESUME_PDF, experience=[exp])
        b = _c(DataSource.JSON, experience=[exp])
        result = self.E.merge(_group(a, b))
        assert len(result.experience) == 1

    def test_education_from_all_sources_combined(self):
        edu1 = Education(institution="MIT", degree="BSc")
        edu2 = Education(institution="Stanford", degree="MSc")
        a = _c(DataSource.RESUME_PDF, education=[edu1])
        b = _c(DataSource.JSON, education=[edu2])
        result = self.E.merge(_group(a, b))
        assert len(result.education) == 2


# ---------------------------------------------------------------------------
# Extra fields
# ---------------------------------------------------------------------------


class TestExtraFields:
    def test_extra_fields_merged_from_all_sources(self):
        a = _c(DataSource.CSV, extra_fields={"key_a": "val_a"})
        b = _c(DataSource.JSON, extra_fields={"key_b": "val_b"})
        result = MergeEngine().merge(_group(a, b))
        assert result.extra_fields["key_a"] == "val_a"
        assert result.extra_fields["key_b"] == "val_b"

    def test_higher_priority_overwrites_lower(self):
        # In _merge_extra, candidates are iterated in reverse so higher-priority
        # source (earlier in list) is applied last → wins
        pdf = _c(DataSource.RESUME_PDF, extra_fields={"score": "HIGH"})
        csv = _c(DataSource.CSV, extra_fields={"score": "LOW"})
        result = MergeEngine().merge(_group(pdf, csv))
        assert result.extra_fields["score"] == "HIGH"


# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_non_empty_none_is_empty(self):
        assert _non_empty(None) is False

    def test_non_empty_blank_string_is_empty(self):
        assert _non_empty("   ") is False

    def test_non_empty_string_is_non_empty(self):
        assert _non_empty("hello") is True

    def test_non_empty_zero_is_non_empty(self):
        assert _non_empty(0) is True

    def test_sort_by_priority_orders_correctly(self):
        pdf = _c(DataSource.RESUME_PDF)
        csv = _c(DataSource.CSV)
        json_ = _c(DataSource.JSON)
        ordered = _sort_by_priority([csv, json_, pdf], ["resume_pdf", "json", "csv"])
        assert ordered[0].source == DataSource.RESUME_PDF
        assert ordered[1].source == DataSource.JSON
        assert ordered[2].source == DataSource.CSV

    def test_sort_by_priority_unknown_source_goes_last(self):
        unk = _c(DataSource.UNKNOWN)
        csv = _c(DataSource.CSV)
        ordered = _sort_by_priority([unk, csv], ["csv"])
        assert ordered[0].source == DataSource.CSV

    def test_already_seen_skills_case_insensitive(self):
        assert _already_seen("python", ["Python", "SQL"], "skills") is True

    def test_already_seen_skills_different(self):
        assert _already_seen("Go", ["Python", "SQL"], "skills") is False

    def test_already_seen_links_by_url(self):
        seen = [Link(url="https://github.com/alice")]
        assert _already_seen(Link(url="https://github.com/alice"), seen, "links") is True

    def test_already_seen_links_url_case_insensitive(self):
        seen = [Link(url="https://GITHUB.com/alice")]
        assert _already_seen(Link(url="https://github.com/alice"), seen, "links") is True

    def test_already_seen_experience_structural(self):
        exp = Experience(company="Acme")
        assert _already_seen(exp, [exp], "experience") is True

    def test_already_seen_experience_different(self):
        e1 = Experience(company="Acme")
        e2 = Experience(company="BigCo")
        assert _already_seen(e2, [e1], "experience") is False


# ---------------------------------------------------------------------------
# _normalize_degree
# ---------------------------------------------------------------------------


class TestNormalizeDegree:
    def test_be_abbreviated(self):
        assert _normalize_degree("B.E") == "bachelor of engineering"

    def test_be_abbreviated_with_trailing_period(self):
        assert _normalize_degree("B.E.") == "bachelor of engineering"

    def test_be_in_field(self):
        assert _normalize_degree("B.E in Computer Science") == "bachelor of engineering"

    def test_bachelor_of_engineering_expanded(self):
        # Leading token "bachelor" is not in canonical → returns cleaned string
        result = _normalize_degree("Bachelor of Engineering (B.E.)")
        assert "bachelor" in result and "engineering" in result

    def test_btech(self):
        assert _normalize_degree("B.Tech") == "bachelor of technology"

    def test_mtech(self):
        assert _normalize_degree("M.Tech") == "master of technology"

    def test_ms(self):
        assert _normalize_degree("M.S.") == "master of science"

    def test_phd(self):
        assert _normalize_degree("Ph.D.") == "doctor of philosophy"

    def test_none_returns_empty(self):
        assert _normalize_degree(None) == ""

    def test_empty_string_returns_empty(self):
        assert _normalize_degree("") == ""


# ---------------------------------------------------------------------------
# _experience_similar
# ---------------------------------------------------------------------------


class TestExperienceSimilar:
    def test_exact_match(self):
        a = Experience(company="Infosys Springboard", title="AI/ML Intern")
        b = Experience(company="Infosys Springboard", title="AI/ML Intern")
        assert _experience_similar(a, b) is True

    def test_different_companies_rejected(self):
        a = Experience(company="Infosys Springboard", title="AI/ML Intern")
        b = Experience(company="Jaaji Technologies",  title="AI/ML Intern")
        assert _experience_similar(a, b) is False

    def test_different_titles_rejected(self):
        a = Experience(company="Acme", title="Software Engineer")
        b = Experience(company="Acme", title="Product Manager")
        assert _experience_similar(a, b) is False

    def test_no_company_no_title_returns_false(self):
        a = Experience(description="did things")
        b = Experience(description="did other things")
        assert _experience_similar(a, b) is False

    def test_company_only_match(self):
        a = Experience(company="Acme")
        b = Experience(company="Acme")
        assert _experience_similar(a, b) is True

    def test_title_only_match(self):
        a = Experience(title="ML Intern")
        b = Experience(title="ML Intern")
        assert _experience_similar(a, b) is True


# ---------------------------------------------------------------------------
# _education_similar
# ---------------------------------------------------------------------------


class TestEducationSimilar:
    def test_exact_match(self):
        a = Education(institution="IIT Bombay", degree="B.Tech CS")
        b = Education(institution="IIT Bombay", degree="B.Tech CS")
        assert _education_similar(a, b) is True

    def test_be_and_bachelor_of_engineering_same_institution(self):
        a = Education(institution="IIT Bombay", degree="B.E in Computer Science")
        b = Education(institution="IIT Bombay", degree="Bachelor of Engineering (B.E.)")
        assert _education_similar(a, b) is True

    def test_different_institutions_rejected(self):
        a = Education(institution="MIT",      degree="BSc")
        b = Education(institution="Stanford", degree="BSc")
        assert _education_similar(a, b) is False

    def test_no_institution_no_degree_returns_false(self):
        a = Education(field_of_study="CS")
        b = Education(field_of_study="CS")
        assert _education_similar(a, b) is False

    def test_institution_only_match(self):
        a = Education(institution="MIT")
        b = Education(institution="MIT")
        assert _education_similar(a, b) is True


# ---------------------------------------------------------------------------
# _find_similar_index
# ---------------------------------------------------------------------------


class TestFindSimilarIndex:
    def test_experience_match_returns_index(self):
        seen = [Experience(company="Acme", title="Engineer")]
        incoming = Experience(company="Acme", title="Engineer")
        assert _find_similar_index(incoming, seen, "experience") == 0

    def test_experience_no_match_returns_none(self):
        seen = [Experience(company="Acme")]
        incoming = Experience(company="BigCo")
        assert _find_similar_index(incoming, seen, "experience") is None

    def test_education_match_returns_index(self):
        seen = [Education(institution="MIT", degree="BSc")]
        incoming = Education(institution="MIT", degree="BSc")
        assert _find_similar_index(incoming, seen, "education") == 0

    def test_skills_field_always_returns_none(self):
        # _find_similar_index only handles experience/education
        assert _find_similar_index("Python", ["Python"], "skills") is None


# ---------------------------------------------------------------------------
# Fuzzy experience deduplication (integration via MergeEngine)
# ---------------------------------------------------------------------------


class TestFuzzyExperienceDedup:
    E = MergeEngine()

    def test_exact_duplicate_merged_to_one(self):
        exp = Experience(company="Infosys Springboard", title="AI/ML Intern")
        a = _c(DataSource.RESUME_PDF, experience=[exp])
        b = _c(DataSource.CSV,        experience=[exp])
        result = self.E.merge(_group(a, b))
        assert len(result.experience) == 1

    def test_merged_entry_fills_description_from_second_source(self):
        exp1 = Experience(company="Infosys Springboard", title="AI/ML Intern")
        exp2 = Experience(company="Infosys Springboard", title="AI/ML Intern",
                          description="Trained NLP models using Python and TensorFlow.")
        a = _c(DataSource.RESUME_PDF, experience=[exp1])
        b = _c(DataSource.CSV,        experience=[exp2])
        result = self.E.merge(_group(a, b))
        assert len(result.experience) == 1
        assert "NLP" in (result.experience[0].description or "")

    def test_merged_entry_preserves_location(self):
        exp1 = Experience(company="Acme", title="Engineer", location="Bangalore")
        exp2 = Experience(company="Acme", title="Engineer")
        a = _c(DataSource.RESUME_PDF, experience=[exp1])
        b = _c(DataSource.CSV,        experience=[exp2])
        result = self.E.merge(_group(a, b))
        assert len(result.experience) == 1
        assert result.experience[0].location == "Bangalore"

    def test_different_companies_kept_separate(self):
        exp1 = Experience(company="Infosys Springboard", title="AI/ML Intern")
        exp2 = Experience(company="Jaaji Technologies",  title="ML Intern")
        a = _c(DataSource.RESUME_PDF, experience=[exp1])
        b = _c(DataSource.CSV,        experience=[exp2])
        result = self.E.merge(_group(a, b))
        assert len(result.experience) == 2

    def test_longer_description_wins_over_shorter(self):
        short = Experience(company="Acme", title="Engineer",
                           description="Built services.")
        long_ = Experience(company="Acme", title="Engineer",
                           description="Built high-throughput backend services "
                                       "using Python, FastAPI, and PostgreSQL.")
        a = _c(DataSource.RESUME_PDF, experience=[short])
        b = _c(DataSource.CSV,        experience=[long_])
        result = self.E.merge(_group(a, b))
        assert len(result.experience) == 1
        desc = result.experience[0].description or ""
        assert "FastAPI" in desc

    def test_duration_filled_from_second_source(self):
        exp1 = Experience(company="Acme", title="Engineer")
        exp2 = Experience(company="Acme", title="Engineer",
                          duration=DateRange(start="2021-01", end="2023-06"))
        a = _c(DataSource.RESUME_PDF, experience=[exp1])
        b = _c(DataSource.CSV,        experience=[exp2])
        result = self.E.merge(_group(a, b))
        assert len(result.experience) == 1
        assert result.experience[0].duration is not None
        assert result.experience[0].duration.start == "2021-01"


# ---------------------------------------------------------------------------
# Fuzzy education deduplication (integration via MergeEngine)
# ---------------------------------------------------------------------------


class TestFuzzyEducationDedup:
    E = MergeEngine()

    def test_identical_degree_merged_to_one(self):
        edu = Education(institution="IIT Bombay", degree="B.Tech CS")
        a = _c(DataSource.RESUME_PDF, education=[edu])
        b = _c(DataSource.CSV,        education=[edu])
        result = self.E.merge(_group(a, b))
        assert len(result.education) == 1

    def test_be_abbreviation_variants_merged(self):
        edu1 = Education(institution="IIT Bombay",
                         degree="B.E in Computer Science", gpa=8.5)
        edu2 = Education(institution="IIT Bombay",
                         degree="Bachelor of Engineering (B.E.)")
        a = _c(DataSource.RESUME_PDF, education=[edu1])
        b = _c(DataSource.CSV,        education=[edu2])
        result = self.E.merge(_group(a, b))
        assert len(result.education) == 1

    def test_merged_education_preserves_gpa(self):
        edu1 = Education(institution="BITS Pilani", degree="B.Tech", gpa=8.5)
        edu2 = Education(institution="BITS Pilani", degree="Bachelor of Technology")
        a = _c(DataSource.RESUME_PDF, education=[edu1])
        b = _c(DataSource.CSV,        education=[edu2])
        result = self.E.merge(_group(a, b))
        assert len(result.education) == 1
        assert result.education[0].gpa == 8.5

    def test_merged_education_fills_field_of_study(self):
        edu1 = Education(institution="MIT", degree="B.Sc")
        edu2 = Education(institution="MIT", degree="B.Sc",
                         field_of_study="Computer Science")
        a = _c(DataSource.RESUME_PDF, education=[edu1])
        b = _c(DataSource.CSV,        education=[edu2])
        result = self.E.merge(_group(a, b))
        assert len(result.education) == 1
        assert result.education[0].field_of_study == "Computer Science"

    def test_different_institutions_kept_separate(self):
        edu1 = Education(institution="MIT",      degree="BSc")
        edu2 = Education(institution="Stanford", degree="BSc")
        a = _c(DataSource.RESUME_PDF, education=[edu1])
        b = _c(DataSource.CSV,        education=[edu2])
        result = self.E.merge(_group(a, b))
        assert len(result.education) == 2

    def test_duration_filled_from_second_source(self):
        edu1 = Education(institution="MIT", degree="BSc")
        edu2 = Education(institution="MIT", degree="BSc",
                         duration=DateRange(start="2018", end="2022"))
        a = _c(DataSource.RESUME_PDF, education=[edu1])
        b = _c(DataSource.CSV,        education=[edu2])
        result = self.E.merge(_group(a, b))
        assert len(result.education) == 1
        assert result.education[0].duration is not None
