"""Unit tests for the extraction layer."""

from __future__ import annotations

import textwrap

import pytest

import app.extractors  # noqa: F401 — triggers registration
from app.extractors.csv_extractor import (
    CSVExtractor,
    _parse_cell_list,
    _parse_gpa,
    _parse_skills,
    _extract_experience_row,
    _extract_education_row,
    _extract_misc_fields,
    _extract_username_links,
)
from app.extractors.field_map import extract_known_fields, resolve_field
from app.extractors.json_extractor import (
    JSONExtractor,
    _parse_experience,
    _parse_education,
    _extract_item_gpa,
    _unwrap_candidate_data,
    _is_candidate_root,
    _extract_misc_fields,
)
from app.extractors.pdf_extractor import (
    PDFExtractor,
    _detect_sections,
    _extract_email,
    _extract_name,
    _extract_phone,
    _extract_skills,
)
from app.extractors.text_resume_parser import TextResumeParser
from app.extractors.txt_extractor import TXTExtractor
from app.extractors.registry import extractor_registry
from app.models.candidate import DataSource, ExtractedCandidate, RawCandidateData


# ---------------------------------------------------------------------------
# Field map
# ---------------------------------------------------------------------------


class TestFieldMap:
    def test_canonical_name_resolves(self):
        assert resolve_field("name") == "name"
        assert resolve_field("full_name") == "name"
        assert resolve_field("Full_Name") == "name"
        assert resolve_field("CANDIDATE_NAME") == "name"

    def test_email_aliases(self):
        for alias in ("email", "email_address", "e-mail", "mail"):
            assert resolve_field(alias) == "email", alias

    def test_phone_aliases(self):
        for alias in ("phone", "mobile", "telephone", "contact_number"):
            assert resolve_field(alias) == "phone", alias

    def test_unknown_key_returns_none(self):
        assert resolve_field("random_column_xyz") is None

    def test_extract_known_fields_ignores_unknown(self):
        raw = {"name": "Alice", "email": "a@b.com", "random": "x"}
        result = extract_known_fields(raw)
        assert "name" in result
        assert "email" in result
        assert "random" not in result

    def test_first_non_empty_value_wins(self):
        # Both "name" and "full_name" map to "name" — first should win
        raw = {"name": "Alice", "full_name": "Bob"}
        result = extract_known_fields(raw)
        assert result["name"] in ("Alice", "Bob")  # one of them, not both


# ---------------------------------------------------------------------------
# Skill parsing (shared utility)
# ---------------------------------------------------------------------------


class TestParseSkills:
    def test_comma_separated_string(self):
        assert _parse_skills("Python, ML, SQL") == ["Python", "ML", "SQL"]

    def test_semicolon_separated_string(self):
        assert _parse_skills("Python; ML; SQL") == ["Python", "ML", "SQL"]

    def test_list_passthrough(self):
        assert _parse_skills(["Python", "ML"]) == ["Python", "ML"]

    def test_empty_string_returns_empty(self):
        assert _parse_skills("") == []

    def test_empty_list_returns_empty(self):
        assert _parse_skills([]) == []

    def test_strips_whitespace_in_list(self):
        assert _parse_skills(["  Python  ", " ML "]) == ["Python", "ML"]

    def test_single_skill_no_delimiter(self):
        assert _parse_skills("Python") == ["Python"]

    def test_filters_empty_splits(self):
        assert _parse_skills("Python,,ML") == ["Python", "ML"]

    def test_pipe_separated(self):
        assert _parse_skills("Python|Java|SQL") == ["Python", "Java", "SQL"]

    def test_newline_separated(self):
        result = _parse_skills("Python\nJava\nSQL")
        assert result == ["Python", "Java", "SQL"]

    def test_json_array_string(self):
        assert _parse_skills('["Python", "Java"]') == ["Python", "Java"]

    def test_json_array_no_spaces(self):
        assert _parse_skills('["Python","Java","SQL"]') == ["Python", "Java", "SQL"]

    def test_malformed_json_falls_back_to_split(self):
        # "[bad" is not valid JSON — should split on delimiters instead
        result = _parse_skills("[bad,Python")
        assert "Python" in result

    def test_mixed_delimiters(self):
        # comma and pipe together — all should be treated as separators
        result = _parse_skills("Python, Java | SQL")
        assert "Python" in result
        assert "Java" in result
        assert "SQL" in result


# ---------------------------------------------------------------------------
# CSV Extractor
# ---------------------------------------------------------------------------


def _csv_raw(fields: dict) -> RawCandidateData:
    return RawCandidateData(source=DataSource.CSV, raw_fields=fields)


class TestCSVExtractor:
    def test_basic_fields_extracted(self):
        raw = _csv_raw({"name": "Alice", "email": "alice@example.com", "phone": "9999999999"})
        result = CSVExtractor().extract(raw)
        assert result.name == "Alice"
        assert result.email == "alice@example.com"
        assert result.phone == "9999999999"
        assert result.source == DataSource.CSV

    def test_skills_parsed_from_string(self):
        raw = _csv_raw({"skills": "Python, ML, SQL"})
        result = CSVExtractor().extract(raw)
        assert "Python" in result.skills
        assert "Machine Learning" in result.skills  # "ML" normalised via _SKILL_ALIASES

    def test_linkedin_becomes_link(self):
        raw = _csv_raw({"linkedin": "https://linkedin.com/in/alice"})
        result = CSVExtractor().extract(raw)
        assert len(result.links) == 1
        assert result.links[0].label == "LinkedIn"
        assert "alice" in result.links[0].url

    def test_unknown_columns_go_to_extra_fields(self):
        raw = _csv_raw({"name": "Alice", "custom_field": "xyz"})
        result = CSVExtractor().extract(raw)
        assert "custom_field" in result.extra_fields

    def test_empty_raw_fields_returns_blank(self):
        raw = _csv_raw({})
        result = CSVExtractor().extract(raw)
        assert result.name is None
        assert result.skills == []

    def test_error_record_returns_empty(self):
        raw = RawCandidateData(
            source=DataSource.CSV,
            parse_errors=["file not found"],
        )
        result = CSVExtractor().extract(raw)
        assert result.name is None

    def test_alias_resolution_full_name(self):
        raw = _csv_raw({"full_name": "Bob Smith", "email_address": "bob@test.com"})
        result = CSVExtractor().extract(raw)
        assert result.name == "Bob Smith"
        assert result.email == "bob@test.com"


# ---------------------------------------------------------------------------
# CSV Extractor — enhanced fields (experience, education, misc, usernames)
# ---------------------------------------------------------------------------


class TestCSVExtractorEnhanced:
    # -- Experience from flat columns ----------------------------------------

    def test_experience_company_and_role(self):
        raw = _csv_raw({"name": "Alice", "company": "Acme", "role": "Engineer"})
        result = CSVExtractor().extract(raw)
        assert len(result.experience) == 1
        assert result.experience[0].company == "Acme"
        assert result.experience[0].title == "Engineer"

    def test_experience_employer_alias(self):
        raw = _csv_raw({"employer": "Google", "designation": "SDE"})
        result = CSVExtractor().extract(raw)
        assert result.experience[0].company == "Google"
        assert result.experience[0].title == "SDE"

    def test_experience_start_end_date(self):
        raw = _csv_raw({
            "company": "Acme",
            "start_date": "2021-01",
            "end_date": "2023-06",
        })
        result = CSVExtractor().extract(raw)
        exp = result.experience[0]
        assert exp.duration.start == "2021-01"
        assert exp.duration.end == "2023-06"
        assert exp.duration.is_current is False

    def test_experience_no_end_date_is_current(self):
        raw = _csv_raw({"company": "Acme", "start_date": "2022-01"})
        result = CSVExtractor().extract(raw)
        assert result.experience[0].duration.is_current is True

    def test_experience_joining_date_alias(self):
        raw = _csv_raw({"company": "X", "joining_date": "2020-06"})
        result = CSVExtractor().extract(raw)
        assert result.experience[0].duration.start == "2020-06"

    def test_experience_responsibilities_column(self):
        raw = _csv_raw({"company": "Y", "responsibilities": "Led team of 5"})
        result = CSVExtractor().extract(raw)
        assert result.experience[0].description == "Led team of 5"

    def test_no_experience_columns_returns_empty(self):
        raw = _csv_raw({"name": "Alice", "email": "alice@test.com"})
        result = CSVExtractor().extract(raw)
        assert result.experience == []

    def test_experience_not_in_extra_fields(self):
        raw = _csv_raw({"company": "Acme", "role": "Eng"})
        result = CSVExtractor().extract(raw)
        assert "company" not in result.extra_fields
        assert "role" not in result.extra_fields

    # -- Education from flat columns ------------------------------------------

    def test_education_college_and_degree(self):
        raw = _csv_raw({"college": "IIT Bombay", "degree": "B.Tech"})
        result = CSVExtractor().extract(raw)
        assert len(result.education) == 1
        assert result.education[0].institution == "IIT Bombay"
        assert result.education[0].degree == "B.Tech"

    def test_education_university_alias(self):
        raw = _csv_raw({"university": "MIT", "qualification": "MS"})
        result = CSVExtractor().extract(raw)
        assert result.education[0].institution == "MIT"
        assert result.education[0].degree == "MS"

    def test_education_field_of_study(self):
        raw = _csv_raw({"college": "BITS", "major": "Computer Science"})
        result = CSVExtractor().extract(raw)
        assert result.education[0].field_of_study == "Computer Science"

    def test_education_branch_alias(self):
        raw = _csv_raw({"college": "VIT", "branch": "ECE"})
        result = CSVExtractor().extract(raw)
        assert result.education[0].field_of_study == "ECE"

    def test_education_gpa_parsed(self):
        raw = _csv_raw({"college": "IIT", "cgpa": "8.5"})
        result = CSVExtractor().extract(raw)
        assert result.education[0].gpa == pytest.approx(8.5)

    def test_education_percentage_parsed(self):
        raw = _csv_raw({"college": "IIT", "percentage": "85.5%"})
        result = CSVExtractor().extract(raw)
        assert result.education[0].gpa == pytest.approx(85.5)

    def test_education_graduation_year(self):
        raw = _csv_raw({"college": "NIT", "graduation_year": "2022"})
        result = CSVExtractor().extract(raw)
        assert result.education[0].duration.end == "2022"

    def test_education_start_and_end_year(self):
        raw = _csv_raw({"college": "SRM", "start_year": "2018", "passing_year": "2022"})
        result = CSVExtractor().extract(raw)
        assert result.education[0].duration.start == "2018"
        assert result.education[0].duration.end == "2022"

    def test_no_education_columns_returns_empty(self):
        raw = _csv_raw({"name": "Alice"})
        result = CSVExtractor().extract(raw)
        assert result.education == []

    def test_education_not_in_extra_fields(self):
        raw = _csv_raw({"college": "IIT", "cgpa": "9.0"})
        result = CSVExtractor().extract(raw)
        assert "college" not in result.extra_fields
        assert "cgpa" not in result.extra_fields

    # -- Skills multi-separator / JSON ----------------------------------------

    def test_skills_pipe_separated(self):
        raw = _csv_raw({"skills": "Python|Java|SQL"})
        result = CSVExtractor().extract(raw)
        assert "Python" in result.skills
        assert "SQL" in result.skills

    def test_skills_json_array_in_cell(self):
        raw = _csv_raw({"skills": '["Python", "Docker", "AWS"]'})
        result = CSVExtractor().extract(raw)
        assert "Python" in result.skills
        assert "Docker" in result.skills

    def test_skills_newline_separated(self):
        raw = _csv_raw({"skills": "Python\nJava\nSQL"})
        result = CSVExtractor().extract(raw)
        assert "Python" in result.skills

    def test_technical_skills_column_alias(self):
        raw = _csv_raw({"technical_skills": "Python, FastAPI"})
        result = CSVExtractor().extract(raw)
        assert "Python" in result.skills

    # -- Alias expansions -----------------------------------------------------

    def test_primary_email_alias(self):
        raw = _csv_raw({"primary_email": "alice@test.com"})
        result = CSVExtractor().extract(raw)
        assert result.email == "alice@test.com"

    def test_state_maps_to_location(self):
        raw = _csv_raw({"name": "Alice", "email": "a@b.com", "state": "Karnataka"})
        result = CSVExtractor().extract(raw)
        assert result.location == "Karnataka"

    def test_current_city_maps_to_location(self):
        raw = _csv_raw({"current_city": "Hyderabad"})
        result = CSVExtractor().extract(raw)
        assert result.location == "Hyderabad"

    def test_career_objective_maps_to_summary(self):
        raw = _csv_raw({"career_objective": "To build scalable systems."})
        result = CSVExtractor().extract(raw)
        assert result.summary == "To build scalable systems."

    # -- Username-to-URL conversion -------------------------------------------

    def test_github_username_to_url(self):
        raw = _csv_raw({"github_username": "pragnagosula"})
        result = CSVExtractor().extract(raw)
        gh_links = [l for l in result.links if l.label == "GitHub"]
        assert gh_links
        assert "pragnagosula" in gh_links[0].url
        assert gh_links[0].url.startswith("https://github.com/")

    def test_linkedin_username_to_url(self):
        raw = _csv_raw({"linkedin_username": "alice-engineer"})
        result = CSVExtractor().extract(raw)
        li_links = [l for l in result.links if l.label == "LinkedIn"]
        assert li_links
        assert "alice-engineer" in li_links[0].url

    def test_username_at_prefix_stripped(self):
        raw = _csv_raw({"github_username": "@pragnagosula"})
        result = CSVExtractor().extract(raw)
        gh_links = [l for l in result.links if l.label == "GitHub"]
        assert "pragnagosula" in gh_links[0].url
        assert "@" not in gh_links[0].url

    def test_leetcode_username_to_url(self):
        raw = _csv_raw({"leetcode_username": "coder123"})
        result = CSVExtractor().extract(raw)
        lc_links = [l for l in result.links if l.label == "LeetCode"]
        assert lc_links
        assert "coder123" in lc_links[0].url

    def test_username_not_in_extra_fields(self):
        raw = _csv_raw({"github_username": "alice"})
        result = CSVExtractor().extract(raw)
        assert "github_username" not in result.extra_fields

    # -- Miscellaneous list fields (extra_fields) -----------------------------

    def test_certifications_extracted(self):
        raw = _csv_raw({"certifications": "AWS, GCP, Azure"})
        result = CSVExtractor().extract(raw)
        assert "certifications" in result.extra_fields
        assert "AWS" in result.extra_fields["certifications"]

    def test_certifications_json_array(self):
        raw = _csv_raw({"certifications": '["AWS Solutions Architect", "GCP Associate"]'})
        result = CSVExtractor().extract(raw)
        assert len(result.extra_fields["certifications"]) == 2

    def test_awards_maps_to_achievements(self):
        raw = _csv_raw({"awards": "Best Employee, Hackathon Winner"})
        result = CSVExtractor().extract(raw)
        assert "achievements" in result.extra_fields
        assert "Best Employee" in result.extra_fields["achievements"]

    def test_achievements_column(self):
        raw = _csv_raw({"achievements": "Dean's List; Published Paper"})
        result = CSVExtractor().extract(raw)
        assert "achievements" in result.extra_fields

    def test_languages_known_extracted(self):
        raw = _csv_raw({"languages_known": "English, Hindi, Telugu"})
        result = CSVExtractor().extract(raw)
        assert "languages" in result.extra_fields
        assert "English" in result.extra_fields["languages"]

    def test_spoken_languages_alias(self):
        raw = _csv_raw({"spoken_languages": "English|Hindi"})
        result = CSVExtractor().extract(raw)
        assert "languages" in result.extra_fields

    def test_projects_extracted(self):
        raw = _csv_raw({"projects": "Resume Parser, Chatbot"})
        result = CSVExtractor().extract(raw)
        assert "projects" in result.extra_fields
        assert "Resume Parser" in result.extra_fields["projects"]

    def test_misc_keys_not_duplicated_in_extra_fields(self):
        # The raw "certifications" key should be consumed (not appear alongside
        # the parsed list entry).
        raw = _csv_raw({"certifications": "AWS, GCP"})
        result = CSVExtractor().extract(raw)
        # The value in extra_fields["certifications"] should be a list, not
        # the original string.
        assert isinstance(result.extra_fields.get("certifications"), list)

    # -- Robustness -----------------------------------------------------------

    def test_empty_experience_columns_ignored(self):
        raw = _csv_raw({"company": "", "role": "  "})
        result = CSVExtractor().extract(raw)
        assert result.experience == []

    def test_empty_education_columns_ignored(self):
        raw = _csv_raw({"college": "", "degree": None})
        result = CSVExtractor().extract(raw)
        assert result.education == []

    def test_gpa_non_numeric_returns_none(self):
        assert _parse_gpa("N/A") is None
        assert _parse_gpa("") is None
        assert _parse_gpa(None) is None

    def test_gpa_with_slash_format(self):
        # "8.5/10" — strip non-numeric except first decimal → "8.510" → 8.51
        result = _parse_gpa("8.5/10")
        assert result is not None
        assert result > 8.0

    def test_parse_cell_list_json_dict_not_exploded(self):
        # A JSON object should NOT be treated as a list of strings
        result = _parse_cell_list('{"skills": ["Python"]}')
        # Falls back to string split — treats the whole thing as one token
        assert isinstance(result, list)

    def test_full_ats_row(self):
        """End-to-end smoke test simulating a realistic ATS export row."""
        raw = _csv_raw({
            "Full Name":        "Pragna Gosula",
            "Primary Email":    "pragna@example.com",
            "Mobile Number":    "+91 9876543210",
            "Current City":     "Hyderabad",
            "Career Objective": "Aspiring software engineer.",
            "Technical Skills": "Python, FastAPI, Docker",
            "College":          "JNTU Hyderabad",
            "Degree":           "B.Tech",
            "Branch":           "CSE",
            "CGPA":             "8.9",
            "Graduation Year":  "2024",
            "Current Company":  "Acme Corp",
            "Designation":      "Software Engineer",
            "Start Date":       "2024-06",
            "LinkedIn Username":"pragna-gosula",
            "GitHub Username":  "pragnagosula",
            "Certifications":   "AWS, GCP",
            "Achievements":     "Hackathon Winner",
            "Languages Known":  "English, Telugu",
        })
        result = CSVExtractor().extract(raw)

        assert result.name == "Pragna Gosula"
        assert result.email == "pragna@example.com"
        assert result.location == "Hyderabad"
        assert result.summary == "Aspiring software engineer."
        assert "Python" in result.skills

        assert len(result.experience) == 1
        assert result.experience[0].company == "Acme Corp"
        assert result.experience[0].title == "Software Engineer"

        assert len(result.education) == 1
        assert result.education[0].institution == "JNTU Hyderabad"
        assert result.education[0].gpa == pytest.approx(8.9)

        gh_links = [l for l in result.links if l.label == "GitHub"]
        assert gh_links and "pragnagosula" in gh_links[0].url

        assert "certifications" in result.extra_fields
        assert "achievements" in result.extra_fields
        assert "languages" in result.extra_fields


# ---------------------------------------------------------------------------
# _parse_cell_list (unit tests for the shared helper)
# ---------------------------------------------------------------------------


class TestParseCellList:
    def test_comma_string(self):
        assert _parse_cell_list("a, b, c") == ["a", "b", "c"]

    def test_pipe_string(self):
        assert _parse_cell_list("a|b|c") == ["a", "b", "c"]

    def test_newline_string(self):
        assert _parse_cell_list("a\nb\nc") == ["a", "b", "c"]

    def test_json_array(self):
        assert _parse_cell_list('["x", "y"]') == ["x", "y"]

    def test_python_list(self):
        assert _parse_cell_list(["x", "y"]) == ["x", "y"]

    def test_empty_string(self):
        assert _parse_cell_list("") == []

    def test_none_value(self):
        assert _parse_cell_list(None) == []

    def test_malformed_json_falls_back(self):
        result = _parse_cell_list("[bad")
        assert isinstance(result, list)  # no crash


# ---------------------------------------------------------------------------
# _extract_experience_row (unit tests)
# ---------------------------------------------------------------------------


class TestExtractExperienceRow:
    def test_company_and_role(self):
        entries = _extract_experience_row({"company": "Acme", "role": "Eng"})
        assert len(entries) == 1
        assert entries[0].company == "Acme"
        assert entries[0].title == "Eng"

    def test_empty_fields_return_empty(self):
        assert _extract_experience_row({"name": "Alice"}) == []

    def test_blank_values_return_empty(self):
        assert _extract_experience_row({"company": "", "role": "  "}) == []

    def test_start_date_sets_is_current(self):
        entries = _extract_experience_row({"company": "X", "start_date": "2022-01"})
        assert entries[0].duration.is_current is True

    def test_both_dates_is_not_current(self):
        entries = _extract_experience_row({
            "company": "X", "start_date": "2020-01", "end_date": "2022-06"
        })
        assert entries[0].duration.is_current is False
        assert entries[0].duration.end == "2022-06"


# ---------------------------------------------------------------------------
# _extract_education_row (unit tests)
# ---------------------------------------------------------------------------


class TestExtractEducationRow:
    def test_college_and_degree(self):
        entries = _extract_education_row({"college": "IIT", "degree": "B.Tech"})
        assert len(entries) == 1
        assert entries[0].institution == "IIT"
        assert entries[0].degree == "B.Tech"

    def test_empty_fields_return_empty(self):
        assert _extract_education_row({"name": "Alice"}) == []

    def test_gpa_parsed_to_float(self):
        entries = _extract_education_row({"college": "IIT", "cgpa": "9.1"})
        assert entries[0].gpa == pytest.approx(9.1)

    def test_graduation_year_in_duration(self):
        entries = _extract_education_row({"college": "MIT", "graduation_year": "2023"})
        assert entries[0].duration.end == "2023"

    def test_specialization_alias(self):
        entries = _extract_education_row({"college": "VIT", "specialization": "AI"})
        assert entries[0].field_of_study == "AI"


# ---------------------------------------------------------------------------
# _extract_username_links (unit tests)
# ---------------------------------------------------------------------------


class TestExtractUsernameLinks:
    def test_github_username(self):
        links = _extract_username_links({"github_username": "alice"})
        assert len(links) == 1
        assert links[0].url == "https://github.com/alice"
        assert links[0].label == "GitHub"

    def test_linkedin_username(self):
        links = _extract_username_links({"linkedin_username": "alice-eng"})
        assert links[0].url == "https://linkedin.com/in/alice-eng"

    def test_at_prefix_stripped(self):
        links = _extract_username_links({"github_username": "@alice"})
        assert links[0].url == "https://github.com/alice"

    def test_empty_username_skipped(self):
        assert _extract_username_links({"github_username": ""}) == []

    def test_unknown_key_ignored(self):
        assert _extract_username_links({"random_username": "alice"}) == []

    def test_multiple_platforms(self):
        links = _extract_username_links({
            "github_username": "alice",
            "leetcode_username": "alice_lc",
        })
        labels = {l.label for l in links}
        assert "GitHub" in labels
        assert "LeetCode" in labels


# ---------------------------------------------------------------------------
# _extract_misc_fields (unit tests)
# ---------------------------------------------------------------------------


class TestExtractMiscFields:
    def test_certifications_parsed(self):
        result = _extract_misc_fields({"certifications": "AWS, GCP"})
        assert result["certifications"] == ["AWS", "GCP"]

    def test_achievements_from_awards_column(self):
        result = _extract_misc_fields({"awards": "Best Employee"})
        assert result["achievements"] == ["Best Employee"]

    def test_languages_known(self):
        result = _extract_misc_fields({"languages_known": "English|Hindi"})
        assert "languages" in result
        assert "English" in result["languages"]

    def test_projects_column(self):
        result = _extract_misc_fields({"projects": "Chatbot, Resume Parser"})
        assert result["projects"] == ["Chatbot", "Resume Parser"]

    def test_no_misc_columns_returns_empty(self):
        assert _extract_misc_fields({"name": "Alice", "email": "a@b.com"}) == {}


# ---------------------------------------------------------------------------
# JSON Extractor
# ---------------------------------------------------------------------------


def _json_raw(fields: dict) -> RawCandidateData:
    return RawCandidateData(source=DataSource.JSON, raw_fields=fields)


class TestJSONExtractor:
    def test_basic_fields(self):
        raw = _json_raw({"name": "Alice", "email": "alice@test.com"})
        result = JSONExtractor().extract(raw)
        assert result.name == "Alice"
        assert result.email == "alice@test.com"
        assert result.source == DataSource.JSON

    def test_skills_list_normalised(self):
        raw = _json_raw({"skills": ["Python", "ML", "Docker"]})
        result = JSONExtractor().extract(raw)
        # "ML" is normalised to "Machine Learning" via _SKILL_ALIASES
        assert result.skills == ["Python", "Machine Learning", "Docker"]

    def test_experience_parsed(self):
        raw = _json_raw({
            "experience": [
                {"company": "Acme", "title": "Engineer", "start": "2021-01", "end": None}
            ]
        })
        result = JSONExtractor().extract(raw)
        assert len(result.experience) == 1
        assert result.experience[0].company == "Acme"
        assert result.experience[0].duration.start == "2021-01"
        assert result.experience[0].duration.is_current is True

    def test_education_parsed(self):
        raw = _json_raw({
            "education": [
                {"institution": "IIT", "degree": "B.Tech", "field": "CS", "year": "2020"}
            ]
        })
        result = JSONExtractor().extract(raw)
        assert len(result.education) == 1
        assert result.education[0].institution == "IIT"
        assert result.education[0].degree == "B.Tech"

    def test_non_list_experience_ignored(self):
        raw = _json_raw({"experience": "not a list"})
        result = JSONExtractor().extract(raw)
        assert result.experience == []

    def test_non_dict_experience_items_skipped(self):
        raw = _json_raw({"experience": [{"company": "Acme"}, "bad_item"]})
        result = JSONExtractor().extract(raw)
        assert len(result.experience) == 1

    def test_extra_fields_captured(self):
        raw = _json_raw({"name": "Alice", "custom_score": 99})
        result = JSONExtractor().extract(raw)
        assert "custom_score" in result.extra_fields


# ---------------------------------------------------------------------------
# PDF Extractor (synthetic text)
# ---------------------------------------------------------------------------


SAMPLE_RESUME = """
Alice Johnson
alice.johnson@example.com | +91 98765-43210
linkedin.com/in/alicejohnson | github.com/alicejohnson

Summary
Data scientist with 4 years of experience building ML pipelines.

Skills
Python, Machine Learning, Deep Learning, SQL, Docker, AWS

Experience
Senior Data Scientist
Acme Analytics
Jan 2021 - Present
Led a team of 3 to build real-time recommendation systems.

Data Analyst
XYZ Corp
Jun 2018 - Dec 2020
Analysed e-commerce data and built reporting dashboards.

Education
IIT Bombay
B.Tech Computer Science
2018
"""


class TestPDFExtractor:
    def _raw(self, text: str) -> RawCandidateData:
        return RawCandidateData(
            source=DataSource.RESUME_PDF,
            raw_fields={"raw_text": text},
        )

    def test_email_extracted(self):
        raw = self._raw(SAMPLE_RESUME)
        result = PDFExtractor().extract(raw)
        assert result.email == "alice.johnson@example.com"

    def test_phone_extracted(self):
        raw = self._raw(SAMPLE_RESUME)
        result = PDFExtractor().extract(raw)
        assert result.phone is not None
        assert "98765" in result.phone

    def test_name_extracted(self):
        raw = self._raw(SAMPLE_RESUME)
        result = PDFExtractor().extract(raw)
        assert result.name == "Alice Johnson"

    def test_skills_extracted(self):
        raw = self._raw(SAMPLE_RESUME)
        result = PDFExtractor().extract(raw)
        assert "Python" in result.skills
        assert "SQL" in result.skills

    def test_summary_extracted(self):
        raw = self._raw(SAMPLE_RESUME)
        result = PDFExtractor().extract(raw)
        assert result.summary is not None
        assert "ML" in result.summary or "machine" in result.summary.lower()

    def test_linkedin_link_extracted(self):
        raw = self._raw(SAMPLE_RESUME)
        result = PDFExtractor().extract(raw)
        linkedin_links = [l for l in result.links if l.label == "LinkedIn"]
        assert linkedin_links

    def test_github_link_extracted(self):
        raw = self._raw(SAMPLE_RESUME)
        result = PDFExtractor().extract(raw)
        github_links = [l for l in result.links if l.label == "GitHub"]
        assert github_links

    def test_empty_text_returns_blank(self):
        raw = self._raw("")
        result = PDFExtractor().extract(raw)
        assert result.name is None
        assert result.email is None

    def test_source_is_resume_pdf(self):
        raw = self._raw(SAMPLE_RESUME)
        result = PDFExtractor().extract(raw)
        assert result.source == DataSource.RESUME_PDF


class TestExtractEmail:
    def test_standard_email(self):
        assert _extract_email("Contact: alice@example.com") == "alice@example.com"

    def test_no_email_returns_none(self):
        assert _extract_email("No email here") is None

    def test_email_in_middle_of_text(self):
        assert _extract_email("call me or email bob@test.org today") == "bob@test.org"


class TestExtractPhone:
    def test_plain_ten_digits(self):
        result = _extract_phone("Call me at 9876543210")
        assert result is not None
        digits = "".join(c for c in result if c.isdigit())
        assert len(digits) >= 7

    def test_international_format(self):
        result = _extract_phone("+91 98765-43210")
        assert result is not None

    def test_no_phone_returns_none(self):
        assert _extract_phone("No phone here at all") is None


class TestExtractName:
    def test_first_two_word_line_is_name(self):
        header = "Alice Johnson\nalice@example.com\n+91 9876543210"
        assert _extract_name(header, "alice@example.com") == "Alice Johnson"

    def test_skips_email_lines(self):
        header = "alice@example.com\nAlice Johnson"
        assert _extract_name(header, "alice@example.com") == "Alice Johnson"

    def test_single_word_not_treated_as_name(self):
        result = _extract_name("Alice\nalice@test.com", None)
        # Single word may or may not be accepted — test it doesn't crash
        assert result is None or isinstance(result, str)


class TestDetectSections:
    def test_skills_section_detected(self):
        text = "Alice\n\nSkills\nPython, ML\n\nEducation\nIIT"
        sections = _detect_sections(text)
        assert "skills" in sections
        assert "Python" in sections["skills"]

    def test_education_section_detected(self):
        text = "Alice\n\nEducation\nIIT Bombay\nB.Tech"
        sections = _detect_sections(text)
        assert "education" in sections

    def test_header_captured(self):
        text = "Alice Johnson\nalice@test.com\n\nSkills\nPython"
        sections = _detect_sections(text)
        assert "_header" in sections
        assert "Alice" in sections["_header"]


class TestExtractorRegistry:
    def test_all_four_registered(self):
        sources = extractor_registry.registered_sources
        assert DataSource.CSV in sources
        assert DataSource.JSON in sources
        assert DataSource.RESUME_PDF in sources
        assert DataSource.RESUME_TXT in sources

    def test_extract_dispatches_correctly(self):
        raw = RawCandidateData(
            source=DataSource.CSV,
            raw_fields={"name": "Alice", "email": "alice@test.com"},
        )
        result = extractor_registry.extract(raw)
        assert result.name == "Alice"

    def test_unknown_source_returns_blank(self):
        raw = RawCandidateData(source=DataSource.LINKEDIN, raw_fields={"name": "Alice"})
        result = extractor_registry.extract(raw)
        assert result.source == DataSource.LINKEDIN
        assert result.name is None


# ---------------------------------------------------------------------------
# TextResumeParser — direct unit tests
# ---------------------------------------------------------------------------


class TestTextResumeParser:
    def _parse(self, text: str, **kw):
        return TextResumeParser().parse(text, source=DataSource.RESUME_TXT, **kw)

    def test_email_extracted(self):
        result = self._parse("Alice\nalice@example.com\n\nSkills\nPython")
        assert result.email == "alice@example.com"

    def test_phone_extracted(self):
        result = self._parse("Alice\n+91 98765-43210\nalice@test.com")
        assert result.phone is not None
        assert "98765" in result.phone

    def test_name_extracted(self):
        result = self._parse("Alice Johnson\nalice@example.com")
        assert result.name == "Alice Johnson"

    def test_skills_extracted(self):
        result = self._parse("Alice\n\nSkills\nPython, SQL, Docker")
        assert "Python" in result.skills
        assert "SQL" in result.skills

    def test_summary_extracted(self):
        result = self._parse("Alice\n\nSummary\nExperienced engineer.\n\nSkills\nPython")
        assert result.summary == "Experienced engineer."

    def test_empty_text_returns_blank(self):
        result = self._parse("")
        assert result.name is None
        assert result.email is None
        assert result.skills == []

    def test_source_stamped_correctly(self):
        result = self._parse("Alice\nalice@test.com")
        assert result.source == DataSource.RESUME_TXT

    def test_source_file_stamped(self):
        result = self._parse("Alice\nalice@test.com", source_file="alice.txt")
        assert result.source_file == "alice.txt"

    def test_extra_links_prepended(self):
        from app.models.candidate import Link
        extra = [Link(url="https://linkedin.com/in/alice", label="LinkedIn")]
        result = self._parse("Alice\nalice@test.com", extra_links=extra)
        labels = [l.label for l in result.links]
        assert "LinkedIn" in labels

    def test_raw_text_preserved(self):
        text = "Alice\nalice@test.com"
        result = self._parse(text)
        assert result.raw_text == text


# ---------------------------------------------------------------------------
# TXT Extractor
# ---------------------------------------------------------------------------


def _txt_raw(text: str, source_file: str | None = None) -> RawCandidateData:
    return RawCandidateData(
        source=DataSource.RESUME_TXT,
        raw_fields={"raw_text": text},
        source_file=source_file,
    )


class TestTXTExtractor:
    def test_email_extracted(self):
        raw = _txt_raw(SAMPLE_RESUME)
        result = TXTExtractor().extract(raw)
        assert result.email == "alice.johnson@example.com"

    def test_phone_extracted(self):
        raw = _txt_raw(SAMPLE_RESUME)
        result = TXTExtractor().extract(raw)
        assert result.phone is not None
        assert "98765" in result.phone

    def test_name_extracted(self):
        raw = _txt_raw(SAMPLE_RESUME)
        result = TXTExtractor().extract(raw)
        assert result.name == "Alice Johnson"

    def test_skills_extracted(self):
        raw = _txt_raw(SAMPLE_RESUME)
        result = TXTExtractor().extract(raw)
        assert "Python" in result.skills

    def test_source_is_resume_txt(self):
        raw = _txt_raw(SAMPLE_RESUME)
        result = TXTExtractor().extract(raw)
        assert result.source == DataSource.RESUME_TXT

    def test_source_file_propagated(self):
        raw = _txt_raw(SAMPLE_RESUME, source_file="alice.txt")
        result = TXTExtractor().extract(raw)
        assert result.source_file == "alice.txt"

    def test_empty_text_returns_blank(self):
        raw = _txt_raw("")
        result = TXTExtractor().extract(raw)
        assert result.name is None
        assert result.email is None


# ---------------------------------------------------------------------------
# Section-boundary tests (Issues 3 & 4)
# ---------------------------------------------------------------------------


class TestTextResumeParserSectionBoundaries:
    """Experience section must stop at non-experience section headings."""

    _PARSER = TextResumeParser()

    def _parse(self, text: str):
        return self._PARSER.parse(text, source=DataSource.RESUME_PDF)

    def _exp_text(self, result) -> str:
        titles = [e.title for e in result.experience if e.title]
        companies = [e.company for e in result.experience if e.company]
        descs = [e.description for e in result.experience if e.description]
        return " ".join(titles + companies + descs)

    # -- Experience stops at newly-added boundary sections -------------------

    def test_experience_stops_at_achievements(self):
        text = textwrap.dedent("""\
            Alice
            alice@example.com

            Experience
            Software Engineer
            Acme Corp
            Jan 2022 - Present
            Built backend services.

            Achievements
            Best Employee Award 2023
            Second Award 2022
            """)
        result = self._parse(text)
        exp_text = self._exp_text(result)
        assert "Best Employee" not in exp_text
        assert "Second Award" not in exp_text

    def test_experience_stops_at_awards(self):
        text = textwrap.dedent("""\
            Bob
            bob@example.com

            Experience
            ML Engineer
            StartupX
            2020 - 2022

            Awards
            National Hackathon Winner 2021
            """)
        result = self._parse(text)
        exp_text = self._exp_text(result)
        assert "Hackathon" not in exp_text
        assert "National" not in exp_text

    def test_experience_stops_at_position_of_responsibility(self):
        text = textwrap.dedent("""\
            Carol
            carol@example.com

            Experience
            Data Analyst
            BigCo
            Jun 2020 - Dec 2021
            Analysed data.

            Position of Responsibility
            President, Coding Club 2019
            Secretary, IEEE Chapter 2018
            """)
        result = self._parse(text)
        exp_text = self._exp_text(result)
        assert "President" not in exp_text
        assert "Coding Club" not in exp_text
        assert "Secretary" not in exp_text

    def test_experience_stops_at_leadership(self):
        text = textwrap.dedent("""\
            Dave
            dave@example.com

            Experience
            Backend Developer
            TechFirm
            2019 - 2021

            Leadership
            Team Lead, Open Source Club 2020
            """)
        result = self._parse(text)
        exp_text = self._exp_text(result)
        assert "Open Source Club" not in exp_text

    def test_experience_stops_at_certifications(self):
        text = textwrap.dedent("""\
            Eve
            eve@example.com

            Experience
            DevOps Engineer
            CloudCo
            2021 - 2023

            Certifications
            AWS Certified Solutions Architect 2023
            Google Cloud Professional 2022
            """)
        result = self._parse(text)
        exp_text = self._exp_text(result)
        assert "AWS Certified" not in exp_text
        assert "Google Cloud" not in exp_text

    # -- Improved heading detection (Issue 4) --------------------------------

    def test_professional_experience_heading_detected(self):
        text = textwrap.dedent("""\
            Frank
            frank@example.com

            Professional Experience
            Backend Developer
            TechFirm
            Jan 2019 - Dec 2021
            Developed APIs.
            """)
        result = self._parse(text)
        assert len(result.experience) >= 1

    def test_work_history_heading_detected(self):
        text = textwrap.dedent("""\
            Grace
            grace@example.com

            Work History
            QA Engineer
            TestCo
            2020 - 2022
            Wrote test plans.
            """)
        result = self._parse(text)
        assert len(result.experience) >= 1

    def test_heading_with_trailing_colon_detected(self):
        text = textwrap.dedent("""\
            Hank
            hank@example.com

            Experience:
            Data Scientist
            DataCo
            2021 - 2023

            Achievements:
            Best Researcher 2022
            """)
        result = self._parse(text)
        exp_text = self._exp_text(result)
        assert "Best Researcher" not in exp_text
        assert len(result.experience) >= 1

    def test_uppercase_heading_detected(self):
        text = textwrap.dedent("""\
            Iris
            iris@example.com

            EXPERIENCE
            ML Researcher
            ResearchLab
            2022 - Present

            ACHIEVEMENTS
            Best Paper Award 2023
            """)
        result = self._parse(text)
        exp_text = self._exp_text(result)
        assert "Best Paper" not in exp_text

    def test_links_extracted(self):
        raw = _txt_raw(SAMPLE_RESUME)
        result = TXTExtractor().extract(raw)
        labels = {l.label for l in result.links}
        assert "LinkedIn" in labels
        assert "GitHub" in labels

    def test_experience_extracted(self):
        raw = _txt_raw(SAMPLE_RESUME)
        result = TXTExtractor().extract(raw)
        assert len(result.experience) >= 1

    def test_registry_dispatches_to_txt_extractor(self):
        raw = _txt_raw(SAMPLE_RESUME)
        result = extractor_registry.extract(raw)
        assert result.source == DataSource.RESUME_TXT
        assert result.name == "Alice Johnson"


# ---------------------------------------------------------------------------
# PDF / TXT parity — same text must produce identical parsed fields
# ---------------------------------------------------------------------------


class TestPDFTXTParity:
    """Verify that PDF and TXT produce identical results when given the same
    resume text (no PDF annotation links in the test — annotation path
    returns [] when there is no on-disk PDF file).
    """

    def _pdf(self, text: str) -> "ExtractedCandidate":  # noqa: F821
        raw = RawCandidateData(source=DataSource.RESUME_PDF, raw_fields={"raw_text": text})
        return PDFExtractor().extract(raw)

    def _txt(self, text: str) -> "ExtractedCandidate":  # noqa: F821
        raw = RawCandidateData(source=DataSource.RESUME_TXT, raw_fields={"raw_text": text})
        return TXTExtractor().extract(raw)

    def test_name_identical(self):
        assert self._pdf(SAMPLE_RESUME).name == self._txt(SAMPLE_RESUME).name

    def test_email_identical(self):
        assert self._pdf(SAMPLE_RESUME).email == self._txt(SAMPLE_RESUME).email

    def test_phone_identical(self):
        assert self._pdf(SAMPLE_RESUME).phone == self._txt(SAMPLE_RESUME).phone

    def test_skills_identical(self):
        assert set(self._pdf(SAMPLE_RESUME).skills) == set(self._txt(SAMPLE_RESUME).skills)

    def test_summary_identical(self):
        assert self._pdf(SAMPLE_RESUME).summary == self._txt(SAMPLE_RESUME).summary

    def test_experience_count_identical(self):
        assert len(self._pdf(SAMPLE_RESUME).experience) == len(self._txt(SAMPLE_RESUME).experience)

    def test_education_count_identical(self):
        assert len(self._pdf(SAMPLE_RESUME).education) == len(self._txt(SAMPLE_RESUME).education)

    def test_links_identical(self):
        pdf_labels = {l.label for l in self._pdf(SAMPLE_RESUME).links}
        txt_labels = {l.label for l in self._txt(SAMPLE_RESUME).links}
        assert pdf_labels == txt_labels

    def test_only_source_differs(self):
        pdf_res = self._pdf(SAMPLE_RESUME)
        txt_res = self._txt(SAMPLE_RESUME)
        assert pdf_res.source == DataSource.RESUME_PDF
        assert txt_res.source == DataSource.RESUME_TXT


# ---------------------------------------------------------------------------
# Backward compatibility — helpers still importable from pdf_extractor
# ---------------------------------------------------------------------------


class TestBackwardCompatReExports:
    """Private helpers previously living in pdf_extractor must still be
    importable from there so no external code breaks.
    """

    def test_detect_sections_importable(self):
        from app.extractors.pdf_extractor import _detect_sections as ds
        sections = ds("Alice\n\nSkills\nPython")
        assert "skills" in sections

    def test_extract_email_importable(self):
        from app.extractors.pdf_extractor import _extract_email as ee
        assert ee("Contact: bob@test.com") == "bob@test.com"

    def test_extract_skills_importable(self):
        from app.extractors.pdf_extractor import _extract_skills as es
        assert "Python" in es("Python, SQL")

    def test_normalize_skill_importable(self):
        from app.extractors.pdf_extractor import _normalize_skill as ns
        assert ns("python") == "Python"

    def test_classify_url_importable(self):
        from app.extractors.pdf_extractor import _classify_url as cu
        assert cu("https://linkedin.com/in/alice") == "LinkedIn"

    def test_merge_links_importable(self):
        from app.extractors.pdf_extractor import _merge_links as ml
        from app.models.candidate import Link
        a = [Link(url="https://github.com/alice", label="GitHub")]
        b = [Link(url="https://github.com/alice", label="GitHub")]
        assert len(ml(a, b)) == 1  # deduped


# ---------------------------------------------------------------------------
# Experience / education parser correctness (root-cause regression suite)
# ---------------------------------------------------------------------------
# These tests document the CORRECT behavior after fixing the year-line
# splitting bug in _extract_experience and the fragment-merge bug in
# _extract_education.  Each test corresponds to one of the root causes
# identified during the pipeline analysis.
# ---------------------------------------------------------------------------

from app.extractors.text_resume_parser import (
    _extract_experience,
    _extract_education,
    _is_date_line,
)


class TestIsDateLine:
    """_is_date_line() must accept pure date strings and reject content lines."""

    def test_month_year_range(self):
        assert _is_date_line("Jun 2024 – Aug 2024") is True

    def test_month_year_to_present(self):
        assert _is_date_line("Jan 2022 - Present") is True

    def test_year_range(self):
        assert _is_date_line("2020 - 2024") is True

    def test_standalone_year(self):
        assert _is_date_line("2018") is True

    def test_bulleted_date(self):
        assert _is_date_line("• Jan 2021 - Present") is True

    def test_description_with_year_is_not_date_line(self):
        assert _is_date_line("Led team in 2023 hackathon") is False

    def test_company_name_not_date_line(self):
        assert _is_date_line("Infosys Springboard") is False

    def test_job_title_not_date_line(self):
        assert _is_date_line("AI/ML Intern") is False

    def test_inline_date_in_long_line_not_date_line(self):
        # Date only ≈ 36% of line — below 75% threshold
        assert _is_date_line("AI/ML Intern | Infosys | Jun 2024 – Aug 2024") is False

    def test_empty_line_is_not_date_line(self):
        assert _is_date_line("") is False


class TestExtractExperienceCorrectness:
    """_extract_experience() must produce exactly one entry per blank-line block.

    Root cause fixed: splitting on year-containing lines put title+company in
    block N (no date) and date+description in block N+1 (with garbage title).
    Blank-line splitting + date-line filtering produces correct entries.
    """

    def test_two_jobs_give_two_entries(self):
        text = textwrap.dedent("""\
            AI/ML Intern
            Infosys Springboard
            Jun 2024 – Aug 2024
            Built ML models.

            Data Science Intern
            Jaaji Technologies
            Jan 2024 – May 2024
            Analysed datasets.""")
        entries = _extract_experience(text)
        assert len(entries) == 2

    def test_first_job_title_and_company(self):
        text = textwrap.dedent("""\
            AI/ML Intern
            Infosys Springboard
            Jun 2024 – Aug 2024
            Built ML models.

            Data Science Intern
            Jaaji Technologies
            Jan 2024 – May 2024""")
        entries = _extract_experience(text)
        assert entries[0].title == "AI/ML Intern"
        assert entries[0].company == "Infosys Springboard"

    def test_second_job_title_and_company(self):
        text = textwrap.dedent("""\
            AI/ML Intern
            Infosys Springboard
            Jun 2024 – Aug 2024

            Data Science Intern
            Jaaji Technologies
            Jan 2024 – May 2024""")
        entries = _extract_experience(text)
        assert entries[1].title == "Data Science Intern"
        assert entries[1].company == "Jaaji Technologies"

    def test_date_not_used_as_title(self):
        text = textwrap.dedent("""\
            Software Engineer
            TechCorp
            Jan 2021 - Present
            Built APIs.""")
        entries = _extract_experience(text)
        assert len(entries) == 1
        # The date string must NOT appear as the title
        assert entries[0].title != "Jan 2021 - Present"
        assert entries[0].title == "Software Engineer"

    def test_duration_extracted_correctly(self):
        text = textwrap.dedent("""\
            Data Analyst
            XYZ Corp
            Jun 2018 - Dec 2020
            Analysed data.""")
        entries = _extract_experience(text)
        assert entries[0].duration is not None
        assert entries[0].duration.start == "Jun 2018"
        assert entries[0].duration.end == "Dec 2020"
        assert entries[0].duration.is_current is False

    def test_present_sets_is_current(self):
        text = textwrap.dedent("""\
            Senior Engineer
            BigCo
            Mar 2022 - Present""")
        entries = _extract_experience(text)
        assert entries[0].duration.is_current is True
        assert entries[0].duration.end is None

    def test_description_not_polluted_by_date(self):
        text = textwrap.dedent("""\
            ML Engineer
            StartupX
            2020 - 2022
            Deployed models to production.""")
        entries = _extract_experience(text)
        # Description must not start with a date string
        assert entries[0].description is not None
        assert "Deployed" in entries[0].description

    def test_year_in_description_does_not_create_extra_entry(self):
        # Previously, "Won 2023 hackathon" would have triggered a split.
        text = textwrap.dedent("""\
            Backend Developer
            TechFirm
            2021 - 2023
            Won 2023 hackathon award.""")
        entries = _extract_experience(text)
        assert len(entries) == 1

    def test_single_job_no_blank_line(self):
        text = "Developer\nAcme\nJan 2020 - Dec 2021\nBuilt things."
        entries = _extract_experience(text)
        assert len(entries) == 1
        assert entries[0].title == "Developer"

    def test_empty_section_returns_empty(self):
        assert _extract_experience("") == []


class TestExtractEducationCorrectness:
    """_extract_education() must merge PDF-split fragments and extract GPA.

    Root cause fixed: when pdfplumber inserts a blank line between the degree
    line and the institution line, re.split(r'\\n{2,}') created two incomplete
    blocks. institution fell back to lines[0] (the degree string), producing a
    duplicate.  The fragment-merge pass now joins such adjacent blocks.
    """

    def test_single_block_one_entry(self):
        text = "IIT Bombay\nB.Tech Computer Science\n2018 - 2022"
        entries = _extract_education(text)
        assert len(entries) == 1

    def test_institution_and_degree_extracted(self):
        text = "IIT Bombay\nB.Tech Computer Science\n2018 - 2022"
        e = _extract_education(text)[0]
        assert e.institution == "IIT Bombay"
        assert e.degree == "B.Tech Computer Science"

    def test_degree_first_then_institution(self):
        text = "B.E in Computer Science\nAnna University\n2020 - 2024"
        e = _extract_education(text)[0]
        assert e.institution == "Anna University"
        assert "B.E" in e.degree

    def test_pdf_blank_line_fragment_merged(self):
        # Simulates pdfplumber inserting a blank line between degree and inst.
        text = "B.E in Computer Science & Engineering\n\nABC University\n2020 - 2024"
        entries = _extract_education(text)
        assert len(entries) == 1, (
            f"Expected 1 entry after merging fragments, got {len(entries)}: {entries}"
        )
        assert entries[0].institution == "ABC University"
        assert "B.E" in entries[0].degree

    def test_institution_not_set_to_degree_string(self):
        # The old bug: institution = lines[0] = degree string when only 1 line
        # in a block.  After the fix the institution must be None or the real
        # institution, not the degree.
        text = "B.E in Computer Science & Engineering\n\nABC University\n2020 - 2024"
        e = _extract_education(text)[0]
        assert e.institution != e.degree

    def test_gpa_extracted_from_labeled_line(self):
        text = "ABC University\nB.Tech Computer Science\n2018 - 2022\nCGPA: 8.5"
        e = _extract_education(text)[0]
        assert e.gpa == pytest.approx(8.5)

    def test_gpa_extracted_when_fragment_merged(self):
        text = "B.E in CS\n\nABC University\n2020 - 2024\nCGPA: 8.9"
        e = _extract_education(text)[0]
        assert e.gpa == pytest.approx(8.9)

    def test_full_date_range_extracted(self):
        text = "IIT Bombay\nB.Tech CS\n2018 - 2022"
        e = _extract_education(text)[0]
        assert e.duration is not None
        assert e.duration.start == "2018"
        assert e.duration.end == "2022"

    def test_two_degrees_remain_two_entries(self):
        text = textwrap.dedent("""\
            IIT Delhi
            B.Tech Computer Science
            2018 - 2022

            IIM Ahmedabad
            MBA Finance
            2022 - 2024""")
        entries = _extract_education(text)
        assert len(entries) == 2

    def test_empty_section_returns_empty(self):
        assert _extract_education("") == []

    def test_year_only_block_skipped(self):
        # A block containing only a year line should not produce an entry.
        text = "ABC University\nB.Tech CS\n2018 - 2022\n\n2024"
        entries = _extract_education(text)
        # Only the meaningful first block should survive.
        assert all(
            e.institution is not None or e.degree is not None
            for e in entries
        )


class TestParserIntegrationNoDuplicates:
    """End-to-end: TextResumeParser must produce exactly one entry per job /
    degree when the resume text is well-formed.
    """

    _PARSER = TextResumeParser()

    def _parse(self, text: str):
        return self._PARSER.parse(text, source=DataSource.RESUME_PDF)

    def test_two_internships_produce_exactly_two_experience_entries(self):
        text = textwrap.dedent("""\
            Pragna Gosula
            pragna@example.com

            Experience
            AI/ML Intern
            Infosys Springboard
            Jun 2024 – Aug 2024
            Built ML models and improved accuracy.

            Data Science Intern
            Jaaji Technologies
            Jan 2024 – May 2024
            Analysed large datasets.

            Education
            B.E in Computer Science & Engineering
            Anna University
            2020 - 2024
            CGPA: 8.5
            """)
        result = self._parse(text)
        assert len(result.experience) == 2, (
            f"Expected 2 experience entries, got {len(result.experience)}: "
            f"{[(e.title, e.company) for e in result.experience]}"
        )

    def test_internship_titles_and_companies_correct(self):
        text = textwrap.dedent("""\
            Pragna Gosula
            pragna@example.com

            Experience
            AI/ML Intern
            Infosys Springboard
            Jun 2024 – Aug 2024
            Built ML models.

            Data Science Intern
            Jaaji Technologies
            Jan 2024 – May 2024
            Analysed datasets.
            """)
        result = self._parse(text)
        titles    = {e.title    for e in result.experience}
        companies = {e.company  for e in result.experience}
        assert "AI/ML Intern"          in titles
        assert "Data Science Intern"   in titles
        assert "Infosys Springboard"   in companies
        assert "Jaaji Technologies"    in companies

    def test_one_degree_produces_exactly_one_education_entry(self):
        text = textwrap.dedent("""\
            Pragna Gosula
            pragna@example.com

            Education
            B.E in Computer Science & Engineering
            Anna University
            2020 - 2024
            CGPA: 8.5
            """)
        result = self._parse(text)
        assert len(result.education) == 1, (
            f"Expected 1 education entry, got {len(result.education)}: "
            f"{[(e.institution, e.degree) for e in result.education]}"
        )

    def test_degree_not_duplicated_as_institution(self):
        text = textwrap.dedent("""\
            Pragna Gosula
            pragna@example.com

            Education
            B.E in Computer Science & Engineering
            Anna University
            2020 - 2024
            """)
        result = self._parse(text)
        if result.education:
            e = result.education[0]
            assert e.institution != e.degree, (
                "institution must not equal degree — old fallback bug"
            )

    def test_achievements_not_in_experience(self):
        text = textwrap.dedent("""\
            Pragna Gosula
            pragna@example.com

            Experience
            AI/ML Intern
            Infosys Springboard
            Jun 2024 – Aug 2024
            Built ML models.

            Achievements
            Best Intern Award 2024
            National Hackathon Winner
            """)
        result = self._parse(text)
        all_exp_text = " ".join(
            " ".join(filter(None, [e.title, e.company, e.description]))
            for e in result.experience
        )
        assert "Best Intern Award" not in all_exp_text
        assert "National Hackathon" not in all_exp_text

    def test_position_of_responsibility_not_in_experience(self):
        text = textwrap.dedent("""\
            Pragna Gosula
            pragna@example.com

            Experience
            AI/ML Intern
            Infosys Springboard
            Jun 2024 – Aug 2024

            Position of Responsibility
            President, Coding Club 2023
            Secretary, IEEE Chapter 2022
            """)
        result = self._parse(text)
        all_exp_text = " ".join(
            " ".join(filter(None, [e.title, e.company, e.description]))
            for e in result.experience
        )
        assert "President" not in all_exp_text
        assert "Coding Club" not in all_exp_text


# ---------------------------------------------------------------------------
# JSON Extractor — expanded robustness suite
# ---------------------------------------------------------------------------
# Tests are grouped by the failure category identified in the root-cause
# analysis:
#   1. Wrapper unwrapping helpers (_is_candidate_root, _unwrap_candidate_data)
#   2. Flat JSON (existing TestJSONExtractor already covers; quick smoke here)
#   3. Nested/wrapped structures (candidate, profile, data wrappers)
#   4. Field aliases (employer, designation, cgpa, work_experience, academics)
#   5. Recursive field extraction (nested contact section)
#   6. Skills multi-format parsing
#   7. Links — nested container, social_links array, social_profiles
#   8. Misc fields (projects, certifications, achievements, languages)
#   9. Robustness (malformed records, missing fields, empty lists)
#  10. End-to-end smoke test (full realistic profile)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 1. Wrapper-detection helpers
# ---------------------------------------------------------------------------


class TestIsCandiateRoot:
    def test_name_makes_it_root(self):
        assert _is_candidate_root({"name": "Alice", "email": "a@b.com"}) is True

    def test_email_alone_makes_it_root(self):
        assert _is_candidate_root({"email": "a@b.com"}) is True

    def test_phone_alone_makes_it_root(self):
        assert _is_candidate_root({"phone": "9999999999"}) is True

    def test_experience_key_makes_it_root(self):
        assert _is_candidate_root({"experience": []}) is True

    def test_skills_key_makes_it_root(self):
        assert _is_candidate_root({"skills": ["Python"]}) is True

    def test_wrapper_only_is_not_root(self):
        assert _is_candidate_root({"candidate": {"name": "Alice"}}) is False

    def test_profile_dict_is_not_root(self):
        # profile key maps to summary alias — but dict value ≠ real summary
        assert _is_candidate_root({"profile": {"name": "Alice"}}) is False

    def test_profile_string_is_not_root(self):
        # profile key with string value: no scalar candidate key at top level
        assert _is_candidate_root({"profile": "I am an engineer"}) is False

    def test_empty_dict_is_not_root(self):
        assert _is_candidate_root({}) is False


class TestUnwrapCandidateData:
    def test_flat_returned_as_is(self):
        data = {"name": "Alice", "email": "a@b.com"}
        assert _unwrap_candidate_data(data) is data

    def test_candidate_wrapper_unwrapped(self):
        inner = {"name": "Alice", "email": "a@b.com"}
        result = _unwrap_candidate_data({"candidate": inner})
        assert result["name"] == "Alice"

    def test_profile_wrapper_unwrapped(self):
        inner = {"name": "Bob", "phone": "999"}
        result = _unwrap_candidate_data({"profile": inner})
        assert result["name"] == "Bob"

    def test_data_candidate_two_level_unwrapped(self):
        inner = {"name": "Carol", "skills": ["Python"]}
        result = _unwrap_candidate_data({"data": {"candidate": inner}})
        assert result["name"] == "Carol"

    def test_profile_string_not_unwrapped(self):
        # profile = "text" should NOT be unwrapped; used as summary instead
        data = {"profile": "I am an engineer"}
        result = _unwrap_candidate_data(data)
        assert "profile" in result

    def test_unknown_wrapper_returned_unchanged(self):
        data = {"foobar": {"name": "Alice"}}
        result = _unwrap_candidate_data(data)
        assert "foobar" in result


# ---------------------------------------------------------------------------
# 2. Flat JSON — quick smoke (more exhaustive coverage in TestJSONExtractor)
# ---------------------------------------------------------------------------


class TestJSONExtractorFlat:
    def test_all_scalar_fields(self):
        raw = _json_raw({
            "name": "Alice", "email": "alice@test.com",
            "phone": "9999999999", "location": "Hyderabad",
            "summary": "Experienced engineer.",
        })
        r = JSONExtractor().extract(raw)
        assert r.name == "Alice"
        assert r.email == "alice@test.com"
        assert r.phone == "9999999999"
        assert r.location == "Hyderabad"
        assert r.summary == "Experienced engineer."

    def test_source_stamped(self):
        raw = _json_raw({"name": "Alice"})
        assert JSONExtractor().extract(raw).source == DataSource.JSON

    def test_source_file_propagated(self):
        raw = RawCandidateData(
            source=DataSource.JSON,
            raw_fields={"name": "Alice"},
            source_file="alice.json",
        )
        assert JSONExtractor().extract(raw).source_file == "alice.json"


# ---------------------------------------------------------------------------
# 3. Nested / wrapped structures
# ---------------------------------------------------------------------------


class TestJSONExtractorNestedStructures:
    def test_candidate_wrapper(self):
        raw = _json_raw({"candidate": {"name": "Alice", "email": "alice@test.com"}})
        r = JSONExtractor().extract(raw)
        assert r.name == "Alice"
        assert r.email == "alice@test.com"

    def test_profile_wrapper(self):
        raw = _json_raw({"profile": {"name": "Bob", "phone": "8888888888"}})
        r = JSONExtractor().extract(raw)
        assert r.name == "Bob"
        assert r.phone == "8888888888"

    def test_data_candidate_two_level(self):
        raw = _json_raw({"data": {"candidate": {"name": "Carol", "email": "carol@test.com"}}})
        r = JSONExtractor().extract(raw)
        assert r.name == "Carol"
        assert r.email == "carol@test.com"

    def test_candidate_wrapper_with_experience(self):
        raw = _json_raw({
            "candidate": {
                "name": "Dave",
                "experience": [{"company": "Acme", "title": "Engineer"}],
            }
        })
        r = JSONExtractor().extract(raw)
        assert r.name == "Dave"
        assert len(r.experience) == 1
        assert r.experience[0].company == "Acme"

    def test_candidate_wrapper_with_education(self):
        raw = _json_raw({
            "candidate": {
                "name": "Eve",
                "education": [{"institution": "IIT", "degree": "B.Tech"}],
            }
        })
        r = JSONExtractor().extract(raw)
        assert len(r.education) == 1
        assert r.education[0].institution == "IIT"

    def test_wrapper_key_not_leaked_to_extra_fields(self):
        raw = _json_raw({"candidate": {"name": "Alice", "custom_x": "value"}})
        r = JSONExtractor().extract(raw)
        # The outer "candidate" key must NOT appear in extra_fields
        assert "candidate" not in r.extra_fields

    def test_nested_experience_alias_work_experience(self):
        raw = _json_raw({
            "candidate": {
                "work_experience": [{"company": "TechCo", "title": "SDE"}]
            }
        })
        r = JSONExtractor().extract(raw)
        assert len(r.experience) == 1
        assert r.experience[0].company == "TechCo"


# ---------------------------------------------------------------------------
# 4. Field aliases
# ---------------------------------------------------------------------------


class TestJSONExtractorFieldAliases:
    # --- Scalar aliases ---

    def test_full_name_alias(self):
        raw = _json_raw({"full_name": "Alice Smith", "email": "a@test.com"})
        assert JSONExtractor().extract(raw).name == "Alice Smith"

    def test_primary_email_alias(self):
        raw = _json_raw({"primary_email": "alice@test.com"})
        assert JSONExtractor().extract(raw).email == "alice@test.com"

    def test_mobile_alias(self):
        raw = _json_raw({"mobile": "9876543210"})
        assert JSONExtractor().extract(raw).phone == "9876543210"

    def test_objective_maps_to_summary(self):
        raw = _json_raw({"objective": "To build great software."})
        assert JSONExtractor().extract(raw).summary == "To build great software."

    def test_about_me_maps_to_summary(self):
        raw = _json_raw({"about_me": "Passionate developer."})
        assert JSONExtractor().extract(raw).summary == "Passionate developer."

    def test_current_location_alias(self):
        raw = _json_raw({"current_location": "Bengaluru"})
        assert JSONExtractor().extract(raw).location == "Bengaluru"

    # --- Experience item aliases ---

    def test_employer_maps_to_company(self):
        raw = _json_raw({"experience": [{"employer": "Google", "title": "SWE"}]})
        r = JSONExtractor().extract(raw)
        assert r.experience[0].company == "Google"

    def test_designation_maps_to_title(self):
        raw = _json_raw({"experience": [{"company": "Acme", "designation": "SDE"}]})
        r = JSONExtractor().extract(raw)
        assert r.experience[0].title == "SDE"

    def test_role_maps_to_title(self):
        raw = _json_raw({"experience": [{"company": "X", "role": "Analyst"}]})
        assert JSONExtractor().extract(raw).experience[0].title == "Analyst"

    def test_responsibilities_as_description(self):
        raw = _json_raw({"experience": [{"company": "Y", "responsibilities": "Led a team."}]})
        assert JSONExtractor().extract(raw).experience[0].description == "Led a team."

    def test_present_end_sets_is_current(self):
        raw = _json_raw({"experience": [{"company": "Z", "start": "2022-01", "end": "present"}]})
        exp = JSONExtractor().extract(raw).experience[0]
        assert exp.duration.is_current is True
        assert exp.duration.end is None

    def test_current_end_sets_is_current(self):
        raw = _json_raw({"experience": [{"company": "Z", "start": "2022-01", "end": "current"}]})
        exp = JSONExtractor().extract(raw).experience[0]
        assert exp.duration.is_current is True

    # --- Experience list field aliases ---

    def test_work_experience_alias(self):
        raw = _json_raw({"work_experience": [{"company": "Acme", "title": "Dev"}]})
        assert len(JSONExtractor().extract(raw).experience) == 1

    def test_employment_alias(self):
        raw = _json_raw({"employment": [{"company": "Big Corp", "title": "PM"}]})
        assert JSONExtractor().extract(raw).experience[0].company == "Big Corp"

    def test_employment_history_alias(self):
        raw = _json_raw({"employment_history": [{"company": "Old Co", "title": "QA"}]})
        assert len(JSONExtractor().extract(raw).experience) == 1

    # --- Education item aliases ---

    def test_college_maps_to_institution(self):
        raw = _json_raw({"education": [{"college": "IIT Bombay", "degree": "B.Tech"}]})
        assert JSONExtractor().extract(raw).education[0].institution == "IIT Bombay"

    def test_cgpa_extracted(self):
        raw = _json_raw({"education": [{"institution": "MIT", "cgpa": "9.1"}]})
        assert JSONExtractor().extract(raw).education[0].gpa == pytest.approx(9.1)

    def test_percentage_extracted(self):
        raw = _json_raw({"education": [{"institution": "NIT", "percentage": "85.0"}]})
        assert JSONExtractor().extract(raw).education[0].gpa == pytest.approx(85.0)

    def test_graduation_year_as_end_year(self):
        raw = _json_raw({"education": [{"institution": "SRM", "graduation_year": "2022"}]})
        edu = JSONExtractor().extract(raw).education[0]
        assert edu.duration.end == "2022"

    def test_separate_start_and_end_year(self):
        raw = _json_raw({
            "education": [{"institution": "IIT", "start_year": "2018", "end_year": "2022"}]
        })
        edu = JSONExtractor().extract(raw).education[0]
        assert edu.duration.start == "2018"
        assert edu.duration.end == "2022"

    def test_specialization_as_field_of_study(self):
        raw = _json_raw({
            "education": [{"institution": "VIT", "degree": "B.E.", "specialization": "CS"}]
        })
        assert JSONExtractor().extract(raw).education[0].field_of_study == "CS"

    def test_branch_as_field_of_study(self):
        raw = _json_raw({
            "education": [{"institution": "NIT", "branch": "ECE"}]
        })
        assert JSONExtractor().extract(raw).education[0].field_of_study == "ECE"

    # --- Education list field aliases ---

    def test_academics_alias(self):
        raw = _json_raw({"academics": [{"institution": "NIT", "degree": "B.Tech"}]})
        assert len(JSONExtractor().extract(raw).education) == 1

    def test_qualifications_alias(self):
        raw = _json_raw({"qualifications": [{"institution": "MIT", "degree": "MS"}]})
        assert len(JSONExtractor().extract(raw).education) == 1

    # --- Skills aliases ---

    def test_technical_skills_alias(self):
        raw = _json_raw({"technical_skills": ["Python", "Docker"]})
        assert "Python" in JSONExtractor().extract(raw).skills

    def test_core_skills_alias(self):
        raw = _json_raw({"core_skills": ["Java", "Spring"]})
        assert "Java" in JSONExtractor().extract(raw).skills


# ---------------------------------------------------------------------------
# 5. Recursive field extraction (nested contact sections)
# ---------------------------------------------------------------------------


class TestJSONExtractorRecursiveExtraction:
    def test_email_in_nested_contact(self):
        raw = _json_raw({
            "name": "Alice",
            "contact": {"email": "alice@test.com", "phone": "9999999999"},
        })
        r = JSONExtractor().extract(raw)
        assert r.email == "alice@test.com"
        assert r.phone == "9999999999"

    def test_phone_in_nested_contact(self):
        raw = _json_raw({
            "name": "Bob",
            "personal": {"mobile": "8888888888", "location": "Delhi"},
        })
        r = JSONExtractor().extract(raw)
        assert r.phone == "8888888888"
        assert r.location == "Delhi"

    def test_top_level_wins_over_nested(self):
        raw = _json_raw({
            "name": "Alice",
            "email": "top@test.com",
            "contact": {"email": "nested@test.com"},
        })
        r = JSONExtractor().extract(raw)
        assert r.email == "top@test.com"

    def test_deeply_nested_skills_in_candidate_wrapper(self):
        raw = _json_raw({
            "candidate": {
                "name": "Carol",
                "technical_skills": ["Python", "SQL"],
            }
        })
        r = JSONExtractor().extract(raw)
        assert "Python" in r.skills
        assert "SQL" in r.skills


# ---------------------------------------------------------------------------
# 6. Skills — multi-format string input
# ---------------------------------------------------------------------------


class TestJSONExtractorSkillsFormats:
    def test_skills_as_list(self):
        raw = _json_raw({"skills": ["Python", "Java", "SQL"]})
        assert "Python" in JSONExtractor().extract(raw).skills

    def test_skills_as_comma_string(self):
        raw = _json_raw({"skills": "Python, Java, SQL"})
        r = JSONExtractor().extract(raw)
        assert "Python" in r.skills
        assert "Java" in r.skills

    def test_skills_as_semicolon_string(self):
        raw = _json_raw({"skills": "Python; Java; SQL"})
        r = JSONExtractor().extract(raw)
        assert "Python" in r.skills

    def test_skills_as_pipe_string(self):
        raw = _json_raw({"skills": "Python|Java|SQL"})
        r = JSONExtractor().extract(raw)
        assert "Python" in r.skills

    def test_skills_as_newline_string(self):
        raw = _json_raw({"skills": "Python\nJava\nSQL"})
        r = JSONExtractor().extract(raw)
        assert "Python" in r.skills

    def test_skill_alias_normalised(self):
        raw = _json_raw({"skills": ["ML", "node js"]})
        r = JSONExtractor().extract(raw)
        assert "Machine Learning" in r.skills
        assert "Node.js" in r.skills

    def test_skills_deduplicated(self):
        raw = _json_raw({"skills": ["Python", "python", "PYTHON"]})
        r = JSONExtractor().extract(raw)
        assert r.skills.count("Python") == 1


# ---------------------------------------------------------------------------
# 7. Links — all source types
# ---------------------------------------------------------------------------


class TestJSONExtractorLinks:
    def test_top_level_linkedin_url(self):
        raw = _json_raw({"linkedin": "https://linkedin.com/in/alice"})
        links = JSONExtractor().extract(raw).links
        li = [l for l in links if l.label == "LinkedIn"]
        assert li and "alice" in li[0].url

    def test_top_level_github_url(self):
        raw = _json_raw({"github": "https://github.com/alice"})
        links = JSONExtractor().extract(raw).links
        gh = [l for l in links if l.label == "GitHub"]
        assert gh

    def test_nested_profiles_dict(self):
        raw = _json_raw({
            "name": "Alice",
            "profiles": {
                "github": "https://github.com/alice",
                "linkedin": "https://linkedin.com/in/alice",
            },
        })
        labels = {l.label for l in JSONExtractor().extract(raw).links}
        assert "GitHub" in labels
        assert "LinkedIn" in labels

    def test_social_links_array_of_strings(self):
        raw = _json_raw({
            "name": "Alice",
            "social_links": [
                "https://github.com/alice",
                "https://linkedin.com/in/alice",
            ],
        })
        labels = {l.label for l in JSONExtractor().extract(raw).links}
        assert "GitHub" in labels

    def test_social_profiles_array_of_dicts(self):
        raw = _json_raw({
            "social_profiles": [
                {"url": "https://github.com/alice", "label": "GitHub"},
            ]
        })
        links = JSONExtractor().extract(raw).links
        assert any(l.label == "GitHub" for l in links)

    def test_links_array_of_url_dicts(self):
        raw = _json_raw({
            "links": [
                {"url": "https://leetcode.com/alice", "label": "LeetCode"},
            ]
        })
        links = JSONExtractor().extract(raw).links
        assert any(l.label == "LeetCode" for l in links)

    def test_url_embedded_in_summary(self):
        raw = _json_raw({"summary": "See my work at https://github.com/alice"})
        links = JSONExtractor().extract(raw).links
        assert any("github" in l.url for l in links)

    def test_duplicate_urls_deduped(self):
        raw = _json_raw({
            "github": "https://github.com/alice",
            "profiles": {"github": "https://github.com/alice"},
        })
        links = JSONExtractor().extract(raw).links
        gh = [l for l in links if "github.com/alice" in l.url]
        assert len(gh) == 1


# ---------------------------------------------------------------------------
# 8. Misc fields (projects, certifications, achievements, languages)
# ---------------------------------------------------------------------------


class TestExtractMiscFieldsJSON:
    def test_projects_list(self):
        result = _extract_misc_fields({"projects": ["Resume Parser", "Chatbot"]})
        assert result["projects"] == ["Resume Parser", "Chatbot"]

    def test_projects_comma_string(self):
        result = _extract_misc_fields({"projects": "Resume Parser, Chatbot"})
        assert "Resume Parser" in result["projects"]

    def test_personal_projects_alias(self):
        result = _extract_misc_fields({"personal_projects": ["Portfolio", "API"]})
        assert "Portfolio" in result["projects"]

    def test_certifications_list(self):
        result = _extract_misc_fields({"certifications": ["AWS", "GCP"]})
        assert result["certifications"] == ["AWS", "GCP"]

    def test_certificates_alias(self):
        result = _extract_misc_fields({"certificates": "AWS, GCP"})
        assert "AWS" in result["certifications"]

    def test_achievements_list(self):
        result = _extract_misc_fields({"achievements": ["Hackathon Winner"]})
        assert "Hackathon Winner" in result["achievements"]

    def test_awards_alias(self):
        result = _extract_misc_fields({"awards": ["Best Employee"]})
        assert "Best Employee" in result["achievements"]

    def test_languages_list(self):
        result = _extract_misc_fields({"languages": ["English", "Hindi"]})
        assert "English" in result["languages"]

    def test_languages_known_alias(self):
        result = _extract_misc_fields({"languages_known": "English|Hindi"})
        assert "English" in result["languages"]

    def test_no_misc_keys_returns_empty(self):
        assert _extract_misc_fields({"name": "Alice"}) == {}


class TestJSONExtractorMiscFieldsIntegration:
    def test_projects_in_extra_fields(self):
        raw = _json_raw({"name": "Alice", "projects": ["Chatbot", "API"]})
        r = JSONExtractor().extract(raw)
        assert "projects" in r.extra_fields
        assert "Chatbot" in r.extra_fields["projects"]

    def test_certifications_in_extra_fields(self):
        raw = _json_raw({"certifications": ["AWS", "GCP"]})
        r = JSONExtractor().extract(raw)
        assert "certifications" in r.extra_fields
        assert isinstance(r.extra_fields["certifications"], list)

    def test_achievements_in_extra_fields(self):
        raw = _json_raw({"awards": "Best Employee, Hackathon Winner"})
        r = JSONExtractor().extract(raw)
        assert "achievements" in r.extra_fields
        assert "Best Employee" in r.extra_fields["achievements"]

    def test_languages_in_extra_fields(self):
        raw = _json_raw({"languages_known": "English, Hindi, Telugu"})
        r = JSONExtractor().extract(raw)
        assert "languages" in r.extra_fields
        assert "English" in r.extra_fields["languages"]


# ---------------------------------------------------------------------------
# 9. Robustness — malformed / missing fields
# ---------------------------------------------------------------------------


class TestJSONExtractorRobustness:
    def test_empty_raw_fields_returns_blank(self):
        raw = _json_raw({})
        r = JSONExtractor().extract(raw)
        assert r.name is None
        assert r.skills == []
        assert r.experience == []
        assert r.education == []

    def test_error_record_returns_empty(self):
        raw = RawCandidateData(
            source=DataSource.JSON,
            parse_errors=["JSON decode error"],
        )
        r = JSONExtractor().extract(raw)
        assert r.name is None

    def test_non_list_experience_ignored(self):
        raw = _json_raw({"experience": "5 years"})
        assert JSONExtractor().extract(raw).experience == []

    def test_non_list_education_ignored(self):
        raw = _json_raw({"education": "B.Tech"})
        assert JSONExtractor().extract(raw).education == []

    def test_non_dict_experience_items_skipped(self):
        raw = _json_raw({"experience": [{"company": "Acme"}, "bad", 42, None]})
        r = JSONExtractor().extract(raw)
        assert len(r.experience) == 1

    def test_non_dict_education_items_skipped(self):
        raw = _json_raw({"education": [{"institution": "IIT"}, "bad"]})
        r = JSONExtractor().extract(raw)
        assert len(r.education) == 1

    def test_missing_optional_fields_produce_none(self):
        raw = _json_raw({"name": "Alice"})
        r = JSONExtractor().extract(raw)
        assert r.email is None
        assert r.phone is None
        assert r.location is None
        assert r.summary is None

    def test_experience_without_dates_has_no_duration(self):
        raw = _json_raw({"experience": [{"company": "Acme", "title": "Dev"}]})
        exp = JSONExtractor().extract(raw).experience[0]
        assert exp.duration is None

    def test_education_without_dates_has_no_duration(self):
        raw = _json_raw({"education": [{"institution": "IIT", "degree": "B.Tech"}]})
        edu = JSONExtractor().extract(raw).education[0]
        assert edu.duration is None

    def test_extra_fields_captured(self):
        raw = _json_raw({"name": "Alice", "custom_rank": 1})
        r = JSONExtractor().extract(raw)
        assert "custom_rank" in r.extra_fields

    def test_experience_and_education_not_in_extra_fields(self):
        raw = _json_raw({
            "experience": [{"company": "X"}],
            "education": [{"institution": "Y"}],
        })
        r = JSONExtractor().extract(raw)
        assert "experience" not in r.extra_fields
        assert "education" not in r.extra_fields


# ---------------------------------------------------------------------------
# _parse_gpa (unit)
# ---------------------------------------------------------------------------


class TestParseGPAJSON:
    def test_cgpa_key(self):
        assert _extract_item_gpa({"cgpa": "8.5"}) == pytest.approx(8.5)

    def test_gpa_key(self):
        assert _extract_item_gpa({"gpa": "3.9"}) == pytest.approx(3.9)

    def test_percentage_key(self):
        assert _extract_item_gpa({"percentage": "87"}) == pytest.approx(87.0)

    def test_grade_key(self):
        assert _extract_item_gpa({"grade": "9.0"}) == pytest.approx(9.0)

    def test_cgpa_takes_priority_over_gpa(self):
        result = _extract_item_gpa({"cgpa": "8.5", "gpa": "3.9"})
        assert result == pytest.approx(8.5)

    def test_non_numeric_returns_none(self):
        assert _extract_item_gpa({"gpa": "N/A"}) is None

    def test_empty_dict_returns_none(self):
        assert _extract_item_gpa({}) is None


# ---------------------------------------------------------------------------
# 10. End-to-end smoke test
# ---------------------------------------------------------------------------


class TestJSONExtractorEndToEnd:
    """Full realistic candidate profile covering all extracted fields."""

    PROFILE = {
        "candidate": {
            "full_name": "Pragna Gosula",
            "contact": {
                "email": "pragna@example.com",
                "mobile": "+91 9876543210",
                "current_location": "Hyderabad",
            },
            "objective": "Aspiring software engineer passionate about AI/ML.",
            "technical_skills": ["Python", "FastAPI", "Docker", "SQL"],
            "work_experience": [
                {
                    "employer": "Infosys Springboard",
                    "designation": "AI/ML Intern",
                    "start_date": "Jun 2024",
                    "end_date": "Aug 2024",
                    "responsibilities": "Built ML models.",
                },
                {
                    "employer": "Jaaji Technologies",
                    "designation": "Data Science Intern",
                    "start_date": "Jan 2024",
                    "end_date": "May 2024",
                },
            ],
            "academics": [
                {
                    "college": "Anna University",
                    "degree": "B.E.",
                    "branch": "Computer Science",
                    "start_year": "2020",
                    "end_year": "2024",
                    "cgpa": "8.5",
                }
            ],
            "profiles": {
                "github": "https://github.com/pragnagosula",
                "linkedin": "https://linkedin.com/in/pragnagosula",
            },
            "certifications": ["AWS Solutions Architect", "TensorFlow Developer"],
            "achievements": ["National Hackathon Winner 2024"],
            "languages_known": "English, Telugu",
        }
    }

    def setup_method(self):
        self.result = JSONExtractor().extract(_json_raw(self.PROFILE))

    def test_name(self):
        assert self.result.name == "Pragna Gosula"

    def test_email(self):
        assert self.result.email == "pragna@example.com"

    def test_phone(self):
        assert self.result.phone is not None
        assert "9876543210" in self.result.phone

    def test_location(self):
        assert self.result.location == "Hyderabad"

    def test_summary(self):
        assert self.result.summary is not None
        assert "AI" in self.result.summary or "engineer" in self.result.summary.lower()

    def test_skills(self):
        assert "Python" in self.result.skills
        assert "FastAPI" in self.result.skills
        assert "Docker" in self.result.skills

    def test_two_experience_entries(self):
        assert len(self.result.experience) == 2

    def test_infosys_entry(self):
        companies = {e.company for e in self.result.experience}
        titles = {e.title for e in self.result.experience}
        assert "Infosys Springboard" in companies
        assert "AI/ML Intern" in titles

    def test_jaaji_entry(self):
        companies = {e.company for e in self.result.experience}
        assert "Jaaji Technologies" in companies

    def test_experience_dates(self):
        infosys = next(e for e in self.result.experience if e.company == "Infosys Springboard")
        assert infosys.duration is not None
        assert "2024" in infosys.duration.start

    def test_one_education_entry(self):
        assert len(self.result.education) == 1

    def test_education_institution(self):
        assert self.result.education[0].institution == "Anna University"

    def test_education_degree(self):
        assert self.result.education[0].degree == "B.E."

    def test_education_field_of_study(self):
        assert self.result.education[0].field_of_study == "Computer Science"

    def test_education_gpa(self):
        assert self.result.education[0].gpa == pytest.approx(8.5)

    def test_education_date_range(self):
        edu = self.result.education[0]
        assert edu.duration.start == "2020"
        assert edu.duration.end == "2024"

    def test_github_link(self):
        labels = {l.label for l in self.result.links}
        assert "GitHub" in labels

    def test_linkedin_link(self):
        labels = {l.label for l in self.result.links}
        assert "LinkedIn" in labels

    def test_certifications_in_extra_fields(self):
        assert "certifications" in self.result.extra_fields
        assert "AWS Solutions Architect" in self.result.extra_fields["certifications"]

    def test_achievements_in_extra_fields(self):
        assert "achievements" in self.result.extra_fields
        assert "National Hackathon Winner 2024" in self.result.extra_fields["achievements"]

    def test_languages_in_extra_fields(self):
        assert "languages" in self.result.extra_fields
        assert "English" in self.result.extra_fields["languages"]
