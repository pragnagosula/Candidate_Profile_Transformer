"""CSV extractor — maps column names to typed ExtractedCandidate fields.

CSV sources are fully structured; extraction is mostly field-name resolution
plus splitting multi-value cells.

Skills and links are normalised the same way as the PDF and JSON extractors
(shared helpers imported from :mod:`app.extractors.text_resume_parser`) so a
candidate gets consistent results regardless of which source format they
came from.

Experience and education are extracted from either:
  - A text-blob column (``Work Experience``, ``Work History``), parsed via
    :func:`~app.extractors.text_resume_parser._extract_experience`, or
  - Flat sub-field columns (``Company``, ``Role``, ``Start Date``, ``College``,
    ``Degree``, ``CGPA``, …) assembled into typed model objects.

Miscellaneous list fields (projects, certifications, achievements, spoken
languages) are parsed from common column aliases and stored as structured
lists inside ``extra_fields`` so they survive the pipeline without requiring
model changes.

NOTE: ``_parse_skills`` is also imported by :mod:`app.extractors.json_extractor`
— its signature and list/string splitting behaviour must remain unchanged.
"""

from __future__ import annotations

import json
import re
from typing import Any

from app.extractors.base import BaseExtractor
from app.extractors.field_map import extract_known_fields
from app.extractors.text_resume_parser import (
    _classify_url,
    _extract_education as _parse_education_text,
    _extract_experience as _parse_experience_text,
    _extract_links_from_text,
    _merge_links,
    _normalize_link_url,
    _normalize_skill,
    _scan_tech_dictionary,
)
from app.extractors.registry import extractor_registry
from app.models.candidate import (
    DataSource,
    DateRange,
    Education,
    Experience,
    ExtractedCandidate,
    Link,
    RawCandidateData,
)
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Cell-level multi-value parsing
# ---------------------------------------------------------------------------

_CELL_SPLIT_RE = re.compile(r"[,;|\n]+")


def _parse_cell_list(value: Any) -> list[str]:
    """Split a CSV cell that may hold multiple items.

    Handles: Python lists, JSON array strings (``["a","b"]``), and strings
    delimited by comma, semicolon, pipe, or newline.
    """
    if isinstance(value, list):
        return [str(s).strip() for s in value if str(s).strip()]
    if not isinstance(value, str) or not value.strip():
        return []
    text = value.strip()
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(s).strip() for s in parsed if str(s).strip()]
        except (ValueError, TypeError):
            pass
    return [s.strip() for s in _CELL_SPLIT_RE.split(text) if s.strip()]


def _parse_skills(raw: object) -> list[str]:
    """Split a raw skills value into individual skill strings.

    Handles comma/semicolon/pipe/newline strings, JSON array strings, and
    Python lists.

    NOTE: imported by :mod:`app.extractors.json_extractor` — signature is
    part of the public API and must remain stable.
    """
    return _parse_cell_list(raw)


# ---------------------------------------------------------------------------
# Skills — alias-normalised, deduplicated, with a tech-dictionary fallback
# ---------------------------------------------------------------------------


def _normalize_and_dedupe_skills(raw_skills: list[str]) -> list[str]:
    """Apply the shared skill-normalisation/alias map and remove duplicates
    case-insensitively while preserving first-seen order and casing.
    """
    normalized: list[str] = []
    seen: set[str] = set()
    for skill in raw_skills:
        norm = _normalize_skill(skill)
        if not norm:
            continue
        key = norm.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(norm)
    return normalized


def _row_text_blob(raw_fields: dict) -> str:
    """Join every string-valued column into one blob for fallback scanning."""
    parts: list[str] = []
    for value in raw_fields.values():
        if isinstance(value, str) and value.strip():
            parts.append(value)
        elif isinstance(value, list):
            parts.extend(v for v in value if isinstance(v, str) and v.strip())
    return "\n".join(parts)


def _extract_skills(fields: dict, raw_fields: dict) -> list[str]:
    """Extract a normalised, deduplicated skill list, with a fallback scan.

    Mirrors the PDF/JSON extractors: if the structured ``skills`` column is
    missing or sparse (<5 entries), the whole row is scanned for known
    technology names so sparse CSV exports still yield useful skills.
    """
    raw_skills = _parse_skills(fields.get("skills", ""))
    skills = _normalize_and_dedupe_skills(raw_skills)

    if len(skills) < 5:
        text_blob = _row_text_blob(raw_fields)
        if text_blob:
            seen = {s.lower() for s in skills}
            for hit in _scan_tech_dictionary(text_blob):
                key = hit.lower()
                if key in seen:
                    continue
                seen.add(key)
                skills.append(hit)

    return skills


# ---------------------------------------------------------------------------
# Links — known URL columns + username handles + free-text scan
# ---------------------------------------------------------------------------

# Normalised column key → display label for direct URL columns
_LINK_FIELD_ALIASES: dict[str, str] = {
    "linkedin": "LinkedIn", "linkedin_url": "LinkedIn",
    "linkedin_profile": "LinkedIn", "linkedin_link": "LinkedIn",
    "github": "GitHub", "github_url": "GitHub",
    "github_profile": "GitHub", "github_link": "GitHub",
    "portfolio": "Portfolio", "portfolio_url": "Portfolio",
    "portfolio_link": "Portfolio",
    "website": "Website", "personal_website": "Website",
    "personal_site": "Website",
    "leetcode": "LeetCode", "leetcode_url": "LeetCode",
    "leetcode_profile": "LeetCode",
    "codechef": "CodeChef", "codechef_url": "CodeChef",
    "codechef_profile": "CodeChef",
    "hackerrank": "HackerRank", "hackerrank_url": "HackerRank",
    "hackerrank_profile": "HackerRank",
    "kaggle": "Kaggle", "kaggle_url": "Kaggle",
    "kaggle_profile": "Kaggle",
    "medium": "Medium", "medium_url": "Medium",
    "medium_profile": "Medium",
    "stackoverflow": "StackOverflow", "stackoverflow_url": "StackOverflow",
    "stackoverflow_profile": "StackOverflow",
}

# Normalised column key → (URL prefix, display label) for username columns
_USERNAME_PLATFORMS: dict[str, tuple[str, str]] = {
    "github_username":     ("https://github.com/",             "GitHub"),
    "github_handle":       ("https://github.com/",             "GitHub"),
    "linkedin_username":   ("https://linkedin.com/in/",        "LinkedIn"),
    "linkedin_handle":     ("https://linkedin.com/in/",        "LinkedIn"),
    "leetcode_username":   ("https://leetcode.com/",           "LeetCode"),
    "leetcode_handle":     ("https://leetcode.com/",           "LeetCode"),
    "codechef_username":   ("https://codechef.com/users/",     "CodeChef"),
    "hackerrank_username": ("https://hackerrank.com/profile/", "HackerRank"),
    "kaggle_username":     ("https://kaggle.com/",             "Kaggle"),
    "medium_username":     ("https://medium.com/@",            "Medium"),
}

def _normalize_field_key(key: str) -> str:
    return key.strip().lower().replace(" ", "_").replace("-", "_")


def _parse_structured_links(raw_fields: dict) -> list[Link]:
    """Pick up well-known single-URL columns, e.g. ``linkedin_url``."""
    links: list[Link] = []
    for raw_key, value in raw_fields.items():
        if not isinstance(value, str) or not value.strip():
            continue
        label = _LINK_FIELD_ALIASES.get(_normalize_field_key(str(raw_key)))
        if label:
            links.append(Link(url=_normalize_link_url(value.strip()), label=label))
    return links


def _extract_username_links(raw_fields: dict) -> list[Link]:
    """Convert username/handle columns into full profile URLs.

    Example: ``github_username = pragnagosula``
             → ``Link(url="https://github.com/pragnagosula", label="GitHub")``
    """
    links: list[Link] = []
    for raw_key, value in raw_fields.items():
        if not isinstance(value, str) or not value.strip():
            continue
        platform = _USERNAME_PLATFORMS.get(_normalize_field_key(str(raw_key)))
        if not platform:
            continue
        username = value.strip().lstrip("@")
        if username:
            base_url, label = platform
            links.append(Link(url=base_url + username, label=label))
    return links


def _extract_links(fields: dict, raw_fields: dict | None = None) -> list[Link]:
    """Build Link objects from well-known URL columns, username columns, and
    any URLs found in free-text columns.

    Priority order (first occurrence wins on duplicate URLs):
        1. Well-known single-URL columns (``linkedin``, ``github_url``, …)
        2. Username/handle columns (``github_username`` → constructed URL)
        3. URLs found anywhere in the row's text values
    """
    raw_fields = raw_fields if raw_fields is not None else fields
    structured = _parse_structured_links(raw_fields)
    username_links = _extract_username_links(raw_fields)
    text_blob = _row_text_blob(raw_fields)
    text_links = _extract_links_from_text(text_blob) if text_blob else []
    return _merge_links(structured, username_links, text_links)


# ---------------------------------------------------------------------------
# Experience extraction from flat sub-field columns
# ---------------------------------------------------------------------------

_EXP_COMPANY_KEYS = frozenset({
    "company", "employer", "organization", "organisation",
    "current_company", "previous_company", "last_company",
})
_EXP_TITLE_KEYS = frozenset({
    "role", "position", "designation", "job_title",
    "current_role", "current_position", "current_designation",
    "last_role", "last_designation",
})
_EXP_START_KEYS = frozenset({
    "start_date", "from_date", "joining_date", "date_of_joining",
    "employment_start",
})
_EXP_END_KEYS = frozenset({
    "end_date", "to_date", "relieving_date", "date_of_leaving",
    "employment_end",
})
_EXP_DESC_KEYS = frozenset({
    "responsibilities", "job_description", "work_description",
    "work_summary", "role_description", "duties",
})
_EXP_TYPE_KEYS = frozenset({
    "employment_type", "job_type", "work_type", "type_of_employment",
})
_EXP_LOC_KEYS = frozenset({
    "job_location", "work_location", "office_location",
})


def _first_value(raw_fields: dict, keys: frozenset) -> str | None:
    """Return the first non-empty string value whose normalised key is in *keys*."""
    for raw_key, value in raw_fields.items():
        if _normalize_field_key(str(raw_key)) in keys:
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _extract_experience_row(raw_fields: dict) -> list[Experience]:
    """Build a single Experience object from flat sub-field columns.

    Returns an empty list when no recognisable experience column is present.
    Common column names: Company, Employer, Role, Designation, Start Date, …
    """
    company = _first_value(raw_fields, _EXP_COMPANY_KEYS)
    title = _first_value(raw_fields, _EXP_TITLE_KEYS)
    start = _first_value(raw_fields, _EXP_START_KEYS)
    end = _first_value(raw_fields, _EXP_END_KEYS)
    description = _first_value(raw_fields, _EXP_DESC_KEYS)
    location = _first_value(raw_fields, _EXP_LOC_KEYS)

    if not any([company, title, start, description]):
        return []

    is_current = start is not None and end is None
    duration = DateRange(start=start, end=end, is_current=is_current) if (start or end) else None

    return [Experience(
        company=company,
        title=title,
        duration=duration,
        description=description,
        location=location,
    )]


# ---------------------------------------------------------------------------
# Education extraction from flat sub-field columns
# ---------------------------------------------------------------------------

_EDU_INST_KEYS = frozenset({
    "college", "university", "institute", "school", "institution",
    "alma_mater", "college_name", "university_name",
})
_EDU_DEGREE_KEYS = frozenset({
    "degree", "qualification", "course", "degree_name", "program",
})
_EDU_FIELD_KEYS = frozenset({
    "major", "specialization", "specialisation", "branch",
    "field_of_study", "stream", "subject",
})
_EDU_GPA_KEYS = frozenset({
    "cgpa", "gpa", "percentage", "grade", "marks", "score",
})
_EDU_GRAD_KEYS = frozenset({
    "graduation_year", "passing_year", "end_year", "year_of_passing",
    "year_of_graduation", "batch", "passout_year",
})
_EDU_START_KEYS = frozenset({
    "start_year", "admission_year", "year_of_admission", "joining_year",
})

_GPA_CLEANUP_RE = re.compile(r"[^\d.]")


def _parse_gpa(value: str | None) -> float | None:
    """Parse a GPA or percentage string to float; returns None if unparseable."""
    if not value:
        return None
    cleaned = _GPA_CLEANUP_RE.sub("", value)[:10]
    try:
        return float(cleaned) if cleaned else None
    except ValueError:
        return None


def _extract_education_row(raw_fields: dict) -> list[Education]:
    """Build a single Education object from flat sub-field columns.

    Returns an empty list when no recognisable education column is present.
    Common column names: College, University, Degree, CGPA, Graduation Year, …
    """
    institution = _first_value(raw_fields, _EDU_INST_KEYS)
    degree = _first_value(raw_fields, _EDU_DEGREE_KEYS)
    field_of_study = _first_value(raw_fields, _EDU_FIELD_KEYS)
    gpa_raw = _first_value(raw_fields, _EDU_GPA_KEYS)
    grad_year = _first_value(raw_fields, _EDU_GRAD_KEYS)
    start_year = _first_value(raw_fields, _EDU_START_KEYS)

    if not any([institution, degree, field_of_study]):
        return []

    duration = (
        DateRange(start=start_year, end=grad_year)
        if (start_year or grad_year)
        else None
    )

    return [Education(
        institution=institution,
        degree=degree,
        field_of_study=field_of_study,
        gpa=_parse_gpa(gpa_raw),
        duration=duration,
    )]


# ---------------------------------------------------------------------------
# Miscellaneous list fields → stored as structured lists in extra_fields
# ---------------------------------------------------------------------------

_PROJ_KEYS = frozenset({
    "project", "projects", "project_name", "project_names",
    "project_title", "project_titles", "project_details",
})
_CERT_KEYS = frozenset({
    "certification", "certifications", "certificate", "certificates",
    "credential", "credentials", "course_certification", "training",
    "courses",
})
_ACHIEVE_KEYS = frozenset({
    "achievements", "achievement", "awards", "honors", "honours",
    "recognition", "accomplishments", "accolades",
})
_LANG_KEYS = frozenset({
    "languages_known", "known_languages", "spoken_languages",
    "language_proficiency", "language_skills", "languages_spoken",
    "language",
})


def _extract_misc_fields(raw_fields: dict) -> dict[str, list[str]]:
    """Parse projects, certifications, achievements, and spoken languages
    into named string lists, stored under ``extra_fields``.

    Each category takes the first matching column; values are split with
    :func:`_parse_cell_list` (comma, semicolon, pipe, newline, JSON array).
    """
    result: dict[str, list[str]] = {}

    def _collect(keys: frozenset, out_key: str) -> None:
        for raw_key, value in raw_fields.items():
            if _normalize_field_key(str(raw_key)) in keys:
                items = _parse_cell_list(value)
                if items:
                    result[out_key] = items
                    return

    _collect(_PROJ_KEYS, "projects")
    _collect(_CERT_KEYS, "certifications")
    _collect(_ACHIEVE_KEYS, "achievements")
    _collect(_LANG_KEYS, "languages")
    return result


# ---------------------------------------------------------------------------
# Keys consumed by extraction — excluded from extra_fields
# ---------------------------------------------------------------------------

_ALL_CONSUMED_KEYS: frozenset[str] = (
    frozenset({"name", "email", "phone", "location", "summary", "skills",
               "experience", "education"})
    | frozenset(_LINK_FIELD_ALIASES)
    | frozenset(_USERNAME_PLATFORMS)
    | _EXP_COMPANY_KEYS | _EXP_TITLE_KEYS | _EXP_START_KEYS
    | _EXP_END_KEYS | _EXP_DESC_KEYS | _EXP_TYPE_KEYS | _EXP_LOC_KEYS
    | _EDU_INST_KEYS | _EDU_DEGREE_KEYS | _EDU_FIELD_KEYS
    | _EDU_GPA_KEYS | _EDU_GRAD_KEYS | _EDU_START_KEYS
    | _PROJ_KEYS | _CERT_KEYS | _ACHIEVE_KEYS | _LANG_KEYS
)


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


@extractor_registry.register(DataSource.CSV)
class CSVExtractor(BaseExtractor):
    """Extract typed fields from a CSV RawCandidateData record."""

    def extract(self, raw: RawCandidateData) -> ExtractedCandidate:
        """Map CSV columns to ExtractedCandidate fields.

        Args:
            raw: Record produced by CSVParser.

        Returns:
            Populated :class:`~app.models.candidate.ExtractedCandidate`.
        """
        if raw.parse_errors and not raw.raw_fields:
            logger.warning("Skipping extraction for errored CSV record: %s", raw.parse_errors)
            return self._empty(raw)

        fields = extract_known_fields(raw.raw_fields)
        rf = raw.raw_fields

        # Experience: only treat as a text blob when it contains newlines
        # (a genuine multi-line field); otherwise use flat sub-field columns.
        # This avoids misclassifying a short alias value (e.g. "3 years").
        exp_blob = fields.get("experience")
        if isinstance(exp_blob, str) and "\n" in exp_blob:
            experience = _parse_experience_text(exp_blob)
            if not experience:
                experience = _extract_experience_row(rf)
        else:
            experience = _extract_experience_row(rf)

        # Education: same dual-path strategy (newline check avoids treating
        # "qualification → MS" as a full education text blob).
        edu_blob = fields.get("education")
        if isinstance(edu_blob, str) and "\n" in edu_blob:
            education = _parse_education_text(edu_blob)
            if not education:
                education = _extract_education_row(rf)
        else:
            education = _extract_education_row(rf)

        misc = _extract_misc_fields(rf)

        extra = {
            k: v for k, v in rf.items()
            if _normalize_field_key(str(k)) not in _ALL_CONSUMED_KEYS
        }
        extra.update(misc)

        candidate = ExtractedCandidate(
            source=DataSource.CSV,
            source_file=raw.source_file,
            name=fields.get("name") or None,
            email=fields.get("email") or None,
            phone=fields.get("phone") or None,
            location=fields.get("location") or None,
            summary=fields.get("summary") or None,
            skills=_extract_skills(fields, rf),
            experience=experience,
            education=education,
            links=_extract_links(fields, rf),
            extra_fields=extra,
        )

        logger.debug(
            "CSVExtractor: name=%r email=%r skills=%d experience=%d education=%d links=%d",
            candidate.name,
            candidate.email,
            len(candidate.skills),
            len(candidate.experience),
            len(candidate.education),
            len(candidate.links),
        )
        return candidate
