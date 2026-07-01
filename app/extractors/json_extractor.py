"""JSON extractor — maps JSON keys to typed ExtractedCandidate fields.

JSON sources may carry nested experience/education arrays, which this
extractor maps to typed Experience and Education model objects.

Skills and links are normalised the same way as the PDF extractor (shared
helpers imported from :mod:`app.extractors.pdf_extractor`) so a candidate
gets consistent results regardless of which source format they came from:

- Skills are cleaned, alias-normalised (``"node js"`` -> ``"Node.js"``,
  ``"cpp"`` -> ``"C++"``, etc.) and deduplicated. If a record has very few
  (or no) skills listed explicitly, a resume-wide technology-dictionary scan
  over all string values in the record fills in the gaps.
- Links recognise the same site set as the PDF extractor (LinkedIn, GitHub,
  Portfolio/Website, LeetCode, CodeChef, HackerRank, Kaggle, Medium,
  StackOverflow, plus generic URLs), are gathered from well-known field
  names, nested link containers (e.g. ``{"profiles": {"github": "..."}}``),
  generic link arrays (``links``, ``social_links``, ``social_profiles``),
  and any URLs embedded in free-text fields, then merged with duplicates
  removed.

Nested / wrapped structures are handled transparently.  Common wrappers
such as ``{"candidate": {...}}``, ``{"profile": {...}}``, and
``{"data": {"candidate": {...}}}`` are unwrapped before extraction.
Nested contact sections (e.g. ``{"contact": {"email": ..., "phone": ...}}``)
are searched one level deep for all scalar candidate fields.
"""

from __future__ import annotations

import re
from typing import Any

from app.extractors.base import BaseExtractor
from app.extractors.csv_extractor import _parse_skills
from app.extractors.field_map import extract_known_fields
from app.extractors.pdf_extractor import (
    _classify_url,
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
# JSON-structure constants
# ---------------------------------------------------------------------------

# Top-level dict keys that wrap the actual candidate object one level down.
# "profile" is intentionally included: {"profile": {...}} is a wrapper
# when the value is a dict (detected by _is_candidate_root).
_CANDIDATE_WRAPPER_KEYS: frozenset[str] = frozenset({
    "candidate", "profile", "applicant", "person", "data", "record", "resume",
})

# All alias spellings for the experience list field.
_EXPERIENCE_FIELD_KEYS: frozenset[str] = frozenset({
    "experience", "work_experience", "employment", "employment_history", "work_history",
})

# All alias spellings for the education list field.
_EDUCATION_FIELD_KEYS: frozenset[str] = frozenset({
    "education", "academics", "qualifications", "academic", "qualification",
})

# All alias spellings for the skills field (used in _is_candidate_root).
_SKILLS_FIELD_KEYS: frozenset[str] = frozenset({
    "skills", "skill", "technical_skills", "core_skills", "technology_stack",
    "technologies", "tech_stack",
})

# Misc structured list fields stored in extra_fields.
_PROJECTS_FIELD_KEYS: frozenset[str] = frozenset({
    "projects", "personal_projects", "project",
})
_CERT_FIELD_KEYS: frozenset[str] = frozenset({
    "certifications", "certificates", "certification", "certificate",
    "credentials", "courses", "training",
})
_ACHIEVE_FIELD_KEYS: frozenset[str] = frozenset({
    "achievements", "achievement", "awards", "honors", "honours", "accomplishments",
})
_LANG_FIELD_KEYS: frozenset[str] = frozenset({
    "languages", "languages_known", "spoken_languages", "language",
})

# Keys whose (list or dict) values hold link information.
_LINKS_CONTAINER_KEYS: frozenset[str] = frozenset({
    "links", "social_links", "social_profiles", "profiles",
})

# Scalar field names whose presence strongly implies a dict IS the candidate root,
# not a wrapper.  "profile" is deliberately absent so {"profile": {...}} is
# treated as a wrapper rather than a summary field.
_CANDIDATE_SCALAR_KEYS: frozenset[str] = frozenset({
    "name", "full_name", "fullname", "candidate_name", "applicant_name",
    "email", "email_address", "emailaddress", "e_mail", "mail",
    "phone", "phone_number", "mobile", "mobile_number", "telephone",
    "location", "city", "address", "current_location",
})

# Everything the extractor consumes — these keys are excluded from extra_fields.
_ALL_JSON_CONSUMED_KEYS: frozenset[str] = (
    frozenset({
        "name", "email", "phone", "location", "summary",
        "skills", "experience", "education",
    })
    | _EXPERIENCE_FIELD_KEYS
    | _EDUCATION_FIELD_KEYS
    | _SKILLS_FIELD_KEYS
    | _LINKS_CONTAINER_KEYS
    | _PROJECTS_FIELD_KEYS
    | _CERT_FIELD_KEYS
    | _ACHIEVE_FIELD_KEYS
    | _LANG_FIELD_KEYS
)

# Regex for stripping non-numeric chars from a GPA string.
_GPA_CLEANUP_RE = re.compile(r"[^\d.]")

# Regex for detecting "present / current / now / ongoing" end-date strings.
_PRESENT_RE = re.compile(r"^(?:present|current|now|ongoing)$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def _coerce_str(value: Any) -> str | None:
    """Return stripped string or None for blank/None values."""
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def _normalize_field_key(key: str) -> str:
    return key.strip().lower().replace(" ", "_").replace("-", "_")


def _first_in_item(item: dict, *keys: str) -> str | None:
    """Return the first non-empty coerced string for any of *keys* in *item*."""
    for key in keys:
        v = _coerce_str(item.get(key))
        if v:
            return v
    return None


# ---------------------------------------------------------------------------
# Wrapper unwrapping + field collection
# ---------------------------------------------------------------------------


def _is_candidate_root(data: dict) -> bool:
    """Return True if *data* likely contains candidate fields at the top level.

    True when any recognised scalar key (name, email, phone, …) has a
    non-empty string value, OR when the experience/education/skills list
    keys are present.  This distinguishes a real candidate dict from a
    single-key wrapper like ``{"candidate": {...}}``.
    """
    for key, value in data.items():
        nk = _normalize_field_key(key)
        if nk in _CANDIDATE_SCALAR_KEYS and isinstance(value, str) and value.strip():
            return True
        if nk in (_EXPERIENCE_FIELD_KEYS | _EDUCATION_FIELD_KEYS | _SKILLS_FIELD_KEYS):
            return True
    return False


def _unwrap_candidate_data(raw_fields: dict, _depth: int = 0) -> dict:
    """Recursively unwrap common JSON wrapper structures to reach the candidate dict.

    Handles:
      ``{"candidate": {...}}``           → inner dict
      ``{"profile":   {...}}``           → inner dict (when value is a dict)
      ``{"data": {"candidate": {...}}}`` → two-level unwrap

    Stops when *raw_fields* already contains known candidate fields, when
    no recognised wrapper key with a dict value is found, or after three
    levels of recursion.
    """
    if not isinstance(raw_fields, dict) or _depth > 3:
        return raw_fields if isinstance(raw_fields, dict) else {}
    if _is_candidate_root(raw_fields):
        return raw_fields
    for key, value in raw_fields.items():
        nk = _normalize_field_key(key)
        if nk in _CANDIDATE_WRAPPER_KEYS and isinstance(value, dict):
            return _unwrap_candidate_data(value, _depth + 1)
    return raw_fields


_SCALAR_CANONICALS: frozenset[str] = frozenset({"name", "email", "phone", "location", "summary"})


def _collect_candidate_fields(data: dict) -> dict:
    """Collect all recognised scalar fields from *data* and one level of nesting.

    Handles nested contact sections such as
    ``{"contact": {"email": ..., "phone": ...}}`` that ``extract_known_fields``
    would miss at the top level.

    NOTE: "contact" is a phone alias in the field map, so when a key maps to
    a scalar canonical field but the value is a dict (e.g. a contact section),
    that mapping is discarded and the dict is explored one level deeper instead.
    """
    raw = extract_known_fields(data)
    # Keep only string values for scalar fields — a dict value means the key
    # accidentally matched an alias for a nested section (e.g. "contact" → phone).
    fields: dict = {
        canon: val for canon, val in raw.items()
        if not (canon in _SCALAR_CANONICALS and not isinstance(val, str))
    }
    # One level deeper — any dict-valued key that is not a structured list.
    skip = _EXPERIENCE_FIELD_KEYS | _EDUCATION_FIELD_KEYS | _LINKS_CONTAINER_KEYS
    for key, value in data.items():
        if not isinstance(value, dict):
            continue
        if _normalize_field_key(key) in skip:
            continue
        for canon, val in extract_known_fields(value).items():
            if canon in fields:
                continue
            if canon in _SCALAR_CANONICALS and not isinstance(val, str):
                continue
            fields[canon] = val
    return fields


def _find_list_field(data: dict, field_keys: frozenset[str]) -> list:
    """Return the first list value found under any recognised alias key in *data*."""
    for key, value in data.items():
        if _normalize_field_key(key) in field_keys and isinstance(value, list):
            return value
    return []


# ---------------------------------------------------------------------------
# Experience parsing
# ---------------------------------------------------------------------------


def _parse_experience(raw_list: Any) -> list[Experience]:
    """Convert a JSON experience array to typed Experience objects.

    Supports common field aliases:
      company   → company / organization / employer
      title     → title / role / position / designation
      start     → start / start_date / from / from_date
      end       → end / end_date / to / to_date (``present``/``current`` → is_current)
      desc      → description / summary / responsibilities / details
    """
    if not isinstance(raw_list, list):
        return []

    result: list[Experience] = []
    for item in raw_list:
        if not isinstance(item, dict):
            continue

        start = _first_in_item(item, "start", "start_date", "from", "from_date")
        end_raw = _first_in_item(item, "end", "end_date", "to", "to_date")

        # Detect explicit is_current flags
        is_current_flag = _coerce_str(
            item.get("is_current") or item.get("current") or item.get("currently_working")
        )
        is_current = bool(
            is_current_flag
            and is_current_flag.lower() in ("true", "yes", "1", "current")
        )

        if end_raw:
            if _PRESENT_RE.match(end_raw):
                is_current = True
                end: str | None = None
            else:
                end = end_raw
        else:
            end = None
            if start and not is_current:
                is_current = True

        result.append(
            Experience(
                company=_first_in_item(item, "company", "organization", "employer"),
                title=_first_in_item(item, "title", "role", "position", "designation"),
                location=_coerce_str(item.get("location")),
                description=_first_in_item(
                    item, "description", "summary", "responsibilities", "details"
                ),
                duration=DateRange(start=start, end=end, is_current=is_current)
                if (start or end)
                else None,
            )
        )
    return result


# ---------------------------------------------------------------------------
# Education parsing
# ---------------------------------------------------------------------------


def _extract_item_gpa(item: dict) -> float | None:
    """Extract GPA/percentage from a JSON education item, checking all aliases."""
    for key in ("cgpa", "gpa", "percentage", "grade", "marks", "score"):
        raw = item.get(key)
        if raw is None:
            continue
        try:
            cleaned = _GPA_CLEANUP_RE.sub("", str(raw))[:10]
            if cleaned:
                return float(cleaned)
        except ValueError:
            pass
    return None


def _parse_education(raw_list: Any) -> list[Education]:
    """Convert a JSON education array to typed Education objects.

    Supports common field aliases:
      institution → institution / university / school / college / alma_mater / institute
      degree      → degree / qualification / program / course
      field       → field / field_of_study / major / specialization / branch / stream
      gpa         → cgpa / gpa / percentage / grade / marks / score
      start year  → start_year / admission_year / joining_year / from_year
      end year    → end_year / graduation_year / passing_year / year_of_passing /
                    year_of_graduation / year
    """
    if not isinstance(raw_list, list):
        return []

    result: list[Education] = []
    for item in raw_list:
        if not isinstance(item, dict):
            continue

        start_year = _first_in_item(
            item, "start_year", "admission_year", "joining_year", "from_year"
        )
        end_year = _first_in_item(
            item,
            "end_year", "graduation_year", "passing_year",
            "year_of_passing", "year_of_graduation", "year",
        )

        result.append(
            Education(
                institution=_first_in_item(
                    item, "institution", "university", "school", "college",
                    "alma_mater", "institute",
                ),
                degree=_first_in_item(item, "degree", "qualification", "program", "course"),
                field_of_study=_first_in_item(
                    item, "field", "field_of_study", "major",
                    "specialization", "specialisation", "branch", "stream",
                ),
                duration=DateRange(start=start_year, end=end_year)
                if (start_year or end_year)
                else None,
                gpa=_extract_item_gpa(item),
            )
        )
    return result


# ---------------------------------------------------------------------------
# Misc fields — projects, certifications, achievements, spoken languages
# ---------------------------------------------------------------------------


def _extract_misc_fields(candidate_data: dict) -> dict[str, list[str]]:
    """Extract projects, certifications, achievements, and spoken languages.

    Values are returned as string lists to be stored in ``extra_fields``.
    """
    result: dict[str, list[str]] = {}

    def _collect(field_keys: frozenset[str], out_key: str) -> None:
        for key, value in candidate_data.items():
            if _normalize_field_key(key) not in field_keys:
                continue
            if isinstance(value, list):
                items = [str(s).strip() for s in value if isinstance(s, str) and s.strip()]
                if not items:
                    # list of dicts — pull all leaf string values
                    items = [
                        str(v).strip()
                        for entry in value
                        if isinstance(entry, dict)
                        for v in entry.values()
                        if isinstance(v, str) and v.strip()
                    ]
            elif isinstance(value, str) and value.strip():
                items = _parse_skills(value)   # reuse delimiter-splitting
            else:
                continue
            if items:
                result[out_key] = items
                return

    _collect(_PROJECTS_FIELD_KEYS, "projects")
    _collect(_CERT_FIELD_KEYS, "certifications")
    _collect(_ACHIEVE_FIELD_KEYS, "achievements")
    _collect(_LANG_FIELD_KEYS, "languages")
    return result


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


def _flatten_strings(value: Any) -> list[str]:
    """Recursively collect every string leaf value out of a JSON-like structure."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        out: list[str] = []
        for v in value.values():
            out.extend(_flatten_strings(v))
        return out
    if isinstance(value, list):
        out = []
        for v in value:
            out.extend(_flatten_strings(v))
        return out
    return []


def _extract_skills(fields: dict, raw_fields: dict) -> list[str]:
    """Extract a normalised, deduplicated skill list, with a fallback scan.

    Mirrors the PDF extractor's behaviour: if the structured ``skills``
    field is missing or sparse (<5 entries), the whole record is scanned
    for known technology names so sparse JSON exports still yield useful
    skills.
    """
    raw_skills = _parse_skills(fields.get("skills", []))
    skills = _normalize_and_dedupe_skills(raw_skills)

    if len(skills) < 5:
        text_blob = "\n".join(_flatten_strings(raw_fields))
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
# Links — structured fields + nested containers + link arrays + free-text scan
# ---------------------------------------------------------------------------

# JSON field name (normalised) → display label for direct URL fields
_LINK_FIELD_ALIASES: dict[str, str] = {
    "linkedin": "LinkedIn", "linkedin_url": "LinkedIn", "linkedin_profile": "LinkedIn",
    "github": "GitHub", "github_url": "GitHub", "github_profile": "GitHub",
    "portfolio": "Portfolio", "portfolio_url": "Portfolio",
    "website": "Website", "personal_website": "Website", "personal_site": "Website",
    "leetcode": "LeetCode", "leetcode_url": "LeetCode", "leetcode_profile": "LeetCode",
    "codechef": "CodeChef", "codechef_url": "CodeChef", "codechef_profile": "CodeChef",
    "hackerrank": "HackerRank", "hackerrank_url": "HackerRank",
    "hackerrank_profile": "HackerRank",
    "kaggle": "Kaggle", "kaggle_url": "Kaggle", "kaggle_profile": "Kaggle",
    "medium": "Medium", "medium_url": "Medium", "medium_profile": "Medium",
    "stackoverflow": "StackOverflow", "stackoverflow_url": "StackOverflow",
    "stackoverflow_profile": "StackOverflow",
}


def _parse_structured_links(candidate_data: dict) -> list[Link]:
    """Pick up well-known single-URL fields, e.g. ``"linkedin": "..."``."""
    links: list[Link] = []
    for raw_key, value in candidate_data.items():
        if not isinstance(value, str) or not value.strip():
            continue
        label = _LINK_FIELD_ALIASES.get(_normalize_field_key(str(raw_key)))
        if label:
            links.append(Link(url=_normalize_link_url(value.strip()), label=label))
    return links


def _parse_nested_link_container(candidate_data: dict) -> list[Link]:
    """Extract links from nested dict containers.

    Handles ``{"profiles": {"github": "...", "linkedin": "..."}}`` and
    similar structures where a container key holds a dict of named URLs.
    """
    links: list[Link] = []
    for key, value in candidate_data.items():
        if _normalize_field_key(key) not in _LINKS_CONTAINER_KEYS:
            continue
        if not isinstance(value, dict):
            continue
        for inner_key, inner_value in value.items():
            if not isinstance(inner_value, str) or not inner_value.strip():
                continue
            label = _LINK_FIELD_ALIASES.get(_normalize_field_key(inner_key))
            if label:
                links.append(Link(url=_normalize_link_url(inner_value.strip()), label=label))
            elif inner_value.strip().startswith(("http://", "https://")):
                url = _normalize_link_url(inner_value.strip())
                links.append(Link(url=url, label=_classify_url(url)))
    return links


def _parse_links_array(candidate_data: dict) -> list[Link]:
    """Handle link arrays under any recognised container key.

    Supports ``links``, ``social_links``, ``social_profiles``, and ``profiles``
    when the value is a list of URL strings or ``{"url": ..., "label": ...}``
    dicts.
    """
    links: list[Link] = []
    for key, raw_list in candidate_data.items():
        if _normalize_field_key(key) not in _LINKS_CONTAINER_KEYS:
            continue
        if not isinstance(raw_list, list):
            continue
        for item in raw_list:
            if isinstance(item, str) and item.strip():
                url = _normalize_link_url(item.strip())
                links.append(Link(url=url, label=_classify_url(url)))
            elif isinstance(item, dict):
                raw_url = item.get("url") or item.get("href") or item.get("link")
                if raw_url and isinstance(raw_url, str) and raw_url.strip():
                    url = _normalize_link_url(raw_url.strip())
                    label = item.get("label") or item.get("type") or _classify_url(url)
                    links.append(Link(url=url, label=str(label)))
    return links


def _parse_links(candidate_data: dict) -> list[Link]:
    """Extract candidate links from all recognised sources.

    Priority order (first occurrence wins on duplicate URLs):
        1. Well-known single-URL fields (``linkedin``, ``github``, …)
        2. Nested dict containers (``{"profiles": {"github": "..."}}`` etc.)
        3. Array containers (``links``, ``social_links``, ``social_profiles``)
        4. URLs found anywhere in the record's text values
    """
    structured = _parse_structured_links(candidate_data)
    nested     = _parse_nested_link_container(candidate_data)
    array_links = _parse_links_array(candidate_data)
    text_blob  = "\n".join(_flatten_strings(candidate_data))
    text_links = _extract_links_from_text(text_blob) if text_blob else []
    return _merge_links(structured + nested, array_links, text_links)


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


@extractor_registry.register(DataSource.JSON)
class JSONExtractor(BaseExtractor):
    """Extract typed fields from a JSON RawCandidateData record."""

    def extract(self, raw: RawCandidateData) -> ExtractedCandidate:
        """Map JSON keys to ExtractedCandidate fields.

        Handles flat, wrapped (``candidate`` / ``profile`` / ``data``), and
        nested JSON structures transparently.

        Args:
            raw: Record produced by JSONParser.

        Returns:
            Populated :class:`~app.models.candidate.ExtractedCandidate`.
        """
        if raw.parse_errors and not raw.raw_fields:
            logger.warning("Skipping extraction for errored JSON record: %s", raw.parse_errors)
            return self._empty(raw)

        # Step 1: Unwrap common wrapper structures to reach the candidate dict.
        candidate_data = _unwrap_candidate_data(raw.raw_fields)

        # Step 2: Collect scalar fields at top level and one level of nesting.
        fields = _collect_candidate_fields(candidate_data)

        # Step 3: Find structured list fields using all recognised aliases.
        exp_list = _find_list_field(candidate_data, _EXPERIENCE_FIELD_KEYS)
        edu_list = _find_list_field(candidate_data, _EDUCATION_FIELD_KEYS)

        # Step 4: Extract misc list fields (projects, certs, achievements, languages).
        misc = _extract_misc_fields(candidate_data)

        # Step 5: Build extra_fields from candidate_data, excluding consumed keys.
        extra: dict[str, Any] = {
            k: v for k, v in candidate_data.items()
            if _normalize_field_key(str(k)) not in _ALL_JSON_CONSUMED_KEYS
        }
        extra.update(misc)

        candidate = ExtractedCandidate(
            source=DataSource.JSON,
            source_file=raw.source_file,
            name=_coerce_str(fields.get("name")),
            email=_coerce_str(fields.get("email")),
            phone=_coerce_str(fields.get("phone")),
            location=_coerce_str(fields.get("location")),
            summary=_coerce_str(fields.get("summary")),
            skills=_extract_skills(fields, raw.raw_fields),
            experience=_parse_experience(exp_list),
            education=_parse_education(edu_list),
            links=_parse_links(candidate_data),
            extra_fields=extra,
        )

        logger.debug(
            "JSONExtractor: name=%r email=%r skills=%d experience=%d education=%d links=%d",
            candidate.name,
            candidate.email,
            len(candidate.skills),
            len(candidate.experience),
            len(candidate.education),
            len(candidate.links),
        )
        return candidate
