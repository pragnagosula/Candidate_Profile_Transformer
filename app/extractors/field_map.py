"""Flexible field-name resolution for structured sources (CSV, JSON).

Maps the many ways a source might spell a field to our canonical name.
Lookup is case-insensitive and strips common punctuation/whitespace.
"""

from __future__ import annotations


def _normalise_key(key: str) -> str:
    """Lowercase, strip, replace spaces/hyphens with underscores."""
    return key.lower().strip().replace(" ", "_").replace("-", "_")


# Canonical field name → list of accepted aliases (lowercase, stripped)
FIELD_ALIASES: dict[str, list[str]] = {
    "name": [
        "name", "full_name", "fullname", "candidate_name", "candidatename",
        "applicant_name", "first_last", "person_name",
    ],
    "email": [
        "email", "email_address", "emailaddress", "e-mail", "mail",
        "contact_email", "work_email", "primary_email",
    ],
    "phone": [
        "phone", "phone_number", "phonenumber", "mobile", "mobile_number",
        "contact", "contact_number", "telephone", "cell", "mobile_no",
        "phone_no",
    ],
    "location": [
        "location", "city", "state", "country", "address", "current_location",
        "place", "residence", "city_state", "city_country", "current_city",
        "current_address", "hometown", "native_place",
    ],
    "summary": [
        "summary", "bio", "about", "profile", "objective", "about_me",
        "professional_summary", "overview", "professional_profile",
        "career_objective", "career_summary",
    ],
    "skills": [
        "skills", "skill", "technologies", "tech_stack", "expertise",
        "competencies", "tools", "languages", "technical_skills",
        "core_skills", "key_skills", "programming_languages", "frameworks",
        "libraries", "technical_expertise", "tech_skills",
        "core_competencies", "technical_competencies",
    ],
    "linkedin": [
        "linkedin", "linkedin_url", "linkedin_profile", "linkedin_link",
    ],
    "github": [
        "github", "github_url", "github_profile", "github_link",
    ],
    "portfolio": [
        "portfolio", "website", "personal_website", "portfolio_url",
    ],
    "experience": [
        "experience", "work_experience", "employment", "work_history",
    ],
    "education": [
        "education", "academic", "academics", "qualification", "qualifications",
    ],
}

# Inverted index: normalised_alias → canonical name
# Keys are normalised at build time so lookups are invariant to casing/punctuation.
_ALIAS_INDEX: dict[str, str] = {
    _normalise_key(alias): canonical
    for canonical, aliases in FIELD_ALIASES.items()
    for alias in aliases
}


def resolve_field(raw_key: str) -> str | None:
    """Map a raw source field name to its canonical name.

    Args:
        raw_key: The column/key name as it appears in the source file.

    Returns:
        Canonical field name (e.g. ``"email"``) or ``None`` if unknown.
    """
    return _ALIAS_INDEX.get(_normalise_key(raw_key))


def extract_known_fields(raw_fields: dict) -> dict[str, object]:
    """Resolve all known fields from a raw dict, ignoring unknown keys.

    Returns:
        Dict mapping canonical field name → raw value.
        If a canonical field appears via multiple aliases, the first wins.
    """
    result: dict[str, object] = {}
    for raw_key, value in raw_fields.items():
        canonical = resolve_field(str(raw_key))
        if canonical and canonical not in result:
            result[canonical] = value
    return result
