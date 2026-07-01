"""Shared text-resume parsing logic.

Single source of truth for all resume section detection, field extraction,
skill normalisation, and link recognition.  Both :class:`PDFExtractor` and
:class:`TXTExtractor` delegate to :class:`TextResumeParser` — neither
contains any parsing logic of its own.

Usage::

    from app.extractors.text_resume_parser import TextResumeParser

    candidate = TextResumeParser().parse(
        text,
        source=DataSource.RESUME_PDF,
        source_file="resume.pdf",
        extra_links=annotation_links,   # PDF-only, omit for TXT
    )
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from app.models.candidate import (
    DataSource,
    DateRange,
    Education,
    Experience,
    ExtractedCandidate,
    Link,
)
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Compiled patterns (module-level — compiled once)
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
    re.IGNORECASE,
)

_PHONE_RE = re.compile(
    r"(?:\+?\d{1,3}[\s\-.]?)?"          # optional country code
    r"(?:\(?\d{2,4}\)?[\s\-.]?)?"       # optional area code
    r"\d{3,5}[\s\-.]?\d{3,5}"           # local number
    r"(?:[\s\-.]?\d{1,4})?",            # optional extension
    re.IGNORECASE,
)

# Section header patterns — order matters (most specific first).
# Used with fullmatch against the *normalised* heading text (lowercase, no
# leading/trailing punctuation) produced by _normalize_heading_text().
_SECTION_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("experience",    re.compile(
        r"(?:work\s+|professional\s+)?experience"
        r"|employment(?:\s+history)?"
        r"|work\s+history",
        re.IGNORECASE,
    )),
    ("education",     re.compile(
        r"education(?:al?\s+background)?"
        r"|academic(?:s|(?:\s+background))?",
        re.IGNORECASE,
    )),
    ("skills",        re.compile(
        r"(?:technical\s+)?skills?"
        r"|technologies|competenc(?:y|ies)|expertise",
        re.IGNORECASE,
    )),
    ("summary",       re.compile(
        r"(?:professional\s+)?summary"
        r"|profile|objective|about(?:\s+me)?",
        re.IGNORECASE,
    )),
    ("projects",      re.compile(
        r"projects?|personal\s+projects?",
        re.IGNORECASE,
    )),
    ("certifications", re.compile(
        r"certifications?|certificates?|licenses?",
        re.IGNORECASE,
    )),
    # Boundary-only sections: content not parsed by TextResumeParser but must
    # terminate the preceding section (e.g. experience) so their text is not
    # mis-attributed as work entries.
    ("achievements",  re.compile(
        r"achievements?|awards?|honors?|honours?"
        r"|accolades?|accomplishments?|recognition",
        re.IGNORECASE,
    )),
    ("leadership",    re.compile(
        r"positions?\s+of\s+responsibility"
        r"|leadership(?:\s+(?:roles?|experience))?"
        r"|extracurricular(?:\s+activities?)?",
        re.IGNORECASE,
    )),
    ("other",         re.compile(
        r"publications?|references?|interests?"
        r"|hobbies|volunteer(?:ing)?|activities",
        re.IGNORECASE,
    )),
]

# Date patterns within experience/education blocks
_DATE_RE = re.compile(
    r"(?:"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s,]+\d{4}"
    r"|(?:0?[1-9]|1[0-2])[/\-]\d{4}"
    r"|\d{4}"
    r")"
    r"(?:\s*[-–—]\s*"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s,]+\d{4}"
    r"|(?:0?[1-9]|1[0-2])[/\-]\d{4}"
    r"|present|current|now|ongoing"
    r"|\d{4}"
    r")?",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Skills section detection constants
# ---------------------------------------------------------------------------

_SKILLS_HEADING_PHRASES: list[str] = [
    "skills", "skill",
    "technical skills", "technical skill",
    "technical skills and coursework", "technical skills & coursework",
    "technical expertise",
    "core skills", "professional skills",
    "technical competencies", "competencies", "core competencies",
    "professional competencies",
    "areas of expertise", "expertise",
    "technology stack", "tech stack",
    "tools", "technologies",
    "programming languages", "programming skills",
    "languages & technologies", "languages and technologies",
    "software skills", "software proficiency",
    "development skills", "engineering skills",
    "technical proficiencies",
    "key skills", "primary skills",
    "skills summary", "relevant skills",
    "relevant technologies", "technology summary",
    "technical knowledge", "knowledge areas",
    "technology experience",
    "languages", "frameworks", "libraries",
    "developer tools", "tool stack",
]
_SKILLS_HEADING_SET = {p.strip().lower() for p in _SKILLS_HEADING_PHRASES}

_OTHER_SECTION_HEADING_PHRASES: list[str] = [
    "experience", "work experience", "professional experience", "work history",
    "employment", "employment history",
    "education", "educational background", "academic background", "academics",
    "projects", "personal projects",
    "certifications", "certificates", "licenses",
    "summary", "professional summary", "profile", "objective", "about", "about me",
    "awards", "achievements", "achievement", "honors", "honours", "accolades",
    "accomplishments", "recognition",
    "publications", "interests", "references",
    "volunteer", "volunteering", "extracurricular activities", "activities",
    "hobbies", "languages spoken",
    "position of responsibility", "positions of responsibility",
    "leadership", "leadership roles", "leadership experience",
]
_OTHER_SECTION_HEADING_SET = {p.strip().lower() for p in _OTHER_SECTION_HEADING_PHRASES}

# Category labels that appear *inside* a skills section but are not skills.
_NOISE_LABELS = {
    "programming languages", "frameworks", "libraries", "databases",
    "developer tools", "machine learning", "cloud", "tools", "technologies",
    "frontend", "backend", "devops", "languages", "tool stack",
    "core cs", "ai/ml", "ai", "ml",
}

_HEADING_TRAILING_RE = re.compile(r"[\s:\-|•]+$")
_HEADING_LEADING_RE  = re.compile(r"^[\s:\-|•]+")
_WHITESPACE_RE        = re.compile(r"\s+")
_LEADING_MARKER_RE    = re.compile(r"^[\s•▪◦‣➜o\*\-]+")
_TOKEN_SPLIT_RE       = re.compile(r"[,;|/]+")

# Strips label prefixes like "Role:", "Title:", "Company:", "Duration:" that some
# text/JSON sources inject into experience field values.
_EXP_LABEL_PREFIX_RE = re.compile(
    r"^(?:role|title|position|designation|company|organization|employer|duration)\s*:\s*",
    re.IGNORECASE,
)

# Detects lines that read as a job title rather than a company name.
# Used to correct swapped company/title (Indian resume format puts company first).
_JOB_TITLE_KEYWORDS_RE = re.compile(
    r"\b(?:intern|engineer|developer|analyst|manager|lead|architect|consultant|"
    r"associate|junior|senior|designer|scientist|researcher|officer|specialist|"
    r"director|head|coordinator|executive|trainee|apprentice|programmer|"
    r"technician|administrator|assistant)\b",
    re.IGNORECASE,
)

# Degree abbreviations/keywords anchored at line-start; used to detect the
# beginning of a new education entry within a block of lines.
_DEGREE_HEADING_RE = re.compile(
    r"^(?:B\.?Tech|B\.?E\.?|M\.?Tech|M\.?E\.?|M\.?S(?:c)?|B\.?Sc|B\.?Com|"
    r"B\.?A|MBA|Ph\.?D|MCA|BCA|Bachelor|Master|Diploma|Associate)\b",
    re.IGNORECASE,
)

# Lines that carry a GPA/CGPA label (education extraction only).
_GPA_LINE_RE = re.compile(
    r"\b(?:cgpa|gpa|percentage|grade|marks|score)\b",
    re.IGNORECASE,
)

# Label prefix on the same line as values, e.g. "Frameworks: React, Django"
_LABEL_PREFIX_RE = re.compile(
    r"^\s*(" + "|".join(sorted((re.escape(l) for l in _NOISE_LABELS), key=len, reverse=True)) + r")\s*[:\-|•]\s*",
    re.IGNORECASE,
)

_INLINE_SKILLS_HEADING_RE = re.compile(
    r"^\s*(?:"
    + "|".join(sorted((re.escape(p) for p in _SKILLS_HEADING_PHRASES), key=len, reverse=True))
    + r")\s*[:\-|•]\s*(\S.*)$",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Skill normalisation
# ---------------------------------------------------------------------------

_SKILL_ALIASES: dict[str, str] = {
    "python": "Python",
    "node.js": "Node.js", "nodejs": "Node.js", "node js": "Node.js",
    "react js": "React.js", "reactjs": "React.js", "react.js": "React.js", "react": "React",
    "c sharp": "C#", "csharp": "C#", "c#": "C#",
    "cpp": "C++", "c plus plus": "C++", "c++": "C++",
    "machine learning": "Machine Learning", "ml": "Machine Learning",
    "deep learning": "Deep Learning", "dl": "Deep Learning",
    "javascript": "JavaScript", "js": "JavaScript",
    "typescript": "TypeScript", "ts": "TypeScript",
    "html5": "HTML", "html": "HTML",
    "css3": "CSS", "css": "CSS",
    "sql": "SQL", "postgresql": "PostgreSQL", "postgres": "PostgreSQL",
    "mongodb": "MongoDB", "mysql": "MySQL",
    "aws": "AWS", "gcp": "GCP", "azure": "Azure",
    "docker": "Docker", "kubernetes": "Kubernetes", "k8s": "Kubernetes",
    "git": "Git", "github": "GitHub",
    "tensorflow": "TensorFlow", "pytorch": "PyTorch",
    "numpy": "NumPy", "pandas": "Pandas",
    "oop": "OOP", "dbms": "DBMS",
}

_TECH_DICTIONARY: list[str] = [
    "Python", "Java", "C", "C++", "C#", "JavaScript", "TypeScript", "Go", "Rust", "Kotlin", "Swift",
    "React", "Angular", "Vue", "Next.js", "HTML", "CSS", "Bootstrap", "Tailwind",
    "Node.js", "Express", "FastAPI", "Flask", "Django", "Spring Boot", "ASP.NET MVC",
    "TensorFlow", "PyTorch", "Scikit-learn", "Keras", "OpenCV", "NumPy", "Pandas",
    "Matplotlib", "Seaborn", "XGBoost", "LightGBM",
    "MongoDB", "MySQL", "PostgreSQL", "SQLite", "Redis", "Oracle",
    "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Jenkins",
    "Git", "GitHub", "VS Code", "Visual Studio", "Linux",
    "OOP", "DBMS", "Operating Systems", "Computer Networks", "Algorithms", "System Design",
]


def _make_term_pattern(term: str) -> re.Pattern:
    """Build a whole-word-ish, case-insensitive pattern for a dictionary term."""
    escaped = re.escape(term)
    return re.compile(
        r"(?<![A-Za-z0-9])" + escaped + r"(?![A-Za-z0-9+#.])",
        re.IGNORECASE,
    )


_TECH_PATTERNS: dict[str, re.Pattern] = {term: _make_term_pattern(term) for term in _TECH_DICTIONARY}

# ---------------------------------------------------------------------------
# Link recognition constants
# ---------------------------------------------------------------------------

_KNOWN_LINK_SITES: list[tuple[str, re.Pattern]] = [
    ("LinkedIn",       re.compile(r"(?:^|//|\.)linkedin\.com",      re.IGNORECASE)),
    ("GitHub",         re.compile(r"(?:^|//|\.)github\.com",        re.IGNORECASE)),
    ("LeetCode",       re.compile(r"(?:^|//|\.)leetcode\.com",      re.IGNORECASE)),
    ("CodeChef",       re.compile(r"(?:^|//|\.)codechef\.com",      re.IGNORECASE)),
    ("HackerRank",     re.compile(r"(?:^|//|\.)hackerrank\.com",    re.IGNORECASE)),
    ("Kaggle",         re.compile(r"(?:^|//|\.)kaggle\.com",        re.IGNORECASE)),
    ("Medium",         re.compile(r"(?:^|//|\.)medium\.com",        re.IGNORECASE)),
    ("StackOverflow",  re.compile(r"(?:^|//|\.)stackoverflow\.com", re.IGNORECASE)),
]

_KNOWN_SITE_TEXT_PATTERNS: dict[str, re.Pattern] = {
    "LinkedIn":      re.compile(r"(?:https?://)?(?:www\.)?linkedin\.com/[\w\-_%/]+",      re.IGNORECASE),
    "GitHub":        re.compile(r"(?:https?://)?(?:www\.)?github\.com/[\w\-_%/]+",        re.IGNORECASE),
    "LeetCode":      re.compile(r"(?:https?://)?(?:www\.)?leetcode\.com/[\w\-_%/]+",      re.IGNORECASE),
    "CodeChef":      re.compile(r"(?:https?://)?(?:www\.)?codechef\.com/[\w\-_%/]+",      re.IGNORECASE),
    "HackerRank":    re.compile(r"(?:https?://)?(?:www\.)?hackerrank\.com/[\w\-_%/]+",    re.IGNORECASE),
    "Kaggle":        re.compile(r"(?:https?://)?(?:www\.)?kaggle\.com/[\w\-_%/]+",        re.IGNORECASE),
    "Medium":        re.compile(r"(?:https?://)?(?:www\.)?medium\.com/[\w\-_%/@]+",       re.IGNORECASE),
    "StackOverflow": re.compile(r"(?:https?://)?(?:www\.)?stackoverflow\.com/[\w\-_%/]+", re.IGNORECASE),
}

_GENERIC_URL_RE = re.compile(r"https?://[^\s()<>\"']+", re.IGNORECASE)

_HYPERLINK_LABEL_WORDS = {
    "linkedin", "github", "portfolio", "website", "leetcode",
    "codechef", "hackerrank", "kaggle", "medium", "stackoverflow",
}

# ---------------------------------------------------------------------------
# Heading helpers
# ---------------------------------------------------------------------------


def _normalize_heading_text(line: str) -> str:
    """Strip surrounding markers/whitespace and collapse internal spaces."""
    core = _HEADING_TRAILING_RE.sub("", line.strip())
    core = _HEADING_LEADING_RE.sub("", core)
    return _WHITESPACE_RE.sub(" ", core).strip().lower()


def _is_skills_heading(line: str) -> bool:
    return _normalize_heading_text(line) in _SKILLS_HEADING_SET


def _is_any_section_heading(line: str) -> bool:
    """True if ``line`` looks like any resume section heading."""
    stripped = line.strip()
    if not stripped or len(stripped) > 60:
        return False
    compact = stripped.replace(" ", "")
    if not compact:
        return False
    alpha_chars = sum(1 for c in compact if c.isalpha())
    if alpha_chars / len(compact) < 0.7:
        return False
    norm = _normalize_heading_text(stripped)
    return norm in _SKILLS_HEADING_SET or norm in _OTHER_SECTION_HEADING_SET


# ---------------------------------------------------------------------------
# Skills extraction helpers
# ---------------------------------------------------------------------------


def _find_skills_section_flexible(text: str) -> str:
    """Locate a skills section using flexible heading matching (fallback)."""
    lines = text.splitlines()
    start = None
    inline_remainder = None
    for i, line in enumerate(lines):
        if _is_skills_heading(line):
            start = i
            break
        inline_match = _INLINE_SKILLS_HEADING_RE.match(line.strip())
        if inline_match:
            start = i
            inline_remainder = inline_match.group(1)
            break
    if start is None:
        return ""

    end = len(lines)
    for j in range(start + 1, len(lines)):
        if _is_any_section_heading(lines[j]):
            end = j
            break

    rest = "\n".join(lines[start + 1:end]).strip()
    if inline_remainder:
        return (inline_remainder + "\n" + rest).strip()
    return rest


def _strip_label_prefix(line: str) -> str:
    """Remove a leading category label (e.g. 'Frameworks: ') from a line."""
    return _LABEL_PREFIX_RE.sub("", line)


def _normalize_skill(raw: str) -> str:
    """Clean whitespace/punctuation and apply alias normalisation."""
    cleaned = _WHITESPACE_RE.sub(" ", raw).strip().strip(".,;:|•-")
    if not cleaned:
        return ""
    key = cleaned.lower()
    return _SKILL_ALIASES.get(key, cleaned)


def _parse_skill_tokens(section_text: str) -> list[str]:
    """Parse raw skill tokens out of a skills-section block."""
    tokens: list[str] = []
    for raw_line in section_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = _strip_label_prefix(line)
        if not line.strip():
            continue
        if _normalize_heading_text(line) in _NOISE_LABELS:
            continue
        line = _LEADING_MARKER_RE.sub("", line)
        if not line.strip():
            continue
        for part in _TOKEN_SPLIT_RE.split(line):
            part = part.strip()
            if not part or len(part) <= 1:
                continue
            if _normalize_heading_text(part) in _NOISE_LABELS:
                continue
            tokens.append(part)
    return tokens


def _scan_tech_dictionary(text: str) -> list[str]:
    """Whole-word, case-insensitive scan for known tech terms."""
    return [term for term, pattern in _TECH_PATTERNS.items() if pattern.search(text)]


def _extract_skills(section_text: str, full_text: str = "") -> list[str]:
    """Extract a deduplicated, normalised list of skills.

    Falls back to flexible heading detection and dictionary scanning when
    the primary section text is absent or sparse.
    """
    raw_tokens = _parse_skill_tokens(section_text) if section_text else []

    if not section_text and full_text:
        flexible_section = _find_skills_section_flexible(full_text)
        if flexible_section:
            raw_tokens = _parse_skill_tokens(flexible_section)

    normalized: list[str] = []
    seen: set[str] = set()
    for tok in raw_tokens:
        norm = _normalize_skill(tok)
        if not norm:
            continue
        key = norm.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(norm)

    if len(normalized) < 5 and full_text:
        for hit in _scan_tech_dictionary(full_text):
            key = hit.lower()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(hit)

    return normalized


# ---------------------------------------------------------------------------
# Section detection
# ---------------------------------------------------------------------------


@dataclass
class _Section:
    name: str
    start: int
    end: int
    content: str


def _detect_sections(text: str) -> dict[str, str]:
    """Split resume text into named sections.

    Returns a dict mapping section name → section content.
    Everything before the first detected heading is stored under ``_header``.

    Headings are normalised before matching (trailing ``:``, ``-``, ``|``
    stripped; internal whitespace collapsed; lowercased) so that headings
    like ``"Achievements:"`` or ``"WORK EXPERIENCE"`` are recognised the
    same as their plain forms.
    """
    lines = text.splitlines()
    boundaries: list[tuple[int, str]] = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or len(stripped) > 60:
            continue
        norm = _normalize_heading_text(stripped)
        if not norm:
            continue
        for section_name, pattern in _SECTION_PATTERNS:
            if pattern.fullmatch(norm):
                boundaries.append((i, section_name))
                break

    sections: dict[str, str] = {}
    for idx, (line_idx, name) in enumerate(boundaries):
        next_line = boundaries[idx + 1][0] if idx + 1 < len(boundaries) else len(lines)
        content = "\n".join(lines[line_idx + 1: next_line]).strip()
        sections[name] = content

    if boundaries:
        sections["_header"] = "\n".join(lines[: boundaries[0][0]]).strip()
    else:
        sections["_header"] = text.strip()

    return sections


# ---------------------------------------------------------------------------
# Field extractors
# ---------------------------------------------------------------------------


def _extract_email(text: str) -> Optional[str]:
    match = _EMAIL_RE.search(text)
    return match.group(0) if match else None


def _extract_phone(text: str) -> Optional[str]:
    """Return the first plausible phone number (7–15 digits)."""
    for match in _PHONE_RE.finditer(text):
        candidate = match.group(0).strip()
        digits = re.sub(r"\D", "", candidate)
        if 7 <= len(digits) <= 15:
            return candidate
    return None


def _extract_name(header_block: str, email: Optional[str]) -> Optional[str]:
    """Heuristic: first 2–6-word, alpha-only line in the header that isn't a
    contact detail.
    """
    skip_patterns = re.compile(
        r"@|http|www\.|linkedin|github|^\+?\d[\d\s\-().]{6,}$|^\d{10}$",
        re.IGNORECASE,
    )
    for line in header_block.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if skip_patterns.search(stripped):
            continue
        words = stripped.split()
        if 1 < len(words) <= 6 and all(re.match(r"^[A-Za-z'\-\.]+$", w) for w in words):
            return stripped
    return None


# ---------------------------------------------------------------------------
# Link extraction
# ---------------------------------------------------------------------------


def _normalize_url_for_dedupe(url: str) -> str:
    """Produce a comparison key that ignores scheme, www, and trailing slash."""
    key = url.strip().lower()
    key = re.sub(r"^https?://", "", key)
    key = re.sub(r"^www\.", "", key)
    key = key.rstrip("/")
    return key


def _classify_url(url: str) -> str:
    """Return a display label for a URL based on its domain."""
    for label, pattern in _KNOWN_LINK_SITES:
        if pattern.search(url):
            return label
    return "Website"


def _normalize_link_url(url: str) -> str:
    """Ensure a URL has an explicit https scheme."""
    url = url.strip()
    if not re.match(r"^https?://", url, re.IGNORECASE):
        url = "https://" + url
    return url


def _extract_links_from_text(text: str) -> list[Link]:
    """Find known-site and generic URLs directly present in resume text."""
    found: list[Link] = []
    seen_spans: list[tuple[int, int]] = []

    def _overlaps(start: int, end: int) -> bool:
        return any(start < e and end > s for s, e in seen_spans)

    for label, pattern in _KNOWN_SITE_TEXT_PATTERNS.items():
        for match in pattern.finditer(text):
            if _overlaps(match.start(), match.end()):
                continue
            seen_spans.append((match.start(), match.end()))
            found.append(Link(url=_normalize_link_url(match.group(0)), label=label))

    for match in _GENERIC_URL_RE.finditer(text):
        if _overlaps(match.start(), match.end()):
            continue
        seen_spans.append((match.start(), match.end()))
        url = match.group(0).rstrip(".,;:)")
        found.append(Link(url=url, label=_classify_url(url)))

    return found


_GENERIC_LINK_LABELS: frozenset[str] = frozenset({"website", "other", ""})


def _merge_links(*link_lists: list[Link]) -> list[Link]:
    """Merge Link lists; deduplicates by normalized URL and by platform label.

    URL dedup handles scheme/www/slash differences.  Label dedup handles the
    case where two sources store the same platform profile under slightly
    different paths (e.g. leetcode.com/u/x vs leetcode.com/x).  Generic
    labels like "Website" are excluded from label dedup because a candidate
    may legitimately have multiple personal sites.
    """
    merged: list[Link] = []
    seen_urls: set[str] = set()
    seen_labels: set[str] = set()
    for links in link_lists:
        for link in links:
            url_key   = _normalize_url_for_dedupe(link.url)
            label_key = (link.label or "").strip().lower()
            if url_key in seen_urls:
                continue
            if label_key not in _GENERIC_LINK_LABELS and label_key in seen_labels:
                continue
            seen_urls.add(url_key)
            if label_key not in _GENERIC_LINK_LABELS:
                seen_labels.add(label_key)
            merged.append(link)
    return merged


# ---------------------------------------------------------------------------
# Experience & education extraction
# ---------------------------------------------------------------------------


# _DATE_RE's optional group only includes the leading separator (–/—/-) in its
# first alternative (Month Year → Month Year).  The "present" and bare-year
# alternatives have no separator, so "Jan 2022 - Present" and "2020 - 2024"
# are matched as just "Jan 2022" / "2020" respectively.  This suffix pattern
# catches the remaining " - Present" / " - 2024" tail so _is_date_line works.
_DATE_RANGE_SUFFIX_RE = re.compile(
    r"^\s*[-–—]\s*(?:present|current|now|ongoing|(?:19|20)\d{2})\s*$",
    re.IGNORECASE,
)


def _is_date_line(line: str) -> bool:
    """True when a line is primarily a date range with little other content.

    Used to filter date lines out of title/company/description extraction.
    A line must have its ``_DATE_RE`` match cover ≥ 75 % of the stripped
    content to be a "date line" — this prevents description lines that merely
    *mention* a year ("Won 2023 hackathon") from being discarded.

    Special case: ``_DATE_RE`` only matches up to "Jan 2022" in a string like
    "Jan 2022 - Present" because the `present` alternative in its optional
    group lacks the leading separator.  We handle this by checking whether the
    remainder after the date match is a ``- present``-style suffix.
    """
    stripped = _LEADING_MARKER_RE.sub("", line.strip())
    if not stripped:
        return False
    m = _DATE_RE.search(stripped)
    if not m:
        return False
    remainder = stripped[m.end():]
    # "Jan 2022 - Present": date match = "Jan 2022", remainder = " - Present"
    # "2020 - 2024":        date match = "2020",     remainder = " - 2024"
    if remainder and _DATE_RANGE_SUFFIX_RE.match(remainder):
        return True
    return len(m.group(0)) / len(stripped) >= 0.75


def _extract_experience(section_text: str) -> list[Experience]:
    """Extract experience entries by splitting on blank lines.

    **Root-cause fix**: the previous implementation split the section text at
    every newline that preceded a year-containing line.  For a resume formatted
    as::

        AI/ML Intern
        Infosys Springboard
        Jun 2024 – Aug 2024    ← contains year → WRONG split point
        • Built ML models

    the old regex produced a first block with just the title and company (no
    date) and a second block whose *first line* was the date string (becoming a
    garbage "title").  This created two entries per job and the merge engine
    could not reconcile them because titles/companies were completely different.

    The fix: split on blank lines (paragraph-based).  Each paragraph = one job
    entry.  Within the paragraph the date range is identified and those lines
    are excluded from title/company/description so the content is not polluted.
    """
    if not section_text:
        return []

    entries: list[Experience] = []
    # Paragraph-based splitting: each blank-line-separated block is one job.
    blocks = re.split(r"\n{2,}", section_text)

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        all_lines = [l.strip() for l in block.splitlines() if l.strip()]
        if not all_lines:
            continue

        # Find the date range by searching the full block text.
        date_match = _DATE_RE.search(block)
        raw_date = date_match.group(0) if date_match else None

        start = end = None
        is_current = False
        if raw_date:
            parts = re.split(r"\s*[-–—]\s*", raw_date, maxsplit=1)
            start = parts[0].strip() if parts else None
            if len(parts) > 1:
                end_raw = parts[1].strip()
                is_current = bool(re.match(r"present|current|now|ongoing", end_raw, re.IGNORECASE))
                end = None if is_current else end_raw
            else:
                is_current = True

        # Remove lines that are primarily dates so they don't pollute
        # title / company / description.  Fallback: keep all lines if every
        # line would be filtered (prevents total data loss).
        content_lines = [l for l in all_lines if not _is_date_line(l)]
        if not content_lines:
            content_lines = all_lines

        title       = content_lines[0] if content_lines else None
        company     = content_lines[1] if len(content_lines) > 1 else None
        description = " ".join(content_lines[2:]) if len(content_lines) > 2 else None

        # Strip label prefixes ("Role: ...", "Duration: ...") that some sources
        # inject into structured fields.
        if title:
            title = _EXP_LABEL_PREFIX_RE.sub("", title).strip() or title
        if company:
            company = _EXP_LABEL_PREFIX_RE.sub("", company).strip() or company

        # Indian-format resumes list company first, role second.  Swap when the
        # first line has no job-title keywords and the second line does.
        if (
            title and company
            and not _JOB_TITLE_KEYWORDS_RE.search(title)
            and _JOB_TITLE_KEYWORDS_RE.search(company)
        ):
            title, company = company, title

        entries.append(
            Experience(
                title=title,
                company=company,
                description=description,
                duration=DateRange(start=start, end=end, is_current=is_current) if start else None,
            )
        )

    logger.debug(
        "_extract_experience: %d char(s) → %d block(s) → %d entry/entries",
        len(section_text), len(blocks), len(entries),
    )
    for i, e in enumerate(entries):
        logger.debug(
            "  exp[%d] title=%r company=%r duration=%s",
            i, e.title, e.company, e.duration,
        )

    return entries


def _extract_education(section_text: str) -> list[Education]:
    """Extract education entries from the EDUCATION section.

    **Root-cause fix**: the previous implementation had two bugs:

    1. ``institution or lines[0]`` — when a blank-line split left a block
       containing only the degree line (e.g. "B.E in Computer Science"), the
       institution fell back to ``lines[0]``, which is the *degree* string.
       The same degree then appeared as both degree and institution.

    2. No GPA extraction at all; only a single year was captured for duration
       (not a start–end range).

    **Fix strategy**: blank-line splitting is preserved (so truly separate
    degrees are still split correctly).  After the first parse pass, adjacent
    *fragment* blocks are merged: if block N has a degree but no institution
    and block N+1 has an institution but no degree (or vice-versa), they
    describe the same credential and are joined into one record.  This handles
    the common PDF case where the extractor inserts a blank line between the
    degree name and the institution name.
    """
    if not section_text:
        return []

    # ------------------------------------------------------------------ #
    # Pass 1 — parse each blank-line block into a raw dict                #
    # ------------------------------------------------------------------ #
    raw: list[dict] = []
    for block in re.split(r"\n{2,}", section_text):
        block = block.strip()
        if not block:
            continue

        block_lines = [l.strip() for l in block.splitlines() if l.strip()]
        if not block_lines:
            continue

        # Extract date range.  _DATE_RE's optional group omits the separator
        # before the bare-year alternative, so "2018 - 2022" is matched as
        # just "2018".  Use a dedicated year-range pattern first, then fall
        # back to a single year.
        year_range_m = re.search(
            r"\b((?:19|20)\d{2})\s*[-–—]\s*((?:19|20)\d{2}|present|current|now|ongoing)\b",
            block,
            re.IGNORECASE,
        )
        start_year = end_year = None
        if year_range_m:
            start_year = year_range_m.group(1)
            end_raw    = year_range_m.group(2)
            end_year   = (
                None
                if re.match(r"present|current|now|ongoing", end_raw, re.IGNORECASE)
                else end_raw
            )
        else:
            single_m = re.search(r"\b((?:19|20)\d{2})\b", block)
            if single_m:
                start_year = single_m.group(1)

        degree = institution = None
        gpa: Optional[float] = None

        for line in block_lines:
            if _DEGREE_HEADING_RE.search(line) and degree is None:
                degree = line
            elif _GPA_LINE_RE.search(line):
                nums = re.findall(r"\d+\.?\d*", line)
                if nums:
                    try:
                        gpa = float(nums[0])
                    except ValueError:
                        pass
            elif institution is None and not _is_date_line(line):
                # Accept first non-degree, non-GPA, non-date line as institution.
                institution = line

        raw.append({
            "institution": institution,
            "degree":      degree,
            "start_year":  start_year,
            "end_year":    end_year,
            "gpa":         gpa,
        })

    # ------------------------------------------------------------------ #
    # Pass 2 — merge adjacent complementary fragments                     #
    # A PDF blank line between the degree line and the institution line   #
    # splits one credential into two half-populated blocks.  When block N #
    # has a degree but no institution and block N+1 has an institution    #
    # but no degree (or vice-versa), merge them.                          #
    # ------------------------------------------------------------------ #
    merged: list[dict] = []
    i = 0
    while i < len(raw):
        curr = raw[i]
        if i + 1 < len(raw):
            nxt = raw[i + 1]
            c_deg  = curr["degree"]      is not None
            c_inst = curr["institution"] is not None
            n_deg  = nxt["degree"]       is not None
            n_inst = nxt["institution"]  is not None

            should_merge = (
                (c_deg and not c_inst and n_inst and not n_deg)
                or
                (c_inst and not c_deg and not n_inst and n_deg)
            )
            if should_merge:
                merged.append({
                    "institution": curr["institution"] or nxt["institution"],
                    "degree":      curr["degree"]      or nxt["degree"],
                    "start_year":  curr["start_year"]  or nxt["start_year"],
                    "end_year":    curr["end_year"]     or nxt["end_year"],
                    "gpa":         curr["gpa"] if curr["gpa"] is not None else nxt["gpa"],
                })
                i += 2
                continue
        merged.append(curr)
        i += 1

    # ------------------------------------------------------------------ #
    # Pass 3 — convert to Education objects, skip empty blocks            #
    # ------------------------------------------------------------------ #
    entries: list[Education] = []
    for m in merged:
        if m["institution"] is None and m["degree"] is None:
            continue  # year-only or entirely empty block
        start, end = m["start_year"], m["end_year"]
        entries.append(
            Education(
                institution=m["institution"],
                degree=m["degree"],
                duration=DateRange(start=start, end=end) if start else None,
                gpa=m["gpa"],
            )
        )

    logger.debug(
        "_extract_education: %d char(s) → %d raw block(s) → %d entry/entries",
        len(section_text), len(raw), len(entries),
    )
    for i, e in enumerate(entries):
        logger.debug(
            "  edu[%d] institution=%r degree=%r gpa=%s duration=%s",
            i, e.institution, e.degree, e.gpa, e.duration,
        )

    return entries


# ---------------------------------------------------------------------------
# Public parser class
# ---------------------------------------------------------------------------


class TextResumeParser:
    """Parse plain-text resume content into a structured :class:`ExtractedCandidate`.

    Shared by :class:`~app.extractors.pdf_extractor.PDFExtractor` and
    :class:`~app.extractors.txt_extractor.TXTExtractor` so all resume
    parsing logic lives in exactly one place.
    """

    def parse(
        self,
        text: str,
        source: DataSource = DataSource.UNKNOWN,
        source_file: Optional[str] = None,
        extra_links: Optional[list[Link]] = None,
    ) -> ExtractedCandidate:
        """Extract structured fields from raw resume text.

        Args:
            text:         Full plain-text content of the resume.
            source:       DataSource to stamp on the returned record.
            source_file:  Original file path (for traceability).
            extra_links:  Pre-extracted links (e.g. PDF hyperlink annotations)
                          merged in before plain-text URL scanning.  First
                          occurrence of each URL wins, so pass
                          higher-priority links here.

        Returns:
            :class:`~app.models.candidate.ExtractedCandidate` with all
            detectable fields populated.
        """
        if not text:
            logger.warning("TextResumeParser received empty text (source=%s, file=%s)", source, source_file)
            return ExtractedCandidate(source=source, source_file=source_file)

        sections = _detect_sections(text)
        header   = sections.get("_header", "")

        email     = _extract_email(text)
        phone     = _extract_phone(text)
        name      = _extract_name(header, email)
        links     = _merge_links(extra_links or [], _extract_links_from_text(text))
        skills    = _extract_skills(sections.get("skills", ""), text)
        experience = _extract_experience(sections.get("experience", ""))
        education  = _extract_education(sections.get("education", ""))
        summary    = sections.get("summary", "") or None

        candidate = ExtractedCandidate(
            source=source,
            source_file=source_file,
            name=name,
            email=email,
            phone=phone,
            summary=summary,
            skills=skills,
            experience=experience,
            education=education,
            links=links,
            raw_text=text,
        )

        # Pipeline stage trace — visible at INFO so callers can confirm counts
        # at the ExtractedCandidate boundary without enabling DEBUG.
        logger.info(
            "STAGE ExtractedCandidate | source=%s file=%s | "
            "education=%d experience=%d skills=%d",
            source,
            source_file or "<text>",
            len(candidate.education),
            len(candidate.experience),
            len(candidate.skills),
        )
        for i, exp in enumerate(candidate.experience):
            logger.info(
                "  exp[%d] title=%r company=%r duration=%s",
                i, exp.title, exp.company, exp.duration,
            )
        for i, edu in enumerate(candidate.education):
            logger.info(
                "  edu[%d] institution=%r degree=%r gpa=%s duration=%s",
                i, edu.institution, edu.degree, edu.gpa, edu.duration,
            )

        logger.debug(
            "TextResumeParser: source=%s name=%r email=%r skills=%d experience=%d",
            source,
            candidate.name,
            candidate.email,
            len(candidate.skills),
            len(candidate.experience),
        )
        return candidate
