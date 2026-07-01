"""Merge Engine.

Takes a CandidateGroup (one or more NormalizedCandidate records from different
sources) and produces a single MergedCandidate by applying config-driven rules
from merge_rules.yaml.

Three merge strategies per field
─────────────────────────────────
priority      Pick value from the highest-priority source that has a non-null,
              non-blank value.  Falls back down the priority list; if none has
              it, falls back to any non-null value in the group.

most_complete Pick the longest non-empty string.  Used for fields where more
              detail is better regardless of source (e.g. summary, phone).

union         Combine items from all sources in priority order, then deduplicate.
              Used for list fields (skills, experience, education, links).
              - skills: case-insensitive exact dedup.
              - links:  URL-based dedup (case-insensitive).
              - experience/education: structural equality (Pydantic __eq__).
"""

from __future__ import annotations

import re
from typing import Any

from rapidfuzz import fuzz as _fuzz

from app.config.loader import get_config
from app.config.models import MergeRulesConfig
from app.extractors.text_resume_parser import _normalize_url_for_dedupe
from app.mergers.candidate_group import CandidateGroup
from app.models.candidate import Education, Experience, MergedCandidate, NormalizedCandidate
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Fuzzy-deduplication helpers for structured list fields
# ---------------------------------------------------------------------------

_EXP_FUZZY_THRESHOLD = 80.0    # minimum token-sort-ratio for company / title
_EDU_FUZZY_THRESHOLD = 80.0    # minimum ratio for institution
_EDU_DEGREE_THRESHOLD = 70.0   # lower bar for degrees (abbreviation variations)

# Strips inter-letter periods so "B.E." → "BE", "M.Tech." → "MTech"
_DEGREE_PERIOD_RE = re.compile(r"(?<=[A-Za-z])\.(?=[A-Za-z])")

# Maps leading abbreviation tokens to canonical degree names for comparison.
_DEGREE_CANONICAL: dict[str, str] = {
    "be":    "bachelor of engineering",
    "btech": "bachelor of technology",
    "bsc":   "bachelor of science",
    "bcom":  "bachelor of commerce",
    "ba":    "bachelor of arts",
    "mtech": "master of technology",
    "me":    "master of engineering",
    "ms":    "master of science",
    "msc":   "master of science",
    "mba":   "master of business administration",
    "phd":   "doctor of philosophy",
    "mca":   "master of computer applications",
    "bca":   "bachelor of computer applications",
}

# ---------------------------------------------------------------------------
# Pure helper functions (fuzzy dedup)
# ---------------------------------------------------------------------------


def _fuzzy_score(a: str | None, b: str | None) -> float:
    """Token-sort fuzzy ratio [0–100]; 0.0 when either argument is absent."""
    if not a or not b:
        return 0.0
    return _fuzz.token_sort_ratio(a, b)


def _normalize_degree(text: str | None) -> str:
    """Expand common degree abbreviations to a canonical form for comparison.

    "B.E in Computer Science" and "Bachelor of Engineering (B.E.)" both
    normalise to "bachelor of engineering".
    """
    if not text:
        return ""
    cleaned = _DEGREE_PERIOD_RE.sub("", text).lower().strip()
    # Strip embedded field-of-study so "B.E. in CSE", "BE, Computer Science",
    # and "Bachelor of Engineering (B.E.) in CSE" all reduce to the same root
    # degree type before fuzzy comparison.
    core = re.sub(r"[,\(].*$|\s+in\s+.*$", "", cleaned).strip()
    alpha_only = re.sub(r"[^a-z\s]", "", core)
    m = re.match(r"^([a-z]+)", alpha_only)
    if m:
        expanded = _DEGREE_CANONICAL.get(m.group(1))
        if expanded:
            return expanded
    return core if core else cleaned


def _pick_longer(a: str | None, b: str | None) -> str | None:
    """Return the longer of two strings; whichever is non-None when one is."""
    if not a:
        return b
    if not b:
        return a
    return a if len(a) >= len(b) else b


def _experience_similar(a: Experience, b: Experience) -> bool:
    """True when two Experience records look like the same job.

    Both company AND title must agree (when both are present in each record).
    At least one of the two signals must score above the threshold.
    """
    if not a.company and not a.title:
        return False
    if not b.company and not b.title:
        return False

    company_score = _fuzzy_score(a.company, b.company)
    title_score   = _fuzzy_score(a.title,   b.title)

    # Company mismatch: before rejecting, try cross-field comparison.
    # Handles the case where one source has company/title swapped.
    if a.company and b.company and company_score < _EXP_FUZZY_THRESHOLD:
        cross = max(
            _fuzzy_score(a.company, b.title),
            _fuzzy_score(a.title,   b.company),
        )
        return cross >= _EXP_FUZZY_THRESHOLD

    # Strong company match: use partial_ratio for titles so that abbreviated
    # variants ("AI/ML Intern" vs "AI/ML Intern (Virtual Internship 6.0)")
    # are not treated as different jobs.
    if a.company and b.company and company_score >= 90:
        if not (a.title and b.title):
            return True
        partial = _fuzz.partial_ratio(
            (a.title or "").lower(), (b.title or "").lower()
        )
        return partial >= 80

    if a.title and b.title and title_score < _EXP_FUZZY_THRESHOLD:
        return False

    # At least one signal must positively agree
    return company_score >= _EXP_FUZZY_THRESHOLD or title_score >= _EXP_FUZZY_THRESHOLD


def _education_similar(a: Education, b: Education) -> bool:
    """True when two Education records look like the same academic credential."""
    if not a.institution and not a.degree:
        return False
    if not b.institution and not b.degree:
        return False

    inst_score = _fuzzy_score(a.institution, b.institution)
    deg_score  = _fuzzy_score(
        _normalize_degree(a.degree),
        _normalize_degree(b.degree),
    )

    if a.institution and b.institution and inst_score < _EDU_FUZZY_THRESHOLD:
        return False
    if a.degree and b.degree and deg_score < _EDU_DEGREE_THRESHOLD:
        return False

    return inst_score >= _EDU_FUZZY_THRESHOLD or deg_score >= _EDU_DEGREE_THRESHOLD


def _merge_experience_pair(existing: Experience, incoming: Experience) -> Experience:
    """Merge two similar Experience records, keeping the best value per field.

    Provenance from both records is preserved: missing fields in *existing*
    are filled from *incoming*, and the longer description wins.
    """
    return existing.model_copy(update={
        "company":     _pick_longer(existing.company,     incoming.company),
        "title":       _pick_longer(existing.title,       incoming.title),
        "description": _pick_longer(existing.description, incoming.description),
        "location":    existing.location or incoming.location,
        "duration":    existing.duration or incoming.duration,
    })


def _merge_education_pair(existing: Education, incoming: Education) -> Education:
    """Merge two similar Education records, keeping the best value per field."""
    return existing.model_copy(update={
        "institution":   _pick_longer(existing.institution, incoming.institution),
        "degree":        _pick_longer(existing.degree,      incoming.degree),
        "field_of_study": existing.field_of_study or incoming.field_of_study,
        "gpa":           existing.gpa if existing.gpa is not None else incoming.gpa,
        "duration":      existing.duration or incoming.duration,
    })


def _find_similar_index(item: Any, seen: list, field: str) -> int | None:
    """Return the index of a fuzzy-matching entry in *seen*, or ``None``.

    Logs every comparison at DEBUG level so callers can trace which pairs
    are merged and which are kept separate, together with their similarity
    scores (Step 6 of the pipeline root-cause analysis).
    """
    for i, existing in enumerate(seen):
        if field == "experience":
            c_score = _fuzzy_score(existing.company, item.company)
            t_score = _fuzzy_score(existing.title,   item.title)
            matched = _experience_similar(existing, item)
            if matched:
                logger.debug(
                    "MERGE exp[%d] ← incoming | "
                    "existing=%r/%r | incoming=%r/%r | "
                    "company=%.0f title=%.0f → MERGED",
                    i,
                    existing.company, existing.title,
                    item.company,     item.title,
                    c_score, t_score,
                )
                return i
            logger.debug(
                "KEEP exp[%d] separate | "
                "existing=%r/%r | incoming=%r/%r | "
                "company=%.0f title=%.0f → NOT merged",
                i,
                existing.company, existing.title,
                item.company,     item.title,
                c_score, t_score,
            )

        elif field == "education":
            i_score = _fuzzy_score(existing.institution, item.institution)
            d_score = _fuzzy_score(
                _normalize_degree(existing.degree),
                _normalize_degree(item.degree),
            )
            matched = _education_similar(existing, item)
            if matched:
                logger.debug(
                    "MERGE edu[%d] ← incoming | "
                    "existing=%r/%r | incoming=%r/%r | "
                    "institution=%.0f degree=%.0f → MERGED",
                    i,
                    existing.institution, existing.degree,
                    item.institution,     item.degree,
                    i_score, d_score,
                )
                return i
            logger.debug(
                "KEEP edu[%d] separate | "
                "existing=%r/%r | incoming=%r/%r | "
                "institution=%.0f degree=%.0f → NOT merged",
                i,
                existing.institution, existing.degree,
                item.institution,     item.degree,
                i_score, d_score,
            )

    return None


# ---------------------------------------------------------------------------
# Engine constants
# ---------------------------------------------------------------------------

# Scalar fields that the engine handles explicitly.
_SCALAR_FIELDS = ("name", "email", "phone", "location", "summary")
# List fields that the engine handles explicitly.
_LIST_FIELDS = ("skills", "experience", "education", "links")


class MergeEngine:
    """Merge a CandidateGroup into a single MergedCandidate.

    Stateless between calls; safe to instantiate once and reuse.
    """

    def __init__(self, config: MergeRulesConfig | None = None) -> None:
        self._rules: MergeRulesConfig = (
            config if config is not None else get_config().merge_rules
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def merge(self, group: CandidateGroup) -> MergedCandidate:
        """Merge all candidates in *group* into one :class:`MergedCandidate`.

        Args:
            group: A cluster produced by :class:`EntityResolver`.  Must
                contain at least one candidate.

        Returns:
            A fully-merged candidate with ``source_records`` set.  Never
            raises — missing values produce ``None`` / empty lists.
        """
        candidates = group.candidates
        if not candidates:
            return MergedCandidate()

        scalar_values: dict[str, Any] = {
            field: self._merge_scalar(field, candidates)
            for field in _SCALAR_FIELDS
        }
        list_values: dict[str, list] = {
            field: self._merge_list(field, candidates)
            for field in _LIST_FIELDS
        }

        merged = MergedCandidate(
            **scalar_values,
            **list_values,
            extra_fields=self._merge_extra(candidates),
            source_records=candidates,
        )

        logger.debug(
            "MergeEngine: merged %d record(s) → name=%r email=%r skills=%d",
            len(candidates),
            merged.name,
            merged.email,
            len(merged.skills),
        )
        return merged

    # ------------------------------------------------------------------
    # Scalar merging
    # ------------------------------------------------------------------

    def _merge_scalar(
        self, field: str, candidates: list[NormalizedCandidate]
    ) -> Any:
        rule = self._rules.get_rule(field)
        if rule.strategy == "most_complete":
            return self._most_complete_scalar(field, candidates)
        return self._priority_scalar(field, rule.priority, candidates)

    def _priority_scalar(
        self,
        field: str,
        priority: list[str],
        candidates: list[NormalizedCandidate],
    ) -> Any:
        """Return the first non-blank value walking down *priority*."""
        by_source: dict[str, Any] = {
            c.source.value: getattr(c, field, None) for c in candidates
        }
        for source in priority:
            value = by_source.get(source)
            if _non_empty(value):
                return value

        # Fallback: any non-empty value from any candidate
        for c in candidates:
            value = getattr(c, field, None)
            if _non_empty(value):
                return value

        return None

    def _most_complete_scalar(
        self, field: str, candidates: list[NormalizedCandidate]
    ) -> Any:
        """Return the longest non-empty string value across all candidates."""
        best: str | None = None
        best_len = -1
        for c in candidates:
            value = getattr(c, field, None)
            if not isinstance(value, str) or not value.strip():
                continue
            if len(value) > best_len:
                best = value
                best_len = len(value)
        return best

    # ------------------------------------------------------------------
    # List merging
    # ------------------------------------------------------------------

    def _merge_list(
        self, field: str, candidates: list[NormalizedCandidate]
    ) -> list:
        rule = self._rules.get_rule(field)
        ordered = _sort_by_priority(candidates, rule.priority)

        if rule.strategy == "union":
            # Experience and education use fuzzy dedup + field-level merging
            # to preserve provenance from all contributing sources.
            if field in ("experience", "education"):
                return self._merge_structured_list(field, ordered)
            seen: list = []
            for c in ordered:
                for item in (getattr(c, field, None) or []):
                    if not _already_seen(item, seen, field):
                        seen.append(item)
            return seen

        # "priority" or "most_complete" for lists: return the first non-empty list
        for c in ordered:
            items = getattr(c, field, None) or []
            if items:
                return list(items)
        return []

    def _merge_structured_list(
        self, field: str, ordered: list[NormalizedCandidate]
    ) -> list:
        """Merge experience / education with fuzzy dedup and field-level merging.

        When two records are judged to describe the same job / degree, we merge
        them rather than discarding one.  The higher-priority record's values
        take precedence for any field both records carry; missing values are
        filled from the lower-priority record (provenance preservation).
        """
        result: list = []
        for c in ordered:
            for item in (getattr(c, field, None) or []):
                # Fast path: exact structural equality (Pydantic __eq__)
                if item in result:
                    continue
                idx = _find_similar_index(item, result, field)
                if idx is not None:
                    if field == "experience":
                        result[idx] = _merge_experience_pair(result[idx], item)
                    else:
                        result[idx] = _merge_education_pair(result[idx], item)
                else:
                    result.append(item)
        return result

    # ------------------------------------------------------------------
    # Extra fields
    # ------------------------------------------------------------------

    def _merge_extra(self, candidates: list[NormalizedCandidate]) -> dict[str, Any]:
        """Merge extra_fields dicts; lower-priority sources are overwritten."""
        result: dict[str, Any] = {}
        for c in reversed(candidates):          # highest priority last → wins
            result.update(c.extra_fields or {})
        return result


# ---------------------------------------------------------------------------
# Module-level helpers (pure functions, no coupling to engine state)
# ---------------------------------------------------------------------------


def _non_empty(value: Any) -> bool:
    """True for any value that is not None and not a blank string."""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _sort_by_priority(
    candidates: list[NormalizedCandidate], priority: list[str]
) -> list[NormalizedCandidate]:
    """Return candidates sorted highest-priority first."""
    index = {src: i for i, src in enumerate(priority)}
    return sorted(
        candidates,
        key=lambda c: index.get(c.source.value, len(priority)),
    )


def _already_seen(item: Any, seen: list, field: str) -> bool:
    """Deduplication check for union merging, field-type aware."""
    if field == "skills":
        item_key = item.lower() if isinstance(item, str) else str(item)
        return any(
            (s.lower() if isinstance(s, str) else str(s)) == item_key
            for s in seen
        )
    if field == "links":
        item_url   = _normalize_url_for_dedupe(item.url) if hasattr(item, "url") else ""
        item_label = (item.label or "").strip().lower() if hasattr(item, "label") else ""
        for s in seen:
            if item_url and _normalize_url_for_dedupe(getattr(s, "url", "")) == item_url:
                return True
            s_label = (s.label or "").strip().lower() if hasattr(s, "label") else ""
            if (
                item_label
                and item_label not in ("website", "other")
                and s_label == item_label
            ):
                return True
        return False
    # experience / education: structural equality via Pydantic __eq__
    return item in seen
