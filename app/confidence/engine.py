"""

Takes a MergedCandidate and produces a ConfidenceReport that quantifies how
much we trust the merged data.

Improvements
──────────────────
1.  Weighted field importance   — per-field weights from ConfidenceConfig
2.  Semantic / fuzzy agreement  — rapidfuzz (fallback: difflib) replaces ==
3.  Conflict penalty            — configurable deduction when sources diverge
4.  Dynamic agreement bonus     — similarity × MAX_AGREEMENT_BONUS (not flat)
5.  Field freshness             — source_timestamp → recency multiplier
6.  Extraction confidence       — OCR / parser confidence multiplier
7.  Cross-field validation      — email format, location drift, LinkedIn/name
8.  Adaptive overall weights    — scale field/completeness/agreement by N sources
9.  Explainability              — ConfidenceReport.explanations human-readable list
10. Configuration-driven        — every magic number lives in ConfidenceConfig

Score anatomy (overall)
───────────────────────
overall = w_field        × weighted_field_average
        + w_completeness × completeness
        + w_agreement    × source_agreement
        − validation_penalty

where (w_field, w_completeness, w_agreement) are adaptive (see Improvement 8).

Per-field effective weight
  source_reliability × recency_multiplier × extraction_confidence_multiplier

Agreement bonus / conflict penalty (scalar fields only)
  similarity >= full_threshold  → bonus = similarity × max_agreement_bonus
  similarity in [partial, full) → bonus = similarity × max_agreement_bonus (smaller)
  similarity < partial          → penalty = -conflict_penalty
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Optional

from app.config.loader import get_config
from app.config.models import (
    ConfidenceConfig,
    FieldSimilarityConfig,
    SourceReliabilityConfig,
)
from app.models.candidate import (
    ConfidenceReport,
    FieldConfidence,
    MergedCandidate,
    NormalizedCandidate,
    ValidationIssue,
    ValidationResult,
)
from app.utils.logging_config import get_logger

# Optional rapidfuzz — fall back to difflib if unavailable.
try:
    from rapidfuzz import fuzz as _rfuzz
    _HAS_RAPIDFUZZ = True
except ImportError:  # pragma: no cover
    _HAS_RAPIDFUZZ = False

logger = get_logger(__name__)

# ── Public constants (kept stable for external consumers) ────────────────────

_KEY_SCALAR_FIELDS = ("name", "email", "phone", "location", "summary")
_KEY_LIST_FIELDS   = ("skills", "experience", "education", "links")
_KEY_FIELDS        = _KEY_SCALAR_FIELDS + _KEY_LIST_FIELDS

# Legacy module-level constants — retained so external code that imports them
# does not break.  The engine now reads these values from ConfidenceConfig.
_W_FIELD             = 0.50
_W_COMPLETENESS      = 0.25
_W_AGREEMENT         = 0.25
_ERROR_PENALTY       = 0.05
_WARNING_PENALTY     = 0.01
_MAX_ERROR_PENALTY   = 0.20
_MAX_WARNING_PENALTY = 0.10
_AGREEMENT_BONUS     = 0.05

# ── Internal helpers ─────────────────────────────────────────────────────────

_EMAIL_RE = re.compile(r'^[^@\s]+@[^@\s]+\.[^@\s]+$')


def _non_empty(value: object) -> bool:
    """True for any value that is not None and not empty/blank."""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict)):
        return bool(value)
    return True


def _canonical(value: object) -> object:
    """Normalise a value for agreement comparison (lowercase strings).

    Kept for backward compatibility; the engine uses _normalize_for_similarity
    internally for fuzzy comparisons.
    """
    if isinstance(value, str):
        return value.lower().strip()
    return value


def _normalize_for_similarity(value: str) -> str:
    """Pre-process a string before fuzzy comparison.

    1. Lowercase and strip whitespace.
    2. Remove punctuation (keep word characters and spaces).
    3. Collapse runs of whitespace.

    Examples::

        "Software Eng."  -> "software eng"
        "Hyderabad, IN"  -> "hyderabad in"
        "ALICE@EX.COM"   -> "aliceexcom"
    """
    s = value.lower().strip()
    s = re.sub(r'[^\w\s]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def _fuzzy_similarity(a: str, b: str) -> float:
    """Similarity in [0, 1] between two pre-processed strings.

    Uses rapidfuzz when available, falls back to difflib.SequenceMatcher.
    """
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    if _HAS_RAPIDFUZZ:
        return _rfuzz.ratio(a, b) / 100.0
    from difflib import SequenceMatcher  # pragma: no cover
    return SequenceMatcher(None, a, b).ratio()  # pragma: no cover


def _min_pairwise_similarity(values: list[str]) -> float:
    """Minimum fuzzy similarity over all pairs in *values*.

    Complexity: O(N²) on source count — fine for N ≤ ~20.
    Short-circuits at 0.0.
    """
    if len(values) < 2:
        return 1.0
    min_sim = 1.0
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            sim = _fuzzy_similarity(values[i], values[j])
            if sim < min_sim:
                min_sim = sim
                if min_sim == 0.0:
                    return 0.0
    return min_sim


# ── New module-level helpers (Improvements 1–4, 7, 11) ──────────────────────

# Titles/roles that imply self-employment so missing company is not flagged.
_SELF_EMPLOYED_RE = re.compile(
    r'\b(freelance[rd]?|self[‑\-\s]employed|consultant|founder|co[‑\-\s]founder'
    r'|independent\s+contractor|contract(?:or)?)\b',
    re.IGNORECASE,
)


def _normalize_email(value: str) -> str:
    """Normalise an email for exact comparison: lowercase + strip."""
    return value.lower().strip()


def _normalize_phone(value: str) -> str:
    """Strip all formatting from a phone number, leaving only digits.

    Heuristic: drop a leading '0' introduced by some national formats so that
    '0123456789' and '123456789' are treated as the same local number.
    """
    digits = re.sub(r'[^\d]', '', value)
    if len(digits) > 10 and digits.startswith('0'):
        digits = digits[1:]
    return digits


def _normalize_location(value: str, aliases: dict[str, str]) -> str:
    """Normalise a location string for agreement comparison.

    1. Apply :func:`_normalize_for_similarity` (lowercase, strip punctuation).
    2. Substitute each token via the alias dictionary (token-level).
    3. Check the full normalised string against the alias dictionary.

    Examples::

        _normalize_location("Bengaluru, India", {"bengaluru": "bangalore"})
        → "bangalore india"

        _normalize_location("NYC", {"nyc": "new york city"})
        → "new york city"
    """
    norm = _normalize_for_similarity(value)
    if not aliases:
        return norm
    # Token-level substitution
    tokens = [aliases.get(t, t) for t in norm.split()]
    result = ' '.join(tokens)
    # Full-string alias (e.g. multi-word abbreviation)
    return aliases.get(result, result)


def _similarity_for_field(
    field: str,
    a: str,
    b: str,
    location_aliases: dict[str, str],
) -> float:
    """Field-aware similarity in [0, 1].

    Improvement 2: each field uses the comparison strategy that makes
    semantic sense for its data type.

    - ``email``    → exact after lowercase (any formatting diff = mismatch)
    - ``phone``    → exact after digit normalisation; partial suffix match for
                     with/without country code
    - ``location`` → alias-aware fuzzy with substring-containment shortcut
    - others       → generic :func:`_fuzzy_similarity` after normalisation
    """
    if field == "email":
        return 1.0 if _normalize_email(a) == _normalize_email(b) else 0.0

    if field == "phone":
        pa, pb = _normalize_phone(a), _normalize_phone(b)
        if pa == pb:
            return 1.0
        # One number might include a country code the other omits
        short, long = (pa, pb) if len(pa) <= len(pb) else (pb, pa)
        if len(short) >= 7 and long.endswith(short):
            return 0.95
        return _fuzzy_similarity(pa, pb)

    if field == "location":
        la = _normalize_location(a, location_aliases)
        lb = _normalize_location(b, location_aliases)
        if la == lb:
            return 1.0
        # "Hyderabad" contained in "Hyderabad, India" → strong match
        if la in lb or lb in la:
            return 0.90
        return _fuzzy_similarity(la, lb)

    # name, summary, skills, experience, education, links → generic fuzzy
    return _fuzzy_similarity(
        _normalize_for_similarity(a),
        _normalize_for_similarity(b),
    )


def _probabilistic_combine(weights: list[float]) -> float:
    """Combine independent reliability estimates probabilistically.

    Improvement 1 — Formula:  P = 1 − Π(1 − Ri)

    A single weak source does not drag down a strong one; instead, each
    additional trusted source can only increase the combined reliability.

    Examples::

        _probabilistic_combine([0.90, 0.80]) → 1 − (0.10 × 0.20) = 0.98
        _probabilistic_combine([0.70])       → 0.70  (N=1 is unchanged)
    """
    if not weights:
        return 0.0
    complement = 1.0
    for w in weights:
        complement *= (1.0 - w)
    return 1.0 - complement


def _continuous_agreement_adjustment(
    similarity: float,
    full_threshold: float,
    partial_threshold: float,
    max_bonus: float,
    conflict_penalty: float,
) -> float:
    """Smooth agreement adjustment in [−conflict_penalty, max_bonus].

    Improvement 4 — replaces binary cliff (≥0.95 → +bonus, else penalty)
    with a continuous function:

    - similarity ≥ full_threshold → bonus scales up to *max_bonus*
    - similarity ∈ [partial, full) → bonus interpolated from 0 → max_bonus×full
    - similarity < partial         → penalty grows linearly to *conflict_penalty*

    The function is C⁰ continuous at both threshold boundaries.
    """
    if similarity >= full_threshold:
        return max_bonus * similarity

    if similarity >= partial_threshold:
        span = max(full_threshold - partial_threshold, 1e-9)
        t = (similarity - partial_threshold) / span
        # At boundary with full_threshold: max_bonus × full_threshold (continuous)
        return max_bonus * full_threshold * t

    # Conflict zone — penalty from 0 (at partial_threshold) to conflict_penalty (at 0)
    t = 1.0 - (similarity / max(partial_threshold, 1e-9))
    return -(conflict_penalty * t)


# ── Engine ───────────────────────────────────────────────────────────────────


class ConfidenceEngine:
    """Compute a ConfidenceReport for a merged candidate.

    Stateless between calls; safe to instantiate once and reuse.

    Args:
        reliability_config: Source reliability weights.
                            Defaults to the active pipeline config.
        confidence_config:  All confidence-engine tuning parameters.
                            Defaults to the active pipeline config.
    """

    def __init__(
        self,
        reliability_config: Optional[SourceReliabilityConfig] = None,
        confidence_config: Optional[ConfidenceConfig] = None,
    ) -> None:
        cfg = get_config()
        self._reliability: SourceReliabilityConfig = (
            reliability_config if reliability_config is not None
            else cfg.source_reliability
        )
        self._cfg: ConfidenceConfig = (
            confidence_config if confidence_config is not None
            else cfg.confidence
        )

    # ── Public API ───────────────────────────────────────────────────────────

    def score(
        self,
        merged: MergedCandidate,
        validation: Optional[ValidationResult] = None,
    ) -> ConfidenceReport:
        """Produce a :class:`ConfidenceReport` for *merged*.

        Args:
            merged:     The merged candidate record.
            validation: Optional caller-supplied validation result.
                        Cross-field issues (Improvement 7) are appended to it
                        before the penalty is applied.

        Returns:
            A fully populated :class:`ConfidenceReport`.  Never raises.
        """
        # Improvement 7: cross-field consistency checks
        cross_issues = self._cross_field_validate(merged)
        combined_val = self._merge_validation(validation, cross_issues)

        # Improvements 1–6 — field-level scoring
        field_scores = [self._score_field(field, merged) for field in _KEY_FIELDS]

        # Improvement 6 (completeness) — unchanged weighted sum
        completeness = self._score_completeness(merged)

        # Improvements 2 & 3 — field-specific source agreement
        source_agreement = self._score_source_agreement(merged)

        # Improvement 8 — adaptive overall weights
        overall_weights = self._adaptive_weights(len(merged.source_records))

        # Improvement 8 (extended) — source diversity bonus
        diversity_bonus = self._score_diversity_bonus(merged)

        # Improvement 10 — validation penalty as inspectable component
        validation_penalty = self._calc_validation_penalty(combined_val)

        overall = self._overall_score(
            field_scores, completeness, source_agreement,
            combined_val, overall_weights, diversity_bonus,
        )

        # Improvement 9 — richer explanations
        explanations = self._build_explanations(
            merged, field_scores, completeness, source_agreement,
            combined_val, diversity_bonus,
        )

        report = ConfidenceReport(
            overall_score=overall,
            field_scores=field_scores,
            completeness=completeness,
            source_agreement=source_agreement,
            explanations=explanations,
            diversity_bonus=round(diversity_bonus, 3),
            validation_penalty=round(validation_penalty, 3),
        )

        logger.debug(
            "ConfidenceEngine: overall=%.3f completeness=%.3f agreement=%.3f sources=%d",
            overall,
            completeness,
            source_agreement,
            len(merged.source_records),
        )
        return report

    # ── Per-field scoring ────────────────────────────────────────────────────

    def _score_field(self, field: str, merged: MergedCandidate) -> FieldConfidence:
        """Score one key field.

        Pipeline
        ────────
        1. Return 0.0 if the merged value is absent.
        2. Compute per-contributor effective weight:
               source_reliability × freshness(field) × extraction_confidence
        3. Combine effective weights probabilistically (Improvement 1):
               P = 1 − Π(1 − wi)
        4. For scalar fields with ≥ 2 contributors and agreement not skipped:
               - Use field-specific comparison (Improvement 2).
               - Apply continuous agreement curve (Improvement 4).
        """
        value = getattr(merged, field, None)
        if not _non_empty(value):
            return FieldConfidence(
                field_name=field,
                score=0.0,
                contributing_sources=[],
                reason="field absent",
            )

        contributors = [
            c for c in merged.source_records
            if _non_empty(getattr(c, field, None))
        ]
        if not contributors:
            return FieldConfidence(
                field_name=field,
                score=0.5,
                contributing_sources=[],
                reason="no source metadata",
            )

        # Improvements 1, 5, 6: probabilistic combination of effective weights
        effective_weights = [
            self._reliability.get(c.source)
            * self._freshness_weight(c, field)
            * self._extraction_conf(c)
            for c in contributors
        ]
        combined = _probabilistic_combine(effective_weights)

        reason_parts = [f"{len(contributors)} source(s) contributing"]
        similarity: Optional[float] = None

        # Improvements 2–4: field-specific similarity + continuous adjustment
        fs_cfg: Optional[FieldSimilarityConfig] = self._cfg.field_similarity.get(field)
        skip = fs_cfg.skip_agreement if fs_cfg else False

        if len(contributors) > 1 and field in _KEY_SCALAR_FIELDS and not skip:
            raw_values = [str(getattr(c, field)) for c in contributors]
            pairs = [
                _similarity_for_field(field, raw_values[i], raw_values[j],
                                      self._cfg.location_aliases)
                for i in range(len(raw_values))
                for j in range(i + 1, len(raw_values))
            ]
            similarity = min(pairs) if pairs else 1.0

            ft = fs_cfg.full_threshold    if fs_cfg else self._cfg.similarity_full_threshold
            pt = fs_cfg.partial_threshold if fs_cfg else self._cfg.similarity_partial_threshold

            adj = _continuous_agreement_adjustment(
                similarity, ft, pt,
                self._cfg.max_agreement_bonus,
                self._cfg.conflict_penalty,
            )
            combined = max(0.0, min(1.0, combined + adj))

            if adj > 0:
                if similarity >= ft:
                    reason_parts.append("sources agree")
                else:
                    reason_parts.append(f"sources partially agree ({similarity:.0%})")
            elif adj < 0:
                reason_parts.append(f"sources conflict (similarity={similarity:.0%})")

        return FieldConfidence(
            field_name=field,
            score=round(min(1.0, combined), 3),
            contributing_sources=[c.source for c in contributors],
            reason="; ".join(reason_parts),
            similarity=round(similarity, 3) if similarity is not None else None,
        )

    # ── Weighted completeness (Improvement 1) ────────────────────────────────

    def _score_completeness(self, merged: MergedCandidate) -> float:
        """Sum of field_weights for non-empty key fields.

        Because field_weights sum to 1.0 the result is in [0, 1].
        A missing high-importance field (experience: 0.20) penalises more
        than a missing low-importance one (links: 0.05).
        """
        total = sum(
            self._cfg.field_weights.get(field, 0.0)
            for field in _KEY_FIELDS
            if _non_empty(getattr(merged, field, None))
        )
        return round(total, 3)

    # ── Fuzzy source agreement (Improvement 2) ───────────────────────────────

    def _score_source_agreement(self, merged: MergedCandidate) -> float:
        """Average min-pairwise field-specific similarity across comparable scalar fields.

        Improvements 2 & 3:
        - Uses :func:`_similarity_for_field` (email exact, phone digit, location alias-aware).
        - Skips fields configured with ``skip_agreement=True`` (e.g. *summary*).
        - Single-source records always return 1.0.
        """
        if len(merged.source_records) < 2:
            return 1.0

        agreements: list[float] = []
        for field in _KEY_SCALAR_FIELDS:
            fs_cfg = self._cfg.field_similarity.get(field)
            if fs_cfg and fs_cfg.skip_agreement:
                continue

            values = [
                str(getattr(c, field))
                for c in merged.source_records
                if _non_empty(getattr(c, field, None))
            ]
            if len(values) < 2:
                continue

            pairs = [
                _similarity_for_field(field, values[i], values[j],
                                      self._cfg.location_aliases)
                for i in range(len(values))
                for j in range(i + 1, len(values))
            ]
            agreements.append(min(pairs))

        return round(sum(agreements) / len(agreements), 3) if agreements else 1.0

    # ── Overall score ─────────────────────────────────────────────────────────

    def _calc_validation_penalty(self, validation: Optional[ValidationResult]) -> float:
        """Compute total validation deduction (capped per type)."""
        if not validation:
            return 0.0
        error_pen   = min(self._cfg.max_error_penalty,
                          len(validation.errors)   * self._cfg.error_penalty)
        warning_pen = min(self._cfg.max_warning_penalty,
                          len(validation.warnings) * self._cfg.warning_penalty)
        return error_pen + warning_pen

    def _score_diversity_bonus(self, merged: MergedCandidate) -> float:
        """Bonus for multiple independent trusted sources (Improvement 8 extension).

        Sources with reliability below ``min_reliability`` are excluded — they
        do not count as independent confirmation.
        """
        dc = self._cfg.diversity
        if not dc.enabled:
            return 0.0
        n_diverse = sum(
            1 for c in merged.source_records
            if self._reliability.get(c.source) >= dc.min_reliability
        )
        if n_diverse < dc.min_sources:
            return 0.0
        ratio = min(1.0, n_diverse / dc.min_sources)
        return dc.max_bonus * ratio

    def _overall_score(
        self,
        field_scores: list[FieldConfidence],
        completeness: float,
        source_agreement: float,
        validation: Optional[ValidationResult],
        weights: tuple[float, float, float],
        diversity_bonus: float,
    ) -> float:
        """Combine component scores using adaptive weights, diversity bonus, and penalties.

        Improvement 10: each component is computed separately so the caller
        can expose the breakdown in the ConfidenceReport.
        """
        w_field, w_completeness, w_agreement = weights

        weighted_field_avg = sum(
            fs.score * self._cfg.field_weights.get(fs.field_name, 0.0)
            for fs in field_scores
        )

        score = (
            weighted_field_avg * w_field
            + completeness     * w_completeness
            + source_agreement * w_agreement
            + diversity_bonus
        )
        score -= self._calc_validation_penalty(validation)
        return round(max(0.0, min(1.0, score)), 3)

    # ── Adaptive overall weights (Improvement 8) ─────────────────────────────

    def _adaptive_weights(self, source_count: int) -> tuple[float, float, float]:
        """Return (w_field, w_completeness, w_agreement) scaled to source count.

        With more sources the agreement component carries more signal, so its
        weight increases at the expense of the field-average weight.
        """
        aw = self._cfg.adaptive_weights
        if source_count <= 1:
            return (aw.one_source_field,  aw.one_source_completeness,  aw.one_source_agreement)
        if source_count < aw.many_threshold:
            return (aw.two_source_field,  aw.two_source_completeness,  aw.two_source_agreement)
        return     (aw.many_source_field, aw.many_source_completeness, aw.many_source_agreement)

    # ── Field freshness (Improvement 5) ──────────────────────────────────────

    def _freshness_weight(self, candidate: NormalizedCandidate, field: str = "") -> float:
        """Exponential freshness multiplier in [min_freshness, 1.0] — Improvement 5.

        weight = max(min_freshness, 0.5 ^ (days_old / half_life_days))

        Only applies to fields listed in ``FreshnessDecayConfig.staleness_fields``.
        Stable fields (name, education, skills) always return 1.0.

        Args:
            candidate: The normalised source record.
            field:     The field being scored.  Empty string disables the
                       staleness-field guard (backward-compat shim).
        """
        fc = self._cfg.freshness
        # Stable fields — freshness is irrelevant
        if field and field not in fc.staleness_fields:
            return 1.0

        ts: Optional[datetime] = getattr(candidate, 'source_timestamp', None)
        if ts is None:
            return 1.0

        now = datetime.now(timezone.utc).replace(tzinfo=None)
        ts_naive = ts.replace(tzinfo=None) if ts.tzinfo is not None else ts
        if ts_naive > now:
            return 1.0  # future timestamp — treat as fresh

        days_old = (now - ts_naive).days
        weight = 0.5 ** (days_old / fc.half_life_days)
        return max(fc.min_freshness, weight)

    # ── Extraction confidence (Improvement 6) ────────────────────────────────

    def _extraction_conf(self, candidate: NormalizedCandidate) -> float:
        """Return OCR / parser confidence for this record, defaulting to 1.0."""
        conf: Optional[float] = getattr(candidate, 'extraction_confidence', None)
        return conf if conf is not None else 1.0

    # ── Cross-field validation (Improvement 7) ───────────────────────────────

    def _cross_field_validate(self, merged: MergedCandidate) -> list[ValidationIssue]:
        """Structural and consistency checks that span multiple fields.

        Improvements 7 & 12:
        - Self-employed / freelance / founder roles exempt from "missing company" warning.
        - Location inconsistency is NOT raised here to avoid double-penalising
          what the field-level conflict penalty already captures; it appears
          instead in :meth:`_build_explanations`.
        - Dedup guard: identical (field, message) pairs are emitted at most once.

        Checks:
        1. Email format (error)
        2. Experience without company — exempt if title implies self-employment
        3. LinkedIn URL vs candidate name mismatch (warning)
        """
        issues: list[ValidationIssue] = []
        seen: set[tuple[str, str]] = set()

        def _add(issue: ValidationIssue) -> None:
            key = (issue.field, issue.message)
            if key not in seen:
                seen.add(key)
                issues.append(issue)

        # 1. Email format
        if merged.email and not _EMAIL_RE.match(merged.email):
            _add(ValidationIssue(
                field="email", severity="error",
                message=f"Invalid email format: {merged.email!r}",
                value=merged.email,
            ))

        # 2. Experience without a company name
        for idx, exp in enumerate(merged.experience):
            if exp.title and not exp.company:
                # Improvement 7: founders/freelancers/consultants do not need a company.
                if _SELF_EMPLOYED_RE.search(exp.title or ""):
                    continue
                _add(ValidationIssue(
                    field="experience", severity="warning",
                    message=(
                        f"Experience entry {idx + 1} ({exp.title!r}) "
                        "is missing a company name"
                    ),
                ))

        # 3. LinkedIn URL vs candidate name
        #    Avoids penalising when a LinkedIn slug uses a nickname or initials.
        linkedin = next(
            (lnk for lnk in merged.links
             if lnk.label and "linkedin" in lnk.label.lower()),
            None,
        )
        if linkedin and merged.name:
            meaningful = [p for p in merged.name.lower().split() if len(p) > 2]
            if meaningful and not any(p in linkedin.url.lower() for p in meaningful):
                _add(ValidationIssue(
                    field="links", severity="warning",
                    message="LinkedIn URL does not appear to match candidate name",
                    value=linkedin.url,
                ))

        return issues

    # ── Merge validation ─────────────────────────────────────────────────────

    def _merge_validation(
        self,
        user_val: Optional[ValidationResult],
        cross_issues: list[ValidationIssue],
    ) -> Optional[ValidationResult]:
        """Combine caller-supplied validation with cross-field issues.

        Returns None (not a ValidationResult with 0 issues) when both inputs
        are empty, so that callers that check `if validation:` see no change
        and no penalty is applied.
        """
        if not cross_issues:
            return user_val
        all_issues = (user_val.issues if user_val else []) + cross_issues
        return ValidationResult(
            is_valid=not any(i.severity == "error" for i in all_issues),
            issues=all_issues,
        )

    # ── Explainability (Improvement 9) ───────────────────────────────────────

    def _build_explanations(
        self,
        merged: MergedCandidate,
        field_scores: list[FieldConfidence],
        completeness: float,
        source_agreement: float,
        validation: Optional[ValidationResult],
        diversity_bonus: float,
    ) -> list[str]:
        """Return a human-readable list of reasons behind the overall score.

        Improvement 9: every bonus and deduction is referenced by field, reason,
        and adjustment so a downstream consumer can display a full audit trail.
        """
        out: list[str] = []
        n = len(merged.source_records)

        # ── Source summary ──────────────────────────────────────────────────
        if n == 0:
            out.append("No source records — score estimated from merged data only")
        elif n == 1:
            src = merged.source_records[0].source
            rel = self._reliability.get(src)
            out.append(f"Single source: {src.value} (reliability {rel:.0%})")
        else:
            out.append(f"{n} source records merged")

        # ── Diversity bonus ─────────────────────────────────────────────────
        if diversity_bonus > 0:
            dc = self._cfg.diversity
            out.append(
                f"✔ Source diversity bonus +{diversity_bonus:.0%} "
                f"({n} sources ≥ {dc.min_reliability:.0%} reliability)"
            )

        # ── Field-level highlights ──────────────────────────────────────────
        for fs in field_scores:
            if fs.score == 0.0:
                out.append(f"✖ {fs.field_name}: absent")
            elif "conflict" in (fs.reason or ""):
                sim_str = (
                    f" (similarity={fs.similarity:.0%})"
                    if fs.similarity is not None else ""
                )
                # Use "⚠ Warning:" so downstream can filter by severity keyword
                out.append(f"⚠ Warning: {fs.field_name} sources conflict{sim_str}")
            elif "partially agree" in (fs.reason or ""):
                out.append(
                    f"~ {fs.field_name}: partial agreement "
                    f"({fs.similarity:.0%} similarity, score {fs.score:.0%})"
                )
            elif fs.score >= 0.85:
                out.append(f"✔ {fs.field_name}: high confidence ({fs.score:.0%})")

        # ── Agreement summary ───────────────────────────────────────────────
        if n >= 2:
            if source_agreement >= 0.90:
                out.append(f"✔ Sources agree strongly ({source_agreement:.0%})")
            elif source_agreement < 0.70:
                out.append(
                    f"✖ Sources disagree on key fields ({source_agreement:.0%} agreement)"
                )

        # ── Completeness summary ────────────────────────────────────────────
        if completeness >= 0.80:
            out.append(f"✔ Profile {completeness:.0%} complete")
        elif completeness < 0.40:
            out.append(
                f"✖ Profile only {completeness:.0%} complete — key fields missing"
            )

        # ── Validation findings (capped at 3 each to keep output readable) ─
        if validation:
            for issue in validation.errors[:3]:
                out.append(f"✖ Error: {issue.message}")
            for issue in validation.warnings[:3]:
                out.append(f"⚠ Warning: {issue.message}")
            for issue in validation.info[:3]:
                out.append(f"ℹ {issue.message}")

        return out
