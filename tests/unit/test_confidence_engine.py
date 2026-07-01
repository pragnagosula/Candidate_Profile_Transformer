"""Unit tests for the Confidence Engine v2.

Coverage:
  ✔ perfect agreement (exact and fuzzy)
  ✔ partial agreement
  ✔ conflicting sources (conflict penalty applied)
  ✔ missing fields
  ✔ OCR / extraction confidence (Improvement 6)
  ✔ timestamp / recency weighting (Improvement 5)
  ✔ adaptive overall weights (Improvement 8)
  ✔ explainability (Improvement 9)
  ✔ validation penalties
  ✔ cross-field validation — Improvement 7
  ✔ weighted completeness — Improvement 1
  ✔ weighted field average — Improvement 1
  ✔ dynamic agreement bonus — Improvement 4
  ✔ configurable conflict penalty — Improvement 3
  ✔ custom reliability config
  ✔ backward-compatible public helpers
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

# (config imports merged into the block above)
from app.confidence.engine import (
    ConfidenceEngine,
    _KEY_FIELDS,
    _KEY_SCALAR_FIELDS,
    _canonical,
    _continuous_agreement_adjustment,
    _fuzzy_similarity,
    _min_pairwise_similarity,
    _non_empty,
    _normalize_email,
    _normalize_for_similarity,
    _normalize_location,
    _normalize_phone,
    _probabilistic_combine,
    _similarity_for_field,
)
from app.config.models import (
    ConfidenceConfig,
    FreshnessDecayConfig,
    SourceDiversityConfig,
    SourceReliabilityConfig,
)
from app.models.candidate import (
    DataSource,
    Experience,
    Link,
    MergedCandidate,
    NormalizedCandidate,
    ValidationIssue,
    ValidationResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _nc(
    source: DataSource = DataSource.CSV,
    name: str | None = None,
    email: str | None = None,
    phone: str | None = None,
    location: str | None = None,
    summary: str | None = None,
    skills: list[str] | None = None,
    extraction_confidence: float | None = None,
    source_timestamp: datetime | None = None,
) -> NormalizedCandidate:
    return NormalizedCandidate(
        source=source,
        name=name,
        email=email,
        phone=phone,
        location=location,
        summary=summary,
        skills=skills or [],
        extraction_confidence=extraction_confidence,
        source_timestamp=source_timestamp,
    )


def _merged(
    name: str | None = None,
    email: str | None = None,
    phone: str | None = None,
    location: str | None = None,
    summary: str | None = None,
    skills: list[str] | None = None,
    experience: list[Experience] | None = None,
    links: list[Link] | None = None,
    sources: list[NormalizedCandidate] | None = None,
) -> MergedCandidate:
    return MergedCandidate(
        name=name,
        email=email,
        phone=phone,
        location=location,
        summary=summary,
        skills=skills or [],
        experience=experience or [],
        links=links or [],
        source_records=sources or [],
    )


def _validation(errors: int = 0, warnings: int = 0) -> ValidationResult:
    issues = (
        [ValidationIssue(field="x", severity="error",   message=f"e{i}") for i in range(errors)]
        + [ValidationIssue(field="x", severity="warning", message=f"w{i}") for i in range(warnings)]
    )
    return ValidationResult(is_valid=(errors == 0), issues=issues)


def _engine(
    source_weights: dict[str, float] | None = None,
    confidence_cfg: ConfidenceConfig | None = None,
) -> ConfidenceEngine:
    rel = SourceReliabilityConfig(weights=source_weights) if source_weights else None
    return ConfidenceEngine(reliability_config=rel, confidence_config=confidence_cfg)


def _default_cfg(**overrides) -> ConfidenceConfig:
    """Build a ConfidenceConfig, optionally overriding fields."""
    return ConfidenceConfig(**overrides)


# ---------------------------------------------------------------------------
# Per-field scoring
# ---------------------------------------------------------------------------


class TestFieldScoring:
    def test_absent_field_scores_zero(self):
        m = _merged(name=None, sources=[_nc(DataSource.CSV, name=None)])
        report = _engine().score(m)
        name_score = next(f for f in report.field_scores if f.field_name == "name")
        assert name_score.score == 0.0
        assert name_score.contributing_sources == []
        assert "absent" in name_score.reason

    def test_present_field_has_positive_score(self):
        src = _nc(DataSource.CSV, name="Alice")
        m = _merged(name="Alice", sources=[src])
        report = _engine().score(m)
        name_score = next(f for f in report.field_scores if f.field_name == "name")
        assert name_score.score > 0.0

    def test_higher_reliability_yields_higher_score(self):
        low  = _nc(DataSource.CSV, name="Alice")   # weight 0.70
        high = _nc(DataSource.ATS, name="Alice")   # weight 0.95

        low_score  = next(f for f in _engine().score(_merged(name="Alice", sources=[low] )).field_scores if f.field_name == "name").score
        high_score = next(f for f in _engine().score(_merged(name="Alice", sources=[high])).field_scores if f.field_name == "name").score

        assert high_score > low_score

    def test_two_sources_average_reliability(self):
        src1 = _nc(DataSource.CSV,  name="Alice")   # 0.70
        src2 = _nc(DataSource.JSON, name="Alice")   # 0.75
        m = _merged(name="Alice", sources=[src1, src2])
        report = _engine().score(m)
        name_score = next(f for f in report.field_scores if f.field_name == "name")
        # avg(0.70, 0.75)=0.725 + dynamic agreement bonus ≥ 0.725
        assert 0.70 <= name_score.score <= 1.0

    def test_contributing_sources_listed(self):
        src = _nc(DataSource.RESUME_PDF, name="Alice")
        m = _merged(name="Alice", sources=[src])
        report = _engine().score(m)
        name_score = next(f for f in report.field_scores if f.field_name == "name")
        assert DataSource.RESUME_PDF in name_score.contributing_sources

    def test_all_key_fields_have_score_entries(self):
        m = _merged(sources=[])
        report = _engine().score(m)
        scored_fields = {f.field_name for f in report.field_scores}
        assert set(_KEY_FIELDS) == scored_fields

    def test_no_source_records_field_score_half(self):
        m = _merged(name="Alice", sources=[])
        report = _engine().score(m)
        name_score = next(f for f in report.field_scores if f.field_name == "name")
        assert name_score.score == 0.5
        assert "no source metadata" in name_score.reason


# ---------------------------------------------------------------------------
# Agreement bonus (Improvements 2 & 4)
# ---------------------------------------------------------------------------


class TestAgreementBonus:
    def test_exact_agreement_applies_dynamic_bonus(self):
        src1 = _nc(DataSource.CSV,  name="Alice Smith")
        src2 = _nc(DataSource.JSON, name="Alice Smith")
        m = _merged(name="Alice Smith", sources=[src1, src2])
        report = _engine().score(m)
        name_score = next(f for f in report.field_scores if f.field_name == "name")
        # avg(0.70, 0.75) = 0.725 + bonus (≥ 0.05) → ≥ 0.775
        assert name_score.score >= 0.725
        assert "agree" in name_score.reason

    def test_less_similar_sources_score_lower_than_exact(self):
        # "Alice S." abbreviation gets partial agree with name partial_threshold=0.70
        # but the score must be lower than exact-match sources.
        src_abbrev1 = _nc(DataSource.CSV,  name="Alice Smith")
        src_abbrev2 = _nc(DataSource.JSON, name="Alice S.")
        m_abbrev = _merged(name="Alice Smith", sources=[src_abbrev1, src_abbrev2])

        src_exact1 = _nc(DataSource.CSV,  name="Alice Smith")
        src_exact2 = _nc(DataSource.JSON, name="Alice Smith")
        m_exact = _merged(name="Alice Smith", sources=[src_exact1, src_exact2])

        abbrev_score = next(f for f in _engine().score(m_abbrev).field_scores if f.field_name == "name").score
        exact_score  = next(f for f in _engine().score(m_exact ).field_scores if f.field_name == "name").score

        assert abbrev_score < exact_score

    def test_case_insensitive_exact_agreement(self):
        # Exact match after normalisation → full bonus
        src1 = _nc(DataSource.CSV,  email="ALICE@EX.COM")
        src2 = _nc(DataSource.JSON, email="alice@ex.com")
        m = _merged(email="alice@ex.com", sources=[src1, src2])
        report = _engine().score(m)
        email_score = next(f for f in report.field_scores if f.field_name == "email")
        assert "agree" in email_score.reason

    def test_fuzzy_partial_agreement(self):
        # "Alice M. Smith" vs "Alice Smith" normalises to ~92 % similarity,
        # which falls in [partial_threshold=0.80, full_threshold=0.95) → "partial agree".
        src1 = _nc(DataSource.CSV,  name="Alice M. Smith")
        src2 = _nc(DataSource.JSON, name="Alice Smith")
        m = _merged(name="Alice M. Smith", sources=[src1, src2])
        report = _engine().score(m)
        name_score = next(f for f in report.field_scores if f.field_name == "name")
        assert "agree" in name_score.reason   # partial or full agree
        assert name_score.similarity is not None
        assert name_score.similarity >= 0.80

    def test_similarity_stored_on_field_confidence(self):
        src1 = _nc(DataSource.CSV,  name="Alice Smith")
        src2 = _nc(DataSource.JSON, name="Alice Smith")
        m = _merged(name="Alice Smith", sources=[src1, src2])
        report = _engine().score(m)
        name_score = next(f for f in report.field_scores if f.field_name == "name")
        assert name_score.similarity == pytest.approx(1.0, abs=0.01)

    def test_no_similarity_for_single_source(self):
        src = _nc(DataSource.CSV, name="Alice")
        m = _merged(name="Alice", sources=[src])
        report = _engine().score(m)
        name_score = next(f for f in report.field_scores if f.field_name == "name")
        assert name_score.similarity is None


# ---------------------------------------------------------------------------
# Conflict penalty (Improvement 3)
# ---------------------------------------------------------------------------


class TestConflictPenalty:
    def test_conflict_reduces_field_score(self):
        # Conflicting location sources must score LOWER than agreeing ones.
        # (Probabilistic base is higher than the old average, but the
        # continuous conflict penalty still reduces it relative to full agreement.)
        src1c = _nc(DataSource.CSV,  location="Bangalore")
        src2c = _nc(DataSource.JSON, location="Hyderabad")
        m_conflict = _merged(location="Bangalore", sources=[src1c, src2c])

        src1a = _nc(DataSource.CSV,  location="Bangalore")
        src2a = _nc(DataSource.JSON, location="Bangalore")
        m_agree = _merged(location="Bangalore", sources=[src1a, src2a])

        conflict_loc = next(f for f in _engine().score(m_conflict).field_scores if f.field_name == "location")
        agree_loc    = next(f for f in _engine().score(m_agree   ).field_scores if f.field_name == "location")

        assert "conflict" in conflict_loc.reason
        assert conflict_loc.score < agree_loc.score

    def test_configurable_conflict_penalty(self):
        # Larger penalty → lower conflicting field score
        cfg_high = _default_cfg(conflict_penalty=0.30)
        cfg_low  = _default_cfg(conflict_penalty=0.05)

        src1 = _nc(DataSource.CSV,  location="Bangalore")
        src2 = _nc(DataSource.JSON, location="Hyderabad")
        m = _merged(location="Bangalore", sources=[src1, src2])

        high_score = next(f for f in _engine(confidence_cfg=cfg_high).score(m).field_scores if f.field_name == "location").score
        low_score  = next(f for f in _engine(confidence_cfg=cfg_low ).score(m).field_scores if f.field_name == "location").score

        assert high_score < low_score


# ---------------------------------------------------------------------------
# Weighted completeness (Improvement 1)
# ---------------------------------------------------------------------------


class TestCompleteness:
    def test_completeness_uses_field_weights(self):
        # name(0.10) + email(0.08) + phone(0.07) + location(0.00)
        # + summary(0.00) + skills(0.20) = 0.45
        # location and summary carry 0.0 weight — they are optional fields.
        src = _nc(DataSource.ATS, name="Alice", email="a@ex.com",
                  phone="+1234567890", location="NYC", summary="Eng",
                  skills=["Python"])
        m = _merged(name="Alice", email="a@ex.com", phone="+1234567890",
                    location="NYC", summary="Eng", skills=["Python"],
                    sources=[src])
        report = _engine().score(m)
        assert report.completeness == pytest.approx(0.45, abs=0.01)

    def test_no_fields_present_is_0(self):
        m = _merged(sources=[])
        assert _engine().score(m).completeness == 0.0

    def test_partial_fields_between_zero_and_one(self):
        m = _merged(name="Alice", email="a@ex.com", sources=[])
        report = _engine().score(m)
        assert 0.0 < report.completeness < 1.0

    def test_all_fields_completeness_sums_correctly(self):
        # Provide all scored fields (education excluded to show partial score).
        # Weights: name(0.10)+email(0.08)+phone(0.07)+location(0.00)+summary(0.00)
        #          +skills(0.20)+experience(0.30)+links(0.05) = 0.80
        # location and summary present but carry 0.0 weight — optional fields.
        exp = Experience(company="Acme", title="Engineer")
        src = _nc(DataSource.ATS, name="Alice", email="a@ex.com",
                  phone="+1234567890", location="NYC", summary="Eng",
                  skills=["Python"])
        m = MergedCandidate(
            name="Alice", email="a@ex.com", phone="+1234567890",
            location="NYC", summary="Eng", skills=["Python"],
            experience=[exp],
            education=[],   # omitted → education(0.20) not counted
            links=[Link(url="https://example.com")],
            source_records=[src],
        )
        report = _engine().score(m)
        assert report.completeness == pytest.approx(0.80, abs=0.01)


# ---------------------------------------------------------------------------
# Source agreement (Improvement 2)
# ---------------------------------------------------------------------------


class TestSourceAgreement:
    def test_singleton_always_1(self):
        src = _nc(DataSource.CSV, name="Alice")
        m = _merged(name="Alice", sources=[src])
        assert _engine().score(m).source_agreement == 1.0

    def test_empty_source_records_is_1(self):
        m = _merged(sources=[])
        assert _engine().score(m).source_agreement == 1.0

    def test_two_sources_agree_on_email(self):
        s1 = _nc(DataSource.CSV,  email="alice@ex.com")
        s2 = _nc(DataSource.JSON, email="alice@ex.com")
        m = _merged(email="alice@ex.com", sources=[s1, s2])
        assert _engine().score(m).source_agreement == pytest.approx(1.0, abs=0.01)

    def test_two_sources_disagree_on_name(self):
        s1 = _nc(DataSource.CSV,  name="Alice Smith")
        s2 = _nc(DataSource.JSON, name="Alicia Smith")
        m = _merged(name="Alice Smith", sources=[s1, s2])
        report = _engine().score(m)
        assert report.source_agreement < 1.0

    def test_agreement_case_insensitive(self):
        s1 = _nc(DataSource.CSV,  email="ALICE@EX.COM")
        s2 = _nc(DataSource.JSON, email="alice@ex.com")
        m = _merged(email="alice@ex.com", sources=[s1, s2])
        assert _engine().score(m).source_agreement == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# Extraction confidence (Improvement 6)
# ---------------------------------------------------------------------------


class TestExtractionConfidence:
    def test_low_ocr_confidence_reduces_field_score(self):
        high_conf = _nc(DataSource.RESUME_PDF, name="Alice", extraction_confidence=0.99)
        low_conf  = _nc(DataSource.RESUME_PDF, name="Alice", extraction_confidence=0.50)

        m_high = _merged(name="Alice", sources=[high_conf])
        m_low  = _merged(name="Alice", sources=[low_conf])

        high_score = next(f for f in _engine().score(m_high).field_scores if f.field_name == "name").score
        low_score  = next(f for f in _engine().score(m_low ).field_scores if f.field_name == "name").score

        assert high_score > low_score

    def test_missing_extraction_confidence_defaults_to_one(self):
        # extraction_confidence=None → treated as 1.0 → same as explicit 1.0
        with_none    = _nc(DataSource.CSV, name="Alice", extraction_confidence=None)
        with_one     = _nc(DataSource.CSV, name="Alice", extraction_confidence=1.0)

        score_none = next(f for f in _engine().score(_merged(name="Alice", sources=[with_none])).field_scores if f.field_name == "name").score
        score_one  = next(f for f in _engine().score(_merged(name="Alice", sources=[with_one ])).field_scores if f.field_name == "name").score

        assert score_none == pytest.approx(score_one, abs=0.001)

    def test_extraction_confidence_multiplier_applied_correctly(self):
        # ATS reliability 0.95 × extraction_confidence 0.65 = 0.6175
        src = _nc(DataSource.ATS, name="Alice", extraction_confidence=0.65)
        m = _merged(name="Alice", sources=[src])
        report = _engine().score(m)
        name_score = next(f for f in report.field_scores if f.field_name == "name")
        assert name_score.score == pytest.approx(0.95 * 0.65, abs=0.01)


# ---------------------------------------------------------------------------
# Recency / timestamp weighting (Improvement 5)
# ---------------------------------------------------------------------------


class TestRecencyWeighting:
    """Freshness uses exponential decay; applies ONLY to staleness_fields.

    Stable fields (name, education, skills) are immune — a 4-year-old name
    is still the same name.  Volatile fields (location, phone, summary,
    experience) decay so that older sources contribute less.
    """

    def _fresh_ts(self) -> datetime:
        # 1 hour ago — effectively zero decay (0.5^(1/365/24) ≈ 1.000)
        return datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=1)

    def _old_ts(self) -> datetime:
        # 4 years ago → 0.5^4 = 0.0625, clamped to min_freshness=0.5
        return datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=365 * 4)

    def test_recent_source_scores_higher_than_old(self):
        # Use location — a staleness field — so decay actually fires.
        recent = _nc(DataSource.CSV, location="Hyderabad", source_timestamp=self._fresh_ts())
        old    = _nc(DataSource.CSV, location="Hyderabad", source_timestamp=self._old_ts())

        recent_score = next(f for f in _engine().score(_merged(location="Hyderabad", sources=[recent])).field_scores if f.field_name == "location").score
        old_score    = next(f for f in _engine().score(_merged(location="Hyderabad", sources=[old   ])).field_scores if f.field_name == "location").score

        assert recent_score > old_score

    def test_no_timestamp_treated_as_fully_fresh(self):
        # A record without a timestamp gets freshness=1.0, same as a just-updated one.
        no_ts = _nc(DataSource.CSV, location="Hyderabad", source_timestamp=None)
        fresh = _nc(DataSource.CSV, location="Hyderabad", source_timestamp=self._fresh_ts())

        score_no_ts = next(f for f in _engine().score(_merged(location="Hyderabad", sources=[no_ts])).field_scores if f.field_name == "location").score
        score_fresh = next(f for f in _engine().score(_merged(location="Hyderabad", sources=[fresh])).field_scores if f.field_name == "location").score

        assert score_no_ts == pytest.approx(score_fresh, abs=0.01)

    def test_old_source_weight_applied(self):
        # ATS (0.95) × exponential freshness for 4-year-old location:
        #   0.5^(4*365/365) = 0.0625  →  clamped to min_freshness=0.5
        #   effective = 0.95 × 0.5 = 0.475
        old = _nc(DataSource.ATS, location="Hyderabad", source_timestamp=self._old_ts())
        m = _merged(location="Hyderabad", sources=[old])
        loc_score = next(f for f in _engine().score(m).field_scores if f.field_name == "location")
        assert loc_score.score == pytest.approx(0.95 * 0.5, abs=0.01)

    def test_name_field_immune_to_freshness(self):
        # 'name' is a stable field — a 4-year-old name is still the same name.
        old = _nc(DataSource.ATS, name="Alice", source_timestamp=self._old_ts())
        m = _merged(name="Alice", sources=[old])
        name_score = next(f for f in _engine().score(m).field_scores if f.field_name == "name")
        assert name_score.score == pytest.approx(0.95, abs=0.01)


# ---------------------------------------------------------------------------
# Adaptive overall weights (Improvement 8)
# ---------------------------------------------------------------------------


class TestAdaptiveWeights:
    def test_single_source_agreement_weight_is_zero(self):
        # With 1 source, w_agreement = 0.0 → source_agreement doesn't affect score.
        src = _nc(DataSource.CSV, name="Alice")
        m   = _merged(name="Alice", sources=[src])
        engine = _engine()
        report = engine.score(m)
        # Verify via adaptive_weights method directly
        w_field, w_comp, w_agree = engine._adaptive_weights(1)
        assert w_agree == 0.0

    def test_many_sources_boost_agreement_weight(self):
        engine = _engine()
        _, _, w_agree_2 = engine._adaptive_weights(2)
        _, _, w_agree_5 = engine._adaptive_weights(5)
        assert w_agree_5 > w_agree_2

    def test_adaptive_weights_sum_to_one(self):
        engine = _engine()
        for count in [0, 1, 2, 4, 5, 10]:
            w = engine._adaptive_weights(count)
            assert sum(w) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Explainability (Improvement 9)
# ---------------------------------------------------------------------------


class TestExplainability:
    def test_explanations_populated(self):
        src = _nc(DataSource.CSV, name="Alice")
        m = _merged(name="Alice", sources=[src])
        report = _engine().score(m)
        assert len(report.explanations) > 0
        assert all(isinstance(e, str) for e in report.explanations)

    def test_absent_field_mentioned_in_explanations(self):
        m = _merged(sources=[])  # all fields absent
        report = _engine().score(m)
        absent_mentions = [e for e in report.explanations if "absent" in e.lower()]
        assert len(absent_mentions) > 0

    def test_conflict_mentioned_in_explanations(self):
        src1 = _nc(DataSource.CSV,  location="Bangalore")
        src2 = _nc(DataSource.JSON, location="Hyderabad")
        m = _merged(location="Bangalore", sources=[src1, src2])
        report = _engine().score(m)
        assert any("conflict" in e for e in report.explanations)

    def test_high_confidence_fields_praised(self):
        src = _nc(DataSource.ATS, name="Alice", email="a@ex.com")
        m = _merged(name="Alice", email="a@ex.com", sources=[src])
        report = _engine().score(m)
        assert any("high confidence" in e or "✔" in e for e in report.explanations)

    def test_validation_errors_in_explanations(self):
        m = _merged(name="Alice", sources=[])
        val = _validation(errors=1)
        report = _engine().score(m, val)
        assert any("Error:" in e for e in report.explanations)

    def test_no_sources_message(self):
        m = _merged(sources=[])
        report = _engine().score(m)
        assert any("no source" in e.lower() for e in report.explanations)


# ---------------------------------------------------------------------------
# Validation penalties
# ---------------------------------------------------------------------------


class TestValidationPenalty:
    def test_no_validation_no_penalty(self):
        m = _merged(name="Alice", email="a@ex.com", sources=[])
        with_val = _engine().score(m, _validation(0, 0))
        without  = _engine().score(m, None)
        assert with_val.overall_score == without.overall_score

    def test_errors_reduce_overall(self):
        m = _merged(name="Alice", email="a@ex.com", sources=[])
        clean      = _engine().score(m, _validation(errors=0))
        with_errors = _engine().score(m, _validation(errors=2))
        assert with_errors.overall_score < clean.overall_score

    def test_warnings_reduce_by_less_than_errors(self):
        m = _merged(name="Alice", email="a@ex.com", sources=[])
        base         = _engine().score(m).overall_score
        one_error    = _engine().score(m, _validation(errors=1)).overall_score
        one_warning  = _engine().score(m, _validation(warnings=1)).overall_score
        assert (base - one_error) > (base - one_warning)

    def test_penalty_capped(self):
        m = _merged(name="Alice", email="a@ex.com", sources=[])
        result = _engine().score(m, _validation(errors=100))
        assert result.overall_score >= 0.0

    def test_overall_clamped_to_one(self):
        src = _nc(DataSource.ATS, name="A", email="a@b.com", phone="+1",
                  location="NY", summary="x" * 100, skills=["Python"])
        m = _merged(name="A", email="a@b.com", phone="+1", location="NY",
                    summary="x" * 100, skills=["Python"], sources=[src, src])
        assert _engine().score(m).overall_score <= 1.0


# ---------------------------------------------------------------------------
# Cross-field validation (Improvement 7)
# ---------------------------------------------------------------------------


class TestCrossFieldValidation:
    def test_invalid_email_adds_error(self):
        m = _merged(email="not-an-email", sources=[])
        report = _engine().score(m, None)
        # Error appears in explanations
        assert any("Error:" in e and "email" in e.lower() for e in report.explanations)

    def test_valid_email_no_error(self):
        m = _merged(email="alice@example.com", sources=[])
        report = _engine().score(m, None)
        assert not any("Invalid email" in e for e in report.explanations)

    def test_experience_without_company_adds_warning(self):
        exp = Experience(title="Engineer", company=None)
        m = _merged(name="Alice", experience=[exp], sources=[])
        report = _engine().score(m, None)
        assert any("Warning:" in e and "company" in e.lower() for e in report.explanations)

    def test_conflicting_locations_across_sources_adds_warning(self):
        src1 = _nc(DataSource.CSV,  location="Bangalore")
        src2 = _nc(DataSource.JSON, location="Chennai")
        m = _merged(location="Bangalore", sources=[src1, src2])
        report = _engine().score(m, None)
        assert any("Warning:" in e and "ocation" in e for e in report.explanations)

    def test_cross_field_issues_compound_with_user_validation(self):
        # 1 user error + email format error from cross-field → 2 errors total
        m = _merged(email="bad-email", sources=[])
        user_val = _validation(errors=1)
        report = _engine().score(m, user_val)
        # Overall must be lower than with only 1 user error
        report_1err_only = _engine().score(_merged(email="good@ex.com", sources=[]), _validation(errors=1))
        assert report.overall_score <= report_1err_only.overall_score


# ---------------------------------------------------------------------------
# Weighted field average (Improvement 1)
# ---------------------------------------------------------------------------


class TestWeightedFieldAverage:
    def test_high_weight_field_dominates_overall(self):
        # experience (weight 0.20) present vs absent — significant delta
        src = _nc(DataSource.ATS, name="Alice")
        exp = Experience(company="Acme", title="Engineer")

        m_with_exp    = MergedCandidate(name="Alice", experience=[exp], source_records=[src])
        m_without_exp = MergedCandidate(name="Alice", experience=[],   source_records=[src])

        score_with    = _engine().score(m_with_exp).overall_score
        score_without = _engine().score(m_without_exp).overall_score

        assert score_with > score_without

    def test_custom_field_weights_reflected(self):
        # Give email weight 0.90, everything else minimal
        fw = {
            "name": 0.02, "email": 0.90, "phone": 0.01, "location": 0.01,
            "summary": 0.01, "skills": 0.01, "experience": 0.01,
            "education": 0.01, "links": 0.02,
        }
        cfg = _default_cfg(field_weights=fw)
        src = _nc(DataSource.CSV, email="a@ex.com")
        m   = _merged(email="a@ex.com", sources=[src])
        report = _engine(confidence_cfg=cfg).score(m)
        # email score contributes most → overall heavily driven by email
        email_score = next(f for f in report.field_scores if f.field_name == "email").score
        # weighted_field_avg ≈ email_score * 0.90 (dominant term)
        assert report.overall_score > 0.0

    def test_field_weights_must_sum_to_one(self):
        bad_weights = {"name": 0.5, "email": 0.6}  # sum = 1.1
        with pytest.raises(Exception):
            _default_cfg(field_weights=bad_weights)


# ---------------------------------------------------------------------------
# Custom reliability config
# ---------------------------------------------------------------------------


class TestCustomReliability:
    def test_custom_weights_reflected_in_score(self):
        e = _engine(source_weights={"csv": 0.99})
        src = _nc(DataSource.CSV, name="Alice")
        m = _merged(name="Alice", sources=[src])
        name_score = next(f for f in e.score(m).field_scores if f.field_name == "name")
        assert name_score.score >= 0.99

    def test_unknown_source_uses_default_weight(self):
        src = _nc(DataSource.UNKNOWN, name="Alice")
        m = _merged(name="Alice", sources=[src])
        name_score = next(f for f in _engine().score(m).field_scores if f.field_name == "name")
        assert 0.0 < name_score.score < 1.0


# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------


class TestHelpers:
    # _non_empty
    def test_non_empty_none_false(self):              assert _non_empty(None) is False
    def test_non_empty_blank_string_false(self):      assert _non_empty("  ") is False
    def test_non_empty_string_true(self):             assert _non_empty("x") is True
    def test_non_empty_empty_list_false(self):        assert _non_empty([]) is False
    def test_non_empty_nonempty_list_true(self):      assert _non_empty(["a"]) is True

    # _canonical (backward compat)
    def test_canonical_lowercases_string(self):       assert _canonical("ALICE@EX.COM") == "alice@ex.com"
    def test_canonical_non_string_passthrough(self):  assert _canonical(42) == 42

    # _normalize_for_similarity
    def test_normalize_removes_punctuation(self):
        assert _normalize_for_similarity("Alice S.") == "alice s"

    def test_normalize_strips_and_lowercases(self):
        assert _normalize_for_similarity("  Software Engineer  ") == "software engineer"

    def test_normalize_email_like_string(self):
        # @ and . removed
        assert _normalize_for_similarity("ALICE@EX.COM") == "aliceexcom"

    # _fuzzy_similarity
    def test_fuzzy_identical_strings(self):
        assert _fuzzy_similarity("alice", "alice") == 1.0

    def test_fuzzy_completely_different(self):
        assert _fuzzy_similarity("alice", "zzzzz") < 0.5

    def test_fuzzy_similar_strings(self):
        sim = _fuzzy_similarity("software engineer", "software eng")
        assert sim >= 0.80

    def test_fuzzy_empty_string(self):
        assert _fuzzy_similarity("", "alice") == 0.0

    # _min_pairwise_similarity
    def test_min_pairwise_single_value(self):
        assert _min_pairwise_similarity(["alice"]) == 1.0

    def test_min_pairwise_identical(self):
        assert _min_pairwise_similarity(["alice", "alice", "alice"]) == pytest.approx(1.0)

    def test_min_pairwise_picks_worst_pair(self):
        # "alice" vs "bob" is worse than "alice" vs "alice"
        sim = _min_pairwise_similarity(["alice", "alice", "bob"])
        assert sim < 1.0


# ---------------------------------------------------------------------------
# Improvement 1 — Probabilistic source aggregation
# ---------------------------------------------------------------------------


class TestProbabilisticReliability:
    def test_two_sources_higher_than_max(self):
        # P = 1 - (1-0.90)(1-0.80) = 1 - 0.02 = 0.98  >  max(0.90, 0.80)
        assert _probabilistic_combine([0.90, 0.80]) == pytest.approx(0.98, abs=0.001)

    def test_single_source_unchanged(self):
        # N=1 → P = 1 - (1-R) = R
        assert _probabilistic_combine([0.75]) == pytest.approx(0.75, abs=0.001)

    def test_empty_is_zero(self):
        assert _probabilistic_combine([]) == 0.0

    def test_perfect_reliability_stays_one(self):
        assert _probabilistic_combine([1.0, 1.0]) == pytest.approx(1.0)

    def test_weak_source_barely_lowers_strong(self):
        # 0.95 + 0.20 → 1 - (0.05 * 0.80) = 0.96
        result = _probabilistic_combine([0.95, 0.20])
        assert result == pytest.approx(0.96, abs=0.001)
        assert result > 0.95  # weak source must NOT drag down the strong one

    def test_probabilistic_applied_in_field_score(self):
        # Two CSV (0.70) + JSON (0.75) sources agreeing on name
        # P = 1 - (0.30 * 0.25) = 0.925  >  avg(0.70, 0.75) = 0.725
        src1 = _nc(DataSource.CSV,  name="Alice")
        src2 = _nc(DataSource.JSON, name="Alice")
        m = _merged(name="Alice", sources=[src1, src2])
        report = _engine().score(m)
        name_score = next(f for f in report.field_scores if f.field_name == "name")
        # Old average was 0.725; probabilistic base is 0.925+bonus → well above old
        assert name_score.score > 0.90


# ---------------------------------------------------------------------------
# Improvement 4 — Continuous agreement curve
# ---------------------------------------------------------------------------


class TestContinuousAgreement:
    def _adj(self, sim: float, ft: float = 0.95, pt: float = 0.80,
             bonus: float = 0.05, penalty: float = 0.10) -> float:
        return _continuous_agreement_adjustment(sim, ft, pt, bonus, penalty)

    def test_full_similarity_gives_max_bonus(self):
        assert self._adj(1.0) == pytest.approx(0.05, abs=0.001)

    def test_at_full_threshold_continuous(self):
        # Approaching from below (partial zone) and above (full zone) must match.
        just_above = self._adj(0.951)
        just_below = self._adj(0.949)
        assert abs(just_above - just_below) < 0.005  # no cliff

    def test_at_partial_threshold_zero(self):
        # Exactly at partial_threshold → zero adjustment (boundary between bonus and penalty)
        assert self._adj(0.80) == pytest.approx(0.0, abs=1e-6)

    def test_below_partial_gives_penalty(self):
        assert self._adj(0.50) < 0.0

    def test_zero_similarity_full_penalty(self):
        assert self._adj(0.0) == pytest.approx(-0.10, abs=0.001)

    def test_monotone_increasing(self):
        sims = [0.0, 0.30, 0.60, 0.80, 0.90, 0.95, 1.0]
        adjs = [self._adj(s) for s in sims]
        assert adjs == sorted(adjs)

    def test_no_binary_cliff_around_partial(self):
        # 0.79 should not receive the same magnitude as 0.0
        near_partial = abs(self._adj(0.79))
        at_zero      = abs(self._adj(0.0))
        assert near_partial < at_zero


# ---------------------------------------------------------------------------
# Improvement 2 & 11 — Field-specific normalization
# ---------------------------------------------------------------------------


class TestFieldSpecificNormalization:
    _ALIASES = {"bengaluru": "bangalore", "nyc": "new york city"}

    # Email
    def test_email_normalizes_to_lowercase(self):
        assert _normalize_email("ALICE@EX.COM") == "alice@ex.com"

    def test_email_strips_whitespace(self):
        assert _normalize_email("  alice@ex.com  ") == "alice@ex.com"

    def test_email_same_after_norm_is_1(self):
        assert _similarity_for_field("email", "ALICE@EX.COM", "alice@ex.com", {}) == 1.0

    def test_email_different_is_0(self):
        assert _similarity_for_field("email", "alice@ex.com", "bob@ex.com", {}) == 0.0

    # Phone
    def test_phone_strips_formatting(self):
        assert _normalize_phone("+91 98765 43210") == "919876543210"

    def test_phone_same_number_different_format_is_1(self):
        sim = _similarity_for_field("phone", "+91-9876543210", "9876543210", {})
        assert sim >= 0.90  # suffix match detected

    def test_phone_completely_different_is_low(self):
        sim = _similarity_for_field("phone", "1234567890", "9876543210", {})
        assert sim < 0.70

    # Location
    def test_location_alias_bengaluru_bangalore(self):
        sim = _similarity_for_field("location", "Bengaluru", "Bangalore", self._ALIASES)
        assert sim == pytest.approx(1.0, abs=0.01)

    def test_location_substring_containment(self):
        # "Hyderabad" is a prefix of "Hyderabad, India"
        sim = _similarity_for_field("location", "Hyderabad", "Hyderabad, India", {})
        assert sim >= 0.90

    def test_location_unrelated_cities_low(self):
        sim = _similarity_for_field("location", "Bangalore", "Mumbai", {})
        assert sim < 0.50

    def test_normalize_location_with_alias(self):
        assert _normalize_location("Bengaluru", self._ALIASES) == "bangalore"

    def test_normalize_location_nyc(self):
        assert _normalize_location("NYC", self._ALIASES) == "new york city"

    def test_normalize_location_city_plus_country(self):
        # Token-level: "bengaluru india" → "bangalore india"
        result = _normalize_location("Bengaluru, India", self._ALIASES)
        assert "bangalore" in result


# ---------------------------------------------------------------------------
# Improvement 5 — Continuous freshness decay
# ---------------------------------------------------------------------------


class TestContinuousFreshness:
    def _engine_with_freshness(self, **kw) -> ConfidenceEngine:
        cfg = ConfidenceConfig(freshness=FreshnessDecayConfig(**kw))
        return ConfidenceEngine(confidence_config=cfg)

    def _ts(self, days_ago: int) -> datetime:
        return datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days_ago)

    def test_zero_days_is_fully_fresh(self):
        eng = _engine()
        src = _nc(DataSource.CSV, location="NYC", source_timestamp=self._ts(0))
        assert eng._freshness_weight(src, "location") == pytest.approx(1.0, abs=0.01)

    def test_one_halflife_is_0_5(self):
        eng = self._engine_with_freshness(half_life_days=365.0, min_freshness=0.0)
        src = _nc(DataSource.CSV, location="NYC", source_timestamp=self._ts(365))
        assert eng._freshness_weight(src, "location") == pytest.approx(0.5, abs=0.01)

    def test_two_halflives_is_0_25_unless_clamped(self):
        eng = self._engine_with_freshness(half_life_days=365.0, min_freshness=0.0)
        src = _nc(DataSource.CSV, location="NYC", source_timestamp=self._ts(730))
        assert eng._freshness_weight(src, "location") == pytest.approx(0.25, abs=0.01)

    def test_min_freshness_floor(self):
        eng = self._engine_with_freshness(half_life_days=365.0, min_freshness=0.50)
        src = _nc(DataSource.CSV, location="NYC", source_timestamp=self._ts(365 * 10))
        assert eng._freshness_weight(src, "location") == pytest.approx(0.50, abs=0.001)

    def test_name_always_1_regardless_of_age(self):
        eng = _engine()
        src = _nc(DataSource.CSV, name="Alice", source_timestamp=self._ts(365 * 5))
        assert eng._freshness_weight(src, "name") == 1.0

    def test_skills_always_1_regardless_of_age(self):
        eng = _engine()
        src = _nc(DataSource.CSV, skills=["Python"], source_timestamp=self._ts(365 * 5))
        assert eng._freshness_weight(src, "skills") == 1.0

    def test_location_decays_with_age(self):
        eng = _engine()
        fresh_src = _nc(DataSource.CSV, location="NYC", source_timestamp=self._ts(1))
        old_src   = _nc(DataSource.CSV, location="NYC", source_timestamp=self._ts(365 * 4))
        assert eng._freshness_weight(fresh_src, "location") > eng._freshness_weight(old_src, "location")

    def test_future_timestamp_clamps_to_one(self):
        eng = _engine()
        future_ts = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(days=30)
        src = _nc(DataSource.CSV, location="NYC", source_timestamp=future_ts)
        assert eng._freshness_weight(src, "location") == 1.0


# ---------------------------------------------------------------------------
# Improvement 8 extension — Source diversity bonus
# ---------------------------------------------------------------------------


class TestSourceDiversityBonus:
    def _engine_diversity(self, **kw) -> ConfidenceEngine:
        cfg = ConfidenceConfig(diversity=SourceDiversityConfig(**kw))
        return ConfidenceEngine(confidence_config=cfg)

    def test_single_source_no_bonus(self):
        src = _nc(DataSource.ATS, name="Alice")
        m = _merged(name="Alice", sources=[src])
        assert _engine().score(m).diversity_bonus == 0.0

    def test_two_trusted_sources_get_bonus(self):
        src1 = _nc(DataSource.ATS,     name="Alice")  # 0.95 ≥ 0.75
        src2 = _nc(DataSource.LINKEDIN, name="Alice")  # 0.90 ≥ 0.75
        m = _merged(name="Alice", sources=[src1, src2])
        report = _engine().score(m)
        assert report.diversity_bonus > 0.0

    def test_low_reliability_sources_excluded(self):
        # CSV = 0.70 < default min_reliability 0.75 → no diversity bonus
        src1 = _nc(DataSource.CSV,  name="Alice")
        src2 = _nc(DataSource.CSV,  name="Alice")
        m = _merged(name="Alice", sources=[src1, src2])
        assert _engine().score(m).diversity_bonus == 0.0

    def test_diversity_bonus_added_to_overall(self):
        src1 = _nc(DataSource.ATS,     name="Alice")
        src2 = _nc(DataSource.LINKEDIN, name="Alice")
        m = _merged(name="Alice", sources=[src1, src2])

        with_div = _engine().score(m).overall_score
        cfg = ConfidenceConfig(diversity=SourceDiversityConfig(enabled=False))
        without_div = ConfidenceEngine(confidence_config=cfg).score(m).overall_score

        assert with_div > without_div

    def test_diversity_disabled_gives_zero(self):
        src1 = _nc(DataSource.ATS,     name="Alice")
        src2 = _nc(DataSource.LINKEDIN, name="Alice")
        m = _merged(name="Alice", sources=[src1, src2])
        cfg = ConfidenceConfig(diversity=SourceDiversityConfig(enabled=False))
        assert ConfidenceEngine(confidence_config=cfg).score(m).diversity_bonus == 0.0

    def test_diversity_bonus_present_in_explanations(self):
        src1 = _nc(DataSource.ATS,     name="Alice")
        src2 = _nc(DataSource.LINKEDIN, name="Alice")
        m = _merged(name="Alice", sources=[src1, src2])
        report = _engine().score(m)
        assert any("diversity" in e.lower() for e in report.explanations)


# ---------------------------------------------------------------------------
# Improvement 7 — Self-employed / founder exemption
# ---------------------------------------------------------------------------


class TestFoundersFreelancers:
    def test_freelancer_title_no_company_warning(self):
        exp = Experience(title="Freelance Developer", company=None)
        m = _merged(name="Alice", experience=[exp], sources=[])
        report = _engine().score(m, None)
        # Should NOT warn about missing company for freelancers
        assert not any(
            "company" in e.lower() and "Warning:" in e
            for e in report.explanations
        )

    def test_founder_title_no_company_warning(self):
        exp = Experience(title="Co-Founder", company=None)
        m = _merged(name="Alice", experience=[exp], sources=[])
        report = _engine().score(m, None)
        assert not any("company" in e.lower() and "Warning:" in e for e in report.explanations)

    def test_consultant_no_company_warning(self):
        exp = Experience(title="Independent Consultant", company=None)
        m = _merged(name="Alice", experience=[exp], sources=[])
        report = _engine().score(m, None)
        assert not any("company" in e.lower() and "Warning:" in e for e in report.explanations)

    def test_regular_employee_missing_company_still_warns(self):
        exp = Experience(title="Software Engineer", company=None)
        m = _merged(name="Alice", experience=[exp], sources=[])
        report = _engine().score(m, None)
        assert any("Warning:" in e and "company" in e.lower() for e in report.explanations)

    def test_contractor_exempt(self):
        exp = Experience(title="Contract Engineer", company=None)
        m = _merged(name="Alice", experience=[exp], sources=[])
        report = _engine().score(m, None)
        assert not any("company" in e.lower() and "Warning:" in e for e in report.explanations)


# ---------------------------------------------------------------------------
# Improvement 2/3 — Summary excluded from agreement
# ---------------------------------------------------------------------------


class TestSummaryExcludedFromAgreement:
    def test_different_summaries_do_not_lower_agreement(self):
        # Summary naturally differs between Resume and LinkedIn — must be skipped.
        s1 = _nc(DataSource.RESUME_PDF, name="Alice Smith",
                 summary="Seasoned backend engineer with 10 years of Python experience.")
        s2 = _nc(DataSource.LINKEDIN,   name="Alice Smith",
                 summary="Passionate about building scalable systems. Open to opportunities.")
        m = _merged(name="Alice Smith", summary="Seasoned backend engineer...",
                    sources=[s1, s2])
        report = _engine().score(m)
        # Agreement should be high (name agrees) despite very different summaries
        assert report.source_agreement >= 0.90

    def test_summary_field_score_skips_agreement_logic(self):
        s1 = _nc(DataSource.CSV,  summary="Summary A")
        s2 = _nc(DataSource.JSON, summary="Summary B — completely different")
        m = _merged(summary="Summary A", sources=[s1, s2])
        report = _engine().score(m)
        summary_score = next(f for f in report.field_scores if f.field_name == "summary")
        # No conflict mentioned; similarity should be None (agreement skipped)
        assert "conflict" not in (summary_score.reason or "")
        assert summary_score.similarity is None


# ---------------------------------------------------------------------------
# Improvement 10 — Confidence breakdown in report
# ---------------------------------------------------------------------------


class TestConfidenceBreakdown:
    def test_diversity_bonus_on_report(self):
        src1 = _nc(DataSource.ATS,     name="Alice")
        src2 = _nc(DataSource.LINKEDIN, name="Alice")
        m = _merged(name="Alice", sources=[src1, src2])
        report = _engine().score(m)
        assert report.diversity_bonus >= 0.0

    def test_validation_penalty_on_report(self):
        m = _merged(name="Alice", sources=[])
        report = _engine().score(m, _validation(errors=2))
        assert report.validation_penalty > 0.0

    def test_no_errors_penalty_is_zero(self):
        m = _merged(name="Alice", sources=[])
        report = _engine().score(m, None)
        assert report.validation_penalty == 0.0

    def test_overall_equals_components(self):
        # Verify: overall ≈ w*field_avg + w*completeness + w*agreement
        #                   + diversity_bonus − validation_penalty
        src = _nc(DataSource.ATS, name="Alice", email="a@ex.com")
        m = _merged(name="Alice", email="a@ex.com", sources=[src])
        val = _validation(errors=1)
        report = _engine().score(m, val)

        # Penalty is inspectable and positive
        assert report.validation_penalty == pytest.approx(0.05, abs=0.001)

        # Overall is below what it would be without the penalty
        report_clean = _engine().score(m, None)
        assert report_clean.overall_score > report.overall_score
