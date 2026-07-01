"""Unit tests for the Entity Resolution Engine.

Coverage targets:
  - Email exact-match grouping (case-insensitive)
  - Name fuzzy-match grouping
  - Transitivity via union-find (A matches B, B matches C → one group)
  - Singletons when no match is found
  - Config override (threshold, email_exact_match flag)
  - CandidateGroup properties: is_singleton, sources, primary_email, primary_name
  - Every candidate appears in exactly one group (partition invariant)
"""

from __future__ import annotations

import pytest

from app.config.models import EntityResolutionConfig
from app.mergers.candidate_group import CandidateGroup
from app.mergers.entity_resolver import EntityResolver
from app.models.candidate import DataSource, NormalizedCandidate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _candidate(
    name: str | None = "Alice Smith",
    email: str | None = "alice@example.com",
    source: DataSource = DataSource.CSV,
    **kwargs,
) -> NormalizedCandidate:
    return NormalizedCandidate(source=source, name=name, email=email, **kwargs)


def _resolver(
    name_threshold: float = 0.85,
    email_exact: bool = True,
) -> EntityResolver:
    cfg = EntityResolutionConfig(
        name_similarity_threshold=name_threshold,
        email_exact_match=email_exact,
    )
    return EntityResolver(config=cfg)


def _all_candidates(groups: list[CandidateGroup]) -> list[NormalizedCandidate]:
    return [c for g in groups for c in g.candidates]


# ---------------------------------------------------------------------------
# Empty / single-candidate edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_list_returns_empty(self):
        assert _resolver().resolve([]) == []

    def test_single_candidate_is_singleton(self):
        c = _candidate()
        groups = _resolver().resolve([c])
        assert len(groups) == 1
        assert groups[0].candidates == [c]
        assert groups[0].is_singleton is True
        assert groups[0].match_reasons == []

    def test_partition_invariant_two_groups(self):
        a = _candidate(name="Alice Smith", email="alice@ex.com")
        b = _candidate(name="Bob Jones", email="bob@ex.com")
        groups = _resolver().resolve([a, b])
        assert len(groups) == 2
        ids = {id(c) for c in _all_candidates(groups)}
        assert ids == {id(a), id(b)}

    def test_partition_invariant_one_group(self):
        a = _candidate(email="shared@ex.com")
        b = _candidate(email="shared@ex.com", source=DataSource.JSON)
        groups = _resolver().resolve([a, b])
        assert len(groups) == 1
        ids = {id(c) for c in _all_candidates(groups)}
        assert ids == {id(a), id(b)}


# ---------------------------------------------------------------------------
# Email matching
# ---------------------------------------------------------------------------


class TestEmailMatching:
    def test_same_email_groups(self):
        a = _candidate(name="Alice S.", email="alice@example.com", source=DataSource.CSV)
        b = _candidate(name="Alice Smith", email="alice@example.com", source=DataSource.JSON)
        groups = _resolver().resolve([a, b])
        assert len(groups) == 1

    def test_email_match_case_insensitive(self):
        a = _candidate(email="Alice@EXAMPLE.COM", source=DataSource.CSV)
        b = _candidate(email="alice@example.com", source=DataSource.JSON)
        groups = _resolver().resolve([a, b])
        assert len(groups) == 1

    def test_different_emails_no_email_match(self):
        a = _candidate(name="Alice Smith", email="alice@foo.com")
        b = _candidate(name="Bob Jones", email="bob@bar.com")
        groups = _resolver().resolve([a, b])
        assert len(groups) == 2

    def test_email_match_reason_is_email_exact(self):
        a = _candidate(email="shared@ex.com", source=DataSource.CSV)
        b = _candidate(email="shared@ex.com", source=DataSource.JSON)
        groups = _resolver().resolve([a, b])
        assert len(groups) == 1
        assert any("email:exact" in r for r in groups[0].match_reasons)

    def test_email_exact_match_disabled_skips_email(self):
        # Same email but email_exact_match=False → must fall through to name
        a = _candidate(name="Alice Smith", email="shared@ex.com", source=DataSource.CSV)
        b = _candidate(name="Bob Jones", email="shared@ex.com", source=DataSource.JSON)
        groups = _resolver(email_exact=False).resolve([a, b])
        # Names differ → two groups (email check skipped)
        assert len(groups) == 2

    def test_null_email_skips_email_check(self):
        # No email on either side; must use name
        a = _candidate(name="Alice Smith", email=None)
        b = _candidate(name="Alice Smith", email=None, source=DataSource.JSON)
        groups = _resolver().resolve([a, b])
        assert len(groups) == 1
        assert any("name:fuzzy" in r for r in groups[0].match_reasons)

    def test_one_null_email_skips_email_check(self):
        # Only one has email; cannot do exact match
        a = _candidate(name="Alice Smith", email="alice@ex.com")
        b = _candidate(name="Alice Smith", email=None, source=DataSource.JSON)
        groups = _resolver().resolve([a, b])
        # Name similarity should still group them
        assert len(groups) == 1


# ---------------------------------------------------------------------------
# Name fuzzy matching
# ---------------------------------------------------------------------------


class TestNameMatching:
    def test_identical_names_match(self):
        a = _candidate(name="John Smith", email="a@foo.com")
        b = _candidate(name="John Smith", email="b@foo.com", source=DataSource.JSON)
        groups = _resolver().resolve([a, b])
        assert len(groups) == 1
        assert any("name:fuzzy" in r for r in groups[0].match_reasons)

    def test_token_sort_handles_middle_initial(self):
        # "John A. Smith" vs "John Smith" → token_sort_ratio handles reordering
        a = _candidate(name="John A. Smith", email="j1@foo.com")
        b = _candidate(name="John Smith", email="j2@foo.com", source=DataSource.JSON)
        groups = _resolver().resolve([a, b])
        assert len(groups) == 1

    def test_completely_different_names_no_match(self):
        a = _candidate(name="Alice Johnson", email="a@foo.com")
        b = _candidate(name="Roberto Fernandez", email="r@foo.com", source=DataSource.JSON)
        groups = _resolver().resolve([a, b])
        assert len(groups) == 2

    def test_name_match_reason_contains_score(self):
        a = _candidate(name="John Smith", email="a@foo.com")
        b = _candidate(name="John Smith", email="b@foo.com", source=DataSource.JSON)
        groups = _resolver().resolve([a, b])
        reason = groups[0].match_reasons[0]
        assert reason.startswith("name:fuzzy(")
        score = float(reason[len("name:fuzzy("):-1])
        assert 0.0 <= score <= 1.0

    def test_null_name_skips_name_check(self):
        a = _candidate(name=None, email="a@foo.com")
        b = _candidate(name=None, email="b@foo.com", source=DataSource.JSON)
        # No email match, no name match → two singletons
        groups = _resolver().resolve([a, b])
        assert len(groups) == 2

    def test_one_null_name_skips_name_check(self):
        a = _candidate(name="Alice Smith", email="a@foo.com")
        b = _candidate(name=None, email="b@foo.com", source=DataSource.JSON)
        groups = _resolver().resolve([a, b])
        assert len(groups) == 2


# ---------------------------------------------------------------------------
# Threshold configuration
# ---------------------------------------------------------------------------


class TestThresholds:
    def test_high_threshold_rejects_similar_names(self):
        # "John Smith" vs "John Smyth" — similar but likely below 0.99
        a = _candidate(name="John Smith", email="a@foo.com")
        b = _candidate(name="John Smyth", email="b@foo.com", source=DataSource.JSON)
        groups = _resolver(name_threshold=0.99).resolve([a, b])
        assert len(groups) == 2

    def test_low_threshold_accepts_different_names(self):
        # Even "Alice Johnson" vs "Alice Johnsone" should match at threshold=0.40
        a = _candidate(name="Alice Johnson", email="a@foo.com")
        b = _candidate(name="Alice Johnsone", email="b@foo.com", source=DataSource.JSON)
        groups = _resolver(name_threshold=0.40).resolve([a, b])
        assert len(groups) == 1


# ---------------------------------------------------------------------------
# Transitivity (union-find correctness)
# ---------------------------------------------------------------------------


class TestTransitivity:
    def test_a_matches_b_b_matches_c_all_in_one_group(self):
        # A and B share email; B and C share name — A, B, C must all be one group
        a = _candidate(name="Alice Smith", email="shared@ex.com", source=DataSource.CSV)
        b = _candidate(name="Alice Smith", email="shared@ex.com", source=DataSource.JSON)
        c = _candidate(name="Alice Smith", email="other@ex.com", source=DataSource.RESUME_PDF)
        groups = _resolver().resolve([a, b, c])
        assert len(groups) == 1
        assert len(groups[0].candidates) == 3

    def test_two_separate_chains(self):
        # Pair 1: a–b (email)   Pair 2: c–d (name)
        a = _candidate(name="Alice Smith", email="same@ex.com", source=DataSource.CSV)
        b = _candidate(name="Alice Smith", email="same@ex.com", source=DataSource.JSON)
        c = _candidate(name="Bob Jones", email="c@ex.com", source=DataSource.RESUME_PDF)
        d = _candidate(name="Bob Jones", email="d@ex.com", source=DataSource.ATS)
        groups = _resolver().resolve([a, b, c, d])
        assert len(groups) == 2
        sizes = sorted(len(g.candidates) for g in groups)
        assert sizes == [2, 2]

    def test_all_candidates_in_exactly_one_group(self):
        candidates = [
            _candidate(name="Alice Smith", email="alice@ex.com", source=DataSource.CSV),
            _candidate(name="Alice Smith", email="alice@ex.com", source=DataSource.JSON),
            _candidate(name="Bob Jones", email="bob@ex.com", source=DataSource.RESUME_PDF),
        ]
        groups = _resolver().resolve(candidates)
        found = _all_candidates(groups)
        assert len(found) == len(candidates)
        assert {id(c) for c in found} == {id(c) for c in candidates}


# ---------------------------------------------------------------------------
# CandidateGroup properties
# ---------------------------------------------------------------------------


class TestCandidateGroupProperties:
    def _two_candidate_group(self) -> CandidateGroup:
        a = _candidate(name="Alice Smith", email="alice@ex.com", source=DataSource.CSV)
        b = _candidate(name="Alice Smith", email="alice@ex.com", source=DataSource.JSON)
        return _resolver().resolve([a, b])[0]

    def test_is_singleton_true(self):
        group = _resolver().resolve([_candidate()])[0]
        assert group.is_singleton is True

    def test_is_singleton_false(self):
        assert self._two_candidate_group().is_singleton is False

    def test_sources_lists_all(self):
        group = self._two_candidate_group()
        assert DataSource.CSV in group.sources
        assert DataSource.JSON in group.sources

    def test_primary_email_returns_first_non_null(self):
        a = _candidate(name="Alice", email=None, source=DataSource.CSV)
        b = _candidate(name="Alice", email="alice@ex.com", source=DataSource.JSON)
        group = CandidateGroup(candidates=[a, b])
        assert group.primary_email == "alice@ex.com"

    def test_primary_email_all_null(self):
        a = _candidate(email=None, source=DataSource.CSV)
        b = _candidate(email=None, source=DataSource.JSON)
        group = CandidateGroup(candidates=[a, b])
        assert group.primary_email is None

    def test_primary_name_returns_first_non_null(self):
        a = _candidate(name=None, source=DataSource.CSV)
        b = _candidate(name="Alice Smith", source=DataSource.JSON)
        group = CandidateGroup(candidates=[a, b])
        assert group.primary_name == "Alice Smith"

    def test_singleton_match_reasons_empty(self):
        group = _resolver().resolve([_candidate()])[0]
        assert group.match_reasons == []
