"""Entity Resolution Engine.

Determines which NormalizedCandidate records from different sources describe
the same real-world person, and groups them into CandidateGroup objects for
the Merge Engine.

Matching signals (checked in order, first hit wins):
  1. Email — exact, case-insensitive (configurable via email_exact_match flag)
  2. Name  — fuzzy token_sort_ratio via rapidfuzz (threshold from YAML config)

Algorithm: Union-Find with path-halving compression.
O(n²) pair comparisons; fine for realistic batch sizes (2–20 candidates).
"""

from __future__ import annotations

from itertools import combinations

from rapidfuzz import fuzz

from app.config.loader import get_config
from app.config.models import EntityResolutionConfig
from app.mergers.candidate_group import CandidateGroup
from app.models.candidate import NormalizedCandidate
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Union-Find helpers (module-level, pure functions)
# ---------------------------------------------------------------------------


def _find(parent: list[int], x: int) -> int:
    """Path-halving find — reduces tree height without full compression."""
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _union(parent: list[int], a: int, b: int) -> None:
    """Merge the sets containing *a* and *b*."""
    parent[_find(parent, b)] = _find(parent, a)


# ---------------------------------------------------------------------------
# EntityResolver
# ---------------------------------------------------------------------------


class EntityResolver:
    """Group NormalizedCandidate records that represent the same person.

    The instance is stateless between calls — safe to instantiate once and
    reuse across many batches.
    """

    def __init__(self, config: EntityResolutionConfig | None = None) -> None:
        self._config: EntityResolutionConfig = (
            config if config is not None else get_config().entity_resolution
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(self, candidates: list[NormalizedCandidate]) -> list[CandidateGroup]:
        """Cluster *candidates* into identity groups.

        Args:
            candidates: NormalizedCandidate records, typically one per input
                source for a single pipeline run.

        Returns:
            A list of :class:`CandidateGroup` objects.  Every candidate
            appears in exactly one group.  Records with no matching peer
            become singleton groups.  Never raises.
        """
        n = len(candidates)
        if n == 0:
            return []

        parent = list(range(n))

        # edge_reasons[( min(i,j), max(i,j) )] = reason string
        edge_reasons: dict[tuple[int, int], str] = {}

        for i, j in combinations(range(n), 2):
            matched, reason = self._is_same_person(candidates[i], candidates[j])
            if matched:
                edge_reasons[(i, j)] = reason
                _union(parent, i, j)

        # Collect indices by root
        buckets: dict[int, list[int]] = {}
        for i in range(n):
            buckets.setdefault(_find(parent, i), []).append(i)

        result: list[CandidateGroup] = []
        for indices in buckets.values():
            group_candidates = [candidates[i] for i in indices]
            group_reasons = [
                edge_reasons[edge]
                for edge in combinations(sorted(indices), 2)
                if edge in edge_reasons
            ]
            group = CandidateGroup(
                candidates=group_candidates, match_reasons=group_reasons
            )
            result.append(group)
            logger.debug(
                "EntityResolver: group size=%d sources=%s reasons=%s",
                len(group_candidates),
                [str(c.source) for c in group_candidates],
                group_reasons,
            )

        logger.info(
            "EntityResolver: %d candidate(s) -> %d group(s)", n, len(result)
        )
        return result

    # ------------------------------------------------------------------
    # Matching logic
    # ------------------------------------------------------------------

    def _is_same_person(
        self, a: NormalizedCandidate, b: NormalizedCandidate
    ) -> tuple[bool, str]:
        """Return *(is_match, reason)* for two candidates.

        Email is checked first because it is a stronger identity signal than
        a name match; two people can share a name but not an email address.
        """
        # 1. Email: exact, case-insensitive ---------------------------------
        if self._config.email_exact_match and a.email and b.email:
            if a.email.lower() == b.email.lower():
                return True, "email:exact"

        # 2. Name: fuzzy token_sort_ratio -----------------------------------
        # token_sort_ratio sorts tokens before comparing, so
        # "John A. Smith" and "Smith, John" both normalise to the same token
        # multiset before the ratio is computed.
        if a.name and b.name:
            score = fuzz.token_sort_ratio(a.name, b.name) / 100.0
            if score >= self._config.name_similarity_threshold:
                return True, f"name:fuzzy({score:.2f})"

        return False, ""
