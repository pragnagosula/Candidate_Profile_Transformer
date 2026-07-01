"""CandidateGroup — a cluster of NormalizedCandidate records believed to
represent the same real-world person, produced by the EntityResolver.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.models.candidate import DataSource, NormalizedCandidate


class CandidateGroup(BaseModel):
    """A set of NormalizedCandidates that entity resolution determined describe
    the same real-world candidate.

    Passed directly to the Merge Engine; it never raises on construction.
    """

    model_config = {"arbitrary_types_allowed": True}

    candidates: list[NormalizedCandidate]
    match_reasons: list[str] = Field(default_factory=list)

    @property
    def is_singleton(self) -> bool:
        """True when only one source contributed (no duplicate was detected)."""
        return len(self.candidates) == 1

    @property
    def sources(self) -> list[DataSource]:
        return [c.source for c in self.candidates]

    @property
    def primary_email(self) -> str | None:
        """First non-null email found among candidates."""
        for c in self.candidates:
            if c.email:
                return c.email
        return None

    @property
    def primary_name(self) -> str | None:
        """First non-null name found among candidates."""
        for c in self.candidates:
            if c.name:
                return c.name
        return None
