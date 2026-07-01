"""Abstract base for all field-level normalizers (Strategy pattern).

Each normalizer handles exactly one type of value (str, list[str], etc.)
and returns a cleaned canonical form of that value.

Contract:
- Never raise.  Return the original value unchanged on any failure.
- Accept None / empty inputs and return them unchanged.
- Be idempotent: normalize(normalize(x)) == normalize(x).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from app.config.models import NormalizationConfig

T = TypeVar("T")


class BaseFieldNormalizer(ABC, Generic[T]):
    """Strategy interface for normalizing a single field type."""

    @abstractmethod
    def normalize(self, value: T, config: NormalizationConfig) -> T:
        """Return the canonical form of *value*.

        Args:
            value: Raw field value from an ExtractedCandidate.
            config: Pipeline normalization configuration.

        Returns:
            Cleaned value, or *value* unchanged if normalization fails.
        """
