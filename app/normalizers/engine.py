"""NormalizationEngine — applies all field-level normalizers to a candidate.

Each field in ExtractedCandidate is routed to the correct normalizer.
The engine also normalizes nested structures (DateRange inside Experience/Education)
and tracks which fields actually changed for downstream provenance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.config.models import NormalizationConfig
from app.models.candidate import (
    DateRange,
    Education,
    Experience,
    ExtractedCandidate,
    Link,
    NormalizedCandidate,
)
from app.normalizers.date_normalizer import DateNormalizer
from app.normalizers.email_normalizer import EmailNormalizer
from app.normalizers.name_normalizer import NameNormalizer
from app.normalizers.phone_normalizer import PhoneNormalizer
from app.normalizers.skill_normalizer import SkillNormalizer
from app.normalizers.url_normalizer import URLNormalizer
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class NormalizationDiff:
    """Records which fields changed during normalization for provenance."""
    changes: dict[str, tuple[Any, Any]] = field(default_factory=dict)

    def record(self, field_name: str, original: Any, normalized: Any) -> None:
        if original != normalized:
            self.changes[field_name] = (original, normalized)


class NormalizationEngine:
    """Orchestrate all normalizers and produce a NormalizedCandidate.

    Instantiate once per pipeline run (or share the instance — it is
    stateless after construction).
    """

    def __init__(self, config: NormalizationConfig) -> None:
        self._config = config
        self._email = EmailNormalizer()
        self._phone = PhoneNormalizer()
        self._date = DateNormalizer()
        self._name = NameNormalizer()
        self._skill = SkillNormalizer()
        self._url = URLNormalizer()

    def normalize(
        self, extracted: ExtractedCandidate
    ) -> tuple[NormalizedCandidate, NormalizationDiff]:
        """Normalize all fields of an ExtractedCandidate.

        Args:
            extracted: Output from an extractor.

        Returns:
            Tuple of (NormalizedCandidate, NormalizationDiff).
            The diff records every (original, normalized) pair where a
            value actually changed.
        """
        cfg = self._config
        diff = NormalizationDiff()

        name = self._name.normalize(extracted.name, cfg)
        diff.record("name", extracted.name, name)

        email = self._email.normalize(extracted.email, cfg)
        diff.record("email", extracted.email, email)

        phone = self._phone.normalize(extracted.phone, cfg)
        diff.record("phone", extracted.phone, phone)

        skills = self._skill.normalize(extracted.skills, cfg)
        diff.record("skills", tuple(extracted.skills), tuple(skills))

        experience = [self._normalize_experience(exp, diff) for exp in extracted.experience]
        education = [self._normalize_education(edu, diff) for edu in extracted.education]
        links = [self._normalize_link(link, diff) for link in extracted.links]

        normalized = NormalizedCandidate(
            source=extracted.source,
            source_file=extracted.source_file,
            name=name,
            email=email,
            phone=phone,
            location=extracted.location,  # normalization: whitespace only
            summary=extracted.summary,
            skills=skills,
            experience=experience,
            education=education,
            links=links,
            extra_fields=extracted.extra_fields,
        )

        logger.debug(
            "NormalizationEngine: %d fields changed for source=%s",
            len(diff.changes),
            extracted.source,
        )
        return normalized, diff

    # ------------------------------------------------------------------
    # Private helpers for nested structures
    # ------------------------------------------------------------------

    def _normalize_date_range(self, dr: DateRange | None) -> DateRange | None:
        if dr is None:
            return None
        return DateRange(
            start=self._date.normalize(dr.start, self._config),
            end=self._date.normalize(dr.end, self._config),
            is_current=dr.is_current,
        )

    def _normalize_experience(self, exp: Experience, diff: NormalizationDiff) -> Experience:
        return Experience(
            company=exp.company,
            title=exp.title,
            location=exp.location,
            description=exp.description,
            duration=self._normalize_date_range(exp.duration),
        )

    def _normalize_education(self, edu: Education, diff: NormalizationDiff) -> Education:
        return Education(
            institution=edu.institution,
            degree=edu.degree,
            field_of_study=edu.field_of_study,
            duration=self._normalize_date_range(edu.duration),
            gpa=edu.gpa,
        )

    def _normalize_link(self, link: Link, diff: NormalizationDiff) -> Link:
        normalized_url = self._url.normalize(link.url, self._config)
        diff.record(f"link:{link.label}", link.url, normalized_url)
        return Link(url=normalized_url or link.url, label=link.label)
