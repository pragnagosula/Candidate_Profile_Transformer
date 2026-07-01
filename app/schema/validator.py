"""Schema Validator.

Performs a final shape-check on a CandidateProfile before it is written
to disk.  Checks three things:

  1. Presence  — every field with include=True in the ProjectionConfig must
                 appear in profile.fields.

  2. Type      — source fields known to be lists (skills, experience,
                 education, links) must have list values in the output.
                 Source fields known to be scalars must not be lists/dicts
                 when they have a non-null value.

  3. Bounds    — if a ConfidenceReport is attached, overall_score and all
                 field-level scores must be in [0.0, 1.0].

Returns a ValidationResult using the same model as the rest of the pipeline.
Never raises — broken input becomes error-severity ValidationIssues.
"""

from __future__ import annotations

from app.config.loader import get_config
from app.config.models import ProjectionConfig
from app.models.candidate import (
    CandidateProfile,
    ValidationIssue,
    ValidationResult,
)
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

# Expected value category per MergedCandidate source field name.
_LIST_SOURCE_FIELDS = {"skills", "experience", "education", "links"}
_SCALAR_SOURCE_FIELDS = {"name", "email", "phone", "location", "summary"}


class SchemaValidator:
    """Validate the final shape of a CandidateProfile.

    Stateless between calls; safe to instantiate once and reuse.
    """

    def __init__(self, config: ProjectionConfig | None = None) -> None:
        self._config: ProjectionConfig = (
            config if config is not None else get_config().projection
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, profile: CandidateProfile) -> ValidationResult:
        """Validate the shape and content of *profile*.

        Args:
            profile: The fully projected output profile.

        Returns:
            A :class:`ValidationResult` describing any structural issues.
            Never raises.
        """
        issues: list[ValidationIssue] = []

        try:
            issues.extend(self._check_field_presence(profile))
            issues.extend(self._check_field_types(profile))
            issues.extend(self._check_confidence_bounds(profile))
        except Exception as exc:  # noqa: BLE001
            logger.error("SchemaValidator raised unexpectedly: %s", exc)
            issues.append(
                ValidationIssue(
                    field="_schema",
                    severity="error",
                    message=f"Schema validation crashed: {exc}",
                )
            )

        has_errors = any(i.severity == "error" for i in issues)
        result = ValidationResult(is_valid=not has_errors, issues=issues)

        logger.debug(
            "SchemaValidator: %d error(s), %d warning(s)",
            len(result.errors),
            len(result.warnings),
        )
        return result

    # ------------------------------------------------------------------
    # Checks
    # ------------------------------------------------------------------

    def _check_field_presence(
        self, profile: CandidateProfile
    ) -> list[ValidationIssue]:
        """Every included projected field must exist in profile.fields."""
        issues: list[ValidationIssue] = []
        for pf in self._config.fields:
            if not pf.include:
                continue
            if pf.output_name not in profile.fields:
                issues.append(
                    ValidationIssue(
                        field=pf.output_name,
                        severity="error",
                        message=(
                            f"Required output field '{pf.output_name}' is missing "
                            f"(source: '{pf.source}')"
                        ),
                    )
                )
        return issues

    def _check_field_types(
        self, profile: CandidateProfile
    ) -> list[ValidationIssue]:
        """Type-check values for known source-field categories."""
        issues: list[ValidationIssue] = []
        for pf in self._config.fields:
            if not pf.include or pf.output_name not in profile.fields:
                continue

            value = profile.fields[pf.output_name]

            if pf.source in _LIST_SOURCE_FIELDS:
                if value is not None and not isinstance(value, list):
                    issues.append(
                        ValidationIssue(
                            field=pf.output_name,
                            severity="error",
                            message=(
                                f"Field '{pf.output_name}' must be a list "
                                f"(source: '{pf.source}'), got {type(value).__name__}"
                            ),
                            value=value,
                        )
                    )

            elif pf.source in _SCALAR_SOURCE_FIELDS:
                if isinstance(value, (list, dict)):
                    issues.append(
                        ValidationIssue(
                            field=pf.output_name,
                            severity="warning",
                            message=(
                                f"Field '{pf.output_name}' is a scalar field "
                                f"(source: '{pf.source}') but received "
                                f"{type(value).__name__}"
                            ),
                            value=value,
                        )
                    )

        return issues

    def _check_confidence_bounds(
        self, profile: CandidateProfile
    ) -> list[ValidationIssue]:
        """overall_score and all field scores must be in [0.0, 1.0]."""
        if profile.confidence is None:
            return []

        issues: list[ValidationIssue] = []
        conf = profile.confidence

        if not 0.0 <= conf.overall_score <= 1.0:
            issues.append(
                ValidationIssue(
                    field="confidence.overall_score",
                    severity="error",
                    message=(
                        f"overall_score {conf.overall_score} is outside [0.0, 1.0]"
                    ),
                    value=conf.overall_score,
                )
            )

        for fs in conf.field_scores:
            if not 0.0 <= fs.score <= 1.0:
                issues.append(
                    ValidationIssue(
                        field=f"confidence.field_scores[{fs.field_name}]",
                        severity="error",
                        message=(
                            f"Field score for '{fs.field_name}' is {fs.score}, "
                            f"outside [0.0, 1.0]"
                        ),
                        value=fs.score,
                    )
                )

        return issues
