"""ValidationEngine — runs all validators and aggregates results.

Design:
- Validators are registered in a list; order determines issue order.
- Each validator is isolated in a try/except — a buggy validator cannot
  crash the pipeline or suppress results from other validators.
- Issues are deduplicated on (field, severity, message) before the result
  is assembled, preventing duplicate warnings when multiple source records
  for the same candidate pass through the engine.
- is_valid = True when there are no error-severity issues.
  Warnings and info messages alone do not make a candidate invalid.
"""

from __future__ import annotations

from app.models.candidate import NormalizedCandidate, ValidationIssue, ValidationResult
from app.validators.base import BaseValidator
from app.validators.completeness_validator import CompletenessValidator
from app.validators.date_validator import DateValidator
from app.validators.email_validator import EmailValidator
from app.validators.phone_validator import PhoneValidator
from app.validators.required_fields import RequiredFieldsValidator
from app.validators.skills_validator import SkillsValidator
from app.validators.url_validator import URLValidator
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


class ValidationEngine:
    """Orchestrate all field-level validators.

    Instantiate once; the instance is stateless and thread-safe.

    Args:
        field_categories: Optional field category config forwarded to
            :class:`~app.validators.completeness_validator.CompletenessValidator`.
            When *None* the standard required/recommended/optional split is used.
        extra_validators: Additional :class:`BaseValidator` instances appended
            to the default set.
    """

    def __init__(
        self,
        field_categories=None,
        extra_validators: list[BaseValidator] | None = None,
    ) -> None:
        self._validators: list[BaseValidator] = [
            RequiredFieldsValidator(),
            EmailValidator(),
            PhoneValidator(),
            DateValidator(),
            SkillsValidator(),
            URLValidator(),
            CompletenessValidator(field_categories),
            *(extra_validators or []),
        ]

    def validate(self, candidate: NormalizedCandidate) -> ValidationResult:
        """Run all validators against *candidate* and return a combined result.

        Duplicate issues (same field + severity + message) are silently dropped
        so the caller receives each distinct finding exactly once.

        Args:
            candidate: A normalised candidate record.

        Returns:
            :class:`~app.models.candidate.ValidationResult` aggregating all
            issues from every registered validator.  Never raises.
        """
        seen: set[tuple[str, str, str]] = set()
        all_issues: list[ValidationIssue] = []

        for validator in self._validators:
            try:
                for issue in validator.validate(candidate):
                    key = (issue.field, issue.severity, issue.message)
                    if key not in seen:
                        seen.add(key)
                        all_issues.append(issue)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Validator %s raised unexpectedly: %s",
                    type(validator).__name__,
                    exc,
                )

        has_errors = any(i.severity == "error" for i in all_issues)
        result = ValidationResult(is_valid=not has_errors, issues=all_issues)

        logger.debug(
            "ValidationEngine: %d error(s), %d warning(s), %d info for source=%s name=%r",
            len(result.errors),
            len(result.warnings),
            len(result.info),
            candidate.source,
            candidate.name,
        )
        return result
