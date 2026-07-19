"""Structured platform error hierarchy.

Every failure raised anywhere in the engine — registry lookup, YAML
definition validation, input validation, parameter resolution, strategy
execution — is a PlatformError subclass carrying a stable `code` and
machine-readable `details`. The engine catches PlatformError uniformly and
turns it into a schemas.error.CalculationError, so callers never have to
parse free-text error strings to know what went wrong.
"""

from typing import Any, Dict, Optional


class PlatformError(Exception):
    code = "platform_error"

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class CalculatorNotFoundError(PlatformError):
    code = "calculator_not_found"


class CalculatorNotApplicableError(PlatformError):
    code = "calculator_not_applicable"


class DefinitionValidationError(PlatformError):
    """Raised at registry load time when a formula-pack YAML file is
    structurally invalid — e.g. references an unknown variable in an
    expression, or an unknown strategy."""

    code = "definition_invalid"


class InputValidationError(PlatformError):
    code = "input_invalid"


class ParameterResolutionError(PlatformError):
    code = "parameter_unresolved"


class StrategyExecutionError(PlatformError):
    code = "strategy_execution_failed"
