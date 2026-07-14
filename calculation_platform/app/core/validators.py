from dataclasses import dataclass, field
from datetime import date as date_cls
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List

from ..schemas.calculator_definition import CalculatorDefinition
from .errors import InputValidationError

# Kept as an alias so any external code still catching ValidationError
# keeps working — InputValidationError is the structured replacement.
ValidationError = InputValidationError

# Strict boolean parsing: bool("false") is True in Python, and a truthy
# string silently flipping a legal regime flag (an exemption, cedolare
# secca, urgency) is exactly the kind of wrong-number this platform exists
# to prevent. Accepted string spellings cover JSON, form posts, and the
# Italian-language planner ("sì"/"no").
_TRUE_STRINGS = {"true", "1", "yes", "si", "sì"}
_FALSE_STRINGS = {"false", "0", "no"}


def _coerce_boolean(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, int) and v in (0, 1):
        return bool(v)
    if isinstance(v, str):
        norm = v.strip().lower()
        if norm in _TRUE_STRINGS:
            return True
        if norm in _FALSE_STRINGS:
            return False
    raise ValueError(
        f"not a recognized boolean: {v!r} (use true/false, 1/0, sì/no)"
    )


_TYPE_COERCERS = {
    "decimal": lambda v: Decimal(str(v)),
    "integer": lambda v: int(v),
    "boolean": _coerce_boolean,
    "string": lambda v: str(v),
    "date": lambda v: v if isinstance(v, date_cls) else date_cls.fromisoformat(str(v)),
}


@dataclass
class ValidatedInputs:
    values: Dict[str, Any] = field(default_factory=dict)
    # Human-readable notes about defaults silently applied — surfaced on
    # CalculationResult.assumptions so a caller can see what was assumed.
    assumptions: List[str] = field(default_factory=list)


def validate_inputs(definition: CalculatorDefinition, raw_inputs: Dict[str, Any]) -> ValidatedInputs:
    """Coerce raw request inputs to the types declared in the calculator's
    definition, applying declared defaults (and recording that as an
    assumption) and reporting every missing required input by name."""
    result = ValidatedInputs()
    missing = []
    for spec in definition.inputs:
        used_default = False
        if spec.name in raw_inputs:
            raw_value = raw_inputs[spec.name]
        elif spec.default is not None:
            raw_value = spec.default
            used_default = True
        elif spec.required:
            missing.append(spec.name)
            continue
        else:
            continue

        coercer = _TYPE_COERCERS.get(spec.type)
        if coercer is None:
            raise InputValidationError(
                f"Unknown input type {spec.type!r} for {spec.name!r}",
                details={"input": spec.name, "type": spec.type},
            )
        try:
            coerced = coercer(raw_value)
        except (InvalidOperation, ValueError, TypeError) as e:
            raise InputValidationError(
                f"Invalid value for {spec.name!r}: {raw_value!r} ({e})",
                details={"input": spec.name, "value": raw_value},
            ) from e

        if spec.min_value is not None and coerced < coercer(spec.min_value):
            raise InputValidationError(
                f"{spec.name} must be >= {spec.min_value}, got {coerced}",
                details={"input": spec.name, "value": str(coerced), "min_value": spec.min_value},
            )
        if spec.max_value is not None and coerced > coercer(spec.max_value):
            raise InputValidationError(
                f"{spec.name} must be <= {spec.max_value}, got {coerced}",
                details={"input": spec.name, "value": str(coerced), "max_value": spec.max_value},
            )
        result.values[spec.name] = coerced

        if used_default:
            result.assumptions.append(f"{spec.name} not provided; assumed default {raw_value!r}")

    if missing:
        raise InputValidationError(
            f"Missing required input(s): {', '.join(missing)}",
            details={"missing_inputs": missing},
        )
    return result
