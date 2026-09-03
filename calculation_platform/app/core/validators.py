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
_MAX_EXACT_FLOAT_INTEGER = 2**53


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


def _coerce_string_list(v: Any) -> List[str]:
    if isinstance(v, str):
        items = [part.strip() for part in v.split(",")]
    elif isinstance(v, (list, tuple)):
        items = [str(part).strip() for part in v]
    else:
        raise ValueError(f"not a list of strings: {v!r}")
    items = [item for item in items if item]
    if not items:
        raise ValueError("the list must contain at least one element")
    return items


def _coerce_decimal(v: Any) -> Decimal:
    # NaN/Infinity parse as valid Decimals but poison every later
    # comparison and arithmetic step with raw InvalidOperation errors —
    # reject them at the boundary like any other invalid value.
    value = Decimal(str(v))
    if not value.is_finite():
        raise ValueError(f"not a finite number: {v!r}")
    return value


def _coerce_integer(v: Any) -> int:
    """Accept only values that ARE whole numbers; never truncate one.

    `int(1.9)` is 1, and an integer input is how this platform spells a
    count of days, months or circumstances: a procedural deadline quietly
    becoming one day shorter is a wrong legal answer delivered with no
    warning at all. A caller who means 2 can say 2, 2.0 or "2"; one who
    writes 1.9 has made an error that only they can resolve.

    Routed through _coerce_decimal so the check is exact (float 0.1
    arithmetic never enters it) and so non-finite values are rejected
    there rather than escaping as OverflowError from int(float("inf")),
    which _coerce_scalar does not catch.
    """
    # bool is an int subclass; True silently becoming 1 is the same class
    # of type confusion the boolean coercer exists to prevent.
    if isinstance(v, bool):
        raise ValueError(f"not a whole number: {v!r} (use a number, not a boolean)")
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        if not v.is_integer():
            raise ValueError(
                f"not a whole number: {v!r} (a fractional value would be truncated; "
                "use a whole number)"
            )
        if abs(v) >= _MAX_EXACT_FLOAT_INTEGER:
            raise ValueError(
                f"integer-valued float {v!r} is outside the exactly representable range; "
                "send it as an integer or numeric string"
            )
    value = _coerce_decimal(v)
    if value != value.to_integral_value():
        raise ValueError(
            f"not a whole number: {v!r} (a fractional value would be truncated; "
            "use a whole number)"
        )
    return int(value)


_TYPE_COERCERS = {
    "decimal": _coerce_decimal,
    "integer": _coerce_integer,
    "boolean": _coerce_boolean,
    "string": lambda v: str(v),
    "date": lambda v: v if isinstance(v, date_cls) else date_cls.fromisoformat(str(v)),
    "string_list": _coerce_string_list,
}


@dataclass
class ValidatedInputs:
    values: Dict[str, Any] = field(default_factory=dict)
    # Human-readable notes about defaults silently applied — surfaced on
    # CalculationResult.assumptions so a caller can see what was assumed.
    assumptions: List[str] = field(default_factory=list)
    # The same information machine-readable: {"path", "value"} per applied
    # default, where `path` addresses the input exactly as the request
    # spells it ("storico_sinistri", "polizze[0].franchigia") and `value`
    # is the exact decimal/boolean/string the platform substituted. Prose
    # cannot be branched on; a comparator has to know *which* field was
    # assumed to decide whether its scoring is provisional.
    defaults_applied: List[Dict[str, Any]] = field(default_factory=list)

    def record_default(self, path: str, value: Any) -> None:
        self.assumptions.append(f"{path} not provided; assumed default {value!r}")
        self.defaults_applied.append({"path": path, "value": _default_repr(value)})


def _default_repr(value: Any) -> Any:
    """JSON-safe rendering of a defaulted value.

    Booleans stay booleans — collapsing `false` into the string "false"
    would make an explicitly-declared false default indistinguishable from
    a string field defaulted to the word. Everything else (numbers,
    dates, lists) becomes its exact string form, matching the platform's
    Decimal-as-string serialization contract.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (list, tuple)):
        return [_default_repr(item) for item in value]
    if isinstance(value, Decimal):
        return format(value, "f")
    return str(value)


def _coerce_scalar(spec, raw_value: Any, label: str) -> Any:
    """Coerce one scalar value to spec.type and enforce declared bounds.
    `label` names the input in error messages (may carry an item index for
    object_list fields, e.g. "candidates[1].premio_annuo")."""
    coercer = _TYPE_COERCERS.get(spec.type)
    if coercer is None:
        raise InputValidationError(
            f"Unknown input type {spec.type!r} for {label!r}",
            details={"input": label, "type": spec.type},
        )
    try:
        coerced = coercer(raw_value)
    except (InvalidOperation, ValueError, TypeError) as e:
        raise InputValidationError(
            f"Invalid value for {label!r}: {raw_value!r} ({e})",
            details={"input": label, "value": raw_value, "expected": missing_input_spec(spec)},
        ) from e

    if spec.min_value is not None and coerced < coercer(spec.min_value):
        raise InputValidationError(
            f"{label} must be >= {spec.min_value}, got {coerced}",
            details={"input": label, "value": str(coerced), "min_value": spec.min_value},
        )
    if spec.max_value is not None and coerced > coercer(spec.max_value):
        raise InputValidationError(
            f"{label} must be <= {spec.max_value}, got {coerced}",
            details={"input": label, "value": str(coerced), "max_value": spec.max_value},
        )
    return coerced


def _coerce_object_list(spec, raw_value: Any, result: "ValidatedInputs") -> List[Dict[str, Any]]:
    """Validate a list-of-objects input: each item is a dict validated
    against spec.item_fields with the same required/default/bounds
    semantics as top-level inputs."""
    if not isinstance(raw_value, (list, tuple)):
        raise InputValidationError(
            f"{spec.name} must be a list of objects, got {type(raw_value).__name__}",
            details={"input": spec.name, "value": raw_value},
        )
    items = list(raw_value)
    min_items = spec.min_items if spec.min_items is not None else 1
    if len(items) < min_items:
        raise InputValidationError(
            f"{spec.name} needs at least {min_items} item(s), got {len(items)}",
            details={"input": spec.name, "min_items": min_items, "items_given": len(items)},
        )

    declared = {field_spec.name for field_spec in spec.item_fields or []}
    validated_items: List[Dict[str, Any]] = []
    for index, raw_item in enumerate(items):
        if not isinstance(raw_item, dict):
            raise InputValidationError(
                f"{spec.name}[{index}] must be an object, got {type(raw_item).__name__}",
                details={"input": f"{spec.name}[{index}]", "value": raw_item},
            )
        # An undeclared key is more likely a typo of a declared field than
        # noise (the generated tool schemas declare additionalProperties:
        # false) — silently dropping it would silently change the score.
        unknown = sorted(set(raw_item) - declared)
        if unknown:
            raise InputValidationError(
                f"{spec.name}[{index}] has undeclared field(s): {', '.join(unknown)}; "
                f"valid fields: {', '.join(sorted(declared))}",
                details={"input": spec.name, "item_index": index, "unknown_fields": unknown},
            )
        item_values: Dict[str, Any] = {}
        item_missing = []
        for field_spec in spec.item_fields or []:
            label = f"{spec.name}[{index}].{field_spec.name}"
            used_default = False
            if field_spec.name in raw_item:
                raw_field = raw_item[field_spec.name]
            elif field_spec.default is not None:
                raw_field = field_spec.default
                used_default = True
            elif field_spec.required:
                item_missing.append(field_spec.name)
                continue
            else:
                continue
            item_values[field_spec.name] = _coerce_scalar(field_spec, raw_field, label)
            if used_default:
                result.record_default(label, raw_field)
        if item_missing:
            missing_set = set(item_missing)
            raise InputValidationError(
                f"{spec.name}[{index}]: missing required field(s): {', '.join(item_missing)}",
                details={
                    "input": spec.name,
                    "item_index": index,
                    "missing_inputs": [f"{spec.name}[{index}].{name}" for name in item_missing],
                    "missing": [
                        missing_input_spec(field_spec)
                        for field_spec in (spec.item_fields or [])
                        if field_spec.name in missing_set
                    ],
                },
            )
        validated_items.append(item_values)
    return validated_items


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

        if spec.type == "object_list":
            result.values[spec.name] = _coerce_object_list(spec, raw_value, result)
            if used_default:
                result.record_default(spec.name, raw_value)
            continue

        result.values[spec.name] = _coerce_scalar(spec, raw_value, spec.name)

        if used_default:
            result.record_default(spec.name, raw_value)

    if missing:
        missing_set = set(missing)
        raise InputValidationError(
            f"Missing required input(s): {', '.join(missing)}",
            details={
                "missing_inputs": missing,
                # Machine-actionable spec of each missing input, so an LLM
                # (or any client) can formulate the clarification question
                # itself instead of parsing Italian prose.
                "missing": [
                    missing_input_spec(spec)
                    for spec in definition.inputs
                    if spec.name in missing_set
                ],
            },
        )
    return result


def missing_input_spec(spec) -> Dict[str, Any]:
    """A JSON-safe description of one missing input for clarification
    payloads: name, declared type, and whatever constraints the pack
    declares (unit, bounds, description)."""
    entry: Dict[str, Any] = {"name": spec.name, "type": spec.type, "required": spec.required}
    if spec.description:
        entry["description"] = spec.description
    if spec.unit:
        entry["unit"] = spec.unit
    if spec.min_value is not None:
        entry["min_value"] = spec.min_value
    if spec.max_value is not None:
        entry["max_value"] = spec.max_value
    if getattr(spec, "item_fields", None):
        entry["item_fields"] = [missing_input_spec(f) for f in spec.item_fields]
    if getattr(spec, "min_items", None) is not None:
        entry["min_items"] = spec.min_items
    return entry
