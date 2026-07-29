"""Deterministic, human-readable narration of calculation audit steps."""

from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Dict, List


MAX_EXPLANATION_LINES = 12
_INTERNAL_KEYS = {"step", "type"}


def _format_decimal_it(value: Any, *, minimum_decimals: int = 0) -> str:
    """Format a decimal without ever passing through binary floating point."""
    try:
        raw = format(Decimal(str(value)), "f")
    except (InvalidOperation, ValueError):
        return str(value)

    sign = ""
    if raw.startswith("-"):
        sign, raw = "-", raw[1:]
    integer, separator, fraction = raw.partition(".")
    grouped = f"{int(integer):,}".replace(",", ".")
    if separator:
        fraction = fraction + ("0" * max(0, minimum_decimals - len(fraction)))
    elif minimum_decimals:
        fraction = "0" * minimum_decimals
    return f"{sign}{grouped}{',' + fraction if fraction else ''}"


def _format_percent_it(value: Any) -> str:
    try:
        percentage = Decimal(str(value)) * Decimal("100")
    except (InvalidOperation, ValueError):
        return f"{value}%"
    rendered = format(percentage, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return f"{rendered.replace('.', ',')}%"


def _label(key: Any) -> str:
    return str(key).replace("_", " ")


def _render_value(value: Any) -> str:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, bool):
        return "sì" if value else "no"
    if value is None:
        return "null"
    if isinstance(value, dict):
        return "{" + ", ".join(
            f"{_label(key)}: {_render_value(item)}" for key, item in value.items()
        ) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_render_value(item) for item in value) + "]"
    return str(value)


def _generic(step: Any) -> str:
    if not isinstance(step, dict):
        return _render_value(step)
    details = [
        f"{_label(key)}: {_render_value(value)}"
        for key, value in step.items()
        if key not in _INTERNAL_KEYS
    ]
    return "; ".join(details) if details else "Passaggio registrato."


def _progressive_bracket(step: Any) -> str:
    if not isinstance(step, dict) or step.get("type") != "bracket":
        return _generic(step)
    required = ("taxable_in_bracket", "rate", "tax_in_bracket")
    if any(step.get(key) is None for key in required):
        return _generic(step)
    taxable = _format_decimal_it(step["taxable_in_bracket"])
    rate = _format_percent_it(step["rate"])
    tax = _format_decimal_it(step["tax_in_bracket"], minimum_decimals=2)
    return f"{taxable} al {rate} = {tax}"


def _dm55(step: Any) -> str:
    if isinstance(step, dict) and step.get("note"):
        # The strategy records an authored, deterministic note for every
        # stage, including each accessory's amount and running subtotal.
        return str(step["note"])
    return _generic(step)


_STRATEGY_RENDERERS: Dict[str, Callable[[Any], str]] = {
    "progressive_brackets": _progressive_bracket,
    "dm55_fees": _dm55,
}


def narrate(steps: list, strategy: str) -> List[str]:
    """Render audit steps without changing them or recomputing any value."""
    if not steps:
        return []

    renderer = _STRATEGY_RENDERERS.get(strategy, _generic)
    lines = [renderer(step) for step in steps]
    if len(lines) <= MAX_EXPLANATION_LINES:
        return lines

    omitted = len(lines) - MAX_EXPLANATION_LINES
    return lines[:MAX_EXPLANATION_LINES] + [f"(+{omitted} passaggi nel report)"]
