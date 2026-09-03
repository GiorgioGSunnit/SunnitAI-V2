from datetime import date, datetime
from decimal import (
    ROUND_CEILING,
    ROUND_DOWN,
    ROUND_FLOOR,
    ROUND_HALF_EVEN,
    ROUND_HALF_UP,
    ROUND_UP,
    Decimal,
)
from typing import Any, Dict, List

# Legal rounding rules differ by domain: tax amounts truncate or round to
# the euro under specific norms, fees round half-up to the cent, day counts
# floor. A calculator declares its rule in YAML (`output.rounding`); the
# platform default stays half_up.
ROUNDING_MODES = {
    "half_up": ROUND_HALF_UP,
    "half_even": ROUND_HALF_EVEN,
    "down": ROUND_DOWN,       # truncate toward zero
    "up": ROUND_UP,
    "floor": ROUND_FLOOR,
    "ceiling": ROUND_CEILING,
}


def round_decimal(value: Decimal, places: int = 2, mode: str = "half_up") -> Decimal:
    """Single place rounding happens — every strategy rounds through here
    so rounding behavior stays consistent across the whole platform."""
    rounding = ROUNDING_MODES.get(mode)
    if rounding is None:
        raise ValueError(f"unknown rounding mode {mode!r}; valid: {', '.join(sorted(ROUNDING_MODES))}")
    quantum = Decimal("1").scaleb(-places)
    return value.quantize(quantum, rounding=rounding)


def round_output(value: Decimal, output_spec: Dict[str, Any]) -> Decimal:
    """Round a strategy's output per the calculator's declared policy:
    `output.round_to` (decimal places, default 2) and `output.rounding`
    (a ROUNDING_MODES key, default half_up)."""
    return round_decimal(
        value,
        int(output_spec.get("round_to", 2)),
        output_spec.get("rounding", "half_up"),
    )


def to_jsonable(value: Any) -> Any:
    """Convert Decimal-bearing structures to plain JSON-safe types for the API response.

    Decimals become strings (e.g. "616.44"), never floats: the engine's
    precision guarantee must survive serialization, and a float would
    reintroduce binary rounding noise at the exact boundary where results
    leave the module. Dates/datetimes become ISO-8601 strings.
    """
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value
