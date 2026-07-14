"""Deterministic value-extraction primitives for the simulated LLM layer.

The real integration will have an LLM read the user's sentence and emit a
structured tool call (calculator_id + named inputs). planner.py mimics
that contract; this module supplies its raw extraction with fully
deterministic, deliberately naive heuristics:

  - numbers (Italian formats), a tax-year candidate, ISO date periods,
    and yes/no words are pulled out of free text;
  - bind_values() assigns them to a calculator's still-missing required
    inputs IN DECLARATION ORDER.

The heuristics are a test fixture, not a feature: they handle clean demo
sentences ("reddito di 42000 euro nel 2026") and intentionally nothing
more. Every limitation here is exactly the work the real LLM will take
over — none of this module should ever run in a production path.
"""

import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional

from app.schemas.calculator_definition import CalculatorDefinition

_NUMBER_RE = re.compile(r"-?\d[\d.]*(?:,\d+)?")
_ISO_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
_YES_WORDS = {"si", "sì", "yes", "true", "vero"}
_NO_WORDS = {"no", "false", "falso"}


def parse_number(token: str) -> Decimal:
    """'42.000,50' -> 42000.50; '42.000' -> 42000; '0,5' -> 0.5."""
    s = token
    if "," in s:
        s = s.replace(".", "").replace(",", ".")
    elif s.count(".") == 1 and len(s.split(".")[1]) == 3:
        s = s.replace(".", "")
    elif s.count(".") > 1:
        s = s.replace(".", "")
    return Decimal(s)


def extract_values(text: str) -> Dict[str, Any]:
    """All the raw material the binder works with: dates, a tax-year
    candidate, remaining numbers (in order of appearance), and a yes/no."""
    lowered = text.lower()
    dates = _ISO_DATE_RE.findall(text)
    remainder = _ISO_DATE_RE.sub(" ", text)

    numbers: List[Decimal] = []
    tax_year: Optional[int] = None
    for token in _NUMBER_RE.findall(remainder):
        value = parse_number(token)
        if value == int(value) and 2000 <= int(value) <= 2099:
            tax_year = int(value)
        else:
            numbers.append(value)

    words = set(re.findall(r"[a-zà-ù]+", text.lower()))
    boolean: Optional[bool] = None
    if words & _YES_WORDS:
        boolean = True
    elif words & _NO_WORDS:
        boolean = False

    period = None
    if len(dates) >= 2:
        start, end = sorted(dates[:2])
        period = {"start_date": start, "end_date": end}

    amount_frequency = None
    if any(marker in lowered for marker in ("al mese", "mensile", "mensili", "mese")):
        amount_frequency = "monthly"
    elif any(marker in lowered for marker in ("annuo", "annua", "annuale", "all'anno", "l'anno")):
        amount_frequency = "annual"

    boolean_hints: Dict[str, bool] = {}
    if "prima registrazione" in lowered or "registrazione iniziale" in lowered:
        boolean_hints["first_registration"] = True
    if "cedolare secca" in lowered:
        boolean_hints["cedolare_secca"] = True

    return {
        "numbers": numbers,
        "tax_year": tax_year,
        "period": period,
        "boolean": boolean,
        "amount_frequency": amount_frequency,
        "boolean_hints": boolean_hints,
    }


@dataclass
class SimulatedToolCall:
    """What the real LLM would emit: one structured /calculate request."""

    calculator_id: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    tax_year: Optional[int] = None
    period: Optional[Dict[str, str]] = None


def bind_values(definition: CalculatorDefinition, inputs: Dict[str, Any], values: Dict[str, Any]) -> None:
    """Fill still-missing required inputs from extracted values, in the
    order the definition declares them. Naive on purpose — see module
    docstring."""
    numbers = list(values["numbers"])
    for spec in definition.inputs:
        if spec.name in inputs:
            continue
        if spec.type in ("decimal", "integer") and numbers and spec.required:
            number = numbers.pop(0)
            if spec.name == "annual_rent" and values.get("amount_frequency") == "monthly":
                number = number * Decimal("12")
            if spec.unit == "rate" and number > 1:
                number = number / Decimal("100")
            inputs[spec.name] = int(number) if spec.type == "integer" else number
        elif spec.type == "boolean" and spec.name in values.get("boolean_hints", {}):
            inputs[spec.name] = values["boolean_hints"][spec.name]
        elif spec.type == "boolean" and spec.required and values["boolean"] is not None:
            inputs[spec.name] = values["boolean"]
