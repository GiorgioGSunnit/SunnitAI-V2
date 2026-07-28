from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, ClassVar, Dict, List, Optional


def data_quality_warning(pv, describe: str) -> Optional[str]:
    """A single data-quality warning for a resolved parameter value, or None.

    Distinguishes two very different states so the message is never
    misleading:
      - `placeholder` → a synthetic stand-in value: the result is NOT usable;
      - `verified is False` (but not a placeholder) → a real value already
        loaded from the source but awaiting the final human sign-off against
        the primary artifact (Gazzetta Ufficiale / ISTAT): usable with care.
    A value with `verified` true (or unset) and no placeholder yields None.
    `describe` is a short human label for the value (e.g. the FOI month or
    the parameter name)."""
    if pv.placeholder:
        return (
            f"{describe}: valore SEGNAPOSTO sintetico, non verificato contro la "
            "fonte ufficiale; il risultato non e utilizzabile operativamente."
        )
    if pv.verified is False:
        return (
            f"{describe}: valore reale non ancora verificato in via definitiva "
            "contro la fonte primaria (G.U./ISTAT); verificare prima dell'uso "
            "operativo."
        )
    return None


@dataclass
class StrategyOutcome:
    result: Dict[str, Any]
    derived_values: Dict[str, Any] = field(default_factory=dict)
    parameters_used: Dict[str, Any] = field(default_factory=dict)
    date_resolution: Any = None
    steps: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)


class CalculationStrategy(ABC):
    """A pluggable calculation shape (expression, progressive brackets,
    percentage-of-base, etc). New calculators reuse an existing strategy by
    declaring it in YAML; a genuinely new *shape* of calculation gets a new
    strategy class registered in strategies/__init__.py — existing
    calculators and the engine never need to change."""

    requires_period: ClassVar[bool] = False

    def __init__(self, parameter_store):
        self.parameter_store = parameter_store
        # Set by the engine to the request's ValidatedInputs before `run`.
        # `run` receives only the coerced values, which no longer say which
        # of them the caller actually supplied and which the platform
        # defaulted — a distinction a strategy needs to report the quality
        # of its own output (see ComparatorStrategy). The strategy instance
        # is built fresh per request, so this stays request-scoped.
        self.validated_inputs: Optional[Any] = None

    @abstractmethod
    def run(self, definition, inputs: Dict[str, Any], request) -> StrategyOutcome:
        ...
