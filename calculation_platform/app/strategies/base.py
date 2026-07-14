from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List


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

    def __init__(self, parameter_store):
        self.parameter_store = parameter_store

    @abstractmethod
    def run(self, definition, inputs: Dict[str, Any], request) -> StrategyOutcome:
        ...
