from decimal import Decimal
from typing import Any, Dict

from ..core.audit import AuditTrail
from ..core.errors import StrategyExecutionError
from .base import CalculationStrategy, StrategyOutcome

_COMPARATORS = {
    "equals": lambda actual, expected: actual == expected,
    "not_equals": lambda actual, expected: actual != expected,
    "greater_than": lambda actual, expected: Decimal(str(actual)) > Decimal(str(expected)),
    "less_than": lambda actual, expected: Decimal(str(actual)) < Decimal(str(expected)),
    "at_least": lambda actual, expected: Decimal(str(actual)) >= Decimal(str(expected)),
    "at_most": lambda actual, expected: Decimal(str(actual)) <= Decimal(str(expected)),
}


class DecisionTableStrategy(CalculationStrategy):
    """Evaluates an ordered list of {when, value} rules and returns the
    first match's value. `when` is a simple dict condition
    ({"input": name, "equals": value}) — or one of the other comparators
    above for numeric thresholds — rather than a full expression, kept
    deliberately small; a rule with no `when` always matches (use it last,
    as a default). Intentionally minimal — no current calculator needs
    more than this; extend the comparator table if a real need arises
    rather than reaching for a general rule-engine.
    """

    def run(self, definition, inputs: Dict[str, Any], request) -> StrategyOutcome:
        rules = definition.formula.get("rules", [])
        trail = AuditTrail()
        chosen = None
        for rule in rules:
            condition = rule.get("when")
            matched = condition is None or self._matches(condition, inputs)
            trail.record("rule_evaluated", condition=condition, matched=matched)
            if matched:
                chosen = rule["value"]
                break
        if chosen is None:
            raise StrategyExecutionError(
                "No decision_table rule matched and no default rule was provided",
                details={"calculator_id": definition.id, "rules_evaluated": len(rules)},
            )

        output_name = definition.output.get("name", "result")
        return StrategyOutcome(result={output_name: chosen}, steps=trail.steps)

    @staticmethod
    def _matches(condition: Dict[str, Any], inputs: Dict[str, Any]) -> bool:
        actual = inputs.get(condition["input"])
        for comparator_name, comparator in _COMPARATORS.items():
            if comparator_name in condition:
                return comparator(actual, condition[comparator_name])
        raise StrategyExecutionError(
            f"decision_table rule has no recognized comparator: {condition!r}",
            details={"condition": condition, "valid_comparators": sorted(_COMPARATORS)},
        )
