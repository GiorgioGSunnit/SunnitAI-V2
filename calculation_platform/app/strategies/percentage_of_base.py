from decimal import Decimal
from typing import Any, Dict

from ..core.audit import AuditTrail
from ..core.result_builder import round_decimal
from ..core.safe_evaluator import safe_eval
from ..resolvers.date_parameter_resolver import resolve_parameters
from .base import CalculationStrategy, StrategyOutcome


class PercentageOfBaseStrategy(CalculationStrategy):
    """base * rate, with an optional minimum and an optional "zero_if"
    escape hatch for an alternative regime that replaces this tax entirely
    (e.g. cedolare secca replacing registration tax) — used by
    legal_it.registration_tax_leases."""

    def run(self, definition, inputs: Dict[str, Any], request) -> StrategyOutcome:
        formula_cfg = definition.formula
        resolution = resolve_parameters(definition, self.parameter_store, request)
        variables: Dict[str, Any] = dict(inputs)
        variables.update(resolution.values)

        zero_if = formula_cfg.get("zero_if")
        output_name = definition.output.get("name", "result")
        if zero_if and bool(inputs.get(zero_if["input"])) == zero_if.get("equals", True):
            trail = AuditTrail()
            trail.record("zero_if_triggered", input=zero_if["input"])
            return StrategyOutcome(
                result={output_name: Decimal("0.00")},
                parameters_used=resolution.parameters_used(),
                date_resolution=resolution.date_resolution,
                steps=trail.steps,
                warnings=[formula_cfg.get(
                    "zero_if_warning",
                    "Formula not applicable under the selected regime; result forced to 0.",
                )],
            )

        base_value = safe_eval(formula_cfg["base"], variables)
        rate = resolution.values[formula_cfg["rate_parameter"]]
        tax = base_value * rate

        trail = AuditTrail()
        trail.record("base", expression=formula_cfg["base"], value=str(base_value))
        trail.record("tax_before_minimum", value=str(tax))
        warnings = []

        minimum_name = formula_cfg.get("minimum_parameter")
        minimum_flag_name = formula_cfg.get("apply_minimum_if")
        if minimum_name and minimum_flag_name:
            if bool(inputs.get(minimum_flag_name)):
                minimum_value = resolution.values[minimum_name]
                if tax < minimum_value:
                    trail.record("minimum_applied", minimum=str(minimum_value))
                    tax = minimum_value
            else:
                warnings.append(formula_cfg.get(
                    "no_minimum_warning",
                    "Minimum threshold not applied; separate rules may apply.",
                ))

        rounded = round_decimal(tax, definition.output.get("round_to", 2))
        return StrategyOutcome(
            result={output_name: rounded},
            parameters_used=resolution.parameters_used(),
            date_resolution=resolution.date_resolution,
            steps=trail.steps,
            warnings=warnings,
        )
