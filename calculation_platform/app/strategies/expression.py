from decimal import Decimal
from typing import Any, Dict

from ..core.audit import AuditTrail
from ..core.result_builder import round_decimal
from ..core.safe_evaluator import safe_eval
from ..resolvers.date_parameter_resolver import resolve_parameters
from .base import CalculationStrategy, StrategyOutcome


def _num(value: Decimal) -> str:
    return str(value)


class ExpressionStrategy(CalculationStrategy):
    """Evaluates declared derived_variables in order, then a final formula
    expression — covers invoice totals, VAT, discounts, loan payments, and
    any other formula expressible as plain arithmetic over named variables.

    Supports two optional escape hatches, both declared in `formula`:
    - `zero_case`: for formulas that would divide by zero under a
      particular input (e.g. a zero-interest loan) — if the named
      variable equals the given value, a fallback expression runs instead.
    - output.min / output.max on the calculator's `output` block: clamps
      the final result into a sane range (e.g. a result can never be
      negative) after rounding.
    """

    def run(self, definition, inputs: Dict[str, Any], request) -> StrategyOutcome:
        variables: Dict[str, Any] = dict(inputs)
        resolution = resolve_parameters(definition, self.parameter_store, request)
        variables.update(resolution.values)

        trail = AuditTrail()
        for name, expr in definition.derived_variables.items():
            value = safe_eval(expr, variables)
            variables[name] = value
            trail.record("derived_variable", variable=name, expression=expr, value=_num(value))

        formula_cfg = definition.formula
        expr = formula_cfg["expression"]
        zero_case = formula_cfg.get("zero_case")
        if zero_case and variables.get(zero_case["when_variable"]) == Decimal(str(zero_case.get("equals", 0))):
            expr = zero_case["expression"]
            trail.record("zero_case_triggered", when_variable=zero_case["when_variable"])

        raw_result = safe_eval(expr, variables)
        round_to = definition.output.get("round_to", 2)
        rounded = round_decimal(raw_result, round_to)
        output_name = definition.output.get("name", "result")
        trail.record("formula", variable=output_name, expression=expr, value=_num(rounded))

        output_min = definition.output.get("min")
        output_max = definition.output.get("max")
        if output_min is not None and rounded < Decimal(str(output_min)):
            trail.record("clamped_to_minimum", minimum=str(output_min), pre_clamp_value=_num(rounded))
            rounded = round_decimal(Decimal(str(output_min)), round_to)
        if output_max is not None and rounded > Decimal(str(output_max)):
            trail.record("clamped_to_maximum", maximum=str(output_max), pre_clamp_value=_num(rounded))
            rounded = round_decimal(Decimal(str(output_max)), round_to)

        return StrategyOutcome(
            result={output_name: rounded},
            derived_values={k: variables[k] for k in definition.derived_variables},
            parameters_used=resolution.parameters_used(),
            date_resolution=resolution.date_resolution,
            steps=trail.steps,
        )
