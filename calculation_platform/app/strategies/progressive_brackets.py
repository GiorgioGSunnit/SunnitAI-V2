from decimal import Decimal
from typing import Any, Dict, List

from ..core.audit import AuditTrail
from ..core.result_builder import round_decimal
from ..resolvers.date_parameter_resolver import resolve_parameters
from .base import CalculationStrategy, StrategyOutcome


class ProgressiveBracketsStrategy(CalculationStrategy):
    """Tiered/progressive tax calculation (e.g. IRPEF). The bracket table
    itself is a resolved parameter (date/tax_year versioned), not hardcoded
    here — a new year's brackets means a new parameter entry in YAML, not a
    code change."""

    def run(self, definition, inputs: Dict[str, Any], request) -> StrategyOutcome:
        base = inputs[definition.formula["base_input"]]
        brackets_param_name = definition.formula["brackets_parameter"]

        resolution = resolve_parameters(definition, self.parameter_store, request)
        brackets: List[Dict[str, Any]] = resolution.values[brackets_param_name]

        total = Decimal("0")
        trail = AuditTrail()
        previous_threshold = Decimal("0")
        for bracket in brackets:
            upper_raw = bracket["up_to"]
            upper = Decimal(str(upper_raw)) if upper_raw is not None else None
            rate = Decimal(str(bracket["rate"]))

            if upper is None:
                slice_amount = max(Decimal("0"), base - previous_threshold)
            else:
                slice_amount = max(Decimal("0"), min(base, upper) - previous_threshold)

            if slice_amount > 0:
                bracket_tax = slice_amount * rate
                total += bracket_tax
                trail.record(
                    "bracket",
                    bracket_up_to=str(upper) if upper is not None else "no limit",
                    rate=str(rate),
                    taxable_in_bracket=str(slice_amount),
                    tax_in_bracket=str(round_decimal(bracket_tax, 2)),
                )

            if upper is not None:
                previous_threshold = upper
                if base <= upper:
                    break

        output_name = definition.output.get("name", "result")
        rounded = round_decimal(total, definition.output.get("round_to", 2))
        return StrategyOutcome(
            result={output_name: rounded},
            parameters_used=resolution.parameters_used(),
            date_resolution=resolution.date_resolution,
            steps=trail.steps,
        )
