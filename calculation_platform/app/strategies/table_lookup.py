from decimal import Decimal
from typing import Any, Dict, List, Optional

from ..core.audit import AuditTrail
from ..core.errors import StrategyExecutionError
from ..core.result_builder import round_output
from ..resolvers.date_parameter_resolver import resolve_parameters
from .base import CalculationStrategy, StrategyOutcome


class TableLookupStrategy(CalculationStrategy):
    """Fixed amount looked up from a value-banded table, with an optional
    exemption escape hatch, an optional "indeterminable value" row, and an
    optional categorical multiplier — the shape of the contributo unificato
    (a fee per case-value band, increased by half on appeal, doubled in
    cassation).

    Expected formula keys:
      table_parameter        name of a declared parameter whose resolved
                             value is {"bands": [{"up_to": num|null,
                             "amount": num}, ...], "indeterminable_amount":
                             num} — bands sorted ascending, `up_to`
                             inclusive, the last band open-ended (null).
      amount_input           decimal input matched against the bands.
      indeterminable_input   optional boolean input selecting
                             indeterminable_amount instead of a band.
      multiplier             optional {"input": name, "values": {key: factor}}.
      zero_if                optional {"input": name, "warning": text} —
                             a true flag forces the result to 0.
    """

    def run(self, definition, inputs: Dict[str, Any], request) -> StrategyOutcome:
        cfg = definition.formula
        output_name = definition.output.get("name", "amount")
        resolution = resolve_parameters(definition, self.parameter_store, request)
        trail = AuditTrail()

        zero_if = cfg.get("zero_if")
        if zero_if and inputs.get(zero_if["input"]) is True:
            trail.record("exemption_applied", input=zero_if["input"])
            return StrategyOutcome(
                result={output_name: Decimal("0.00")},
                parameters_used=resolution.parameters_used(),
                date_resolution=resolution.date_resolution,
                steps=trail.steps,
                warnings=[zero_if.get(
                    "warning",
                    "Exemption flag set; amount forced to 0. Verify the exemption's legal basis applies.",
                )],
            )

        table = resolution.values[cfg["table_parameter"]]
        if not isinstance(table, dict) or "bands" not in table:
            raise StrategyExecutionError(
                f"table parameter {cfg['table_parameter']!r} must resolve to a dict with a 'bands' list",
                details={"calculator_id": definition.id},
            )

        assumptions: List[str] = []
        indeterminable_input = cfg.get("indeterminable_input")
        amount_input = cfg["amount_input"]
        base_amount: Optional[Decimal] = None

        if indeterminable_input and inputs.get(indeterminable_input) is True:
            raw = table.get("indeterminable_amount")
            if raw is None:
                raise StrategyExecutionError(
                    "table has no 'indeterminable_amount' but the indeterminable flag was set",
                    details={"calculator_id": definition.id},
                )
            base_amount = Decimal(str(raw))
            trail.record("indeterminable_value_row", amount=str(base_amount))
            if amount_input in inputs:
                assumptions.append(
                    f"'{amount_input}' was provided but ignored because "
                    f"'{indeterminable_input}' is true."
                )
        else:
            if amount_input not in inputs:
                raise StrategyExecutionError(
                    f"either {amount_input!r} or the {indeterminable_input!r} flag is required",
                    details={"calculator_id": definition.id, "missing_input": amount_input},
                )
            value = Decimal(str(inputs[amount_input]))
            for band in table["bands"]:
                up_to = band.get("up_to")
                if up_to is None or value <= Decimal(str(up_to)):
                    base_amount = Decimal(str(band["amount"]))
                    trail.record(
                        "band_matched",
                        value=str(value),
                        band_up_to=str(up_to) if up_to is not None else "open-ended",
                        amount=str(base_amount),
                    )
                    break
            if base_amount is None:
                raise StrategyExecutionError(
                    f"no band matched value {value} — the table's last band should have up_to: null",
                    details={"calculator_id": definition.id, "value": str(value)},
                )

        amount = base_amount
        multiplier_cfg = cfg.get("multiplier")
        if multiplier_cfg:
            key = str(inputs.get(multiplier_cfg["input"]))
            factors = multiplier_cfg["values"]
            if key not in factors:
                raise StrategyExecutionError(
                    f"{multiplier_cfg['input']!r} has no multiplier for {key!r}; "
                    f"valid values: {', '.join(sorted(factors))}",
                    details={"calculator_id": definition.id, "input": multiplier_cfg["input"], "value": key},
                )
            factor = Decimal(str(factors[key]))
            if factor != 1:
                amount = amount * factor
            trail.record("multiplier_applied", input=multiplier_cfg["input"], key=key, factor=str(factor))

        rounded = round_output(amount, definition.output)
        return StrategyOutcome(
            result={output_name: rounded},
            parameters_used=resolution.parameters_used(),
            date_resolution=resolution.date_resolution,
            steps=trail.steps,
            assumptions=assumptions,
        )
