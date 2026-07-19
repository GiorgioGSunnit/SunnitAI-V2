from datetime import date
from decimal import Decimal
from typing import Any, Dict

from ..core.audit import AuditTrail
from ..core.errors import ParameterResolutionError, StrategyExecutionError
from ..core.result_builder import round_output
from ..schemas.parameter_value import ParameterValue
from ..schemas.resolved_parameter import ResolvedParameter
from .base import CalculationStrategy, StrategyOutcome


def resolve_foi_index(parameter_store, parameter_id: str, as_of: date) -> ParameterValue:
    """The FOI index for the calendar month of `as_of`, as a structured
    failure when the month is missing from the series — never interpolated,
    never defaulted."""
    try:
        return parameter_store.resolve_monthly(parameter_id, as_of)
    except KeyError as e:
        raise ParameterResolutionError(
            str(e),
            details={
                "parameter_id": parameter_id,
                "year": as_of.year,
                "month": as_of.month,
            },
        ) from e


def foi_resolved_parameter(name: str, pv: ParameterValue) -> ResolvedParameter:
    return ResolvedParameter(
        name=name, value=Decimal(str(pv.value)), origin="parameter_store",
        parameter_id=pv.parameter_id, source=pv.source,
        effective_from=pv.effective_from.isoformat(),
        effective_to=pv.effective_to.isoformat() if pv.effective_to else None,
        official=pv.official, last_verified_at=pv.last_verified_at,
        citations=pv.citations,
    )


def placeholder_warnings(*values: ParameterValue) -> list:
    warnings = []
    for pv in values:
        if pv.placeholder or pv.verified is False:
            warnings.append(
                f"L'indice FOI {pv.effective_from.year}-{pv.effective_from.month:02d} "
                "e un valore SEGNAPOSTO non verificato contro la serie ufficiale ISTAT; "
                "il risultato non e utilizzabile operativamente."
            )
    return warnings


class FoiRevaluationStrategy(CalculationStrategy):
    """Rivalutazione monetaria on the ISTAT FOI (senza tabacchi) monthly
    series: importo × FOI(mese di data_finale) / FOI(mese di data_iniziale).

    Final-month convention: the index of the calendar month each date falls
    in — the same convention as ISTAT's own calculator (rivaluta.istat.it),
    so a human can verify the coefficient there. The coefficient is computed
    in Decimal at full precision; quantization happens only on the final
    amount per the pack's declared rounding policy.
    """

    def run(self, definition, inputs: Dict[str, Any], request) -> StrategyOutcome:
        amount = inputs[definition.formula.get("amount_input", "importo")]
        start = inputs[definition.formula.get("start_date_input", "data_iniziale")]
        end = inputs[definition.formula.get("end_date_input", "data_finale")]
        index_parameter_id = definition.formula["index_parameter_id"]

        if start > end:
            raise StrategyExecutionError(
                "data_iniziale must not be after data_finale",
                details={"data_iniziale": start.isoformat(), "data_finale": end.isoformat()},
            )

        pv_start = resolve_foi_index(self.parameter_store, index_parameter_id, start)
        pv_end = resolve_foi_index(self.parameter_store, index_parameter_id, end)
        index_start = Decimal(str(pv_start.value))
        index_end = Decimal(str(pv_end.value))
        coefficient = index_end / index_start
        revalued = amount * coefficient

        trail = AuditTrail()
        trail.record(
            "revaluation_coefficient",
            index_initial_month=f"{start.year}-{start.month:02d}",
            index_initial=str(index_start),
            index_final_month=f"{end.year}-{end.month:02d}",
            index_final=str(index_end),
            coefficient=str(coefficient),
            note=(
                f"Coefficiente di rivalutazione = FOI({end.year}-{end.month:02d}) / "
                f"FOI({start.year}-{start.month:02d}) = {index_end} / {index_start} "
                "(convenzione del mese finale, indice FOI senza tabacchi)"
            ),
        )
        rounded = round_output(revalued, definition.output)
        trail.record(
            "revalued_amount",
            amount=str(amount),
            coefficient=str(coefficient),
            revalued_amount=str(rounded),
            note=f"Importo rivalutato = {amount} x {coefficient} = {rounded} (arrotondato)",
        )

        start_key = f"foi_index_{start.year}_{start.month:02d}"
        end_key = f"foi_index_{end.year}_{end.month:02d}"
        parameters_used = {
            start_key: foi_resolved_parameter(start_key, pv_start).model_dump(),
        }
        if end_key != start_key:
            parameters_used[end_key] = foi_resolved_parameter(end_key, pv_end).model_dump()

        output_name = definition.output.get("name", "importo_rivalutato")
        return StrategyOutcome(
            result={output_name: rounded},
            derived_values={"coefficiente_rivalutazione": coefficient},
            parameters_used=parameters_used,
            steps=trail.steps,
            warnings=placeholder_warnings(pv_start, pv_end),
        )
