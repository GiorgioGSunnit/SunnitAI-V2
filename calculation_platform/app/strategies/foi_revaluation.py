from datetime import date
from decimal import Decimal
from typing import Any, Dict

from ..core.audit import AuditTrail
from ..core.errors import ParameterResolutionError, StrategyExecutionError
from ..core.result_builder import round_output
from ..schemas.parameter_value import ParameterValue
from ..schemas.resolved_parameter import ResolvedParameter
from .base import CalculationStrategy, StrategyOutcome, data_quality_warning


def relinked_end_value(parameter_store, parameter_id: str, pv_start: ParameterValue, pv_end: ParameterValue) -> Decimal:
    """The end index value expressed in the START month's base. When the two
    months carry different `base_year`s the end index is relinked via the
    store's base-link coefficient; if no link exists this is refused as a
    structured error rather than silently mixing incompatible bases. When bases
    match (or are unset, i.e. legacy single-base data) it is the raw value.

    Returning the *relinked value* (not a pre-divided coefficient) lets callers
    keep their original Decimal operation order — e.g. ``capital * v_end /
    v_start`` — so a base change never perturbs the last rounding digit."""
    v_end = Decimal(str(pv_end.value))
    b_start, b_end = pv_start.base_year, pv_end.base_year
    if b_start is not None and b_end is not None and b_start != b_end:
        try:
            link = parameter_store.base_link_coefficient(parameter_id, b_end, b_start)
        except KeyError as e:
            raise ParameterResolutionError(
                str(e),
                details={"parameter_id": parameter_id, "base_start": b_start, "base_end": b_end},
            ) from e
        v_end = v_end * link  # v_end now expressed in the start's base
    return v_end


def foi_coefficient(parameter_store, parameter_id: str, pv_start: ParameterValue, pv_end: ParameterValue) -> Decimal:
    """The revaluation coefficient FOI(end)/FOI(start), relinking across a base
    change first (see relinked_end_value)."""
    return relinked_end_value(parameter_store, parameter_id, pv_start, pv_end) / Decimal(str(pv_start.value))


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
    seen = set()
    for pv in values:
        label = f"L'indice FOI {pv.effective_from.year}-{pv.effective_from.month:02d}"
        if label in seen:
            continue
        seen.add(label)
        message = data_quality_warning(pv, label)
        if message:
            warnings.append(message)
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
        coefficient = foi_coefficient(self.parameter_store, index_parameter_id, pv_start, pv_end)
        revalued = amount * coefficient

        cross_base = (
            pv_start.base_year is not None
            and pv_end.base_year is not None
            and pv_start.base_year != pv_end.base_year
        )
        base_note = (
            f" (raccordo base {pv_end.base_year}->{pv_start.base_year} applicato)"
            if cross_base else ""
        )
        trail = AuditTrail()
        trail.record(
            "revaluation_coefficient",
            index_initial_month=f"{start.year}-{start.month:02d}",
            index_initial=str(index_start),
            index_initial_base=pv_start.base_year,
            index_final_month=f"{end.year}-{end.month:02d}",
            index_final=str(index_end),
            index_final_base=pv_end.base_year,
            coefficient=str(coefficient),
            note=(
                f"Coefficiente di rivalutazione = FOI({end.year}-{end.month:02d}) / "
                f"FOI({start.year}-{start.month:02d}) = {coefficient} "
                f"(convenzione del mese finale, indice FOI senza tabacchi){base_note}"
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
