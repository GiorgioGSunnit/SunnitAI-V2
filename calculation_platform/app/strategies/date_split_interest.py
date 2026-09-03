from datetime import date
from decimal import Decimal
from typing import Any, ClassVar, Dict, List, Tuple

from ..core.audit import AuditTrail
from ..core.errors import ParameterResolutionError, StrategyExecutionError
from ..core.result_builder import round_decimal
from ..schemas.parameter_value import ParameterValue
from ..schemas.resolved_parameter import ResolvedParameter
from .base import CalculationStrategy, StrategyOutcome


def covered_rate_segments(
    parameter_store, parameter_id: str, start: date, end: date
) -> List[Tuple[ParameterValue, date, date, int]]:
    """The rate parameter's effective ranges clamped to [start, end], as
    (value, segment_start, segment_end, days) with inclusive day counts.

    Shared by every strategy that splits a period across rate changes, so
    day-count and coverage logic exists exactly once. A rate table that
    stops mid-period must not silently shrink the result — the whole
    requested period has to be covered, or this raises.
    """
    try:
        segments = sorted(
            parameter_store.all_effective_ranges(parameter_id, start, end),
            key=lambda pv: pv.effective_from,
        )
    except KeyError as e:
        raise ParameterResolutionError(str(e), details={"parameter_id": parameter_id}) from e
    if not segments:
        raise ParameterResolutionError(
            f"No rate segments found for {parameter_id!r} in the requested period",
            details={"parameter_id": parameter_id, "start_date": start.isoformat(), "end_date": end.isoformat()},
        )

    clamped = []
    covered_days = 0
    for pv in segments:
        seg_start = max(start, pv.effective_from)
        seg_end = min(end, pv.effective_to or end)
        if seg_start > seg_end:
            continue
        days = (seg_end - seg_start).days + 1
        covered_days += days
        clamped.append((pv, seg_start, seg_end, days))

    period_days = (end - start).days + 1
    if covered_days != period_days:
        raise ParameterResolutionError(
            f"Rate parameter {parameter_id!r} covers only {covered_days} of the "
            f"{period_days} days in {start.isoformat()}..{end.isoformat()}; "
            "update the rate table before computing interest for this period",
            details={
                "parameter_id": parameter_id,
                "covered_days": covered_days,
                "period_days": period_days,
            },
        )
    return clamped


class DateSplitInterestStrategy(CalculationStrategy):
    """Simple interest over a period, splitting the period wherever the
    rate parameter's effective range changes (e.g. a period that spans a
    Jan-1 rate change is computed as two segments, not one).

    Honors an explicit caller override before consulting the date-versioned
    parameter store: if the rate's parameter name is present in
    request.caller_supplied_values, that single flat rate is used for the
    whole period (no date-splitting, since the caller is overriding the
    historical table entirely) — matching the engine's documented
    resolution order (caller value > parameter store > static default).
    """

    requires_period: ClassVar[bool] = True

    def run(self, definition, inputs: Dict[str, Any], request) -> StrategyOutcome:
        if request.period is None:
            raise StrategyExecutionError(
                "date_split_interest requires request.period.start_date/end_date",
                details={"calculator_id": definition.id},
            )
        start = request.period.start_date
        end = request.period.end_date
        if start > end:
            raise StrategyExecutionError(
                "period.start_date must not be after period.end_date",
                details={"start_date": start.isoformat(), "end_date": end.isoformat()},
            )

        capital = inputs[definition.formula.get("capital_input", "capital")]
        rate_parameter_id = definition.formula["rate_parameter_id"]
        day_count = int(definition.formula.get("day_count", 365))

        rate_param_name = next(
            (ref.name for ref in definition.parameters if ref.parameter_id == rate_parameter_id),
            None,
        )
        if rate_param_name and rate_param_name in request.caller_supplied_values:
            return self._run_with_flat_rate(
                capital, start, end, day_count, definition,
                rate_param_name, Decimal(str(request.caller_supplied_values[rate_param_name])),
            )

        segments = covered_rate_segments(self.parameter_store, rate_parameter_id, start, end)

        total_interest = Decimal("0")
        trail = AuditTrail()
        resolved_params: Dict[str, ResolvedParameter] = {}
        for pv, seg_start, seg_end, days in segments:
            rate = Decimal(str(pv.value))
            segment_interest = capital * rate * Decimal(days) / Decimal(day_count)
            total_interest += segment_interest
            trail.record(
                "interest_segment",
                segment_start=seg_start.isoformat(),
                segment_end=seg_end.isoformat(),
                days=days,
                rate=str(rate),
                interest=str(round_decimal(segment_interest, 2)),
            )
            key = f"{rate_param_name or rate_parameter_id}_from_{seg_start.isoformat()}"
            resolved_params[key] = ResolvedParameter(
                name=key, value=rate, origin="parameter_store",
                parameter_id=rate_parameter_id, source=pv.source,
                effective_from=pv.effective_from.isoformat(),
                effective_to=pv.effective_to.isoformat() if pv.effective_to else None,
                official=pv.official, last_verified_at=pv.last_verified_at,
                citations=pv.citations,
            )

        output_name = definition.output.get("name", "interest")
        rounded_interest = round_decimal(total_interest, definition.output.get("round_to", 2))
        return StrategyOutcome(
            result={
                output_name: rounded_interest,
                "capital_plus_interest": round_decimal(capital + total_interest, 2),
            },
            parameters_used={name: rp.model_dump() for name, rp in resolved_params.items()},
            steps=trail.steps,
        )

    def _run_with_flat_rate(self, capital, start, end, day_count, definition, rate_param_name, rate) -> StrategyOutcome:
        days = (end - start).days + 1
        segment_interest = capital * rate * Decimal(days) / Decimal(day_count)
        trail = AuditTrail()
        trail.record(
            "interest_segment",
            segment_start=start.isoformat(),
            segment_end=end.isoformat(),
            days=days,
            rate=str(rate),
            interest=str(round_decimal(segment_interest, 2)),
            note="caller-supplied rate override; parameter store not consulted",
        )
        output_name = definition.output.get("name", "interest")
        rounded_interest = round_decimal(segment_interest, definition.output.get("round_to", 2))
        resolved = ResolvedParameter(name=rate_param_name, value=rate, origin="caller_supplied")
        return StrategyOutcome(
            result={
                output_name: rounded_interest,
                "capital_plus_interest": round_decimal(capital + segment_interest, 2),
            },
            parameters_used={rate_param_name: resolved.model_dump()},
            steps=trail.steps,
        )
