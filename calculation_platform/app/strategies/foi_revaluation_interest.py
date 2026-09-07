import calendar
from datetime import date
from decimal import Decimal
from typing import Any, Dict, List, Tuple

from ..core.audit import AuditTrail
from ..core.errors import StrategyExecutionError
from ..core.result_builder import round_output
from ..schemas.parameter_value import ParameterValue
from ..schemas.resolved_parameter import ResolvedParameter
from .base import CalculationStrategy, StrategyOutcome
from .date_split_interest import covered_rate_segments
from .foi_revaluation import (
    foi_resolved_parameter,
    placeholder_warnings,
    relinked_end_value,
    resolve_foi_index,
)

TWO = Decimal("2")


def year_slices(start: date, end: date) -> List[Tuple[date, date]]:
    """[start, end] split into calendar-year slices, inclusive at both ends.
    Partial first/last years are simply shorter slices — pro rata comes from
    the actual day counts downstream."""
    slices = []
    slice_start = start
    while slice_start.year < end.year:
        slices.append((slice_start, date(slice_start.year, 12, 31)))
        slice_start = date(slice_start.year + 1, 1, 1)
    slices.append((slice_start, end))
    return slices


def _year_divisor(year: int) -> Decimal:
    return Decimal(366 if calendar.isleap(year) else 365)


class FoiRevaluationInterestStrategy(CalculationStrategy):
    """Rivalutazione + interessi on a debito di valore per Cass. SS.UU.
    26/02/1995 n. 1712, year by year:

    - the capital is revalued slice by slice on the FOI chain (index of the
      month of data_iniziale, then December of each intervening year, then
      the month of data_finale — telescoping to FOI(final)/FOI(initial));
    - each slice's legal interest is computed on the MEAN between the
      capital at slice start and the revalued capital at slice end
      (criterio della media), at that year's legal rate, for the slice's
      actual days over a 365/366 divisor;
    - interest accumulates separately and is never added to the capital
      base (no anatocismo).
    """

    def run(self, definition, inputs: Dict[str, Any], request) -> StrategyOutcome:
        amount = inputs[definition.formula.get("amount_input", "importo")]
        start = inputs[definition.formula.get("start_date_input", "data_iniziale")]
        end = inputs[definition.formula.get("end_date_input", "data_finale")]
        index_parameter_id = definition.formula["index_parameter_id"]
        rate_parameter_id = definition.formula["rate_parameter_id"]

        if start > end:
            raise StrategyExecutionError(
                "data_iniziale must not be after data_finale",
                details={"data_iniziale": start.isoformat(), "data_finale": end.isoformat()},
            )

        trail = AuditTrail()
        parameters_used: Dict[str, Any] = {}
        foi_values: List[ParameterValue] = []

        def foi_at(as_of: date) -> ParameterValue:
            pv = resolve_foi_index(self.parameter_store, index_parameter_id, as_of)
            key = f"foi_index_{as_of.year}_{as_of.month:02d}"
            if key not in parameters_used:
                parameters_used[key] = foi_resolved_parameter(key, pv).model_dump()
                foi_values.append(pv)
            return pv

        capital = amount  # running revalued capital, full precision
        total_interest = Decimal("0")
        pv_start = foi_at(start)
        index_start = Decimal(str(pv_start.value))

        for slice_start, slice_end in year_slices(start, end):
            pv_end = foi_at(slice_end)
            index_end = Decimal(str(pv_end.value))
            # Relink the end index into the start's base (a no-op when bases
            # match), then keep the original multiply-then-divide order so a
            # base change never perturbs the last rounding digit.
            index_end_relinked = relinked_end_value(
                self.parameter_store, index_parameter_id, pv_start, pv_end
            )
            revalued = capital * index_end_relinked / index_start
            mean_base = (capital + revalued) / TWO
            divisor = _year_divisor(slice_start.year)

            for pv, seg_start, seg_end, days in covered_rate_segments(
                self.parameter_store, rate_parameter_id, slice_start, slice_end
            ):
                rate = Decimal(str(pv.value))
                segment_interest = mean_base * rate * Decimal(days) / divisor
                total_interest += segment_interest
                rate_key = f"legal_interest_rate_from_{pv.effective_from.isoformat()}"
                if rate_key not in parameters_used:
                    parameters_used[rate_key] = ResolvedParameter(
                        name=rate_key, value=rate, origin="parameter_store",
                        parameter_id=rate_parameter_id, source=pv.source,
                        effective_from=pv.effective_from.isoformat(),
                        effective_to=pv.effective_to.isoformat() if pv.effective_to else None,
                        official=pv.official, last_verified_at=pv.last_verified_at,
                        citations=pv.citations,
                    ).model_dump()
                trail.record(
                    "year_slice",
                    slice_start=seg_start.isoformat(),
                    slice_end=seg_end.isoformat(),
                    index_initial=str(index_start),
                    index_final=str(index_end),
                    capital_at_slice_start=str(capital),
                    capital_revalued=str(revalued),
                    mean_base=str(mean_base),
                    rate=str(rate),
                    days=days,
                    divisor=str(divisor),
                    interest=str(segment_interest),
                    note=(
                        f"Dal {seg_start.isoformat()} al {seg_end.isoformat()}: capitale rivalutato "
                        f"da {capital} a {revalued} (FOI {index_start} -> {index_end}); interessi legali "
                        f"al {rate} sulla media tra i due valori ({mean_base}) per {days} giorni "
                        f"su {divisor}: {segment_interest}"
                    ),
                )
            # Only the revalued capital rolls forward — interest never
            # enters the base (no anatocismo).
            capital = revalued
            pv_start = pv_end
            index_start = index_end

        revalued_rounded = round_output(capital, definition.output)
        interest_rounded = round_output(total_interest, definition.output)
        # The total is the sum of the two ROUNDED components, so the three
        # stated figures always add up exactly on the liquidation.
        total_rounded = revalued_rounded + interest_rounded
        trail.record(
            "totals",
            capitale_rivalutato=str(revalued_rounded),
            interessi_totali=str(interest_rounded),
            totale=str(total_rounded),
            note=(
                f"Capitale rivalutato {revalued_rounded} + interessi {interest_rounded} "
                f"= totale {total_rounded} (interessi mai capitalizzati)"
            ),
        )

        warnings = [
            (
                "Criterio applicato: interessi legali sulla MEDIA tra il capitale a inizio "
                "anno e il capitale rivalutato a fine anno (Cass. SS.UU. 26/02/1995 n. 1712). "
                "Esistono varianti giurisprudenziali (es. interessi sul capitale via via "
                "rivalutato, senza media)."
            ),
        ] + placeholder_warnings(*foi_values)

        return StrategyOutcome(
            result={
                "capitale_rivalutato": revalued_rounded,
                "interessi_totali": interest_rounded,
                "totale": total_rounded,
            },
            parameters_used=parameters_used,
            steps=trail.steps,
            warnings=warnings,
        )
