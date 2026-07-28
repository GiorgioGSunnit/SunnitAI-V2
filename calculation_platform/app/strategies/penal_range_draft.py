"""DRAFT penal-range strategy — mechanical demonstration only.

Chains the tested penal arithmetic operations (app/penal/) into the
pipeline order from the design document, for ONE offence whose base
range is declared in the calculator's YAML. Circumstances are plain
COUNTS (n aggravating, n mitigating), not the legally-gated catalog —
this deliberately demonstrates the mechanics without asserting any
legal qualification. Not validated by a lawyer; every result carries
the draft warnings from the definition.

Pipeline: base range -> discretionary envelopes per circumstance
(sequential, art. 63) -> art. 69 scenario (selected or full matrix)
-> art. 66 caps -> art. 442 abbreviato reduction -> formatted ranges.
"""

from decimal import Decimal
from fractions import Fraction
from typing import Any, Dict, List

from ..core.audit import AuditTrail
from ..core.errors import StrategyExecutionError
from ..penal.duration import format_duration_it, years
from ..penal.operations import (
    apply_abbreviato,
    apply_fraction_envelope,
    apply_reclusione_caps,
)
from ..penal.penalty import PenalRange, Penalty
from .base import CalculationStrategy, StrategyOutcome

_THIRD = Fraction(1, 3)
_SCENARIOS = ("aggravanti_prevalenti", "attenuanti_prevalenti", "equivalenza")


class PenalRangeDraftStrategy(CalculationStrategy):
    def run(self, definition, inputs: Dict[str, Any], request) -> StrategyOutcome:
        # str() first so YAML may hold ints (21), decimals or exact
        # fractions ("1/2" for a 6-month minimum) — always exact rationals.
        base = PenalRange(
            min_years=years(Fraction(str(definition.formula["base_min_years"]))),
            max_years=years(Fraction(str(definition.formula["base_max_years"]))),
        )
        statutory_max = base.max_years
        n_aggravating = int(inputs["aggravanti_comuni"])
        n_mitigating = int(inputs["attenuanti_comuni"])
        abbreviato = bool(inputs.get("rito_abbreviato", False))
        scenario = inputs.get("scenario_art_69") or None
        if scenario is not None and scenario not in _SCENARIOS:
            raise StrategyExecutionError(
                f"scenario_art_69 must be one of: {', '.join(_SCENARIOS)}",
                details={"scenario_art_69": scenario},
            )

        trail = AuditTrail()

        def compute(agg: int, mit: int, label: str) -> Dict[str, str]:
            rng = base
            trail.record("base_edittale", scenario=label,
                         min=format_duration_it(rng.min_years),
                         max=format_duration_it(rng.max_years),
                         norm="cornice edittale del reato")
            for _ in range(agg):
                rng = apply_fraction_envelope(rng, "increase", Fraction(0), _THIRD)
                trail.record("aggravante_comune", scenario=label,
                             effetto="aumento discrezionale fino a 1/3",
                             max=format_duration_it(rng.max_years),
                             norm="artt. 63-64 c.p.")
            for _ in range(mit):
                rng = apply_fraction_envelope(rng, "decrease", Fraction(0), _THIRD)
                trail.record("attenuante_comune", scenario=label,
                             effetto="diminuzione discrezionale fino a 1/3",
                             min=format_duration_it(rng.min_years),
                             norm="artt. 63, 65 c.p.")
            capped = apply_reclusione_caps(rng, statutory_max_years=statutory_max)
            if capped != rng:
                trail.record("tetto_applicato", scenario=label,
                             max=format_duration_it(capped.max_years), norm="art. 66 c.p.")
            penalty = Penalty(species="reclusione", range=capped)
            if abbreviato:
                penalty = apply_abbreviato(penalty, offence_kind="delitto")
                trail.record("rito_abbreviato", scenario=label,
                             effetto="riduzione di 1/3",
                             min=format_duration_it(penalty.range.min_years),
                             max=format_duration_it(penalty.range.max_years),
                             norm="art. 442 c.p.p.")
            return {
                "specie": penalty.species,
                "pena_minima": format_duration_it(penalty.range.min_years),
                "pena_massima": format_duration_it(penalty.range.max_years),
            }

        both_present = n_aggravating > 0 and n_mitigating > 0
        if both_present and scenario is None:
            # Art. 69: no scenario selected -> full matrix, never one number.
            result: Dict[str, Any] = {
                "tipo": "matrice_scenari_art_69",
                "aggravanti_prevalenti": compute(n_aggravating, 0, "aggravanti_prevalenti"),
                "attenuanti_prevalenti": compute(0, n_mitigating, "attenuanti_prevalenti"),
                "equivalenza": compute(0, 0, "equivalenza"),
            }
        else:
            if scenario == "aggravanti_prevalenti":
                agg, mit = n_aggravating, 0
            elif scenario == "attenuanti_prevalenti":
                agg, mit = 0, n_mitigating
            elif scenario == "equivalenza":
                agg, mit = 0, 0
            else:
                agg, mit = n_aggravating, n_mitigating
            result = compute(agg, mit, scenario or "unico")

        warnings: List[str] = []
        self._attach_multa(definition, result, trail, warnings)

        return StrategyOutcome(result=result, steps=trail.steps, warnings=warnings)

    @staticmethod
    def _attach_multa(definition, result: Dict[str, Any], trail: AuditTrail, warnings: List[str]) -> None:
        """The pecuniary penalty (multa) that most patrimonial delitti carry
        alongside reclusione. Reported as its DECLARED statutory frame only:
        this draft does not adjust the fine for circumstances, the art. 66
        n. 3 ceilings, or the rito reduction — those fine-specific rules are
        not modeled, so surfacing a computed fine would be a guess. A pack
        omits the two keys for an offence with no fine (e.g. omicidio)."""
        formula = definition.formula or {}
        if "multa_min_eur" not in formula and "multa_max_eur" not in formula:
            return
        multa_min = Decimal(str(formula["multa_min_eur"]))
        multa_max = Decimal(str(formula["multa_max_eur"]))
        if multa_min > multa_max:
            raise StrategyExecutionError(
                "multa_min_eur cannot exceed multa_max_eur",
                details={"multa_min_eur": str(multa_min), "multa_max_eur": str(multa_max)},
            )
        result["multa_specie"] = "multa"
        result["multa_base_minima_eur"] = f"{multa_min:.2f}"
        result["multa_base_massima_eur"] = f"{multa_max:.2f}"
        trail.record("multa_edittale_base",
                     min=f"{multa_min:.2f}", max=f"{multa_max:.2f}",
                     norm="cornice edittale della multa (art. 24 c.p.)")
        warnings.append(
            "La multa e riportata nella sua cornice edittale base (art. 24 c.p.): "
            "questa bozza NON la adegua per circostanze, tetti (art. 66 n. 3) o "
            "riduzione per rito abbreviato."
        )
