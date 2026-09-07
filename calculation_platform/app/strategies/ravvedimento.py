from datetime import timedelta
from decimal import Decimal
from typing import Any, Dict

from ..core.audit import AuditTrail
from ..core.errors import ParameterResolutionError, StrategyExecutionError
from ..core.result_builder import round_output
from ..schemas.resolved_parameter import ResolvedParameter
from .base import CalculationStrategy, StrategyOutcome


class RavvedimentoStrategy(CalculationStrategy):
    """Ravvedimento operoso for a late/omitted tax payment: reduced penalty
    picked from a delay-tier table plus legal interest accrued day by day
    (date-split across rate changes), on top of the tax itself.

    The tier table is resolved by the ORIGINAL DUE DATE (the violation
    date), not the request's as_of date — the applicable sanction regime is
    the one in force when the violation was committed (favor rei aside,
    which is out of scope and flagged in the pack's warnings).

    Expected formula keys:
      principal_input        the unpaid tax amount input.
      due_date_input         the original payment deadline input.
      payment_date_input     the date the taxpayer will pay.
      tiers_parameter_id     parameter whose value is {"tiers": [{"max_days":
                             int|null, "type": "per_day"|"flat", "rate": num,
                             "label": str}, ...]} sorted by max_days.
      interest_parameter_id  the legal interest rate parameter (date-split).
      day_count              optional, default 365.
    """

    def run(self, definition, inputs: Dict[str, Any], request) -> StrategyOutcome:
        cfg = definition.formula
        principal = Decimal(str(inputs[cfg["principal_input"]]))
        due_date = inputs[cfg["due_date_input"]]
        payment_date = inputs[cfg["payment_date_input"]]
        day_count = Decimal(str(cfg.get("day_count", 365)))

        delay_days = (payment_date - due_date).days
        if delay_days <= 0:
            raise StrategyExecutionError(
                "data_pagamento must be after scadenza_originaria — with no delay there is nothing to regularize",
                details={
                    "calculator_id": definition.id,
                    "scadenza_originaria": due_date.isoformat(),
                    "data_pagamento": payment_date.isoformat(),
                },
            )

        trail = AuditTrail()
        parameters_used: Dict[str, Any] = {}

        # Tier table resolved as of the violation (original due) date.
        tiers_parameter_id = cfg["tiers_parameter_id"]
        try:
            tiers_pv = self.parameter_store.resolve_by_date(tiers_parameter_id, due_date)
        except KeyError as e:
            raise ParameterResolutionError(
                f"No sanction tier table in force on {due_date.isoformat()} — "
                "violations under earlier sanction regimes are not covered",
                details={"parameter_id": tiers_parameter_id, "violation_date": due_date.isoformat()},
            ) from e
        parameters_used["ravvedimento_tiers"] = ResolvedParameter(
            name="ravvedimento_tiers", value=tiers_pv.value, origin="parameter_store",
            parameter_id=tiers_parameter_id, source=tiers_pv.source,
            effective_from=tiers_pv.effective_from.isoformat(),
            effective_to=tiers_pv.effective_to.isoformat() if tiers_pv.effective_to else None,
            official=tiers_pv.official, last_verified_at=tiers_pv.last_verified_at,
            citations=tiers_pv.citations,
        ).model_dump()

        # The "long" tier legally runs to the declaration deadline for the
        # year of the violation, not a fixed number of days. When the caller
        # supplies it, use it; otherwise fall back to the tier's declared
        # fallback_max_days and record the approximation.
        declaration_deadline = None
        deadline_input = cfg.get("declaration_deadline_input")
        if deadline_input and deadline_input in inputs:
            declaration_deadline = inputs[deadline_input]

        assumptions = []
        tier = None
        for candidate in tiers_pv.value["tiers"]:
            if candidate.get("boundary") == "declaration_deadline":
                if declaration_deadline is not None:
                    if payment_date <= declaration_deadline:
                        tier = candidate
                        break
                    continue
                if delay_days <= int(candidate["fallback_max_days"]):
                    tier = candidate
                    assumptions.append(
                        "Termine di presentazione della dichiarazione non indicato: lo scaglione "
                        f"'{candidate.get('label', '1/8')}' è stato applicato approssimandolo a "
                        f"{candidate['fallback_max_days']} giorni. Per i tributi dichiarativi "
                        "indicare 'termine_dichiarazione' per un calcolo esatto."
                    )
                    break
                continue
            max_days = candidate.get("max_days")
            if max_days is None or delay_days <= int(max_days):
                tier = candidate
                break
        if tier is None:
            raise StrategyExecutionError(
                "no sanction tier matched — the table's last tier should have max_days: null",
                details={"calculator_id": definition.id, "delay_days": delay_days},
            )

        rate = Decimal(str(tier["rate"]))
        if tier.get("type") == "per_day":
            sanzione = principal * rate * Decimal(delay_days)
        else:
            sanzione = principal * rate
        trail.record(
            "sanction_tier",
            delay_days=delay_days,
            tier_label=tier.get("label", ""),
            rate=str(rate),
            per_day=tier.get("type") == "per_day",
            sanzione=str(round_output(sanzione, definition.output)),
        )

        # Legal interest, day by day from the day after the due date through
        # the payment date, split across rate changes.
        interest_parameter_id = cfg["interest_parameter_id"]
        interest_start = due_date + timedelta(days=1)
        try:
            segments = sorted(
                self.parameter_store.all_effective_ranges(interest_parameter_id, interest_start, payment_date),
                key=lambda pv: pv.effective_from,
            )
        except KeyError as e:
            raise ParameterResolutionError(str(e), details={"parameter_id": interest_parameter_id}) from e
        if not segments:
            raise ParameterResolutionError(
                f"No legal interest rate found for {interest_parameter_id!r} in the delay period",
                details={
                    "parameter_id": interest_parameter_id,
                    "start_date": interest_start.isoformat(),
                    "end_date": payment_date.isoformat(),
                },
            )

        interessi = Decimal("0")
        covered_days = 0
        for pv in segments:
            seg_start = max(interest_start, pv.effective_from)
            seg_end = min(payment_date, pv.effective_to or payment_date)
            if seg_start > seg_end:
                continue
            days = (seg_end - seg_start).days + 1
            covered_days += days
            seg_rate = Decimal(str(pv.value))
            segment_interest = principal * seg_rate * Decimal(days) / day_count
            interessi += segment_interest
            trail.record(
                "interest_segment",
                segment_start=seg_start.isoformat(),
                segment_end=seg_end.isoformat(),
                days=days,
                rate=str(seg_rate),
                interest=str(round_output(segment_interest, definition.output)),
            )
            key = f"legal_interest_rate_from_{seg_start.isoformat()}"
            parameters_used[key] = ResolvedParameter(
                name=key, value=seg_rate, origin="parameter_store",
                parameter_id=interest_parameter_id, source=pv.source,
                effective_from=pv.effective_from.isoformat(),
                effective_to=pv.effective_to.isoformat() if pv.effective_to else None,
                official=pv.official, last_verified_at=pv.last_verified_at,
                citations=pv.citations,
            ).model_dump()

        # Undercounting interest because the rate table stops mid-period is
        # a silent wrong number — refuse instead.
        if covered_days != delay_days:
            raise ParameterResolutionError(
                f"Legal interest rates cover only {covered_days} of the {delay_days} delay days "
                f"({interest_start.isoformat()}..{payment_date.isoformat()}); "
                "update the rate table before computing this ravvedimento",
                details={
                    "parameter_id": interest_parameter_id,
                    "covered_days": covered_days,
                    "delay_days": delay_days,
                },
            )

        warnings = []
        if inputs.get("violazione_gia_constatata") is True:
            warnings.append(
                "Violazione già constatata/contestata: il ravvedimento può essere precluso "
                "dalla notifica di atti di liquidazione o accertamento (art. 13 co. 1 "
                "D.Lgs. 472/1997). Verificare prima di procedere al versamento."
            )

        # F24 practice: each component is rounded separately, the total is
        # the sum of the rounded components.
        sanzione_r = round_output(sanzione, definition.output)
        interessi_r = round_output(interessi, definition.output)
        totale = principal + sanzione_r + interessi_r
        return StrategyOutcome(
            result={
                "totale_da_versare": round_output(totale, definition.output),
                "sanzione_ridotta": sanzione_r,
                "interessi": interessi_r,
                "tributo": round_output(principal, definition.output),
            },
            derived_values={"giorni_di_ritardo": delay_days},
            parameters_used=parameters_used,
            steps=trail.steps,
            warnings=warnings,
            assumptions=assumptions,
        )
