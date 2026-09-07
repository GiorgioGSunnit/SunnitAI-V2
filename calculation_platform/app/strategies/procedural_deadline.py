from datetime import date, timedelta
from typing import Any, Dict, Set

from ..core.audit import AuditTrail
from ..core.errors import StrategyExecutionError
from ..resolvers.date_parameter_resolver import resolve_parameters
from .base import CalculationStrategy, StrategyOutcome


class ProceduralDeadlineStrategy(CalculationStrategy):
    """Italian civil procedural deadline arithmetic (termini a giorni,
    art. 155 c.p.c. + L. 742/1969):

    - dies a quo excluded, dies ad quem included (art. 155 co. 1);
    - "giorni liberi": both endpoint days excluded, i.e. one extra day;
    - feriale suspension: days in the 1-31 August window are not counted
      (L. 742/1969, window reduced by D.L. 132/2014);
    - a deadline landing on a Sunday/holiday is extended to the next
      working day (art. 155 co. 4), and Saturday likewise for procedural
      acts (art. 155 co. 5); for backward terms (termini a ritroso) the
      deadline is anticipated to the previous working day instead;
    - a deadline pushed into the feriale window by the rolling rule keeps
      moving until it exits the suspension.

    Expected formula keys:
      holidays_parameter   name of a declared parameter whose resolved value
                           is {"dates": [ISO...], "covers_through": ISO} —
                           national holidays, explicit per year.
      feriale              optional {"start": "MM-DD", "end": "MM-DD"},
                           default 08-01..08-31.
      saturday_rolls       optional bool, default true (art. 155 co. 5).

    Expected inputs (names fixed by convention, declared in the pack):
      data_decorrenza, giorni, giorni_liberi, sospensione_feriale,
      termine_a_ritroso.
    """

    def run(self, definition, inputs: Dict[str, Any], request) -> StrategyOutcome:
        cfg = definition.formula
        resolution = resolve_parameters(definition, self.parameter_store, request)
        holidays_value = resolution.values[cfg["holidays_parameter"]]
        if not isinstance(holidays_value, dict) or "dates" not in holidays_value:
            raise StrategyExecutionError(
                f"holidays parameter {cfg['holidays_parameter']!r} must resolve to a dict with a 'dates' list",
                details={"calculator_id": definition.id},
            )
        holidays: Set[date] = {date.fromisoformat(str(d)) for d in holidays_value["dates"]}
        covers_through = (
            date.fromisoformat(str(holidays_value["covers_through"]))
            if holidays_value.get("covers_through") else None
        )
        covers_from = (
            date.fromisoformat(str(holidays_value["covers_from"]))
            if holidays_value.get("covers_from") else None
        )

        start: date = inputs["data_decorrenza"]
        days = int(inputs["giorni"])
        if days < 1:
            raise StrategyExecutionError(
                "giorni must be at least 1", details={"calculator_id": definition.id, "giorni": days},
            )
        free_days = bool(inputs.get("giorni_liberi", False))
        feriale_on = bool(inputs.get("sospensione_feriale", True))
        backward = bool(inputs.get("termine_a_ritroso", False))
        saturday_rolls = bool(cfg.get("saturday_rolls", True))

        feriale_cfg = cfg.get("feriale", {})
        feriale_start = str(feriale_cfg.get("start", "08-01"))
        feriale_end = str(feriale_cfg.get("end", "08-31"))

        def suspended(d: date) -> bool:
            return feriale_on and feriale_start <= d.strftime("%m-%d") <= feriale_end

        def nonworking(d: date) -> bool:
            if d.weekday() == 6 or d in holidays:
                return True
            return saturday_rolls and d.weekday() == 5

        # Giorni liberi exclude the dies ad quem as well: one more counted day.
        effective_days = days + (1 if free_days else 0)
        step = -1 if backward else 1

        current = start
        counted = 0
        suspension_skipped = 0
        while counted < effective_days:
            current += timedelta(days=step)
            if suspended(current):
                suspension_skipped += 1
                continue
            counted += 1
        raw_deadline = current

        rolled_over = 0
        deadline = raw_deadline
        while nonworking(deadline) or suspended(deadline):
            deadline += timedelta(days=step)
            rolled_over += 1

        trail = AuditTrail()
        trail.record(
            "term_counted",
            data_decorrenza=start.isoformat(),
            giorni=days,
            giorni_liberi=free_days,
            direction="a_ritroso" if backward else "in_avanti",
            counted_days=effective_days,
            feriale_days_skipped=suspension_skipped,
            raw_deadline=raw_deadline.isoformat(),
        )
        if rolled_over:
            trail.record(
                "holiday_roll",
                from_date=raw_deadline.isoformat(),
                to_date=deadline.isoformat(),
                days_moved=rolled_over,
                direction="anticipata" if backward else "prorogata",
            )

        warnings = []
        if covers_through and max(deadline, start) > covers_through:
            warnings.append(
                f"La scadenza {deadline.isoformat()} cade oltre il {covers_through.isoformat()}, "
                "ultima data coperta dal calendario festività caricato: il rinvio per giorni "
                "festivi potrebbe non essere stato applicato. Verificare manualmente."
            )
        if covers_from and min(deadline, start) < covers_from:
            warnings.append(
                f"Il termine coinvolge date anteriori al {covers_from.isoformat()}, prima data "
                "coperta dal calendario festività caricato: il rinvio per giorni festivi "
                "potrebbe non essere stato applicato. Verificare manualmente."
            )
        if backward and rolled_over and deadline.weekday() == 4:
            # Anticipating a backward term that fell on Saturday is the
            # prudent reading but not textually settled — flag it.
            warnings.append(
                "Termine a ritroso anticipato da un sabato al venerdì precedente: "
                "orientamento prevalente ma non unanime; verificare la prassi del foro."
            )

        output_name = definition.output.get("name", "scadenza")
        return StrategyOutcome(
            result={
                output_name: deadline.isoformat(),
                "scadenza_senza_rinvii": raw_deadline.isoformat(),
                "giorni_feriale_sospesi": suspension_skipped,
            },
            parameters_used=resolution.parameters_used(),
            date_resolution=resolution.date_resolution,
            steps=trail.steps,
            warnings=warnings,
        )
