"""End-to-end simulation of one calculation conversation.

This is the full production loop with the LLM replaced by the
deterministic planner, exercising every platform seam the real
integration will use:

  user sentence
    -> planner picks a calculator and extracts what it can (PlanResult)
    -> if the plan is ready: engine.calculate() with a structured request
    -> if the plan (or the platform's structured errors) reports
       missing/invalid inputs, that becomes a clarifying question; the
       user's next message fills the gap and the calculation retries
    -> on success, a deterministic "synthesis" renders the result with
       sources, warnings and assumptions — the shape of the final answer
       the real LLM would write in prose.

The conversation object is intentionally the ONLY stateful piece: the
platform itself stays stateless per request, exactly as in production.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.core.engine import CalculationEngine
from app.schemas.calculation_request import CalculationRequest
from app.schemas.calculation_result import CalculationResult

from .planner import PlanResult, _clarification_question, plan_sentence
from .scripted_llm import SimulatedToolCall, bind_offer, bind_values, extract_values

# Words that close the offer-collection loop of a comparator conversation.
_FINISH_WORDS = {"confronta", "calcola", "basta", "fine", "procedi"}


def _render_default_value(value: Any) -> str:
    """A defaulted boolean prints as sì/no, not as Python's True/False —
    the point of showing defaults is that a reader can check them."""
    if isinstance(value, bool):
        return "sì" if value else "no"
    return str(value)


@dataclass
class Reply:
    kind: str  # "answer" | "question" | "ambiguous" | "no_match"
    text: str
    calculation: Optional[CalculationResult] = None
    tool_call: Optional[SimulatedToolCall] = None
    plan: Optional[PlanResult] = None


@dataclass
class _PendingCalculation:
    tool_call: SimulatedToolCall
    asked_for: List[str] = field(default_factory=list)
    # Comparator conversations collect candidates one message at a time:
    # `list_spec` is the calculator's object_list InputSpec, `offers` the
    # candidates gathered so far (finalized into the tool call on a finish
    # word). None for ordinary single-scenario calculators.
    list_spec: Any = None
    offers: List[Dict[str, Any]] = field(default_factory=list)


class SimulatedConversation:
    def __init__(self, engine: CalculationEngine):
        self.engine = engine
        self._pending: Optional[_PendingCalculation] = None

    def send(self, message: str) -> Reply:
        if self._pending is not None:
            return self._continue_pending(message)
        return self._start_new(message)

    # ------------------------------------------------------------------

    def _start_new(self, sentence: str) -> Reply:
        plan = plan_sentence(sentence, self.engine.registry.definitions())

        if plan.status == "no_match":
            return Reply(
                kind="no_match",
                text=(
                    "Nessun calcolo disponibile corrisponde alla richiesta. "
                    "Il sistema non tenta di indovinare: la domanda va riformulata "
                    "oppure il calcolo non è (ancora) supportato."
                ),
                plan=plan,
            )

        if plan.status == "ambiguous":
            lines = ["La richiesta potrebbe corrispondere a più calcoli. Quale intendi?"]
            for index, candidate in enumerate(plan.candidates, 1):
                lines.append(f"  {index}) {candidate.calculator_id} — {candidate.name}")
            return Reply(kind="ambiguous", text="\n".join(lines), plan=plan)

        tool_call = SimulatedToolCall(
            calculator_id=plan.calculator_id,
            inputs=dict(plan.inputs),
            tax_year=plan.tax_year,
            period=plan.period,
        )
        self._pending = _PendingCalculation(tool_call=tool_call)

        definition = self.engine.registry.get(plan.calculator_id)
        list_spec = next((s for s in definition.inputs if s.type == "object_list"), None)
        if list_spec is not None:
            self._pending.list_spec = list_spec
            reply = self._advance_comparison(definition)
            reply.plan = plan
            return reply

        if plan.status == "needs_clarification":
            self._pending.asked_for = plan.missing_inputs
            return Reply(kind="question", text=plan.question, tool_call=tool_call, plan=plan)

        reply = self._attempt()
        reply.plan = plan
        return reply

    def _continue_pending(self, message: str) -> Reply:
        pending = self._pending
        definition = self.engine.registry.get(pending.tool_call.calculator_id)

        if pending.list_spec is not None and not self._missing_scalars(definition):
            return self._offer_message(definition, message)

        values = extract_values(message)
        bind_values(definition, pending.tool_call.inputs, values)
        if pending.tool_call.tax_year is None and values["tax_year"]:
            pending.tool_call.tax_year = values["tax_year"]
        if pending.tool_call.period is None and values["period"]:
            pending.tool_call.period = values["period"]
        if pending.list_spec is not None:
            return self._advance_comparison(definition)
        return self._attempt()

    # ------------------------------------------------- comparator collection

    def _missing_scalars(self, definition) -> List[str]:
        inputs = self._pending.tool_call.inputs
        return [
            s.name for s in definition.inputs
            if s.required and s.type != "object_list" and s.name not in inputs
        ]

    def _advance_comparison(self, definition) -> Reply:
        """Next question in a comparator conversation: shared scalar inputs
        first (the applicant/consumption facts every offer shares), then the
        offers, one message each, closed by a finish word."""
        pending = self._pending
        missing = self._missing_scalars(definition)
        if missing:
            pending.asked_for = missing
            return Reply(
                kind="question",
                text=_clarification_question(definition, missing),
                tool_call=pending.tool_call,
            )
        required_fields = [
            s.description or s.name
            for s in pending.list_spec.item_fields
            if s.required
        ]
        return Reply(
            kind="question",
            text=(
                f"Confronto '{definition.name}'. Dimmi la prima offerta in un solo "
                f"messaggio, indicando: {'; '.join(required_fields)}. "
                "Puoi aggiungere anche gli altri dati che conosci (coperture, sconti, servizi, voto)."
            ),
            tool_call=pending.tool_call,
        )

    def _offer_message(self, definition, message: str) -> Reply:
        pending = self._pending
        normalized = message.strip().lower().rstrip("!.")
        min_items = pending.list_spec.min_items or 2

        if normalized in _FINISH_WORDS:
            if len(pending.offers) < min_items:
                return Reply(
                    kind="question",
                    text=(
                        f"Per un confronto servono almeno {min_items} offerte — finora "
                        f"ne ho registrate {len(pending.offers)}. Dimmi la prossima offerta."
                    ),
                    tool_call=pending.tool_call,
                )
            pending.tool_call.inputs[pending.list_spec.name] = list(pending.offers)
            return self._attempt()

        offer = bind_offer(pending.list_spec.item_fields, message)
        missing_required = [
            s for s in pending.list_spec.item_fields
            if s.required and s.name not in offer
        ]
        if missing_required:
            labels = [f"{s.name} ({s.description})" if s.description else s.name for s in missing_required]
            return Reply(
                kind="question",
                text=(
                    "Per registrare questa offerta manca ancora: " + "; ".join(labels)
                    + ". Ripetila completa in un solo messaggio (i numeri vanno "
                    "etichettati, es. 'premio 450 euro' — il sistema non abbina "
                    "numeri senza etichetta)."
                ),
                tool_call=pending.tool_call,
            )

        pending.offers.append(offer)
        summary = ", ".join(
            f"{k}={'sì' if v is True else 'no' if v is False else v}"
            for k, v in offer.items()
        )
        next_step = (
            f"Dimmi la prossima offerta, oppure scrivi 'confronta' per procedere"
            if len(pending.offers) >= min_items
            else "Dimmi la prossima offerta"
        )
        return Reply(
            kind="question",
            text=f"Registrata offerta {len(pending.offers)}: {summary}.\n{next_step}.",
            tool_call=pending.tool_call,
        )

    # ------------------------------------------------------------------

    def _attempt(self) -> Reply:
        tool_call = self._pending.tool_call
        definition = self.engine.registry.get(tool_call.calculator_id)

        needs_period = definition.requires_period and tool_call.period is None
        if needs_period:
            return Reply(
                kind="question",
                text=(
                    f"Per calcolare '{definition.name}' serve il periodo di riferimento: "
                    "indica data di inizio e di fine nel formato YYYY-MM-DD."
                ),
                tool_call=tool_call,
            )

        request = CalculationRequest(
            calculator_id=tool_call.calculator_id,
            inputs={k: v for k, v in tool_call.inputs.items()},
            tax_year=tool_call.tax_year,
            period=tool_call.period,
        )
        result = self.engine.calculate(request)

        if result.status == "success":
            self._pending = None
            return Reply(
                kind="answer",
                text=self._synthesize(definition, result, tool_call),
                calculation=result,
                tool_call=tool_call,
            )

        error = result.errors[0]
        if error.code == "input_invalid":
            missing = error.details.get("missing_inputs")
            if missing:
                descriptions = []
                for name in missing:
                    spec = next((s for s in definition.inputs if s.name == name), None)
                    label = f"{name} ({spec.description})" if spec and spec.description else name
                    descriptions.append(label)
                self._pending.asked_for = missing
                return Reply(
                    kind="question",
                    text=(
                        f"Per calcolare '{definition.name}' manca ancora: "
                        + "; ".join(descriptions) + ". Puoi indicarlo?"
                    ),
                    calculation=result,
                    tool_call=tool_call,
                )
            # A supplied value was invalid (e.g. negative income): drop it
            # so the next user message can replace it, and relay the
            # platform's own explanation as the question.
            bad_input = error.details.get("input")
            if bad_input:
                self._pending.tool_call.inputs.pop(bad_input, None)
            return Reply(
                kind="question",
                text=f"Valore non valido: {error.message}. Puoi correggerlo?",
                calculation=result,
                tool_call=tool_call,
            )

        # Any other structured error ends the conversation attempt honestly.
        self._pending = None
        return Reply(
            kind="no_match",
            text=f"Il calcolo non è stato possibile: {error.message}",
            calculation=result,
            tool_call=tool_call,
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _render_ranking(
        ranking: List[Dict[str, Any]],
        best: Optional[str],
        comparison: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Render a comparison so the reader can tell a recommendation from
        a coin flip.

        The verdict leads, the money leads each candidate, and the 0-100
        score comes after both: a synthetic score printed first reads as an
        objective grade, which it is not. On an effective tie no offer is
        called "the best" anywhere in the output, and a provisional result
        says so before the numbers rather than in a footnote.
        """
        comparison = comparison or {}
        status = comparison.get("decision_status")
        lines: List[str] = []

        if status == "effective_tie":
            tied = ", ".join(comparison.get("best_candidates") or [])
            lines.append(
                f"  Esito: PARITA SOSTANZIALE tra {tied}. Con il modello di punteggio "
                f"configurato non c'e una differenza materiale (distacco "
                f"{comparison.get('score_gap')} punti, entro la tolleranza di "
                f"{comparison.get('tie_tolerance')}): il confronto non indica un vincitore."
            )
        elif status == "clear_winner":
            winner = (comparison.get("best_candidates") or [best])[0]
            lines.append(
                f"  Esito: {winner} e in testa in modo netto secondo il modello configurato "
                f"(distacco {comparison.get('score_gap')} punti su 100, tolleranza di parita "
                f"{comparison.get('tie_tolerance')})."
            )
        elif best:
            lines.append(f"  Migliore offerta: {best}")

        if comparison.get("provisional"):
            assumed = ", ".join(
                str(entry.get("path")) for entry in comparison.get("scoring_defaults_applied") or []
            )
            lines.append(
                "  ATTENZIONE — risultato PROVVISORIO, non definitivo: alcuni campi che "
                f"incidono sul punteggio non sono stati forniti e sono stati assunti per "
                f"default ({assumed}). Completezza dei dati di punteggio: "
                f"{comparison.get('scoring_completeness')}."
            )

        cost_variable = (comparison.get("cost_basis") or {}).get("variable")
        for entry in ranking:
            derived = entry.get("derived") or {}
            cost = derived.get(cost_variable) if cost_variable else None
            head = f"  {entry['rank']}. {entry['label']} — "
            if cost is not None:
                head += f"costo stimato {cost} ({cost_variable}); "
            head += f"punteggio totale {entry['total_score']}/100"
            lines.append(head)
            components = ", ".join(f"{name} {value}" for name, value in entry.get("scores", {}).items())
            if components:
                lines.append(f"     dettaglio: {components}")
            if derived:
                lines.append("     valori: " + ", ".join(f"{k} {v}" for k, v in derived.items()))
            lines += SimulatedConversation._render_data_quality(entry.get("data_quality") or {})

        lines.append(
            "  Nota: il punteggio 0-100 e relativo alle sole offerte confrontate e ai pesi "
            "configurati in questo calcolatore, non una misura oggettiva di mercato."
        )
        return lines

    @staticmethod
    def _render_data_quality(quality: Dict[str, Any]) -> List[str]:
        if not quality:
            return []
        parts = [f"{len(quality.get('provided_fields') or [])} forniti"]
        assumed = quality.get("assumed_fields") or []
        if assumed:
            parts.append(f"{len(assumed)} assunti per default ({', '.join(assumed)})")
        unknown = quality.get("unknown_fields") or []
        if unknown:
            # Not the same as a defaulted field: nothing stood in for these.
            label = "non dichiarato" if len(unknown) == 1 else "non dichiarati"
            parts.append(f"{len(unknown)} {label} ({', '.join(unknown)})")
        return [
            f"     dati: {'; '.join(parts)}; completezza sui campi valutati "
            f"{quality.get('scoring_completeness')}"
        ]

    @staticmethod
    def _synthesize(definition, result: CalculationResult, tool_call: SimulatedToolCall) -> str:
        lines = [f"Risultato — {definition.name}:"]
        unit = f" {definition.output_unit}" if definition.output_unit else ""
        ranking = result.result.get("ranking")
        if (
            isinstance(ranking, list) and ranking
            and isinstance(ranking[0], dict) and "total_score" in ranking[0]
        ):
            lines += SimulatedConversation._render_ranking(
                ranking, result.result.get("best"), result.result.get("comparison")
            )
        else:
            for key, value in result.result.items():
                if isinstance(value, (int, float)):
                    lines.append(f"  {key}: {value:,.2f}{unit}")
                else:
                    lines.append(f"  {key}: {value}")
        if tool_call.tax_year:
            lines.append(f"Anno d'imposta: {tool_call.tax_year}")
        if result.citations:
            lines.append("Fonti: " + "; ".join(c.reference for c in result.citations))
        if result.warnings:
            lines.append("Avvertenze: " + " ".join(w.message for w in result.warnings))
        if result.assumptions:
            lines.append("Assunzioni: " + " ".join(a.message for a in result.assumptions))
        if result.defaults_applied:
            lines.append(
                "Default applicati: "
                + "; ".join(
                    f"{entry.get('path')}={_render_default_value(entry.get('value'))}"
                    for entry in result.defaults_applied
                )
            )
        # Its own section, never merged into the warnings: what a calculator
        # deliberately leaves out is a scope statement the reader has to see
        # before acting, not a caveat about precision.
        if result.exclusions:
            lines.append("Non incluso:")
            lines += [f"  - {item}" for item in result.exclusions]
        lines.append(f"(Calcolo verificabile: {len(result.steps)} passaggi registrati.)")
        return "\n".join(lines)
