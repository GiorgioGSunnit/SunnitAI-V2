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

from .planner import PlanResult, plan_sentence
from .scripted_llm import SimulatedToolCall, bind_values, extract_values


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

        if plan.status == "needs_clarification":
            self._pending.asked_for = plan.missing_inputs
            return Reply(kind="question", text=plan.question, tool_call=tool_call, plan=plan)

        reply = self._attempt()
        reply.plan = plan
        return reply

    def _continue_pending(self, message: str) -> Reply:
        pending = self._pending
        definition = self.engine.registry.get(pending.tool_call.calculator_id)
        values = extract_values(message)
        bind_values(definition, pending.tool_call.inputs, values)
        if pending.tool_call.tax_year is None and values["tax_year"]:
            pending.tool_call.tax_year = values["tax_year"]
        if pending.tool_call.period is None and values["period"]:
            pending.tool_call.period = values["period"]
        return self._attempt()

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
    def _synthesize(definition, result: CalculationResult, tool_call: SimulatedToolCall) -> str:
        lines = [f"Risultato — {definition.name}:"]
        unit = f" {definition.output_unit}" if definition.output_unit else ""
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
        lines.append(f"(Calcolo verificabile: {len(result.steps)} passaggi registrati.)")
        return "\n".join(lines)
