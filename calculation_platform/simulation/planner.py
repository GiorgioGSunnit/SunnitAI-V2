"""Deterministic planner — the hardcoded stand-in for the LLM/router.

Takes a user sentence and produces the structured planning result the
real LLM/router will eventually produce: which calculator to use, the
named inputs it could extract, what is still missing (with a ready-made
clarification question), or an honest "ambiguous"/"no_match". It never
computes anything — the calculation stays entirely in the engine.

Statuses:
  ready_to_calculate — calculator identified, every required input bound.
  needs_clarification — calculator identified, but required inputs (or
      the period, for date-based calculators) are missing; `question`
      holds a deterministic clarification prompt.
  ambiguous — several calculators tie; `candidates` lists at most 3.
  no_match — nothing matched; the platform does not guess.

Like everything in simulation/, this is a dev/test artifact: the value
extraction is deliberately naive (see scripted_llm.py) and the whole
module is replaced by the real LLM at integration time — but the
PlanResult *shape* is the contract that integration will keep.
"""

from typing import Any, Dict, Iterable, List, Optional

from pydantic import BaseModel, Field

from app.core.matcher import match_query
from app.core.result_builder import to_jsonable
from app.schemas.calculator_definition import CalculatorDefinition
from app.schemas.match_result import MatchCandidate

from .scripted_llm import bind_values, extract_values

_MAX_AMBIGUOUS_CANDIDATES = 3
_HIGH_CONFIDENCE_MARGIN = 2


class PlanResult(BaseModel):
    status: str  # ready_to_calculate | needs_clarification | ambiguous | no_match
    calculator_id: Optional[str] = None
    extracted_values: Dict[str, Any] = Field(default_factory=dict)
    normalized_inputs: Dict[str, Any] = Field(default_factory=dict)
    inputs: Dict[str, Any] = Field(default_factory=dict)
    tax_year: Optional[int] = None
    period: Optional[Dict[str, str]] = None
    missing_inputs: List[str] = Field(default_factory=list)
    question: Optional[str] = None
    clarification_questions: List[str] = Field(default_factory=list)
    confidence: Optional[str] = None  # high | medium
    matched_terms: List[str] = Field(default_factory=list)
    candidates: List[MatchCandidate] = Field(default_factory=list)
    required_context: List[str] = Field(default_factory=list)


def _confidence(candidates: List[MatchCandidate]) -> str:
    top = candidates[0]
    has_phrase_evidence = any(" " in term for term in top.matched_terms)
    clear_margin = len(candidates) == 1 or top.score - candidates[1].score >= _HIGH_CONFIDENCE_MARGIN
    return "high" if has_phrase_evidence and clear_margin else "medium"


def _clarification_question(definition: CalculatorDefinition, missing: List[str]) -> str:
    parts = []
    for name in missing:
        if name == "period":
            parts.append("il periodo di riferimento (date di inizio e fine, formato YYYY-MM-DD)")
            continue
        spec = next((s for s in definition.inputs if s.name == name), None)
        if spec and spec.description:
            parts.append(f"'{spec.description}' ({name})")
        else:
            parts.append(name)
    if len(parts) == 1:
        return f"Per calcolare '{definition.name}' serve ancora: {parts[0]}. Puoi indicarlo?"
    return f"Per calcolare '{definition.name}' servono ancora: {'; '.join(parts)}. Puoi indicarli?"


def plan_sentence(sentence: str, definitions: Iterable[CalculatorDefinition]) -> PlanResult:
    definitions = list(definitions)
    response = match_query(sentence, definitions)

    if response.status == "no_match":
        return PlanResult(status="no_match")

    if response.status == "ambiguous":
        return PlanResult(
            status="ambiguous",
            candidates=response.candidates[:_MAX_AMBIGUOUS_CANDIDATES],
        )

    top = response.candidates[0]
    definition = next(d for d in definitions if d.id == top.calculator_id)

    values = extract_values(sentence)
    inputs: Dict[str, Any] = {}
    bind_values(definition, inputs, values)

    missing = [
        spec.name for spec in definition.inputs
        if spec.required and spec.name not in inputs
    ]
    if definition.strategy == "date_split_interest" and values["period"] is None:
        missing.append("period")

    common = {
        "calculator_id": definition.id,
        "extracted_values": to_jsonable(values),
        "normalized_inputs": to_jsonable(inputs),
        "inputs": to_jsonable(inputs),
        "tax_year": values["tax_year"],
        "period": values["period"],
        "matched_terms": top.matched_terms,
        "required_context": definition.required_context,
    }

    if missing:
        return PlanResult(
            status="needs_clarification",
            missing_inputs=missing,
            question=_clarification_question(definition, missing),
            clarification_questions=[_clarification_question(definition, missing)],
            **common,
        )

    return PlanResult(
        status="ready_to_calculate",
        confidence=_confidence(response.candidates),
        **common,
    )
