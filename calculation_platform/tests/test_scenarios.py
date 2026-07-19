"""Pins the three canonical demo sentences (simulation/scenarios.py) to
their expected flow outcomes, so the scripted demo can never silently
drift if keywords, matcher scoring, or extraction heuristics change."""

from app.main import engine
from simulation.conversation import SimulatedConversation
from simulation.scenarios import (
    SCENARIO_1_SENTENCE,
    SCENARIO_2_SENTENCES,
    SCENARIO_3_SENTENCE,
)


def test_scenario_1_direct_recognition_and_calculation():
    reply = SimulatedConversation(engine).send(SCENARIO_1_SENTENCE)
    assert reply.kind == "answer"
    assert reply.tool_call.calculator_id == "legal_it.irpef"
    assert reply.tool_call.tax_year == 2026
    # the payload that would go back to the LLM is complete and auditable
    assert reply.calculation.result["gross_tax"] == "11060.00"
    assert reply.calculation.citations
    assert reply.calculation.steps


def test_scenario_2_missing_variable_asks_then_completes():
    conversation = SimulatedConversation(engine)

    first = conversation.send(SCENARIO_2_SENTENCES[0])
    assert first.kind == "question"
    assert "taxable_income" in first.text
    # the question is driven by the planner's structured result, not guesswork
    assert first.plan.status == "needs_clarification"
    assert first.plan.missing_inputs == ["taxable_income"]
    assert first.plan.tax_year == 2026

    second = conversation.send(SCENARIO_2_SENTENCES[1])
    assert second.kind == "answer"
    assert second.calculation.result["gross_tax"] == "11060.00"


def test_scenario_3_ambiguity_offers_exactly_three_named_choices():
    reply = SimulatedConversation(engine).send(SCENARIO_3_SENTENCE)
    assert reply.kind == "ambiguous"
    numbered_lines = [l for l in reply.text.splitlines() if l.strip().startswith(("1)", "2)", "3)"))]
    assert len(numbered_lines) == 3
    assert "4)" not in reply.text
