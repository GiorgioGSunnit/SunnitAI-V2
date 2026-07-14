"""Tests for the deterministic planner (the hardcoded LLM/router stand-in)
and the conversation flows built on it — the required minimum cases from
the routing-metadata development spec."""

from app.main import engine
from simulation.conversation import SimulatedConversation
from simulation.planner import plan_sentence


def _plan(sentence: str):
    return plan_sentence(sentence, engine.registry.definitions())


# ---------------------------------------------------------------------------
# Planner-level routing decisions
# ---------------------------------------------------------------------------

def test_complete_irpef_sentence_is_ready_to_calculate():
    plan = _plan("calcolo irpef su 42000 euro nel 2026")
    assert plan.status == "ready_to_calculate"
    assert plan.calculator_id == "legal_it.irpef"
    assert plan.inputs == {"taxable_income": 42000.0}
    assert plan.normalized_inputs == {"taxable_income": 42000.0}
    assert plan.extracted_values["numbers"] == [42000.0]
    assert plan.tax_year == 2026
    assert plan.missing_inputs == []
    assert plan.confidence in ("high", "medium")
    assert plan.matched_terms


def test_irpef_without_income_needs_clarification():
    plan = _plan("quanto pago di IRPEF nel 2026?")
    assert plan.status == "needs_clarification"
    assert plan.calculator_id == "legal_it.irpef"
    assert plan.missing_inputs == ["taxable_income"]
    assert plan.tax_year == 2026
    assert plan.question and "taxable_income" in plan.question
    assert plan.clarification_questions == [plan.question]


def test_legal_interest_without_period_needs_clarification():
    plan = _plan("interessi legali su 10000 euro")
    assert plan.status == "needs_clarification"
    assert plan.calculator_id == "legal_it.legal_interest"
    assert plan.missing_inputs == ["period"]
    assert "YYYY-MM-DD" in plan.question


def test_ravvedimento_one_shot_is_ready_to_calculate():
    plan = _plan("ravvedimento operoso per 1000 euro di iva scadenza 2026-06-16 pagamento 2026-07-01")
    assert plan.status == "ready_to_calculate"
    assert plan.calculator_id == "legal_it.ravvedimento_operoso"
    assert plan.inputs["tributo_non_versato"] == 1000.0
    assert plan.inputs["scadenza_originaria"] == "2026-06-16"
    assert plan.inputs["data_pagamento"] == "2026-07-01"
    assert plan.missing_inputs == []


def test_ravvedimento_without_dates_needs_clarification():
    plan = _plan("ravvedimento operoso per 1000 euro di iva")
    assert plan.status == "needs_clarification"
    assert plan.calculator_id == "legal_it.ravvedimento_operoso"
    assert plan.missing_inputs == ["scadenza_originaria", "data_pagamento"]


def test_termini_processuali_one_shot_is_ready_to_calculate():
    plan = _plan("termine processuale di 30 giorni dal 2026-07-01")
    assert plan.status == "ready_to_calculate"
    assert plan.calculator_id == "legal_it.termini_processuali_civili"
    assert plan.inputs["data_decorrenza"] == "2026-07-01"
    assert plan.inputs["giorni"] == 30
    assert plan.missing_inputs == []


def test_optional_date_inputs_are_not_auto_filled():
    # A third date in the sentence must not be guessed into the optional
    # termine_dichiarazione — silently filling an optional legal field is
    # a hidden default.
    plan = _plan(
        "ravvedimento operoso per 1000 euro scadenza 2026-06-16 "
        "pagamento 2026-07-01 dichiarazione 2026-10-31"
    )
    assert plan.status == "ready_to_calculate"
    assert plan.calculator_id == "legal_it.ravvedimento_operoso"
    assert "termine_dichiarazione" not in plan.inputs


def test_vague_sentence_is_ambiguous_not_guessed():
    plan = _plan("quanto costa?")
    assert plan.status == "ambiguous"
    assert plan.calculator_id is None
    assert 2 <= len(plan.candidates) <= 3


def test_unrelated_requests_are_no_match():
    assert _plan("che tempo farà domani a Milano?").status == "no_match"
    assert _plan("scrivimi una lettera di disdetta").status == "no_match"


def test_negative_example_blocks_cedolare_secca_misrouting():
    # Asking about the cedolare secca tax itself must NOT route to the
    # registration-tax calculator (whose keywords include "cedolare secca"
    # only to describe its exemption case).
    plan = _plan("cedolare secca quanto pago")
    if plan.status in ("ready_to_calculate", "needs_clarification"):
        assert plan.calculator_id != "legal_it.registration_tax_leases"
    elif plan.status == "ambiguous":
        top_ids = [c.calculator_id for c in plan.candidates]
        assert "legal_it.registration_tax_leases" not in top_ids


def test_genuine_registration_tax_query_survives_the_negative_example():
    # The penalty must not knock out a query that clearly IS about
    # registration tax, even when it also mentions cedolare secca.
    plan = _plan("imposta di registro per un affitto")
    assert plan.status in ("ready_to_calculate", "needs_clarification")
    assert plan.calculator_id == "legal_it.registration_tax_leases"


def test_intent_example_routes_tasse_sul_reddito_to_irpef():
    plan = _plan("tasse sul reddito")
    assert plan.status == "needs_clarification"  # no amount given
    assert plan.calculator_id == "legal_it.irpef"


def test_high_confidence_requires_phrase_evidence_and_margin():
    plan = _plan("calcolo irpef su 42000 euro nel 2026")
    assert plan.confidence == "high"


def test_required_context_is_surfaced():
    plan = _plan("quanto pago di irpef nel 2026?")
    assert any("anno d'imposta" in c for c in plan.required_context)


def test_planner_normalizes_monthly_lease_amount_to_annual_rent():
    plan = _plan("imposta di registro per affitto da 800 euro al mese per 4 anni, prima registrazione")
    assert plan.status == "ready_to_calculate"
    assert plan.calculator_id == "legal_it.registration_tax_leases"
    assert plan.extracted_values["numbers"] == [800.0, 4.0]
    assert plan.extracted_values["amount_frequency"] == "monthly"
    assert plan.inputs["annual_rent"] == 9600.0
    assert plan.normalized_inputs["annual_rent"] == 9600.0
    assert plan.inputs["years"] == 4.0
    assert plan.inputs["first_registration"] is True


def test_planner_normalizes_percent_values_for_rate_inputs():
    plan = _plan("totale fattura 1000 euro con iva 22")
    assert plan.status == "ready_to_calculate"
    assert plan.calculator_id == "business.invoice_total"
    assert plan.inputs["net_amount"] == 1000.0
    assert plan.inputs["vat_rate"] == 0.22


def test_planner_keeps_decimal_rate_values_when_already_canonical():
    plan = _plan("totale fattura 1000 euro con iva 0,22")
    assert plan.status == "ready_to_calculate"
    assert plan.calculator_id == "business.invoice_total"
    assert plan.inputs["vat_rate"] == 0.22


def test_planner_never_computes_anything():
    # The plan for a fully-specified sentence carries inputs, never results.
    plan = _plan("calcolo irpef su 42000 euro nel 2026")
    assert not hasattr(plan, "result")
    assert "gross_tax" not in plan.inputs


# ---------------------------------------------------------------------------
# Conversation flows driven by the planner
# ---------------------------------------------------------------------------

def test_conversation_clarification_followup_completes_irpef():
    conversation = SimulatedConversation(engine)
    first = conversation.send("quanto pago di IRPEF nel 2026?")
    assert first.kind == "question"
    second = conversation.send("42000 euro")
    assert second.kind == "answer"
    assert second.calculation.result["gross_tax"] == 11060.00


def test_conversation_period_followup_completes_and_splits_rates():
    conversation = SimulatedConversation(engine)
    first = conversation.send("interessi legali su 10000 euro")
    assert first.kind == "question"
    assert "YYYY-MM-DD" in first.text
    second = conversation.send("dal 2025-10-01 al 2026-03-31")
    assert second.kind == "answer"
    assert second.calculation.result["interest"] == 89.86
    assert len(second.calculation.steps) == 2  # two rate segments


def test_conversation_reply_carries_the_plan():
    conversation = SimulatedConversation(engine)
    reply = conversation.send("calcolo irpef su 42000 euro nel 2026")
    assert reply.plan is not None
    assert reply.plan.status == "ready_to_calculate"
    assert reply.calculation.result["gross_tax"] == 11060.00


# ---------------------------------------------------------------------------
# Metadata sufficiency — every pack must be routable
# ---------------------------------------------------------------------------

def test_every_formula_pack_has_routing_metadata():
    for definition in engine.registry.definitions():
        assert definition.keywords, f"{definition.id} has no keywords"
        assert definition.tags, f"{definition.id} has no tags"
        assert definition.intent_examples, f"{definition.id} has no intent_examples"


def test_every_intent_example_routes_to_its_own_calculator():
    # Self-consistency: a calculator's own intent examples must reach it
    # (ready or needs_clarification), never no_match or a different winner.
    for definition in engine.registry.definitions():
        for example in definition.intent_examples:
            plan = _plan(example)
            assert plan.status != "no_match", f"{definition.id}: {example!r} matched nothing"
            if plan.status in ("ready_to_calculate", "needs_clarification"):
                assert plan.calculator_id == definition.id, (
                    f"{definition.id}: intent example {example!r} routed to {plan.calculator_id}"
                )
