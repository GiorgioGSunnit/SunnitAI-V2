"""End-to-end tests of the simulated LLM<->platform loop, focused on one
formula (legal_it.irpef) exercised to full depth: happy path, regime
selection, clarification loop, invalid-value correction, ambiguity and
no-match honesty, and the catalog artifact for the future system prompt.
"""

from decimal import Decimal

from app.main import engine
from simulation.catalog import render_catalog
from simulation.conversation import SimulatedConversation
from simulation.scripted_llm import extract_values, parse_number


def _conversation():
    return SimulatedConversation(engine)


# ---------------------------------------------------------------------------
# Value extraction primitives
# ---------------------------------------------------------------------------

def test_parse_number_handles_italian_formats():
    assert parse_number("42000") == Decimal("42000")
    assert parse_number("42.000") == Decimal("42000")
    assert parse_number("42.000,50") == Decimal("42000.50")
    assert parse_number("0,5") == Decimal("0.5")
    assert parse_number("-100") == Decimal("-100")


def test_extract_values_separates_tax_year_from_amounts():
    values = extract_values("un reddito di 42000 euro nel 2026")
    assert values["tax_year"] == 2026
    assert values["numbers"] == [Decimal("42000")]


def test_extract_values_finds_period_dates():
    values = extract_values("dal 2025-10-01 al 2026-03-31")
    assert values["period"] == {"start_date": "2025-10-01", "end_date": "2026-03-31"}


def test_extract_values_keeps_dates_in_appearance_order():
    # The period pair is sorted, but binding to date inputs must follow the
    # sentence's own order ("scadenza X pagamento Y" == declaration order).
    values = extract_values("scadenza 2026-07-01 pagamento 2026-06-16")
    assert values["dates"] == ["2026-07-01", "2026-06-16"]
    assert values["period"] == {"start_date": "2026-06-16", "end_date": "2026-07-01"}


def test_extract_values_marks_monthly_amounts_and_named_booleans():
    values = extract_values("affitto da 900 euro al mese, prima registrazione")
    assert values["numbers"] == [Decimal("900")]
    assert values["amount_frequency"] == "monthly"
    assert values["boolean_hints"]["first_registration"] is True


def test_extract_values_marks_cedolare_secca_as_named_boolean():
    values = extract_values("contratto con cedolare secca")
    assert values["boolean_hints"]["cedolare_secca"] is True


# ---------------------------------------------------------------------------
# IRPEF end to end — the "one solid formula" exercised in full
# ---------------------------------------------------------------------------

def test_irpef_happy_path_single_sentence():
    reply = _conversation().send("quanto pago di tasse su un reddito di 42000 euro nel 2026")
    assert reply.kind == "answer"
    assert reply.calculation.result["gross_tax"] == 11060.00
    assert reply.tool_call.calculator_id == "legal_it.irpef"
    assert reply.tool_call.tax_year == 2026
    assert "Fonti:" in reply.text
    assert "Avvertenze:" in reply.text


def test_irpef_regime_switches_with_year_in_sentence():
    reply = _conversation().send("quanto pago di tasse su un reddito di 42000 euro nel 2024")
    assert reply.kind == "answer"
    assert reply.calculation.result["gross_tax"] == 11340.00  # 35% bracket, not 33%


def test_irpef_missing_income_triggers_clarification_then_completes():
    conversation = _conversation()
    first = conversation.send("quanto pago di irpef nel 2026?")
    assert first.kind == "question"
    assert "taxable_income" in first.text or "Reddito" in first.text

    second = conversation.send("il reddito è di 42000 euro")
    assert second.kind == "answer"
    assert second.calculation.result["gross_tax"] == 11060.00


def test_irpef_invalid_value_is_reported_and_correctable():
    conversation = _conversation()
    first = conversation.send("calcolo irpef 2026 su un reddito di -100 euro")
    assert first.kind == "question"
    assert "taxable_income" in first.text  # platform's own validation message relayed

    second = conversation.send("scusa, intendevo 42000")
    assert second.kind == "answer"
    assert second.calculation.result["gross_tax"] == 11060.00


def test_irpef_answer_carries_full_audit_payload():
    reply = _conversation().send("calcolo irpef su 42000 euro nel 2026")
    calc = reply.calculation
    assert calc.parameters_used["brackets"]["origin"] == "parameter_store"
    assert calc.date_resolution["source"] == "derived_from_tax_year"
    assert calc.formula_version == "1"
    assert len(calc.steps) == 2  # two brackets touched at 42000


# ---------------------------------------------------------------------------
# The unhappy paths a routing layer must handle honestly
# ---------------------------------------------------------------------------

def test_unrelated_sentence_is_refused_not_guessed():
    reply = _conversation().send("che tempo fa domani a Milano?")
    assert reply.kind == "no_match"
    assert reply.calculation is None


def test_ambiguous_sentence_asks_user_to_choose():
    reply = _conversation().send("quanto costa")
    assert reply.kind == "ambiguous"
    assert "più calcoli" in reply.text


def test_period_calculator_asks_for_period_then_completes():
    conversation = _conversation()
    first = conversation.send("interessi legali su un capitale di 10000 euro")
    assert first.kind == "question"
    assert "YYYY-MM-DD" in first.text

    second = conversation.send("dal 2025-10-01 al 2026-03-31")
    assert second.kind == "answer"
    assert second.calculation.result["interest"] == 89.86
    assert len(second.calculation.steps) == 2  # split across the rate change


def test_registration_tax_normalizes_monthly_rent_before_calculation():
    reply = _conversation().send(
        "imposta di registro per affitto da 800 euro al mese per 4 anni, prima registrazione"
    )
    assert reply.kind == "answer"
    assert reply.tool_call.inputs["annual_rent"] == Decimal("9600")
    assert reply.tool_call.inputs["years"] == Decimal("4")
    assert reply.tool_call.inputs["first_registration"] is True
    assert reply.calculation.result["tax_due"] == 768.00


def test_registration_tax_cedolare_secca_hint_sets_optional_flag():
    reply = _conversation().send(
        "imposta di registro per affitto da 800 euro al mese per 4 anni, prima registrazione, cedolare secca"
    )
    assert reply.kind == "answer"
    assert reply.tool_call.inputs["cedolare_secca"] is True
    assert reply.calculation.result["tax_due"] == 0.00


# ---------------------------------------------------------------------------
# Date-typed inputs bound from free text (ravvedimento, termini processuali)
# ---------------------------------------------------------------------------

def test_ravvedimento_one_shot_conversation():
    reply = _conversation().send(
        "ravvedimento operoso per 1000 euro di iva scadenza 2026-06-16 pagamento 2026-07-01"
    )
    assert reply.kind == "answer"
    assert reply.tool_call.inputs["scadenza_originaria"] == "2026-06-16"
    assert reply.tool_call.inputs["data_pagamento"] == "2026-07-01"
    # Assert on structure and day count, not euro amounts, so the test
    # survives parameter-table updates.
    for key in ("totale_da_versare", "sanzione_ridotta", "interessi", "tributo"):
        assert key in reply.calculation.result
    assert reply.calculation.derived_values["giorni_di_ritardo"] == 15


def test_ravvedimento_clarification_loop_completes_with_dates():
    conversation = _conversation()
    first = conversation.send("ravvedimento operoso per 1000 euro di iva")
    assert first.kind == "question"
    assert "scadenza_originaria" in first.text
    assert "data_pagamento" in first.text
    assert first.calculation is None  # nothing was guessed

    second = conversation.send("scadenza 2026-06-16, pagato il 2026-07-01")
    assert second.kind == "answer"
    assert second.calculation.derived_values["giorni_di_ritardo"] == 15


def test_termini_processuali_one_shot_conversation():
    reply = _conversation().send("termine processuale di 30 giorni dal 2026-07-01")
    assert reply.kind == "answer"
    assert reply.tool_call.inputs["data_decorrenza"] == "2026-07-01"
    assert reply.tool_call.inputs["giorni"] == 30
    assert reply.calculation.result["scadenza"] == "2026-07-31"


def test_ravvedimento_swapped_dates_fail_honestly():
    # Appearance order binds as written; a payment date before the due date
    # is the platform's structured error, relayed verbatim — never fixed up
    # or reordered by the mimic.
    reply = _conversation().send(
        "ravvedimento operoso per 1000 euro scadenza 2026-07-01 pagamento 2026-06-16"
    )
    assert reply.kind == "no_match"
    assert reply.calculation.status == "error"
    assert reply.calculation.errors[0].code == "strategy_execution_failed"


# ---------------------------------------------------------------------------
# Catalog artifact for the future system prompt
# ---------------------------------------------------------------------------

def test_catalog_renders_every_calculator_with_inputs():
    catalog = render_catalog(engine.registry.definitions())
    assert "### legal_it.irpef" in catalog
    assert "taxable_income" in catalog
    assert "richiede tax_year" in catalog
    assert "### legal_it.legal_interest" in catalog
    assert "period.start_date" in catalog


def test_catalog_lists_exclusions_so_the_llm_knows_the_limits():
    catalog = render_catalog(engine.registry.definitions())
    assert "Non copre:" in catalog
    assert "addizionale regionale" in catalog
