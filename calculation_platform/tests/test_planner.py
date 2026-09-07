"""Tests for the deterministic planner (the hardcoded LLM/router stand-in)
and the conversation flows built on it — the required minimum cases from
the routing-metadata development spec."""

import pytest

from app.core.registry import CalculatorRegistry
from app.main import FORMULA_PACKS_DIR, engine
from simulation.conversation import SimulatedConversation
from simulation.planner import plan_sentence


def _plan(sentence: str):
    return plan_sentence(sentence, engine.registry.definitions())


def _plan_with_drafts(sentence: str):
    """Route against a drafts-enabled catalogue.

    The penal drafts are withheld from the default registry (see
    test_draft_gate.py), but the disambiguation they exercise — an
    aggravated phrasing must never fall through to the simple-offence
    frame — is matcher behaviour worth keeping under test for the day
    they are validated.
    """
    return plan_sentence(sentence, _DRAFT_DEFINITIONS)


_DRAFT_DEFINITIONS = CalculatorRegistry(FORMULA_PACKS_DIR, enable_drafts=True).definitions()


# ---------------------------------------------------------------------------
# Planner-level routing decisions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("sentence", "calculator_id", "expected_inputs"),
    [
        (
            "Calcola il totale di una fattura con netto 1.000 euro, IVA 22% e sconto 10%.",
            "business.invoice_total",
            {"net_amount": "1000", "vat_rate": "0.22", "discount_rate": "0.1"},
        ),
        (
            "Qual è la rata mensile per un prestito di 10.000 euro al 6% annuo per 12 mesi?",
            "business.loan_payment",
            {"principal": "10000", "annual_rate": "0.06", "months": 12},
        ),
        (
            "Prestito senza interessi: 12.000 euro da rimborsare in 12 mesi. Calcola la rata.",
            "business.loan_payment",
            {"principal": "12000", "annual_rate": "0", "months": 12},
        ),
        (
            "Calcola i compensi DM 55 per una causa da 30.000 euro, fasi studio, "
            "introduttiva e decisionale, con CPA e IVA.",
            "legal_it.compensi_dm55",
            {
                "valore_causa": "30000",
                "fasi": ["studio", "introduttiva", "decisionale"],
                "applica_cpa": True,
                "applica_iva": True,
            },
        ),
        (
            "Stesso calcolo DM 55: causa da 30.000 euro, fasi studio, introduttiva "
            "e decisionale, ma applica un aumento del 20%, con CPA e IVA.",
            "legal_it.compensi_dm55",
            {
                "valore_causa": "30000",
                "fasi": ["studio", "introduttiva", "decisionale"],
                "aumento_pct": "20",
                "applica_cpa": True,
                "applica_iva": True,
            },
        ),
        (
            "Calcola il contributo unificato civile per una causa da 15.000 euro in primo grado.",
            "legal_it.contributo_unificato_civile",
            {"valore_causa": "15000", "grado": "primo_grado"},
        ),
        (
            "Per una causa da 15.000 euro, calcola il contributo unificato in appello.",
            "legal_it.contributo_unificato_civile",
            {"valore_causa": "15000", "grado": "appello"},
        ),
        (
            "Calcola i contributi INPS su 2.000 euro lordi: aliquota lavoratore 9,19% "
            "e aliquota datore 23,81%.",
            "legal_it.inps_contributions",
            {
                "retribuzione_lorda": "2000",
                "aliquota_lavoratore": "0.0919",
                "aliquota_datore_lavoro": "0.2381",
            },
        ),
        (
            "Stesse aliquote INPS, ma su 3.000 euro lordi: 9,19% lavoratore e 23,81% datore.",
            "legal_it.inps_contributions",
            {
                "retribuzione_lorda": "3000",
                "aliquota_lavoratore": "0.0919",
                "aliquota_datore_lavoro": "0.2381",
            },
        ),
        (
            "Retribuzione mensile globale 2.500 euro e due mesi di preavviso non "
            "lavorati: calcola l'indennità.",
            "legal_it.notice_indemnity",
            {"retribuzione_mensile_globale": "2500", "mesi_preavviso": "2"},
        ),
        (
            "Calcola l'imposta di registro per un affitto annuo di 9.600 euro, durata "
            "4 anni, prima registrazione, senza cedolare secca.",
            "legal_it.registration_tax_leases",
            {
                "annual_rent": "9600",
                "years": "4",
                "first_registration": True,
                "cedolare_secca": False,
            },
        ),
        (
            "Rivalutazione e interessi criterio Cassazione 1712: 1.000 euro dal "
            "15 marzo 2021 al 15 giugno 2021.",
            "legal_it.rivalutazione_interessi_1712",
            {
                "importo": "1000",
                "data_iniziale": "2021-03-15",
                "data_finale": "2021-06-15",
            },
        ),
    ],
)
def test_pm_live_prompts_bind_their_named_inputs(sentence, calculator_id, expected_inputs):
    plan = _plan(sentence)
    assert plan.status == "ready_to_calculate"
    assert plan.calculator_id == calculator_id
    assert plan.inputs == expected_inputs


def test_pm_live_irpef_prompt_keeps_the_requested_tax_year():
    plan = _plan("Calcola l'IRPEF lorda su 42.000 euro, ma per l'anno d'imposta 2024.")
    assert plan.status == "ready_to_calculate"
    assert plan.calculator_id == "legal_it.irpef"
    assert plan.inputs == {"taxable_income": "42000"}
    assert plan.tax_year == 2024


def test_complete_irpef_sentence_is_ready_to_calculate():
    plan = _plan("calcolo irpef su 42000 euro nel 2026")
    assert plan.status == "ready_to_calculate"
    assert plan.calculator_id == "legal_it.irpef"
    assert plan.inputs == {"taxable_income": "42000"}
    assert plan.normalized_inputs == {"taxable_income": "42000"}
    assert plan.extracted_values["numbers"] == ["42000"]
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
    assert plan.inputs["tributo_non_versato"] == "1000"
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
    plan = _plan("quanto pago di interessi")
    assert plan.status == "ambiguous"
    assert plan.calculator_id is None
    assert 2 <= len(plan.candidates) <= 3


def test_unrelated_requests_are_no_match():
    assert _plan("che tempo farà domani a Milano?").status == "no_match"
    assert _plan("scrivimi una lettera di disdetta").status == "no_match"
    assert _plan("quanto costa?").status == "no_match"
    assert _plan("quanto dista la luna dalla terra?").status == "no_match"


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


def test_uncovered_penal_offences_still_no_match():
    # Offences with no calculator at all must never be forced onto an
    # adjacent draft — the router stays silent rather than guessing.
    for sentence in (
        "omicidio colposo",
        "omicidio colposo pena",
        "omicidio stradale",
    ):
        plan = _plan(sentence)
        assert plan.status == "no_match"
        assert plan.calculator_id is None
        assert plan.candidates == []


def test_aggravated_offences_route_to_the_aggravated_draft_not_the_simple_one():
    # Aggravated phrasings now have their own draft calculators; they must
    # route there and NEVER fall through to the simple-offence draft (whose
    # statutory frame would understate the penalty).
    for sentence, expected in (
        ("furto in abitazione", "legal_it.furto_aggravato_draft"),
        ("pena per furto in abitazione", "legal_it.furto_aggravato_draft"),
        ("furto in casa", "legal_it.furto_aggravato_draft"),
        ("rapina a mano armata pena", "legal_it.rapina_aggravata_draft"),
    ):
        plan = _plan_with_drafts(sentence)
        assert plan.status == "needs_clarification"
        assert plan.calculator_id == expected


def test_penal_simple_offences_still_route_to_their_drafts():
    furto = _plan_with_drafts("pena per furto")
    assert furto.status == "needs_clarification"
    assert furto.calculator_id == "legal_it.furto_pena_draft"

    omicidio = _plan_with_drafts("che pena rischia per omicidio")
    assert omicidio.status == "needs_clarification"
    assert omicidio.calculator_id == "legal_it.omicidio_pena_draft"


def test_furto_narrative_past_tense_routes_to_simple_draft():
    plan = _plan_with_drafts(
        "il mio assistito ha rubato un portafoglio al mercato, che pena rischia?"
    )
    assert plan.status == "needs_clarification"
    assert plan.calculator_id == "legal_it.furto_pena_draft"
    assert plan.missing_inputs == ["aggravanti_comuni", "attenuanti_comuni"]


def test_furto_home_variants_remain_blocked():
    for sentence in (
        "furto in casa",
        "rubare in casa",
        "ha rubato in casa",
    ):
        plan = _plan(sentence)
        assert plan.calculator_id != "legal_it.furto_pena_draft"
        assert all(
            candidate.calculator_id != "legal_it.furto_pena_draft"
            for candidate in plan.candidates
        )


def test_lawyer_fee_queries_route_to_dm55():
    precise = _plan("compenso avvocato per una causa civile da 30000 euro")
    assert precise.status == "needs_clarification"
    assert precise.calculator_id == "legal_it.compensi_dm55"
    assert precise.missing_inputs == ["fasi"]

    vague = _plan("quanto costa l'avvocato per una causa da 30000 euro")
    assert vague.status in ("needs_clarification", "ambiguous")
    if vague.status == "needs_clarification":
        assert vague.calculator_id == "legal_it.compensi_dm55"
    else:
        assert vague.candidates[0].calculator_id == "legal_it.compensi_dm55"


def test_tfr_retribution_queries_route_to_tfr():
    for sentence in (
        "calcolo tfr per retribuzione di 30000 euro e 10 anni",
        "tfr per una retribuzione annua di 30000 euro",
    ):
        plan = _plan(sentence)
        assert plan.status == "ready_to_calculate"
        assert plan.calculator_id == "legal_it.tfr"


def test_contributo_unificato_queries_remain_on_court_tax():
    for sentence in (
        "contributo unificato per una causa da 30000 euro",
        "devo iscrivere a ruolo una causa da 80000 euro in tribunale, quanto pago?",
    ):
        plan = _plan(sentence)
        assert plan.status == "ready_to_calculate"
        assert plan.calculator_id == "legal_it.contributo_unificato_civile"


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
    assert plan.extracted_values["numbers"] == ["800", "4"]
    assert plan.extracted_values["amount_frequency"] == "monthly"
    assert plan.inputs["annual_rent"] == "9600"
    assert plan.normalized_inputs["annual_rent"] == "9600"
    assert plan.inputs["years"] == "4"
    assert plan.inputs["first_registration"] is True


def test_planner_normalizes_percent_values_for_rate_inputs():
    plan = _plan("totale fattura 1000 euro con iva 22")
    assert plan.status == "ready_to_calculate"
    assert plan.calculator_id == "business.invoice_total"
    assert plan.inputs["net_amount"] == "1000"
    assert plan.inputs["vat_rate"] == "0.22"


def test_planner_keeps_decimal_rate_values_when_already_canonical():
    plan = _plan("totale fattura 1000 euro con iva 0,22")
    assert plan.status == "ready_to_calculate"
    assert plan.calculator_id == "business.invoice_total"
    assert plan.inputs["vat_rate"] == "0.22"


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
    assert second.calculation.result["gross_tax"] == "11060.00"


def test_conversation_period_followup_completes_and_splits_rates():
    conversation = SimulatedConversation(engine)
    first = conversation.send("interessi legali su 10000 euro")
    assert first.kind == "question"
    assert "YYYY-MM-DD" in first.text
    second = conversation.send("dal 2025-10-01 al 2026-03-31")
    assert second.kind == "answer"
    assert second.calculation.result["interest"] == "89.86"
    assert len(second.calculation.steps) == 2  # two rate segments


def test_conversation_reply_carries_the_plan():
    conversation = SimulatedConversation(engine)
    reply = conversation.send("calcolo irpef su 42000 euro nel 2026")
    assert reply.plan is not None
    assert reply.plan.status == "ready_to_calculate"
    assert reply.calculation.result["gross_tax"] == "11060.00"


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
