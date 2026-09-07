"""Guided comparator conversations through the simulated LLM layer.

The comparator cannot be filled by one-shot extraction (a list of offers in
a single sentence is real-LLM work), so the simulation collects offers one
message at a time: shared applicant/consumption facts first, then one offer
per message, closed by a finish word ('confronta'). These tests script the
exact demo conversations, including the never-guess refusals.
"""

from app.core.engine import CalculationEngine
from app.core.registry import CalculatorRegistry
from app.main import FORMULA_PACKS_DIR, PARAMETERS_DIR
from app.resolvers.parameter_store import ParameterStore
from simulation.conversation import SimulatedConversation

engine = CalculationEngine(
    CalculatorRegistry(FORMULA_PACKS_DIR, enable_drafts=True),
    ParameterStore(PARAMETERS_DIR),
)


def _conversation():
    return SimulatedConversation(engine)


def test_full_policy_comparison_conversation_matches_pack_example():
    chat = _conversation()

    reply = chat.send("confronta due polizze auto, conducente di 22 anni")
    assert reply.kind == "question"
    assert "prima offerta" in reply.text  # age was bound, straight to offers

    reply = chat.send(
        "Polizza A: premio 450 euro, franchigia 150, kasko sì, cristalli sì, "
        "guida esclusiva, sconto no sinistri 10% e sconto fedeltà 5%, app sì, voto 4,2"
    )
    assert reply.kind == "question"
    assert "Registrata offerta 1" in reply.text
    assert "Polizza A" in reply.text

    reply = chat.send(
        "Polizza B: premio 500 euro, franchigia 400, kasko sì, sconto no sinistri 10%, voto 3,8"
    )
    assert reply.kind == "question"
    assert "Registrata offerta 2" in reply.text

    reply = chat.send("confronta")
    assert reply.kind == "answer", reply.text
    assert reply.calculation.status == "success"
    assert reply.calculation.result["best"] == "Polizza A"
    totals = [e["total_score"] for e in reply.calculation.result["ranking"]]
    assert totals == ["70.67", "49.78"]  # the pack's own worked example
    # A clear winner may be named, and only "under the configured model".
    assert reply.calculation.result["comparison"]["decision_status"] == "clear_winner"
    assert "Polizza A e in testa in modo netto secondo il modello configurato" in reply.text
    # The calculated annual premium leads the synthetic score, and the
    # score's relativity is stated.
    assert reply.text.index("premio annuo calcolato 1326.45 EUR/anno") < reply.text.index(
        "punteggio totale 70.67"
    )
    assert "non e una misura oggettiva di mercato" in reply.text
    assert "pesi" in reply.text.lower()  # the demo-weights warning is surfaced
    # Defaults were applied to scored fields, so nothing may read as final.
    assert "PROVVISORIO" in reply.text
    assert "Non incluso:" not in reply.text
    assert "Metodo:" not in reply.text
    assert "Calcolo verificabile" not in reply.text
    # The presentation is shortened, not the structured audit payload.
    assert reply.calculation.exclusions
    assert reply.calculation.methodology


def test_complete_identical_policy_offers_produce_effective_tie():
    chat = _conversation()
    opening = chat.send(
        "Confronta due polizze auto per un conducente di 40 anni senza sinistri."
    )
    assert opening.kind == "question"

    offer = (
        "premio 500 euro, franchigia 200, massimale 5.000.000, kasko sì, "
        "cristalli sì, infortuni sì, assistenza stradale sì, guida esclusiva sì, "
        "sconto no sinistri 10%, sconto fedeltà 5%, telemedicina sì, app sì, voto 4,5."
    )
    assert chat.send(f"Alfa: {offer}").kind == "question"
    assert chat.send(f"Beta: {offer}").kind == "question"

    reply = chat.send("confronta")
    assert reply.kind == "answer"
    comparison = reply.calculation.result["comparison"]
    assert comparison["decision_status"] == "effective_tie"
    assert comparison["best_candidates"] == ["Alfa", "Beta"]
    assert comparison["score_gap"] == "0.00"
    assert comparison["scoring_completeness"] == "1.0000"
    assert comparison["scoring_defaults_applied"] == []


def test_comparison_asks_for_missing_shared_facts_first():
    chat = _conversation()
    reply = chat.send("quale polizza assicurativa mi conviene tra queste")
    assert reply.kind == "question"
    assert "eta_conducente" in reply.text

    reply = chat.send("il conducente ha 40 anni")
    assert reply.kind == "question"
    assert "prima offerta" in reply.text


def test_comparison_refuses_fewer_than_two_offers():
    chat = _conversation()
    chat.send("confronta queste polizze auto, conducente di 40 anni")
    chat.send("Polizza Unica: premio 300 euro")
    reply = chat.send("confronta")
    assert reply.kind == "question"
    assert "almeno 2" in reply.text


def test_offer_with_unlabeled_optional_numbers_is_not_guessed():
    chat = _conversation()
    chat.send("confronta due polizze auto, conducente di 30 anni")
    # Premium and franchise are optional. Bare numbers must still not be
    # guessed into either field; only the explicit leading name is recorded.
    reply = chat.send("Polizza X: 450 e 150")
    assert reply.kind == "question"
    assert "Registrata offerta 1: nome=Polizza X" in reply.text
    assert "premio_annuo=" not in reply.text
    assert "franchigia=" not in reply.text


def test_offer_name_accepts_comma_before_first_labeled_field():
    chat = _conversation()
    chat.send("confronta due polizze auto, conducente di 35 anni")

    reply = chat.send(
        "Generali GroupAma, premio annuo 500 euro, franchigia 300 euro, "
        "senza kasko, senza cristalli, senza infortuni, senza assistenza stradale, "
        "guida esclusiva sì, sconto no sinistri 0%, sconto fedeltà 0%, "
        "senza telemedicina, senza app e voto utenti 3."
    )

    assert reply.kind == "question"
    assert "Registrata offerta 1" in reply.text
    assert "nome=Generali GroupAma" in reply.text
    assert "premio_annuo=500" in reply.text
    assert "franchigia=300" in reply.text


def test_unlabeled_field_list_is_not_mistaken_for_comma_separated_name():
    chat = _conversation()
    chat.send("confronta due polizze auto, conducente di 35 anni")

    reply = chat.send("premio 500 euro, franchigia 300 euro")

    assert reply.kind == "question"
    assert "manca ancora: nome" in reply.text


def test_gas_luce_conversation_collects_consumption_then_offers():
    chat = _conversation()
    reply = chat.send("confronta due offerte gas e luce e dimmi quale conviene")
    assert reply.kind == "question"
    assert "consumo" in reply.text.lower()

    reply = chat.send("consumo annuo 2700 kWh di luce e 1200 Smc di gas")
    assert reply.kind == "question"
    assert "prima offerta" in reply.text

    reply = chat.send(
        "Fornitore X: luce 0,25 euro al kWh, gas 1,10 euro a Smc, costo fisso 10 euro al mese, "
        "sconto primo anno 10%, sconto rid 5%, vincolo di 12 mesi, energia verde, app sì, voto 4,5"
    )
    assert reply.kind == "question", reply.text
    assert "Registrata offerta 1" in reply.text

    reply = chat.send(
        "Fornitore Y: luce 0,22 euro al kWh, gas 1,05 euro a Smc, costo fisso 12 euro al mese, "
        "vincolo di 24 mesi, penale 50 euro, voto 3,9"
    )
    assert reply.kind == "question", reply.text
    assert "Registrata offerta 2" in reply.text

    reply = chat.send("confronta")
    assert reply.kind == "answer", reply.text
    assert reply.calculation.status == "success"
    assert reply.calculation.result["best"] == "Fornitore X"
    by_label = {e["label"]: e for e in reply.calculation.result["ranking"]}
    assert by_label["Fornitore X"]["derived"]["costo_annuo_scontato"] == "1797.75"
    assert by_label["Fornitore Y"]["derived"]["costo_annuo_scontato"] == "1998.00"
