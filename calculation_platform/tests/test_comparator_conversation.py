"""Guided comparator conversations through the simulated LLM layer.

The comparator cannot be filled by one-shot extraction (a list of offers in
a single sentence is real-LLM work), so the simulation collects offers one
message at a time: shared applicant/consumption facts first, then one offer
per message, closed by a finish word ('confronta'). These tests script the
exact demo conversations, including the never-guess refusals.
"""

from app.main import engine
from simulation.conversation import SimulatedConversation


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
    assert totals == ["85.34", "67.39"]  # the pack's own worked example
    # A clear winner may be named, and only "under the configured model".
    assert reply.calculation.result["comparison"]["decision_status"] == "clear_winner"
    assert "Polizza A e in testa in modo netto secondo il modello configurato" in reply.text
    # Money leads the synthetic score, and the score's relativity is stated.
    assert reply.text.index("costo stimato 382.50") < reply.text.index("punteggio totale 85.34")
    assert "non una misura oggettiva di mercato" in reply.text
    assert "pesi" in reply.text.lower()  # the demo-weights warning is surfaced
    # Defaults were applied to scored fields, so nothing may read as final.
    assert "PROVVISORIO" in reply.text
    assert "Non incluso:" in reply.text


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


def test_offer_with_unlabeled_numbers_is_not_guessed():
    chat = _conversation()
    chat.send("confronta due polizze auto, conducente di 30 anni")
    # premium AND franchise as bare numbers -> ambiguous -> must ask, not map
    reply = chat.send("Polizza X: 450 e 150")
    assert reply.kind == "question"
    assert "manca ancora" in reply.text
    assert "premio_annuo" in reply.text


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
