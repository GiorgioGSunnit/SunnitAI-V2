"""ROUTER_ENABLED: narrative | direct | ccnl | document, deterministic first (src/rag/query_router.py).
Invented texts only."""
import os

os.environ.setdefault("EMBEDDING_PROVIDER", "openai")
os.environ.setdefault("EMBEDDING_API_KEY", "test-key")
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test-password")

import pytest  # noqa: E402

from src.rag import query_router as qr  # noqa: E402

NARRATIVES = [
    "Tizio acquista da Caio un'auto usata il 3 marzo 2024. Dopo due settimane il motore si guasta e Tizio "
    "porta l'auto in officina, dove scopre che il contachilometri era stato manomesso. Caio rifiuta ogni "
    "rimborso. Il candidato, assunte le vesti del legale di Tizio, individui le azioni esperibili.",
    "Mia madre è stata ricoverata nell'ospedale di zona per un intervento al ginocchio. Durante il ricovero "
    "ha contratto un'infezione e il medico non ha disposto alcun esame per tre giorni. È stata dimessa "
    "con febbre alta e poi ricoverata di nuovo. Quali azioni possiamo intraprendere contro la struttura?",
    "La società Alfa S.r.l. ha consegnato alla Beta S.p.A. un macchinario difettoso. Beta ha contestato i "
    "vizi con una PEC del 10 gennaio, ma Alfa non ha risposto. Successivamente Beta ha sospeso i pagamenti. "
    "Qual è la strategia migliore per Beta?",
]
DIRECTS = [
    "Qual è il termine per proporre appello contro una sentenza penale?",
    "Quali sono i requisiti della legittima difesa?",
    "Come funziona l'accettazione tacita dell'eredità?",
    "Cosa prevede l'art. 1490 c.c. sulla garanzia per vizi?",
    "Entro quanto si prescrive il diritto al risarcimento del danno da fatto illecito?",
]


@pytest.fixture(autouse=True)
def no_llm(monkeypatch):
    calls = []

    def fake_tie(text, lean):
        calls.append(text)
        return lean, "test: leaning"
    monkeypatch.setattr(qr, "_tie_break", fake_tie)
    qr._cache.clear()
    return calls


@pytest.mark.parametrize("text", NARRATIVES)
def test_a_fact_pattern_with_a_request_is_a_narrative_without_the_llm(text, no_llm):
    result = qr.classify(text, log=False)
    assert result["route"] == "narrative", result["reasons"]
    assert not result["tie_break"] and no_llm == []


@pytest.mark.parametrize("text", DIRECTS)
def test_a_question_about_a_norm_is_direct_without_the_llm(text, no_llm):
    result = qr.classify(text, log=False)
    assert result["route"] == "direct", result["reasons"]
    assert no_llm == []


@pytest.mark.parametrize("text", [
    "Quanti giorni di ferie spettano secondo il CCNL Commercio?",
    "Sono un dipendente a tempo pieno: quanti giorni di preavviso devo dare se mi dimetto?",
])
def test_a_collective_agreement_or_an_employment_question_on_its_matters_is_ccnl(text):
    assert qr.classify(text, log=False)["route"] == "ccnl"


def test_a_document_keeps_its_path():
    assert qr.classify("Riassumi il contratto allegato", has_document=True, log=False)["route"] == "document"


def test_facts_without_a_request_go_to_the_tie_break(no_llm):
    text = ("Tizio vende a Caio un terreno nel 2019. Nel 2021 Caio scopre che sul terreno grava una servitù di "
            "passaggio a favore di Sempronio. Caio chiede a Tizio la riduzione del prezzo.")
    result = qr.classify(text, log=False)
    assert result["tie_break"] and len(no_llm) == 1
    assert qr.classify(text, log=False) is result and len(no_llm) == 1          # once per text


def test_the_reasons_name_the_signals():
    reasons = qr.classify(NARRATIVES[0], log=False)["reasons"]
    assert any("parties" in r for r in reasons) and any("events" in r for r in reasons)


def test_parties_count_school_names_companies_roles_and_the_user():
    assert {"tizio", "caio"} <= set(qr.parties("Tizio presta a Caio mille euro."))
    assert len(qr.parties("La società Alfa S.r.l. e la Beta S.p.A. firmano un contratto.")) >= 2
    assert "io" in qr.parties("Ho comprato una casa dal venditore.")


def test_without_the_flag_the_pipeline_is_the_narrative_path(monkeypatch):
    monkeypatch.delenv("ROUTER_ENABLED", raising=False)
    assert qr.route_for_pipeline("Quali sono i requisiti della legittima difesa?")["route"] == "narrative"


def test_with_the_flag_a_frame_with_one_party_does_not_confirm_a_narrative(monkeypatch):
    monkeypatch.setenv("ROUTER_ENABLED", "true")
    analysis = {"frame": {"facts": ["un solo fatto"]}, "fact_signals": {"parties": [{"name": "Tizio"}]}}
    result = qr.route_for_pipeline(NARRATIVES[0], analysis)
    assert result["route"] == "direct" and result["frame_confirms"] is False
    analysis = {"frame": {"facts": ["acquisto", "guasto", "rifiuto"]},
                "fact_signals": {"parties": [{"name": "Tizio"}, {"name": "Caio"}]}}
    assert qr.route_for_pipeline(NARRATIVES[0], analysis)["route"] == "narrative"
    assert qr.route_for_pipeline(NARRATIVES[0], None)["route"] == "narrative"   # no frame: the router's own


def test_the_tie_break_reads_one_word(monkeypatch):
    import src.rag.ai_chat as ai_chat
    monkeypatch.undo()
    monkeypatch.setattr(ai_chat, "_call_chat", lambda *a, **k: "Narrativa")
    assert qr._tie_break("testo", "direct")[0] == "narrative"
    monkeypatch.setattr(ai_chat, "_call_chat", lambda *a, **k: "boh")
    assert qr._tie_break("testo", "direct")[0] == "direct"


def test_the_chat_sends_a_narrative_rag_request_to_the_strategy_and_keeps_earlier_intents(monkeypatch):
    import src.chatbot.api as api
    monkeypatch.setenv("ROUTER_ENABLED", "true")
    assert api._routed_intent(NARRATIVES[1], "rag") == "defensive_draft"
    assert api._routed_intent(NARRATIVES[1], "generate") == "generate"
    assert api._routed_intent(DIRECTS[0], "rag") == "rag"
    assert api._routed_intent(DIRECTS[1], "defensive_draft") == "defensive_draft"     # precedence
    assert api._routed_intent(NARRATIVES[1], "rag", pending_calculation={"id": "x"}) == "rag"
    assert api._routed_intent(NARRATIVES[1] + " Calcola anche gli interessi legali.", "rag") == "rag"
    monkeypatch.setenv("ROUTER_ENABLED", "false")
    assert api._routed_intent(NARRATIVES[1], "rag") == "rag"
