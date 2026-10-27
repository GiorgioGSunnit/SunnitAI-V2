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


def test_parties_without_enough_events_or_a_request_go_to_the_tie_break(no_llm):
    text = ("Tizio e Caio sono comproprietari di un fondo agricolo. Il fondo confina con la strada provinciale. "
            "Vorrei capire la disciplina della comunione.")
    result = qr.classify(text, log=False)
    assert result["tie_break"] and len(no_llm) == 1
    assert qr.classify(text, log=False) is result and len(no_llm) == 1          # once per text


# --- the routes' rules (invented texts, borderline ones included) ------------------------------------------

@pytest.mark.parametrize("text", [
    # two or more events, no explicit request: the facts alone make it a narrative
    "Tizio vende a Caio un terreno nel 2019. Nel 2021 Caio scopre che sul terreno grava una servitù di "
    "passaggio a favore di Sempronio. Caio chiede a Tizio la riduzione del prezzo.",
    # present tense, the names opening their sentences
    "Marco affitta a Luca un negozio in centro. Luca non paga il canone da sei mesi. Marco vuole riavere "
    "l'immobile: quali azioni può intraprendere?",
    # a public body as a party
    "Il Comune di Ferrara ha negato a Giulia il permesso di costruire. Giulia ha presentato una nuova domanda, "
    "ma il Comune non ha risposto entro novanta giorni. Che strategia conviene?",
    # labour facts inside a wider narrative (an accident, a third company): not ccnl
    "Anna lavora come impiegata presso la Gamma S.r.l. Il 5 maggio, mentre rientra dal lavoro, viene investita "
    "da un furgone della Delta S.p.A., il cui conducente fugge. Anna resta in malattia per tre mesi. Quali "
    "azioni può intraprendere Anna?",
    # borderline: two parties and a request, hardly any event
    "Mio fratello mi deve 5.000 euro da anni e non risponde ai messaggi. Posso agire contro di lui?",
])
def test_two_parties_and_events_or_a_request_are_a_narrative(text, no_llm):
    result = qr.classify(text, log=False)
    assert result["route"] == "narrative", result["reasons"]
    assert no_llm == []


@pytest.mark.parametrize("text", [
    "Tizio riceve una cartella esattoriale. Il candidato, assunte le vesti del legale di Tizio, rediga parere "
    "motivato.",
    "Assunte le vesti del difensore, si illustrino le questioni sottese al caso.",
])
def test_an_exams_markers_make_a_narrative_regardless(text):
    result = qr.classify(text, log=False)
    assert result["route"] == "narrative" and any("exam" in r for r in result["reasons"])


@pytest.mark.parametrize("text", [
    "Qual è la differenza tra dolo eventuale e colpa cosciente?",
    "Cosa si intende per possesso di buona fede ai sensi dell'art. 1147 c.c.?",
    "Come si notifica un atto di citazione a una società con sede all'estero?",
    # borderline: two generic roles and a request, one sentence -- a question about the rule
    "Il conduttore può chiedere al locatore la riduzione del canone per vizi dell'immobile?",
    # borderline: employer and worker, but not pay, leave, dismissal or contract terms
    "Il datore di lavoro può controllare la posta elettronica aziendale del dipendente?",
])
def test_a_norm_or_procedure_without_a_fact_pattern_is_direct(text, no_llm):
    result = qr.classify(text, log=False)
    assert result["route"] == "direct", result["reasons"]
    assert no_llm == []


@pytest.mark.parametrize("text", [
    "Il datore di lavoro può licenziare un dipendente durante il periodo di comporto per malattia?",
    "Sono una colf e la mia datrice di lavoro non mi paga la tredicesima: cosa prevede il contratto?",
    # a fact pattern whose question itself names the agreement
    "Luca è assunto dalla Omega S.r.l. dal 2018 come magazziniere. Nel 2024 l'azienda lo ha trasferito in "
    "un'altra sede e gli ha ridotto l'orario. Quale livello di inquadramento gli spetta secondo il CCNL "
    "Logistica?",
])
def test_an_employment_relationship_the_question_is_about_is_ccnl(text):
    result = qr.classify(text, log=False)
    assert result["route"] == "ccnl", result["reasons"]


def test_party_kinds_tell_concrete_parties_from_generic_roles():
    kinds = qr.party_kinds("Il Comune di Ferrara nega a Giulia il permesso; il conduttore e mia madre no.")
    assert kinds["comune di ferrara"] == "pa" and kinds["giulia"] == "name"
    assert kinds["condut"] == "role" and kinds["madre"] == "own role" and "comune" not in kinds
    assert "secondo" not in qr.parties("Quante ferie spettano secondo il CCNL? In primo grado no.")
    assert qr.parties("Anna lavora presso la Gamma S.r.l. da anni.")[:1] == ["gamma s.r.l"]


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


def test_with_the_flag_the_frame_turns_a_narrative_down_only_with_too_few_parties_and_events(monkeypatch):
    monkeypatch.setenv("ROUTER_ENABLED", "true")
    one_and_one = {"frame": {"facts": ["un solo fatto"]}, "fact_signals": {"parties": [{"name": "Mario"}]}}
    result = qr.route_for_pipeline(NARRATIVES[1], one_and_one)
    assert result["route"] == "direct" and result["frame_confirms"] is False
    one_party_three_events = {"frame": {"facts": ["ricovero", "infezione", "dimissione"]},
                              "fact_signals": {"parties": [{"name": "Mario"}]}}
    assert qr.route_for_pipeline(NARRATIVES[1], one_party_three_events)["route"] == "narrative"
    two_parties_one_event = {"frame": {"facts": ["ricovero"]},
                             "fact_signals": {"parties": [{"role": "paziente"}, {"role": "ospedale"}]}}
    assert qr.route_for_pipeline(NARRATIVES[1], two_parties_one_event)["route"] == "narrative"
    assert qr.route_for_pipeline(NARRATIVES[1], None)["route"] == "narrative"   # no frame: the router's own
    # an exam is never turned down
    assert qr.route_for_pipeline(NARRATIVES[0], one_and_one)["route"] == "narrative"


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
