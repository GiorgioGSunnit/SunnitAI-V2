"""v61: the intake (src/rag/intake.py) -- what the user wants, read once per request before the route decision: the
deliverable, the explicit asks in order, the rito, the client's role, the stage and the prior decisions, the client;
each field's confidence (rule | model | default); the model asked only for the fields the rules leave empty; the
defaults (parere for a narrated case, pre_filing, role and rito left to the client logic and the area frame, resolved
from the analysis); the route each deliverable takes through /api/chat; the 31 sim cases in their default cell and the
grid (tests/data/intake_grid.yaml) on the rules alone. The model and the graph are stubbed."""

import json
import os
import sys
from pathlib import Path

import pytest

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts"))

from src.rag import intake  # noqa: E402
from src.rag import interaction_copy as C  # noqa: E402
from src.rag import interactive_flow as flow  # noqa: E402
from src.rag import plan_then_write as ptw  # noqa: E402

SEMPRONIO_ASK = ("Il PM contesta a Sempronio l'art 584 cp. Il candidato assunte le vesti del legale di Sempronio, premessi "
                 "brevi cenni sull'omicidio preterintenzionale, il nesso causale e l'elemento soggettivo, delinea una "
                 "strategia difensiva, ipotizzando anche possibili reati alternativi configurabili nel caso di specie")


def rules(message, upload="", **kwargs):
    return intake.detect(message, upload, model=False, **kwargs)


# --- the deliverable ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("message, deliverable", [
    ("Predisponi l'atto di opposizione a decreto ingiuntivo per la Alfa S.r.l.", "atto"),
    ("Mi serve un ricorso per decreto ingiuntivo contro Beta S.p.A.", "atto"),
    ("Diffida ad adempiere nei confronti del conduttore moroso.", "atto"),
    ("Brevi cenni sulla servitù di passaggio coattivo.", "cenni"),
    ("Che cos'è la clausola risolutiva espressa?", "cenni"),
    ("Qual è la pena per il furto in abitazione?", "quesito"),
    ("Chi è competente per l'opposizione a sanzione amministrativa?", "quesito"),
    ("Calcola la rivalutazione monetaria di 10.000 euro dal 2015.", "calcolo"),
    ("Cerca giurisprudenza recente sull'abuso del diritto nei contratti bancari.", "ricerca_giurisprudenza"),
    ("Come si è pronunciata la Cassazione sul mantenimento del figlio maggiorenne?", "ricerca_giurisprudenza"),
    ("Ok, grazie.", "altro"),
])
def test_each_deliverable_by_its_rule(message, deliverable):
    result = rules(message)
    assert result["deliverable"] == deliverable
    assert result["confidence"]["deliverable"] == (intake.DEFAULT if deliverable == "altro" else intake.RULE)
    assert result["route"] == intake.route_of(result)


@pytest.mark.parametrize("message", [
    # the act named inside the story or as the parere's subject is not the request
    "Tizio viene condannato. Nella prospettiva della redazione dell'atto d'appello, il candidato esponga la traccia "
    "delle possibili argomentazioni difensive.",
    "La Società Alfa ha ricevuto il provvedimento di revoca dal Comune e intende proporre ricorso. Il candidato, "
    "assunte le vesti del legale della società, rediga parere.",
    "Caio ha presentato querela contro Tizio per lesioni. Il candidato, assunte le vesti del difensore di Tizio, rediga "
    "motivato parere sull'atto di citazione ricevuto.",
])
def test_an_act_named_in_the_story_or_by_the_parere_is_not_an_atto(message):
    assert rules(message)["deliverable"] == "parere"


def test_a_case_with_a_case_law_aside_stays_a_parere_and_case_law_alone_is_out_of_scope():
    exam = ("Tizio vende a Caio un'auto con vizi occulti; Caio scopre i vizi dopo un anno. Il candidato, assunte le vesti "
            "del legale di Caio, rediga parere illustrando l'orientamento della Cassazione in materia.")
    assert rules(exam)["deliverable"] == "parere"
    alone = rules("Quali sono le sentenze più recenti sulla responsabilità medica?")
    assert alone["deliverable"] == "ricerca_giurisprudenza" and alone["route"] == intake.OUT_OF_SCOPE


def test_an_upload_to_fill_to_review_or_to_argue():
    form = "MODULO DI DELEGA. Il sottoscritto ____ delega ____ a ritirare ____."
    contract = "CONTRATTO DI LOCAZIONE tra Alfa e Beta. Art. 3 Canone. Art. 7 Recesso."
    citation = ("TRIBUNALE DI MILANO. ATTO DI CITAZIONE. Alfa S.r.l. cita in giudizio Beta S.p.A. per sentirla "
                "condannare al pagamento di euro 40.000 per forniture non pagate nel 2024.")
    fill = rules("Compila il modulo con i dati di Mario Rossi.", form, refers_to_upload=True)
    assert fill["deliverable"] == "atto" and fill["route"] == intake.DOCUMENT and intake.doc_intent(fill) == "generate"
    review = rules("Rivedi il contratto allegato.", contract, refers_to_upload=True)
    assert review["deliverable"] == "revisione_documento" and intake.doc_intent(review) == "analyse"
    argue = rules("Sono il legale di Beta: come mi difendo da questa citazione?", citation, refers_to_upload=True)
    assert argue["deliverable"] == "parere" and argue["route"] == intake.CASE_FIRST and argue["case_in"] == "upload"
    assert intake.doc_intent(argue) == "defensive_draft"
    # a review asked with no upload: the plain answer
    assert intake.route_of({"deliverable": "revisione_documento", "confidence": {}}, has_upload=False) == intake.RAG


def test_a_narrative_beside_an_unrelated_upload_is_the_case_and_its_own_path(monkeypatch):
    monkeypatch.setattr(flow, "routes_here", lambda text: "Leonardo" in text)
    read = rules("Leonardo è proprietario di un fondo; Domitilla nel 2016 ha chiuso il cancello. Come agire?",
                 "Fattura n. 12 del 2025.")
    assert read["deliverable"] == "parere" and read["case_in"] == "message" and intake.doc_intent(read) == "rag"


# --- the requested blocks ------------------------------------------------------------------------------------------------

def test_the_exams_asks_in_order_three_cenni_one_parere_one_alternative():
    blocks = rules(SEMPRONIO_ASK)["requested_blocks"]
    assert [b["kind"] for b in blocks] == ["cenni", "cenni", "cenni", "parere", "alternative"]
    assert [b["subject"] for b in blocks[:3]] == ["omicidio preterintenzionale", "nesso causale", "elemento soggettivo"]
    assert blocks[3]["subject"].startswith("strategia difensiva")
    assert "reati alternativi" in blocks[4]["subject"]


def test_a_plain_narrative_or_its_one_parere_ask_has_no_blocks():
    assert rules("Tizio vende a Caio un terreno nel 2010; Caio non paga il prezzo. Cosa può fare Tizio?")[
        "requested_blocks"] == []
    assert rules("Tizio colpisce Caio con un pugno. Il candidato rediga motivato parere.")["requested_blocks"] == []


# --- rito, role, stage, decisions ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("message, rito", [
    ("Tizio è imputato di rapina aggravata. Il candidato ne assuma la difesa.", "penale"),
    ("Il giudice di pace ha fissato l'udienza per le percosse denunciate da Caio contro Tizio. Come difendere Tizio?",
     "penale_gdp"),
    ("Tizio, cliente dell'hotel, subisce il furto del computer; chiede il risarcimento al titolare, che invoca l'art. "
     "1227 c.c. Quali iniziative?", "civile"),
    ("La Alfa S.r.l. chiede la penale contrattuale per il ritardo nella consegna; il cliente contesta la clausola penale. "
     "Come difendersi?", "civile"),
    ("Il TAR ha respinto il ricorso contro il provvedimento di esclusione dalla gara. Quali rimedi?", "amministrativo"),
    ("Mario è stato licenziato senza preavviso dal datore di lavoro dopo vent'anni. Come impugnare il licenziamento?",
     "lavoro"),
    ("Caio, piccolo imprenditore, accetta un prestito a tassi usurari per evitare il fallimento. Tizio viene denunciato. "
     "Il candidato assuma la difesa di Tizio.", "penale"),
])
def test_the_rito_by_its_markers(message, rito):
    assert rules(message)["rito"] == rito


def test_a_tax_or_insolvency_matter_is_out_of_the_corpus_and_the_answer_says_so():
    tax = rules("L'Agenzia delle Entrate ha notificato a Tizio una cartella esattoriale per IVA non versata. Come "
                "contestarla?")
    assert tax["rito"] == "tributario" and tax["route"] == intake.OUT_OF_SCOPE
    insolvency = rules("La Beta S.r.l. è stata dichiarata fallita e il curatore fallimentare chiede la restituzione dei "
                       "pagamenti. Sono il legale del fornitore: come difenderlo?")
    assert insolvency["rito"] == "fallimentare" and insolvency["route"] == intake.OUT_OF_SCOPE
    assert "{materia}" in C.OUT_OF_CORPUS_MATTER and set(C.OUT_OF_CORPUS_MATTERS) == set(intake.OUT_OF_CORPUS)
    # a question or a calculation on a tax keeps the plain answer (its calculation gate: the tax calculators)
    question = rules("Come funziona l'imposta di registro sulle locazioni?")
    assert question["rito"] == "tributario" and question["route"] == intake.RAG
    assert rules("Calcola l'imposta di registro per un canone di 400 euro al mese.")["route"] == intake.RAG


@pytest.mark.parametrize("message, role", [
    (SEMPRONIO_ASK, "imputato"),
    ("Il candidato, assunte le vesti del legale della persona offesa, indichi le iniziative.", "persona_offesa"),
    ("Nel processo a carico di Caio, Mevia si è costituita parte civile. Come legale della parte civile, quale "
     "strategia?", "parte_civile"),
    ("Sono l'avvocato di Anna, che ha presentato querela contro il vicino per minacce. Cosa fare?", "persona_offesa"),
    ("Assisto Paolo, che ha citato in giudizio il vicino per le immissioni. Come procedere?", "attore"),
    ("Il mio cliente, Giorgio Blu, ha ricevuto un atto di citazione dalla Alfa S.r.l. Come difenderlo?", "convenuto"),
    ("Caio ha presentato querela contro Tizio per lesioni. Il candidato, assunte le vesti del difensore di Tizio, "
     "indichi la linea difensiva.", "imputato"),
])
def test_the_clients_role(message, role):
    assert rules(message)["role"] == role


def test_the_other_sides_lawyer_in_the_story_is_not_the_users():
    read = rules("Nel processo a carico di Sempronio il difensore dell'imputato chiede il rito abbreviato; i genitori "
                 "della vittima si sono costituiti parte civile. Come legale della parte civile, quale strategia?")
    assert read["role"] == "parte_civile"


@pytest.mark.parametrize("message, stage", [
    ("Il PM contesta a Tizio il reato di truffa. Il candidato assuma la difesa di Tizio.", "indagini"),
    ("Tizio, rinviato a giudizio per furto, attende l'udienza preliminare. Quale linea difensiva?", "primo_grado"),
    ("Tizio è stato condannato in primo grado per truffa. Il candidato prospetti il mezzo di impugnazione.",
     "impugnazione"),
    ("Al signor Rossi è stato notificato un atto di precetto. Come lo difendo?", "esecuzione"),
    ("Tizio, svegliato di notte, impugna la pistola e spara al ladro. Il candidato assuma la difesa di Tizio.",
     "pre_filing"),
])
def test_the_stage(message, stage):
    assert rules(message)["stage"] == stage


def test_prior_decisions_are_captured_with_court_outcome_and_date():
    decisions = rules("Il Tribunale di Milano con sentenza del 3 marzo 2024 ha condannato Tizio per truffa. Tizio vuole "
                      "proporre appello. Quale strategia?")["prior_decisions"]
    assert decisions[0] == {"court": "Tribunale di Milano", "outcome": "condanna", "date": "3 marzo 2024"}
    assert rules("Tizio rischia una condanna per truffa. Quale strategia?")["prior_decisions"] == []


def test_the_client_its_side_and_the_parties():
    read = rules(SEMPRONIO_ASK)
    assert read["client"]["name"] == "Sempronio" and read["client"]["side"] == "accused"
    assert intake.side_of("persona_offesa") == "victim" and intake.side_of("terzo") == "terzo"
    assert intake.side_of("sconosciuto") == ""


# --- the model, the defaults, the fallback ---------------------------------------------------------------------------------

def test_the_model_is_asked_only_for_the_fields_the_rules_leave_empty(monkeypatch):
    asked = []

    def ask(message, upload, history, fields):
        asked.append(list(fields))
        return {"role": "imputato", "stage": "primo_grado", "rito": "nonsense", "deliverable": "parere"}
    monkeypatch.setattr(intake, "ask_model", ask)
    read = intake.detect("Tizio colpisce Caio con un pugno al volto, causandogli lesioni. Il candidato, assunte le "
                         "vesti del difensore di Tizio, rediga motivato parere.")
    assert asked == [["role", "stage"]] or asked == [["rito", "role", "stage"]]
    assert read["role"] == "imputato" and read["confidence"]["role"] == intake.MODEL
    assert read["confidence"]["deliverable"] == intake.RULE
    asked.clear()
    assert intake.detect("Cos'è l'usucapione?")["deliverable"] == "cenni" and asked == []      # a question: no call
    asked.clear()
    full = intake.detect(SEMPRONIO_ASK)                                                    # every field by rule
    assert asked == [] and set(full["confidence"].values()) == {intake.RULE}
    asked.clear()
    unread = intake.detect("Vorrei capire meglio.")
    assert asked and asked[0][0] == "deliverable" and unread["deliverable"] == "parere"
    assert unread["confidence"]["deliverable"] == intake.MODEL and unread["rito"] == intake.UNKNOWN   # «nonsense»


_ASK_MODEL = intake.ask_model          # the real one: conftest stubs it for every test


def test_the_models_reply_is_read_as_json_and_kept_to_the_allowed_values(monkeypatch):
    import src.rag.ai_chat as ai_chat
    monkeypatch.setattr(ai_chat, "_call_chat", lambda *a, **k: 'Ecco: {"deliverable": "quesito", "stage": "boh", '
                                                              '"role": "sconosciuto"}')
    assert _ASK_MODEL("x", "", [], ["deliverable", "role", "stage"]) == {"deliverable": "quesito"}
    monkeypatch.setattr(ai_chat, "_call_chat", lambda *a, **k: "non è JSON")
    assert _ASK_MODEL("x", "", [], ["deliverable"]) == {}

    def broken(*a, **k):
        raise RuntimeError("down")
    monkeypatch.setattr(ai_chat, "_call_chat", broken)
    assert _ASK_MODEL("x", "", [], ["deliverable"]) == {}


def test_the_defaults_when_nothing_says_otherwise():
    case = rules("Tizio vende a Caio un terreno nel 2010; nel 2020 Caio scopre che il terreno non è edificabile. Cosa può "
                 "fare Caio?")
    assert case["deliverable"] == "parere" and case["stage"] == "pre_filing"
    assert case["confidence"]["stage"] == intake.DEFAULT and case["role"] == intake.UNKNOWN
    question = rules("Grazie, a domani.")
    assert question["deliverable"] == "altro" and question["stage"] == intake.UNKNOWN and question["route"] == intake.RAG


def test_resolve_fills_the_default_role_and_rito_from_the_analysis():
    read = rules("Tizio colpisce Caio con un pugno. Il candidato rediga motivato parere.")
    analysis = {"frame": {"areas": ["penale"]}, "fact_signals": {"facts": {"client_side": "Tizio (imputato)"}}}
    resolved = intake.resolve(read, analysis)
    assert resolved["rito"] in ("penale", read["rito"]) and resolved["role"] == "imputato"
    assert resolved["confidence"]["role"] == intake.DEFAULT and resolved["client"]["side"] == "accused"
    civil = intake.resolve(rules("Tizio non paga il canone a Caio da mesi. Cosa può fare Caio?"),
                           {"frame": {"areas": ["civile"]}, "fact_signals": {"facts": {"client_side": "Caio (locatore, "
                                                                                                       "danneggiato)"}}})
    assert civil["role"] == "attore"
    ruled = intake.resolve(rules(SEMPRONIO_ASK), {"frame": {"areas": ["civile"]}})
    assert ruled["rito"] == "penale" and ruled["role"] == "imputato"                     # a rule is never changed


def test_safe_detect_falls_back_on_a_failure(monkeypatch):
    monkeypatch.setattr(intake, "deliverable_rule", lambda *a, **k: 1 / 0)
    monkeypatch.setattr(flow, "routes_here", lambda text: True)
    read = intake.safe_detect("Tizio e Caio litigano.")
    assert read["deliverable"] == "parere" and read["route"] == intake.CASE_FIRST and "ZeroDivisionError" in read["error"]


def test_the_short_topic_rule_lives_in_the_intake():
    assert intake.topic_question("contratto firmato sotto minaccia")
    assert not intake.topic_question("ho firmato un contratto sotto minaccia")
    assert not intake.topic_question("strategia per un contratto firmato sotto minaccia")


# --- the sim's 31 and the grid, on the rules alone -------------------------------------------------------------------------

def test_the_31_sim_cases_keep_the_default_cell_on_the_rules_alone():
    from diagnose_long import HELDOUT_IDS, TUNING_IDS
    cases = {c["id"]: c for c in json.load(open(HERE / "eval_narrative_cases.json", encoding="utf-8"))["cases"]}
    ids = list(TUNING_IDS) + list(HELDOUT_IDS)
    assert len(ids) == 31
    off = [cid for cid in ids if (lambda r: (r["deliverable"], r["route"]))(rules(cases[cid]["question"]))
           != ("parere", intake.CASE_FIRST)]
    assert off == []


def test_the_grid_on_the_rules_alone():
    import yaml
    grid = yaml.safe_load(open(HERE / "data" / "intake_grid.yaml", encoding="utf-8"))
    assert len(grid) >= 24
    deliverables = {row["expected"]["deliverable"] for row in grid}
    assert deliverables == set(intake.DELIVERABLES)
    wrong = []
    for row in grid:
        result = rules(row["prompt"], row.get("upload") or "", refers_to_upload=bool(row.get("upload")))
        for key, value in row["expected"].items():
            got = ",".join(b["kind"] for b in result["requested_blocks"]) if key == "blocks" else result[key]
            if got != value:
                wrong.append((row["id"], key, got, value))
    assert len({w[0] for w in wrong}) <= len(grid) * 2 // 24, wrong        # the acceptance: ≥ 22/24, scaled


def test_the_check_script_prints_the_tables_and_the_totals(capsys):
    import intake_check
    assert intake_check.main(["--rules-only", "--env", "/nonexistent"]) == 0
    out = capsys.readouterr().out
    assert "## (a) the 31 sim cases" in out and "## (b) the grid" in out
    assert "**sim 31/31 ·" in out and "| **total** |" in out


# --- through /api/chat ---------------------------------------------------------------------------------------------------

@pytest.fixture
def api_client(monkeypatch, tmp_path):
    for name, value in (("LLM_API_KEY", "k"), ("OPENAI_API_KEY", "k"), ("EMBEDDING_PROVIDER", "openai"),
                        ("EMBEDDING_API_KEY", "k")):
        monkeypatch.setenv(name, value)
    import src.chatbot.api as api
    from fastapi.testclient import TestClient
    from src.db.base import get_db
    monkeypatch.setattr(api.chatbot, "_sessions", {})
    monkeypatch.setattr(api.chatbot, "_sessions_file", str(tmp_path / "sessions.json"))
    monkeypatch.setattr(api, "_classify_legal_intent", lambda *a: pytest.fail("the old classifier decided"))
    monkeypatch.setattr(api, "_classify_top_level_intent", lambda *a: pytest.fail("the old classifier decided"))
    logged = []
    monkeypatch.setattr(api, "vlog", lambda event, data=None, *a: logged.append((event, data)))
    monkeypatch.setattr(intake, "vlog", lambda event, data=None, *a: logged.append((event, data)))   # v73: logged there
    api.app.dependency_overrides[get_db] = lambda: None
    try:
        yield api, TestClient(api.app), logged
    finally:
        api.app.dependency_overrides.pop(get_db, None)


def _no_rag(api, monkeypatch):
    monkeypatch.setattr(api.chatbot, "chat", lambda *a, **k: pytest.fail("the plain answer ran"))


def test_case_law_gets_the_out_of_scope_copy_and_the_intake_is_logged(api_client, monkeypatch):
    api, client, logged = api_client
    _no_rag(api, monkeypatch)
    body = client.post("/api/chat", json={"message": "Trovami sentenze della Cassazione sul danno da vacanza "
                                                     "rovinata."}).json()
    assert body["answer"] == C.CASE_LAW_OUT_OF_SCOPE and body["status_messages"] == ["out_of_scope"]
    entry = [d for e, d in logged if e == "intake"]
    assert len(entry) == 1 and entry[0]["deliverable"] == "ricerca_giurisprudenza"
    session = api.chatbot._sessions[body["session_id"]]
    assert session.messages[0].metadata["intake"]["route"] == intake.OUT_OF_SCOPE


def test_a_tax_case_gets_the_out_of_corpus_copy(api_client, monkeypatch):
    api, client, _ = api_client
    _no_rag(api, monkeypatch)
    body = client.post("/api/chat", json={"message": "L'Agenzia delle Entrate ha notificato a Tizio un avviso di "
                                                     "accertamento IRPEF. Come contestarlo?"}).json()
    assert body["answer"] == C.OUT_OF_CORPUS_MATTER.format(materia=C.OUT_OF_CORPUS_MATTERS["tributario"])


def test_cenni_and_quesito_take_the_plain_answer_with_the_intake(api_client, monkeypatch):
    api, client, _ = api_client
    monkeypatch.setenv("CASE_FIRST_SINGLE_SHOT", "true")
    monkeypatch.setattr(flow, "single_shot", lambda *a, **k: pytest.fail("a question ran the case pipeline"))
    seen = []

    def plain(session_id, message, **kwargs):
        seen.append(kwargs.get("intake"))
        return {"session_id": session_id, "answer": "RISPOSTA", "original_query": message, "resolved_query": message,
                "session_language": "it", "status_messages": []}
    monkeypatch.setattr(api.chatbot, "chat", plain)
    for message in ("Cenni sull'usucapione abbreviata.", "Entro quanto si propone appello civile?"):
        assert client.post("/api/chat", json={"message": message}).json()["answer"] == "RISPOSTA"
    assert [s["deliverable"] for s in seen] == ["cenni", "quesito"]


def test_a_parere_runs_the_single_shot_with_the_intake_in_its_state(api_client, monkeypatch):
    api, client, logged = api_client
    monkeypatch.setenv("CASE_FIRST_SINGLE_SHOT", "true")
    _no_rag(api, monkeypatch)
    got = {}

    def single_shot(text, lang="it", request="", intake=None):
        got["intake"] = intake
        state = {"mode": "single_shot", "state": flow.ANSWERED, "created": "x", "version": 1, "intake": intake}
        return {"answer": "IL PARERE", "interaction": None, "state": state, "citations": [], "kind": "answer"}
    monkeypatch.setattr(flow, "single_shot", single_shot)
    body = client.post("/api/chat", json={"message": SEMPRONIO_ASK}).json()
    assert body["answer"] == "IL PARERE" and body["status_messages"] == ["interactive_single_shot"]
    assert got["intake"]["deliverable"] == "parere" and len(got["intake"]["requested_blocks"]) == 5
    assert len([e for e, _ in logged if e == "intake"]) == 1                                  # once per request


def test_an_atto_on_a_narrated_case_is_generated_with_the_case_record(api_client, monkeypatch):
    api, client, _ = api_client
    from src.rag import case_read
    monkeypatch.setattr(case_read, "read", lambda text, request="", lang="it": {
        "client": {"name": "Mario Rossi", "role": "creditore"}, "goal": "recuperare il credito",
        "facts": ["Beta non ha pagato tre fatture"], "unknown": ["la data delle fatture"], "hypotheses": []})
    monkeypatch.setattr(api, "classify_system_template", lambda *a, **k: "atto_citazione")
    monkeypatch.setattr(api, "rag_run", lambda *a, **k: {"raw_result": []})
    seen = {}

    def generate(message, doc_type, lang, citations=None, studio_name="", section_hint="", **kwargs):
        seen.update(kwargs)
        return {"draft": "ATTO DI CITAZIONE", "case_details": {}, "doc_type": doc_type}
    monkeypatch.setattr(api, "generate_document", generate)
    body = client.post("/api/chat", json={"message": "Redigi un atto di citazione per Mario Rossi contro la Beta "
                                                     "S.r.l., che nel 2024 non gli ha pagato tre fatture."}).json()
    assert body["status_messages"] == ["generation_mode"] and body["draft"].startswith("ATTO DI CITAZIONE")
    assert "Cliente: Mario Rossi (creditore)" in seen["case_record"] and "- Beta non ha pagato" in seen["case_record"]
    assert "Il caso non dice:" in seen["case_record"]


def test_the_case_record_is_empty_when_the_read_fails(monkeypatch):
    from src.rag import case_read
    monkeypatch.setattr(case_read, "read", lambda *a, **k: {})
    assert intake.case_record("Tizio e Caio.") == ""

    def broken(*a, **k):
        raise RuntimeError("down")
    monkeypatch.setattr(case_read, "read", broken)
    assert intake.case_record("Tizio e Caio.") == ""


# --- the flow: the intake in the state, the blocks in the plan ---------------------------------------------------------------

def test_the_single_shot_keeps_the_intake_and_passes_the_blocks_to_the_plan(monkeypatch):
    import test_v60k
    state, seen = test_v60k._single(monkeypatch)
    plans = []
    real = ptw.write_sections

    def capture(plan, jobs, *a, **k):
        plans.append(plan)
        return real(plan, jobs, *a, **k)
    monkeypatch.setattr(ptw, "write_sections", capture)
    read = rules(SEMPRONIO_ASK)
    out = flow.single_shot("Il caso.", intake=read)
    assert out["state"]["intake"]["requested_blocks"] == read["requested_blocks"]
    assert plans and plans[0]["requested_blocks"] == read["requested_blocks"]
    assert any(e == "intake_state" for e, _ in seen["events"])


def test_the_single_shot_reads_the_intake_itself_when_the_api_gave_none(monkeypatch):
    import test_v60k
    test_v60k._single(monkeypatch)
    out = flow.single_shot("Tizio vende a Caio un fondo nel 2010; Caio non paga. Cosa può fare Tizio?")
    assert out["state"]["intake"]["deliverable"] == "parere"


def test_an_upload_the_message_argues_is_the_case_of_the_single_shot(api_client, monkeypatch, tmp_path):
    import asyncio
    from types import SimpleNamespace
    import src.db.base as db_base
    import src.utils.document as document
    api, _, logged = api_client
    monkeypatch.setenv("CASE_FIRST_SINGLE_SHOT", "true")
    _no_rag(api, monkeypatch)
    citation = ("TRIBUNALE DI MILANO. ATTO DI CITAZIONE. Alfa S.r.l. cita in giudizio Beta S.p.A. per sentirla "
                "condannare al pagamento di euro 40.000 per forniture non pagate nel 2024.")
    path = tmp_path / "citazione.txt"
    path.write_text(citation, encoding="utf-8")
    doc = SimpleNamespace(id="00000000-0000-0000-0000-0000000000d1", original_filename="citazione.txt",
                          storage_path=str(path), document_role="document", uploaded_at=None)
    monkeypatch.setattr(db_base, "get_db", lambda: iter([None]))
    monkeypatch.setattr(api, "get_user_document", lambda *a, **k: doc)
    monkeypatch.setattr(api, "find_user_document_by_name", lambda *a, **k: doc)
    monkeypatch.setattr(api, "link_documents_to_conversation", lambda *a, **k: 0)
    monkeypatch.setattr(api, "get_user_settings", lambda *a, **k: {"tone": 2, "standing": 2, "response_length": 2})
    monkeypatch.setattr(api, "enforce_tenant_product_access", lambda *a: None)
    monkeypatch.setattr(document, "extract_text_from_file", lambda p: open(p, encoding="utf-8").read())
    got = {}

    def single_shot(text, lang="it", request="", intake=None):
        got.update(text=text, request=request, intake=intake)
        state = {"mode": "single_shot", "state": flow.ANSWERED, "created": "x", "version": 1}
        return {"answer": "IL PARERE", "interaction": None, "state": state, "citations": [], "kind": "answer"}
    monkeypatch.setattr(flow, "single_shot", single_shot)
    user = {"sub": "00000000-0000-0000-0000-000000000001", "tenant_id": "00000000-0000-0000-0000-000000000002"}
    message = "Sono il legale di Beta: come mi difendo da questa citazione?"
    body = asyncio.run(api._chat_impl(api.ChatRequest(message=message, document_id=doc.id), user, None))
    assert body.answer == "IL PARERE" and got["text"] == citation and got["request"] == message
    assert got["intake"]["deliverable"] == "parere" and got["intake"]["case_in"] == "upload"
    assert [d["deliverable"] for e, d in logged if e == "intake"] == ["parere"]                # once, with the file
    assert any(e == "interactive_deferred" for e, _ in logged)
