"""v60: the case-first method of the interactive flow -- A the first read, B the topics from the graded rows, C the
procedural facts from the article's text, D the client settled before writing, E the read-back pass, F the rubric, G the
grading limits, H replies in the user's words. The model and the graph are stubbed: what is tested is the code around
them; what the model answers needs the server."""

import importlib.util
import os
import sys
import time
from pathlib import Path

import pytest

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))

from src.rag import case_read, interaction_copy as C, procedural_facts, readback, render_computed, topic_groups  # noqa
from src.rag import defensive_generation as dg  # noqa: E402
from src.rag import interactive_flow as flow  # noqa: E402
import test_interactive_flow as base  # noqa: E402


def _graded(number, code, grade, rubrica, doc="LEGAL_DOC::cp", rank=None):
    return {"_article": (number, code), "_grade": grade, "_fused_rank": rank,
            "s": {"name": f"{number}.0.0", "titolo_articolo": rubrica, "plain_text": f"Testo dell'art. {number}."},
            "d": {"id": doc, "name": "Codice Penale"}}


# --- A. the first read -------------------------------------------------------------------------------------------------

READ = {"cliente": {"nome": "Tizio", "ruolo": "indagato"}, "obiettivo": "evitare la condanna",
        "fatti": ["Tizio spara al ladro entrato in casa", "Qual è l'atto più opportuno?", "x" * 400] + ["f"] * 20,
        "non_detto": ["se l'arma era detenuta legalmente"],
        "ipotesi": [{"linea": "principale", "tesi": "Legittima difesa domiciliare", "istituti": ["legittima difesa",
                                                                                               "art. 52 c.p."],
                     "da_provare": ["l'intrusione"], "prove": ["sopralluogo"]},
                    {"linea": "subordinata", "tesi": "Eccesso colposo", "istituti": ["eccesso colposo"]},
                    {"linea": "controparte", "tesi": "Omicidio volontario", "istituti": ["omicidio"]},
                    {"linea": "boh", "tesi": "scartata", "istituti": []}]}


def test_the_first_read_is_bounded_and_never_records_the_request_as_a_fact():
    read = case_read.clean(READ, request="Qual è l'atto più opportuno?")
    assert read["client"] == {"name": "Tizio", "role": "indagato"}
    assert "Qual è l'atto più opportuno?" not in read["facts"] and len(read["facts"]) <= case_read.MAX_FACTS
    assert all(len(f) <= case_read.TEXT_CHARS + 1 for f in read["facts"])
    assert [h["line"] for h in read["hypotheses"]] == ["principale", "subordinata", "controparte"]
    assert read["hypotheses"][0]["institutions"] == ["legittima difesa"]          # never an article number
    assert case_read.clean("not json") == {} and case_read.clean({"ipotesi": "x"}) == {}
    assert not case_read.not_request("Tizio chiede quale sia l'atto giudiziario più opportuno",
                                     "quale sia l'atto giudiziario più opportuno")


def test_the_first_read_runs_beside_the_rest_and_never_waits_past_its_budget(monkeypatch):
    monkeypatch.setattr(case_read, "READ_SECONDS", 0.2)
    slow = case_read.start("caso", reader=lambda *a: time.sleep(1) or {"x": 1})
    t0 = time.time()
    assert slow.result() == {} and slow.timed_out and time.time() - t0 < 0.5
    quick = case_read.start("caso", reader=lambda *a: {"hypotheses": []})
    time.sleep(0.05)
    assert quick.result() == {"hypotheses": []} and not quick.timed_out and quick.ms is not None


# --- B. topics from the graded rows ---------------------------------------------------------------------------------

UNITS = {("LEGAL_DOC::cp", "52"): {"SEZIONE": "", "CAPO": "CAPO I – Della legittima difesa"},
         ("LEGAL_DOC::cp", "55"): {"CAPO": "CAPO I – Della legittima difesa"},
         ("LEGAL_DOC::cp", "575"): {"CAPO": "CAPO I – Dei delitti contro la vita e l'incolumità individuale"},
         ("LEGAL_DOC::cp", "584"): {"CAPO": "CAPO I – Dei delitti contro la vita e l'incolumità individuale"},
         ("LEGAL_DOC::cp", "589"): {"CAPO": "CAPO I – Dei delitti contro la vita e l'incolumità individuale"},
         ("LEGAL_DOC::cp", "157"): {"CAPO": "CAPO I – Disposizioni generali"},
         ("LEGAL_DOC::cp", "624"): {"CAPO": "CAPO I – Dei delitti contro il patrimonio"}}


def test_every_row_graded_two_or_more_makes_or_joins_a_topic_named_by_its_unit_or_leading_rubrica():
    rows = [_graded("575", "c.p.", 3, "Omicidio", rank=1), _graded("589", "c.p.", 2, "Omicidio colposo", rank=4),
            _graded("584", "c.p.", 2, "Omicidio preterintenzionale", rank=3),
            _graded("52", "c.p.", 3, "Difesa legittima", rank=2), _graded("55", "c.p.", 2, "Eccesso colposo", rank=5),
            _graded("157", "c.p.", 2, "Prescrizione. Tempo necessario a prescrivere", rank=6),
            _graded("624", "c.p.", 1, "Furto", rank=7), _graded("1", "c.p.", 0, "Reati e pene", rank=8)]
    rule_topics = [{"title": "Legittima difesa ed eccesso colposo", "anchor": "55 c.p.", "matched": ["52 c.p."],
                    "questions": ["arma_detenuta"], "facts": ["da verificare: l'arma"], "source": "tabella legittima_difesa"},
                   {"title": "Rapina", "anchor": "628 c.p.", "matched": [], "questions": ["data_fatto"], "source": "tabella"}]
    hypotheses = case_read.clean(READ)["hypotheses"] + [{"line": "subordinata", "claim": "Stato di necessità",
                                                         "institutions": ["stato di necessità"]}]
    built = topic_groups.build(rows, UNITS, hypotheses, rule_topics)
    titles = [t["title"] for t in built["topics"]]
    # the Capo's title; the generic «Disposizioni generali» named by its rubrica instead
    assert set(titles) == {"Delitti contro la vita e l'incolumità individuale", "Legittima difesa",
                           "Prescrizione. Tempo necessario a prescrivere"}
    assert titles[0] == "Delitti contro la vita e l'incolumità individuale"          # v60c: 3 graded rows first
    first = next(t for t in built["topics"] if t["title"] == "Legittima difesa")
    assert [h["claim"] for h in first["hypotheses"]] == ["Legittima difesa domiciliare", "Eccesso colposo"]
    assert first["questions"] == ["arma_detenuta"] and "da verificare: l'arma" in first["facts"]   # the table attaches
    assert built["lost_rules"] == ["Rapina"]                                         # and decides no topic
    life = next(t for t in built["topics"] if t["title"].startswith("Delitti"))
    assert life["articles"][:3] == ["575 c.p.", "584 c.p.", "589 c.p."] and life["graded"] == 3
    assert [h["claim"] for h in life["hypotheses"]] == ["Omicidio volontario"]
    assert [h["claim"] for h in built["unanchored"]] == ["Stato di necessità"]
    assert [o["articles"] for o in built["others"]] == [["624 c.p."]]                # grade 1: altri profili
    assert all("1 c.p." not in t["articles"] for t in built["topics"] + built["others"])   # grade 0: nowhere


def test_round_one_shows_the_hypotheses_under_their_topic_and_the_unanchored_as_to_verify(monkeypatch):
    state = base._state(monkeypatch)
    state["topics"][0]["hypotheses"] = [{"line": "principale", "claim": "Servitù per destinazione"}]
    state["unanchored"] = [{"line": "subordinata", "claim": "Abuso del diritto", "institutions": ["abuso"]}]
    reply, _ = flow.round1(state)
    assert "   - linea principale: Servitù per destinazione" in reply
    assert C.UNANCHORED_TITLE in reply and "- Abuso del diritto" in reply
    assert state["round1_text"] == reply


def test_analyse_case_reads_the_case_beside_retrieval_and_anchors_the_hypotheses(monkeypatch):
    monkeypatch.setenv("INTERACTIVE_CASE_FIRST", "true")
    monkeypatch.setattr(dg, "_analyse_case_for_pipeline", lambda text, lang, request="": {"frame": {"areas": ["penale"]}})
    started = {}
    monkeypatch.setattr(case_read, "read", lambda text, request="", lang="it": started.setdefault(
        "read", case_read.clean(READ, request)))
    rows = [_graded("52", "c.p.", 3, "Difesa legittima"), _graded("575", "c.p.", 3, "Omicidio")]
    built = topic_groups.build(rows, UNITS)

    def retrieve(text, request, lang, analysis, pending=None):
        time.sleep(0.05)
        return {"article_rows": rows, "regime_rows": [], "topics": built["topics"], "others": [], "topic_rows": rows,
                "rubriche": topic_groups.rubriche_of(rows), "grading": {"s": 1.0, "topics_s": 0.1}, "reserve": []}
    monkeypatch.setattr(flow, "_retrieve", retrieve)
    monkeypatch.setattr(flow, "_case_branches_early", lambda *a: __import__("contextlib").nullcontext())
    monkeypatch.setattr(render_computed, "render", lambda *a, **k: "")
    logged = []
    monkeypatch.setattr(flow, "_log", lambda event, data, *a: logged.append((event, data)))
    state = flow.analyse_case("Tizio spara al ladro.", "Sei l'avvocato di Tizio. Che atto?")
    assert started["read"] and state["read"]["client"]["name"] == "Tizio"
    assert state["topics"][0]["title"] == "Legittima difesa" and state["topics"][0]["hypotheses"]
    entry = next(d for e, d in logged if e == "hypotheses")
    assert (entry["n"], entry["anchored"], entry["unanchored"]) == (3, 2, 1) and entry["ms"] >= 0   # eccesso: no row
    assert [h["claim"] for h in state["unanchored"]] == ["Eccesso colposo"]
    assert state["timings"]["first_read_s"] is not None and state["timings"]["grading_s"] == 1.0


# --- C. the procedural facts ----------------------------------------------------------------------------------------

ART_582 = [(1, "Chiunque cagiona ad alcuno una lesione personale, dalla quale deriva una malattia nel corpo o nella mente, è "
               "punito con la reclusione da sei mesi a tre anni."),
           (2, "Il delitto è punibile a querela della persona offesa, salvo che ricorra taluna delle circostanze "
               "aggravanti previste negli articoli 583 e 585. Si procede tuttavia d'ufficio se il fatto è commesso contro "
               "persona incapace.")]
ART_575 = [(None, "Chiunque cagiona la morte di un uomo è punito con la reclusione non inferiore ad anni ventuno.")]
ART_624 = [(1, "Chiunque s'impossessa della cosa mobile altrui, sottraendola a chi la detiene, al fine di trarne profitto "
               "per sé o per altri, è punito con la reclusione da due a sei anni e con la multa da euro 927 a euro 1.500."),
           (2, "Il delitto è punibile a querela della persona offesa. Si procede, tuttavia, d'ufficio se la persona "
               "offesa è incapace.")]


def test_the_procedural_facts_come_from_the_articles_own_text():
    lesioni = procedural_facts.extract("582", "c.p.", ART_582)
    assert lesioni["procedibilita"].startswith("a querela della persona offesa (co. 2)") and lesioni["querela"]
    assert lesioni["querela_termine"] == procedural_facts.QUERELA_DEFAULT
    assert lesioni["remissione"] == procedural_facts.REMISSIONE_DEFAULT
    assert lesioni["pena"] == "reclusione da sei mesi a tre anni (co. 1)" and "giudice di pace" in lesioni["competenza"]
    omicidio = procedural_facts.extract("575", "c.p.", ART_575)
    assert not omicidio["querela"] and omicidio["competenza"].startswith("Corte d'assise")
    furto = procedural_facts.extract("624", "c.p.", ART_624)
    assert furto["querela"] and furto["competenza"].startswith("tribunale in composizione monocratica")
    assert procedural_facts.extract("52", "c.p.", [(1, "Non è punibile chi ha commesso il fatto…")]) == {}   # no offence
    assert procedural_facts.extract("309", "c.p.p.", [(1, "Entro dieci giorni dalla esecuzione o notificazione del "
                                                          "provvedimento, l'imputato può proporre richiesta di riesame.")]
                                    )["termine"] == "dieci giorni (co. 1)"


def test_the_procedural_section_renders_the_facts_never_the_generic_default():
    facts = {"582": procedural_facts.extract("582", "c.p.", ART_582)}
    lines = render_computed.procedure_lines([], [("582", "Lesione personale"), ("612", "Minaccia")], False, facts)
    assert lines[0].startswith("- Lesione personale (art. 582 c.p.): procedibile a querela della persona offesa (co. 2)")
    assert "querela entro tre mesi (art. 124 c.p.), remissibile (art. 152 c.p.)" in lines[0]
    assert lines[1] == f"- Minaccia (art. 612 c.p.): {C.PROCEDURE_UNKNOWN}."
    assert "procedibile d'ufficio; competenza: tribunale" not in "\n".join(lines)
    assert "il termine per proporla" in lines[-1]


def test_a_writer_sentence_contradicting_the_procedural_facts_is_replaced_by_them():
    facts = {"582": procedural_facts.extract("582", "c.p.", ART_582)}
    sections = [("Risposta e strategia", "Il reato di lesioni (art. 582 c.p.) è procedibile d'ufficio. La difesa regge."),
                ("Conclusioni", "Si procede d'ufficio salvo aggravanti (art. 582 c.p.). La competenza è della Corte d'assise.")]
    log = []
    out = procedural_facts.consistency(sections, facts, {"582": "Lesione personale"}, log)
    assert out[0][1].startswith("Quanto ai profili processuali: Lesione personale (art. 582 c.p.): procedibile a querela")
    assert out[0][1].endswith("La difesa regge.")
    assert "Si procede d'ufficio salvo aggravanti" in out[1][1]                    # an exception stated: no conflict
    assert "Corte d'assise" not in out[1][1] and [x["why"].split(",")[0] for x in log] == [
        "d'ufficio", "competenza Corte d'assise"]


def test_attach_reads_the_store_and_computes_what_it_lacks():
    class Session:
        def run(self, query, **params):
            class Result:
                def data(self_inner):
                    return [{"doc": "LEGAL_DOC::cp", "num": "575", "facts": '{"procedibilita": "d\'ufficio", "querela": false}',
                             "version": procedural_facts.VERSION}]
            return Result()
    rows = [{"_article": ("575", "c.p."), "s": {"name": "575.0.0", "plain_text": ART_575[0][1]}, "d": {"id": "LEGAL_DOC::cp"}},
            {"_article": ("582", "c.p."), "_commi": [t for _, t in ART_582], "s": {"name": "582.0.0"},
             "d": {"id": "LEGAL_DOC::cp"}},
            {"_article": ("1062", "c.c."), "s": {"name": "1062.0.0", "plain_text": "x"}, "d": {"id": "LEGAL_DOC::cc"}}]
    counts = procedural_facts.attach(Session(), rows)
    assert counts == {"store": 1, "computed": 1}
    assert rows[0]["_proc"]["source"] == "store" and rows[1]["_proc"]["querela"] and "_proc" not in rows[2]
    assert flow._compact_row(rows[1])["_proc"]["source"] == "computed"           # kept in the session


# --- D. the client ----------------------------------------------------------------------------------------------------

def test_an_unanswered_client_question_assumes_the_party_charged_never_the_first_name(monkeypatch):
    facts = {**base.FACTS, "parties": [{"name": "Caio", "role": "vicino"}, {"name": "Tizio", "role": "convenuto"}]}
    state = base._state(monkeypatch, facts)
    analysis = flow._restored_analysis(state["analysis"])
    state["questions"] = [{"id": "parte_assistita"}]
    name, line = flow.settle_client(state, analysis)
    assert name == "Tizio" and line.startswith("Non mi hai indicato chi assisti: imposto il parere per Tizio (convenuto)")
    assert analysis["parte_assistita"] == "Tizio"
    nobody = {**base.FACTS, "parties": [{"name": "Caio", "role": "vicino"}, {"name": "Sempronio", "role": "vicino"}]}
    state2 = base._state(monkeypatch, nobody)
    state2["questions"] = [{"id": "parte_assistita"}]
    assert flow.settle_client(state2, flow._restored_analysis(state2["analysis"])) == ("", C.CLIENT_UNKNOWN)
    state2["answered"] = {"parte_assistita": "Sempronio"}
    assert flow.settle_client(state2, {"parte_assistita": "Sempronio"}) == ("Sempronio", "")
    assert flow.settle_client(base._state(monkeypatch), {}) == ("", "")              # the facts name the client


# --- E. the read-back pass --------------------------------------------------------------------------------------------

def test_the_read_back_drops_a_contradiction_and_marks_a_fact_in_no_source():
    draft = ("**1. Risposta e strategia**\n\nL'art. 52 c.p. si applica al caso. Il fatto è avvenuto il 3 marzo 2021 e "
             "il danno è di euro 12.000. Tizio era in casa.\n\n**2. Conclusioni**\n\nL'art. 52 c.p. non si applica. "
             "Se l'intrusione è provata, l'art. 52 c.p. non si applica soltanto in caso di eccesso.\n\n"
             "**3. Profili processuali**\n\nTermine nel 2031.")
    log = {}
    out = readback.read_back(draft, ["Tizio era in casa quando il ladro è entrato."], {"52 c.p.": ["provato"]},
                             ("Profili processuali",), log)
    assert "L'art. 52 c.p. si applica al caso." in out and "L'art. 52 c.p. non si applica. " not in out
    assert "Se l'intrusione è provata" in out                                        # a conditional is no claim
    assert "euro 12.000 (non risulta dal caso)." in out and "Tizio era in casa." in out
    assert "Termine nel 2031." in out                                                # the code's section: not read
    assert log["contradictions"][0]["dropped"].startswith("L'art. 52 c.p. non si applica")
    assert {a for x in log["not_in_case"] for a in x["not_in_case"]} >= {"3 marzo 2021", "euro 12.000"}
    assert isinstance(log["lint"], list)


def test_the_read_back_keeps_the_negation_when_a_requisite_does_not_hold():
    sections = [("A", "L'art. 1062 c.c. si applica."), ("B", "L'art. 1062 c.c. non si applica.")]
    log = []
    out = readback.contradictions(sections, {"1062 c.c.": ["non risulta", "provato"]}, log)
    assert out[0][1].strip() == "" and out[1][1] == "L'art. 1062 c.c. non si applica."


# --- G. the grading limits ------------------------------------------------------------------------------------------

def test_the_prefilter_grades_only_the_top_n_by_similarity_and_fails_open():
    texts = {"52 c.p.": "difesa legittima", "575 c.p.": "omicidio", "1 c.p.": "reati e pene", "624 c.p.": "furto"}
    vectors = {"52": [1, 0], "575": [0.9, 0.1], "1": [0, 1], "624": [0.2, 0.8]}
    kept, dropped = dg._prefiltered(texts, "caso", 2, embed_query=lambda t: [1, 0],
                                    embed_documents=lambda docs: [vectors[d.split(" ")[0]] for d in docs])
    assert list(kept) == ["52 c.p.", "575 c.p."] and dropped == ["1 c.p.", "624 c.p."]
    assert dg._prefiltered(texts, "caso", 10)[1] == []
    broken = dg._prefiltered(texts, "caso", 2, embed_query=lambda t: 1 / 0)
    assert broken == (texts, [])


def test_round_one_grades_in_two_rounds_and_no_final_one(monkeypatch):
    texts = {f"{n} c.p.": "t" for n in range(1, 30)}
    by_label = {label: tuple(label.split(" ")) for label in texts}
    calls = []
    monkeypatch.setattr(dg, "_prompt_reach", lambda rows, grades, *a: {tuple(l.split(" ")) for l in texts})

    def grade_these(chosen):
        calls.append(len(chosen))
        return {}, []
    monkeypatch.setattr(dg, "_GRADE_ALL_PROMPT_CAP", 120)
    with dg.grading_limits({"rounds": 2}) as limits:
        dg._grade_prompt_rows(texts, by_label, [], grade_these, "f", "", {}, "it")
    assert limits["rounds_run"] == 1 and len(calls) == 1          # one round grades all 29: no second, no final


def test_the_answer_step_grades_the_chosen_topics_rest(monkeypatch):
    state = base._state(monkeypatch)
    state["topics"][0]["unit"] = "Sezione delle servitù"
    reserve = [base._row("1063", "Il proprietario del fondo servente…"), base._row("1170", "Chi è stato molestato…")]
    reserve[0]["_unit"], reserve[1]["_unit"] = "Sezione delle servitù", "Capo del possesso"
    state["reserve"] = [flow._compact_row(r) for r in reserve]

    def graded(text, request, rows, analysis):
        return [{**r, "_grade": 3} for r in rows]
    monkeypatch.setattr(dg, "_filter_applicable_articles", graded)
    added = flow.grade_chosen(state, [state["topics"][0]])
    assert added == ["1063 c.c."] and "1063 c.c." in state["topics"][0]["articles"]
    assert "1063 c.c." in state["topics"][0]["matched"] and len(state["reserve"]) == 1
    assert any(r["_article"] == ["1063", "c.c."] for r in state["rows"])


# --- H. replies in the user's words ----------------------------------------------------------------------------------

@pytest.mark.parametrize("message", ["procedi", "vai", "ok così", "dammi il parere", "scrivi il parere", "vai avanti",
                                     "Procedi pure", "scrivimi un parere", "avanti"])
def test_asking_for_the_answer_needs_no_model_for_the_usual_words(monkeypatch, message):
    monkeypatch.setattr(flow, "_reply_call", lambda *a: pytest.fail("the model read a shortcut"))
    parsed = flow.parse_reply(message, base._state(monkeypatch))
    assert parsed["confirm"] and parsed["intents"] == ["procedi"]


def _model(monkeypatch, **reply):
    raw = {"intents": [], "topics": [], "confirm": False, "answers": {}, "angle": "", "corrections": [],
           "statements": [], "question": "", **reply}
    monkeypatch.setattr(flow, "_reply_call", lambda *a: raw)


@pytest.mark.parametrize("message", ["mi interessa di più il passaggio coattivo", "quello sul passaggio",
                                     "parliamo del secondo argomento"])
def test_choosing_a_topic_by_name_or_description(monkeypatch, message):
    state = base._state(monkeypatch)
    state["state"] = flow.TOPICS_OFFERED
    _model(monkeypatch, intents=["scegli"], topics=[2])
    turn = flow.follow_up(message, state)
    assert turn["kind"] == "round2" and turn["state"]["chosen"] == [2]


@pytest.mark.parametrize("message", ["perfetto, scrivilo pure", "sì va bene, mandami la risposta", "ok, il parere"])
def test_asking_for_the_answer_in_other_words(monkeypatch, message):
    state = base._state(monkeypatch)
    state["state"] = flow.TOPICS_OFFERED
    _model(monkeypatch, intents=["procedi"])
    monkeypatch.setattr(flow, "write_answer", lambda st, numbers: (f"RISPOSTA {numbers}", []))
    turn = flow.follow_up(message, state)
    assert turn["kind"] == "answer" and turn["answer"] == "RISPOSTA [1]"


@pytest.mark.parametrize("message", ["la strada esisteva già nel 1950", "Domitilla ha chiuso il cancello nel 2018, non nel 2016"])
def test_adding_or_correcting_facts(monkeypatch, message):
    state = base._state(monkeypatch)
    state["state"] = flow.TOPICS_OFFERED
    _model(monkeypatch, intents=["fatti"], statements=[message])
    turn = flow.follow_up(message, state)
    assert turn["kind"] == "round2" and message in turn["state"]["additions"]


@pytest.mark.parametrize("message", ["e l'usucapione della strada?", "vorrei che guardassi anche la tolleranza",
                                     "concentrati sull'usucapione"])
def test_a_topic_not_listed_or_a_new_focus(monkeypatch, message):
    state = base._state(monkeypatch)
    state["state"] = flow.TOPICS_OFFERED
    _model(monkeypatch, intents=["nuovo_argomento"], angle="usucapione")
    monkeypatch.setattr(flow, "_angle_topic", lambda *a: None)
    turn = flow.follow_up(message, state)
    assert turn["kind"] == "round2" and turn["state"]["angle"] == "usucapione" and turn["state"]["chosen"] == [3]


@pytest.mark.parametrize("message", ["mi mandi la bozza in word?", "voglio il file", "posso avere il documento"])
def test_asking_for_the_document_in_other_words(monkeypatch, message):
    state = base._state(monkeypatch)
    state["state"] = flow.ANSWERED
    _model(monkeypatch, intents=["documento"])
    turn = flow.follow_up(message, state)
    assert turn["kind"] == "document_request" and turn["document"]
    state["state"] = flow.TOPICS_OFFERED          # before any answer: the answer first, then the document
    monkeypatch.setattr(flow, "write_answer", lambda st, numbers: ("RISPOSTA", []))
    turn = flow.follow_up(message, state)
    assert turn["kind"] == "answer" and turn["document"]


@pytest.mark.parametrize("message", ["come funziona?", "cosa succede se scelgo il 2?", "quanto ci mette?"])
def test_a_side_question_about_the_flow(monkeypatch, message):
    state = base._state(monkeypatch)
    state["state"] = flow.TOPICS_OFFERED
    _model(monkeypatch, intents=["domanda_flusso"], question=message)
    turn = flow.follow_up(message, state)
    assert turn["kind"] == "flow_help" and turn["answer"] == C.FLOW_HELP


@pytest.mark.parametrize("message", ["boh", "non saprei", "asdf"])
def test_anything_else_gets_one_clarifying_question_never_round_one(monkeypatch, message):
    state = base._state(monkeypatch)
    state["state"] = flow.TOPICS_OFFERED
    _model(monkeypatch)
    monkeypatch.setattr(flow, "start", lambda *a, **k: pytest.fail("a fresh round 1"))
    turn = flow.follow_up(message, state)
    assert turn["kind"] == "unclear" and turn["answer"].endswith(C.CLARIFY_TOPICS)


def test_a_short_new_case_claim_is_clarified_a_new_narrative_restarts(monkeypatch):
    state = base._state(monkeypatch)
    state["state"] = flow.ANSWERED
    _model(monkeypatch, intents=["nuovo_caso"])
    assert flow.follow_up("ho un altro caso", state)["kind"] == "unclear"
    monkeypatch.setattr(flow, "routes_here", lambda m: True)
    assert flow.follow_up("Tizio e Caio " * 60, state) is None


def test_the_closing_lines_dictate_no_string():
    for text in (C.INSTRUCTION, C.CLOSING, C.NEW_FACT, C.ANGLE_STATED_CHOSEN, C.NO_TOPIC_TO_ANSWER, C.CLARIFY_TOPICS,
                 C.CLARIFY_CONFIRM, C.CLARIFY_ANSWERED, C.FLOW_HELP):
        assert '"' not in text and "(sì / no)" not in text and "scrivi " not in text.lower()


# --- F. the rubric ----------------------------------------------------------------------------------------------------

def _rubric():
    spec = importlib.util.spec_from_file_location("rubric_score", HERE.parent / "scripts" / "rubric_score.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_rubric_scores_ten_rows_and_keeps_the_benchmark_out_of_the_mean(tmp_path):
    rubric = _rubric()
    assert len(rubric.ROWS) == 10
    (tmp_path / "po1").mkdir()
    (tmp_path / "po1" / "case.txt").write_text("Caso nuovo", encoding="utf-8")
    (tmp_path / "po1" / "answer.md").write_text("Parere", encoding="utf-8")
    (tmp_path / "h3.case.txt").write_text("Caso h3", encoding="utf-8")
    (tmp_path / "h3.answer.md").write_text("Parere h3", encoding="utf-8")
    cases = rubric.from_dir(str(tmp_path))
    assert set(cases) == {"po1", "h3"}
    reply = {k: {"score": 1 if k != "prova" else 0.5, "why": "ok"} for k, _, _ in rubric.ROWS}
    reply["contraddizioni"] = {"score": 7, "why": "fuori scala"}
    scored = {cid: rubric.score(text, answer, caller=lambda system, human: reply) for cid, (text, answer) in cases.items()}
    assert scored["po1"]["contraddizioni"]["score"] == 1.0 and scored["po1"]["prova"]["score"] == 0.5
    table = rubric.table(scored, rubric.benchmark_ids())
    assert "| h3 (benchmark: regression only) |" in table and "**mean, unseen cases (1)**" in table
    assert "9.5/10" in table and "| strumento/atto |" in table


def test_the_sim_reports_the_expected_norms_the_prefilter_dropped():
    spec = importlib.util.spec_from_file_location("diagnose_long", HERE.parent / "scripts" / "diagnose_long.py")
    dl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dl)
    expected = [{"article": "1051", "code": "cc"}, {"article": "52", "code": "cp", "any_of": ["52", "55"]}]
    assert dl.prefilter_expected(["1051 c.c.", "55 c.p.", "2043 c.c."], expected) == ["1051 c.c.", "55 c.p."]


# --- v60b: instrumentation, the first read's timer, anchoring by similarity -------------------------------------------

from src.rag import plan_then_write as ptw, stage_meter  # noqa: E402


def test_the_stage_meter_times_and_counts_each_stage_and_is_a_no_op_without_one():
    stage_meter.mark("nowhere")                      # no active meter: nothing happens
    meter = stage_meter.Meter()

    class Session:
        def run(self, query, **params):
            return "ok"
    with meter.active():
        meter.mark("select")
        graph = stage_meter.counted(Session())
        graph.run("MATCH 1"), graph.run("MATCH 2")
        stage_meter.count("llm")
        time.sleep(0.02)
        stage_meter.mark("grading")
        stage_meter.count("llm", 3)
    meter.stop()
    log = meter.as_log()
    assert log["select_graph"] == 2 and log["select_llm"] == 1 and log["select_s"] >= 0.0
    assert log["grading_llm"] == 3 and "grading_graph" not in log
    assert (log["total_llm"], log["total_graph"]) == (4, 2) and list(log)[:2] == ["select_s", "select_llm"]


def test_a_review_pass_is_a_stage_of_its_own():
    meter = stage_meter.Meter()
    with meter.active():
        ptw._step("term_consistency", [("A", "x")], lambda secs: secs)
    meter.stop()
    assert "review:term_consistency" in meter.stages


def test_the_answer_logs_its_stages_in_one_line(monkeypatch):
    logged = []
    monkeypatch.setattr(flow, "_log", lambda event, data, *a: logged.append((event, data)))

    def fake(state, numbers):
        stage_meter.mark("plan")
        stage_meter.count("llm")
        stage_meter.mark("write")
        stage_meter.count("llm", 4)
        ptw._step("procedural_consistency", [("A", "x")], lambda secs: secs)
        stage_meter.mark("readback")
        return "RISPOSTA", []
    monkeypatch.setattr(flow, "_write_answer", fake)
    assert flow.write_answer({}, [1]) == ("RISPOSTA", [])
    stages = next(d for e, d in logged if e == "answer_stages")
    assert stages["plan_llm"] == 1 and stages["write_llm"] == 4
    assert {"plan_s", "write_s", "review_procedural_consistency_s", "readback_s", "total_s"} <= set(stages)


def test_the_first_read_timer_measures_the_call_and_the_join_stays_bounded(monkeypatch):
    done = case_read.start("caso", reader=lambda *a: time.sleep(0.05) or {"hypotheses": []})
    time.sleep(0.3)                                  # the rest of round 1 takes longer than the call
    assert done.result() == {"hypotheses": []}
    assert 40 <= done.ms <= 200 and done.wait_ms < 50 and not done.timed_out     # the call, not the collection
    monkeypatch.setattr(case_read, "READ_SECONDS", 0.1)
    slow = case_read.start("caso", reader=lambda *a: time.sleep(0.6) or {"x": 1})
    t0 = time.time()
    assert slow.result() == {} and slow.timed_out and time.time() - t0 < 0.2 and slow.ms is None


VECTORS = {"legittima difesa": [1.0, 0.0, 0.0], "Legittima difesa": [0.98, 0.1, 0.0], "Difesa legittima": [0.97, 0.2, 0.0],
           "omicidio": [0.0, 1.0, 0.0], "Omicidio": [0.05, 0.99, 0.0],
           "Delitti contro la vita e l'incolumità individuale": [0.1, 0.9, 0.2],
           "stato di necessità": [0.0, 0.0, 1.0]}


def _embed(texts):
    return [VECTORS.get(t, [0.3, 0.3, 0.3]) for t in texts]


def test_hypotheses_anchor_on_the_graded_rubriche_first_then_on_the_titles_at_0_75():
    rows = [_graded("52", "c.p.", 3, "Difesa legittima"), _graded("575", "c.p.", 3, "Omicidio")]
    topics = topic_groups.build(rows, UNITS)["topics"]
    vectors = topic_groups.topic_vectors(None, topics, rows)
    calls = []

    def embed(texts):
        calls.append(list(texts))
        return _embed(texts)
    hypotheses = [{"line": "principale", "claim": "Difesa", "institutions": ["legittima difesa"]},
                  {"line": "controparte", "claim": "Omicidio volontario", "institutions": ["delitti contro la vita"]},
                  {"line": "subordinata", "claim": "Necessità", "institutions": ["stato di necessità"]}]
    VECTORS["delitti contro la vita"] = [0.1, 0.9, 0.2]
    unanchored, log = topic_groups.anchor_by_similarity(topics, vectors, hypotheses, embed=embed,
                                                        rubriche=topic_groups.rubriche_of(rows))
    by_title = {t["title"]: t for t in topics}
    assert by_title["Legittima difesa"]["hypotheses"][0]["anchor_path"] == "rubrica"          # «Difesa legittima»
    assert by_title["Delitti contro la vita e l'incolumità individuale"]["hypotheses"][0]["anchor_path"] == "titolo"
    assert [h["claim"] for h in unanchored] == ["Necessità"]
    assert [(x["path"], x["attached"]) for x in log] == [("rubrica", True), ("titolo", True), ("none", False)]
    assert len(calls) == 1 and "legittima difesa" not in calls[0]      # one call, only for what the stems left
    assert topic_groups.ANCHOR_MIN == 0.75


def test_a_title_below_0_75_does_not_anchor(monkeypatch):
    rows = [_graded("575", "c.p.", 3, "Omicidio")]
    topics = topic_groups.build(rows, UNITS)["topics"]
    VECTORS["vagamente vicino"] = [0.9, 0.1, 0.4]
    unanchored, log = topic_groups.anchor_by_similarity(
        topics, topic_groups.topic_vectors(None, topics), [{"line": "principale", "claim": "x",
                                                            "institutions": ["vagamente vicino"]}], embed=_embed)
    assert unanchored and log[0]["path"] == "none" and 0 < log[0]["score"] < 0.75


# --- v60c: the cap, the order, the offences, the immediate answer ---------------------------------------------------

def _rows_many():
    rows, units = [], {}
    for i in range(7):                     # seven c.c. groups (primary), one row each graded 2
        n = str(1100 + i * 10)
        rows.append(_graded(n, "c.c.", 2, f"Rubrica {i} servitù", doc="LEGAL_DOC::cc"))
        units[("LEGAL_DOC::cc", n)] = {"CAPO": f"CAPO {i} – Capo numero {i}"}
    for i in range(3):                     # three special-code groups
        n = str(10 + i)
        grade = 3 if i == 0 else 2
        row = _graded(n, "Codice del Consumo", grade, f"Garanzia {i} consumatore", doc="LEGAL_DOC::cons")
        rows.append(row)
        units[("LEGAL_DOC::cons", n)] = {"CAPO": f"CAPO {i} – Tutela {i}"}
    return rows, units


def test_primary_code_groups_are_never_capped_and_special_ones_are():
    rows, units = _rows_many()
    built = topic_groups.build(rows, units)
    codes = [t["code"] for t in built["topics"]]
    assert codes.count("c.c.") == 7 and len(built["topics"]) == 7          # all seven primary groups kept
    assert {o["code"] for o in built["others"]} == {"Codice del Consumo"}   # the special ones spilled (6 - 7 < 0)
    few = [r for r in rows if r["_article"][1] != "c.c."] + rows[:2]
    built = topic_groups.build(few, units)
    assert [t["code"] for t in built["topics"]][:3] == ["Codice del Consumo", "c.c.", "c.c."]   # graded 3: weight 0
    assert len(built["topics"]) == 5


def test_the_order_is_code_weight_then_graded_rows_then_main_line_then_angle():
    def t(title, primary, graded, best=2, main=False):
        return {"title": title, "primary": primary, "graded": graded, "best_grade": best,
                "hypotheses": [{"line": "principale"}] if main else []}
    topics = [t("speciale", False, 5), t("speciale forte", False, 1, best=3), t("uno", True, 1, main=True),
              t("due", True, 2), t("angolo", True, 1)]
    topic_groups.order(topics, angle=[5])
    assert [x["title"] for x in topics] == ["due", "uno", "angolo", "speciale forte", "speciale"]


ART_612BIS = [(1, "Salvo che il fatto costituisca più grave reato, è punito con la reclusione da un anno a sei anni e sei "
                  "mesi chiunque, con condotte reiterate, minaccia o molesta taluno in modo da cagionare un perdurante e "
                  "grave stato di ansia o di paura."),
              (4, "Il delitto è punito a querela della persona offesa. Il termine per la proposizione della querela è di "
                  "sei mesi. La remissione della querela può essere soltanto processuale.")]


def test_an_identified_offence_is_always_in_a_topic_and_in_the_immediate_answer():
    rows, units = _rows_many()
    stalking = {**_graded("612-bis", "c.p.", 1, "Atti persecutori", doc="LEGAL_DOC::cp"),
                "_commi": [t for _, t in ART_612BIS]}
    lesioni = {**_graded("582", "c.p.", 2, "Lesione personale", doc="LEGAL_DOC::cp"), "_commi": [t for _, t in ART_582]}
    rows += [stalking, lesioni]
    units[("LEGAL_DOC::cp", "582")] = {"CAPO": "CAPO I – Dei delitti contro la vita"}
    built = topic_groups.build(rows, units)
    hypotheses = [{"line": "principale", "claim": "Stalking", "institutions": ["atti persecutori"]}]
    others = built["others"]
    found = topic_groups.ensure_offences(built["topics"], others, rows, units, hypotheses)
    assert found == ["582 c.p.", "612-bis c.p."]                       # grade 2, and grade 1 the first read names
    holders = [t for t in built["topics"] if t.get("offences")]
    assert {o for t in holders for o in t["offences"]} == {"582 c.p.", "612-bis c.p."}
    topic_groups.order(built["topics"])
    chosen = topic_groups.immediate(built["topics"])
    picked = [built["topics"][i - 1] for i in chosen]
    assert any("612-bis c.p." in t["offences"] for t in picked) and any("582 c.p." in t["offences"] for t in picked)
    assert 1 in chosen
    assert topic_groups.ensure_offences(built["topics"], others, rows, units, []) == ["582 c.p."]   # not named: grade 1 out


def test_the_immediate_answer_takes_the_top_groups_within_eight_articles():
    topics = [{"planned": 5}, {"planned": 2}, {"planned": 3}, {"planned": 1}]
    assert topic_groups.immediate(topics) == [1, 2, 4]


def test_procedi_takes_the_immediate_answer_on_case_first_topics(monkeypatch):
    state = base._state(monkeypatch)
    state["state"] = flow.TOPICS_OFFERED
    state["topics"] = [{**t, "graded": 1, "planned": 4, "offences": []} for t in state["topics"]]
    state["topics"][2]["offences"] = ["1158 c.c."]
    seen = []
    monkeypatch.setattr(flow, "write_answer", lambda st, numbers: (seen.append(list(numbers)) or "R", []))
    flow.follow_up("procedi", state)
    assert seen == [[3, 1]]


def test_a_rule_anchor_not_graded_is_listed_but_never_planned_nor_written():
    rows = [_graded("52", "c.p.", 3, "Difesa legittima")]
    rule = [{"title": "Tabella", "anchor": "52 c.p.", "matched": ["55 c.p.", "59 c.p."], "questions": ["arma_detenuta"],
             "source": "tabella"}]
    topic = topic_groups.build(rows, UNITS, (), rule)["topics"][0]
    assert topic["articles"] == ["52 c.p.", "55 c.p.", "59 c.p."] and topic["planned"] == 1
    assert topic["matched"] == ["52 c.p."] and flow._topic_refs(topic) == [("52", "c.p.")]
    assert flow.capped_refs([topic]) == [("52", "c.p.")]


def test_the_first_reads_gaps_become_questions_after_the_rules_ones(monkeypatch):
    asked = [{"id": "data_fatto", "text": "Quando è avvenuto il fatto?", "short": "data del fatto"}]
    out = flow.with_read_questions(asked, ["la data del fatto", "se l'arma era detenuta legalmente",
                                           "chi ha chiamato i soccorsi", "a", "b", "c", "d"], {})
    # v60k: the generic «Quando è avvenuto il fatto?» after the read's own questions, dropped first at the cap
    assert [q["id"] for q in out][:2] == ["read_2", "read_3"] and len(out) == flow.MAX_QUESTIONS
    assert "data_fatto" not in [q["id"] for q in out]
    assert out[0]["text"] == "Il caso non dice se l'arma era detenuta legalmente: puoi precisarlo?"
    few = flow.with_read_questions(asked, ["se l'arma era detenuta legalmente"], {})
    assert [q["id"] for q in few] == ["read_1", "data_fatto"]                    # room left: the generic one last
    assert flow.with_read_questions(asked, ["se l'arma era detenuta legalmente"], {"read_1": "sì"}) == asked
    state = base._state(monkeypatch)
    state["questions"] = out
    parsed = flow._clean_reply({"answers": {"read_2": "sì, regolarmente denunciata"}}, state)
    assert parsed["answers"] == {"read_2": "sì, regolarmente denunciata"}


# --- v60c: abrogated articles, the rubrica check, the read-back -----------------------------------------------------

def test_an_abrogated_article_is_never_an_anchor_nor_written():
    from src.rag.article_text import is_abrogated
    assert is_abrogated("(abrogato)") and is_abrogated("Articolo abrogato da R.D.L. 9 agosto 1943, n. 721")
    assert not is_abrogated("Comma abrogato dall'art. 46, comma 1") and not is_abrogated(ART_575[0][1])
    dead = {**_graded("216", "c.c.", 3, "Diritti dei coniugi"), "s": {"name": "216.0.0", "plain_text": "(abrogato)",
                                                                      "titolo_articolo": "Diritti"}}
    built = topic_groups.build([dead, _graded("52", "c.p.", 3, "Difesa legittima")], UNITS)
    assert all("216 c.c." not in t["articles"] for t in built["topics"])
    topics = [{"title": "T", "anchor": "216 c.c.", "articles": ["216 c.c.", "217 c.c."], "planned": 2,
               "matched": ["216 c.c."]}]
    alive = {"_article": ("217", "c.c."), "s": {"plain_text": "Testo vigente."}}
    pruned = flow.prune_abrogated(topics, [{"_article": ("216", "c.c."), "s": {"plain_text": "(abrogato)"}}, alive])
    assert pruned == [{"title": "T", "anchor": "217 c.c.", "articles": ["217 c.c."], "planned": 1, "matched": []}]


def test_one_stem_with_the_rubrica_or_the_first_comma_clears_the_rubrica_check():
    assert dg.one_stem_clears("la querela va presentata entro tre mesi", "Diritto di querela", "")
    assert dg.one_stem_clears("chi sottrae la cosa altrui", "Furto",
                              "Chiunque s'impossessa della cosa mobile altrui, sottraendola")
    assert not dg.one_stem_clears("le misure cautelari vanno richieste", "Querela", "Il diritto di querela")


def test_the_read_back_ignores_citation_numbers_and_marks_a_derived_amount_calculated():
    case = "Il canone è di euro 800 al mese; Tizio non paga da tre mesi. Il danno è di euro 1.500."
    draft = ("**1. Risposta e strategia**\n\nAi sensi dell'art. 2043 c.c. e dell'art. 1453, co. 2, c.c. (d.lgs. "
             "28/2010) il credito è di euro 2.400 per i canoni. Il danno totale è di euro 2.300. Tizio deve euro 9.999. "
             "Non rileva l'art. 3 Legge sul Procedimento Amministrativo 2026.")
    log = {}
    out = readback.read_back(draft, [case], {}, (), log, case=case)
    assert "euro 2.400 per i canoni (calcolato)." in out and "euro 2.300 (calcolato)." in out
    assert "euro 9.999 (non risulta dal caso)." in out
    atoms = [a for x in log["not_in_case"] for a in x["not_in_case"] + x["calculated"]]
    assert atoms == ["euro 2.400", "euro 2.300", "euro 9.999"] or sorted(atoms) == sorted(
        ["euro 2.400", "euro 2.300", "euro 9.999"])                         # no 2043, 1453, 28, 2010, 2026


def test_the_read_back_drops_a_claim_the_next_paragraph_denies():
    sections = [("Risposta e strategia", "Caia ha presentato querela tempestivamente presso la Procura competente.\n\n"
                                         "Caia non ha presentato querela tempestivamente presso la Procura competente."),
                ("Conclusioni", "Il fondo è intercluso.")]
    log = []
    out = readback.contradictions_across(sections, log)
    assert out[0][1].strip() == "Caia ha presentato querela tempestivamente presso la Procura competente."
    assert log and log[0]["dropped"].startswith("Caia non ha presentato")
    far = [("A", "Caia ha presentato querela tempestivamente presso la Procura competente.\n\nAltro.\n\nAncora "
                 "un altro paragrafo qui.\n\nCaia non ha presentato querela tempestivamente presso la Procura competente.")]
    assert readback.contradictions_across(far, []) == far                    # not consecutive


def test_a_sentence_stating_what_an_article_provides_must_share_a_stem_with_its_text():
    art120 = ("Diritto di querela Ogni persona offesa da un reato per cui non debba procedersi d'ufficio o dietro "
              "richiesta o istanza ha diritto di querela.")
    norms = ("**Art. 120 c.p. — Diritto di querela**\n- testo: da provare.\n\nLa querela costituisce atto di citazione. "
             "L'art. 120 c.p. prevede il diritto di querela della persona offesa. Caia ha subito un reato e la prova "
             "del danno richiede una perizia.")
    log = []
    out = readback.ungrounded([("Norme applicate ai fatti", norms)], {"120 c.p.": art120}, log,
                              rubriche={"120 c.p.": "Diritto di querela"})
    text = out[0][1]
    assert "atto di citazione" not in text and "prevede il diritto di querela" in text and "perizia" in text
    assert [x["article"] for x in log] == ["120 c.p."]
