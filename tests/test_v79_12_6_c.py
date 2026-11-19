"""v79.12.7 (items 7 and 8 of the brief), on synthetic sentences, records, rulings and article texts only:

7. the text residue. (a) element codes and a ruling's stance («(contro)») outside the tables: the path that bypassed
   compose's strip (the «Risposte ai quesiti» block, built after the sections' reader checks; the head; a step after
   compose) and an invented code («R9»); (b) a headless reference's date left behind («11.04.2025)»); (c) a procedural
   act narrated (the PM's contestazione, a notifica, a decreto) is a certain fact, and proves only the act; (d) the
   «elenco interno» note: no path prints it; (e) the main line printed at most once per section; (f) a «Cenni»
   section's rule statement grounded in the cited article's text or record. Counters for each.
8. the version footer: the version is the response's metadata; FOOTER_IN_BODY (default on) keeps it in the body, off
   drops it.
"""

import json
import os
import re
from pathlib import Path

import pytest

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("OPENAI_API_KEY", "k"), ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

from src import version  # noqa: E402
from src.rag import interactive_flow as flow  # noqa: E402
from src.rag import issue_records as IR  # noqa: E402
from src.rag import lawyer_method as M  # noqa: E402
from src.rag import posture  # noqa: E402
from src.rag import rule_statements  # noqa: E402
from src.rag import rulings  # noqa: E402
from src.rag import text_hygiene as T  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("THESIS_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("THESIS_DETERMINISTIC", "")
    from src.rag import plan_then_write
    monkeypatch.setattr(plan_then_write, "negation_review", lambda items, known, call=None: [])


IMPUTATO = {"rito": "penale", "family": "penale", "role": "imputato", "stage": "primo_grado",
            "entry": posture.entry_for("penale", "imputato", "primo_grado")}

# a synthetic case: a quarrel, a slap, a fall, a death; the PM's contestazione
QUARREL = "Gamma, durante un diverbio davanti a un bar, colpisce Delta con uno schiaffo."
FALL = "Delta cade all'indietro e batte la testa sul selciato; muore tre giorni dopo."
CHARGE = "Il pubblico ministero ha contestato a Gamma il reato di omicidio preterintenzionale."
QUESTION = "Quale strategia difensiva conviene a Gamma?"
# an invented ruling of the registry and its date (not a real one)
HEADLESS = "L'accusa richiamerà (Sez. 4, n. 11111/2025, 11.04.2025) sul nesso causale."
# a synthetic text in the form of a general-part article on causation (the cenni's material)
CAUSE_TEXT = ("1. Il concorso di cause preesistenti o simultanee o sopravvenute, anche se indipendenti dall'azione del "
              "colpevole, non esclude il rapporto di causalità fra l'azione e l'evento.\n2. Le cause sopravvenute "
              "escludono il rapporto di causalità quando sono state da sole sufficienti a determinare l'evento.")


def _sents(texts):
    return [{"id": f"S{i}", "text": t, "source": "caso", "certainty": "accertato"} for i, t in enumerate(texts, 1)]


def _records():
    return {"584 c.p.": {"rubrica": "Omicidio preterintenzionale", "side": "threat", "charged": 1, "requisites": [
        {"id": "R1", "comma": 1, "kind": "requisito", "status": "provato", "supporting": ["S1"],
         "text": "con atti diretti a commettere uno dei delitti preveduti dagli articoli 581 e 582"},
        {"id": "R2", "comma": 1, "kind": "requisito", "status": "provato", "supporting": ["S2"],
         "text": "cagiona la morte di un uomo"}]}}


def _row():
    return {"_article": ("584", "c.p."), "s": {"plain_text": "Chiunque commette il fatto (584)."},
            "_proc": {"version": 4, "pena": "reclusione da dieci a diciotto anni (co. 1)", "querela": False}}


def _ctx(texts=(QUARREL, FALL)):
    recs = _records()
    ctx = M.prepare({"records": recs, "sentences": _sents(list(texts))}, IMPUTATO, [_row()], " ".join(texts), "Gamma",
                    "", list(recs), [], [], [], [], link_call=lambda *a: {}, case_id_="v79126c")
    ctx["asked"] = [QUESTION]
    return ctx


def _plan(ctx, objection="L'accusa insisterà sul nesso.", reply="Il nesso va provato."):
    def call(messages):
        return {"valutazioni": {c["id"]: {"pro": "p", "contro": "c", "argomento": "a", "realismo": "medio"}
                                for c in ctx["candidates"]},
                "obiezioni": [{"obiezione": objection, "risposta": reply}]}
    return M.method_plan(ctx, call=call)


def _section(answer, n):
    return dict(M.answer_sections(answer)).get(M.title(n)) if isinstance(n, str) else None


# --- 7 (a): element codes and stance marks ---------------------------------------------------------------------------------

def test_the_answers_block_was_the_path_the_codes_leaked_by_and_is_now_checked():
    ctx = _ctx()
    plan = _plan(ctx)
    written = "Conviene contestare R2 (contro). La difesa resta sul nesso (verso l'integrazione: contro)."
    # the bypass: the block is built after the sections' reader checks and only ran Block B's prose check
    raw, _ = M.answers_block(ctx, [(QUESTION, written)], "", [], 0, {}, "Gamma", lambda x: x)
    assert "R2" in raw and "(contro)" in raw
    log = {}
    answer = M.compose([("Norme applicate ai fatti", "La condotta è R1."), (QUESTION, written)], ctx, plan, log=log)
    block = "\n".join(a for _, a in M.answers_of(answer))
    assert "R2" not in block and "(contro)" not in block and "verso l'integrazione" not in block
    assert "«cagiona la morte di un uomo»" in block
    assert M.output_check(answer, ctx["output_sources"])["wording"] == []
    assert log["residue_out"]["stance"]


def test_an_invented_code_drops_its_sentence_and_its_objection_and_a_stance_mark_goes_everywhere():
    ctx = _ctx()
    plan = _plan(ctx, "L'accusa insisterà sul nesso (contro).", "Il nesso tra R1 e R9 (contro) non regge.")
    log = {}
    answer = M.compose([("Norme applicate ai fatti", "La condotta è R1 (contro).")], ctx, plan, log=log)
    assert "R9" not in answer and "(contro)" not in answer
    assert "L'accusa insisterà sul nesso" not in answer          # its reply emptied: the fallback fills §7
    assert any("R9" in x for x in log["unknown_codes_out"])
    # the tables keep their codes: they are the tables'
    assert "| Contestare" in answer


def test_the_last_pass_strips_what_a_late_step_or_the_head_left_and_counts_it():
    ctx = _ctx()
    state = {"records": _records(), "sentences": _sents([QUARREL, FALL, CHARGE])}
    draft = ("Nota iniziale sull'elemento R2 (contro).\n\n**1. Sintesi**\n\nTesto (R1).\n\n| R1 | integrato |\n\n"
             "**5. Orientamenti rilevanti**\n\n- «passaggio citato con R7 e (contro) al suo interno» (contro).")
    out = flow.text_residue(state, draft, [ctx])
    assert "R2" not in out.split("**1.")[0] and "(contro)" not in out.replace("(contro) al suo", "")
    assert "| R1 | integrato |" in out                                    # a table is left as it is
    assert "«passaggio citato con R7 e (contro) al suo interno»" in out   # a quoted passage is left as it is
    counters = state["block_counters"]
    assert counters["residue_codes_stripped"] >= 2 and counters["residue_stance_stripped"] == 2
    assert counters["residue_left"] == 0


# --- 7 (b): a headless reference's date -------------------------------------------------------------------------------------

def test_a_headless_reference_goes_with_its_date_and_an_orphan_date_is_found():
    log = []
    out = M.whole_citations(HEADLESS, log)
    assert "11.04.2025" not in out and "11111" not in out and out == "L'accusa richiamerà sul nesso causale."
    assert log and "11.04.2025" in log[0]
    # the residue as the live answer showed it: a date and a parenthesis without their reference
    orphan = "L'accusa richiamerà l'orientamento, 11.04.2025) sul nesso causale."
    assert T.orphan_dates(orphan) and any("data senza la pronuncia" in x for x in rulings.printed_problems(orphan))
    assert M.whole_citations(orphan) == "L'accusa richiamerà l'orientamento sul nesso causale."
    # a date with its head, or in its own parenthesis, or in a quoted passage, is no residue
    whole = "Cass. pen., Sez. 4, n. 11111/2025 (11.04.2025) afferma il principio."
    assert M.whole_citations(whole) == whole and not rulings.printed_problems(whole)
    quoted = "Il passaggio: «come già detto in Sez. 2, n. 22222/2024, 01.02.2024)» resta citato."
    assert M.whole_citations(quoted) == quoted


def test_the_objections_section_has_no_headless_date():
    ctx = _ctx()
    plan = _plan(ctx, HEADLESS, "La difesa risponde sul piano soggettivo.")
    answer = M.compose([("Norme applicate ai fatti", "La condotta è R1.")], ctx, plan, log={})
    objections = dict(M.answer_sections(answer))[M.title("obiezioni")]
    assert "11.04.2025" not in objections and not rulings.printed_problems(objections)


# --- 7 (c): a procedural act narrated is a certain fact ---------------------------------------------------------------------

@pytest.mark.parametrize("sentence,expected", [
    (CHARGE, IR.ACCERTATO),
    ("Il PM contesta a Gamma l'omicidio preterintenzionale, ipotizzando che lo schiaffo abbia causato la caduta.",
     IR.ACCERTATO),
    ("A Gamma è stato notificato il decreto di citazione a giudizio.", IR.ACCERTATO),
    ("Gamma è stato rinviato a giudizio davanti alla Corte d'assise.", IR.ACCERTATO),
    ("Gamma sostiene di non aver mai ricevuto la notifica del decreto.", IR.ALLEGATO),
    ("Forse il decreto penale non è stato notificato.", IR.IPOTIZZATO),
])
def test_a_procedural_act_is_certain_unless_the_rest_is_attributed_or_hedged(sentence, expected):
    assert T.procedural_act(sentence)
    assert IR.certainty(sentence) == expected


def test_without_the_act_rule_the_contestazione_read_as_alleged():
    # the cause: «contest… a <Nome>» is an attribution verb of _ALLEGED_RE -- the act's own words
    assert IR._ALLEGED_RE.search(CHARGE) and not IR._ALLEGED_RE.search(T.act_remainder(CHARGE, "contestazione"))


def test_the_act_is_a_fatto_certo_in_the_facts_and_proves_only_the_act():
    sents = IR.sentences(" ".join([QUARREL, FALL, CHARGE]))
    act = next(s for s in sents if s["text"] == CHARGE)
    assert act["act"] == "contestazione" and act["certainty"] == IR.ACCERTATO
    split = M.facts_split(sents, {})
    assert CHARGE in split["certi"] and not any(x["text"] == CHARGE for x in split["ipotizzabili"])
    # it never proves what it charges ...
    death = {"kind": "requisito", "text": "cagiona la morte di un uomo", "status": "provato",
             "supporting": [act["id"]]}
    IR.settle(death, {s["id"]: s for s in sents})
    assert death["status"] == "da provare"
    # ... only the act itself
    done = {"kind": "requisito", "text": "previa contestazione dell'addebito", "status": "provato",
            "supporting": [act["id"]]}
    IR.settle(done, {s["id"]: s for s in sents})
    assert done["status"] == "provato"


def test_the_act_counters():
    state = {"sentences": IR.sentences(" ".join([QUARREL, CHARGE,
                                                  "Gamma sostiene di non aver mai ricevuto la notifica del decreto."]))}
    flow.text_residue(state, "Testo.", [])
    assert state["block_counters"]["procedural_acts_certain"] == 1
    assert state["block_counters"]["procedural_acts_uncertain"] == 1


# --- 7 (d): the «elenco interno» note ---------------------------------------------------------------------------------------

def _printable_json_strings(value, key=""):
    if isinstance(value, dict):
        for k, v in value.items():
            if k.startswith("_") or k == "status" or k.endswith("_status"):
                continue                                   # a draft's status and the files' own notes: never printed
            yield from _printable_json_strings(v, k)
    elif isinstance(value, list):
        for v in value:
            yield from _printable_json_strings(v, key)
    elif isinstance(value, str):
        yield value


def test_no_data_file_or_code_string_prints_a_draft_status():
    rx = T.internal_re()
    assert rx is not None
    for path in sorted((ROOT / "data").glob("*.json")):
        if path.name == "text_hygiene.json":
            continue                                       # the table of the wording itself
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        hits = [x for x in _printable_json_strings(data) if rx.search(x)]
        assert not hits, f"{path.name}: {hits[:2]}"
    for path in sorted((ROOT / "src" / "rag").glob("*.py")):
        literals = re.findall(r"\"[^\"\n]*\"|'[^'\n]*'", path.read_text(encoding="utf-8"))
        assert not [x for x in literals if rx.search(x)], path.name


def test_a_sentence_with_a_draft_status_is_removed_and_counted():
    state = {}
    draft = ("**8. Profili processuali**\n\n- Messa alla prova: preclusa, secondo l'elenco interno, ancora in bozza. "
             "Il rito ordinario resta.")
    out = flow.text_residue(state, draft, [])
    assert "elenco interno" not in out and "Il rito ordinario resta." in out
    assert state["block_counters"]["internal_notes_removed"] == 1 and state["block_counters"]["residue_left"] == 0


# --- 7 (e): the main line once per section ----------------------------------------------------------------------------------

def test_the_main_line_is_printed_once_per_section():
    ctx = _ctx()
    plan = _plan(ctx)
    log = {}
    answer = M.compose([("Norme applicate ai fatti", "La condotta è R1."), ("Quale pena rischia Gamma?", ""),
                        (QUESTION, "")], ctx, plan, log=log)
    tesi = ctx and log["main_thesis"]
    assert log["main_line_repeated"] == [] and ctx["main_line_repeated"] == 0
    conclusions = dict(M.answer_sections(answer))[M.title("conclusioni")]
    assert T.main_line_count(conclusions, tesi) == 1
    solutions = dict(M.answer_sections(answer))[M.title("soluzioni")]
    assert T.main_line_count(solutions, tesi) == 1 and M.STRATEGY_MAIN_IN_TABLE in solutions


def test_a_second_answer_on_the_main_line_refers_to_the_first():
    sols = [{"tesi": "Contestare la prevedibilità in concreto dell'evento morte", "realismo": "medio"}]
    block, _ = M.answers_block({"asked": ["Quale linea conviene?", "Che esito avrà la difesa?"]}, [], "", sols, 0,
                               {"esito": M.INCERTO, "why": "un elemento è da provare"}, "Gamma", lambda x: x)
    assert T.main_line_count(block, sols[0]["tesi"]) == 1
    assert M.MAIN_LINE_SEE_ANSWER.format(n=1) in block


def test_the_counter_reads_tables_too():
    tesi = "Ammettere la condotta e contenere la pena"
    text = f"| {tesi} (tesi principale) | p |\n\n- Linea principale: {tesi.lower()}."
    assert T.main_line_count(text, tesi) == 2 and T.main_line_repeated([("6", text)], tesi) == ["6"]


# --- 7 (f): the cenni's rules grounded in the cited article -----------------------------------------------------------------

WRONG = "L'art. 41 c.p. stabilisce che la responsabilità dipende dalla prevedibilità in concreto dell'evento."
RIGHT = ("L'art. 41 c.p. dispone che le cause sopravvenute escludono il rapporto di causalità quando sono state da sole "
         "sufficienti a determinare l'evento.")
UNNAMED = "Sul punto si richiede la prevedibilità della morte di un uomo come conseguenza della condotta."


def test_a_cenno_states_only_the_rule_its_article_holds():
    src = {"elements": {}, "store": {}, "passages": {}}
    log = []
    out = T.cenni_check(f"{WRONG} {RIGHT}", {"41 c.p.": CAUSE_TEXT}, src, log)
    assert WRONG not in out and RIGHT in out and len(log) == 1 and "41 c.p." in log[0]["reason"]
    # the bypass: Block B's check keeps it -- «stabilisce» is no rule verb of its table -- and, with no record on the
    # article, removes the rule the article's text does hold
    assert rule_statements.check(WRONG, src) == WRONG and rule_statements.check(RIGHT, src) == ""


def test_a_cenno_naming_no_article_is_read_against_its_sections_articles():
    src = rule_statements.sources({"records": _records(), "circumstances": {}})
    # Block B alone: any record element of the answer grounds a sentence naming no article
    assert rule_statements.source_of(UNNAMED, src)
    text = f"{RIGHT} {UNNAMED}"
    out = T.cenni_check(text, {"41 c.p.": CAUSE_TEXT}, src, [])
    assert UNNAMED not in out and RIGHT in out


def test_compose_checks_the_cenni_sections_and_counts():
    ctx = _ctx()
    ctx["cenni_texts"] = {"41 c.p.": CAUSE_TEXT}
    plan = _plan(ctx)
    log = {}
    answer = M.compose([("Norme applicate ai fatti", "La condotta è R1."),
                        ("Cenni: Nesso causale", f"{RIGHT} {WRONG}")], ctx, plan, log=log)
    assert RIGHT in answer and WRONG not in answer
    assert ctx["cenni_rules_removed"] == 1 and ctx["cenni_rules_unsourced"] == 0
    assert log["cenni_rules_removed"][0]["sentence"] == WRONG


def test_the_cenni_texts_come_from_the_cenni_rows():
    grounds = {"nesso causale": [{"_article": ("41", "c.p."), "s": {"plain_text": CAUSE_TEXT}}]}
    assert flow.cenni_article_texts(grounds) == {"41 c.p.": CAUSE_TEXT}


# --- 8: the version footer --------------------------------------------------------------------------------------------------

def test_the_footer_flag_defaults_on_and_off_drops_the_footer(monkeypatch):
    assert flow.footer_in_body() is True
    assert version.has_footer(flow.with_version("Il parere."), version.current())
    monkeypatch.setenv("FOOTER_IN_BODY", "false")
    assert flow.with_version("Il parere.") == "Il parere."
    assert flow.with_version(f"Il parere.\n\n{version.footer()}") == "Il parere."


def _api(monkeypatch, tmp_path):
    for name, value in (("LLM_API_KEY", "k"), ("OPENAI_API_KEY", "k"), ("EMBEDDING_PROVIDER", "openai"),
                        ("EMBEDDING_API_KEY", "k")):
        monkeypatch.setenv(name, value)
    import src.chatbot.api as api
    from fastapi.testclient import TestClient
    from src.db.base import get_db
    monkeypatch.setattr(api.chatbot, "_sessions", {})
    monkeypatch.setattr(api.chatbot, "_sessions_file", str(tmp_path / "sessions.json"))
    monkeypatch.setattr(api.chatbot, "chat", lambda *a, **k: pytest.fail("the plain answer ran"))
    monkeypatch.setenv("CASE_FIRST_SINGLE_SHOT", "true")

    def single_shot(text, lang="it", request="", intake=None):
        state = {"mode": "single_shot", "state": flow.ANSWERED, "created": "x", "version": 1, "intake": intake}
        return {"answer": "IL PARERE", "interaction": None, "state": state, "citations": [], "kind": "answer"}
    monkeypatch.setattr(flow, "single_shot", single_shot)
    api.app.dependency_overrides[get_db] = lambda: None
    return api, TestClient(api.app), get_db


ASK = ("Sono il difensore di Gamma, imputato di omicidio preterintenzionale per uno schiaffo a Delta, caduto e morto. "
       "Delinea una strategia difensiva.")


@pytest.mark.parametrize("flag", ["", "false"])
def test_the_api_returns_the_version_as_metadata(monkeypatch, tmp_path, flag):
    if flag:
        monkeypatch.setenv("FOOTER_IN_BODY", flag)
    api, client, get_db = _api(monkeypatch, tmp_path)
    try:
        body = client.post("/api/chat", json={"message": ASK}).json()
    finally:
        api.app.dependency_overrides.pop(get_db, None)
    assert body["version"] == version.current()
    if flag:
        assert body["answer"] == "IL PARERE" and not version.has_footer(body["answer"])
    else:
        assert body["answer"] == version.with_footer("IL PARERE")


def test_the_nightly_footer_check_reads_the_metadata_with_the_flag_off(monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location("nightly_sim_c", ROOT / "scripts" / "nightly_sim.py")
    nightly = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nightly)
    meta = {"version": version.current()}
    assert nightly.footer_check(version.with_footer("Il parere."), meta)["ok"]
    monkeypatch.setenv("FOOTER_IN_BODY", "false")
    assert nightly.footer_check("Il parere.", meta)["ok"]
    assert not nightly.footer_check(version.with_footer("Il parere."), meta)["ok"]
    assert not nightly.footer_check("Il parere.", {"version": "0.0.0+old"})["ok"]

