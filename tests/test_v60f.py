"""v60f: the intermediate representation between round 1 and the answer -- the per-article records and the case's
sentences (b), the plan bound to the chosen topics (c), the answer's ledger (no deletion without a recorded reason, every
planned article in exactly one bucket), the sim's saved events (a), and the audit's bugs X7, X4, P9, M8, B8, A4. The
model and the graph are stubbed."""

import copy
import os
import re
import sys
from pathlib import Path

import pytest

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))

from src.rag import answer_ledger, citation_ledger, issue_records, render_computed  # noqa: E402
from src.rag import interaction_copy as C  # noqa: E402
from src.rag import interactive_flow as flow  # noqa: E402
from src.rag import plan_then_write as ptw  # noqa: E402
import test_interactive_flow as base  # noqa: E402


def _graded_state(monkeypatch):
    """The base case with case-first topics: grades on the topics' articles, the records and the sentences built."""
    state = base._state(monkeypatch)
    for t, grades in zip(state["topics"], ({"1062 c.c.": 3, "1061 c.c.": 2}, {"1051 c.c.": 2, "1052 c.c.": 2,
                                                                             "1054 c.c.": 2},
                                           {"1158 c.c.": 3, "1159 c.c.": 1})):
        t["grades"], t["graded"], t["planned"] = grades, len(grades), len(t["articles"])
        t["matched"] = list(t["articles"])
    rows = flow._restored_rows(state["rows"])
    for r in rows:
        r["_grade"] = 3
    state["others"] = [{"title": "Altro", "articles": ["2043 c.c."], "why": "grado 1"}]
    state["grading"] = {"prefilter_dropped": ["1027 c.c."]}
    flow.build_records(state, rows)
    return state


# --- b. the per-article record ----------------------------------------------------------------------------------------

def test_the_record_holds_grade_reasons_requisites_and_the_dropped(monkeypatch):
    state = _graded_state(monkeypatch)
    rec = state["records"]["1062 c.c."]
    assert rec["grade"] == 3 and "grado 3" in rec["kept"] and "ancora del tema" in rec["kept"] and rec["dropped"] is None
    assert rec["requisites"] and all(r["id"].startswith("R") and r["status"] in (C.STATUS_PROVATO, C.STATUS_DA_PROVARE,
                                                                                C.STATUS_NON_RISULTA)
                                     for r in rec["requisites"])
    assert state["records"]["2043 c.c."]["dropped"] == "grado 1"
    assert state["records"]["1027 c.c."]["dropped"].startswith("prefiltro")
    assert state["sentences"][0]["id"] == "S1" and state["timings"]["records_s"] is not None


def test_each_case_sentence_has_its_certainty():
    sents = issue_records.sentences("Mario ha chiuso il cancello nel 2016. Domitilla sostiene che la strada fosse sua. "
                                    "Forse la strada esisteva già prima della divisione.",
                                    ["Il cliente aggiunge che il passaggio era quotidiano."],
                                    ["Parte assistita assunta: Mario"])
    assert [(s["certainty"], s["source"]) for s in sents] == [
        ("accertato", "caso"), ("allegato", "caso"), ("ipotizzato", "caso"), ("accertato", "precisazione"),
        ("assunto", "sistema")]


def test_an_open_requisite_names_the_question_that_asks_it(monkeypatch):
    state = _graded_state(monkeypatch)
    asked = {q["id"] for q in state["questions"]}
    linked = [r for rec in state["records"].values() for r in rec["requisites"] if r["question"]]
    assert linked and all(r["question"] in asked and r["status"] == C.STATUS_DA_PROVARE for r in linked)


def test_the_link_is_one_call_per_topic_in_parallel_and_cached(monkeypatch):
    state = _graded_state(monkeypatch)
    calls = []

    def call(topic, reqs, sents):
        calls.append(topic)
        return {reqs[0][1]["id"]: {"sostiene": ["S1", "S99"], "contraddice": [], "manca": "la data della divisione"}}
    topics = state["topics"][:2]
    labels = [["1062 c.c.", "1061 c.c."], ["1051 c.c."]]
    log = issue_records.link(state["records"], topics, labels, state["sentences"], call=call)
    assert sorted(calls) == sorted(t["title"] for t in topics) and log["total_ms"] >= 0
    first = state["records"]["1062 c.c."]["requisites"][0]
    assert first["supporting"] == ["S1"] and first["missing"] == "la data della divisione"      # S99: no such sentence
    again = issue_records.link(state["records"], topics, labels, state["sentences"], call=call)
    assert len(calls) == 2 and all(e.get("cached") for e in again["topics"])                    # same sentences: cached


def test_a_slow_link_is_bounded_and_leaves_the_record(monkeypatch):
    import time
    state = _graded_state(monkeypatch)
    log = issue_records.link(state["records"], state["topics"][:1], [["1062 c.c."]], state["sentences"],
                             call=lambda *a: time.sleep(0.5) or {}, seconds=0.05)
    assert log["topics"][0]["error"].startswith("timeout") and not state["records"]["1062 c.c."]["linked"]


# --- c. the plan bound to the chosen topics ---------------------------------------------------------------------------

def test_the_plan_prompt_gets_the_chosen_issues_and_no_list_that_adds_one():
    issues = [{"titolo": "Servitù", "norme": [("1062", "c.c.")], "scheda": "- art. 1062 c.c.\n  · R1 co. 1: «x» — da provare"}]
    analysis = {ptw.PLAN_ISSUES_KEY: issues, ptw.PLAN_ANGLE_KEY: "Come ottengo il passaggio?"}
    human = ptw._plan_human("caso", "", analysis, "NORME", "", "", [], [("1", "c.c.")], [("1062", "c.c.")], True)
    assert ptw.FIXED_ISSUES_TITLE in human and "R1 co. 1" in human and "DOMANDA DELL'UTENTE: Come ottengo" in human
    assert "NORME GIUDICATE APPLICABILI" not in human and "NORME APPLICATE DALLA GIURISPRUDENZA" not in human
    system = re.sub(r"Le questioni per prime, da due a \w+, in ordine di forza per la parte assistita\.",
                    ptw.FIXED_ISSUES_RULE, ptw._plan_system(1))
    assert "ESATTAMENTE quelle di QUESTIONI SCELTE" in system and "da due a" not in system


def test_the_plan_argues_the_chosen_issues_and_never_selects_them():
    issues = [{"titolo": "Servitù per destinazione", "norme": [("1062", "c.c."), ("1061", "c.c.")]},
              {"titolo": "Passaggio coattivo", "norme": [("1051", "c.c.")]}]
    plan = {"parte_assistita": "Leonardo", "questioni": [
        {"titolo": "Usucapione", "norme": [("1158", "c.c.")], "argomento": "ventennio"},
        {"titolo": "Destinazione del padre di famiglia", "norme": [("1062", "c.c.")], "argomento": "opera visibile",
         "obiezione": "o", "replica": "r"}]}
    log = []
    out = issue_records.fix_plan(plan, issues, log)
    assert [q["titolo"] for q in out["questioni"]] == ["Servitù per destinazione", "Passaggio coattivo"]
    assert out["questioni"][0]["argomento"] == "opera visibile" and out["questioni"][0]["norme"] == issues[0]["norme"]
    assert out["questioni"][1]["argomento"] == "" and any(e.get("dropped_plan_issue") == "Usucapione" for e in log)


# --- A4: the cap by legal weight -------------------------------------------------------------------------------------

def test_the_cap_ranks_by_grade_and_element_support_not_position():
    grades = {f"{n} c.c.": 2 for n in range(1450, 1460)}
    grades["1460 c.c."] = 3
    topic = {"title": "Contratto", "articles": list(grades), "planned": 11, "grades": grades, "matched": list(grades)}
    other = {"title": "Sfratto", "articles": ["658 c.p.c.", "657 c.p.c."], "planned": 2,
             "grades": {"658 c.p.c.": 2, "657 c.p.c.": 2}, "matched": []}
    records = {"1452 c.c.": {"grade": 2, "requisites": [{"status": C.STATUS_PROVATO}]}}
    dropped = []
    out = flow.capped_refs([topic, other], records=records, dropped=dropped)
    labels = [f"{n} {c}" for n, c in out]
    assert labels[:2] == ["1460 c.c.", "658 c.p.c."]                 # each topic's best first
    assert labels[2] == "1452 c.c." and len(out) == flow.TOPIC_ARTICLES   # then by grade, then by element support
    assert ("1459", "c.c.") in dropped and ("1460", "c.c.") not in dropped
    assert flow.capped_refs([dict(topic, grades={})])[0] == ("1450", "c.c.")    # no grades: the old order


# --- X4, P9 --------------------------------------------------------------------------------------------------------

def test_a_deleted_first_sentence_keeps_its_paragraph_mark():
    text = "[1] L'art. 2043 c.c. si applica qui. La strada esiste dal 1900.\n\n[2] Seconda norma, secondo paragrafo."
    spans = ptw._sentence_spans(text)
    assert all(not text[s:e].startswith("[") for s, e in spans)
    out = flow.citation_check([("Norme applicate ai fatti", text)], [("1062", "c.c.")], [], [])
    assert out[0][1].startswith("[1] La strada esiste dal 1900.") and "[2] Seconda norma" in out[0][1]


def test_the_writer_is_told_the_answers_own_sections_and_no_name_is_repaired_after():
    note = flow.sections_note()
    assert note.startswith("La risposta ha cinque sezioni (risposta e strategia;") and "profili processuali;" in note
    system = ptw._section_system({"parte_assistita": "Leonardo", "posizione": ""}, "it", note)
    assert "cinque sezioni" in system and "sei sezioni" not in system
    assert "«Profili processuali»" in render_computed.read_only_block("x", flow.ANSWER_TITLES[3])
    assert "sei sezioni" in ptw._section_system({"parte_assistita": "", "posizione": ""}, "it")   # the parere unchanged
    assert not hasattr(flow, "_references_fixed")


# --- the answer: ledger, buckets, X7, M8, B8 ------------------------------------------------------------------------

def _answer(monkeypatch, state, writer=None, seen=None):
    seen = seen if seen is not None else {}
    monkeypatch.setattr(flow, "_documents", lambda: [{"id": "LEGAL_DOC::cc", "name": "Codice Civile 2026"}])

    def fake_plan(facts, request, analysis, articles, rulings, principles, rows, references):
        seen["plan_analysis"] = dict(analysis)
        return {"parte_assistita": "Leonardo", "posizione": "", "obiettivo": "", "questione_centrale": "",
                "risposte": [], "subordinate": [], "punti_deboli": [],
                "questioni": [{"titolo": "Altro tema", "norme": [("1158", "c.c.")], "pronunce": [], "argomento": "a",
                               "obiezione": "", "replica": ""}]}
    monkeypatch.setattr(ptw, "make_plan", fake_plan)
    monkeypatch.setattr(render_computed, "render", lambda analysis, party="": "RENDERED BLOCK")
    monkeypatch.setattr(issue_records, "_link_call", lambda topic, reqs, sents: {})

    def fake_write(plan, jobs, lang, named, documents, rows, case, **kwargs):
        seen["plan"], seen["note"] = plan, kwargs.get("sections_note")
        return writer(jobs) if writer else [(t, f"Testo di {t}.") for t, *_ in jobs]
    monkeypatch.setattr(ptw, "write_sections", fake_write)
    from src.rag import text_tidy
    real_tidy = text_tidy.tidy

    def tidy(draft):
        seen["protected"] = list(getattr(citation_ledger._local, "plan", None) or [])
        return real_tidy(draft)
    monkeypatch.setattr(text_tidy, "tidy", tidy)
    return flow.write_answer(state, [1])


WRITER = {
    "Risposta e strategia": "Leonardo può agire. L'art. 2043 c.c. regola il danno extracontrattuale. Leonardo può agire.",
    "Norme applicate ai fatti": "[1] L'art. 1062 c.c. fa sorgere la servitù per destinazione. La strada esisteva alla "
                                "divisione dei fondi.\n\n[2] L'art. 1061 c.c. vuole opere visibili e permanenti, qui "
                                "presenti sul fondo.",
    "Obiezioni e repliche": "[1] La replica di Leonardo è che la strada era visibile.",
    "Conclusioni": "L'esito è favorevole a Leonardo. L'art. 2051 c.c. non c'entra con la vicenda dei fondi."}


def test_no_deletion_without_a_recorded_reason_and_every_planned_article_in_one_bucket(monkeypatch):
    state = _graded_state(monkeypatch)
    seen = {}
    text, _ = _answer(monkeypatch, state, lambda jobs: [(t, WRITER.get(t, f"Testo di {t}.")) for t, *_ in jobs], seen)
    ledger = state["ledger"]
    assert ledger["unaccounted"] == []                                           # invariant 1
    removed = [e for e in ledger["entries"] if e["kind"] == "removed"]
    # v60g (e): citation_check only proposes; the one check, last, removes against the record, with its reason
    assert any("2043" in e["sentence"] and e["step"] == "record_check" and e["rule"] == "fuori_dal_record"
               and "citation_check" in e["proposed_by"] for e in removed)
    assert any("2051" in e["sentence"] for e in removed)
    planned = [label for label, rec in state["records"].items() if (rec.get("answer") or {}).get("bucket") in
               (flow.BUCKET_WRITTEN, flow.BUCKET_NOT_APPLICABLE, flow.BUCKET_NOT_TREATED)]
    assert set(planned) == {"1062 c.c.", "1061 c.c."}                            # invariant 2: each in one bucket
    for label in planned:
        answer = state["records"][label]["answer"]
        assert answer["bucket"] in (flow.BUCKET_WRITTEN, flow.BUCKET_NOT_APPLICABLE, flow.BUCKET_NOT_TREATED)
        assert answer["bucket"] != flow.BUCKET_NOT_TREATED or answer["reason"] != flow.REASON_LOST
    assert "2043" not in text and "2051" not in text


def test_every_pass_of_the_answer_is_in_the_ledger(monkeypatch):
    state = _graded_state(monkeypatch)
    _answer(monkeypatch, state, lambda jobs: [(t, WRITER.get(t, f"Testo di {t}.")) for t, *_ in jobs])
    steps = state["ledger"]["steps"]
    for step in ("scaffolding_out", "cosmetic_fixes", "term_consistency", "rule_quote", "strict_status", "phantom_commi",
                 "conditional_on_unknown", "unanswered_out", "repetition_guard", "citation_check", "headless",
                 "assemble", "empty_sections", "tidy", "answer_final", "readback", "record_check"):
        assert step in steps, step
    assert "unsupported_out" not in steps          # v60g: with records, every planned article stays in §2


def test_the_bucket_of_an_article_the_writer_never_wrote_and_of_one_never_asked():
    ledger = answer_ledger.Ledger([("1062", "c.c.")])
    ledger.record("x", [("Norme applicate ai fatti", "[1] L'art. 1062 c.c. si applica ai fondi.")],
                  [("Norme applicate ai fatti", "[1] L'art. 1062 c.c. si applica ai fondi.")])
    draft = ("**2. Norme applicate ai fatti**\n\n**Art. 1062 c.c. — Destinazione**\n\n- testo: da provare.\n\n"
             "**Art. 1061 c.c. — Opere**\n\n- testo: da provare.")
    out = flow.bucket_articles([("1062", "c.c."), ("1061", "c.c."), ("1051", "c.c.")], [("1062", "c.c."), ("1051", "c.c.")],
                               draft, ledger, [("1159", "c.c.")])
    assert out["1062 c.c."]["reason"] == "perso senza rimozione registrata"      # written, no gloss now, no removal
    assert out["1061 c.c."]["reason"] == flow.REASON_UNEARNED
    assert out["1051 c.c."] == {"bucket": flow.BUCKET_NOT_TREATED, "reason": flow.REASON_NOT_WRITTEN,
                                "where": "assente dal §2", "removed_sentences": 0}
    assert out["1159 c.c."]["bucket"] == flow.BUCKET_CAP and len(out) == 4


def test_the_ledger_tells_removed_rewritten_and_moved():
    ledger = answer_ledger.Ledger()
    before = [("A", "Primo enunciato del testo qui. Secondo enunciato che resterà."), ("B", "Terzo enunciato del caso.")]
    after = [("A", "Secondo enunciato che resterà (da verificare: data)."),
             ("B", "Terzo enunciato del caso. Primo enunciato del testo qui.")]
    ledger.record("step", before, after)
    kinds = {e["kind"] for e in ledger.entries}
    assert kinds == {"moved"} and ledger.unaccounted(after) == []                 # a marker is no change
    ledger.record("drop", after, [("A", "Secondo enunciato che resterà."), ("B", "")])
    assert [e["step"] for e in ledger.entries if e["kind"] == "removed"] == ["drop", "drop"]


def test_tidy_runs_with_the_planned_articles_protected_and_the_guard_is_cleared_after(monkeypatch):
    state = _graded_state(monkeypatch)
    seen = {}
    _answer(monkeypatch, state, None, seen)
    assert ("1062", "c.c.") in seen["protected"]
    assert not getattr(citation_ledger._local, "plan", None)


def test_the_plan_gets_the_chosen_topics_records_and_the_angle(monkeypatch):
    state = _graded_state(monkeypatch)
    state["angle"] = "Come ottengo il passaggio?"
    seen = {}
    _answer(monkeypatch, state, None, seen)
    assert seen["plan_analysis"][ptw.PLAN_ANGLE_KEY] == "Come ottengo il passaggio?"
    issues = seen["plan_analysis"][ptw.PLAN_ISSUES_KEY]
    assert [i["titolo"] for i in issues] == [state["topics"][0]["title"]] and "R1" in issues[0]["scheda"]
    assert [q["titolo"] for q in seen["plan"]["questioni"]] == [state["topics"][0]["title"]]     # «Altro tema» gone
    assert seen["note"] == flow.sections_note()
    assert ptw.PLAN_ISSUES_KEY not in state["analysis"]


def test_the_settled_client_is_kept_in_the_state(monkeypatch):
    facts = {**base.NO_CLIENT, "parties": [{"name": "Leonardo", "role": "attore"},
                                           {"name": "Domitilla", "role": "convenuta"}]}
    state = base._state(monkeypatch, facts)
    analysis = flow._restored_analysis(state["analysis"])
    analysis.pop("parte_assistita", None)
    state["analysis"] = flow._jsonable(analysis)
    _answer(monkeypatch, state)
    assert state["client"] == {"name": "Domitilla", "how": "assunta"}
    assert state["analysis"]["parte_assistita"] == "Domitilla"


# --- a. the sim keeps the events, the records and the ledger --------------------------------------------------------

def test_the_sim_keeps_the_writer_and_check_events():
    import diagnose_long as dl
    cap = {"events": [["interactive_answer", {"x": 1}, 3.0], ["case_analysis_citation_discipline", {}, None],
                      ["fact_rules", {}, None], ["post_writing_step_failed", {"step": "tidy"}, None]]}
    assert [e[0] for e in dl.sim_events(cap)] == ["interactive_answer", "case_analysis_citation_discipline",
                                                  "post_writing_step_failed"]


def test_the_funnel_reads_the_exact_buckets_when_the_case_has_them():
    import measure_preservation as mp
    records = {"1062 c.c.": {"answer": {"bucket": "scritto"}},
               "1061 c.c.": {"answer": {"bucket": "non trattato", "reason": "rimosso da tidy", "where": "assente dal §2"}},
               "1159 c.c.": {"answer": {"bucket": "oltre il limite"}}}
    topics = [{"title": "T", "articles": ["1062 c.c.", "1061 c.c.", "1159 c.c."], "planned": 3}]
    r = mp.measure_exact("run/h3", topics, [1], "**2. Norme applicate ai fatti**\n\n", records)
    assert (r["C"], r["P"], r["W"], r["NT"]) == (3, 2, 1, 1)
    assert sorted(why for _, why, _ in r["not_written"]) == ["cap", "rimosso da tidy"]
