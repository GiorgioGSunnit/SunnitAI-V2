"""v79.19.9 (branch v79.19) -- three rail changes, the opinion writer behind OPINION_WRITER. Synthetic fixtures only.

Part 1 (always on): A -- a record outside its topic's first 8 with grade 2 is writable, grade 1 is not, logged
writable_by_grade; the link, the plan issues and the fallback plan read it with its topic. B -- grade 3 + contradicted
-> kept and logged kept_grade3; grade 2 + contradicted -> demoted; grade 3 + fact-excluded -> demoted. C -- the cap is
50, the kept sets unchanged (test_v79_8a pins it), the cost logged.
Part 2: flag off -> no call, no opinion keys, the fixture's answer is the page; flag on -> the prose is the answer, an
uncited article's sentence removed, «Cass. n. 123» stripped, a term absent from the rendered section flagged, both texts
in the result and in the event; the system prompt as shipped.
"""

import json
import os
import re
import sys
import types
from pathlib import Path

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("OPENAI_API_KEY", "k"), ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import _fixture_79_15_0 as H  # noqa: E402
import test_v79_15_0_fixture_penal as P  # noqa: E402
from src.rag import defensive_generation as dg  # noqa: E402
from src.rag import interactive_flow as flow  # noqa: E402
from src.rag import issue_records as I  # noqa: E402
from src.rag import opinion_writer as O  # noqa: E402
from src.rag import plan_then_write as ptw  # noqa: E402
from src.rag import verbose_logger  # noqa: E402

ROOT = HERE.parent


def _events(monkeypatch):
    events = []
    monkeypatch.setattr(verbose_logger, "vlog", lambda e, d=None, ms=None: events.append((e, d)))
    return events


def _case(**over):
    keys = ("TEXT", "ARTICLES", "FRAME_REPLY", "fit_reply", "link", "analysis_reply", "TITLE", "ANCHOR", "RUBRICA",
            "DOCUMENTS", "CLIENT", "writer_line", "CASE_ID")
    return types.SimpleNamespace(**{**{k: getattr(P, k) for k in keys}, **over})


# --- A. writable by grade --------------------------------------------------------------------------------------------------

def _row(n, code="c.c.", text="Il debitore che non esegue esattamente la prestazione dovuta è tenuto al risarcimento "
                              "del danno, se non prova che l'inadempimento è stato determinato da impossibilità."):
    return {"_article": (str(n), code), "s": {"name": f"{n}.0.0", "plain_text": text, "titolo_articolo": f"Rubrica {n}"},
            "d": {"id": "LEGAL_DOC::cc", "name": "Codice Civile 2026"}}


def _topic_ten():
    arts = [f"{1200 + i} c.c." for i in range(1, 11)]        # ten listed, planned ten: the first eight are «planned»
    grades = {a: (3 if i < 3 else 1) for i, a in enumerate(arts[:8])}     # three graded 3, five graded 1 (planned)
    grades[arts[8]] = 2                                        # the ninth: graded 2, beyond the cut
    grades[arts[9]] = 1                                        # the tenth: graded 1
    return {"title": "Inadempimento", "anchor": arts[0], "articles": arts, "planned": 10, "matched": arts[:1],
            "grades": grades, "graded": 10, "source": "frame"}, arts


def test_a_record_beyond_the_topics_first_eight_is_writable_by_grade_2_not_by_grade_1(monkeypatch):
    events = _events(monkeypatch)
    topic, arts = _topic_ten()
    rows = [_row(1200 + i) for i in range(1, 11)]
    records = I.build([topic], rows, {})
    assert flow.TOPIC_ARTICLES == 8 and len(flow._topic_refs(topic)) == 8
    ninth, tenth = records[arts[8]], records[arts[9]]
    assert ninth["grade"] == 2 and ninth["dropped"] is None and ninth["requisites"]       # writable: split into elements
    assert tenth["grade"] == 1 and tenth["dropped"] == "elencato, non scrivibile (né graduato né del tema)"
    assert tenth["requisites"] == []
    assert all(records[a]["dropped"] is None for a in arts[:8])                           # the planned ones as before
    logged = [d for e, d in events if e == "interactive_writable_by_grade"]
    assert logged == [{"articles": [arts[8]]}]
    # no record writable by grade: nothing logged
    events.clear()
    topic2 = {**topic, "grades": {a: 3 for a in arts[:8]}}
    I.build([topic2], rows, {})
    assert not [d for e, d in events if e == "interactive_writable_by_grade"]


def test_the_record_writable_by_grade_is_written_with_its_topic(monkeypatch):
    """The audit of the _topic_refs sites that feed the writer: the writer's refs, the link's labels, the plan issues
    and the fallback plan all carry the ninth article with its topic."""
    topic, arts = _topic_ten()
    rows = [_row(1200 + i) for i in range(1, 11)]
    records = I.build([topic], rows, {})
    available = {flow._key(r["_article"]) for r in rows}
    refs = flow.capped_refs([topic], cap=flow.SINGLE_SHOT_ARTICLES, records=records, available=available)
    labels = [I._label(r) for r in refs]
    # the writer's refs rank every graded article by grade (capped_refs): the ninth (grade 2) is in, before the planned
    # grade-1 ones; the cap of SINGLE_SHOT_ARTICLES then decides among the grade-1 ones
    assert arts[8] in labels and labels.index(arts[8]) == 3 and len(labels) == 10 <= flow.SINGLE_SHOT_ARTICLES
    seen = {}

    def link(records_, topics, refs_by_topic, sents):
        seen["labels"] = [list(x) for x in refs_by_topic]
        return {"topics": [], "total_ms": 0}
    monkeypatch.setattr(I, "link", link)
    state = {"records": records, "sentences": [{"id": "S1", "text": "Alfa non ha pagato.", "certainty": "accertato",
                                                "source": "caso"}], "text": "Alfa non ha pagato."}
    flow.link_records(state, [topic], set())
    assert arts[8] in seen["labels"][0] and arts[9] not in seen["labels"][0] and arts[0] in seen["labels"][0]
    plan = flow.fallback_plan({"parte_assistita": "Alfa"}, [topic], refs) if hasattr(flow, "fallback_plan") else None
    src = (ROOT / "src" / "rag" / "interactive_flow.py").read_text(encoding="utf-8")
    assert '"norme": [r for r in _topic_refs(t, all_=True) if r in refs]' in src          # the fallback plan
    assert "by_topic = [[r for r in _topic_refs(t, all_=True) if _key(r) in wanted] for t in topics]" in src
    if plan is not None:
        assert flow._parse_label(arts[8]) in plan["questioni"][0]["norme"]


# --- B. a grade of 3 is not demoted by one contradicted element ----------------------------------------------------------------

def _sent(n, text):
    return {"id": f"S{n}", "text": text, "certainty": "accertato", "source": "caso"}


def _demote(monkeypatch, grade, excluded=False):
    rows = [{"_article": ("9999", "c.c."), "_grade": grade, "s": {}}]
    el = {"comma": 1, "text": "con violenza o minaccia", "kind": "requisito"}
    monkeypatch.setattr(I, "distinctive", lambda rows, analysis: {"9999 c.c.": el})
    monkeypatch.setattr(I, "sentences", lambda case: [_sent(1, "Alfa si è comportato correttamente con Beta.")])
    if excluded:
        monkeypatch.setattr(flow, "facts_status", lambda requisite, analysis: flow.NON_RISULTA)
        monkeypatch.setattr(flow, "excluding_facts", lambda element, analysis: ["nessuna violenza"])
    log = I.demote_contradicted(rows, "x", {}, call=lambda topic, reqs, sents: {
        r["id"]: {"sostiene": [], "contraddice": ["S1"], "manca": ""} for _, r in reqs})
    return rows[0], log


def test_grade_3_contradicted_is_kept_grade_2_demoted_grade_3_fact_excluded_demoted(monkeypatch):
    row, log = _demote(monkeypatch, 3)
    assert row["_grade"] == 3 and not row.get("_demoted") and log["demoted"] == [] and log["kept_grade3"] == ["9999 c.c."]
    row, log = _demote(monkeypatch, 2)
    assert row["_grade"] == 1 and row["_demoted"] and row["_grade_before"] == 2 and "kept_grade3" not in log
    assert log["demoted"][0]["reason"] == I.DEMOTED_CONTRADICTED
    row, log = _demote(monkeypatch, 3, excluded=True)
    assert row["_grade"] == 1 and row["_demoted"] and row["_grade_before"] == 3 and "kept_grade3" not in log
    assert log["demoted"][0]["reason"] == I.DEMOTED_EXCLUDED


# --- C. the cap -------------------------------------------------------------------------------------------------------------

def test_the_cap_is_fifty_below_the_prefilter_and_the_cost_is_logged():
    assert flow.GRADE_CAP == 50 and dg.GRADE_PREFILTER_N == 50 and flow.GRADE_CAP <= dg.grade_prefilter_n()
    src = (ROOT / "src" / "rag" / "interactive_flow.py").read_text(encoding="utf-8")
    assert '"cap": GRADE_CAP, "hits": hits_n' in src and '"cap": GRADE_CAP}' in src       # grading_calls, grade_funnel
    rows = [{"_article": (str(100 + i), "c.c."), "_rank": i} for i in range(1, 71)]
    out, log = flow.grade_cap(rows, ["170 c.c."], anchors=["169 c.c."])
    kept = [f"{r['_article'][0]} {r['_article'][1]}" for r in out]
    assert len(kept) == 50 and "170 c.c." in kept and "169 c.c." in kept and log["anchors_kept"] == ["169 c.c."]


# --- Part 2: the opinion writer --------------------------------------------------------------------------------------------

PROSE = ("Dario Melis, cassiere della Ferramenta Lunardi, è indagato per appropriazione indebita ai sensi dell'art. 646 "
         "c.p. La condotta richiede l'appropriazione del denaro altrui e il fine di ingiusto profitto. "
         "L'art. 2043 c.c. impone comunque il risarcimento del danno. "
         "La giurisprudenza (Cass. n. 123/2020) esclude il reato quando manca il profitto. "
         "La querela va proposta entro 90 giorni dalla conoscenza del fatto. "
         "Il fatto che Dario Melis custodisse la cassa resta da provare. "
         "Le attenuanti generiche dell'art. 62-bis c.p. possono ridurre la pena.")


def test_flag_off_no_call_no_opinion_keys_the_answer_is_the_page(monkeypatch, tmp_path):
    def never(*a, **k):
        raise AssertionError("the opinion writer was called with the flag off")
    monkeypatch.setattr(O, "write", never)
    run = H.run(monkeypatch, tmp_path, _case(CASE_ID="v79199_off"))
    assert "opinion_text" not in run["out"] and "page_text" not in run["out"]
    assert "opinion_text" not in run["state"] and "opinion_writer" not in run["state"]
    assert not [e for e, _, _ in run["events"] if e == "interactive_opinion_writer"]
    assert "Linea principale: art. 646 c.p." in H.section(run["answer"], "sintesi")


def test_flag_on_the_opinion_is_the_answer_checked_the_page_kept(monkeypatch, tmp_path):
    seen = {}

    def call(messages, max_tokens):
        seen["system"], seen["human"], seen["max_tokens"] = messages[0].content, messages[1].content, max_tokens
        return PROSE
    monkeypatch.setattr(O, "_default_call", call)
    run = H.run(monkeypatch, tmp_path, _case(CASE_ID="v79199_on", ENV={"OPINION_WRITER": "true"}))
    out, state, answer = run["out"], run["state"], run["answer"]
    assert [x for x in run["failed"] if x.split(":")[0] in ("opinion_writer", "opinion_answer")] == []
    # the call: the shipped system prompt, the case, the article texts (the line first), the rendered section
    assert seen["system"] == O.SYSTEM and seen["max_tokens"] == 3000
    assert seen["system"].startswith("Sei un avvocato italiano con lunga esperienza.")
    assert seen["system"].endswith('"Profili processuali (calcolati)" riportato sotto, senza aggiungerne altri.')
    human = seen["human"]
    assert human.startswith("CASO:\n" + P.TEXT)
    assert human.index("ARTICOLO: 646 c.p.") < human.index("ARTICOLO: 62-bis c.p.")
    assert P.ART_62BIS[:60] in human and f"{O.PROCEDURE_HEADING}:\nRENDERED BLOCK" in human
    # both texts in the result and in the state; the answer is the opinion
    assert out["opinion_text"] and out["page_text"] and out["opinion_text"] == state["opinion_text"]
    assert "Linea principale: art. 646 c.p." in out["page_text"] and "Linea principale" not in out["opinion_text"]
    assert answer.startswith(out["opinion_text"][:80])
    # the checks: the uncited article's sentence removed, the ruling stripped, the term flagged, the fact counted
    assert "2043 c.c." not in out["opinion_text"] and "risarcimento del danno" not in out["opinion_text"]
    assert "Cass." not in out["opinion_text"] and "123/2020" not in out["opinion_text"]
    assert "esclude il reato quando manca il profitto" in out["opinion_text"]
    assert "entro 90 giorni dalla conoscenza del fatto. [da verificare]" in out["opinion_text"]
    assert "art. 646 c.p." in out["opinion_text"] and "art. 62-bis c.p." in out["opinion_text"]
    # the code-rendered procedural section as its own section, after the prose
    assert f"**5. {ptw.SECTION_TITLES[4]}**\n\nRENDERED BLOCK" in out["opinion_text"]
    assert out["opinion_text"].index("attenuanti generiche") < out["opinion_text"].index("RENDERED BLOCK")
    logged = [d for e, d, _ in run["events"] if e == "interactive_opinion_writer"]
    assert len(logged) == 1
    checks = logged[0]["opinion_checks"]
    assert checks["articles_removed"][0]["articles"] == ["2043 c.c."]
    assert checks["rulings_stripped"] == ["(Cass. n. 123/2020)"]
    assert checks["terms_flagged"][0]["terms"] == ["entro 90 giorni"]
    assert len(checks["facts_contested"]) == 1 and checks["facts_contested"][0]["article"] == "646 c.p."
    assert "custodisse la cassa" in checks["facts_contested"][0]["sentence"]
    assert logged[0]["opinion_text"] == out["opinion_text"] and logged[0]["articles"] == ["646 c.p.", "62-bis c.p."]
    assert logged[0]["lines"] == ["646 c.p."] and logged[0]["error"] == ""
    assert "opinion_text" in verbose_logger.FULL_KEYS and "opinion_checks" in verbose_logger.FULL_KEYS
    # the step-aside path's log for the prose
    assert [d for e, d, _ in run["events"] if e == "interactive_step_aside_answer"]


def test_a_failed_opinion_call_leaves_the_page(monkeypatch, tmp_path):
    def broken(messages, max_tokens):
        raise RuntimeError("the opinion model is down")
    monkeypatch.setattr(O, "_default_call", broken)
    run = H.run(monkeypatch, tmp_path, _case(CASE_ID="v79199_broken", ENV={"OPINION_WRITER": "true"}))
    assert "opinion_text" not in run["out"] and "Linea principale: art. 646 c.p." in H.section(run["answer"], "sintesi")
    assert run["state"]["opinion_writer"]["text"] == "" and "RuntimeError" in run["state"]["opinion_writer"]["error"]


def test_the_checks_alone():
    verified = {"analysis": {"elementi": {"646 c.p.": [{"elemento": "A646cp:E3", "esito": "provato", "cita_fatti": ["S2"]}]}},
                "sentences": {"S2": "Dario Melis lavorava come cassiere e custodiva la cassa del negozio."}}
    text, log = O.checks(PROSE, ["646 c.p.", "62-bis c.p."], "Querela entro 90 giorni.", verified)
    assert "2043" not in text and "Cass." not in text and "[da verificare]" not in text     # the term is in rendered
    assert log["terms_flagged"] == [] and len(log["facts_contested"]) == 1
    text, log = O.checks("La prescrizione è di cinque anni. L'azione si propone entro sei mesi dal 12/03/2026.",
                         [], "", None)
    assert text.count("[da verificare]") == 2 and [t["terms"] for t in log["terms_flagged"]] == [
        ["prescrizione", "cinque anni"], ["entro sei mesi", "dal 12/03/2026"]]
    assert O.checks("", [], "", None) == ("", {"articles_removed": [], "rulings_stripped": [], "terms_flagged": [],
                                              "facts_contested": []})
    text, log = O.checks("Lo dice la sent. n. 44/2019 della Corte.", [], "", None)
    assert text == "Lo dice la della Corte." or text == "Lo dice la  della Corte." or "44/2019" not in text
    assert log["rulings_stripped"] == ["sent. n. 44/2019"]


def test_the_material_cuts_each_article_and_the_whole():
    long = "x" * 10000
    rows = [{"_article": ("1", "c.c."), "s": {"plain_text": long, "titolo_articolo": "Uno"}},
            {"_article": ("2", "c.c."), "s": {"plain_text": "y" * 9000}},
            {"_article": ("3", "c.c."), "s": {"plain_text": "z" * 9000}},
            {"_article": ("4", "c.c."), "s": {"plain_text": "w" * 9000}},
            {"_article": ("5", "c.c."), "s": {"plain_text": "v" * 9000}},
            {"_article": ("6", "c.c."), "s": {"plain_text": "u" * 100}}]
    got = O.article_texts([f"{n} c.c." for n in range(1, 8)], rows, {})
    assert got[0]["chars"] == 9000 and got[0]["cut"] is True and got[0]["rubrica"] == "Uno"
    assert sum(t["chars"] for t in got) == 45000 and got[5]["omitted"] == "total" and got[6]["omitted"] == "no_text"
    state = {"records": {"640 c.p.": {"grade": 3}, "62-bis c.p.": {"grade": 2}, "2043 c.c.": {"grade": 1},
                         "1218 c.c.": {"grade": 3, "dropped": "prefiltro"}, "641 c.p.": {"grade": None}}}
    verified = {"analysis": {"linea_principale": {"articolo": "641 c.p."}, "linea_subordinata": {"articolo": "640 c.p."}}}
    assert O.article_labels(state, verified) == ["641 c.p.", "640 c.p.", "62-bis c.p."]
    assert O.article_labels(state, None) == ["640 c.p.", "62-bis c.p."]


def test_the_flag_is_registered_and_the_prompt_is_the_row_112_text():
    import conftest
    assert "OPINION_WRITER" in conftest.FLAGS and not O.enabled()
    row = (Path("/Users/flavioporocani/Downloads/strategy_review_ceiling_28nov.md"))
    assert O.SYSTEM_ROW_112.startswith("Sei un avvocato italiano con lunga esperienza.")
    assert O.SYSTEM_ROW_112.endswith("Massimo 1.200 parole.")
    assert O.SYSTEM == O.SYSTEM_ROW_112 + "\n" + O.SYSTEM_RAILS and O.SYSTEM_RAILS.count("\n") == 2
    docs = (ROOT / "docs" / "FLAGS.md").read_text(encoding="utf-8")
    assert "| `OPINION_WRITER` | false |" in docs
