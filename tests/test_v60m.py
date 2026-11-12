"""v60m: round 1 demotes an article to grade 1 only when its distinctive element is contradicted by a sentence of the case
narrated as a fact or excluded by the extracted facts -- an element the case is silent on never demotes, the article
keeps its grade and the earned-block rule decides whether it is written; distinctive_link lists each demotion with its
reason and the sentence (or the facts) behind it; the sim's report sets the topic recall beside an earlier run's. The
model and the graph are stubbed."""

import copy
import importlib.util
import json
import os
import sys
from pathlib import Path

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from src.rag import answer_ledger, issue_records  # noqa: E402
from src.rag import interaction_copy as C  # noqa: E402
from src.rag import interactive_flow as flow  # noqa: E402
from test_v60f import _graded_state  # noqa: E402
from test_v60k import WRITER, _single  # noqa: E402

NORMS = flow.ANSWER_TITLES[1]
NT_PREFIX = C.OTHER_NORMS_LINE.split("{")[0]
CASE = "Leonardo passa sul fondo. Domitilla ha chiuso il cancello nel 2016. Domitilla sostiene che la strada fosse sua."


def _rows(state):
    rows = flow._restored_rows(state["rows"])
    for r in rows:
        r["_grade"] = 2
    return rows


def _link(contradicting=()):
    return lambda topic, reqs, sents: {r["id"]: {"sostiene": [], "contraddice": list(contradicting), "manca": "x"}
                                       for _, r in reqs}


def test_an_element_the_case_is_silent_on_never_demotes(monkeypatch):
    state = _graded_state(monkeypatch)
    rows = _rows(state)
    log = issue_records.demote_contradicted(rows, CASE, {}, call=_link())
    assert log["demoted"] == [] and all(r["_grade"] == 2 and not r.get("_demoted") for r in rows)
    assert log["silent"] == log["answered"] == log["elements"] > 0
    assert all(r.get("_distinctive") for r in rows if issue_records._label(r["_article"]) in
               issue_records.distinctive(rows, {}))                    # the link still seeds the record


def test_a_sentence_narrated_as_fact_contradicting_it_demotes_and_the_log_cites_it(monkeypatch):
    state = _graded_state(monkeypatch)
    rows = _rows(state)
    log = issue_records.demote_contradicted(rows, CASE, {}, call=_link(["S3"]))     # S3: an allegation
    assert log["demoted"] == [] and all(r["_grade"] == 2 for r in rows)
    rows = _rows(state)
    log = issue_records.demote_contradicted(rows, CASE, {}, call=_link(["S2"]), keep=["1158 c.c."])
    by = {issue_records._label(r["_article"]): r for r in rows}
    entry = next(d for d in log["demoted"] if d["article"] == "1062 c.c.")
    assert entry["reason"] == issue_records.DEMOTED_CONTRADICTED and entry["grade"] == 2
    assert entry["sentences"] == [{"id": "S2", "text": "Domitilla ha chiuso il cancello nel 2016.",
                                   "certainty": "accertato"}]
    assert by["1062 c.c."]["_grade"] == 1 and by["1062 c.c."]["_grade_issue"].startswith(
        f"elemento distintivo {issue_records.DEMOTED_CONTRADICTED}: «")
    assert by["1158 c.c."]["_grade"] == 2 and "1158 c.c." in log["kept_named"]          # named by the case


def test_an_element_the_extracted_facts_exclude_demotes_even_without_the_link(monkeypatch):
    state = _graded_state(monkeypatch)
    analysis = copy.deepcopy(flow._restored_analysis(state["analysis"]))
    analysis["fact_signals"]["facts"]["access"] = [{"landlocked": "no"}]               # the fondo has an access
    rows = _rows(state)
    log = issue_records.demote_contradicted(rows, CASE, analysis,
                                            call=lambda *a: (_ for _ in ()).throw(RuntimeError("down")))
    assert [d["article"] for d in log["demoted"]] == ["1051 c.c."] and log["errors"]
    entry = log["demoted"][0]
    assert entry["reason"] == issue_records.DEMOTED_EXCLUDED and entry["facts"] == ["landlocked"]
    rows = _rows(state)
    log = issue_records.demote_contradicted(rows, CASE, analysis, call=_link(["S2"]))
    by = {d["article"]: d for d in log["demoted"]}
    assert "1061 c.c." not in by                         # the facts prove it: a contradiction does not override them
    assert by["1051 c.c."]["reason"] == issue_records.DEMOTED_EXCLUDED and by["1051 c.c."]["sentences"][0]["id"] == "S2"


def test_a_kept_silent_article_is_written_only_when_its_block_is_earned(monkeypatch):
    state, seen = _single(monkeypatch)
    state["topics"][1]["hypotheses"] = [{"line": "principale", "claim": "Passaggio coattivo"}]    # v60n: covered
    state["questions"] = [q for q in state["questions"] if q["id"] not in ("accesso_alternativo", "opera_separazione_0")]
    flow.refresh_records(state)                 # 1061 and 1051: silent, no support, no open question
    seen["writer"] = {**WRITER, NORMS: "[1] L'art. 1062 c.c. fa sorgere la servitù.\n\n[2] L'art. 1051 c.c. dà il "
                                       "passaggio.\n\n[3] L'art. 1158 c.c. vuole vent'anni di possesso."}
    norms = dict(answer_ledger.sections_of(flow.single_shot(CASE)["answer"]))[NORMS]
    assert "**Art. 1061 c.c.**" not in norms and NT_PREFIX not in norms    # not an anchor: no block; v60r: no line
    heads = {r["article"]: r for r in state["ledger_rows"] if r["requisite"] is None and r["comma"] is None}
    assert heads["1061 c.c."]["status"] == "non trattata" and not heads["1061 c.c."]["in_body"]
    assert "**Art. 1051 c.c.**\nEsito sui fatti del caso: da provare" in norms   # v60p: a covered topic's anchor
    assert "**Art. 1158 c.c.**" in norms        # silent too, but its missing fact is an open question


# --- the sim: topic recall beside an earlier run, the demotions per case ---------------------------------------------------

def _dl():
    spec = importlib.util.spec_from_file_location("diagnose_long", HERE.parent / "scripts" / "diagnose_long.py")
    dl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dl)
    return dl


def _run(path, recalls, demoted=()):
    (path / "cases").mkdir(parents=True)
    for cid, (hit, top3, expected) in recalls.items():
        record = {"id": cid, "kind": "interactive", "status": "ok", "rounds": [],
                  "topic_recall": {"in_topics": hit, "in_top3": top3, "expected": expected},
                  "grading": {"distinctive": {"demoted": list(demoted)}}}
        (path / "cases" / f"{cid}.json").write_text(json.dumps(record), encoding="utf-8")


def test_the_sim_report_sets_topic_recall_beside_an_earlier_run_and_lists_the_demotions(tmp_path):
    dl = _dl()

    class Ev:
        @staticmethod
        def mentions(answer):
            return []
    _run(tmp_path / "v66", {"h3": (4, 3, 5), "h5": (3, 2, 4)})
    demoted = [{"article": "1062 c.c.", "reason": issue_records.DEMOTED_CONTRADICTED,
                "sentences": [{"id": "S2", "text": "Domitilla ha chiuso il cancello nel 2016.", "certainty": "accertato"}]},
               {"article": "1051 c.c.", "reason": issue_records.DEMOTED_EXCLUDED, "facts": ["landlocked"], "sentences": []}]
    _run(tmp_path / "now", {"h3": (4, 3, 5), "h5": (2, 2, 4)}, demoted)
    report = Path(dl.write_interactive_report(str(tmp_path / "now"), Ev(), str(tmp_path / "v66"))).read_text("utf-8")
    assert "## Topic recall vs v66" in report
    assert "| h3 | 4/5 (3) | 4/5 (3) | +0 | +0 |" in report and "| h5 | 3/4 (2) | 2/4 (2) | -1 | +0 |" in report
    assert "| **all** | 7/9 (5) | 6/9 (5) | -1 | +0 |" in report and "Below v66: h5 (-1, top 3 +0)" in report
    assert ("Demoted to grade 1 (distinctive element): 1062 c.c. — contraddetto da una frase del caso narrata come "
            "fatto: S2 «Domitilla ha chiuso il cancello nel 2016.» (accertato); 1051 c.c. — escluso dai fatti estratti "
            "[fatti: landlocked]") in report
    assert dl.demotions_line({"grading": {"distinctive": {"demoted": ["1061 c.c."]}}}) == "1061 c.c."   # an older run
