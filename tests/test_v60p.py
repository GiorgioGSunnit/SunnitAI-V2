"""v60p: every grading call of round 1 at once, bounded by _APPLICABILITY_WORKERS (tests/test_v60o.py, its v60p part);
the anchor article of a covered topic always earns its §2 block, its silent elements «da verificare» (1); §4's terms
and trascrizione line also triggered by the parere's own analysis -- a hypothesis line or a covered topic naming the
institution -- as «da verificare: …» with the fact that would settle it, each line's trigger logged (2). The model and
the graph are stubbed."""

import os
import sys
from pathlib import Path

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from src.rag import answer_ledger, issue_records, render_computed  # noqa: E402
from src.rag import interaction_copy as C  # noqa: E402
from src.rag import interactive_flow as flow  # noqa: E402
from test_v60f import _graded_state  # noqa: E402
from test_v60k import WRITER, _event, _single  # noqa: E402

NORMS = flow.ANSWER_TITLES[1]
NT_PREFIX = C.OTHER_NORMS_LINE.split("{")[0]
CASE = "Leonardo passa sul fondo. Domitilla ha chiuso il cancello nel 2016."


# --- 1. the anchor of a covered topic always earns its block -------------------------------------------------------------

def _silent(verdict_status=C.STATUS_DA_PROVARE):
    return {"requisites": [{"kind": "requisito", "comma": 1, "text": "fondo intercluso", "status": verdict_status,
                            "supporting": [], "contradicting": [], "question": ""}]}


def test_the_anchor_flag_earns_the_block_never_for_a_non_applicable_article():
    rec = _silent()
    assert not issue_records.block_due(rec)                                       # silent: one line
    rec["anchor_due"] = "Passaggio coattivo"
    assert issue_records.block_due(rec) and issue_records.line_reason(rec) is None
    gone = {**_silent(C.STATUS_NON_RISULTA), "anchor_due": "x"}
    gone["requisites"][0]["distinctive"] = True
    assert not issue_records.block_due(gone)                                      # the case excludes it: still a line


def test_anchor_blocks_flags_the_covered_topics_anchors_only_and_anew(monkeypatch):
    state = _graded_state(monkeypatch)
    seen = []
    monkeypatch.setattr(flow, "_log", lambda event, data, *a: seen.append((event, data)))
    state["records"]["1158 c.c."]["anchor_due"] = "stale"
    out = flow.anchor_blocks(state, [state["topics"][1]])
    flagged = {label for label, rec in state["records"].items() if rec.get("anchor_due")}
    assert flagged == {"1051 c.c."} and out[0]["anchor"] == "1051 c.c." and out[0]["block"] is True
    assert dict(seen)["anchor_blocks"] == out


def test_a_covered_topics_silent_anchor_has_its_block_and_its_silent_elements_are_da_provare(monkeypatch):
    state, seen = _single(monkeypatch)                  # v60r (6): «da provare» in the payload, no longer «da verificare»
    state["topics"][1]["hypotheses"] = [{"line": "principale", "claim": "Passaggio coattivo"}]
    state["questions"] = [q for q in state["questions"] if q["id"] != "accesso_alternativo"]
    flow.refresh_records(state)
    seen["writer"] = {**WRITER, NORMS: "[1] L'art. 1062 c.c. fa sorgere la servitù.\n\n[2] L'art. 1051 c.c. dà il "
                                       "passaggio.\n\n[3] L'art. 1158 c.c. vuole vent'anni di possesso."}
    norms = dict(answer_ledger.sections_of(flow.single_shot(CASE)["answer"]))[NORMS]
    block = norms.split("**Art. 1051 c.c.**", 1)[1].split("**Art.", 1)[0]
    assert block.startswith("\nEsito sui fatti del caso: da provare") and "- co. 1" not in block
    rows = [r for r in state["ledger_rows"] if r["article"] == "1051 c.c." and r["requisite"]]
    assert rows and all(r["status"] == C.STATUS_DA_PROVARE for r in rows)
    entry = next(e for e in _event(seen, "anchor_blocks") if e["anchor"] == "1051 c.c.")
    assert entry["earned_before"] is False and entry["block"] is True and entry["silent"] >= 1


# --- 2. §4: the parere's own analysis triggers a term and the trascrizione ------------------------------------------------

USUCAPIONE = {"id": "usucapione_ordinaria", "fact": "il possesso ad usucapionem (il possesso o l'uso del bene e la sua "
              "durata, detti dal caso)", "label": "Usucapione ordinaria", "ref": "1158 c.c."}


def _analysis(own=None, case="Leonardo passa sul fondo.", terms=(), untriggered=(USUCAPIONE,), tools=()):
    return {"store_offences_only": True, flow.CASE_KEY: case, "own_analysis": own or {},
            "fact_signals": {"terms": list(terms), "terms_untriggered": list(untriggered), "tools": list(tools)}}


def _logged(monkeypatch):
    seen = []
    import src.rag.verbose_logger as vl
    monkeypatch.setattr(vl, "vlog", lambda event, data=None, **k: seen.append((event, data)))
    return seen


def test_a_hypothesis_line_naming_the_institution_renders_the_term_to_verify(monkeypatch):
    seen = _logged(monkeypatch)
    own = {"lines": ["Leonardo ha acquistato la servitù per usucapione usucapione"], "topics": []}
    blocks = dict(render_computed.blocks(_analysis(own)))
    assert blocks["Termini"] == ("- Usucapione ordinaria (art. 1158 c.c.): da verificare: il caso non dice il possesso "
                                 "ad usucapionem (il possesso o l'uso del bene e la sua durata, detti dal caso).")
    lines = dict(seen)["computed_triggers"]["lines"]
    assert lines == [{"line": "usucapione_ordinaria", "trigger": "ipotesi",
                      "by": "Leonardo ha acquistato la servitù per usucapione usucapione"}]


def test_a_covered_topic_naming_it_or_holding_its_article_renders_it_too(monkeypatch):
    seen = _logged(monkeypatch)
    own = {"lines": [], "topics": [{"title": "Possesso e acquisto", "text": "Possesso", "articles": ["1158 c.c."]}]}
    assert "da verificare: il caso non dice il possesso" in dict(render_computed.blocks(_analysis(own)))["Termini"]
    assert dict(seen)["computed_triggers"]["lines"][0]["trigger"] == "argomento"
    assert "Termini" not in dict(render_computed.blocks(_analysis({"lines": ["Una servitù di passaggio."],
                                                                   "topics": []})))    # nothing names it: none
    assert "Termini" not in dict(render_computed.blocks(_analysis()))               # no own analysis: as before


def test_a_case_sentence_trigger_still_renders_the_computed_term(monkeypatch):
    seen = _logged(monkeypatch)
    computed = {"id": "usucapione_ordinaria", "text": "Usucapione ordinaria: 20 anni, DECORSO", "ref": "1158 c.c.",
                "trigger": "Leonardo passa sul fondo da vent'anni."}
    own = {"lines": ["usucapione"], "topics": []}
    terms = dict(render_computed.blocks(_analysis(own, terms=[computed], untriggered=())))["Termini"]
    assert "da verificare" not in terms and terms.startswith("- Usucapione ordinaria")
    assert dict(seen)["computed_triggers"]["lines"] == [{"line": "usucapione_ordinaria", "trigger": "frase del caso",
                                                         "by": "Leonardo passa sul fondo da vent'anni."}]


TRASCRIZIONE = {"id": "trascrizione_domanda", "refs": [["2653", "c.c."]], "text": "Trascrizione della domanda giudiziale.",
                "matched": ["trascrizione della domanda giudiziale (azione confessoria) ai sensi dell'art. 2653, n. 1, c.c."]}


def test_the_trascrizione_line_is_triggered_by_a_covered_topic_as_to_verify(monkeypatch):
    seen = _logged(monkeypatch)
    own = {"lines": [], "topics": [{"title": "Azioni a difesa delle servitù",
                                    "text": "Accertamento della servitù e altri provvedimenti di tutela",
                                    "articles": ["1079 c.c."]}]}
    analysis = _analysis(own, case="La lite riguarda la servitù di passaggio sul fondo.", untriggered=(),
                         tools=[TRASCRIZIONE])
    body = dict(render_computed.blocks(analysis))[render_computed.TRASCRIZIONE_TITLE]
    assert body == C.capital(C.fill(C.OPEN_POINT, fact=C.TRASCRIZIONE_FACT)) + "."
    assert {"line": "trascrizione_domanda", "trigger": "argomento", "by": "Azioni a difesa delle servitù"} in \
        dict(seen)["computed_triggers"]["lines"]
    analysis[flow.CASE_KEY] = "Leonardo vuole far accertare la servitù sul fondo."      # the case's own sentence
    assert dict(render_computed.blocks(analysis))[render_computed.TRASCRIZIONE_TITLE] == \
        "Va trascritta la domanda giudiziale (azione confessoria)."
    analysis.update({flow.CASE_KEY: "La lite riguarda la servitù di passaggio sul fondo.", "own_analysis": {}})
    assert render_computed.TRASCRIZIONE_TITLE not in dict(render_computed.blocks(analysis))     # v60i: as before


def test_own_analysis_reads_the_hypotheses_and_the_covered_topics(monkeypatch):
    state = _graded_state(monkeypatch)
    state["read"] = {"hypotheses": [{"line": "principale", "claim": "Servitù per destinazione",
                                     "institutions": ["destinazione del padre di famiglia"], "rubricated": []}]}
    own = flow.own_analysis(state, [state["topics"][0]])
    assert own["lines"] == ["Servitù per destinazione destinazione del padre di famiglia"]
    assert own["topics"][0]["articles"][:2] == ["1062 c.c.", "1061 c.c."] and own["topics"][0]["title"]
