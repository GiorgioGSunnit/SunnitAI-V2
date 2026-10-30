"""The steps after writing, fail-safe (plan_then_write._step, defensive_generation._safe_step), and the second-round
configuration -- ANCHORED_WRITING + ROLE_BLOCKS + GROUNDED_REVIEW + GROUNDED_REVIEW_STRICT -- on an administrative-style
case with no case law and nothing to remove (h10, h11, n4 crashed with KeyError: 'removed'). Every article here is
invented; no model, no database."""
import os

os.environ.setdefault("EMBEDDING_PROVIDER", "openai")
os.environ.setdefault("EMBEDDING_API_KEY", "test-key")
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test-password")

import pytest  # noqa: E402

from src.rag import anchored_writing as aw  # noqa: E402
from src.rag import comma_material as cm  # noqa: E402
from src.rag import defensive_generation as dg  # noqa: E402
from src.rag import fact_signals  # noqa: E402
from src.rag import grounded_review as gr  # noqa: E402
from src.rag import plan_then_write as ptw  # noqa: E402

TERM = "Il ricorso immaginario si propone entro sessanta giorni dalla notificazione del provvedimento."
STAY = "Con il ricorso immaginario si può chiedere la sospensione del provvedimento impugnato."
CASE = "Il Comune di Borgoalto ha negato a Sempronia il permesso di costruire il 3 marzo; la notifica è del 5 marzo."
SECTIONS = {
    "Norme e giurisprudenza": f"{TERM[:-1]} (art. 9029 c.p.a.) [A1]. {STAY[:-1]} (art. 9055 c.p.a.) [A2].",
    "Applicazione al caso": "Sempronia ha ricevuto la notifica il 5 marzo e può ancora impugnare il diniego del Comune.",
    "Conclusioni": "Sempronia può impugnare il diniego entro il termine e chiederne la sospensione.",
}
MATERIAL = f"TESTI DELLE NORME:\n[art. 9029 c.p.a.]\n{TERM}\n[art. 9055 c.p.a.]\n{STAY}"


def _rows():
    return [{"_article": ("9029", "c.p.a."), "_grade": 3, "s": {"plain_text": TERM, "titolo_articolo": "Termine"},
             cm.COMMI_KEY: [TERM], cm.FROM_KEY: "sections"},
            {"_article": ("9055", "c.p.a."), "_grade": 2, "s": {"plain_text": STAY, "titolo_articolo": "Sospensione"},
             cm.COMMI_KEY: [STAY], cm.FROM_KEY: "sections"}]


def _plan():
    return {"parte_assistita": "Sempronia", "posizione": "ricorrente", "obiettivo": "", "questione_centrale": "",
            "risposte": [], "subordinate": [], "punti_deboli": [], "pronunce_contrarie": [], "questioni": [
                {"titolo": "Termine e sospensione", "norme": [("9029", "c.p.a."), ("9055", "c.p.a.")],
                 "pronunce": [], "argomento": TERM + " " + STAY, "obiezione": "", "replica": "", "passi": []}]}


@pytest.fixture
def second_round(monkeypatch):
    for flag in (aw.FLAG, fact_signals.ROLE_BLOCKS_FLAG, gr.FLAG, gr.STRICT_FLAG):
        monkeypatch.setenv(flag, "true")
    events = []
    monkeypatch.setattr(dg, "vlog", lambda event, data=None, **_: events.append((event, data)))
    monkeypatch.setattr(aw, "vlog", lambda event, data=None, **_: events.append((event, data)))
    monkeypatch.setattr(gr, "vlog", lambda event, data=None, **_: events.append((event, data)))
    return events


def _write(monkeypatch, judge):
    monkeypatch.setattr(dg, "_call_chat", lambda messages, max_tokens=None, **k: SECTIONS[
        next(t for t in SECTIONS if f"{t}-istr" in messages[0].content)])
    monkeypatch.setattr(dg, "_chat_json", judge)
    plan = _plan()
    cat = aw.build(plan, _rows(), {})
    cat.section_ids = {t: ["A1", "A2"] for t in SECTIONS}
    cat.checked = [("9029", "c.p.a."), ("9055", "c.p.a.")]
    jobs = [(t, f"{t}-istr", MATERIAL, 400) for t in SECTIONS]
    return ptw.write_sections(plan, jobs, "it", case_text=CASE, checked=cat.checked, anchors=cat,
                              article_rows=_rows())


def _all_supported(messages, max_tokens, **k):
    system, human = messages[0].content, messages[1].content
    assert system == gr.JUDGE_SYSTEM, "no rewrite call: nothing fails"
    lines = [ln for ln in human.split("FRASI:\n", 1)[1].splitlines() if ln.strip()]
    return [{"i": int(ln.split(".", 1)[0]), "tipo": "applicazione" if "Sempronia" in ln else "regola",
             "esito": "supportata"} for ln in lines]


def test_all_four_flags_on_an_administrative_case_without_case_law_or_removals(monkeypatch, second_round):
    out = _write(monkeypatch, _all_supported)
    assert [t for t, _ in out] == list(SECTIONS)
    norme = dict(out)["Norme e giurisprudenza"]
    assert norme == f"{TERM[:-1]} (art. 9029 c.p.a.). {STAY[:-1]} (art. 9055 c.p.a.)."      # anchors stripped only
    assert dict(out)["Applicazione al caso"] == SECTIONS["Applicazione al caso"]
    anchored = [d for e, d in second_round if e == "anchored_writing"][-1]["totals"]
    assert anchored["removed"] == 0 and anchored["case_law"] == 0 and anchored["rewritten"] == 0
    grounded = [d for e, d in second_round if e == "grounded_review"][-1]
    assert grounded["counts"]["dropped"] == 0 and grounded["counts"]["citations_removed"] == 0
    assert not [d for e, d in second_round if e == "post_writing_step_failed"]


def test_a_dangling_entry_in_the_citation_discipline_log_no_longer_raises(monkeypatch, second_round):
    monkeypatch.setenv(ptw.KEEP_CHECKED_FLAG, "true")
    monkeypatch.setenv(dg.CHECK_GRADED_FLAG, "true")
    # what drop_dangling logs: no "removed" list (the KeyError of h10, h11, n4)
    monkeypatch.setattr(ptw, "_cite_only_material", lambda title, text, *a: (text, [
        {"section": title, "kind": "dangling", "how": "sentence dropped", "sentence": "Tale norma si applica."}]))
    out = _write(monkeypatch, _all_supported)
    assert out is not None and not [d for e, d in second_round if e == "post_writing_step_failed"]
    log = [d for e, d in second_round if e == "case_analysis_citation_discipline"][-1]
    assert log["count"] == 0 and log["dangling"] == 3


@pytest.mark.parametrize("module,name", [(aw, "verify"), (gr, "review")])
def test_a_failing_step_keeps_the_sections_as_they_were(monkeypatch, second_round, module, name):
    def boom(*a, **k):
        raise KeyError("removed")
    monkeypatch.setattr(module, name, boom)
    out = _write(monkeypatch, _all_supported)
    norme = dict(out)["Norme e giurisprudenza"]
    assert "[A" not in norme and "(art. 9029 c.p.a.)" in norme and "(art. 9055 c.p.a.)" in norme
    failed = [d for e, d in second_round if e == "post_writing_step_failed"]
    assert [f["step"] for f in failed] == ["anchored_writing" if module is aw else "grounded_review"]
    assert "KeyError" in failed[0]["error"]


def test_safe_step_on_the_answer(monkeypatch):
    events = []
    monkeypatch.setattr(dg, "vlog", lambda event, data=None, **_: events.append((event, data)))

    def boom(d):
        raise KeyError("removed")
    assert dg._safe_step("text_tidy", "La risposta.", boom) == "La risposta."
    assert dg._safe_step("role", "La risposta.", lambda d: "") == "La risposta."       # an empty answer is a failure
    assert dg._safe_step("ok", "La risposta.", lambda d: d + " Altro.") == "La risposta. Altro."
    assert [d["step"] for e, d in events if e == "post_writing_step_failed"] == ["text_tidy", "role"]
