"""v49: protected citations (the five v48 ledger removals, rebuilt from the ledger's own sentence starts and reasons --
the v48 dumps are not here), the qualification check's reason in full, and PLAN_CAP. No model, no database."""
import json
import os
import re

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

import pytest  # noqa: E402

from src.rag import anchored_writing as aw, citation_ledger, text_tidy  # noqa: E402
import src.rag.defensive_generation as dg  # noqa: E402
import src.rag.plan_then_write as ptw  # noqa: E402


@pytest.fixture
def events(monkeypatch):
    seen = []
    monkeypatch.setattr(dg, "vlog", lambda name, data=None, *a, **k: seen.append((name, data)))
    monkeypatch.setattr(aw, "vlog", lambda name, data=None, *a, **k: seen.append((name, data)))
    yield seen
    citation_ledger.protect(())


def kept_events(events):
    return [d for n, d in events if n == citation_ledger.EVENT and d.get("kept") == citation_ledger.KEPT]


# --- (2) answer_tidy: c3 1587 c.c., h4 485 c.c. -- "cut clause inside a sentence" -----------------------------------------

C3_1587 = ("L’art. 1587 c.c. impone al conduttore l’obbligo di prendere in consegna la cosa e di osservare la diligenza "
           "del buon padre di famiglia nel servirsene. per l'uso determinato nel contratto o per quello presunto.")
H4_485 = ("L'art. 485 c.c. disciplina l'obbligo del chiamato all'eredità, in possesso di beni ereditari, di fare "
          "l'inventario entro tre mesi dall'apertura della successione. trascorso tale termine senza che "
          "l'inventario sia stato compiuto.")


@pytest.mark.parametrize("sentence, ref, first", [(C3_1587, ("1587", "c.c."), "L’art. 1587 c.c. impone"),
                                                 (H4_485, ("485", "c.c."), "L'art. 485 c.c. disciplina")])
def test_a_cut_clause_in_a_plan_article_s_sentence_is_trimmed_not_dropped(events, sentence, ref, first):
    assert text_tidy.unreadable(sentence) == "cut clause inside a sentence"
    assert text_tidy.tidy(sentence)[0] == ""                       # v48: no plan, the sentence goes
    citation_ledger.protect([ref])
    text, log = text_tidy.tidy(sentence)
    assert text.startswith(first) and text.endswith(".") and not text_tidy.unreadable(text)
    assert log[-1]["kind"] == "cut clause inside a sentence: trimmed"
    [e] = kept_events(events)
    assert e["pass"] == "answer_tidy" and e["article"] == f"{ref[0]} {ref[1]}" and e["dropped"] is False


def test_a_plan_article_s_sentence_that_cannot_be_trimmed_is_kept_as_written(events):
    sentence = "L'art. 1587 c.c. si applica sempre. quando il conduttore riceve la cosa."
    citation_ledger.protect([("1587", "c.c.")])
    text, _ = text_tidy.tidy(sentence)
    assert "1587" in text and kept_events(events)


def test_an_article_outside_the_plan_is_still_dropped(events):
    citation_ledger.protect([("1588", "c.c.")])
    assert text_tidy.tidy(C3_1587)[0] == "" and not kept_events(events)


# --- (2) final_checks: l3 2051 c.c. -- "a copy inside a sentence: sentence dropped" ------------------------------------

A2051 = "Ciascuno è responsabile del danno cagionato dalle cose che ha in custodia, salvo che provi il caso fortuito."
L3_2051 = ("L'art. 2051 c.c. dispone che ciascuno è responsabile del danno cagionato dalle cose che ha in custodia, "
           "salvo che provi il caso fortuito, e pertanto il condominio risponde del danno alla signora Bianchi.")
ROWS_2051 = [{"_article": ("2051", "c.c."), "s": {"plain_text": A2051, "titolo_articolo": "Danno cagionato da cosa in custodia"}}]


def test_a_copy_in_a_plan_article_s_sentence_is_quoted_in_place(events):
    out = ptw.final_checks([("Norme e giurisprudenza", L3_2051)], ROWS_2051, {})
    assert out[0][1].strip() == ""                                  # v48: no plan, the sentence goes
    citation_ledger.protect([("2051", "c.c.")])
    out = ptw.final_checks([("Norme e giurisprudenza", L3_2051)], ROWS_2051, {})
    text = out[0][1]
    assert text.startswith("L'art. 2051 c.c. dispone che «ciascuno") and "caso fortuito», e pertanto il condominio" in text
    [e] = kept_events(events)
    assert e["pass"] == "final_checks" and e["article"] == "2051 c.c."
    log = [d for n, d in events if n == "case_analysis_final_checks"][-1]
    assert log["entries"][0]["kind"] == text_tidy._QUOTED_IN_PLACE


def test_a_wrong_statement_in_a_plan_article_s_sentence_loses_only_its_clause(events):
    sentence = ("L'art. 2043 c.c. fonda la pretesa risarcitoria della signora Bianchi per il danno subito, "
                "dovuto all'inadempimento del condominio.")
    context = {"contract": False}             # l4 v48: "a breach with no contract in the facts"
    assert ptw._wrong_statement(sentence, context)
    citation_ledger.protect([("2043", "c.c.")])
    out = ptw.final_checks([("Applicazione al caso", sentence)], [], context)
    assert out[0][1] == "L'art. 2043 c.c. fonda la pretesa risarcitoria della signora Bianchi per il danno subito."
    assert kept_events(events)[0]["now"].startswith("L'art. 2043 c.c. fonda")


# --- (2) anchored_writing: h2 1382 c.c. -- "removed: law unanchored citation" ------------------------------------------

def test_a_plan_article_s_unanchored_statement_is_rewritten_from_its_text_or_kept(events, monkeypatch):
    from test_anchored_writing import _catalogue
    rows = [{"_article": ("1382", "c.c."), "s": {"plain_text": "La clausola, con cui si conviene che, in caso "
                                                               "d'inadempimento o di ritardo, uno dei contraenti è "
                                                               "tenuto a una determinata prestazione, ha l'effetto di "
                                                               "limitare il risarcimento.",
                                                 "titolo_articolo": "Effetti della clausola penale"}}]
    sentence = "Si applica l’art. 1382 c.c."
    monkeypatch.setattr(dg, "_chat_json", lambda *a, **k: [])        # the rewrite gives nothing back
    _, cat = _catalogue([])
    cat.checked = []
    out = aw.verify([("Norme e giurisprudenza", sentence)], cat, [], rows)
    assert "1382" not in out[0][1]                                   # v48: outside the checked material, removed
    _, cat = _catalogue([])
    cat.checked = [("1382", "c.c.")]                                 # v49: write_parere adds the plan's articles
    citation_ledger.protect([("1382", "c.c.")])
    out = aw.verify([("Norme e giurisprudenza", sentence)], cat, [], rows)
    assert "1382 c.c." in out[0][1]
    assert [e["article"] for e in kept_events(events) if e["pass"] == "anchored_writing"] == ["1382 c.c."]


# --- (2) qualification: n2 416-bis c.p. -- dropped, the reason in full ----------------------------------------------------

def test_the_qualification_check_keeps_its_drop_and_logs_its_reason_in_full(monkeypatch, events):
    monkeypatch.setenv("FLAGGED_REMOVE_ENABLED", "true")
    citation_ledger.protect([("416-bis", "c.p.")])
    reason = ("L'art. 416-bis c.p. punisce l'associazione di tipo mafioso, mentre la frase lo usa per un concorso "
              "esterno in un reato contro la pubblica amministrazione: altra materia, e la frase ne trae una pena.")
    rows = [{"d": {"id": "doc-c.p.", "name": "c.p."}, "_article": ("416-bis", "c.p."),
             "s": {"name": "416-bis.0.0", "plain_text": "Chiunque fa parte di un'associazione di tipo mafioso...",
                   "titolo_articolo": "Associazioni di tipo mafioso anche straniere"}}]

    def model(messages, max_tokens=None, temperature=None, **kwargs):
        labels = re.findall(r"^\[([^\]]+)\]$", messages[-1].content, re.MULTILINE)
        return json.dumps([{"articolo": label, "compatibile": False, "perche": reason, "tipo": "materia"}
                           for label in labels])
    monkeypatch.setattr(dg, "_call_chat", model)
    draft = "L’art. 416-bis c.p. prevede la pena della reclusione per l’associazione armata, applicabile a Tizio."
    out = dg._check_qualifications(draft, {}, rows, None, [])
    assert "416-bis" not in out                                      # its drop kept
    [e] = [d for n, d in events if n == citation_ledger.EVENT and d["pass"] == "qualification"]
    assert e["dropped"] and reason in e["reason"] and "qualification mismatch (materia)" in e["reason"]


# --- (3) closure and the eval reader are in test_citation_closure / test_eval_narrative --------------------------------


# --- (1) PLAN_CAP ----------------------------------------------------------------------------------------------------

def _row(n, code, grade=None, signals=(), **tags):
    return {"_article": (n, code), "_grade": grade, "_signals": list(signals),
            "s": {"plain_text": f"Testo {n}.", "titolo_articolo": f"Rubrica {n}"}, **tags}


def test_the_cap_keeps_named_then_grade_3_then_grade_2_with_two_signals_then_rule_insertions():
    rows = ([_row(str(n), "c.c.", 3, ["frame"]) for n in range(1, 6)]                       # 5 grade 3
            + [_row(str(n), "c.c.", 2, ["frame", "distill"]) for n in range(10, 15)]       # 5 grade 2, 2 signals
            + [_row(str(n), "c.c.", 2, ["distill"]) for n in range(20, 24)]               # 4 grade 2, 1 signal: rule
            + [_row("30", "c.c.", 1, ["distill", "frame", "institution"]),                 # the plan's own grade 1
               _row("40", "c.c.", 1), _row("163", "c.p.c.", 2)])                          # named by the user; procedural
    refs = [r["_article"] for r in rows]
    plan = {"questioni": [{"titolo": "A", "argomento": "", "norme": refs, "pronunce": []}],
            "plan_source": {f"{n} c.c.": "rule" for n in range(20, 24)}}
    out, log = ptw.cap_plan(plan, rows, [("40", "c.c.")], 12)
    kept = ptw._plan_refs(out)
    assert ("40", "c.c.") in kept and ("163", "c.p.c.") in kept                 # named always; procedural uncounted
    assert all((str(n), "c.c.") in kept for n in range(1, 6)) and all((str(n), "c.c.") in kept for n in range(10, 15))
    assert log["before"] == 16 and log["after"] == 12 and log["procedural"] == 1
    assert [e["article"] for e in log["available"]][-1] == "30 c.c."          # the plan's other last, after rule ones
    assert out["disponibili"] == [e["article"] for e in log["available"]] and len(out["disponibili"]) == 4
    assert all(r.get(ptw.AVAILABLE_KEY) for r in rows if f"{r['_article'][0]} {r['_article'][1]}" in out["disponibili"])
    assert "21 c.c." not in out["plan_source"]


def test_a_plan_within_the_cap_is_untouched():
    rows = [_row("1", "c.c.", 3), _row("2", "c.c.", 2)]
    plan = {"questioni": [{"titolo": "A", "argomento": "", "norme": [("1", "c.c."), ("2", "c.c.")], "pronunce": []}]}
    out, log = ptw.cap_plan(plan, rows, [], 12)
    assert out is plan and log["available"] == []


def test_no_writer_prompt_names_the_capped_out_articles_and_the_material_keeps_them():
    """v50: the articles past the cap are in the material (tagged, for the closure), in no section's prompt."""
    rows = [_row(str(n), "c.c.", 3, ["frame", "distill"]) for n in range(1, 4)] + [_row("21", "c.c.", 1)]
    plan = {"parte_assistita": "Tizio", "posizione": "", "obiettivo": "", "questione_centrale": "", "risposte": [],
            "questioni": [{"titolo": "A", "argomento": "x", "obiezione": "", "replica": "",
                           "norme": [r["_article"] for r in rows], "pronunce": []}],
            "subordinate": [], "punti_deboli": []}
    capped, log = ptw.cap_plan(plan, rows, [], 3)
    assert capped["disponibili"] == ["21 c.c."] and rows[-1].get(ptw.AVAILABLE_KEY)
    for title, instruction, material, _ in ptw.section_jobs(capped, "Fatti.", "", {}, rows, {}, ""):
        assert "art. 21 c.c." not in material and "Testo 21." not in material, title
    assert any("art. 1 c.c." in material for _, _, material, _ in ptw.section_jobs(capped, "Fatti.", "", {}, rows, {}, ""))


def test_plan_cap_is_off_by_default_and_its_size_can_be_set(monkeypatch):
    from src.rag import feature_flags
    monkeypatch.delenv("PLAN_CAP", raising=False)
    monkeypatch.delenv("PLAN_CAP_MAX", raising=False)
    assert not feature_flags.enabled(ptw.PLAN_CAP_FLAG) and ptw.plan_cap_max() == 12
    monkeypatch.setenv("PLAN_CAP_MAX", "20")
    assert ptw.plan_cap_max() == 20


# --- a plan without a usable issue: made once more before the one-call fallback --------------------------------------

from test_v48 import FIRST  # noqa: E402

UNUSABLE = {"parte_assistita": "Leonardo", "questioni": []}


def _plan_with(monkeypatch, replies):
    calls = []

    def plan_json(messages):
        calls.append(messages)
        reply = replies[min(len(calls), len(replies)) - 1]
        if isinstance(reply, Exception):
            raise reply
        return reply
    monkeypatch.setattr(ptw, "_plan_json", plan_json)
    monkeypatch.setattr(dg, "_opposing_arguments", lambda *a, **k: [])
    monkeypatch.setattr(ptw, "_issue_check", lambda issue, rows, headings: [
        {"articolo": "art. 1062 c.c.", "pertinente": True, "contraddice": False}])
    monkeypatch.setattr(dg, "_chat_json", lambda *a, **k: [])
    rows = [{"_article": ("1062", "c.c."), "_grade": 3, "s": {"plain_text": "La destinazione del padre di famiglia.",
                                                              "titolo_articolo": "R"}}]
    return ptw.make_plan("Leonardo passa sulla strada.", "", {}, "[art. 1062 c.c.]", "", "", rows, {}), calls


def test_a_plan_without_a_usable_issue_is_made_once_more_from_the_same_inputs(monkeypatch, events):
    plan, calls = _plan_with(monkeypatch, [UNUSABLE, FIRST])
    assert plan and plan["questioni"] and len(calls) == 2
    assert calls[0][0].content == calls[1][0].content and calls[0][1].content == calls[1][1].content
    [retry] = [d for n, d in events if n == ptw.PLAN_RETRY_EVENT]
    assert retry["outcome"] == "plan" and retry["issues"] == len(plan["questioni"])
    assert not [n for n, _ in events if n == "plan_then_write_fallback"]


def test_the_retry_failing_too_falls_back_once(monkeypatch, events):
    plan, calls = _plan_with(monkeypatch, [UNUSABLE, UNUSABLE])
    assert plan is None and len(calls) == 2
    assert [d["outcome"] for n, d in events if n == ptw.PLAN_RETRY_EVENT] == ["no plan"]
    [fallback] = [d for n, d in events if n == "plan_then_write_fallback"]
    assert fallback["step"] == "plan without a usable issue" and "after one retry" in fallback["error"]


def test_a_failed_plan_call_is_not_retried(monkeypatch, events):
    plan, calls = _plan_with(monkeypatch, [ValueError("cut reply")])
    assert plan is None and len(calls) == 1 and not [n for n, _ in events if n == ptw.PLAN_RETRY_EVENT]


def test_diagnose_counts_one_call_fallbacks():
    import importlib.util
    import pathlib
    spec = importlib.util.spec_from_file_location("dl", pathlib.Path(__file__).parent.parent / "scripts" / "diagnose_long.py")
    dl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dl)
    from collections import Counter
    f = {"cases": 32, "one_call": 2, "inferred": False, "steps": Counter({"plan without a usable issue": 1,
                                                                       "plan call": 1}),
         "retried": 2, "retry_ok": 1}
    assert dl._fallback_cell(f) == "2 of 32 (plan without a usable issue 1, plan call 1); 2 / 1"
    assert dl._fallback_cell({**f, "inferred": True, "steps": Counter()}) == "2 of 32 (inferred: no plan)"
    assert dl._fallback_cell({}) == "off"
