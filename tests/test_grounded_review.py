"""GROUNDED_REVIEW_ENABLED (src/rag/grounded_review.py), ROLE_BLOCKS_ENABLED (fact_signals.fit_to_role, the
offended party's posture checks) and fetch_audit's role filter, captured-but-lost funnel and grounded-review reading.
Every article here is invented; no model, no database."""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("EMBEDDING_PROVIDER", "openai")
os.environ.setdefault("EMBEDDING_API_KEY", "test-key")
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test-password")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import pytest  # noqa: E402

from src.rag import defensive_generation as dg  # noqa: E402
from src.rag import fact_signals, practical_checklist  # noqa: E402
from src.rag import grounded_review as gr  # noqa: E402
from src.rag import plan_then_write as ptw  # noqa: E402

import fetch_audit as fa  # noqa: E402

MATERIAL = ("CASO:\nTizio è caduto nel cortile di Caio.\n\nTESTI DELLE NORME:\n[art. 9002 c.c.]\nIl custode della cosa "
            "risponde del danno cagionato dalla cosa, salvo che provi il caso fortuito.")
SUPPORTED = "Caio risponde del danno cagionato dalla cosa in custodia, salvo il caso fortuito (art. 9002 c.c.)."
FROM_FETCH = "La domanda si prescrive in cinque anni dal giorno del fatto secondo l'art. 9001 c.c. applicabile."
INVENTED = "Il danneggiato deve sempre provare la colpa grave del custode secondo l'art. 9002 c.c. vigente."
MEMORY = "La giurisprudenza ammette comunque un risarcimento forfettario per ogni caduta in un cortile privato."
FILLER = "Si tratta di una questione certamente delicata e meritevole di grande attenzione da parte di tutti."
POINTLESS_CITE = "La materia è regolata con grande cura dal legislatore anche all'art. 9005 c.c. qui."
SECTION = "\n\n".join([SUPPORTED, FROM_FETCH, INVENTED, MEMORY, FILLER, POINTLESS_CITE])
FETCHED = {("9001", "c.c."): "[art. 9001 c.c.]\nIl diritto al risarcimento si prescrive in cinque anni dal fatto."}


def fake_chat_json(messages, max_tokens, **kwargs):
    system, human = messages[0].content, messages[1].content
    if system == gr.JUDGE_SYSTEM:
        lines = [ln for ln in human.split("FRASI:\n", 1)[1].splitlines() if ln.strip()]
        out = []
        for ln in lines:
            n, sentence = ln.split(". ", 1)
            if sentence == SUPPORTED:
                out.append({"i": int(n), "tipo": "regola", "esito": "supportata"})
            elif sentence == FROM_FETCH:
                ok = "si prescrive in cinque anni" in human.split("FRASI:", 1)[0]
                out.append({"i": int(n), "tipo": "regola", "esito": "supportata" if ok else "assente"})
            elif sentence == INVENTED:
                out.append({"i": int(n), "tipo": "regola", "esito": "contraddetta"})
            elif sentence in (MEMORY, POINTLESS_CITE):
                out.append({"i": int(n), "tipo": "massima" if sentence == MEMORY else "regola", "esito": "assente"})
            elif sentence == FILLER:
                out.append({"i": int(n), "tipo": "istituto", "esito": "assente"})
            else:
                out.append({"i": int(n), "tipo": "altro", "esito": None})
        return out
    if system == gr.REWRITE_SYSTEM:
        assert gr.REWRITE_MARK in human
        lines = [ln for ln in human.split(gr.REWRITE_MARK + "\n", 1)[1].splitlines() if ln.strip()]
        out = []
        for ln in lines:
            n, sentence = ln.split(". ", 1)
            if sentence == INVENTED:
                out.append({"i": int(n), "azione": "riscritta", "frase": "Il custode risponde del danno salvo che "
                            "provi il caso fortuito (art. 9002 c.c.)"})
            elif sentence == MEMORY:
                out.append({"i": int(n), "azione": "rinvio", "frase": "Va verificato se la giurisprudenza ammetta un "
                            "risarcimento forfettario."})
            elif sentence == FILLER:
                out.append({"i": int(n), "azione": "riempitivo", "frase": ""})
            elif sentence == POINTLESS_CITE:      # a rewriting that loses its citation: the code's pointer instead
                out.append({"i": int(n), "azione": "riscritta", "frase": "La materia è regolata dal legislatore."})
        return out
    raise AssertionError(system[:40])


@pytest.fixture
def logged(monkeypatch):
    events = []
    monkeypatch.setattr(gr, "vlog", lambda event, data=None, **_: events.append((event, data)))
    monkeypatch.setattr(dg, "_chat_json", fake_chat_json)
    return events


def test_the_review_keeps_supported_rewrites_points_and_drops_only_filler(logged):
    reads = []

    def fetch(refs):
        reads.append(list(refs))
        return {r: FETCHED[r] for r in refs if r in FETCHED}
    jobs = [("Norme e giurisprudenza", "istr", MATERIAL, 900), ("Conclusioni", "istr", "PIANO:\n- nulla", 300)]
    sections = [("Norme e giurisprudenza", SECTION), ("Conclusioni", "Conviene agire subito.")]
    out = gr.review(sections, jobs, [], [], fetch=fetch)
    text = out[0][1]
    assert SUPPORTED in text and FROM_FETCH in text                      # never touched / kept after the read
    assert INVENTED not in text and "salvo che provi il caso fortuito (art. 9002 c.c.)." in text
    assert "Va verificato se la giurisprudenza ammetta un risarcimento forfettario." in text
    assert FILLER not in text
    assert "Su questo punto va verificata la disciplina dell'art. 9005 c.c." in text
    assert out[1] == sections[1]
    # one read for the whole parere, only of the articles cited and not in the material (9002 is given)
    assert reads == [[("9001", "c.c."), ("9005", "c.c.")]]
    log = [d for e, d in logged if e == "grounded_review"][-1]
    assert log["counts"] == {"rewritten": 1, "pointer": 2, "dropped": 1, "kept": 0, "kept_fetched": 1, "abstained": 0,
                             "citations_removed": 0} and not log["strict"]
    assert log["totals"]["rule"] == 6 and log["totals"]["supportata"] == 1 and log["totals"]["contraddetta"] == 1
    assert log["fetched"] == {"art. 9001 c.c.": FETCHED[("9001", "c.c.")]} and log["not_found"] == ["art. 9005 c.c."]
    kept = next(e for e in log["entries"] if e["how"] == "kept_fetched")
    assert kept["fetched"] == ["art. 9001 c.c."] and kept["section"] == "Norme e giurisprudenza"
    assert gr.totals([["grounded_review", log, None], ["other", {}, None]])["pointer"] == 2


def test_a_failed_judge_changes_nothing(monkeypatch):
    def boom(*a, **k):
        raise ValueError("no JSON")
    monkeypatch.setattr(dg, "_chat_json", boom)
    monkeypatch.setattr(gr, "vlog", lambda *a, **k: None)
    sections = [("Norme e giurisprudenza", SECTION)]
    assert gr.review(sections, [("Norme e giurisprudenza", "", MATERIAL, 900)], [], [], fetch=lambda r: {}) == sections


def test_choose_never_loses_a_citation_and_never_invents_a_pointer():
    old = "Il termine è di dieci giorni (art. 9003 c.c.)."
    refs = [("9003", "c.c.")]
    assert gr.choose(old, {"azione": "riscritta", "frase": "Il termine è indicato dall'art. 9003 c.c."}, refs) == \
        ("rewritten", "Il termine è indicato dall'art. 9003 c.c.")
    assert gr.choose(old, {"azione": "riscritta", "frase": "Il termine è breve."}, refs) == \
        ("pointer", "Su questo punto va verificata la disciplina dell'art. 9003 c.c.")
    assert gr.choose(old, {"azione": "riempitivo"}, refs)[0] == "pointer"          # it cites: never dropped
    assert gr.choose("Una regola detta a memoria senza alcuna fonte.", None, []) == ("kept", None)
    assert gr.choose("Una frase di puro riempimento senza alcuna fonte.", {"azione": "riempitivo"}, []) == ("dropped", None)
    assert gr.pointer([("9001", "c.c."), ("9002", "Codice Civile")]) == \
        "Su questo punto va verificata la disciplina degli artt. 9001 c.c. e 9002 c.c."


def test_candidates_skip_short_sentences_and_keep_bullets_and_bold_leads():
    text = "**Profilo penale**: la querela va proposta nel termine di legge davanti al giudice.\n- Breve.\n" \
           "- Le prove da acquisire sono i testimoni e le fotografie del cortile."
    got = [c[2] for c in gr.candidates(text)]
    assert got == ["la querela va proposta nel termine di legge davanti al giudice.",
                   "Le prove da acquisire sono i testimoni e le fotografie del cortile."]
    start, end, sentence = gr.candidates(text)[0]
    assert text[start:end] == sentence


def test_apply_mixes_drops_and_replacements():
    text = "Prima frase da tenere qui. Seconda frase da togliere. Terza frase da cambiare adesso."
    a, b = text.index("Seconda"), text.index("Terza")
    out = gr.apply(text, {(a, b - 1): None, (b, len(text)): "Terza frase nuova e più lunga di prima."})
    assert out == "Prima frase da tenere qui. Terza frase nuova e più lunga di prima."


def test_the_judge_is_fetch_audit_s_own():
    assert gr.JUDGE_SYSTEM == fa.PROVENANCE_SYSTEM


def test_write_sections_runs_the_review_only_with_the_flag(monkeypatch):
    calls = []
    monkeypatch.setattr(dg, "_call_chat", lambda messages, max_tokens=None, **k: "Testo della sezione scritto.")
    monkeypatch.setattr(gr, "review", lambda sections, *a, **k: calls.append(1) or sections)
    plan = {"parte_assistita": "Caio", "posizione": "convenuto", "questioni": [], "pronunce_contrarie": []}
    jobs = [("Conclusioni", "istr", "PIANO", 300)]
    monkeypatch.delenv(gr.FLAG, raising=False)
    ptw.write_sections(plan, jobs, "it")
    assert calls == []
    monkeypatch.setenv(gr.FLAG, "true")
    ptw.write_sections(plan, jobs, "it")
    assert calls == [1]


# --- ROLE_BLOCKS_ENABLED --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("party,side", [
    ("Tizio (imputato di lesioni colpose ai danni della persona offesa)", "accused"),
    ("I genitori di Sempronio (persona offesa, familiari della vittima)", "victim"),
    ("Caio (convenuto in giudizio dai familiari della vittima)", "accused"),
    ("Mevia (attrice, danneggiata)", "victim"),
    ("la parte assistita", ""),
])
def test_client_side_by_the_first_role_word(party, side):
    assert fact_signals.client_side(party) == side


def _signals():
    return {"tools": [{"id": "profilo_penale", "refs": [["590", "c.p."], ["76", "c.p.p."], ["2947", "c.c."]],
                       "text": "Profilo penale: valutare la querela e la costituzione di parte civile.",
                       "why": "lesioni", "matched": ["Tizio ferito"]},
                      {"id": "accertamento_preventivo", "refs": [["696", "c.p.c."]], "text": "ATP.", "matched": []}]}


def test_the_criminal_profile_follows_the_client_s_side(monkeypatch):
    signals = _signals()
    assert fact_signals.fit_to_role(signals, "Tizio (imputato)") == ""          # flag off: unchanged
    assert signals == _signals()
    monkeypatch.setenv(fact_signals.ROLE_BLOCKS_FLAG, "true")
    assert fact_signals.fit_to_role(signals, "Tizio (imputato)") == "accused"
    tool = signals["tools"][0]
    assert ["589", "c.p."] not in tool["refs"] and ["590", "c.p."] in tool["refs"]      # the harm's offence kept
    assert ["2044", "c.c."] in tool["refs"] and ["74", "c.p.p."] in tool["refs"] and ["76", "c.p.p."] not in tool["refs"]
    assert "parte civile contro la parte assistita" in tool["text"] and "2044 c.c." in tool["text"]
    assert signals["tools"][1]["text"] == "ATP." and signals["side"] == "accused"
    assert "contro la parte assistita" in ptw._criminal_what({"fact_signals": signals})
    victim = _signals()
    assert fact_signals.fit_to_role(victim, "Mevia (persona offesa)") == "victim"
    assert ["76", "c.p.p."] in victim["tools"][0]["refs"] and ["336", "c.p.p."] in victim["tools"][0]["refs"]
    assert "querela" in victim["tools"][0]["text"] and "2044" not in victim["tools"][0]["text"]
    unknown = _signals()
    assert fact_signals.fit_to_role(unknown, "la parte assistita") == "" and unknown == _signals()


def test_an_offended_party_gets_its_own_criminal_posture_checks():
    defence = practical_checklist.checks_for(["penale"], "indagini_preliminari")
    victim = practical_checklist.checks_for(["penale"], "indagini_preliminari", "victim")
    assert any("Riti alternativi" in c for c in defence) and not any("Riti alternativi" in c for c in victim)
    assert any("Querela" in c for c in victim) and any("opposizione" in c for c in victim)
    assert practical_checklist.checks_for(["penale"], "indagini_preliminari", "accused") == defence
    assert practical_checklist.checks_for(["contratti"], "", "victim") == practical_checklist.checks_for(["contratti"], "")


# --- fetch_audit ----------------------------------------------------------------------------------------------------

def test_role_flags_on_the_other_side_s_move_or_the_client_s_own_act_are_not_kept():
    accused = {"party": "Tizio", "position": "imputato"}
    victim = {"party": "Mevia", "position": "persona offesa"}
    assert fa.role_flag_filtered(accused, "La persona offesa potrà costituirsi parte civile nel processo.")
    assert fa.role_pattern_hits(accused, "La persona offesa potrà costituirsi parte civile nel processo.") is None
    assert fa.role_flag_filtered(victim, "Mevia può proporre querela entro il termine di legge.")
    # a real contradiction stays: the accused advised the victim's act
    assert fa.role_flag_filtered(accused, "Tizio deve costituirsi parte civile per il risarcimento.") is None
    assert fa.role_pattern_hits(accused, "Tizio deve costituirsi parte civile per il risarcimento.")


def test_the_funnel_locates_where_a_captured_article_was_lost():
    resolved_elsewhere = {"stage": "LOST_AT_UNDERSTANDING", "understanding": {}, "in_pool": False, "rank": None}
    f = fa.funnel(resolved_elsewhere)
    assert f["lost_at"] == "resolved" and not any(f["steps"].values())
    not_planned = {"stage": "NOT_IN_PLAN", "in_pool": True, "rank": 4, "check": "applicabile", "in_prompt": True,
                   "in_plan": False, "channels": ["vector"]}
    f = fa.funnel(not_planned)
    assert f["lost_at"] == "in_plan" and f["steps"]["graded"] and f["channels"] == ["vector"]
    rejected = {"stage": "REJECTED_BY_CHECK", "in_pool": True, "check": "rejected", "in_plan": None}
    assert fa.funnel(rejected)["lost_at"] == "graded"
    assert fa.capture_kind(["istituto: Custodia -> art. 2043 c.c."]) == "institution_elsewhere"
    assert fa.capture_kind(["istituto: Custodia -> non risolto"]) == "institution_unresolved"
    assert fa.capture_kind(["questione: Responsabilità del custode — istituti []"]) == "issue"
    assert fa.capture_kind(["fatto: caduta nel cortile"]) == "fact"


def test_the_audit_reads_the_grounded_review_s_rewrites_and_reads():
    log = {"entries": [{"section": "Norme e giurisprudenza", "how": "kept_fetched", "fetched": ["art. 9001 c.c."]}],
           "fetched": {"art. 9001 c.c.": FETCHED[("9001", "c.c.")]}, "totals": {"rule": 3, "assente": 1},
           "counts": {"pointer": 1}}
    stage = fa.section_stage("Norme e giurisprudenza")
    case = {"proxy": False, "record": {"events": [["grounded_review", log, None]]}, "calls": [
        {"stage": stage, "system": "S", "user": MATERIAL, "reply": SECTION},
        {"stage": "grounded_rewrite", "system": gr.REWRITE_SYSTEM, "user": f"MATERIALE:\n{MATERIAL}\n\n"
         f"{gr.REWRITE_MARK}\n1. {INVENTED}", "reply": json.dumps([{"i": 1, "frase": "Nuova frase dal materiale."}])}]}
    calls = fa.writer_calls(case)
    assert [c["stage"] for c in calls] == [stage, "grounded_rewrite"]
    assert calls[0]["material"].startswith(FETCHED[("9001", "c.c.")])        # read for the section: given to it
    assert INVENTED not in calls[1]["material"] and MATERIAL in calls[1]["material"]
    assert fa.grounded_case_totals(case) == {"rule": 3, "assente": 1, "pointer": 1, "fetched": 1, "not_found": 0}


# --- GROUNDED_REVIEW_STRICT_ENABLED ---------------------------------------------------------------------------------

def strict_chat_json(messages, max_tokens, **kwargs):
    system, human = messages[0].content, messages[1].content
    if system == gr.JUDGE_SYSTEM:
        return fake_chat_json(messages, max_tokens)
    assert system == gr.STRICT_REWRITE_SYSTEM
    lines = [ln for ln in human.split(gr.REWRITE_MARK + "\n", 1)[1].splitlines() if ln.strip()]
    out = []
    for ln in lines:
        n, sentence = ln.split(". ", 1)
        if sentence.startswith(INVENTED):
            out.append({"i": int(n), "azione": "riscritta", "frase": "Il custode risponde del danno salvo che provi "
                        "il caso fortuito (art. 9002 c.c.)"})
        elif sentence.startswith(FROM_FETCH):
            assert "[da togliere: art. 9001 c.c.]" in sentence
            out.append({"i": int(n), "azione": "rinvio", "frase": "Va verificata la disciplina dell'art. 9001 c.c."})
        else:
            out.append({"i": int(n), "azione": "elimina", "frase": ""})
    return out


def test_strict_never_fetches_nor_points_outside_the_checked_material(monkeypatch):
    events = []
    monkeypatch.setattr(gr, "vlog", lambda event, data=None, **_: events.append((event, data)))
    monkeypatch.setattr(dg, "_chat_json", strict_chat_json)
    reads = []
    jobs = [("Norme e giurisprudenza", "istr", MATERIAL, 900)]
    out = gr.review([("Norme e giurisprudenza", SECTION)], jobs, [], [],
                    fetch=lambda refs: reads.append(refs) or {}, checked=[("9002", "c.c.")])
    text = out[0][1]
    assert reads == []                                                   # nothing read
    assert SUPPORTED in text                                             # supported: untouched
    assert "salvo che provi il caso fortuito (art. 9002 c.c.)." in text  # rewritten, its in-material citation kept
    assert "9001" not in text and "9005" not in text                     # outside: never pointed to
    assert MEMORY not in text and FILLER not in text and FROM_FETCH not in text
    log = [d for e, d in events if e == "grounded_review"][-1]
    assert log["strict"] and log["counts"]["kept"] == 0 and log["counts"]["kept_fetched"] == 0
    assert log["counts"]["citations_removed"] == 2 and log["counts"]["rewritten"] == 1
    removed = {e["sentence"][:20]: e.get("removed") for e in log["entries"]}
    assert removed[FROM_FETCH[:20]] == ["art. 9001 c.c."]


def test_choose_strict_keeps_inside_citations_and_points_only_inside():
    good, bad = [("9002", "c.c.")], [("9001", "c.c.")]
    old = "Il custode risponde sempre (artt. 9001 e 9002 c.c.)."
    assert gr.choose_strict(old, {"azione": "riscritta", "frase": "Il custode risponde (art. 9002 c.c.)."}, good, bad) \
        == ("rewritten", "Il custode risponde (art. 9002 c.c.).")
    # the rewriting keeps the outside citation, or adds a new one: the pointer to the inside article instead
    assert gr.choose_strict(old, {"azione": "riscritta", "frase": "Risponde (artt. 9001 e 9002 c.c.)."}, good, bad) \
        == ("pointer", "Su questo punto va verificata la disciplina dell'art. 9002 c.c.")
    assert gr.choose_strict(old, {"azione": "riscritta", "frase": "Risponde (artt. 9002 e 9009 c.c.)."}, good, bad)[0] \
        == "pointer"
    assert gr.choose_strict("Il termine è breve (art. 9001 c.c.).", {"azione": "rinvio", "frase": "Va verificato "
                            "l'art. 9001 c.c."}, [], bad) == ("dropped", None)
    assert gr.choose_strict("Una regola detta a memoria senza fonte.", None, [], []) == ("dropped", None)


def test_strict_refs_graded_named_tools_and_terms():
    rows = [{"_article": ("9002", "c.c."), "_grade": 3}, {"_article": ("9003", "c.c."), "_grade": 1},
            {"_article": ("9004", "c.c."), "_grade": 2}]
    analysis = {"fact_signals": {"tools": [{"id": "x", "refs": [["696", "c.p.c."]]}],
                                 "terms": [{"id": "t", "ref": "1495 c.c."}]}}
    got = gr.strict_refs(rows, [("9010", "c.c.")], analysis)
    assert ("9002", "c.c.") in got and ("9004", "c.c.") in got and ("9003", "c.c.") not in got
    assert ("9010", "c.c.") in got and ("696", "c.p.c.") in got and ("1495", "c.c.") in got
    ungraded = [{"_article": ("9003", "c.c."), "_applicability": "applicabile"}, {"_article": ("9005", "c.c.")}]
    assert gr.strict_refs(ungraded) == [("9003", "c.c.")]


# --- FACT_ISSUES_ENABLED --------------------------------------------------------------------------------------------

from src.rag import fact_issues as fi  # noqa: E402


def test_fact_issues_clean_caps_dedupes_and_refuses_numbers():
    raw = {"istituti": [{"fatto": "infezione dopo l'intervento", "nome": "responsabilità della struttura sanitaria",
                         "codice": "c.c.", "area": "responsabilita_civile", "ruolo_cliente": "danneggiato"},
                        {"fatto": "x", "nome": "art. 2236 c.c.", "codice": "c.c."},
                        {"fatto": "x", "nome": "Garanzia per i vizi", "codice": "c.c.", "area": "boh"},
                        {"fatto": "x", "nome": "garanzia per i vizi", "codice": "c.c."}]
           + [{"fatto": "f", "nome": f"istituto numero {w}", "codice": "c.c."} for w in "abcdefghij"]}
    got = fi.clean(raw, ["Garanzia per i vizi"])
    assert got[0] == {"nome": "responsabilità della struttura sanitaria", "codice": "c.c.", "fonte": "fatto",
                      "fatto": "infezione dopo l'intervento", "area": "responsabilita_civile", "ruolo": "danneggiato"}
    assert len(got) == fi.MAX_INSTITUTIONS and not any("2236" in g["nome"] or "vizi" in g["nome"] for g in got)


def test_fact_issues_join_the_central_issues_and_reach_the_resolver_with_their_fact(monkeypatch):
    events = []
    monkeypatch.setattr(fi, "vlog", lambda event, data=None, **_: events.append((event, data)))
    monkeypatch.setattr(dg, "_chat_json", lambda messages, max_tokens, **k: {"istituti": [
        {"fatto": "la lavatrice si guasta dopo tre mesi", "nome": "difetto di conformità del bene",
         "codice": "Codice del Consumo", "area": "consumo", "ruolo_cliente": "acquirente consumatore"}]})
    frame = {"facts": ["Acquisto di una lavatrice", "Guasto dopo tre mesi"], "tasks": ["che fare"],
             "central_issue_candidates": [{"nome": "compravendita", "codice": "c.c."}]}
    signals = {"facts": {"events": [{"what": "guasto", "date_iso": "2026-01-01"}], "parties": []}}
    assert fi.add_to_frame(frame, "caso", signals) is frame                  # flag off: unchanged, no call
    monkeypatch.setenv(fi.FLAG, "true")
    got = fi.add_to_frame(frame, "caso", signals)
    assert [c["nome"] for c in got["central_issue_candidates"]] == ["compravendita", "difetto di conformità del bene"]
    assert events[-1][0] == "case_analysis_fact_issues" and events[-1][1]["facts_read"] == 3
    assert "- events: {\"what\": \"guasto\"}" in fi.fact_lines(frame, signals)
    analysis = dg._with_central_questions({"questioni": []}, got)
    q = next(q for q in analysis["questioni"] if q["istituti"][0]["nome"] == "difetto di conformità del bene")
    assert q["istituti"][0]["fonte"] == "fatto" and q["fatto"] == "la lavatrice si guasta dopo tre mesi"
    block = dg._frame_block(got)
    assert "dal fatto: la lavatrice si guasta dopo tre mesi; parte assistita: acquirente consumatore" in block
    # the resolver's rows kept only where the rubrica shares a word with the institution
    rows = [{"_article": ("129", "Codice del Consumo"), "s": {"titolo_articolo": "Conformità al contratto"}},
            {"_article": ("33", "Codice del Consumo"), "s": {"titolo_articolo": "Clausole vessatorie"}}]
    assert fact_signals.keep_resolved(q["istituti"][0], rows) == rows[:1]


# --- fetch_audit: ROLE_BLOCKS alone -------------------------------------------------------------------------------

def test_role_counts_per_run_only_over_parsed_replies():
    replies = iter(['{"parte": "Tizio", "ruolo": "imputato", "contrasti": [{"i": 1, "perche": "x"}]}', "non JSON"])
    judge = fa.Judge(None, call=lambda system, human, mt: next(replies))
    sentences = [f"Tizio deve costituirsi parte civile per il risarcimento numero {i}." for i in range(3)]
    case = {"case_text": "caso", "role": {"party": "Tizio", "position": "imputato"}}
    r = fa.audit_role(case, [{"sentence": s} for s in sentences], judge, 2)
    assert r["batches"] == 2 and r["failed_batches"] == 1 and r["judged_sentences"] == 2
    by = fa.role_by_run([{"run": "rb-on", "role": r}, {"run": "rb-off", "role": {**r, "skipped": "not run"}}])
    assert list(by) == ["rb-on"] and by["rb-on"]["judge"] == 1 and by["rb-on"]["rate"] == 50.0
    assert by["rb-on"]["pattern"] == 3


def test_mechanisms_role_alone_makes_only_role_calls(tmp_path):
    asked = []

    def judge_call(system, human, mt):
        asked.append(system)
        return '{"parte": "Tizio", "ruolo": "imputato", "contrasti": []}'
    case = {"key": "r/x", "run": "r", "id": "x", "kind": "eval", "proxy": True, "answer": "Tizio può chiedere il rito "
            "abbreviato davanti al giudice competente.", "record": {}, "calls": [], "case_text": "caso", "role": {},
            "citations": []}

    class A:
        batch, material_chars, mechanisms = 12, 14000, ["role"]
    out = fa.audit_case(case, fa.Judge(None, call=judge_call), None, None, A)
    assert set(asked) == {fa.ROLE_SYSTEM} and out["provenance"]["skipped"] == "not run"
    assert "per 100 judged" in fa.render([out], {"runs": ["r"], "judge": "x", "commi": "x", "when": "now"})


# --- GROUNDED_REVIEW_POLICY=abstain ---------------------------------------------------------------------------------

def test_abstain_policy_replaces_unsupported_rules_with_one_abstention_per_topic(monkeypatch):
    events = []
    monkeypatch.setattr(gr, "vlog", lambda event, data=None, **_: events.append((event, data)))
    monkeypatch.setenv(gr.FLAG, "true")
    monkeypatch.setenv(gr.POLICY_VAR, "abstain")

    def judge_only(messages, max_tokens, **k):
        assert messages[0].content == gr.JUDGE_SYSTEM, "abstain: no rewrite call"
        return fake_chat_json(messages, max_tokens)
    monkeypatch.setattr(dg, "_chat_json", judge_only)
    rows = [{"_article": ("9001", "c.c."), "s": {"titolo_articolo": "Prescrizione immaginaria"}},
            {"_article": ("9002", "c.c."), "s": {"titolo_articolo": "Danno cagionato da cose in custodia"}}]
    application = "Caio non ha recintato il cortile e Tizio vi è caduto scendendo dalle scale di sera."
    text = SECTION + "\n\n" + application
    out = gr.review([("Norme e giurisprudenza", text)], [("Norme e giurisprudenza", "istr", MATERIAL, 400)], [],
                    rows, fetch=lambda refs: pytest.fail("abstain: nothing fetched"), checked=[("9002", "c.c.")],
                    topics=["Caduta nel cortile"])
    start = gr.ABSTENTION_START
    assert out[0][1] == (
        f"{SUPPORTED}\n\n"
        f"{start} prescrizione immaginaria: da verificare.\n"                         # 9001: outside the checked
        f"{start} danno cagionato da cose in custodia (art. 9002 c.c.): da verificare.\n"   # 9002: checked, kept
        f"{start} caduta nel cortile: da verificare.\n"                                # no citation: the plan issue
        f"{start} questo profilo: da verificare.\n\n"                                  # filler + 9005, merged
        f"{application}")
    log = [d for e, d in events if e == "grounded_review"][-1]
    assert log["policy"] == "abstain" and log["counts"]["abstained"] == 5 and log["counts"]["abstentions"] == 4
    assert gr.abstentions(out[0][1]) == 4


def test_abstain_policy_is_off_by_default_and_reads_only_its_values(monkeypatch):
    monkeypatch.setenv(gr.FLAG, "true")
    monkeypatch.delenv(gr.POLICY_VAR, raising=False)
    assert gr.policy() == "rewrite" and not gr.abstain()
    monkeypatch.setenv(gr.POLICY_VAR, "whatever")
    assert gr.policy() == "rewrite"
    monkeypatch.setenv(gr.POLICY_VAR, "Abstain")
    assert gr.abstain()
    assert gr.abstention("termine per ricorrere", [("9029", "c.p.a."), ("9029", "c.p.a.")]) == \
        f"{gr.ABSTENTION_START} termine per ricorrere (art. 9029 c.p.a.): da verificare."
