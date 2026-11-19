"""v75b -- synthetic records and texts.

1. an attore with elements still to prove gets «fondare la domanda sull'art. X, provando …» (realismo basso when
   nothing is proven), never the standard body
2. «Procedibilità e competenza», «Profilo penale» and the querela's term from the intake rito only
3. the store's regulations each gated by its own signal
4. the penal-defence points: candidates from the records (never the read's sì / no); the charged article's paragraph
   from its sussunzione rows only; the negation review on the objections' replies; the containment line's next steps
   with no element; §10 only the open questions
"""

import os
import sys
from pathlib import Path

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from src.rag import fact_signals, lawyer_method as M, posture as posture_mod, render_computed, special_laws  # noqa: E402

ATTORE = {"rito": "civile", "family": "civile", "role": "attore", "stage": "pre_filing", "intake_rito": "civile"}
IMPUTATO = {"rito": "penale", "family": "penale", "role": "imputato", "stage": "primo_grado", "intake_rito": "penale",
            "entry": posture_mod.entry_for("penale", "imputato", "primo_grado")}
SENTS = [{"id": "S1", "text": "Sempronio colpisce Caio, che cade e muore.", "source": "caso", "certainty": "accertato"}]


def _req(i, text, status, **extra):
    return {"id": f"R{i}", "comma": 1, "kind": "requisito", "text": text, "status": status, **extra}


def _ctx(recs, posture, questions=()):
    return M.prepare({"records": recs, "sentences": SENTS}, posture, [], "Nel 2025 Sempronio colpisce Caio.", "Caio",
                     "", list(recs), [], [], [], [], link_call=lambda *a: {}, questions=list(questions),
                     case_id_="t75b")


# --- 1. the attore's claim to prove -----------------------------------------------------------------------------------------

def test_the_attore_with_elements_to_prove_gets_the_claim_never_the_standard_body():
    recs = {"2043 c.c.": {"rubrica": "Risarcimento per fatto illecito", "side": "weapon",
                          "requisites": [_req(1, "fatto doloso o colposo", "da provare"),
                                         _req(2, "cagiona ad altri un danno ingiusto", "da provare")]}}
    ctx = _ctx(recs, ATTORE)
    cands = M.thesis_candidates(ctx)
    assert cands[0]["tesi"] == ("Fondare la domanda sull'art. 2043 c.c. (Risarcimento per fatto illecito), provando "
                                "«fatto doloso o colposo» (R1), «cagiona ad altri un danno ingiusto» (R2)")
    assert cands[0]["code_argument"]["realismo"] == "basso"
    assert cands[0]["code_argument"]["argomento"].startswith("La domanda sull'art. 2043 c.c. richiede la prova di:")
    assert all(c["code_argument"]["realismo"] == "basso" for c in cands if c["kind"] == "remedy")
    assert M.placeholder_main(ctx, M.clean_method({}, cands)) == ""
    recs["2043 c.c."]["requisites"][0]["status"] = "provato"               # one proven: the claim, realismo medio
    cand = M.thesis_candidates(_ctx(recs, ATTORE))[0]
    assert cand["tesi"] == "Fondare la domanda sull'art. 2043 c.c. (Risarcimento per fatto illecito)"
    assert cand["code_argument"]["realismo"] == "medio"


# --- 2. the penal lines of the computed section by the intake rito ----------------------------------------------------------

def test_the_penal_lines_of_the_computed_section_follow_the_intake_rito(monkeypatch):
    monkeypatch.setattr(render_computed, "criminal_procedure", lambda analysis: "Procedibilità d'ufficio.")
    signals = {"tools": [{"id": fact_signals.CRIMINAL_TOOL, "text": "Profilo penale del fatto.", "applies": True}],
               "terms": [{"id": "querela", "text": "Querela: tre mesi dalla notizia del fatto (art. 124 c.p.)."}]}
    civil = {"rito": "civile", "family": "civile", "role": "attore", "stage": "pre_filing", "intake_rito": "civile"}
    penal = {**civil, "rito": "penale", "family": "penale", "role": "persona_offesa", "intake_rito": "penale"}
    titles = lambda posture: [t for t, _ in render_computed.blocks({"fact_signals": signals, "posture": posture})]
    assert render_computed.PROCEDURE_TITLE not in titles(civil) and "Profilo penale" not in titles(civil)
    assert render_computed.PROCEDURE_TITLE in titles(penal) and "Profilo penale" in titles(penal)
    assert render_computed.rito_terms(signals["terms"], civil) == []
    assert render_computed.rito_terms(signals["terms"], penal) == signals["terms"]
    assert render_computed.penal_rito({}) and render_computed.penal_rito({"rito": "civile"})     # as before
    assert not render_computed.penal_rito({**civil, "intake_rito": ""})


# --- 3. the store's regulations ---------------------------------------------------------------------------------------------

def test_each_regulation_of_the_store_reaches_grading_only_with_its_own_signal():
    cases = [("Reg. 692/2014", "La società esportava verso la Crimea beni soggetti a embargo."),
             ("Direttiva 94/25/CE", "Caio ha acquistato un'imbarcazione da diporto difettosa."),
             ("Regolamento generale Banca d'Italia", "La banca ha omesso le segnalazioni di vigilanza."),
             ("Provvedimento Banca d'Italia 2022/1917", "Le segnalazioni statistiche della banca erano errate.")]
    bank = "Caio ha ricevuto una mail che sembrava della sua banca e ha disposto un bonifico."
    for code, signal in cases:
        row = {"_article": ("1", code)}
        assert special_laws.law_of(code), code
        assert special_laws.gate([row], bank, set())[0] == [], code
        assert special_laws.gate([row], signal, set())[0] == [row], code


# --- 4. the penal-defence points --------------------------------------------------------------------------------------------

def _charge(*statuses):
    texts = ["atti diretti a percuotere o ledere", "la morte come conseguenza degli atti"]
    return {"584 c.p.": {"rubrica": "Omicidio preterintenzionale", "side": "threat", "charged": 1,
                         "requisites": [_req(i, texts[i - 1], st, distinctive=(i == 1)) for i, st in
                                        enumerate(statuses, 1)]}}


def test_the_riqualificazione_comes_from_the_record_never_from_the_reads_no():
    recs = _charge("provato", "provato")
    recs["589 c.p."] = {"rubrica": "Omicidio colposo", "side": "weapon", "alternative": True, "alternative_facts": "no",
                        "requisites": [_req(9, "cagiona la morte di una persona", "provato"),
                                       _req(10, "per colpa", "provato")]}
    ctx = _ctx(recs, IMPUTATO)
    # v79.12.7 (item 4): 584 -> 589 is offered only with a fact on the differing element (a spinta, an impulse)
    ctx["sentences"] = list(ctx["sentences"]) + [{"id": "S9", "text": "Sempronio dice di avere reagito d'impulso, con una spinta.", "source": "caso", "certainty": "accertato"}]
    assert any(c["kind"] == "requalify" for c in M.thesis_candidates(ctx))


def test_the_charged_articles_paragraph_is_its_sussunzione_rows():
    ctx = _ctx(_charge("provato", "da provare"), IMPUTATO)
    assert ctx["charged_prose"]["584 c.p."] == (
        "Sui fatti narrati, degli elementi dell'art. 584 c.p. risultano provati «atti diretti a percuotere o ledere» "
        "(R1); restano da provare «la morte come conseguenza degli atti» (R2) (v. l'analisi per elementi).")
    norms = ("**Art. 584 c.p. — Omicidio preterintenzionale**\nLa morte non è conseguenza dello schiaffo: manca il "
             "nesso.\nIl dolo c'era.\n\n**Art. 41 c.p.**\nLe concause non escludono il nesso.")
    log = []
    out = M.charged_blocks(norms, ctx["charged_prose"], log)
    assert "manca il nesso" not in out and "Il dolo c'era" not in out and ctx["charged_prose"]["584 c.p."] in out
    assert "Le concause non escludono il nesso." in out and log == ["584 c.p."]
    plan = M.clean_method({}, M.thesis_candidates(ctx))
    answer = M.compose([("Norme applicate ai fatti", norms)], ctx, plan)
    assert "manca il nesso" not in answer


def _down(messages, **kw):
    raise RuntimeError("down")


def test_the_objections_replies_never_negate_past_the_main_lines_contesta():
    known = {"R1": {"article": "584 c.p.", "text": "atti diretti a percuotere o ledere", "status": "provato"},
             "R2": {"article": "584 c.p.", "text": "la morte come conseguenza degli atti", "status": "da provare"}}
    objections = [{"obiezione": "L'accusa: la morte deriva dallo schiaffo",
                   "risposta": "La morte non è provata come conseguenza degli atti. Gli atti diretti a percuotere o "
                               "ledere non ci sono."}]
    log = M.bind_replies(objections, {"contesta": ["R2"]}, known, lambda messages, **kw: [
        {"i": 2, "elemento": "R1", "perche": "nega R1"}])
    assert objections[0]["risposta"] == "La morte non è provata come conseguenza degli atti."
    assert log["how"] == "review" and log["dropped"][0]["elemento"] == "R1"
    ctx = _ctx(_charge("provato", "da provare"), IMPUTATO)
    plan = M.clean_method({}, M.thesis_candidates(ctx))
    out = M.objections_plan(ctx, plan, lambda messages: {"obiezioni": [
        {"obiezione": "Lo schiaffo era lesivo", "risposta": "Gli atti diretti a percuotere o ledere non ci sono."}]},
        review_call=_down)                                         # the review fails: the word test decides
    assert out["obiezioni"][0]["risposta"] == ""


def test_the_containment_lines_next_steps_carry_no_element():
    ctx = _ctx(_charge("provato", "provato"), IMPUTATO)
    cands = M.thesis_candidates(ctx)
    sols, main, _ = M.selection(ctx, M.clean_method({}, cands))
    assert sols[main]["kind"] == "contenimento"
    steps = M.steps_from_selection(ctx, sols, main)
    assert steps and not any("R1" in s["passo"] + s["scopo"] or "R2" in s["passo"] + s["scopo"] for s in steps)


def test_section_ten_holds_only_the_open_questions():
    questions = [{"id": "q1", "text": "In che data Caio ha avuto notizia del fatto?"}]
    ctx = _ctx(_charge("provato", "da provare"), IMPUTATO, questions)
    plan = M.clean_method({"orientamenti": [{"punto": "causa sopravvenuta", "pronuncia": ""}]},
                          M.thesis_candidates(ctx))
    answer = M.compose([("Norme applicate ai fatti", "La regola vale (da verificare).\n\n**Profili**\n(da verificare)")],
                       ctx, plan)
    points = dict(M.answer_sections(answer))["Punti da verificare"]
    assert "In che data Caio ha avuto notizia del fatto?" in points and "La regola vale." in points
    assert "causa sopravvenuta" not in points and "Norme e analisi" not in points and "Profili" not in points
