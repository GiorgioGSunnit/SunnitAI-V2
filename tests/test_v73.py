"""v73: the sim is the live path -- synthetic texts, the model stubbed.

1. the sims (diagnose_long --interactive-sim, nightly_sim) read the intake as api.py does -- intake.request_intake, the
   same inputs -- before the flow, and log it; a sim run without an intake line fails loudly
2. the in-flow charge fallback reads the case text with the shared tolerant parser: the charge is marked with or
   without intake
3. the next steps only from the selected line's elements; an article without decomposed elements yields none
4. the objections written after the selection, given the selected line and the barred list
"""

import asyncio
import copy
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("OPENAI_API_KEY", "k"), ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts"))

from src.rag import intake, lawyer_method as M, posture as posture_mod  # noqa: E402
from src.rag import interactive_flow as flow  # noqa: E402

CASE = ("Sempronio colpisce Caio con un pugno al volto durante una lite. Caio cade, batte la testa sul marciapiede e "
        "muore due giorni dopo. Il PM contesta a Sempronio l’art 584 cp. Assisto Sempronio: come impostare la difesa?")
ENTRY = posture_mod.entry_for("penale", "imputato", "primo_grado")
IMPUTATO = {"rito": "penale", "family": "penale", "role": "imputato", "stage": "primo_grado", "entry": ENTRY}
SENTS = [{"id": "S1", "text": "Sempronio colpisce Caio, che cade e muore.", "source": "caso", "certainty": "accertato"}]


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Stop(Exception):
    """Raised right after the intake is in the state: what follows is the same flow for both paths."""


@pytest.fixture
def after_intake(monkeypatch):
    """The flow stopped once the intake is in the state; each run's state kept (a copy). The analysis stubbed (the same
    for every path), the intake's model call stubbed (deterministic): what differs between the paths is only how they
    read the intake and start the flow."""
    monkeypatch.setenv("CASE_FIRST_SINGLE_SHOT", "true")
    monkeypatch.setattr(intake, "ask_model", lambda *a, **k: {})
    monkeypatch.setattr(flow, "analyse_case", lambda text, request="", lang="it": {
        "text": text, "request": request, "lang": lang, "analysis": {}})
    states = []
    real = flow.take_intake

    def take(state, read, message, request=""):
        real(state, read, message, request)
        states.append(copy.deepcopy(state))
    monkeypatch.setattr(flow, "take_intake", take)

    def stop(*a, **k):
        raise _Stop()
    monkeypatch.setattr(flow, "covered_topics", stop)
    lines = []
    monkeypatch.setattr(intake, "vlog", lambda event, data=None, ms=None, **k: lines.append((event, data)))
    return states, lines


def _without_case_id(state):
    return {k: v for k, v in state.items() if k != "case_id"}


# --- 1. the sims read the intake as the API does ---------------------------------------------------------------------------

def test_the_sims_state_after_intake_equals_the_apis_for_one_fixture(after_intake, monkeypatch, tmp_path):
    states, lines = after_intake
    import src.chatbot.api as api
    monkeypatch.setattr(api, "vlog", lambda *a, **k: None)
    session = SimpleNamespace(messages=[], session_language="it")
    read = asyncio.run(api._request_intake({}, CASE, session))
    assert asyncio.run(api._interactive_start(session, "sid", CASE, session.session_language, intake=read)) is None
    api_state = states[-1]

    with pytest.raises(_Stop):                              # the shared entry both sims call
        flow.request_case_first(CASE, "it", case_id="cell")
    entry_state = states[-1]

    N = _script("nightly_sim")                              # the nightly's cell
    N.run_cell({"id": "penale_imputato", "cell": IMPUTATO, "text": CASE})
    nightly_state = states[-1]

    dl = _script("diagnose_long")                           # diagnose_long --interactive-sim --single-shot
    monkeypatch.setattr(dl, "_eval_case", lambda cid: None)
    dl.run_interactive_sim({"id": "c1", "out": str(tmp_path / "c1.json"), "single_shot": True}, CASE, "", None,
                           {"calls": 0, "seconds": 0, "errors": 0}, lambda message: None)
    sim_state = states[-1]

    assert len(states) == 4 and api_state["intake_source"] == "request"
    for other in (entry_state, nightly_state, sim_state):
        assert _without_case_id(other) == _without_case_id(api_state)
    assert api_state["intake"]["prior_decisions"][0]["articles"] == ["584 c.p."]
    assert api_state["intake"]["role"] == "imputato" and api_state["text"] == CASE and api_state["request"] == ""
    assert [e for e, _ in lines] == ["intake"] * 4          # one [VERBOSE:intake] line per run, the shared function's


def test_a_file_case_reaches_the_flow_as_the_apis_upload_does(after_intake, monkeypatch):
    states, _ = after_intake
    seen = []
    real = intake.request_intake
    monkeypatch.setattr(intake, "request_intake", lambda *a, **k: (seen.append((a, k)), real(*a, **k))[1])
    dl = _script("diagnose_long")
    message, upload = dl.sim_request(CASE, "Analizza il caso allegato")
    assert (message, upload) == ("Analizza il caso allegato", CASE) and dl.sim_request(CASE, "") == (CASE, "")
    with pytest.raises(_Stop):
        flow.request_case_first(message, "it", upload)
    assert seen[0][0][:3] == (message, CASE, ()) and seen[0][1]["refers_to_upload"] is True
    if states[-1]["intake"].get("case_in") == "upload":     # the upload the case, the message its request (api's own)
        assert (states[-1]["text"], states[-1]["request"]) == (CASE, message)
    assert flow.entry_args("m") == ("m", "") and flow.entry_args("m", "file") == ("file", "m")


def test_a_sim_run_without_an_intake_line_fails_loudly(after_intake, monkeypatch, tmp_path, capsys):
    states, _ = after_intake
    dl = _script("diagnose_long")
    assert dl.intake_missing({"intake": {"route": "case_first"}}, {"intake_source": "request"}, None) == ""
    assert "no [VERBOSE:intake]" in dl.intake_missing({"intake": {"x": 1}}, {"intake_source": "request"}, {"events": []})
    assert dl.intake_missing({"intake": {"x": 1}}, {"intake_source": "request"},
                             {"events": [["intake", {}, 1.0]]}) == ""
    assert "holds no intake" in dl.intake_missing({}, None, None)
    assert "read the intake itself" in dl.intake_missing({"intake": {"x": 1}}, {"intake_source": "flow"}, None)

    # a sim that skips the intake (calls the flow bare): the record's status, stderr
    monkeypatch.setattr(dl, "_eval_case", lambda cid: None)
    monkeypatch.setattr(flow, "request_case_first", lambda message, lang="it", upload="", case_id="": flow.single_shot(
        message, lang, "", case_id=case_id))
    out = tmp_path / "c2.json"
    dl.run_interactive_sim({"id": "c2", "out": str(out), "single_shot": True}, CASE, "", None,
                           {"calls": 0, "seconds": 0, "errors": 0}, lambda message: None)
    import json
    record = json.loads(out.read_text(encoding="utf-8"))
    assert record["status"].startswith("ERROR no intake line")
    assert "FAIL c2: ERROR no intake line" in capsys.readouterr().err

    N = _script("nightly_sim")
    assert N.intake_line_check([], {"intake_source": "request"}, "case_first")["ok"] is False
    assert N.intake_line_check([{}], {"intake_source": "flow"}, "case_first")["ok"] is False
    assert "not case-first" in N.intake_line_check([{}], {"intake_source": "request"}, "rag")["why"]
    assert N.intake_line_check([{}], {"intake_source": "request"}, intake.CASE_FIRST)["ok"] is True


# --- 2. the charge marked with or without intake ---------------------------------------------------------------------------

def _charge_state(text):
    return {"text": text, "records": {
        "584 c.p.": {"rubrica": "Omicidio preterintenzionale", "requisites": [{"id": "R1", "status": "provato"}]},
        "582 c.p.": {"rubrica": "Lesione personale", "requisites": [{"id": "R2", "status": "provato"}]}}}


@pytest.mark.parametrize("sentence", [
    "Il PM contesta a X l’art 584 cp.", "il PM contesta a Sempronio l'art 584 cp.", "Sempronio è imputato ex art. 584 c.p.",
    "Il pubblico ministero gli contesta l'omicidio preterintenzionale (art 584 cp).", "il PM gli contesta il 584 cp."])
def test_the_charge_is_marked_from_the_case_text_without_intake(sentence):
    state = _charge_state(f"Sempronio colpisce Caio, che muore. {sentence} Assisto Sempronio.")
    log = posture_mod.mark_sides(state, {"role": "imputato", "prior_decisions": []})       # no intake
    assert state["records"]["584 c.p."]["charged"] == 1 and "charged" not in state["records"]["582 c.p."]
    assert state["records"]["584 c.p."]["side"] == "threat"
    assert log["charge_from"] == {"intake": [], "case": ["584 c.p."], "first_read": []} and \
        log["charge_unmarked"] == []


def test_the_charge_from_intake_and_from_the_text_agree_and_an_unmarked_one_is_logged():
    text = "Sempronio colpisce Caio, che muore. Il PM contesta a Sempronio l’art 584 cp."
    read = {"role": "imputato", "prior_decisions": intake.prior_decisions(text)}
    state = _charge_state(text)
    log = posture_mod.mark_sides(state, read)
    assert log["charge_from"]["intake"] == log["charge_from"]["case"] == ["584 c.p."]
    state = {"text": text, "records": {"582 c.p.": {"rubrica": "Lesione personale", "requisites": []}}}
    assert posture_mod.mark_sides(state, {"role": "imputato"})["charge_unmarked"] == ["584 c.p."]


# --- 3. the next steps from the selected line's elements -------------------------------------------------------------------

def _req(i, text, status, **extra):
    return {"id": f"R{i}", "comma": 1, "kind": "requisito", "text": text, "status": status, **extra}


def _records(*statuses, alt=None, alt_requisites=True):
    texts = ["atti diretti a percuotere o ledere", "la morte come conseguenza degli atti", "il nesso causale"]
    recs = {"584 c.p.": {"rubrica": "Omicidio preterintenzionale", "side": "threat", "charged": 1,
                         "requisites": [_req(i, texts[i - 1], st, distinctive=(i == 1))
                                        for i, st in enumerate(statuses, 1)]}}
    if alt:
        recs["589 c.p."] = {"rubrica": "Omicidio colposo", "side": "weapon", "alternative": True,
                            "requisites": [_req(9, "condotta colposa", alt)] if alt_requisites else []}
    return recs


def _ctx(recs, posture=IMPUTATO):
    return M.prepare({"records": recs, "sentences": SENTS}, posture, [], "Nel 2025 Sempronio colpisce Caio.",
                     "Sempronio", "", list(recs), [], [], [], [], link_call=lambda *a: {}, case_id_="t73")


def test_the_steps_come_only_from_the_selected_lines_elements():
    recs = _records("provato", "da provare", "da provare")
    recs["584 c.p."]["requisites"][1]["missing"] = "la causa della morte"
    recs["584 c.p."]["requisites"][2]["missing"] = "la perizia sul nesso"
    ctx = _ctx(recs)
    main = {"tesi": "R2 non è provato", "contesta": ["R2"], "poggia_su": []}
    steps = M.steps_from_selection(ctx, [main], 0)
    assert len(steps) == 1 and "R2" in steps[0]["scopo"] and "la causa della morte" in steps[0]["passo"]
    assert not any("R3" in s["passo"] + s["scopo"] or "perizia" in s["passo"] for s in steps)   # not the line's
    rests = {"tesi": "Si regge su R1 e R3", "contesta": [], "poggia_su": ["R1", "R3"]}
    steps = M.steps_from_selection(ctx, [rests], 0)
    assert steps[0]["passo"].startswith("Documentare l'elemento «atti diretti") and "(R1)" in steps[0]["passo"]
    assert steps[1]["passo"] == "Acquisire la prova: la perizia sul nesso" and "R3" in steps[1]["scopo"]
    integrated = {"tesi": "Contestare R1", "contesta": ["R1"], "poggia_su": [], "minoritaria": True}
    assert "integrano davvero" in M.steps_from_selection(ctx, [integrated], 0)[0]["passo"]


def test_an_article_without_decomposed_elements_yields_no_steps():
    ctx = _ctx(_records("provato", "provato", alt="provato", alt_requisites=False))
    requalify = {"tesi": "Riqualificazione nell'art. 589 c.p.", "contesta": [], "poggia_su": [], "realismo": "medio"}
    assert M.steps_from_selection(ctx, [requalify], 0) == []
    assert M.steps_from_selection(ctx, [{"tesi": "x", "contesta": ["R99"], "poggia_su": ["R98"]}], 0) == []
    assert M.steps_text([]) == ""                           # no steps: no «Passi successivi» heading


def test_the_containment_line_steps_are_its_own_riti():
    ctx = _ctx(_records("provato", "provato"))
    sols, main = M.coherent_solutions({"soluzioni": []}, ctx["records"], ctx["assessment"], ctx["riti"], [],
                                      ctx["posture"])
    assert sols[main].get("fallback")
    steps = M.steps_from_selection(ctx, sols, main)
    assert steps and steps[0]["passo"] == "Valutare i riti alternativi e le attenuanti"
    assert all("Acquisire" not in s["passo"] for s in steps)


# --- 4. the objections written for the selected line -----------------------------------------------------------------------

def _barred_plan():
    """The v71 path (a model's own theses): one contesting the integrated R1 unlabelled -- barred --, one sound."""
    return M.clean_method({"soluzioni": [
        {"tesi": "Assenza di dolo negli atti di percosse", "contesta": ["R1"], "realismo": "alto"},
        {"tesi": "Il nesso causale non è provato", "contesta": ["R3"], "realismo": "medio"},
    ], "obiezioni": [{"obiezione": "Dal piano del modello", "risposta": "prima della selezione"}]})


def test_the_objections_writer_receives_the_selected_line_and_the_barred_list(tmp_path, monkeypatch):
    monkeypatch.setenv(M.THESIS_LOG_ENV, str(tmp_path))
    ctx = _ctx(_records("provato", "provato", "da provare"))
    ctx["thesis_log"] = M.thesis_log(ctx, [], "", {}, {}, 0.0)
    got = {}

    def call(messages):
        got["system"], got["human"] = messages[0].content, messages[1].content
        return {"obiezioni": [
            {"obiezione": "L'accusa: il nesso è provato dalla sequenza dei fatti", "risposta": "Manca la perizia."},
            {"obiezione": "L'accusa: gli atti erano dolosi",
             "risposta": "Assenza di dolo negli atti di percosse."},                   # argues the barred thesis
            {"obiezione": "L'accusa: la caduta era prevedibile", "risposta": "Non rileva fuori dalla colpa."},
            {"obiezione": "L'accusa: la morte è conseguenza degli atti", "risposta": "Il nesso resta da provare."}]}
    plan = M.objections_plan(ctx, _barred_plan(), call)
    assert "LINEA PRINCIPALE: Il nesso causale non è provato; contesta R3.\n" in got["human"]
    assert "TESI ESCLUSE:\n- Assenza di dolo negli atti di percosse: contesta R1, integrato" in got["human"]
    assert "non usare mai una TESI ESCLUSA" in got["system"] and "l'accusa opporrà" in got["system"]
    assert M.data()["honesty_rules"][0] in got["system"]               # his rules, as in every writer call
    texts = [o["obiezione"] for o in plan["obiezioni"]]
    assert "Dal piano del modello" not in texts and "L'accusa: gli atti erano dolosi" not in texts
    assert len(texts) == 3 and ctx["timings"]["objections_s"] is not None
    import json
    log = json.loads(Path(ctx["thesis_log"]).read_text(encoding="utf-8"))
    assert log["objections"]["main"] == "Il nesso causale non è provato"
    assert log["objections"]["barred"] == ["Assenza di dolo negli atti di percosse"]
    assert log["objections"]["dropped"][0]["argues"] == "Assenza di dolo negli atti di percosse"
    answer = M.compose([("Norme applicate ai fatti", "regola")], ctx, plan)
    section = dict(M.answer_sections(answer))["Obiezioni e risposte"]
    assert "**Obiezione 1.** L'accusa: il nesso è provato dalla sequenza dei fatti." in section
    assert "dolosi" not in section


def test_a_failed_objections_call_leaves_compose_its_fallbacks(monkeypatch):
    ctx = _ctx(_records("provato", "provato", "da provare"))

    def broken(messages):
        raise RuntimeError("down")
    plan = M.objections_plan(ctx, _barred_plan(), broken)
    assert plan["obiezioni"] == []
    section = dict(M.answer_sections(M.compose([("Norme applicate ai fatti", "x")], ctx, plan)))["Obiezioni e risposte"]
    assert section.count("**Obiezione ") == 3


def test_the_thesis_call_no_longer_writes_the_objections():
    assert "\"obiezioni\"" not in M.METHOD_SYSTEM and "valutazioni" in M.METHOD_SYSTEM
    cands = [{"id": "T1", "label": M.SUSTAINABLE, "poggia_su": [], "contesta": []}]
    assert M.score_sample({"valutazioni": {"T1": {"realismo": "medio"}}}, cands, {}) == (10.0, [])


def test_the_flow_runs_the_objections_after_the_thesis_call(monkeypatch):
    calls = []
    monkeypatch.setattr(M, "prepare", lambda *a, **k: {"timings": {"thesis_selection_s": 1.0}})
    monkeypatch.setattr(M, "method_plan", lambda ctx: calls.append("thesis") or {"soluzioni": [], "obiezioni": []})

    def objections(ctx, plan):
        calls.append(("objections", plan))
        ctx["timings"]["objections_s"] = 2.0
        return {**plan, "obiezioni": [{"obiezione": "o", "risposta": "r"}]}
    monkeypatch.setattr(M, "objections_plan", objections)
    monkeypatch.setattr(flow, "objections", lambda *a: [])
    state = {}
    ctx, plan = flow.method_inputs(state, IMPUTATO, {}, [], [], [], "caso")
    assert calls[0] == "thesis" and calls[1][0] == "objections" and plan["obiezioni"][0]["obiezione"] == "o"
    assert state["timings"] == {"thesis_selection_s": 1.0, "objections_s": 2.0}
