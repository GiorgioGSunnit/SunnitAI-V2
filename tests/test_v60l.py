"""v60l: open points are facts a client can state -- a legal characterisation is never a point (dropped «qualificazione»,
in §2 «da verificare» without a question, the writer told to settle it), and the fact after «il caso non dice» starts
lowercase. (v60l's open-point protection of a demotion was removed in v60m: a silent element no longer demotes.) The
model and the graph are stubbed."""

import contextlib
import os
import sys
from pathlib import Path

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from src.rag import answer_ledger, case_read, render_computed, topic_groups  # noqa: E402
from src.rag import defensive_generation as dg  # noqa: E402
from src.rag import interaction_copy as C  # noqa: E402
from src.rag import interactive_flow as flow  # noqa: E402
import test_case_first as cf  # noqa: E402
from test_v60f import _graded_state  # noqa: E402
from test_v60k import _event, _single  # noqa: E402

NORMS = flow.ANSWER_TITLES[1]
CASE = "Leonardo, proprietario di un terreno, ... Domitilla ha chiuso il cancello nel 2016."


def test_analyse_case_drops_a_qualification_from_the_questions(monkeypatch):
    monkeypatch.setenv("INTERACTIVE_CASE_FIRST", "true")
    monkeypatch.setattr(dg, "_analyse_case_for_pipeline", lambda text, lang, request="": {"frame": {"areas": ["penale"]}})
    read = {**cf.READ, "non_detto": ["la qualificazione giuridica dello sparo", "se l'arma era detenuta legalmente"]}
    monkeypatch.setattr(case_read, "read", lambda text, request="", lang="it": case_read.clean(read, request))
    rows = [cf._graded("52", "c.p.", 3, "Difesa legittima")]
    built = topic_groups.build(rows, cf.UNITS)
    monkeypatch.setattr(flow, "_retrieve", lambda text, request, lang, analysis, pending=None: {
        "article_rows": rows, "regime_rows": [], "topics": built["topics"], "others": [], "topic_rows": rows,
        "rubriche": topic_groups.rubriche_of(rows), "grading": {"s": 1.0, "topics_s": 0.1}, "reserve": []})
    monkeypatch.setattr(flow, "_case_branches_early", lambda *a: contextlib.nullcontext())
    monkeypatch.setattr(render_computed, "render", lambda *a, **k: "")
    monkeypatch.setattr(flow, "questions", lambda *a, **k: [])
    logged = []
    monkeypatch.setattr(flow, "_log", lambda event, data, *a: logged.append((event, data)))
    state = flow.analyse_case("Tizio spara al ladro.", "")
    entry = next(d for e, d in logged if e == "first_read_questions")
    assert {"point": "la qualificazione giuridica dello sparo", "dropped": "qualificazione"} in entry["dropped"]
    assert entry["asked"] == ["read_2"] and state["qualifications"] == ["la qualificazione giuridica dello sparo"]
    assert not any("qualificazione" in q["text"] for q in state["questions"])


# --- 2. open points are facts a client can state -----------------------------------------------------------------------------

def test_a_legal_characterisation_is_never_a_point():
    for point in ("la natura giuridica del rapporto tra le parti", "se si tratta di servitù o di obbligazione",
                  "La qualificazione del contratto", "come qualificare il passaggio", "l'inquadramento giuridico della "
                  "condotta", "se si tratti di un'occupazione abusiva", "Si tratta di locazione o di comodato?"):
        assert flow.qualification(point), point
    for point in ("la data del rogito", "se Leonardo ha ricevuto la diffida", "quante volte è passato",
                  "se esiste un atto scritto", "chi ha chiuso il cancello", "se il contratto è stato registrato",
                  "l'importo del canone", "se la società è in liquidazione"):
        assert not flow.qualification(point), point
    dropped = []
    out = flow.with_read_questions([], ["la natura giuridica del passaggio", "la data del rogito"], {}, dropped)
    assert [q["point"] for q in out] == ["la data del rogito"]
    assert dropped == [{"point": "la natura giuridica del passaggio", "dropped": "qualificazione"}]


def test_open_points_drop_a_qualification_and_the_writer_settles_it(monkeypatch):
    monkeypatch.setenv(flow.OPEN_POINTS_ENV, "3")          # v60q: no block by default (OPEN_POINTS_MAX 0)
    state = _graded_state(monkeypatch)
    state["mode"] = "single_shot"
    state["qualifications"] = ["la natura giuridica del passaggio"]
    state["questions"] = [
        {"id": "read_1", "kind": "free", "family": "lettura", "text": "Il caso non dice se si tratta di servitù o di "
         "comodato: puoi precisarlo?", "point": "se si tratta di servitù o di comodato", "unlocks": ["requisito: x"]},
        {"id": "read_2", "kind": "free", "family": "lettura", "text": "Il caso non dice la data del rogito: puoi "
         "precisarlo?", "point": "La data del rogito", "unlocks": ["requisito: x"]}]
    got = flow.open_points(state, [])
    assert {"point": "la natura giuridica del passaggio", "reason": "qualificazione"} in got["dropped"]
    assert {"question": "read_1", "reason": "qualificazione"} in got["dropped"]
    assert [p["question"] for p in got["points"]] == ["read_2"]
    assert got["points"][0]["point"] == "da verificare: il caso non dice la data del rogito"     # lowercase
    assert [a["question"] for a in got["asked"]] == ["read_2"] and "servitù o di comodato" not in got["block"]
    block = flow.unknown_block(state)
    assert "- da verificare: il caso non dice la data del rogito" in block and "servitù o di comodato" not in \
        block.split(C.QUALIFICATIONS_WRITER_TITLE)[0]
    assert f"{C.QUALIFICATIONS_WRITER_TITLE}:\n- la natura giuridica del passaggio\n" in block
    assert C.QUALIFICATIONS_WRITER_RULE in block


def test_the_fact_after_il_caso_non_dice_starts_lowercase():
    assert flow.open_point_fact({"point": "La data del rogito."}) == "la data del rogito"
    assert flow.open_point_fact({"point": "Se il contratto è stato registrato?"}) == "se il contratto è stato registrato"
    assert flow.open_point_fact({"point": "Leonardo quando ha ricevuto la diffida"}, ["Leonardo Rossi"]) == \
        "Leonardo quando ha ricevuto la diffida"                                                     # a party's name
    assert flow.open_point_fact({"point": "INPS: la data della domanda"}) == "INPS: la data della domanda"   # acronym
    assert flow.open_point_label({"point": "Quando è stata notificata la diffida"}) == \
        "il caso non dice quando è stata notificata la diffida"


def test_a_requisite_a_qualification_leaves_open_reads_da_verificare_without_a_question(monkeypatch):
    state, seen = _single(monkeypatch)
    state["questions"] = [q for q in state["questions"] if q["id"] != "atto_separazione"]
    flow.refresh_records(state)
    state["qualifications"] = ["la qualificazione del possesso dei due fondi da parte dello stesso proprietario"]
    text = flow.single_shot(CASE)["answer"]
    norms = dict(answer_ledger.sections_of(text))[NORMS]
    assert "requisito «consta" not in norms                          # v60r (1): the rows go to the citations only
    row = next(r for r in state["ledger_rows"] if (r["requisite"] or "").startswith("consta, mediante"))
    assert row["status"] == C.STATUS_DA_PROVARE and row["supporting_sentence"] is None      # v60r (6)
    assert "il caso non dice la qualificazione" not in text and "qualificazione del possesso" not in \
        text.split(f"**{C.OPEN_POINTS_TITLE}**")[-1]
    points = _event(seen, "open_points")
    assert {"point": state["qualifications"][0], "reason": "qualificazione"} in points["dropped"]
    assert C.QUALIFICATIONS_WRITER_TITLE in " ".join(str(j) for j in seen["jobs"][-1])


def test_the_sim_report_names_the_dropped_qualifications(monkeypatch, tmp_path):
    import importlib.util
    import json
    spec = importlib.util.spec_from_file_location("diagnose_long", HERE.parent / "scripts" / "diagnose_long.py")
    dl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dl)
    state, seen = _single(monkeypatch)
    state["qualifications"] = ["la natura giuridica del passaggio"]
    monkeypatch.setattr(dl, "answer_extras", lambda *a, **k: {})
    monkeypatch.setattr(dl, "_eval_case", lambda cid: None)

    class Ev:
        @staticmethod
        def score(case, answer, citations):
            return {"expected": [], "failures": []}

        @staticmethod
        def mentions(answer):
            return []
    (tmp_path / "cases").mkdir()
    out = tmp_path / "cases" / "c1.json"
    dl.run_interactive_sim({"id": "c1", "out": str(out), "single_shot": True}, CASE, "", Ev(),
                           {"calls": 0, "seconds": 0, "errors": 0}, lambda message: None)
    report = Path(dl.write_interactive_report(str(tmp_path), Ev())).read_text(encoding="utf-8")
    assert "«la natura giuridica del passaggio»: qualificazione" in report and "None:" not in report
