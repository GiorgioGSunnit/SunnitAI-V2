"""r13 (21 Oct): CASE_FRAME_ENABLED, METHOD_CANDIDATES_ENABLED, FUSION_CLEAN_ENABLED
and the new STRATEGY_DRAFT_RULES_ENABLED rules. Offline: the model's replies
and the searches are mocked; the cases are synthetic, never the held-out ones."""

import json
import os

import pytest

os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test-password")
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("EMBEDDING_PROVIDER", "openai")
os.environ.setdefault("EMBEDDING_API_KEY", "test-key")

import src.rag.defensive_generation as dg  # noqa: E402
from src.rag import method_candidates as mc  # noqa: E402

FRAME_REPLY = {
    "facts": ["Tizio compra un'auto", "l'auto ha un difetto al motore"],
    "procedural_posture": "controversia contrattuale prima del giudizio",
    "posture_type": "precontenzioso",
    "areas": ["contratti", "consumo", "astrologia"],
    "tasks": ["quali rimedi ha Tizio", "quali rimedi ha Tizio", "entro quando deve agire"],
    "central_issue_candidates": [{"nome": "garanzia per i vizi della cosa venduta", "codice": "c.c."},
                                 {"nome": "art. 1490 c.c.", "codice": "c.c."}, "difetto di conformità"],
}


def _reply(monkeypatch, reply):
    monkeypatch.setattr(dg, "_chat", lambda *a, **k: json.dumps(reply))


# --- the frame ----------------------------------------------------------------

def test_the_frame_is_normalised(monkeypatch):
    _reply(monkeypatch, FRAME_REPLY)
    frame = dg._case_frame("facts", "")
    assert frame["posture_type"] == "precontenzioso"
    assert frame["areas"] == ["contratti", "consumo"]                     # unknown area dropped
    assert frame["tasks"] == ["quali rimedi ha Tizio", "entro quando deve agire"]   # deduplicated
    # A "central issue" that is an article number is not an institution in words.
    assert frame["central_issue_candidates"] == [
        {"nome": "garanzia per i vizi della cosa venduta", "codice": "c.c."},
        {"nome": "difetto di conformità", "codice": ""}]


def test_an_unknown_posture_is_altro_and_an_empty_frame_is_none(monkeypatch):
    _reply(monkeypatch, {**FRAME_REPLY, "posture_type": "boh"})
    assert dg._case_frame("facts", "")["posture_type"] == "altro"
    _reply(monkeypatch, {"facts": ["x"], "tasks": [], "central_issue_candidates": []})
    assert dg._case_frame("facts", "") is None
    monkeypatch.setattr(dg, "_chat", lambda *a, **k: "not json")
    assert dg._case_frame("facts", "") is None


def test_the_frame_queries_are_tasks_posture_then_central_issues():
    frame = {"tasks": ["a", "b"], "procedural_posture": "fase", "central_issue_candidates": [{"nome": "a"},
                                                                                           {"nome": "c"}]}
    assert dg._frame_queries(frame) == ["a", "b", "fase", "c"]
    assert dg._frame_queries({}) == []


def test_central_issues_become_first_questions_unless_already_named():
    analysis = {"questioni": [{"titolo": "q", "istituti": [{"nome": "Delitto tentato", "codice": "c.p."}]}]}
    frame = {"central_issue_candidates": [{"nome": "delitto  tentato", "codice": "c.p."},
                                          {"nome": "atti preparatori", "codice": "c.p."}]}
    out = dg._with_central_questions(analysis, frame)
    assert [q["titolo"] for q in out["questioni"]] == ["Questione centrale: atti preparatori", "q"]
    assert out["questioni"][0]["istituti"] == [{"nome": "atti preparatori", "codice": "c.p.",
                                                "numero_indicativo": ""}]
    assert dg._with_central_questions(analysis, {}) is analysis


def test_the_pipeline_analysis_reads_the_frame_only_when_enabled(monkeypatch):
    seen = {}

    def v2(facts, lang, request="", frame=None):
        seen["frame"] = frame
        return {"area": "civile", "questioni": []}
    monkeypatch.setattr(dg, "_analyse_case_v2", v2)
    monkeypatch.setattr(dg, "_case_frame", lambda facts, request: {
        "tasks": ["t"], "central_issue_candidates": [{"nome": "x", "codice": "c.c."}]})
    monkeypatch.setenv("CASE_ANALYSIS_ENABLED", "true")
    monkeypatch.delenv("STRUCTURED_ANALYSIS_ENABLED", raising=False)

    monkeypatch.delenv("CASE_FRAME_ENABLED", raising=False)
    assert "frame" not in dg._analyse_case_for_pipeline("f", "it")
    assert seen["frame"] is None

    monkeypatch.setenv("CASE_FRAME_ENABLED", "true")
    out = dg._analyse_case_for_pipeline("f", "it")
    assert seen["frame"]["tasks"] == ["t"] and out["frame"]["tasks"] == ["t"]
    assert out["questioni"][0]["titolo"] == "Questione centrale: x"


def test_the_frame_reaches_the_analysis_prompt(monkeypatch):
    prompts = []

    def fake(messages, **k):
        prompts.append(messages[1].content)
        return json.dumps({"area": "civile", "questioni": []})
    monkeypatch.setattr(dg, "_chat", fake)
    dg._analyse_case_v2("fatti", "it", "", frame={"procedural_posture": "prima del giudizio", "tasks": ["quali rimedi"],
                                                  "central_issue_candidates": [{"nome": "risoluzione", "codice": ""}]})
    assert "QUADRO DEL CASO" in prompts[0] and "- quali rimedi" in prompts[0] and "- risoluzione" in prompts[0]
    dg._analyse_case_v2("fatti", "it", "")
    assert "QUADRO DEL CASO" not in prompts[1]


def test_the_structured_grid_gets_a_central_slot_first():
    assert dg._central_slot(None) == []
    slot = dg._central_slot({"central_issue_candidates": [{"nome": "giudizio di ottemperanza"}]})
    assert slot[0][0] == "questione_centrale" and "giudizio di ottemperanza" in slot[0][2]


# --- fusion: clean searches and the frame as a fourth signal ----------------------

class _Result:
    def __init__(self, rows):
        self._rows = rows

    def data(self):
        return self._rows


class _Session:
    """Returns `vector` rows for the vector query and `fulltext` rows for the fulltext one."""
    def __init__(self, vector, fulltext):
        self.vector, self.fulltext, self.params = vector, fulltext, []

    def run(self, query, **params):
        self.params.append(params)
        return _Result(self.vector if "vector" in query else self.fulltext)


def test_clean_searches_drop_non_laws_by_name_and_keep_k(monkeypatch):
    monkeypatch.setattr(dg, "embedding_model", type("E", (), {"embed_query": staticmethod(lambda text: [0.1])}))
    vector = [{"name": "1.0.0", "doc_name": "Lineamenti di diritto civile"},
              {"name": "1453.0.0", "doc_name": "Codice Civile"},
              {"name": "3.0.0", "doc_name": "LOCAZIONE ABITATIVA DI NATURA TRANSITORIA"},
              {"name": "1455.0.0", "doc_name": "Codice Civile"},
              {"name": "1460.0.0", "doc_name": "Codice Civile"}]
    fulltext = [{"name": "108.0.0", "doc_name": "Codice dei contratti pubblici"}]
    session = _Session(vector, fulltext)
    v, f = dg._fusion_clean_searches(session, "inadempimento", "idx", 2)
    assert v == [("1453", "c.c."), ("1455", "c.c.")]
    assert f == [("108", "Codice dei contratti pubblici")]          # a law, not a contract model
    assert session.params[0]["fetch"] == 3 * 2 * dg._CLEAN_OVERFETCH and session.params[0]["limit"] == 6


@pytest.fixture
def fusion_env(monkeypatch):
    monkeypatch.setattr(dg, "_select_articles_v2", lambda *a, **k: {
        "first_rulings": [], "similar": [], "keep": [], "named": [], "candidates": [], "chosen": [],
        "selected": [("1", "c.c.")], "article_rows": [], "spotted": [], "institutions_resolved": [],
        "heading_matched": []})
    monkeypatch.setattr(dg, "_fusion_distill_bullets", lambda case: [])
    import src.rag.doc_lookup as doc_lookup
    monkeypatch.setattr(doc_lookup, "_fetch_doc_names", lambda driver, database: [])
    monkeypatch.setattr(dg, "_fetch_any", lambda session, documents, refs: [
        {"d": {"id": "doc"}, "s": {"name": f"{n}.0.0", "plain_text": "t"}, "_article": (n, c)} for n, c in refs])
    import src.rag.nodes.retrieval as retrieval_module
    monkeypatch.setattr(retrieval_module, "broaden_article_context",
                        lambda state, driver, database: {"raw_result": state["raw_result"]})
    return monkeypatch


def test_fusion_uses_the_clean_searches_only_when_enabled(fusion_env):
    calls = []
    fusion_env.setattr(dg, "_fusion_text_search", lambda s, text, i, k: calls.append("plain") or [])
    fusion_env.setattr(dg, "_fusion_clean_searches", lambda s, text, i, k: calls.append("clean") or ([], []))
    fusion_env.delenv("FUSION_CLEAN_ENABLED", raising=False)
    dg._select_articles_fusion(object(), object(), "db", "facts", "", {"questioni": []})
    assert calls == ["plain"]
    calls.clear()
    fusion_env.setenv("FUSION_CLEAN_ENABLED", "true")
    dg._select_articles_fusion(object(), object(), "db", "facts", "", {"questioni": []})
    assert calls == ["clean"]


def test_the_frame_queries_are_a_fourth_signal(fusion_env):
    found = {"quali rimedi": [("2", "c.c."), ("1", "c.c.")]}
    fusion_env.delenv("FUSION_CLEAN_ENABLED", raising=False)
    fusion_env.setattr(dg, "_fusion_text_search", lambda s, text, i, k: found.get(text, []))
    picked = dg._select_articles_fusion(object(), object(), "db", "facts", "",
                                        {"questioni": [], "frame": {"tasks": ["quali rimedi"]}})
    # 1 c.c. is in CURRENT and in the frame search: two signals, first.
    assert picked["candidates"] == [("1", "c.c."), ("2", "c.c.")]
    multi = {r["_article"] for r in picked["article_rows"] if r.get("_multi_signal")}
    assert multi == {("1", "c.c.")}


# --- method candidates ---------------------------------------------------------------

def test_the_lists_hold_the_briefs_procedural_articles():
    appeal = mc.POSTURE_ARTICLES["condanna_primo_grado"][1]
    assert {(n, "c.p.p.") for n in ("581", "585", "593", "597", "603")} <= set(appeal)
    assert {("606", "c.p.p."), ("569", "c.p.p.")} <= set(mc.POSTURE_ARTICLES["sentenza_penale_appello"][1])
    assert [n for n, _ in mc.POSTURE_ARTICLES["giudicato_amministrativo_non_eseguito"][1]] == \
        ["112", "113", "114", "117"]
    assert ("56", "c.p.") in mc.AREA_ARTICLES["penale"][1]
    # Every posture the frame can return has a list, or is "altro".
    assert set(dg._POSTURE_TYPES) - set(mc.POSTURE_ARTICLES) == {"altro"}
    assert set(mc.AREA_ARTICLES) <= set(dg._FRAME_AREAS)


def test_candidate_refs_dedupe_and_fall_back_to_the_analysis_area():
    refs = mc.candidate_refs(["contratti", "locazione"], "precontenzioso")
    assert len(refs) == len(set(refs)) and refs.index(("1453", "c.c.")) < refs.index(("1571", "c.c."))
    assert mc.areas_for([], "penale") == ["penale"]
    assert mc.areas_for(["lavoro"], "civile") == []
    assert mc.areas_for(["famiglia"], "penale") == ["famiglia"]


def _picked(rows):
    return {"article_rows": rows, "selected": [r["_article"] for r in rows]}


def test_method_candidates_need_both_flags_and_are_never_exempt(monkeypatch):
    import src.rag.doc_lookup as doc_lookup
    monkeypatch.setattr(doc_lookup, "_fetch_doc_names", lambda driver, database: [])
    monkeypatch.setattr(dg, "_fetch_any", lambda session, documents, refs: [
        {"d": {"id": "doc"}, "s": {"name": f"{n}.0.0", "plain_text": "testo"}, "_article": (n, c)}
        for n, c in refs if n != "114"])
    analysis = {"area": "amministrativo",
                "frame": {"areas": [], "posture_type": "giudicato_amministrativo_non_eseguito"}}
    picked = _picked([{"d": {}, "s": {}, "_article": ("112", "c.p.a.")}])

    monkeypatch.delenv("METHOD_CANDIDATES_ENABLED", raising=False)
    assert dg._add_method_candidates(None, None, "db", analysis, picked) is picked
    monkeypatch.setenv("METHOD_CANDIDATES_ENABLED", "true")
    monkeypatch.delenv("APPLICABILITY_CHECK_ENABLED", raising=False)
    assert dg._add_method_candidates(None, None, "db", analysis, picked) is picked

    monkeypatch.setenv("APPLICABILITY_CHECK_ENABLED", "true")
    out = dg._add_method_candidates(None, None, "db", analysis, picked)
    added = out["method_candidates"]
    assert ("112", "c.p.a.") not in added and ("113", "c.p.a.") in added and ("114", "c.p.a.") not in added
    assert ("21-septies", mc.L241) in added                        # the area's own list
    assert out["selected"][0] == ("112", "c.p.a.") and ("113", "c.p.a.") in out["selected"]
    to_check, _ = dg._applicability_candidates("fatti", "", out["article_rows"])
    assert ("113", "c.p.a.") in to_check                                  # filtered by the check


# --- draft rules ------------------------------------------------------------------------

def test_the_draft_rules_ask_for_the_answers_first_and_forbid_concessions():
    for phrase in ("**Risposte ai quesiti**", "mai \"art. 585\" da solo", "Non concedere mai un elemento contestato",
                   "Niente sezioni o paragrafi generici", "ometti il riferimento"):
        assert phrase in dg._DRAFT_RULES


def test_the_questions_reach_the_draft_prompt():
    human = dg._parere_human("fatti", {"frame": {"tasks": ["quale reato", "quali rimedi"]}}, "", "")
    assert "QUESITI DEL CASO:\n- quale reato\n- quali rimedi" in human
    assert "QUESITI DEL CASO" not in dg._parere_human("fatti", {}, "", "")


def test_a_wrong_code_citation_names_the_code_the_sources_hold():
    text = "Il termine dell'art. 585 c.p. decorre dal deposito; l'art. 339 c.p. aggrava la rapina."
    known = [("585", "c.p.p."), ("628", "c.p.")]
    plain = dg._mark_unverified(text, known, "")
    assert plain.count(dg._UNVERIFIED) == 2
    marked = dg._mark_unverified(text, known, "", wrong_code=True)
    assert "art. 585 c.p. [DA VERIFICARE: nelle fonti è l'art. 585 c.p.p.]" in marked
    assert "art. 339 c.p." + dg._UNVERIFIED in marked
