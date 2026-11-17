"""v79 block D (79.3) -- defences and candidates, on synthetic rows and records only:

D1. a general-part defence article (40, 41, 43, 50-56, 83, 131-bis c.p.) enters the record when the first read names
    its concept (data/general_concepts.json) or the retrieval ranks it among the top; with its elements proven it is a
    defence line among the candidates; the gate still drops the rest
D2. the retrieval's top-ranked articles not in the record are graded as candidates, never records; one that grades to
    nothing is dropped silently
D3. the citation checks never remove a sentence citing an article of the record or of the candidates
D4. the nightly counts the defence lines, the defence articles the gate dropped, the ranked candidates
"""

import importlib.util
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    for name, value in (("LLM_API_KEY", "k"), ("OPENAI_API_KEY", "k"), ("EMBEDDING_PROVIDER", "openai"),
                        ("EMBEDDING_API_KEY", "k"), ("NEO4J_URI", "bolt://localhost:7687"), ("NEO4J_USER", "n"),
                        ("NEO4J_PASSWORD", "x")):
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("THESIS_LOG_DIR", str(tmp_path / "logs"))


def _script(name):
    spec = importlib.util.spec_from_file_location(name, HERE.parent / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _req(i, text, status="provato"):
    return {"id": f"R{i}", "comma": 1, "kind": "requisito", "text": text, "status": status}


def _row(number, code, rank=None, grade=None, rubrica=""):
    row = {"_article": (number, code), "s": {"name": number, "titolo_articolo": rubrica}, "d": {"id": code}}
    if rank is not None:
        row["_fused_rank"] = rank
    if grade is not None:
        row["_grade"] = grade
    return row


IMPUTATO = {"rito": "penale", "family": "penale", "role": "imputato", "stage": "primo_grado"}
ATTORE = {"rito": "civile", "family": "civile", "role": "attore", "stage": "pre_filing"}


# --- D1: entry -----------------------------------------------------------------------------------------------------------

def test_a_defence_article_ranked_among_the_top_enters_and_a_low_one_is_counted_as_dropped():
    from src.rag import interactive_flow as flow
    rows = [_row(str(600 + i), "c.p.", rank=i) for i in range(1, 9)] + [_row("52", "c.p.", rank=3.5),
                                                                       _row("55", "c.p.", rank=40),
                                                                       _row("62", "c.p.", rank=2)]
    got = flow.defence_ranked(rows)
    assert got == {"ranked": ["52 c.p."], "dropped": ["55 c.p."]}           # 62 c.p. is no defence article


def test_the_concept_the_first_read_names_brings_its_defence_article():
    from src.rag import interactive_flow as flow
    state = {"read": {"facts": ["Alfa ha reagito per legittima difesa e ha ecceduto in eccesso colposo."],
                      "hypotheses": [{"claim": "tentativo di furto", "institutions": ["tentativo"]}]},
             "defence_gate": {"ranked": ["41 c.p."]}}
    assert flow.defence_wanted(state) == ["52 c.p.", "55 c.p.", "56 c.p.", "41 c.p."]


def test_the_gate_still_drops_the_rest_and_a_defence_article_off_its_route():
    from src.rag import issue_records
    assert issue_records.decomposed_by_gate("55 c.p.", "Eccesso colposo", defence=True)
    assert not issue_records.decomposed_by_gate("55 c.p.", "Eccesso colposo")           # a topic's: as before
    assert not issue_records.decomposed_by_gate("62 c.p.", "Circostanze attenuanti comuni", defence=True)
    assert not issue_records.decomposed_by_gate("405 c.p.p.", "", defence=True)
    assert not issue_records.counted("55 c.p.")                                       # never a fattispecie
    assert issue_records.DEFENCE_WHY == "difesa di parte generale"


def _decompose(monkeypatch, state, posture, rows, reqs):
    from src.rag import interactive_flow as flow, issue_records, posture as posture_mod
    monkeypatch.setattr(flow, "case_text", lambda s: "Alfa colpisce Beta che lo aggrediva.")
    monkeypatch.setattr(flow, "_restored_analysis", lambda a: {})
    monkeypatch.setattr(flow, "user_named", lambda s: [])
    monkeypatch.setattr(posture_mod, "intake_charges", lambda p: [])
    monkeypatch.setattr(posture_mod, "charge_labels", lambda case: [])
    monkeypatch.setattr(posture_mod, "claim_from_read", lambda *a, **k: [])
    monkeypatch.setattr(posture_mod, "alternative_labels", lambda s, rows=(): [])
    monkeypatch.setattr(issue_records, "_requisites", lambda r, a: [dict(x) for x in reqs])
    monkeypatch.setattr(issue_records, "refresh", lambda *a, **k: None)
    monkeypatch.setattr(issue_records, "seed", lambda rec, row: None)
    return flow.decompose_records(state, posture, rows, set(), fetch=lambda keys: [])


def test_a_ranked_defence_article_gets_its_elements_even_over_a_topics_general_part_record(monkeypatch):
    from src.rag import interactive_flow as flow
    state = {"records": {"55 c.p.": {"rubrica": "Eccesso colposo", "requisites": [], "not_fattispecie": "x"}},
             "analysis": {}, "read": {}, "defence_gate": {"ranked": ["55 c.p."]}}
    link, _ = _decompose(monkeypatch, state, IMPUTATO, [_row("55", "c.p.", rubrica="Eccesso colposo")],
                         [{"id": "", "comma": 1, "kind": "requisito", "text": "eccede i limiti", "status": "da provare"}])
    rec = state["records"]["55 c.p."]
    assert link == ["55 c.p."] and rec["decomposed_for"] == flow.DECOMPOSE_WHY["defence"]
    assert rec["requisites"] and "not_fattispecie" not in rec


def test_no_defence_route_in_a_civil_case(monkeypatch):
    state = {"records": {}, "analysis": {}, "read": {}, "defence_gate": {"ranked": ["52 c.p."]}}
    link, _ = _decompose(monkeypatch, state, ATTORE, [_row("52", "c.p.")], [
        {"id": "", "comma": 1, "kind": "requisito", "text": "x", "status": "da provare"}])
    assert link == [] and state["records"] == {}


# --- D1: the defence line ---------------------------------------------------------------------------------------------

def _ctx(recs, posture=IMPUTATO):
    from src.rag import lawyer_method as M
    return M.prepare({"records": recs, "sentences": [{"id": "S1", "text": "Alfa colpisce Beta.", "source": "caso",
                                                      "certainty": "accertato"}]}, posture, [], "Alfa colpisce Beta.",
                     "Alfa", "", list(recs), [], [], [], [], link_call=lambda *a: {}, case_id_="v79d")


def _charge(*statuses):
    texts = ["cagiona una lesione personale", "dalla quale deriva una malattia"]
    return {"582 c.p.": {"rubrica": "Lesione personale", "side": "threat", "charged": 1,
                         "requisites": [_req(i, texts[i - 1], st) for i, st in enumerate(statuses, 1)]}}


@pytest.mark.parametrize("article, rubrica", [("55 c.p.", "Eccesso colposo"), ("56 c.p.", "Delitto tentato"),
                                              ("41 c.p.", "Concorso di cause"), ("131-bis c.p.", "Esclusione della "
                                                                                                 "punibilità per particolare tenuità del fatto")])
def test_a_defence_article_with_its_elements_proven_is_a_defence_line(article, rubrica):
    from src.rag import lawyer_method as M
    recs = {**_charge("provato", "da provare"),
            article: {"rubrica": rubrica, "side": "weapon", "decomposed_for": "difesa di parte generale",
                      "requisites": [_req(9, "il presupposto della difesa")]}}
    cands = M.thesis_candidates(_ctx(recs))
    defence = [c for c in cands if c["kind"] == "defence"]
    assert len(defence) == 1 and defence[0]["article"] == article and defence[0]["poggia_su"] == ["R9"]
    assert defence[0]["argued"] == "code" and defence[0]["code_argument"]["argomento"]


def test_no_defence_line_with_an_element_excluded_nor_for_a_victim():
    from src.rag import lawyer_method as M
    recs = {**_charge("provato", "da provare"),
            "55 c.p.": {"rubrica": "Eccesso colposo", "side": "weapon", "decomposed_for": "difesa di parte generale",
                        "requisites": [_req(9, "il presupposto"), _req(10, "l'eccesso colposo", "non risulta")]}}
    assert not [c for c in M.thesis_candidates(_ctx(recs)) if c["kind"] == "defence"]
    recs["55 c.p."]["requisites"] = [_req(9, "il presupposto")]
    victim = {**IMPUTATO, "role": "persona_offesa"}
    assert not [c for c in M.thesis_candidates(_ctx(recs, victim)) if c["kind"] == "defence"]


def test_a_defence_article_is_never_the_charge_nor_a_subsumption_table():
    from src.rag import lawyer_method as M
    recs = {**_charge("provato", "provato"),
            "55 c.p.": {"rubrica": "Eccesso colposo", "side": "weapon", "decomposed_for": "difesa di parte generale",
                        "requisites": [_req(9, "il presupposto")]}}
    ctx = _ctx(recs)
    assert ctx["assessment"]["article"] == "582 c.p." and "55 c.p." not in ctx["subsumption"]


# --- D2 ------------------------------------------------------------------------------------------------------------------

def test_the_ranked_labels_are_the_top_graded_fattispecie():
    from src.rag import interactive_flow as flow
    rows = [_row("640", "c.p.", rank=5, grade=2), _row("646", "c.p.", rank=1, grade=3), _row("52", "c.p.", rank=1,
                                                                                          grade=3),
            _row("624", "c.p.", rank=2, grade=1), _row("405", "c.p.p.", rank=1, grade=3)]
    rows += [_row(str(700 + i), "c.p.", rank=10 + i, grade=2) for i in range(12)]
    got = flow.ranked_labels(rows)
    assert got[:2] == ["646 c.p.", "640 c.p."] and len(got) == flow.RANKED_TOP
    assert "624 c.p." not in got and "52 c.p." not in got and "405 c.p.p." not in got


def test_a_ranked_article_not_in_the_record_is_decomposed_as_a_candidate(monkeypatch):
    state = {"records": {"646 c.p.": {"rubrica": "Appropriazione indebita", "requisites": [_req(1, "x")]}},
             "analysis": {}, "read": {}, "ranked": ["646 c.p.", "640 c.p."]}
    link, _ = _decompose(monkeypatch, state, IMPUTATO, [_row("640", "c.p.", rubrica="Truffa")],
                         [{"id": "", "comma": 1, "kind": "requisito", "text": "artifizi o raggiri",
                           "status": "da provare"}])
    assert link == ["640 c.p."] and state["records"]["640 c.p."]["ranked_candidate"]
    assert not state["records"]["646 c.p."].get("ranked_candidate")


def _ranked(*statuses):
    return {"640 c.p.": {"rubrica": "Truffa", "side": "weapon", "ranked_candidate": True,
                         "decomposed_for": "candidata dalla classifica del recupero",
                         "requisites": [_req(20 + i, t, st) for i, (t, st) in
                                        enumerate(zip(["artifizi o raggiri", "induce in errore"], statuses))]}}


def test_a_ranked_candidate_with_a_proven_element_is_a_candidate_never_a_record():
    from src.rag import lawyer_method as M
    recs = {**_charge("provato", "da provare"), **_ranked("provato", "da provare")}
    ctx = _ctx(recs)
    requalify = [c for c in M.thesis_candidates(ctx) if c["kind"] == "requalify"]
    assert [c["article"] for c in requalify] == ["640 c.p."]
    assert "640 c.p." not in ctx["subsumption"] and ctx["assessment"]["article"] == "582 c.p."
    assert all(m.get("article") != "640 c.p." for m in ctx["facts"]["mancanti"])


def test_a_ranked_candidate_that_grades_to_nothing_is_dropped_silently():
    from src.rag import lawyer_method as M
    log = {}
    view = M.method_records({**_charge("provato", "da provare"), **_ranked("da provare", "da provare")}, log)
    assert "640 c.p." not in view and log["ranked_dropped"] == ["640 c.p."]
    view = M.method_records({**_charge("provato", "da provare"), **_ranked("provato", "non risulta")}, {})
    assert "640 c.p." not in view


def test_a_claimants_ranked_candidate_is_a_claim_only_with_a_proven_element():
    from src.rag import lawyer_method as M
    recs = {"1218 c.c.": {"rubrica": "Responsabilità del debitore", "side": "weapon",
                          "requisites": [_req(1, "l'inadempimento", "provato")]},
            "2043 c.c.": {"rubrica": "Risarcimento per fatto illecito", "side": "weapon", "ranked_candidate": True,
                          "requisites": [_req(5, "il fatto doloso o colposo", "provato")]}}
    cands = M.thesis_candidates(_ctx(recs, ATTORE))
    assert any(c["kind"] == "claim" and c["article"] == "2043 c.c." for c in cands)


# --- D3 ------------------------------------------------------------------------------------------------------------------

def test_the_candidates_articles_are_never_removed_for_scope():
    from src.rag import lawyer_method as M
    ctx = {"candidates": [{"article": "640 c.p.", "tesi": "Riqualificazione nell'art. 640 c.p."},
                          {"tesi": "Costituirsi parte civile (artt. 74-79 c.p.p.)", "articles": "art. 79 c.p.p."}]}
    got = M.candidate_articles(ctx)
    assert "640 c.p." in got and "79 c.p.p." in got
    scope = M.article_scope({**ctx, "records": {"582 c.p.": {}}})
    text = "La truffa (art. 640 c.p.) richiede artifizi. La costituzione segue l'art. 79 c.p.p."
    assert M.articles_out(text, scope) == text


def test_the_writers_check_allows_the_record_and_the_candidates():
    from src.rag import interactive_flow as flow
    state = {"records": {"55 c.p.": {}, "2043 c.c.": {"dropped": "grado 1"}}, "candidate_articles": ["640 c.p."],
             "sentences": []}
    got = flow.check_context(state, [], [], [], [], [], "", [])
    assert {"55 c.p.", "640 c.p."} <= got["allowed"] and "2043 c.c." not in got["allowed"]     # a dropped one is not


# --- D4 ------------------------------------------------------------------------------------------------------------------

def test_the_counts_carry_the_defence_lines_and_the_ranked_candidates():
    from src.rag import lawyer_method as M
    ctx = {"records": {"640 c.p.": {"ranked_candidate": True}}, "assessment": {}, "posture": IMPUTATO,
           "candidates": [{"id": "T1", "kind": "defence", "argued": "code", "article": "55 c.p."},
                          {"id": "T2", "kind": "requalify", "argued": "model", "article": "640 c.p."}]}
    plan = {"soluzioni": [{"tesi": "Difesa", "realismo": "medio", "contesta": [], "poggia_su": ["R9"],
                           "candidate": "T1", "kind": "defence"}]}
    counts = M.selection_counts(ctx, plan, {})
    assert counts["defence_lines"] == 1 and counts["ranked_candidates"] == 1


def test_the_nightly_prints_the_defence_gate_per_criminal_cell():
    nightly = _script("nightly_sim")
    rows = [{"id": "penale_x", "checks": [], "thesis_counts": {"offered": 3, "defence_lines": 1,
                                                               "defence_gate_dropped": 2, "ranked_candidates": 1,
                                                               "main_is_fallback": False},
             "defence_gate": {"dropped": ["55 c.p.", "83 c.p."]}}]
    text = "\n".join(nightly.counts_lines(rows))
    assert "| 1 | 2 | 1 |" in text and "penale_x: 55 c.p., 83 c.p." in text
