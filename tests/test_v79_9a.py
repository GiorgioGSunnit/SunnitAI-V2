"""v79.9 block A -- grading: mental elements and polarity, on synthetic sentences and records only:

A1. a culpability or state-of-mind element is integrato only by a fact stating the state of mind or its recognised
    indicators (an admission, a warning ignored, a false statement knowingly made); a conduct fact leaves it «da provare»
A2. a duty-phrased element is graded on its breach: the breaching fact integrates it, never excludes it
A3. a riqualificazione only into an offence of the charge's own code, and only when the model's argument names the
    element that differs
A4. the counters: mental_integrated_by_conduct (0), duty_polarity_errors (0), riqualificazione_out_of_code (0)
"""

import pytest


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    for name, value in (("LLM_API_KEY", "k"), ("OPENAI_API_KEY", "k"), ("EMBEDDING_PROVIDER", "openai"),
                        ("EMBEDDING_API_KEY", "k"), ("NEO4J_URI", "bolt://localhost:7687"), ("NEO4J_USER", "n"),
                        ("NEO4J_PASSWORD", "x")):
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("THESIS_LOG_DIR", str(tmp_path / "logs"))


def _settled(text, sents, sup=(), con=(), facts="da provare"):
    from src.rag import issue_records
    by_id = {s["id"]: s for s in sents}
    req = {"kind": "requisito", "text": text, "facts_status": facts,
           "link": {"supporting": list(sup), "contradicting": list(con)}}
    issue_records.settle(req, by_id)
    return req


# --- A1 ---------------------------------------------------------------------------------------------------------------------

PUSH = "Tizio spinge Caio, che cade a terra e si frattura il polso."


def test_a_push_integrates_582_and_leaves_590s_per_colpa_to_prove():
    from src.rag import issue_records
    sents = issue_records.sentences(PUSH)
    assert _settled("cagiona ad alcuno una lesione personale", sents, ["S1"])["status"] == "provato"
    colpa = _settled("cagiona ad altri per colpa una lesione personale", sents, ["S1"])
    assert colpa["status"] == "da provare" and colpa["mental_conduct"] and colpa["why"] == issue_records.MENTAL_CONDUCT_WHY
    assert _settled("per colpa", sents, ["S1"], facts="provato")["status"] == "da provare"     # the facts' read too


@pytest.mark.parametrize("element", ["per colpa", "con dolo", "in mala fede", "in buona fede", "per negligenza",
                                     "per imprudenza", "con la consapevolezza dell'altruità", "con l'intenzione di nuocere",
                                     "un fatto doloso o colposo"])
@pytest.mark.parametrize("conduct", ["Tizio colpisce Caio con un pugno.", "Alfa vende a Beta un'auto usata.",
                                     "Tizio spinge Caio contro il muro."])
def test_a_conduct_fact_never_integrates_a_mental_element(element, conduct):
    from src.rag import issue_records
    sents = issue_records.sentences(conduct)
    got = _settled(element, sents, ["S1"])
    assert got["status"] == "da provare"
    assert _settled(element, sents, [], ["S1"])["status"] == "da provare"         # never «escluso» on a conduct either


@pytest.mark.parametrize("fact", [
    "Tizio ha ammesso di aver colpito Caio di proposito.",
    "Tizio era stato avvertito del pericolo e ha proseguito ignorando l'avviso.",
    "Alfa ha descritto falsamente l'auto come mai incidentata, pur sapendo dell'incidente.",
    "Tizio guidava distratto dal cellulare e passa col rosso.",
])
def test_a_fact_stating_the_state_of_mind_integrates_it(fact):
    from src.rag import issue_records
    sents = issue_records.sentences(fact)
    assert sents[0]["certainty"] == "accertato"
    assert _settled("per colpa o con dolo", sents, ["S1"])["status"] == "provato"


def test_a_fact_stating_the_contrary_state_of_mind_excludes_it():
    from src.rag import issue_records
    sents = issue_records.sentences("Tizio non sapeva che la cosa fosse altrui.")
    assert _settled("con la consapevolezza dell'altruità della cosa", sents, [], ["S1"])["status"] == "non risulta"


# --- A2 ---------------------------------------------------------------------------------------------------------------------

DEFECT = "Alfa vende a Beta una lavatrice che dopo una settimana smette di funzionare per un difetto del motore."


def test_1490s_duty_element_is_integrated_by_the_breach():
    from src.rag import issue_records
    sents = issue_records.sentences(DEFECT)
    for sup, con in (([], ["S1"]), (["S1"], [])):
        got = _settled("la cosa venduta sia immune da vizi che la rendano inidonea all'uso", sents, sup, con)
        assert got["status"] == "provato" and "S1" in got["supporting"] and "S1" not in got["contradicting"]


@pytest.mark.parametrize("element", ["il venditore deve consegnare la cosa", "il debitore è tenuto a eseguire i lavori",
                                     "la cosa sia conforme al contratto"])
def test_a_breached_duty_is_never_excluded(element):
    from src.rag import issue_records
    sents = issue_records.sentences("I lavori non sono stati eseguiti e la merce non è stata consegnata.")
    got = _settled(element, sents, [], ["S1"])
    assert got["status"] == "provato" and got["duty_breach"]


def test_a_fact_denying_the_duty_itself_still_excludes_it():
    from src.rag import issue_records
    sents = issue_records.sentences("Tra Alfa e Beta non è mai stato concluso alcun contratto.")
    assert _settled("il venditore deve consegnare la cosa", sents, [], ["S1"])["status"] == "non risulta"


def test_the_polarity_counters():
    from src.rag import interactive_flow as flow, issue_records
    sents = issue_records.sentences(PUSH + " " + DEFECT)
    by_id = {s["id"]: s for s in sents}
    recs = {"590 c.p.": {"requisites": [{"text": "per colpa", "status": "provato", "supporting": ["S1"]}]},
            "1490 c.c.": {"requisites": [{"text": "la cosa sia immune da vizi", "status": "non risulta",
                                          "contradicting": ["S2"]}]}}
    assert issue_records.polarity_counters(recs, by_id) == {"mental_integrated_by_conduct": 1, "duty_polarity_errors": 1}
    for rec in recs.values():
        for r in rec["requisites"]:
            r["link"] = {"supporting": r.get("supporting", []), "contradicting": r.get("contradicting", [])}
            r["kind"] = "requisito"
            issue_records.settle(r, by_id)
    got = flow.grading_counters({"sentences": sents, "records": recs})
    assert got["mental_integrated_by_conduct"] == 0 and got["duty_polarity_errors"] == 0


# --- A3 ---------------------------------------------------------------------------------------------------------------------

def _req(i, text, status="provato", **kw):
    return {"id": f"R{i}", "comma": 1, "kind": "requisito", "text": text, "status": status, **kw}


def _ctx(recs):
    from src.rag import posture
    p = {"rito": "penale", "family": "penale", "role": "imputato", "stage": "indagini",
         "entry": posture.entry_for("penale", "imputato", "indagini")}
    return {"records": recs, "circumstances": {}, "posture": p, "case": "Tizio è indagato.",
            "assessment": {"article": "640 c.p."}, "institutions": []}


def _recs(alt_label="646 c.p.", alt_rubrica="Appropriazione indebita"):
    return {"640 c.p.": {"rubrica": "Truffa", "side": "threat", "decomposed_for": "x", "requisites": [
                _req(1, "con artifizi o raggiri induce taluno in errore", distinctive=True),
                _req(2, "procura a sé un ingiusto profitto con altrui danno")]},
            alt_label: {"rubrica": alt_rubrica, "side": "weapon", "alternative": True, "decomposed_for": "x",
                        "requisites": [_req(3, "si appropria del denaro o della cosa mobile altrui"),
                                       _req(4, "procura a sé un ingiusto profitto")]}}


def test_a_riqualificazione_into_the_charges_code_names_what_differs():
    from src.rag import lawyer_method as M
    ctx = _ctx(_recs())
    cands = M.thesis_candidates(ctx)
    req = next(c for c in cands if c["kind"] == "requalify")
    assert req["article"] == "646 c.p." and req["charge"] == "640 c.p."
    assert req["differs"][0] == "con artifizi o raggiri induce taluno in errore"
    assert any("appropria" in x for x in req["differs"])
    assert "l'argomento deve nominare l'elemento che distingue" in M.candidates_block(cands)


def test_a_riqualificazione_out_of_the_charges_code_is_never_offered():
    from src.rag import lawyer_method as M
    ctx = _ctx(_recs("2043 c.c.", "Risarcimento per fatto illecito"))
    assert not any(c["kind"] == "requalify" for c in M.thesis_candidates(ctx))
    assert ctx["candidates_trace"]["requalify_out_of_code"] == ["2043 c.c."]


def test_an_argument_naming_no_differing_element_is_not_offered():
    from src.rag import lawyer_method as M
    ctx = _ctx(_recs())
    ctx["candidates"] = M.thesis_candidates(ctx)
    tid = next(c["id"] for c in ctx["candidates"] if c["kind"] == "requalify")
    vague = {"valutazioni": {c["id"]: {"realismo": "medio", "argomento": "Una diversa qualificazione è possibile."}
                             for c in ctx["candidates"]}}
    log = {}
    plan = M.clean_method(vague, ctx["candidates"], log)
    assert tid not in [t["candidate"] for t in plan["soluzioni"]] and log["requalify_unnamed"] == [tid]
    counts = M.selection_counts(ctx, plan, vague)
    assert counts["requalify_unnamed"] == 1 and counts["riqualificazione_out_of_code"] == 0
    named = {"valutazioni": {c["id"]: {"realismo": "medio", "argomento": "Mancano gli artifizi e i raggiri: Tizio si "
                                       "appropria del denaro che già possedeva."} for c in ctx["candidates"]}}
    plan = M.clean_method(named, ctx["candidates"])
    assert tid in [t["candidate"] for t in plan["soluzioni"]]
