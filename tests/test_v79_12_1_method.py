"""v79.12.1 -- defence records, negated elements, rubriche, pertinence; synthetic records and sentences only:

1. a scriminante is never written with the charge template: a defence record (52-55, 59, 384, 131-bis …) all integrato
   -> the outcome is favourable; its open elements «da provare dalla difesa», never «per l'accusa»; «Rischio residuo»
   never names a conviction under the defence article (DEFENCE_WORDS: a separate wording set); a defence record is
   never a charged one
2. negated elements: an element phrased as an absence («non vi è desistenza», «non ha autorizzato», «senza il
   consenso») is integrato only by a fact stating that absence; a fact stating the presence of the thing, or silence,
   leaves it «da provare»; counter negated_integrated_by_presence (must be 0)
3. the rubrica of a general-part article is its own heading from the store, never the topic it is filed under (52 and
   55 c.p. under «Omicidio colposo», 61 c.p. under «Furto»)
4. pertinence before realismo (row 16): the main line first by pertinence -- the first read's institution, the more
   specific norm over the general -- then realismo and proven count; a more pertinent subordinate line is promoted,
   logged promoted_by_pertinence; counter main_outside_institution
"""

import re

import pytest


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    for name, value in (("LLM_API_KEY", "k"), ("OPENAI_API_KEY", "k"), ("EMBEDDING_PROVIDER", "openai"),
                        ("EMBEDDING_API_KEY", "k"), ("NEO4J_URI", "bolt://localhost:7687"), ("NEO4J_USER", "n"),
                        ("NEO4J_PASSWORD", "x")):
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("THESIS_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("STATE_DIR", str(tmp_path / "state"))


# --- 1. a defence record (synthetic 52 c.p.) ----------------------------------------------------------------------------

def _req(i, text, status="provato", sup=("S1",)):
    return {"id": f"R{i}", "comma": 1, "kind": "requisito", "text": text, "status": status,
            "supporting": list(sup) if status == "provato" else [], "missing": "" if status == "provato" else text}


def _records(defence=("provato", "provato")):
    from src.rag import lawyer_method as M
    recs = {"575 c.p.": {"rubrica": "Omicidio", "side": "threat", "charged": 1, "decomposed_for": "x",
                         "requisites": [_req(1, "cagiona la morte di un uomo")]},
            "52 c.p.": {"rubrica": "Difesa legittima", "decomposed_for": "difesa di parte generale",
                        "requisites": [_req(5, "la necessità di difendere un diritto proprio contro il pericolo "
                                               "attuale di un'offesa ingiusta", defence[0]),
                                       _req(6, "la difesa sia proporzionata all'offesa", defence[1])]}}
    return M.method_records(recs)


def _posture():
    from src.rag import posture
    return {"rito": "penale", "family": "penale", "role": "imputato", "stage": "indagini",
            "entry": posture.entry_for("penale", "imputato", "indagini")}


def _ctx(records):
    from src.rag import lawyer_method as M
    a = M.assessment(records, _posture(), client="Tizio")
    return {"records": records, "circumstances": {}, "posture": _posture(), "case": "Tizio colpisce Caio.",
            "assessment": a, "institutions": [], "client": "Tizio", "facts": {"mancanti": []}}


def _scriminante_main(records):
    from src.rag import lawyer_method as M
    ctx = _ctx(records)
    ctx["candidates"] = M.thesis_candidates(ctx)
    sc = next(c for c in ctx["candidates"] if c["kind"] == "scriminante")
    plan = M.clean_method({"valutazioni": {c["id"]: {"realismo": "alto" if c is sc else "basso"}
                                           for c in ctx["candidates"]}}, ctx["candidates"])
    sols, main, sub = M.selection(ctx, plan)
    assert sols[main].get("candidate") == sc["id"]
    return ctx, sols, main, sub


def _conclusions(records):
    from src.rag import lawyer_method as M
    ctx, sols, main, sub = _scriminante_main(records)
    log = {}
    a = M.settle_objective(M.forecast(ctx, sols[main], ctx["assessment"], log), sols, records)
    a = M.certainty_of(a, records)
    return a, M.conclusions_from_selection(a, sols, main, sub, ctx, "Tizio"), log


def _risk(text):
    return next((ln for ln in text.split("\n") if ln.startswith("**Rischio residuo**")), "")


def test_a_52_record_all_integrato_is_a_favourable_outcome_and_its_risk_names_no_conviction_under_52():
    records = _records()
    a, text, log = _conclusions(records)
    assert log["forecast"]["now"] == "52 c.p." and log["forecast"]["defence"] is True
    assert a["esito"] == "favorevole" and a["article"] == "52 c.p." and a["defence"] is True
    assert "scriminante" in a["why"] and "non è punibile" in a["why"]
    assert "Esito probabile**: favorevole a Tizio" in text
    risk = _risk(text)
    assert risk and "presupposti dell'art. 52 c.p." in risk
    assert not re.search(r"condanna per l['’]art\. 52", text)
    assert "per l'accusa" not in text and "575 c.p." in risk          # the charge it answers, never a 52 conviction


def test_a_52_record_with_an_open_element_is_da_provare_dalla_difesa_never_per_l_accusa():
    from src.rag import lawyer_method as M
    records = _records(("provato", "da provare"))
    a, text, _ = _conclusions(records)
    assert a["esito"] == "incerto" and "da provare dalla difesa" in a["why"]
    assert "per l'accusa" not in text and "accusa provi" not in text
    assert "che la difesa non provi gli elementi R6 dell'art. 52 c.p." in _risk(text)
    assert "da parte della difesa" in text
    assert not re.search(r"condanna per l['’]art\. 52", text)
    # the missing fact's line says who proves it
    facts = M.facts_split([], records, True)
    line = next(x for x in facts["mancanti"] if x["elemento"] == "R6")
    assert "da provare dalla difesa" in line["cambierebbe"]
    # the assessment read directly on the defence article is the defence's, never the charge template's
    direct = M.assessment(records, _posture(), client="Tizio", article="52 c.p.")
    assert direct["defence"] and "per l'accusa" not in direct["why"] and direct["esito"] != "sfavorevole"


def test_a_52_record_with_an_element_excluded_fails_without_a_conviction_under_52():
    records = _records(("provato", "non risulta"))
    a = __import__("src.rag.lawyer_method", fromlist=["x"]).defence_assessment(records, "52 c.p.", "Tizio", "575 c.p.")
    assert a["esito"] == "sfavorevole" and "non regge" in a["why"]
    from src.rag import lawyer_method as M
    depends, risk = M.defence_risk(a, [])
    assert "condanna" not in risk and "art. 52 c.p." in risk and "575 c.p." in risk


def test_a_defence_record_is_never_charged_and_its_objections_are_its_own():
    from src.rag import lawyer_method as M
    records = _records(("provato", "da provare"))
    records["52 c.p."]["charged"] = 2
    assert M._charged(records) == ["575 c.p."]
    out = M.objections_fallback(records, {"article": "52 c.p."}, True, {"S1": {"id": "S1", "text": "Caio aggredisce "
                                                                                                 "Tizio con un coltello."}})
    joined = " ".join(o["obiezione"] + " " + o["risposta"] for o in out)
    assert "dalla difesa" in joined and "spetta all'accusa" not in joined
    assert M.defence_record("52 c.p. co. 2") and M.defence_record("384 c.p.") and M.defence_record("131-bis c.p.")
    assert M.defence_record("59 c.p.") and not M.defence_record("575 c.p.") and not M.defence_record("52 c.c.")


# --- 2. negated elements -------------------------------------------------------------------------------------------------

def _settled(text, story, sup=("S1",), facts="da provare"):
    from src.rag import issue_records
    sents = issue_records.sentences(story)
    by_id = {s["id"]: s for s in sents}
    req = {"kind": "requisito", "text": text, "facts_status": facts,
           "link": {"supporting": list(sup), "contradicting": []}}
    issue_records.settle(req, by_id)
    return req, by_id


@pytest.mark.parametrize("text, absent, present", [
    ("non vi è desistenza volontaria", "Tizio non ha desistito e ha continuato a colpire.",
     "Tizio ha desistito spontaneamente dall'azione."),
    ("il titolare non ha autorizzato l'accesso al sistema", "Il titolare non aveva mai dato alcuna autorizzazione.",
     "Il titolare aveva autorizzato Caio ad accedere al sistema."),
    ("l'ingresso avviene senza il consenso dell'avente diritto", "Tizio è entrato senza il suo consenso.",
     "Caio aveva dato il consenso all'ingresso di Tizio."),
])
def test_a_negated_element_is_integrato_only_by_a_fact_stating_the_absence(text, absent, present):
    from src.rag import issue_records
    req, by_id = _settled(text, absent)
    assert req["status"] == "provato" and req["negated"] is True
    req, by_id = _settled(text, present)
    assert req["status"] == "da provare" and req["negated_presence"] == "S1"
    assert issue_records.negated_counters({"x": {"requisites": [req]}}, by_id)["negated_integrated_by_presence"] == 0


def test_silence_leaves_a_negated_element_da_provare_and_the_counter_sees_a_wrong_one():
    from src.rag import issue_records
    # the fact extraction says «provato», the trigger sentence shares the element's nouns but states no absence
    req, by_id = _settled("non vi è desistenza volontaria", "Tizio è fuggito dopo il fatto.", sup=(), facts="provato")
    assert req["status"] == "da provare"
    story = "Tizio ha desistito spontaneamente dall'azione."
    by_id = {s["id"]: s for s in issue_records.sentences(story)}
    wrong = {"kind": "requisito", "text": "non vi è desistenza volontaria", "status": "provato", "supporting": ["S1"]}
    got = issue_records.negated_counters({"x": {"requisites": [wrong]}}, by_id)
    assert got == {"negated_integrated_by_presence": 1, "negated_elements": 1}


def test_a_prohibition_or_a_positive_element_is_no_negated_element():
    from src.rag import issue_records
    assert issue_records.negated_element("il conduttore non può sublocare la cosa") is None
    assert issue_records.negated_element("cagiona la morte di un uomo") is None
    assert issue_records.negated_element("per colpa, senza dolo") is None
    req, _ = _settled("cagiona la morte di un uomo", "Tizio cagiona la morte di Caio con un coltello.")
    assert req["status"] == "provato" and "negated" not in req


def test_the_grading_counters_carry_negated_integrated_by_presence():
    from src.rag import interactive_flow, issue_records
    sents = issue_records.sentences("Tizio non ha desistito e ha continuato a colpire.")
    state = {"sentences": sents, "records": {"56 c.p.": {"requisites": [
        {"kind": "requisito", "text": "non vi è desistenza volontaria", "status": "provato", "supporting": ["S1"]}]}}}
    assert interactive_flow.grading_counters(state)["negated_integrated_by_presence"] == 0


# --- 3. the rubrica of general-part articles -------------------------------------------------------------------------------

def _row(n, rubrica=""):
    return {"_article": (n, "c.p."), "_grade": 2, "s": {"name": n, **({"titolo_articolo": rubrica} if rubrica else {})},
            "d": {"id": "CP"}}


@pytest.mark.parametrize("topic, anchor, anchor_rubrica, general, store", [
    ("Omicidio colposo", "589", "Omicidio colposo", "52", "Difesa legittima"),
    ("Omicidio colposo", "589", "Omicidio colposo", "55", "Eccesso colposo"),
    ("Furto", "624", "Furto", "61", "Circostanze aggravanti comuni"),
])
def test_a_general_part_article_keeps_its_own_rubrica_never_its_topic(topic, anchor, anchor_rubrica, general, store):
    from src.rag import issue_records
    t = {"title": topic, "anchor": f"{anchor} c.p.", "rubrica": anchor_rubrica,
         "articles": [f"{anchor} c.p.", f"{general} c.p."], "matched": [f"{anchor} c.p.", f"{general} c.p."],
         "planned": 2}
    rows = [_row(anchor, anchor_rubrica), _row(general)]          # the general article's row has no heading
    recs = issue_records.build([t], rows, {})
    assert recs[f"{anchor} c.p."]["rubrica"] == anchor_rubrica
    assert recs[f"{general} c.p."]["rubrica"] == ""               # never the topic's «{topic}»
    recs = issue_records.build([t], rows, {}, rubriche={f"{general} c.p.": store})
    assert recs[f"{general} c.p."]["rubrica"] == store
    rows = [_row(anchor, anchor_rubrica), _row(general, store)]   # the row's own heading first
    recs = issue_records.build([t], rows, {}, rubriche={f"{general} c.p.": "altro"})
    assert recs[f"{general} c.p."]["rubrica"] == store != topic


# --- 4. pertinence before realismo (row 16) ---------------------------------------------------------------------------------

def _civil(recs, case, institutions=()):
    from src.rag import posture
    p = {"rito": "civile", "family": "civile", "role": "attore", "stage": "pre_filing",
         "entry": posture.entry_for("civile", "attore", "pre_filing")}
    return {"records": recs, "circumstances": {}, "posture": p, "case": case, "assessment": {},
            "institutions": list(institutions)}


def _crec(rubrica, *reqs, **extra):
    return {"rubrica": rubrica, "side": "weapon", "requisites": list(reqs), **extra}


def test_a_more_pertinent_subordinate_line_is_promoted_over_a_more_realistic_general_one():
    from src.rag import lawyer_method as M
    case = "Il cane di Beta ha morso Tizio al parco; Tizio chiede il risarcimento."
    recs = {"2043 c.c.": _crec("Risarcimento per fatto illecito", _req(1, "un fatto doloso o colposo")),
            "2052 c.c.": _crec("Danno cagionato da animali", _req(2, "l'animale in custodia del convenuto",
                                                                   "da provare"))}
    ctx = _civil(recs, case, [{"name": "danno cagionato da animali", "rubricated": "Danno cagionato da animali",
                               "claim": "Tizio chiede il risarcimento a Beta"}])
    ctx["candidates"] = M.thesis_candidates(ctx)
    by_art = {c["article"]: c for c in ctx["candidates"] if c["kind"] == "claim"}
    assert set(by_art) >= {"2043 c.c.", "2052 c.c."}
    plan = M.clean_method({"valutazioni": {c["id"]: {"realismo": "alto" if c is by_art["2043 c.c."] else "medio"}
                                           for c in ctx["candidates"]}}, ctx["candidates"])
    log = []
    import copy
    sols, main = M.coherent_solutions(copy.deepcopy(plan), recs, {}, (), log, ctx["posture"], {})
    assert sols[main]["candidate"] == by_art["2043 c.c."]["id"]           # realismo alone: the general norm
    sols, main, sub = M.selection(ctx, plan)
    assert sols[main]["candidate"] == by_art["2052 c.c."]["id"]           # pertinence first: the institution's
    assert sols[main]["promoted_by_pertinence"] == {"over": by_art["2043 c.c."]["id"], "from": "subordinata"}
    counts = M.selection_counts(ctx, plan, {})
    assert counts["promoted_by_pertinence"] == 1 and counts["promoted_from"] == "subordinata"
    assert counts["main_outside_institution"] is False


def test_a_line_outside_the_first_read_s_institution_never_leads_over_one_inside_it():
    from src.rag import lawyer_method as M
    case = "Tizio ha prestato la sua auto a Caio, che non la restituisce; Tizio chiede la restituzione."
    recs = {"1809 c.c.": _crec("Restituzione", _req(1, "la scadenza del termine convenuto", "da provare")),
            "1218 c.c.": _crec("Responsabilità del debitore", _req(2, "l'inadempimento della prestazione"))}
    inst = [{"name": "comodato", "rubricated": "Comodato", "claim": "Tizio chiede la restituzione dell'auto"}]
    ctx = _civil(recs, case, inst)
    ctx["candidates"] = M.thesis_candidates(ctx)
    claims = {c["article"]: c for c in ctx["candidates"] if c["kind"] == "claim"}
    assert set(claims) == {"1809 c.c.", "1218 c.c."} and ctx["claim_modules"] == ["comodato"]
    assert M.inside_institution("1809 c.c.", ctx) is True and M.inside_institution("1218 c.c.", ctx) is False
    plan = M.clean_method({"valutazioni": {c["id"]: {"realismo": "alto" if c is claims["1218 c.c."] else "basso"}
                                           for c in ctx["candidates"]}}, ctx["candidates"])
    counts = M.selection_counts(ctx, plan, {})
    assert counts["chosen"] == claims["1809 c.c."]["id"] and counts["promoted_by_pertinence"] == 1
    assert counts["main_outside_institution"] is False


def test_pertinence_judges_only_the_client_s_own_norms_and_needs_an_institution_or_a_module():
    from src.rag import lawyer_method as M
    assert M.line_pertinence({"kind": "contest_open", "article": "575 c.p."}, {"candidates": []}) is None
    assert M.line_pertinence({"kind": "claim", "article": "2043 c.c."}, None) is None
    ctx = {"records": {}, "institutions": [], "candidates": []}
    assert M.inside_institution("2043 c.c.", ctx) is None
    assert M.specificity("2043 c.c.", ctx) == 2 and M.specificity("1490 c.c.", ctx) == 1
    ctx["records"] = {"1490 c.c. co. 2": {"comma_of": "1490 c.c."}}
    assert M.specificity("1490 c.c. co. 2", ctx) == 0
