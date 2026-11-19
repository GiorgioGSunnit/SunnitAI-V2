"""v79.12.7 (items 1, 3, 4, 5 of the brief), on synthetic sentences, records and ruling passages only:

1. a jurisprudential element (584: prevedibilità in concreto dell'evento morte) a note or a kept ruling passage states
   and the text does not enters the record inferred, «da provare», its source named; contesting it is sostenibile; a
   reviewed schema carrying it is used as is. On the synthetic slap-and-death facts the main line is that contest, the
   passage under it, the riqualificazione and the attenuanti subordinate. Counters jurisprudential_elements,
   jurisprudential_minoritaria (0), jurisprudential_main.
3. data/penal_lines.json's absorption: 584 absorbs 581/582; 589/590 beside 584 only as the riqualificazione
   alternative; 593 out where a soccorso is narrated, the soccorso a favourable element after the fact. Counter
   absorbed_as_autonomous (0), absorbed_titles, soccorso_post_fact.
4. a riqualificazione 584 -> 589/590 names the element that differs and the facts bearing on it; without facts it is
   not offered. Counters requalify_no_facts, requalify_without_facts (0).
5. a causation line on an ordinary consequence (a fall and a trauma after a slap) is capped at realismo basso unless
   the facts narrate a pathology, a medical error or an abnormal cause; a reply denying the causal link is replaced by
   one arguing the subjective element. Counters causation_capped, causation_cap_lifted, causation_over_basso (0),
   causation_replies_rewritten.
"""

import os

import pytest

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("OPENAI_API_KEY", "k"), ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

from src.rag import issue_records  # noqa: E402
from src.rag import lawyer_method as M  # noqa: E402
from src.rag import penal_lines as P  # noqa: E402
from src.rag import posture  # noqa: E402


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("THESIS_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("THESIS_DETERMINISTIC", "")
    from src.rag import plan_then_write
    monkeypatch.setattr(plan_then_write, "negation_review", lambda items, known, call=None: [])


IMPUTATO = {"rito": "penale", "family": "penale", "role": "imputato", "stage": "primo_grado",
            "entry": posture.entry_for("penale", "imputato", "primo_grado")}

# a synthetic case: a slap, a fall, a death -- and the help called at once
SLAP = ["Alfa, durante un diverbio in un parcheggio, colpisce Beta con uno schiaffo al volto.",
        "Beta perde l'equilibrio, cade all'indietro e batte la testa sul marciapiede.",
        "Beta muore due giorni dopo per il trauma cranico riportato nella caduta.",
        "Alfa chiama subito il 118 e attende l'ambulanza accanto a Beta."]
IMPULSE = "Alfa sostiene di avere reagito d'impulso, con una spinta, quando Beta lo ha afferrato per il bavero."
# a synthetic ruling passage on the prevedibilità in concreto (no real ruling)
PASSAGE = ("Ai fini della responsabilità per omicidio preterintenzionale occorre accertare la prevedibilità in "
           "concreto dell'evento morte, da valutare in relazione alle circostanze del fatto.")
ESTREMI = "Cass. pen., Sez. 5, n. 11111/2025 (03.03.2025)"


def _sents(texts):
    return [{"id": f"S{i}", "text": t, "source": "caso", "certainty": "accertato"} for i, t in enumerate(texts, 1)]


def _req(i, text, status="provato", **extra):
    return {"id": f"R{i}", "comma": 1, "kind": "requisito", "text": text, "status": status, **extra}


def _records(with_582=False, with_593=False, with_589=True, with_41=False):
    recs = {"584 c.p.": {"rubrica": "Omicidio preterintenzionale", "side": "threat", "charged": 1, "requisites": [
        _req(1, "con atti diretti a commettere uno dei delitti preveduti dagli articoli 581 e 582", supporting=["S1"]),
        _req(2, "cagiona la morte di un uomo", "da provare", supporting=["S3"])]}}
    if with_582:
        recs["582 c.p."] = {"rubrica": "Lesione personale", "side": "threat", "charged": 1, "requisites": [
            _req(3, "cagiona ad alcuno una lesione personale", supporting=["S2"])]}
    if with_589:
        recs["589 c.p."] = {"rubrica": "Omicidio colposo", "alternative": True, "requisites": [
            _req(4, "cagiona per colpa la morte di una persona", "da provare", supporting=["S3"])]}
    if with_593:
        recs["593 c.p."] = {"rubrica": "Omissione di soccorso", "side": "threat", "charged": 1, "requisites": [
            _req(5, "trovando una persona ferita omette di prestare l'assistenza occorrente", "da provare",
                 supporting=["S4"])]}
    if with_41:
        recs["41 c.p."] = {"rubrica": "Concorso di cause", "decomposed_for": issue_records.DEFENCE_WHY,
                           "requisites": [_req(6, "il concorso di cause preesistenti o simultanee o sopravvenute",
                                               supporting=["S2"])]}
    return recs


def _ctx(records, texts, rows=()):
    sents = _sents(texts)
    case = " ".join(texts)
    return M.prepare({"records": records, "sentences": sents}, IMPUTATO, list(rows), case, "Alfa", "", list(records),
                     [], [], [], [], link_call=lambda *a: {}, case_id_="v79126")


def _record_with_passage(passage=PASSAGE):
    return {"kept": [{"id": "g1", "estremi": ESTREMI, "article": "584 c.p.", "element_id": "R2",
                      "element": "cagiona la morte di un uomo", "passage": passage, "verso": "integrato"}],
            "orientations": [], "landmarks": {}}


def _valued(ctx, real="medio", **by_kind):
    def call(messages):
        return {"valutazioni": {c["id"]: {"pro": "p", "contro": "c", "argomento": (
            "la reazione impulsiva, una spinta senza la volontà di percuotere o di ledere"),
            "realismo": by_kind.get(c["kind"], real)} for c in ctx["candidates"]}}
    return call


# --- 1. jurisprudential elements ---------------------------------------------------------------------------------------

def test_a_kept_passage_adds_the_prevedibilita_inferred_da_provare_with_its_source():
    ctx = _ctx(_records(), SLAP)
    assert not any(r.get("jurisprudential") for r in ctx["records"]["584 c.p."]["requisites"])
    ctx["rulings_record"] = _record_with_passage()
    got = M.ruling_elements(ctx)
    assert [x["how"] for x in got] == ["added"]
    el = next(r for r in ctx["records"]["584 c.p."]["requisites"] if r.get("jurisprudential"))
    assert el["text"] == "prevedibilità in concreto dell'evento morte" and el["status"] == "da provare"
    assert el["inferred"] and el["source"] == ESTREMI and el["sources"][0]["passage"] == PASSAGE
    assert el["id"] not in {"R1", "R2", "R4"}
    # the assessment and §4 refreshed: the charge is no longer fully integrato, the element is in the table
    assert not ctx["assessment"].get("all_integrated")
    assert "prevedibilità in concreto dell'evento morte" in ctx["subsumption"]


def test_the_slap_and_death_main_line_is_the_prevedibilita_contest_with_the_passage_under_it():
    ctx = _ctx(_records(), SLAP + [IMPULSE])
    ctx["rulings_record"] = _record_with_passage()
    plan = M.method_plan(ctx, call=_valued(ctx))
    juris = next(c for c in ctx["candidates"] if c.get("jurisprudential"))
    assert juris["kind"] == "contest_open" and juris["label"] == M.SUSTAINABLE
    assert ESTREMI in juris["tesi"] and "prevedibilità in concreto" in juris["tesi"]
    sols, main, sub = M.selection(ctx, plan)
    assert sols[main].get("jurisprudential") and not sols[main].get("minoritaria")
    # containment -- the riqualificazione, the containment line -- subordinate
    kinds = {t.get("kind") for i, t in enumerate(sols) if i != main}
    assert "requalify" in kinds and "contenimento" in kinds
    assert sub is not None and sols[sub].get("kind") in ("requalify", "contenimento", "contest_open")
    assert all(t.get("main_demoted") for t in sols if t.get("kind") in ("requalify", "contenimento"))
    text = M.solutions_section(sols, main, plan, [], "nessuna")
    head, _, rest = text.partition("- Linea principale:")
    first = rest.split("\n- ")[0]
    assert "passaggio della motivazione «" in first and "prevedibilità in concreto dell'evento morte" in first
    assert ESTREMI in first
    counts = ctx["selection_counts"]
    assert counts["jurisprudential_elements"] == 1 and counts["jurisprudential_main"] == 1
    assert counts["jurisprudential_minoritaria"] == 0


def test_the_note_feeds_the_element_on_the_text_path_and_names_itself():
    note = ("Errori da evitare: per la preterintenzione la giurisprudenza richiede la prevedibilità in concreto "
            "dell'evento morte.")
    row = {"_article": ("584", "c.p."), "_note": {"nota": note, "numeri": {}}, "d": {"id": "CP"},
           "s": {"name": "584", "plain_text": "testo"}}
    ctx = _ctx(_records(), SLAP, rows=[row])
    el = next(r for r in ctx["records"]["584 c.p."]["requisites"] if r.get("jurisprudential"))
    assert el["source"] == P.SOURCE_NOTE and el["status"] == "da provare"
    assert ctx["jurisprudential"][0]["how"] == "added"
    # a kept passage on the same element extends its sources, never a second copy
    ctx["rulings_record"] = _record_with_passage()
    M.ruling_elements(ctx)
    same = [r for r in ctx["records"]["584 c.p."]["requisites"] if r.get("jurisprudential")]
    assert len(same) == 1 and [s["kind"] for s in same[0]["sources"]] == ["note", "ruling"]


def test_a_note_warning_against_prevedibilita_outside_colpa_adds_nothing():
    recs = {"584 c.p.": {"rubrica": "Omicidio preterintenzionale", "requisites": [_req(1, "cagiona la morte di un uomo")],
                         "note_di_inquadramento": "Usare la prevedibilità del rischio come argomento fuori dalla colpa."}}
    assert P.jurisprudential_elements(recs) == []


def test_contesting_the_jurisprudential_element_is_sustainable_even_where_the_rest_is_integrato():
    recs = {"584 c.p.": {"rubrica": "Omicidio preterintenzionale", "side": "threat", "requisites": [
        _req(1, "con atti diretti a commettere uno dei delitti preveduti dagli articoli 581 e 582"),
        _req(2, "cagiona la morte di un uomo"),
        _req(9, "prevedibilità in concreto dell'evento morte", "da provare", jurisprudential=True, inferred=True,
             source=ESTREMI, sources=[{"kind": "ruling", "source": ESTREMI, "estremi": ESTREMI, "passage": PASSAGE}])]}}
    ctx = {"records": recs, "circumstances": {}, "posture": IMPUTATO, "case": " ".join(SLAP),
           "assessment": M.assessment(recs, IMPUTATO), "client": "Alfa", "sentences": _sents(SLAP)}
    cands = M.thesis_candidates(ctx)
    juris = next(c for c in cands if c.get("jurisprudential"))
    assert juris["label"] == M.SUSTAINABLE and juris["contesta"] == ["R9"]
    integrated = [c for c in cands if c["kind"] == "contest_integrated"]
    assert integrated and all(c["label"] == M.MINORITY for c in integrated)


def test_a_reviewed_schema_carrying_the_element_is_used_as_is():
    el = _req(7, "prevedibilità in concreto dell'evento morte", "provato", split="schema", schema_grounding="inferred")
    recs = {"584 c.p.": {"rubrica": "Omicidio preterintenzionale", "schema": {"reviewed": True, "source": "reviewed"},
                         "requisites": [_req(1, "cagiona la morte di un uomo"), el]}}
    got = P.jurisprudential_elements(recs, [{"article": "584 c.p.", "estremi": ESTREMI, "passage": PASSAGE}],
                                     schemas_on=True)
    reqs = recs["584 c.p."]["requisites"]
    assert len(reqs) == 2 and got[0]["how"] == "tagged"
    assert el["jurisprudential"] and el["status"] == "da provare"
    assert [s["kind"] for s in el["sources"]] == ["schema", "ruling"]
    # schemas off: the same record's schema is not read as reviewed -- the text states it, nothing is added
    recs2 = {"584 c.p.": {"rubrica": "x", "requisites": [_req(1, "prevedibilità in concreto dell'evento morte")]}}
    assert P.jurisprudential_elements(recs2, [{"article": "584 c.p.", "estremi": ESTREMI, "passage": PASSAGE}]) == []


# --- 3. absorption ------------------------------------------------------------------------------------------------------

def test_584_absorbs_582_and_keeps_589_as_the_alternative_only():
    recs = _records(with_582=True)
    recs["589 c.p."] = {**recs["589 c.p."], "side": "threat", "charged": 1}
    recs["589 c.p."].pop("alternative")
    out, report = P.absorb(recs, _sents(SLAP))
    assert "582 c.p." not in out and "584 c.p." in out
    alt = out["589 c.p."]
    assert alt["alternative"] and alt["unprinted"] and "side" not in alt and "charged" not in alt
    assert recs["589 c.p."]["side"] == "threat"          # the input untouched
    assert {(x["article"], x["mode"]) for x in report["absorbed"]} == {("582 c.p.", "absorbed"),
                                                                        ("589 c.p.", "alternative_only")}


def test_582_is_not_absorbed_when_584_is_only_an_alternative():
    recs = {"582 c.p.": {"rubrica": "Lesione personale", "side": "threat", "requisites": [_req(1, "lesione")]},
            "584 c.p.": {"rubrica": "Omicidio preterintenzionale", "alternative": True, "requisites": [_req(2, "morte")]}}
    out, report = P.absorb(recs, [])
    assert "582 c.p." in out and not report["absorbed"]


def test_593_is_out_where_the_soccorso_is_narrated_and_becomes_a_favourable_element():
    ctx = _ctx(_records(with_582=True, with_593=True), SLAP)
    assert "593 c.p." not in ctx["records"] and "582 c.p." not in ctx["records"]
    assert ctx["post_fact"] and "118" in ctx["post_fact"][0]["text"]
    ctx["rulings_record"] = None
    plan = M.method_plan(ctx, call=_valued(ctx))
    assert not any(c.get("article") in ("593 c.p.", "582 c.p.") for c in ctx["candidates"])
    counts = ctx["selection_counts"]
    assert counts["absorbed_as_autonomous"] == 0 and counts["absorbed_titles"] == 3      # 582, 589 (alternative), 593
    assert counts["soccorso_post_fact"] == 1
    sols, main, _ = M.selection(ctx, plan)
    text = M.solutions_section(sols, main, plan, [], "nessuna", None, M.post_fact_lines(ctx))
    assert "Elemento favorevole successivo al fatto" in text and "art. 62-bis c.p." in text


def test_593_stays_where_the_soccorso_is_denied():
    texts = SLAP[:3] + ["Alfa si allontana senza chiamare i soccorsi."]
    ctx = _ctx(_records(with_593=True), texts)
    assert "593 c.p." in ctx["records"] and not ctx["post_fact"]


def test_absorbed_as_autonomous_counts_a_title_left_beside_its_absorber():
    recs = _records(with_582=True)
    ctx = {"records": recs, "candidates": [{"kind": "contest_open", "contesta": ["R3"], "poggia_su": []}],
           "sentences": _sents(SLAP)}
    assert P.absorbed_as_autonomous(ctx) == 3      # 582, the contest on its element, 589 printed beside 584
    out, _ = P.absorb(recs, _sents(SLAP))
    assert P.absorbed_as_autonomous({"records": out, "candidates": [
        {"kind": "requalify", "article": "589 c.p."}], "sentences": _sents(SLAP)}) == 0


# --- 4. riqualificazione through the differing element ------------------------------------------------------------------

def test_the_riqualificazione_names_the_element_and_the_facts():
    ctx = _ctx(_records(), SLAP + [IMPULSE])
    cands = M.thesis_candidates(ctx)
    req = next(c for c in cands if c["kind"] == "requalify")
    assert req["article"] == "589 c.p."
    assert "atti diretti a percuotere o a ledere" in req["tesi"] and "reazione impulsiva" in req["tesi"]
    assert "d'impulso, con una spinta" in req["tesi"] and req["facts"][0]["sid"] == "S5"
    assert req["differs"][:2] == ["atti diretti a percuotere o a ledere",
                                  "una spinta o una reazione impulsiva senza la volontà di percuotere o di ledere"]


def test_without_facts_on_the_element_the_riqualificazione_is_not_offered():
    ctx = _ctx(_records(), SLAP)
    cands = M.thesis_candidates(ctx)
    assert not any(c["kind"] == "requalify" for c in cands)
    assert ctx["candidates_trace"]["requalify_no_facts"] == ["589 c.p."]
    plan = M.method_plan(ctx, call=_valued(ctx))
    assert ctx["selection_counts"]["requalify_no_facts"] == 1
    assert ctx["selection_counts"]["requalify_without_facts"] == 0
    assert plan["soluzioni"]


def test_requalify_rule_reads_the_pairs_from_data():
    assert P.requalify_rule("584 c.p.", "590 c.p.")["id"] == "584_to_589_590"
    assert P.requalify_rule("582 c.p.", "590 c.p.") is None


# --- 5. art. 41 realism -------------------------------------------------------------------------------------------------

def test_the_cap_reads_an_ordinary_consequence_and_lifts_on_a_trigger():
    assert P.causation_cap(" ".join(SLAP))["capped"]
    lifted = P.causation_cap(" ".join(SLAP) + " Beta soffriva di una grave cardiopatia.")
    assert not lifted["capped"] and "patologia" in lifted["triggers"]
    assert not P.causation_cap(" ".join(SLAP) + " In ospedale un errore medico ritarda l'intervento.")["capped"]
    assert not P.causation_cap(" ".join(SLAP) + " L'ambulanza ha un incidente stradale durante il trasporto.")["capped"]
    assert not P.causation_cap("Alfa sottrae il portafoglio a Beta.")["capped"]      # no ordinary consequence read
    assert not P.causation_cap("Il cadavere di Beta è trovato in casa.")["capped"]    # a body is no fall


def test_an_art_41_defence_and_the_causal_contest_are_basso_on_the_slap_facts():
    ctx = _ctx(_records(with_41=True), SLAP)
    plan = M.method_plan(ctx, call=_valued(ctx, real="alto"))
    defence = next(c for c in ctx["candidates"] if c["kind"] == "defence")
    causal = next(c for c in ctx["candidates"] if c["kind"] == "contest_open" and c["contesta"] == ["R2"])
    assert defence["causation_capped"] and causal["causation_capped"]
    assert defence["code_argument"]["contro"].startswith("La caduta e il trauma sono conseguenze ordinarie")
    for c in (defence, causal):
        sol = next(t for t in plan["soluzioni"] if t["candidate"] == c["id"])
        assert sol["realismo"] == M.BASSO
    counts = ctx["selection_counts"]
    assert counts["causation_capped"] == 2 and counts["causation_over_basso"] == 0
    assert counts["causation_cap_lifted"] == 0


def test_a_pathology_in_the_facts_leaves_the_causation_line_uncapped():
    ctx = _ctx(_records(with_41=True), SLAP + ["Beta soffriva da anni di una cardiopatia non diagnosticata."])
    plan = M.method_plan(ctx, call=_valued(ctx, real="alto"))
    defence = next(c for c in ctx["candidates"] if c["kind"] == "defence")
    assert not defence["causation_capped"]
    assert next(t for t in plan["soluzioni"] if t["candidate"] == defence["id"])["realismo"] != M.BASSO
    assert ctx["selection_counts"]["causation_cap_lifted"] == 2


def test_a_reply_denying_the_causal_link_is_replaced_by_the_subjective_element():
    ctx = _ctx(_records(with_41=True), SLAP)
    plan = M.method_plan(ctx, call=_valued(ctx))
    objections = {"obiezioni": [
        {"obiezione": "La caduta è conseguenza ordinaria dello schiaffo.",
         "risposta": "La caduta interrompe il nesso causale tra lo schiaffo e la morte."},
        {"obiezione": "Lo schiaffo è provato.", "risposta": "Resta da provare la prevedibilità dell'evento."}]}
    out = M.objections_plan(ctx, plan, call=lambda messages: objections, review_call=lambda *a, **k: [])
    replies = [o["risposta"] for o in out["obiezioni"]]
    assert replies[0] == P.CAUSATION_REPLY + "."
    assert "elemento soggettivo" in replies[0] and replies[1].startswith("Resta da provare")
    assert ctx["selection_counts"]["causation_replies_rewritten"] == 1


def test_no_cap_no_rewrite():
    ctx = _ctx(_records(), ["Alfa sottrae il portafoglio a Beta."])
    ctx["candidates"] = M.thesis_candidates(ctx)
    objs = [{"obiezione": "x", "risposta": "Il nesso causale è interrotto."}]
    assert M.causation_replies(ctx, objs) == 0 and objs[0]["risposta"] == "Il nesso causale è interrotto."


def test_the_penal_counters_reach_the_case_counters():
    from src.rag import interactive_flow as flow
    keys = set(M.PENAL_COUNT_KEYS)
    out = flow.case_counters({"thesis_counts": {k: 1 for k in keys}})
    assert all(out[k] == 1 for k in keys)
