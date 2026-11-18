"""v79.12.1 review -- the reviewers' findings on items 1, 2 and 4; synthetic records and sentences only:

1. causation (40, 41) and fault (43 c.p.) are no defence records: their elements integrated establish the charge, so a
   fully proven 589 with proven causation or fault is never forecast «favorevole … la difesa regge»; only 41 co. 2 (a
   later cause sufficient by itself) is one
2. a negated element: the absence must govern the thing («non ha autorizzato», «senza il suo consenso»); a fact stating
   the presence with an unrelated «non» / «senza» nearby leaves it «da provare» and the counter sees it; an auxiliary or
   a light verb is never the thing; a fact stating the absence in other words (pagare for adempiere, riconsegnare for
   restituire) keeps it «provato»
4. among equally pertinent lines, realismo (then the proven count) decides which one is promoted, never the pool's order
"""

import pytest


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    for name, value in (("LLM_API_KEY", "k"), ("OPENAI_API_KEY", "k"), ("EMBEDDING_PROVIDER", "openai"),
                        ("EMBEDDING_API_KEY", "k"), ("NEO4J_URI", "bolt://localhost:7687"), ("NEO4J_USER", "n"),
                        ("NEO4J_PASSWORD", "x")):
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("THESIS_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("STATE_DIR", str(tmp_path / "state"))


def _req(i, text, status="provato", sup=("S1",)):
    return {"id": f"R{i}", "comma": 1, "kind": "requisito", "text": text, "status": status,
            "supporting": list(sup) if status == "provato" else [], "missing": "" if status == "provato" else text}


# --- 1. causation and fault are no defence records ------------------------------------------------------------------------

def _posture():
    from src.rag import posture
    return {"rito": "penale", "family": "penale", "role": "imputato", "stage": "indagini",
            "entry": posture.entry_for("penale", "imputato", "indagini")}


@pytest.mark.parametrize("article, element", [
    ("40 c.p.", "l'evento dannoso è conseguenza dell'azione od omissione dell'imputato"),
    ("43 c.p.", "l'evento si verifica a causa di negligenza, imprudenza o imperizia"),
])
def test_proven_causation_or_fault_is_never_a_favourable_defence(article, element):
    from src.rag import lawyer_method as M
    recs = M.method_records({
        "589 c.p.": {"rubrica": "Omicidio colposo", "side": "threat", "charged": 1, "decomposed_for": "x",
                     "requisites": [_req(1, "cagiona per colpa la morte di una persona")]},
        article: {"rubrica": "", "decomposed_for": "difesa di parte generale", "requisites": [_req(5, element)]}})
    assert not M.defence_record(article, recs[article])
    direct = M.assessment(recs, _posture(), client="Tizio", article=article)
    assert not direct.get("defence") and direct["esito"] != "favorevole"
    ctx = {"records": recs, "circumstances": {}, "posture": _posture(), "case": "Tizio investe Caio.",
           "assessment": M.assessment(recs, _posture(), client="Tizio"), "institutions": [], "client": "Tizio",
           "facts": {"mancanti": []}}
    ctx["candidates"] = M.thesis_candidates(ctx)
    plan = M.clean_method({"valutazioni": {c["id"]: {"realismo": "alto" if c.get("article") == article else "basso"}
                                           for c in ctx["candidates"]}}, ctx["candidates"])
    sols, main, sub = M.selection(ctx, plan)
    log = {}
    a = M.forecast(ctx, sols[main], ctx["assessment"], log)
    assert a["esito"] != "favorevole" and "la difesa sull'art." not in a["why"]
    assert not (log.get("forecast") or {}).get("defence")
    assert M._charged({**recs, article: {**recs[article], "charged": 2}}) == ["589 c.p.", article]


def test_only_the_defence_comma_of_art_41_is_a_defence_record():
    from src.rag import lawyer_method as M
    assert not M.defence_record("41 c.p.") and not M.defence_record("41 c.p. co. 1")
    assert M.defence_record("41 c.p. co. 2", {"comma_of": "41 c.p."})
    recs = {"41 c.p. co. 2": {"comma_of": "41 c.p.", "requisites": [
        _req(7, "una causa sopravvenuta da sola sufficiente a determinare l'evento")]}}
    a = M.defence_assessment(M.method_records(recs), "41 c.p. co. 2", "Tizio", "589 c.p.")
    assert a["esito"] == "favorevole" and a["obiettivo"] == M.PIENO and "nesso causale" in a["why"]
    assert M.defence_record("52 c.p.") and M.defence_record("55 c.p.") and M.defence_record("131-bis c.p.")


# --- 2. the absence must govern the thing -------------------------------------------------------------------------------------

def _settled(text, story):
    from src.rag import issue_records
    by_id = {s["id"]: s for s in issue_records.sentences(story)}
    req = {"kind": "requisito", "text": text, "facts_status": "da provare",
           "link": {"supporting": ["S1"], "contradicting": []}}
    issue_records.settle(req, by_id)
    return req, issue_records.negated_counters({"x": {"requisites": [req]}}, by_id)


AUTH = "il titolare non ha autorizzato l'accesso al sistema"
CONSENT = "l'ingresso avviene senza il consenso dell'avente diritto"


@pytest.mark.parametrize("text, presence", [
    (AUTH, "Il titolare ha autorizzato l'accesso e non ha sollevato obiezioni."),
    (AUTH, "Il titolare, che non conosceva Tizio, lo ha autorizzato ad entrare."),
    (AUTH, "Il titolare aveva autorizzato l'accesso e non lo aveva mai revocato."),
    (CONSENT, "Caio ha dato il consenso senza alcuna riserva."),
    (CONSENT, "Caio aveva prestato il consenso, non opponendosi all'ingresso di Tizio."),
    ("non vi è desistenza volontaria", "Tizio ha desistito spontaneamente, senza che nessuno lo fermasse."),
    ("non vi è desistenza volontaria", "Tizio ha desistito non appena ha visto la polizia."),
    ("si introduce nel sistema senza essere autorizzato",
     "Tizio, autorizzato dal titolare, è entrato senza essere notato."),
    ("il titolare non ha prestato il consenso", "Il titolare ha prestato il consenso, non senza esitazioni."),
    ("il titolare non ha prestato il consenso", "Il titolare non ha rifiutato il consenso."),
])
def test_a_presence_fact_with_a_stray_negation_leaves_the_negated_element_da_provare(text, presence):
    req, counters = _settled(text, presence)
    assert req["status"] == "da provare" and req["negated_presence"] == "S1"
    assert counters == {"negated_integrated_by_presence": 0, "negated_elements": 1}


def test_the_counter_sees_a_presence_fact_with_a_stray_negation():
    from src.rag import issue_records
    by_id = {s["id"]: s for s in issue_records.sentences("Caio ha dato il consenso senza alcuna riserva.")}
    wrong = {"kind": "requisito", "text": CONSENT, "status": "provato", "supporting": ["S1"]}
    assert issue_records.negated_counters({"x": {"requisites": [wrong]}}, by_id)["negated_integrated_by_presence"] == 1


def test_an_auxiliary_or_a_light_verb_is_never_the_thing():
    from src.rag import issue_records
    assert "autoriz" in issue_records.negated_element("si introduce nel sistema senza essere autorizzato")
    got = issue_records.negated_element("il titolare non ha prestato il consenso")
    assert "consen" in got and not any(p.startswith("prest") for p in got)


@pytest.mark.parametrize("text, absence", [
    ("il debitore non ha adempiuto l'obbligazione di pagamento", "Il conduttore non ha pagato i canoni di marzo e aprile."),
    ("il comodatario non ha restituito il bene alla scadenza", "Alla scadenza Caio si è rifiutato di riconsegnare l'auto."),
    ("si introduce nel sistema senza essere autorizzato",
     "Tizio è entrato nel sistema senza essere stato autorizzato da nessuno."),
    ("il titolare non ha prestato il consenso", "Il consenso del titolare non è mai stato prestato."),
    ("il titolare non ha prestato il consenso", "Il titolare non ha mai prestato alcun consenso."),
    (CONSENT, "Tizio è entrato in assenza di qualsiasi consenso."),
    (CONSENT, "Caio non aveva dato nessun consenso all'ingresso."),
])
def test_a_fact_stating_the_absence_keeps_the_negated_element_provato(text, absence):
    req, counters = _settled(text, absence)
    assert req["status"] == "provato" and req["negated"] is True
    assert counters == {"negated_integrated_by_presence": 0, "negated_elements": 1}


# --- 4. realismo breaks a tie in pertinence -------------------------------------------------------------------------------------

def test_among_equally_pertinent_lines_the_more_realistic_one_is_promoted():
    from src.rag import lawyer_method as M, posture
    p = {"rito": "civile", "family": "civile", "role": "attore", "stage": "pre_filing",
         "entry": posture.entry_for("civile", "attore", "pre_filing")}
    recs = {"1809 c.c.": {"rubrica": "Restituzione", "side": "weapon",
                          "requisites": [_req(1, "la scadenza del termine convenuto", "da provare")]},
            "1810 c.c.": {"rubrica": "Comodato senza determinazione di durata", "side": "weapon",
                          "requisites": [_req(3, "il comodatario richiesto dal comodante", "da provare")]},
            "1218 c.c.": {"rubrica": "Responsabilità del debitore", "side": "weapon",
                          "requisites": [_req(2, "l'inadempimento della prestazione")]}}
    ctx = {"records": recs, "circumstances": {}, "posture": p, "assessment": {},
           "case": "Tizio ha prestato la sua auto a Caio, che non la restituisce; Tizio chiede la restituzione.",
           "institutions": [{"name": "comodato", "rubricated": "Comodato",
                             "claim": "Tizio chiede la restituzione dell'auto"}]}
    ctx["candidates"] = M.thesis_candidates(ctx)
    claims = {c["article"]: c for c in ctx["candidates"] if c["kind"] == "claim"}
    assert set(claims) == {"1809 c.c.", "1810 c.c.", "1218 c.c."}
    assert M.line_pertinence({"candidate": claims["1809 c.c."]["id"]}, ctx) == \
        M.line_pertinence({"candidate": claims["1810 c.c."]["id"]}, ctx)
    rated = {"1218 c.c.": "alto", "1810 c.c.": "medio", "1809 c.c.": "basso"}
    plan = M.clean_method({"valutazioni": {c["id"]: {"realismo": rated.get(c.get("article"), "basso")}
                                           for c in ctx["candidates"]}}, ctx["candidates"])
    sols, main, sub = M.selection(ctx, plan)
    assert sols[main]["candidate"] == claims["1810 c.c."]["id"]
    assert sols[main]["promoted_by_pertinence"]["over"] == claims["1218 c.c."]["id"]
