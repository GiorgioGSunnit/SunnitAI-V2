"""v78b: the PO's review of the first v78 answers -- synthetic tests per point, no database, no model.

1. the passage picker quotes principles, not mentions: a quotable sentence states a rule, holds the graded element's
   nouns, names no procedural step; the citing abstract shares the article's rubrica nouns or the ruling is dropped for
   it; Sez. 7 and the ordinanze last
2. a ruling is cited only for its article and on its graded element; §7's objections are reviewed (an article outside the
   answer's scope removes its sentence: 103, 105, 113 c.p.p.)
3. a reported counterparty claim never makes an element «non risulta» (582's «cagiona una lesione» excluded by «il
   difensore apprendeva che Tizio aveva negato…, sostenendo che…» flipped a victim's outcome to «sfavorevole»)
4. a victim's outcome is the querela's offence; records no fact touches leave the method (590-bis, 583-quater, 2044 c.c.)
"""

import pytest

from src.rag import issue_records, lawyer_method, rulings


@pytest.fixture(autouse=True)
def _model_env(monkeypatch):
    for name, value in (("LLM_API_KEY", "k"), ("OPENAI_API_KEY", "k"), ("EMBEDDING_PROVIDER", "openai"),
                        ("EMBEDDING_API_KEY", "k"), ("NEO4J_URI", "bolt://localhost:7687")):
        monkeypatch.setenv(name, value)


# --- 3. the regression: a reported claim never excludes an element ------------------------------------------------------------

CASE = ("Mevio, persona offesa, è stato colpito da Tizio con un pugno al volto il 3 marzo 2026, riportando una frattura "
        "del setto nasale guaribile in 30 giorni. Il difensore apprendeva che Tizio aveva negato di aver colpito Mevio, "
        "sostenendo che Mevio avesse iniziato la lite. Mevio vuole sporgere querela per lesioni personali.")
REPORTED = ("Il difensore apprendeva che Tizio aveva negato di aver colpito Mevio, sostenendo che Mevio avesse iniziato "
            "la lite.")


def test_the_reported_denial_is_an_allegation_and_its_speaker_the_counterparty():
    assert issue_records.certainty(REPORTED) == issue_records.ALLEGATO
    assert issue_records.certainty("Tizio ha sostenuto che la lite fosse iniziata da Mevio.") == issue_records.ALLEGATO
    assert issue_records.certainty("Mevio ha sostenuto le spese mediche.") == issue_records.ACCERTATO   # paid, not said
    from src.rag import interactive_flow as flow
    assert flow.attributed_to(REPORTED, ["Tizio", "Mevio"]) == "Tizio"


def _582(status_sentence_ids):
    sents = issue_records.sentences(CASE)
    by_id = {s["id"]: s for s in sents}
    reported = next(s for s in sents if s["text"].startswith("Il difensore apprendeva"))
    req = {"id": "R1", "text": "cagiona una lesione personale", "kind": "requisito",
           "facts_status": lawyer_method.NON_RISULTA,
           "link": {"supporting": [], "contradicting": [reported["id"]]}}
    issue_records.settle(req, by_id)
    return req, reported


def test_582_cagiona_una_lesione_is_never_escluso_on_a_reported_claim_and_the_victims_outcome_stays():
    req, reported = _582(())
    assert reported["certainty"] == issue_records.ALLEGATO
    assert req["status"] != lawyer_method.NON_RISULTA and req["status"] == lawyer_method.DA_PROVARE
    records = {"582 c.p.": {"side": "weapon", "charged": 1, "rubrica": "Lesione personale", "requisites": [
        {**req, "kind": "requisito"},
        {"id": "R2", "text": "malattia nel corpo o nella mente", "status": lawyer_method.PROVATO, "kind": "requisito"}]}}
    a = lawyer_method.assessment(records, {"role": "persona_offesa", "family": "penale"}, main="590 c.p.")
    assert a["article"] == "582 c.p." and a["esito"] != lawyer_method.SFAVOREVOLE
    # the same element narrated as fact the other way would exclude it -- the rule is about the reported claim only
    records["582 c.p."]["requisites"][0] = {**req, "status": lawyer_method.NON_RISULTA}
    assert lawyer_method.assessment(records, {"role": "persona_offesa"})["esito"] == lawyer_method.SFAVOREVOLE


# --- 4. a victim's charge, and the records no fact touches ---------------------------------------------------------------------

def _req(i, status=lawyer_method.DA_PROVARE, sup=(), con=()):
    return {"id": f"R{i}", "text": f"elemento {i}", "status": status, "kind": "requisito", "supporting": list(sup),
            "contradicting": list(con)}


def test_a_victims_outcome_is_the_querela_offence_not_the_main_topic():
    records = {"590 c.p.": {"side": "weapon", "requisites": [_req(1, lawyer_method.PROVATO, ["S1"])]},
               "582 c.p.": {"side": "weapon", "charged": 1, "requisites": [_req(2, lawyer_method.PROVATO, ["S1"]),
                                                                           _req(3)]}}
    a = lawyer_method.assessment(records, {"role": "persona_offesa"}, main="590 c.p.")
    assert a["article"] == "582 c.p." and "582" in a["why"]
    assert lawyer_method.assessment(records, {"role": "attore"}, main="590 c.p.")["article"] == "590 c.p."   # not a victim


def test_records_no_fact_touches_leave_the_method_unless_charged_threatening_or_named(monkeypatch):
    monkeypatch.setattr(issue_records, "decomposed_by_gate", lambda label, rubrica="": True)
    monkeypatch.setattr(issue_records, "decomposed", lambda rec: True)
    records = {"582 c.p.": {"side": "weapon", "charged": 1, "requisites": [_req(1)]},
               "590-bis c.p.": {"side": "weapon", "requisites": [_req(2), _req(3)]},
               "583-quater c.p.": {"side": "weapon", "requisites": [_req(4)]},
               "2044 c.c.": {"side": "weapon", "requisites": [_req(5)]},
               "52 c.p.": {"side": "weapon", "requisites": [_req(6, lawyer_method.DA_PROVARE, ["S2"])]},
               "610 c.p.": {"side": "weapon", "decomposed_for": "nominata dall'utente", "requisites": [_req(7)]},
               "612 c.p.": {"side": "threat", "requisites": [_req(8)]}}
    log = {}
    kept = lawyer_method.method_records(records, log)
    assert set(kept) == {"582 c.p.", "52 c.p.", "610 c.p.", "612 c.p."}
    assert set(log["no_fact"]) == {"590-bis c.p.", "583-quater c.p.", "2044 c.c."}


# --- 1. the passage picker ----------------------------------------------------------------------------------------------------

LESIONE = {"id": "R1", "text": "cagiona una lesione personale dalla quale deriva una malattia"}


@pytest.mark.parametrize("sentence", [
    "Con un terzo motivo di ricorso, si lamenta che la lesione non sussisterebbe perché la malattia era lieve.",
    "L'addebito mosso all'imputata è quello di aver cagionato una lesione personale con una malattia di 20 giorni.",
    "La Corte territoriale ha ritenuto sussistente la lesione personale e la malattia (dep. 12/03/2025, Rv. 28812).",
    "L'imputato lamenta che la lesione sarebbe esclusa dalla malattia preesistente.",
])
def test_mentions_and_procedural_sentences_are_never_quoted(sentence):
    assert not rulings.quotable(sentence, LESIONE)


def test_a_rule_on_the_element_is_quotable_and_one_on_another_element_is_not():
    rule = ("Ai fini della configurabilità del delitto di lesione personale, la malattia sussiste in presenza di "
            "un'alterazione funzionale dell'organismo, anche non definitiva.")
    assert rulings.quotable(rule, LESIONE)
    assert not rulings.quotable("Il dolo generico è sufficiente e non è necessaria la premeditazione.", LESIONE)
    assert not rulings.quotable("La lesione personale fu subita da Caio nel corso della lite.", LESIONE)   # no rule


def test_an_abstract_off_the_rubrica_drops_the_ruling_for_the_article():
    assert rulings.on_rubrica("Lesioni personali: la malattia richiede un'alterazione funzionale.", "Lesione personale")
    assert not rulings.on_rubrica("Maltrattamento di animali: lesione all'animale da compagnia.", "Lesione personale")
    assert rulings.on_rubrica("qualunque testo", "")


class Session:
    def __init__(self, rows, texts):
        self.rows, self.text_of = rows, texts

    def run(self, query, **kw):
        rows = self.rows if query == rulings.FETCH_QUERY else [
            {"section": i, "plain_text": self.text_of.get(i, "")} for i in kw["ids"]]

        class R:
            def data(self_inner):
                return rows
        return R()


PRINCIPLE = ("In tema di lesione personale, la malattia sussiste quando vi sia un'alterazione funzionale "
             "dell'organismo, anche modesta.")
MOTIVO = "Con il primo motivo di ricorso il ricorrente lamenta che la lesione personale e la malattia non sussistono."


def _row(i, section="50", act="S", abstract="Lesione personale: malattia e alterazione funzionale", when="2025-03-01"):
    return {"ruling": f"R{i}", "file_name": f"snpen@s{section}@a2025@n{i:05d}@t{act}.clean.pdf", "name": "",
            "document_number": "", "date_enacted": when, "linked": True, "section": f"S{i}", "abstract": abstract}


def _ctx():
    return {"records": {"582 c.p.": {"side": "weapon", "charged": 1, "rubrica": "Lesione personale",
                                     "requisites": [{**LESIONE, "status": "da provare", "kind": "requisito"}]}},
            "candidates": [{"id": "T1", "contesta": ["R1"]}], "case": "Mevio colpito da Tizio."}


ROWS = [{"_article": ("582", "c.p."), "d": {"id": "CP"}, "s": {"name": "582"}}]


def test_the_build_drops_the_off_rubrica_ruling_and_ranks_sez_7_and_ordinanze_last():
    rows = [_row(1, section="70", when="2025-12-01"), _row(2, act="O", when="2025-11-01"), _row(3, when="2025-01-01"),
            _row(4, abstract="Maltrattamento di animali: lesione al cane")]
    session = Session(rows, {f"S{i}": f"{MOTIVO} {PRINCIPLE}" for i in range(1, 5)})
    verdicts = {f"P{i}": {"frase": f"P{i}.1", "elemento": "R1", "pertinenza": 2, "verso": "a favore"} for i in range(1, 4)}
    record = rulings.build(_ctx(), ROWS, session, call=lambda m, t: verdicts)
    art = record["log"]["articles"]["582 c.p."]
    assert art["off_rubrica"] == 1 and "R4" not in [r["id"] for r in record["kept"]]
    assert [r["id"] for r in record["kept"]] == ["R3", "R1", "R2"]      # the plain sentenza first, though older
    assert all(r["passage"] == PRINCIPLE for r in record["kept"])       # the motivo sentence was never a candidate


def test_a_passage_the_call_puts_on_an_element_whose_nouns_it_lacks_is_not_quoted():
    ctx = _ctx()
    ctx["records"]["582 c.p."]["requisites"].append({"id": "R2", "text": "dolo generico di ledere",
                                                     "status": "da provare", "kind": "requisito"})
    ctx["candidates"] = [{"id": "T1", "contesta": ["R1", "R2"]}]
    session = Session([_row(1)], {"S1": PRINCIPLE})
    record = rulings.build(ctx, ROWS, session,
                           call=lambda m, t: {"P1": {"frase": "P1.1", "elemento": "R2", "pertinenza": 2,
                                                     "verso": "a favore"}})
    assert record["kept"][0]["passage"] == "" and record["kept"][0]["element_id"] == ""


# --- 2. citations: the ruling's article and element, the articles' scope in §7 --------------------------------------------------

ALLOWED = [{"id": "R1", "label": "Cass. pen., Sez. 5, n. 3/2025 (01.03.2025)", "massima": PRINCIPLE,
            "article": "582 c.p.", "element": LESIONE["text"]},
           {"id": "R9", "label": "Cass. pen., Sez. 3, n. 9/2025 (data n.d.)", "massima": "", "article": "582 c.p.",
            "element": ""}]


def test_a_ruling_is_cited_only_on_its_article_and_its_graded_element():
    on = ("La malattia derivante dalla lesione personale è integrata anche da un'alterazione modesta (Cass. pen., "
          "Sez. 5, n. 3/2025).")
    assert lawyer_method.rulings_scope_out(on, ALLOWED) == on
    other_article = "L'omicidio preterintenzionale (art. 584 c.p.) è escluso, come afferma Cass. pen., Sez. 5, n. 3/2025."
    log = []
    assert lawyer_method.rulings_scope_out(other_article, ALLOWED, log) == "" and "584 c.p." in log[0]
    off_element = "Il dolo è generico (Cass. pen., Sez. 5, n. 3/2025), e la condotta resta punibile."
    assert lawyer_method.rulings_scope_out(off_element, ALLOWED) == "Il dolo è generico, e la condotta resta punibile."
    no_passage = "La lesione personale e la malattia sono provate, come dice Cass. pen., Sez. 3, n. 9/2025."
    assert lawyer_method.rulings_scope_out(no_passage, ALLOWED) == ""


def test_the_objections_invented_articles_leave_with_their_sentence():
    scope = lawyer_method.article_scope({"records": {"582 c.p.": {}}, "case": "querela per lesioni (art. 582 c.p.)"},
                                        ["Procedibilità a querela: art. 336 c.p.p."])
    text = ("La difesa eccepirà la nullità della notifica ex artt. 103, 105 e 113 c.p.p. La lesione resta provata "
            "(art. 582 c.p.).\nSul termine: art. 336 c.p.p.")
    log = []
    out = lawyer_method.articles_out(text, scope, log)
    assert out == "La lesione resta provata (art. 582 c.p.).\nSul termine: art. 336 c.p.p." and "103 c.p.p." in log[0]
    assert lawyer_method.articles_out("Nulla da togliere.\n\nSecondo paragrafo.", scope) == \
        "Nulla da togliere.\n\nSecondo paragrafo."            # untouched text keeps its paragraphs


def test_compose_reviews_the_objection_itself(monkeypatch):
    """§7: an objection whose sentences all cite articles outside the scope is removed, its answer with it."""
    seen = {}

    def objections_section(plan, fallback=()):
        seen["obiezioni"] = plan.get("obiezioni")
        return "§7"
    monkeypatch.setattr(lawyer_method, "objections_section", objections_section)
    monkeypatch.setattr(lawyer_method, "final_check", lambda *a, **k: [])
    records = {"582 c.p.": {"side": "weapon", "charged": 1, "rubrica": "Lesione personale",
                            "requisites": [{**LESIONE, "status": "provato", "kind": "requisito", "supporting": ["S1"]}]}}
    ctx = {"records": records, "assessment": {"article": "582 c.p.", "esito": "incerto", "grado": "medio",
                                              "obiettivo": "esito pieno", "why": "x"},
           "posture": {"role": "persona_offesa", "family": "penale"}, "case": "Lesioni (art. 582 c.p.).",
           "rulings": ALLOWED, "facts": {}, "riti": [], "objections_fallback": []}
    plan = {"soluzioni": [{"tesi": "Querela per lesioni", "pro": "", "contro": "", "realismo": "medio",
                           "argomento": "", "contesta": [], "poggia_su": ["R1"], "candidate": "T1", "kind": "weapon"}],
            "obiezioni": [{"obiezione": "Nullità ex artt. 103, 105 e 113 c.p.p.", "risposta": "Infondata."},
                          {"obiezione": "La lesione personale non sussiste per l'art. 582 c.p.",
                           "risposta": "La malattia derivante dalla lesione personale è integrata (Cass. pen., Sez. 5, "
                                       "n. 3/2025)."}]}
    log = {}
    lawyer_method.compose([("", "")], ctx, plan, log=log)
    assert [o["obiezione"] for o in seen["obiezioni"]] == ["La lesione personale non sussiste per l'art. 582 c.p."]
    assert "n. 3/2025" in seen["obiezioni"][0]["risposta"]
    assert any("103 c.p.p." in x for x in log["articles_out_of_scope"])
