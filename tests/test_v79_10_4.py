"""v79.10.4 (Block D: the civil rule found and then lost), on synthetic sentences, rows and records only:

D1. time conditions: an element stating a term («entro dieci anni», «entro due anni dalla consegna», «entro sessanta
    giorni dalla scoperta») graded against the facts' durations (stated spans, dated events) -- inside «provato»
    (integrato), beyond «non risulta» (escluso), no duration «da provare», never «non risulta» while a duration is
    inside; the text path and a schema «termine» alike; the counters termini_graded and termine_excluded_inside
D2. the grade cap keeps the anchors: what the similar rulings point to and the anchor's own section, whatever the rank
D3. a norm the counterparty violated is never «contraddetto» on the client's record; the client's own duty still is
D4. fragments: a category, a confiscation or medicinal clause, a modal permission -- fragment_why and the
    fragments_in_tables check
"""

import os

import pytest

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("OPENAI_API_KEY", "k"), ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

from src.rag import interactive_flow as flow  # noqa: E402
from src.rag import issue_records as I  # noqa: E402
from src.rag import posture, rule_quote  # noqa: E402

PROVATO, DA_PROVARE, NON_RISULTA = flow.PROVATO, flow.DA_PROVARE, flow.NON_RISULTA


def _settled(text, case, sup=(), con=(), facts=DA_PROVARE, **extra):
    sents = I.sentences(case)
    by_id = {s["id"]: s for s in sents}
    req = {"kind": "requisito", "text": text, "facts_status": facts,
           "link": {"supporting": list(sup), "contradicting": list(con)}, **extra}
    I.settle(req, by_id)
    return req


# --- D1 ------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("text,n,unit,anchor", [
    ("l'azione si prescrive entro dieci anni", 10, "ann", ""),
    ("l'azione va proposta entro due anni dalla consegna", 2, "ann", "conse"),
    ("la denunzia va fatta entro sessanta giorni dalla scoperta", 60, "giorn", "scope"),
    ("entro il termine di trenta giorni dalla data della notifica", 30, "giorn", "notif"),
    ("se la rovina si manifesta nel corso di dieci anni dal compimento", 10, "ann", "compi"),
    ("entro un anno", 1, "ann", ""),
])
def test_the_term_an_element_states(text, n, unit, anchor):
    term = I.term_of(text)
    assert (term["n"], term["unit"], term["anchor"]) == (n, unit, anchor)


def test_no_term_without_a_duration():
    assert I.term_of("il venditore è tenuto a garantire la cosa") is None
    assert I.term_of("entro i limiti del valore") is None


def test_the_spans_the_facts_state():
    assert [(x["bound"], round(x["days"])) for x in I.spans_of("Le crepe sono comparse oltre sei anni prima.")] == \
        [("lower", 2192)]
    assert [x["bound"] for x in I.spans_of("Il vizio è stato scoperto tre mesi fa.")] == ["exact"]
    assert I.spans_of("Tizio, un bambino di dieci anni, gioca in cortile.") == []      # an age is no span
    assert len(I.dates_of("Consegna il 3 marzo 2019, causa nel 2022 (L. 92/2012).")) == 2


def test_a_span_inside_the_term_integrates_and_never_excludes():
    text = "il diritto si esercita entro dieci anni"
    req = _settled(text, "Il fatto risale a oltre sei anni prima della domanda.", sup=["S1"])
    assert req["status"] == PROVATO and req["termine"]["verdict"] == I.INSIDE
    assert req["supporting"] == ["S1"] and req["why"].startswith("termine rispettato")
    # a contradicting sentence of the link never excludes it while a duration is inside
    req = _settled(text, "Il fatto risale a oltre sei anni prima della domanda.", con=["S1"])
    assert req["status"] == PROVATO and "S1" not in req["contradicting"]


def test_dated_events_from_the_anchor_beyond_the_term_exclude():
    text = "l'azione va proposta entro due anni dalla consegna"
    case = "La consegna dell'immobile è avvenuta il 3 marzo 2019. Il 5 aprile 2022 l'acquirente ha agito in giudizio."
    req = _settled(text, case, sup=["S2"])
    assert req["status"] == NON_RISULTA and req["termine"]["verdict"] == I.BEYOND
    assert "S2" in req["contradicting"] and "S2" not in req["supporting"]
    inside = case.replace("2022", "2020")
    req = _settled(text, inside, sup=["S2"])      # v79.10.7: the act's sentence linked -- its date measures the term
    assert req["status"] == PROVATO and req["termine"]["span"] == "dal 03/03/2019 al 05/04/2020"
    # v79.10.7 (the user's ruling on D1): the act's date not found, a later date never measures it -- «da provare»
    assert _settled(text, inside)["status"] == DA_PROVARE


def test_an_inside_duration_wins_over_a_beyond_one():
    text = "la denunzia va fatta entro sessanta giorni dalla scoperta"
    case = "Il vizio è stato scoperto oltre un anno fa. La scoperta del vizio è stata denunciata dopo venti giorni."
    assert _settled(text, case)["status"] == PROVATO


def test_no_duration_leaves_it_to_prove_even_with_a_supporting_sentence():
    text = "l'azione va proposta entro due anni dalla consegna"
    req = _settled(text, "La consegna è avvenuta. L'acquirente ha agito in giudizio.", sup=["S1"])
    assert req["status"] == DA_PROVARE and req["why"] == I.TERMINE_OPEN_WHY
    # the computed terms keep a «provato» they decided
    assert _settled(text, "La consegna è avvenuta.", sup=["S1"], facts=PROVATO)["status"] == PROVATO


def test_only_the_anchor_s_facts_and_certain_facts_count():
    text = "la denunzia va fatta entro sessanta giorni dalla scoperta"
    assert _settled(text, "Tizio lavora presso Caio da oltre sei anni.")["status"] == DA_PROVARE    # not the anchor's
    alleged = _settled(text, "Caio sostiene che la scoperta del vizio risale a tre anni fa.")
    assert alleged["status"] == DA_PROVARE                                                       # an allegazione


def test_the_schema_kind_selects_and_the_text_selects_otherwise():
    text = "l'azione va proposta entro due anni dalla consegna"
    assert I.element_rule({"kind": "requisito", "text": text}) == ("termine", "regex")
    assert I.element_rule({"kind": "requisito", "text": text, "schema_kind": "termine"}) == ("termine", "schema")
    assert I.element_rule({"kind": "requisito", "text": text, "schema_kind": "condotta"})[0] != "termine"
    # a term phrased as a duty is graded on its durations
    duty = "il compratore deve denunziare i vizi entro otto giorni dalla scoperta"
    assert I.element_rule({"kind": "requisito", "text": duty}) == ("termine", "regex")
    req = _settled(duty, "Il compratore ha scoperto i vizi il 2 maggio 2024 e li ha denunciati il 6 maggio 2024.",
                   schema_kind="termine")
    assert req["status"] == PROVATO and req["grading_rule"]["rule"] == "termine"


def test_a_procedibilita_term_is_graded_too():
    req = {"kind": "procedibilita", "text": "la querela è proposta entro tre mesi dalla notizia del fatto",
           "facts_status": DA_PROVARE, "link": {"supporting": [], "contradicting": []}}
    sents = I.sentences("La notizia del fatto è giunta il 10 gennaio 2025. La querela è stata presentata il 20 giugno "
                        "2025.")
    I.settle(req, {s["id"]: s for s in sents})
    assert req["status"] == NON_RISULTA


def test_the_termine_counters():
    sents = I.sentences("Il fatto risale a oltre sei anni prima della domanda.")
    by_id = {s["id"]: s for s in sents}
    ok = _settled("entro dieci anni", "Il fatto risale a oltre sei anni prima della domanda.")
    wrong = {"kind": "requisito", "text": "entro dieci anni", "status": NON_RISULTA, "contradicting": ["S1"]}
    got = I.termine_counters({"a": {"requisites": [ok, wrong, {"kind": "requisito", "text": "cagiona una lesione",
                                                               "status": DA_PROVARE}]}}, by_id)
    assert got == {"termini_graded": {"integrato": 1, "escluso": 1, "da provare": 0}, "termine_excluded_inside": 1}
    state = {"records": {"a": {"requisites": [ok]}}, "sentences": sents}
    counters = flow.grading_counters(state)
    assert counters["termine_excluded_inside"] == 0 and counters["termini_graded"]["integrato"] == 1


def test_an_unrelated_later_date_never_measures_the_term():
    text = "la denunzia va fatta entro sessanta giorni dalla scoperta"
    case = ("Il 1 marzo 2024 il committente ha scoperto i vizi dell'opera. Il 10 marzo 2024 il committente ha denunciato "
            "i vizi all'appaltatore. Il 5 dicembre 2024 l'appaltatore ha risposto con una lettera.")
    for sup in (["S2"], []):
        req = _settled(text, case, sup=sup)
        assert req["status"] == PROVATO and req["termine"]["span"] == "dal 01/03/2024 al 10/03/2024"
    case = ("Il 2 maggio 2024 il compratore ha scoperto i vizi. Il 6 maggio 2024 ha inviato la denuncia. Il 9 gennaio "
            "2025 il venditore ha risposto rifiutando.")
    assert _settled("il compratore denunzia i vizi entro otto giorni dalla scoperta", case)["status"] == PROVATO
    # the act's date not found: only a verdict every later date agrees on
    case = "Il vizio è stato scoperto il 1 marzo 2024. Il 10 marzo 2024 è arrivata una lettera. Il 5 dicembre 2024 un'altra."
    assert _settled("il diritto si esercita entro sessanta giorni dalla scoperta", case)["status"] == DA_PROVARE


def test_a_state_s_length_or_an_unrelated_span_never_grades_a_bare_term():
    req = _settled("l'azione è proposta entro un anno", "La società è in attività da quindici anni. Il cliente ha "
                   "proposto l'azione.", sup=["S2"], facts=PROVATO)
    assert req["status"] == PROVATO and req["termine"]["verdict"] == ""
    req = _settled("il diritto è esercitato entro un anno", "Mario lavora nell'azienda da dieci anni.")
    assert req["status"] == DA_PROVARE


def test_a_negated_act_is_integrated_by_the_late_act():
    late = _settled("il compratore non denunzia i vizi entro otto giorni dalla scoperta",
                    "Il compratore ha scoperto i vizi e li ha denunciati dopo tre mesi dalla scoperta.", sup=["S1"])
    assert late["status"] == PROVATO and late["why"].startswith("termine superato")
    on_time = _settled("il compratore decade dalla garanzia se non denunzia i vizi entro otto giorni dalla scoperta",
                       "Il compratore ha scoperto i vizi e li ha denunciati dopo tre giorni dalla scoperta.", sup=["S1"])
    assert on_time["status"] == NON_RISULTA and on_time["why"].startswith("termine rispettato")
    assert I.term_of("la domanda è proponibile se non entro un anno")["negated"] is False
    assert I.term_of("il venditore che non ha consegnato la cosa deve agire entro un anno")["negated"] is False


def test_a_stipulated_deadline_is_no_time_elapsed():
    assert I.spans_of("Il contratto prevedeva il compimento dell'opera entro due anni.") == []
    assert [x["bound"] for x in I.spans_of("Il compratore ha denunciato i vizi entro tre giorni.")] == ["upper"]
    req = _settled("la rovina si manifesta entro dieci anni dal compimento",
                   "Il contratto prevedeva il compimento dell'opera entro due anni.")
    assert req["status"] == DA_PROVARE


def test_every_number_word_is_a_duration():
    assert [I._number(w) for w in ("tredici", "cinquanta", "settanta", "ottanta", "venticinque", "ventotto",
                                   "trentuno", "ventitré", "centoventi")] == [13, 50, 70, 80, 25, 28, 31, 23, 120]
    req = _settled("il vizio è denunziato entro sessanta giorni dalla scoperta",
                   "Mario ha scoperto il vizio cinquanta giorni fa e lo ha denunciato subito.")
    assert req["status"] == PROVATO
    assert round(I.spans_of("la scoperta risale a settanta giorni fa")[0]["days"]) == 70


# --- D2 ------------------------------------------------------------------------------------------------------------------

def _row(n, rank, **tags):
    return {"_article": (str(n), "c.c."), "_fused_rank": rank, **tags}


def test_the_cap_keeps_the_anchors_whatever_their_rank():
    rows = [_row(1000 + i, i) for i in range(1, 6)] + [_row(1495, 99)]
    capped, log = flow.grade_cap(rows, [], cap=3)
    assert "1495 c.c." in log["dropped"] and log["anchors_kept"] == []
    capped, log = flow.grade_cap(rows, [], cap=3, anchors=["1495 c.c."])
    kept = {r["_article"][0] for r in capped}
    assert "1495" in kept and log["anchors_kept"] == ["1495 c.c."] and log["kept"] == 3
    # an anchor the rank keeps anyway is not counted
    _, log = flow.grade_cap(rows, [], cap=3, anchors=["1001 c.c."])
    assert log["anchors_kept"] == []


def test_the_cap_anchors_are_the_similar_rulings_and_the_anchor_s_section():
    rows = [_row(1490, 1), _row(1495, 40), _row(2043, 41, _vote=True), _row(1218, 42, _map=True),
            _row(1176, 43, _vote_stats={"rulings": 3}), _row(1175, 44, _vote_stats={"rulings": 1}), _row(1337, 45)]
    topics = [{"anchor": "1490 c.c.", "articles": ["1490 c.c.", "1495 c.c.", "1497 c.c."], "planned": 3},
              {"anchor": "1337 c.c.", "articles": ["1337 c.c."], "planned": 1}]
    got = flow.cap_anchors(rows, topics, ["1490 c.c."])
    assert set(got) == {"2043 c.c.", "1218 c.c.", "1176 c.c.", "1490 c.c.", "1495 c.c."}   # 1497: no row; 1337 gated
    assert "1175 c.c." not in got


def test_the_flow_passes_the_anchors_to_the_cap_and_the_prefilter():
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent / "src" / "rag" / "interactive_flow.py").read_text(encoding="utf-8")
    assert "grade_cap(article_rows, keep_now, anchors=held)" in src
    assert "+ list(held)))}" in src
    state = {"grade_funnel": {"anchors_kept": ["1495 c.c."], "capped": []}}
    assert flow.case_counters(state)["cap_anchors_kept"] == 1


# --- D3 ------------------------------------------------------------------------------------------------------------------

WARRANTY = "il venditore è tenuto a garantire che la cosa venduta sia immune da vizi"
NO_SUBLET = "il conduttore non può sublocare la cosa locata senza il consenso del locatore"
PAY = "il compratore deve pagare il prezzo pattuito"


def test_who_must_satisfy_an_element():
    roles = I.role_stems(["acquirente, compratore"])
    assert I.binds({"kind": "requisito", "text": WARRANTY}, roles) == I.COUNTERPARTY_BINDS
    assert I.binds({"kind": "requisito", "text": PAY}, roles) == I.CLIENT_BINDS
    assert I.binds({"kind": "requisito", "text": NO_SUBLET}, I.role_stems(["locatore"])) == I.COUNTERPARTY_BINDS
    assert I.binds({"kind": "requisito", "text": "cagiona ad altri un danno ingiusto"}, roles) == ""     # descriptive
    assert I.binds({"kind": "requisito", "text": "entro due anni dalla consegna"}, roles) == ""          # a term (D1)
    assert I.role_stems(["cliente, attore"]) == set()


def _weapon(text, case, con, role_words, side="weapon", role="attore"):
    sents = I.sentences(case)
    by_id = {s["id"]: s for s in sents}
    req = {"id": "R1", "comma": 1, "kind": "requisito", "text": text, "facts_status": DA_PROVARE,
           "link": {"supporting": [], "contradicting": list(con)}}
    I.settle(req, by_id)
    rec = {"requisites": [req], "side": side}
    rec["verdict"] = I.verdict(rec)
    counts = I.client_bindings({"x": rec}, I.role_stems(role_words), by_id, role)
    return rec, req, counts, by_id


def test_the_counterparty_s_breach_is_never_a_contradiction_on_the_client_s_record():
    case = "Il conduttore ha sublocato l'appartamento a un terzo senza chiedere nulla."
    rec, req, counts, by_id = _weapon(NO_SUBLET, case, ["S1"], ["locatore"])
    assert req["binds"] == I.COUNTERPARTY_BINDS and req["status"] == PROVATO and req["counterparty_breach"]
    assert not I.contradicted(rec) and counts["counterparty_breaches"] == 1
    assert I.counterparty_contradicted({"x": rec}, by_id) == 0
    # an allegation against it leaves it open, never contradicted
    rec, req, _, _ = _weapon(NO_SUBLET, "Il locatore sostiene che il conduttore abbia sublocato.", ["S1"], ["locatore"])
    assert req["status"] == DA_PROVARE and not I.contradicted(rec)


def test_the_duty_itself_denied_still_excludes():
    rec, req, _, by_id = _weapon(WARRANTY, "Nessun contratto di vendita è stato concluso tra le parti.", ["S1"],
                                 ["compratore"])
    assert req["status"] == NON_RISULTA and I.contradicted(rec)
    assert I.counterparty_contradicted({"x": rec}, by_id) == 0


def test_the_client_s_own_duty_contradicted_stays_unmet():
    case = "Il compratore non ha pagato il prezzo pattuito alla scadenza."
    rec, req, counts, by_id = _weapon(PAY, case, ["S1"], ["compratore"])
    assert req["binds"] == I.CLIENT_BINDS and req["status"] == NON_RISULTA and I.contradicted(rec)
    assert I.polarity_counters({"x": rec}, by_id)["duty_polarity_errors"] == 0
    # off the client's record (a threat) no binding: v79.9's A2 conversion as before
    rec, req, _, _ = _weapon(PAY, case, ["S1"], ["compratore"], side="threat")
    assert "binds" not in req and req["status"] == PROVATO and req["duty_breach"]


def test_an_accused_client_s_weapons_get_no_binding():
    rec, req, counts, _ = _weapon(NO_SUBLET, "Il conduttore ha sublocato l'appartamento.", ["S1"], [],
                                  role="imputato")
    assert "binds" not in req and counts["counterparty_elements"] == 0


def test_mark_sides_binds_the_client_s_records_and_counts():
    sents = I.sentences("Il conduttore ha sublocato l'appartamento a un terzo senza chiedere nulla.")
    req = {"id": "R1", "comma": 1, "kind": "requisito", "text": NO_SUBLET, "facts_status": DA_PROVARE,
           "link": {"supporting": [], "contradicting": ["S1"]}}
    I.settle(req, {s["id"]: s for s in sents})
    assert req["status"] == NON_RISULTA                                    # before the sides: a contradiction
    state = {"records": {"1594 c.c.": {"requisites": [req], "rubrica": "Sublocazione"}}, "sentences": sents,
             "text": "Il conduttore ha sublocato.", "read": {"client": {"name": "Tizio", "role": "locatore"}}}
    posture.mark_sides(state, {"role": "attore", "family": "civile"})
    assert req["status"] == PROVATO and req["binds"] == I.COUNTERPARTY_BINDS
    assert state["block_counters"]["counterparty_breaches"] == 1
    assert state["block_counters"]["counterparty_contradicted"] == 0
    assert state["records"]["1594 c.c."]["verdict"] == I.APPLICABLE


def _client(role, name="Tizio"):
    return {"fact_signals": {"facts": {"client_side": f"{name} ({role})", "parties": [{"name": name, "role": role}]}}}


def _round_one(monkeypatch, text, case, analysis):
    rows = [{"_article": ("1490", "c.c."), "_grade": 2}]
    monkeypatch.setattr(I, "distinctive", lambda rows, analysis: {"1490 c.c.": {"comma": 1, "text": text,
                                                                                 "kind": "requisito"}})
    call = lambda topic, reqs, sents: {r["id"]: {"sostiene": [], "contraddice": ["S1"], "manca": ""}
                                       for _, r in reqs}
    return rows, I.demote_contradicted(rows, case, analysis, call=call)


def test_round_one_never_demotes_on_a_breach(monkeypatch):
    rows, log = _round_one(monkeypatch, WARRANTY, "La caldaia venduta presentava gravi difetti di funzionamento.",
                           _client("acquirente"))
    assert rows[0]["_grade"] == 2 and log["demoted"] == [] and log["kept_breach"] == ["1490 c.c."]
    rows, log = _round_one(monkeypatch, WARRANTY, "Nessun contratto di vendita è stato concluso tra le parti.",
                           _client("acquirente"))
    assert rows[0]["_grade"] == 1 and log["demoted"][0]["reason"] == I.DEMOTED_CONTRADICTED


def test_round_one_keeps_the_breach_only_of_the_counterparty_s_norm(monkeypatch):
    # the client's own duty contradicted, an accused client's norm, no role known: demoted as before
    case = "Il conduttore non ha pagato il canone da sei mesi."
    rows, log = _round_one(monkeypatch, "il conduttore deve pagare il canone", case, _client("conduttore"))
    assert rows[0]["_grade"] == 1 and not log.get("kept_breach")
    sober = "Il conducente era sobrio: l'alcoltest ha dato esito negativo."
    rows, log = _round_one(monkeypatch, "è vietato guidare in stato di ebbrezza alcolica", sober, {})
    assert rows[0]["_grade"] == 1 and not log.get("kept_breach")
    rows, log = _round_one(monkeypatch, "è vietato guidare in stato di ebbrezza alcolica", sober, _client("imputato"))
    assert rows[0]["_grade"] == 1 and not log.get("kept_breach")
    assert I.analysis_client_roles(_client("indagato")) == set()
    assert I.analysis_client_roles(_client("acquirente")) == {"acqui", "compr"}


def test_a_statutory_synonym_of_the_client_s_role_binds_the_client():
    assert I.binds({"kind": "requisito", "text": "il compratore è tenuto a pagare il prezzo"},
                   I.role_stems(["acquirente"])) == I.CLIENT_BINDS
    assert I.binds({"kind": "requisito", "text": "il conduttore deve pagare il canone"},
                   I.role_stems(["inquilino"])) == I.CLIENT_BINDS
    # no role known: who must satisfy it cannot be told -- no binding, the grading as before
    assert I.binds({"kind": "requisito", "text": PAY}, set()) == ""
    rec, req, counts, _ = _weapon(PAY, "Il compratore non ha pagato il prezzo pattuito.", ["S1"], ["acquirente"])
    assert req["binds"] == I.CLIENT_BINDS and req["status"] == NON_RISULTA and I.contradicted(rec)


# --- D4 ------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("text,kind", [
    ("I precettori e coloro che insegnano un mestiere o un'arte", "category"),
    ("un mestiere o un'arte", "category"),
    ("i genitori e i tutori", "category"),
    ("è sempre ordinata la confisca delle cose che servirono a commettere il reato", "confiscation"),
    ("sono confiscati i beni che ne costituiscono il profitto", "confiscation"),
    ("i medicinali compresi nella tabella dei medicinali, sezione A", "medicinal"),
    ("la misura può essere disposta dal giudice", "modal"),
    ("è consentito al conduttore il recesso", "modal"),
    ("ha facoltà di recedere dal contratto", "modal"),
    ("può chiedere la risoluzione", "modal"),                                 # v79.9 (D2), as before
])
def test_the_new_fragments(text, kind):
    assert rule_quote.fragment_kind(text) == kind
    assert rule_quote.fragment_why(text) == rule_quote.FRAGMENT_WHY[kind]


@pytest.mark.parametrize("text", [
    "con violenza o minaccia", "la morte di una persona", "cagiona ad alcuno una lesione personale", WARRANTY,
    NO_SUBLET, "chiunque somministra medicinali guasti", "entro due anni dalla consegna",
    "se la misura può essere disposta quando il fatto è grave", "non è consentito al conduttore sublocare la cosa",
    # a conduct naming confiscated goods or medicinals, an event that may follow, what the law allows: elements
    "sottrae cose sottoposte a sequestro o confiscate", "vende medicinali senza ricetta",
    "il farmacista vende medicinali senza ricetta medica", "detiene per il commercio medicinali guasti in farmacia",
    "detiene medicinali in quantità superiore a quella prevista dalla tabella",
    "somministra medicinali in dose diversa da quella prescritta dalla ricetta",
    "dal fatto può derivarne un pericolo per la pubblica incolumità", "il fatto è consentito dalla legge",
])
def test_elements_stay_elements(text):
    assert rule_quote.fragment_why(text) == ""


def test_a_d4_fragment_never_rejects_a_generated_schema():
    from src.rag import element_schemas as E
    text = ("Chiunque sottrae cose sottoposte a sequestro è punito con la reclusione fino a tre anni. È sempre ordinata la "
            "confisca delle cose che servirono a commettere il reato.")
    els = [{"id": eid, "testo": t, "kind": kind, "kind_note": None, "comma": "1", "branches": [], "connective": None,
            "parti": [], "grounding": "explicit", "fonte": {"comma": "1", "testo": t}, "spiegazione": None}
           for eid, t, kind in (("E1", "sottrae cose sottoposte a sequestro", "condotta"),
                                ("E2", "È sempre ordinata la confisca delle cose che servirono a commettere il reato",
                                 "quantificazione"))]
    schema = {"tipo": ["sanction"], "destinatario": [{"ruolo": "chiunque", "condizione": None}], "legittimato": [],
              "soggetto_passivo": [], "rapporto": "sottrazione", "elementi": els,
              "logic": {"op": "all_of", "args": [{"ref": "E1"}, {"ref": "E2"}]}, "eccezioni": [],
              "conseguenza": {"testo": "è punito con la reclusione fino a tre anni", "tipo": "pena", "comma": "1"},
              "commi_speciali": [], "collegati": []}
    assert E.check(schema, text)["fragments"] == []
    assert rule_quote.fragment_kind(els[1]["testo"]) == "confiscation"            # dropped alone by requisites()
    assert rule_quote.fragment_kind(els[1]["testo"], wider=False) == ""
    assert rule_quote.fragment_kind("può chiedere la risoluzione", wider=False) == "modal"     # v79.9's, as before


def test_the_fragments_in_tables_check_reads_the_new_kinds():
    state = {"records": {"2048 c.c.": {"requisites": [
        {"kind": "requisito", "text": "I precettori e coloro che insegnano un mestiere o un'arte"},
        {"kind": "requisito", "text": "il danno cagionato dal fatto illecito dei loro allievi"}]},
        "73 d.P.R. 309/1990": {"requisites": [{"kind": "requisito", "text": "è sempre ordinata la confisca"}]}}}
    got = flow.surviving_fragments(state)
    assert len(got) == 2 and any("categoria" in x for x in got) and any("confisca" in x for x in got)
    assert flow.grading_counters(state)["fragments_in_tables"] == 2


def test_the_client_s_role_words_scoped_to_one_client():
    read = {"client": {"name": "Tizio", "role": "locatore"},
            "clients": [{"name": "Tizio", "role": "locatore"}, {"name": "Caia", "role": "fideiussore"}]}
    assert set(posture.client_role_words({"read": read})) == {"locatore", "fideiussore"}
    assert posture.client_role_words({"read": read, "client_scope": {"name": "Caia", "others": ["Tizio"]}}) == \
        ["fideiussore"]
    # no sentences given: the element is marked, never settled on nothing
    req = {"id": "R1", "kind": "requisito", "text": NO_SUBLET, "status": NON_RISULTA, "contradicting": ["S1"]}
    I.client_bindings({"x": {"requisites": [req], "side": "weapon"}}, I.role_stems(["locatore"]), {})
    assert req["binds"] == I.COUNTERPARTY_BINDS and req["status"] == NON_RISULTA
