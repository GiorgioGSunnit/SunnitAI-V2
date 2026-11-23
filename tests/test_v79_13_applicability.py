"""v79.13.0 build A -- the applicability judgment (src/rag/applicability.py, data/applicability.json), the model faked:

1. the call: batches of at most 8 norms (balanced), bounded concurrency, thinking off and temperature 0 in the request,
   the facts labelled certo / allegazione with their ids, the schema's tipo / rapporto as context only (ELEMENT_SCHEMAS)
2. the verification: v79.15.0 (item 3) by ids -- a «si» cites the article's numbered periods (cita_norma), a «no» and
   a «dipende» the facts (cita_fatti), every id existing; no content beyond the cited periods, the facts and the
   reviewed note (fit_added_content), a dipende's condizione a grounded question, the elements it touches
3. the brief's synthetic cases: a rissa between two people, a loan with no instalments (1819 c.c., «rateale»), an
   advance payment (646 c.p., «altrui»), an invented condition
4. malformed replies, missing or unknown fact ids, failed calls, the deadline: the norm unjudged, never a crash

The article texts are statutory (public law); the facts are synthetic.
"""

import json
import threading
import time

import pytest

from src.rag import applicability as ap


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    for name, value in (("LLM_API_KEY", "k"), ("OPENAI_API_KEY", "k"), ("EMBEDDING_PROVIDER", "openai"),
                        ("EMBEDDING_API_KEY", "k"), ("NEO4J_URI", "bolt://localhost:7687"), ("NEO4J_USER", "n"),
                        ("NEO4J_PASSWORD", "x")):
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("ARTICLE_SCHEMAS_FILE", str(tmp_path / "schemas.jsonl"))
    monkeypatch.setenv("ARTICLE_SCHEMAS_REVIEWED_FILE", str(tmp_path / "reviewed.jsonl"))


ART_588 = ("1. Chiunque partecipa a una rissa è punito con la multa fino a euro 2.000.\n"
           "2. Se nella rissa taluno rimane ucciso, o riporta lesione personale, la pena, per il solo fatto della "
           "partecipazione alla rissa, è della reclusione da sei mesi a sei anni. La stessa pena si applica se "
           "l'uccisione, o la lesione personale, avviene immediatamente dopo la rissa e in conseguenza di essa.")
ART_1819 = ("Se è stata convenuta la restituzione rateale delle cose mutuate e il mutuatario non adempie l'obbligo del "
            "pagamento anche di una sola rata, il mutuante può chiedere, secondo le circostanze, l'immediata "
            "restituzione dell'intero.")
ART_646 = ("1. Chiunque, per procurare a sé o ad altri un ingiusto profitto, si appropria il denaro o la cosa mobile "
           "altrui di cui abbia, a qualsiasi titolo, il possesso, è punito, a querela della persona offesa, con la "
           "reclusione da due a cinque anni e con la multa da euro 1.000 a euro 3.000.\n"
           "2. Se il fatto è commesso su cose possedute a titolo di deposito necessario, la pena è aumentata.")

RISSA = {"article": "588 c.p.", "rubrica": "Rissa", "text": ART_588,
         "elements": [{"id": "E1", "testo": "partecipa a una rissa", "comma": "1"},
                      {"id": "E2", "testo": "taluno rimane ucciso o riporta lesione personale", "comma": "2"}]}
MUTUO = {"article": "1819 c.c.", "rubrica": "Restituzione rateale", "text": ART_1819}
APPROPRIAZIONE = {"article": "646 c.p.", "rubrica": "Appropriazione indebita", "text": ART_646}

FACTS_RISSA = [{"id": "S1", "text": "Tizio e Caio, due persone, si sono colpiti a vicenda davanti a un bar.",
                "certainty": "accertato"},
               {"id": "S2", "text": "Caio sostiene che Tizio lo abbia colpito per primo e di aver riportato una ferita.",
                "certainty": "allegato"}]
FACTS_MUTUO = [{"id": "S1", "text": "Tizio ha prestato a Caio 5.000 euro da restituire in un'unica soluzione entro "
                                    "il 30 giugno.", "certainty": "accertato"},
               {"id": "S2", "text": "Caio non ha restituito la somma alla scadenza.", "certainty": "accertato"}]
FACTS_ACCONTO = [{"id": "S1", "text": "Il cliente ha versato a Beta 3.000 euro come acconto sul prezzo dei lavori.",
                  "certainty": "accertato"},
                 {"id": "S2", "text": "Il cliente afferma che Beta non ha eseguito i lavori e trattiene l'acconto.",
                  "certainty": "allegato"}]


def _item(article, fit, perche, ambito="fattispecie", condizione="", cf=None, cn=None):
    """v79.15.0 (item 3): a «no» or a «dipende» cites its facts by id (cf: «cita_fatti»), a «si» the article's numbered
    periods (cn: «cita_norma», applicability.article_ids); None: no such key."""
    out = {"articolo": article, "fit": fit, "perche": perche, "condizione": condizione, "ambito": ambito}
    if cf is not None:
        out["cita_fatti"] = cf
    if cn is not None:
        out["cita_norma"] = cn
    return out


RISSA_1 = ["A588cp.1"]          # 588 c.p.: comma 1, «Chiunque partecipa a una rissa è punito …»


def _reply(*items):
    return json.dumps({"giudizi": list(items)}, ensure_ascii=False)


def _fake(reply_for):
    """A model stand-in: reply_for(norm labels in the human message) -> reply text; every request recorded."""
    seen = []

    def call(messages, kwargs):
        human = messages[-1].content
        labels = [line.split(": ", 1)[1] for line in human.splitlines() if line.startswith("ARTICOLO: ")]
        seen.append({"messages": messages, "kwargs": kwargs, "labels": labels})
        return reply_for(labels)
    call.seen = seen
    return call


def _one(norm, facts, item, role="imputato"):
    res = ap.judge([norm], facts, role, call=_fake(lambda labels: _reply(item)))
    return res, res["judgments"][0]


# --- 3. the brief's cases -------------------------------------------------------------------------------------------

def test_rissa_si_on_the_quoted_text_passes():
    res, rec = _one(RISSA, FACTS_RISSA, _item(
        "588 c.p.", "si", "Tizio e Caio si sono colpiti a vicenda davanti a un bar (S1): «Chiunque partecipa a una "
                          "rissa è punito».", cf=["S1"], cn=RISSA_1))       # v79.16.1 (item 2): a «si» cites its facts
    assert rec["verified"] and rec["fit"] == "si" and rec["facts"] == ["S1"] and rec["cita_norma"] == RISSA_1
    assert rec["articolo"] == "588 c.p."
    assert rec["passage"] == "Chiunque partecipa a una rissa è punito con la multa fino a euro 2.000." and \
        rec["comma"] == "1"
    assert res["counters"]["fit_si"] == 1 and res["counters"]["fit_added_content"] == 0


def test_rissa_three_persons_rule_not_in_the_text_is_rejected_and_counted():
    res, rec = _one(RISSA, FACTS_RISSA, _item(
        "588 c.p.", "no", "La rissa richiede almeno tre persone, mentre i fatti descrivono «Tizio e Caio, due "
                          "persone» (S1).", cf=["S1"]))
    # v79.16.1 (item 2): verified by ids only -- the perché is logged, never read for content (the «tre persone» rule
    # it adds is no reason to discard the «no» that cites its fact)
    assert rec["verified"] and not rec["added_content"] and rec["reason"] == ""
    c = res["counters"]
    assert c["fit_no"] == 1 and c["fit_no_verified"] == 1 and c["fit_unverified"] == 0 and c["fit_added_content"] == 0


def test_rissa_no_citing_a_fact_that_does_not_exist_is_rejected():
    res, rec = _one(RISSA, FACTS_RISSA, _item(
        "588 c.p.", "no", "Tizio e Caio sono due persone (S1), non una rissa.", cf=["S7"]))
    # v79.15.0 (item 3): a «no» citing a fact id not given -- unverified, counted as an id failure
    assert not rec["verified"] and rec["reason"] == "cita_fatti: fatto inesistente: S7" and rec["ids_failed"]
    assert res["counters"]["fit_unverified"] == 1 and res["counters"]["fit_added_content"] == 0
    assert res["counters"]["fit_no_unverified_ids"] == 1 and res["counters"]["fit_unverified_ids"] == 1


def test_rissa_dipende_with_a_grounded_question_passes_and_touches_its_element():
    rec = _one(RISSA, FACTS_RISSA, _item(
        "588 c.p.", "dipende", "Caio sostiene di aver riportato una ferita (S2): dipende se taluno «riporta lesione "
                               "personale».", condizione="Caio ha riportato una lesione personale?",
        cf=["S2"], cn=["A588cp.2"]))[1]
    assert rec["verified"] and rec["fit"] == "dipende" and rec["fact_quotes"] == [FACTS_RISSA[1]["text"]]
    assert rec["condizione"] == "Caio ha riportato una lesione personale?"
    assert rec["touches"]["elements"] == ["E2"] and rec["touches"]["commi"] == ["2"]


def test_rissa_dipende_reasoning_adding_the_penalty_rule_is_rejected():
    rec = _one(RISSA, FACTS_RISSA, _item(
        "588 c.p.", "dipende", "Caio sostiene di aver riportato una ferita (S2), e la pena cambia se taluno «riporta "
                               "lesione personale».", condizione="Caio ha riportato una lesione personale?",
        cf=["S2"]))[1]
    assert not rec["verified"] and rec["reason"] == "cita_norma mancante"     # v79.16.1 (item 2): a «dipende» cites
    rec = _one(RISSA, FACTS_RISSA, _item(                                      # its periods and its facts; nothing else
        "588 c.p.", "dipende", "Caio sostiene di aver riportato una ferita (S2), e la pena cambia se taluno «riporta "
                               "lesione personale».", condizione="Caio ha riportato una lesione personale?",
        cf=["S2"], cn=RISSA_1))[1]
    assert rec["verified"] and not rec["added_content"] and rec["condizione"] == "Caio ha riportato una lesione personale?"


def test_rissa_invented_condition_is_rejected_and_counted():
    res, rec = _one(RISSA, FACTS_RISSA, _item(
        "588 c.p.", "dipende", "Tizio e Caio si sono colpiti a vicenda (S1): «Chiunque partecipa a una rissa».",
        condizione="Alla rissa hanno partecipato almeno tre persone?", cf=["S1"], cn=RISSA_1))
    # v79.16.1 (item 2): the condizione is recorded, never verified for content
    assert rec["verified"] and not rec["added_content"] and rec["condizione"].startswith("Alla rissa")
    assert res["counters"]["fit_added_content"] == 0 and res["counters"]["fit_dipende"] == 1
    assert res["counters"]["fit_dipende_verified"] == 1


def test_dipende_condition_must_be_a_question():
    rec = _one(RISSA, FACTS_RISSA, _item(
        "588 c.p.", "dipende", "Tizio e Caio si sono colpiti a vicenda (S1): «Chiunque partecipa a una rissa».",
        condizione="Bisogna sapere chi ha colpito per primo.", cf=["S1"], cn=RISSA_1))[1]
    assert rec["verified"] and rec["condizione"] == "Bisogna sapere chi ha colpito per primo."   # v79.16.1: ids only


def test_loan_without_instalments_1819_no_citing_its_fact_passes():
    res, rec = _one(MUTUO, FACTS_MUTUO, _item(
        "1819 c.c.", "no", "Il prestito è «da restituire in un'unica soluzione» (S1), non «rateale».",
        ambito="rimedio", cf=["S1"], cn=["A1819cc.1"]), role="attore")
    assert rec["verified"] and rec["fit"] == "no" and rec["ambito"] == "rimedio"
    assert rec["passage"].startswith("Se è stata convenuta la restituzione rateale") and rec["comma"] == "1"
    assert rec["fact_quotes"] == [FACTS_MUTUO[0]["text"]] and rec["facts"] == ["S1"]
    assert res["counters"]["fit_no"] == 1 and res["counters"]["fit_no_verified"] == 1
    # v79.15.0 (item 3): a «no» citing no fact (the article's period only) is no verified «no»
    old = _one(MUTUO, FACTS_MUTUO, _item("1819 c.c.", "no", "Il prestito (S1) è da restituire in un'unica soluzione, "
                                                            "non «rateale».", ambito="rimedio", cn=["A1819cc.1"]))
    assert old[1]["reason"] == "cita_fatti mancante"
    assert old[0]["counters"]["fit_no"] == 0 and old[0]["counters"]["fit_no_unverified_ids"] == 1


def test_1819_naming_the_article_itself_is_no_added_content():
    rec = _one(MUTUO, FACTS_MUTUO, _item(
        "1819 c.c.", "no", "L'art. 1819 c.c. non si applica: il prestito è «da restituire in un'unica soluzione» "
                           "(S1), non «rateale».", ambito="rimedio", cf=["S1"]))[1]
    assert rec["verified"], rec["reason"]


def test_another_norms_number_is_added_content():
    rec = _one(MUTUO, FACTS_MUTUO, _item(
        "1819 c.c.", "no", "Il prestito è «da restituire in un'unica soluzione» (S1), non «rateale», come art. 1820.",
        ambito="rimedio", cf=["S1"]))[1]
    assert rec["verified"] and not rec["added_content"]           # v79.16.1 (item 2): the perché's words never read


def test_advance_payment_646_judged_on_altrui_with_the_reasoning_verified():
    res, rec = _one(APPROPRIAZIONE, FACTS_ACCONTO, _item(
        "646 c.p.", "no", "L'acconto versato a Beta «sul prezzo dei lavori» (S1) non è «il denaro o la cosa mobile "
                          "altrui»: è di Beta.", cf=["S1"], cn=["A646cp.1"]))
    assert rec["verified"] and rec["fit"] == "no" and "altrui" in rec["passage"] and rec["comma"] == "1"
    assert res["counters"]["fit_no"] == 1


def test_646_citing_a_period_of_another_article_fails():
    # v79.15.0 (item 3): a period id exists only in its own article -- 588's first period is no period of 646
    for ids in (["A588cp.1"], ["A646cp.1", "A646cp.9"], ["646.1"]):
        rec = _one(APPROPRIAZIONE, FACTS_ACCONTO, _item(
            "646 c.p.", "si", "L'acconto versato a Beta (S1) è il denaro altrui.", cn=ids))[1]
        assert not rec["verified"] and rec["reason"].startswith("cita_norma: periodo non dell'articolo"), ids
        assert rec["ids_failed"] and rec["passage"] == ""


def test_646_reasoning_adding_a_rule_fails():
    res, rec = _one(APPROPRIAZIONE, FACTS_ACCONTO, _item(
        "646 c.p.", "no", "L'acconto «sul prezzo dei lavori» (S1) non è «altrui» perché la proprietà del denaro "
                          "passa a chi lo riceve.", cf=["S1"], cn=["A646cp.1"]))
    assert rec["verified"] and not rec["added_content"]           # v79.16.1 (item 2): the perché's words never read
    assert res["counters"]["fit_added_content"] == 0 and res["counters"]["fit_no_verified"] == 1


def test_reviewed_note_grounds_the_reasoning():
    norm = dict(APPROPRIAZIONE, note="Nell'acconto la proprietà del denaro passa a chi lo riceve.")
    rec = _one(norm, FACTS_ACCONTO, _item(
        "646 c.p.", "no", "L'acconto «sul prezzo dei lavori» (S1) non è «altrui» perché la proprietà del denaro "
                          "passa a chi lo riceve.", cf=["S1"], cn=["A646cp.1"]))[1]
    assert rec["verified"], rec["reason"]


def test_reviewed_note_reads_the_lawyer_note_and_the_reviewed_schema(monkeypatch, tmp_path):
    from src.rag import article_notes
    path = tmp_path / "reviewed.jsonl"
    path.write_text(json.dumps({"label": "646 c.p.", "reviewed": True, "note_di_inquadramento": [{"testo": "nota A"}],
                                "schema": {"elementi": [{"id": "E1", "testo": "cosa mobile altrui"}]}}) + "\n",
                    encoding="utf-8")
    monkeypatch.setattr(article_notes, "for_label", lambda label, rows=(): ("nota del legale", "file"))
    note = ap.reviewed_note("646 c.p.")
    assert "nota del legale" in note and "nota A" in note and "cosa mobile altrui" in note


# --- 2. the verification's other checks -------------------------------------------------------------------------------

def test_stem_tolerant_passage_is_found():
    assert ap.passage_found(ART_588, "riportato lesioni personali")         # participle, plural
    assert ap.passage_found(ART_588, "chiunque partecipa a una rissa")       # case, verbatim
    assert not ap.passage_found(ART_588, "riporta lesione grave")
    assert not ap.passage_found(ART_588, "a una")                            # no content word
    assert not ap.passage_found(ART_588, "rissa multa")                       # words not contiguous
    assert ap.passage_found(ART_588, "partecipa rissa è punito")           # articles, short words aside


def test_a_no_citing_no_fact_is_discarded():
    rec = _one(RISSA, FACTS_RISSA, _item("588 c.p.", "no", "Tizio e Caio: non una rissa.", cf=[], cn=RISSA_1))[1]
    assert not rec["verified"] and rec["reason"] == "cita_fatti mancante"


def test_unknown_fact_id_is_discarded():
    # v79.16.1 (item 2): an id named in the perché is not read; an id cited in cita_fatti must exist
    rec = _one(RISSA, FACTS_RISSA, _item("588 c.p.", "si", "Tizio e Caio (S1, S9): «Chiunque partecipa a una rissa».",
                                         cf=["S1"], cn=RISSA_1))[1]
    assert rec["verified"]
    rec = _one(RISSA, FACTS_RISSA, _item("588 c.p.", "si", "Tizio e Caio (S1): «Chiunque partecipa a una rissa».",
                                         cf=["S9"], cn=RISSA_1))[1]
    assert not rec["verified"] and rec["reason"] == "cita_fatti: fatto inesistente: S9" and rec["ids_failed"]


def test_a_si_citing_no_period_is_discarded():
    rec = _one(RISSA, FACTS_RISSA, _item("588 c.p.", "si", "Tizio e Caio partecipano a una rissa (S1)."))[1]
    assert not rec["verified"] and rec["reason"] == "cita_norma mancante"


def test_out_of_schema_values_are_discarded():
    ok = "Tizio e Caio (S1): «Chiunque partecipa a una rissa»."
    # v79.16.1 (item 2): only the fit value and the ids decide -- an unknown ambito, an empty perché, a «dipende»
    # with no condizione are recorded, not discarded; a «si» or a «dipende» with no cita_fatti / cita_norma is
    for item, why in ((_item("588 c.p.", "forse", ok, cf=["S1"], cn=RISSA_1), "fit fuori schema"),
                      (_item("588 c.p.", "dipende", ok, cn=RISSA_1), "cita_fatti mancante"),       # v79.15.0 (item 3)
                      (_item("588 c.p.", "si", ok, cn=RISSA_1), "cita_fatti mancante"),
                      (_item("588 c.p.", "si", ok, cf=["S1"]), "cita_norma mancante"),
                      (_item("588 c.p.", "dipende", ok, cf=["S1"]), "cita_norma mancante")):
        rec = _one(RISSA, FACTS_RISSA, item)[1]
        assert not rec["verified"] and rec["reason"].startswith(why), (item, rec["reason"])
    for item in (_item("588 c.p.", "si", ok, ambito="altro", cf=["S1"], cn=RISSA_1),
                 _item("588 c.p.", "si", "", cf=["S1"], cn=RISSA_1),
                 _item("588 c.p.", "dipende", ok, cf=["S1"], cn=RISSA_1),
                 _item("588 c.p.", "Sì", ok, cf=["S1"], cn=RISSA_1),
                 _item("588 c.p.", "no", ok, cf=["S1"])):                   # a «no»: its facts suffice
        rec = _one(RISSA, FACTS_RISSA, item)[1]
        assert rec["verified"], (item, rec["reason"])
    assert _one(RISSA, FACTS_RISSA, _item("588 c.p.", "si", ok, ambito="altro", cf=["S1"], cn=RISSA_1))[1]["ambito"] == "altro"


def test_two_sentences_are_discarded():
    rec = _one(RISSA, FACTS_RISSA, _item("588 c.p.", "si", "Tizio e Caio (S1): «Chiunque partecipa a una rissa». "
                                                           "Caio sostiene (S2).", cf=["S1"], cn=RISSA_1))[1]
    assert rec["verified"]                      # v79.16.1 (item 2): the perché's shape is not read either
    rec = _one(MUTUO, FACTS_MUTUO, _item(
        "1819 c.c.", "no", "Secondo l'art. 1819 c.c. Il prestito è «da restituire in un'unica soluzione» (S1), non "
                           "«rateale».", ambito="rimedio", cf=["S1"]))[1]
    assert rec["verified"], rec["reason"]


# --- 4. replies and calls that fail ----------------------------------------------------------------------------------

def test_malformed_reply_leaves_the_norms_unjudged():
    res = ap.judge([RISSA, MUTUO], FACTS_RISSA, "imputato", call=lambda m, k: "non è JSON")
    assert [r["reason"] for r in res["judgments"]] == ["risposta non JSON"] * 2
    assert res["counters"]["fit_unverified"] == 2 and res["timing"]["calls"][0]["status"] == "malformed"


def test_missing_norm_and_non_object_items():
    reply = _reply("rubbish", _item("588 c.p.", "si", "Tizio e Caio (S1): «Chiunque partecipa a una rissa».",
                                    cf=["S1"], cn=RISSA_1))
    res = ap.judge([RISSA, MUTUO], FACTS_RISSA, "imputato", call=lambda m, k: reply)
    assert res["by_article"]["588 c.p."]["verified"]
    assert res["by_article"]["1819 c.c."]["reason"] == "giudizio mancante nella risposta"
    assert res["counters"]["fit_judged"] == 2 and res["counters"]["fit_si"] == 1 and res["counters"]["fit_unverified"] == 1


def test_failed_call_keeps_the_current_path():
    def boom(messages, kwargs):
        raise RuntimeError("connection refused")
    res = ap.judge([RISSA], FACTS_RISSA, "imputato", call=boom)
    assert res["judgments"][0]["reason"] == "chiamata fallita" and not res["judgments"][0]["verified"]
    assert res["counters"]["fit_failed_calls"] == 1 and res["counters"]["fit_unverified"] == 1


def test_no_facts_or_no_text_sends_nothing():
    calls = []
    res = ap.judge([RISSA], [], "imputato", call=lambda m, k: calls.append(1) or "")
    assert not calls and res["counters"]["fit_skipped"] == 1 and res["counters"]["fit_judged"] == 0
    res = ap.judge([dict(RISSA, text="")], FACTS_RISSA, "imputato", call=lambda m, k: calls.append(1) or "")
    assert not calls and res["judgments"][0]["reason"] == "testo dell'articolo mancante"


def test_deadline_cuts_a_slow_batch(monkeypatch):
    cfg = dict(ap.config(), deadline_s=0.2)
    monkeypatch.setattr(ap, "config", lambda: cfg)

    def slow(messages, kwargs):
        time.sleep(1.0)
        return _reply()
    t0 = time.time()
    res = ap.judge([RISSA], FACTS_RISSA, "imputato", call=slow)
    assert time.time() - t0 < 0.9
    assert res["counters"]["fit_timeouts"] == 1 and res["judgments"][0]["reason"] == "chiamata oltre il tempo"


# --- 1. the call ---------------------------------------------------------------------------------------------------

def _thirty():
    return [dict(RISSA, article=f"{100 + i} c.p.", elements=[]) for i in range(30)]


def test_batches_are_at_most_eight_and_balanced():
    assert [len(b) for b in ap.batches(list(range(30)), 8)] == [8, 8, 7, 7]
    assert [len(b) for b in ap.batches(list(range(8)), 8)] == [8]
    assert [len(b) for b in ap.batches(list(range(9)), 8)] == [5, 4]
    assert ap.batches([], 8) == []


def test_thirty_norms_four_calls_concurrent_and_bounded():
    lock, state = threading.Lock(), {"now": 0, "peak": 0}

    def reply_for(labels):
        with lock:
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
        time.sleep(0.2)
        with lock:
            state["now"] -= 1
        return _reply(*[_item(l, "si", "Tizio e Caio (S1): «Chiunque partecipa a una rissa».", cf=["S1"],
                              cn=[f"A{ap.article_token(l)}.1"]) for l in labels])
    call = _fake(reply_for)
    res = ap.judge(_thirty(), FACTS_RISSA, "imputato", call=call)
    assert len(call.seen) == 4 and max(len(s["labels"]) for s in call.seen) <= 8
    assert sorted(l for s in call.seen for l in s["labels"]) == sorted(n["article"] for n in _thirty())
    assert 1 < state["peak"] <= ap.config()["concurrency"]
    assert res["counters"]["fit_si"] == 30 and res["counters"]["fit_calls"] == 4
    assert res["timing"]["total_s"] < 0.2 * 2 and len(res["timing"]["calls"]) == 4


def test_request_is_thinking_off_temperature_zero_json():
    call = _fake(lambda labels: _reply())
    ap.judge([RISSA], FACTS_RISSA, "imputato", call=call)
    kwargs = call.seen[0]["kwargs"]
    assert kwargs["temperature"] == 0
    assert kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    system = call.seen[0]["messages"][0].content
    assert "JSON" in system and "giudizi" in system and system == ap.prompt_system()
    assert len(ap.prompt_hash()) == 12


def test_default_call_binds_the_request_to_the_repo_chat_model(monkeypatch):
    import types
    import src.rag as pkg
    bound = {}

    class Model:
        def bind(self, **kw):
            bound.update(kw)
            return self

        def invoke(self, messages):
            return types.SimpleNamespace(content=_reply(), response_metadata={})
    fake = types.SimpleNamespace(chat_model=Model(), _stage_count=lambda: None,
                                 _stage_call=lambda *a: None, _reply_text=lambda r, m=None: r.content)
    monkeypatch.setattr(pkg, "ai_chat", fake, raising=False)
    assert ap._default_call([], ap.request_kwargs()) == _reply()
    assert bound["temperature"] == 0 and bound["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False


def test_human_message_labels_facts_and_numbers_the_periods():
    call = _fake(lambda labels: _reply())
    ap.judge([RISSA], FACTS_RISSA, "imputato", call=call)
    human = call.seen[0]["messages"][1].content
    assert "S1 [certo] Tizio e Caio" in human and "S2 [allegazione] Caio sostiene" in human
    assert "RUOLO DEL CLIENTE: imputato" in human and "Rubrica: Rissa" in human
    # v79.15.0 (item 3): the article as its numbered periods (applicability.article_ids), what cita_norma cites
    assert "Testo (periodi numerati):" in human and "[A588cp.1] Chiunque partecipa" in human
    assert "[A588cp.2] Se nella rissa" in human and "[A588cp.3] La stessa pena" in human
    assert "Contesto dello schema" not in human


def test_schema_context_only_with_schemas_on(monkeypatch):
    norm = dict(RISSA, schema={"tipo": ["reato"], "rapporto": "rissa"})
    call = _fake(lambda labels: _reply())
    ap.judge([norm], FACTS_RISSA, "imputato", call=call)
    assert "Contesto dello schema" not in call.seen[0]["messages"][1].content
    monkeypatch.setenv("ELEMENT_SCHEMAS", "true")
    ap.judge([norm], FACTS_RISSA, "imputato", call=call)
    human = call.seen[1]["messages"][1].content
    assert "Contesto dello schema (solo contesto, non il giudizio): tipo: reato; rapporto: rissa" in human


def test_counters_add_up_and_the_flag_defaults_off(monkeypatch):
    items = [_item("588 c.p.", "si", "Tizio e Caio (S1): «Chiunque partecipa a una rissa».", cf=["S1"], cn=RISSA_1),
             _item("1819 c.c.", "no", "«Tizio e Caio, due persone» (S1): non «rateale» con tre persone.")]   # no ids
    res = ap.judge([RISSA, MUTUO], FACTS_RISSA, "imputato", call=lambda m, k: _reply(*items))
    c = res["counters"]
    assert c["fit_judged"] == c["fit_si"] + c["fit_no"] + c["fit_dipende"] + c["fit_unverified"] == 2
    assert c["fit_added_content"] == 0 and c["fit_unverified_ids"] == 1      # v79.16.1 (item 2): ids only
    assert c["fit_si_verified"] == 1 and c["fit_dipende_verified"] == 0 and c["fit_no_verified"] == 0
    assert all(not r["added_content"] for r in res["judgments"] if r["verified"])
    assert [x["article"] for x in res["log"]] == ["1819 c.c."]
    assert not ap.enabled()
    monkeypatch.setenv("APPLICABILITY_FIT", "true")
    assert ap.enabled()


def test_timing_script_with_a_stub_model_never_calls_the_model(capsys, monkeypatch):
    import importlib.util
    import pathlib
    path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "fit_timing.py"
    spec = importlib.util.spec_from_file_location("fit_timing", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(ap, "_default_call", lambda *a: (_ for _ in ()).throw(AssertionError("model called")))
    assert mod.main(["--stub-latency", "0", "--n", "30"]) == 0
    out = capsys.readouterr().out
    assert "fit_judged=30" in out and "fit_si=30" in out and out.count("  batch ") == 4
