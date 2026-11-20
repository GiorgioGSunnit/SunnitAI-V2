"""v79.13.0 review -- the applicability judgment's findings, fixed (the model faked, the facts synthetic; statutory text
of 588 c.p., 646 c.p., 1813 and 1819 c.c.; an invented article text where that suffices):

1. verification -- a «no» or a «dipende» rests on its quoted passage: no count or restriction the passage does not set
   («due persone», «solo»), no bare «non riguarda», no content-free quote («Chiunque»); the quoted passage matched word
   for word (a number, «o»/«e», «da»/«a» exactly) and never cut after a negation («è punibile» out of «non è
   punibile»); a perché never contradicts the fact it names; a condizione never smuggles a count; another code's
   article with the same number is another norm; auxiliaries and plain words are no added content
2. the call -- the reply's «articolo» however spelt, conflicting duplicates unjudged, a queued batch cancelled
3. the flow -- a special law's charge kept under its store alias; a removed norm the user names later, or removed on
   facts since changed, put back and judged again; the flag off after an answer with it on undoes its effects; a kept
   named norm with no method record, or on the prose path, still printed
4. the nightly -- a removed norm brought back by any path fails the check; the follow-up checked too
"""

import json
import sys
import threading
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))

import test_v79_13_flow as F  # noqa: E402
from test_v79_13_applicability import (ART_588, FACTS_ACCONTO, FACTS_MUTUO, FACTS_RISSA, MUTUO,  # noqa: E402
                                       APPROPRIAZIONE, RISSA, _reply)
from src.rag import applicability as ap  # noqa: E402
from src.rag import interactive_flow as flow  # noqa: E402
from src.rag import lawyer_method as M  # noqa: E402

_env = F._env           # the flow tests' environment (no real model, temporary schema files)
_item = F._item


def _v(norm, facts, fit, perche, condizione="", fatto=None):
    return ap.verify(_item(norm["article"], fit, perche, condizione=condizione, fatto=fatto), norm, facts)


LOAN_FACT = "da restituire in un'unica soluzione"     # v79.14.3 (item 2): the «fatto_citato» of each «no»/«dipende»
ACCONTO_FACT = "sul prezzo dei lavori"
FERITA_FACT = "di aver riportato una ferita"
COLPITI_FACT = "si sono colpiti a vicenda"


# --- 1. verification ----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("perche,fatto", [      # v79.14.2 (A): each «no» quotes its fact -- the invented count still
    # out (v79.14.3, item 2: the fact quote in «fatto_citato»)
    ("«Chiunque partecipa a una rissa» non si applica perché «Tizio e Caio» (S1) sono solo due persone.", "Tizio e Caio"),
    ("La rissa non sussiste tra due persone: «Tizio e Caio, due persone» (S1).", "Tizio e Caio, due persone"),
    ("La rissa non ricorre: solo due persone «si sono colpiti a vicenda» (S1).", COLPITI_FACT),
    ("«Tizio e Caio, due persone» (S1): la rissa non sussiste tra due persone.", "Tizio e Caio, due persone"),
    ("«Tizio e Caio» (S1) solo, quindi la rissa manca.", "Tizio e Caio"),
    ("«Tizio e Caio, due persone» (S1): la rissa richiede più di due persone.", "Tizio e Caio, due persone"),
])
def test_rissa_no_on_an_unstated_count_is_rejected_and_counted(perche, fatto):
    rec = _v(RISSA, FACTS_RISSA, "no", perche, fatto=fatto)
    assert not rec["verified"] and rec["added_content"], rec["reason"]
    res = ap.judge([RISSA], FACTS_RISSA, "imputato", call=lambda m, k: _reply(_item("588 c.p.", "no", perche,
                                                                                     fatto=fatto)))
    assert res["counters"]["fit_no"] == 0 and res["counters"]["fit_added_content"] == 1
    assert ap.effects(res, [])["removed"] == []


def test_646_only_restriction_not_in_the_passage_is_rejected():
    rec = _v(APPROPRIAZIONE, FACTS_ACCONTO, "no", "La norma si applica solo al denaro altrui, non al «prezzo dei "
                                                  "lavori» (S1).", fatto=ACCONTO_FACT)
    assert not rec["verified"] and rec["added_content"] and "solo" in rec["reason"]


def test_a_no_with_no_fact_quoted_or_a_content_free_quote_is_rejected():
    # v79.14.2 (A): a «no» rests on the facts it quotes -- a quote of the article alone, or a quote with no content
    # word, is no verified «no» (the quote failure, never a removal). v79.14.3 (item 2): the quote is «fatto_citato»'s
    bare = _v(RISSA, FACTS_RISSA, "no", "«Chiunque partecipa a una rissa» non riguarda (S1).")
    assert not bare["verified"] and bare["quote_failed"] and bare["reason"] == "fatto_citato mancante"
    word = _v(APPROPRIAZIONE, FACTS_ACCONTO, "no", "La norma non si applica al caso: «Il» (S1).", fatto="Il")
    assert not word["verified"] and word["quote_failed"] and word["reason"].startswith("fatto_citato non nei fatti")
    fact = "Caio non ha restituito la somma alla scadenza"
    said = _v(MUTUO, FACTS_MUTUO, "no", f"«{fact}» (S2).", fatto=fact)
    assert said["verified"] and said["facts"] == ["S2"] and said["passage"] == ""      # the fact alone: enough
    unread = _v(MUTUO, FACTS_MUTUO, "no", f"«{fact}» (S2).")         # the same quote in the perché only: not read
    assert not unread["verified"] and unread["quote_failed"] and unread["reason"] == "fatto_citato mancante"


def test_the_grounded_no_judgments_still_pass():
    for norm, facts, perche, fatto in (
            (MUTUO, FACTS_MUTUO, "Il prestito è «da restituire in un'unica soluzione» (S1), non «rateale».", LOAN_FACT),
            (MUTUO, FACTS_MUTUO, "Tizio e Caio hanno previsto «un'unica soluzione» (S1), non la «restituzione "
                                 "rateale».", "un'unica soluzione"),
            (APPROPRIAZIONE, FACTS_ACCONTO, "L'acconto versato a Beta «sul prezzo dei lavori» (S1) non è «il denaro o "
                                            "la cosa mobile altrui»: è di Beta.", ACCONTO_FACT),
            (APPROPRIAZIONE, FACTS_ACCONTO, "«Il cliente ha versato a Beta 3.000 euro come acconto» (S1), non «il "
                                            "denaro o la cosa mobile altrui».",
             "Il cliente ha versato a Beta 3.000 euro come acconto")):
        rec = _v(norm, facts, "no", perche, fatto=fatto)
        assert rec["verified"], (perche, rec["reason"])


@pytest.mark.parametrize("norm,facts,quote", [
    (RISSA, FACTS_RISSA, "Chiunque partecipa a una rissa è punito con la multa fino a euro 50.000"),
    (APPROPRIAZIONE, FACTS_ACCONTO, "con la multa da euro 100.000 a euro 300.000"),
    (RISSA, FACTS_RISSA, "taluno rimane ucciso e riporta lesione personale"),
    (APPROPRIAZIONE, FACTS_ACCONTO, "per procurare da sé o ad altri"),
    (APPROPRIAZIONE, FACTS_ACCONTO, "per procurare a sé e ad altri"),
])
def test_a_changed_number_or_short_word_in_the_quote_is_not_the_text(norm, facts, quote):
    assert not ap.passage_found(norm["text"], quote)
    rec = _v(norm, facts, "si", f"Beta trattiene l'acconto (S2), Tizio e Caio (S1): «{quote}».")
    assert not rec["verified"] and rec["reason"].startswith("passo non nel testo")


def test_the_text_as_quoted_is_still_found():
    assert ap.passage_found(ART_588, "Chiunque partecipa a una rissa è punito con la multa fino a euro 2.000")
    assert ap.passage_found(APPROPRIAZIONE["text"], "per procurare a sé o ad altri un ingiusto profitto")
    assert ap.passage_found(ART_588, "taluno rimane ucciso, o riporta lesione personale")
    assert ap.passage_found(ART_588, "riportato lesioni personali")           # stem-tolerant
    assert ap.passage_found(ART_588, "partecipa rissa è punito")              # the text's articles left out
    assert not ap.passage_found(ART_588, "rissa è punita multa")              # a word skipped: no


def test_a_quote_cut_after_a_negation_is_not_the_text():
    assert not ap.passage_found(MUTUO["text"], "adempie l'obbligo del pagamento")
    assert ap.passage_found(MUTUO["text"], "non adempie l'obbligo del pagamento")
    rec = _v(MUTUO, FACTS_MUTUO, "no", "Caio (S2) «adempie l'obbligo del pagamento».",
             fatto="adempie l'obbligo del pagamento")
    assert not rec["verified"] and rec["reason"].startswith("fatto_citato non nei fatti")    # v79.14.3 (item 2)
    assert ap.fact_quote_at("Caio non ha restituito la somma.", "ha restituito la somma") is None    # nor a fact
    art = "1. Non è punibile chi ha commesso alcuno dei fatti previsti a danno del coniuge."      # invented text
    assert not ap.passage_found(art, "è punibile chi ha commesso alcuno dei fatti")
    assert not ap.passage_found(art, "punibile chi ha commesso")
    assert ap.passage_found(art, "Non è punibile chi ha commesso alcuno dei fatti")
    assert ap.passage_found("Chi non adempie, è punito con la multa.", "è punito con la multa")


def test_a_perche_contradicting_the_fact_it_names_is_rejected():
    # v79.14.2 (A, review): a fact quote of two content words at least, the rest on the facts' words
    rec = _v(APPROPRIAZIONE, FACTS_ACCONTO, "no", "Il cliente non ha versato «a Beta 3.000 euro» (S1).",
             fatto="a Beta 3.000 euro")
    assert not rec["verified"] and rec["reason"] == "contraddice il fatto nominato: versato"
    assert ap.contradicted("Caio ha restituito la somma (S2).", ["Caio non ha restituito la somma alla scadenza."]) \
        == ["restituito"]
    assert ap.contradicted("Caio non ha restituito la somma (S2).", ["Caio non ha restituito la somma."]) == []
    assert ap.contradicted("non si applica: il cliente ha versato", ["Il cliente ha versato a Beta 3.000 euro."]) == []


def test_a_condition_smuggling_a_count_is_rejected_and_counted():
    for cond in ("Alla rissa ha partecipato anche altri, non solo Tizio e Caio?",
                 "Alla rissa hanno partecipato più di due persone?"):
        res = ap.judge([RISSA], FACTS_RISSA, "imputato", call=lambda m, k: _reply(_item(
            "588 c.p.", "dipende", "Tizio e Caio si sono colpiti (S1): «Chiunque partecipa a una rissa».",
            condizione=cond, fatto=COLPITI_FACT)))
        rec = res["judgments"][0]
        assert not rec["verified"] and rec["added_content"] and "condizione" in rec["reason"], cond
        assert res["counters"]["fit_added_content"] == 1 and res["counters"]["fit_dipende"] == 0


def test_another_codes_article_with_the_same_number_is_added_content():
    other = _v(APPROPRIAZIONE, FACTS_ACCONTO, "no", "Si applica invece l'art. 646 c.c. al denaro «sul prezzo dei "
                                                    "lavori» (S1).", fatto=ACCONTO_FACT)
    assert not other["verified"] and other["added_content"] and "646" in other["reason"]
    own = _v(MUTUO, FACTS_MUTUO, "no", "L'art. 1819 cod. civ. non si applica: il prestito è «da restituire in "
                                       "un'unica soluzione» (S1), non «rateale».", fatto=LOAN_FACT)
    assert own["verified"], own["reason"]


def test_auxiliaries_and_plain_words_are_no_added_content():
    for norm, facts, fit, perche, cond, fatto in (
            (MUTUO, FACTS_MUTUO, "dipende", "Il prestito (S1) è da restituire in un'unica soluzione: dipende dalla "
                                            "«restituzione rateale».", "Le parti hanno convenuto la restituzione rateale?",
             LOAN_FACT),
            (MUTUO, FACTS_MUTUO, "dipende", "Il prestito (S1) è da restituire in un'unica soluzione: dipende dalla "
                                            "«restituzione rateale».", "Tizio e Caio avevano convenuto una restituzione "
                                                                       "rateale?", LOAN_FACT),
            (APPROPRIAZIONE, FACTS_ACCONTO, "si", "Beta trattiene l'acconto (S2) che il cliente aveva versato (S1): «si "
                                                  "appropria il denaro o la cosa mobile altrui».", "", None),
            (RISSA, FACTS_RISSA, "dipende", "Caio sostiene una ferita (S2): conta se taluno «riporta lesione personale».",
             "Caio ha riportato una lesione personale?", FERITA_FACT)):
        rec = _v(norm, facts, fit, perche, cond, fatto)
        assert rec["verified"], (perche, rec["reason"])


# --- 2. the call ------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("label", ["1819 c.c.", "Art. 1819 c.c.", "art. 1819 c.c.", "1819 cc", "articolo 1819 c.c.",
                                   "1819 c.c. (Restituzione rateale)", "1819", "Art. 1819 codice civile"])
def test_the_reply_article_however_spelt_is_matched(label):
    item = _item(label, "no", "Il prestito è «da restituire in un'unica soluzione» (S1), non «rateale».", "rimedio",
                 fatto=LOAN_FACT)
    res = ap.judge([MUTUO], FACTS_MUTUO, "convenuto", call=lambda m, k: _reply(item))
    assert res["counters"]["fit_no"] == 1 and res["counters"]["fit_unverified"] == 0, res["judgments"][0]["reason"]


def test_a_reply_naming_another_codes_article_is_not_matched():
    item = _item("1819 c.p.", "no", "Il prestito è «da restituire in un'unica soluzione» (S1), non «rateale».", "rimedio",
                 fatto=LOAN_FACT)
    res = ap.judge([MUTUO], FACTS_MUTUO, "convenuto", call=lambda m, k: _reply(item))
    assert res["counters"]["fit_unverified"] == 1 and res["judgments"][0]["reason"] == "giudizio mancante nella risposta"


def test_conflicting_judgments_for_one_norm_leave_it_unjudged():
    no = _item("1819 c.c.", "no", "Il prestito è «da restituire in un'unica soluzione» (S1), non «rateale».", "rimedio",
               fatto=LOAN_FACT)
    si = _item("1819 c.c.", "si", "Il prestito (S1) «rateale».", "rimedio")
    res = ap.judge([MUTUO], FACTS_MUTUO, "convenuto", call=lambda m, k: _reply(no, si))
    rec = res["judgments"][0]
    assert not rec["verified"] and rec["reason"].startswith("giudizi contrastanti")
    assert res["counters"]["fit_no"] == 0 and res["counters"]["fit_unverified"] == 1
    assert [x["article"] for x in res["log"]] == ["1819 c.c."]
    same = ap.judge([MUTUO], FACTS_MUTUO, "convenuto", call=lambda m, k: _reply(no, dict(no, perche="altro")))
    assert same["counters"]["fit_no"] == 1                          # the same fit twice: the first


def test_a_batch_still_queued_at_the_deadline_never_calls_the_model(monkeypatch):
    cfg = dict(ap.config(), deadline_s=0.5)
    monkeypatch.setattr(ap, "config", lambda: cfg)
    started: list = []
    lock = threading.Lock()

    def slow(messages, kwargs):
        with lock:
            started.append(time.time())
        time.sleep(1.0)
        return _reply()
    norms = [dict(RISSA, article=f"{600 + i} c.p.") for i in range(36)]     # 5 batches, 4 at once
    res = ap.judge(norms, FACTS_RISSA, "imputato", call=slow)
    assert res["counters"]["fit_timeouts"] == 5
    time.sleep(1.3)
    assert len(started) == 4                                         # the fifth was cancelled, never sent


# --- 3. the flow ------------------------------------------------------------------------------------------------------

def test_a_special_laws_charge_is_kept_under_its_store_alias(monkeypatch):
    from src.rag import issue_records, posture
    monkeypatch.setenv("APPLICABILITY_FIT", "true")
    store = "Testo Unico sugli Stupefacenti 2026"                    # an invented store document
    text73 = "Chiunque, senza l'autorizzazione, detiene sostanze stupefacenti è punito con la reclusione."
    row = {"_article": ("73", store), "d": {"id": "TU", "name": store},
           "s": {"name": "73", "titolo_articolo": "Detenzione", "plain_text": text73}}
    case = "Il PM contesta ad Alfa l'art. 73 d.P.R. 309/1990. Alfa custodiva in casa un pacco chiuso di un amico."
    state = {"records": {"40 c.p.": {"article": "40 c.p.", "requisites": []}}, "analysis": {}, "case": case,
             "rows": [], "sentences": [{"id": "S1", "text": "Alfa custodiva in casa un pacco chiuso di un amico.",
                                        "certainty": "accertato"}]}
    monkeypatch.setattr(flow, "case_text", lambda s: s["case"])
    monkeypatch.setattr(flow, "_restored_analysis", lambda a: {})
    monkeypatch.setattr(posture, "alternative_labels", lambda s, rows=(): [])
    monkeypatch.setattr(issue_records, "_requisites", lambda r, a, *_: [
        {"id": "", "comma": 1, "kind": "requisito", "text": "detiene sostanze stupefacenti", "status": "da provare"}])
    monkeypatch.setattr(issue_records, "refresh", lambda *a, **k: None)
    canonical = flow._label(flow._key(row["_article"]))
    reply = json.dumps({"giudizi": [_item(canonical, "no", "«Alfa custodiva in casa un pacco chiuso» (S1): non "
                                                           "«detiene sostanze stupefacenti».",
                                          fatto="Alfa custodiva in casa un pacco chiuso")]})
    monkeypatch.setattr(ap, "_default_call", lambda m, k: reply)
    p = {"rito": "penale", "family": "penale", "role": "imputato", "stage": "primo_grado"}
    dead: set = set()
    flow.decompose_records(state, p, [], dead, fetch=lambda keys: [row],
                           fit=lambda wanted, have: flow.applicability_fit(state, p, [], dead, wanted, have))
    assert [j["verified"] for j in state["applicability"]["judgments"] if j["article"] == canonical] == [True]
    assert canonical in state["records"] and not state.get("fit_removed")
    assert [e["article"] for e in state["applicability"]["kept_named"]] == [canonical]
    assert canonical in state["applicability"]["charged"]
    assert state["block_counters"]["fit_kept_named"] == 1 and state["block_counters"]["norms_removed_by_fit"] == 0


def _loan_model(monkeypatch, seen=None):
    F._model(monkeypatch,
             _item("1813 c.c.", "si", "Tizio ha prestato a Caio 5.000 euro (S1): «una parte consegna all'altra una "
                                      "determinata quantità di danaro»."),
             _item("1819 c.c.", "no", "Il prestito è «da restituire in un'unica soluzione» (S1), non «rateale».",
                   ambito="rimedio", fatto=LOAN_FACT), seen=seen)


def test_a_removed_norm_the_user_names_later_is_put_back(monkeypatch):
    monkeypatch.setenv("APPLICABILITY_FIT", "true")
    _loan_model(monkeypatch)
    case = F._loan()
    before = json.loads(json.dumps(case["state"]["topics"]))
    got = F._flow(case)
    state = got["state"]
    assert "1819 c.c." not in state["records"]
    state["text"] += " Il cliente chiede espressamente se si applica l'art. 1819 c.c."     # the follow-up names it
    assert "1819 c.c." in flow.fit_protected(state, F.CONVENUTO)
    assert flow.fit_readmit(state, F.CONVENUTO) == ["1819 c.c."]            # as _write_answer does, first
    assert not state["fit_removed"] and flow.fit_removed_keys(state) == set()
    assert list(state["records"]) == ["1813 c.c.", "1819 c.c."] and state["topics"] == before
    assert state["block_counters"]["fit_readmitted"] == 1
    out = flow.fit_removed_out(state, "Sull'art. 1819 c.c. che ci chiede: non si applica. L'art. 1813 c.c. resta.")
    assert out == "Sull'art. 1819 c.c. che ci chiede: non si applica. L'art. 1813 c.c. resta."
    again = F._flow({**case, "state": state})                               # judged again: kept, the contest
    assert "1819 c.c." in again["state"]["records"] and not again["state"].get("fit_removed")
    assert again["state"]["block_counters"]["fit_kept_named"] == 1 and "1819" in again["answer"]


def test_a_removal_on_facts_since_changed_is_judged_again(monkeypatch):
    monkeypatch.setenv("APPLICABILITY_FIT", "true")
    _loan_model(monkeypatch)
    got = F._flow(F._loan())
    state = got["state"]
    assert flow.fit_readmit(state, F.CONVENUTO) == []                       # the same facts: still removed
    state["sentences"].append({"id": "S3", "text": "Le parti hanno poi convenuto la restituzione rateale in tre rate.",
                               "source": "caso", "certainty": "accertato"})
    assert flow.fit_readmit(state, F.CONVENUTO) == ["1819 c.c."]
    seen: list = []
    F._model(monkeypatch, _item("1819 c.c.", "si", "Le parti hanno poi convenuto la restituzione rateale (S3): «è stata "
                                                   "convenuta la restituzione rateale».", ambito="rimedio"), seen=seen)
    dead: set = set()
    flow.applicability_fit(state, F.CONVENUTO, got["state"].get("rows") or [
        F._row("1813", "c.c.", F.ART_1813, "Nozione"), F._row("1819", "c.c.", F.ART_1819, "Restituzione rateale")],
        dead, [])
    assert "1819 c.c." in seen[0]["labels"] and "1819 c.c." in state["records"]
    assert state["block_counters"]["fit_si"] >= 1 and not state.get("fit_removed")


def test_the_flag_off_after_an_answer_with_it_on_undoes_its_effects(monkeypatch):
    monkeypatch.setenv("APPLICABILITY_FIT", "true")
    F._model(monkeypatch, _item("588 c.p.", "dipende", "Caio sostiene di aver riportato una ferita (S2): dipende se "
                                                       "taluno «riporta lesione personale».",
                                condizione="Caio ha riportato una lesione personale?", fatto=FERITA_FACT))
    case = F._rissa()
    state = F._flow(case)["state"]
    monkeypatch.setenv("APPLICABILITY_FIT", "false")
    again = F._flow({**case, "state": state})               # read only with the flag on, even before fit_clear
    fresh = F._flow(F._rissa(), judge=False)
    assert again["answer"] == fresh["answer"]
    assert "Caio ha riportato una lesione personale?" not in again["answer"]


def test_fit_clear_puts_every_removal_back_where_it_stood(monkeypatch):
    monkeypatch.setenv("APPLICABILITY_FIT", "true")
    _loan_model(monkeypatch)
    case = F._loan()
    records_before = json.loads(json.dumps(case["state"]["records"]))
    topics_before = json.loads(json.dumps(case["state"]["topics"]))
    state = F._flow(case)["state"]
    assert list(state["records"]) == ["1813 c.c."]
    monkeypatch.setenv("APPLICABILITY_FIT", "false")
    flow.fit_clear(state)
    assert list(state["records"]) == list(records_before) and state["topics"] == topics_before
    assert "applicability" not in state and "fit_removed" not in state
    assert not any(k.startswith("fit_") or k == "norms_removed_by_fit" for k in state["block_counters"])
    assert state["records"]["1819 c.c."]["rubrica"] == "Restituzione rateale"


def test_others_topic_dropped_for_a_removal_comes_back():
    key_of = lambda x: flow._key(flow._parse_label(x))
    others = [{"title": "A", "articles": ["1819 c.c."]}, {"title": "B", "articles": ["1813 c.c."]}]
    marks: dict = {}
    left = ap.strip_topics(json.loads(json.dumps(others)), {("1819", "c.c.")}, key_of, drop_empty=True, marks=marks)
    assert left == [{"title": "B", "articles": ["1813 c.c."]}]
    assert ap.restore_topics(left, ("1819", "c.c."), marks[("1819", "c.c.")], key_of) == others


def test_a_kept_named_norm_with_no_method_record_is_still_printed(monkeypatch):
    monkeypatch.setenv("APPLICABILITY_FIT", "true")
    _loan_model(monkeypatch)
    case = F._loan(text_extra="Il cliente chiede se si applica l'art. 1819 c.c.")
    case["state"]["records"]["1819 c.c."]["not_fattispecie"] = True
    got = F._flow(case)
    assert got["state"]["block_counters"]["fit_kept_named"] == 1
    printed = [c for c in got["ctx"]["candidates"] if c["kind"] == "contest_fit"] + \
        [o for o in got["ctx"]["objections_fallback"] if "non governa i fatti narrati" in o["obiezione"]]
    assert printed and "rateale" in got["answer"]


def test_the_prose_answer_prints_a_kept_named_norms_no(monkeypatch):
    monkeypatch.setenv("APPLICABILITY_FIT", "true")
    _loan_model(monkeypatch)
    case = F._loan(text_extra="Il cliente chiede se si applica l'art. 1819 c.c.")
    state = case["state"]
    flow.applicability_fit(state, case["posture"], case["rows"], set(), [])
    out = flow.fit_prose(state, case["posture"], "Il parere.")
    assert out.startswith("Il parere.\n\n**" + ap.FIT_PROSE_TITLE + "**\n\n- ")
    assert "L'art. 1819 c.c. non governa i fatti narrati: il prestito è «da restituire in un'unica soluzione», " \
           "non «rateale»." in out and "(S1)" not in out
    monkeypatch.setenv("APPLICABILITY_FIT", "false")
    assert flow.fit_prose(state, case["posture"], "Il parere.") == "Il parere."


# --- 4. the nightly ---------------------------------------------------------------------------------------------------

def test_the_nightly_check_fails_when_a_path_brought_a_removed_norm_back(monkeypatch):
    import nightly_sim
    monkeypatch.setenv("APPLICABILITY_FIT", "true")
    _loan_model(monkeypatch)
    state = F._flow(F._loan())["state"]
    draft = flow.fit_removed_out(state, "L'art. 1819 c.c. dà l'intero. L'art. 1813 c.c. resta.")
    assert draft == "L'art. 1813 c.c. resta."
    check = nightly_sim.fit_check(draft, state)
    assert not check["ok"] and "stripped by the last pass" in check["why"]
    state["block_counters"].update(fit_removed_stripped=0, fit_removed_left=1)
    assert not nightly_sim.fit_check(draft, state)["ok"]
    state["block_counters"]["fit_removed_left"] = 0
    assert nightly_sim.fit_check(draft, state)["ok"]


def test_the_nightly_checks_the_follow_up_answer_too(monkeypatch):
    import nightly_sim
    from src.rag import interactive_flow
    monkeypatch.setenv("APPLICABILITY_FIT", "true")
    _loan_model(monkeypatch)
    state = F._flow(F._loan())["state"]
    monkeypatch.setattr(interactive_flow, "follow_up", lambda text, st: {
        "answer": "L'art. 1819 c.c. torna nella replica.", "state": st})
    out = {"checks": []}
    nightly_sim.run_followup({"followup": "E l'intero?"}, "primo", state, out)
    fit = [c for c in out["checks"] if c["check"] == "applicability_fit_followup"]
    assert fit and not fit[0]["ok"] and fit[0]["cited"] == ["1819 c.c."]
