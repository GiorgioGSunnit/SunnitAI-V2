"""_filter_applicable_articles drops a candidate article only when the model
shows it is incompatible with the facts: it names the requirement in the
article's own text and the fact of the case that contradicts it (e.g. art. 116
c.p., concorso anomalo, needs several agents; Sempronio acted alone).

Round 3, item 3. On production 6db1125's version kept 116 and dropped 589/590
for Sempronio: 116 was exempt because the rulings agreed on it (it was in
`keep`), and 589/590 were judged against the prosecutor's charge instead of
as alternative qualifications. Now only articles the user's own text names
(and crossref rows) are exempt, an alternative qualification is kept, and the
check runs only with APPLICABILITY_CHECK_ENABLED.
"""

import json
import os

import pytest

os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test-password")
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("EMBEDDING_PROVIDER", "openai")
os.environ.setdefault("EMBEDDING_API_KEY", "test-key")

import src.rag.defensive_generation as dg  # noqa: E402
from src.rag.nodes.cypher import CROSSREF_SOURCE  # noqa: E402

FACTS = "Sempronio, agendo da solo, schiaffeggia Caio, che cade e muore. Il PM contesta l'art. 584 c.p."


def _row(name, text, source=None):
    row = {"d": {"id": "LEGAL_DOC::cp", "name": "Codice Penale 2026"},
           "s": {"id": name, "name": f"{name}.0.0", "plain_text": text}, "_article": (name, "c.p.")}
    if source:
        row["_source"] = source
    return row


@pytest.fixture
def rows():
    return [
        _row("584", "Chiunque, con atti diretti a commettere uno dei delitti preveduti dagli articoli 581 e 582..."),
        _row("581", "Chiunque percuote taluno...", source=CROSSREF_SOURCE),
        _row("116", "Qualora il reato commesso sia diverso da quello voluto da taluno dei concorrenti..."),
        _row("43", "Il delitto è doloso, o secondo l'intenzione..."),
        _row("589", "Chiunque cagiona per colpa la morte di una persona..."),
        _row("582", "Chiunque cagiona ad alcuno una lesione personale..."),
    ]


@pytest.fixture
def model(monkeypatch):
    seen, logged = {}, []

    def answer(reply):
        def fake_call_chat(messages, max_tokens=None, **kwargs):
            seen.update(system=messages[0].content, human=messages[-1].content)
            return reply if isinstance(reply, str) else json.dumps(reply)
        monkeypatch.setattr(dg, "_call_chat", fake_call_chat)
    monkeypatch.setattr(dg, "vlog", lambda event, data=None, *a, **k: logged.append((event, data)))
    return answer, seen, logged


def test_116_is_dropped_589_kept_as_an_alternative(rows, model):
    answer, seen, logged = model
    answer([
        {"article": "116 c.p.", "verdict": "incompatibile", "requisito": "più concorrenti nel reato",
         "fatto": "Sempronio ha agito da solo", "reason": "Il concorso anomalo richiede più agenti."},
        {"article": "Art. 589 c.p.", "verdict": "alternativa", "reason": "Qualificazione subordinata."},
        {"article": "43 c.p.", "verdict": "applicabile", "reason": "Elemento soggettivo."},
        {"article": "582 c.p.", "verdict": "applicabile", "reason": "Lesioni."},
    ])
    kept = dg._filter_applicable_articles(FACTS, "", rows)
    assert [r["_article"][0] for r in kept] == ["584", "581", "43", "589", "582"]   # order unchanged
    # Exempt and never sent: 584 (named in the user's text) and 581 (cross-reference).
    assert "[584 c.p.]" not in seen["human"] and "[581 c.p.]" not in seen["human"]
    assert "[116 c.p.]" in seen["human"] and "[589 c.p.]" in seen["human"]
    assert "alternativa" in seen["system"] and "requisito" in seen["system"]
    # On production the model kept 116 as an "alternativa": that verdict is only for offences/claims.
    assert "solo per le norme che descrivono una fattispecie" in seen["system"]
    event, data = logged[-1]
    assert event == "case_analysis_applicability"
    assert data["dropped"] == [{"article": "116 c.p.", "requisito": "più concorrenti nel reato",
                                "fatto": "Sempronio ha agito da solo",
                                "reason": "Il concorso anomalo richiede più agenti."}]
    assert data["verdicts"]["589 c.p."] == "alternativa"


def test_incompatible_without_a_stated_fact_is_kept(rows, model):
    answer, _, _ = model
    answer([{"article": "589 c.p.", "verdict": "incompatibile", "requisito": "colpa", "fatto": "",
             "reason": "Il PM contesta il 584."},
            {"article": "116 c.p.", "verdict": "incompatibile", "reason": "Non pertinente."}])
    kept = [r["_article"][0] for r in dg._filter_applicable_articles(FACTS, "", rows)]
    assert "589" in kept and "116" in kept


@pytest.mark.parametrize("requisito, article_text", [
    # Prod, 20 Oct: a negation in the stated requirement used to keep the article
    # whatever its text; these were all irrelevant and must now go.
    ("caso di non punibilità per la moglie", "Non è punibile la moglie che commette il fatto per..."),
    ("Il Conduttore si obbliga a non arrecare danno o molestia al vicinato",
     "Il Conduttore si obbliga a non arrecare danno o molestia al vicinato e a rispettare il regolamento."),
])
def test_a_negation_in_the_requirement_no_longer_keeps_an_article(requisito, article_text):
    verdict = {"verdict": "incompatibile", "requisito": requisito, "fatto": "Il conduttore non paga da tre mesi"}
    assert dg._drops_article(verdict, article_text)


def test_a_limit_rule_is_still_kept_by_its_own_text():
    verdict = {"verdict": "incompatibile", "requisito": "l'inadempimento ha scarsa importanza",
               "fatto": "Il conduttore non paga da tre mesi"}
    text_1455 = ("Il contratto non si può risolvere se l'inadempimento di una delle parti ha scarsa importanza, "
                 "avuto riguardo all'interesse dell'altra.")
    assert not dg._drops_article(verdict, text_1455)
    assert "non si può" in dg._APPLICABILITY_SYSTEM and "il limite non opera" in dg._APPLICABILITY_SYSTEM


@pytest.mark.parametrize("text, limit", [
    ("Il contratto non si può risolvere se l'inadempimento di una delle parti ha scarsa importanza, avuto "
     "riguardo all'interesse dell'altra.", True),                                                   # 1455 c.c.
    ("Non è dovuta la garanzia se al momento del contratto il compratore conosceva i vizi della cosa; "
     "parimenti non è dovuta, se i vizi erano facilmente riconoscibili", True),                     # 1491 c.c.
    ("Il debitore non può essere considerato in mora, se tempestivamente ha fatto offerta della "
     "prestazione dovuta", True),                                                                   # 1220 c.c.
    ("Non è punibile chi ha commesso il fatto per esservi stato costretto dalla necessità di salvare sé od "
     "altri dal pericolo attuale di un danno grave alla persona, pericolo da lui non volontariamente "
     "causato", False),                                                                             # 54 c.p.
    ("Nei contratti indicati dall'articolo 1420 l'inadempimento di una delle parti non importa la "
     "risoluzione del contratto rispetto alle altre, salvo che la prestazione mancata debba", False),  # 1459 c.c.
    ("Qualora il reato commesso sia diverso da quello voluto da taluno dei concorrenti, anche questi ne "
     "risponde, se l'evento è conseguenza della sua azione od omissione.", False),                  # 116 c.p.
    ("Chiunque cagiona ad altri per colpa una lesione personale è punito con la reclusione", False),  # 590 c.p.
])
def test_limit_rules_are_recognised_by_their_text(text, limit):
    assert dg._is_limit_rule(text) is limit


def test_a_limit_rule_is_kept_even_when_the_model_rewords_it(model):
    """Production, 15 Oct: the requirement came back without the negation."""
    answer, _, logged = model
    rows = [{"d": {"id": "cc", "name": "Codice Civile 2026"}, "_article": ("1455", "c.c."),
             "s": {"name": "1455.0.0", "plain_text": "Il contratto non si può risolvere se l'inadempimento di una "
                                                     "delle parti ha scarsa importanza, avuto riguardo all'interesse dell'altra."}},
            _row("116", "Qualora il reato commesso sia diverso da quello voluto da taluno dei concorrenti...")]
    answer([{"article": "1455 c.c.", "verdict": "incompatibile",
             "requisito": "l'inadempimento di una delle parti ha scarsa importanza",
             "fatto": "Il conduttore non paga il canone da tre mesi"},
            {"article": "116 c.p.", "verdict": "incompatibile", "requisito": "presenza di concorrenti",
             "fatto": "Sempronio agisce da solo"}])
    kept = dg._filter_applicable_articles("Il conduttore non paga da tre mesi.", "", rows)
    assert [r["_article"][0] for r in kept] == ["1455"]
    assert [d["article"] for d in logged[-1][1]["dropped"]] == ["116 c.p."]


def test_articles_the_rulings_agree_on_are_checked_not_exempt(model):
    """On production 116 reached `keep` through the rulings and skipped the check."""
    answer, seen, _ = model
    answer([])
    dg._filter_applicable_articles("Sempronio schiaffeggia Caio.", "", [_row("116", "Qualora il reato...")])
    assert "[116 c.p.]" in seen["human"]


# --- Round 9 (19 Oct): _heading_match / _multi_signal are also exempt -------
#
# A prod diagnostic (sempronio_no_hint) found the model's "incompatibile"
# judgement wrong about art. 584 c.p. itself, a core qualification the facts
# raise, not a speculative candidate. The prompt already asks for a concrete
# requirement+fact before dropping anything; this is the code-level backstop
# for articles _select_articles_fusion marks strong enough that the check
# should never get the final say -- resolved by the article's own rubrica
# (_heading_match) or found by two or more independent signals (_multi_signal).

def test_heading_match_rows_are_exempt_and_never_sent_to_the_model(rows, model):
    rows[3]["_heading_match"] = True   # "43", normally checked (see the rows fixture)
    answer, seen, _ = model
    answer([
        {"article": "116 c.p.", "verdict": "incompatibile", "requisito": "più concorrenti nel reato",
         "fatto": "Sempronio ha agito da solo", "reason": "Il concorso anomalo richiede più agenti."},
        {"article": "589 c.p.", "verdict": "applicabile", "reason": "alternativa"},
        {"article": "582 c.p.", "verdict": "applicabile", "reason": "alternativa"},
    ])
    kept_articles = {r["_article"] for r in dg._filter_applicable_articles(FACTS, "", rows)}
    assert ("43", "c.p.") in kept_articles          # exempt: never checked, so never dropped
    assert ("116", "c.p.") not in kept_articles      # genuinely dropped, unaffected by the exemption
    assert "[43 c.p.]" not in seen["human"]          # proof it was never even sent to the model


def test_multi_signal_rows_are_checked_like_any_other(rows, model):
    """19 Oct, later: on Sempronio (prod) the fusion pool's multi-signal 116 and
    593 were never checked and reached the draft. Searches agreeing is not
    evidence an article applies."""
    rows[2]["_multi_signal"] = True   # "116"
    answer, seen, _ = model
    answer([
        {"article": "116 c.p.", "verdict": "incompatibile", "requisito": "più concorrenti nel reato",
         "fatto": "Sempronio ha agito da solo", "reason": "Il concorso anomalo richiede più agenti."},
        {"article": "589 c.p.", "verdict": "applicabile", "reason": "alternativa"},
        {"article": "582 c.p.", "verdict": "applicabile", "reason": "alternativa"},
    ])
    kept_articles = {r["_article"] for r in dg._filter_applicable_articles(FACTS, "", rows)}
    assert "[116 c.p.]" in seen["human"]
    assert ("116", "c.p.") not in kept_articles


def test_an_exemption_covers_every_row_of_the_article(rows, model):
    """581 is cross-referenced from 584's text; a second, untagged 581 row (the
    fusion pool's own) used to send it to the check anyway."""
    rows.append({**rows[1], "_source": "fusion"})
    answer, seen, _ = model
    answer([{"article": "581 c.p.", "verdict": "incompatibile", "requisito": "nessuna malattia derivata",
             "fatto": "Caio è morto"}])
    kept_articles = [r["_article"] for r in dg._filter_applicable_articles(FACTS, "", rows)]
    assert "[581 c.p.]" not in seen["human"] and kept_articles.count(("581", "c.p.")) == 2


@pytest.mark.parametrize("reply", ["not json", json.dumps({"article": "116 c.p."})])
def test_fails_open_on_a_malformed_reply(rows, model, reply):
    answer, _, _ = model
    answer(reply)
    assert dg._filter_applicable_articles(FACTS, "", rows) == rows


def test_fails_open_on_an_error(rows, monkeypatch):
    monkeypatch.setattr(dg, "_call_chat", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("timeout")))
    monkeypatch.setattr(dg, "vlog", lambda *a, **k: None)
    assert dg._filter_applicable_articles(FACTS, "", rows) == rows


def test_fails_open_per_article_when_a_verdict_is_missing(rows, model):
    answer, _, _ = model
    answer([{"article": "43 c.p.", "verdict": "applicabile", "reason": "ok"}])
    kept = [r["_article"][0] for r in dg._filter_applicable_articles(FACTS, "", rows)]
    assert "116" in kept and "589" in kept


def test_alternativa_must_match_the_actual_outcome(model):
    """prod, 17 Oct (v1 diagnostic): the model proposed 590 (lesioni colpose,
    which presupposes the victim survives) as an "alternativa" to 589 for
    Sempronio although the victim died — a contradiction the old prompt had
    no rule against. The prompt now asks for the requisito phrased
    positively ("presuppone la sopravvivenza") rather than as a negation
    ("non ne è derivata la morte"), a habit worth keeping: until 20 Oct a
    negation in the requirement kept the article (_LIMIT_WORDING, since
    removed), and it would have kept 590 again."""
    answer, _, _ = model
    row_590 = [_row("590", "Chiunque cagiona ad altri per colpa una lesione personale è punito con la "
                           "reclusione da uno a tre anni")]
    answer([{"article": "590 c.p.", "verdict": "incompatibile",
             "requisito": "presuppone la sopravvivenza della vittima",
             "fatto": "la vittima è deceduta", "reason": "L'esito è la morte, non una lesione: non è un'alternativa."}])
    assert dg._filter_applicable_articles(FACTS, "", row_590) == []
    assert "esito effettivo narrato nei fatti" in dg._APPLICABILITY_SYSTEM
    assert "non come negazione" in dg._APPLICABILITY_SYSTEM
