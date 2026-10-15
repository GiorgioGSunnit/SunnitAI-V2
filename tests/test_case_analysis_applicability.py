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


def test_articles_the_rulings_agree_on_are_checked_not_exempt(model):
    """On production 116 reached `keep` through the rulings and skipped the check."""
    answer, seen, _ = model
    answer([])
    dg._filter_applicable_articles("Sempronio schiaffeggia Caio.", "", [_row("116", "Qualora il reato...")])
    assert "[116 c.p.]" in seen["human"]


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
