"""_filter_applicable_articles drops a candidate article whose own real text
doesn't apply to the case facts (e.g. art. 116 c.p. - concorso anomalo,
requires more than one agent - surfacing via semantic/keyword search for a
single-defendant case), while never even asking about articles the user's
own text cites by number (`keep`) or crossref rows (deterministic, grounded
in the cited article's own text via broaden_article_context).
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


def _row(sid, name, abbr, text, source=None):
    row = {
        "d": {"id": "LEGAL_DOC::cp", "name": "Codice Penale 2026"},
        "s": {"id": sid, "name": name, "plain_text": text},
        "_article": (name.split(".")[0], abbr),
    }
    if source:
        row["_source"] = source
    return row


@pytest.fixture
def sempronio_rows():
    keep = [("584", "c.p.")]
    rows = [
        _row("584", "584.0.0", "c.p.", "Chiunque partecipa a una rissa e' punito..."),
        _row("581", "581.0.0", "c.p.", "Chiunque percuote taluno...", source=CROSSREF_SOURCE),
        _row("582", "582.0.0", "c.p.", "Chiunque cagiona lesione personale...", source=CROSSREF_SOURCE),
        _row("116", "116.0.0", "c.p.", "Se il reato commesso e' diverso da quello voluto da taluno dei concorrenti..."),
        _row("43", "43.0.0", "c.p.", "Il delitto e' doloso, o secondo l'intenzione..."),
        _row("589", "589.0.0", "c.p.", "Chiunque cagiona per colpa la morte di una persona..."),
    ]
    return rows, keep


def test_116_is_dropped_while_584_581_582_43_589_stay(sempronio_rows, monkeypatch):
    rows, keep = sempronio_rows
    calls = []

    def fake_call_chat(messages, max_tokens=None, **kwargs):
        calls.append(messages[-1].content)
        return json.dumps([
            {"article": "116 c.p.", "applicable": False,
             "reason": "Concorso anomalo richiede piu' agenti; un solo imputato."},
            {"article": "43 c.p.", "applicable": True, "reason": "Rilevante per dolo/colpa."},
            {"article": "589 c.p.", "applicable": True, "reason": "Ipotesi alternativa pertinente."},
        ])

    vlog_events = []
    monkeypatch.setattr(dg, "_call_chat", fake_call_chat)
    monkeypatch.setattr(dg, "vlog", lambda event, data=None, *a, **k: vlog_events.append((event, data)))

    filtered = dg._filter_applicable_articles("Sempronio, agendo da solo...", "", rows, keep)
    kept_articles = sorted({r["_article"][0] for r in filtered})

    assert "116" not in kept_articles
    assert kept_articles == ["43", "581", "582", "584", "589"]
    # Only the 3 non-exempt articles were sent to the LLM.
    assert "116 c.p." in calls[0] and "43 c.p." in calls[0] and "589 c.p." in calls[0]
    assert "Chiunque partecipa a una rissa" not in calls[0]  # 584 (keep) never asked about
    dropped = next(data["dropped"] for ev, data in vlog_events if ev == "case_analysis_applicability")
    assert dropped == [{"article": "116 c.p.", "reason": "Concorso anomalo richiede piu' agenti; un solo imputato."}]


def test_fails_open_on_llm_error(sempronio_rows, monkeypatch):
    rows, keep = sempronio_rows
    monkeypatch.setattr(dg, "_call_chat", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("timeout")))
    monkeypatch.setattr(dg, "vlog", lambda *a, **k: None)
    filtered = dg._filter_applicable_articles("facts", "", rows, keep)
    assert len(filtered) == len(rows)


def test_fails_open_on_malformed_reply(sempronio_rows, monkeypatch):
    rows, keep = sempronio_rows
    monkeypatch.setattr(dg, "_call_chat", lambda *a, **k: "not json")
    monkeypatch.setattr(dg, "vlog", lambda *a, **k: None)
    filtered = dg._filter_applicable_articles("facts", "", rows, keep)
    assert len(filtered) == len(rows)


def test_fails_open_per_article_when_verdict_is_missing(sempronio_rows, monkeypatch):
    rows, keep = sempronio_rows
    # Reply only covers 43, omitting 116 and 589 entirely.
    monkeypatch.setattr(dg, "_call_chat", lambda *a, **k: json.dumps(
        [{"article": "43 c.p.", "applicable": True, "reason": "ok"}]))
    monkeypatch.setattr(dg, "vlog", lambda *a, **k: None)
    filtered = dg._filter_applicable_articles("facts", "", rows, keep)
    kept = sorted({r["_article"][0] for r in filtered})
    assert "116" in kept and "589" in kept  # no verdict for these -> kept, not dropped
