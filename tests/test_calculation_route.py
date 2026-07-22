import os
import threading

import pytest
import requests

# Importing ChatSession also imports the existing RAG module-level clients.
# Keep that persistence-only import local and offline during test collection.
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("EMBEDDING_PROVIDER", "openai")
os.environ.setdefault("EMBEDDING_API_KEY", "test-key")
os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test-password")

from src.chatbot.session import ChatBot, ChatSession, Message
from src.rag.calculation import (
    calculation_gate,
    calculation_node,
    route_after_calculation,
)


class _Response:
    def __init__(self, body, status_code=200):
        self._body = body
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self._body


def _mock_http(monkeypatch, *responses):
    queued = iter(responses)
    calls = []

    def fake_post(url, **kwargs):
        calls.append({"url": url, **kwargs})
        response = next(queued)
        if isinstance(response, Exception):
            raise response
        return _Response(response)

    monkeypatch.setattr("src.rag.calculation.requests.post", fake_post)
    return calls


def test_gate_routes_high_scoring_match_to_calculation(monkeypatch):
    candidate = {
        "calculator_id": "legal_it.irpef",
        "score": 3,
        "required_inputs": [{"name": "taxable_income", "type": "decimal"}],
    }
    calls = _mock_http(
        monkeypatch,
        {"query": "calcolo irpef", "status": "matched", "candidates": [candidate]},
    )

    update = calculation_gate({"query": "calcolo irpef"})

    assert update == {"calc_route": "calculate", "calculation_match": candidate}
    assert calls[0]["url"].endswith("/match")
    assert calls[0]["json"] == {"query": "calcolo irpef"}


@pytest.mark.parametrize(
    "body",
    [
        {"status": "matched", "candidates": [{"calculator_id": "x", "score": 2}]},
        {"status": "ambiguous", "candidates": [{"calculator_id": "x", "score": 5}]},
        {"status": "no_match", "candidates": []},
    ],
)
def test_gate_keeps_low_ambiguous_and_empty_matches_on_normal_route(monkeypatch, body):
    _mock_http(monkeypatch, body)

    assert calculation_gate({"query": "question"}) == {"calc_route": "normal"}


def test_gate_fails_safe_when_platform_is_unreachable(monkeypatch):
    _mock_http(monkeypatch, requests.ConnectionError("offline"))

    assert calculation_gate({"query": "calcolo irpef"}) == {"calc_route": "normal"}


def test_calculation_node_success_narrates_result_and_sources(monkeypatch):
    calls = _mock_http(
        monkeypatch,
        {
            "calculator_id": "legal_it.irpef",
            "status": "success",
            "result": {"gross_tax": "11060.00"},
            "citations": [
                {
                    "reference": "Art. 11 TUIR",
                    "source_name": "Normattiva",
                    "url": "https://example.test/tuir",
                }
            ],
        },
    )
    state = {
        "query": "Calcola l'IRPEF 2026 su 42.000,00 euro",
        "session_language": "it",
        "calculation_match": {
            "calculator_id": "legal_it.irpef",
            "required_inputs": [
                {"name": "taxable_income", "type": "decimal", "unit": "EUR"}
            ],
            "optional_inputs": [],
            "supports_tax_year": True,
        },
    }

    update = calculation_node(state)

    assert "11060.00" in update["answer"]
    assert "Fonti:" in update["answer"]
    assert "Art. 11 TUIR" in update["answer"]
    assert update["calculation_result"] == {"gross_tax": "11060.00"}
    assert update["pending_calculation"] is None
    assert update["awaiting_clarification"] is False
    assert update["pending_sections"] == []
    assert calls[0]["url"].endswith("/calculate")
    assert calls[0]["json"] == {
        "calculator_id": "legal_it.irpef",
        "inputs": {"taxable_income": "42000.00"},
        "tax_year": 2026,
    }


def test_calculation_node_missing_inputs_sets_one_clarification_and_pending_state(
    monkeypatch,
):
    missing = [
        {
            "name": "capital",
            "type": "decimal",
            "required": True,
            "description": "Capitale su cui calcolare gli interessi",
        },
        {
            "name": "period",
            "type": "period",
            "required": True,
            "description": "Periodo di riferimento",
        },
    ]
    _mock_http(
        monkeypatch,
        {
            "calculator_id": "legal_it.legal_interest",
            "status": "error",
            "result": {},
            "errors": [
                {
                    "code": "input_invalid",
                    "message": "Missing inputs",
                    "details": {
                        "missing_inputs": ["capital", "period"],
                        "missing": missing,
                    },
                }
            ],
        },
    )

    update = calculation_node(
        {
            "query": "Calcola gli interessi legali",
            "session_language": "it",
            "calculation_match": {
                "calculator_id": "legal_it.legal_interest",
                "required_inputs": [{"name": "capital", "type": "decimal"}],
                "optional_inputs": [],
                "requires_period": True,
            },
        }
    )

    assert update["answer"].count("?") == 1
    assert "capital" in update["answer"]
    assert "period" in update["answer"]
    assert update["pending_calculation"] == {
        "calculator_id": "legal_it.legal_interest",
        "inputs_so_far": {},
        "round": 1,
        "missing_inputs": missing,
    }
    assert update["awaiting_clarification"] is False
    assert update["pending_sections"] == []


def test_calculation_continuation_merges_follow_up_values_and_completes(monkeypatch):
    missing = [
        {"name": "capital", "type": "decimal", "required": True},
        {"name": "period", "type": "period", "required": True},
    ]
    calls = _mock_http(
        monkeypatch,
        {
            "calculator_id": "legal_it.legal_interest",
            "status": "success",
            "result": {"interest": "160.00"},
            "citations": [{"reference": "Art. 1284 c.c."}],
        },
    )

    update = calculation_node(
        {
            "query": "10.000,00 euro dal 01-01-2025 al 31-12-2025",
            "session_language": "it",
            "pending_calculation": {
                "calculator_id": "legal_it.legal_interest",
                "inputs_so_far": {},
                "round": 1,
                "missing_inputs": missing,
            },
        }
    )

    assert update["pending_calculation"] is None
    assert "160.00" in update["answer"]
    assert update["awaiting_clarification"] is False
    assert update["pending_sections"] == []
    assert calls[0]["json"] == {
        "calculator_id": "legal_it.legal_interest",
        "inputs": {"capital": "10000.00"},
        "period": {"start_date": "2025-01-01", "end_date": "2025-12-31"},
    }


def test_pending_calculation_survives_session_round_trip():
    pending = {
        "calculator_id": "legal_it.legal_interest",
        "inputs_so_far": {"capital": "10000"},
        "round": 1,
        "missing_inputs": [{"name": "period", "type": "period"}],
    }
    session = ChatSession(session_id="calc-session")
    session.messages = [
        Message(
            role="assistant",
            content="Periodo?",
            metadata={"pending_calculation": pending},
        )
    ]

    serialized = session.to_dict()
    restored = ChatSession.from_dict(serialized)

    assert serialized["messages"][0]["pending_calculation"] == pending
    assert restored.messages[0].metadata["pending_calculation"] == pending


def test_fresh_calculation_outage_falls_back_without_setting_an_answer(monkeypatch):
    _mock_http(monkeypatch, requests.ConnectionError("offline"))

    update = calculation_node(
        {
            "query": "Calcola l'IRPEF su 42000 euro",
            "calculation_match": {
                "calculator_id": "legal_it.irpef",
                "required_inputs": [{"name": "taxable_income", "type": "decimal"}],
            },
        }
    )

    assert update == {
        "calc_route": "normal",
        "calculation_match": None,
        "pending_calculation": None,
    }
    assert route_after_calculation(update) == "fallback"


def test_continuation_outage_preserves_pending_inputs_and_clears_legal_state(
    monkeypatch,
):
    pending = {
        "calculator_id": "legal_it.legal_interest",
        "inputs_so_far": {"capital": "8500"},
        "round": 1,
        "missing_inputs": [{"name": "period", "type": "period"}],
    }
    _mock_http(monkeypatch, requests.ConnectionError("offline"))

    update = calculation_node(
        {
            "query": "rewritten query with the wrong dates",
            "raw_query": "dal 01/01/2024 al 31/12/2025",
            "session_language": "it",
            "pending_calculation": pending,
            "awaiting_clarification": True,
            "pending_sections": [{"stale": True}],
        }
    )

    assert "Riprova" in update["answer"]
    assert update["pending_calculation"] == pending
    assert update["awaiting_clarification"] is False
    assert update["pending_sections"] == []
    assert route_after_calculation(update) == "end"


def test_genuine_platform_error_message_is_included_in_failure(monkeypatch):
    message = "Nessun parametro disponibile per la data richiesta."
    _mock_http(
        monkeypatch,
        {
            "calculator_id": "legal_it.legal_interest",
            "status": "error",
            "errors": [{"code": "parameter_unresolved", "message": message}],
        },
    )

    update = calculation_node(
        {
            "query": "8500 euro dal 01/01/1900 al 31/12/1900",
            "calculation_match": {
                "calculator_id": "legal_it.legal_interest",
                "required_inputs": [{"name": "capital", "type": "decimal"}],
                "requires_period": True,
            },
        }
    )

    assert message in update["answer"]
    assert update["pending_calculation"] is None
    assert update["awaiting_clarification"] is False
    assert update["pending_sections"] == []


def test_continuation_without_new_values_escapes_to_normal_rag(monkeypatch):
    calls = _mock_http(monkeypatch)

    update = calculation_node(
        {
            "query": "Cosa prevede l'articolo 1284 del codice civile?",
            "raw_query": "Parlami invece della prescrizione",
            "pending_calculation": {
                "calculator_id": "legal_it.legal_interest",
                "inputs_so_far": {},
                "round": 1,
                "missing_inputs": [
                    {"name": "capital", "type": "decimal"},
                    {"name": "period", "type": "period"},
                ],
            },
        }
    )

    assert update == {
        "calc_route": "normal",
        "pending_calculation": None,
        "awaiting_clarification": False,
        "pending_sections": [],
    }
    assert route_after_calculation(update) == "fallback"
    assert calls == []


def test_raw_query_wins_over_rewritten_query_for_value_extraction(monkeypatch):
    calls = _mock_http(
        monkeypatch,
        {
            "calculator_id": "legal_it.legal_interest",
            "status": "success",
            "result": {"interest": "300.00"},
            "citations": [{"reference": "Art. 1284 c.c."}],
        },
    )

    calculation_node(
        {
            "query": "Calcola gli interessi su 99999 euro nel 2030",
            "raw_query": "8500 euro dal 01/01/2024 al 31/12/2025",
            "calculation_match": {
                "calculator_id": "legal_it.legal_interest",
                "required_inputs": [{"name": "capital", "type": "decimal"}],
                "requires_period": True,
            },
        }
    )

    assert calls[0]["json"] == {
        "calculator_id": "legal_it.legal_interest",
        "inputs": {"capital": "8500"},
        "period": {"start_date": "2024-01-01", "end_date": "2025-12-31"},
    }


def test_chat_passes_literal_user_text_as_raw_query(monkeypatch):
    session = ChatSession(
        session_id="raw-query-session",
        messages=[
            Message(role="user", content="Calcoliamo gli interessi legali"),
            Message(role="assistant", content="Dimmi capitale e periodo"),
        ],
        _language_fixed_from_first_turn=True,
    )
    bot = ChatBot.__new__(ChatBot)
    bot._sessions = {session.session_id: session}
    bot._lock = threading.Lock()
    bot._save_sessions = lambda: None
    captured = {}

    monkeypatch.setattr(
        "src.chatbot.session._rewrite_query_with_context",
        lambda *args: "rewritten query with garbled numbers",
    )
    monkeypatch.setattr(
        "src.chatbot.session.get_user_settings_by_email",
        lambda *args: {"tone": 2, "standing": 2, "response_length": 2},
    )

    def fake_rag_run(query, **kwargs):
        captured["query"] = query
        captured.update(kwargs)
        return {"answer": "ok", "retrieval_quality_ok": True}

    monkeypatch.setattr("src.chatbot.session.rag_run", fake_rag_run)
    raw = "8500 euro dal 01/01/2024 al 31/12/2025"

    bot.chat(session.session_id, raw)

    assert captured["query"] == "rewritten query with garbled numbers"
    assert captured["raw_query"] == raw
