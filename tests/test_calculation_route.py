import builtins
import importlib
import json
import os
import sys
import threading
from types import SimpleNamespace

import pytest
import requests

os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test-password")

import src.rag.calculation as calculation
from src.rag.calculation import (
    calculation_gate,
    calculation_node,
    route_after_calculation,
)

_REAL_EXTRACT_VALUES_LLM = calculation._extract_values_llm


@pytest.fixture(autouse=True)
def offline_llm_extraction(monkeypatch):
    """Keep existing route tests on the deterministic fallback tier."""
    monkeypatch.setattr(calculation, "_extract_values_llm", lambda *args, **kwargs: None)


@pytest.fixture
def chat_classes(monkeypatch):
    """Import persistence classes without leaving test LLM credentials configured."""
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("EMBEDDING_API_KEY", "test-key")
    from src.chatbot.session import ChatBot, ChatSession, Message

    sys.modules.pop("src.rag.ai_chat", None)
    return ChatBot, ChatSession, Message


class _Response:
    def __init__(self, body, status_code=200):
        self._body = body
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)

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
        if isinstance(response, _Response):
            return response
        return _Response(response)

    monkeypatch.setattr("src.rag.calculation.requests.post", fake_post)
    return calls


def _legal_interest_tool_schema():
    return {
        "name": "legal_it__legal_interest",
        "calculator_id": "legal_it.legal_interest",
        "description": "Calcola gli interessi legali.",
        "input_schema": {
            "type": "object",
            "properties": {
                "capital": {"type": ["number", "string"]},
                "period": {
                    "type": "object",
                    "properties": {
                        "start_date": {"type": "string", "format": "date"},
                        "end_date": {"type": "string", "format": "date"},
                    },
                    "required": ["start_date", "end_date"],
                    "additionalProperties": False,
                },
                "tax_year": {"type": "integer"},
            },
            "required": ["capital", "period"],
            "additionalProperties": False,
        },
    }


def _relaxed_legal_interest_schema():
    schema = _legal_interest_tool_schema()["input_schema"]
    schema.pop("required")
    schema["properties"]["period"].pop("required")
    return schema


def _install_fake_ai_chat(monkeypatch, *, tool_call, chat_model=None):
    fake_module = SimpleNamespace(
        _call_chat_with_tools=tool_call,
        chat_model=chat_model,
    )
    monkeypatch.setitem(sys.modules, "src.rag.ai_chat", fake_module)


@pytest.fixture
def ai_chat_module(monkeypatch):
    """Load ai_chat with inert OpenAI clients, restoring any prior module afterward."""
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("EMBEDDING_API_KEY", "test-key")
    previous = sys.modules.pop("src.rag.ai_chat", None)
    module = importlib.import_module("src.rag.ai_chat")
    try:
        yield module
    finally:
        sys.modules.pop("src.rag.ai_chat", None)
        if previous is not None:
            sys.modules["src.rag.ai_chat"] = previous


def test_platform_tool_schema_is_cached(monkeypatch):
    calls = []
    schema = _legal_interest_tool_schema()

    def fake_get(url, **kwargs):
        calls.append({"url": url, **kwargs})
        return _Response(schema)

    monkeypatch.setattr(calculation, "_TOOL_SCHEMA_CACHE", {})
    monkeypatch.setattr(calculation.requests, "get", fake_get)
    client = calculation.PlatformClient(base_url="http://schema.test", timeout=1.25)

    assert client.tool_schema("legal_it.legal_interest") == schema
    assert client.tool_schema("legal_it.legal_interest") == schema
    assert calls == [
        {
            "url": "http://schema.test/calculators/legal_it.legal_interest/tool-schema",
            "timeout": 1.25,
        }
    ]


def test_unavailable_platform_tool_schema_is_not_cached(monkeypatch):
    calls = []
    responses = iter(
        [requests.ConnectionError("offline"), _Response(_legal_interest_tool_schema())]
    )

    def fake_get(url, **kwargs):
        calls.append({"url": url, **kwargs})
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(calculation, "_TOOL_SCHEMA_CACHE", {})
    monkeypatch.setattr(calculation.requests, "get", fake_get)
    client = calculation.PlatformClient(base_url="http://schema.test", timeout=1.25)

    assert client.tool_schema("legal_it.legal_interest").get("platform_unavailable")
    assert client.tool_schema("legal_it.legal_interest") == _legal_interest_tool_schema()
    assert client.tool_schema("legal_it.legal_interest") == _legal_interest_tool_schema()
    assert len(calls) == 2


def test_call_chat_with_tools_disables_parallel_calls(monkeypatch, ai_chat_module):
    response = SimpleNamespace(content="ok")

    class BoundModel:
        def bind(self, **kwargs):
            assert kwargs == {"max_tokens": 1000}
            return self

        def invoke(self, messages):
            assert messages == ["message"]
            return response

    class ChatModel:
        def bind_tools(self, tools, **kwargs):
            assert tools == [{"name": "expected"}]
            assert kwargs == {
                "tool_choice": "expected",
                "parallel_tool_calls": False,
            }
            return BoundModel()

    monkeypatch.setattr(ai_chat_module, "chat_model", ChatModel())

    assert ai_chat_module._call_chat_with_tools(
        ["message"],
        [{"name": "expected"}],
        tool_choice="expected",
        max_tokens=1000,
    ) is response


def test_call_chat_with_tools_falls_back_for_older_bind_tools(
    monkeypatch, ai_chat_module
):
    calls = []
    response = SimpleNamespace(content="ok")

    class BoundModel:
        def invoke(self, messages):
            return response

    class ChatModel:
        def bind_tools(self, tools, **kwargs):
            calls.append(kwargs)
            if "parallel_tool_calls" in kwargs:
                raise TypeError("unsupported keyword")
            return BoundModel()

    monkeypatch.setattr(ai_chat_module, "chat_model", ChatModel())

    assert ai_chat_module._call_chat_with_tools(
        [], [{"name": "expected"}], tool_choice="expected"
    ) is response
    assert calls == [
        {"tool_choice": "expected", "parallel_tool_calls": False},
        {"tool_choice": "expected"},
    ]


def test_llm_tool_args_are_used_without_consulting_regex(monkeypatch):
    period = {"start_date": "2025-01-01", "end_date": "2025-12-31"}
    monkeypatch.setattr(
        calculation,
        "_extract_values_llm",
        lambda *args, **kwargs: {"capital": "8500", "period": period},
    )
    monkeypatch.setattr(
        calculation,
        "_extract_values",
        lambda *args, **kwargs: pytest.fail("regex fallback should not run"),
    )
    calls = _mock_http(
        monkeypatch,
        {
            "calculator_id": "legal_it.legal_interest",
            "status": "success",
            "result": {"interest": "100.00"},
        },
    )

    calculation_node(
        {
            "query": "capitale ottomilacinquecento per tutto il 2025",
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
        "period": period,
    }


def test_llm_drops_unknown_and_null_tool_arguments(monkeypatch):
    captured = {}

    def tool_call(messages, tools, **kwargs):
        captured["messages"] = messages
        captured["tools"] = tools
        captured.update(kwargs)
        return SimpleNamespace(
            tool_calls=[
                {
                    "name": "legal_it__legal_interest",
                    "args": {
                        "capital": "8500",
                        "period": None,
                        "invented": "discard me",
                    }
                }
            ]
        )

    monkeypatch.setattr(calculation, "_extract_values_llm", _REAL_EXTRACT_VALUES_LLM)
    monkeypatch.setattr(
        calculation.PlatformClient,
        "tool_schema",
        lambda self, calculator_id: _legal_interest_tool_schema(),
    )
    _install_fake_ai_chat(monkeypatch, tool_call=tool_call)

    values = calculation._extract_values_llm(
        "Il capitale è 8.500 euro",
        "legal_it.legal_interest",
    )

    assert values == {"capital": "8500"}
    assert captured["tools"] == [
        {
            "name": "legal_it__legal_interest",
            "description": "Calcola gli interessi legali.",
            "parameters": _relaxed_legal_interest_schema(),
        }
    ]
    assert captured["tool_choice"] == "legal_it__legal_interest"
    schema_text = json.dumps(
        _relaxed_legal_interest_schema(), ensure_ascii=False, sort_keys=True
    )
    assert schema_text in captured["messages"][0].content
    assert "required" in _legal_interest_tool_schema()["input_schema"]


def test_llm_parses_string_tool_arguments(monkeypatch):
    def tool_call(*args, **kwargs):
        return SimpleNamespace(
            tool_calls=[
                {
                    "name": "legal_it__legal_interest",
                    "args": '{"capital": "8500"}',
                }
            ]
        )

    monkeypatch.setattr(calculation, "_extract_values_llm", _REAL_EXTRACT_VALUES_LLM)
    monkeypatch.setattr(
        calculation.PlatformClient,
        "tool_schema",
        lambda self, calculator_id: _legal_interest_tool_schema(),
    )
    _install_fake_ai_chat(monkeypatch, tool_call=tool_call)

    assert calculation._extract_values_llm(
        "Il capitale è 8.500 euro", "legal_it.legal_interest"
    ) == {"capital": "8500"}


def test_llm_merges_all_matching_tool_calls_with_later_values_winning(monkeypatch):
    def tool_call(*args, **kwargs):
        return SimpleNamespace(
            tool_calls=[
                {"name": "legal_it__legal_interest", "args": {}},
                {"name": "unrelated_tool", "args": {"capital": "invented"}},
                {
                    "name": "legal_it__legal_interest",
                    "args": {"capital": "8500", "tax_year": 2025},
                },
                {
                    "name": "legal_it__legal_interest",
                    "args": {"capital": "9000"},
                },
            ]
        )

    monkeypatch.setattr(calculation, "_extract_values_llm", _REAL_EXTRACT_VALUES_LLM)
    monkeypatch.setattr(
        calculation.PlatformClient,
        "tool_schema",
        lambda self, calculator_id: _legal_interest_tool_schema(),
    )
    _install_fake_ai_chat(monkeypatch, tool_call=tool_call)

    assert calculation._extract_values_llm(
        "Il capitale è novemila euro", "legal_it.legal_interest"
    ) == {"capital": "9000", "tax_year": 2025}


def test_clean_extracted_values_prunes_nested_values_and_incomplete_periods():
    properties = _legal_interest_tool_schema()["input_schema"]["properties"]

    assert calculation._clean_extracted_values(
        {
            "capital": "8500",
            "period": {
                "start_date": "2025-01-01",
                "end_date": "2025-12-31",
                "timezone": "Europe/Rome",
                "note": None,
            },
        },
        properties,
    ) == {
        "capital": "8500",
        "period": {"start_date": "2025-01-01", "end_date": "2025-12-31"},
    }
    assert calculation._clean_extracted_values(
        {
            "capital": "8500",
            "period": {
                "start_date": "2025-01-01",
                "end_date": None,
                "timezone": "Europe/Rome",
            },
        },
        properties,
    ) == {"capital": "8500"}


def test_llm_tiers_fail_then_regex_fallback_is_used(monkeypatch):
    class FailingStructuredModel:
        def invoke(self, messages):
            raise RuntimeError("json mode unavailable")

    class FailingChatModel:
        def bind(self, **kwargs):
            return self

        def with_structured_output(self, schema, method):
            assert schema == _relaxed_legal_interest_schema()
            assert method == "json_mode"
            return FailingStructuredModel()

    def failing_tool_call(*args, **kwargs):
        raise RuntimeError("tool calling unavailable")

    monkeypatch.setattr(calculation, "_extract_values_llm", _REAL_EXTRACT_VALUES_LLM)
    monkeypatch.setattr(
        calculation.PlatformClient,
        "tool_schema",
        lambda self, calculator_id: _legal_interest_tool_schema(),
    )
    _install_fake_ai_chat(
        monkeypatch,
        tool_call=failing_tool_call,
        chat_model=FailingChatModel(),
    )
    calls = _mock_http(
        monkeypatch,
        {
            "calculator_id": "legal_it.legal_interest",
            "status": "success",
            "result": {"interest": "100.00"},
        },
    )

    calculation_node(
        {
            "query": "8500 euro dal 01/01/2025 al 31/12/2025",
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
        "period": {"start_date": "2025-01-01", "end_date": "2025-12-31"},
    }


def test_llm_json_mode_is_the_second_extraction_tier(monkeypatch):
    extraction_schema = _relaxed_legal_interest_schema()

    class StructuredModel:
        def invoke(self, messages):
            return {"capital": "9000", "period": None, "unknown": 1}

    class ChatModel:
        def bind(self, **kwargs):
            return self

        def with_structured_output(self, schema, method):
            assert schema == extraction_schema
            assert method == "json_mode"
            return StructuredModel()

    monkeypatch.setattr(calculation, "_extract_values_llm", _REAL_EXTRACT_VALUES_LLM)
    monkeypatch.setattr(
        calculation.PlatformClient,
        "tool_schema",
        lambda self, calculator_id: _legal_interest_tool_schema(),
    )
    _install_fake_ai_chat(
        monkeypatch,
        tool_call=lambda *args, **kwargs: SimpleNamespace(
            tool_calls=[{"name": "wrong_tool", "args": {"capital": "1"}}]
        ),
        chat_model=ChatModel(),
    )

    assert calculation._extract_values_llm(
        "novemila euro", "legal_it.legal_interest"
    ) == {"capital": "9000"}


def test_calculation_import_and_llm_import_failure_are_safe(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delitem(sys.modules, "src.rag.ai_chat", raising=False)
    monkeypatch.delitem(sys.modules, "src.rag.calculation", raising=False)
    module = importlib.import_module("src.rag.calculation")
    assert module is not calculation
    assert "src.rag.ai_chat" not in sys.modules

    monkeypatch.setattr(
        module.PlatformClient,
        "tool_schema",
        lambda self, calculator_id: _legal_interest_tool_schema(),
    )
    original_import = builtins.__import__

    def fail_ai_chat_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.endswith("ai_chat"):
            raise ImportError("LLM credentials unavailable")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fail_ai_chat_import)

    assert module._extract_values_llm("8500 euro", "legal_it.legal_interest") is None


def test_platform_4xx_is_request_invalid_not_unavailable(monkeypatch):
    _mock_http(
        monkeypatch,
        _Response({"detail": "Invalid calculator payload"}, status_code=422),
    )

    response = calculation.PlatformClient(base_url="http://platform.test").calculate(
        {"calculator_id": "legal_it.legal_interest", "inputs": {}}
    )

    assert response == {
        "status": "error",
        "errors": [
            {"code": "request_invalid", "message": "Invalid calculator payload"}
        ],
    }
    assert "platform_unavailable" not in response


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


def test_llm_continuation_preserves_prior_inputs_and_merges_new_values(monkeypatch):
    period = {"start_date": "2024-01-01", "end_date": "2025-12-31"}
    missing = [{"name": "period", "type": "period", "required": True}]
    captured = {}

    def extract(query, calculator_id, missing_specs=None, prior_inputs=None):
        captured.update(
            query=query,
            calculator_id=calculator_id,
            missing_specs=missing_specs,
            prior_inputs=dict(prior_inputs or {}),
        )
        return {"period": period}

    monkeypatch.setattr(calculation, "_extract_values_llm", extract)
    monkeypatch.setattr(
        calculation,
        "_extract_values",
        lambda *args, **kwargs: pytest.fail("regex fallback should not run"),
    )
    calls = _mock_http(
        monkeypatch,
        {
            "calculator_id": "legal_it.legal_interest",
            "status": "success",
            "result": {"interest": "300.00"},
        },
    )

    calculation_node(
        {
            "raw_query": "dal primo gennaio 2024 al 31 dicembre 2025",
            "pending_calculation": {
                "calculator_id": "legal_it.legal_interest",
                "inputs_so_far": {"capital": "8500"},
                "round": 1,
                "missing_inputs": missing,
            },
        }
    )

    assert captured == {
        "query": "dal primo gennaio 2024 al 31 dicembre 2025",
        "calculator_id": "legal_it.legal_interest",
        "missing_specs": missing,
        "prior_inputs": {"capital": "8500"},
    }
    assert calls[0]["json"] == {
        "calculator_id": "legal_it.legal_interest",
        "inputs": {"capital": "8500"},
        "period": period,
    }


def test_pending_calculation_survives_session_round_trip(chat_classes):
    _, ChatSession, Message = chat_classes
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
    monkeypatch.setattr(calculation, "_extract_values_llm", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        calculation,
        "_extract_values",
        lambda *args, **kwargs: pytest.fail("regex fallback should not run"),
    )
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


def test_chat_passes_literal_user_text_as_raw_query(monkeypatch, chat_classes):
    ChatBot, ChatSession, Message = chat_classes
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
