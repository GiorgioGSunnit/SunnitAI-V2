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


@pytest.mark.parametrize(
    "lang,heading",
    [
        ("it", "Come e stato calcolato"),
        ("es", "Como se ha calculado"),
        ("en", "How it was computed"),
    ],
)
def test_success_answer_renders_methodology_after_caveats(lang, heading):
    methodology = "Scaglioni progressivi: ogni fascia è tassata alla propria aliquota."
    answer = calculation._success_answer(
        lang,
        {
            "status": "success",
            "result": {"gross_tax": "11060.00"},
            "warnings": [{"code": "definition", "message": "Avvertenza del pack."}],
            "exclusions": ["Addizionali non incluse."],
            "methodology": methodology,
            "citations": [{"reference": "Art. 11 TUIR"}],
        },
    )

    assert f"{heading}:\n{methodology}" in answer
    assert answer.index(calculation._COPY[lang]["warnings"]) < answer.index(heading)
    assert answer.index(calculation._COPY[lang]["exclusions"]) < answer.index(heading)
    assert answer.index(heading) < answer.index(calculation._COPY[lang]["sources"])


def test_success_answer_renders_scalar_explanation_under_methodology_heading():
    answer = calculation._success_answer(
        "it",
        {
            "status": "success",
            "result": {"gross_tax": "11060.00"},
            "methodology": "Scaglioni progressivi.",
            "explanation": [
                "28.000 al 23% = 6.440,00",
                "14.000 al 33% = 4.620,00",
            ],
        },
    )

    assert (
        "Come e stato calcolato:\n"
        "Scaglioni progressivi.\n"
        "- 28.000 al 23% = 6.440,00\n"
        "- 14.000 al 33% = 4.620,00"
    ) in answer


def test_success_answer_suppresses_comparator_step_explanation():
    response = _comparison_result(provisional=False)
    response["methodology"] = "Punteggio relativo alle offerte confrontate."
    response["explanation"] = ["candidate: Alfa; exact total: 85.00"]

    answer = calculation._success_answer("it", response)

    assert response["methodology"] in answer
    assert response["explanation"][0] not in answer


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


def test_success_answer_surfaces_warnings_assumptions_and_disclaimer(monkeypatch):
    _mock_http(
        monkeypatch,
        {
            "calculator_id": "legal_it.irpef",
            "status": "success",
            "result": {"gross_tax": "11340.00"},
            "warnings": [
                {"code": "definition", "message": "Calcola solo l'IRPEF lorda nazionale."},
                {"code": "parameter_verification_missing", "message": "Parametro non verificato."},
            ],
            "assumptions": [
                {"code": "definition", "message": "Reddito già al netto delle deduzioni."},
            ],
            "citations": [{"reference": "Art. 11 TUIR"}],
        },
    )
    state = {
        "query": "Calcola l'IRPEF su 42.000 euro nel 2025",
        "session_language": "it",
        "calculation_match": {
            "calculator_id": "legal_it.irpef",
            "required_inputs": [{"name": "taxable_income", "type": "decimal"}],
            "optional_inputs": [],
            "supports_tax_year": True,
        },
    }

    answer = calculation_node(state)["answer"]

    # The computed figure, the warnings, the assumption, and the disclaimer must
    # all be present — none may be silently dropped.
    assert "11340.00" in answer
    assert "Avvisi:" in answer
    assert "Calcola solo l'IRPEF lorda nazionale." in answer
    assert "Parametro non verificato." in answer
    assert "Assunzioni:" in answer
    assert "Reddito già al netto delle deduzioni." in answer
    assert "non sostituisce la verifica di un professionista" in answer


def test_continuation_ignores_irrelevant_number_and_escapes(monkeypatch):
    """An abandoned calc + unrelated numbered question must route to normal RAG,
    not mis-bind the stray number into a calculator input (W1)."""
    # The missing field is a period; the follow-up ("e l'art. 2043 c.c.?") yields
    # only a value for capital, which is NOT what we were waiting for.
    monkeypatch.setattr(
        calculation,
        "_extract_values_llm",
        lambda *args, **kwargs: {"capital": "2043"},
    )
    calls = _mock_http(monkeypatch)  # no HTTP response queued: /calculate must NOT be called

    update = calculation_node(
        {
            "raw_query": "e l'art. 2043 c.c. cosa prevede?",
            "session_language": "it",
            "pending_calculation": {
                "calculator_id": "legal_it.legal_interest",
                "inputs_so_far": {"capital": "8500"},
                "round": 1,
                "missing_inputs": [{"name": "period", "type": "period", "required": True}],
            },
        }
    )

    assert update["calc_route"] == "normal"
    assert update["pending_calculation"] is None
    assert "answer" not in update  # falls through to RAG, does not answer here
    assert calls == []  # the platform was never asked to calculate a mis-bound value


# --- Simulation/production parity: ambiguous calculator match ----------------
# The dev simulation asks the user to choose whenever a match is ambiguous.
# Production now does the same, but only when every tied candidate would have
# auto-routed on its own (see the strong/weak ambiguity tests at the end of
# this file). A tie below the auto-route threshold still falls back silently:
# prompting there would turn any passing mention of a legal topic into a menu.

def test_calculation_gate_ambiguous_match_falls_back_to_normal_rag(monkeypatch):
    _mock_http(
        monkeypatch,
        {
            "status": "ambiguous",
            "candidates": [
                {"calculator_id": "legal_it.furto_pena_draft", "score": 1},
                {"calculator_id": "legal_it.rapina_pena_draft", "score": 1},
            ],
        },
    )
    result = calculation_gate({"query": "furto e rapina"})
    # Silent fallback: no clarification, no calculation_match carried forward.
    assert result == {"calc_route": "normal"}


def test_calculation_gate_high_score_match_routes_to_calculate(monkeypatch):
    _mock_http(
        monkeypatch,
        {
            "status": "matched",
            "candidates": [{"calculator_id": "legal_it.irpef", "score": 3}],
        },
    )
    result = calculation_gate({"query": "quanto pago di irpef su 42000 nel 2026"})
    assert result["calc_route"] == "calculate"
    assert result["calculation_match"]["calculator_id"] == "legal_it.irpef"


def _comparator_tool_schema():
    """An object_list calculator, as tool_schemas.build_tool_schema emits it."""
    return {
        "name": "business__confronto_polizze",
        "calculator_id": "business.confronto_polizze",
        "description": "Confronta polizze assicurative.",
        "input_schema": {
            "type": "object",
            "properties": {
                "offerte": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "nome": {"type": "string"},
                            "premio": {"type": ["number", "string"]},
                            "massimale": {"type": ["number", "string"]},
                            "assistenza_stradale": {"type": "boolean"},
                        },
                        "required": ["nome", "premio"],
                        "additionalProperties": False,
                    },
                    "minItems": 2,
                },
                "tax_year": {"type": "integer"},
            },
            "required": ["offerte"],
            "additionalProperties": False,
        },
    }


def test_relaxed_extraction_schema_drops_required_at_every_depth():
    schema = _comparator_tool_schema()["input_schema"]

    relaxed = calculation._relaxed_extraction_schema(schema)

    # Candidate objects must be partially extractable: with items.required left
    # in place the model has to invent fields for an offer described in part.
    assert "required" not in relaxed
    assert "required" not in relaxed["properties"]["offerte"]["items"]
    # Cleaning runs against the unrelaxed schema, so it must survive untouched.
    assert schema["required"] == ["offerte"]
    assert schema["properties"]["offerte"]["items"]["required"] == ["nome", "premio"]
    # Pre-existing period relaxation still holds.
    assert "required" not in calculation._relaxed_extraction_schema(
        _legal_interest_tool_schema()["input_schema"]
    )["properties"]["period"]


def test_clean_extracted_values_prunes_incomplete_comparator_candidates():
    properties = _comparator_tool_schema()["input_schema"]["properties"]

    cleaned = calculation._clean_extracted_values(
        {
            "offerte": [
                {"nome": "Alfa", "premio": "420", "assistenza_stradale": True},
                {"nome": "Beta", "premio": None},
                {"nome": "Gamma", "premio": "510", "sconto": "10"},
                "Delta",
            ]
        },
        properties,
    )

    # Beta never stated a premium and Delta is a bare scalar; both are unusable
    # to the comparator. Gamma's unknown field is dropped, the offer is kept.
    assert cleaned == {
        "offerte": [
            {"nome": "Alfa", "premio": "420", "assistenza_stradale": True},
            {"nome": "Gamma", "premio": "510"},
        ]
    }


def test_clean_extracted_values_drops_a_candidate_list_with_nothing_usable():
    properties = _comparator_tool_schema()["input_schema"]["properties"]

    # Nothing survives, so the field is omitted and /calculate reports the list
    # itself as missing — the ordinary clarification, not a per-field rejection.
    assert calculation._clean_extracted_values(
        {"offerte": [{"nome": "Alfa"}, {"premio": "420"}]}, properties
    ) == {}


def test_clean_extracted_values_leaves_scalar_arrays_untouched():
    properties = {"voci": {"type": "array", "items": {"type": "string"}, "minItems": 1}}

    assert calculation._clean_extracted_values(
        {"voci": ["nolo", "trasporto"]}, properties
    ) == {"voci": ["nolo", "trasporto"]}


def test_generation_reply_must_carry_pending_calculation_to_stay_resumable(chat_classes):
    _, ChatSession, _ = chat_classes
    from src.chatbot.session import last_pending_calculation

    pending = {
        "calculator_id": "legal_it.legal_interest",
        "inputs_so_far": {"capital": "10000"},
        "round": 1,
        "missing_inputs": [{"name": "period", "type": "period"}],
    }

    def _session_after_generation(reply_metadata):
        session = ChatSession(session_id="gen-session")
        session.add_message(
            "assistant", "Per quale periodo?", metadata={"pending_calculation": pending}
        )
        session.add_message("user", "scrivimi una diffida per quel credito")
        session.add_message("assistant", "Documento generato.", metadata=reply_metadata)
        return session

    # What the generation branch wrote before the fix: the calculation is
    # orphaned, because only the newest assistant turn is consulted.
    assert last_pending_calculation(_session_after_generation({"sources": []})) is None
    # Carried across, route_entry can still resume it on the next turn.
    assert last_pending_calculation(
        _session_after_generation({"sources": [], "pending_calculation": pending})
    ) == pending


def test_validation_error_with_no_missing_field_still_explains_itself(monkeypatch):
    # A comparator list below its pack minimum reports no missing field, so the
    # clarification path has nothing to ask for. Suppressing the message here
    # left the user with a bare "calculation failed" and no way to recover.
    message = "polizze needs at least 2 item(s), got 1"
    _mock_http(
        monkeypatch,
        {
            "calculator_id": "business.confronto_polizze",
            "status": "error",
            "errors": [{"code": "input_invalid", "message": message, "details": {}}],
        },
    )

    update = calculation_node(
        {
            "query": "confronta questa polizza: Alfa 420 euro",
            "calculation_match": {
                "calculator_id": "business.confronto_polizze",
                "required_inputs": [{"name": "polizze", "type": "object_list"}],
            },
        }
    )

    assert message in update["answer"]
    assert update["pending_calculation"] is None


def test_ordinary_missing_input_still_asks_instead_of_quoting_the_platform():
    # The clarification path states missing fields better, so an input_invalid
    # that does name a field must not leak the raw platform message.
    response = {
        "status": "error",
        "errors": [
            {
                "code": "input_invalid",
                "message": "Missing required input(s): capital",
                "details": {"missing": [{"name": "capital", "type": "decimal"}]},
            }
        ],
    }

    assert calculation._platform_error_message(response) is None
    assert calculation._missing_specs(response) == [
        {"name": "capital", "type": "decimal"}
    ]


def test_llm_tool_call_extracts_a_candidate_array(monkeypatch):
    captured = {}

    def tool_call(messages, tools, **kwargs):
        captured["tools"] = tools
        return SimpleNamespace(
            tool_calls=[
                {
                    "name": "business__confronto_polizze",
                    "args": {
                        "offerte": [
                            {"nome": "Alfa", "premio": "420", "assistenza_stradale": True},
                            {"nome": "Beta", "premio": "510"},
                            {"nome": "Gamma"},
                        ]
                    },
                }
            ]
        )

    monkeypatch.setattr(calculation, "_extract_values_llm", _REAL_EXTRACT_VALUES_LLM)
    monkeypatch.setattr(
        calculation.PlatformClient,
        "tool_schema",
        lambda self, calculator_id: _comparator_tool_schema(),
    )
    _install_fake_ai_chat(monkeypatch, tool_call=tool_call)

    values = calculation._extract_values_llm(
        "confronta Alfa a 420 euro con assistenza stradale e Beta a 510 euro",
        "business.confronto_polizze",
    )

    # Gamma never got a premium and is dropped; the other two survive whole.
    assert values == {
        "offerte": [
            {"nome": "Alfa", "premio": "420", "assistenza_stradale": True},
            {"nome": "Beta", "premio": "510"},
        ]
    }
    # The model must receive the relaxed item schema, or it has to invent a
    # premium for every offer the user described only in part.
    item_schema = captured["tools"][0]["parameters"]["properties"]["offerte"]["items"]
    assert "required" not in item_schema


def test_no_llm_comparator_degrades_to_clarification_without_inventing_offers(monkeypatch):
    # The autouse offline_llm_extraction fixture forces the regex tier, which
    # is the production path whenever LLM credentials are absent.
    calls = _mock_http(
        monkeypatch,
        {
            "calculator_id": "business.confronto_polizze",
            "status": "error",
            "errors": [
                {
                    "code": "input_invalid",
                    "message": "Missing required input(s): polizze",
                    "details": {
                        "missing": [
                            {"name": "polizze", "type": "object_list", "required": True}
                        ]
                    },
                }
            ],
        },
    )

    update = calculation_node(
        {
            "query": "confronta Alfa a 420 euro con Beta a 510 euro",
            "calculation_match": {
                "calculator_id": "business.confronto_polizze",
                "required_inputs": [{"name": "polizze", "type": "object_list"}],
            },
        }
    )

    # The regex tier reads dates and bare numbers; it cannot assemble offers,
    # so it must send none rather than a candidate built from stray figures.
    assert "polizze" not in calls[0]["json"]["inputs"]
    assert update["pending_calculation"]["calculator_id"] == "business.confronto_polizze"
    assert update["answer"]


# --- Incremental candidate collection (object_list comparators) --------------
# The comparator takes an array of offers. Rebuilding that array from the
# newest message on every turn loses whatever the model fails to re-derive,
# so the array is state owned by the route: one candidate per turn, prior
# candidates kept verbatim, and explicit review/confirm gates before the
# number is shown. These tests drive that protocol end to end with a scripted
# extractor standing in for the LLM.


def _polizze_match():
    """A comparator match as the platform's /match now reports it, with the
    object_list's item fields included (a router cannot collect candidates
    one at a time without knowing which per-item fields are required)."""
    return {
        "calculator_id": "business.confronto_polizze",
        "name": "Confronto polizze assicurative",
        "required_inputs": [
            {
                "name": "eta_conducente",
                "type": "integer",
                "required": True,
                "description": "Eta del conducente",
            },
            {
                "name": "polizze",
                "type": "object_list",
                "required": True,
                "min_items": 2,
                "item_fields": [
                    {"name": "nome", "type": "string", "required": True,
                     "description": "Nome della polizza"},
                    {"name": "premio_annuo", "type": "decimal", "required": True,
                     "description": "Premio annuo", "unit": "EUR"},
                    {"name": "franchigia", "type": "decimal", "required": False,
                     "default": 0},
                    {"name": "copertura_kasko", "type": "boolean", "required": False,
                     "default": False},
                ],
            },
        ],
        "optional_inputs": [],
    }


def _scripted_extractor(monkeypatch, replies):
    """Stand in for the LLM: map a substring of the user message to the
    arguments the model would have returned. A message matching nothing
    yields {} (nothing extractable), None means the LLM is unavailable."""
    seen = []

    def fake(query, calculator_id, missing_specs=None, prior_inputs=None,
             keep_partial_items=False):
        seen.append({"query": query, "keep_partial_items": keep_partial_items})
        for needle, payload in replies.items():
            if needle in query:
                return payload
        return {}

    monkeypatch.setattr(calculation, "_extract_values_llm", fake)
    return seen


def _offer(name, premium, **extra):
    return {"nome": name, "premio_annuo": premium, **extra}


def _comparison_result(*, provisional=True, status="clear_winner", tie=None):
    comparison = {
        "decision_status": status,
        "best_candidates": tie or ["Alfa"],
        "cost_basis": {"component": "punteggio_costo", "variable": "premio_netto"},
        "score_gap": "0.00" if status == "effective_tie" else "12.50",
        "tie_tolerance": "0.50",
        "provisional": provisional,
        "provisional_status": "provisional_unconfirmed" if provisional else "none",
        "assumptions_confirmed": False,
        "scoring_completeness": "0.6000" if provisional else "1.0000",
        "scoring_defaults_applied": (
            [{"path": "polizze[0].franchigia", "value": "0"}] if provisional else []
        ),
        "scored_fields": ["franchigia", "premio_annuo"],
        "candidates_compared": 2,
    }
    return {
        "status": "success",
        "calculator_id": "business.confronto_polizze",
        "result": {
            "best": "Alfa",
            "comparison": comparison,
            "ranking": [
                {"rank": 1, "label": "Alfa", "total_score": "85.00",
                 "scores": {"punteggio_costo": "100.00"},
                 "derived": {"premio_netto": "400.00"}},
                {"rank": 2, "label": "Beta", "total_score": "72.50",
                 "scores": {"punteggio_costo": "80.00"},
                 "derived": {"premio_netto": "500.00"}},
            ],
        },
        "exclusions": ["Il massimale non entra nel punteggio."],
        "defaults_applied": [{"path": "polizze[0].franchigia", "value": "0"}],
        "assumptions": [{"code": "input_default", "message": "franchigia assunta 0"}],
        "warnings": [],
        "citations": [],
    }


def _pending(update):
    pending = update.get("pending_calculation")
    assert pending is not None, update
    return pending


def _collecting(candidates, **overrides):
    """A pending comparison parked in the candidate-collection phase."""
    descriptor = calculation._descriptor_from_specs(
        calculation._match_specs(_polizze_match())
    )
    pending = {
        "calculator_id": "business.confronto_polizze",
        "calculator_name": "Confronto polizze assicurative",
        "phase": "collect_candidates",
        "inputs_so_far": {"eta_conducente": 40},
        "candidate_field": "polizze",
        "candidate_descriptor": descriptor,
        "candidates": list(candidates),
        "candidate_draft": {},
        "missing_inputs": [],
        "shared_specs": calculation._shared_specs(
            calculation._match_specs(_polizze_match()), "polizze"
        ),
        "round": 0,
    }
    pending.update(overrides)
    return pending


def test_one_shot_extraction_seeds_both_offers_without_calculating(monkeypatch):
    calls = _mock_http(monkeypatch)
    _scripted_extractor(monkeypatch, {
        "confronta": {
            "eta_conducente": 40,
            "polizze": [_offer("Alfa", "400"), _offer("Beta", "500")],
        }
    })

    update = calculation_node({
        "query": "confronta Alfa a 400 euro con Beta a 500 euro, ho 40 anni",
        "calculation_match": _polizze_match(),
    })

    pending = _pending(update)
    assert pending["phase"] == "collect_candidates"
    assert [c["nome"] for c in pending["candidates"]] == ["Alfa", "Beta"]
    assert pending["inputs_so_far"] == {"eta_conducente": 40}
    # Nothing is calculated until the user reviews and confirms.
    assert calls == []


def test_shared_inputs_are_collected_before_any_offer(monkeypatch):
    _mock_http(monkeypatch)
    _scripted_extractor(monkeypatch, {"quale polizza": {}, "ho 35 anni": {"eta_conducente": 35}})

    first = calculation_node({
        "query": "quale polizza mi conviene",
        "calculation_match": _polizze_match(),
    })
    assert _pending(first)["phase"] == "collect_shared_inputs"
    assert "eta_conducente" in first["answer"]

    second = calculation_node({
        "query": "ho 35 anni",
        "pending_calculation": _pending(first),
    })
    pending = _pending(second)
    assert pending["phase"] == "collect_candidates"
    assert pending["inputs_so_far"] == {"eta_conducente": 35}
    assert "prima offerta" in second["answer"].lower() or "Descrivimi" in second["answer"]


def test_one_offer_per_turn_keeps_the_previous_offers(monkeypatch):
    _mock_http(monkeypatch)
    _scripted_extractor(monkeypatch, {
        "Alfa": {"polizze": [_offer("Alfa", "400")]},
        "Beta": {"polizze": [_offer("Beta", "500")]},
    })

    first = calculation_node({
        "query": "la prima è Alfa, premio 400 euro",
        "pending_calculation": _collecting([]),
    })
    assert [c["nome"] for c in _pending(first)["candidates"]] == ["Alfa"]

    second = calculation_node({
        "query": "la seconda è Beta, premio 500 euro",
        "pending_calculation": _pending(first),
    })
    # The turn only described Beta; Alfa must survive from state, not from
    # the model re-deriving it.
    assert [c["nome"] for c in _pending(second)["candidates"]] == ["Alfa", "Beta"]
    assert _pending(second)["candidates"][0]["premio_annuo"] == "400"


def test_three_offers_accumulate_across_turns(monkeypatch):
    _mock_http(monkeypatch)
    _scripted_extractor(monkeypatch, {"Gamma": {"polizze": [_offer("Gamma", "600")]}})

    update = calculation_node({
        "query": "aggiungi Gamma a 600 euro",
        "pending_calculation": _collecting([_offer("Alfa", "400"), _offer("Beta", "500")]),
    })

    assert [c["nome"] for c in _pending(update)["candidates"]] == ["Alfa", "Beta", "Gamma"]
    assert "3" in update["answer"]


def test_offer_missing_a_required_field_is_held_as_a_draft_and_asked_about(monkeypatch):
    _mock_http(monkeypatch)
    _scripted_extractor(monkeypatch, {"Delta": {"polizze": [{"nome": "Delta"}]}})

    update = calculation_node({
        "query": "poi c'è Delta",
        "pending_calculation": _collecting([_offer("Alfa", "400")]),
    })

    pending = _pending(update)
    assert pending["candidates"] == [_offer("Alfa", "400")]  # not accepted yet
    assert pending["candidate_draft"] == {"nome": "Delta"}
    assert "premio_annuo" in update["answer"]


def test_a_draft_is_completed_by_the_next_message(monkeypatch):
    _mock_http(monkeypatch)
    _scripted_extractor(monkeypatch, {"450": {"polizze": [{"premio_annuo": "450"}]}})

    update = calculation_node({
        "query": "il premio è 450 euro",
        "pending_calculation": _collecting(
            [_offer("Alfa", "400")], candidate_draft={"nome": "Delta"}
        ),
    })

    assert [c["nome"] for c in _pending(update)["candidates"]] == ["Alfa", "Delta"]
    assert _pending(update)["candidate_draft"] == {}


def test_restating_an_offer_by_label_corrects_it_instead_of_duplicating(monkeypatch):
    _mock_http(monkeypatch)
    _scripted_extractor(monkeypatch, {
        "correggo": {"polizze": [_offer("Alfa", "380", copertura_kasko=True)]}
    })

    update = calculation_node({
        "query": "correggo Alfa: premio 380 euro, con kasko",
        "pending_calculation": _collecting([_offer("Alfa", "400"), _offer("Beta", "500")]),
    })

    pending = _pending(update)
    assert [c["nome"] for c in pending["candidates"]] == ["Alfa", "Beta"]
    assert pending["candidates"][0]["premio_annuo"] == "380"
    assert pending["candidates"][0]["copertura_kasko"] is True
    assert "Aggiornata" in update["answer"]


def test_an_offer_can_be_removed_by_label(monkeypatch):
    _mock_http(monkeypatch)
    seen = _scripted_extractor(monkeypatch, {})

    update = calculation_node({
        "query": "rimuovi Beta",
        "pending_calculation": _collecting([_offer("Alfa", "400"), _offer("Beta", "500")]),
    })

    assert [c["nome"] for c in _pending(update)["candidates"]] == ["Alfa"]
    assert "Rimossa" in update["answer"]
    assert seen == []  # a command, not something to extract from


def test_removing_an_unknown_offer_says_so_and_keeps_the_set(monkeypatch):
    _mock_http(monkeypatch)
    _scripted_extractor(monkeypatch, {})

    update = calculation_node({
        "query": "rimuovi Omega",
        "pending_calculation": _collecting([_offer("Alfa", "400"), _offer("Beta", "500")]),
    })

    assert len(_pending(update)["candidates"]) == 2
    assert "Omega" in update["answer"]


def test_finishing_with_one_offer_asks_for_another_instead_of_calculating(monkeypatch):
    calls = _mock_http(monkeypatch)
    _scripted_extractor(monkeypatch, {})

    update = calculation_node({
        "query": "confronta",
        "pending_calculation": _collecting([_offer("Alfa", "400")]),
    })

    assert calls == []
    assert "almeno 2" in update["answer"] or "2" in update["answer"]
    assert _pending(update)["phase"] == "collect_candidates"


def test_finishing_presents_a_structured_review_before_calculating(monkeypatch):
    calls = _mock_http(monkeypatch)
    _scripted_extractor(monkeypatch, {})

    update = calculation_node({
        "query": "confronta",
        "pending_calculation": _collecting([_offer("Alfa", "400"), _offer("Beta", "500")]),
    })

    assert calls == []  # review first, calculation only after confirmation
    assert _pending(update)["phase"] == "review"
    answer = update["answer"]
    assert "eta_conducente" in answer
    assert "Alfa" in answer and "Beta" in answer
    assert "premio_annuo=400" in answer


def test_confirmation_is_required_when_scoring_defaults_were_applied(monkeypatch):
    calls = _mock_http(monkeypatch, _comparison_result(provisional=True),
                       _comparison_result(provisional=True))
    _scripted_extractor(monkeypatch, {})

    review = _collecting([_offer("Alfa", "400"), _offer("Beta", "500")], phase="review")
    first = calculation_node({"query": "confermo", "pending_calculation": review})

    # The platform ran, but a provisional result is not shown as final.
    assert calls[0]["json"].get("confirm_assumptions") is None
    assert _pending(first)["phase"] == "confirm"
    assert "polizze[0].franchigia" in first["answer"]

    second = calculation_node({"query": "confermo", "pending_calculation": _pending(first)})
    assert calls[1]["json"]["confirm_assumptions"] is True
    assert second["pending_calculation"] is None
    assert "Alfa" in second["answer"]


def test_a_non_provisional_comparison_answers_straight_after_review(monkeypatch):
    calls = _mock_http(monkeypatch, _comparison_result(provisional=False))
    _scripted_extractor(monkeypatch, {})

    review = _collecting([_offer("Alfa", "400"), _offer("Beta", "500")], phase="review")
    update = calculation_node({"query": "confermo", "pending_calculation": review})

    assert len(calls) == 1
    assert update["pending_calculation"] is None
    assert "Vincitore chiaro" in update["answer"]


def test_more_than_twenty_offers_is_refused_with_a_clear_message(monkeypatch):
    _mock_http(monkeypatch)
    _scripted_extractor(monkeypatch, {"Extra": {"polizze": [_offer("Extra", "999")]}})

    twenty = [_offer(f"Offerta {n}", str(400 + n)) for n in range(20)]
    update = calculation_node({
        "query": "aggiungi anche Extra a 999 euro",
        "pending_calculation": _collecting(twenty),
    })

    pending = _pending(update)
    assert len(pending["candidates"]) == 20
    assert "20" in update["answer"]
    # The offer is not lost: it waits for a slot.
    assert pending["candidate_draft"]["nome"] == "Extra"


def test_a_freed_slot_admits_the_held_offer(monkeypatch):
    _mock_http(monkeypatch)
    _scripted_extractor(monkeypatch, {})

    twenty = [_offer(f"Offerta {n}", str(400 + n)) for n in range(20)]
    update = calculation_node({
        "query": "rimuovi Offerta 3",
        "pending_calculation": _collecting(
            twenty, candidate_draft=_offer("Extra", "999")
        ),
    })

    names = [c["nome"] for c in _pending(update)["candidates"]]
    assert "Offerta 3" not in names
    assert "Extra" in names
    assert len(names) == 20


def test_topic_change_during_collection_escapes_to_normal_rag(monkeypatch):
    calls = _mock_http(monkeypatch)
    _scripted_extractor(monkeypatch, {})  # nothing in the message belongs here

    update = calculation_node({
        "query": "in realtà, cosa dice l'art. 1284 c.c. sugli interessi legali?",
        "pending_calculation": _collecting([_offer("Alfa", "400")]),
    })

    assert update["calc_route"] == "normal"
    assert update["pending_calculation"] is None
    assert "answer" not in update
    assert calls == []


def test_candidate_turns_do_not_consume_the_clarification_round_budget(monkeypatch):
    _mock_http(monkeypatch)
    _scripted_extractor(monkeypatch, {
        "Beta": {"polizze": [_offer("Beta", "500")]},
        "Gamma": {"polizze": [_offer("Gamma", "600")]},
        "Delta": {"polizze": [_offer("Delta", "700")]},
        "Epsilon": {"polizze": [_offer("Epsilon", "800")]},
    })

    pending = _collecting([_offer("Alfa", "400")])
    for name, premium in (("Beta", 500), ("Gamma", 600), ("Delta", 700), ("Epsilon", 800)):
        update = calculation_node({
            "query": f"{name} a {premium} euro",
            "pending_calculation": pending,
        })
        pending = _pending(update)

    # Five offers over five turns — well past the three-round limit that
    # governs ordinary missing-input clarifications.
    assert len(pending["candidates"]) == 5
    assert pending["round"] == 0


def test_llm_unavailable_never_invents_an_offer_from_stray_numbers(monkeypatch):
    _mock_http(monkeypatch)
    monkeypatch.setattr(calculation, "_extract_values_llm", lambda *a, **k: None)

    update = calculation_node({
        "query": "450 e 150",
        "pending_calculation": _collecting([_offer("Alfa", "400")]),
    })

    pending = _pending(update)
    assert pending["candidates"] == [_offer("Alfa", "400")]  # nothing added
    assert pending["candidate_draft"] == {}
    assert "campo: valore" in update["answer"]
    assert "premio_annuo" in update["answer"]


def test_llm_unavailable_still_accepts_the_structured_form(monkeypatch):
    _mock_http(monkeypatch)
    monkeypatch.setattr(calculation, "_extract_values_llm", lambda *a, **k: None)

    update = calculation_node({
        "query": "nome: Beta, premio_annuo: 500, copertura_kasko: si",
        "pending_calculation": _collecting([_offer("Alfa", "400")]),
    })

    candidates = _pending(update)["candidates"]
    assert [c["nome"] for c in candidates] == ["Alfa", "Beta"]
    assert candidates[1]["premio_annuo"] == "500"
    assert candidates[1]["copertura_kasko"] is True


def test_pending_comparison_survives_a_session_round_trip(chat_classes):
    _, ChatSession, _ = chat_classes

    pending = _collecting([_offer("Alfa", "400"), _offer("Beta", "500")], phase="review")
    session = ChatSession(session_id="comparison-session")
    session.add_message("assistant", "Confermi?", metadata={"pending_calculation": pending})

    restored = ChatSession.from_dict(json.loads(json.dumps(session.to_dict())))
    from src.chatbot.session import last_pending_calculation

    carried = last_pending_calculation(restored)
    assert carried == pending
    # Every phase field survives, not just the calculator id.
    assert carried["phase"] == "review"
    assert carried["candidate_descriptor"]["required_fields"] == ["nome", "premio_annuo"]
    assert [c["nome"] for c in carried["candidates"]] == ["Alfa", "Beta"]


def test_full_mocked_production_comparison_conversation(monkeypatch):
    """The whole flow one turn at a time, exactly as the graph drives it."""
    calls = _mock_http(
        monkeypatch,
        _comparison_result(provisional=True),
        _comparison_result(provisional=True),
    )
    _scripted_extractor(monkeypatch, {
        "quale polizza": {},
        "40 anni": {"eta_conducente": 40},
        "Alfa": {"polizze": [_offer("Alfa", "400")]},
        "Beta": {"polizze": [_offer("Beta", "500")]},
    })

    state = {"query": "quale polizza mi conviene", "calculation_match": _polizze_match()}
    update = calculation_node(state)
    assert _pending(update)["phase"] == "collect_shared_inputs"

    for message in ("ho 40 anni", "Alfa a 400 euro", "Beta a 500 euro"):
        update = calculation_node({"query": message, "pending_calculation": _pending(update)})
    assert _pending(update)["phase"] == "collect_candidates"
    assert len(_pending(update)["candidates"]) == 2

    update = calculation_node({"query": "confronta", "pending_calculation": _pending(update)})
    assert _pending(update)["phase"] == "review"

    update = calculation_node({"query": "confermo", "pending_calculation": _pending(update)})
    assert _pending(update)["phase"] == "confirm"

    update = calculation_node({"query": "confermo", "pending_calculation": _pending(update)})
    assert update["pending_calculation"] is None
    answer = update["answer"]
    assert "Vincitore chiaro secondo il modello di punteggio configurato: Alfa" in answer
    # Money before the synthetic score, and the score's relativity stated.
    assert answer.index("costo stimato 400.00") < answer.index("punteggio 85.00/100")
    assert "non è una misura oggettiva del mercato" in answer
    assert "Non incluso" in answer
    assert len(calls) == 2


def test_effective_tie_answer_never_names_a_best_offer(monkeypatch):
    _mock_http(monkeypatch, _comparison_result(
        provisional=False, status="effective_tie", tie=["Alfa", "Beta"]
    ))
    _scripted_extractor(monkeypatch, {})

    review = _collecting([_offer("Alfa", "400"), _offer("Beta", "500")], phase="review")
    update = calculation_node({"query": "confermo", "pending_calculation": review})

    answer = update["answer"]
    assert "Sostanziale parità tra Alfa, Beta" in answer
    assert "Vincitore chiaro" not in answer
    assert "nessuna differenza materiale" in answer.lower()


@pytest.mark.parametrize("lang,marker", [
    ("it", "Vincitore chiaro"),
    ("es", "Ganador claro"),
    ("en", "Clear winner under the configured scoring model"),
])
def test_comparison_answer_is_aligned_across_languages(lang, marker):
    answer = calculation._success_answer(lang, _comparison_result(provisional=True))
    assert marker in answer
    assert calculation._COPY[lang]["exclusions"] in answer
    assert "Il massimale non entra nel punteggio." in answer
    assert calculation._COMPARISON_COPY[lang]["relative_note"] in answer
    # A provisional comparison says so in every language.
    assert calculation._COMPARISON_COPY[lang]["provisional"].split("{")[0].strip() in answer


# --- Ambiguous routing -------------------------------------------------------


def test_strong_ambiguity_asks_the_user_to_choose(monkeypatch):
    _mock_http(monkeypatch, {
        "status": "ambiguous",
        "candidates": [
            {"calculator_id": "business.confronto_polizze", "name": "Confronto polizze", "score": 4},
            {"calculator_id": "business.confronto_gas_luce", "name": "Confronto gas e luce", "score": 4},
        ],
    })

    result = calculation_gate({"query": "confronta queste due offerte e dimmi quale conviene"})

    assert result["calc_route"] == "calculate"
    assert result["calculation_match"] is None
    assert [c["calculator_id"] for c in result["calculation_choices"]] == [
        "business.confronto_polizze", "business.confronto_gas_luce",
    ]


def test_weak_ambiguity_still_falls_back_to_normal_rag_without_prompting(monkeypatch):
    # Two calculators scraping one incidental token each is not evidence
    # that the user asked for a calculation at all.
    _mock_http(monkeypatch, {
        "status": "ambiguous",
        "candidates": [
            {"calculator_id": "legal_it.furto_pena_draft", "score": 2},
            {"calculator_id": "legal_it.rapina_pena_draft", "score": 2},
        ],
    })

    assert calculation_gate({"query": "furto e rapina"}) == {"calc_route": "normal"}


def test_choice_question_is_asked_and_persisted(monkeypatch):
    _mock_http(monkeypatch)
    _scripted_extractor(monkeypatch, {})

    update = calculation_node({
        "query": "confronta queste offerte",
        "calculation_choices": [
            {"calculator_id": "business.confronto_polizze", "name": "Confronto polizze"},
            {"calculator_id": "business.confronto_gas_luce", "name": "Confronto gas e luce"},
        ],
    })

    pending = _pending(update)
    assert pending["phase"] == "choose_calculator"
    assert len(pending["choices"]) == 2
    assert "1)" in update["answer"] and "2)" in update["answer"]


@pytest.mark.parametrize("answer", ["1", "business.confronto_polizze", "Confronto polizze"])
def test_a_choice_is_accepted_by_number_id_or_name(monkeypatch, answer):
    _mock_http(monkeypatch)
    _scripted_extractor(monkeypatch, {})

    pending = {
        "phase": "choose_calculator",
        "calculator_id": None,
        "choices": [_polizze_match(), {"calculator_id": "business.confronto_gas_luce",
                                       "name": "Confronto gas e luce"}],
        "raw_query": "confronta Alfa e Beta",
        "round": 0,
    }
    pending["choices"][0]["name"] = "Confronto polizze"

    update = calculation_node({"query": answer, "pending_calculation": pending})

    assert _pending(update)["calculator_id"] == "business.confronto_polizze"


def test_an_unrelated_reply_to_the_choice_returns_to_normal_rag(monkeypatch):
    _mock_http(monkeypatch)
    _scripted_extractor(monkeypatch, {})

    update = calculation_node({
        "query": "lascia stare, parlami invece della prescrizione",
        "pending_calculation": {
            "phase": "choose_calculator",
            "calculator_id": None,
            "choices": [
                {"calculator_id": "business.confronto_polizze", "name": "Confronto polizze"},
                {"calculator_id": "business.confronto_gas_luce", "name": "Confronto gas e luce"},
            ],
            "raw_query": "confronta",
            "round": 0,
        },
    })

    assert update["calc_route"] == "normal"
    assert update["pending_calculation"] is None


def test_finishing_while_a_shared_fact_is_missing_asks_for_it(monkeypatch):
    # "confronta" here is a request to proceed, not a change of subject:
    # escaping to ordinary RAG would abandon the offers already collected.
    calls = _mock_http(monkeypatch)
    _scripted_extractor(monkeypatch, {})

    pending = _collecting(
        [_offer("Alfa", "400"), _offer("Beta", "500")],
        phase="collect_shared_inputs",
        inputs_so_far={},
    )
    update = calculation_node({"query": "confronta", "pending_calculation": pending})

    assert calls == []
    assert _pending(update)["phase"] == "collect_shared_inputs"
    assert "eta_conducente" in update["answer"]
    assert len(_pending(update)["candidates"]) == 2


def test_a_finish_word_after_the_review_confirms_instead_of_re_reviewing(monkeypatch):
    calls = _mock_http(monkeypatch, _comparison_result(provisional=False))
    _scripted_extractor(monkeypatch, {})

    review = _collecting([_offer("Alfa", "400"), _offer("Beta", "500")], phase="review")
    update = calculation_node({"query": "calcola", "pending_calculation": review})

    assert len(calls) == 1
    assert update["pending_calculation"] is None
    assert "Vincitore chiaro" in update["answer"]


# --- Regressions found by external review ---------------------------------


def test_structured_form_keeps_italian_decimal_commas(monkeypatch):
    """A value ran to the next comma, so "0,25" became "0". Zero is a valid
    price, so this silently invented a free offer and could hand it the win."""
    _mock_http(monkeypatch)
    monkeypatch.setattr(calculation, "_extract_values_llm", lambda *a, **k: None)

    update = calculation_node({
        "query": "nome: Beta, premio_annuo: 1.234,56, franchigia: 0,50",
        "pending_calculation": _collecting([_offer("Alfa", "400")]),
    })

    beta = _pending(update)["candidates"][1]
    assert beta["premio_annuo"] == "1234.56"
    assert beta["franchigia"] == "0.50"


def test_a_longer_name_is_a_new_offer_not_a_correction_of_a_shorter_one(monkeypatch):
    """"Alfa Plus" contains "Alfa", and substring matching swallowed the new
    product into the base one — the comparison lost a candidate silently."""
    _mock_http(monkeypatch)
    _scripted_extractor(monkeypatch, {"Plus": {"polizze": [_offer("Alfa Plus", "600")]}})

    update = calculation_node({
        "query": "aggiungi anche Alfa Plus a 600 euro",
        "pending_calculation": _collecting([_offer("Alfa", "400")]),
    })

    names = [c["nome"] for c in _pending(update)["candidates"]]
    assert names == ["Alfa", "Alfa Plus"]
    assert _pending(update)["candidates"][0]["premio_annuo"] == "400"


def test_a_complete_offer_does_not_swallow_an_unfinished_draft(monkeypatch):
    # Draft Delta merged field-by-field with a complete Beta, so Delta was
    # overwritten and disappeared without a word.
    _mock_http(monkeypatch)
    _scripted_extractor(monkeypatch, {"Beta": {"polizze": [_offer("Beta", "500")]}})

    update = calculation_node({
        "query": "intanto aggiungi Beta a 500 euro",
        "pending_calculation": _collecting(
            [_offer("Alfa", "400")], candidate_draft={"nome": "Delta"}
        ),
    })

    pending = _pending(update)
    assert [c["nome"] for c in pending["candidates"]] == ["Alfa", "Beta"]
    assert pending["candidate_draft"] == {"nome": "Delta"}  # not lost
    assert "Delta" in update["answer"]  # and the user is told it is pending


def test_correcting_an_offer_does_not_discard_an_unrelated_draft(monkeypatch):
    _mock_http(monkeypatch)
    _scripted_extractor(monkeypatch, {"correggo": {"polizze": [_offer("Alfa", "380")]}})

    update = calculation_node({
        "query": "correggo Alfa: premio 380 euro",
        "pending_calculation": _collecting(
            [_offer("Alfa", "400")], candidate_draft={"nome": "Delta"}
        ),
    })

    pending = _pending(update)
    assert pending["candidates"][0]["premio_annuo"] == "380"
    assert pending["candidate_draft"] == {"nome": "Delta"}


def test_offline_a_question_containing_a_number_is_not_read_as_a_shared_input(monkeypatch):
    """The positional extractor bound by order, not by name: while waiting
    for the driver's age, "articolo 40" was accepted as a 40-year-old."""
    _mock_http(monkeypatch)
    monkeypatch.setattr(calculation, "_extract_values_llm", lambda *a, **k: None)

    pending = _collecting([], phase="collect_shared_inputs", inputs_so_far={})
    update = calculation_node({
        "query": "cosa dice l'articolo 40 del codice penale?",
        "pending_calculation": pending,
    })

    carried = update.get("pending_calculation")
    assert carried is None or "eta_conducente" not in carried["inputs_so_far"]


def test_offline_a_repeated_off_topic_message_escapes_instead_of_trapping(monkeypatch):
    _mock_http(monkeypatch)
    monkeypatch.setattr(calculation, "_extract_values_llm", lambda *a, **k: None)

    pending = _collecting([_offer("Alfa", "400")])
    first = calculation_node({"query": "e l'articolo 1284 c.c.?", "pending_calculation": pending})
    # Explained once...
    assert "campo: valore" in first["answer"]

    second = calculation_node({
        "query": "dimmi della prescrizione decennale",
        "pending_calculation": _pending(first),
    })
    # ...then it lets go rather than asking forever.
    assert second["calc_route"] == "normal"
    assert second["pending_calculation"] is None


@pytest.mark.parametrize("reply", ["1Password", "1. no, dimmi altro", "12345"])
def test_a_number_must_be_the_whole_answer_to_select_a_choice(reply):
    choices = [
        {"calculator_id": "business.confronto_polizze", "name": "Confronto polizze"},
        {"calculator_id": "business.confronto_gas_luce", "name": "Confronto gas e luce"},
    ]
    assert calculation._resolve_choice(choices, reply) is None


@pytest.mark.parametrize("reply", ["1", "2", "1.", "2)"])
def test_a_bare_option_number_still_selects(reply):
    choices = [
        {"calculator_id": "business.confronto_polizze", "name": "Confronto polizze"},
        {"calculator_id": "business.confronto_gas_luce", "name": "Confronto gas e luce"},
    ]
    assert calculation._resolve_choice(choices, reply) is not None
