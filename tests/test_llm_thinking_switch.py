"""LLM_THINKING=false (ai_chat.py): the chat model asks vLLM's chat template not to reason, so Qwen3's reasoning no
longer eats the answer's max_tokens (prod, 2-4 Nov: 5 cut answer calls in 8 chat requests). The setting must reach
the request body on every path the app uses: a plain call, a call with its own max_tokens (_call_chat binds it),
bind_tools (_call_chat_with_tools) and with_structured_output (the extractor); and it must send nothing when unset,
since OpenAI's API rejects the field."""

import os

os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test-password")
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

import pytest  # noqa: E402
from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402

import src.rag.ai_chat as ai_chat  # noqa: E402

NO_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}
MESSAGES = [SystemMessage(content="Sei un assistente."), HumanMessage(content="Ciao")]
TOOLS = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {}}}}]


def _model(setting):
    kwargs = {k: v for k, v in ai_chat._chat_kwargs.items() if k != "extra_body"}
    return ChatOpenAI(**{**kwargs, "base_url": "http://llm.test/v1", **ai_chat._thinking_kwargs(setting)})


@pytest.mark.parametrize("setting", ["false", "False", " FALSE ", "0", "no", "off"])
def test_the_switch_reads_every_spelling_of_no(setting):
    assert ai_chat._thinking_kwargs(setting) == {"extra_body": NO_THINKING}


@pytest.mark.parametrize("setting", [None, "", "true", "1", "yes", "default"])
def test_anything_else_keeps_the_models_default(setting):
    assert ai_chat._thinking_kwargs(setting) == {}


def test_off_reaches_the_request_body_with_the_calls_own_max_tokens():
    payload = _model("false")._get_request_payload(MESSAGES, max_tokens=250)
    assert payload["extra_body"] == NO_THINKING
    assert 250 in (payload.get("max_tokens"), payload.get("max_completion_tokens"))
    assert payload["messages"][-1]["content"] == "Ciao"


def test_unset_sends_no_extra_body():
    payload = _model(None)._get_request_payload(MESSAGES, max_tokens=250)
    assert "extra_body" not in payload


def test_off_survives_bind_tools_and_structured_output():
    model = _model("false")
    with_tools = model.bind_tools(TOOLS, tool_choice=None, parallel_tool_calls=False).bind(max_tokens=100)
    assert with_tools.bound.extra_body == NO_THINKING
    structured = model.with_structured_output({"title": "Out", "type": "object", "properties": {}})
    assert structured.first.bound.extra_body == NO_THINKING


def test_call_chat_sends_the_switch(monkeypatch):
    """The app's own path: _call_chat binds max_tokens and invokes; the body the OpenAI client would send must
    carry both. The client is replaced by a recorder that answers like vLLM does."""
    sent = {}

    class Raw:
        def parse(self):
            return {"id": "r", "model": "x", "object": "chat.completion",
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": "Buongiorno."}}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}}

    class Completions:
        """What langchain-openai 1.1 calls: client.with_raw_response.create(**payload).parse()."""
        with_raw_response = None

        def create(self, **kwargs):
            sent.update(kwargs)
            return Raw()

    Completions.with_raw_response = Completions()
    model = _model("false")
    model.client = Completions()
    monkeypatch.setattr(ai_chat, "chat_model", model)
    assert ai_chat._call_chat(MESSAGES, max_tokens=250) == "Buongiorno."
    assert sent["extra_body"] == NO_THINKING
    assert 250 in (sent.get("max_tokens"), sent.get("max_completion_tokens"))


def test_the_startup_notice_is_printed_so_the_journal_shows_it(capsys):
    """uvicorn leaves the root logger unconfigured, so only a print reaches journalctl (the 4 Nov deploy check
    counted 0 lines from the INFO version)."""
    assert ai_chat._announce_thinking("false") == "LLM_THINKING=false: the chat model is asked not to reason before answering"
    assert "LLM_THINKING=false" in capsys.readouterr().out
    assert ai_chat._announce_thinking(None) is None
    assert capsys.readouterr().out == ""
