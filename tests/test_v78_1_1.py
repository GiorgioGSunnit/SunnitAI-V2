"""v78.1.1: LLM_REASONING_EFFORT -- with thinking on, chat_template_kwargs["reasoning_effort"] beside enable_thinking when
the env is set, absent when it is not; --reasoning-effort on thesis_replay and nightly_sim; the thesis log records it."""

import importlib.util
import json
import os
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for name, value in (("LLM_API_KEY", "k"), ("OPENAI_API_KEY", "k"), ("EMBEDDING_PROVIDER", "openai"),
                        ("EMBEDDING_API_KEY", "k"), ("NEO4J_URI", "bolt://localhost:7687")):
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("LLM_REASONING_EFFORT", "")      # v79: set (empty = unset) so a script's override is undone


def _script(name):
    spec = importlib.util.spec_from_file_location(name, HERE.parent / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_key_is_absent_when_the_env_is_unset_and_beside_enable_thinking_when_set(monkeypatch):
    from src.rag import ai_chat
    unset = ai_chat.thinking_bind(True, 100, "per_call")
    assert unset["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}} and unset["max_tokens"] > 100
    monkeypatch.setenv("LLM_REASONING_EFFORT", " low ")
    got = ai_chat.thinking_bind(True, 100, "per_call")
    assert got["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "low"}}
    assert ai_chat._THINKING == {"chat_template_kwargs": {"enable_thinking": True}}        # the shared body untouched


def test_nothing_is_sent_without_thinking_on(monkeypatch):
    from src.rag import ai_chat
    monkeypatch.setenv("LLM_REASONING_EFFORT", "high")
    assert ai_chat.thinking_bind(False, 100, "per_call") == {}
    assert ai_chat.thinking_bind(True, 100, "false") == {} and ai_chat.thinking_bind(True, 100, "") == {}
    assert "reasoning_effort" not in json.dumps(ai_chat._thinking_kwargs("per_call"))   # the no-thinking body


def test_the_replay_sends_it_only_with_thinking_on():
    replay = _script("thesis_replay")
    assert replay.template_kwargs("on", "low") == {"chat_template_kwargs": {"enable_thinking": True,
                                                                            "reasoning_effort": "low"}}
    assert replay.template_kwargs("on", "") == {"chat_template_kwargs": {"enable_thinking": True}}
    assert replay.template_kwargs("off", "low") == {"chat_template_kwargs": {"enable_thinking": False}}
    assert replay.template_kwargs("default", "low") == {}


def test_the_replay_flag_overrides_the_env_and_is_recorded(monkeypatch, tmp_path):
    replay = _script("thesis_replay")
    monkeypatch.setenv("LLM_REASONING_EFFORT", "high")
    log = tmp_path / "thesis_c1.json"
    log.write_text(json.dumps({"case_id": "c1", "messages": [], "max_tokens": 100, "reasoning_effort": "high"}))
    seen = {}

    def call(log_, base_url, model, api_key, thinking, max_tokens, n=1, temperature=0.0, effort=""):
        seen["effort"] = effort
        return {"raw_reply": "{}", "parsed_best": {}, "samples": [], "best": 0, "finish_reason": "stop", "seconds": 0}
    monkeypatch.setattr(replay, "call", call)
    monkeypatch.setattr(replay, "check", lambda raw, log_: {"main": "", "main_is_fallback": False, "barred": [],
                                                           "theses": 0, "declared": 0, "unknown_ids": []})
    out = tmp_path / "ab.json"
    replay.main(["--log", str(log), "--base-url", "http://x/v1", "--model", "m", "--reasoning-effort", "low",
                 "--out", str(out)])
    got = json.loads(out.read_text())
    assert seen["effort"] == "low" and got["replay"]["reasoning_effort"] == "low"
    assert got["original"]["reasoning_effort"] == "high"


def test_the_nightly_flag_overrides_the_env_file_for_its_run(monkeypatch, tmp_path):
    nightly = _script("nightly_sim")
    env = tmp_path / ".env"
    env.write_text("LLM_REASONING_EFFORT=high\n")
    monkeypatch.setattr(nightly, "cells", lambda: [])
    monkeypatch.setattr(nightly, "stage_check", lambda: None)
    nightly.main(["--env", str(env), "--no-stages", "--reasoning-effort", "medium"])
    assert os.environ["LLM_REASONING_EFFORT"] == "medium"


def test_the_thesis_log_records_the_effort_next_to_thinking(monkeypatch, tmp_path):
    from src.rag import lawyer_method
    monkeypatch.setenv("THESIS_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("LLM_THINKING", "per_call")
    monkeypatch.setattr(lawyer_method, "thinking_supported", lambda: True)
    monkeypatch.setenv("LLM_REASONING_EFFORT", "low")
    entry = json.loads(Path(lawyer_method.thesis_log({"case_id": "c1"}, [], "{}", {}, {}, 1.0)).read_text())
    assert entry["thinking"] is True and entry["reasoning_effort"] == "low"
    monkeypatch.delenv("LLM_REASONING_EFFORT")
    entry = json.loads(Path(lawyer_method.thesis_log({"case_id": "c2"}, [], "{}", {}, {}, 1.0)).read_text())
    assert entry["reasoning_effort"] == ""
