"""Qwen3's replies: reasoning, fences and prose around the JSON (src/rag/llm_reply.py), for every path through _call_chat."""
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

for key, value in (("NEO4J_URI", "bolt://localhost:7687"), ("NEO4J_USER", "neo4j"), ("NEO4J_PASSWORD", "x"),
                   ("LLM_API_KEY", "k"), ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

from src.rag import ai_chat  # noqa: E402
from src.rag.llm_reply import clean_json_text, parse_json_reply, strip_think  # noqa: E402
import src.rag.defensive_generation as dg  # noqa: E402

PLAN = '{"parte_assistita": "Tizio", "questioni": [{"titolo": "Rapina", "norme": ["art. 628 c.p."]}]}'
VALUE = json.loads(PLAN)
THINK = "<think>\nLa norma [art. 628] e {forse} altro: {\"bozza\": 1}.\n</think>\n\n"


@pytest.mark.parametrize("raw", [
    PLAN, f"<think>\n\n</think>\n\n{PLAN}", THINK + PLAN, f"```json\n{PLAN}\n```", f"<think>\n\n</think>\n\n```json\n{PLAN}\n```",
    THINK + f"Ecco il piano:\n```\n{PLAN}\n```\nFine.", f"<THINK></THINK>{PLAN}", f"ragiono ancora</think>{PLAN}",
    f"Vedi [1] e poi {PLAN}",
])
def test_the_outermost_json_is_read(raw):
    assert parse_json_reply(raw) == VALUE


def test_arrays_too_and_the_first_of_equals():
    assert parse_json_reply(f'<think>\n\n</think>\n["a", "b"]') == ["a", "b"]
    assert parse_json_reply('["a"]\n\n["b"]') == ["a"]


@pytest.mark.parametrize("raw", ["<think>solo ragionamento [1, 2], mai la risposta", "<think>\n\n</think>", "<think>{\"bozza\": 1}</think>Nessuno."])
def test_json_inside_the_reasoning_is_not_the_answer(raw):
    with pytest.raises(ValueError):
        parse_json_reply(raw)


def test_free_text_loses_its_reasoning_and_nothing_else():
    assert strip_think("<think>\n\n</think>\n\nIl parere.") == "Il parere."
    assert strip_think("<think>a</think>Uno <think>b</think>due") == "Uno due"
    assert strip_think("Il parere [1], ```x```.") == "Il parere [1], ```x```."
    assert strip_think(None) == ""
    assert clean_json_text("<think></think>```json\n[1]\n```") == "[1]"


class _Model:
    def __init__(self, content, finish="stop"):
        self.reply = SimpleNamespace(content=content, response_metadata={"finish_reason": finish})

    def bind(self, **kwargs):
        return self

    def bind_tools(self, *args, **kwargs):
        return self

    def invoke(self, messages):
        return self.reply


def test_call_chat_strips_every_reply_keeps_the_raw_for_the_dump_and_logs_it(monkeypatch, caplog):
    monkeypatch.setattr(ai_chat, "chat_model", _Model("<think>\n\n</think>\n\nIl parere."))
    with caplog.at_level("INFO", logger=ai_chat.logger.name):
        assert ai_chat._call_chat([]) == "Il parere."
    assert "reasoning removed" in caplog.text
    assert ai_chat.last_reply.raw.startswith("<think>") and ai_chat.last_reply.finish_reason == "stop"
    caplog.clear()
    monkeypatch.setattr(ai_chat, "chat_model", _Model("Pulito."))
    with caplog.at_level("INFO", logger=ai_chat.logger.name):
        assert ai_chat._call_chat([]) == "Pulito."
    assert "reasoning removed" not in caplog.text


def test_a_reply_empty_after_stripping_is_a_warning_with_its_finish_reason(monkeypatch, caplog):
    monkeypatch.setattr(ai_chat, "chat_model", _Model("<think>lungo ragionamento", "length"))
    assert ai_chat._call_chat([]) == ""
    assert "finish_reason=length" in caplog.text


def test_tool_replies_lose_it_too(monkeypatch):
    monkeypatch.setattr(ai_chat, "chat_model", _Model("<think>x</think>Testo"))
    assert ai_chat._call_chat_with_tools([], []).content == "Testo"


def test_chat_json_needs_no_retry_for_a_wrapped_reply(monkeypatch):
    calls = []
    monkeypatch.setattr(dg, "_call_chat", lambda *a, **k: calls.append(1) or strip_think(THINK + f"```json\n{PLAN}\n```"))
    assert dg._chat_json([SimpleNamespace(content="s")], max_tokens=10) == VALUE and len(calls) == 1


def _report_module():
    path = Path(__file__).resolve().parent.parent / "scripts" / "plan_fallback_report.py"
    spec = importlib.util.spec_from_file_location("plan_fallback_report", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _case(root, name, events, dumps):
    cases = root / "cases"
    (cases / name / "llm").mkdir(parents=True)
    (cases / f"{name}.json").write_text(json.dumps({"events": events}), encoding="utf-8")
    for i, (stage, reply, status, finish) in enumerate(dumps, 1):
        (cases / name / "llm" / f"{i:02d}_{stage}.txt").write_text(
            f"stage: {stage}\nmax_tokens: 2000\nstatus: {status}\nfinish_reason: {finish}\n\n===== SYSTEM =====\ns\n\n"
            f"===== REPLY =====\n{reply}\n", encoding="utf-8")


DRAFT = ["case_analysis_draft_llm", None, 5]


def test_the_report_tells_the_causes_apart(tmp_path):
    good = [["case_analysis_plan", {"issues": 2, "removed": []}, 5]]
    fragment = '{"parte_assistita": "Tizio", "risposte": [{"quesito": "q", "risposta": "r"}], "questioni": [{"titolo": "tagl'
    _case(tmp_path, "ok", good, [("plan", PLAN, "ok", "stop"), ("section_a", "x", "ok", "stop"),
                                 ("draft", "un controllo senza stadio, non una bozza", "ok", "stop")])
    _case(tmp_path, "cut", [DRAFT], [("plan", fragment, "ok", "?")])
    _case(tmp_path, "empty", [DRAFT], [("plan", "", "ok", "stop")])
    _case(tmp_path, "rejected", [["case_analysis_plan", {"issues": 0, "removed": ["art. 9"]}, 5], DRAFT],
          [("plan", PLAN, "ok", "stop")])
    _case(tmp_path, "skipped", [DRAFT], [("analysis", "x", "ok", "stop")])
    _case(tmp_path, "later", good + [["plan_then_write_fallback", {"step": "section call", "error": "empty reply"}, None], DRAFT],
          [("plan", f"<think>\n\n</think>{PLAN}", "ok", "stop"), ("section_a", "", "ok", "length")])
    _case(tmp_path, "older", good + [DRAFT], [("plan", PLAN, "ok", "stop"), ("section_a", "", "ok", "stop")])
    _case(tmp_path, "safe", good + [["post_writing_step_failed", {"step": "writer_review", "error": "boom"}, None]],
          [("plan", PLAN, "ok", "stop"), ("section_a", "x", "ok", "stop")])
    text = "\n".join(_report_module().report(str(tmp_path), parse_json_reply))
    assert "8 cases: 2 written from a plan, 6 in one call" in text and "--- ok" not in text and "--- safe" not in text
    assert "--- cut: plan reply cut at max_tokens" in text and "; cut" in text
    assert "--- empty: plan reply has no JSON" in text and "new parser reads it: no" in text
    assert "--- rejected: plan complete, no usable issue" in text and "art. 9" in text
    assert "--- skipped: no plan call" in text
    assert "--- later: after the plan: section call" in text and "fallback at section call: empty reply" in text
    assert "--- older: after the plan: section call" in text
    assert "writer_review: boom" in text and "think block 1" in text and "labelled draft" in text


# --- the plan and the sections -----------------------------------------------------------------------------------

import src.rag.plan_then_write as ptw  # noqa: E402
from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402

MESSAGES = [SystemMessage(content="s"), HumanMessage(content="h")]


def _replies(monkeypatch, *replies):
    """dg._call_chat answering in turn (text, finish_reason); the messages each call got."""
    seen, queue = [], list(replies)

    def fake(messages, **kwargs):
        seen.append((messages, kwargs))
        text, finish = queue.pop(0)
        ai_chat.last_reply.finish_reason = finish
        return text
    monkeypatch.setattr(dg, "_call_chat", fake)
    return seen


def test_the_issues_come_first_and_a_plan_has_room():
    system = ptw._plan_system()
    assert system.index('"questioni"') < system.index('"parte_assistita"') and "Sii conciso" in system
    assert ptw._PLAN_TOKENS == 4000
    assert '"norme": ["art. N codice"], ' in system


def test_a_cut_plan_is_never_read_and_asked_once_more_shorter(monkeypatch):
    events = []
    monkeypatch.setattr(dg, "vlog", lambda name, data=None, **k: events.append((name, data)))
    seen = _replies(monkeypatch, ('{"a": 1, "questioni": [{"titolo": "t', "length"), (PLAN, "stop"))
    assert ptw._plan_json(MESSAGES) == VALUE
    assert len(seen) == 2 and "più conciso" in seen[1][0][0].content and seen[0][1]["max_tokens"] == 4000
    assert [d["finish_reason"] for n, d in events if n == "case_analysis_plan_call"] == ["length", "stop"]


def test_a_fragment_from_a_cut_plan_is_not_taken_for_the_plan(monkeypatch):
    seen = _replies(monkeypatch, ('{"risposte": [{"quesito": "q"}], "questioni": [{"titolo": "tagl', "length"), (PLAN, "stop"))
    assert ptw._plan_json(MESSAGES) == VALUE and len(seen) == 2


def test_a_plan_cut_twice_fails_after_one_retry(monkeypatch):
    seen = _replies(monkeypatch, ('{"a": "x', "length"), ('{"a": "x', "length"))
    with pytest.raises(ValueError, match="twice"):
        ptw._plan_json(MESSAGES)
    assert len(seen) == 2                    # the stricter-JSON retry cannot shorten a reply


def test_a_plan_that_finishes_takes_one_call(monkeypatch):
    seen = _replies(monkeypatch, (PLAN, "stop"))
    assert ptw._plan_json(MESSAGES) == VALUE and len(seen) == 1


def test_an_empty_section_is_tried_again_and_a_second_says_where(monkeypatch):
    events = []
    monkeypatch.setattr(dg, "vlog", lambda name, data=None, **k: events.append((name, data)))
    jobs = [("Fatti", "scrivi", "materiale", 100)]
    plan = {"parte_assistita": "Tizio", "questioni": []}
    monkeypatch.setattr(ptw, "_section_system", lambda *a: "s")
    _replies(monkeypatch, ("", "stop"), ("", "length"))
    assert ptw.write_sections(plan, jobs, "it", [], [], []) is None
    fallback = [d for n, d in events if n == "plan_then_write_fallback"]
    assert fallback[0]["step"] == "section call" and "Fatti" in fallback[0]["error"] and "finish_reason=length" in fallback[0]["error"]
    assert [d["finish_reason"] for n, d in events if n == "case_analysis_section_call"] == ["stop", "length"]
