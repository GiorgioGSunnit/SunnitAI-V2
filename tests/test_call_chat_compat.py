"""Guards against the 17 Oct prod incident: defensive_generation.py started
passing temperature=0 to the chat call before ai_chat.py's matching change
had deployed everywhere it runs, and every case-analysis LLM call crashed
with "_call_chat() got an unexpected keyword argument 'temperature'" -- on
the flag-off path too, since _analyse_case (v1) also passes it.

_chat() (defensive_generation.py) now drops any kwarg the resolved
_call_chat doesn't accept instead of crashing. This file pins that
behaviour, plus a static check that no call site in the module passes
_call_chat itself a kwarg it doesn't accept.
"""

import ast
import inspect
import os

os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test-password")
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

import src.rag.ai_chat as ai_chat  # noqa: E402
import src.rag.defensive_generation as dg  # noqa: E402


def test_chat_drops_a_kwarg_a_stale_call_chat_does_not_accept(monkeypatch):
    """Simulates a partial deploy: ai_chat.py on prod still has the old
    signature, without `temperature`."""
    seen = {}

    def old_call_chat(messages, max_tokens=None, stop=None):
        seen["kwargs"] = {"max_tokens": max_tokens, "stop": stop}
        return "ok"

    monkeypatch.setattr(dg, "_call_chat", old_call_chat)
    monkeypatch.setattr(dg, "_CALL_CHAT_PARAMS", frozenset(inspect.signature(old_call_chat).parameters))
    assert dg._chat([], max_tokens=10, temperature=0) == "ok"
    assert seen["kwargs"] == {"max_tokens": 10, "stop": None}


def test_chat_passes_every_kwarg_a_current_call_chat_accepts(monkeypatch):
    seen = {}

    def new_call_chat(messages, max_tokens=None, stop=None, temperature=None):
        seen["kwargs"] = {"max_tokens": max_tokens, "stop": stop, "temperature": temperature}
        return "ok"

    monkeypatch.setattr(dg, "_call_chat", new_call_chat)
    monkeypatch.setattr(dg, "_CALL_CHAT_PARAMS", frozenset(inspect.signature(new_call_chat).parameters))
    dg._chat([], max_tokens=10, temperature=0)
    assert seen["kwargs"]["temperature"] == 0


def test_no_call_site_passes_call_chat_a_kwarg_it_does_not_accept():
    """Static scan of defensive_generation.py: every direct _call_chat(...)
    call (not routed through _chat) must only use kwargs the real _call_chat
    signature accepts, so a reintroduced mismatch fails at test time, not on
    prod."""
    accepted = set(inspect.signature(ai_chat._call_chat).parameters)
    with open(dg.__file__, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_call_chat":
            for kw in node.keywords:
                if kw.arg and kw.arg not in accepted:
                    bad.append(kw.arg)
    assert bad == [], f"defensive_generation.py calls _call_chat directly with unsupported kwargs: {bad}"
