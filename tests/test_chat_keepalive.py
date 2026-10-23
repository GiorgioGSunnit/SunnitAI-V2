"""/api/chat kept alive past Vercel's 120 seconds (CHAT_KEEPALIVE_ENABLED).

Prod, 23 Oct 2026: a strategy took 142 s on the server and the page, behind a Vercel rewrite
that cancels a request after 120 s without data, showed "Si è verificato un errore durante la
richiesta". With the flag on, a reply not ready after the grace period is streamed: spaces
while it is prepared, then the usual JSON, which the page parses unchanged.
"""

import asyncio
import json
import time

import pytest
from fastapi import HTTPException
from fastapi.responses import StreamingResponse


@pytest.fixture
def chat_api(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("EMBEDDING_API_KEY", "test-key")
    import src.chatbot.api as api
    monkeypatch.setattr(api, "_KEEPALIVE_GRACE_SECONDS", 0.05)
    monkeypatch.setattr(api, "_KEEPALIVE_INTERVAL_SECONDS", 0.05)
    return api


class _Db:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


@pytest.fixture
def own_sessions(monkeypatch):
    """The sessions the wrapper opens for itself (src.db.base.SessionLocal)."""
    import src.db.base as base
    opened = []

    def fake_session_local():
        opened.append(_Db())
        return opened[-1]
    monkeypatch.setattr(base, "SessionLocal", fake_session_local)
    return opened


def _reply(api, answer="Ecco la strategia."):
    return api.ChatResponse(session_id="s1", answer=answer, original_query="q", resolved_query="q")


def _impl(api, delay, outcome, seen=None):
    async def fake_impl(request, current_user, db):
        if seen is not None:
            seen.append(db)
        await asyncio.sleep(delay)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome
    return fake_impl


def _request(api):
    return api.ChatRequest(message="Prepara una strategia difensiva.", session_id="s1")


async def _body(response):
    return [chunk async for chunk in response.body_iterator]


def test_flag_off_is_the_endpoint_as_before(chat_api, own_sessions, monkeypatch):
    monkeypatch.delenv("CHAT_KEEPALIVE_ENABLED", raising=False)
    seen, injected = [], object()
    monkeypatch.setattr(chat_api, "_chat_impl", _impl(chat_api, 0.2, _reply(chat_api), seen))
    result = asyncio.run(chat_api.chat(_request(chat_api), None, injected))
    assert isinstance(result, chat_api.ChatResponse) and result.answer == "Ecco la strategia."
    assert seen == [injected] and own_sessions == []                 # the injected session, as before


def test_a_quick_reply_is_returned_as_usual(chat_api, own_sessions, monkeypatch):
    monkeypatch.setenv("CHAT_KEEPALIVE_ENABLED", "true")
    monkeypatch.setattr(chat_api, "_chat_impl", _impl(chat_api, 0.0, _reply(chat_api)))
    result = asyncio.run(chat_api.chat(_request(chat_api), None, object()))
    assert isinstance(result, chat_api.ChatResponse)
    assert own_sessions[0].closed


def test_an_early_error_keeps_its_status_code(chat_api, own_sessions, monkeypatch):
    monkeypatch.setenv("CHAT_KEEPALIVE_ENABLED", "true")
    monkeypatch.setattr(chat_api, "_chat_impl",
                        _impl(chat_api, 0.0, HTTPException(status_code=404, detail="Session not found")))
    with pytest.raises(HTTPException) as caught:
        asyncio.run(chat_api.chat(_request(chat_api), None, object()))
    assert caught.value.status_code == 404


def test_a_slow_reply_is_streamed_spaces_then_the_json(chat_api, own_sessions, monkeypatch):
    monkeypatch.setenv("CHAT_KEEPALIVE_ENABLED", "true")
    monkeypatch.setattr(chat_api, "_chat_impl", _impl(chat_api, 0.25, _reply(chat_api)))

    async def run():
        response = await chat_api.chat(_request(chat_api), None, object())
        assert isinstance(response, StreamingResponse) and response.media_type == "application/json"
        return await _body(response)
    chunks = asyncio.run(run())
    assert chunks[0] == b" " and chunks.count(b" ") >= 3                # sent at once, then while waiting
    assert json.loads(b"".join(chunks)) == json.loads(_reply(chat_api).model_dump_json())
    assert own_sessions[0].closed


def test_a_late_failure_becomes_a_normal_error_reply(chat_api, own_sessions, monkeypatch):
    monkeypatch.setenv("CHAT_KEEPALIVE_ENABLED", "true")
    monkeypatch.setattr(chat_api, "_chat_impl", _impl(chat_api, 0.15, RuntimeError("model unreachable")))

    async def run():
        return await _body(await chat_api.chat(_request(chat_api), None, object()))
    body = json.loads(b"".join(asyncio.run(run())))
    assert body["answer"] == chat_api._LATE_ERROR_ANSWER and body["session_id"] == "s1"
    assert body["original_query"] == "Prepara una strategia difensiva."
    assert own_sessions[0].closed


def test_over_http_the_page_reads_the_streamed_reply_unchanged(chat_api, own_sessions, monkeypatch):
    from fastapi.testclient import TestClient
    from src.db.base import get_db

    monkeypatch.setenv("CHAT_KEEPALIVE_ENABLED", "true")
    monkeypatch.setattr(chat_api, "_chat_impl", _impl(chat_api, 0.2, _reply(chat_api)))
    chat_api.app.dependency_overrides[get_db] = lambda: None
    try:
        t0 = time.time()
        response = TestClient(chat_api.app).post("/api/chat", json={"message": "Strategia?", "session_id": "s1"})
    finally:
        chat_api.app.dependency_overrides.pop(get_db, None)
    assert response.status_code == 200 and time.time() - t0 >= 0.2
    assert response.text.startswith(" ") and response.json()["answer"] == "Ecco la strategia."
