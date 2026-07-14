import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from src.chatbot.auth import create_access_token
from src.chatbot.tracking_ws import queue_tracking_event, router, tracking_hub


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(router, prefix="/api")
    return TestClient(app)


def setup_function():
    tracking_hub.clear()


def teardown_function():
    tracking_hub.clear()


def test_tracking_websocket_delivers_queued_event_for_authenticated_user():
    user_id = "user_123"
    token = create_access_token({"sub": user_id})
    assert queue_tracking_event(user_id, {"event": "free_trial", "user_id": user_id, "event_id": "evt_trial"})

    with _client().websocket_connect(f"/api/tracking/ws?token={token}") as websocket:
        message = websocket.receive_json()
        assert message == {
            "type": "tracking_event",
            "payload": {"event": "free_trial", "user_id": user_id, "event_id": "evt_trial"},
        }

        websocket.send_json({"type": "ack", "event_id": "evt_trial"})
        for _ in range(20):
            if tracking_hub.pending_count(user_id) == 0:
                break
            time.sleep(0.01)

    assert tracking_hub.pending_count(user_id) == 0


def test_tracking_websocket_rejects_missing_token():
    with pytest.raises(WebSocketDisconnect):
        with _client().websocket_connect("/api/tracking/ws"):
            pass
