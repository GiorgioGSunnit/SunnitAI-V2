import asyncio
import json
import logging
import uuid
from collections import OrderedDict, defaultdict
from threading import RLock
from typing import Any, Optional

from fastapi import APIRouter, Header, HTTPException, Request, WebSocket, WebSocketDisconnect, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .auth import decode_access_token

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tracking", tags=["tracking"])
MAX_PENDING_EVENTS_PER_USER = 100

TrackingPayload = dict[str, Any]


class TrackingEventHub:
    def __init__(self, max_pending_events_per_user: int = MAX_PENDING_EVENTS_PER_USER):
        self.max_pending_events_per_user = max_pending_events_per_user
        self._pending: dict[str, OrderedDict[str, TrackingPayload]] = defaultdict(OrderedDict)
        self._connections: dict[str, set[WebSocket]] = defaultdict(set)
        self._lock = RLock()

    def queue_event(self, user_id: Any, payload: TrackingPayload) -> bool:
        user_id_str = _optional_str(user_id)
        if not user_id_str or not payload:
            return False

        event = dict(payload)
        event_id = _optional_str(event.get("event_id")) or uuid.uuid4().hex
        event["event_id"] = event_id
        event["user_id"] = _optional_str(event.get("user_id")) or user_id_str

        with self._lock:
            user_events = self._pending[user_id_str]
            user_events[event_id] = event
            while len(user_events) > self.max_pending_events_per_user:
                user_events.popitem(last=False)
            connections = list(self._connections.get(user_id_str, ()))

        if connections:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                pass
            else:
                for websocket in connections:
                    loop.create_task(self._send_event(user_id_str, websocket, event))

        return True

    async def connect(self, user_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        with self._lock:
            self._connections[user_id].add(websocket)
            pending = list(self._pending.get(user_id, {}).values())

        try:
            for event in pending:
                await self._send_event(user_id, websocket, event)

            while True:
                message = await websocket.receive_json()
                if message.get("type") == "ack":
                    self.ack_event(user_id, message.get("event_id"))
        except WebSocketDisconnect:
            pass
        finally:
            self.disconnect(user_id, websocket)

    def ack_event(self, user_id: Any, event_id: Any) -> bool:
        user_id_str = _optional_str(user_id)
        event_id_str = _optional_str(event_id)
        if not user_id_str or not event_id_str:
            return False
        with self._lock:
            user_events = self._pending.get(user_id_str)
            if not user_events or event_id_str not in user_events:
                return False
            del user_events[event_id_str]
            if not user_events:
                self._pending.pop(user_id_str, None)
            return True

    def disconnect(self, user_id: str, websocket: WebSocket) -> None:
        with self._lock:
            connections = self._connections.get(user_id)
            if not connections:
                return
            connections.discard(websocket)
            if not connections:
                self._connections.pop(user_id, None)

    def pending_events(self, user_id: Any) -> list[TrackingPayload]:
        user_id_str = _optional_str(user_id)
        if not user_id_str:
            return []
        with self._lock:
            return list(self._pending.get(user_id_str, {}).values())

    def pending_count(self, user_id: Any) -> int:
        user_id_str = _optional_str(user_id)
        if not user_id_str:
            return 0
        with self._lock:
            return len(self._pending.get(user_id_str, ()))

    def clear(self) -> None:
        with self._lock:
            self._pending.clear()
            self._connections.clear()

    async def _send_event(self, user_id: str, websocket: WebSocket, event: TrackingPayload) -> None:
        try:
            await websocket.send_json({"type": "tracking_event", "payload": event})
        except RuntimeError as exc:
            logger.warning("Tracking websocket send failed for user %s: %s", user_id, exc)
            self.disconnect(user_id, websocket)


tracking_hub = TrackingEventHub()


def _optional_str(value: Any) -> Optional[str]:
    if value is None or value == "":
        return None
    return str(value)


def queue_tracking_event(user_id: Any, payload: TrackingPayload) -> bool:
    return tracking_hub.queue_event(user_id, payload)


class TrackingAckRequest(BaseModel):
    event_id: str


def _user_id_from_bearer(authorization: Optional[str]) -> Optional[str]:
    if not authorization or not authorization.lower().startswith("bearer "):
        return None
    payload = decode_access_token(authorization.split(" ", 1)[1])
    return _optional_str(payload.get("sub")) if payload else None


def _sse_tracking_event(event: TrackingPayload) -> str:
    event_id = _optional_str(event.get("event_id")) or ""
    data = json.dumps(event, separators=(",", ":"))
    return f"id: {event_id}\nevent: tracking_event\ndata: {data}\n\n"


@router.get("/events")
async def tracking_events_sse(request: Request):
    token = request.query_params.get("token")
    payload = decode_access_token(token) if token else None
    user_id = _optional_str(payload.get("sub")) if payload else None
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")

    async def event_stream():
        while not await request.is_disconnected():
            pending = tracking_hub.pending_events(user_id)
            if pending:
                for event in pending:
                    yield _sse_tracking_event(event)
            else:
                yield ": keepalive\n\n"
            await asyncio.sleep(1)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/events/ack")
def ack_tracking_event(request: TrackingAckRequest, authorization: Optional[str] = Header(default=None)):
    user_id = _user_id_from_bearer(authorization)
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    return {"acknowledged": tracking_hub.ack_event(user_id, request.event_id)}


@router.websocket("/ws")
async def tracking_events_ws(websocket: WebSocket):
    token = websocket.query_params.get("token")
    payload = decode_access_token(token) if token else None
    user_id = payload.get("sub") if payload else None
    if not user_id:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await tracking_hub.connect(str(user_id), websocket)
