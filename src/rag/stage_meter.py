"""v60 (instrumentation): seconds, chat-model calls, embedding calls and graph queries per stage of one request.

    meter = stage_meter.Meter()
    with meter.active():
        meter.mark("select")            # every mark closes the stage before it
        ...
        meter.mark("grading")
        ...
    meter.stop(); meter.as_log()        # {"select_s": 12.3, "select_llm": 4, "select_graph": 31, ...}

A mark anywhere in the code (stage_meter.mark("fusion:distill")) acts on the request's active meter and is a no-op with
none, so library code can name its sub-steps without knowing who measures. The stages form a timeline: time and calls
between two marks go to the first; a name marked twice adds up.

v60h (11): each chat call also records itself (call(): the asking function, its seconds, input and output tokens,
its start from the meter's), so a stage's calls are listed one by one -- the retrieval line carries the analysis
stage's ("analysis_calls").

Counting: ai_chat._call_chat (and _call_chat_with_tools) calls count("llm"); the embedding model's embed_query and
embed_documents are wrapped once (count("embed")); graph queries are counted through counted(session) / counted(driver)
(each .run()). A call made on a worker thread (the graded check's batches) has no context of its own: it counts on the
meter last made active -- exact for one request at a time (the simulation), approximate under concurrent requests.
"""

import contextlib
import contextvars
import threading
import time
from typing import Any, Dict, List, Optional

_ACTIVE: "contextvars.ContextVar[Optional[Meter]]" = contextvars.ContextVar("stage_meter", default=None)
_LAST: List[Optional["Meter"]] = [None]
_LOCK = threading.Lock()
_INSTALLED = [False]


class Meter:
    def __init__(self):
        self.stages: Dict[str, Dict[str, float]] = {}
        self.order: List[str] = []
        self.current: Optional[str] = None
        self.started = time.time()
        self.t = self.started
        self.calls: List[Dict[str, Any]] = []

    def _bucket(self, name: str) -> Dict[str, float]:
        if name not in self.stages:
            self.stages[name] = {"s": 0.0, "llm": 0, "embed": 0, "graph": 0}
            self.order.append(name)
        return self.stages[name]

    def mark(self, name: str) -> None:
        now = time.time()
        with _LOCK:
            if self.current is not None:
                self._bucket(self.current)["s"] += now - self.t
            self.current, self.t = name, now
            self._bucket(name)

    def count(self, kind: str, n: int = 1) -> None:
        with _LOCK:
            self._bucket(self.current or "other")[kind] += n

    def call(self, name: str, seconds: float, tokens_in: Optional[int], tokens_out: Optional[int],
             started: Optional[float] = None) -> None:
        with _LOCK:
            self.calls.append({"stage": self.current or "other", "call": name, "s": round(seconds, 1),
                               "in": tokens_in, "out": tokens_out,
                               "at": round((started or time.time() - seconds) - self.started, 1)})

    def stage_calls(self, stage: str) -> List[Dict[str, Any]]:
        return [{k: v for k, v in c.items() if k != "stage"} for c in self.calls if c["stage"] == stage]

    def stop(self) -> None:
        self.mark("__end__")
        with _LOCK:
            self.stages.pop("__end__", None)
            if "__end__" in self.order:
                self.order.remove("__end__")
            self.current = None

    def total(self, kind: str) -> float:
        return sum(b[kind] for b in self.stages.values())

    def as_log(self) -> Dict[str, Any]:
        """Flat keys, in stage order: <stage>_s, and <stage>_llm / _embed / _graph when not 0."""
        out: Dict[str, Any] = {}
        for name in self.order:
            b = self.stages[name]
            key = name.replace(":", "_").replace(" ", "_")
            out[f"{key}_s"] = round(b["s"], 1)
            for kind in ("llm", "embed", "graph"):
                if b[kind]:
                    out[f"{key}_{kind}"] = int(b[kind])
        out["total_s"] = round(sum(b["s"] for b in self.stages.values()), 1)
        if "analysis" in self.stages:            # v60h (11): the analysis stage's calls, one by one
            out["analysis_calls"] = self.stage_calls("analysis")
            out["analysis_tokens_in"] = sum(c["in"] or 0 for c in out["analysis_calls"])
            out["analysis_tokens_out"] = sum(c["out"] or 0 for c in out["analysis_calls"])
        out["total_llm"], out["total_embed"], out["total_graph"] = (int(self.total("llm")), int(self.total("embed")),
                                                                    int(self.total("graph")))
        return out

    @contextlib.contextmanager
    def active(self):
        install()
        token = _ACTIVE.set(self)
        _LAST[0] = self
        try:
            yield self
        finally:
            _ACTIVE.reset(token)


@contextlib.contextmanager
def own(meter: "Meter"):
    """A side task's own meter (a thread of its own: the first read), never the request's fallback (_LAST)."""
    token = _ACTIVE.set(meter)
    try:
        yield meter
    finally:
        _ACTIVE.reset(token)


def current() -> Optional[Meter]:
    return _ACTIVE.get() or _LAST[0]


def mark(name: str) -> None:
    meter = _ACTIVE.get()
    if meter is not None:
        meter.mark(name)


def call(name: str, seconds: float, tokens_in: Optional[int] = None, tokens_out: Optional[int] = None,
         started: Optional[float] = None) -> None:
    """v60h (11): one chat call on the request's meter (the active one, else the last made active); never fails."""
    meter = current()
    if meter is not None:
        try:
            meter.call(name, seconds, tokens_in, tokens_out, started)
        except Exception:
            pass


def count(kind: str, n: int = 1) -> None:
    meter = current()
    if meter is not None:
        try:
            meter.count(kind, n)
        except Exception:
            pass


class _CountedSession:
    def __init__(self, inner):
        self._inner = inner

    def run(self, *args, **kwargs):
        count("graph")
        return self._inner.run(*args, **kwargs)

    def __enter__(self):
        entered = self._inner.__enter__()
        return _CountedSession(entered)

    def __exit__(self, *exc):
        return self._inner.__exit__(*exc)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _CountedDriver:
    def __init__(self, inner):
        self._inner = inner

    def session(self, *args, **kwargs):
        return _CountedSession(self._inner.session(*args, **kwargs))

    def execute_query(self, *args, **kwargs):
        count("graph")
        return self._inner.execute_query(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def counted(obj):
    """A graph session or driver whose queries count on the active meter; the object itself otherwise unchanged."""
    if isinstance(obj, (_CountedSession, _CountedDriver)) or obj is None:
        return obj
    return _CountedDriver(obj) if hasattr(obj, "session") and not hasattr(obj, "run") else _CountedSession(obj)


def install() -> None:
    """Once: the embedding model's calls counted (the chat calls count in ai_chat itself)."""
    if _INSTALLED[0]:
        return
    _INSTALLED[0] = True
    try:
        from . import ai_chat
        model = ai_chat.embedding_model
        for name in ("embed_query", "embed_documents"):
            real = getattr(model, name)

            def wrapped(*args, _real=real, **kwargs):
                count("embed")
                return _real(*args, **kwargs)
            object.__setattr__(model, name, wrapped)
    except Exception:
        pass
