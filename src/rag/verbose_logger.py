import json
import os
import time
import logging
import traceback
from typing import Any, Optional

VERBOSE = os.getenv("VERBOSE_LOGGING", "false").lower() == "true"
logger = logging.getLogger("sunnitai.verbose")
# Printed whole, as JSON, instead of cut to 100 characters: the typed facts of a case (fact_signals), which a
# dry run cannot be read without (27 Oct).
FULL_KEYS = {"facts", "rules", "issues", "traceback"}
# Printed whatever VERBOSE_LOGGING says: an exception the pipeline caught, with its traceback (v45d: a run failed with
# only "'bool' object is not iterable" to go on).
ALWAYS_EVENTS = {"pipeline_exception"}


# What traces a run: scripts/diagnose_long.py's case logs carry these whatever VERBOSE_LOGGING says.
TRACE_EVENTS = {"fact_rules", "case_analysis_plan_final", "case_analysis_plan_call", "case_analysis_plan_remake",
                "llm_reply_cut", "plan_then_write_fallback", "pipeline_exception"}


def trace(event: str, data: dict = None, duration_ms: float = None):
    """For a path that captures vlog in place of printing it: the event printed all the same, if verbose or a trace."""
    if VERBOSE or event in TRACE_EVENTS:
        _emit(event, data, duration_ms)


def vlog(event: str, data: dict = None, duration_ms: float = None):
    """Log verbose event if VERBOSE_LOGGING=true (an ALWAYS_EVENTS one in any case)."""
    if VERBOSE or event in ALWAYS_EVENTS:
        _emit(event, data, duration_ms)


def exception_data(stage: str, exc: BaseException, **detail: Any) -> dict:
    """The pipeline_exception event's data for an exception being handled: where, what, and the full traceback."""
    return {"stage": stage, "error": f"{type(exc).__name__}: {exc}"[:500], "traceback": traceback.format_exc(), **detail}


def _emit(event: str, data: dict = None, duration_ms: float = None):
    parts = [f"[VERBOSE:{event}]"]
    if duration_ms is not None:
        parts.append(f"duration={duration_ms:.1f}ms")
    if data:
        for k, v in data.items():
            if k in FULL_KEYS:
                parts.append(f"{k}={json.dumps(v, ensure_ascii=False, default=str)}")
            else:
                parts.append(f"{k}={repr(v)[:100]}")
    msg = " | ".join(parts)
    logger.info(msg)
    print(msg, flush=True)  # ensure visibility in journalctl


class Timer:
    """Context manager for timing code blocks."""
    def __init__(self, event: str, data: dict = None):
        self.event = event
        self.data = data or {}
        self.start = None

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        duration_ms = (time.time() - self.start) * 1000
        vlog(self.event, self.data, duration_ms)
