import os
import time
import logging
from typing import Any, Optional

VERBOSE = os.getenv("VERBOSE_LOGGING", "false").lower() == "true"
logger = logging.getLogger("sunnitai.verbose")


def vlog(event: str, data: dict = None, duration_ms: float = None):
    """Log verbose event if VERBOSE_LOGGING=true."""
    if not VERBOSE:
        return
    parts = [f"[VERBOSE:{event}]"]
    if duration_ms is not None:
        parts.append(f"duration={duration_ms:.1f}ms")
    if data:
        for k, v in data.items():
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
