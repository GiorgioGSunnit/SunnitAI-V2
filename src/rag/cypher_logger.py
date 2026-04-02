"""Append-only Cypher pipeline log with size-based archival (timestamped suffix)."""

from __future__ import annotations

import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

_lock = threading.Lock()

_DEFAULT_MAX_BYTES = 5 * 1024 * 1024  # 5 MiB


def _project_root() -> str:
    """Resolve deploy directory (folder that contains pyproject.toml), not site-packages."""
    env = os.getenv("CHATBOT_PROJECT_ROOT")
    if env:
        return os.path.abspath(env)
    here = os.path.abspath(os.path.dirname(__file__))
    p = here
    while True:
        if os.path.isfile(os.path.join(p, "pyproject.toml")):
            return p
        parent = os.path.dirname(p)
        if parent == p:
            break
        p = parent
    # Fallback: src/rag -> two levels up (historical layout)
    return os.path.abspath(os.path.join(here, "..", ".."))


def _log_dir() -> str:
    return os.path.join(_project_root(), "data", "logs")


def _log_path() -> str:
    custom = os.getenv("CYPHER_LOG_PATH")
    if custom:
        return os.path.abspath(custom)
    return os.path.join(_log_dir(), "cypher_queries.log")


def _max_bytes() -> int:
    try:
        return int(os.getenv("CYPHER_LOG_MAX_BYTES", str(_DEFAULT_MAX_BYTES)))
    except ValueError:
        return _DEFAULT_MAX_BYTES


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _rotate_if_needed(path: str, max_bytes: int) -> None:
    if not os.path.isfile(path):
        return
    try:
        if os.path.getsize(path) < max_bytes:
            return
    except OSError:
        return
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    base, ext = os.path.splitext(path)
    archive = f"{base}_{ts}{ext or '.log'}"
    try:
        os.replace(path, archive)
        logger.info("Cypher log rotated: archived to %s", archive)
    except OSError as e:
        logger.warning("Could not rotate cypher log %s: %s", path, e)


def get_cypher_log_path() -> str:
    """Absolute path to the active Cypher log file (for diagnostics)."""
    return _log_path()


def ensure_cypher_log_ready() -> str:
    """Create data/logs and the log file with a startup line. Call once when the API starts."""
    path = _log_path()
    with _lock:
        _ensure_parent(path)
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(
                    f"{datetime.now(timezone.utc).isoformat()} [startup] "
                    f"cypher log ready at {path}\n"
                )
        except OSError as e:
            logger.warning("Cypher log startup write failed: %s", e)
    return path


def log_cypher_event(
    phase: str,
    message: str,
    *,
    detail: Optional[Any] = None,
) -> None:
    """Thread-safe line append; rotates file when over max size."""
    path = _log_path()
    max_bytes = _max_bytes()
    line = (
        f"{datetime.now(timezone.utc).isoformat()} [{phase}] {message}"
    )
    if detail is not None:
        line += f" | {detail!r}"
    line += "\n"

    with _lock:
        _ensure_parent(path)
        _rotate_if_needed(path, max_bytes)
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
        except OSError as e:
            logger.warning("Cypher log write failed: %s", e)


def log_cypher_multiline(
    phase: str,
    headline: str,
    body: str,
    *,
    delimiter_label: str = "CYPHER",
) -> None:
    """Append a header plus exact multi-line text (no truncation). Use delimiter_label e.g. LLM_VERDICT for non-Cypher."""
    path = _log_path()
    max_bytes = _max_bytes()
    ts = datetime.now(timezone.utc).isoformat()
    text = body if body is not None else ""
    if text and not text.endswith("\n"):
        text = text + "\n"
    block = (
        f"{ts} [{phase}] {headline}\n"
        f"---BEGIN {delimiter_label}---\n"
        f"{text}"
        f"---END {delimiter_label}---\n"
    )

    with _lock:
        _ensure_parent(path)
        _rotate_if_needed(path, max_bytes)
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(block)
        except OSError as e:
            logger.warning("Cypher log write failed: %s", e)
