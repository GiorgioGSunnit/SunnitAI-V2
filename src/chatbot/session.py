"""Conversation session management with multi-turn memory.

Each ChatSession maintains a history of user/assistant messages and feeds
relevant context from prior turns into the RAG pipeline so the agent can
resolve follow-up questions (e.g. "tell me more about that decree").
"""

import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..rag.main import run as rag_run
from ..rag.ai_chat import _call_chat
from ..rag.language import (
    DEFAULT_LANGUAGE,
    detect_explicit_language_switch,
    detect_language_llm,
    normalize_lang,
)
from ..rag.prompts import query_rewriter_system

from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

MAX_HISTORY_TURNS = 20  # Max conversation turns to keep in memory
MAX_CONTEXT_TURNS = 6   # Max recent turns to feed into query rewriting
SESSION_TTL_SECONDS = 3600  # Evict sessions idle for more than 1 hour
SESSION_CLEANUP_INTERVAL = 300  # Run cleanup every 5 minutes
DEFAULT_SESSION_LANGUAGE = os.getenv("CHAT_DEFAULT_LANGUAGE", DEFAULT_LANGUAGE)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# In questa fase assumiamo italiano: language detect disattivato di default.
CHAT_ENABLE_LANGUAGE_DETECT = _env_bool("CHAT_ENABLE_LANGUAGE_DETECT", False)


def _sanitize_user_error_message(err: Exception) -> str:
    raw = str(err or "")
    low = raw.lower()
    if "<!doctype html" in low or "<html" in low or "cloudflare" in low:
        return "Upstream LLM gateway error. Please try again in a few moments."
    if "bad gateway" in low or "502" in low or "503" in low or "504" in low:
        return "Temporary upstream service error. Please try again shortly."
    if "timeout" in low or "timed out" in low:
        return "Request timed out while contacting the language model. Please retry."
    return "An internal processing error occurred. Please retry."


@dataclass
class Message:
    role: str  # "user" or "assistant"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ChatSession:
    """A single conversation session with history tracking."""

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    messages: List[Message] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    session_language: str = field(default=DEFAULT_SESSION_LANGUAGE)
    _language_fixed_from_first_turn: bool = field(default=False)
    _last_active: float = field(default_factory=time.monotonic)

    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> Message:
        self._last_active = time.monotonic()
        msg = Message(role=role, content=content, metadata=metadata)
        self.messages.append(msg)
        # Trim old messages to prevent unbounded growth
        if len(self.messages) > MAX_HISTORY_TURNS * 2:
            self.messages = self.messages[-(MAX_HISTORY_TURNS * 2):]
        return msg

    def get_recent_context(self, n_turns: int = MAX_CONTEXT_TURNS) -> List[Message]:
        return self.messages[-(n_turns * 2):]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "message_count": len(self.messages),
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    "timestamp": m.timestamp,
                }
                for m in self.messages
            ],
        }


def _rewrite_query_with_context(
    query: str, history: List[Message], session_language: str
) -> str:
    """Use the LLM to resolve references in the user query given conversation history.

    For example: "tell me more about that" → "tell me more about Decree No. 46/2025"
    """
    if not history:
        return query

    lang = normalize_lang(session_language)
    history_text = "\n".join(
        f"{'User' if m.role == 'user' else 'Assistant'}: {m.content[:300]}"
        for m in history
    )

    rewritten = _call_chat(
        [
            SystemMessage(content=query_rewriter_system(lang)),
            HumanMessage(
                content=(
                    "Conversation history:\n{history}\n\n"
                    "Latest user message: {query}\n\n"
                    "Rewritten question:"
                ).format(history=history_text, query=query)
            ),
        ]
    )
    logger.info("Query rewritten: '%s' → '%s'", query, rewritten)
    return rewritten


class ChatBot:
    """Stateful chatbot that wraps the RAG pipeline with conversation memory."""

    def __init__(self):
        self._sessions: Dict[str, ChatSession] = {}
        self._lock = threading.Lock()
        # Start background cleanup daemon
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop, daemon=True
        )
        self._cleanup_thread.start()

    def _cleanup_loop(self) -> None:
        """Periodically evict sessions that have been idle beyond SESSION_TTL_SECONDS."""
        while True:
            time.sleep(SESSION_CLEANUP_INTERVAL)
            self._evict_expired_sessions()

    def _evict_expired_sessions(self) -> None:
        now = time.monotonic()
        with self._lock:
            expired = [
                sid
                for sid, s in self._sessions.items()
                if now - s._last_active > SESSION_TTL_SECONDS
            ]
            for sid in expired:
                del self._sessions[sid]
        if expired:
            logger.info("Evicted %d idle session(s)", len(expired))

    def create_session(self) -> ChatSession:
        session = ChatSession()
        self._sessions[session.session_id] = session
        logger.info("Created session %s", session.session_id)
        return session

    def get_session(self, session_id: str) -> Optional[ChatSession]:
        return self._sessions.get(session_id)

    def delete_session(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def list_sessions(self) -> List[Dict[str, Any]]:
        return [
            {
                "session_id": s.session_id,
                "created_at": s.created_at,
                "message_count": len(s.messages),
            }
            for s in self._sessions.values()
        ]

    def chat(self, session_id: str, user_message: str) -> Dict[str, Any]:
        """Process a user message within a session and return the response.

        Returns:
            {
                "session_id": str,
                "answer": str,
                "references": list,
                "original_query": str,
                "resolved_query": str,
            }
        """
        trace_id = str(uuid.uuid4())[:8]
        t0 = time.perf_counter()
        logger.info(
            "[chat.trace:%s] start session_id=%s message_len=%d",
            trace_id,
            session_id,
            len(user_message or ""),
        )

        session = self._sessions.get(session_id)
        if not session:
            session = self.create_session()
            session.session_id = session_id
            self._sessions[session_id] = session
            logger.info("[chat.trace:%s] created missing session object", trace_id)

        # Record the user message
        session.add_message("user", user_message)

        # Language phase
        lang_t0 = time.perf_counter()
        if CHAT_ENABLE_LANGUAGE_DETECT:
            switch = detect_explicit_language_switch(
                user_message, normalize_lang(session.session_language)
            )
            if switch:
                session.session_language = switch
            if not session._language_fixed_from_first_turn:
                if not switch:
                    session.session_language = detect_language_llm(user_message)
                session._language_fixed_from_first_turn = True
            logger.info(
                "[chat.trace:%s] language_detect enabled lang=%s elapsed_ms=%d",
                trace_id,
                session.session_language,
                int((time.perf_counter() - lang_t0) * 1000),
            )
        else:
            session.session_language = "it"
            session._language_fixed_from_first_turn = True
            logger.info(
                "[chat.trace:%s] language_detect skipped (CHAT_ENABLE_LANGUAGE_DETECT=false) lang=it elapsed_ms=%d",
                trace_id,
                int((time.perf_counter() - lang_t0) * 1000),
            )

        # Rewrite query with conversation context for follow-ups
        # Skip rewrite entirely on first message (no prior context to resolve against)
        rw_t0 = time.perf_counter()
        if len(session.messages) <= 1:
            resolved_query = user_message
        else:
            recent_history = session.get_recent_context()
            # Exclude the message we just added (last one) from rewrite context
            context_for_rewrite = recent_history[:-1] if len(recent_history) > 1 else []
            resolved_query = _rewrite_query_with_context(
                user_message, context_for_rewrite, session.session_language
            )
        logger.info(
            "[chat.trace:%s] rewrite done elapsed_ms=%d resolved_len=%d",
            trace_id,
            int((time.perf_counter() - rw_t0) * 1000),
            len(resolved_query or ""),
        )

        # Run through the RAG pipeline
        rag_t0 = time.perf_counter()
        try:
            result = rag_run(
                resolved_query,
                session_language=session.session_language,
                trace_id=trace_id,
            )
            answer = result.get("answer", "I couldn't find an answer to your question.")
            references = result.get("references", [])
            status_messages = result.get("status_messages") or []
            logger.info(
                "[chat.trace:%s] rag ok elapsed_ms=%d refs=%d",
                trace_id,
                int((time.perf_counter() - rag_t0) * 1000),
                len(references),
            )
        except Exception as e:
            logger.error("RAG pipeline error: %s", e, exc_info=True)
            user_err = _sanitize_user_error_message(e)
            answer = f"I'm sorry, I encountered an error processing your question. {user_err}"
            references = []
            status_messages = []
            logger.error(
                "[chat.trace:%s] rag error elapsed_ms=%d",
                trace_id,
                int((time.perf_counter() - rag_t0) * 1000),
                exc_info=True,
            )

        # Record the assistant response
        session.add_message("assistant", answer, metadata={"references": references})
        logger.info(
            "[chat.trace:%s] end total_ms=%d message_count=%d",
            trace_id,
            int((time.perf_counter() - t0) * 1000),
            len(session.messages),
        )

        return {
            "session_id": session.session_id,
            "answer": answer,
            "references": references,
            "original_query": user_message,
            "resolved_query": resolved_query,
            "session_language": session.session_language,
            "status_messages": status_messages,
        }
