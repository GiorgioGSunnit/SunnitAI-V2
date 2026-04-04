"""Conversation session management with multi-turn memory.

Each ChatSession maintains a history of user/assistant messages and feeds
relevant context from prior turns into the RAG pipeline so the agent can
resolve follow-up questions (e.g. "tell me more about that decree").
"""

import logging
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
    detect_explicit_language_switch_llm,
    detect_language_llm,
    normalize_lang,
    should_auto_detect_language,
)
from ..rag.prompts import query_rewriter_system

from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

MAX_HISTORY_TURNS = 20  # Max conversation turns to keep in memory
MAX_CONTEXT_TURNS = 6   # Max recent turns to feed into query rewriting
SESSION_TTL_SECONDS = 3600  # Evict sessions idle for more than 1 hour
SESSION_CLEANUP_INTERVAL = 300  # Run cleanup every 5 minutes


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
    session_language: str = field(default=DEFAULT_LANGUAGE)
    _language_fixed_from_first_turn: bool = field(default=False)
    _last_active: float = field(default_factory=time.monotonic)

    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> Message:
        self._last_active = time.monotonic()
        msg = Message(role=role, content=content, metadata=metadata)
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
        session = self._sessions.get(session_id)
        if not session:
            session = self.create_session()
            session.session_id = session_id
            self._sessions[session_id] = session

        # Record the user message
        session.add_message("user", user_message)

        # Session language: explicit switch, or auto-detect on first long message (default Italian)
        switch = detect_explicit_language_switch_llm(
            user_message, normalize_lang(session.session_language)
        )
        if switch:
            session.session_language = switch
        if not session._language_fixed_from_first_turn:
            if not switch and should_auto_detect_language(user_message):
                session.session_language = detect_language_llm(user_message)
            session._language_fixed_from_first_turn = True

        # Rewrite query with conversation context for follow-ups
        recent_history = session.get_recent_context()
        # Exclude the message we just added (last one) from rewrite context
        context_for_rewrite = recent_history[:-1] if len(recent_history) > 1 else []
        resolved_query = _rewrite_query_with_context(
            user_message, context_for_rewrite, session.session_language
        )

        # Run through the RAG pipeline
        try:
            result = rag_run(resolved_query, session_language=session.session_language)
            answer = result.get("answer", "I couldn't find an answer to your question.")
            references = result.get("references", [])
            status_messages = result.get("status_messages") or []
        except Exception as e:
            logger.error("RAG pipeline error: %s", e, exc_info=True)
            answer = f"I'm sorry, I encountered an error processing your question. Error: {e}"
            references = []
            status_messages = []

        # Record the assistant response
        session.add_message("assistant", answer, metadata={"references": references})

        return {
            "session_id": session.session_id,
            "answer": answer,
            "references": references,
            "original_query": user_message,
            "resolved_query": resolved_query,
            "session_language": session.session_language,
            "status_messages": status_messages,
        }
