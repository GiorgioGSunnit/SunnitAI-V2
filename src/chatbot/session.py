"""Conversation session management with multi-turn memory.

Each ChatSession maintains a history of user/assistant messages and feeds
relevant context from prior turns into the RAG pipeline so the agent can
resolve follow-up questions (e.g. "tell me more about that decree").
"""

import logging
import re
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
    title: str = field(default="Nuova conversazione")
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
            "title": self.title,
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


def _generate_session_title(first_message: str) -> str:
    try:
        raw = _call_chat(
            [
                SystemMessage(
                    content=(
                        "Generate a concise 4-6 word title for a legal conversation "
                        "that starts with this message. Reply with only the title, no punctuation. "
                        "Always generate the title in the same language as the user message."
                    )
                ),
                HumanMessage(content=first_message[:200]),
            ],
            max_tokens=10,
        )
        return re.sub(r'["""\'\.\,\:\;\!\?]', "", raw).strip()
    except Exception:
        return first_message[:50]


def _strip_embeddings(records: list) -> list:
    cleaned = []
    for record in records:
        clean_record = {}
        for key, value in record.items():
            if isinstance(value, dict):
                clean_record[key] = {k: v for k, v in value.items() if k != "embedding"}
            else:
                clean_record[key] = value
        cleaned.append(clean_record)
    return cleaned


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


def detect_topic_drift(current_query: str, history: List[Message], lang: str) -> bool:
    """Return True if the current query is clearly about a different legal topic than recent history."""
    try:
        recent = history[-6:]  # up to 3 turns
        history_text = "\n".join(
            f"{'User' if m.role == 'user' else 'Assistant'}: {m.content[:300]}"
            for m in recent
        )
        response = _call_chat(
            [
                SystemMessage(
                    content=(
                        "You are a conversation topic analyzer. Given a conversation history and a new question, "
                        "return only YES if the new question is about a completely different legal topic than the "
                        "conversation so far, or NO if it is related or a natural continuation. "
                        "Be conservative — only return YES for clear topic changes, not for follow-up questions "
                        "or related subtopics."
                    )
                ),
                HumanMessage(
                    content=(
                        "Conversation history:\n{history}\n\n"
                        "New question: {query}\n\n"
                        "Answer YES or NO:"
                    ).format(history=history_text, query=current_query)
                ),
            ],
            max_tokens=5,
        )
        return response.strip().upper().startswith("YES")
    except Exception:
        return False


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
        with self._lock:
            self._sessions[session.session_id] = session
        logger.info("Created session %s", session.session_id)
        return session

    def get_session(self, session_id: str) -> Optional[ChatSession]:
        with self._lock:
            return self._sessions.get(session_id)

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
        return False

    def list_sessions(self) -> List[Dict[str, Any]]:
        with self._lock:
            sessions = list(self._sessions.values())
        return [
            {
                "session_id": s.session_id,
                "created_at": s.created_at,
                "title": s.title,
                "message_count": len(s.messages),
            }
            for s in sessions
        ]

    def chat(self, session_id: str, user_message: str) -> Dict[str, Any]:
        """Process a user message within a session and return the response.

        Returns:
            {
                "session_id": str,
                "answer": str,
                "original_query": str,
                "resolved_query": str,
            }
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                session = ChatSession(session_id=session_id)
                self._sessions[session_id] = session

        # Record the user message
        session.add_message("user", user_message)

        # Generate title on the first message
        if len(session.messages) == 1:
            session.title = _generate_session_title(user_message)

        # Session language: explicit switch, or auto-detect on first long message (default Italian)
        switch = detect_explicit_language_switch(
            user_message, normalize_lang(session.session_language)
        )
        if switch:
            session.session_language = switch
        if not session._language_fixed_from_first_turn:
            if not switch and should_auto_detect_language(user_message):
                session.session_language = detect_language_llm(user_message)
            session._language_fixed_from_first_turn = True

        # Rewrite query with conversation context for follow-ups
        # Skip rewrite entirely on first message (no prior context to resolve against)
        if len(session.messages) <= 1:
            resolved_query = user_message
        else:
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
            references = _strip_embeddings(result.get("references", []))
            status_messages = result.get("status_messages") or []
            citations = result.get("citations") or []
        except Exception as e:
            logger.error("RAG pipeline error: %s", e, exc_info=True)
            answer = f"I'm sorry, I encountered an error processing your question. Error: {e}"
            references = []
            status_messages = []
            citations = []

        # Detect topic drift and append a note when the user switches topics mid-session
        _DRIFT_NOTES = {
            "it": "\n\n---\n💡 Questo argomento sembra diverso dalla conversazione precedente. Considera di aprire una nuova chat per mantenere il contesto separato.",
            "es": "\n\n---\n💡 Este tema parece diferente a la conversación anterior. Considera abrir un nuevo chat para mantener el contexto separado.",
        }
        _drift_note = _DRIFT_NOTES.get(session.session_language, "\n\n---\n💡 This topic seems different from your previous conversation. Consider opening a new chat to keep the context separate.")
        if result.get("off_topic"):
            if not result.get("answer", "").startswith("💡"):
                answer += _drift_note
        elif len(session.messages) > 4:
            drift_context = session.get_recent_context()
            # Exclude the message we just added (the current user message)
            drift_context = drift_context[:-1] if len(drift_context) > 1 else []
            if detect_topic_drift(user_message, drift_context, session.session_language) and not result.get("answer", "").startswith("💡"):
                answer += _drift_note

        # Record the assistant response
        session.add_message("assistant", answer, metadata={"references": references})

        return {
            "session_id": session.session_id,
            "answer": answer,
            "original_query": user_message,
            "resolved_query": resolved_query,
            "session_language": session.session_language,
            "title": session.title,
            "status_messages": status_messages,
            "citations": citations,
        }
