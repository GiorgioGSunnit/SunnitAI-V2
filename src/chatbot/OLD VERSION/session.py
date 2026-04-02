"""Conversation session management with multi-turn memory.

Each ChatSession maintains a history of user/assistant messages and feeds
relevant context from prior turns into the RAG pipeline so the agent can
resolve follow-up questions (e.g. "tell me more about that decree").
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..rag.main import run as rag_run
from ..rag.ai_chat import _call_chat

from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

MAX_HISTORY_TURNS = 20  # Max conversation turns to keep in memory
MAX_CONTEXT_TURNS = 6   # Max recent turns to feed into query rewriting


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

    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> Message:
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


def _rewrite_query_with_context(query: str, history: List[Message]) -> str:
    """Use the LLM to resolve references in the user query given conversation history.

    For example: "tell me more about that" → "tell me more about Decree No. 46/2025"
    """
    if not history:
        return query

    history_text = "\n".join(
        f"{'User' if m.role == 'user' else 'Assistant'}: {m.content[:300]}"
        for m in history
    )

    rewritten = _call_chat(
        [
            SystemMessage(
                content=(
                    "You are a query rewriter for a legal document chatbot. "
                    "Given the conversation history and the latest user message, rewrite the user message "
                    "into a fully self-contained question that resolves all pronouns and references. "
                    "If the message is already self-contained, return it unchanged. "
                    "Return ONLY the rewritten question, nothing else."
                )
            ),
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

        # Rewrite query with conversation context for follow-ups
        recent_history = session.get_recent_context()
        # Exclude the message we just added (last one) from rewrite context
        context_for_rewrite = recent_history[:-1] if len(recent_history) > 1 else []
        resolved_query = _rewrite_query_with_context(user_message, context_for_rewrite)

        # Run through the RAG pipeline
        try:
            result = rag_run(resolved_query)
            answer = result.get("answer", "I couldn't find an answer to your question.")
            references = result.get("references", [])
        except Exception as e:
            logger.error("RAG pipeline error: %s", e, exc_info=True)
            answer = f"I'm sorry, I encountered an error processing your question. Error: {e}"
            references = []

        # Record the assistant response
        session.add_message("assistant", answer, metadata={"references": references})

        return {
            "session_id": session.session_id,
            "answer": answer,
            "references": references,
            "original_query": user_message,
            "resolved_query": resolved_query,
        }
