"""Conversation session management with multi-turn memory.

Each ChatSession maintains a history of user/assistant messages and feeds
relevant context from prior turns into the RAG pipeline so the agent can
resolve follow-up questions (e.g. "tell me more about that decree").
"""

import json
import logging
import os
import re
import tempfile
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
from .user_store import get_user_settings, get_user_settings_by_email

#remove once the frontend sends data
_FALLBACK_USER_EMAIL = os.environ.get("FALLBACK_USER_EMAIL", "admin@studiorossi.it")

from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

SESSIONS_FILE = os.environ.get("SESSIONS_FILE", "/opt/chatbot/data/sessions.json")

MAX_HISTORY_TURNS = 20  # Max conversation turns to keep in memory
MAX_CONTEXT_TURNS = 6   # Max recent turns to feed into query rewriting

@dataclass
class Message:
    role: str  # "user" or "assistant"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {"role": self.role, "content": self.content, "timestamp": self.timestamp}
        if self.metadata:
            for key in ("citations", "document_id", "document_name", "documents",
                        "generated_document_id", "generated_document_name"):
                if key in self.metadata:
                    d[key] = self.metadata[key]
        return d


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
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None

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
            "session_language": self.session_language,
            "_language_fixed_from_first_turn": self._language_fixed_from_first_turn,
            "last_active_at": datetime.now(timezone.utc).isoformat(),
            "message_count": len(self.messages),
            "messages": [m.to_dict() for m in self.messages],
            "user_id": self.user_id,
            "tenant_id": self.tenant_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatSession":
        try:
            stored_wall = datetime.fromisoformat(data["last_active_at"]).timestamp()
            elapsed = max(0.0, time.time() - stored_wall)
            last_active = time.monotonic() - elapsed
        except (KeyError, ValueError):
            last_active = time.monotonic()
        session = cls(
            session_id=data["session_id"],
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            title=data.get("title", "Nuova conversazione"),
            session_language=data.get("session_language", DEFAULT_LANGUAGE),
            _language_fixed_from_first_turn=data.get("_language_fixed_from_first_turn", False),
            _last_active=last_active,
            user_id=data.get("user_id"),
            tenant_id=data.get("tenant_id"),
        )
        session.messages = [
            Message(
                role=m["role"],
                content=m["content"],
                timestamp=m.get("timestamp", datetime.now(timezone.utc).isoformat()),
                metadata={
                    k: m[k]
                    for k in ("citations", "document_id", "document_name", "documents",
                              "generated_document_id", "generated_document_name")
                    if k in m
                } or None,
            )
            for m in data.get("messages", [])
        ]
        return session


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
    # Use more content from recent turns and less from older ones.
    # Recent turns get 500 chars; older turns get 150 chars.
    # This keeps the rewriter focused on the current topic without
    # losing the thread of the conversation.
    total = len(history)
    history_text = "\n".join(
        f"{'User' if m.role == 'user' else 'Assistant'}: "
        f"{m.content[:500 if i >= total - 4 else 150]}"
        for i, m in enumerate(history)
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
        self._sessions_file = SESSIONS_FILE
        self._load_sessions()

    def _load_sessions(self) -> None:
        try:
            with open(self._sessions_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            loaded = 0
            for session_data in data.get("sessions", []):
                try:
                    session = ChatSession.from_dict(session_data)
                    self._sessions[session.session_id] = session
                    loaded += 1
                except Exception as e:
                    logger.warning("Skipping corrupt session entry: %s", e)
            logger.info("Loaded %d session(s) from %s", loaded, self._sessions_file)
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning("Could not load sessions from %s: %s", self._sessions_file, e)

    def _save_sessions(self) -> None:
        with self._lock:
            snapshot = [s.to_dict() for s in self._sessions.values()]
            db_sessions = [
                s for s in self._sessions.values()
                if s.user_id and s.tenant_id
            ]

        # Write to JSON file
        try:
            dir_ = os.path.dirname(self._sessions_file) or "."
            os.makedirs(dir_, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(dir=dir_, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump({"sessions": snapshot}, f, ensure_ascii=False, indent=2)
                os.replace(tmp_path, self._sessions_file)
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except Exception as e:
            logger.error("Failed to save sessions to %s: %s", self._sessions_file, e)

        # Persist to PostgreSQL (failures are logged but never break the JSON save)
        if db_sessions:
            try:
                from ..db.base import SessionLocal
                from ..db.crud import upsert_conversation_from_session
                db = SessionLocal()
                try:
                    for s in db_sessions:
                        upsert_conversation_from_session(
                            db,
                            session_id=s.session_id,
                            user_id=s.user_id,
                            tenant_id=s.tenant_id,
                            title=s.title,
                            messages=[m.to_dict() for m in s.messages],
                            session_language=s.session_language,
                        )
                finally:
                    db.close()
            except Exception as e:
                logger.warning("Failed to persist sessions to PostgreSQL: %s", e)

    def create_session(self, user_id: Optional[str] = None, tenant_id: Optional[str] = None) -> ChatSession:
        session = ChatSession(user_id=user_id, tenant_id=tenant_id)
        with self._lock:
            self._sessions[session.session_id] = session
        logger.info("Created session %s", session.session_id)
        self._save_sessions()
        return session

    def get_session(self, session_id: str, user_id: Optional[str] = None) -> Optional[ChatSession]:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            return None
        if user_id is not None and session.user_id != user_id:
            return None
        return session

    def delete_session(self, session_id: str, user_id: Optional[str] = None) -> bool:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return False
            if user_id is not None and session.user_id != user_id:
                return False
            deleted = session_id in self._sessions
            if deleted:
                del self._sessions[session_id]
        if deleted:
            self._save_sessions()
            try:
                from ..db.base import SessionLocal
                from ..db.crud import delete_conversation_by_session_id
                db = SessionLocal()
                try:
                    delete_conversation_by_session_id(db, session_id)
                finally:
                    db.close()
            except Exception as e:
                logger.warning("Failed to delete session from PostgreSQL: %s", e)
        return deleted

    def list_sessions(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock:
            sessions = sorted(self._sessions.values(), key=lambda s: s.created_at, reverse=True)
            if user_id is not None:
                sessions = [s for s in sessions if s.user_id == user_id]
        return [
            {
                "session_id": s.session_id,
                "created_at": s.created_at,
                "title": s.title,
                "message_count": len(s.messages),
            }
            for s in sessions
        ]

    def chat(self, session_id: str, user_message: str,
             user_id: Optional[str] = None,
             tenant_id: Optional[str] = None) -> Dict[str, Any]:
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

        if session.user_id is not None and session.user_id != user_id:
            raise PermissionError(f"Session {session_id} does not belong to this user")

        if user_id and session.user_id is None:
            session.user_id = user_id
            session.tenant_id = tenant_id

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

        # Fetch AI behaviour settings from DB; fall back to test user when unauthenticated
        if session.user_id:
            settings = get_user_settings(session.user_id)
        else:
            settings = get_user_settings_by_email(_FALLBACK_USER_EMAIL)

        # Run through the RAG pipeline
        try:
            result = rag_run(
                resolved_query,
                session_language=session.session_language,
                user_id=session.user_id,
                tenant_id=session.tenant_id,
                tone=settings["tone"],
                standing=settings["standing"],
                response_length=settings["response_length"],
            )
            answer = result.get("answer", "I couldn't find an answer to your question.")
            references = _strip_embeddings(result.get("references", []))
            status_messages = result.get("status_messages") or []
            citations = result.get("citations") or []
        except Exception as e:
            _e_str = str(e)
            if "maximum context length" in _e_str or "input_tokens" in _e_str:
                logger.warning("RAG pipeline context overflow: %s", e)
                answer = (
                    "La domanda ha generato troppo contesto per essere elaborata. "
                    "Si prega di riformulare la domanda in modo più specifico."
                )
            else:
                logger.error("RAG pipeline error: %s", e, exc_info=True)
                answer = (
                    "Si è verificato un errore durante l'elaborazione della domanda. "
                    "Riprova tra qualche istante."
                )
            result = {}
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
            # off_topic: query classified as outside legal/professional domain
            _generic_fallback = {
                "it": "La domanda sembra esulare dall'ambito legale e professionale. Sono qui per assistere su questioni giuridiche, contrattuali o normative.",
                "es": "La pregunta parece estar fuera del ámbito legal y profesional. Estoy aquí para ayudar con cuestiones jurídicas, contractuales o normativas.",
                "en": "The question seems to fall outside the legal and professional domain. I am here to assist with legal, contractual, or regulatory matters.",
            }
            answer = _generic_fallback.get(session.session_language, _generic_fallback["it"])
        elif not result.get("retrieval_quality_ok"):
            # retrieval_quality_ok=False — query too broad or no quality results.
            # Extract document titles from references to show the user what topics exist.
            _references = result.get("references", [])
            _doc_titles = []
            for ref in _references:
                sources = ref.get("sources", []) if isinstance(ref, dict) else []
                for src in sources:
                    title = src.get("document_title") if isinstance(src, dict) else None
                    if title and title not in _doc_titles:
                        _doc_titles.append(title)
                if len(_doc_titles) >= 3:
                    break
            if _doc_titles:
                _topics = ", ".join(_doc_titles[:3])
                _broad_msg = {
                    "it": f"Avrei bisogno di una domanda più precisa perché nella base dati trovo notizie in {_topics}. Cosa ti interessa in particolare?",
                    "es": f"Necesitaría una pregunta más precisa porque en la base de datos encuentro información sobre {_topics}. ¿Qué le interesa en particular?",
                    "en": f"I would need a more precise question because in the knowledge base I find information on {_topics}. What are you specifically interested in?",
                }
                answer = _broad_msg.get(session.session_language, _broad_msg["it"])
            else:
                _no_data_fallback = {
                    "it": "Non ho trovato informazioni sufficienti nel database per rispondere a questa domanda. Prova a specificare meglio l'argomento o a citare un articolo o una norma specifica.",
                    "es": "No he encontrado información suficiente en la base de datos para responder a esta pregunta. Intenta especificar mejor el tema o citar un artículo o norma específica.",
                    "en": "I could not find sufficient information in the knowledge base to answer this question. Try specifying the topic further or referencing a specific article or regulation.",
                }
                answer = _no_data_fallback.get(session.session_language, _no_data_fallback["it"])
                # Replacing the answer with a "not found" fallback without
                # clearing citations leaves the original RAG-generated
                # citations attached to a message that explicitly says
                # nothing was found — actively misleading. Clear them here.
                citations = []
        elif len(session.messages) > 4:
            drift_context = session.get_recent_context()
            drift_context = drift_context[:-1] if len(drift_context) > 1 else []
            if detect_topic_drift(user_message, drift_context, session.session_language) and not result.get("answer", "").startswith("💡"):
                answer += _drift_note

        # Record the assistant response
        session.add_message("assistant", answer, metadata={"references": references, "citations": citations})
        self._save_sessions()

        return {
            "session_id": session.session_id,
            "answer": answer,
            "original_query": user_message,
            "resolved_query": resolved_query,
            "session_language": session.session_language,
            "title": session.title,
            "status_messages": status_messages,
            "citations": citations,
            "is_comparison": bool(result.get("is_comparison")),
        }
