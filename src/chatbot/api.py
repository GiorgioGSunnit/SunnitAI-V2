"""FastAPI backend for the SunnitAI chatbot.

Endpoints:
    POST   /api/chat              — Send a message (creates session if needed)
    POST   /api/sessions          — Create a new session
    GET    /api/sessions          — List all sessions
    GET    /api/sessions/{id}     — Get session history
    DELETE /api/sessions/{id}     — Delete a session
    GET    /api/health            — Health check
"""

import asyncio
import ast
import logging
import os
import re
from contextlib import asynccontextmanager
from functools import partial
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .session import ChatBot
from ..rag.main import run_diagnostics, run_diagnostics_full

logger = logging.getLogger(__name__)
CHAT_ENDPOINT_TIMEOUT_SECONDS = int(os.getenv("CHAT_ENDPOINT_TIMEOUT_SECONDS", "600"))


@asynccontextmanager
async def _lifespan(app: FastAPI):
    from ..rag.cypher_logger import ensure_cypher_log_ready

    log_path = ensure_cypher_log_ready()
    logger.info("Cypher query log file: %s", log_path)
    yield


# ---------------------------------------------------------------------------
# App & chatbot singleton
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SunnitAI ChatBot API",
    description="RAG-powered chatbot over legal documents stored in a Neo4j knowledge graph.",
    version="1.0.0",
    lifespan=_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chatbot = ChatBot()


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="The user's message")
    session_id: Optional[str] = Field(
        None,
        description="Session ID for multi-turn conversation. Omit to auto-create a new session.",
    )


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    references: list
    original_query: str
    resolved_query: str
    session_language: str = Field(
        default="it",
        description="Active session language code: it, en, or es.",
    )
    status_messages: list = Field(
        default_factory=list,
        description="Pipeline status lines (e.g. retrieval evaluation phase).",
    )


class SessionResponse(BaseModel):
    session_id: str
    created_at: str
    message_count: int


class RagDiagRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Diagnostic user message")
    session_language: Optional[str] = Field(
        default="it",
        description="Language code (default it)",
    )
    max_transitions: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Max graph transitions for full diagnostics loop",
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health_check():
    return {"status": "ok"}


@app.get("/api/debug")
async def debug_check():
    """Check connectivity to LLM and Neo4j."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _debug_check_sync)


def _debug_check_sync() -> dict:
    """Synchronous debug checks (runs in thread pool to avoid blocking event loop)."""
    import os
    from dotenv import load_dotenv

    from ..rag.cypher_logger import get_cypher_log_path

    load_dotenv()

    checks = {}

    cypher_log = get_cypher_log_path()
    checks["cypher_log_path"] = cypher_log
    checks["cypher_log_exists"] = os.path.isfile(cypher_log)
    try:
        with open(cypher_log, "a", encoding="utf-8"):
            pass
        checks["cypher_log_writable"] = True
    except OSError as e:
        checks["cypher_log_writable"] = f"error: {e}"

    # Check Neo4j
    try:
        from neo4j import GraphDatabase
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USER")
        pwd = os.getenv("NEO4J_PASSWORD")
        checks["neo4j_config"] = {"uri": uri, "user": user, "password_set": bool(pwd)}
        driver = GraphDatabase.driver(uri, auth=(user, pwd))
        with driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j")) as session:
            result = session.run("RETURN 1 AS n").single()
            checks["neo4j"] = "ok" if result else "query returned nothing"
        driver.close()
    except Exception as e:
        checks["neo4j"] = f"error: {e}"

    # Check LLM
    try:
        base_url = os.getenv("LLM_BASE_URL")
        model = os.getenv("LLM_MODEL")
        api_key = os.getenv("LLM_API_KEY")
        checks["llm_config"] = {
            "base_url": base_url,
            "model": model,
            "api_key_set": bool(api_key),
        }
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=model, api_key=api_key, base_url=base_url, temperature=0
        )
        resp = llm.invoke("Say hello in one word.")
        checks["llm"] = f"ok: {resp.content[:100]}"
    except Exception as e:
        checks["llm"] = f"error: {e}"

    # Check embeddings
    try:
        from ..rag.ai_chat import embedding_model
        test = embedding_model.embed_query("test")
        checks["embeddings"] = f"ok: dim={len(test)}"
    except Exception as e:
        checks["embeddings"] = f"error: {e}"

    return checks


def _extract_last_cypher_block(text: str) -> str:
    pattern = re.compile(
        r"\[c_execute\].*?---BEGIN CYPHER---\n(.*?)\n---END CYPHER---",
        re.DOTALL,
    )
    matches = list(pattern.finditer(text or ""))
    if not matches:
        return ""
    return (matches[-1].group(1) or "").strip()


def _extract_recent_timing_events(text: str, phase: str, limit: int = 5) -> list:
    rows = []
    for line in (text or "").splitlines():
        if f"[{phase}]" not in line or " | " not in line:
            continue
        ts = line.split(" [", 1)[0].strip()
        _, detail_str = line.split(" | ", 1)
        detail = {}
        try:
            detail = ast.literal_eval(detail_str.strip())
        except Exception:
            detail = {"raw_detail": detail_str.strip()}
        rows.append({"timestamp": ts, "detail": detail})
    return rows[-limit:]


def _diag_last_run_sync(max_tail_bytes: int = 350000) -> dict:
    from ..rag.cypher_logger import get_cypher_log_path

    path = get_cypher_log_path()
    if not os.path.isfile(path):
        return {"cypher_log_path": path, "error": "cypher log file not found"}

    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - max_tail_bytes))
        chunk = f.read().decode("utf-8", errors="replace")

    return {
        "cypher_log_path": path,
        "last_cypher_query": _extract_last_cypher_block(chunk),
        "recent_llm_calls": _extract_recent_timing_events(chunk, "llm_call", limit=8),
        "recent_neo4j_exec": _extract_recent_timing_events(chunk, "c_execute", limit=8),
    }


@app.post("/api/sessions", response_model=SessionResponse)
def create_session():
    session = chatbot.create_session()
    return SessionResponse(
        session_id=session.session_id,
        created_at=session.created_at,
        message_count=0,
    )


@app.get("/api/sessions")
def list_sessions():
    return chatbot.list_sessions()


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str):
    session = chatbot.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session.to_dict()


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str):
    if not chatbot.delete_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted", "session_id": session_id}


@app.post("/api/diag/rag")
async def diag_rag(request: RagDiagRequest):
    """Diagnostics endpoint: run early RAG steps with timings.

    Executes decompose -> linking -> context -> cypher generation (no execute/synthesis).
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        partial(
            run_diagnostics,
            request.message,
            request.session_language or "it",
        ),
    )


@app.post("/api/diag/chat")
async def diag_chat(request: RagDiagRequest):
    """Diagnostics endpoint: run full RAG flow with per-step timings."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        partial(
            run_diagnostics_full,
            request.message,
            request.session_language or "it",
            request.max_transitions,
        ),
    )


@app.get("/api/diag/last-run")
async def diag_last_run():
    """Diagnostics endpoint: inspect last Cypher + recent LLM/Neo4j timing events."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _diag_last_run_sync)


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a message and get a response.

    If session_id is provided, continues the conversation.
    If omitted, a new session is created automatically.
    """
    session_id = request.session_id
    if not session_id:
        session = chatbot.create_session()
        session_id = session.session_id

    try:
        loop = asyncio.get_event_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(None, partial(chatbot.chat, session_id, request.message)),
            timeout=CHAT_ENDPOINT_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.error(
            "Chat timeout after %ss (session_id=%s)",
            CHAT_ENDPOINT_TIMEOUT_SECONDS,
            session_id,
        )
        raise HTTPException(
            status_code=504,
            detail=(
                "Chat request timed out. "
                "Try again with a shorter question or check upstream LLM/Neo4j latency."
            ),
        )
    except Exception as e:
        logger.error("Chat error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    return ChatResponse(**result)


# ---------------------------------------------------------------------------
# Server runner
# ---------------------------------------------------------------------------

def start_server(host: str = "0.0.0.0", port: int = 8000):
    """Start the FastAPI server with uvicorn."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)
