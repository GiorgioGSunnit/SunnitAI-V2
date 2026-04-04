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
import logging
from contextlib import asynccontextmanager
from functools import partial
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .session import ChatBot

logger = logging.getLogger(__name__)


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


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health_check():
    return {"status": "ok"}


@app.get("/api/debug")
def debug_check():
    """Check connectivity to LLM and Neo4j."""
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
        result = await loop.run_in_executor(
            None, partial(chatbot.chat, session_id, request.message)
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
