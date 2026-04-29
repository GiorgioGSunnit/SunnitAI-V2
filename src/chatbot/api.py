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
import io
import logging
import re
from contextlib import asynccontextmanager
from functools import partial
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .session import ChatBot
from ..rag.main import run as rag_run
from ..rag.document_generation import (
    extract_case_details,
    generate_opposition_act,
    is_generation_request,
)

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


class GenerateRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Free-text request describing the opposition case.")
    session_id: Optional[str] = Field(None, description="Session ID for context. Omit to auto-create.")


class GenerateResponse(BaseModel):
    draft: str
    case_details: dict
    sources: list
    session_id: str


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


_PH_PATTERN = re.compile(r"\[[A-ZÀÁÂÄÉÈÊËÍÌÎÏÓÒÔÖÚÙÛÜ\s]+\](?:\s*\([^)]*\))?")

_DOC_TITLES = {
    "es": "ESCRITO DE OPOSICIÓN A DECRETO MONITORIO",
    "en": "OPPOSITION TO PAYMENT ORDER",
}


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

def _raw_result_to_sections(raw_result: list) -> list:
    """Convert raw Neo4j result records to flat dicts for _format_retrieved_sections."""
    sections = []
    seen: set = set()
    for record in raw_result:
        for value in record.values():
            if not isinstance(value, dict) or "properties" not in value:
                continue
            props = value["properties"]
            labels = value.get("labels", [])
            title = props.get("heading") or props.get("title") or ""
            text = props.get("text_en") or props.get("text_it") or props.get("text") or ""
            source = props.get("document_title") or props.get("document_id") or ""
            key = (title, source)
            if key in seen or not (title or text):
                continue
            seen.add(key)
            sections.append({"title": title, "text": text, "document_title": source, "labels": labels})
    return sections


def _get_cached_sections(session) -> Optional[list]:
    """Return converted sections from the most recent RAG-backed assistant message (last 2 turns).

    The normal chat flow stores raw_result in metadata["references"]. A record is
    RAG-backed when its values are dicts containing a "properties" key.
    """
    if not session:
        return None
    assistant_msgs = [m for m in reversed(session.messages) if m.role == "assistant"]
    for msg in assistant_msgs[:2]:
        refs = (msg.metadata or {}).get("references") or []
        if refs and isinstance(refs[0], dict) and "properties" in refs[0]:
            sections = _raw_result_to_sections(refs)
            if sections:
                return sections
    return None


def _run_generation_sync(message: str, session_lang: str, cached_sections: Optional[list] = None) -> dict:
    case_details = extract_case_details(message)
    if cached_sections is not None:
        retrieved_sections = cached_sections
        sources = sorted({s["document_title"] for s in retrieved_sections if s.get("document_title")})
    else:
        try:
            rag_state = rag_run(message, session_language=session_lang)
            raw_result = rag_state.get("raw_result") or []
            retrieved_sections = _raw_result_to_sections(raw_result)
            sources = sorted({s["document_title"] for s in retrieved_sections if s.get("document_title")})
        except Exception as exc:
            logger.warning("RAG retrieval for generation failed: %s", exc)
            retrieved_sections = []
            sources = []
    draft = generate_opposition_act(case_details, retrieved_sections, session_lang)
    return {"draft": draft, "case_details": case_details, "sources": sources}


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


@app.post("/api/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate an Italian opposition act (atto di opposizione a decreto ingiuntivo).

    Extracts case details from the free-text message, retrieves relevant sections
    from the knowledge base, and returns a structured draft act.
    """
    if not is_generation_request(request.message):
        raise HTTPException(status_code=400, detail="Not a generation request")

    session_id = request.session_id
    if not session_id:
        session = chatbot.create_session()
        session_id = session.session_id

    session = chatbot.get_session(session_id)
    if not session:
        session = chatbot.create_session()
        session.session_id = session_id
        chatbot._sessions[session_id] = session

    session_lang = session.session_language
    cached = _get_cached_sections(session)
    session.add_message("user", request.message)

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, partial(_run_generation_sync, request.message, session_lang, cached)
        )
    except Exception as exc:
        logger.error("Generation error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

    session.add_message("assistant", result["draft"], metadata={"sources": result["sources"]})
    return GenerateResponse(session_id=session_id, **result)


@app.post("/api/generate/download")
async def generate_download(request: GenerateRequest):
    """Generate opposition act and return as a downloadable .docx file."""
    try:
        from docx import Document
        from docx.shared import Cm, Pt
        from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_COLOR_INDEX
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="python-docx is not installed. Add 'python-docx>=1.1.0' to pyproject.toml and reinstall.",
        )

    if not is_generation_request(request.message):
        raise HTTPException(status_code=400, detail="Not a generation request")

    session_id = request.session_id
    if not session_id:
        session = chatbot.create_session()
        session_id = session.session_id
    session = chatbot.get_session(session_id)
    if not session:
        session = chatbot.create_session()
        session.session_id = session_id
        chatbot._sessions[session_id] = session
    session_lang = session.session_language
    cached = _get_cached_sections(session)
    session.add_message("user", request.message)

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, partial(_run_generation_sync, request.message, session_lang, cached)
        )
    except Exception as exc:
        logger.error("Generation error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))
    session.add_message("assistant", result["draft"], metadata={"sources": result["sources"]})

    doc_title = _DOC_TITLES.get(session_lang, "ATTO DI OPPOSIZIONE A DECRETO INGIUNTIVO")
    doc = Document()
    for section in doc.sections:
        section.top_margin = Cm(2.5)
        section.bottom_margin = Cm(2.5)
        section.left_margin = Cm(2.5)
        section.right_margin = Cm(2.5)

    title_para = doc.add_paragraph()
    title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_run = title_para.add_run(doc_title)
    title_run.bold = True
    title_run.font.size = Pt(14)
    doc.add_paragraph()

    for line in result["draft"].split("\n"):
        para = doc.add_paragraph()
        parts = _PH_PATTERN.split(line)
        matches = _PH_PATTERN.findall(line)
        for i, part in enumerate(parts):
            if part:
                para.add_run(part)
            if i < len(matches):
                hl_run = para.add_run(matches[i])
                hl_run.font.highlight_color = WD_COLOR_INDEX.YELLOW

    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": 'attachment; filename="atto_opposizione.docx"'},
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a message and get a response.

    If session_id is provided, continues the conversation.
    If omitted, a new session is created automatically.
    If the message is a generation request, redirects to the opposition act generation flow.
    """
    session_id = request.session_id
    if not session_id:
        session = chatbot.create_session()
        session_id = session.session_id

    if is_generation_request(request.message):
        session = chatbot.get_session(session_id)
        if not session:
            session = chatbot.create_session()
            session.session_id = session_id
            chatbot._sessions[session_id] = session
        session_lang = session.session_language
        cached = _get_cached_sections(session)
        session.add_message("user", request.message)
        try:
            loop = asyncio.get_event_loop()
            gen_result = await loop.run_in_executor(
                None, partial(_run_generation_sync, request.message, session_lang, cached)
            )
        except Exception as exc:
            logger.error("Generation error: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc))
        session.add_message("assistant", gen_result["draft"], metadata={"sources": gen_result["sources"]})
        return ChatResponse(
            session_id=session_id,
            answer=gen_result["draft"],
            references=gen_result["sources"],
            original_query=request.message,
            resolved_query=request.message,
            session_language=session_lang,
            status_messages=["generation_mode"],
        )

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
