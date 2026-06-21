"""FastAPI backend for the SunnitAI chatbot.

Endpoints:
    POST   /api/chat              — Send a message (creates session if needed)
    POST   /api/sessions          — Create a new session
    GET    /api/sessions          — List all sessions
    GET    /api/sessions/{id}     — Get session history
    DELETE /api/sessions/{id}     — Delete a session
    GET    /api/health            — Health check
    GET    /api/documents         — List all documents with section counts
    GET    /api/documents/{id}/sections/{name} — Get full section content
"""

import asyncio
import io
import logging
logging.getLogger("src.rag").setLevel(logging.INFO)
import os
import re
import time
import urllib.parse
from contextlib import asynccontextmanager
from functools import partial
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .session import ChatBot, ChatSession, _generate_session_title
from ..rag.main import run as rag_run, driver as neo4j_driver, NEO4J_DATABASE
from ..rag.verbose_logger import vlog
from ..rag.document_generation import (
    DOCUMENT_TYPE_REGISTRY,
    classify_document_type,
    classify_system_template,
    generate_document,
    is_generation_request,
)
from ..rag.graph_nodes import _extract_citations
from .auth import get_current_user, require_user, create_access_token, verify_password, hash_password
from .user_store import (get_user_by_email, get_user_by_id,
    get_tenant_by_id, create_studio_and_admin,
    create_user_with_invite, get_tenant_invite_code, update_user_profile)

logger = logging.getLogger(__name__)


async def _background_embedding_job():
    """Periodically generate embeddings for Section nodes missing them."""
    _running = False
    while True:
        await asyncio.sleep(120)  # every 2 minutes
        if _running:
            logger.debug("Background embedding job: previous run still in progress, skipping")
            continue
        _running = True
        try:
            from ..preprocessing.generate_embeddings import embed_missing
            count = await asyncio.get_event_loop().run_in_executor(
                None, embed_missing, NEO4J_DATABASE
            )
            if count > 0:
                logger.info(f"Background embedding job: generated {count} embeddings")
        except Exception as e:
            logger.warning(f"Background embedding job failed: {e}")
        finally:
            _running = False


@asynccontextmanager
async def _lifespan(app: FastAPI):
    from ..rag.cypher_logger import ensure_cypher_log_ready

    log_path = ensure_cypher_log_ready()
    logger.info("Cypher query log file: %s", log_path)
    task = asyncio.create_task(_background_embedding_job())
    yield
    task.cancel()


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

from .routes.auth import router as auth_router
from .routes.totp import router as totp_router
from .routes.users import router as users_router
from .routes.documents import router as documents_router
app.include_router(auth_router, prefix="/api")
app.include_router(totp_router, prefix="/api")
app.include_router(users_router, prefix="/api")
app.include_router(documents_router, prefix="/api")

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


DEFAULT_LOGO_PATH = "/opt/chatbot/assets/studio_logo.jpeg"


class GenerateRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Free-text request describing the opposition case.")
    session_id: Optional[str] = Field(None, description="Session ID for context. Omit to auto-create.")
    studio_name: str = Field("", description="Legal studio or organization name for document header.")
    studio_logo_path: str = Field("", description="Absolute server path to logo image file.")
    doc_type: str = Field("", description="Pre-classified document type; skips classify_document_type if provided.")
    draft: str = Field("", description="Pre-generated draft text; skips generation if provided.")


class GenerateResponse(BaseModel):
    draft: str
    case_details: dict
    sources: list
    session_id: str
    doc_type: str = ""


class ChatResponse(BaseModel):
    session_id: str
    answer: str
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
    citations: list = Field(
        default_factory=list,
        description="Structured citation list: [{document_name, document_id, sections}].",
    )
    title: str = Field(
        default="Nuova conversazione",
        description="Auto-generated session title (set after first message).",
    )
    is_comparison: bool = Field(
        default=False,
        description="True when the answer was produced by the comparison retrieval path.",
    )


class SessionResponse(BaseModel):
    session_id: str
    created_at: str
    message_count: int
    title: str = "Nuova conversazione"


class LoginRequest(BaseModel):
    email: str
    password: str


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    email: str
    studio_name: str = ""


class UserProfileResponse(BaseModel):
    user_id: str
    email: str
    role: str
    studio_name: str = ""
    first_name: str = ""
    last_name: str = ""
    tenant_id: str


class UpdateProfileRequest(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    display_name: Optional[str] = None
    professional_title: Optional[str] = None
    phone: Optional[str] = None


class RegisterStudioRequest(BaseModel):
    email: str
    password: str
    first_name: str
    last_name: str
    studio_name: str


class RegisterUserRequest(BaseModel):
    email: str
    password: str
    first_name: str
    last_name: str
    invite_code: str


class RegisterStudioResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    email: str
    studio_name: str
    invite_code: str


_PH_PATTERN = re.compile(r"\[[A-ZÀÁÂÄÉÈÊËÍÌÎÏÓÒÔÖÚÙÛÜ\s]+\](?:\s*\([^)]*\))?")

_DOC_FILENAMES = {
    "opposition_act": ("ATTO DI OPPOSIZIONE A DECRETO INGIUNTIVO", "ESCRITO DE OPOSICIÓN A DECRETO MONITORIO", "OPPOSITION TO PAYMENT ORDER", "atto_opposizione"),
    "rental_basic": ("CONTRATTO DI LOCAZIONE CON CEDOLARE SECCA", "CONTRATO DE ARRENDAMIENTO", "RENTAL AGREEMENT", "contratto_locazione_cedolare"),
    "rental_standard": ("CONTRATTO DI LOCAZIONE ABITATIVA", "CONTRATO DE ARRENDAMIENTO RESIDENCIAL", "RESIDENTIAL RENTAL AGREEMENT", "contratto_locazione_abitativa"),
    "rental_student": ("LOCAZIONE ABITATIVA PER STUDENTI UNIVERSITARI", "CONTRATO DE ARRENDAMIENTO PARA ESTUDIANTES", "STUDENT RENTAL AGREEMENT", "contratto_locazione_studenti"),
    "rental_transitional": ("LOCAZIONE ABITATIVA DI NATURA TRANSITORIA", "CONTRATO DE ARRENDAMIENTO TRANSITORIO", "TRANSITIONAL RENTAL AGREEMENT", "contratto_locazione_transitoria"),
    "rental_free_rent": ("CONTRATTO DI LOCAZIONE A CANONE LIBERO", "CONTRATO DE ARRENDAMIENTO A PRECIO LIBRE", "FREE RENT AGREEMENT", "contratto_locazione_canone_libero"),
    "rental_commercial": ("CONTRATTO DI LOCAZIONE AD USO COMMERCIALE", "CONTRATO DE ARRENDAMIENTO COMERCIAL", "COMMERCIAL LEASE AGREEMENT", "contratto_locazione_commerciale"),
    "rental_cancellation": ("DISDETTA CONTRATTO DI LOCAZIONE", "RESCISIÓN CONTRATO DE ARRENDAMIENTO", "RENTAL CANCELLATION NOTICE", "disdetta_locazione"),
    "insurance_cancellation": ("DISDETTA POLIZZA ASSICURATIVA", "RESCISIÓN PÓLIZA DE SEGUROS", "INSURANCE CANCELLATION NOTICE", "disdetta_polizza"),
    "insurance_declaration": ("DICHIARAZIONE SOSTITUTIVA DI POLIZZA ASSICURATIVA", "DECLARACIÓN SUSTITUTIVA DE PÓLIZA", "INSURANCE SUBSTITUTIVE DECLARATION", "dichiarazione_polizza"),
    "employment_dismissal_appeal": ("IMPUGNATIVA DI LICENZIAMENTO", "IMPUGNACIÓN DE DESPIDO", "DISMISSAL APPEAL", "impugnativa_licenziamento"),
    "employment_termination": ("LETTERA DI LICENZIAMENTO PER GIUSTA CAUSA", "CARTA DE DESPIDO POR CAUSA JUSTIFICADA", "TERMINATION LETTER FOR CAUSE", "lettera_licenziamento"),
    "franchising_contract": ("CONTRATTO DI FRANCHISING", "CONTRATO DE FRANQUICIA", "FRANCHISING AGREEMENT", "contratto_franchising"),
    "demand_letter": ("LETTERA DI DIFFIDA", "CARTA DE REQUERIMIENTO", "DEMAND LETTER", "lettera_diffida"),
    "appeal": ("RICORSO", "RECURSO", "APPEAL", "ricorso"),
    "power_of_attorney": ("PROCURA", "PODER NOTARIAL", "POWER OF ATTORNEY", "procura"),
    "sale_agreement": ("CONTRATTO DI COMPRAVENDITA", "CONTRATO DE COMPRAVENTA", "SALE AGREEMENT", "contratto_compravendita"),
    "verbale_assemblea": ("VERBALE DI ASSEMBLEA CONDOMINIALE", "ACTA DE JUNTA DE PROPIETARIOS", "CONDOMINIUM ASSEMBLY MINUTES", "verbale_assemblea"),
    "nota_contestazione": ("NOTA ALLA CONTESTAZIONE", "NOTA A LA CONTESTACIÓN", "NOTICE CONTESTING TRAFFIC VIOLATION", "nota_contestazione"),
    "comparison": ("CONFRONTO TRA DOCUMENTI", "COMPARACIÓN DE DOCUMENTOS", "DOCUMENT COMPARISON", "confronto_documenti"),
}


def _strip_markdown(text: str) -> str:
    text = re.sub(r'\*\*|__', '', text)
    text = re.sub(r'[*_]', '', text)
    return text


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


def _build_clarification_message() -> str:
    types_list = "\n".join(
        f"- **{entry['label']}**"
        for entry in DOCUMENT_TYPE_REGISTRY.values()
    )
    return (
        "Non ho capito che tipo di documento vuoi generare. "
        "Puoi specificare meglio la tua richiesta? I tipi di documento disponibili sono:\n\n"
        f"{types_list}\n\n"
        "Indica quale documento desideri e fornisci i dettagli necessari."
    )


def _run_generation_sync(message: str, session_lang: str, doc_type: str, cached_sections: Optional[list] = None, studio_name: str = "") -> dict:
    citations = None
    if cached_sections is not None:
        sources = sorted({s["document_title"] for s in cached_sections if s.get("document_title")})
    else:
        try:
            rag_state = rag_run(message, session_language=session_lang)
            raw_result = rag_state.get("raw_result") or []
            retrieved_sections = _raw_result_to_sections(raw_result)
            sources = sorted({s["document_title"] for s in retrieved_sections if s.get("document_title")})
            citations = _extract_citations(rag_state)
        except Exception as exc:
            logger.warning("RAG retrieval for generation failed: %s", exc)
            sources = []
    gen = generate_document(message, doc_type, session_lang, citations, studio_name)
    return {"draft": gen["draft"], "case_details": gen["case_details"], "sources": sources, "doc_type": gen["doc_type"]}


def _run_comparison_sync(message: str, session_lang: str, cached_sections=None) -> dict:
    """Run a comparison query through the RAG graph and return answer + citations."""
    rag_state = rag_run(message, session_language=session_lang)
    return {
        "answer": rag_state.get("answer", ""),
        "citations": rag_state.get("citations", []),
        "is_comparison": bool(rag_state.get("is_comparison")),
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/auth/login", response_model=AuthResponse)
async def login(request: LoginRequest):
    user = get_user_by_email(request.email)
    if not user or not user.get("is_active"):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not verify_password(request.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token({
        "sub": str(user["id"]),
        "email": user["email"],
        "role": user["role"],
        "tenant_id": str(user["tenant_id"]),
    })
    return AuthResponse(
        access_token=token,
        user_id=str(user["id"]),
        email=user["email"],
        studio_name=user.get("studio_name") or "",
    )


@app.post("/api/auth/register/studio", response_model=RegisterStudioResponse)
async def register_studio(request: RegisterStudioRequest):
    existing = get_user_by_email(request.email)
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")
    hashed = hash_password(request.password)
    user = create_studio_and_admin(
        email=request.email,
        hashed_password=hashed,
        first_name=request.first_name,
        last_name=request.last_name,
        studio_name=request.studio_name,
    )
    token = create_access_token({
        "sub": user["id"],
        "email": user["email"],
        "role": user["role"],
        "tenant_id": user["tenant_id"],
    })
    return RegisterStudioResponse(
        access_token=token,
        user_id=user["id"],
        email=user["email"],
        studio_name=user["studio_name"],
        invite_code=user["invite_code"],
    )


@app.post("/api/auth/register", response_model=AuthResponse)
async def register_user(request: RegisterUserRequest):
    existing = get_user_by_email(request.email)
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")
    hashed = hash_password(request.password)
    try:
        user = create_user_with_invite(
            email=request.email,
            hashed_password=hashed,
            first_name=request.first_name,
            last_name=request.last_name,
            invite_code=request.invite_code,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    token = create_access_token({
        "sub": user["id"],
        "email": user["email"],
        "role": user["role"],
        "tenant_id": user["tenant_id"],
    })
    return AuthResponse(
        access_token=token,
        user_id=user["id"],
        email=user["email"],
        studio_name=user.get("studio_name") or "",
    )


@app.get("/api/auth/me", response_model=UserProfileResponse)
async def get_me(current_user: dict = Depends(require_user)):
    user = get_user_by_id(current_user["sub"])
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return UserProfileResponse(
        user_id=str(user["id"]),
        email=user["email"],
        role=user["role"],
        studio_name=user.get("studio_name") or "",
        first_name=user.get("first_name") or "",
        last_name=user.get("last_name") or "",
        tenant_id=str(user["tenant_id"]),
    )


@app.put("/api/auth/me", response_model=UserProfileResponse)
async def update_me(
    request: UpdateProfileRequest,
    current_user: dict = Depends(require_user),
):
    try:
        user = update_user_profile(
            user_id=current_user["sub"],
            first_name=request.first_name,
            last_name=request.last_name,
            display_name=request.display_name,
            professional_title=request.professional_title,
            phone=request.phone,
        )
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return UserProfileResponse(
            user_id=str(user["id"]),
            email=user["email"],
            role=user["role"],
            studio_name=user.get("studio_name") or "",
            first_name=user.get("first_name") or "",
            last_name=user.get("last_name") or "",
            tenant_id=str(user["tenant_id"]),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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


@app.get("/api/documents")
def list_documents():
    """List all Document nodes with their section counts."""
    try:
        with neo4j_driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(
                "MATCH (d:Document)-[:CONTAINS]->(s:Section) "
                "RETURN d.name AS document_name, d.id AS document_id, count(s) AS section_count "
                "ORDER BY d.name"
            )
            documents = [
                {
                    "document_name": r["document_name"],
                    "document_id": r["document_id"],
                    "section_count": r["section_count"],
                }
                for r in result
            ]
    except Exception as e:
        logger.error("list_documents error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    return {"documents": documents}


@app.get("/api/documents/{document_id}/sections/{section_name}")
def get_section(document_id: str, section_name: str):
    """Return full content of a specific section."""
    decoded_doc_id = urllib.parse.unquote(document_id)
    decoded_section = urllib.parse.unquote(section_name)
    parts = decoded_doc_id.split("::")
    doc_hash = parts[1] if len(parts) >= 2 else decoded_doc_id
    try:
        with neo4j_driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(
                "MATCH (d:Document {id: $doc_id})-[:CONTAINS]->(s:Section) "
                "WHERE s.name = $section_name "
                "RETURN d.name AS document_name, s.name AS section_name, "
                "s.abstract AS abstract, s.plain_text AS plain_text "
                "LIMIT 1",
                doc_id=decoded_doc_id,
                section_name=decoded_section,
            )
            row = result.single()
    except Exception as e:
        logger.error("get_section error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    if not row:
        raise HTTPException(status_code=404, detail="Section not found")
    return {
        "document_name": row["document_name"],
        "section_name": row["section_name"],
        "abstract": row["abstract"] or "",
        "plain_text": row["plain_text"] or "",
    }


@app.post("/api/sessions", response_model=SessionResponse)
def create_session(current_user: dict = Depends(require_user)):
    session = chatbot.create_session(
        user_id=current_user["sub"],
        tenant_id=current_user.get("tenant_id"),
    )
    return SessionResponse(
        session_id=session.session_id,
        created_at=session.created_at,
        message_count=0,
        title=session.title,
    )


@app.get("/api/sessions")
def list_sessions(current_user: dict = Depends(require_user)):
    return chatbot.list_sessions(user_id=current_user["sub"])


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str, current_user: dict = Depends(require_user)):
    session = chatbot.get_session(session_id, user_id=current_user["sub"])
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session.to_dict()


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str, current_user: dict = Depends(require_user)):
    if not chatbot.delete_session(session_id, user_id=current_user["sub"]):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted", "session_id": session_id}


@app.post("/api/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest, current_user: Optional[dict] = Depends(get_current_user)):
    """Generate a legal document draft from a free-text request.

    Classifies the document type, extracts case details, retrieves relevant sections
    from the knowledge base, and returns a structured draft.
    """
    is_comparison_request = bool(re.search(
        r'\b(confronta|confronto|differenze?\s+tra|compara|paragona|versus|vs\.?)\b',
        request.message, re.IGNORECASE
    )) or request.doc_type == "comparison"

    if not is_comparison_request and not is_generation_request(request.message):
        raise HTTPException(status_code=400, detail="Not a generation request")

    session_id = request.session_id
    if not session_id:
        session = chatbot.create_session()
        session_id = session.session_id

    session = chatbot.get_session(session_id)
    if not session:
        session = ChatSession(session_id=session_id)
        chatbot._sessions[session_id] = session

    session_lang = session.session_language

    if current_user and not request.studio_name:
        _profile = get_user_by_id(current_user["sub"])
        if _profile:
            request = request.model_copy(update={"studio_name": _profile.get("studio_name") or ""})

    if is_comparison_request:
        doc_type = "comparison"
    else:
        doc_type = classify_document_type(request.message, session_lang)
        if doc_type == "unknown":
            doc_type = classify_system_template(request.message, session_lang)
        if doc_type == "unknown":
            clarification = _build_clarification_message()
            session.add_message("user", request.message)
            session.add_message("assistant", clarification)
            return GenerateResponse(
                session_id=session_id, draft=clarification, case_details={}, sources=[], doc_type="unknown"
            )

    cached = _get_cached_sections(session)
    session.add_message("user", request.message)

    try:
        loop = asyncio.get_event_loop()
        if is_comparison_request:
            comparison_result = await loop.run_in_executor(
                None, partial(_run_comparison_sync, request.message, session_lang, cached)
            )
            result = {
                "draft": comparison_result.get("answer", ""),
                "case_details": {},
                "sources": comparison_result.get("citations", []),
                "doc_type": "comparison",
                "studio_name": request.studio_name,
            }
        else:
            result = await loop.run_in_executor(
                None, partial(_run_generation_sync, request.message, session_lang, doc_type, cached, request.studio_name)
            )
    except Exception as exc:
        logger.error("Generation error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

    session.add_message("assistant", result["draft"], metadata={"sources": result.get("sources", [])})
    return GenerateResponse(session_id=session_id, **result)


@app.post("/api/generate/download")
async def generate_download(request: GenerateRequest, current_user: Optional[dict] = Depends(get_current_user)):
    """Generate opposition act and return as a downloadable .docx file."""
    try:
        from docx import Document
        from docx.shared import Cm, Inches, Pt
        from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_COLOR_INDEX
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="python-docx is not installed. Add 'python-docx>=1.1.0' to pyproject.toml and reinstall.",
        )

    is_comparison_request = bool(re.search(
        r'\b(confronta|confronto|differenze?\s+tra|compara|paragona|versus|vs\.?)\b',
        request.message, re.IGNORECASE
    )) or request.doc_type == "comparison" or bool(request.draft)

    if not is_comparison_request and not is_generation_request(request.message):
        raise HTTPException(status_code=400, detail="Not a generation request")

    session_id = request.session_id
    if not session_id:
        session = chatbot.create_session()
        session_id = session.session_id
    session = chatbot.get_session(session_id)
    if not session:
        session = ChatSession(session_id=session_id)
        chatbot._sessions[session_id] = session
    session_lang = session.session_language

    if current_user and not request.studio_name:
        _profile = get_user_by_id(current_user["sub"])
        if _profile:
            request = request.model_copy(update={"studio_name": _profile.get("studio_name") or ""})

    if is_comparison_request:
        doc_type = "comparison"
    elif request.doc_type:
        doc_type = request.doc_type
    else:
        doc_type = classify_document_type(request.message, session_lang)
        if doc_type == "unknown":
            doc_type = classify_system_template(request.message, session_lang)
        if doc_type == "unknown":
            raise HTTPException(status_code=400, detail=_build_clarification_message())
    cached = _get_cached_sections(session)
    session.add_message("user", request.message)

    loop = asyncio.get_event_loop()
    if is_comparison_request and not request.draft:
        comparison_result = await loop.run_in_executor(
            None, partial(_run_comparison_sync, request.message, session_lang)
        )
        request = request.model_copy(update={"draft": comparison_result.get("answer", "")})

    try:
        if request.draft:
            result = {"draft": request.draft, "doc_type": doc_type, "sources": []}
        else:
            result = await loop.run_in_executor(
                None, partial(_run_generation_sync, request.message, session_lang, doc_type, cached, request.studio_name)
            )
    except Exception as exc:
        logger.error("Generation error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))
    session.add_message("assistant", result["draft"], metadata={"sources": result.get("sources", [])})

    lang_idx = {"it": 0, "es": 1, "en": 2}.get(session_lang, 0)
    doc_info = _DOC_FILENAMES.get(doc_type, _DOC_FILENAMES["opposition_act"])
    doc_title = doc_info[lang_idx]
    filename = f"{doc_info[3]}.docx"
    doc = Document()
    for section in doc.sections:
        section.top_margin = Cm(2.5)
        section.bottom_margin = Cm(2.5)
        section.left_margin = Cm(2.5)
        section.right_margin = Cm(2.5)

    studio_logo_path = request.studio_logo_path or ""
    studio_name_val = request.studio_name or ""
    if not studio_logo_path and os.path.exists(DEFAULT_LOGO_PATH):
        studio_logo_path = DEFAULT_LOGO_PATH

    if studio_logo_path and os.path.exists(studio_logo_path):
        logo_para = doc.add_paragraph()
        logo_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        logo_run = logo_para.add_run()
        logo_run.add_picture(studio_logo_path, width=Inches(3.0))
        doc.add_paragraph()

    title_para = doc.add_paragraph()
    title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_run = title_para.add_run(doc_title)
    title_run.bold = True
    title_run.font.size = Pt(14)
    doc.add_paragraph()

    for line in _strip_markdown(result["draft"]).split("\n"):
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
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, current_user: Optional[dict] = Depends(get_current_user)):
    """Send a message and get a response.

    If session_id is provided, continues the conversation.
    If omitted, a new session is created automatically.
    If the message is a generation request, redirects to the opposition act generation flow.
    """
    session_id = request.session_id
    if not session_id:
        session = chatbot.create_session()
        session_id = session.session_id

    _req_start = time.time()
    vlog("request_start", {"session_id": session_id, "message_length": len(request.message)})

    if is_generation_request(request.message):
        session = chatbot.get_session(session_id)
        if not session:
            session = ChatSession(session_id=session_id)
            chatbot._sessions[session_id] = session
        session_lang = session.session_language
        doc_type = classify_document_type(request.message, session_lang)
        if doc_type == "unknown":
            doc_type = classify_system_template(request.message, session_lang)
        if doc_type == "unknown":
            clarification = _build_clarification_message()
            session.add_message("user", request.message)
            session.add_message("assistant", clarification)
            return ChatResponse(
                session_id=session_id,
                answer=clarification,
                original_query=request.message,
                resolved_query=request.message,
                session_language=session_lang,
                status_messages=["generation_mode"],
            )
        cached = _get_cached_sections(session)
        session.add_message("user", request.message)
        if len(session.messages) == 1:
            session.title = _generate_session_title(request.message)
        try:
            loop = asyncio.get_event_loop()
            gen_result = await loop.run_in_executor(
                None, partial(_run_generation_sync, request.message, session_lang, doc_type, cached)
            )
        except Exception as exc:
            logger.error("Generation error: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc))
        gen_result["draft"] = _strip_markdown(gen_result["draft"])
        session.add_message("assistant", gen_result["draft"], metadata={"sources": gen_result["sources"]})
        return ChatResponse(
            session_id=session_id,
            answer=gen_result["draft"],
            original_query=request.message,
            resolved_query=request.message,
            session_language=session_lang,
            status_messages=["generation_mode"],
            title=session.title,
        )

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, partial(chatbot.chat, session_id, request.message,
                          user_id=current_user.get("sub") if current_user else None,
                          tenant_id=current_user.get("tenant_id") if current_user else None)
        )
    except Exception as e:
        logger.error("Chat error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    vlog("request_end", {"session_id": session_id, "message_length": len(request.message)}, (time.time() - _req_start) * 1000)
    return ChatResponse(**result)


# ---------------------------------------------------------------------------
# Server runner
# ---------------------------------------------------------------------------

def start_server(host: str = "0.0.0.0", port: int = 8000):
    """Start the FastAPI server with uvicorn."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)
