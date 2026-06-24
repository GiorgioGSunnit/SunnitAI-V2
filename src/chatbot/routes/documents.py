"""User document endpoints — upload, list, download, delete, and analyse."""

import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from ...constants import PRIVATE_DOCS_BASE, SUPPORTED_EXTENSIONS
from ...db.base import get_db
from ...db.models import User
from ...db.crud import (
    create_user_document,
    delete_user_document,
    get_tenant_expiry_hours,
    get_user_document,
    get_user_documents,
)
from ...utils.document import extract_text_from_file
from .auth import get_current_user

router = APIRouter(prefix="/user/documents", tags=["documents"])

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TEMPLATE_EXTENSIONS = {".pdf", ".docx"}
_DOCUMENT_EXTENSIONS = {".txt"}


def _infer_document_role(ext: str, explicit_role: Optional[str]) -> str:
    if explicit_role in ("template", "document"):
        return explicit_role
    return "document" if ext in _DOCUMENT_EXTENSIONS else "template"


# ---------------------------------------------------------------------------
# POST /user/documents  — upload
# ---------------------------------------------------------------------------

@router.post("", status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    scope: str = Form("personal"),
    role: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # ── 1. All validation before any I/O ───────────────────────────────────
    if scope not in ("personal", "tenant"):
        raise HTTPException(status_code=400, detail="scope must be 'personal' or 'tenant'")

    if scope == "tenant" and current_user.role not in ("admin", "superadmin"):
        raise HTTPException(
            status_code=403, detail="Only admins can upload tenant-scope documents"
        )

    if role is not None and role not in ("template", "document"):
        raise HTTPException(
            status_code=400, detail="role must be 'template' or 'document'"
        )

    ext = os.path.splitext(file.filename or "")[1].lower()

    if ext not in SUPPORTED_EXTENSIONS:
        if ext == ".doc":
            raise HTTPException(
                status_code=400,
                detail=(
                    "Il formato .doc non è supportato. "
                    "Converti il file in .docx (Salva con nome → Word .docx) "
                    "e ricaricalo."
                ),
            )
        raise HTTPException(
            status_code=400,
            detail=(
                f"Formato non supportato: '{ext}'. "
                f"Formati accettati: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            ),
        )

    document_role = _infer_document_role(ext, role)

    # ── 2. Read into memory (no disk write yet) ────────────────────────────
    contents = await file.read()

    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Il file caricato è vuoto.")

    # ── 3. Atomic write: disk → DB, with cleanup on DB failure ────────────
    folder = os.path.join(
        PRIVATE_DOCS_BASE, str(current_user.tenant_id), str(current_user.id)
    )
    os.makedirs(folder, exist_ok=True)

    unique_name = f"{uuid.uuid4()}{ext}"
    storage_path = os.path.join(folder, unique_name)

    try:
        with open(storage_path, "wb") as f:
            f.write(contents)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Impossibile salvare il file: {exc}")

    expiry_hours = get_tenant_expiry_hours(db, current_user.tenant_id)
    expires_at = datetime.now(timezone.utc) + timedelta(hours=expiry_hours)

    try:
        doc = create_user_document(
            db,
            user_id=current_user.id,
            tenant_id=current_user.tenant_id,
            original_filename=file.filename,
            storage_path=storage_path,
            file_size_bytes=len(contents),
            scope=scope,
            expires_at=expires_at,
            document_role=document_role,
        )
    except Exception:
        try:
            os.remove(storage_path)
        except OSError:
            pass
        raise HTTPException(
            status_code=500,
            detail="Errore durante la registrazione del documento. Riprova.",
        )

    return {
        "id": str(doc.id),
        "original_filename": doc.original_filename,
        "file_size_bytes": doc.file_size_bytes,
        "scope": doc.scope,
        "document_role": doc.document_role,
        "expires_at": doc.expires_at.isoformat() if doc.expires_at else None,
        "uploaded_at": doc.uploaded_at.isoformat(),
    }


# ---------------------------------------------------------------------------
# GET /user/documents  — list
# ---------------------------------------------------------------------------

@router.get("")
def list_documents(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    docs = get_user_documents(db, current_user.id, current_user.tenant_id)
    return [
        {
            "id": str(d.id),
            "original_filename": d.original_filename,
            "file_size_bytes": d.file_size_bytes,
            "scope": d.scope,
            "document_role": getattr(d, "document_role", "template"),
            "expires_at": d.expires_at.isoformat() if d.expires_at else None,
            "uploaded_at": d.uploaded_at.isoformat(),
        }
        for d in docs
    ]


# ---------------------------------------------------------------------------
# GET /user/documents/{doc_id}/download
# ---------------------------------------------------------------------------

@router.get("/{doc_id}/download")
def download_document(
    doc_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    doc = get_user_document(db, doc_id, current_user.id, current_user.tenant_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    if not os.path.exists(doc.storage_path):
        raise HTTPException(status_code=404, detail="File not found on disk")
    return FileResponse(
        doc.storage_path,
        filename=doc.original_filename,
        media_type="application/octet-stream",
    )


# ---------------------------------------------------------------------------
# POST /user/documents/{doc_id}/analyse  — NEW
# ---------------------------------------------------------------------------

@router.post("/{doc_id}/analyse")
async def analyse_document(
    doc_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    import asyncio
    from functools import partial

    doc = get_user_document(db, doc_id, current_user.id, current_user.tenant_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    role = getattr(doc, "document_role", "template")
    if role != "document":
        raise HTTPException(
            status_code=400,
            detail=(
                "Questo file è stato caricato come template da compilare, "
                "non come documento da analizzare. "
                "Usa /api/generate/from-user-template per compilarlo."
            ),
        )

    if not os.path.exists(doc.storage_path):
        raise HTTPException(status_code=404, detail="File not found on disk")

    try:
        text = extract_text_from_file(doc.storage_path)
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Impossibile estrarre il testo dal documento: {exc}",
        )

    if not text.strip():
        raise HTTPException(
            status_code=422,
            detail="Il documento non contiene testo estraibile.",
        )

    _MAX_CHARS = 12_000
    truncated = text[:_MAX_CHARS]
    was_truncated = len(text) > _MAX_CHARS

    def _call_llm() -> dict:
        from langchain_core.messages import HumanMessage, SystemMessage
        from ..rag.document_generation import _call_chat

        system = (
            "Sei un assistente legale italiano. Analizza il documento fornito e rispondi "
            "SOLO in JSON valido con questa struttura esatta:\n"
            "{\n"
            '  "sommario": "Riassunto breve (3-5 frasi) del contenuto e scopo del documento.",\n'
            '  "tipo_documento": "Tipo di documento (es. Contratto di locazione, Diffida, ecc.)",\n'
            '  "parti": ["Lista delle parti coinvolte"],\n'
            '  "date_rilevanti": ["Lista di date rilevanti nel formato DD/MM/YYYY"],\n'
            '  "importi": ["Lista di importi o valori economici menzionati"],\n'
            '  "obblighi_principali": ["Lista breve degli obblighi o clausole principali"],\n'
            '  "note": "Eventuali avvertenze o informazioni mancanti rilevanti"\n'
            "}\n"
            "Non aggiungere testo fuori dal JSON. Se un campo non è applicabile usa [] o null."
        )
        human = f"Documento da analizzare:\n\n{truncated}"

        raw = _call_chat(
            [SystemMessage(content=system), HumanMessage(content=human)],
            max_tokens=1000,
        )
        import json, re
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        try:
            return json.loads(clean)
        except json.JSONDecodeError:
            return {"sommario": raw, "parse_error": True}

    loop = asyncio.get_event_loop()
    try:
        analysis = await loop.run_in_executor(None, _call_llm)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Errore durante l'analisi del documento: {exc}",
        )

    return {
        "doc_id": str(doc_id),
        "original_filename": doc.original_filename,
        "was_truncated": was_truncated,
        "analysis": analysis,
    }


# ---------------------------------------------------------------------------
# DELETE /user/documents/{doc_id}
# ---------------------------------------------------------------------------

@router.delete("/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    doc_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    doc = get_user_document(db, doc_id, current_user.id, current_user.tenant_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    if doc.scope == "platform":
        raise HTTPException(status_code=403, detail="Platform templates cannot be deleted")
    if doc.scope == "tenant" and current_user.role not in ("admin", "superadmin"):
        raise HTTPException(status_code=403, detail="Only admins can delete tenant documents")

    if os.path.exists(doc.storage_path):
        os.remove(doc.storage_path)

    delete_user_document(db, doc_id, current_user.id, current_user.tenant_id)
