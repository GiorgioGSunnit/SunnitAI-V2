"""User document endpoints — upload, list, and delete private documents."""

import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
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
from .auth import get_current_user

router = APIRouter(prefix="/user/documents", tags=["documents"])


@router.post("", status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    scope: str = Form("personal"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if scope not in ("personal", "tenant"):
        raise HTTPException(status_code=400, detail="scope must be 'personal' or 'tenant'")
    if scope == "tenant" and current_user.role not in ("admin", "superadmin"):
        raise HTTPException(status_code=403, detail="Only admins can upload tenant-scope documents")

    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    folder = os.path.join(PRIVATE_DOCS_BASE, str(current_user.tenant_id), str(current_user.id))
    os.makedirs(folder, exist_ok=True)

    unique_name = f"{uuid.uuid4()}{ext}"
    storage_path = os.path.join(folder, unique_name)

    contents = await file.read()
    with open(storage_path, "wb") as f:
        f.write(contents)

    expiry_hours = get_tenant_expiry_hours(db, current_user.tenant_id)
    expires_at = datetime.now(timezone.utc) + timedelta(hours=expiry_hours)

    doc = create_user_document(
        db,
        user_id=current_user.id,
        tenant_id=current_user.tenant_id,
        original_filename=file.filename,
        storage_path=storage_path,
        file_size_bytes=len(contents),
        scope=scope,
        expires_at=expires_at,
    )
    return {
        "id": str(doc.id),
        "original_filename": doc.original_filename,
        "file_size_bytes": doc.file_size_bytes,
        "scope": doc.scope,
        "expires_at": doc.expires_at.isoformat() if doc.expires_at else None,
        "uploaded_at": doc.uploaded_at.isoformat(),
    }


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
            "expires_at": d.expires_at.isoformat() if d.expires_at else None,
            "uploaded_at": d.uploaded_at.isoformat(),
        }
        for d in docs
    ]


@router.delete("/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    doc_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    doc = get_user_document(db, doc_id, current_user.id, current_user.tenant_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    if doc.scope == "tenant" and current_user.role not in ("admin", "superadmin"):
        raise HTTPException(status_code=403, detail="Only admins can delete tenant documents")

    if os.path.exists(doc.storage_path):
        os.remove(doc.storage_path)

    delete_user_document(db, doc_id, current_user.id, current_user.tenant_id)


