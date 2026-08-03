"""Database-backed job queue and audit trail.

The queue is a table rather than an in-process structure so that killing the
worker mid-run loses nothing: a claimed job whose lease has expired is simply
reclaimed by the next worker.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from ..db import GUID, Base, JSONColumn


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class QuoteJob(Base):
    """A unit of work for the worker: contact one provider for one request."""

    __tablename__ = "pc_quote_jobs"
    __table_args__ = (
        # One live job per (request, provider, kind). Re-queuing an identical
        # job is a no-op rather than a duplicate provider submission.
        UniqueConstraint("dedupe_key", name="uq_pc_job_dedupe_key"),
    )

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(GUID(), nullable=False, index=True)
    quote_request_id: Mapped[uuid.UUID] = mapped_column(GUID(), nullable=False, index=True)
    provider_attempt_id: Mapped[uuid.UUID] = mapped_column(GUID(), nullable=False, index=True)
    provider_id: Mapped[str] = mapped_column(String(48), nullable=False)

    #: "request" for a first submission, "resume" after missing info was filled.
    kind: Mapped[str] = mapped_column(String(16), default="request")
    dedupe_key: Mapped[str] = mapped_column(String(160), nullable=False)

    status: Mapped[str] = mapped_column(String(16), default="queued", index=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3)

    #: Not before this time — used to schedule the exponential backoff.
    run_after: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    #: Lease held by the claiming worker. Past-due leases are reclaimable.
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    claimed_by: Mapped[str | None] = mapped_column(String(80), nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )

    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    payload: Mapped[dict] = mapped_column(JSONColumn(), default=dict)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class AuditEvent(Base):
    """Append-only record of who did what.

    ``metadata_json`` holds structured context but never personal data: field
    *names* are recorded, field *values* are not.
    """

    __tablename__ = "pc_audit_events"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(GUID(), nullable=False, index=True)
    actor_user_id: Mapped[uuid.UUID | None] = mapped_column(GUID(), nullable=True, index=True)
    actor_email: Mapped[str | None] = mapped_column(String(255), nullable=True)

    action: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    #: e.g. "quote_request", "provider_attempt".
    entity_type: Mapped[str | None] = mapped_column(String(48), nullable=True)
    entity_id: Mapped[uuid.UUID | None] = mapped_column(GUID(), nullable=True, index=True)
    provider_id: Mapped[str | None] = mapped_column(String(48), nullable=True)

    metadata_json: Mapped[dict] = mapped_column(JSONColumn(), default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True
    )


class StaffUser(Base):
    """Staff account for this tool.

    The application also accepts JWTs minted by the parent platform (same
    secret and claim names), so this table exists mainly so the tool can run
    standalone — a fresh checkout with no parent database still has logins.
    """

    __tablename__ = "pc_staff_users"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(GUID(), nullable=False, index=True)
    email: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str | None] = mapped_column(String(160), nullable=True)
    role: Mapped[str] = mapped_column(String(24), default="staff")  # staff | admin
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
