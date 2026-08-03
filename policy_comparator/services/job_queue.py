"""Database-backed job queue.

The queue is a table, so a worker that is killed mid-flight loses nothing: its
claimed jobs simply have their lease expire and the next worker picks them up.
That is also why a job carries a ``dedupe_key`` — re-queuing identical work is a
no-op rather than a second submission to the provider.

Claiming uses ``SELECT ... FOR UPDATE SKIP LOCKED`` on PostgreSQL so several
workers can share a queue. SQLite has no such clause; there, writes are
serialized by the database itself and the lease check does the rest.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..models import QuoteJob
from ..models.enums import JobStatus

DEFAULT_LEASE_SECONDS = 300


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _aware(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def dedupe_key(attempt_id: uuid.UUID, kind: str, sequence: int) -> str:
    """Identity of one unit of work.

    Includes the attempt sequence so a *deliberate* retry is a new job, while a
    duplicated enqueue of the same round is not.
    """
    return f"{attempt_id}:{kind}:{sequence}"


def enqueue(
    db: Session,
    *,
    tenant_id: uuid.UUID,
    quote_request_id: uuid.UUID,
    provider_attempt_id: uuid.UUID,
    provider_id: str,
    kind: str = "request",
    sequence: int = 0,
    max_attempts: int = 3,
    run_after: datetime | None = None,
    payload: dict | None = None,
) -> QuoteJob:
    """Add a job, or return the existing one with the same identity."""
    key = dedupe_key(provider_attempt_id, kind, sequence)

    existing = db.execute(
        select(QuoteJob).where(QuoteJob.dedupe_key == key)
    ).scalar_one_or_none()
    if existing is not None:
        return existing

    job = QuoteJob(
        tenant_id=tenant_id,
        quote_request_id=quote_request_id,
        provider_attempt_id=provider_attempt_id,
        provider_id=provider_id,
        kind=kind,
        dedupe_key=key,
        status=JobStatus.QUEUED.value,
        max_attempts=max_attempts,
        run_after=run_after or _now(),
        payload=payload or {},
    )
    db.add(job)
    try:
        db.flush()
    except IntegrityError:
        # Another worker or request created the same job concurrently.
        db.rollback()
        return db.execute(select(QuoteJob).where(QuoteJob.dedupe_key == key)).scalar_one()
    return job


def claim_batch(
    db: Session,
    *,
    worker_id: str,
    limit: int = 4,
    lease_seconds: int = DEFAULT_LEASE_SECONDS,
    now: datetime | None = None,
) -> list[QuoteJob]:
    """Claim up to ``limit`` runnable jobs and lease them to this worker."""
    now = now or _now()

    stmt = (
        select(QuoteJob)
        .where(
            QuoteJob.status.in_([JobStatus.QUEUED.value, JobStatus.CLAIMED.value]),
            QuoteJob.run_after <= now,
        )
        .order_by(QuoteJob.run_after, QuoteJob.created_at)
        .limit(limit)
    )
    if db.bind is not None and db.bind.dialect.name == "postgresql":
        stmt = stmt.with_for_update(skip_locked=True)

    candidates = list(db.execute(stmt).scalars())

    claimed: list[QuoteJob] = []
    for job in candidates:
        lease = _aware(job.lease_expires_at)
        # A claimed job is only reclaimable once its lease has actually expired.
        if job.status == JobStatus.CLAIMED.value and lease is not None and lease > now:
            continue
        job.status = JobStatus.CLAIMED.value
        job.claimed_at = now
        job.claimed_by = worker_id
        job.lease_expires_at = now + timedelta(seconds=lease_seconds)
        job.attempts += 1
        claimed.append(job)

    if claimed:
        db.commit()
    return claimed


def complete(db: Session, job: QuoteJob) -> None:
    job.status = JobStatus.DONE.value
    job.finished_at = _now()
    job.lease_expires_at = None
    db.flush()


def reschedule(db: Session, job: QuoteJob, *, delay_seconds: float, error: str | None) -> None:
    """Put a job back on the queue for another attempt."""
    job.status = JobStatus.QUEUED.value
    job.run_after = _now() + timedelta(seconds=delay_seconds)
    job.claimed_at = None
    job.claimed_by = None
    job.lease_expires_at = None
    job.last_error = error
    db.flush()


def fail(db: Session, job: QuoteJob, *, error: str | None) -> None:
    job.status = JobStatus.FAILED.value
    job.finished_at = _now()
    job.lease_expires_at = None
    job.last_error = error
    db.flush()


def cancel_pending_for_request(db: Session, quote_request_id: uuid.UUID) -> int:
    """Cancel everything still outstanding for a request. Returns the count."""
    jobs = list(
        db.execute(
            select(QuoteJob).where(
                QuoteJob.quote_request_id == quote_request_id,
                QuoteJob.status.in_([JobStatus.QUEUED.value, JobStatus.CLAIMED.value]),
            )
        ).scalars()
    )
    for job in jobs:
        job.status = JobStatus.CANCELLED.value
        job.finished_at = _now()
        job.lease_expires_at = None
    db.flush()
    return len(jobs)


def pending_count(db: Session, quote_request_id: uuid.UUID) -> int:
    return len(
        list(
            db.execute(
                select(QuoteJob.id).where(
                    QuoteJob.quote_request_id == quote_request_id,
                    QuoteJob.status.in_([JobStatus.QUEUED.value, JobStatus.CLAIMED.value]),
                )
            ).scalars()
        )
    )
