"""The quote worker.

Providers are contacted here and nowhere else — never inside an HTTP request,
because a portal round trip can take a minute and browser automation cannot run
in a request thread at all.

The loop is deliberately split into three phases so synchronous database work
and asynchronous provider calls never interleave:

1. **prepare** (sync) — claim jobs and read everything the adapters will need;
2. **execute** (async) — call every provider in parallel, bounded by
   ``PC_MAX_CONCURRENT_PROVIDERS``, each under its own timeout;
3. **finalize** (sync) — persist results, apply retry/backoff, recompute status.

Restarting mid-flight is safe: work lives in the ``pc_quote_jobs`` table, a
killed worker's lease expires, and the idempotency key means a provider sees a
replayed submission as the same one.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import socket
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from ..config import Settings, get_settings
from ..db import session_scope
from ..models import ProviderAttempt, QuoteJob, QuoteRequest
from ..models.enums import AttemptStatus, JobStatus, QuoteOutcome
from ..providers import registry
from ..schemas.profile import QuotationProfile
from ..schemas.quotes import ProviderResult
from ..services import audit, job_queue, orchestrator, profile_service

logger = logging.getLogger("policy_comparator.worker")


@dataclass
class PreparedJob:
    """Everything one provider call needs, read out of the database up front."""

    job_id: uuid.UUID
    attempt_id: uuid.UUID
    request_id: uuid.UUID
    tenant_id: uuid.UUID
    provider_id: str
    kind: str
    idempotency_key: str
    profile: QuotationProfile | None
    resume_token: dict | None
    timeout_seconds: int
    #: Set when preparation itself failed; the job is finalized without a call.
    preparation_error: ProviderResult | None = None


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Worker:
    def __init__(self, settings: Settings | None = None, worker_id: str | None = None) -> None:
        self.settings = settings or get_settings()
        self.worker_id = worker_id or f"{socket.gethostname()}:{os.getpid()}"
        self._stopping = False

    # -- lifecycle -----------------------------------------------------------

    def request_stop(self, *_: Any) -> None:
        """Finish the batch in flight, then exit."""
        self._stopping = True
        logger.info("worker %s stopping after the current batch", self.worker_id)

    def run_forever(self, poll_interval: float = 1.0) -> None:
        signal.signal(signal.SIGINT, self.request_stop)
        signal.signal(signal.SIGTERM, self.request_stop)
        logger.info(
            "worker %s started (concurrency=%s, provider timeout=%ss, live=%s)",
            self.worker_id,
            self.settings.max_concurrent_providers,
            self.settings.provider_timeout_seconds,
            self.settings.live_provider_automation,
        )
        while not self._stopping:
            try:
                processed = self.run_once()
            except Exception:  # a bad batch must not kill the worker
                logger.exception("worker batch failed")
                processed = 0
            if processed == 0:
                time.sleep(poll_interval)
        logger.info("worker %s stopped", self.worker_id)

    # -- one batch -----------------------------------------------------------

    def run_once(self) -> int:
        """Claim, execute and finalize one batch. Returns the job count."""
        with session_scope() as db:
            jobs = job_queue.claim_batch(
                db,
                worker_id=self.worker_id,
                limit=self.settings.max_concurrent_providers,
            )
            if not jobs:
                return 0
            prepared = [self._prepare(db, job) for job in jobs]
            db.commit()

        results = asyncio.run(self._execute_all(prepared))

        with session_scope() as db:
            for job_prepared, result in results:
                self._finalize(db, job_prepared, result)
            db.commit()

        return len(prepared)

    # -- phase 1: prepare ----------------------------------------------------

    def _prepare(self, db: Session, job: QuoteJob) -> PreparedJob:
        attempt = db.get(ProviderAttempt, job.provider_attempt_id)
        request = db.get(QuoteRequest, job.quote_request_id)

        base = PreparedJob(
            job_id=job.id,
            attempt_id=job.provider_attempt_id,
            request_id=job.quote_request_id,
            tenant_id=job.tenant_id,
            provider_id=job.provider_id,
            kind=job.kind,
            idempotency_key="",
            profile=None,
            resume_token=None,
            timeout_seconds=self.settings.provider_timeout_seconds,
        )

        if attempt is None or request is None:
            base.preparation_error = ProviderResult(
                provider_id=job.provider_id,
                outcome=QuoteOutcome.FAILED,
                error_category="orphaned_job",
                error_message="The quotation request no longer exists",
            )
            return base

        try:
            bundle = profile_service.load_bundle(db, request.tenant_id, request)
            profile = profile_service.build_profile(
                bundle,
                policy_start_date=request.policy_start_date,
                email=bundle.customer.email,
            )
        except (LookupError, PermissionError, ValueError) as exc:
            base.preparation_error = ProviderResult(
                provider_id=job.provider_id,
                outcome=QuoteOutcome.FAILED,
                error_category="profile_unavailable",
                error_message=f"Could not assemble the profile ({type(exc).__name__})",
            )
            return base

        attempt.status = AttemptStatus.RUNNING.value
        attempt.started_at = _now()
        db.flush()

        config = self.settings.provider(job.provider_id)
        base.idempotency_key = attempt.idempotency_key
        base.profile = profile
        base.resume_token = attempt.resume_token
        base.timeout_seconds = config.timeout_seconds or self.settings.provider_timeout_seconds
        return base

    # -- phase 2: execute ----------------------------------------------------

    async def _execute_all(
        self, prepared: list[PreparedJob]
    ) -> list[tuple[PreparedJob, ProviderResult]]:
        semaphore = asyncio.Semaphore(max(1, self.settings.max_concurrent_providers))

        async def run(job: PreparedJob) -> tuple[PreparedJob, ProviderResult]:
            async with semaphore:
                return job, await self._execute_one(job)

        return list(await asyncio.gather(*(run(job) for job in prepared)))

    async def _execute_one(self, job: PreparedJob) -> ProviderResult:
        if job.preparation_error is not None:
            return job.preparation_error
        assert job.profile is not None

        try:
            adapter = registry.build_adapter(job.provider_id, self.settings)
        except registry.UnknownProvider:
            return ProviderResult(
                provider_id=job.provider_id,
                outcome=QuoteOutcome.CONFIGURATION_ERROR,
                error_category="unknown_provider",
                error_message=f"No adapter is registered for '{job.provider_id}'",
            )

        try:
            call = (
                adapter.resume_quote(job.resume_token, job.profile, job.idempotency_key)
                if job.kind == "resume"
                else adapter.request_quote(job.profile, job.idempotency_key)
            )
            return await asyncio.wait_for(call, timeout=job.timeout_seconds)
        except asyncio.TimeoutError:
            return ProviderResult(
                provider_id=job.provider_id,
                outcome=QuoteOutcome.TIMED_OUT,
                error_category="provider_timeout",
                error_message=(
                    f"{job.provider_id}: no response within {job.timeout_seconds}s"
                ),
            )
        except Exception as exc:
            # One adapter blowing up must never affect the others in the batch.
            # The type name only — an exception message can echo back the data
            # that was sent to the provider.
            logger.warning(
                "adapter %s raised %s", job.provider_id, type(exc).__name__, exc_info=False
            )
            return ProviderResult(
                provider_id=job.provider_id,
                outcome=QuoteOutcome.FAILED,
                error_category="adapter_exception",
                error_message=f"{job.provider_id}: adapter error ({type(exc).__name__})",
            )
        finally:
            try:
                await adapter.close()
            except Exception:  # pragma: no cover - cleanup must not mask results
                logger.warning("adapter %s failed to close cleanly", job.provider_id)

    # -- phase 3: finalize ---------------------------------------------------

    def _finalize(self, db: Session, prepared: PreparedJob, result: ProviderResult) -> None:
        job = db.get(QuoteJob, prepared.job_id)
        attempt = db.get(ProviderAttempt, prepared.attempt_id)
        request = db.get(QuoteRequest, prepared.request_id)

        if job is None or attempt is None or request is None:
            return
        if job.status == JobStatus.CANCELLED.value:
            return

        if self._should_retry(job, result):
            delay = self.settings.provider_retry_backoff_seconds * (2 ** (job.attempts - 1))
            attempt.status = AttemptStatus.RETRYING.value
            attempt.error_category = result.error_category
            attempt.error_message = audit.scrub(result.error_message)
            job_queue.reschedule(
                db, job, delay_seconds=delay, error=audit.scrub(result.error_message)
            )
            audit.log_event(
                logging.INFO,
                "provider attempt scheduled for retry",
                provider_id=prepared.provider_id,
                attempt=job.attempts,
                delay_seconds=delay,
                error_category=result.error_category,
            )
            orchestrator.refresh_request_status(db, request)
            return

        try:
            adapter = registry.build_adapter(prepared.provider_id, self.settings)
        except registry.UnknownProvider:
            adapter = None

        if adapter is not None:
            orchestrator.record_attempt_result(
                db, attempt, result, adapter, settings=self.settings
            )
        job_queue.complete(db, job)
        orchestrator.refresh_request_status(db, request)

        audit.log_event(
            logging.INFO,
            "provider attempt finished",
            provider_id=prepared.provider_id,
            outcome=result.outcome.value,
            quotes=len(result.raw_quotes),
        )

    def _should_retry(self, job: QuoteJob, result: ProviderResult) -> bool:
        """Retry only transient provider conditions, and only within budget."""
        if not result.outcome.is_retryable:
            return False
        return job.attempts < job.max_attempts


def main() -> None:  # pragma: no cover - process entry point
    logging.basicConfig(
        level=os.getenv("PC_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    Worker().run_forever()
