"""Worker behaviour: timeouts, retries, the circuit breaker, restart and resume."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from policy_comparator.config import get_settings
from policy_comparator.models import ProviderAttempt, ProviderHealthState, QuoteJob, QuoteRequest
from policy_comparator.models.enums import AttemptStatus, JobStatus, QuoteOutcome, RequestStatus
from policy_comparator.providers import registry
from policy_comparator.services import circuit_breaker, job_queue, orchestrator
from policy_comparator.services.orchestrator import NewRequestInput
from policy_comparator.tests.conftest import FULL_ANSWERS, TENANT_A
from policy_comparator.worker.runner import Worker


def _now():
    return datetime.now(timezone.utc)


def make_request(db, providers=("zurich",), **overrides) -> QuoteRequest:
    data = NewRequestInput(
        vehicle_plate="AB123CD",
        owner_date_of_birth=datetime(1985, 3, 4).date(),
        customer_email="cliente@esempio.it",
        policy_start_date=(_now() + timedelta(days=7)).date(),
        privacy_accepted=True,
        provider_data_transfer_accepted=True,
        selected_provider_ids=list(providers),
        **overrides,
    )
    request = orchestrator.create_request(
        db, tenant_id=TENANT_A, actor_user_id=None, actor_email="staff@example.com", data=data
    )
    db.commit()
    return request


def fill_profile(db, request) -> None:
    """Answer everything the mock providers ask for, without resuming them."""
    from policy_comparator.models.enums import FieldSource
    from policy_comparator.services import profile_service

    bundle = profile_service.load_bundle(db, TENANT_A, request)
    profile_service.apply_updates(db, bundle, dict(FULL_ANSWERS), source=FieldSource.STAFF)
    db.commit()


def worker_with(**overrides) -> Worker:
    """A worker with its own settings copy.

    Settings are a frozen, process-wide singleton, so a test must never mutate
    them in place — that would leak into every later test.
    """
    return Worker(settings=replace(get_settings(), **overrides))


def drain(worker: Worker, limit: int = 12) -> int:
    """Run the worker until the queue is empty."""
    total = 0
    for _ in range(limit):
        processed = worker.run_once()
        if processed == 0:
            break
        total += processed
    return total


class TestHappyPath:
    def test_a_complete_profile_produces_quotes(self, db):
        request = make_request(db, providers=("zurich", "allianz"))
        fill_profile(db, request)
        orchestrator.start_request(db, request, actor_user_id=None, actor_email=None)
        db.commit()

        drain(Worker())

        db.expire_all()
        attempts = db.query(ProviderAttempt).filter_by(quote_request_id=request.id).all()
        assert {a.status for a in attempts} == {AttemptStatus.QUOTED.value}
        assert db.get(QuoteRequest, request.id).status == RequestStatus.COMPLETED.value

    def test_an_incomplete_profile_stops_at_missing_information(self, db):
        request = make_request(db)
        orchestrator.start_request(db, request, actor_user_id=None, actor_email=None)
        db.commit()

        drain(Worker())

        db.expire_all()
        attempt = db.query(ProviderAttempt).filter_by(quote_request_id=request.id).one()
        assert attempt.status == AttemptStatus.MISSING_INFORMATION.value
        assert attempt.missing_fields
        assert (
            db.get(QuoteRequest, request.id).status == RequestStatus.AWAITING_INFORMATION.value
        )

    def test_supplying_the_answers_resumes_and_quotes(self, db):
        request = make_request(db)
        orchestrator.start_request(db, request, actor_user_id=None, actor_email=None)
        db.commit()
        drain(Worker())

        db.expire_all()
        request = db.get(QuoteRequest, request.id)
        resumed = orchestrator.supply_missing_information(
            db, request, dict(FULL_ANSWERS), actor_user_id=None, actor_email=None
        )
        db.commit()
        assert resumed == ["zurich"]

        drain(Worker())
        db.expire_all()
        attempt = db.query(ProviderAttempt).filter_by(quote_request_id=request.id).one()
        assert attempt.status == AttemptStatus.QUOTED.value
        assert attempt.quotes


class TestTimeout:
    def test_a_slow_provider_times_out_without_blocking_the_others(self, db, monkeypatch):
        # A one-second budget keeps the test fast; the mechanism is identical.
        worker = worker_with(provider_timeout_seconds=1, provider_retry_count=0)

        request = make_request(db, providers=("zurich", "allianz"))
        fill_profile(db, request)
        orchestrator.start_request(
            db, request, actor_user_id=None, actor_email=None, settings=worker.settings
        )
        db.commit()

        original = registry.build_adapter

        def slow_zurich(provider_id, settings=None):
            adapter = original(provider_id, settings)
            if provider_id == "zurich":
                async def never(*_args, **_kwargs):
                    await asyncio.sleep(30)
                adapter.request_quote = never  # type: ignore[method-assign]
            return adapter

        monkeypatch.setattr(registry, "build_adapter", slow_zurich)

        drain(worker)

        db.expire_all()
        statuses = {
            a.provider_id: a.status
            for a in db.query(ProviderAttempt).filter_by(quote_request_id=request.id)
        }
        assert statuses["zurich"] == AttemptStatus.TIMED_OUT.value
        assert statuses["allianz"] == AttemptStatus.QUOTED.value

    def test_a_partially_failed_request_is_marked_partially_completed(self, db, monkeypatch):
        monkeypatch.setenv("PC_MOCK_FORCE_OUTCOME_ZURICH", "unavailable")
        worker = worker_with(provider_retry_count=0)

        request = make_request(db, providers=("zurich", "allianz"))
        fill_profile(db, request)
        orchestrator.start_request(
            db, request, actor_user_id=None, actor_email=None, settings=worker.settings
        )
        db.commit()

        drain(worker)

        db.expire_all()
        assert (
            db.get(QuoteRequest, request.id).status == RequestStatus.PARTIALLY_COMPLETED.value
        )


class TestRetry:
    def test_a_transient_failure_is_retried_with_backoff(self, db, monkeypatch):
        monkeypatch.setenv("PC_MOCK_FORCE_OUTCOME_ZURICH", "unavailable")
        worker = worker_with(provider_retry_count=2)

        request = make_request(db)
        fill_profile(db, request)
        orchestrator.start_request(
            db, request, actor_user_id=None, actor_email=None, settings=worker.settings
        )
        db.commit()

        worker.run_once()

        db.expire_all()
        job = db.query(QuoteJob).filter_by(quote_request_id=request.id).one()
        attempt = db.query(ProviderAttempt).filter_by(quote_request_id=request.id).one()

        assert job.status == JobStatus.QUEUED.value
        assert attempt.status == AttemptStatus.RETRYING.value
        # Backoff pushes the next run into the future rather than hot-looping.
        run_after = job.run_after.replace(tzinfo=timezone.utc) if job.run_after.tzinfo is None else job.run_after
        assert run_after > _now()

    def test_retries_stop_at_the_configured_budget(self, db, monkeypatch):
        monkeypatch.setenv("PC_MOCK_FORCE_OUTCOME_ZURICH", "unavailable")
        worker = worker_with(provider_retry_count=1, provider_retry_backoff_seconds=0.0)

        request = make_request(db)
        fill_profile(db, request)
        orchestrator.start_request(
            db, request, actor_user_id=None, actor_email=None, settings=worker.settings
        )
        db.commit()

        for _ in range(6):
            if worker.run_once() == 0:
                break

        db.expire_all()
        job = db.query(QuoteJob).filter_by(quote_request_id=request.id).one()
        attempt = db.query(ProviderAttempt).filter_by(quote_request_id=request.id).one()
        assert job.status == JobStatus.DONE.value
        assert attempt.status == AttemptStatus.UNAVAILABLE.value

    def test_human_gated_outcomes_are_never_retried(self, db, monkeypatch):
        """A CAPTCHA will not resolve itself, so retrying only wastes a call."""
        monkeypatch.setenv("PC_MOCK_FORCE_OUTCOME_ZURICH", "manual_action_required")
        request = make_request(db)
        fill_profile(db, request)
        orchestrator.start_request(db, request, actor_user_id=None, actor_email=None)
        db.commit()

        worker = worker_with(provider_retry_count=3)
        worker.run_once()

        db.expire_all()
        job = db.query(QuoteJob).filter_by(quote_request_id=request.id).one()
        assert job.status == JobStatus.DONE.value
        assert job.attempts == 1


class TestCircuitBreaker:
    def test_it_opens_after_the_threshold(self, db):
        settings = get_settings()
        for _ in range(settings.circuit_breaker_threshold - 1):
            circuit_breaker.record_failure(db, TENANT_A, "zurich", error_category="api_timeout")
        assert circuit_breaker.is_open(db, TENANT_A, "zurich") is False

        tripped = circuit_breaker.record_failure(
            db, TENANT_A, "zurich", error_category="api_timeout"
        )
        assert tripped is True
        assert circuit_breaker.is_open(db, TENANT_A, "zurich") is True

    def test_one_success_closes_it(self, db):
        for _ in range(5):
            circuit_breaker.record_failure(db, TENANT_A, "zurich")
        assert circuit_breaker.is_open(db, TENANT_A, "zurich") is True

        circuit_breaker.record_success(db, TENANT_A, "zurich")
        assert circuit_breaker.is_open(db, TENANT_A, "zurich") is False

    def test_human_gated_outcomes_do_not_trip_it(self, db):
        for outcome in (
            QuoteOutcome.MISSING_INFORMATION,
            QuoteOutcome.MANUAL_ACTION_REQUIRED,
            QuoteOutcome.AUTHENTICATION_REQUIRED,
            QuoteOutcome.CONFIGURATION_ERROR,
        ):
            circuit_breaker.record_outcome(db, TENANT_A, "zurich", outcome)

        state = circuit_breaker.get_state(db, TENANT_A, "zurich", create=False)
        assert state is None or state.consecutive_failures == 0

    def test_an_open_circuit_skips_the_provider_instead_of_calling_it(self, db):
        for _ in range(10):
            circuit_breaker.record_failure(db, TENANT_A, "zurich")
        db.commit()

        request = make_request(db, providers=("zurich", "allianz"))
        fill_profile(db, request)
        orchestrator.start_request(db, request, actor_user_id=None, actor_email=None)
        db.commit()

        db.expire_all()
        attempts = {
            a.provider_id: a
            for a in db.query(ProviderAttempt).filter_by(quote_request_id=request.id)
        }
        assert attempts["zurich"].status == AttemptStatus.SKIPPED_CIRCUIT_OPEN.value
        assert attempts["zurich"].error_category == "circuit_open"
        # No job was queued for the skipped provider.
        assert db.query(QuoteJob).filter_by(provider_id="zurich").count() == 0
        assert db.query(QuoteJob).filter_by(provider_id="allianz").count() == 1

    def test_state_survives_a_worker_restart(self, db):
        circuit_breaker.record_failure(db, TENANT_A, "zurich")
        db.commit()

        # A fresh session stands in for a restarted process.
        from policy_comparator.db import session_scope

        with session_scope() as other:
            state = other.query(ProviderHealthState).filter_by(provider_id="zurich").one()
            assert state.consecutive_failures == 1


class TestIdempotencyAndRestart:
    def test_the_same_round_is_not_queued_twice(self, db):
        request = make_request(db)
        fill_profile(db, request)
        orchestrator.start_request(db, request, actor_user_id=None, actor_email=None)
        db.commit()

        attempt = db.query(ProviderAttempt).filter_by(quote_request_id=request.id).one()
        job_queue.enqueue(
            db,
            tenant_id=TENANT_A,
            quote_request_id=request.id,
            provider_attempt_id=attempt.id,
            provider_id="zurich",
            kind="request",
            sequence=attempt.attempt_count,
        )
        db.commit()

        assert db.query(QuoteJob).filter_by(quote_request_id=request.id).count() == 1

    def test_a_deliberate_retry_creates_a_new_job(self, db):
        request = make_request(db)
        fill_profile(db, request)
        orchestrator.start_request(db, request, actor_user_id=None, actor_email=None)
        db.commit()
        drain(Worker())

        db.expire_all()
        request = db.get(QuoteRequest, request.id)
        orchestrator.retry_provider(
            db, request, "zurich", actor_user_id=None, actor_email=None
        )
        db.commit()

        assert db.query(QuoteJob).filter_by(quote_request_id=request.id).count() == 2

    def test_the_idempotency_key_is_stable_per_round(self):
        request_id = uuid.uuid4()
        assert orchestrator.idempotency_key(request_id, "zurich", 0) == (
            orchestrator.idempotency_key(request_id, "zurich", 0)
        )
        assert orchestrator.idempotency_key(request_id, "zurich", 0) != (
            orchestrator.idempotency_key(request_id, "zurich", 1)
        )
        assert orchestrator.idempotency_key(request_id, "zurich", 0) != (
            orchestrator.idempotency_key(request_id, "allianz", 0)
        )

    def test_an_expired_lease_is_reclaimed_by_the_next_worker(self, db):
        """A killed worker must not strand its jobs."""
        request = make_request(db)
        fill_profile(db, request)
        orchestrator.start_request(db, request, actor_user_id=None, actor_email=None)
        db.commit()

        stranded = job_queue.claim_batch(db, worker_id="worker-that-died", limit=5)
        assert len(stranded) == 1

        # Nothing is reclaimable while the lease is still valid.
        assert job_queue.claim_batch(db, worker_id="worker-2", limit=5) == []

        stranded[0].lease_expires_at = _now() - timedelta(seconds=1)
        db.commit()

        reclaimed = job_queue.claim_batch(db, worker_id="worker-2", limit=5)
        assert len(reclaimed) == 1
        assert reclaimed[0].claimed_by == "worker-2"

    def test_work_survives_a_restart_and_still_completes(self, db):
        request = make_request(db)
        fill_profile(db, request)
        orchestrator.start_request(db, request, actor_user_id=None, actor_email=None)
        db.commit()

        # Claim without processing, as if the worker died mid-flight.
        claimed = job_queue.claim_batch(db, worker_id="worker-1", limit=5)
        claimed[0].lease_expires_at = _now() - timedelta(seconds=1)
        db.commit()

        drain(Worker(worker_id="worker-2"))

        db.expire_all()
        attempt = db.query(ProviderAttempt).filter_by(quote_request_id=request.id).one()
        assert attempt.status == AttemptStatus.QUOTED.value


class TestRetryIsolation:
    def test_retrying_one_provider_leaves_the_others_untouched(self, db, monkeypatch):
        monkeypatch.setenv("PC_MOCK_FORCE_OUTCOME_ZURICH", "unavailable")
        request = make_request(db, providers=("zurich", "allianz"))
        fill_profile(db, request)
        orchestrator.start_request(db, request, actor_user_id=None, actor_email=None)
        db.commit()

        worker = worker_with(provider_retry_count=0)
        drain(worker)

        db.expire_all()
        allianz = db.query(ProviderAttempt).filter_by(
            quote_request_id=request.id, provider_id="allianz"
        ).one()
        allianz_quote_ids = {q.id for q in allianz.quotes}
        assert allianz_quote_ids

        # Zurich recovers; retry it alone.
        monkeypatch.delenv("PC_MOCK_FORCE_OUTCOME_ZURICH")
        request = db.get(QuoteRequest, request.id)
        orchestrator.retry_provider(db, request, "zurich", actor_user_id=None, actor_email=None)
        db.commit()
        drain(Worker())

        db.expire_all()
        allianz = db.query(ProviderAttempt).filter_by(
            quote_request_id=request.id, provider_id="allianz"
        ).one()
        zurich = db.query(ProviderAttempt).filter_by(
            quote_request_id=request.id, provider_id="zurich"
        ).one()

        assert zurich.status == AttemptStatus.QUOTED.value
        # Allianz was neither re-run nor had its quotes replaced.
        assert {q.id for q in allianz.quotes} == allianz_quote_ids
        assert allianz.attempt_count == 0


class TestCancellation:
    def test_cancelling_stops_pending_work_but_keeps_results(self, db):
        request = make_request(db, providers=("zurich", "allianz"))
        fill_profile(db, request)
        orchestrator.start_request(db, request, actor_user_id=None, actor_email=None)
        db.commit()

        cancelled = orchestrator.cancel_request(
            db, request, actor_user_id=None, actor_email=None
        )
        db.commit()

        assert cancelled == 2
        assert db.get(QuoteRequest, request.id).status == RequestStatus.CANCELLED.value
        # The worker must not pick cancelled jobs back up.
        assert Worker().run_once() == 0
