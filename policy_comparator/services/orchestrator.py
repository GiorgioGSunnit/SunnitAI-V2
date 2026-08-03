"""Coordination of a quotation request across providers.

The orchestrator owns the request lifecycle: creating it, gating the moment it
may be transmitted, fanning it out into per-provider jobs, folding each result
back in, and deciding when the request as a whole is finished.

It never calls a provider itself — that happens in the worker. Everything here
is a short database transaction, safe to run inside an HTTP request.

The live-submission gate lives in :func:`start_request`. All five conditions
must hold before anything leaves the building: provider configured, provider
authorized, ``LIVE_PROVIDER_AUTOMATION`` on, a valid consent record, and an
explicit start by a staff user. A provider that fails the gate still runs — in
mock mode, clearly labelled — rather than silently doing nothing.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Iterable

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import Settings, get_settings
from ..crypto import blind_index
from ..models import (
    ConsentRecord,
    CoveragePreference,
    Customer,
    CustomerProfile,
    InsuranceHistory,
    NormalizedQuote,
    ProviderAttempt,
    ProviderMissingFieldRequest,
    ProviderRawResponse,
    QuoteCoverage,
    QuoteRequest,
    Vehicle,
)
from ..models.enums import (
    AttemptStatus,
    AuditAction,
    ConsentType,
    FieldSource,
    QuoteOutcome,
    RequestStatus,
)
from ..providers import registry
from ..providers.base import ProviderAdapter
from ..schemas.quotes import ProviderResult
from . import audit, circuit_breaker, job_queue, profile_service

#: Namespace for deriving stable idempotency keys.
_IDEMPOTENCY_NS = uuid.UUID("6f1f9a4e-4a1b-4c2f-9c1a-3e0a2b5d7c11")


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ConsentMissing(PermissionError):
    """A request was started without the consent needed to transmit data."""


class RequestNotStartable(ValueError):
    """The request is not in a state where it can be started."""


@dataclass
class NewRequestInput:
    """The minimal form the staff user fills in first."""

    vehicle_plate: str
    owner_date_of_birth: date
    customer_email: str
    policy_start_date: date
    privacy_accepted: bool
    provider_data_transfer_accepted: bool
    selected_provider_ids: list[str]
    marketing_accepted: bool = False


def idempotency_key(request_id: uuid.UUID, provider_id: str, sequence: int) -> str:
    """Stable across worker restarts, distinct per deliberate retry."""
    return str(uuid.uuid5(_IDEMPOTENCY_NS, f"{request_id}:{provider_id}:{sequence}"))


# ---------------------------------------------------------------------------
# Creation
# ---------------------------------------------------------------------------


def create_request(
    db: Session,
    *,
    tenant_id: uuid.UUID,
    actor_user_id: uuid.UUID | None,
    actor_email: str | None,
    data: NewRequestInput,
    settings: Settings | None = None,
) -> QuoteRequest:
    """Create a request from the four initial fields plus consent."""
    settings = settings or get_settings()

    unknown = [
        pid for pid in data.selected_provider_ids if pid not in registry.available_provider_ids()
    ]
    if unknown:
        raise ValueError(f"Unknown provider(s): {', '.join(sorted(unknown))}")
    if not data.selected_provider_ids:
        raise ValueError("Select at least one provider")
    if not data.privacy_accepted:
        raise ConsentMissing("Privacy processing consent is required")
    if not data.provider_data_transfer_accepted:
        raise ConsentMissing("Consent to transmit data to the selected providers is required")

    fingerprint = blind_index(data.customer_email)
    customer = db.execute(
        select(Customer).where(
            Customer.tenant_id == tenant_id, Customer.email_fingerprint == fingerprint
        )
    ).scalar_one_or_none()
    if customer is None:
        customer = Customer(
            tenant_id=tenant_id,
            created_by_user_id=actor_user_id,
            email=data.customer_email,
            email_fingerprint=fingerprint,
        )
        db.add(customer)
        db.flush()

    staff = FieldSource.STAFF.value
    profile = CustomerProfile(
        tenant_id=tenant_id,
        customer_id=customer.id,
        owner_date_of_birth=data.owner_date_of_birth,
        field_sources={"customer.owner_date_of_birth": staff},
    )
    vehicle = Vehicle(
        tenant_id=tenant_id,
        plate="".join(ch for ch in data.vehicle_plate.upper() if ch.isalnum()),
        field_sources={"vehicle.plate": staff},
    )
    history = InsuranceHistory(tenant_id=tenant_id, field_sources={})
    preferences = CoveragePreference(tenant_id=tenant_id, field_sources={})
    db.add_all([profile, vehicle, history, preferences])
    db.flush()

    request = QuoteRequest(
        tenant_id=tenant_id,
        created_by_user_id=actor_user_id,
        customer_id=customer.id,
        customer_profile_id=profile.id,
        vehicle_id=vehicle.id,
        insurance_history_id=history.id,
        coverage_preference_id=preferences.id,
        policy_start_date=data.policy_start_date,
        selected_provider_ids=list(data.selected_provider_ids),
        status=RequestStatus.DRAFT.value,
        demonstration_data=True,
    )
    db.add(request)
    db.flush()

    # Mandatory processing consent and optional marketing consent are separate
    # records — they are never bundled behind one checkbox.
    _record_consent(
        db,
        tenant_id=tenant_id,
        customer_id=customer.id,
        request_id=request.id,
        consent_type=ConsentType.PRIVACY_PROCESSING,
        granted=True,
        scope=[],
        actor_user_id=actor_user_id,
    )
    _record_consent(
        db,
        tenant_id=tenant_id,
        customer_id=customer.id,
        request_id=request.id,
        consent_type=ConsentType.PROVIDER_DATA_TRANSFER,
        granted=True,
        scope=list(data.selected_provider_ids),
        actor_user_id=actor_user_id,
    )
    if data.marketing_accepted:
        _record_consent(
            db,
            tenant_id=tenant_id,
            customer_id=customer.id,
            request_id=request.id,
            consent_type=ConsentType.MARKETING,
            granted=True,
            scope=[],
            actor_user_id=actor_user_id,
        )

    for provider_id in data.selected_provider_ids:
        adapter_cls = registry.adapter_class(provider_id)
        db.add(
            ProviderAttempt(
                tenant_id=tenant_id,
                quote_request_id=request.id,
                provider_id=provider_id,
                provider_type=adapter_cls.provider_type.value,
                provider_mode=settings.provider(provider_id).mode,
                status=AttemptStatus.WAITING.value,
                idempotency_key=idempotency_key(request.id, provider_id, 0),
            )
        )

    audit.record(
        db,
        tenant_id=tenant_id,
        action=AuditAction.REQUEST_CREATED,
        actor_user_id=actor_user_id,
        actor_email=actor_email,
        entity_type="quote_request",
        entity_id=request.id,
        metadata={"providers": list(data.selected_provider_ids)},
    )
    db.flush()
    return request


def _record_consent(
    db: Session,
    *,
    tenant_id: uuid.UUID,
    customer_id: uuid.UUID,
    request_id: uuid.UUID,
    consent_type: ConsentType,
    granted: bool,
    scope: list[str],
    actor_user_id: uuid.UUID | None,
) -> ConsentRecord:
    record = ConsentRecord(
        tenant_id=tenant_id,
        customer_id=customer_id,
        quote_request_id=request_id,
        consent_type=consent_type.value,
        granted=granted,
        scope_provider_ids=scope,
        recorded_by_user_id=actor_user_id,
        policy_version="1.0",
    )
    db.add(record)
    db.flush()
    audit.record(
        db,
        tenant_id=tenant_id,
        action=AuditAction.CONSENT_RECORDED,
        actor_user_id=actor_user_id,
        entity_type="consent_record",
        entity_id=record.id,
        metadata={"consent_type": consent_type.value, "scope": scope},
    )
    return record


def has_transfer_consent(
    db: Session, request: QuoteRequest, provider_ids: Iterable[str]
) -> bool:
    """Whether a valid transfer consent covers every named provider."""
    records = list(
        db.execute(
            select(ConsentRecord).where(
                ConsentRecord.tenant_id == request.tenant_id,
                ConsentRecord.quote_request_id == request.id,
                ConsentRecord.consent_type == ConsentType.PROVIDER_DATA_TRANSFER.value,
                ConsentRecord.granted.is_(True),
            )
        ).scalars()
    )
    if not records:
        return False
    covered: set[str] = set()
    for record in records:
        covered.update(record.scope_provider_ids or [])
    return set(provider_ids).issubset(covered)


# ---------------------------------------------------------------------------
# Starting and resuming
# ---------------------------------------------------------------------------


def start_request(
    db: Session,
    request: QuoteRequest,
    *,
    actor_user_id: uuid.UUID | None,
    actor_email: str | None,
    settings: Settings | None = None,
) -> list[ProviderAttempt]:
    """Queue every waiting provider. Requires consent and an explicit start."""
    settings = settings or get_settings()

    if request.status == RequestStatus.CANCELLED.value:
        raise RequestNotStartable("This request has been cancelled")

    attempts = _attempts(db, request)
    startable = [a for a in attempts if a.status == AttemptStatus.WAITING.value]
    if not startable:
        raise RequestNotStartable("No provider is waiting to be started")

    if not has_transfer_consent(db, request, [a.provider_id for a in startable]):
        raise ConsentMissing(
            "No valid consent record covers every selected provider for this request"
        )

    queued: list[ProviderAttempt] = []
    for attempt in startable:
        if _skip_for_open_circuit(db, request, attempt, settings):
            continue
        _queue_attempt(db, request, attempt, kind="request", settings=settings)
        queued.append(attempt)

    request.started_at = request.started_at or _now()
    request.status = RequestStatus.RUNNING.value
    request.demonstration_data = all(
        settings.provider(a.provider_id).is_mock for a in attempts
    )

    audit.record(
        db,
        tenant_id=request.tenant_id,
        action=AuditAction.PROVIDERS_STARTED,
        actor_user_id=actor_user_id,
        actor_email=actor_email,
        entity_type="quote_request",
        entity_id=request.id,
        metadata={
            "queued": [a.provider_id for a in queued],
            "skipped_circuit_open": [
                a.provider_id
                for a in startable
                if a.status == AttemptStatus.SKIPPED_CIRCUIT_OPEN.value
            ],
            "live_provider_automation": settings.live_provider_automation,
        },
    )
    refresh_request_status(db, request)
    db.flush()
    return queued


def _skip_for_open_circuit(
    db: Session, request: QuoteRequest, attempt: ProviderAttempt, settings: Settings
) -> bool:
    """Mark an attempt skipped when the provider's breaker is open."""
    if not circuit_breaker.is_open(db, request.tenant_id, attempt.provider_id):
        return False
    opens_at = circuit_breaker.open_until(db, request.tenant_id, attempt.provider_id)
    attempt.status = AttemptStatus.SKIPPED_CIRCUIT_OPEN.value
    attempt.outcome = QuoteOutcome.UNAVAILABLE.value
    attempt.error_category = "circuit_open"
    attempt.error_message = (
        f"{attempt.provider_id}: temporarily skipped after repeated failures"
        + (f" (retry after {opens_at.isoformat()})" if opens_at else "")
    )
    attempt.finished_at = _now()
    db.flush()
    return True


def _queue_attempt(
    db: Session,
    request: QuoteRequest,
    attempt: ProviderAttempt,
    *,
    kind: str,
    settings: Settings,
    delay_seconds: float = 0.0,
) -> None:
    attempt.status = AttemptStatus.WAITING.value
    attempt.outcome = None
    attempt.error_category = None
    attempt.error_message = None
    attempt.idempotency_key = idempotency_key(
        request.id, attempt.provider_id, attempt.attempt_count
    )
    db.flush()

    run_after = _now()
    if delay_seconds:
        from datetime import timedelta

        run_after = run_after + timedelta(seconds=delay_seconds)

    job_queue.enqueue(
        db,
        tenant_id=request.tenant_id,
        quote_request_id=request.id,
        provider_attempt_id=attempt.id,
        provider_id=attempt.provider_id,
        kind=kind,
        sequence=attempt.attempt_count,
        max_attempts=settings.provider_retry_count + 1,
        run_after=run_after,
    )


def supply_missing_information(
    db: Session,
    request: QuoteRequest,
    updates: dict[str, Any],
    *,
    actor_user_id: uuid.UUID | None,
    actor_email: str | None,
    settings: Settings | None = None,
) -> list[str]:
    """Apply staff-supplied answers and resume the providers that were waiting.

    Only the attempts that actually asked for something are resumed; providers
    that already returned a quote are left alone.
    """
    settings = settings or get_settings()
    bundle = profile_service.load_bundle(db, request.tenant_id, request)

    changed = profile_service.apply_updates(db, bundle, updates, source=FieldSource.STAFF)

    audit.record(
        db,
        tenant_id=request.tenant_id,
        action=AuditAction.PROFILE_UPDATED,
        actor_user_id=actor_user_id,
        actor_email=actor_email,
        entity_type="quote_request",
        entity_id=request.id,
        # Field names only. Never the values the staff member typed.
        metadata={"fields": sorted(changed)},
    )

    resumed: list[str] = []
    for attempt in _attempts(db, request):
        if attempt.status != AttemptStatus.MISSING_INFORMATION.value:
            continue
        for row in attempt.missing_fields:
            if row.field_path in updates:
                row.resolved = True
        attempt.attempt_count += 1
        _queue_attempt(db, request, attempt, kind="resume", settings=settings)
        resumed.append(attempt.provider_id)

    if resumed:
        request.status = RequestStatus.RUNNING.value
    refresh_request_status(db, request)
    db.flush()
    return resumed


def retry_provider(
    db: Session,
    request: QuoteRequest,
    provider_id: str,
    *,
    actor_user_id: uuid.UUID | None,
    actor_email: str | None,
    settings: Settings | None = None,
) -> ProviderAttempt:
    """Re-run one provider, leaving every other provider's result intact."""
    settings = settings or get_settings()
    attempt = db.execute(
        select(ProviderAttempt).where(
            ProviderAttempt.quote_request_id == request.id,
            ProviderAttempt.tenant_id == request.tenant_id,
            ProviderAttempt.provider_id == provider_id,
        )
    ).scalar_one_or_none()
    if attempt is None:
        raise LookupError(f"Provider '{provider_id}' is not part of this request")
    if attempt.status in {AttemptStatus.WAITING.value, AttemptStatus.RUNNING.value}:
        raise ValueError(f"Provider '{provider_id}' is already running")
    if not has_transfer_consent(db, request, [provider_id]):
        raise ConsentMissing(f"No consent record covers transmitting data to '{provider_id}'")

    # Discard only this provider's previous quotes; the others are untouched.
    for quote in list(attempt.quotes):
        db.delete(quote)

    # A staff-initiated retry is an explicit override of the breaker.
    circuit_breaker.reset(db, request.tenant_id, provider_id)

    attempt.attempt_count += 1
    attempt.diagnostic_artifact_path = None
    _queue_attempt(db, request, attempt, kind="request", settings=settings)

    audit.record(
        db,
        tenant_id=request.tenant_id,
        action=AuditAction.PROVIDER_RETRIED,
        actor_user_id=actor_user_id,
        actor_email=actor_email,
        entity_type="provider_attempt",
        entity_id=attempt.id,
        provider_id=provider_id,
        metadata={"attempt_count": attempt.attempt_count},
    )
    request.status = RequestStatus.RUNNING.value
    db.flush()
    return attempt


def cancel_request(
    db: Session,
    request: QuoteRequest,
    *,
    actor_user_id: uuid.UUID | None,
    actor_email: str | None,
) -> int:
    """Cancel outstanding work. Results already received are kept."""
    cancelled = job_queue.cancel_pending_for_request(db, request.id)
    for attempt in _attempts(db, request):
        if AttemptStatus(attempt.status).is_pending:
            attempt.status = AttemptStatus.CANCELLED.value
            attempt.finished_at = _now()

    request.status = RequestStatus.CANCELLED.value
    request.cancelled_at = _now()
    audit.record(
        db,
        tenant_id=request.tenant_id,
        action=AuditAction.REQUEST_CANCELLED,
        actor_user_id=actor_user_id,
        actor_email=actor_email,
        entity_type="quote_request",
        entity_id=request.id,
        metadata={"cancelled_jobs": cancelled},
    )
    db.flush()
    return cancelled


# ---------------------------------------------------------------------------
# Folding results back in
# ---------------------------------------------------------------------------


def record_attempt_result(
    db: Session,
    attempt: ProviderAttempt,
    result: ProviderResult,
    adapter: ProviderAdapter,
    *,
    settings: Settings | None = None,
) -> None:
    """Persist one provider's answer and update every derived state."""
    settings = settings or get_settings()
    request = db.get(QuoteRequest, attempt.quote_request_id)
    assert request is not None

    attempt.outcome = result.outcome.value
    attempt.status = AttemptStatus.from_outcome(result.outcome).value
    attempt.error_category = result.error_category
    attempt.error_message = audit.scrub(result.error_message)
    attempt.finished_at = _now()
    if attempt.started_at is not None:
        started = attempt.started_at
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        attempt.duration_ms = int((attempt.finished_at - started).total_seconds() * 1000)
    if result.resume_token is not None:
        attempt.resume_token = result.resume_token
    if isinstance(result.raw_payload, dict):
        artifact = result.raw_payload.get("diagnostic_artifact")
        if artifact:
            attempt.diagnostic_artifact_path = artifact

    db.add(
        ProviderRawResponse(
            tenant_id=attempt.tenant_id,
            provider_attempt_id=attempt.id,
            provider_id=attempt.provider_id,
            attempt_number=attempt.attempt_count + 1,
            raw_status=result.raw_status,
            payload={"quotes": result.raw_quotes, "meta": result.raw_payload},
        )
    )

    # The provider's current ask replaces the previous one outright.
    for row in list(attempt.missing_fields):
        db.delete(row)
    db.flush()
    for field in result.missing_fields:
        db.add(
            ProviderMissingFieldRequest(
                tenant_id=attempt.tenant_id,
                provider_attempt_id=attempt.id,
                field_path=field.field_path,
                label=field.label,
                input_type=field.input_type,
                choices=field.choices,
                required=field.required,
                help_text=field.help_text,
            )
        )

    if result.outcome is QuoteOutcome.QUOTED:
        for raw_quote in result.raw_quotes:
            _persist_quote(db, request, attempt, adapter, raw_quote)

    circuit_breaker.record_outcome(
        db,
        attempt.tenant_id,
        attempt.provider_id,
        result.outcome,
        error_category=result.error_category,
        settings=settings,
    )

    audit.record(
        db,
        tenant_id=attempt.tenant_id,
        action=AuditAction.PROVIDER_RESULT,
        entity_type="provider_attempt",
        entity_id=attempt.id,
        provider_id=attempt.provider_id,
        metadata={
            "outcome": result.outcome.value,
            "quotes": len(result.raw_quotes),
            "missing_fields": [f.field_path for f in result.missing_fields],
            "error_category": result.error_category,
        },
    )
    db.flush()


def _persist_quote(
    db: Session,
    request: QuoteRequest,
    attempt: ProviderAttempt,
    adapter: ProviderAdapter,
    raw_quote: dict,
) -> None:
    """Normalize and store one quote, tolerating a single malformed payload."""
    try:
        data = adapter.normalize_result(raw_quote)
    except Exception as exc:  # a broken payload must not lose the other quotes
        audit.log_event(
            30,
            "quote normalization failed",
            provider_id=attempt.provider_id,
            error=type(exc).__name__,
        )
        return

    quote = NormalizedQuote(
        tenant_id=attempt.tenant_id,
        quote_request_id=request.id,
        provider_attempt_id=attempt.id,
        provider_id=data.provider_id,
        insurer_name=data.insurer_name,
        source_channel=data.source_channel,
        product_name=data.product_name,
        provider_quote_reference=data.provider_quote_reference,
        annual_total_premium=data.annual_total_premium,
        instalment_count=data.instalment_count,
        instalment_amount=data.instalment_amount,
        instalment_total_cost=data.instalment_total_cost,
        currency=data.currency,
        liability_limit_people=data.liability_limit_people,
        liability_limit_property=data.liability_limit_property,
        driving_formula=data.driving_formula,
        deductible=data.deductible,
        percentage_excess=data.percentage_excess,
        requires_black_box=data.requires_black_box,
        requires_approved_repair_network=data.requires_approved_repair_network,
        important_exclusions=list(data.important_exclusions),
        quote_expires_at=data.quote_expires_at,
        purchase_url=data.purchase_url,
        product_document_url=data.product_document_url,
        precontractual_document_url=data.precontractual_document_url,
        raw_provider_status=data.raw_provider_status,
        is_demonstration=data.is_demonstration,
        calculation_source=data.calculation_source,
        calculation_breakdown=(
            data.calculation_breakdown.model_dump(mode="json")
            if data.calculation_breakdown is not None
            else None
        ),
    )
    db.add(quote)
    db.flush()

    for coverage in data.coverages:
        db.add(
            QuoteCoverage(
                tenant_id=attempt.tenant_id,
                quote_id=quote.id,
                code=coverage.code,
                label=coverage.label,
                included=coverage.included,
                price=coverage.price,
                limit_amount=coverage.limit_amount,
                deductible=coverage.deductible,
                notes=coverage.notes,
            )
        )
    db.flush()


def refresh_request_status(db: Session, request: QuoteRequest) -> str:
    """Recompute the request status from its attempts and outstanding jobs."""
    if request.status == RequestStatus.CANCELLED.value:
        return request.status

    attempts = _attempts(db, request)
    if not attempts:
        return request.status

    statuses = [AttemptStatus(a.status) for a in attempts]

    if any(s.is_pending for s in statuses) or job_queue.pending_count(db, request.id):
        request.status = RequestStatus.RUNNING.value
    elif any(s is AttemptStatus.MISSING_INFORMATION for s in statuses):
        request.status = RequestStatus.AWAITING_INFORMATION.value
    elif all(s is AttemptStatus.QUOTED for s in statuses):
        request.status = RequestStatus.COMPLETED.value
        request.completed_at = request.completed_at or _now()
    elif any(s is AttemptStatus.QUOTED for s in statuses):
        request.status = RequestStatus.PARTIALLY_COMPLETED.value
        request.completed_at = request.completed_at or _now()
    else:
        request.status = RequestStatus.FAILED.value
        request.completed_at = request.completed_at or _now()

    db.flush()
    return request.status


def _attempts(db: Session, request: QuoteRequest) -> list[ProviderAttempt]:
    return list(
        db.execute(
            select(ProviderAttempt)
            .where(
                ProviderAttempt.quote_request_id == request.id,
                ProviderAttempt.tenant_id == request.tenant_id,
            )
            .order_by(ProviderAttempt.provider_id)
        ).scalars()
    )
