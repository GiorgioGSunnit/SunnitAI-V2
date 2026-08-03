"""Quotation request endpoints.

Every route is authenticated and tenant-scoped. Personal data travels in
request bodies only — never in a path or query string — so it cannot end up in
an access log or a browser history entry.
"""

from __future__ import annotations

import uuid
from decimal import InvalidOperation

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select

from ..models import AuditEvent, ConsentRecord, QuoteRequest
from ..models.enums import AuditAction, FieldSource
from ..schemas.api import (
    CreatedRequestResponse,
    CreateQuoteRequest,
    RetryProviderRequest,
    UpdateMissingFieldsRequest,
    UpdatePreferencesRequest,
)
from ..services import audit, orchestrator, profile_service, results
from ..services.orchestrator import ConsentMissing, NewRequestInput, RequestNotStartable
from .deps import (
    CurrentIdentity,
    DbSession,
    enforce_quote_rate_limit,
    get_request_for_identity,
)

router = APIRouter(prefix="/api/quotes", tags=["quotes"])


@router.post("", response_model=CreatedRequestResponse, status_code=status.HTTP_201_CREATED)
def create_quote_request(
    body: CreateQuoteRequest,
    db: DbSession,
    identity: CurrentIdentity,
) -> CreatedRequestResponse:
    """Create a request from the four initial fields plus consent."""
    enforce_quote_rate_limit(identity)
    try:
        request = orchestrator.create_request(
            db,
            tenant_id=identity.tenant_id,
            actor_user_id=identity.user_id,
            actor_email=identity.email,
            data=NewRequestInput(
                vehicle_plate=body.vehicle_plate,
                owner_date_of_birth=body.owner_date_of_birth,
                customer_email=str(body.customer_email),
                policy_start_date=body.policy_start_date,
                privacy_accepted=body.privacy_accepted,
                provider_data_transfer_accepted=body.provider_data_transfer_accepted,
                selected_provider_ids=body.selected_provider_ids,
                marketing_accepted=body.marketing_accepted,
            ),
        )
    except ConsentMissing as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    db.commit()
    return CreatedRequestResponse(
        request_id=request.id,
        status=request.status,
        selected_provider_ids=list(request.selected_provider_ids),
        demonstration_data=bool(request.demonstration_data),
    )


@router.get("/{request_id}")
def read_quote_request(
    request_id: uuid.UUID, db: DbSession, identity: CurrentIdentity
) -> dict:
    """The request itself, without provider results."""
    request = get_request_for_identity(request_id, db, identity)
    bundle = profile_service.load_bundle(db, identity.tenant_id, request)

    return {
        "request_id": str(request.id),
        "status": request.status,
        "policy_start_date": request.policy_start_date.isoformat(),
        "selected_provider_ids": list(request.selected_provider_ids),
        "demonstration_data": bool(request.demonstration_data),
        "created_at": request.created_at.isoformat(),
        "started_at": request.started_at.isoformat() if request.started_at else None,
        "vehicle_plate": bundle.vehicle.plate,
        "customer_email": bundle.customer.email,
        "recommended_quote_id": (
            str(request.recommended_quote_id) if request.recommended_quote_id else None
        ),
    }


@router.post("/{request_id}/start")
def start_quote_request(
    request_id: uuid.UUID, db: DbSession, identity: CurrentIdentity
) -> dict:
    """Explicitly transmit the request to the selected providers."""
    request = get_request_for_identity(request_id, db, identity)
    try:
        queued = orchestrator.start_request(
            db,
            request,
            actor_user_id=identity.user_id,
            actor_email=identity.email,
        )
    except ConsentMissing as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except RequestNotStartable as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    db.commit()
    return {
        "request_id": str(request.id),
        "status": request.status,
        "queued_providers": [a.provider_id for a in queued],
    }


@router.get("/{request_id}/progress")
def read_progress(request_id: uuid.UUID, db: DbSession, identity: CurrentIdentity) -> dict:
    """Per-provider progress. Polled by the UI while a request is running."""
    request = get_request_for_identity(request_id, db, identity)
    orchestrator.refresh_request_status(db, request)
    payload = results.progress(db, request)
    db.commit()
    return payload


@router.get("/{request_id}/missing-fields")
def read_missing_fields(
    request_id: uuid.UUID, db: DbSession, identity: CurrentIdentity
) -> dict:
    """Outstanding questions, deduplicated across providers."""
    request = get_request_for_identity(request_id, db, identity)
    return results.missing_information(db, request)


@router.post("/{request_id}/missing-fields")
def submit_missing_fields(
    request_id: uuid.UUID,
    body: UpdateMissingFieldsRequest,
    db: DbSession,
    identity: CurrentIdentity,
) -> dict:
    """Supply the missing answers and resume the providers that asked."""
    request = get_request_for_identity(request_id, db, identity)
    try:
        resumed = orchestrator.supply_missing_information(
            db,
            request,
            body.updates,
            actor_user_id=identity.user_id,
            actor_email=identity.email,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    db.commit()
    return {
        "request_id": str(request.id),
        "status": request.status,
        "resumed_providers": resumed,
    }


@router.put("/{request_id}/preferences")
def update_preferences(
    request_id: uuid.UUID,
    body: UpdatePreferencesRequest,
    db: DbSession,
    identity: CurrentIdentity,
) -> dict:
    """Edit the coverage requirements that decide eligibility."""
    request = get_request_for_identity(request_id, db, identity)
    bundle = profile_service.load_bundle(db, identity.tenant_id, request)

    updates: dict = {}
    for field, value in body.model_dump(exclude_unset=True).items():
        if field == "required_optional_covers":
            bundle.preferences.required_optional_covers = list(value or [])
            continue
        updates[f"preferences.{field}"] = value

    try:
        changed = profile_service.apply_updates(
            db, bundle, updates, source=FieldSource.STAFF
        )
    except (ValueError, InvalidOperation) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    audit.record(
        db,
        tenant_id=identity.tenant_id,
        action=AuditAction.PROFILE_UPDATED,
        actor_user_id=identity.user_id,
        actor_email=identity.email,
        entity_type="quote_request",
        entity_id=request.id,
        metadata={"fields": sorted(changed)},
    )
    db.commit()
    return {"request_id": str(request.id), "updated_fields": changed}


@router.get("/{request_id}/results")
def read_results(request_id: uuid.UUID, db: DbSession, identity: CurrentIdentity) -> dict:
    """Recommendation, comparison, and every provider that produced nothing."""
    request = get_request_for_identity(request_id, db, identity)
    payload = results.results(db, request)

    audit.record(
        db,
        tenant_id=identity.tenant_id,
        action=AuditAction.RESULTS_VIEWED,
        actor_user_id=identity.user_id,
        actor_email=identity.email,
        entity_type="quote_request",
        entity_id=request.id,
        metadata={
            "eligible": len(payload["eligible_quotes"]),
            "ineligible": len(payload["ineligible_quotes"]),
            "unavailable": len(payload["unavailable_providers"]),
        },
    )
    db.commit()
    return payload


@router.post("/{request_id}/retry")
def retry_provider(
    request_id: uuid.UUID,
    body: RetryProviderRequest,
    db: DbSession,
    identity: CurrentIdentity,
) -> dict:
    """Re-run one provider. Providers that already succeeded are untouched."""
    request = get_request_for_identity(request_id, db, identity)
    try:
        attempt = orchestrator.retry_provider(
            db,
            request,
            body.provider_id,
            actor_user_id=identity.user_id,
            actor_email=identity.email,
        )
    except LookupError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ConsentMissing as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    db.commit()
    return {
        "request_id": str(request.id),
        "provider_id": attempt.provider_id,
        "status": attempt.status,
        "attempt_count": attempt.attempt_count,
    }


@router.post("/{request_id}/cancel")
def cancel_request(
    request_id: uuid.UUID, db: DbSession, identity: CurrentIdentity
) -> dict:
    """Cancel outstanding work. Results already received are kept."""
    request = get_request_for_identity(request_id, db, identity)
    cancelled = orchestrator.cancel_request(
        db, request, actor_user_id=identity.user_id, actor_email=identity.email
    )
    db.commit()
    return {"request_id": str(request.id), "status": request.status, "cancelled_jobs": cancelled}


@router.get("")
def list_quote_requests(
    db: DbSession, identity: CurrentIdentity, limit: int = 50
) -> dict:
    """Recent requests for this tenant, for the save/recover list."""
    limit = max(1, min(limit, 200))
    rows = list(
        db.execute(
            select(QuoteRequest)
            .where(QuoteRequest.tenant_id == identity.tenant_id)
            .order_by(QuoteRequest.created_at.desc())
            .limit(limit)
        ).scalars()
    )
    return {
        "requests": [
            {
                "request_id": str(r.id),
                "status": r.status,
                "created_at": r.created_at.isoformat(),
                "policy_start_date": r.policy_start_date.isoformat(),
                "providers": list(r.selected_provider_ids),
                "demonstration_data": bool(r.demonstration_data),
            }
            for r in rows
        ]
    }


@router.get("/{request_id}/audit")
def read_audit_history(
    request_id: uuid.UUID, db: DbSession, identity: CurrentIdentity
) -> dict:
    """Audit trail for one request. Administrators only."""
    if not identity.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Administrator role required"
        )
    request = get_request_for_identity(request_id, db, identity)

    # Events hang off three different entities — the request itself, each
    # provider attempt, and each consent record — so all three id sets have to
    # be gathered or the trail silently omits, for example, the consent events.
    consent_ids = list(
        db.execute(
            select(ConsentRecord.id).where(
                ConsentRecord.tenant_id == identity.tenant_id,
                ConsentRecord.quote_request_id == request.id,
            )
        ).scalars()
    )
    related_ids = (
        [request.id] + [a.id for a in results.attempts_for(db, request)] + consent_ids
    )

    events = list(
        db.execute(
            select(AuditEvent)
            .where(
                AuditEvent.tenant_id == identity.tenant_id,
                AuditEvent.entity_id.in_(related_ids),
            )
            .order_by(AuditEvent.created_at)
        ).scalars()
    )
    return {
        "request_id": str(request.id),
        "events": [
            {
                "action": e.action,
                "actor_email": e.actor_email,
                "provider_id": e.provider_id,
                "entity_type": e.entity_type,
                "metadata": e.metadata_json,
                "created_at": e.created_at.isoformat(),
            }
            for e in events
        ],
    }
