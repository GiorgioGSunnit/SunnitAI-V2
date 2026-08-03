"""Read models for the progress, missing-information and results screens.

Everything the UI shows is assembled here, so the API layer stays a thin
translation of HTTP to these functions.

Two presentation rules are enforced at this level rather than in the frontend,
because they are correctness properties rather than styling:

* a provider that failed, timed out or was skipped is always present in the
  payload — results are never quietly narrowed to the providers that worked;
* the same question asked by two providers appears once, with both providers
  attributed.
"""

from __future__ import annotations

import uuid
from decimal import Decimal
from typing import Any
from urllib.parse import urlparse

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import NormalizedQuote, ProviderAttempt, QuoteRequest
from ..models.enums import AttemptStatus
from ..providers import registry
from ..schemas.profile import CoveragePreferenceData
from ..schemas.quotes import CalculationBreakdown, CoverageData, NormalizedQuoteData
from . import deduplication, field_catalog, profile_service, recommendation

#: Statuses that mean "this provider produced nothing usable".
_UNAVAILABLE_STATUSES = {
    AttemptStatus.UNAVAILABLE,
    AttemptStatus.TIMED_OUT,
    AttemptStatus.FAILED,
    AttemptStatus.MANUAL_ACTION_REQUIRED,
    AttemptStatus.AUTHENTICATION_REQUIRED,
    AttemptStatus.CONFIGURATION_ERROR,
    AttemptStatus.SKIPPED_CIRCUIT_OPEN,
    AttemptStatus.CANCELLED,
}

_STATUS_LABELS_IT = {
    AttemptStatus.WAITING: "In attesa",
    AttemptStatus.RUNNING: "In corso",
    AttemptStatus.RETRYING: "Nuovo tentativo",
    AttemptStatus.QUOTED: "Preventivo ricevuto",
    AttemptStatus.MISSING_INFORMATION: "Servono altri dati",
    AttemptStatus.UNAVAILABLE: "Non disponibile",
    AttemptStatus.TIMED_OUT: "Tempo scaduto",
    AttemptStatus.MANUAL_ACTION_REQUIRED: "Intervento manuale necessario",
    AttemptStatus.AUTHENTICATION_REQUIRED: "Credenziali non valide",
    AttemptStatus.CONFIGURATION_ERROR: "Errore di configurazione",
    AttemptStatus.FAILED: "Errore",
    AttemptStatus.CANCELLED: "Annullato",
    AttemptStatus.SKIPPED_CIRCUIT_OPEN: "Sospeso dopo errori ripetuti",
}


def _display_name(provider_id: str) -> str:
    try:
        return registry.adapter_class(provider_id).display_name
    except KeyError:
        return provider_id


def _money(value: Decimal | None) -> str | None:
    return str(value) if value is not None else None


def attempts_for(db: Session, request: QuoteRequest) -> list[ProviderAttempt]:
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


def progress(db: Session, request: QuoteRequest) -> dict[str, Any]:
    """Per-provider state for the progress screen."""
    attempts = attempts_for(db, request)
    providers = []
    for attempt in attempts:
        status = AttemptStatus(attempt.status)
        providers.append(
            {
                "provider_id": attempt.provider_id,
                "display_name": _display_name(attempt.provider_id),
                "provider_type": attempt.provider_type,
                "mode": attempt.provider_mode,
                "status": status.value,
                "status_label": _STATUS_LABELS_IT.get(status, status.value),
                "outcome": attempt.outcome,
                "error_category": attempt.error_category,
                "error_message": attempt.error_message,
                "attempt_count": attempt.attempt_count,
                "duration_ms": attempt.duration_ms,
                "quotes": len(attempt.quotes),
                "missing_field_count": len([f for f in attempt.missing_fields if not f.resolved]),
                "retryable": status.is_finished and status is not AttemptStatus.QUOTED,
                "finished": status.is_finished,
            }
        )

    return {
        "request_id": str(request.id),
        "status": request.status,
        "demonstration_data": bool(request.demonstration_data),
        "providers": providers,
        "pending": sum(1 for p in providers if not p["finished"]),
    }


def missing_information(db: Session, request: QuoteRequest) -> dict[str, Any]:
    """Outstanding questions, deduplicated across providers and grouped."""
    questions: dict[str, dict[str, Any]] = {}

    for attempt in attempts_for(db, request):
        if AttemptStatus(attempt.status) is not AttemptStatus.MISSING_INFORMATION:
            continue
        for row in attempt.missing_fields:
            if row.resolved:
                continue
            entry = questions.get(row.field_path)
            if entry is None:
                entry = {
                    "field_path": row.field_path,
                    "label": row.label,
                    "input_type": row.input_type,
                    "choices": row.choices,
                    "required": row.required,
                    "help_text": row.help_text,
                    "group": field_catalog.group_for(row.field_path),
                    "requested_by": [],
                }
                questions[row.field_path] = entry
            name = _display_name(attempt.provider_id)
            if name not in entry["requested_by"]:
                entry["requested_by"].append(name)
            # A field only one provider asks for is still required for *that*
            # provider, so the strictest requirement wins.
            entry["required"] = entry["required"] or row.required

    groups: dict[str, list[dict]] = {}
    for entry in questions.values():
        groups.setdefault(entry["group"], []).append(entry)

    return {
        "request_id": str(request.id),
        "status": request.status,
        "groups": [
            {
                "group": group,
                "label": field_catalog.GROUP_LABELS.get(group, group),
                "fields": sorted(fields, key=lambda f: f["label"]),
            }
            for group, fields in sorted(groups.items())
        ],
        "total_questions": len(questions),
    }


def _to_schema(quote: NormalizedQuote) -> NormalizedQuoteData:
    return NormalizedQuoteData(
        provider_id=quote.provider_id,
        insurer_name=quote.insurer_name,
        source_channel=quote.source_channel,
        product_name=quote.product_name,
        provider_quote_reference=quote.provider_quote_reference,
        annual_total_premium=quote.annual_total_premium,
        instalment_count=quote.instalment_count,
        instalment_amount=quote.instalment_amount,
        instalment_total_cost=quote.instalment_total_cost,
        currency=quote.currency,
        liability_limit_people=quote.liability_limit_people,
        liability_limit_property=quote.liability_limit_property,
        driving_formula=quote.driving_formula,
        deductible=quote.deductible,
        percentage_excess=quote.percentage_excess,
        requires_black_box=quote.requires_black_box,
        requires_approved_repair_network=quote.requires_approved_repair_network,
        coverages=[
            CoverageData(
                code=c.code,
                label=c.label,
                included=c.included,
                price=c.price,
                limit_amount=c.limit_amount,
                deductible=c.deductible,
                notes=c.notes,
            )
            for c in quote.coverages
        ],
        important_exclusions=list(quote.important_exclusions or []),
        quote_expires_at=quote.quote_expires_at,
        purchase_url=quote.purchase_url,
        product_document_url=quote.product_document_url,
        precontractual_document_url=quote.precontractual_document_url,
        raw_provider_status=quote.raw_provider_status,
        is_demonstration=quote.is_demonstration,
        calculation_source=quote.calculation_source,
        calculation_breakdown=(
            CalculationBreakdown.model_validate(quote.calculation_breakdown)
            if quote.calculation_breakdown
            else None
        ),
    )


#: Only these schemes may ever reach an href. A provider-supplied
#: `javascript:` or `data:` URL rendered into a link would be a script
#: injection with the provider as the attacker.
_SAFE_URL_SCHEMES = {"https"}


def safe_external_url(raw: str | None) -> str | None:
    """Return the URL only if it is safe to put in an ``href``.

    Anything that is not an absolute ``https`` URL with a host is dropped
    rather than sanitized — a partially-repaired URL is a guess, and the UI
    degrades cleanly to no link at all.
    """
    if not raw or not isinstance(raw, str):
        return None
    try:
        parsed = urlparse(raw.strip())
    except ValueError:
        return None
    if parsed.scheme.lower() not in _SAFE_URL_SCHEMES:
        return None
    if not parsed.netloc:
        return None
    return raw.strip()


def _satisfied_requirements(
    data: NormalizedQuoteData, preferences: CoveragePreferenceData
) -> list[str]:
    """The customer requirements this quote demonstrably meets.

    Only requirements the customer actually set are listed — telling someone a
    quote satisfies a constraint they never asked for is noise.
    """
    satisfied: list[str] = []

    if preferences.min_liability_limit_people is not None and data.liability_limit_people:
        satisfied.append(
            f"Massimale danni a persone ≥ {_money(preferences.min_liability_limit_people)} €"
        )
    if preferences.min_liability_limit_property is not None and data.liability_limit_property:
        satisfied.append(
            f"Massimale danni a cose ≥ {_money(preferences.min_liability_limit_property)} €"
        )
    if preferences.max_acceptable_deductible is not None and data.deductible is not None:
        satisfied.append(
            f"Franchigia entro {_money(preferences.max_acceptable_deductible)} €"
        )
    if preferences.driving_formula:
        satisfied.append(
            "Formula di guida richiesta: "
            + {
                "free": "guida libera",
                "expert": "guida esperta",
                "exclusive": "guida esclusiva",
            }.get(preferences.driving_formula, preferences.driving_formula)
        )
    for code in preferences.required_optional_covers or []:
        label = next(
            (c.label for c in data.coverages if c.code == code and c.included), code
        )
        satisfied.append(f"Garanzia richiesta inclusa: {label}")
    if preferences.accepts_black_box is False:
        satisfied.append("Nessun obbligo di scatola nera")
    if preferences.accepts_approved_repair_network is False:
        satisfied.append("Nessun obbligo di carrozzerie convenzionate")

    return satisfied


def _serialize_quote(
    quote: NormalizedQuote,
    data: NormalizedQuoteData,
    *,
    channels: list[str],
    preferences: CoveragePreferenceData | None = None,
) -> dict[str, Any]:
    purchase_url = safe_external_url(quote.purchase_url)
    return {
        "calculation_source": quote.calculation_source,
        "calculation_breakdown": (
            data.calculation_breakdown.model_dump(mode="json")
            if data.calculation_breakdown is not None
            else None
        ),
        "satisfied_requirements": (
            _satisfied_requirements(data, preferences) if preferences else []
        ),
        #: True when the purchase link is a demonstration placeholder that must
        #: not be followed. The UI intercepts the click instead of navigating.
        "purchase_url_is_demonstration": bool(quote.is_demonstration),
        "quote_id": str(quote.id),
        "provider_id": quote.provider_id,
        "provider_display_name": _display_name(quote.provider_id),
        "insurer_name": quote.insurer_name,
        "source_channel": quote.source_channel,
        "also_available_via": [c for c in channels if c != quote.source_channel],
        "product_name": quote.product_name,
        "quote_reference": quote.provider_quote_reference,
        "annual_total_premium": _money(quote.annual_total_premium),
        "currency": quote.currency,
        "instalments": (
            {
                "count": quote.instalment_count,
                "amount": _money(quote.instalment_amount),
                "total": _money(quote.instalment_total_cost),
            }
            if quote.instalment_count
            else None
        ),
        "liability_limit_people": _money(quote.liability_limit_people),
        "liability_limit_property": _money(quote.liability_limit_property),
        "driving_formula": quote.driving_formula,
        "deductible": _money(quote.deductible),
        "percentage_excess": quote.percentage_excess,
        "requires_black_box": quote.requires_black_box,
        "requires_approved_repair_network": quote.requires_approved_repair_network,
        "included_coverages": [
            {"code": c.code, "label": c.label, "price": _money(c.price)}
            for c in data.included_coverages
        ],
        "optional_coverages": [
            {"code": c.code, "label": c.label, "price": _money(c.price)}
            for c in data.optional_coverages
        ],
        "important_exclusions": list(quote.important_exclusions or []),
        "quote_expires_at": (
            quote.quote_expires_at.isoformat() if quote.quote_expires_at else None
        ),
        "purchase_url": purchase_url,
        "product_document_url": safe_external_url(quote.product_document_url),
        "precontractual_document_url": safe_external_url(quote.precontractual_document_url),
        "is_demonstration": quote.is_demonstration,
    }


def results(db: Session, request: QuoteRequest) -> dict[str, Any]:
    """The full results payload: recommendation, comparison, and every gap."""
    quotes = list(
        db.execute(
            select(NormalizedQuote).where(
                NormalizedQuote.quote_request_id == request.id,
                NormalizedQuote.tenant_id == request.tenant_id,
            )
        ).scalars()
    )

    pairs = [(str(q.id), _to_schema(q)) for q in quotes]
    by_id = {str(q.id): q for q in quotes}

    dedupe = deduplication.deduplicate(pairs)
    # Persist the link so the duplicate stays traceable to its primary.
    for duplicate_id, primary_id in dedupe.duplicate_to_primary.items():
        by_id[duplicate_id].duplicate_of_quote_id = uuid.UUID(primary_id)

    bundle = profile_service.load_bundle(db, request.tenant_id, request)
    preferences = CoveragePreferenceData(
        base_rc=bool(bundle.preferences.base_rc),
        min_liability_limit_people=bundle.preferences.min_liability_limit_people,
        min_liability_limit_property=bundle.preferences.min_liability_limit_property,
        driving_formula=bundle.preferences.driving_formula,
        max_acceptable_deductible=bundle.preferences.max_acceptable_deductible,
        required_optional_covers=list(bundle.preferences.required_optional_covers or []),
        accepts_black_box=bundle.preferences.accepts_black_box,
        accepts_approved_repair_network=bundle.preferences.accepts_approved_repair_network,
        payment_frequency=bundle.preferences.payment_frequency,
    )

    outcome = recommendation.recommend(
        pairs, preferences, duplicate_ids=dedupe.duplicate_ids
    )
    request.recommended_quote_id = (
        uuid.UUID(outcome.recommended_quote_id) if outcome.recommended_quote_id else None
    )

    evaluations = {e.quote_id: e for e in outcome.evaluations}

    eligible_payload = []
    ineligible_payload = []
    for quote_id, data in pairs:
        quote = by_id[quote_id]
        channels = dedupe.channels_by_primary.get(quote_id, [quote.source_channel])
        serialized = _serialize_quote(
            quote, data, channels=channels, preferences=preferences
        )
        evaluation = evaluations[quote_id]

        if evaluation.eligible:
            serialized["recommended"] = quote_id == outcome.recommended_quote_id
            eligible_payload.append(serialized)
        else:
            serialized["ineligible_reasons"] = [
                {"code": r.code, "message": r.message, "detail": r.detail}
                for r in evaluation.reasons
            ]
            serialized["duplicate_of_quote_id"] = dedupe.duplicate_to_primary.get(quote_id)
            ineligible_payload.append(serialized)

    order = {qid: i for i, qid in enumerate([outcome.recommended_quote_id or ""] + outcome.alternatives)}
    eligible_payload.sort(key=lambda q: order.get(q["quote_id"], 10_000))

    unavailable = []
    for attempt in attempts_for(db, request):
        status = AttemptStatus(attempt.status)
        if status not in _UNAVAILABLE_STATUSES:
            continue
        unavailable.append(
            {
                "provider_id": attempt.provider_id,
                "display_name": _display_name(attempt.provider_id),
                "status": status.value,
                "status_label": _STATUS_LABELS_IT.get(status, status.value),
                "error_category": attempt.error_category,
                "error_message": attempt.error_message,
                "retryable": True,
            }
        )

    db.flush()

    return {
        "request_id": str(request.id),
        "status": request.status,
        "demonstration_data": bool(request.demonstration_data)
        or any(q.is_demonstration for q in quotes),
        "recommended_quote_id": outcome.recommended_quote_id,
        "recommendation_explanation": outcome.explanation,
        "recommendation_code": outcome.explanation_code,
        "eligible_quotes": eligible_payload,
        "ineligible_quotes": ineligible_payload,
        "unavailable_providers": unavailable,
        "requirements": {
            "min_liability_limit_people": _money(preferences.min_liability_limit_people),
            "min_liability_limit_property": _money(preferences.min_liability_limit_property),
            "driving_formula": preferences.driving_formula,
            "max_acceptable_deductible": _money(preferences.max_acceptable_deductible),
            "required_optional_covers": list(preferences.required_optional_covers),
            "accepts_black_box": preferences.accepts_black_box,
            "accepts_approved_repair_network": preferences.accepts_approved_repair_network,
            "payment_frequency": preferences.payment_frequency,
        },
    }
