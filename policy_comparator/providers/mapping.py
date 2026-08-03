"""Mapping from the demonstration payload shape onto the common quote structure.

Every mock adapter shares this mapper because they all emit the same payload
shape. A live adapter maps its own provider's response instead — that is the
whole point of ``normalize_result`` being part of the adapter contract.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from ..schemas.quotes import (
    CALCULATION_SOURCE_DEMONSTRATION,
    CALCULATION_SOURCE_PROVIDER,
    CalculationBreakdown,
    CalculationStep,
    CoverageData,
    NormalizedQuoteData,
)


def _normalize_calculation(
    payload: dict[str, Any], *, is_demonstration: bool
) -> tuple[str, CalculationBreakdown | None]:
    """Read a calculation breakdown off a provider payload, if there is one.

    A breakdown is only ever accepted for a demonstration quote. A live
    provider response claiming to carry our demonstration formula is ignored
    outright: the only honest statement about a real quote is that the insurer
    supplied the price.
    """
    if not is_demonstration:
        return CALCULATION_SOURCE_PROVIDER, None

    raw = payload.get("calculation")
    if not isinstance(raw, dict):
        return CALCULATION_SOURCE_PROVIDER, None

    source = raw.get("source") or CALCULATION_SOURCE_DEMONSTRATION
    if source != CALCULATION_SOURCE_DEMONSTRATION:
        return CALCULATION_SOURCE_PROVIDER, None

    steps = [
        CalculationStep(
            code=str(step.get("code") or ""),
            label=str(step.get("label") or ""),
            kind=str(step.get("kind") or "factor"),
            factor=_decimal_string(step.get("factor")),
            value=_decimal_string(step.get("value")),
            running=_decimal_string(step.get("running")),
            detail=step.get("detail"),
        )
        for step in raw.get("steps", [])
        if isinstance(step, dict)
    ]

    return source, CalculationBreakdown(
        source=source,
        currency=raw.get("currency") or "EUR",
        rounding=raw.get("rounding"),
        annual_total=_decimal_string(raw.get("annual_total")),
        steps=steps,
    )


def _decimal_string(value: Any) -> str | None:
    """Keep money and factors as exact decimal strings, never floats."""
    if value is None or value == "":
        return None
    if isinstance(value, float):
        raise TypeError("Calculation values must be decimal strings, not floats")
    return str(value)


def _decimal(value: Any) -> Decimal | None:
    if value is None or value == "":
        return None
    if isinstance(value, float):
        raise TypeError("Provider payloads must carry money as strings, not floats")
    try:
        return Decimal(str(value))
    except InvalidOperation:
        return None


def _dt(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def normalize_offer_payload(
    payload: dict[str, Any],
    *,
    provider_id: str,
    default_source_channel: str = "direct",
) -> NormalizedQuoteData:
    premium = payload.get("premium") or {}
    instalments = premium.get("instalments") or {}
    liability = payload.get("liability") or {}
    conditions = payload.get("conditions") or {}
    links = payload.get("links") or {}

    coverages = [
        CoverageData(
            code=c["code"],
            label=c["label"],
            included=bool(c.get("included", True)),
            price=_decimal(c.get("price")),
            limit_amount=_decimal(c.get("limit_amount")),
            deductible=_decimal(c.get("deductible")),
            notes=c.get("notes"),
        )
        for c in payload.get("coverages", [])
    ]

    is_demonstration = bool(payload.get("demonstration", False))
    calculation_source, breakdown = _normalize_calculation(
        payload, is_demonstration=is_demonstration
    )

    return NormalizedQuoteData(
        provider_id=provider_id,
        insurer_name=payload.get("insurer") or provider_id,
        source_channel=payload.get("source_channel") or default_source_channel,
        product_name=payload.get("product"),
        provider_quote_reference=payload.get("quote_reference"),
        annual_total_premium=_decimal(premium.get("annual_total")),
        instalment_count=instalments.get("count"),
        instalment_amount=_decimal(instalments.get("amount")),
        instalment_total_cost=_decimal(instalments.get("total")),
        currency=premium.get("currency") or "EUR",
        liability_limit_people=_decimal(liability.get("people")),
        liability_limit_property=_decimal(liability.get("property")),
        driving_formula=payload.get("driving_formula"),
        deductible=_decimal(payload.get("deductible")),
        percentage_excess=payload.get("percentage_excess"),
        requires_black_box=conditions.get("black_box_required"),
        requires_approved_repair_network=conditions.get("approved_repair_network_required"),
        coverages=coverages,
        important_exclusions=list(payload.get("exclusions", [])),
        quote_expires_at=_dt(payload.get("expires_at")),
        purchase_url=links.get("purchase"),
        product_document_url=links.get("product_document"),
        precontractual_document_url=links.get("precontractual_document"),
        raw_provider_status=payload.get("status"),
        is_demonstration=is_demonstration,
        calculation_source=calculation_source,
        calculation_breakdown=breakdown,
    )
