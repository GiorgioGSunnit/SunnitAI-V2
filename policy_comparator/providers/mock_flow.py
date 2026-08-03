"""Shared mock behaviour for every provider adapter.

All four providers run the same three-phase conversation in mock mode:

1. the fast-quote fields are checked locally,
2. the provider asks for the extra fields it needs (the missing-information
   round trip the real portals also perform),
3. the quote(s) come back.

Keeping this in one place means the four adapters differ only in *what* they
ask for and *which* insurers they can quote — which is the real difference
between them.
"""

from __future__ import annotations

import asyncio

from ..models.enums import QuoteOutcome
from ..schemas.profile import QuotationProfile
from ..schemas.quotes import MissingField, ProviderResult
from ..services import field_catalog
from . import mock_engine


def _forced(provider_id: str) -> ProviderResult | None:
    """Honour ``PC_MOCK_FORCE_OUTCOME_<PROVIDER>`` if it is set."""
    raw = mock_engine.forced_outcome(provider_id)
    if not raw:
        return None
    try:
        outcome = QuoteOutcome(raw.strip().lower())
    except ValueError:
        return None
    if outcome is QuoteOutcome.QUOTED:
        return None
    return ProviderResult(
        provider_id=provider_id,
        outcome=outcome,
        error_category=f"simulated_{outcome.value}",
        error_message=(
            f"{provider_id}: simulated '{outcome.value}' outcome "
            "(PC_MOCK_FORCE_OUTCOME is set for this provider)"
        ),
    )


async def run_mock_flow(
    *,
    provider_id: str,
    profile: QuotationProfile,
    insurer_keys: tuple[str, ...],
    required_paths: tuple[str, ...],
    second_stage_paths: tuple[str, ...],
    source_channel: str = "direct",
) -> ProviderResult:
    forced = _forced(provider_id)
    if forced is not None:
        return forced

    await asyncio.sleep(mock_engine.mock_latency_seconds())

    missing_paths = profile.missing_paths(list(required_paths) + list(second_stage_paths))
    if missing_paths:
        fields: list[MissingField] = [field_catalog.describe(p) for p in missing_paths]
        return ProviderResult(
            provider_id=provider_id,
            outcome=QuoteOutcome.MISSING_INFORMATION,
            missing_fields=fields,
            raw_status="NEEDS_INFO",
            resume_token={"stage": "awaiting_details", "asked_for": missing_paths},
        )

    raw_quotes = [mock_engine.build_offer(key, profile) for key in insurer_keys]
    for quote in raw_quotes:
        quote["source_channel"] = source_channel

    return ProviderResult(
        provider_id=provider_id,
        outcome=QuoteOutcome.QUOTED,
        raw_quotes=raw_quotes,
        raw_status="QUOTED",
    )
