"""Provider catalogue and health endpoints."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter

from ..config import get_settings
from ..providers import registry
from ..services import circuit_breaker
from .deps import CurrentIdentity, DbSession

router = APIRouter(prefix="/api/providers", tags=["providers"])


@router.get("")
def list_providers(identity: CurrentIdentity) -> dict:
    """Providers available for selection, with their configured mode."""
    settings = get_settings()
    return {
        "providers": registry.describe_providers(settings),
        "live_provider_automation": settings.live_provider_automation,
    }


@router.get("/health")
def provider_health(db: DbSession, identity: CurrentIdentity) -> dict:
    """Configuration and circuit-breaker state, per provider, for this tenant."""
    settings = get_settings()

    async def gather() -> list:
        adapters = [
            registry.build_adapter(pid, settings) for pid in registry.available_provider_ids()
        ]
        try:
            return list(await asyncio.gather(*(a.health_check() for a in adapters)))
        finally:
            await asyncio.gather(*(a.close() for a in adapters), return_exceptions=True)

    healths = asyncio.run(gather())

    payload = []
    for health in healths:
        state = circuit_breaker.get_state(
            db, identity.tenant_id, health.provider_id, create=False
        )
        item = health.model_dump(mode="json")
        item["display_name"] = registry.adapter_class(health.provider_id).display_name
        if state is not None:
            item.update(
                circuit_open=circuit_breaker.is_open(
                    db, identity.tenant_id, health.provider_id
                ),
                circuit_open_until=(
                    state.circuit_open_until.isoformat() if state.circuit_open_until else None
                ),
                consecutive_failures=state.consecutive_failures,
                last_success_at=(
                    state.last_success_at.isoformat() if state.last_success_at else None
                ),
                last_failure_at=(
                    state.last_failure_at.isoformat() if state.last_failure_at else None
                ),
                last_error_category=state.last_error_category,
                total_successes=state.total_successes,
                total_failures=state.total_failures,
            )
        payload.append(item)

    return {"providers": payload}
