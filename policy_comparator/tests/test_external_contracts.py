"""Contract tests against the real providers.

**Skipped by default, and skipped in CI.** They only run when someone
deliberately sets both flags:

    PC_RUN_EXTERNAL_CONTRACT_TESTS=true LIVE_PROVIDER_AUTOMATION=true \
        pytest policy_comparator/tests/test_external_contracts.py

Even then a provider is only exercised if it is configured *and* marked
authorized, so an unauthorized provider can never be contacted by a test run.
These exist to catch a provider changing its API or portal — they are not part
of the normal suite, because a test that depends on somebody else's website is
not a test, it is a monitor.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from policy_comparator.config import get_settings
from policy_comparator.providers import registry

RUN_EXTERNAL = os.getenv("PC_RUN_EXTERNAL_CONTRACT_TESTS", "").lower() in {"1", "true", "yes"}

pytestmark = pytest.mark.skipif(
    not RUN_EXTERNAL,
    reason="External provider contract tests are disabled. Set PC_RUN_EXTERNAL_CONTRACT_TESTS=true.",
)


def _authorized_providers() -> list[str]:
    settings = get_settings()
    if not settings.live_provider_automation:
        return []
    return [
        provider_id
        for provider_id in registry.available_provider_ids()
        if settings.provider(provider_id).authorized
        and not settings.provider(provider_id).is_mock
    ]


@pytest.mark.parametrize("provider_id", registry.available_provider_ids())
def test_provider_is_reachable(provider_id):
    """A configured, authorized provider answers a health check."""
    if provider_id not in _authorized_providers():
        pytest.skip(f"{provider_id} is not configured and authorized for live access")

    adapter = registry.build_adapter(provider_id)
    health = asyncio.run(adapter.health_check())
    try:
        assert health.live_enabled is True
        assert health.configured is True
    finally:
        asyncio.run(adapter.close())


def test_no_provider_is_live_unless_explicitly_enabled():
    """A guard that runs even here: nothing is live by accident."""
    settings = get_settings()
    if not settings.live_provider_automation:
        assert _authorized_providers() == []
