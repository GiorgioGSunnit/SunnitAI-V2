"""Common dispatch shared by the auto-insurance adapters.

Chooses between mock, API and portal automation, and enforces the
configuration half of the live-submission gate. Everything provider-specific —
the URLs, the selectors, the payload mapping, which insurers can be quoted —
stays in the individual provider modules.

The priority order for reaching a provider is fixed: official/partner API
first, an authorized portal second, browser automation only where no API exists
and the client is authorized to automate that portal.
"""

from __future__ import annotations

from typing import Any

from ..models.enums import ProviderMode, QuoteOutcome
from ..schemas.profile import QuotationProfile
from ..schemas.quotes import NormalizedQuoteData, ProviderHealth, ProviderResult
from . import api_client, automation, mock_flow
from .base import ProviderAdapter
from .mapping import normalize_offer_payload


class StandardAutoAdapter(ProviderAdapter):
    """Base for the four auto-insurance providers."""

    #: Insurer keys in :data:`mock_engine.INSURERS` this provider can quote.
    mock_insurer_keys: tuple[str, ...] = ()
    #: Fields the provider asks for *after* the initial fast quote.
    second_stage_paths: tuple[str, ...] = ()
    #: "direct" for an insurer, "aggregator" when quotes are relayed.
    source_channel: str = "direct"

    # -- provider-specific hooks --------------------------------------------

    def api_contract(self) -> api_client.ApiContract:
        """The provider's API shape. Override in the provider module."""
        return api_client.ApiContract()

    def browser_flow(self, profile: QuotationProfile) -> automation.BrowserFlow | None:
        """The provider's portal flow. Override in the provider module."""
        return None

    # -- contract implementation --------------------------------------------

    async def health_check(self) -> ProviderHealth:
        allowed, reason = self.live_submission_allowed()
        configured = self.config.is_mock or bool(
            self.config.api_base_url or self.config.portal_url
        )
        return ProviderHealth(
            provider_id=self.provider_id,
            provider_type=self.provider_type.value,
            mode=self.config.mode,
            configured=configured,
            authorized=self.config.authorized,
            live_enabled=allowed,
            # Mock adapters are always ready; live reachability is not probed
            # here because an unsolicited request to a provider portal is
            # itself a form of traffic we should not generate on a health poll.
            reachable=True if self.config.is_mock else None,
            detail=(
                "Demonstration data — no provider is contacted."
                if self.config.is_mock
                else reason
            ),
        )

    async def request_quote(
        self, profile: QuotationProfile, idempotency_key: str
    ) -> ProviderResult:
        return await self._dispatch(profile, idempotency_key, resume_token=None)

    async def resume_quote(
        self,
        resume_token: dict | None,
        profile: QuotationProfile,
        idempotency_key: str,
    ) -> ProviderResult:
        return await self._dispatch(profile, idempotency_key, resume_token=resume_token)

    async def _dispatch(
        self,
        profile: QuotationProfile,
        idempotency_key: str,
        *,
        resume_token: dict | None,
    ) -> ProviderResult:
        if self.config.is_mock:
            return await mock_flow.run_mock_flow(
                provider_id=self.provider_id,
                profile=profile,
                insurer_keys=self.mock_insurer_keys,
                required_paths=self.required_paths,
                second_stage_paths=self.second_stage_paths,
                source_channel=self.source_channel,
            )

        allowed, reason = self.live_submission_allowed()
        if not allowed:
            return self._not_configured(reason or "live submission is not enabled")

        if self.config.mode == ProviderMode.API:
            return await api_client.submit_quote_request(
                provider_id=self.provider_id,
                display_name=self.display_name,
                base_url=self.config.api_base_url or "",
                api_key_env=self.config.api_key_env,
                contract=self.api_contract(),
                profile=profile,
                idempotency_key=idempotency_key,
                timeout_seconds=self.timeout_seconds,
            )

        flow = self.browser_flow(profile)
        if flow is None:
            return self._not_configured("no portal flow is defined for this provider")
        return await automation.run_browser_flow(
            flow=flow,
            profile=profile,
            provider_id=self.provider_id,
            settings=self.settings,
            attempt_id=idempotency_key,
            timeout_seconds=self.timeout_seconds,
        )

    def normalize_result(self, raw_quote: dict[str, Any]) -> NormalizedQuoteData:
        return normalize_offer_payload(
            raw_quote,
            provider_id=self.provider_id,
            default_source_channel=self.source_channel,
        )

    # -- helpers -------------------------------------------------------------

    def _unsupported_live_mode(self) -> ProviderResult:
        return ProviderResult(
            provider_id=self.provider_id,
            outcome=QuoteOutcome.CONFIGURATION_ERROR,
            error_category="live_mode_not_implemented",
            error_message=(
                f"{self.display_name}: this provider has no verified live "
                "integration yet."
            ),
        )
