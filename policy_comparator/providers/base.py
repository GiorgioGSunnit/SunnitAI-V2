"""The provider adapter contract.

Every provider — insurer or aggregator, API-backed or portal-automated — is
reached only through :class:`ProviderAdapter`. The orchestrator knows nothing
about any specific provider, and an exception raised inside one adapter is
converted to that adapter's own failed result without touching the others.

Adapters must not raise for ordinary provider conditions. A timeout, an outage,
a CAPTCHA or a missing field are all *results*, expressed through
:class:`~policy_comparator.models.enums.QuoteOutcome`.
"""

from __future__ import annotations

import abc
from typing import Any

from ..config import ProviderSettings, Settings
from ..models.enums import ProviderType, QuoteOutcome
from ..schemas.profile import QuotationProfile
from ..schemas.quotes import MissingField, NormalizedQuoteData, ProviderHealth, ProviderResult


class ProviderNotAuthorized(RuntimeError):
    """Raised when a live submission is attempted without every precondition."""


class ProviderAdapter(abc.ABC):
    """Base class for all provider adapters."""

    #: Stable identifier, also the config key (``PC_PROVIDER_<ID>_*``).
    provider_id: str = ""
    #: Human-readable name shown in the UI.
    display_name: str = ""
    provider_type: ProviderType = ProviderType.INSURER
    #: Public reference page. Documentation only — never fetched automatically.
    reference_url: str | None = None

    #: Profile paths this provider needs before it can return a quote. The
    #: initial fast-quote fields are deliberately few; everything else is asked
    #: for only after the provider says it needs it.
    required_paths: tuple[str, ...] = ()

    def __init__(self, settings: Settings, provider_settings: ProviderSettings) -> None:
        self.settings = settings
        self.config = provider_settings

    # -- capability / configuration -----------------------------------------

    @property
    def mode(self) -> str:
        return self.config.mode

    @property
    def timeout_seconds(self) -> int:
        return self.config.timeout_seconds or self.settings.provider_timeout_seconds

    @property
    def retry_count(self) -> int:
        rc = self.config.retry_count
        return self.settings.provider_retry_count if rc is None or rc < 0 else rc

    def live_submission_allowed(self) -> tuple[bool, str | None]:
        """Whether this adapter may contact the real provider.

        Every condition must hold. Consent and explicit staff initiation are
        checked by the orchestrator, which owns that context; this covers the
        configuration half of the gate.
        """
        if self.config.is_mock:
            return False, "provider is configured in mock mode"
        if not self.settings.live_provider_automation:
            return False, "LIVE_PROVIDER_AUTOMATION is not enabled"
        if not self.config.authorized:
            return False, f"provider '{self.provider_id}' is not marked authorized"
        if self.config.mode == "api" and not self.config.api_base_url:
            return False, "no API base URL configured"
        if self.config.mode == "browser" and not self.config.portal_url:
            return False, "no portal URL configured"
        return True, None

    # -- contract ------------------------------------------------------------

    @abc.abstractmethod
    async def health_check(self) -> ProviderHealth:
        """Report configuration and, where cheap and safe, reachability."""

    def validate_profile(self, profile: QuotationProfile) -> list[MissingField]:
        """Fields this provider knows it needs but hasn't been given.

        Checked locally before any network call, so an obviously incomplete
        profile costs nothing. Providers may still ask for more afterwards.
        """
        return [
            self.describe_field(path) for path in profile.missing_paths(list(self.required_paths))
        ]

    def describe_field(self, path: str) -> MissingField:
        """Render a profile path as a UI question.

        Overridable per provider for wording, but the shared catalogue is what
        makes the same question from two providers deduplicate into one.
        """
        from ..services.field_catalog import describe

        return describe(path)

    @abc.abstractmethod
    async def request_quote(
        self,
        profile: QuotationProfile,
        idempotency_key: str,
    ) -> ProviderResult:
        """Submit a first quotation request."""

    @abc.abstractmethod
    async def resume_quote(
        self,
        resume_token: dict | None,
        profile: QuotationProfile,
        idempotency_key: str,
    ) -> ProviderResult:
        """Continue after the missing information has been supplied."""

    @abc.abstractmethod
    def normalize_result(self, raw_quote: dict[str, Any]) -> NormalizedQuoteData:
        """Map one provider-shaped payload onto the common quote structure."""

    async def close(self) -> None:
        """Release adapter-held resources (HTTP clients, browser contexts)."""
        return None

    # -- helpers for subclasses ---------------------------------------------

    def _not_configured(self, detail: str) -> ProviderResult:
        return ProviderResult(
            provider_id=self.provider_id,
            outcome=QuoteOutcome.CONFIGURATION_ERROR,
            error_category="provider_not_configured",
            error_message=f"{self.display_name}: {detail}",
        )

    def _manual_action(self, detail: str) -> ProviderResult:
        """Anti-bot, CAPTCHA or MFA was encountered. Never bypassed."""
        return ProviderResult(
            provider_id=self.provider_id,
            outcome=QuoteOutcome.MANUAL_ACTION_REQUIRED,
            error_category="manual_action_required",
            error_message=f"{self.display_name}: {detail}",
        )

    def _missing(self, fields: list[MissingField], resume_token: dict | None = None) -> ProviderResult:
        return ProviderResult(
            provider_id=self.provider_id,
            outcome=QuoteOutcome.MISSING_INFORMATION,
            missing_fields=fields,
            resume_token=resume_token,
        )
