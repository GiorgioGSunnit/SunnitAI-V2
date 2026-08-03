"""Provider adapter contract, mock behaviour and isolation."""

from __future__ import annotations

import asyncio
from datetime import date
from decimal import Decimal

import pytest

from policy_comparator.config import get_settings
from policy_comparator.models.enums import ProviderType, QuoteOutcome
from policy_comparator.providers import registry
from policy_comparator.providers.base import ProviderAdapter
from policy_comparator.schemas.profile import (
    CoveragePreferenceData,
    CustomerProfileData,
    InsuranceHistoryData,
    QuotationProfile,
    VehicleData,
)


def minimal_profile() -> QuotationProfile:
    """Only the four fields the initial form collects."""
    return QuotationProfile(
        customer_email="cliente@esempio.it",
        policy_start_date=date(2026, 9, 1),
        customer=CustomerProfileData(owner_date_of_birth=date(1985, 3, 4)),
        vehicle=VehicleData(plate="AB123CD"),
    )


def complete_profile(**preference_overrides) -> QuotationProfile:
    """Everything the four mock providers ask for."""
    return QuotationProfile(
        customer_email="cliente@esempio.it",
        policy_start_date=date(2026, 9, 1),
        customer=CustomerProfileData(
            owner_date_of_birth=date(1985, 3, 4),
            first_name="Mario",
            last_name="Rossi",
            tax_code="RSSMRA85C04H501Z",
            mobile_number="3331234567",
            municipality="Roma",
            province="RM",
            postcode="00184",
        ),
        vehicle=VehicleData(
            plate="AB123CD",
            first_registration_date=date(2019, 5, 10),
            make="Fiat",
            model="Panda",
            power_kw=51,
        ),
        history=InsuranceHistoryData(universal_merit_class=3),
        preferences=CoveragePreferenceData(driving_formula="expert", **preference_overrides),
    )


ALL_PROVIDERS = list(registry.available_provider_ids())


class TestContract:
    @pytest.mark.parametrize("provider_id", ALL_PROVIDERS)
    def test_adapter_implements_the_contract(self, provider_id):
        adapter = registry.build_adapter(provider_id)
        assert isinstance(adapter, ProviderAdapter)
        assert adapter.provider_id == provider_id
        assert adapter.display_name
        assert adapter.required_paths, "every adapter must declare its fast-quote fields"
        for method in ("health_check", "request_quote", "resume_quote", "normalize_result", "close"):
            assert callable(getattr(adapter, method))

    @pytest.mark.parametrize("provider_id", ALL_PROVIDERS)
    def test_health_check_reports_mock_mode(self, provider_id):
        adapter = registry.build_adapter(provider_id)
        health = asyncio.run(adapter.health_check())
        assert health.provider_id == provider_id
        assert health.mode == "mock"
        assert health.live_enabled is False
        assert health.configured is True

    def test_cercassicurazioni_is_modelled_as_an_aggregator(self):
        adapter = registry.build_adapter("cercassicurazioni")
        assert adapter.provider_type is ProviderType.AGGREGATOR
        assert adapter.source_channel == "aggregator"

    def test_the_three_insurers_are_not_aggregators(self):
        for provider_id in ("zurich", "allianz", "generali"):
            assert registry.build_adapter(provider_id).provider_type is ProviderType.INSURER

    def test_unknown_provider_raises(self):
        with pytest.raises(registry.UnknownProvider):
            registry.build_adapter("nonexistent")


class TestMissingInformationRoundTrip:
    @pytest.mark.parametrize("provider_id", ALL_PROVIDERS)
    def test_minimal_profile_asks_for_more(self, provider_id):
        adapter = registry.build_adapter(provider_id)
        result = asyncio.run(adapter.request_quote(minimal_profile(), "key-1"))

        assert result.outcome is QuoteOutcome.MISSING_INFORMATION
        assert result.missing_fields, "a provider must say what it needs"
        assert result.resume_token is not None
        assert not result.raw_quotes

    @pytest.mark.parametrize("provider_id", ALL_PROVIDERS)
    def test_complete_profile_quotes(self, provider_id):
        adapter = registry.build_adapter(provider_id)
        result = asyncio.run(adapter.request_quote(complete_profile(), "key-1"))

        assert result.outcome is QuoteOutcome.QUOTED
        assert result.raw_quotes

    @pytest.mark.parametrize("provider_id", ALL_PROVIDERS)
    def test_resume_after_the_gap_is_filled(self, provider_id):
        adapter = registry.build_adapter(provider_id)
        first = asyncio.run(adapter.request_quote(minimal_profile(), "key-1"))
        assert first.outcome is QuoteOutcome.MISSING_INFORMATION

        resumed = asyncio.run(
            adapter.resume_quote(first.resume_token, complete_profile(), "key-1")
        )
        assert resumed.outcome is QuoteOutcome.QUOTED

    def test_generali_accepts_tax_code_instead_of_date_of_birth(self):
        """Generali identifies the policyholder by either field, not both."""
        adapter = registry.build_adapter("generali")

        without_either = QuotationProfile(
            customer_email="cliente@esempio.it",
            policy_start_date=date(2026, 9, 1),
            customer=CustomerProfileData(),
            vehicle=VehicleData(plate="AB123CD"),
        )
        paths = {f.field_path for f in adapter.validate_profile(without_either)}
        assert "customer.owner_date_of_birth" in paths

        with_tax_code_only = QuotationProfile(
            customer_email="cliente@esempio.it",
            policy_start_date=date(2026, 9, 1),
            customer=CustomerProfileData(tax_code="RSSMRA85C04H501Z"),
            vehicle=VehicleData(plate="AB123CD"),
        )
        paths = {f.field_path for f in adapter.validate_profile(with_tax_code_only)}
        assert "customer.owner_date_of_birth" not in paths


class TestDeterminism:
    def test_same_profile_gives_the_same_premium(self):
        adapter = registry.build_adapter("zurich")
        first = asyncio.run(adapter.request_quote(complete_profile(), "a"))
        second = asyncio.run(adapter.request_quote(complete_profile(), "b"))

        assert adapter.normalize_result(first.raw_quotes[0]).annual_total_premium == (
            adapter.normalize_result(second.raw_quotes[0]).annual_total_premium
        )

    def test_a_worse_merit_class_costs_more(self):
        adapter = registry.build_adapter("zurich")

        def premium(merit_class: int) -> Decimal:
            profile = complete_profile()
            profile.history.universal_merit_class = merit_class
            result = asyncio.run(adapter.request_quote(profile, "k"))
            return adapter.normalize_result(result.raw_quotes[0]).annual_total_premium

        assert premium(14) > premium(1)

    def test_quotes_are_flagged_as_demonstration_data(self):
        adapter = registry.build_adapter("zurich")
        result = asyncio.run(adapter.request_quote(complete_profile(), "k"))
        assert adapter.normalize_result(result.raw_quotes[0]).is_demonstration is True

    def test_premiums_are_decimal_never_float(self):
        adapter = registry.build_adapter("allianz")
        result = asyncio.run(adapter.request_quote(complete_profile(), "k"))
        quote = adapter.normalize_result(result.raw_quotes[0])
        assert isinstance(quote.annual_total_premium, Decimal)
        assert isinstance(quote.deductible, Decimal)


class TestAggregatorRelay:
    def test_aggregator_returns_several_insurers_and_names_them(self):
        adapter = registry.build_adapter("cercassicurazioni")
        result = asyncio.run(adapter.request_quote(complete_profile(), "k"))
        quotes = [adapter.normalize_result(q) for q in result.raw_quotes]

        assert len(quotes) > 1
        # The insurer is the company carrying the risk, not the aggregator.
        assert all(q.insurer_name.lower() != "cercassicurazioni" for q in quotes)
        assert all(q.source_channel == "aggregator" for q in quotes)

    def test_the_same_offer_matches_across_channels(self):
        """A direct quote and its aggregator copy must be recognisably one offer."""
        direct = registry.build_adapter("zurich")
        aggregator = registry.build_adapter("cercassicurazioni")

        direct_quote = direct.normalize_result(
            asyncio.run(direct.request_quote(complete_profile(), "k")).raw_quotes[0]
        )
        relayed = [
            aggregator.normalize_result(q)
            for q in asyncio.run(aggregator.request_quote(complete_profile(), "k")).raw_quotes
        ]
        zurich_via_aggregator = next(q for q in relayed if q.insurer_name == "Zurich")

        assert zurich_via_aggregator.dedupe_signature() == direct_quote.dedupe_signature()
        assert zurich_via_aggregator.source_channel != direct_quote.source_channel


class TestFaultInjectionAndIsolation:
    def test_forced_outcome_makes_one_provider_fail(self, monkeypatch):
        monkeypatch.setenv("PC_MOCK_FORCE_OUTCOME_ZURICH", "unavailable")
        result = asyncio.run(
            registry.build_adapter("zurich").request_quote(complete_profile(), "k")
        )
        assert result.outcome is QuoteOutcome.UNAVAILABLE
        assert result.error_category == "simulated_unavailable"

    def test_one_provider_failing_does_not_affect_the_others(self, monkeypatch):
        monkeypatch.setenv("PC_MOCK_FORCE_OUTCOME_ZURICH", "failed")

        async def run_all():
            adapters = [registry.build_adapter(p) for p in ALL_PROVIDERS]
            return await asyncio.gather(
                *(a.request_quote(complete_profile(), "k") for a in adapters)
            )

        results = {r.provider_id: r for r in asyncio.run(run_all())}
        assert results["zurich"].outcome is QuoteOutcome.FAILED
        for other in ("allianz", "generali", "cercassicurazioni"):
            assert results[other].outcome is QuoteOutcome.QUOTED


class TestLiveSubmissionGate:
    """Configuration alone must never be enough to reach a real provider."""

    def _adapter(self, monkeypatch, **env):
        from policy_comparator import config

        for key, value in env.items():
            monkeypatch.setenv(key, value)
        config.reset_settings_cache()
        settings = get_settings()
        return registry.build_adapter("zurich", settings)

    def teardown_method(self):
        from policy_comparator import config

        config.reset_settings_cache()

    def test_mock_mode_is_never_live(self, monkeypatch):
        adapter = self._adapter(monkeypatch, PC_PROVIDER_ZURICH_MODE="mock")
        allowed, reason = adapter.live_submission_allowed()
        assert allowed is False and "mock" in reason

    def test_authorized_but_master_switch_off_is_not_live(self, monkeypatch):
        adapter = self._adapter(
            monkeypatch,
            PC_PROVIDER_ZURICH_MODE="api",
            PC_PROVIDER_ZURICH_AUTHORIZED="true",
            PC_PROVIDER_ZURICH_API_BASE_URL="https://example.invalid",
            LIVE_PROVIDER_AUTOMATION="false",
        )
        allowed, reason = adapter.live_submission_allowed()
        assert allowed is False and "LIVE_PROVIDER_AUTOMATION" in reason

    def test_master_switch_on_but_unauthorized_is_not_live(self, monkeypatch):
        adapter = self._adapter(
            monkeypatch,
            PC_PROVIDER_ZURICH_MODE="api",
            PC_PROVIDER_ZURICH_AUTHORIZED="false",
            PC_PROVIDER_ZURICH_API_BASE_URL="https://example.invalid",
            LIVE_PROVIDER_AUTOMATION="true",
        )
        allowed, reason = adapter.live_submission_allowed()
        assert allowed is False and "authorized" in reason

    def test_authorized_without_a_url_is_not_live(self, monkeypatch):
        adapter = self._adapter(
            monkeypatch,
            PC_PROVIDER_ZURICH_MODE="api",
            PC_PROVIDER_ZURICH_AUTHORIZED="true",
            LIVE_PROVIDER_AUTOMATION="true",
        )
        allowed, reason = adapter.live_submission_allowed()
        assert allowed is False and "API base URL" in reason

    def test_every_condition_together_opens_the_gate(self, monkeypatch):
        adapter = self._adapter(
            monkeypatch,
            PC_PROVIDER_ZURICH_MODE="api",
            PC_PROVIDER_ZURICH_AUTHORIZED="true",
            PC_PROVIDER_ZURICH_API_BASE_URL="https://example.invalid",
            LIVE_PROVIDER_AUTOMATION="true",
        )
        allowed, _ = adapter.live_submission_allowed()
        assert allowed is True

    def test_a_non_mock_provider_that_fails_the_gate_reports_configuration_error(
        self, monkeypatch
    ):
        adapter = self._adapter(
            monkeypatch,
            PC_PROVIDER_ZURICH_MODE="api",
            PC_PROVIDER_ZURICH_AUTHORIZED="false",
            LIVE_PROVIDER_AUTOMATION="true",
        )
        result = asyncio.run(adapter.request_quote(complete_profile(), "k"))
        # It must not silently fall back to mock data and look like a success.
        assert result.outcome is QuoteOutcome.CONFIGURATION_ERROR
        assert not result.raw_quotes
