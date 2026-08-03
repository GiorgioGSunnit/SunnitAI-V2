"""PII handling, field provenance, and the live API transport.

The transport tests run against an in-process stub, never a real provider.
Contract tests against real providers live in
``test_external_contracts.py`` and are skipped unless explicitly enabled.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date

import httpx
import pytest

from policy_comparator.models.enums import FieldSource, QuoteOutcome
from policy_comparator.providers import api_client
from policy_comparator.providers.zurich import ZurichAdapter
from policy_comparator.schemas.profile import (
    CustomerProfileData,
    QuotationProfile,
    VehicleData,
)
from policy_comparator.services import audit, field_catalog, profile_service


def _profile() -> QuotationProfile:
    return QuotationProfile(
        customer_email="cliente@esempio.it",
        policy_start_date=date(2026, 9, 1),
        customer=CustomerProfileData(
            owner_date_of_birth=date(1985, 3, 4), tax_code="RSSMRA85C04H501Z"
        ),
        vehicle=VehicleData(plate="AB123CD"),
    )


class TestScrubbing:
    @pytest.mark.parametrize(
        "text, secret",
        [
            ("Errore per RSSMRA85C04H501Z", "RSSMRA85C04H501Z"),
            ("Contatto cliente@esempio.it fallito", "cliente@esempio.it"),
            ("Numero 3331234567 non valido", "3331234567"),
            ("Targa AB123CD sconosciuta", "AB123CD"),
        ],
    )
    def test_personal_data_is_removed(self, text, secret):
        cleaned = audit.scrub(text)
        assert secret not in cleaned

    def test_ordinary_text_survives(self):
        assert audit.scrub("Provider timeout after 60s") == "Provider timeout after 60s"

    def test_none_passes_through(self):
        assert audit.scrub(None) is None

    def test_log_events_are_scrubbed(self, caplog):
        with caplog.at_level(logging.INFO, logger="policy_comparator"):
            audit.log_event(logging.INFO, "attempt failed", detail="email cliente@esempio.it")
        assert "cliente@esempio.it" not in caplog.text


class TestFieldProvenance:
    """A provider may fill a blank, but must not overwrite a staff answer."""

    def _bundle(self, db):
        from policy_comparator.services.orchestrator import NewRequestInput, create_request
        from policy_comparator.tests.conftest import TENANT_A

        request = create_request(
            db,
            tenant_id=TENANT_A,
            actor_user_id=None,
            actor_email=None,
            data=NewRequestInput(
                vehicle_plate="AB123CD",
                owner_date_of_birth=date(1985, 3, 4),
                customer_email="cliente@esempio.it",
                policy_start_date=date(2026, 9, 1),
                privacy_accepted=True,
                provider_data_transfer_accepted=True,
                selected_provider_ids=["zurich"],
            ),
        )
        db.commit()
        return profile_service.load_bundle(db, TENANT_A, request), request

    def test_a_provider_can_fill_an_empty_field(self, db):
        bundle, _ = self._bundle(db)
        changed = profile_service.apply_updates(
            db, bundle, {"vehicle.make": "Fiat"}, source=FieldSource.PROVIDER
        )
        assert changed == ["vehicle.make"]
        assert bundle.vehicle.make == "Fiat"

    def test_a_provider_cannot_overwrite_a_staff_value(self, db):
        bundle, _ = self._bundle(db)
        profile_service.apply_updates(
            db, bundle, {"vehicle.make": "Fiat"}, source=FieldSource.STAFF
        )

        changed = profile_service.apply_updates(
            db, bundle, {"vehicle.make": "Lancia"}, source=FieldSource.PROVIDER
        )
        assert changed == []
        assert bundle.vehicle.make == "Fiat"
        assert profile_service.protected_paths(bundle, {"vehicle.make": "Lancia"}) == [
            "vehicle.make"
        ]

    def test_staff_can_correct_a_provider_value(self, db):
        bundle, _ = self._bundle(db)
        profile_service.apply_updates(
            db, bundle, {"vehicle.make": "Fiat"}, source=FieldSource.PROVIDER
        )
        changed = profile_service.apply_updates(
            db, bundle, {"vehicle.make": "Lancia"}, source=FieldSource.STAFF
        )
        assert changed == ["vehicle.make"]
        assert bundle.vehicle.make == "Lancia"

    def test_values_are_coerced_to_the_column_type(self, db):
        bundle, _ = self._bundle(db)
        profile_service.apply_updates(
            db,
            bundle,
            {
                "history.universal_merit_class": "3",
                "vehicle.first_registration_date": "2019-05-10",
                "vehicle.towing_hook": "true",
            },
            source=FieldSource.STAFF,
        )
        assert bundle.history.universal_merit_class == 3
        assert bundle.vehicle.first_registration_date == date(2019, 5, 10)
        assert bundle.vehicle.towing_hook is True

    def test_an_unknown_path_is_rejected(self, db):
        bundle, _ = self._bundle(db)
        with pytest.raises(ValueError):
            profile_service.apply_updates(
                db, bundle, {"vehicle.colour": "red"}, source=FieldSource.STAFF
            )


class TestFieldCatalogue:
    def test_the_same_path_always_yields_the_same_question(self):
        """Deduplication across providers depends on this."""
        first = field_catalog.describe("history.universal_merit_class")
        second = field_catalog.describe("history.universal_merit_class")
        assert first == second
        assert first.label == "Classe di merito universale (CU)"

    def test_choice_fields_carry_their_options(self):
        formula = field_catalog.describe("preferences.driving_formula")
        assert formula.input_type == "choice"
        assert {c["value"] for c in formula.choices} == {"free", "expert", "exclusive"}

    def test_an_unknown_path_degrades_to_a_text_input(self):
        described = field_catalog.describe("vehicle.something_new")
        assert described.input_type == "text"
        assert described.field_path == "vehicle.something_new"

    def test_every_adapter_asks_only_for_catalogued_fields(self):
        from policy_comparator.providers import registry

        for provider_id in registry.available_provider_ids():
            adapter = registry.build_adapter(provider_id)
            for path in adapter.required_paths + adapter.second_stage_paths:
                assert field_catalog.is_known(path), f"{provider_id} asks for uncatalogued {path}"


class TestApiTransport:
    """The HTTP layer of the live adapters, against an in-process stub."""

    def _adapter(self) -> ZurichAdapter:
        from policy_comparator.config import get_settings

        settings = get_settings()
        return ZurichAdapter(settings, settings.provider("zurich"))

    def _call(self, handler, monkeypatch, *, contract=None, api_key="secret"):
        transport = httpx.MockTransport(handler)
        original = httpx.AsyncClient

        def patched(*args, **kwargs):
            kwargs["transport"] = transport
            return original(*args, **kwargs)

        monkeypatch.setattr(httpx, "AsyncClient", patched)
        monkeypatch.setenv("ZURICH_API_KEY", api_key)

        adapter = self._adapter()
        return asyncio.run(
            api_client.submit_quote_request(
                provider_id="zurich",
                display_name="Zurich",
                base_url="https://api.example.invalid",
                api_key_env="ZURICH_API_KEY" if api_key else None,
                contract=contract or adapter.api_contract(),
                profile=_profile(),
                idempotency_key="idem-1",
                timeout_seconds=5,
            )
        )

    def test_a_successful_response_becomes_a_quote(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.headers["Idempotency-Key"] == "idem-1"
            assert request.headers["Authorization"] == "Bearer secret"
            return httpx.Response(200, json={"quotes": [{"insurer": "Zurich"}]})

        result = self._call(handler, monkeypatch)
        assert result.outcome is QuoteOutcome.QUOTED
        assert result.raw_quotes == [{"insurer": "Zurich"}]

    def test_a_401_asks_for_credentials_not_a_retry(self, monkeypatch):
        result = self._call(lambda r: httpx.Response(401, json={}), monkeypatch)
        assert result.outcome is QuoteOutcome.AUTHENTICATION_REQUIRED
        assert result.outcome.is_retryable is False

    def test_a_500_is_unavailable_and_retryable(self, monkeypatch):
        result = self._call(lambda r: httpx.Response(503, json={}), monkeypatch)
        assert result.outcome is QuoteOutcome.UNAVAILABLE
        assert result.outcome.is_retryable is True

    def test_rate_limiting_is_unavailable(self, monkeypatch):
        result = self._call(lambda r: httpx.Response(429, json={}), monkeypatch)
        assert result.outcome is QuoteOutcome.UNAVAILABLE
        assert result.error_category == "api_rate_limited"

    def test_a_422_with_field_names_becomes_missing_information(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                422, json={"missing_fields": ["tax_code", "merit_class"], "quote_id": "Q1"}
            )

        result = self._call(handler, monkeypatch)
        assert result.outcome is QuoteOutcome.MISSING_INFORMATION
        assert {f.field_path for f in result.missing_fields} == {
            "customer.tax_code",
            "history.universal_merit_class",
        }
        assert result.resume_token == {"quote_id": "Q1"}

    def test_a_timeout_is_reported_as_a_timeout(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.TimeoutException("too slow", request=request)

        result = self._call(handler, monkeypatch)
        assert result.outcome is QuoteOutcome.TIMED_OUT

    def test_a_connection_failure_is_unavailable(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        result = self._call(handler, monkeypatch)
        assert result.outcome is QuoteOutcome.UNAVAILABLE

    def test_an_unexpected_shape_is_a_configuration_error_not_a_quote(self, monkeypatch):
        """A mapping that no longer matches must fail loudly, not invent a price."""
        contract = api_client.ApiContract(
            build_payload=lambda p: {},
            extract_quotes=lambda body: body["not_here"],
        )
        result = self._call(handler=lambda r: httpx.Response(200, json={}), monkeypatch=monkeypatch, contract=contract)
        assert result.outcome is QuoteOutcome.CONFIGURATION_ERROR

    def test_an_empty_result_set_is_unavailable(self, monkeypatch):
        result = self._call(lambda r: httpx.Response(200, json={"quotes": []}), monkeypatch)
        assert result.outcome is QuoteOutcome.UNAVAILABLE

    def test_a_missing_credential_is_reported_before_any_request(self, monkeypatch):
        called = False

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal called
            called = True
            return httpx.Response(200, json={"quotes": []})

        monkeypatch.delenv("ZURICH_API_KEY", raising=False)
        result = self._call(handler, monkeypatch, api_key="")
        assert result.outcome is QuoteOutcome.AUTHENTICATION_REQUIRED
        assert called is False, "no request should be sent without a credential"

    def test_an_unimplemented_contract_never_pretends_to_work(self):
        result = asyncio.run(
            api_client.submit_quote_request(
                provider_id="zurich",
                display_name="Zurich",
                base_url="https://api.example.invalid",
                api_key_env=None,
                contract=api_client.ApiContract(),  # no mappings supplied
                profile=_profile(),
                idempotency_key="idem",
                timeout_seconds=5,
            )
        )
        assert result.outcome is QuoteOutcome.CONFIGURATION_ERROR
        assert "verified" in result.error_message


class TestBrowserAutomationIsOptional:
    def test_the_application_runs_without_playwright(self):
        from policy_comparator.providers import automation

        # The import must not fail, whatever is installed.
        assert isinstance(automation.playwright_available(), bool)

    def test_a_browser_provider_without_playwright_reports_configuration_error(self):
        from policy_comparator.providers import automation

        if automation.playwright_available():
            pytest.skip("Playwright is installed in this environment")

        from policy_comparator.config import get_settings

        result = asyncio.run(
            automation.run_browser_flow(
                flow=automation.BrowserFlow(url="https://example.invalid"),
                profile=_profile(),
                provider_id="zurich",
                settings=get_settings(),
                attempt_id="a",
                timeout_seconds=1,
            )
        )
        assert result.outcome is QuoteOutcome.CONFIGURATION_ERROR
        assert result.error_category == "playwright_not_installed"

    def test_protection_markers_cover_the_common_walls(self):
        from policy_comparator.providers import automation

        for marker in ("recaptcha", "hcaptcha", "datadome", "turnstile"):
            assert marker in automation.CAPTCHA_MARKERS
        assert any("otp" in m for m in automation.MFA_MARKERS)
