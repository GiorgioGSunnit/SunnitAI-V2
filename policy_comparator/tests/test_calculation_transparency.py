"""Calculation transparency: the breakdown, its provenance, and safe links.

The central guarantee: the published steps replay to exactly the premium the
customer is shown. If the formula and its explanation could drift apart, the
explanation would be worse than none.
"""

from __future__ import annotations

import re
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pytest

from policy_comparator.providers import registry
from policy_comparator.providers.mapping import normalize_offer_payload
from policy_comparator.providers.mock_engine import (
    CALCULATION_SOURCE_DEMONSTRATION,
    INSURERS,
    build_offer,
    compute_offer_pricing,
)
from policy_comparator.schemas.profile import (
    CoveragePreferenceData,
    CustomerProfileData,
    InsuranceHistoryData,
    QuotationProfile,
    VehicleData,
)
from policy_comparator.schemas.quotes import CALCULATION_SOURCE_PROVIDER
from policy_comparator.services.results import safe_external_url

CENT = Decimal("0.01")


def deterministic_profile(**preference_overrides) -> QuotationProfile:
    """A fixed profile, so every amount below is an exact expected value."""
    return QuotationProfile(
        customer_email="cliente@esempio.it",
        policy_start_date=date(2026, 9, 1),
        customer=CustomerProfileData(
            owner_date_of_birth=date(1985, 3, 4),
            tax_code="RSSMRA85C04H501Z",
            municipality="Roma",
            province="RM",
            postcode="00184",
            first_name="Mario",
            last_name="Rossi",
            mobile_number="3331234567",
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


def steps_by_code(breakdown) -> dict:
    return {s["code"]: s for s in breakdown["steps"]}


class TestBreakdownReproducesThePremium:
    def test_replaying_the_steps_lands_on_the_published_premium(self):
        """The whole point: the explanation must be the calculation."""
        pricing = compute_offer_pricing(INSURERS["zurich"], deterministic_profile())
        steps = pricing.breakdown_payload()["steps"]

        running = None
        base_rc = None
        for step in steps:
            if step["kind"] == "base":
                running = Decimal(step["value"])
            elif step["kind"] == "factor" and base_rc is None:
                running *= Decimal(step["factor"])
            elif step["kind"] == "rounding":
                base_rc = running.quantize(CENT, rounding=ROUND_HALF_UP)
                assert base_rc == Decimal(step["value"])
                break

        total = base_rc
        for step in steps:
            if step["kind"] == "addition":
                total += Decimal(step["value"])

        assert total.quantize(CENT, rounding=ROUND_HALF_UP) == pricing.annual_total

    def test_the_documented_factors_match_the_formula(self):
        pricing = compute_offer_pricing(INSURERS["zurich"], deterministic_profile())
        steps = steps_by_code(pricing.breakdown_payload())

        assert steps["base_premium"]["value"] == "310.00"
        assert steps["insurer_multiplier"]["factor"] == "1.00"
        # CU 3 -> 1.00 + (3-1) * 0.055
        assert steps["merit_class"]["factor"] == "1.110"
        # Born 1985-03-04, policy starts 2026-09-01 -> 41 years old
        assert steps["driver_age"]["factor"] == "1.00"
        assert steps["vehicle_power"]["factor"] == "0.92"  # 51 kW
        assert steps["province"]["factor"] == "1.16"  # RM
        assert steps["claims"]["factor"] == "1.00"  # no claims declared
        assert steps["driving_formula"]["factor"] == "0.97"  # guida esperta
        assert steps["annual_total"]["value"] == str(pricing.annual_total)

    def test_the_expected_premium_is_stable(self):
        """A regression guard on the worked example used in the README."""
        pricing = compute_offer_pricing(INSURERS["zurich"], deterministic_profile())
        assert pricing.base_rc_total == Decimal("342.31")
        assert pricing.annual_total == Decimal("342.31")

    def test_every_step_carries_a_label_and_a_kind(self):
        pricing = compute_offer_pricing(INSURERS["conte"], deterministic_profile())
        for step in pricing.breakdown_payload()["steps"]:
            assert step["label"]
            assert step["kind"] in {
                "base",
                "factor",
                "rounding",
                "addition",
                "subtotal",
                "total",
            }

    def test_a_worse_merit_class_shows_a_higher_factor(self):
        low = compute_offer_pricing(INSURERS["zurich"], deterministic_profile())
        profile = deterministic_profile()
        profile.history.universal_merit_class = 14
        high = compute_offer_pricing(INSURERS["zurich"], profile)

        assert Decimal(steps_by_code(high.breakdown_payload())["merit_class"]["factor"]) > Decimal(
            steps_by_code(low.breakdown_payload())["merit_class"]["factor"]
        )
        assert high.annual_total > low.annual_total

    def test_an_undeclared_merit_class_is_labelled_as_a_default(self):
        profile = deterministic_profile()
        profile.history.universal_merit_class = None
        steps = steps_by_code(compute_offer_pricing(INSURERS["zurich"], profile).breakdown_payload())
        assert "predefinito" in steps["merit_class"]["detail"]


class TestOptionalCoveragesAndInstalments:
    def test_a_requested_cover_appears_as_an_addition_and_raises_the_total(self):
        without = compute_offer_pricing(INSURERS["zurich"], deterministic_profile())
        with_cover = compute_offer_pricing(
            INSURERS["zurich"], deterministic_profile(required_optional_covers=["furto_incendio"])
        )

        steps = steps_by_code(with_cover.breakdown_payload())
        assert "cover_furto_incendio" in steps
        addition = Decimal(steps["cover_furto_incendio"]["value"])

        # 148.00 * 1.00 insurer multiplier
        assert addition == Decimal("148.00")
        assert with_cover.annual_total == without.annual_total + addition
        assert with_cover.optional_total == addition

    def test_a_bundled_cover_is_recorded_at_zero(self):
        pricing = compute_offer_pricing(INSURERS["zurich"], deterministic_profile())
        steps = steps_by_code(pricing.breakdown_payload())
        # Zurich bundles roadside assistance into the base premium.
        assert steps["cover_assistenza_stradale"]["value"] == "0.00"
        assert pricing.optional_total == Decimal("0")

    def test_the_optional_subtotal_equals_the_sum_of_the_additions(self):
        pricing = compute_offer_pricing(
            INSURERS["allianz"],
            deterministic_profile(required_optional_covers=["furto_incendio", "cristalli"]),
        )
        payload = pricing.breakdown_payload()
        additions = sum(
            Decimal(s["value"]) for s in payload["steps"] if s["kind"] == "addition"
        )
        assert additions == pricing.optional_total
        assert Decimal(steps_by_code(payload)["optional_subtotal"]["value"]) == pricing.optional_total

    def test_instalments_add_a_surcharge_over_the_annual_premium(self):
        pricing = compute_offer_pricing(INSURERS["zurich"], deterministic_profile())
        assert pricing.instalment_total > pricing.annual_total
        assert pricing.instalment_amount * pricing.instalment_count == pricing.instalment_total

    def test_the_instalment_steps_match_the_computed_values(self):
        pricing = compute_offer_pricing(INSURERS["zurich"], deterministic_profile())
        steps = steps_by_code(pricing.breakdown_payload())
        assert Decimal(steps["instalment_surcharge"]["value"]) == pricing.instalment_total
        assert Decimal(steps["instalment_amount"]["value"]) == pricing.instalment_amount


class TestMoneyStaysDecimal:
    def test_no_float_appears_anywhere_in_the_payload(self):
        payload = build_offer("zurich", deterministic_profile(required_optional_covers=["kasko"]))

        def walk(node, path="offer"):
            if isinstance(node, float):
                pytest.fail(f"float found at {path}: {node!r}")
            if isinstance(node, dict):
                for k, v in node.items():
                    walk(v, f"{path}.{k}")
            elif isinstance(node, list):
                for i, v in enumerate(node):
                    walk(v, f"{path}[{i}]")

        walk(payload)

    def test_every_monetary_step_parses_as_an_exact_decimal(self):
        payload = build_offer("allianz", deterministic_profile())["calculation"]
        for step in payload["steps"]:
            for key in ("factor", "value", "running"):
                if step[key] is not None:
                    assert Decimal(step[key]) == Decimal(step[key])

    def test_the_mapper_refuses_a_float_in_a_calculation(self):
        payload = build_offer("zurich", deterministic_profile())
        payload["calculation"]["steps"][0]["value"] = 310.0
        with pytest.raises(TypeError):
            normalize_offer_payload(payload, provider_id="zurich")


class TestCalculationSource:
    def test_a_mock_quote_is_labelled_demonstration(self):
        quote = normalize_offer_payload(
            build_offer("zurich", deterministic_profile()), provider_id="zurich"
        )
        assert quote.calculation_source == CALCULATION_SOURCE_DEMONSTRATION
        assert quote.calculation_breakdown is not None
        assert quote.calculation_breakdown.steps

    def test_a_provider_payload_is_labelled_provider_supplied(self):
        payload = build_offer("zurich", deterministic_profile())
        payload["demonstration"] = False  # as a live adapter would report it

        quote = normalize_offer_payload(payload, provider_id="zurich")
        assert quote.calculation_source == CALCULATION_SOURCE_PROVIDER
        # The demonstration formula must never be attached to a real price.
        assert quote.calculation_breakdown is None

    def test_a_live_payload_cannot_smuggle_in_the_demo_formula(self):
        """A provider response claiming our formula is ignored, not trusted."""
        payload = build_offer("zurich", deterministic_profile())
        payload["demonstration"] = False
        payload["calculation"]["source"] = CALCULATION_SOURCE_DEMONSTRATION

        quote = normalize_offer_payload(payload, provider_id="zurich")
        assert quote.calculation_source == CALCULATION_SOURCE_PROVIDER
        assert quote.calculation_breakdown is None

    def test_a_payload_without_a_calculation_defaults_to_provider_supplied(self):
        payload = build_offer("zurich", deterministic_profile())
        payload.pop("calculation")
        quote = normalize_offer_payload(payload, provider_id="zurich")
        assert quote.calculation_source == CALCULATION_SOURCE_PROVIDER
        assert quote.calculation_breakdown is None

    def test_every_mock_adapter_produces_a_breakdown(self):
        import asyncio

        for provider_id in registry.available_provider_ids():
            adapter = registry.build_adapter(provider_id)
            result = asyncio.run(adapter.request_quote(deterministic_profile(), "k"))
            for raw in result.raw_quotes:
                quote = adapter.normalize_result(raw)
                assert quote.calculation_source == CALCULATION_SOURCE_DEMONSTRATION
                assert quote.calculation_breakdown is not None
                assert Decimal(
                    quote.calculation_breakdown.annual_total
                ) == quote.annual_total_premium


class TestPurchaseUrlSafety:
    @pytest.mark.parametrize(
        "unsafe",
        [
            "javascript:alert(1)",
            "JavaScript:alert(1)",
            "data:text/html;base64,PHNjcmlwdD4=",
            "vbscript:msgbox(1)",
            "file:///etc/passwd",
            "http://example.com/quote",  # plain http is not accepted either
            "//example.com/quote",
            "not a url",
            "",
            None,
        ],
    )
    def test_unsafe_urls_are_rejected(self, unsafe):
        assert safe_external_url(unsafe) is None

    def test_https_urls_are_allowed(self):
        url = "https://example-demo.invalid/zurich/quote/ZUR-123"
        assert safe_external_url(url) == url

    def test_surrounding_whitespace_is_trimmed(self):
        assert safe_external_url("  https://example.com/x  ") == "https://example.com/x"


class TestFrontendRequirementsForm:
    """Guards the exact regression: the button was outside the form.

    There is no JS runtime in this suite, so this asserts the structural
    property that broke — the submit control must be built into the form.
    """

    @property
    def source(self) -> str:
        return (
            Path(__file__).resolve().parent.parent / "frontend" / "assets" / "app.js"
        ).read_text(encoding="utf-8")

    def test_the_save_button_is_a_submit_control(self):
        assert re.search(
            r"id:\s*'save-requirements'", self.source
        ), "the requirements save button should be identifiable"
        block = self.source.split("const save = el('button', {")[1].split("});")[0]
        assert "type: 'submit'" in block

    def test_the_button_is_appended_inside_the_form(self):
        form_block = self.source.split("const form = el('form', { id: 'form-requirements'")[1]
        form_children = form_block.split("]);")[0]
        assert "save" in form_children, "the submit button must be a child of the form"

    def test_the_form_submits_to_the_preferences_endpoint(self):
        assert "/preferences`, { method: 'PUT'" in self.source

    def test_the_button_is_disabled_while_saving(self):
        assert "save.disabled = true" in self.source

    def test_a_failure_is_reported_in_an_accessible_status_region(self):
        assert "'aria-live': 'polite'" in self.source
        assert "form-status is-error" in self.source

    def test_results_are_reloaded_after_a_successful_save(self):
        # The handler runs from its registration to where the form is mounted.
        handler = self.source.split("form.addEventListener('submit'")[1].split(
            "host.appendChild"
        )[0]
        assert "loadResults()" in handler
        assert "PUT" in handler

    def test_false_booleans_are_preserved_not_dropped(self):
        assert "value === 'false' ? false" in self.source

    def test_untrusted_values_are_never_written_as_html(self):
        """No value from the API may be assigned as markup.

        Matches actual assignments and object keys, so the word appearing in a
        comment explaining the rule does not trip the check.
        """
        assert not re.search(r"\.innerHTML\s*=", self.source)
        assert not re.search(r"\binnerHTML\s*:", self.source)
        assert not re.search(r"insertAdjacentHTML|document\.write", self.source)
