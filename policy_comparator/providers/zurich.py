"""Zurich adapter.

Public reference: https://www.zurich.it/

Zurich's public fast quote starts from the vehicle plate and the owner's date
of birth, then expands into the full questionnaire. The adapter mirrors that:
two fields to start, the rest asked only once the provider needs them.

**Integration status:** mock mode is complete and deterministic. The API
contract and portal selectors below are unverified placeholders — see the
production-readiness checklist in the README.
"""

from __future__ import annotations

from typing import Any

from ..models.enums import ProviderType
from ..schemas.profile import QuotationProfile
from ..schemas.quotes import MissingField
from ..services import field_catalog
from . import api_client, automation
from .standard import StandardAutoAdapter

#: Everything Zurich-specific about reaching Zurich lives in this block.
PORTAL_URL = "https://www.zurich.it/"
API_QUOTE_PATH = "/v1/motor/quotes"

#: UNVERIFIED. Placeholders until the real portal flow is confirmed under an
#: authorized agreement.
SELECTORS = {
    "plate": "input[name='targa']",
    "date_of_birth": "input[name='dataNascita']",
    "submit": "button[type='submit']",
    "results": "[data-testid='quote-results']",
    "needs_more_info": "[data-testid='additional-questions']",
}


class ZurichAdapter(StandardAutoAdapter):
    provider_id = "zurich"
    display_name = "Zurich"
    provider_type = ProviderType.INSURER
    reference_url = PORTAL_URL

    required_paths = ("vehicle.plate", "customer.owner_date_of_birth")
    second_stage_paths = (
        "customer.tax_code",
        "customer.municipality",
        "customer.postcode",
        "vehicle.first_registration_date",
        "history.universal_merit_class",
    )
    mock_insurer_keys = ("zurich",)

    # -- API scaffolding ------------------------------------------------------

    def api_contract(self) -> api_client.ApiContract:
        return api_client.ApiContract(
            quote_path=API_QUOTE_PATH,
            auth_style="bearer",
            build_payload=_build_payload,
            extract_quotes=_extract_quotes,
            parse_missing_fields=_parse_missing_fields,
        )

    # -- portal scaffolding ---------------------------------------------------

    def browser_flow(self, profile: QuotationProfile) -> automation.BrowserFlow | None:
        url = self.config.portal_url or PORTAL_URL
        return automation.BrowserFlow(
            url=url,
            steps=(
                automation.Step("fill", SELECTORS["plate"], value_path="vehicle.plate"),
                automation.Step(
                    "fill",
                    SELECTORS["date_of_birth"],
                    value_path="customer.owner_date_of_birth",
                ),
                automation.Step("click", SELECTORS["submit"]),
            ),
            result_selector=SELECTORS["results"],
            missing_info_selector=SELECTORS["needs_more_info"],
            extract=_extract_from_page,
        )


def _build_payload(profile: QuotationProfile) -> dict[str, Any]:
    """UNVERIFIED placeholder mapping."""
    return {
        "vehicle": {
            "plate": profile.vehicle.plate,
            "first_registration_date": _iso(profile.vehicle.first_registration_date),
        },
        "policyholder": {
            "date_of_birth": _iso(profile.customer.owner_date_of_birth),
            "tax_code": profile.customer.tax_code,
            "postcode": profile.customer.postcode,
            "municipality": profile.customer.municipality,
        },
        "policy": {
            "start_date": _iso(profile.policy_start_date),
            "merit_class": profile.history.universal_merit_class,
            "driving_formula": profile.preferences.driving_formula,
        },
    }


def _extract_quotes(body: dict[str, Any]) -> list[dict[str, Any]]:
    """UNVERIFIED placeholder mapping."""
    return list(body.get("quotes") or [])


def _parse_missing_fields(body: dict[str, Any]) -> list[MissingField]:
    """Translate the provider's field names back into profile paths."""
    return [
        field_catalog.describe(_API_FIELD_TO_PATH[name])
        for name in body.get("missing_fields", [])
        if name in _API_FIELD_TO_PATH
    ]


_API_FIELD_TO_PATH = {
    "tax_code": "customer.tax_code",
    "postcode": "customer.postcode",
    "municipality": "customer.municipality",
    "merit_class": "history.universal_merit_class",
    "first_registration_date": "vehicle.first_registration_date",
}


async def _extract_from_page(page) -> list[dict[str, Any]]:  # pragma: no cover - needs a portal
    """Read quotes off the rendered results page.

    Left unimplemented on purpose: writing a scraper against selectors nobody
    has verified would produce silently wrong prices. Returning nothing makes
    the attempt report ``unavailable`` instead.
    """
    return []


def _iso(value) -> str | None:
    return value.isoformat() if value is not None else None
