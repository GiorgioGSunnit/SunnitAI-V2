"""Allianz adapter.

Public references:
  https://www.allianz.it/le-soluzioni-per-te/mobilita/auto.html
  https://fastquote.allianz.it/

The Allianz fast quote asks for plate, owner date of birth and an email address
up front, and follows up for a mobile number. The customer email is always
present on our side (it is one of the four initial fields), so the extra ask
here is the mobile number.

**Integration status:** mock mode is complete and deterministic. The API
contract and portal selectors below are unverified placeholders.
"""

from __future__ import annotations

from typing import Any

from ..models.enums import ProviderType
from ..schemas.profile import QuotationProfile
from ..schemas.quotes import MissingField
from ..services import field_catalog
from . import api_client, automation
from .standard import StandardAutoAdapter

PORTAL_URL = "https://fastquote.allianz.it/"
PRODUCT_URL = "https://www.allianz.it/le-soluzioni-per-te/mobilita/auto.html"
API_QUOTE_PATH = "/motor/v2/quotations"

#: UNVERIFIED placeholders.
SELECTORS = {
    "plate": "input#targa",
    "date_of_birth": "input#dataNascita",
    "email": "input#email",
    "submit": "button#calcola-preventivo",
    "results": "[data-qa='risultati-preventivo']",
    "needs_more_info": "[data-qa='dati-mancanti']",
}


class AllianzAdapter(StandardAutoAdapter):
    provider_id = "allianz"
    display_name = "Allianz"
    provider_type = ProviderType.INSURER
    reference_url = PRODUCT_URL

    required_paths = ("vehicle.plate", "customer.owner_date_of_birth")
    second_stage_paths = (
        "customer.mobile_number",
        "customer.tax_code",
        "vehicle.make",
        "vehicle.model",
        "history.universal_merit_class",
    )
    mock_insurer_keys = ("allianz",)

    def api_contract(self) -> api_client.ApiContract:
        return api_client.ApiContract(
            quote_path=API_QUOTE_PATH,
            auth_style="api_key_header",
            api_key_header="X-Allianz-Api-Key",
            build_payload=_build_payload,
            extract_quotes=_extract_quotes,
            parse_missing_fields=_parse_missing_fields,
        )

    def browser_flow(self, profile: QuotationProfile) -> automation.BrowserFlow | None:
        return automation.BrowserFlow(
            url=self.config.portal_url or PORTAL_URL,
            steps=(
                automation.Step("fill", SELECTORS["plate"], value_path="vehicle.plate"),
                automation.Step(
                    "fill",
                    SELECTORS["date_of_birth"],
                    value_path="customer.owner_date_of_birth",
                ),
                automation.Step(
                    "fill", SELECTORS["email"], value_path="customer_email", optional=True
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
        "contactEmail": str(profile.customer_email),
        "mobile": profile.customer.mobile_number,
        "vehicle": {
            "plate": profile.vehicle.plate,
            "make": profile.vehicle.make,
            "model": profile.vehicle.model,
        },
        "owner": {
            "dateOfBirth": _iso(profile.customer.owner_date_of_birth),
            "fiscalCode": profile.customer.tax_code,
        },
        "coverageStart": _iso(profile.policy_start_date),
        "meritClass": profile.history.universal_merit_class,
    }


def _extract_quotes(body: dict[str, Any]) -> list[dict[str, Any]]:
    """UNVERIFIED placeholder mapping."""
    return list(body.get("quotations") or [])


_API_FIELD_TO_PATH = {
    "mobile": "customer.mobile_number",
    "fiscalCode": "customer.tax_code",
    "make": "vehicle.make",
    "model": "vehicle.model",
    "meritClass": "history.universal_merit_class",
}


def _parse_missing_fields(body: dict[str, Any]) -> list[MissingField]:
    return [
        field_catalog.describe(_API_FIELD_TO_PATH[name])
        for name in body.get("requiredFields", [])
        if name in _API_FIELD_TO_PATH
    ]


async def _extract_from_page(page) -> list[dict[str, Any]]:  # pragma: no cover - needs a portal
    """Not implemented — see the note in the Zurich adapter."""
    return []


def _iso(value) -> str | None:
    return value.isoformat() if value is not None else None
