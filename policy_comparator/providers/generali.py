"""Generali adapter (auto product).

Public references:
  https://www.generali.it/assicurazione-auto/immagina-strade-nuove-assicurazione-auto/
  https://www.generali.it/preventivatori/preventivatore-auto-quota-facile

Generali's quote identity accepts either the owner's date of birth *or* their
tax code, so this adapter treats the two as interchangeable rather than
demanding both.

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

PRODUCT_URL = (
    "https://www.generali.it/assicurazione-auto/immagina-strade-nuove-assicurazione-auto/"
)
PORTAL_URL = "https://www.generali.it/preventivatori/preventivatore-auto-quota-facile"
API_QUOTE_PATH = "/quotafacile/auto/preventivi"

#: UNVERIFIED placeholders.
SELECTORS = {
    "plate": "input[name='targa']",
    "tax_code": "input[name='codiceFiscale']",
    "date_of_birth": "input[name='dataDiNascita']",
    "submit": "button[data-action='calcola']",
    "results": "[data-section='risultati']",
    "needs_more_info": "[data-section='dati-aggiuntivi']",
}


class GeneraliAdapter(StandardAutoAdapter):
    provider_id = "generali"
    display_name = "Generali"
    provider_type = ProviderType.INSURER
    reference_url = PRODUCT_URL

    #: Only the plate is unconditionally required — identity is satisfied by
    #: either the date of birth or the tax code, checked in validate_profile.
    required_paths = ("vehicle.plate",)
    second_stage_paths = (
        "customer.first_name",
        "customer.last_name",
        "customer.municipality",
        "vehicle.first_registration_date",
        "history.universal_merit_class",
    )
    mock_insurer_keys = ("generali",)

    def validate_profile(self, profile: QuotationProfile) -> list[MissingField]:
        missing = super().validate_profile(profile)
        # Date of birth or tax code — either identifies the policyholder.
        if not profile.has_path("customer.owner_date_of_birth") and not profile.has_path(
            "customer.tax_code"
        ):
            missing.append(field_catalog.describe("customer.owner_date_of_birth"))
        return missing

    def api_contract(self) -> api_client.ApiContract:
        return api_client.ApiContract(
            quote_path=API_QUOTE_PATH,
            auth_style="bearer",
            build_payload=_build_payload,
            extract_quotes=_extract_quotes,
            parse_missing_fields=_parse_missing_fields,
        )

    def browser_flow(self, profile: QuotationProfile) -> automation.BrowserFlow | None:
        steps = [automation.Step("fill", SELECTORS["plate"], value_path="vehicle.plate")]
        if profile.has_path("customer.tax_code"):
            steps.append(
                automation.Step("fill", SELECTORS["tax_code"], value_path="customer.tax_code")
            )
        else:
            steps.append(
                automation.Step(
                    "fill",
                    SELECTORS["date_of_birth"],
                    value_path="customer.owner_date_of_birth",
                )
            )
        steps.append(automation.Step("click", SELECTORS["submit"]))

        return automation.BrowserFlow(
            url=self.config.portal_url or PORTAL_URL,
            steps=tuple(steps),
            result_selector=SELECTORS["results"],
            missing_info_selector=SELECTORS["needs_more_info"],
            extract=_extract_from_page,
        )


def _build_payload(profile: QuotationProfile) -> dict[str, Any]:
    """UNVERIFIED placeholder mapping."""
    return {
        "targa": profile.vehicle.plate,
        "codiceFiscale": profile.customer.tax_code,
        "dataDiNascita": _iso(profile.customer.owner_date_of_birth),
        "decorrenza": _iso(profile.policy_start_date),
        "anagrafica": {
            "nome": profile.customer.first_name,
            "cognome": profile.customer.last_name,
            "comune": profile.customer.municipality,
        },
        "classeDiMerito": profile.history.universal_merit_class,
    }


def _extract_quotes(body: dict[str, Any]) -> list[dict[str, Any]]:
    """UNVERIFIED placeholder mapping."""
    return list(body.get("preventivi") or [])


_API_FIELD_TO_PATH = {
    "nome": "customer.first_name",
    "cognome": "customer.last_name",
    "comune": "customer.municipality",
    "classeDiMerito": "history.universal_merit_class",
    "dataImmatricolazione": "vehicle.first_registration_date",
}


def _parse_missing_fields(body: dict[str, Any]) -> list[MissingField]:
    return [
        field_catalog.describe(_API_FIELD_TO_PATH[name])
        for name in body.get("datiMancanti", [])
        if name in _API_FIELD_TO_PATH
    ]


async def _extract_from_page(page) -> list[dict[str, Any]]:  # pragma: no cover - needs a portal
    """Not implemented — see the note in the Zurich adapter."""
    return []


def _iso(value) -> str | None:
    return value.isoformat() if value is not None else None
