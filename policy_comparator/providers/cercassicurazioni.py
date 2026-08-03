"""CercAssicurazioni adapter — an aggregator, not an insurer.

Public form: https://www.cercassicurazioni.it/assicurazione-auto/preventivo1.html

This provider relays quotes issued by other companies. Two consequences run
through the whole application:

* the ``insurer_name`` on a returned quote is the company carrying the risk,
  not "cercassicurazioni", and ``source_channel`` records that it arrived
  through an aggregator;
* the same offer can arrive twice — once here, once from that insurer's own
  adapter. Deduplication is handled centrally in
  :mod:`policy_comparator.services.deduplication`; nothing is dropped, the
  duplicate is linked to its primary so both channels stay auditable.

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

PORTAL_URL = "https://www.cercassicurazioni.it/assicurazione-auto/preventivo1.html"
API_QUOTE_PATH = "/api/v1/auto/preventivi"

#: UNVERIFIED placeholders.
SELECTORS = {
    "plate": "input[name='targa']",
    "date_of_birth": "input[name='data_nascita']",
    "submit": "button[type='submit']",
    "results": "#lista-preventivi",
    "needs_more_info": "#dati-mancanti",
}


class CercAssicurazioniAdapter(StandardAutoAdapter):
    provider_id = "cercassicurazioni"
    display_name = "CercAssicurazioni"
    provider_type = ProviderType.AGGREGATOR
    reference_url = PORTAL_URL
    source_channel = "aggregator"

    required_paths = ("vehicle.plate", "customer.owner_date_of_birth")
    second_stage_paths = (
        "customer.tax_code",
        "customer.postcode",
        "history.universal_merit_class",
        "preferences.driving_formula",
    )
    #: Two of these overlap with the direct insurer adapters — which is what
    #: makes deduplication observable in mock mode rather than hypothetical.
    mock_insurer_keys = ("zurich", "allianz", "genertel", "conte")

    def api_contract(self) -> api_client.ApiContract:
        return api_client.ApiContract(
            quote_path=API_QUOTE_PATH,
            auth_style="api_key_header",
            api_key_header="X-Api-Key",
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
                automation.Step("click", SELECTORS["submit"]),
            ),
            result_selector=SELECTORS["results"],
            missing_info_selector=SELECTORS["needs_more_info"],
            extract=_extract_from_page,
        )


def _build_payload(profile: QuotationProfile) -> dict[str, Any]:
    """UNVERIFIED placeholder mapping."""
    return {
        "targa": profile.vehicle.plate,
        "dataNascita": _iso(profile.customer.owner_date_of_birth),
        "codiceFiscale": profile.customer.tax_code,
        "cap": profile.customer.postcode,
        "classeMerito": profile.history.universal_merit_class,
        "formulaGuida": profile.preferences.driving_formula,
        "decorrenza": _iso(profile.policy_start_date),
    }


def _extract_quotes(body: dict[str, Any]) -> list[dict[str, Any]]:
    """UNVERIFIED placeholder mapping."""
    return list(body.get("preventivi") or [])


_API_FIELD_TO_PATH = {
    "codiceFiscale": "customer.tax_code",
    "cap": "customer.postcode",
    "classeMerito": "history.universal_merit_class",
    "formulaGuida": "preferences.driving_formula",
}


def _parse_missing_fields(body: dict[str, Any]) -> list[MissingField]:
    return [
        field_catalog.describe(_API_FIELD_TO_PATH[name])
        for name in body.get("campiMancanti", [])
        if name in _API_FIELD_TO_PATH
    ]


async def _extract_from_page(page) -> list[dict[str, Any]]:  # pragma: no cover - needs a portal
    """Not implemented — see the note in the Zurich adapter."""
    return []


def _iso(value) -> str | None:
    return value.isoformat() if value is not None else None
