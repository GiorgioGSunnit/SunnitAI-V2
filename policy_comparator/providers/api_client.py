"""Generic partner-API client used by the live adapter scaffolding.

This handles the parts that are the same for every insurer API — authentication
headers, idempotency, timeouts, HTTP status to :class:`QuoteOutcome` mapping —
so a provider module only has to supply three provider-specific functions:
how to build the request body, how to pull quotes out of a success response,
and how to read a "we need more data" response.

**Status:** the transport is real and exercised by tests against a stub server.
The per-provider request/response mappings are *unverified placeholders* — none
of the four providers' API contracts were available. Each must be checked
against that provider's real API documentation before its adapter is switched
out of mock mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import httpx

from ..models.enums import QuoteOutcome
from ..schemas.profile import QuotationProfile
from ..schemas.quotes import MissingField, ProviderResult
from ..secrets_provider import SecretNotConfigured, resolve_secret


@dataclass(frozen=True)
class ApiContract:
    """How one provider's quoting API is shaped."""

    quote_path: str = "/quotes"
    resume_path: str = "/quotes/{quote_id}"
    #: "bearer" | "api_key_header" | "none"
    auth_style: str = "bearer"
    api_key_header: str = "X-API-Key"
    build_payload: Callable[[QuotationProfile], dict[str, Any]] | None = None
    extract_quotes: Callable[[dict[str, Any]], list[dict[str, Any]]] | None = None
    parse_missing_fields: Callable[[dict[str, Any]], list[MissingField]] | None = None


def _auth_headers(contract: ApiContract, api_key: str | None) -> dict[str, str]:
    if contract.auth_style == "none" or not api_key:
        return {}
    if contract.auth_style == "api_key_header":
        return {contract.api_key_header: api_key}
    return {"Authorization": f"Bearer {api_key}"}


async def submit_quote_request(
    *,
    provider_id: str,
    display_name: str,
    base_url: str,
    api_key_env: str | None,
    contract: ApiContract,
    profile: QuotationProfile,
    idempotency_key: str,
    timeout_seconds: int,
    path: str | None = None,
) -> ProviderResult:
    """POST one quotation request and map the response onto a result."""
    if contract.build_payload is None or contract.extract_quotes is None:
        return ProviderResult(
            provider_id=provider_id,
            outcome=QuoteOutcome.CONFIGURATION_ERROR,
            error_category="api_contract_incomplete",
            error_message=(
                f"{display_name}: the API request/response mapping has not been "
                "verified against the provider's documentation yet."
            ),
        )

    try:
        api_key = resolve_secret(api_key_env)
        if contract.auth_style != "none" and api_key is None:
            raise SecretNotConfigured(
                f"environment variable {api_key_env or '<unset>'} holds no value"
            )
    except SecretNotConfigured as exc:
        return ProviderResult(
            provider_id=provider_id,
            outcome=QuoteOutcome.AUTHENTICATION_REQUIRED,
            error_category="missing_credentials",
            error_message=f"{display_name}: {exc}",
        )

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        # Lets the provider collapse a duplicate submission after a worker
        # restart instead of issuing a second quote.
        "Idempotency-Key": idempotency_key,
        **_auth_headers(contract, api_key),
    }

    url = base_url.rstrip("/") + (path or contract.quote_path)

    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.post(
                url, json=contract.build_payload(profile), headers=headers
            )
    except httpx.TimeoutException:
        return ProviderResult(
            provider_id=provider_id,
            outcome=QuoteOutcome.TIMED_OUT,
            error_category="api_timeout",
            error_message=f"{display_name}: no response within {timeout_seconds}s",
        )
    except httpx.HTTPError as exc:
        return ProviderResult(
            provider_id=provider_id,
            outcome=QuoteOutcome.UNAVAILABLE,
            error_category="api_unreachable",
            error_message=f"{display_name}: {type(exc).__name__} contacting the provider API",
        )

    return _interpret(
        response,
        provider_id=provider_id,
        display_name=display_name,
        contract=contract,
    )


def _interpret(
    response: httpx.Response,
    *,
    provider_id: str,
    display_name: str,
    contract: ApiContract,
) -> ProviderResult:
    status = response.status_code

    if status in (401, 403):
        return ProviderResult(
            provider_id=provider_id,
            outcome=QuoteOutcome.AUTHENTICATION_REQUIRED,
            error_category="api_unauthorized",
            error_message=f"{display_name}: the provider rejected our credentials ({status})",
        )
    if status == 429:
        return ProviderResult(
            provider_id=provider_id,
            outcome=QuoteOutcome.UNAVAILABLE,
            error_category="api_rate_limited",
            error_message=f"{display_name}: rate limited by the provider",
        )
    if status >= 500:
        return ProviderResult(
            provider_id=provider_id,
            outcome=QuoteOutcome.UNAVAILABLE,
            error_category="api_server_error",
            error_message=f"{display_name}: provider returned HTTP {status}",
        )

    try:
        body = response.json()
    except ValueError:
        return ProviderResult(
            provider_id=provider_id,
            outcome=QuoteOutcome.CONFIGURATION_ERROR,
            error_category="api_invalid_response",
            error_message=f"{display_name}: response was not JSON (HTTP {status})",
        )

    if status in (400, 422) and contract.parse_missing_fields is not None:
        missing = contract.parse_missing_fields(body)
        if missing:
            return ProviderResult(
                provider_id=provider_id,
                outcome=QuoteOutcome.MISSING_INFORMATION,
                missing_fields=missing,
                raw_status=str(status),
                resume_token=(
                    {"quote_id": body.get("quote_id")} if body.get("quote_id") else None
                ),
            )

    if status >= 400:
        return ProviderResult(
            provider_id=provider_id,
            outcome=QuoteOutcome.FAILED,
            error_category="api_client_error",
            error_message=f"{display_name}: provider rejected the request (HTTP {status})",
        )

    try:
        quotes = contract.extract_quotes(body) if contract.extract_quotes else []
    except (KeyError, TypeError, ValueError):
        return ProviderResult(
            provider_id=provider_id,
            outcome=QuoteOutcome.CONFIGURATION_ERROR,
            error_category="api_response_shape_mismatch",
            error_message=(
                f"{display_name}: the response did not match the expected shape — "
                "the response mapping needs re-verifying."
            ),
        )

    if not quotes:
        return ProviderResult(
            provider_id=provider_id,
            outcome=QuoteOutcome.UNAVAILABLE,
            error_category="no_quotes_returned",
            error_message=f"{display_name}: the provider returned no quotes",
        )

    return ProviderResult(
        provider_id=provider_id,
        outcome=QuoteOutcome.QUOTED,
        raw_quotes=quotes,
        raw_status=str(status),
    )
