"""Normalized quote structures shared by adapters, the ranking service and the API."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..models.enums import QuoteOutcome


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _reject_float(v: Any) -> Any:
    # ValueError, not TypeError: pydantic folds ValueError into a
    # ValidationError, while a TypeError escapes as an unhandled crash.
    if isinstance(v, float):
        raise ValueError("Monetary values must be Decimal or string, not float")
    return v


class CoverageData(_Base):
    code: str
    label: str
    included: bool = True
    price: Decimal | None = None
    limit_amount: Decimal | None = None
    deductible: Decimal | None = None
    notes: str | None = None

    @field_validator("price", "limit_amount", "deductible", mode="before")
    @classmethod
    def _no_floats(cls, v: Any) -> Any:
        return _reject_float(v)


#: The price was computed by this application's demonstration formula.
CALCULATION_SOURCE_DEMONSTRATION = "demonstration_formula"
#: The price came from the insurer. No formula is ever shown for these.
CALCULATION_SOURCE_PROVIDER = "provider_supplied"


class CalculationStep(_Base):
    """One auditable line of a price calculation.

    Monetary and factor values are decimal *strings* end to end — they are
    displayed, not arithmetic inputs, and a float here would be exactly the
    rounding drift the rest of the application avoids.
    """

    code: str
    label: str
    #: base | factor | rounding | addition | subtotal | total
    kind: str
    factor: str | None = None
    value: str | None = None
    #: Exact unrounded running total at this point, so the steps can be replayed.
    running: str | None = None
    detail: str | None = None


class CalculationBreakdown(_Base):
    """How a premium was arrived at.

    Only ever populated for demonstration quotes. A price supplied by an
    insurer carries ``calculation_source = provider_supplied`` and no steps:
    reverse-engineering an underwriting formula from a single quoted number
    would be fabrication.
    """

    source: str = CALCULATION_SOURCE_DEMONSTRATION
    currency: str = "EUR"
    rounding: str | None = None
    annual_total: str | None = None
    steps: list[CalculationStep] = Field(default_factory=list)

    @property
    def is_demonstration(self) -> bool:
        return self.source == CALCULATION_SOURCE_DEMONSTRATION


class NormalizedQuoteData(_Base):
    """A quote in the application's common vocabulary.

    Produced by ``adapter.normalize_result`` and validated centrally before it
    is persisted, so a malformed adapter cannot write a half-built quote.
    """

    provider_id: str
    insurer_name: str
    source_channel: str = "direct"
    product_name: str | None = None
    provider_quote_reference: str | None = None

    annual_total_premium: Decimal | None = None
    instalment_count: int | None = None
    instalment_amount: Decimal | None = None
    instalment_total_cost: Decimal | None = None
    currency: str = "EUR"

    liability_limit_people: Decimal | None = None
    liability_limit_property: Decimal | None = None
    driving_formula: str | None = None
    deductible: Decimal | None = None
    percentage_excess: str | None = None

    requires_black_box: bool | None = None
    requires_approved_repair_network: bool | None = None

    coverages: list[CoverageData] = Field(default_factory=list)
    important_exclusions: list[str] = Field(default_factory=list)

    quote_expires_at: datetime | None = None
    purchase_url: str | None = None
    product_document_url: str | None = None
    precontractual_document_url: str | None = None

    raw_provider_status: str | None = None
    is_demonstration: bool = True

    #: Where the price came from. Drives whether a formula may be shown at all.
    calculation_source: str = CALCULATION_SOURCE_PROVIDER
    calculation_breakdown: CalculationBreakdown | None = None

    @field_validator(
        "annual_total_premium",
        "instalment_amount",
        "instalment_total_cost",
        "liability_limit_people",
        "liability_limit_property",
        "deductible",
        mode="before",
    )
    @classmethod
    def _no_floats(cls, v: Any) -> Any:
        return _reject_float(v)

    @property
    def optional_coverages(self) -> list[CoverageData]:
        return [c for c in self.coverages if not c.included]

    @property
    def included_coverages(self) -> list[CoverageData]:
        return [c for c in self.coverages if c.included]

    def dedupe_signature(self) -> tuple:
        """Identity of the underlying offer, independent of how it reached us.

        Two quotes with the same signature are the same product from the same
        insurer with the same configuration — one direct, one via an
        aggregator. The provider quote reference is included because two
        genuinely different offers from one insurer must not collapse.
        """
        return (
            self.insurer_name.strip().lower(),
            (self.product_name or "").strip().lower(),
            (self.provider_quote_reference or "").strip().lower(),
            str(self.annual_total_premium) if self.annual_total_premium is not None else "",
            (self.driving_formula or ""),
            str(self.deductible) if self.deductible is not None else "",
            str(self.liability_limit_people) if self.liability_limit_people is not None else "",
        )


class MissingField(_Base):
    """A field a provider requires before it will quote."""

    field_path: str
    label: str
    input_type: str = "text"
    choices: list[dict] | None = None
    required: bool = True
    help_text: str | None = None


class ProviderResult(_Base):
    """The complete outcome of one adapter call."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    provider_id: str
    outcome: QuoteOutcome
    #: Provider-shaped payloads. Passed back through ``normalize_result``.
    raw_quotes: list[dict] = Field(default_factory=list)
    missing_fields: list[MissingField] = Field(default_factory=list)
    #: Opaque state allowing ``resume_quote`` to continue this conversation.
    resume_token: dict | None = None
    raw_status: str | None = None
    error_category: str | None = None
    #: Staff-facing. Must never contain customer data.
    error_message: str | None = None
    raw_payload: dict = Field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.outcome is QuoteOutcome.QUOTED


class ProviderHealth(_Base):
    provider_id: str
    provider_type: str
    mode: str
    configured: bool
    authorized: bool
    live_enabled: bool
    reachable: bool | None = None
    circuit_open: bool = False
    circuit_open_until: datetime | None = None
    consecutive_failures: int = 0
    last_success_at: datetime | None = None
    last_failure_at: datetime | None = None
    last_error_category: str | None = None
    detail: str | None = None
