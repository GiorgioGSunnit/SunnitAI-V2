"""Quotation request, per-provider attempts, and normalized results.

The shape here is deliberately append-friendly: a request fans out into one
:class:`ProviderAttempt` per provider, each attempt keeps its own status and
history, and nothing about one provider's failure can remove another provider's
result. A provider that never answered stays visible as a row.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from decimal import Decimal

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..db import GUID, Base, JSONColumn, Money
from .customer import Customer


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class QuoteRequest(Base):
    """One staff-initiated quotation run across a set of providers."""

    __tablename__ = "pc_quote_requests"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(GUID(), nullable=False, index=True)
    created_by_user_id: Mapped[uuid.UUID | None] = mapped_column(GUID(), nullable=True)

    customer_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("pc_customers.id", ondelete="CASCADE"), nullable=False, index=True
    )
    customer_profile_id: Mapped[uuid.UUID] = mapped_column(GUID(), nullable=False)
    vehicle_id: Mapped[uuid.UUID] = mapped_column(GUID(), nullable=False)
    insurance_history_id: Mapped[uuid.UUID] = mapped_column(GUID(), nullable=False)
    coverage_preference_id: Mapped[uuid.UUID] = mapped_column(GUID(), nullable=False)

    policy_start_date: Mapped[date] = mapped_column(Date(), nullable=False)
    selected_provider_ids: Mapped[list] = mapped_column(JSONColumn(), default=list)
    status: Mapped[str] = mapped_column(String(32), default="draft", index=True)

    #: Set when the staff user explicitly starts the run — a request is never
    #: transmitted to a provider merely by being created.
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    cancelled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    recommended_quote_id: Mapped[uuid.UUID | None] = mapped_column(GUID(), nullable=True)
    #: True when every contacted provider was running in mock mode.
    demonstration_data: Mapped[bool] = mapped_column(Boolean, default=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    customer: Mapped[Customer] = relationship(back_populates="quote_requests")
    attempts: Mapped[list["ProviderAttempt"]] = relationship(
        back_populates="quote_request", cascade="all, delete-orphan"
    )
    quotes: Mapped[list["NormalizedQuote"]] = relationship(
        back_populates="quote_request", cascade="all, delete-orphan"
    )


class ProviderAttempt(Base):
    """One provider's participation in one request.

    There is exactly one row per (request, provider); retries mutate this row
    and bump ``attempt_count`` rather than creating parallel rows, so a retry of
    a failed provider can never re-run the providers that already succeeded.
    """

    __tablename__ = "pc_provider_attempts"
    __table_args__ = (
        UniqueConstraint("quote_request_id", "provider_id", name="uq_pc_attempt_request_provider"),
    )

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(GUID(), nullable=False, index=True)
    quote_request_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("pc_quote_requests.id", ondelete="CASCADE"), nullable=False, index=True
    )

    provider_id: Mapped[str] = mapped_column(String(48), nullable=False, index=True)
    provider_type: Mapped[str] = mapped_column(String(24), default="insurer")
    provider_mode: Mapped[str] = mapped_column(String(16), default="mock")

    status: Mapped[str] = mapped_column(String(32), default="waiting", index=True)
    outcome: Mapped[str | None] = mapped_column(String(32), nullable=True)
    #: Stable machine-readable error category, e.g. "provider_timeout".
    error_category: Mapped[str | None] = mapped_column(String(64), nullable=True)
    #: Staff-facing message. Sanitized — never contains customer data.
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    attempt_count: Mapped[int] = mapped_column(Integer, default=0)
    #: Deduplication key. A provider must not be charged twice for one logical
    #: submission if the worker restarts mid-flight.
    idempotency_key: Mapped[str] = mapped_column(String(80), nullable=False)
    #: Opaque adapter state (session handle, provider-side quote id) used to
    #: resume after a missing-information round trip.
    resume_token: Mapped[dict | None] = mapped_column(JSONColumn(), nullable=True)

    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    #: Relative path of a sanitized failure screenshot, when diagnostics are on.
    diagnostic_artifact_path: Mapped[str | None] = mapped_column(String(512), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    quote_request: Mapped[QuoteRequest] = relationship(back_populates="attempts")
    missing_fields: Mapped[list["ProviderMissingFieldRequest"]] = relationship(
        back_populates="attempt", cascade="all, delete-orphan"
    )
    raw_responses: Mapped[list["ProviderRawResponse"]] = relationship(
        back_populates="attempt", cascade="all, delete-orphan"
    )
    quotes: Mapped[list["NormalizedQuote"]] = relationship(
        back_populates="attempt", cascade="all, delete-orphan"
    )


class ProviderMissingFieldRequest(Base):
    """A field a provider asked for before it would quote.

    Rows are replaced wholesale each time an attempt reports
    ``missing_information``, so the UI always reflects the provider's current
    ask rather than an accumulated history.
    """

    __tablename__ = "pc_provider_missing_fields"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(GUID(), nullable=False, index=True)
    provider_attempt_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("pc_provider_attempts.id", ondelete="CASCADE"), nullable=False, index=True
    )

    #: Dotted path into the standardized profile, e.g. "vehicle.make".
    field_path: Mapped[str] = mapped_column(String(120), nullable=False)
    label: Mapped[str] = mapped_column(String(200), nullable=False)
    #: "text" | "date" | "number" | "boolean" | "choice"
    input_type: Mapped[str] = mapped_column(String(24), default="text")
    choices: Mapped[list | None] = mapped_column(JSONColumn(), nullable=True)
    required: Mapped[bool] = mapped_column(Boolean, default=True)
    help_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    resolved: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    attempt: Mapped[ProviderAttempt] = relationship(back_populates="missing_fields")


class ProviderRawResponse(Base):
    """The provider's answer as received, for audit and debugging.

    Kept separate from the normalized quote so a normalization bug can be fixed
    and replayed without re-contacting the provider.
    """

    __tablename__ = "pc_provider_raw_responses"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(GUID(), nullable=False, index=True)
    provider_attempt_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("pc_provider_attempts.id", ondelete="CASCADE"), nullable=False, index=True
    )
    provider_id: Mapped[str] = mapped_column(String(48), nullable=False)
    attempt_number: Mapped[int] = mapped_column(Integer, default=1)
    raw_status: Mapped[str | None] = mapped_column(String(64), nullable=True)
    payload: Mapped[dict] = mapped_column(JSONColumn(), default=dict)
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    attempt: Mapped[ProviderAttempt] = relationship(back_populates="raw_responses")


class NormalizedQuote(Base):
    """A provider quote expressed in the application's common vocabulary."""

    __tablename__ = "pc_normalized_quotes"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(GUID(), nullable=False, index=True)
    quote_request_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("pc_quote_requests.id", ondelete="CASCADE"), nullable=False, index=True
    )
    provider_attempt_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("pc_provider_attempts.id", ondelete="CASCADE"), nullable=False, index=True
    )

    #: The adapter that produced the quote.
    provider_id: Mapped[str] = mapped_column(String(48), nullable=False)
    #: The company actually carrying the risk. Differs from ``provider_id``
    #: whenever the quote arrived through an aggregator.
    insurer_name: Mapped[str] = mapped_column(String(120), nullable=False)
    #: "direct" | "aggregator" — how this quote reached us.
    source_channel: Mapped[str] = mapped_column(String(24), default="direct")

    product_name: Mapped[str | None] = mapped_column(String(160), nullable=True)
    provider_quote_reference: Mapped[str | None] = mapped_column(String(120), nullable=True)

    annual_total_premium: Mapped[Decimal | None] = mapped_column(Money(), nullable=True)
    instalment_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    instalment_amount: Mapped[Decimal | None] = mapped_column(Money(), nullable=True)
    instalment_total_cost: Mapped[Decimal | None] = mapped_column(Money(), nullable=True)
    currency: Mapped[str] = mapped_column(String(3), default="EUR")

    liability_limit_people: Mapped[Decimal | None] = mapped_column(Money(), nullable=True)
    liability_limit_property: Mapped[Decimal | None] = mapped_column(Money(), nullable=True)
    driving_formula: Mapped[str | None] = mapped_column(String(24), nullable=True)
    deductible: Mapped[Decimal | None] = mapped_column(Money(), nullable=True)
    percentage_excess: Mapped[str | None] = mapped_column(String(16), nullable=True)

    requires_black_box: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    requires_approved_repair_network: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    important_exclusions: Mapped[list] = mapped_column(JSONColumn(), default=list)
    quote_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    purchase_url: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    product_document_url: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    precontractual_document_url: Mapped[str | None] = mapped_column(String(1024), nullable=True)

    raw_provider_status: Mapped[str | None] = mapped_column(String(64), nullable=True)
    #: True when this quote came from a mock adapter.
    is_demonstration: Mapped[bool] = mapped_column(Boolean, default=True)

    #: "demonstration_formula" when this application computed the price,
    #: "provider_supplied" when the insurer quoted it.
    calculation_source: Mapped[str] = mapped_column(
        String(32), default="provider_supplied", server_default="provider_supplied"
    )
    #: The auditable step-by-step derivation, for demonstration quotes only.
    #: Stored as JSON so the breakdown a user was shown stays reproducible even
    #: if the formula is later changed.
    calculation_breakdown: Mapped[dict | None] = mapped_column(JSONColumn(), nullable=True)

    #: Set when this quote duplicates another one (same insurer/product reached
    #: through a different channel). Kept, never deleted, so the source channel
    #: stays auditable.
    duplicate_of_quote_id: Mapped[uuid.UUID | None] = mapped_column(GUID(), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    quote_request: Mapped[QuoteRequest] = relationship(back_populates="quotes")
    attempt: Mapped[ProviderAttempt] = relationship(back_populates="quotes")
    coverages: Mapped[list["QuoteCoverage"]] = relationship(
        back_populates="quote", cascade="all, delete-orphan"
    )


class QuoteCoverage(Base):
    """One guarantee on a quote — included in the premium or priced separately."""

    __tablename__ = "pc_quote_coverages"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(GUID(), nullable=False, index=True)
    quote_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("pc_normalized_quotes.id", ondelete="CASCADE"), nullable=False, index=True
    )

    #: Stable internal code, e.g. "furto_incendio", "assistenza_stradale".
    code: Mapped[str] = mapped_column(String(64), nullable=False)
    label: Mapped[str] = mapped_column(String(200), nullable=False)
    included: Mapped[bool] = mapped_column(Boolean, default=True)
    price: Mapped[Decimal | None] = mapped_column(Money(), nullable=True)
    limit_amount: Mapped[Decimal | None] = mapped_column(Money(), nullable=True)
    deductible: Mapped[Decimal | None] = mapped_column(Money(), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    quote: Mapped[NormalizedQuote] = relationship(back_populates="coverages")


class ProviderHealthState(Base):
    """Circuit-breaker state, persisted so it survives a worker restart."""

    __tablename__ = "pc_provider_health"
    __table_args__ = (
        UniqueConstraint("tenant_id", "provider_id", name="uq_pc_health_tenant_provider"),
    )

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(GUID(), nullable=False, index=True)
    provider_id: Mapped[str] = mapped_column(String(48), nullable=False, index=True)

    consecutive_failures: Mapped[int] = mapped_column(Integer, default=0)
    #: While set and in the future, the breaker is open and the provider is
    #: skipped rather than contacted.
    circuit_open_until: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_success_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_failure_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_error_category: Mapped[str | None] = mapped_column(String(64), nullable=True)
    total_successes: Mapped[int] = mapped_column(Integer, default=0)
    total_failures: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )
