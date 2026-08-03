"""Customer-side tables: identity, vehicle, insurance history, preferences, consent.

Every table carries ``tenant_id``. Nothing is ever queried without it — see
:mod:`policy_comparator.api.deps`.

Directly identifying values (tax code, phone, address, names) use
:class:`~policy_comparator.db.EncryptedString` so a database dump is not a
plaintext PII spill.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone

from sqlalchemy import Boolean, Date, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..db import GUID, Base, EncryptedString, JSONColumn


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Customer(Base):
    """The end customer a quotation is being prepared for."""

    __tablename__ = "pc_customers"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(GUID(), nullable=False, index=True)
    created_by_user_id: Mapped[uuid.UUID | None] = mapped_column(GUID(), nullable=True)
    email: Mapped[str] = mapped_column(EncryptedString(), nullable=False)
    #: Blind index over the lowercased email so a customer can be looked up
    #: without decrypting every row, and without storing the address in clear.
    email_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    profiles: Mapped[list["CustomerProfile"]] = relationship(
        back_populates="customer", cascade="all, delete-orphan"
    )
    quote_requests: Mapped[list["QuoteRequest"]] = relationship(  # noqa: F821
        back_populates="customer", cascade="all, delete-orphan"
    )


class CustomerProfile(Base):
    """Personal details of the policyholder / vehicle owner.

    Only ``date_of_birth`` is required up front; every other column is filled in
    later, and only if a provider actually asks for it.
    """

    __tablename__ = "pc_customer_profiles"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(GUID(), nullable=False, index=True)
    customer_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("pc_customers.id", ondelete="CASCADE"), nullable=False, index=True
    )

    owner_date_of_birth: Mapped[date | None] = mapped_column(Date(), nullable=True)
    first_name: Mapped[str | None] = mapped_column(EncryptedString(), nullable=True)
    last_name: Mapped[str | None] = mapped_column(EncryptedString(), nullable=True)
    tax_code: Mapped[str | None] = mapped_column(EncryptedString(), nullable=True)
    gender: Mapped[str | None] = mapped_column(String(16), nullable=True)
    mobile_number: Mapped[str | None] = mapped_column(EncryptedString(), nullable=True)

    address_street: Mapped[str | None] = mapped_column(EncryptedString(), nullable=True)
    municipality: Mapped[str | None] = mapped_column(String(120), nullable=True)
    province: Mapped[str | None] = mapped_column(String(8), nullable=True)
    postcode: Mapped[str | None] = mapped_column(String(16), nullable=True)

    #: "individual" or "company".
    subject_type: Mapped[str] = mapped_column(String(16), default="individual")
    company_name: Mapped[str | None] = mapped_column(EncryptedString(), nullable=True)
    vat_number: Mapped[str | None] = mapped_column(EncryptedString(), nullable=True)

    policyholder_same_as_owner: Mapped[bool] = mapped_column(Boolean, default=True)

    #: field name -> FieldSource, so a provider response can never silently
    #: overwrite a value a staff member confirmed.
    field_sources: Mapped[dict] = mapped_column(JSONColumn(), default=dict)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    customer: Mapped[Customer] = relationship(back_populates="profiles")


class Vehicle(Base):
    __tablename__ = "pc_vehicles"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(GUID(), nullable=False, index=True)

    plate: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    ownership_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    first_registration_date: Mapped[date | None] = mapped_column(Date(), nullable=True)
    purchase_date: Mapped[date | None] = mapped_column(Date(), nullable=True)
    make: Mapped[str | None] = mapped_column(String(64), nullable=True)
    model: Mapped[str | None] = mapped_column(String(120), nullable=True)
    trim: Mapped[str | None] = mapped_column(String(160), nullable=True)
    fuel_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    power_kw: Mapped[int | None] = mapped_column(Integer, nullable=True)
    primary_use: Mapped[str | None] = mapped_column(String(48), nullable=True)
    annual_kilometres: Mapped[int | None] = mapped_column(Integer, nullable=True)
    overnight_parking: Mapped[str | None] = mapped_column(String(48), nullable=True)
    anti_theft_system: Mapped[str | None] = mapped_column(String(48), nullable=True)
    towing_hook: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    field_sources: Mapped[dict] = mapped_column(JSONColumn(), default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class InsuranceHistory(Base):
    """Bonus/malus position and claims record — the main premium driver."""

    __tablename__ = "pc_insurance_histories"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(GUID(), nullable=False, index=True)

    current_insurer: Mapped[str | None] = mapped_column(String(120), nullable=True)
    existing_policy_expiry: Mapped[date | None] = mapped_column(Date(), nullable=True)
    #: Classe di merito universale (CU), 1..18.
    universal_merit_class: Mapped[int | None] = mapped_column(Integer, nullable=True)
    first_insurance: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    claims_last_5_years: Mapped[int | None] = mapped_column(Integer, nullable=True)
    claims_full_responsibility: Mapped[int | None] = mapped_column(Integer, nullable=True)
    claims_partial_responsibility: Mapped[int | None] = mapped_column(Integer, nullable=True)

    #: RC Familiare / Legge Bersani — inheriting a relative's merit class.
    bersani_applicable: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    bersani_source_plate: Mapped[str | None] = mapped_column(String(16), nullable=True)
    bersani_source_merit_class: Mapped[int | None] = mapped_column(Integer, nullable=True)
    risk_certificate_reference: Mapped[str | None] = mapped_column(
        EncryptedString(), nullable=True
    )

    field_sources: Mapped[dict] = mapped_column(JSONColumn(), default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class CoveragePreference(Base):
    """What the customer requires. Drives eligibility, not just presentation."""

    __tablename__ = "pc_coverage_preferences"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(GUID(), nullable=False, index=True)

    #: Third-party liability is always included; kept explicit for clarity.
    base_rc: Mapped[bool] = mapped_column(Boolean, default=True)
    min_liability_limit_people: Mapped[str | None] = mapped_column(String(32), nullable=True)
    min_liability_limit_property: Mapped[str | None] = mapped_column(String(32), nullable=True)
    driving_formula: Mapped[str | None] = mapped_column(String(24), nullable=True)
    max_acceptable_deductible: Mapped[str | None] = mapped_column(String(32), nullable=True)
    #: Coverage codes the customer requires, e.g. ["furto_incendio", "kasko"].
    required_optional_covers: Mapped[list] = mapped_column(JSONColumn(), default=list)
    accepts_black_box: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    accepts_approved_repair_network: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    payment_frequency: Mapped[str | None] = mapped_column(String(24), nullable=True)

    field_sources: Mapped[dict] = mapped_column(JSONColumn(), default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class ConsentRecord(Base):
    """Immutable evidence that the customer agreed to a specific processing.

    Mandatory processing consent and optional marketing consent are separate
    rows — they are never bundled into a single checkbox.
    """

    __tablename__ = "pc_consent_records"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[uuid.UUID] = mapped_column(GUID(), nullable=False, index=True)
    customer_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("pc_customers.id", ondelete="CASCADE"), nullable=False, index=True
    )
    quote_request_id: Mapped[uuid.UUID | None] = mapped_column(GUID(), nullable=True, index=True)

    consent_type: Mapped[str] = mapped_column(String(48), nullable=False)
    granted: Mapped[bool] = mapped_column(Boolean, nullable=False)
    #: Provider ids the customer agreed to transmit data to.
    scope_provider_ids: Mapped[list] = mapped_column(JSONColumn(), default=list)
    granted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    recorded_by_user_id: Mapped[uuid.UUID | None] = mapped_column(GUID(), nullable=True)
    #: Wording shown to the customer, so the record stays meaningful when the
    #: consent copy is later reworded.
    policy_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
