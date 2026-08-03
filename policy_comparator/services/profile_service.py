"""Assembling the standardized profile, and applying updates to it.

Two responsibilities:

*Reading* — gather the four customer-side rows into one
:class:`QuotationProfile`, which is the only shape adapters ever see.

*Writing* — apply updates while tracking where each value came from. A value a
staff member entered is authoritative: a later provider response may fill a
blank, but it must never overwrite a staff-confirmed field.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from sqlalchemy.orm import Session

from ..models import (
    CoveragePreference,
    Customer,
    CustomerProfile,
    InsuranceHistory,
    Vehicle,
)
from ..models.enums import FieldSource
from ..schemas.profile import (
    CoveragePreferenceData,
    CustomerProfileData,
    InsuranceHistoryData,
    QuotationProfile,
    VehicleData,
)
from . import field_catalog

#: Which ORM row backs each dotted path prefix.
_SECTION_MODELS = {
    "customer": CustomerProfile,
    "vehicle": Vehicle,
    "history": InsuranceHistory,
    "preferences": CoveragePreference,
}


@dataclass
class ProfileBundle:
    """The four rows that make up one request's profile."""

    customer: Customer
    profile: CustomerProfile
    vehicle: Vehicle
    history: InsuranceHistory
    preferences: CoveragePreference


class StaffValueProtected(ValueError):
    """A provider-sourced write tried to replace a staff-confirmed value."""


def build_profile(bundle: ProfileBundle, *, policy_start_date: date, email: str) -> QuotationProfile:
    """Compose the standardized profile handed to adapters."""
    return QuotationProfile(
        customer_email=email,
        policy_start_date=policy_start_date,
        customer=CustomerProfileData(
            owner_date_of_birth=bundle.profile.owner_date_of_birth,
            first_name=bundle.profile.first_name,
            last_name=bundle.profile.last_name,
            tax_code=bundle.profile.tax_code,
            gender=bundle.profile.gender,
            mobile_number=bundle.profile.mobile_number,
            address_street=bundle.profile.address_street,
            municipality=bundle.profile.municipality,
            province=bundle.profile.province,
            postcode=bundle.profile.postcode,
            subject_type=bundle.profile.subject_type or "individual",
            company_name=bundle.profile.company_name,
            vat_number=bundle.profile.vat_number,
            policyholder_same_as_owner=bool(bundle.profile.policyholder_same_as_owner),
        ),
        vehicle=VehicleData(
            plate=bundle.vehicle.plate,
            ownership_status=bundle.vehicle.ownership_status,
            first_registration_date=bundle.vehicle.first_registration_date,
            purchase_date=bundle.vehicle.purchase_date,
            make=bundle.vehicle.make,
            model=bundle.vehicle.model,
            trim=bundle.vehicle.trim,
            fuel_type=bundle.vehicle.fuel_type,
            power_kw=bundle.vehicle.power_kw,
            primary_use=bundle.vehicle.primary_use,
            annual_kilometres=bundle.vehicle.annual_kilometres,
            overnight_parking=bundle.vehicle.overnight_parking,
            anti_theft_system=bundle.vehicle.anti_theft_system,
            towing_hook=bundle.vehicle.towing_hook,
        ),
        history=InsuranceHistoryData(
            current_insurer=bundle.history.current_insurer,
            existing_policy_expiry=bundle.history.existing_policy_expiry,
            universal_merit_class=bundle.history.universal_merit_class,
            first_insurance=bundle.history.first_insurance,
            claims_last_5_years=bundle.history.claims_last_5_years,
            claims_full_responsibility=bundle.history.claims_full_responsibility,
            claims_partial_responsibility=bundle.history.claims_partial_responsibility,
            bersani_applicable=bundle.history.bersani_applicable,
            bersani_source_plate=bundle.history.bersani_source_plate,
            bersani_source_merit_class=bundle.history.bersani_source_merit_class,
            risk_certificate_reference=bundle.history.risk_certificate_reference,
        ),
        preferences=CoveragePreferenceData(
            base_rc=bool(bundle.preferences.base_rc),
            min_liability_limit_people=_dec(bundle.preferences.min_liability_limit_people),
            min_liability_limit_property=_dec(bundle.preferences.min_liability_limit_property),
            driving_formula=bundle.preferences.driving_formula,
            max_acceptable_deductible=_dec(bundle.preferences.max_acceptable_deductible),
            required_optional_covers=list(bundle.preferences.required_optional_covers or []),
            accepts_black_box=bundle.preferences.accepts_black_box,
            accepts_approved_repair_network=bundle.preferences.accepts_approved_repair_network,
            payment_frequency=bundle.preferences.payment_frequency,
        ),
    )


def _dec(value: Any) -> Decimal | None:
    if value is None or value == "":
        return None
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except InvalidOperation:
        return None


def _coerce(path: str, raw: Any) -> Any:
    """Convert an incoming JSON value to the type the column expects."""
    if raw is None or raw == "":
        return None

    spec_type = field_catalog.describe(path).input_type
    if spec_type == "date":
        if isinstance(raw, date) and not isinstance(raw, datetime):
            return raw
        return date.fromisoformat(str(raw))
    if spec_type == "number":
        text = str(raw).strip()
        # Monetary preferences stay Decimal; counts and classes are ints.
        if path.startswith("preferences."):
            return str(Decimal(text))
        return int(Decimal(text))
    if spec_type == "boolean":
        if isinstance(raw, bool):
            return raw
        return str(raw).strip().lower() in {"1", "true", "yes", "si", "sì", "on"}
    return str(raw).strip()


def _target(bundle: ProfileBundle, path: str) -> tuple[Any, str]:
    section, _, attribute = path.partition(".")
    if section not in _SECTION_MODELS or not attribute:
        raise ValueError(f"Unknown profile path: {path}")
    row = {
        "customer": bundle.profile,
        "vehicle": bundle.vehicle,
        "history": bundle.history,
        "preferences": bundle.preferences,
    }[section]
    if not hasattr(row, attribute):
        raise ValueError(f"Unknown profile path: {path}")
    return row, attribute


def apply_updates(
    db: Session,
    bundle: ProfileBundle,
    updates: dict[str, Any],
    *,
    source: FieldSource,
) -> list[str]:
    """Write values into the profile, honouring provenance.

    Returns the paths that were actually changed. Paths skipped because a staff
    value already occupies them are reported by :func:`protected_paths`.
    """
    changed: list[str] = []

    for path, raw in updates.items():
        row, attribute = _target(bundle, path)
        sources: dict = dict(row.field_sources or {})

        if source is not FieldSource.STAFF and sources.get(path) == FieldSource.STAFF.value:
            # A provider must not silently replace what staff confirmed.
            continue

        value = _coerce(path, raw)
        if getattr(row, attribute) == value and sources.get(path) == source.value:
            continue

        setattr(row, attribute, value)
        sources[path] = source.value
        # Reassign rather than mutate: JSON columns do not track in-place edits.
        row.field_sources = sources
        changed.append(path)

    if changed:
        db.flush()
    return changed


def protected_paths(bundle: ProfileBundle, updates: dict[str, Any]) -> list[str]:
    """Paths a non-staff write would be refused, because staff own them."""
    blocked = []
    for path in updates:
        try:
            row, _ = _target(bundle, path)
        except ValueError:
            continue
        if (row.field_sources or {}).get(path) == FieldSource.STAFF.value:
            blocked.append(path)
    return blocked


def load_bundle(db: Session, tenant_id: uuid.UUID, request) -> ProfileBundle:
    """Load the four profile rows for a request, tenant-scoped throughout."""
    customer = db.get(Customer, request.customer_id)
    profile = db.get(CustomerProfile, request.customer_profile_id)
    vehicle = db.get(Vehicle, request.vehicle_id)
    history = db.get(InsuranceHistory, request.insurance_history_id)
    preferences = db.get(CoveragePreference, request.coverage_preference_id)

    rows = [customer, profile, vehicle, history, preferences]
    if any(row is None for row in rows):
        raise LookupError("Quotation request is missing part of its profile")
    if any(row.tenant_id != tenant_id for row in rows):
        # Defence in depth: the request was already tenant-scoped, so this can
        # only fire on cross-tenant data corruption.
        raise PermissionError("Profile rows belong to a different tenant")

    return ProfileBundle(
        customer=customer,  # type: ignore[arg-type]
        profile=profile,  # type: ignore[arg-type]
        vehicle=vehicle,  # type: ignore[arg-type]
        history=history,  # type: ignore[arg-type]
        preferences=preferences,  # type: ignore[arg-type]
    )
