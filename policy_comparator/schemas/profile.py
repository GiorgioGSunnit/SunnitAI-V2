"""The standardized internal profile handed to every provider adapter.

There is exactly one profile shape in the application. Providers differ in
*which* parts of it they need, not in its structure — an adapter that wants a
field it hasn't been given reports a missing field, it never invents a
provider-specific schema.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CustomerProfileData(_Base):
    """Policyholder / vehicle owner. Only the date of birth is asked up front."""

    owner_date_of_birth: date | None = None
    first_name: str | None = None
    last_name: str | None = None
    tax_code: str | None = None
    gender: str | None = None
    mobile_number: str | None = None
    address_street: str | None = None
    municipality: str | None = None
    province: str | None = None
    postcode: str | None = None
    subject_type: str = "individual"
    company_name: str | None = None
    vat_number: str | None = None
    policyholder_same_as_owner: bool = True

    @field_validator("tax_code")
    @classmethod
    def _upper_tax_code(cls, v: str | None) -> str | None:
        return v.strip().upper() if v else v

    @field_validator("province")
    @classmethod
    def _upper_province(cls, v: str | None) -> str | None:
        return v.strip().upper() if v else v


class VehicleData(_Base):
    plate: str
    ownership_status: str | None = None
    first_registration_date: date | None = None
    purchase_date: date | None = None
    make: str | None = None
    model: str | None = None
    trim: str | None = None
    fuel_type: str | None = None
    power_kw: int | None = None
    primary_use: str | None = None
    annual_kilometres: int | None = None
    overnight_parking: str | None = None
    anti_theft_system: str | None = None
    towing_hook: bool | None = None

    @field_validator("plate")
    @classmethod
    def _normalize_plate(cls, v: str) -> str:
        """Italian plates are compared without spaces or separators."""
        cleaned = "".join(ch for ch in v.upper() if ch.isalnum())
        if not 4 <= len(cleaned) <= 10:
            raise ValueError("Vehicle plate must be between 4 and 10 characters")
        return cleaned


class InsuranceHistoryData(_Base):
    current_insurer: str | None = None
    existing_policy_expiry: date | None = None
    universal_merit_class: int | None = Field(default=None, ge=1, le=18)
    first_insurance: bool | None = None
    claims_last_5_years: int | None = Field(default=None, ge=0)
    claims_full_responsibility: int | None = Field(default=None, ge=0)
    claims_partial_responsibility: int | None = Field(default=None, ge=0)
    bersani_applicable: bool | None = None
    bersani_source_plate: str | None = None
    bersani_source_merit_class: int | None = Field(default=None, ge=1, le=18)
    risk_certificate_reference: str | None = None


class CoveragePreferenceData(_Base):
    """Customer requirements. Everything set here is treated as mandatory."""

    base_rc: bool = True
    min_liability_limit_people: Decimal | None = None
    min_liability_limit_property: Decimal | None = None
    driving_formula: str | None = None
    max_acceptable_deductible: Decimal | None = None
    required_optional_covers: list[str] = Field(default_factory=list)
    accepts_black_box: bool | None = None
    accepts_approved_repair_network: bool | None = None
    payment_frequency: str | None = None

    @field_validator(
        "min_liability_limit_people",
        "min_liability_limit_property",
        "max_acceptable_deductible",
        mode="before",
    )
    @classmethod
    def _no_binary_floats(cls, v: Any) -> Any:
        """Reject float input outright rather than inheriting its rounding error.

        ValueError rather than TypeError so pydantic reports it as a normal
        validation failure instead of letting it escape as a crash.
        """
        if isinstance(v, float):
            raise ValueError("Monetary limits must be Decimal or string, not float")
        return v


class QuotationProfile(_Base):
    """Everything an adapter may read about one quotation request."""

    customer_email: EmailStr
    policy_start_date: date
    customer: CustomerProfileData
    vehicle: VehicleData
    history: InsuranceHistoryData = Field(default_factory=InsuranceHistoryData)
    preferences: CoveragePreferenceData = Field(default_factory=CoveragePreferenceData)

    def get_path(self, path: str) -> Any:
        """Read a dotted field path, e.g. ``"vehicle.make"``.

        Returns ``None`` for any path that does not resolve, so an adapter
        asking about an unknown field is treated as "not provided" rather than
        crashing the run.
        """
        node: Any = self
        for part in path.split("."):
            if node is None:
                return None
            node = getattr(node, part, None)
        return node

    def has_path(self, path: str) -> bool:
        value = self.get_path(path)
        if value is None:
            return False
        if isinstance(value, str) and not value.strip():
            return False
        if isinstance(value, list) and not value:
            return False
        return True

    def missing_paths(self, required: list[str]) -> list[str]:
        return [p for p in required if not self.has_path(p)]
