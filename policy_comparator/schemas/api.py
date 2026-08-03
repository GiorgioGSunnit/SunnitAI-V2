"""Request and response bodies for the HTTP API."""

from __future__ import annotations

import uuid
from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LoginRequest(_Base):
    email: EmailStr
    password: str


class TokenResponse(_Base):
    access_token: str
    token_type: str = "bearer"
    expires_in_minutes: int
    tenant_id: uuid.UUID
    email: str | None = None
    role: str


class CreateQuoteRequest(_Base):
    """The minimal first form. Nothing else is asked until a provider asks."""

    vehicle_plate: str = Field(min_length=4, max_length=12)
    owner_date_of_birth: date
    customer_email: EmailStr
    policy_start_date: date
    privacy_accepted: bool
    provider_data_transfer_accepted: bool
    selected_provider_ids: list[str] = Field(min_length=1)
    marketing_accepted: bool = False

    @field_validator("selected_provider_ids")
    @classmethod
    def _unique(cls, v: list[str]) -> list[str]:
        seen: list[str] = []
        for pid in v:
            if pid not in seen:
                seen.append(pid)
        return seen


class UpdateMissingFieldsRequest(_Base):
    """Answers keyed by dotted profile path, e.g. ``{"vehicle.make": "Fiat"}``."""

    updates: dict[str, Any] = Field(min_length=1)


class UpdatePreferencesRequest(_Base):
    """Coverage requirements. Every value set here becomes mandatory."""

    min_liability_limit_people: str | None = None
    min_liability_limit_property: str | None = None
    driving_formula: str | None = None
    max_acceptable_deductible: str | None = None
    required_optional_covers: list[str] | None = None
    accepts_black_box: bool | None = None
    accepts_approved_repair_network: bool | None = None
    payment_frequency: str | None = None


class RetryProviderRequest(_Base):
    provider_id: str


class CreatedRequestResponse(_Base):
    request_id: uuid.UUID
    status: str
    selected_provider_ids: list[str]
    demonstration_data: bool
