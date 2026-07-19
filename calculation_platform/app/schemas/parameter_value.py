from datetime import date
from typing import Any, List, Optional

from pydantic import BaseModel, Field, field_validator

from .citation import Citation


class ParameterValue(BaseModel):
    """A single date-versioned value for a parameter_id (e.g. one year's tasso legale)."""

    parameter_id: str
    value: Any
    unit: Optional[str] = None
    effective_from: date
    effective_to: Optional[date] = None
    source: Optional[str] = None
    citations: List[Citation] = Field(default_factory=list)
    official: bool = False
    last_verified_at: Optional[str] = None
    # Data-quality marks for values shipped as schema examples awaiting a
    # human check against the official source (see docs/TO_VERIFY.md).
    verified: Optional[bool] = None
    placeholder: bool = False

    @field_validator("last_verified_at", mode="before")
    @classmethod
    def normalize_last_verified_at(cls, value):
        if isinstance(value, date):
            return value.isoformat()
        return value
