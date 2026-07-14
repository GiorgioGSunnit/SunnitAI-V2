from datetime import date
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class Period(BaseModel):
    start_date: date
    end_date: date


class CalculationOptions(BaseModel):
    explain: bool = True
    rounding: Optional[str] = None
    require_sources: bool = False


class CalculationRequest(BaseModel):
    request_id: Optional[str] = None
    calculator_id: str
    jurisdiction: Optional[str] = None
    as_of_date: Optional[date] = None
    tax_year: Optional[int] = None
    period: Optional[Period] = None
    inputs: Dict[str, Any] = Field(default_factory=dict)
    caller_supplied_values: Dict[str, Any] = Field(default_factory=dict)
    options: CalculationOptions = Field(default_factory=CalculationOptions)
    # Reserved for future multi-tenant support — not enforced yet.
    tenant_id: Optional[str] = None
