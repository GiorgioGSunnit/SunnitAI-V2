from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .citation import Citation
from .error import CalculationError
from .warning import Warning as CalcWarning


class CalculationResult(BaseModel):
    request_id: Optional[str] = None
    calculator_id: str
    status: str  # "success" | "error"
    result: Dict[str, Any] = Field(default_factory=dict)
    formula_used: Optional[str] = None
    formula_version: Optional[str] = None
    # Exactly what the caller sent, before type coercion/defaults —
    # preserved for audit even when validation fails.
    raw_inputs: Dict[str, Any] = Field(default_factory=dict)
    inputs_used: Dict[str, Any] = Field(default_factory=dict)
    # Per-parameter: value, origin (caller/store/default), and — when
    # sourced from the parameter store — its citation/effective range.
    parameters_used: Dict[str, Any] = Field(default_factory=dict)
    # How the as-of date used for parameter lookups was determined
    # (explicit as_of_date / derived from tax_year / defaulted to today).
    # None when the calculator resolved no date-versioned parameters.
    date_resolution: Optional[Dict[str, Any]] = None
    derived_values: Dict[str, Any] = Field(default_factory=dict)
    steps: List[Dict[str, Any]] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list)
    warnings: List[CalcWarning] = Field(default_factory=list)
    # Things silently assumed while producing this result — defaults
    # applied for omitted optional inputs, plus any calculator-level
    # declared assumptions (e.g. "assumes a single national employer").
    assumptions: List[CalcWarning] = Field(default_factory=list)
    errors: List[CalculationError] = Field(default_factory=list)
