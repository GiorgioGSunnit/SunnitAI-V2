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
    methodology: Optional[str] = None
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
    explanation: List[str] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list)
    warnings: List[CalcWarning] = Field(default_factory=list)
    # Things silently assumed while producing this result — defaults
    # applied for omitted optional inputs, plus any calculator-level
    # declared assumptions (e.g. "assumes a single national employer").
    assumptions: List[CalcWarning] = Field(default_factory=list)
    # The same defaults, machine-readable: one {"path", "value"} entry per
    # input (or nested object_list item field) that the caller omitted and
    # the platform filled in. `assumptions` keeps the prose form for
    # backward compatibility; this is what a caller should branch on.
    # Paths use the request's own addressing: "storico_sinistri",
    # "polizze[0].franchigia".
    defaults_applied: List[Dict[str, Any]] = Field(default_factory=list)
    # What this calculator explicitly does NOT account for, copied from the
    # definition so every consumer (API, storage, replay, report, chatbot)
    # sees the same structured list instead of re-reading the YAML.
    exclusions: List[str] = Field(default_factory=list)
    errors: List[CalculationError] = Field(default_factory=list)
