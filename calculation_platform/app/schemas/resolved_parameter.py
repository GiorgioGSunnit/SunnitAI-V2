from typing import Any, List, Optional

from pydantic import BaseModel, Field

from .citation import Citation


class ResolvedParameter(BaseModel):
    """Where a single parameter's value actually came from — surfaced on
    CalculationResult.parameters_used so a caller can tell "this rate was
    overridden by the caller" from "this rate came from the official
    2026 table, sourced from X" without re-deriving it."""

    name: str
    value: Any
    origin: str  # "caller_supplied" | "parameter_store" | "static_default"
    parameter_id: Optional[str] = None
    source: Optional[str] = None
    effective_from: Optional[str] = None
    effective_to: Optional[str] = None
    official: bool = False
    last_verified_at: Optional[str] = None
    citations: List[Citation] = Field(default_factory=list)
