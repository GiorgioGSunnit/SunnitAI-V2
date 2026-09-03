from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class MatchRequest(BaseModel):
    query: str


class MatchCandidate(BaseModel):
    calculator_id: str
    name: str
    description: str = ""
    score: int
    matched_terms: List[str] = Field(default_factory=list)
    # Known confusions with other calculators, straight from the
    # definition's metadata — helps a router phrase a disambiguation.
    ambiguity_notes: Optional[str] = None
    # What a caller would still need to supply to actually run this
    # calculator — the "clarification" half of a future routing layer.
    required_inputs: List[Dict[str, Any]] = Field(default_factory=list)
    optional_inputs: List[Dict[str, Any]] = Field(default_factory=list)
    requires_period: bool = False
    supports_tax_year: bool = False


class MatchResponse(BaseModel):
    query: str
    # "matched"   — one candidate clearly scores above the rest
    # "ambiguous" — two or more candidates tie at the top score
    # "no_match"  — nothing scored above zero
    status: str
    candidates: List[MatchCandidate] = Field(default_factory=list)
