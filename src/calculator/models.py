"""Pydantic model contracts for the formula calculation engine.

These models are the single source of truth for all data shapes exchanged
between the calculator components:
  registry   → FormulaRecord, ParameterDefinition
  router     → RouterDecision
  extractor  → ExtractedParams
  executor   → FormulaResult, StepResult
  formatter  → FormulaResult (input), dict (output)
  api        → CalculateRequest, DirectCalculateRequest, MissingParamsError

IMPORTANT:
  - FormulaRecord.similarity_score is ephemeral — populated by the router at
    query time and never stored in the DB.
  - ExtractedParams.clarification_questions contains Italian-language questions
    returned directly to the user when required params are missing.
  - StepResult is the building block of the citation block shown to the user.
"""

from pydantic import BaseModel
from typing import Any, Literal, Optional
from uuid import UUID


class ParameterDefinition(BaseModel):
    """Describes a single input parameter expected by a formula.

    Stored as JSONB in the formulas table (parameter_schema column).
    The 'type' field is used by the extractor to cast extracted values.
    """
    name: str
    name_it: str
    type: Literal["float", "int", "bool", "str", "date"]
    required: bool
    description_it: str
    default_value: Optional[Any] = None
    unit: Optional[str] = None


class FormulaRecord(BaseModel):
    """In-memory representation of a formula loaded from the DB.

    Not a DB model — this is the application-layer view of a formulas row.
    similarity_score is NOT stored in the DB; it is populated by the router
    after computing cosine similarity against the query embedding.
    """
    id: UUID
    slug: str
    name_it: str
    description_it: str
    category: str
    expression_type: Literal["simple", "complex"]
    expression: Optional[str] = None       # None for complex formulas
    plugin_name: Optional[str] = None      # None for simple formulas
    parameter_schema: list[ParameterDefinition]
    source_norm: Optional[str] = None
    similarity_score: Optional[float] = None   # ephemeral — set by router only


class ExtractedParams(BaseModel):
    """Result of the LLM parameter extraction step.

    clarification_questions: Italian-language questions to ask the user
    when required parameters could not be found in the conversation.
    These are returned directly to the frontend when missing_required is non-empty.
    """
    formula_slug: str
    params: dict[str, Any]
    missing_required: list[str]
    clarification_questions: list[str]   # Italian questions to ask user
    confidence: float


class StepResult(BaseModel):
    """A single computation step within a formula result.

    This is the building block of the citation block shown to the user.
    Example:
      label       = "Penale giornaliera"
      computation = "120.000 × 0,5%"
      result      = 600.0
    """
    label: str
    computation: str
    result: float


class FormulaResult(BaseModel):
    """Complete output of a formula execution.

    steps: ordered list of StepResult — shown as the reasoning trace.
    warning: optional Italian-language caveat (e.g. approximate ISTAT data).
    """
    formula_slug: str
    formula_name_it: str
    input_params: dict[str, Any]
    steps: list[StepResult]
    final_result: float
    unit: Optional[str] = None
    source_norm: Optional[str] = None
    warning: Optional[str] = None   # e.g. "ISTAT data for year X is approximate"


class RouterDecision(BaseModel):
    """Decision produced by the semantic router for a given user query.

    is_calculation=False means the query should fall through to the RAG pipeline.
    method='none' means no embedding match was found above the threshold.
    """
    is_calculation: bool
    formula: Optional[FormulaRecord] = None
    confidence: float
    method: Literal["semantic", "llm_fallback", "none"]


class CalculateRequest(BaseModel):
    """Request body for the full pipeline endpoint (/api/calculate).

    conversation_history: list of {"role": ..., "content": ...} dicts,
    passed to the LLM extractor to find parameter values in context.
    """
    query: str
    conversation_history: list[dict[str, str]] = []


class DirectCalculateRequest(BaseModel):
    """Request body for the direct execution endpoint (/api/calculate/direct).

    Caller provides formula_slug + fully resolved params.
    No routing, no LLM — deterministic execution only.
    Used by aiac-be document generation pipeline.
    """
    formula_slug: str
    params: dict[str, Any]


class MissingParamsError(BaseModel):
    """Response shape returned when required parameters are missing.

    questions: Italian-language clarification questions for the user.
    Matches the ExtractedParams.clarification_questions content.
    """
    detail: Literal["missing_params"]
    formula_slug: str
    missing: list[str]
    questions: list[str]
