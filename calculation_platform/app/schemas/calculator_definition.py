from datetime import date
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class InputSpec(BaseModel):
    name: str
    type: str  # decimal | integer | boolean | date | string
    required: bool = True
    default: Optional[Any] = None
    unit: Optional[str] = None
    # Optional inclusive bounds, checked after type coercion — for
    # decimal/integer inputs where a negative or out-of-range value is
    # nonsensical (e.g. a negative taxable income) rather than merely
    # unusual. Left unset for a field means "not bounded".
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    description: Optional[str] = None


class ParameterRef(BaseModel):
    """A parameter a calculator needs resolved before running its strategy."""

    name: str
    parameter_id: Optional[str] = None
    default: Optional[Any] = None
    resolver: Optional[str] = None  # "date" | "static"
    description: Optional[str] = None


class CalculatorExample(BaseModel):
    """A worked example: given inputs, the calculator must produce this
    result. These aren't just documentation — tests/test_examples.py runs
    every declared example through the real engine and asserts it matches,
    so a formula pack's docs can't silently drift from its behavior."""

    description: Optional[str] = None
    inputs: Dict[str, Any] = Field(default_factory=dict)
    tax_year: Optional[int] = None
    as_of_date: Optional[date] = None
    period: Optional[Dict[str, str]] = None
    caller_supplied_values: Dict[str, Any] = Field(default_factory=dict)
    expected_result: Dict[str, Any] = Field(default_factory=dict)


class CalculatorDefinition(BaseModel):
    """A single calculator's declarative definition, loaded from a formula-pack YAML file."""

    id: str
    name: str
    category: str
    strategy: str
    version: str = "1"  # formula/definition version — bump when the formula's logic itself changes
    description: Optional[str] = None
    jurisdiction: Optional[str] = None  # e.g. "IT"
    # The calculator's own validity window — distinct from a parameter's
    # effective_from/effective_to, which versions a single value. This is
    # for the rare case where the whole calculator becomes obsolete or
    # only applies from some date (e.g. a tax introduced/repealed).
    applicable_from: Optional[date] = None
    applicable_to: Optional[date] = None
    output_unit: Optional[str] = None  # e.g. "EUR"
    inputs: List[InputSpec] = Field(default_factory=list)
    parameters: List[ParameterRef] = Field(default_factory=list)
    derived_variables: Dict[str, str] = Field(default_factory=dict)
    formula: Dict[str, Any] = Field(default_factory=dict)
    output: Dict[str, Any] = Field(default_factory=dict)
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    regime_selector: Optional[Dict[str, Any]] = None
    warnings: List[str] = Field(default_factory=list)
    # Declarative, always-true caveats about this calculator's model (e.g.
    # "assumes a single national employer"). Distinct from
    # CalculationResult.assumptions, which also includes per-request
    # runtime facts (e.g. "discount_rate not provided; assumed 0").
    assumptions: List[str] = Field(default_factory=list)
    # Things this calculator explicitly does NOT account for (e.g.
    # "regional/municipal IRPEF surcharges") — stronger and more specific
    # than a free-text warning.
    exclusions: List[str] = Field(default_factory=list)
    examples: List[CalculatorExample] = Field(default_factory=list)
    # Routing metadata, consumed by the deterministic matcher/planner.
    # `keywords`: terms/synonyms describing the calculation itself (legal
    # and plain-language, multiple languages where relevant).
    # `aliases`: short phrases a user might actually type/say to ask for
    # this calculation, closer to natural query fragments than terms.
    # `tags`: single-word topic labels (vocabulary for token matching).
    # `intent_examples`: full example requests that should route here —
    # multi-word ones act as strong phrase evidence.
    # `negative_examples`: requests that look related but should NOT route
    # here (applied as a scoring penalty, e.g. asking about the cedolare
    # secca tax itself vs. registration tax).
    # `ambiguity_notes`: free text documenting known confusions with other
    # calculators, surfaced on match candidates.
    # `required_context`: contextual info (beyond named inputs) a caller
    # must establish before the result is meaningful, e.g. the tax year.
    keywords: List[str] = Field(default_factory=list)
    aliases: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    intent_examples: List[str] = Field(default_factory=list)
    negative_examples: List[str] = Field(default_factory=list)
    ambiguity_notes: Optional[str] = None
    required_context: List[str] = Field(default_factory=list)
