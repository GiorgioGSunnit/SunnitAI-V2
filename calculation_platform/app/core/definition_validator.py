"""Fail-fast structural validation for a loaded CalculatorDefinition.

Pydantic already checks the YAML has the right *shape* (types, required
top-level fields). This module checks the *content* makes sense as a
calculation: that the declared strategy exists, that it has the config
keys that strategy needs, that every parameter can in principle be
resolved, and that every expression only references variables that are
actually declared — so an authoring mistake in a formula pack fails loudly
at startup instead of silently producing a wrong number (or an opaque
UnknownVariableError) the first time someone happens to call it.
"""

from typing import Iterable, Set

from ..schemas.calculator_definition import CalculatorDefinition
from ..schemas.citation import Citation
from .errors import DefinitionValidationError
from .result_builder import ROUNDING_MODES
from .safe_evaluator import UnsafeExpressionError, extract_variable_names

_KNOWN_INPUT_TYPES = {"decimal", "integer", "boolean", "string", "date", "string_list"}

_REQUIRED_FORMULA_KEYS = {
    "expression": ["expression"],
    "progressive_brackets": ["base_input", "brackets_parameter"],
    "percentage_of_base": ["base", "rate_parameter"],
    "date_split_interest": ["rate_parameter_id"],
    "foi_revaluation": ["index_parameter_id"],
    "foi_revaluation_interest": ["index_parameter_id", "rate_parameter_id"],
    "dm55_fees": ["table_parameter_id", "amount_input", "phases_input"],
    "decision_table": ["rules"],
    "penal_range_draft": ["base_min_years", "base_max_years"],
    "table_lookup": ["table_parameter", "amount_input"],
    "procedural_deadline": ["holidays_parameter"],
    "ravvedimento": ["principal_input", "due_date_input", "payment_date_input", "tiers_parameter_id", "interest_parameter_id"],
}


def _fail(message: str, definition: CalculatorDefinition, source_file: str, **details) -> None:
    raise DefinitionValidationError(
        f"{definition.id}: {message}",
        details={"calculator_id": definition.id, "file": source_file, **details},
    )


def _check_expression(expr: str, known: Set[str], definition: CalculatorDefinition, source_file: str, where: str) -> None:
    try:
        referenced = extract_variable_names(expr)
    except UnsafeExpressionError as e:
        _fail(f"invalid expression in {where}: {e}", definition, source_file, expression=expr)
        return
    unknown = referenced - known
    if unknown:
        _fail(
            f"expression in {where} references undeclared variable(s): {', '.join(sorted(unknown))}",
            definition, source_file, expression=expr, unknown_variables=sorted(unknown),
        )


def validate_definition(definition: CalculatorDefinition, source_file: str = "<unknown>") -> None:
    from ..strategies import STRATEGIES  # local import: strategies/__init__ imports core.* leaves, not this module

    if definition.strategy not in STRATEGIES:
        _fail(
            f"unknown strategy {definition.strategy!r}; valid strategies: {', '.join(sorted(STRATEGIES))}",
            definition, source_file,
        )

    for spec in definition.inputs:
        if spec.type not in _KNOWN_INPUT_TYPES:
            _fail(
                f"input {spec.name!r} has unknown type {spec.type!r}; valid types: {', '.join(sorted(_KNOWN_INPUT_TYPES))}",
                definition, source_file, input=spec.name,
            )
        if not spec.required and spec.default is None and spec.type != "boolean":
            # Booleans commonly default to False, which is falsy-but-valid;
            # any other optional input with no default is likely a mistake,
            # but not fatal — surfaced as-is at input-validation time instead.
            pass

    for ref in definition.parameters:
        if not ref.parameter_id and ref.default is None:
            _fail(
                f"parameter {ref.name!r} has neither parameter_id nor default — it can never resolve",
                definition, source_file, parameter=ref.name,
            )

    rounding = definition.output.get("rounding")
    if rounding is not None and rounding not in ROUNDING_MODES:
        _fail(
            f"output.rounding {rounding!r} is not a known mode; valid modes: {', '.join(sorted(ROUNDING_MODES))}",
            definition, source_file, rounding=rounding,
        )

    required_keys = _REQUIRED_FORMULA_KEYS.get(definition.strategy, [])
    for key in required_keys:
        if key not in definition.formula:
            _fail(
                f"strategy {definition.strategy!r} requires formula.{key}, which is missing",
                definition, source_file, missing_formula_key=key,
            )

    for citation in definition.citations:
        try:
            Citation(**citation)
        except Exception as e:
            _fail(f"malformed citation entry: {e}", definition, source_file, citation=citation)

    known: Set[str] = {inp.name for inp in definition.inputs} | {p.name for p in definition.parameters}
    for name, expr in definition.derived_variables.items():
        _check_expression(expr, known, definition, source_file, where=f"derived_variables.{name}")
        known.add(name)

    if definition.strategy == "expression":
        expr = definition.formula.get("expression")
        if expr:
            _check_expression(expr, known, definition, source_file, where="formula.expression")
        zero_case = definition.formula.get("zero_case")
        if zero_case:
            for key in ("when_variable", "expression"):
                if key not in zero_case:
                    _fail(f"formula.zero_case is missing {key!r}", definition, source_file)
            if zero_case.get("expression"):
                _check_expression(zero_case["expression"], known, definition, source_file, where="formula.zero_case.expression")

    if definition.strategy == "percentage_of_base":
        base_expr = definition.formula.get("base")
        if base_expr:
            _check_expression(base_expr, known, definition, source_file, where="formula.base")

    # The newer strategies reference declared inputs/parameters by name from
    # formula config — verify every reference resolves, so a typo in a pack
    # fails at load time instead of on the first request.
    input_names = {inp.name for inp in definition.inputs}
    parameter_names = {p.name for p in definition.parameters}

    def _check_input_ref(key: str, name) -> None:
        if name is not None and name not in input_names:
            _fail(
                f"formula.{key} references undeclared input {name!r}",
                definition, source_file, missing_input=name,
            )

    def _check_parameter_ref(key: str, name) -> None:
        if name is not None and name not in parameter_names:
            _fail(
                f"formula.{key} references undeclared parameter {name!r}",
                definition, source_file, missing_parameter=name,
            )

    if definition.strategy == "table_lookup":
        _check_parameter_ref("table_parameter", definition.formula.get("table_parameter"))
        _check_input_ref("amount_input", definition.formula.get("amount_input"))
        _check_input_ref("indeterminable_input", definition.formula.get("indeterminable_input"))
        zero_if = definition.formula.get("zero_if")
        if zero_if:
            _check_input_ref("zero_if.input", zero_if.get("input"))
        multiplier = definition.formula.get("multiplier")
        if multiplier:
            _check_input_ref("multiplier.input", multiplier.get("input"))
            if not multiplier.get("values"):
                _fail("formula.multiplier has no 'values' map", definition, source_file)

    if definition.strategy == "procedural_deadline":
        _check_parameter_ref("holidays_parameter", definition.formula.get("holidays_parameter"))

    if definition.strategy == "ravvedimento":
        for key in ("principal_input", "due_date_input", "payment_date_input"):
            _check_input_ref(key, definition.formula.get(key))
        _check_input_ref("declaration_deadline_input", definition.formula.get("declaration_deadline_input"))
