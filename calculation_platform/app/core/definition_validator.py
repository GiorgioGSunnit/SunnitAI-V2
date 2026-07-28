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

_KNOWN_INPUT_TYPES = {"decimal", "integer", "boolean", "string", "date", "string_list", "object_list"}
# object_list item fields must be scalars — no nested lists of objects.
_KNOWN_ITEM_FIELD_TYPES = _KNOWN_INPUT_TYPES - {"object_list"}

_REQUIRED_FORMULA_KEYS = {
    "comparator": ["candidates_input", "components"],
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
        if spec.type == "object_list":
            if not spec.item_fields:
                _fail(
                    f"object_list input {spec.name!r} declares no item_fields",
                    definition, source_file, input=spec.name,
                )
            for item_spec in spec.item_fields:
                if item_spec.type not in _KNOWN_ITEM_FIELD_TYPES:
                    _fail(
                        f"input {spec.name!r} item field {item_spec.name!r} has invalid type {item_spec.type!r}; "
                        f"valid item field types: {', '.join(sorted(_KNOWN_ITEM_FIELD_TYPES))}",
                        definition, source_file, input=spec.name, item_field=item_spec.name,
                    )
        elif spec.item_fields or spec.min_items is not None:
            _fail(
                f"input {spec.name!r} declares item_fields/min_items but is not an object_list",
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

    if definition.strategy == "comparator":
        _validate_comparator(definition, source_file)


_AGGREGATE_FUNCTION_NAMES = {"max", "min", "sum", "mean"}
# Exactly one of these per component. Order is the error-message order.
_COMPONENT_KINDS = ("relative_expression", "expression", "points", "rules")
_NUMERIC_TYPES = {"decimal", "integer"}
_NUMERIC_COMPARATOR_NAMES = {"greater_than", "less_than", "at_least", "at_most"}


def _decimal_or_none(value):
    from decimal import Decimal, InvalidOperation

    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None
    return parsed if parsed.is_finite() else None


def _validate_comparator(definition: CalculatorDefinition, source_file: str) -> None:
    from decimal import Decimal

    from ..strategies.comparator import DIRECTIONS, NORMALIZATIONS
    from ..strategies.decision_table import _COMPARATORS  # local import, same reason as STRATEGIES above

    formula = definition.formula
    candidates_name = formula.get("candidates_input")
    candidates_spec = next((s for s in definition.inputs if s.name == candidates_name), None)
    if candidates_spec is None or candidates_spec.type != "object_list":
        _fail(
            f"formula.candidates_input {candidates_name!r} does not reference a declared object_list input",
            definition, source_file, candidates_input=candidates_name,
        )
        return

    # A comparator without candidates cannot run, and comparing fewer than
    # two candidates is meaningless — make both misconfigurations fail at
    # load time instead of on the first request.
    if not candidates_spec.required and candidates_spec.default is None:
        _fail(
            f"comparator candidates input {candidates_name!r} must be required (or have a default)",
            definition, source_file, candidates_input=candidates_name,
        )
    if candidates_spec.min_items is None or candidates_spec.min_items < 2:
        _fail(
            f"comparator candidates input {candidates_name!r} must declare min_items >= 2",
            definition, source_file, candidates_input=candidates_name,
        )
    if definition.output.get("name") in {"best", "comparison"}:
        _fail(
            f"output.name {definition.output.get('name')!r} is reserved (the comparator emits "
            "'best' and 'comparison' keys alongside the ranking)",
            definition, source_file,
        )

    # How close two totals may be before the comparison refuses to name a
    # winner. Invalid here means every future comparison silently picks the
    # wrong decision_status, so it fails at load time like any weight.
    if "tie_tolerance" in formula:
        tolerance = _decimal_or_none(formula.get("tie_tolerance"))
        if tolerance is None or tolerance < 0:
            _fail(
                f"formula.tie_tolerance must be a finite non-negative number, got {formula.get('tie_tolerance')!r}",
                definition, source_file, tie_tolerance=formula.get("tie_tolerance"),
            )

    item_specs = {f.name: f for f in candidates_spec.item_fields or []}
    if len(item_specs) != len(candidates_spec.item_fields or []):
        _fail(
            f"input {candidates_name!r} declares duplicate item field names",
            definition, source_file, input=candidates_name,
        )
    boolean_items = {name for name, f in item_specs.items() if f.type == "boolean"}
    label_field = formula.get("label_field")
    if label_field is not None and label_field not in item_specs:
        _fail(
            f"formula.label_field {label_field!r} is not a declared item field",
            definition, source_file, label_field=label_field,
        )

    scalar_specs = {s.name: s for s in definition.inputs if s.name != candidates_name}
    parameter_names = {p.name for p in definition.parameters}

    # Names must be disjoint across every source that feeds a candidate's
    # expression environment — a collision would silently shadow one value
    # with another (e.g. a candidate field hiding the user's shared input).
    def _require_fresh(name: str, kind: str, taken: Set[str]) -> None:
        if name in taken:
            _fail(
                f"{kind} {name!r} collides with an already-declared variable name",
                definition, source_file, variable=name,
            )

    known: Set[str] = set(item_specs)
    for name in scalar_specs:
        _require_fresh(name, "scalar input", known)
        known.add(name)
    for name in parameter_names:
        _require_fresh(name, "parameter", known)
        known.add(name)

    derived_names = list(formula.get("candidate_derived") or {})
    for name, expr in (formula.get("candidate_derived") or {}).items():
        _require_fresh(name, "candidate_derived variable", known)
        _check_expression(expr, known, definition, source_file, where=f"formula.candidate_derived.{name}")
        known.add(name)

    # numeric-and-always-present names: usable in aggregates and numeric rule
    # comparisons without risking a missing value or a non-numeric operand
    def _numeric_present(name: str) -> bool:
        if name in derived_names or name in (formula.get("aggregates") or {}):
            return True
        spec = item_specs.get(name) or scalar_specs.get(name)
        if spec is None:
            return False
        return spec.type in _NUMERIC_TYPES and (spec.required or spec.default is not None)

    for name, agg in (formula.get("aggregates") or {}).items():
        _require_fresh(name, "aggregate", known)
        if not isinstance(agg, dict) or agg.get("function") not in _AGGREGATE_FUNCTION_NAMES:
            _fail(
                f"formula.aggregates.{name} needs a function among {', '.join(sorted(_AGGREGATE_FUNCTION_NAMES))}",
                definition, source_file, aggregate=name,
            )
        over = agg.get("over")
        if over not in known:
            _fail(
                f"formula.aggregates.{name} is over undeclared variable {over!r}",
                definition, source_file, aggregate=name,
            )
        if not _numeric_present(over):
            _fail(
                f"formula.aggregates.{name} is over {over!r}, which is not a numeric always-present "
                "variable (declare it decimal/integer and required, or give it a default)",
                definition, source_file, aggregate=name,
            )
        known.add(name)

    components = formula.get("components") or []
    if not components:
        _fail("formula.components is empty", definition, source_file)
    seen_names = set()
    weight_total = Decimal("0")
    for component in components:
        if not isinstance(component, dict):
            _fail(f"components entries must be mappings, got {component!r}", definition, source_file)
        comp_name = component.get("name")
        if not comp_name or comp_name in seen_names:
            _fail(
                f"component name {comp_name!r} is missing or duplicated",
                definition, source_file, component=comp_name,
            )
        seen_names.add(comp_name)

        weight = _decimal_or_none(component.get("weight"))
        if weight is None or not (0 <= weight <= 1):
            _fail(
                f"component {comp_name!r} weight must be a finite number in [0, 1], got {component.get('weight')!r}",
                definition, source_file, component=comp_name,
            )
        weight_total += weight

        clamp = component.get("clamp")
        if clamp is not None:
            clamp_min = _decimal_or_none(clamp.get("min")) if isinstance(clamp, dict) else None
            clamp_max = _decimal_or_none(clamp.get("max")) if isinstance(clamp, dict) else None
            if (
                not isinstance(clamp, dict)
                or (clamp.get("min") is not None and clamp_min is None)
                or (clamp.get("max") is not None and clamp_max is None)
                or (clamp_min is not None and not (0 <= clamp_min <= 100))
                or (clamp_max is not None and not (0 <= clamp_max <= 100))
                or (clamp_min is not None and clamp_max is not None and clamp_min > clamp_max)
            ):
                _fail(
                    f"component {comp_name!r} clamp must be a mapping with numeric min <= max within [0, 100]",
                    definition, source_file, component=comp_name, clamp=clamp,
                )

        kinds = [k for k in _COMPONENT_KINDS if k in component]
        if len(kinds) != 1:
            _fail(
                f"component {comp_name!r} must declare exactly one of "
                f"{'/'.join(_COMPONENT_KINDS)}, got {kinds or 'none'}",
                definition, source_file, component=comp_name,
            )
        kind = kinds[0]
        # direction/normalization only mean something to a relative
        # component; tolerating them elsewhere would let a pack author
        # believe a plain expression is being normalized when it is not.
        if kind != "relative_expression":
            stray = [k for k in ("direction", "normalization") if k in component]
            if stray:
                _fail(
                    f"component {comp_name!r} declares {', '.join(stray)} but is a "
                    f"{kind!r} component; those apply to relative_expression only",
                    definition, source_file, component=comp_name,
                )
        if kind == "relative_expression":
            _check_expression(
                component["relative_expression"], known, definition, source_file,
                where=f"components.{comp_name}.relative_expression",
            )
            direction = component.get("direction")
            if direction not in DIRECTIONS:
                _fail(
                    f"components.{comp_name}.direction must be one of "
                    f"{', '.join(sorted(DIRECTIONS))}, got {direction!r}",
                    definition, source_file, component=comp_name, direction=direction,
                )
            normalization = component.get("normalization")
            if normalization not in NORMALIZATIONS:
                _fail(
                    f"components.{comp_name}.normalization must be one of "
                    f"{', '.join(sorted(NORMALIZATIONS))}, got {normalization!r}",
                    definition, source_file, component=comp_name, normalization=normalization,
                )
        elif kind == "expression":
            _check_expression(component["expression"], known, definition, source_file, where=f"components.{comp_name}.expression")
        elif kind == "points":
            entries = component["points"]
            if not isinstance(entries, list) or not entries:
                _fail(
                    f"components.{comp_name}.points must be a non-empty list",
                    definition, source_file, component=comp_name,
                )
            for entry in entries:
                if not isinstance(entry, dict) or entry.get("field") not in boolean_items:
                    _fail(
                        f"components.{comp_name}.points references {entry.get('field') if isinstance(entry, dict) else entry!r}, "
                        "which is not a declared boolean item field",
                        definition, source_file, component=comp_name,
                    )
                if _decimal_or_none(entry.get("points")) is None:
                    _fail(
                        f"components.{comp_name}.points entry for {entry.get('field')!r} needs a numeric 'points' value",
                        definition, source_file, component=comp_name, field=entry.get("field"),
                    )
            scale_max = component.get("scale_max")
            if scale_max is not None:
                parsed = _decimal_or_none(scale_max)
                if parsed is None or parsed <= 0:
                    _fail(
                        f"components.{comp_name}.scale_max must be a positive number, got {scale_max!r}",
                        definition, source_file, component=comp_name,
                    )
        else:  # rules
            rules = component["rules"]
            if not isinstance(rules, list) or not rules:
                _fail(
                    f"components.{comp_name}.rules must be a non-empty list",
                    definition, source_file, component=comp_name,
                )
            if _decimal_or_none(component.get("base", 100)) is None:
                _fail(
                    f"components.{comp_name}.base must be numeric, got {component.get('base')!r}",
                    definition, source_file, component=comp_name,
                )
            for rule in rules:
                when = rule.get("when") if isinstance(rule, dict) else None
                if not isinstance(when, dict) or when.get("field") not in known:
                    _fail(
                        f"components.{comp_name} has a rule whose 'when' is missing or references an undeclared field",
                        definition, source_file, component=comp_name, rule=rule,
                    )
                comparators_used = [c for c in _COMPARATORS if c in when]
                if len(comparators_used) != 1:
                    _fail(
                        f"components.{comp_name} rule must use exactly one comparator among "
                        f"{', '.join(sorted(_COMPARATORS))}, got {comparators_used or 'none'}",
                        definition, source_file, component=comp_name, rule=rule,
                    )
                if comparators_used[0] in _NUMERIC_COMPARATOR_NAMES and not _numeric_present(when["field"]):
                    _fail(
                        f"components.{comp_name} rule compares {when['field']!r} numerically, but that "
                        "variable is not numeric-and-always-present (declare it decimal/integer and "
                        "required, or give it a default)",
                        definition, source_file, component=comp_name, rule=rule,
                    )
                if ("points" in rule) == ("points_per_unit" in rule):
                    _fail(
                        f"components.{comp_name} rule must declare exactly one of points/points_per_unit",
                        definition, source_file, component=comp_name, rule=rule,
                    )
                points_key = "points" if "points" in rule else "points_per_unit"
                if _decimal_or_none(rule.get(points_key)) is None:
                    _fail(
                        f"components.{comp_name} rule {points_key} must be numeric, got {rule.get(points_key)!r}",
                        definition, source_file, component=comp_name, rule=rule,
                    )
                if "points_per_unit" in rule and not _numeric_present(when["field"]):
                    _fail(
                        f"components.{comp_name} points_per_unit rule needs a numeric always-present field, got {when['field']!r}",
                        definition, source_file, component=comp_name, rule=rule,
                    )

    if weight_total != 1:
        _fail(
            f"component weights must sum to 1 so the total stays on the 0-100 scale; they sum to {weight_total}",
            definition, source_file, weight_total=str(weight_total),
        )
