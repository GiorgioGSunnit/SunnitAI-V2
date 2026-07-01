"""Safe deterministic formula executor.

Two execution paths:
  - 'simple'  → SafeExpressionEvaluator (whitelist AST evaluator, no eval/exec)
  - 'complex' → plugin looked up from src.calculator.formulas registry

SECURITY — SafeExpressionEvaluator whitelist:
  Allowed AST node types: Constant (numeric only), Name (variables only),
  BinOp, UnaryOp, IfExp (ternary), Compare, BoolOp, Call (whitelisted funcs).
  Any node NOT in this whitelist raises ValueError unconditionally.
  String literals, attribute access, subscripts, imports, and all other
  constructs are rejected — there are no exceptions to this rule.
"""

import ast
import operator
from typing import Any

from src.calculator.models import FormulaRecord, FormulaResult, StepResult


# ---------------------------------------------------------------------------
# Whitelisted arithmetic/comparison operators and built-in functions
# ---------------------------------------------------------------------------

ALLOWED_OPS = {
    ast.Add:      operator.add,
    ast.Sub:      operator.sub,
    ast.Mult:     operator.mul,
    ast.Div:      operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod:      operator.mod,
    ast.Pow:      operator.pow,
    ast.USub:     operator.neg,
}

ALLOWED_CMP = {
    ast.Eq:    operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt:    operator.lt,
    ast.LtE:   operator.le,
    ast.Gt:    operator.gt,
    ast.GtE:   operator.ge,
}

ALLOWED_FUNCS = {"max", "min", "round", "abs", "int", "float"}

FUNC_MAP = {
    "max":   max,
    "min":   min,
    "round": round,
    "abs":   abs,
    "int":   int,
    "float": float,
}


# ---------------------------------------------------------------------------
# SafeExpressionEvaluator
# ---------------------------------------------------------------------------

class SafeExpressionEvaluator:
    """Evaluate a Python expression string against a param dict.

    Only whitelisted AST node types are permitted. Any unknown or dangerous
    construct (string literals, attribute access, imports, subscripts, etc.)
    raises ValueError before any computation occurs.

    Example:
        ev = SafeExpressionEvaluator({"a": 10, "b": 2})
        ev.eval("a * b")   # → 20
    """

    def __init__(self, params: dict[str, Any]):
        self.params = params

    def eval(self, expression: str) -> float:
        tree = ast.parse(expression, mode="eval")
        return self._eval_node(tree.body)

    def _eval_node(self, node: ast.AST) -> Any:
        # Numeric constants only — string literals are rejected
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float, bool)):
                raise ValueError(f"String literals not allowed: {node.value!r}")
            return node.value

        # Variable lookup — must be a known parameter
        if isinstance(node, ast.Name):
            if node.id not in self.params:
                raise ValueError(f"Unknown variable: {node.id}")
            return self.params[node.id]

        # Binary arithmetic: +, -, *, /, //, %, **
        if isinstance(node, ast.BinOp):
            op = type(node.op)
            if op not in ALLOWED_OPS:
                raise ValueError(f"Operator not allowed: {op}")
            return ALLOWED_OPS[op](
                self._eval_node(node.left),
                self._eval_node(node.right)
            )

        # Unary arithmetic: - (negation only)
        if isinstance(node, ast.UnaryOp):
            if type(node.op) not in ALLOWED_OPS:
                raise ValueError("Unary operator not allowed")
            return ALLOWED_OPS[type(node.op)](self._eval_node(node.operand))

        # Ternary expression: value_if_true if condition else value_if_false
        if isinstance(node, ast.IfExp):
            cond = self._eval_node(node.test)
            return self._eval_node(node.body) if cond else self._eval_node(node.orelse)

        # Chained comparisons: a < b, a == b, etc.
        if isinstance(node, ast.Compare):
            left = self._eval_node(node.left)
            for op, comparator in zip(node.ops, node.comparators):
                op_type = type(op)
                if op_type not in ALLOWED_CMP:
                    raise ValueError(f"Comparison operator not allowed: {op_type}")
                right = self._eval_node(comparator)
                if not ALLOWED_CMP[op_type](left, right):
                    return False
                left = right
            return True

        # Boolean logic: and / or
        if isinstance(node, ast.BoolOp):
            values = [self._eval_node(v) for v in node.values]
            if isinstance(node.op, ast.And):
                return all(values)
            if isinstance(node.op, ast.Or):
                return any(values)

        # Whitelisted built-in function calls only — no attribute access
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Attribute access not allowed in function calls")
            if node.func.id not in ALLOWED_FUNCS:
                raise ValueError(f"Function not allowed: {node.func.id}")
            args = [self._eval_node(a) for a in node.args]
            return FUNC_MAP[node.func.id](*args)

        # Everything else is denied unconditionally
        raise ValueError(f"AST node type not allowed: {type(node).__name__}")


# ---------------------------------------------------------------------------
# Internal execution helpers
# ---------------------------------------------------------------------------

def _run_simple(formula: FormulaRecord, params: dict[str, Any]) -> FormulaResult:
    """Evaluate a simple formula using the SafeExpressionEvaluator."""
    evaluator = SafeExpressionEvaluator(params)
    result = evaluator.eval(formula.expression)
    steps = [StepResult(
        label="Calcolo",
        computation=formula.expression,
        result=round(float(result), 2)
    )]
    return FormulaResult(
        formula_slug=formula.slug,
        formula_name_it=formula.name_it,
        input_params=params,
        steps=steps,
        final_result=round(float(result), 2),
        source_norm=formula.source_norm
    )


def _run_plugin(formula: FormulaRecord, params: dict[str, Any]) -> FormulaResult:
    """Execute a complex formula via its registered plugin function.

    The import of get_plugin is deferred to call time so that the formulas
    package (and all its plugins) is fully loaded before the lookup happens.
    """
    from src.calculator.formulas import get_plugin
    plugin = get_plugin(formula.plugin_name)
    if plugin is None:
        raise ValueError(f"Plugin not found: {formula.plugin_name}")
    return plugin(params)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def execute(formula: FormulaRecord, params: dict[str, Any]) -> FormulaResult:
    """Execute a formula deterministically.

    Routes to SafeExpressionEvaluator for simple formulas or the plugin
    registry for complex formulas. Never calls the LLM.

    Raises:
        ValueError: unknown expression_type, blocked AST node, unknown variable,
                    or plugin not registered.
        IstatDataMissingError: TFR plugin only — missing ISTAT coefficient.
    """
    if formula.expression_type == "simple":
        return _run_simple(formula, params)
    elif formula.expression_type == "complex":
        return _run_plugin(formula, params)
    else:
        raise ValueError(f"Unknown expression_type: {formula.expression_type!r}")
