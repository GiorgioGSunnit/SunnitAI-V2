"""AST-based restricted arithmetic evaluator.

Formulas live in YAML — untrusted-ish data, not code — so expressions are
parsed with `ast.parse(mode="eval")` (which already rejects statements,
imports, and function defs at the syntax level) and then walked by a
visitor that only knows how to handle numeric literals, variable lookups,
arithmetic operators, and a small whitelist of functions. Anything else
(attribute access, subscripting, comprehensions, lambdas, string/list/dict
literals, boolean/comparison operators) has no visitor and is rejected.
"""

import ast
from decimal import Decimal
from typing import Any, Callable, Dict

_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.FloorDiv, ast.Pow)
_ALLOWED_UNARYOPS = (ast.UAdd, ast.USub)

_ALLOWED_FUNCTIONS = frozenset({"min", "max", "abs", "round", "pow", "sum"})


class UnsafeExpressionError(ValueError):
    pass


class UnknownVariableError(ValueError):
    pass


def _to_decimal(value: Any) -> Decimal:
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


class _SafeEvaluator(ast.NodeVisitor):
    def __init__(self, variables: Dict[str, Any]):
        self.variables = variables

    def visit(self, node: ast.AST) -> Decimal:
        method = "visit_" + node.__class__.__name__
        visitor = getattr(self, method, None)
        if visitor is None:
            raise UnsafeExpressionError(f"Disallowed expression element: {node.__class__.__name__}")
        return visitor(node)

    def visit_Expression(self, node: ast.Expression) -> Decimal:
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> Decimal:
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise UnsafeExpressionError(f"Disallowed constant: {node.value!r}")
        return _to_decimal(node.value)

    def visit_Name(self, node: ast.Name) -> Decimal:
        if node.id not in self.variables:
            raise UnknownVariableError(f"Unknown variable: {node.id!r}")
        return _to_decimal(self.variables[node.id])

    def visit_BinOp(self, node: ast.BinOp) -> Decimal:
        if not isinstance(node.op, _ALLOWED_BINOPS):
            raise UnsafeExpressionError(f"Disallowed operator: {node.op.__class__.__name__}")
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Mod):
            return left % right
        if isinstance(node.op, ast.FloorDiv):
            return left // right
        return left ** int(right)  # ast.Pow — integer exponents only

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Decimal:
        if not isinstance(node.op, _ALLOWED_UNARYOPS):
            raise UnsafeExpressionError(f"Disallowed unary operator: {node.op.__class__.__name__}")
        operand = self.visit(node.operand)
        return operand if isinstance(node.op, ast.UAdd) else -operand

    def visit_Call(self, node: ast.Call) -> Decimal:
        if not isinstance(node.func, ast.Name):
            raise UnsafeExpressionError("Only whitelisted function calls are allowed")
        name = node.func.id
        if name not in _ALLOWED_FUNCTIONS:
            raise UnsafeExpressionError(f"Disallowed function: {name!r}")
        if node.keywords:
            raise UnsafeExpressionError("Keyword arguments are not allowed")
        args = [self.visit(arg) for arg in node.args]
        if name == "pow":
            if len(args) != 2:
                raise UnsafeExpressionError("pow() requires exactly 2 arguments")
            return args[0] ** int(args[1])
        if name == "round":
            if len(args) == 2:
                return round(args[0], int(args[1]))
            return round(args[0])
        if name == "min":
            return min(args)
        if name == "max":
            return max(args)
        if name == "abs":
            return abs(args[0])
        if name == "sum":
            return sum(args, Decimal("0"))
        raise UnsafeExpressionError(f"Disallowed function: {name!r}")


def safe_eval(expression: str, variables: Dict[str, Any]) -> Decimal:
    """Parse and evaluate a restricted arithmetic expression against variables."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise UnsafeExpressionError(f"Invalid expression syntax: {e}") from e
    return _SafeEvaluator(variables).visit(tree)


def extract_variable_names(expression: str) -> set:
    """Every bare variable name referenced in an expression — used by
    definition_validator to catch a formula referencing an undeclared
    input/parameter/derived variable at YAML load time, without evaluating
    anything (no variables are bound; this is a pure syntax-tree walk).

    Names used purely as function-call targets (e.g. `pow` in
    `pow(a, b)`) are excluded from the result — they are function
    references, not variables — but are checked against the whitelist so
    a call to a disallowed function is still caught here, at load time.
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise UnsafeExpressionError(f"Invalid expression syntax: {e}") from e

    call_func_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise UnsafeExpressionError("Only whitelisted function calls are allowed")
            if node.func.id not in _ALLOWED_FUNCTIONS:
                raise UnsafeExpressionError(f"Disallowed function: {node.func.id!r}")
            call_func_names.add(node.func.id)

    all_names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    return all_names - call_func_names
