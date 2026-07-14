from decimal import Decimal

import pytest

from app.core.safe_evaluator import UnknownVariableError, UnsafeExpressionError, safe_eval


def test_valid_arithmetic_and_precedence():
    assert safe_eval("a + b * 2", {"a": Decimal("1"), "b": Decimal("3")}) == Decimal("7")


def test_valid_whitelisted_functions():
    variables = {"a": Decimal("5"), "b": Decimal("7")}
    assert safe_eval("min(a, b, 10)", variables) == Decimal("5")
    assert safe_eval("max(a, b)", variables) == Decimal("7")
    assert safe_eval("abs(a - b)", variables) == Decimal("2")
    assert safe_eval("round(a / b, 2)", variables) == round(Decimal("5") / Decimal("7"), 2)


def test_valid_pow_with_negative_integer_exponent():
    result = safe_eval("pow(1 + rate, -months)", {"rate": Decimal("0.005"), "months": Decimal("12")})
    assert result > 0
    assert result < 1


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo pwned')",
        "().__class__.__bases__[0]",
        "[x for x in range(10)]",
        "open('/etc/passwd').read()",
        "lambda: 1",
        "a if True else b",
        "1 == 1",
        "a.__class__",
        "{1: 2}",
        "(1, 2, 3)",
        "'a string'",
        "True",
        "not a",
    ],
)
def test_unsafe_expressions_are_rejected(expression):
    with pytest.raises(UnsafeExpressionError):
        safe_eval(expression, {"a": Decimal("1"), "b": Decimal("2")})


def test_unknown_variables_are_rejected():
    with pytest.raises(UnknownVariableError):
        safe_eval("unknown_var + 1", {})


def test_invalid_syntax_is_rejected():
    with pytest.raises(UnsafeExpressionError):
        safe_eval("1 +", {})
