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
    assert safe_eval("round(a / b, 2)", variables) == Decimal("0.71")


def test_round_ties_use_half_up_not_banker_rounding():
    # 0.125 and 2.5 are exact ties: half-even (Decimal.__round__) would give
    # 0.12 and 2, the platform's declared half_up must give 0.13 and 3.
    assert safe_eval("round(a, 2)", {"a": Decimal("0.125")}) == Decimal("0.13")
    assert safe_eval("round(a, 0)", {"a": Decimal("2.5")}) == Decimal("3")


def test_one_arg_round_returns_decimal_and_rounds_half_up():
    result = safe_eval("round(a)", {"a": Decimal("2.5")})
    assert isinstance(result, Decimal)
    assert result == Decimal("3")


def test_round_rejects_negative_precision():
    with pytest.raises(UnsafeExpressionError):
        safe_eval("round(a, -2)", {"a": Decimal("1234.5")})


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
