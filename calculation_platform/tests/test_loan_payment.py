from decimal import Decimal

from app.main import engine
from app.schemas.calculation_request import CalculationRequest


def test_loan_payment_normal_case_returns_sensible_rounded_decimal():
    request = CalculationRequest(
        calculator_id="business.loan_payment",
        inputs={"principal": 10000, "annual_rate": 0.06, "months": 12},
    )
    result = engine.calculate(request)
    assert result.status == "success"
    payment = result.result["monthly_payment"]
    # Decimals serialize as exact strings, never floats
    assert isinstance(payment, str)
    payment_decimal = Decimal(payment)
    # 12 monthly payments should roughly repay principal plus some interest
    assert 800 < payment_decimal < 900
    # quantized to exactly 2 decimal places
    assert payment_decimal == payment_decimal.quantize(Decimal("0.01"))


def test_loan_payment_zero_rate_equals_principal_over_months():
    request = CalculationRequest(
        calculator_id="business.loan_payment",
        inputs={"principal": 12000, "annual_rate": 0, "months": 12},
    )
    result = engine.calculate(request)
    assert result.status == "success"
    assert result.result["monthly_payment"] == "1000.00"
