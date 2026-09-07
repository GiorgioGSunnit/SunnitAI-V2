from decimal import Decimal

from app.main import engine
from app.schemas.calculation_request import CalculationRequest


def test_invoice_total_with_discount():
    request = CalculationRequest(
        calculator_id="business.invoice_total",
        inputs={"net_amount": 1000, "vat_rate": 0.22, "discount_rate": 0.10},
    )
    result = engine.calculate(request)
    assert result.status == "success"
    assert result.result["total"] == "1098.00"


def test_invoice_total_discount_defaults_to_zero():
    request = CalculationRequest(
        calculator_id="business.invoice_total",
        inputs={"net_amount": 1000, "vat_rate": 0.22},
    )
    result = engine.calculate(request)
    assert result.status == "success"
    assert result.result["total"] == "1220.00"


def test_invoice_total_missing_required_input_errors():
    request = CalculationRequest(
        calculator_id="business.invoice_total",
        inputs={"net_amount": 1000},
    )
    result = engine.calculate(request)
    assert result.status == "error"
    assert result.errors[0].code == "input_invalid"
    assert "vat_rate" in result.errors[0].message
    assert result.errors[0].details["missing_inputs"] == ["vat_rate"]
