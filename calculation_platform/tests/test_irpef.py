from app.main import engine
from app.schemas.calculation_request import CalculationRequest


def test_irpef_2026_regime():
    request = CalculationRequest(
        calculator_id="legal_it.irpef",
        inputs={"taxable_income": 42000},
        tax_year=2026,
    )
    result = engine.calculate(request)
    assert result.status == "success"
    # 28000 * 0.23 = 6440; 14000 * 0.33 = 4620; total = 11060
    assert result.result["gross_tax"] == "11060.00"


def test_irpef_2024_regime():
    request = CalculationRequest(
        calculator_id="legal_it.irpef",
        inputs={"taxable_income": 42000},
        tax_year=2024,
    )
    result = engine.calculate(request)
    assert result.status == "success"
    # 28000 * 0.23 = 6440; 14000 * 0.35 = 4900; total = 11340
    assert result.result["gross_tax"] == "11340.00"


def test_irpef_carries_gross_only_warnings():
    request = CalculationRequest(
        calculator_id="legal_it.irpef",
        inputs={"taxable_income": 42000},
        tax_year=2026,
    )
    result = engine.calculate(request)
    messages = [w.message for w in result.warnings]
    assert any("gross national IRPEF only" in m for m in messages)


def test_irpef_missing_tax_year_falls_back_to_today():
    # No tax_year/as_of_date given — resolver falls back to date.today(),
    # which currently resolves to the 2026 bracket table.
    request = CalculationRequest(
        calculator_id="legal_it.irpef",
        inputs={"taxable_income": 20000},
    )
    result = engine.calculate(request)
    assert result.status == "success"
    assert result.result["gross_tax"] == "4600.00"  # 20000 * 0.23


def test_irpef_zero_income_yields_zero_tax():
    request = CalculationRequest(
        calculator_id="legal_it.irpef",
        inputs={"taxable_income": 0},
        tax_year=2026,
    )
    result = engine.calculate(request)
    assert result.status == "success"
    assert result.result["gross_tax"] == "0.00"
    assert result.steps == []  # no bracket has any taxable amount in it


def test_irpef_income_exactly_at_bracket_boundary():
    # Exactly 28000 should fall entirely in the first bracket (up_to is
    # inclusive: min(base, upper) - previous_threshold).
    request = CalculationRequest(
        calculator_id="legal_it.irpef",
        inputs={"taxable_income": 28000},
        tax_year=2026,
    )
    result = engine.calculate(request)
    assert result.status == "success"
    assert result.result["gross_tax"] == "6440.00"  # 28000 * 0.23
    assert len(result.steps) == 1


def test_irpef_negative_income_rejected():
    request = CalculationRequest(
        calculator_id="legal_it.irpef",
        inputs={"taxable_income": -100},
        tax_year=2026,
    )
    result = engine.calculate(request)
    assert result.status == "error"
