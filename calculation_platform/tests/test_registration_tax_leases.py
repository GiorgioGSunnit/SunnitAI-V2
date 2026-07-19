from app.main import engine
from app.schemas.calculation_request import CalculationRequest


def test_ordinary_lease_applies_two_percent():
    request = CalculationRequest(
        calculator_id="legal_it.registration_tax_leases",
        inputs={"annual_rent": 9600, "years": 4, "first_registration": True},
    )
    result = engine.calculate(request)
    assert result.status == "success"
    assert result.result["tax_due"] == "768.00"


def test_first_registration_applies_minimum_of_67():
    request = CalculationRequest(
        calculator_id="legal_it.registration_tax_leases",
        inputs={"annual_rent": 500, "years": 1, "first_registration": True},
    )
    result = engine.calculate(request)
    assert result.status == "success"
    assert result.result["tax_due"] == "67.00"


def test_non_first_registration_below_minimum_warns_instead_of_flooring():
    request = CalculationRequest(
        calculator_id="legal_it.registration_tax_leases",
        inputs={"annual_rent": 500, "years": 1, "first_registration": False},
    )
    result = engine.calculate(request)
    assert result.status == "success"
    assert result.result["tax_due"] == "10.00"
    messages = [w.message for w in result.warnings]
    assert any("subsequent-year" in m for m in messages)


def test_cedolare_secca_returns_zero_with_warning():
    request = CalculationRequest(
        calculator_id="legal_it.registration_tax_leases",
        inputs={"annual_rent": 9600, "years": 4, "first_registration": True, "cedolare_secca": True},
    )
    result = engine.calculate(request)
    assert result.status == "success"
    assert result.result["tax_due"] == "0.00"
    messages = [w.message for w in result.warnings]
    assert any("cedolare secca" in m for m in messages)


def test_tax_exactly_at_minimum_boundary_is_not_flagged_as_clamped():
    # 3350 * 1 * 0.02 == 67.00 exactly — the minimum equals the calculated
    # tax, so no clamping should occur (tax < minimum is false).
    request = CalculationRequest(
        calculator_id="legal_it.registration_tax_leases",
        inputs={"annual_rent": 3350, "years": 1, "first_registration": True},
    )
    result = engine.calculate(request)
    assert result.status == "success"
    assert result.result["tax_due"] == "67.00"
    assert not any(s.get("type") == "minimum_applied" for s in result.steps)


def test_zero_rent_with_first_registration_still_floors_to_minimum():
    request = CalculationRequest(
        calculator_id="legal_it.registration_tax_leases",
        inputs={"annual_rent": 0, "years": 1, "first_registration": True},
    )
    result = engine.calculate(request)
    assert result.status == "success"
    assert result.result["tax_due"] == "67.00"


def test_parameters_used_includes_source_citation():
    request = CalculationRequest(
        calculator_id="legal_it.registration_tax_leases",
        inputs={"annual_rent": 9600, "years": 4, "first_registration": True},
    )
    result = engine.calculate(request)
    rate_info = result.parameters_used["rate"]
    assert rate_info["origin"] == "parameter_store"
    assert "D.P.R." in rate_info["source"]
