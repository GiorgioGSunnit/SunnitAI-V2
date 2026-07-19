from app.main import engine
from app.reporting import render_report_html
from app.schemas.calculation_request import CalculationRequest
from app.schemas.stored_calculation import StoredCalculation


def test_render_report_html_contains_audit_sections_and_escapes_user_input():
    request = CalculationRequest(
        request_id="report-render-test",
        calculator_id="legal_it.irpef",
        inputs={"taxable_income": 42000},
        tax_year=2026,
    )
    result = engine.calculate(request)
    result.request_id = request.request_id
    result.raw_inputs["malicious"] = "<script>alert(1)</script>"

    stored = StoredCalculation(
        request_id=request.request_id,
        created_at="2026-01-15T12:00:00Z",
        calculator_id=request.calculator_id,
        status=result.status,
        result_preview=result.result,
        request=request.model_dump(mode="json"),
        result=result.model_dump(mode="json"),
    )

    definition = engine.registry.get("legal_it.irpef")
    html = render_report_html(stored, definition)

    assert definition.name in html
    assert "11060" in html
    assert "bracket_up_to" in html
    assert "Art. 11 D.P.R." in html
    assert "non costituisce parere legale" in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "<script>alert(1)</script>" not in html
