import pytest
from fastapi.testclient import TestClient

from app.main import app, engine
from app.reporting import render_report_html
from app.schemas.calculation_request import CalculationRequest
from app.schemas.stored_calculation import StoredCalculation
from simulation.conversation import SimulatedConversation
from src.rag.calculation import _COPY, _success_answer


client = TestClient(app)


def _irpef_result():
    request = CalculationRequest(
        request_id="methodology-test",
        calculator_id="legal_it.irpef",
        inputs={"taxable_income": 42000},
        tax_year=2026,
    )
    return request, engine.calculate(request)


def test_methodology_reaches_calculate_json():
    response = client.post(
        "/calculate",
        json={
            "calculator_id": "legal_it.irpef",
            "inputs": {"taxable_income": 42000},
            "tax_year": 2026,
        },
    )

    assert response.status_code == 200
    assert response.json()["methodology"] == (
        "Scaglioni progressivi: ogni fascia di reddito è tassata alla propria aliquota."
    )


def test_methodology_reaches_html_report_below_warnings():
    request, result = _irpef_result()
    stored = StoredCalculation(
        request_id=request.request_id,
        created_at="2026-01-15T12:00:00Z",
        calculator_id=request.calculator_id,
        status=result.status,
        result_preview=result.result,
        request=request.model_dump(mode="json"),
        result=result.model_dump(mode="json"),
    )

    html = render_report_html(stored, engine.registry.get(request.calculator_id))

    assert "<h2>Metodo</h2>" in html
    assert result.methodology in html
    assert html.index("<h2>Avvertenze / Assunzioni</h2>") < html.index("<h2>Metodo</h2>")


def test_methodology_reaches_development_conversation():
    reply = SimulatedConversation(engine).send(
        "quanto pago di tasse su un reddito di 42000 euro nel 2026"
    )

    assert reply.kind == "answer"
    assert f"Metodo: {reply.calculation.methodology}" in reply.text


@pytest.mark.parametrize(
    "lang,heading",
    [
        ("it", "Come e stato calcolato"),
        ("es", "Como se ha calculado"),
        ("en", "How it was computed"),
    ],
)
def test_success_answer_uses_localized_methodology_heading(lang, heading):
    methodology = "Scaglioni progressivi: ogni fascia è tassata alla propria aliquota."

    answer = _success_answer(
        lang,
        {
            "status": "success",
            "result": {"gross_tax": "11060.00"},
            "methodology": methodology,
        },
    )

    assert _COPY[lang]["methodology"] == heading
    assert f"{heading}:\n{methodology}" in answer
    assert answer.index(heading) < answer.index(_COPY[lang]["sources"])


def test_methodology_is_none_on_error_result():
    result = engine.calculate(
        CalculationRequest(calculator_id="does.not.exist", inputs={"amount": 10})
    )

    assert result.status == "error"
    assert result.methodology is None


def test_every_calculator_declares_a_non_empty_methodology():
    missing = [
        calculator_id
        for calculator_id, definition in engine.registry._definitions.items()
        if not definition.methodology or not definition.methodology.strip()
    ]
    assert not missing, f"Calculators with no methodology: {missing}"
