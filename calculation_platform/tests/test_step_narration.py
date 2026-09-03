import pytest
from fastapi.testclient import TestClient

from app.api.routes import set_store
from app.core.step_narration import MAX_EXPLANATION_LINES, narrate
from app.main import app, engine
from app.schemas.calculation_request import CalculationRequest
from app.storage.sqlite_store import SqliteCalculationStore
from src.rag.calculation import _success_answer


@pytest.fixture
def client(tmp_path):
    set_store(SqliteCalculationStore(tmp_path / "step-narration.db"))
    return TestClient(app)


def test_progressive_brackets_are_narrated_with_exact_request_values():
    result = engine.calculate(
        CalculationRequest(
            calculator_id="legal_it.irpef",
            inputs={"taxable_income": 42000},
            tax_year=2026,
        )
    )

    assert result.explanation == [
        "28.000 al 23% = 6.440,00",
        "14.000 al 33% = 4.620,00",
    ]


def test_dm55_narration_preserves_staged_subtotals():
    result = engine.calculate(
        CalculationRequest(
            calculator_id="legal_it.compensi_dm55",
            inputs={
                "valore_causa": 30000,
                "fasi": ["studio", "introduttiva", "decisionale"],
            },
        )
    )

    assert result.explanation[0] == "Fase studio: valore medio di scaglione 1701 EUR"
    assert (
        "+15% rimborso spese generali (art. 2 DM 55/2014): "
        "871.50 EUR, subtotale 6681.50 EUR"
    ) in result.explanation
    assert result.explanation[-1] == (
        "+22% IVA: 1528.727200 EUR, totale 8477.487200 EUR"
    )


def test_expression_strategy_uses_generic_fallback():
    result = engine.calculate(
        CalculationRequest(
            calculator_id="business.invoice_total",
            inputs={"net_amount": 1000, "vat_rate": "0.22", "discount_rate": "0.10"},
        )
    )

    assert result.explanation == [
        "variable: discount_amount; expression: net_amount * discount_rate; value: 100.00",
        "variable: taxable_amount; expression: net_amount - discount_amount; value: 900.00",
        "variable: vat_amount; expression: taxable_amount * vat_rate; value: 198.0000",
        "variable: total; expression: taxable_amount + vat_amount; value: 1098.00",
    ]
    assert all("step:" not in line and "type:" not in line for line in result.explanation)


def test_narration_caps_long_traces_and_reports_omitted_count():
    steps = [
        {"step": index, "type": "synthetic", "value": str(index)}
        for index in range(1, MAX_EXPLANATION_LINES + 6)
    ]

    lines = narrate(steps, "unknown_strategy")

    assert lines[:MAX_EXPLANATION_LINES] == [
        f"value: {index}" for index in range(1, MAX_EXPLANATION_LINES + 1)
    ]
    assert lines[-1] == "(+5 passaggi nel report)"
    assert len(lines) == MAX_EXPLANATION_LINES + 1


def test_narrate_empty_steps_returns_empty_list():
    assert narrate([], "progressive_brackets") == []


def test_explanation_reaches_calculate_json_and_replay_still_matches(client):
    calculated = client.post(
        "/calculate",
        json={
            "calculator_id": "legal_it.irpef",
            "inputs": {"taxable_income": 42000},
            "tax_year": 2026,
        },
    ).json()

    assert calculated["explanation"] == [
        "28.000 al 23% = 6.440,00",
        "14.000 al 33% = 4.620,00",
    ]

    replay = client.post(f"/calculations/{calculated['request_id']}/replay")
    assert replay.status_code == 200
    assert replay.json()["matches"] is True
    assert replay.json()["replayed_result"]["explanation"] == calculated["explanation"]


def test_comparator_answer_omits_methodology_and_raw_step_narration():
    raw_line = "candidate: Alfa; exact total: 85.00"
    answer = _success_answer(
        "it",
        {
            "status": "success",
            "result": {
                "best": "Alfa",
                "comparison": {
                    "decision_status": "clear_winner",
                    "best_candidates": ["Alfa"],
                    "score_gap": "10.00",
                    "tie_tolerance": "0.50",
                },
                "ranking": [],
            },
            "methodology": "Punteggio relativo alle offerte confrontate.",
            "explanation": [raw_line],
        },
    )

    assert "Punteggio relativo alle offerte confrontate." not in answer
    assert raw_line not in answer
