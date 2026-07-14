from app.main import engine
from app.schemas.calculation_request import CalculationRequest


def test_legal_interest_period_inside_one_year():
    request = CalculationRequest(
        calculator_id="legal_it.legal_interest",
        inputs={"capital": 10000},
        period={"start_date": "2025-06-01", "end_date": "2025-09-30"},
    )
    result = engine.calculate(request)
    assert result.status == "success"
    assert len(result.steps) == 1
    assert result.result["interest"] == 66.85


def test_legal_interest_period_crossing_2025_and_2026_is_split():
    request = CalculationRequest(
        calculator_id="legal_it.legal_interest",
        inputs={"capital": 10000},
        period={"start_date": "2025-10-01", "end_date": "2026-03-31"},
    )
    result = engine.calculate(request)
    assert result.status == "success"
    assert len(result.steps) == 2
    assert result.steps[0]["rate"] == "0.02"
    assert result.steps[1]["rate"] == "0.016"
    # 92 days at 2%, then 90 days at 1.6% — segments must sum to the total
    segment_sum = round(float(result.steps[0]["interest"]) + float(result.steps[1]["interest"]), 2)
    assert segment_sum == result.result["interest"]


def test_legal_interest_requires_period():
    request = CalculationRequest(
        calculator_id="legal_it.legal_interest",
        inputs={"capital": 10000},
    )
    result = engine.calculate(request)
    assert result.status == "error"
    assert result.errors[0].code == "strategy_execution_failed"
    assert "period" in result.errors[0].message


def test_legal_interest_single_day_period():
    request = CalculationRequest(
        calculator_id="legal_it.legal_interest",
        inputs={"capital": 10000},
        period={"start_date": "2026-03-15", "end_date": "2026-03-15"},
    )
    result = engine.calculate(request)
    assert result.status == "success"
    assert result.steps[0]["days"] == 1
    # 10000 * 0.016 * 1/365
    assert result.result["interest"] == 0.44


def test_legal_interest_honors_caller_supplied_rate_override():
    # Regression test: caller_supplied_values must take priority over the
    # date-versioned parameter store, per the engine's documented
    # resolution order — this previously silently ignored the override.
    request = CalculationRequest(
        calculator_id="legal_it.legal_interest",
        inputs={"capital": 10000},
        period={"start_date": "2025-06-01", "end_date": "2025-09-30"},
        caller_supplied_values={"legal_interest_rate": 0.05},
    )
    result = engine.calculate(request)
    assert result.status == "success"
    assert len(result.steps) == 1
    assert result.steps[0]["rate"] == "0.05"
    # 122 days at 5% on 10000, vs 66.85 at the real 2% rate for the same period
    assert result.result["interest"] == 167.12
