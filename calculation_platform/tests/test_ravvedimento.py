from decimal import Decimal

from app.main import engine
from app.schemas.calculation_request import CalculationRequest


def _calculate(**inputs):
    return engine.calculate(CalculationRequest(
        calculator_id="legal_it.ravvedimento_operoso", inputs=inputs,
    ))


def test_sprint_tier_is_per_day():
    # 10 days late: 0.0833%/day * 10 on 10,000 = 83.33
    result = _calculate(
        tributo_non_versato=10000,
        scadenza_originaria="2026-06-30",
        data_pagamento="2026-07-10",
    )
    assert result.status == "success"
    assert result.result["sanzione_ridotta"] == "83.33"
    assert result.derived_values["giorni_di_ritardo"] == 10


def test_tier_boundaries_14_15_30_31():
    def sanzione(days_late):
        from datetime import date, timedelta
        # Due date chosen so even the 366-day case stays inside the
        # 2024-2026 window covered by the legal interest rate table.
        due = date(2025, 6, 30)
        return _calculate(
            tributo_non_versato=10000,
            scadenza_originaria=due.isoformat(),
            data_pagamento=(due + timedelta(days=days_late)).isoformat(),
        ).result["sanzione_ridotta"]

    assert sanzione(14) == "116.67"   # last per-day day: 0.0833% * 14
    assert sanzione(15) == "125.00"   # flat 1/10 of 12.5%
    assert sanzione(30) == "125.00"
    assert sanzione(31) == "138.89"   # flat 1/9 of 12.5%
    assert sanzione(90) == "138.89"
    assert sanzione(91) == "312.50"   # flat 1/8 of 25%
    assert sanzione(365) == "312.50"
    assert sanzione(366) == "357.14"  # flat 1/7 of 25%


def test_interest_splits_across_year_end_rate_change():
    result = _calculate(
        tributo_non_versato=8000,
        scadenza_originaria="2025-11-30",
        data_pagamento="2026-02-28",
    )
    assert result.result["interessi"] == "34.28"
    segments = [s for s in result.steps if s.get("type") == "interest_segment"]
    assert len(segments) == 2
    assert segments[0]["rate"] == "0.02"
    assert segments[1]["rate"] == "0.016"


def test_total_is_sum_of_rounded_components():
    result = _calculate(
        tributo_non_versato=10000,
        scadenza_originaria="2026-06-30",
        data_pagamento="2026-07-10",
    )
    r = result.result
    component_sum = Decimal(r["tributo"]) + Decimal(r["sanzione_ridotta"]) + Decimal(r["interessi"])
    assert Decimal(r["totale_da_versare"]) == component_sum.quantize(Decimal("0.01"))


def test_payment_on_or_before_due_date_is_an_error():
    result = _calculate(
        tributo_non_versato=10000,
        scadenza_originaria="2026-06-30",
        data_pagamento="2026-06-30",
    )
    assert result.status == "error"


def test_violation_before_2024_reform_is_not_covered():
    result = _calculate(
        tributo_non_versato=10000,
        scadenza_originaria="2024-06-30",
        data_pagamento="2026-07-10",
    )
    assert result.status == "error"
    assert "regime" in result.errors[0].message or "not covered" in result.errors[0].message


def test_interest_period_beyond_rate_table_coverage_is_an_error():
    # The rate table currently ends at 2026-12-31: paying in 2027 must
    # refuse rather than silently undercount the 2027 interest days.
    result = _calculate(
        tributo_non_versato=10000,
        scadenza_originaria="2026-12-15",
        data_pagamento="2027-01-15",
    )
    assert result.status == "error"
    assert "cover" in result.errors[0].message


def test_2024_post_reform_violation_is_covered():
    # 20 days late in Oct 2024: per-day tier at the 2024 legal rate (2.5%).
    result = _calculate(
        tributo_non_versato=10000,
        scadenza_originaria="2024-10-01",
        data_pagamento="2024-10-15",
    )
    assert result.status == "success"
    assert result.result["sanzione_ridotta"] == "116.67"   # 0.0833% * 14 days
    assert result.result["interessi"] == "9.59"            # 10000 * 0.025 * 14/365


def test_declaration_deadline_bounds_the_one_eighth_tier():
    common = dict(
        tributo_non_versato=10000,
        scadenza_originaria="2026-01-31",
        data_pagamento="2026-08-31",   # 212 days late
    )
    # Paid before the declaration deadline: 1/8 of 25%.
    within = _calculate(**common, termine_dichiarazione="2026-10-31")
    assert within.result["sanzione_ridotta"] == "312.50"
    # Paid after it: 1/7 of 25%, even though the delay is under a year.
    beyond = _calculate(**common, termine_dichiarazione="2026-07-31")
    assert beyond.result["sanzione_ridotta"] == "357.14"


def test_missing_declaration_deadline_falls_back_with_assumption():
    result = _calculate(
        tributo_non_versato=10000,
        scadenza_originaria="2026-01-31",
        data_pagamento="2026-08-31",
    )
    assert result.result["sanzione_ridotta"] == "312.50"
    assert any("termine_dichiarazione" in a.message for a in result.assumptions)


def test_violation_already_assessed_warns():
    result = _calculate(
        tributo_non_versato=10000,
        scadenza_originaria="2026-06-30",
        data_pagamento="2026-07-10",
        violazione_gia_constatata=True,
    )
    assert result.status == "success"
    assert any("precluso" in w.message for w in result.warnings)
