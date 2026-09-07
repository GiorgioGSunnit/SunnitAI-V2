"""Monthly parameter series (ISTAT FOI): resolution hit/miss, the
missing-month structured failure, and the revaluation coefficient."""

from datetime import date
from decimal import Decimal

import pytest

from support import parameter_store

PARAM = "legal_it.foi_index"


def test_resolve_monthly_returns_the_value_for_the_month_of_the_date():
    pv = parameter_store.resolve_monthly(PARAM, date(2024, 11, 17))
    assert pv.value == "100.0"
    assert pv.effective_from == date(2024, 11, 1)
    assert pv.effective_to == date(2024, 11, 30)
    # placeholder marks survive loading — a consumer can warn on them
    assert pv.placeholder is True
    assert pv.verified is False
    assert "PLACEHOLDER" in pv.source


def test_resolve_monthly_missing_month_fails_loudly_naming_param_and_month():
    with pytest.raises(KeyError) as exc_info:
        parameter_store.resolve_monthly(PARAM, date(2025, 6, 15))
    message = str(exc_info.value)
    assert PARAM in message
    assert "2025-06" in message


def test_resolve_monthly_unknown_series_fails():
    with pytest.raises(KeyError) as exc_info:
        parameter_store.resolve_monthly("legal_it.does_not_exist", date(2024, 11, 1))
    assert "legal_it.does_not_exist" in str(exc_info.value)


def test_monthly_pair_returns_both_values_and_full_precision_coefficient():
    pv_start, pv_end, coefficient = parameter_store.monthly_pair(
        PARAM, date(2024, 11, 5), date(2026, 2, 20)
    )
    assert pv_start.value == "100.0"
    assert pv_end.value == "102.5"
    # 102.5 / 100.0 exactly, computed in Decimal at full precision
    assert coefficient == Decimal("102.5") / Decimal("100.0")
    assert coefficient == Decimal("1.025")


def test_monthly_pair_propagates_missing_month():
    with pytest.raises(KeyError) as exc_info:
        parameter_store.monthly_pair(PARAM, date(2024, 11, 5), date(2026, 3, 1))
    assert "2026-03" in str(exc_info.value)
