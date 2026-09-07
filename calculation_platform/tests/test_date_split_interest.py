from datetime import date

import pytest

from app.resolvers.parameter_store import ParameterStore
from pathlib import Path

PARAMETERS_DIR = Path(__file__).resolve().parent.parent / "parameters"


@pytest.fixture
def parameter_store():
    return ParameterStore(PARAMETERS_DIR)


def test_all_effective_ranges_returns_single_segment_within_one_year(parameter_store):
    segments = parameter_store.all_effective_ranges(
        "legal_it.legal_interest_rate", date(2025, 6, 1), date(2025, 9, 30)
    )
    assert len(segments) == 1
    assert segments[0].value == 0.02


def test_all_effective_ranges_splits_across_a_rate_change(parameter_store):
    segments = parameter_store.all_effective_ranges(
        "legal_it.legal_interest_rate", date(2025, 10, 1), date(2026, 3, 31)
    )
    assert len(segments) == 2
    values = sorted(s.value for s in segments)
    assert values == [0.016, 0.02]


def test_resolve_by_date_picks_the_correct_year(parameter_store):
    pv_2025 = parameter_store.resolve_by_date("legal_it.legal_interest_rate", date(2025, 12, 31))
    pv_2026 = parameter_store.resolve_by_date("legal_it.legal_interest_rate", date(2026, 1, 1))
    assert pv_2025.value == 0.02
    assert pv_2026.value == 0.016


def test_resolve_by_date_raises_for_unknown_parameter(parameter_store):
    with pytest.raises(KeyError):
        parameter_store.resolve_by_date("does.not.exist", date(2026, 1, 1))
