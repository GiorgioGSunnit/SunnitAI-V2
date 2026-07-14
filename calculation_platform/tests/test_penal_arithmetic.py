"""Tests for the penal arithmetic core (app/penal/).

Every numeric expectation here comes from the worked examples in the
design document (Calcolo_Pena_Documento_di_Progettazione.md), i.e. the
*structural* arithmetic of artt. 63, 65, 66, 69 c.p. and 438/442 c.p.p.
No legal catalog content is exercised — that part is gated on lawyer
validation. All arithmetic must be exact rationals: any float leaking in
is a bug.
"""

from fractions import Fraction

import pytest

from app.penal.duration import format_duration_it, years
from app.penal.operations import (
    ComputationBlockedError,
    apply_abbreviato,
    apply_fraction_envelope,
    apply_pinned_fractions_sequentially,
    apply_reclusione_caps,
    balancing_scenario_matrix,
    select_balancing,
    transform_ergastolo_for_ordinary_mitigating,
)
from app.penal.penalty import PenalRange, Penalty

THIRD = Fraction(1, 3)
BASE_575 = PenalRange(min_years=years(21), max_years=years(24))  # art. 575 + art. 23


# ---------------------------------------------------------------------------
# Art. 63 — sequential application (the document's central worked example)
# ---------------------------------------------------------------------------

def test_two_full_thirds_are_sequential_not_additive():
    # 24 years -> +1/3 = 32 -> +1/3 of 32 = 42 years 8 months (128/3)
    result = apply_pinned_fractions_sequentially(
        PenalRange(years(24), years(24)), [("increase", THIRD), ("increase", THIRD)]
    )
    assert result.max_years == Fraction(128, 3)
    # linear addition would give 24 * (1 + 2/3) = 40 — must NOT be that
    assert result.max_years != years(40)


def test_sequential_result_formats_as_years_and_months():
    assert format_duration_it(Fraction(128, 3)) == "42 anni e 8 mesi"


def test_sequential_decreases_also_compound():
    # 24 -> -1/3 = 16 -> -1/3 of 16 = 32/3 = 10 years 8 months
    result = apply_pinned_fractions_sequentially(
        PenalRange(years(24), years(24)), [("decrease", THIRD), ("decrease", THIRD)]
    )
    assert result.max_years == Fraction(32, 3)
    assert format_duration_it(result.max_years) == "10 anni e 8 mesi"


def test_arithmetic_stays_exact_rational():
    result = apply_pinned_fractions_sequentially(
        PenalRange(years(24), years(24)), [("increase", THIRD), ("increase", THIRD)]
    )
    assert isinstance(result.max_years, Fraction)


# ---------------------------------------------------------------------------
# Discretionary envelopes ("fino a un terzo" is a maximum, not a value)
# ---------------------------------------------------------------------------

def test_aggravating_envelope_widens_upward_only():
    # one common aggravating on 21-24: min stays 21 (fraction may be 0),
    # max becomes 24 * 4/3 = 32
    result = apply_fraction_envelope(BASE_575, "increase", Fraction(0), THIRD)
    assert result.min_years == years(21)
    assert result.max_years == years(32)


def test_mitigating_envelope_matches_document_example():
    # the document's JSON example: 0..1/3 decrease on 21-24 -> "14 to 24 years"
    result = apply_fraction_envelope(BASE_575, "decrease", Fraction(0), THIRD)
    assert result.min_years == years(14)
    assert result.max_years == years(24)


def test_fractions_outside_legal_bounds_are_rejected():
    with pytest.raises(ValueError):
        apply_fraction_envelope(BASE_575, "increase", Fraction(0), Fraction(3, 2))
    with pytest.raises(TypeError):
        apply_fraction_envelope(BASE_575, "increase", 0.0, 0.33)  # floats banned


# ---------------------------------------------------------------------------
# Art. 66 — caps AFTER the sequence
# ---------------------------------------------------------------------------

def test_reclusione_cap_of_30_years_applies_after_sequence():
    # 21-24 with two pinned +1/3: 28 -> 37+1/3 (min), 32 -> 42+2/3 (max)
    increased = apply_pinned_fractions_sequentially(
        BASE_575, [("increase", THIRD), ("increase", THIRD)]
    )
    assert increased.max_years == Fraction(128, 3)  # would exceed 30
    capped = apply_reclusione_caps(increased, statutory_max_years=years(24))
    assert capped.min_years == years(30)
    assert capped.max_years == years(30)


def test_cap_after_differs_from_cap_before():
    # capping first and then increasing would give 30 * 4/3 = 40 — proving
    # the order (sequence, THEN cap) is load-bearing, not cosmetic
    capped_first = apply_reclusione_caps(
        PenalRange(years(24), years(24)), statutory_max_years=years(24)
    )
    wrong_order = apply_pinned_fractions_sequentially(capped_first, [("increase", THIRD)])
    assert wrong_order.max_years == years(32)  # > 30: the cap would be violated


def test_triple_of_statutory_max_cap_binds_when_lower_than_30():
    # a hypothetical offence with statutory max 8: triple = 24 < 30
    rng = PenalRange(years(20), years(28))
    capped = apply_reclusione_caps(rng, statutory_max_years=years(8))
    assert capped.max_years == years(24)


def test_cap_leaves_lower_results_untouched():
    capped = apply_reclusione_caps(BASE_575, statutory_max_years=years(24))
    assert capped == BASE_575


# ---------------------------------------------------------------------------
# Art. 69 — balancing is a scenario selector, never a computation
# ---------------------------------------------------------------------------

def test_aggravating_prevalent_drops_mitigating():
    applied = select_balancing(["cp_61_1"], ["cp_62bis"], "aggravating_prevalent")
    assert applied == (["cp_61_1"], [])


def test_mitigating_prevalent_drops_aggravating():
    applied = select_balancing(["cp_61_1"], ["cp_62bis"], "mitigating_prevalent")
    assert applied == ([], ["cp_62bis"])


def test_equivalence_drops_both_leaving_base_penalty():
    applied = select_balancing(["cp_61_1"], ["cp_62bis"], "equivalent")
    assert applied == ([], [])


def test_missing_scenario_with_both_sides_blocks_instead_of_guessing():
    with pytest.raises(ComputationBlockedError) as exc_info:
        select_balancing(["cp_61_1"], ["cp_62bis"], "no_balancing_applicable")
    assert exc_info.value.code == "computation_blocked"


def test_scenario_matrix_returns_all_three_outcomes():
    matrix = balancing_scenario_matrix(["cp_61_1"], ["cp_62bis"])
    assert set(matrix) == {"aggravating_prevalent", "mitigating_prevalent", "equivalent"}
    assert matrix["equivalent"] == ([], [])


# ---------------------------------------------------------------------------
# Ergastolo — a species, not a number
# ---------------------------------------------------------------------------

def test_ergastolo_cannot_carry_a_numeric_range():
    with pytest.raises(ValueError):
        Penalty(species="ergastolo", range=BASE_575)


def test_temporary_penalty_requires_a_range():
    with pytest.raises(ValueError):
        Penalty(species="reclusione")


def test_art65_transforms_ergastolo_into_reclusione_20_24():
    result = transform_ergastolo_for_ordinary_mitigating()
    assert result.species == "reclusione"
    assert result.range == PenalRange(years(20), years(24))


# ---------------------------------------------------------------------------
# Rito abbreviato — procedural gate, then reduction
# ---------------------------------------------------------------------------

def test_abbreviato_reduces_delitto_penalty_by_one_third():
    penalty = Penalty(species="reclusione", range=BASE_575)
    reduced = apply_abbreviato(penalty, offence_kind="delitto")
    assert reduced.range == PenalRange(years(14), years(16))  # 21*2/3, 24*2/3


def test_abbreviato_is_blocked_for_life_punishable_cases():
    with pytest.raises(ComputationBlockedError) as exc_info:
        apply_abbreviato(Penalty(species="ergastolo"))
    assert "438" in exc_info.value.message


# ---------------------------------------------------------------------------
# Duration formatting and range validation
# ---------------------------------------------------------------------------

def test_duration_formatting_singulars_and_units():
    assert format_duration_it(years(21)) == "21 anni"
    assert format_duration_it(years(1)) == "1 anno"
    assert format_duration_it(Fraction(1, 12)) == "1 mese"
    assert format_duration_it(Fraction(1, 18)) == "20 giorni"  # 2/3 of a 30-day month


def test_negative_durations_and_inverted_ranges_are_rejected():
    with pytest.raises(ValueError):
        format_duration_it(years(-1))
    with pytest.raises(ValueError):
        PenalRange(min_years=years(24), max_years=years(21))
