"""Mechanical rule operations for penal computation.

Implements the arithmetic *structure* of the sentencing rules from the
design document — the parts that are deterministic regardless of legal
validation: fractional envelopes, sequential application (art. 63 c.p.),
caps applied after the sequence (art. 66 c.p.), art. 69 balancing as a
scenario selector (never computed), the art. 65 ergastolo transformation,
and the rito abbreviato gate (art. 438 c.p.p.) + reduction (art. 442
c.p.p.).

This module is NOT yet wired to any calculator or strategy: the legal
content (which circumstances exist, their effects, expected outcomes)
is gated on lawyer validation per the design document. Everything here
is exercised only by its own test suite, anchored to the worked numeric
examples in that document.

All arithmetic is exact (`Fraction`); floats never enter or leave.
"""

from fractions import Fraction
from typing import Dict, List, Tuple

from ..core.errors import PlatformError
from .penalty import PenalRange, Penalty


class ComputationBlockedError(PlatformError):
    """Raised when the computation must stop and demand a legal decision
    instead of producing a number ("Calcolo bloccato: necessaria conferma
    legale") — e.g. rito abbreviato requested for a life-punishable case."""

    code = "computation_blocked"


_RECLUSIONE_ABSOLUTE_CAP_YEARS = Fraction(30)  # art. 66, n. 2 c.p.
_ART66_MULTIPLIER = 3  # art. 66, comma 1: never more than triple the statutory max

BALANCING_SCENARIOS = (
    "aggravating_prevalent",
    "mitigating_prevalent",
    "equivalent",
    "no_balancing_applicable",
)


def _check_fraction(fraction: Fraction) -> None:
    if not isinstance(fraction, Fraction):
        raise TypeError(f"penal fractions must be Fraction, got {type(fraction).__name__}")
    if fraction < 0 or fraction > 1:
        raise ValueError(f"a penal fraction must be within [0, 1], got {fraction}")


def apply_fraction_envelope(
    rng: PenalRange, operation: str, min_fraction: Fraction, max_fraction: Fraction
) -> PenalRange:
    """The statutory envelope of one discretionary circumstance
    ("aumento/diminuzione fino a X"): the widest range the law permits.
    An increase leaves the minimum at min_fraction (usually 0) and pushes
    the maximum by max_fraction; a decrease mirrors that downward."""
    _check_fraction(min_fraction)
    _check_fraction(max_fraction)
    if min_fraction > max_fraction:
        raise ValueError("min_fraction cannot exceed max_fraction")

    if operation == "increase":
        return PenalRange(
            min_years=rng.min_years * (1 + min_fraction),
            max_years=rng.max_years * (1 + max_fraction),
        )
    if operation == "decrease":
        return PenalRange(
            min_years=rng.min_years * (1 - max_fraction),
            max_years=rng.max_years * (1 - min_fraction),
        )
    raise ValueError(f"unknown operation: {operation!r}")


def apply_pinned_fractions_sequentially(
    rng: PenalRange, steps: List[Tuple[str, Fraction]]
) -> PenalRange:
    """Art. 63 c.p.: with several circumstances at pinned fractions, each
    increase/decrease operates on the RESULT of the previous one — never
    additively on the original base."""
    current = rng
    for operation, fraction in steps:
        _check_fraction(fraction)
        if operation == "increase":
            factor = 1 + fraction
        elif operation == "decrease":
            factor = 1 - fraction
        else:
            raise ValueError(f"unknown operation: {operation!r}")
        current = PenalRange(
            min_years=current.min_years * factor,
            max_years=current.max_years * factor,
        )
    return current


def apply_reclusione_caps(rng: PenalRange, statutory_max_years: Fraction) -> PenalRange:
    """Art. 66 c.p., applied AFTER the sequential computation: the result
    of multiple aggravating increases cannot exceed the triple of the
    statutory maximum, and in any case 30 years for reclusione."""
    cap = min(_ART66_MULTIPLIER * statutory_max_years, _RECLUSIONE_ABSOLUTE_CAP_YEARS)
    return PenalRange(
        min_years=min(rng.min_years, cap),
        max_years=min(rng.max_years, cap),
    )


def select_balancing(
    aggravating: List[str], mitigating: List[str], scenario: str
) -> Tuple[List[str], List[str]]:
    """Art. 69 c.p. as a scenario SELECTOR (the judgment itself is a user
    input, never computed): returns which circumstance sets remain
    applicable under the declared scenario.

    - aggravating_prevalent: mitigating circumstances are not applied
    - mitigating_prevalent: aggravating circumstances are not applied
    - equivalent: neither set is applied (base penalty stands)
    - no_balancing_applicable: both pass through (used when only one
      side is present and no balancing arises)
    """
    if scenario not in BALANCING_SCENARIOS:
        raise ValueError(
            f"unknown balancing scenario {scenario!r}; valid: {', '.join(BALANCING_SCENARIOS)}"
        )
    if scenario == "no_balancing_applicable" and aggravating and mitigating:
        raise ComputationBlockedError(
            "Both aggravating and mitigating circumstances are present: an "
            "art. 69 balancing scenario must be selected explicitly.",
            details={"aggravating": aggravating, "mitigating": mitigating},
        )
    if scenario == "aggravating_prevalent":
        return aggravating, []
    if scenario == "mitigating_prevalent":
        return [], mitigating
    if scenario == "equivalent":
        return [], []
    return aggravating, mitigating


def balancing_scenario_matrix(
    aggravating: List[str], mitigating: List[str]
) -> Dict[str, Tuple[List[str], List[str]]]:
    """All three art. 69 outcomes side by side — the 'show all scenarios'
    mode from the design document."""
    return {
        scenario: select_balancing(aggravating, mitigating, scenario)
        for scenario in ("aggravating_prevalent", "mitigating_prevalent", "equivalent")
    }


def transform_ergastolo_for_ordinary_mitigating() -> Penalty:
    """Art. 65, n. 2 c.p.: when an ordinary mitigating circumstance applies
    to ergastolo and the law does not determine the decrease, the penalty
    becomes reclusione from 20 to 24 years."""
    return Penalty(
        species="reclusione",
        range=PenalRange(min_years=Fraction(20), max_years=Fraction(24)),
    )


def apply_abbreviato(penalty: Penalty, offence_kind: str = "delitto") -> Penalty:
    """Rito abbreviato: art. 438 c.p.p. excludes it for offences punished
    with ergastolo (a procedural GATE — the computation blocks rather than
    producing a number); art. 442 c.p.p. reduces the determined penalty by
    1/3 for delitti and 1/2 for contravvenzioni, AFTER circumstances."""
    if penalty.is_life:
        raise ComputationBlockedError(
            "Giudizio abbreviato is excluded for offences punished with "
            "ergastolo (art. 438 c.p.p.): no reduction can be computed.",
            details={"norm": "art. 438 c.p.p."},
        )
    if offence_kind == "delitto":
        factor = Fraction(2, 3)
    elif offence_kind == "contravvenzione":
        factor = Fraction(1, 2)
    else:
        raise ValueError(f"unknown offence kind: {offence_kind!r}")
    return Penalty(
        species=penalty.species,
        range=PenalRange(
            min_years=penalty.range.min_years * factor,
            max_years=penalty.range.max_years * factor,
        ),
    )
