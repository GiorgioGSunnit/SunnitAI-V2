"""Rational (never float) penal-duration arithmetic and formatting.

Durations are held as `Fraction` years end to end, because penal fractions
("aumento fino a un terzo") are exact rationals: 1/3 is not 0.3333, and two
sequential +1/3 on 24 years must yield exactly 42 years 8 months (128/3),
not a float approximation.

Conversion conventions (month = 12 per year, day = 30 per month) are the
customary computo conventions and are PROVISIONAL pending legal validation
(art. 14 c.p.) — see the design document's rounding question for the
lawyer. Only exact-fraction cases are exercised by tests until then.
"""

from fractions import Fraction
from math import floor
from typing import Union

MONTHS_PER_YEAR = 12
DAYS_PER_MONTH = 30  # provisional convention — confirm against art. 14 c.p.


def years(value: Union[int, str, Fraction]) -> Fraction:
    """Build an exact duration in years: years(24), years('1/3')."""
    return Fraction(value)


def format_duration_it(value_years: Fraction) -> str:
    """'128/3 years' -> '42 anni e 8 mesi'. Exact components only; a
    residual smaller than one day is dropped (provisional rounding)."""
    if value_years < 0:
        raise ValueError(f"a penal duration cannot be negative: {value_years}")

    whole_years = floor(value_years)
    remainder_months = (value_years - whole_years) * MONTHS_PER_YEAR
    whole_months = floor(remainder_months)
    remainder_days = (remainder_months - whole_months) * DAYS_PER_MONTH
    whole_days = floor(remainder_days + Fraction(1, 2))  # provisional half-up

    parts = []
    if whole_years:
        parts.append(f"{whole_years} anni" if whole_years != 1 else "1 anno")
    if whole_months:
        parts.append(f"{whole_months} mesi" if whole_months != 1 else "1 mese")
    if whole_days:
        parts.append(f"{whole_days} giorni" if whole_days != 1 else "1 giorno")
    if not parts:
        return "0 giorni"
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " e " + parts[-1]
