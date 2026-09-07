"""Penalty as a typed value: species + range, never a bare number.

The design document's central structural rule: `ergastolo` is a different
penalty *species*, not "30 years" — it has no numeric range, and only
explicit transformation rules (e.g. art. 65 c.p.) may convert it into
temporary reclusione. Temporary penalties carry an exact rational
min–max range in years.
"""

from dataclasses import dataclass
from fractions import Fraction
from typing import Optional

SPECIES = ("reclusione", "arresto", "multa", "ammenda", "ergastolo")


@dataclass(frozen=True)
class PenalRange:
    """An exact statutory envelope in years (min <= max, both >= 0)."""

    min_years: Fraction
    max_years: Fraction

    def __post_init__(self) -> None:
        if self.min_years < 0 or self.max_years < 0:
            raise ValueError("penal range bounds cannot be negative")
        if self.min_years > self.max_years:
            raise ValueError(
                f"penal range min ({self.min_years}) cannot exceed max ({self.max_years})"
            )


@dataclass(frozen=True)
class Penalty:
    species: str
    range: Optional[PenalRange] = None

    def __post_init__(self) -> None:
        if self.species not in SPECIES:
            raise ValueError(f"unknown penalty species: {self.species!r}")
        if self.is_life and self.range is not None:
            raise ValueError("ergastolo has no numeric range — it is a distinct species")
        if not self.is_life and self.range is None:
            raise ValueError(f"{self.species} requires a numeric range")

    @property
    def is_life(self) -> bool:
        return self.species == "ergastolo"
