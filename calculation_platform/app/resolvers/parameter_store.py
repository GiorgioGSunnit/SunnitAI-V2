"""Loads date-versioned parameter values from YAML into memory.

This is the swappable data layer the engine depends on — the scalability
requirement to move parameter storage to PostgreSQL later just means
providing another class with the same resolve_by_date /
resolve_by_tax_year / all_effective_ranges interface.
"""

import calendar
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

from ..schemas.parameter_value import ParameterValue


class ParameterStore:
    def __init__(self, parameters_dir: Path):
        self._values: Dict[str, List[ParameterValue]] = {}
        # Monthly series entries, keyed (parameter_id, year, month). A month
        # that is absent here is absent, full stop — resolution fails loudly,
        # it never interpolates or falls back to a neighboring month.
        self._monthly: Dict[Tuple[str, int, int], ParameterValue] = {}
        self._load(parameters_dir)

    def _load(self, parameters_dir: Path) -> None:
        for yml_path in sorted(parameters_dir.rglob("*.yml")):
            data = yaml.safe_load(yml_path.read_text(encoding="utf-8")) or {}
            for entry in data.get("values", []):
                if "year" in entry and "month" in entry:
                    pv = self._load_monthly_entry(entry, yml_path)
                else:
                    pv = ParameterValue(**entry)
                self._values.setdefault(pv.parameter_id, []).append(pv)
        for entries in self._values.values():
            entries.sort(key=lambda pv: pv.effective_from)

    def _load_monthly_entry(self, entry: dict, yml_path: Path) -> ParameterValue:
        """A monthly-series entry is keyed (year, month) with `value` and
        `source_ref`; it normalizes to a ParameterValue effective for exactly
        that calendar month so the date-based resolution APIs keep working."""
        entry = dict(entry)
        year = int(entry.pop("year"))
        month = int(entry.pop("month"))
        if not 1 <= month <= 12:
            raise ValueError(f"Invalid month {month} for {entry.get('parameter_id')!r} in {yml_path}")
        source_ref = entry.pop("source_ref", None)
        if source_ref and not entry.get("source"):
            entry["source"] = source_ref
        pv = ParameterValue(
            **entry,
            effective_from=date(year, month, 1),
            effective_to=date(year, month, calendar.monthrange(year, month)[1]),
        )
        key = (pv.parameter_id, year, month)
        if key in self._monthly:
            raise ValueError(f"Duplicate monthly value for {pv.parameter_id!r} {year}-{month:02d} in {yml_path}")
        self._monthly[key] = pv
        return pv

    def resolve_monthly(self, parameter_id: str, as_of: date) -> ParameterValue:
        """The monthly-series value for the calendar month of `as_of`.
        Raises KeyError for an unknown series or a missing month — callers
        convert that to the platform's structured error."""
        if not any(pid == parameter_id for pid, _, _ in self._monthly):
            raise KeyError(f"Unknown monthly parameter_id: {parameter_id!r}")
        pv = self._monthly.get((parameter_id, as_of.year, as_of.month))
        if pv is None:
            raise KeyError(
                f"No monthly value for {parameter_id!r} for month {as_of.year}-{as_of.month:02d}"
            )
        return pv

    def monthly_pair(
        self, parameter_id: str, start: date, end: date
    ) -> Tuple[ParameterValue, ParameterValue, Decimal]:
        """The monthly values for the months of `start` and `end`, plus their
        ratio end/start as a full-precision Decimal — the revaluation
        coefficient under the final-month convention (index of the month of
        each date; no quantization here, rounding is the caller's policy)."""
        pv_start = self.resolve_monthly(parameter_id, start)
        pv_end = self.resolve_monthly(parameter_id, end)
        coefficient = Decimal(str(pv_end.value)) / Decimal(str(pv_start.value))
        return pv_start, pv_end, coefficient

    def resolve_by_date(self, parameter_id: str, as_of: date) -> ParameterValue:
        entries = self._values.get(parameter_id)
        if not entries:
            raise KeyError(f"Unknown parameter_id: {parameter_id!r}")
        for pv in entries:
            if pv.effective_from <= as_of and (pv.effective_to is None or as_of <= pv.effective_to):
                return pv
        raise KeyError(f"No value for {parameter_id!r} effective on {as_of}")

    def resolve_by_tax_year(self, parameter_id: str, tax_year: int) -> ParameterValue:
        return self.resolve_by_date(parameter_id, date(tax_year, 12, 31))

    def all_effective_ranges(self, parameter_id: str, start: date, end: date) -> List[ParameterValue]:
        """Every ParameterValue whose effective range overlaps [start, end] —
        used by date_split_interest to split a period across a rate change."""
        entries = self._values.get(parameter_id)
        if not entries:
            raise KeyError(f"Unknown parameter_id: {parameter_id!r}")
        overlapping = []
        for pv in entries:
            pv_end = pv.effective_to or date.max
            if pv.effective_from <= end and pv_end >= start:
                overlapping.append(pv)
        return overlapping
