"""Loads date-versioned parameter values from YAML into memory.

This is the swappable data layer the engine depends on — the scalability
requirement to move parameter storage to PostgreSQL later just means
providing another class with the same resolve_by_date /
resolve_by_tax_year / all_effective_ranges interface.
"""

from datetime import date
from pathlib import Path
from typing import Dict, List

import yaml

from ..schemas.parameter_value import ParameterValue


class ParameterStore:
    def __init__(self, parameters_dir: Path):
        self._values: Dict[str, List[ParameterValue]] = {}
        self._load(parameters_dir)

    def _load(self, parameters_dir: Path) -> None:
        for yml_path in sorted(parameters_dir.rglob("*.yml")):
            data = yaml.safe_load(yml_path.read_text(encoding="utf-8")) or {}
            for entry in data.get("values", []):
                pv = ParameterValue(**entry)
                self._values.setdefault(pv.parameter_id, []).append(pv)
        for entries in self._values.values():
            entries.sort(key=lambda pv: pv.effective_from)

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
