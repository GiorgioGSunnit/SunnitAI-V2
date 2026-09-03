"""Resolves every ParameterRef a CalculatorDefinition declares, in priority order:

1. request.caller_supplied_values — an explicit override from the caller.
2. The parameter store, resolved by request.as_of_date (or request.tax_year,
   or today if neither is given).
3. A static default declared directly on the ParameterRef in the YAML.

Returns a ParameterResolution carrying both the plain values (for a
strategy to actually compute with) and the full provenance of each one
(for CalculationResult.parameters_used / audit).
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, Optional

from ..core.errors import ParameterResolutionError
from ..schemas.resolved_parameter import ResolvedParameter
from .static_resolver import resolve_static


def _coerce(value: Any) -> Any:
    if isinstance(value, (list, dict, bool, str)):
        return value
    from decimal import Decimal
    return Decimal(str(value))


def describe_as_of_resolution(request) -> Dict[str, Any]:
    """How the as-of date used for parameter lookups was determined —
    surfaced on CalculationResult.date_resolution."""
    if request.as_of_date:
        return {
            "as_of_date": request.as_of_date.isoformat(),
            "source": "explicit_as_of_date",
            "tax_year": request.tax_year,
        }
    if request.tax_year:
        as_of = date(request.tax_year, 12, 31)
        return {"as_of_date": as_of.isoformat(), "source": "derived_from_tax_year", "tax_year": request.tax_year}
    return {"as_of_date": date.today().isoformat(), "source": "defaulted_to_today", "tax_year": None}


@dataclass
class ParameterResolution:
    values: Dict[str, Any] = field(default_factory=dict)
    resolved: Dict[str, ResolvedParameter] = field(default_factory=dict)
    date_resolution: Optional[Dict[str, Any]] = None

    def parameters_used(self) -> Dict[str, Any]:
        return {name: rp.model_dump() for name, rp in self.resolved.items()}


def resolve_parameters(definition, parameter_store, request) -> ParameterResolution:
    resolution = ParameterResolution()
    as_of_info = describe_as_of_resolution(request)
    as_of = date.fromisoformat(as_of_info["as_of_date"])

    used_parameter_store = False
    for ref in definition.parameters:
        if ref.name in request.caller_supplied_values:
            value = _coerce(request.caller_supplied_values[ref.name])
            resolution.values[ref.name] = value
            resolution.resolved[ref.name] = ResolvedParameter(
                name=ref.name, value=_jsonable(value), origin="caller_supplied",
                parameter_id=ref.parameter_id,
            )
            continue

        if ref.parameter_id:
            try:
                pv = parameter_store.resolve_by_date(ref.parameter_id, as_of)
            except KeyError as e:
                raise ParameterResolutionError(
                    str(e), details={"parameter": ref.name, "parameter_id": ref.parameter_id, "as_of_date": as_of.isoformat()}
                ) from e
            used_parameter_store = True
            value = _coerce(pv.value)
            resolution.values[ref.name] = value
            resolution.resolved[ref.name] = ResolvedParameter(
                name=ref.name, value=_jsonable(value), origin="parameter_store",
                parameter_id=ref.parameter_id, source=pv.source,
                effective_from=pv.effective_from.isoformat(),
                effective_to=pv.effective_to.isoformat() if pv.effective_to else None,
                official=pv.official, last_verified_at=pv.last_verified_at,
                citations=pv.citations,
            )
            continue

        if ref.default is not None:
            value = resolve_static(ref.default)
            resolution.values[ref.name] = value
            resolution.resolved[ref.name] = ResolvedParameter(
                name=ref.name, value=_jsonable(value), origin="static_default",
            )
            continue

        raise ParameterResolutionError(
            f"Cannot resolve parameter {ref.name!r}: no caller value, parameter_id, or default",
            details={"parameter": ref.name},
        )

    if used_parameter_store:
        resolution.date_resolution = as_of_info
    return resolution


def _jsonable(value: Any) -> Any:
    from ..core.result_builder import to_jsonable
    return to_jsonable(value)
