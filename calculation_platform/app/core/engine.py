"""Orchestrates a single calculation:

request -> load definition -> validate inputs -> run strategy -> build result.

This is the one place that knows the end-to-end flow; it delegates the
actual math to a CalculationStrategy and never contains formula logic
itself, so new calculators/strategies never require touching this file.
Every failure anywhere in that flow is caught here and turned into a
structured CalculationError — callers never see a raw Python exception.
"""

from datetime import date

from ..resolvers.parameter_store import ParameterStore
from ..schemas.calculation_request import CalculationRequest
from ..schemas.calculation_result import CalculationResult
from ..schemas.citation import Citation
from ..schemas.error import CalculationError
from ..schemas.warning import Warning as CalcWarning
from ..strategies import STRATEGIES
from .errors import PlatformError
from .registry import CalculatorRegistry
from .result_builder import to_jsonable
from .validators import validate_inputs


class CalculationEngine:
    def __init__(
        self,
        registry: CalculatorRegistry,
        parameter_store: ParameterStore,
        parameter_verification_stale_after_days: int = 365,
    ):
        self.registry = registry
        self.parameter_store = parameter_store
        self.parameter_verification_stale_after_days = parameter_verification_stale_after_days

    def calculate(self, request: CalculationRequest) -> CalculationResult:
        def _error(err: PlatformError, raw_inputs=None, inputs_used=None) -> CalculationResult:
            return CalculationResult(
                request_id=request.request_id,
                calculator_id=request.calculator_id,
                status="error",
                raw_inputs=to_jsonable(raw_inputs) if raw_inputs else {},
                inputs_used=to_jsonable(inputs_used) if inputs_used else {},
                errors=[CalculationError(code=err.code, message=err.message, details=err.details)],
            )

        try:
            definition = self.registry.get(request.calculator_id)
        except PlatformError as e:
            return _error(e, raw_inputs=request.inputs)

        try:
            validated = validate_inputs(definition, request.inputs)
        except PlatformError as e:
            return _error(e, raw_inputs=request.inputs)

        # definition_validator already guarantees definition.strategy is a
        # known key at registry-load time, so a lookup miss here would be
        # an engine bug, not a request-time condition — let it raise.
        strategy_cls = STRATEGIES[definition.strategy]
        strategy = strategy_cls(self.parameter_store)
        try:
            outcome = strategy.run(definition, validated.values, request)
        except PlatformError as e:
            return _error(e, raw_inputs=request.inputs, inputs_used=validated.values)

        citations = [Citation(**c) for c in definition.citations]
        warnings = [CalcWarning(code="definition", message=w) for w in definition.warnings]
        warnings += [CalcWarning(code="calculation", message=w) for w in outcome.warnings]
        warnings += self._parameter_verification_warnings(outcome.parameters_used)

        assumptions = [CalcWarning(code="definition", message=a) for a in definition.assumptions]
        assumptions += [CalcWarning(code="input_default", message=a) for a in validated.assumptions]
        assumptions += [CalcWarning(code="calculation", message=a) for a in outcome.assumptions]

        return CalculationResult(
            request_id=request.request_id,
            calculator_id=request.calculator_id,
            status="success",
            result=to_jsonable(outcome.result),
            formula_used=definition.id,
            formula_version=definition.version,
            raw_inputs=to_jsonable(request.inputs),
            inputs_used=to_jsonable(validated.values),
            parameters_used=to_jsonable(outcome.parameters_used),
            date_resolution=outcome.date_resolution,
            derived_values=to_jsonable(outcome.derived_values),
            steps=outcome.steps,
            citations=citations,
            warnings=warnings,
            assumptions=assumptions,
        )

    def _parameter_verification_warnings(self, parameters_used):
        warnings = []
        today = date.today()
        for name, raw in parameters_used.items():
            if raw.get("origin") != "parameter_store" or not raw.get("official"):
                continue
            parameter_id = raw.get("parameter_id") or name
            last_verified_at = raw.get("last_verified_at")
            if not last_verified_at:
                warnings.append(CalcWarning(
                    code="parameter_verification_missing",
                    message=(
                        f"Il parametro ufficiale '{parameter_id}' non ha una verifica automatica "
                        "registrata; ricontrollare la fonte prima dell'uso operativo."
                    ),
                ))
                continue

            try:
                verified_on = date.fromisoformat(last_verified_at[:10])
            except ValueError:
                warnings.append(CalcWarning(
                    code="parameter_verification_invalid",
                    message=(
                        f"Il parametro ufficiale '{parameter_id}' ha una data di verifica non valida "
                        f"('{last_verified_at}'); ricontrollare la fonte."
                    ),
                ))
                continue

            age_days = (today - verified_on).days
            if age_days > self.parameter_verification_stale_after_days:
                warnings.append(CalcWarning(
                    code="parameter_verification_stale",
                    message=(
                        f"Il parametro ufficiale '{parameter_id}' e stato verificato l'ultima volta "
                        f"il {verified_on.isoformat()}; ricontrollare la fonte prima dell'uso operativo."
                    ),
                ))
        return warnings
