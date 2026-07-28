"""Orchestrates a single calculation:

request -> load definition -> validate inputs -> run strategy -> build result.

This is the one place that knows the end-to-end flow; it delegates the
actual math to a CalculationStrategy and never contains formula logic
itself, so new calculators/strategies never require touching this file.
Every failure anywhere in that flow is caught here and turned into a
structured CalculationError — callers never see a raw Python exception.
"""

from datetime import date

from ..resolvers.date_parameter_resolver import describe_as_of_resolution
from ..resolvers.parameter_store import ParameterStore
from ..schemas.calculation_request import CalculationRequest
from ..schemas.calculation_result import CalculationResult
from ..schemas.citation import Citation
from ..schemas.error import CalculationError
from ..schemas.warning import Warning as CalcWarning
from ..strategies import STRATEGIES
from .errors import (
    CalculatorNotApplicableError,
    InputValidationError,
    PlatformError,
    StrategyExecutionError,
)
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
        # Bound to the definition as soon as one is loaded, so a failed
        # calculation still tells the caller what the calculator does not
        # cover. A validation error on an energy comparison is exactly when
        # someone needs to be told the model excludes VAT and system charges.
        known_exclusions: list = []

        def _error(err: PlatformError, raw_inputs=None, inputs_used=None) -> CalculationResult:
            return CalculationResult(
                request_id=request.request_id,
                calculator_id=request.calculator_id,
                status="error",
                raw_inputs=to_jsonable(raw_inputs) if raw_inputs else {},
                inputs_used=to_jsonable(inputs_used) if inputs_used else {},
                exclusions=list(known_exclusions),
                errors=[CalculationError(code=err.code, message=err.message, details=err.details)],
            )

        try:
            definition = self.registry.get(request.calculator_id)
        except PlatformError as e:
            return _error(e, raw_inputs=request.inputs)
        known_exclusions = list(definition.exclusions)

        try:
            self._ensure_applicable(definition, request)
        except PlatformError as e:
            return _error(e, raw_inputs=request.inputs)

        try:
            validated = validate_inputs(definition, request.inputs)
        except PlatformError as e:
            return _error(e, raw_inputs=request.inputs)

        try:
            self._ensure_period_present(definition, request)
        except PlatformError as e:
            return _error(e, raw_inputs=request.inputs, inputs_used=validated.values)

        # definition_validator already guarantees definition.strategy is a
        # known key at registry-load time, so a lookup miss here would be
        # an engine bug, not a request-time condition — let it raise.
        strategy_cls = STRATEGIES[definition.strategy]
        strategy = strategy_cls(self.parameter_store)
        strategy.validated_inputs = validated
        try:
            outcome = strategy.run(definition, validated.values, request)
        except PlatformError as e:
            return _error(e, raw_inputs=request.inputs, inputs_used=validated.values)
        except (ArithmeticError, ValueError) as e:
            # The module's contract is that callers never see a raw Python
            # exception, and a strategy is not obliged to anticipate every
            # arithmetic edge its declared formula can reach: a legal but
            # absurd input (a price of 1E+100) makes the final display
            # quantize raise decimal.InvalidOperation from inside otherwise
            # correct code. Structure it here rather than let it become a 500.
            return _error(
                StrategyExecutionError(
                    f"Il calcolo non e rappresentabile con i valori forniti: {e}",
                    details={"calculator_id": definition.id, "error_type": type(e).__name__},
                ),
                raw_inputs=request.inputs,
                inputs_used=validated.values,
            )

        citations = [Citation(**c) for c in definition.citations]
        # A draft calculator (version "*-draft") carries a machine-readable,
        # code-emitted caveat regardless of what its pack author wrote, so a
        # downstream renderer can gate on the CODE and never silently drop the
        # "not legally validated" banner. It leads the warning list on purpose.
        warnings = []
        if definition.version.endswith("-draft"):
            warnings.append(CalcWarning(
                code="draft_not_validated",
                message=(
                    "BOZZA NON VALIDATA LEGALMENTE: risultato dimostrativo, non "
                    "uno strumento professionale e non una previsione della "
                    "decisione. Da confermare con un professionista prima di "
                    "qualsiasi uso."
                ),
            ))
        warnings += [CalcWarning(code="definition", message=w) for w in definition.warnings]
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
            defaults_applied=to_jsonable(validated.defaults_applied),
            # Exclusions live on the definition but belong on every result:
            # a caller reading only the payload must see what the number
            # leaves out without going back to the formula pack.
            exclusions=list(definition.exclusions),
        )

    def _ensure_applicable(self, definition, request: CalculationRequest) -> None:
        as_of_info = describe_as_of_resolution(request)
        as_of = date.fromisoformat(as_of_info["as_of_date"])
        if definition.applicable_from and as_of < definition.applicable_from:
            raise self._not_applicable_error(definition, as_of_info)
        if definition.applicable_to and as_of > definition.applicable_to:
            raise self._not_applicable_error(definition, as_of_info)

    def _not_applicable_error(self, definition, as_of_info) -> CalculatorNotApplicableError:
        as_of_date = as_of_info["as_of_date"]
        source = as_of_info["source"]
        source_labels = {
            "explicit_as_of_date": "dalla as_of_date esplicita della richiesta",
            "derived_from_tax_year": "dal tax_year della richiesta",
            "defaulted_to_today": "dalla data odierna perche la richiesta non indica as_of_date o tax_year",
        }
        source_label = source_labels.get(source, f"dalla sorgente {source}")
        return CalculatorNotApplicableError(
            (
                f"Il calcolatore {definition.id!r} non e applicabile alla data {as_of_date}; "
                f"la data e stata determinata {source_label}."
            ),
            details={
                "applicable_from": definition.applicable_from.isoformat() if definition.applicable_from else None,
                "applicable_to": definition.applicable_to.isoformat() if definition.applicable_to else None,
                "as_of_date": as_of_date,
                "as_of_source": source,
            },
        )

    def _ensure_period_present(self, definition, request: CalculationRequest) -> None:
        if definition.requires_period and request.period is None:
            raise InputValidationError(
                f"Per calcolare {definition.name!r} serve il periodo di riferimento.",
                details={
                    "missing_inputs": ["period"],
                    "missing": [{
                        "name": "period",
                        "type": "period",
                        "required": True,
                        "description": "Periodo di riferimento del calcolo (request.period)",
                        "fields": [
                            {"name": "start_date", "type": "date", "required": True},
                            {"name": "end_date", "type": "date", "required": True},
                        ],
                    }],
                },
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
