from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

from ..core.engine import CalculationEngine
from ..core.errors import CalculatorNotFoundError
from ..core.matcher import match_query
from ..schemas.calculation_request import CalculationRequest
from ..schemas.calculation_result import CalculationResult
from ..schemas.calculator_definition import CalculatorDefinition
from ..schemas.match_result import MatchRequest, MatchResponse
from ..schemas.stored_calculation import StoredCalculation, StoredCalculationSummary
from ..schemas.warning import Warning as CalcWarning
from ..storage.base import CalculationRecord, CalculationStore
from ..reporting import render_report_html
from .tool_schemas import build_all_tool_schemas, build_tool_schema

router = APIRouter()
_engine: Optional[CalculationEngine] = None
_store: Optional[CalculationStore] = None


def set_engine(engine: CalculationEngine) -> None:
    global _engine
    _engine = engine


def set_store(store: CalculationStore) -> None:
    global _store
    _store = store


def calculate_and_persist(request: CalculationRequest) -> CalculationResult:
    if request.request_id is None:
        request = request.model_copy(update={"request_id": uuid4().hex})
    result = _engine.calculate(request)
    if result.request_id is None:
        result.request_id = request.request_id

    try:
        _store.save(CalculationRecord.from_models(request, result))
    except Exception as exc:
        result.warnings.append(CalcWarning(
            code="persistence_failed",
            message=f"Il risultato e stato calcolato ma non salvato: {exc}",
        ))
    return result


@router.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@router.get("/calculators")
def list_calculators() -> List[Dict[str, Any]]:
    return _engine.registry.list_all()


@router.get("/tool-schemas")
def list_tool_schemas() -> List[Dict[str, Any]]:
    return build_all_tool_schemas(_engine.registry)


@router.get("/calculators/{calculator_id}/tool-schema")
def get_tool_schema(calculator_id: str) -> Dict[str, Any]:
    try:
        return build_tool_schema(_engine.registry.get(calculator_id))
    except CalculatorNotFoundError as e:
        raise HTTPException(status_code=404, detail=e.message)


@router.get("/calculators/{calculator_id}", response_model=CalculatorDefinition)
def get_calculator(calculator_id: str) -> CalculatorDefinition:
    """Full definition (inputs, parameters, strategy) — used by the dev UI
    to render the right form fields for whichever calculator is selected."""
    try:
        return _engine.registry.get(calculator_id)
    except CalculatorNotFoundError as e:
        raise HTTPException(status_code=404, detail=e.message)


@router.post("/calculate", response_model=CalculationResult)
def calculate(request: CalculationRequest) -> CalculationResult:
    return calculate_and_persist(request)


def _readable_or_404(request_id: str) -> StoredCalculation:
    """Fetch a stored calculation, refusing anything whose calculator is not
    released.

    Gating computation while leaving stored results readable protects
    nobody — the disclosure is the harm. Enforced at READ time rather than
    by migrating or deleting records: no stored data is lost, and turning
    the override on makes the history visible again.

    A withheld record is refused exactly like an absent one, deliberately:
    distinguishing them would confirm to a caller that a criminal-sentencing
    calculation exists for a given request_id.
    """
    stored = _store.get(request_id)
    if stored is None or not _engine.registry.is_disclosable(stored.calculator_id):
        raise HTTPException(status_code=404, detail="Calculation not found")
    return stored


@router.get("/calculations", response_model=List[StoredCalculationSummary])
def list_calculations(limit: int = 50) -> List[StoredCalculationSummary]:
    # Filtered after the fetch, so `limit` caps what is read, not what is
    # returned: a page may come back shorter when it spans withheld records.
    return [
        summary
        for summary in _store.list_recent(limit=min(limit, 200))
        if _engine.registry.is_disclosable(summary.calculator_id)
    ]


@router.get("/calculations/{request_id}", response_model=StoredCalculation)
def get_calculation(request_id: str) -> StoredCalculation:
    return _readable_or_404(request_id)


@router.get("/calculations/{request_id}/report", response_class=HTMLResponse)
def get_calculation_report(request_id: str) -> HTMLResponse:
    stored = _readable_or_404(request_id)

    definition = None
    try:
        definition = _engine.registry.get(stored.calculator_id)
    except CalculatorNotFoundError:
        definition = None
    return HTMLResponse(render_report_html(stored, definition))


@router.post("/calculations/{request_id}/replay")
def replay_calculation(request_id: str) -> Dict[str, Any]:
    stored = _readable_or_404(request_id)

    request = CalculationRequest.model_validate(stored.request)
    replayed = _engine.calculate(request)
    replayed_result = replayed.model_dump(mode="json")
    return {
        "request_id": request_id,
        "stored_result": stored.result,
        "replayed_result": replayed_result,
        "matches": _results_match(stored.result, replayed_result),
    }


def _results_match(stored_result: Dict[str, Any], replayed_result: Dict[str, Any]) -> bool:
    keys = (
        "result",
        "steps",
        "inputs_used",
        "parameters_used",
        "derived_values",
        "status",
        "errors",
    )
    return {key: stored_result.get(key) for key in keys} == {
        key: replayed_result.get(key) for key in keys
    }


@router.post("/match", response_model=MatchResponse)
def match(request: MatchRequest) -> MatchResponse:
    """Deterministic keyword/alias matching of a free-text description to
    candidate calculators — a testable preview of a future routing layer.
    Returns ranked candidates plus each one's still-required inputs."""
    return match_query(request.query, _engine.registry.definitions())
