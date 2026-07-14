from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException

from ..core.engine import CalculationEngine
from ..core.errors import CalculatorNotFoundError
from ..core.matcher import match_query
from ..schemas.calculation_request import CalculationRequest
from ..schemas.calculation_result import CalculationResult
from ..schemas.calculator_definition import CalculatorDefinition
from ..schemas.match_result import MatchRequest, MatchResponse

router = APIRouter()
_engine: Optional[CalculationEngine] = None


def set_engine(engine: CalculationEngine) -> None:
    global _engine
    _engine = engine


@router.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@router.get("/calculators")
def list_calculators() -> List[Dict[str, Any]]:
    return _engine.registry.list_all()


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
    return _engine.calculate(request)


@router.post("/match", response_model=MatchResponse)
def match(request: MatchRequest) -> MatchResponse:
    """Deterministic keyword/alias matching of a free-text description to
    candidate calculators — a testable preview of a future routing layer.
    Returns ranked candidates plus each one's still-required inputs."""
    return match_query(request.query, _engine.registry.definitions())
