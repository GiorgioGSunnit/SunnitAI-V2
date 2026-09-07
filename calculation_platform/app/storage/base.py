from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

from ..schemas.calculation_request import CalculationRequest
from ..schemas.calculation_result import CalculationResult
from ..schemas.stored_calculation import StoredCalculation, StoredCalculationSummary


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class CalculationRecord:
    request_id: str
    created_at: str
    calculator_id: str
    status: str
    request: CalculationRequest
    result: CalculationResult

    @classmethod
    def from_models(
        cls,
        request: CalculationRequest,
        result: CalculationResult,
        created_at: Optional[str] = None,
    ) -> "CalculationRecord":
        request_id = result.request_id or request.request_id
        if not request_id:
            raise ValueError("Stored calculations require a request_id")
        return cls(
            request_id=request_id,
            created_at=created_at or utc_now_iso(),
            calculator_id=result.calculator_id,
            status=result.status,
            request=request,
            result=result,
        )


class CalculationStore(ABC):
    """Swap point for calculation persistence.

    The HTTP layer depends on this interface so a future PostgreSQL-backed
    implementation can replace SQLite without changing the calculation engine.
    """

    @abstractmethod
    def save(self, record: CalculationRecord) -> None:
        raise NotImplementedError

    @abstractmethod
    def get(self, request_id: str) -> Optional[StoredCalculation]:
        raise NotImplementedError

    @abstractmethod
    def list_recent(self, limit: int = 50) -> List[StoredCalculationSummary]:
        raise NotImplementedError
