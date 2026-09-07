from pathlib import Path

from app.schemas.calculation_request import CalculationRequest
from app.schemas.calculation_result import CalculationResult
from app.schemas.error import CalculationError
from app.storage.base import CalculationRecord
from app.storage.sqlite_store import SqliteCalculationStore


def _record(request_id: str, created_at: str, status: str = "success") -> CalculationRecord:
    request = CalculationRequest(
        request_id=request_id,
        calculator_id="business.invoice_total",
        inputs={"net_amount": 100, "vat_rate": 0.22},
    )
    result = CalculationResult(
        request_id=request_id,
        calculator_id="business.invoice_total",
        status=status,
        result={"total": 122} if status == "success" else {},
        errors=[] if status == "success" else [
            CalculationError(code="input_invalid", message="Missing net amount")
        ],
    )
    return CalculationRecord.from_models(request, result, created_at=created_at)


def test_sqlite_store_roundtrip_get_and_recent_ordering(tmp_path: Path):
    store = SqliteCalculationStore(tmp_path / "calculations.db")
    store.save(_record("older", "2026-01-01T00:00:00Z"))
    store.save(_record("newer", "2026-01-02T00:00:00Z", status="error"))

    stored = store.get("older")
    assert stored is not None
    assert stored.request_id == "older"
    assert stored.request["inputs"]["net_amount"] == 100
    assert stored.result["result"]["total"] == 122
    assert stored.result_preview == {"total": 122}

    recent = store.list_recent()
    assert [item.request_id for item in recent] == ["newer", "older"]
    assert recent[0].result_preview == "Missing net amount"


def test_sqlite_store_list_limit_is_capped_at_200(tmp_path: Path):
    store = SqliteCalculationStore(tmp_path / "calculations.db")
    for index in range(205):
        store.save(_record(f"request-{index}", f"2026-01-01T00:00:{index:03d}Z"))

    assert len(store.list_recent(limit=500)) == 200
    assert len(store.list_recent(limit=3)) == 3
