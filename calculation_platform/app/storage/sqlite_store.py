import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..schemas.stored_calculation import StoredCalculation, StoredCalculationSummary
from .base import CalculationRecord, CalculationStore


class SqliteCalculationStore(CalculationStore):
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._create_table()

    def save(self, record: CalculationRecord) -> None:
        request_json = json.dumps(
            record.request.model_dump(mode="json"),
            ensure_ascii=False,
            default=str,
        )
        result_json = json.dumps(
            record.result.model_dump(mode="json"),
            ensure_ascii=False,
            default=str,
        )
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO calculations
                    (request_id, created_at, calculator_id, status, request_json, result_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    record.request_id,
                    record.created_at,
                    record.calculator_id,
                    record.status,
                    request_json,
                    result_json,
                ),
            )

    def get(self, request_id: str) -> Optional[StoredCalculation]:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT request_id, created_at, calculator_id, status, request_json, result_json
                FROM calculations
                WHERE request_id = ?
                """,
                (request_id,),
            ).fetchone()
        if row is None:
            return None
        return self._stored_from_row(row)

    def list_recent(self, limit: int = 50) -> List[StoredCalculationSummary]:
        capped_limit = max(0, min(limit, 200))
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT request_id, created_at, calculator_id, status, result_json
                FROM calculations
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (capped_limit,),
            ).fetchall()
        return [self._summary_from_row(row) for row in rows]

    def _create_table(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS calculations (
                    request_id TEXT PRIMARY KEY,
                    created_at TEXT,
                    calculator_id TEXT,
                    status TEXT,
                    request_json TEXT,
                    result_json TEXT
                )
                """
            )

    def _stored_from_row(self, row: Any) -> StoredCalculation:
        request_id, created_at, calculator_id, status, request_json, result_json = row
        result = json.loads(result_json)
        return StoredCalculation(
            request_id=request_id,
            created_at=created_at,
            calculator_id=calculator_id,
            status=status,
            result_preview=self._preview(status, result),
            request=json.loads(request_json),
            result=result,
        )

    def _summary_from_row(self, row: Any) -> StoredCalculationSummary:
        request_id, created_at, calculator_id, status, result_json = row
        result = json.loads(result_json)
        return StoredCalculationSummary(
            request_id=request_id,
            created_at=created_at,
            calculator_id=calculator_id,
            status=status,
            result_preview=self._preview(status, result),
        )

    def _preview(self, status: str, result: Dict[str, Any]) -> Any:
        if status == "error":
            errors = result.get("errors") or []
            if errors:
                return errors[0].get("message")
            return None
        return result.get("result", {})
