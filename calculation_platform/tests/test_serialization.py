"""Serialization-boundary contract: Decimal values leave the module as exact
strings, never floats. Binary float noise (the 0.1 + 0.2 == 0.30000000000000004
class of error) must be impossible in API responses, stored calculations, and
replays.
"""

from decimal import Decimal

from fastapi.testclient import TestClient

from app.api.routes import set_store
from app.core.result_builder import to_jsonable
from app.main import app
from app.storage.sqlite_store import SqliteCalculationStore

import pytest

client = TestClient(app)


@pytest.fixture()
def store(tmp_path):
    store = SqliteCalculationStore(tmp_path / "calculations.db")
    set_store(store)
    return store


def test_to_jsonable_serializes_decimal_as_exact_string():
    assert to_jsonable(Decimal("616.44")) == "616.44"
    assert to_jsonable(Decimal("0.30")) == "0.30"  # trailing zero preserved
    assert to_jsonable({"a": [Decimal("1.10")]}) == {"a": ["1.10"]}


def test_to_jsonable_serializes_dates_as_iso_strings():
    from datetime import date, datetime

    assert to_jsonable(date(2026, 3, 31)) == "2026-03-31"
    assert to_jsonable(datetime(2026, 3, 31, 12, 30)) == "2026-03-31T12:30:00"


def test_float_noise_cannot_appear_in_api_response(store):
    # 0.1 * (1 + 2.0) in binary floats is 0.30000000000000004; the exact
    # Decimal result is 0.30. The response must carry the exact string.
    response = client.post(
        "/calculate",
        json={
            "calculator_id": "business.invoice_total",
            "inputs": {"net_amount": "0.1", "vat_rate": "2.0"},
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["result"]["total"] == "0.30"
    assert isinstance(body["result"]["total"], str)
    # inputs echo back as exact strings too
    assert body["inputs_used"]["net_amount"] == "0.1"


def test_stored_calculation_replays_losslessly(store):
    calculated = client.post(
        "/calculate",
        json={
            "calculator_id": "business.invoice_total",
            "inputs": {"net_amount": "0.1", "vat_rate": "2.0"},
        },
    ).json()

    stored = client.get(f"/calculations/{calculated['request_id']}").json()
    assert stored["result"]["result"]["total"] == "0.30"

    replay = client.post(f"/calculations/{calculated['request_id']}/replay").json()
    assert replay["matches"] is True
    # exact equality, not float-tolerant comparison
    assert replay["replayed_result"]["result"] == stored["result"]["result"]
    assert replay["replayed_result"]["steps"] == stored["result"]["steps"]
    assert replay["replayed_result"]["parameters_used"] == stored["result"]["parameters_used"]
