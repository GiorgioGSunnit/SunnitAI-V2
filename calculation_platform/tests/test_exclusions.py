"""`exclusions` must survive every hop a result takes.

A calculator's exclusions are the boundary of its answer — "does not
include VAT", "the coverage limit is collected but never scored". They
lived on the definition only, so anyone reading a stored calculation, a
replay, a report, or a chatbot answer saw a confident number with no
statement of what it leaves out. These tests pin the whole chain: engine
result -> /calculate payload -> SQLite -> GET -> replay -> HTML report,
plus the two renderers (dev simulation and production bridge).
"""

import html as html_lib

import pytest
from fastapi.testclient import TestClient

from app.api.routes import set_store
from app.main import app, engine
from app.reporting import render_report_html
from app.schemas.calculation_request import CalculationRequest
from app.schemas.stored_calculation import StoredCalculation
from app.storage.sqlite_store import SqliteCalculationStore

client = TestClient(app)

_REQUEST = {
    "calculator_id": "business.confronto_gas_luce",
    "inputs": {
        "consumo_annuo_kwh": 2000,
        "consumo_annuo_smc": 800,
        "offerte": [
            {"fornitore": "Alfa", "prezzo_kwh_luce": "0.25", "prezzo_smc_gas": "1.10"},
            {"fornitore": "Beta", "prezzo_kwh_luce": "0.22", "prezzo_smc_gas": "1.05"},
        ],
    },
}


@pytest.fixture(autouse=True)
def isolated_calculation_store(tmp_path):
    store = SqliteCalculationStore(tmp_path / "calculations.db")
    set_store(store)
    yield store


def _declared_exclusions():
    return engine.registry.get("business.confronto_gas_luce").exclusions


def test_engine_result_carries_the_definition_exclusions():
    result = engine.calculate(CalculationRequest(**_REQUEST))
    assert result.status == "success", result.errors
    assert result.exclusions == _declared_exclusions()
    # Structured, not prose folded into the warning list.
    assert isinstance(result.exclusions, list)
    assert all(isinstance(item, str) for item in result.exclusions)


def test_calculate_endpoint_returns_exclusions():
    response = client.post("/calculate", json=_REQUEST)
    assert response.status_code == 200
    body = response.json()
    assert body["exclusions"] == _declared_exclusions()


def test_stored_calculation_and_replay_preserve_exclusions():
    request_id = client.post("/calculate", json=_REQUEST).json()["request_id"]

    stored = client.get(f"/calculations/{request_id}").json()
    assert stored["result"]["exclusions"] == _declared_exclusions()

    replay = client.post(f"/calculations/{request_id}/replay").json()
    assert replay["replayed_result"]["exclusions"] == _declared_exclusions()
    assert replay["matches"] is True


def test_report_renders_exclusions_under_their_own_heading():
    request = CalculationRequest(request_id="exclusions-report", **_REQUEST)
    result = engine.calculate(request)
    stored = StoredCalculation(
        request_id=request.request_id,
        created_at="2026-07-28T12:00:00Z",
        calculator_id=request.calculator_id,
        status=result.status,
        result_preview=result.result,
        request=request.model_dump(mode="json"),
        result=result.model_dump(mode="json"),
    )
    html = render_report_html(stored, engine.registry.get(request.calculator_id))

    assert "Non incluso" in html
    for exclusion in _declared_exclusions():
        assert html_lib.escape(exclusion, quote=True) in html


def test_report_falls_back_to_the_definition_for_pre_exclusions_records():
    """An archived result from before results carried exclusions must still
    show them rather than silently claim the calculator excludes nothing."""
    request = CalculationRequest(request_id="legacy-report", **_REQUEST)
    result = engine.calculate(request)
    payload = result.model_dump(mode="json")
    payload.pop("exclusions")

    stored = StoredCalculation(
        request_id=request.request_id,
        created_at="2026-01-01T12:00:00Z",
        calculator_id=request.calculator_id,
        status=result.status,
        result_preview=result.result,
        request=request.model_dump(mode="json"),
        result=payload,
    )
    html = render_report_html(stored, engine.registry.get(request.calculator_id))
    assert html_lib.escape(_declared_exclusions()[0], quote=True) in html


def test_development_conversation_renders_exclusions():
    from simulation.conversation import SimulatedConversation

    chat = SimulatedConversation(engine)
    chat.send("confronta due offerte gas e luce e dimmi quale conviene")
    chat.send("consumo annuo 2700 kWh di luce e 1200 Smc di gas")
    chat.send(
        "Fornitore Alfa: luce 0,25 euro al kWh, gas 1,10 euro a Smc, "
        "costo fisso 10 euro al mese, voto 4,0"
    )
    chat.send(
        "Fornitore Beta: luce 0,22 euro al kWh, gas 1,05 euro a Smc, "
        "costo fisso 12 euro al mese, voto 3,9"
    )
    reply = chat.send("confronta")

    assert reply.kind == "answer", reply.text
    assert "Non incluso:" in reply.text
    for exclusion in _declared_exclusions():
        assert exclusion in reply.text


def test_a_failed_calculation_still_reports_what_the_calculator_excludes():
    """A validation error is exactly when someone needs to be told the model
    excludes VAT and system charges — the error builder dropped them."""
    response = client.post(
        "/calculate", json={"calculator_id": "business.confronto_gas_luce", "inputs": {}}
    )
    body = response.json()
    assert body["status"] == "error"
    assert body["exclusions"] == _declared_exclusions()


def test_an_unknown_calculator_reports_no_exclusions_rather_than_guessing():
    body = client.post(
        "/calculate", json={"calculator_id": "does.not.exist", "inputs": {}}
    ).json()
    assert body["status"] == "error"
    assert body["exclusions"] == []


def test_a_stored_empty_exclusions_list_is_not_overwritten_by_the_definition():
    """An empty list is a statement ("this excluded nothing when it ran"),
    not a missing field; only a genuinely absent key falls back."""
    request = CalculationRequest(request_id="empty-exclusions", **_REQUEST)
    result = engine.calculate(request)
    payload = result.model_dump(mode="json")
    payload["exclusions"] = []

    stored = StoredCalculation(
        request_id=request.request_id,
        created_at="2026-01-01T12:00:00Z",
        calculator_id=request.calculator_id,
        status=result.status,
        result_preview=result.result,
        request=request.model_dump(mode="json"),
        result=payload,
    )
    html = render_report_html(stored, engine.registry.get(request.calculator_id))
    assert "Nessuna esclusione dichiarata" in html
    assert html_lib.escape(_declared_exclusions()[0], quote=True) not in html
