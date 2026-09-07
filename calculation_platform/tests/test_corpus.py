"""Runs the deterministic local verification corpus (tests/corpus/scenarios.yml).

Every scenario executes against a locally reproducible layer only — the
deterministic engine, the in-process local API (temp DB), or the deterministic
simulation. No network, no real LLM. Monetary expectations are exact decimal
strings compared as strings (never binary floats).
"""

from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from app.api.routes import set_store
from app.core.matcher import match_query
from app.main import app, engine
from app.schemas.calculation_request import CalculationRequest
from app.storage.sqlite_store import SqliteCalculationStore
from simulation.conversation import SimulatedConversation

_CORPUS = Path(__file__).parent / "corpus" / "scenarios.yml"
_SCENARIOS = yaml.safe_load(_CORPUS.read_text(encoding="utf-8"))["scenarios"]
_EVIDENCE_LEVELS = {
    "engine_verified", "local_api_verified", "integration_simulated", "production_unverified",
}

_client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def _isolated_api_store(tmp_path_factory):
    """Point the API at a temporary DB so corpus runs never touch the normal one."""
    set_store(SqliteCalculationStore(tmp_path_factory.mktemp("corpus") / "calculations.db"))
    yield


def _messages(items):
    out = []
    for item in items or []:
        message = item.message if hasattr(item, "message") else item.get("message")
        if message:
            out.append(message)
    return out


def _normalize_engine(result):
    return {
        "status": result.status,
        "result": dict(result.result or {}),
        "errors": [{"code": e.code, "message": e.message, "details": e.details or {}} for e in result.errors],
        "assumptions": _messages(result.assumptions),
        "warnings": _messages(result.warnings),
    }


def _normalize_api(body):
    return {
        "status": body.get("status"),
        "result": body.get("result") or {},
        "errors": body.get("errors") or [],
        "assumptions": _messages(body.get("assumptions")),
        "warnings": _messages(body.get("warnings")),
    }


def _request(sc):
    return CalculationRequest(
        calculator_id=sc["calculator_id"],
        inputs=sc.get("inputs", {}),
        tax_year=sc.get("tax_year"),
        as_of_date=sc.get("as_of_date"),
        period=sc.get("period"),
        caller_supplied_values=sc.get("caller_supplied_values", {}),
    )


def _check_calculation(sc, normalized):
    expect = sc["expect"]
    assert normalized["status"] == expect["status"], f"{sc['id']}: {normalized['errors']}"
    for key, value in (expect.get("result") or {}).items():
        assert str(normalized["result"].get(key)) == str(value), (
            f"{sc['id']}: {key} expected {value!r}, got {normalized['result'].get(key)!r}"
        )
    if "error_code" in expect:
        assert normalized["errors"], f"{sc['id']}: expected an error"
        assert normalized["errors"][0]["code"] == expect["error_code"]
    for key, value in (expect.get("error_details") or {}).items():
        assert str(normalized["errors"][0]["details"].get(key)) == str(value)
    if "error_message_contains" in expect:
        assert expect["error_message_contains"] in normalized["errors"][0]["message"]
    for needle in expect.get("assumptions_contains") or []:
        assert any(needle in a for a in normalized["assumptions"]), f"{sc['id']}: assumption {needle!r} missing"
    for needle in expect.get("warnings_contains") or []:
        assert any(needle in w for w in normalized["warnings"]), f"{sc['id']}: warning {needle!r} missing"


@pytest.mark.parametrize("sc", _SCENARIOS, ids=[s["id"] for s in _SCENARIOS])
def test_corpus_scenario(sc):
    assert sc.get("evidence") in _EVIDENCE_LEVELS, f"{sc['id']}: bad evidence level"
    expect = sc["expect"]

    if "route" in sc:
        response = match_query(sc["route"], engine.registry.definitions())
        assert response.status == expect["routing_status"], f"{sc['id']}: {response.status}"
        if "calculator_id" in expect:
            assert response.candidates[0].calculator_id == expect["calculator_id"]
        return

    if "chat" in sc:
        conversation = SimulatedConversation(engine)
        reply = None
        for message in sc["chat"]:
            reply = conversation.send(message)
        assert reply.kind == expect["kind"], f"{sc['id']}: {reply.kind}"
        if "calculator_id" in expect:
            assert reply.tool_call.calculator_id == expect["calculator_id"]
        if "missing" in expect:
            assert reply.plan.missing_inputs == expect["missing"]
        for key, value in (expect.get("result") or {}).items():
            assert str(reply.calculation.result.get(key)) == str(value)
        return

    if "calculator_id" in sc:
        if sc.get("api"):
            body = _client.post("/calculate", json={
                "calculator_id": sc["calculator_id"],
                "inputs": sc.get("inputs", {}),
                **({"period": sc["period"]} if sc.get("period") else {}),
                **({"tax_year": sc["tax_year"]} if sc.get("tax_year") else {}),
            }).json()
            _check_calculation(sc, _normalize_api(body))
        else:
            _check_calculation(sc, _normalize_engine(engine.calculate(_request(sc))))
        return

    raise AssertionError(f"{sc['id']}: scenario declares no runnable layer")
