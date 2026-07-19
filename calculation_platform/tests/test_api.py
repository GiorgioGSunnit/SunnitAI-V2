import pytest
from fastapi.testclient import TestClient

from app.api.routes import set_store
from app.main import app
from app.schemas.calculation_request import CalculationRequest
from app.schemas.calculation_result import CalculationResult
from app.schemas.citation import Citation
from app.storage.base import CalculationRecord
from app.storage.sqlite_store import SqliteCalculationStore

client = TestClient(app)


@pytest.fixture(autouse=True)
def isolated_calculation_store(tmp_path):
    store = SqliteCalculationStore(tmp_path / "calculations.db")
    set_store(store)
    yield store


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_list_calculators_includes_full_registry():
    from tests.test_registry import EXPECTED_CALCULATOR_IDS

    response = client.get("/calculators")
    assert response.status_code == 200
    ids = {c["id"] for c in response.json()}
    assert ids == EXPECTED_CALCULATOR_IDS


def test_calculate_invoice_total_via_api():
    response = client.post(
        "/calculate",
        json={
            "calculator_id": "business.invoice_total",
            "inputs": {"net_amount": 1000, "vat_rate": 0.22, "discount_rate": 0.10},
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["result"]["total"] == "1098.00"


def test_calculate_generates_request_id_when_absent_and_persists(isolated_calculation_store):
    response = client.post(
        "/calculate",
        json={
            "calculator_id": "business.invoice_total",
            "inputs": {"net_amount": 1000, "vat_rate": 0.22, "discount_rate": 0.10},
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["request_id"]

    stored = isolated_calculation_store.get(body["request_id"])
    assert stored is not None
    assert stored.request["request_id"] == body["request_id"]
    assert stored.result["request_id"] == body["request_id"]


def test_calculate_error_result_is_persisted(isolated_calculation_store):
    response = client.post("/calculate", json={"calculator_id": "does.not.exist"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "error"

    stored = isolated_calculation_store.get(body["request_id"])
    assert stored is not None
    assert stored.status == "error"
    assert stored.result_preview == body["errors"][0]["message"]


def test_calculate_unknown_calculator_returns_error_status_not_http_error():
    response = client.post("/calculate", json={"calculator_id": "does.not.exist"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "error"
    assert body["errors"][0]["code"] == "calculator_not_found"
    assert "does.not.exist" in body["errors"][0]["message"]
    assert "does.not.exist" in body["errors"][0]["details"]["calculator_id"]


def test_get_calculator_definition_returns_full_shape():
    response = client.get("/calculators/legal_it.irpef")
    assert response.status_code == 200
    body = response.json()
    assert body["strategy"] == "progressive_brackets"
    assert body["requires_period"] is False
    assert body["inputs"][0]["name"] == "taxable_income"
    assert body["parameters"][0]["parameter_id"] == "legal_it.irpef_brackets"


def test_get_calculator_definition_exposes_requires_period():
    response = client.get("/calculators/legal_it.legal_interest")
    assert response.status_code == 200
    body = response.json()
    assert body["requires_period"] is True


def test_get_calculator_definition_404_for_unknown_id():
    response = client.get("/calculators/does.not.exist")
    assert response.status_code == 404


def test_calculations_list_and_get_happy_path_and_404():
    first = client.post(
        "/calculate",
        json={
            "calculator_id": "business.invoice_total",
            "inputs": {"net_amount": 100, "vat_rate": 0.22},
        },
    ).json()
    second = client.post(
        "/calculate",
        json={
            "calculator_id": "legal_it.irpef",
            "inputs": {"taxable_income": 42000},
            "tax_year": 2026,
        },
    ).json()

    listed = client.get("/calculations?limit=1")
    assert listed.status_code == 200
    summaries = listed.json()
    assert len(summaries) == 1
    assert summaries[0]["request_id"] in {first["request_id"], second["request_id"]}
    assert summaries[0]["result_preview"]

    stored = client.get(f"/calculations/{second['request_id']}")
    assert stored.status_code == 200
    assert stored.json()["request"]["tax_year"] == 2026
    assert stored.json()["result"]["result"]["gross_tax"] == "11060.00"

    missing = client.get("/calculations/does-not-exist")
    assert missing.status_code == 404


def test_replay_deterministic_calculation_matches():
    calculated = client.post(
        "/calculate",
        json={
            "calculator_id": "legal_it.irpef",
            "inputs": {"taxable_income": 42000},
            "tax_year": 2026,
        },
    ).json()

    replay = client.post(f"/calculations/{calculated['request_id']}/replay")
    assert replay.status_code == 200
    body = replay.json()
    assert body["request_id"] == calculated["request_id"]
    assert body["matches"] is True
    assert body["replayed_result"]["result"]["gross_tax"] == "11060.00"


def test_replay_unknown_calculation_404():
    response = client.post("/calculations/does-not-exist/replay")
    assert response.status_code == 404


def test_report_endpoint_returns_printable_html_for_stored_calculation():
    calculated = client.post(
        "/calculate",
        json={
            "calculator_id": "legal_it.irpef",
            "inputs": {"taxable_income": 42000},
            "tax_year": 2026,
        },
    ).json()

    response = client.get(f"/calculations/{calculated['request_id']}/report")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "IRPEF" in response.text
    assert "non costituisce parere legale" in response.text


def test_report_endpoint_404_for_unknown_calculation():
    response = client.get("/calculations/does-not-exist/report")
    assert response.status_code == 404


def test_report_endpoint_renders_when_calculator_definition_is_missing(isolated_calculation_store):
    request = CalculationRequest(
        request_id="missing-definition-report",
        calculator_id="legacy.missing",
        inputs={"amount": 100},
    )
    result = CalculationResult(
        request_id=request.request_id,
        calculator_id=request.calculator_id,
        status="success",
        result={"total": 100},
        formula_version="legacy-1",
        raw_inputs=request.inputs,
        inputs_used=request.inputs,
        steps=[{"step": 1, "type": "legacy", "value": 100}],
        citations=[
            Citation(
                reference="Legacy reference",
                source_name="Archived source",
                publisher="Archivio",
                publication_date="2024-01-01",
                url="https://example.invalid/source",
            )
        ],
    )
    isolated_calculation_store.save(
        CalculationRecord.from_models(request, result, created_at="2026-01-15T12:00:00Z")
    )

    response = client.get("/calculations/missing-definition-report/report")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "legacy.missing" in response.text
    assert "Legacy reference" in response.text
    assert "non costituisce parere legale" in response.text


def test_storage_failure_returns_calculation_with_warning(isolated_calculation_store, monkeypatch):
    def fail_save(record):
        raise OSError("read-only test store")

    monkeypatch.setattr(isolated_calculation_store, "save", fail_save)
    response = client.post(
        "/calculate",
        json={
            "calculator_id": "business.invoice_total",
            "inputs": {"net_amount": 100, "vat_rate": 0.22},
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["result"]["total"] == "122.00"
    assert any(w["code"] == "persistence_failed" for w in body["warnings"])


def test_match_endpoint_returns_ranked_candidates():
    response = client.post("/match", json={"query": "quanto pago di tasse sul reddito"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in ("matched", "ambiguous")
    assert body["candidates"][0]["calculator_id"] == "legal_it.irpef"
    assert body["candidates"][0]["matched_terms"]
    assert any(i["name"] == "taxable_income" for i in body["candidates"][0]["required_inputs"])


def test_match_endpoint_no_match_for_unrelated_text():
    response = client.post("/match", json={"query": "xylophone zebra quantum"})
    assert response.status_code == 200
    assert response.json()["status"] == "no_match"


def test_plan_endpoint_ready_to_calculate():
    response = client.post("/plan", json={"sentence": "calcolo irpef su 42000 euro nel 2026"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready_to_calculate"
    assert body["calculator_id"] == "legal_it.irpef"
    assert body["inputs"] == {"taxable_income": "42000"}
    assert body["tax_year"] == 2026


def test_plan_endpoint_needs_clarification():
    response = client.post("/plan", json={"sentence": "quanto pago di irpef nel 2026?"})
    body = response.json()
    assert body["status"] == "needs_clarification"
    assert body["missing_inputs"] == ["taxable_income"]
    assert body["question"]


def test_simulate_chat_full_clarification_loop_via_http():
    client.post("/simulate/reset")

    first = client.post("/simulate/chat", json={"message": "quanto pago di irpef nel 2026?"}).json()
    assert first["kind"] == "question"
    assert first["tool_call"]["calculator_id"] == "legal_it.irpef"
    assert first["plan"]["status"] == "needs_clarification"
    assert first["plan"]["missing_inputs"] == ["taxable_income"]

    second = client.post("/simulate/chat", json={"message": "il reddito è di 42000 euro"}).json()
    assert second["kind"] == "answer"
    assert second["calculation"]["result"]["gross_tax"] == "11060.00"
    assert second["calculation"]["request_id"]
    assert client.get(f"/calculations/{second['calculation']['request_id']}").status_code == 200

    client.post("/simulate/reset")


def test_simulate_reset_clears_pending_state():
    client.post("/simulate/reset")
    client.post("/simulate/chat", json={"message": "quanto pago di irpef nel 2026?"})  # leaves a pending question
    client.post("/simulate/reset")
    # after reset, the same fresh sentence must be treated as a new conversation
    reply = client.post("/simulate/chat", json={"message": "calcolo irpef su 42000 euro nel 2024"}).json()
    assert reply["kind"] == "answer"
    assert reply["calculation"]["result"]["gross_tax"] == "11340.00"
    client.post("/simulate/reset")
