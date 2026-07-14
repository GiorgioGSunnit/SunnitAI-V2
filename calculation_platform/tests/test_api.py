from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


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
    assert body["result"]["total"] == 1098.00


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
    assert body["inputs"][0]["name"] == "taxable_income"
    assert body["parameters"][0]["parameter_id"] == "legal_it.irpef_brackets"


def test_get_calculator_definition_404_for_unknown_id():
    response = client.get("/calculators/does.not.exist")
    assert response.status_code == 404


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
    assert body["inputs"] == {"taxable_income": 42000.0}
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
    assert second["calculation"]["result"]["gross_tax"] == 11060.00

    client.post("/simulate/reset")


def test_simulate_reset_clears_pending_state():
    client.post("/simulate/reset")
    client.post("/simulate/chat", json={"message": "quanto pago di irpef nel 2026?"})  # leaves a pending question
    client.post("/simulate/reset")
    # after reset, the same fresh sentence must be treated as a new conversation
    reply = client.post("/simulate/chat", json={"message": "calcolo irpef su 42000 euro nel 2024"}).json()
    assert reply["kind"] == "answer"
    assert reply["calculation"]["result"]["gross_tax"] == 11340.00
    client.post("/simulate/reset")
