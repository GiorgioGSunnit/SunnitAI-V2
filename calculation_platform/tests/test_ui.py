"""Smoke tests for the development UI page.

There is no JS runtime in the test environment, so these cannot execute the
page — they pin the contract between the Python side and the browser side:
that the page ships the object_list machinery at all, that the endpoints it
calls exist and return what it reads, and that /simulate/chat stays labelled
development-only. Behavioural verification of the rendering itself is manual,
against a running `uvicorn calculation_platform.app.main:app`.
"""

import re

import pytest
from fastapi.testclient import TestClient

from app.api.routes import set_store
from app.main import app
from app.storage.sqlite_store import SqliteCalculationStore
from app.ui import _PAGE

client = TestClient(app)


@pytest.fixture(autouse=True)
def isolated_calculation_store(tmp_path):
    set_store(SqliteCalculationStore(tmp_path / "calculations.db"))


def test_index_serves_the_page():
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert "Calculation Platform" in response.text


@pytest.mark.parametrize("function_name", [
    "listInputMarkup", "addListItem", "removeListItem", "renumberListItems",
    "readListInput", "readControl", "controlFor", "fieldMarkup", "renderComparison",
])
def test_page_defines_the_object_list_form_functions(function_name):
    assert f"function {function_name}(" in _PAGE


def test_candidate_cards_offer_add_and_remove_controls():
    assert "Aggiungi offerta" in _PAGE
    assert "removeListItem(" in _PAGE
    # Minimum-items validation happens before anything is sent.
    assert "servono almeno ${minItems} elementi" in _PAGE


def test_every_declared_input_type_has_a_control():
    control_source = _PAGE[_PAGE.index("function controlFor("):_PAGE.index("function fieldMarkup(")]
    for kind in ("boolean", "date", "string", "string_list", "integer"):
        assert kind in control_source, f"no control branch for {kind}"
    # A decimal must not go through Number(): the platform's contract is an
    # exact decimal string, and a float round-trip reintroduces binary noise.
    assert "return raw;           // exact decimal string" in _PAGE
    assert "Number(el.value)" not in _PAGE


def test_optional_booleans_are_tri_state_not_checkboxes():
    # Unknown, explicit false and explicit true must stay distinguishable.
    assert "non specificato" in _PAGE
    assert "el.value === '' ? undefined : el.value === 'true'" in _PAGE


def test_required_item_fields_are_marked():
    assert 'class="req"' in _PAGE
    assert "spec.required ?" in _PAGE


def test_comparison_rendering_covers_verdict_provisional_and_quality():
    comparison_source = _PAGE[
        _PAGE.index("function renderComparison("):_PAGE.index("function renderResult(")
    ]
    assert "effective_tie" in comparison_source
    assert "Nessuna offerta è indicata come migliore" in comparison_source
    assert "PROVVISORIO" in comparison_source
    assert "scoring_defaults_applied" in comparison_source
    assert "unknown_fields" in comparison_source
    assert "scoring_completeness" in comparison_source
    assert "costLabel" in comparison_source
    assert "costUnit" in comparison_source
    # The cost is a column of its own, ahead of the synthetic score.
    assert comparison_source.index("cost_basis") < comparison_source.index("total_score")


def test_result_rendering_surfaces_defaults_and_exclusions():
    assert "Non incluso" in _PAGE
    assert "body.exclusions" in _PAGE
    assert "body.defaults_applied" in _PAGE


def test_no_frontend_framework_or_build_step_is_introduced():
    # The page must stay a single self-contained document served by FastAPI.
    assert not re.search(r"<script[^>]+src=", _PAGE)
    assert not re.search(r"<link[^>]+stylesheet", _PAGE)
    for framework in ("react", "vue", "angular", "webpack", "import ", "require("):
        assert framework not in _PAGE.lower()


def test_simulate_chat_is_labelled_development_only():
    assert "Solo per sviluppo" in _PAGE
    response = client.post("/simulate/chat", json={"message": "ciao"})
    assert response.status_code == 200
    assert response.json()["dev_only"] is True
    client.post("/simulate/reset")


def test_the_definition_endpoint_gives_the_page_what_the_list_form_needs():
    """loadDefinition renders candidate cards straight from this payload."""
    definition = client.get("/calculators/business.confronto_gas_luce").json()
    list_input = next(i for i in definition["inputs"] if i["type"] == "object_list")
    assert list_input["min_items"] == 2
    names = {f["name"] for f in list_input["item_fields"]}
    assert {"fornitore", "prezzo_kwh_luce", "prezzo_smc_gas"} <= names
    assert all(f["type"] in {"string", "decimal", "integer", "boolean", "date", "string_list"}
               for f in list_input["item_fields"])


def test_a_calculation_from_the_form_shape_returns_what_the_page_renders():
    """The exact JSON readListInput produces, and every key renderResult reads."""
    response = client.post("/calculate", json={
        "calculator_id": "business.confronto_gas_luce",
        "inputs": {
            "consumo_annuo_kwh": 2700,
            "consumo_annuo_smc": 1200,
            "offerte": [
                {
                    "fornitore": "Alfa",
                    "prezzo_kwh_luce": "0.25",
                    "prezzo_smc_gas": "1.10",
                },
                {
                    "fornitore": "Beta",
                    "prezzo_kwh_luce": "0.22",
                    "prezzo_smc_gas": "1.05",
                },
            ],
        },
    })
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert isinstance(body["exclusions"], list) and body["exclusions"]
    assert isinstance(body["defaults_applied"], list) and body["defaults_applied"]
    comparison = body["result"]["comparison"]
    assert {"decision_status", "best_candidates", "score_gap", "tie_tolerance",
            "provisional", "provisional_status", "scoring_completeness",
            "scoring_defaults_applied", "cost_basis"} <= set(comparison)
    for entry in body["result"]["ranking"]:
        assert {"provided_fields", "assumed_fields", "unknown_fields",
                "scoring_completeness"} <= set(entry["data_quality"])
