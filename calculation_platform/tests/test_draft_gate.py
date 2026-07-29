"""Unreleased packs must be unreachable unless SUNNIT_ENABLE_DRAFT_PACKS is set.

The release manifest, not a version spelling, declares which calculators
have been legally validated. The post-computation `draft_not_validated`
warning documents an override; it does not prevent a user from being handed
a criminal-sentencing range. These tests pin the gate itself, at every entry
point a user can reach: natural-language matching, the planner, discovery
listings, and a direct /calculate by calculator_id.
"""

import pytest
from fastapi.testclient import TestClient

from app.core.errors import CalculatorNotFoundError
from app.core.matcher import match_query
from app.core.registry import CalculatorRegistry
from app.main import FORMULA_PACKS_DIR
from app.schemas.calculation_request import CalculationRequest
from simulation.planner import plan_sentence

DRAFT_IDS = {
    "legal_it.omicidio_pena_draft",
    "legal_it.furto_pena_draft",
    "legal_it.rapina_pena_draft",
    "legal_it.furto_aggravato_draft",
    "legal_it.rapina_aggravata_draft",
}


def _registry(drafts_enabled: bool) -> CalculatorRegistry:
    return CalculatorRegistry(FORMULA_PACKS_DIR, enable_drafts=drafts_enabled)


# ---------------------------------------------------------------------------
# Flag off — drafts do not exist as far as any caller can tell
# ---------------------------------------------------------------------------

def test_drafts_are_absent_from_definitions_and_listings():
    registry = _registry(drafts_enabled=False)
    listed = {d.id for d in registry.definitions()}
    assert not (listed & DRAFT_IDS)
    assert not ({entry["id"] for entry in registry.list_all()} & DRAFT_IDS)


def test_natural_language_does_not_route_to_a_draft():
    registry = _registry(drafts_enabled=False)
    definitions = registry.definitions()
    for sentence in (
        "pena per omicidio con 2 aggravanti e 0 attenuanti",
        "che pena rischia chi ruba",
        "pena per rapina",
        "furto in casa",
        "rapina a mano armata",
    ):
        response = match_query(sentence, definitions)
        assert not ({c.calculator_id for c in response.candidates} & DRAFT_IDS), sentence
        assert plan_sentence(sentence, definitions).calculator_id not in DRAFT_IDS, sentence


@pytest.mark.parametrize("calculator_id", sorted(DRAFT_IDS))
def test_direct_lookup_by_id_is_refused(calculator_id):
    with pytest.raises(CalculatorNotFoundError) as excinfo:
        _registry(drafts_enabled=False).get(calculator_id)
    assert excinfo.value.details.get("released") is False
    assert excinfo.value.details.get("enable_with") == "SUNNIT_ENABLE_DRAFT_PACKS"


def test_calculate_returns_a_structured_error_not_a_result():
    from app.core.engine import CalculationEngine
    from app.main import PARAMETERS_DIR
    from app.resolvers.parameter_store import ParameterStore

    engine = CalculationEngine(_registry(drafts_enabled=False), ParameterStore(PARAMETERS_DIR))
    result = engine.calculate(CalculationRequest(
        calculator_id="legal_it.omicidio_pena_draft",
        inputs={"aggravanti_comuni": 0, "attenuanti_comuni": 0},
    ))
    assert result.status == "error"
    assert result.result == {}
    assert result.errors[0].code == "calculator_not_found"


def test_http_calculate_and_discovery_do_not_expose_drafts(monkeypatch):
    monkeypatch.delenv("SUNNIT_ENABLE_DRAFT_PACKS", raising=False)
    import importlib

    import app.main

    reloaded = importlib.reload(app.main)
    try:
        client = TestClient(reloaded.app)

        listed = {entry["id"] for entry in client.get("/calculators").json()}
        assert not (listed & DRAFT_IDS)
        schema_ids = {entry["name"] for entry in client.get("/tool-schemas").json()}
        assert not any(draft.split(".")[-1] in schema_ids for draft in DRAFT_IDS)

        assert client.get("/calculators/legal_it.omicidio_pena_draft").status_code == 404
        assert client.get(
            "/calculators/legal_it.omicidio_pena_draft/tool-schema"
        ).status_code == 404

        matched = client.post("/match", json={"query": "pena per omicidio"}).json()
        assert not ({c["calculator_id"] for c in matched["candidates"]} & DRAFT_IDS)

        response = client.post("/calculate", json={
            "calculator_id": "legal_it.omicidio_pena_draft",
            "inputs": {"aggravanti_comuni": 0, "attenuanti_comuni": 0},
        })
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "error"
        assert body["errors"][0]["code"] == "calculator_not_found"
    finally:
        importlib.reload(app.main)


# ---------------------------------------------------------------------------
# Flag on — drafts stay developable, and still carry their caveat
# ---------------------------------------------------------------------------

def test_flag_on_makes_drafts_reachable_again():
    registry = _registry(drafts_enabled=True)
    assert DRAFT_IDS <= {d.id for d in registry.definitions()}
    assert registry.get("legal_it.omicidio_pena_draft").id == "legal_it.omicidio_pena_draft"
    assert plan_sentence(
        "pena per omicidio con 2 aggravanti e 0 attenuanti", registry.definitions()
    ).calculator_id == "legal_it.omicidio_pena_draft"


def test_environment_variable_drives_the_default(monkeypatch):
    monkeypatch.setenv("SUNNIT_ENABLE_DRAFT_PACKS", "1")
    assert DRAFT_IDS <= {d.id for d in CalculatorRegistry(FORMULA_PACKS_DIR).definitions()}
    monkeypatch.setenv("SUNNIT_ENABLE_DRAFT_PACKS", "false")
    assert not (DRAFT_IDS & {d.id for d in CalculatorRegistry(FORMULA_PACKS_DIR).definitions()})
    monkeypatch.delenv("SUNNIT_ENABLE_DRAFT_PACKS")
    assert not (DRAFT_IDS & {d.id for d in CalculatorRegistry(FORMULA_PACKS_DIR).definitions()})
