"""Release policy: only human-verified calculators are reachable OR disclosable.

Two properties the "-draft" version heuristic could not give:

  - default-deny. A pack is withheld unless its id appears in the release
    manifest, so a brand-new pack with an ordinary version ("1.0") and no
    draft marker is withheld too. The old check keyed on a version suffix,
    which failed open for every authoring mistake it existed to contain
    ('0.1-DRAFT', '1.0-draft.1', '1.0-draft+build', a trailing space).

  - disclosure, not just computation. Withholding a calculator from
    /calculate while its stored results stay readable protects nobody: a
    user handed a stored criminal-sentencing range is harmed exactly as
    much as one who computed it.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api import routes
from app.core.engine import CalculationEngine
from app.core.errors import CalculatorNotFoundError
from app.core.registry import CalculatorRegistry
from app.main import FORMULA_PACKS_DIR, PARAMETERS_DIR
from app.resolvers.parameter_store import ParameterStore
from app.schemas.calculation_request import CalculationRequest
from app.storage.base import CalculationRecord
from app.storage.sqlite_store import SqliteCalculationStore

DRAFT_ID = "legal_it.omicidio_pena_draft"
RELEASED_ID = "business.invoice_total"


def _engine(drafts_enabled: bool) -> CalculationEngine:
    return CalculationEngine(
        CalculatorRegistry(FORMULA_PACKS_DIR, enable_drafts=drafts_enabled),
        ParameterStore(PARAMETERS_DIR),
    )


@pytest.fixture
def seeded_client(tmp_path: Path, monkeypatch):
    """A store holding one draft and one released result, served by a
    registry that withholds drafts — i.e. the state after the flag is
    turned off on a system that had it on."""
    store = SqliteCalculationStore(tmp_path / "calculations.db")

    seeding_engine = _engine(drafts_enabled=True)
    draft = seeding_engine.calculate(CalculationRequest(
        request_id="draft-req",
        calculator_id=DRAFT_ID,
        inputs={"aggravanti_comuni": 0, "attenuanti_comuni": 0},
    ))
    assert draft.status == "success"
    store.save(CalculationRecord.from_models(
        CalculationRequest(request_id="draft-req", calculator_id=DRAFT_ID, inputs={}), draft
    ))

    released = seeding_engine.calculate(CalculationRequest(
        request_id="ok-req",
        calculator_id=RELEASED_ID,
        inputs={"net_amount": 100, "vat_rate": 0.22},
    ))
    store.save(CalculationRecord.from_models(
        CalculationRequest(request_id="ok-req", calculator_id=RELEASED_ID, inputs={}), released
    ))

    monkeypatch.setattr(routes, "_engine", _engine(drafts_enabled=False))
    monkeypatch.setattr(routes, "_store", store)
    from app.main import app
    return TestClient(app)


# ---------------------------------------------------------------------------
# Finding A — stored results must not leak a withheld calculator
# ---------------------------------------------------------------------------

def test_listing_omits_withheld_records(seeded_client):
    body = seeded_client.get("/calculations").json()
    ids = {entry["calculator_id"] for entry in body}
    assert DRAFT_ID not in ids, f"withheld calculation disclosed in listing: {body}"
    assert RELEASED_ID in ids


def test_stored_detail_refuses_withheld_record(seeded_client):
    response = seeded_client.get("/calculations/draft-req")
    assert response.status_code == 404
    assert response.status_code != 500
    assert "draft-req" not in response.text or "pena_minima" not in response.text
    assert seeded_client.get("/calculations/ok-req").status_code == 200


def test_report_refuses_withheld_record(seeded_client):
    response = seeded_client.get("/calculations/draft-req/report")
    assert response.status_code == 404
    assert "pena_minima" not in response.text
    assert seeded_client.get("/calculations/ok-req/report").status_code == 200


def test_replay_does_not_return_the_stored_withheld_result(seeded_client):
    body = seeded_client.post("/calculations/draft-req/replay").json()
    assert "pena_minima" not in str(body), f"replay disclosed stored draft result: {body}"


def test_override_restores_access_to_stored_records(tmp_path, monkeypatch):
    store = SqliteCalculationStore(tmp_path / "calculations.db")
    engine = _engine(drafts_enabled=True)
    result = engine.calculate(CalculationRequest(
        request_id="draft-req", calculator_id=DRAFT_ID,
        inputs={"aggravanti_comuni": 0, "attenuanti_comuni": 0},
    ))
    store.save(CalculationRecord.from_models(
        CalculationRequest(request_id="draft-req", calculator_id=DRAFT_ID, inputs={}), result
    ))
    monkeypatch.setattr(routes, "_engine", engine)
    monkeypatch.setattr(routes, "_store", store)
    from app.main import app
    client = TestClient(app)

    assert DRAFT_ID in {e["calculator_id"] for e in client.get("/calculations").json()}
    assert client.get("/calculations/draft-req").status_code == 200
    assert client.get("/calculations/draft-req/report").status_code == 200


# ---------------------------------------------------------------------------
# Finding B — detection must not key on a version string
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("version", ["0.1-DRAFT", "1.0-draft.1", "1.0-draft+build", "0.1-draft "])
def test_draft_marker_variants_are_still_withheld(tmp_path, version):
    """Every one of these loaded as VISIBLE under the suffix heuristic."""
    _write_pack(tmp_path, "legal_it.variant_probe", version)
    registry = CalculatorRegistry(tmp_path, enable_drafts=False)
    assert "legal_it.variant_probe" not in {d.id for d in registry.definitions()}
    with pytest.raises(CalculatorNotFoundError):
        registry.get("legal_it.variant_probe")


def test_unmarked_new_pack_is_withheld_by_default(tmp_path):
    """The fail-open case: an author who writes no draft marker at all."""
    _write_pack(tmp_path, "legal_it.unmarked_probe", "1.0")
    registry = CalculatorRegistry(tmp_path, enable_drafts=False)
    assert "legal_it.unmarked_probe" not in {d.id for d in registry.definitions()}
    assert "legal_it.unmarked_probe" not in {e["id"] for e in registry.list_all()}
    with pytest.raises(CalculatorNotFoundError):
        registry.get("legal_it.unmarked_probe")


def test_withheld_pack_is_still_parsed_and_validated(tmp_path):
    """A broken withheld pack must still break the build."""
    (tmp_path / "broken.yml").write_text(
        "id: legal_it.broken_probe\nname: Broken\ncategory: test\n"
        "strategy: expression\nversion: '1.0'\n"
        "inputs:\n  - {name: a, type: decimal, required: true}\n"
        "formula: {expression: 'a + nonexistent_variable'}\n"
        "output: {name: result}\n",
        encoding="utf-8",
    )
    from app.core.errors import DefinitionValidationError
    with pytest.raises(DefinitionValidationError):
        CalculatorRegistry(tmp_path, enable_drafts=False)


def _write_pack(directory: Path, calculator_id: str, version: str) -> None:
    (directory / f"{calculator_id.split('.')[-1]}.yml").write_text(
        f"id: {calculator_id}\nname: Probe\ncategory: test\n"
        f"strategy: expression\nversion: '{version}'\n"
        "inputs:\n  - {name: a, type: decimal, required: true}\n"
        "formula: {expression: 'a * 2'}\n"
        "output: {name: result, round_to: 2}\n",
        encoding="utf-8",
    )
