from pathlib import Path

import pytest

from app.core.errors import CalculatorNotFoundError
from app.core.registry import CalculatorRegistry

FORMULA_PACKS_DIR = Path(__file__).resolve().parent.parent / "formula_packs"


@pytest.fixture
def registry():
    return CalculatorRegistry(FORMULA_PACKS_DIR)


EXPECTED_CALCULATOR_IDS = {
    "business.invoice_total",
    "business.loan_payment",
    "business.confronto_polizze",
    "business.confronto_gas_luce",
    "legal_it.irpef",
    "legal_it.legal_interest",
    "legal_it.registration_tax_leases",
    "legal_it.tfr",
    "legal_it.imu",
    "legal_it.inps_contributions",
    "legal_it.late_payment_interest",
    "legal_it.notice_indemnity",
    "legal_it.omicidio_pena_draft",
    "legal_it.furto_pena_draft",
    "legal_it.furto_aggravato_draft",
    "legal_it.rapina_pena_draft",
    "legal_it.rapina_aggravata_draft",
    "legal_it.contributo_unificato_civile",
    "legal_it.termini_processuali_civili",
    "legal_it.ravvedimento_operoso",
    "legal_it.rivalutazione_istat",
    "legal_it.rivalutazione_interessi_1712",
    "legal_it.compensi_dm55",
}


def test_registry_loads_all_formula_packs(registry):
    ids = {c["id"] for c in registry.list_all()}
    assert ids == EXPECTED_CALCULATOR_IDS


def test_registry_list_all_shape(registry):
    entries = registry.list_all()
    for entry in entries:
        assert set(entry) == {"id", "name", "category", "description", "keywords", "aliases"}


def test_registry_list_all_includes_keywords_for_matching(registry):
    # Not consumed by anything in this platform yet — pure metadata for a
    # future matching layer — but every calculator must have at least one
    # keyword, otherwise it's silently unreachable by name.
    for entry in registry.list_all():
        assert entry["keywords"], f"{entry['id']} has no keywords"


def test_registry_get_returns_full_definition(registry):
    definition = registry.get("legal_it.irpef")
    assert definition.strategy == "progressive_brackets"
    assert definition.requires_period is False
    assert definition.jurisdiction == "IT"
    assert len(definition.examples) >= 1


def test_registry_stamps_requires_period_from_strategy(registry):
    assert registry.get("legal_it.legal_interest").requires_period is True
    assert registry.get("business.invoice_total").requires_period is False


def test_registry_get_raises_structured_error_for_unknown_id(registry):
    with pytest.raises(CalculatorNotFoundError) as exc_info:
        registry.get("does.not.exist")
    assert exc_info.value.code == "calculator_not_found"
    assert "does.not.exist" in exc_info.value.details["calculator_id"]
    assert "legal_it.irpef" in exc_info.value.details["available"]
