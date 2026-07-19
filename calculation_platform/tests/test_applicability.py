from datetime import date
from pathlib import Path

from app.core.engine import CalculationEngine
from app.core.registry import CalculatorRegistry
from app.main import engine as live_engine
from app.resolvers.parameter_store import ParameterStore
from app.schemas.calculation_request import CalculationRequest


def _write_pack(tmp_path: Path, *, applicable_from=None, applicable_to=None) -> Path:
    pack_dir = tmp_path / "packs"
    pack_dir.mkdir()
    lines = [
        "id: test.applicability",
        "name: Applicability test calculator",
        "category: test",
        "strategy: expression",
        'version: "1"',
        "description: Test-only calculator for applicability windows.",
    ]
    if applicable_from is not None:
        lines.append(f"applicable_from: {applicable_from}")
    if applicable_to is not None:
        lines.append(f"applicable_to: {applicable_to}")
    lines.extend([
        "inputs:",
        "  - name: amount",
        "    type: decimal",
        "    required: true",
        "parameters: []",
        "derived_variables: {}",
        "formula:",
        '  expression: "amount * 2"',
        "output:",
        "  name: total",
        "  round_to: 2",
        "citations: []",
        "warnings: []",
        "assumptions: []",
        "exclusions: []",
        "examples: []",
        "keywords:",
        "  - applicability",
    ])
    (pack_dir / "applicability.yml").write_text("\n".join(lines), encoding="utf-8")
    return pack_dir


def _engine(pack_dir: Path) -> CalculationEngine:
    return CalculationEngine(
        CalculatorRegistry(pack_dir),
        ParameterStore(pack_dir / "parameters"),
    )


def _calculate(engine: CalculationEngine, **kwargs):
    request = CalculationRequest(
        calculator_id="test.applicability",
        inputs={"amount": 10},
        **kwargs,
    )
    return engine.calculate(request)


def test_request_before_applicable_from_returns_structured_error(tmp_path):
    engine = _engine(_write_pack(tmp_path, applicable_from="2024-01-01", applicable_to="2024-12-31"))

    result = _calculate(engine, as_of_date=date(2023, 12, 31))

    assert result.status == "error"
    assert result.errors[0].code == "calculator_not_applicable"
    assert "non e applicabile alla data 2023-12-31" in result.errors[0].message
    assert result.errors[0].details == {
        "applicable_from": "2024-01-01",
        "applicable_to": "2024-12-31",
        "as_of_date": "2023-12-31",
        "as_of_source": "explicit_as_of_date",
    }


def test_request_after_applicable_to_returns_structured_error(tmp_path):
    engine = _engine(_write_pack(tmp_path, applicable_from="2024-01-01", applicable_to="2024-12-31"))

    result = _calculate(engine, as_of_date=date(2025, 1, 1))

    assert result.status == "error"
    assert result.errors[0].code == "calculator_not_applicable"
    assert result.errors[0].details == {
        "applicable_from": "2024-01-01",
        "applicable_to": "2024-12-31",
        "as_of_date": "2025-01-01",
        "as_of_source": "explicit_as_of_date",
    }


def test_request_inside_applicability_window_succeeds(tmp_path):
    engine = _engine(_write_pack(tmp_path, applicable_from="2024-01-01", applicable_to="2024-12-31"))

    result = _calculate(engine, as_of_date=date(2024, 6, 15))

    assert result.status == "success"
    assert result.result["total"] == "20.00"


def test_definition_without_applicability_bounds_succeeds_for_any_date(tmp_path):
    engine = _engine(_write_pack(tmp_path))

    result = _calculate(engine, as_of_date=date(1900, 1, 1))

    assert result.status == "success"
    assert result.result["total"] == "20.00"


def test_tax_year_derived_as_of_date_feeds_applicability_check(tmp_path):
    engine = _engine(_write_pack(tmp_path, applicable_from="2024-01-01"))

    result = _calculate(engine, tax_year=2023)

    assert result.status == "error"
    assert result.errors[0].code == "calculator_not_applicable"
    assert result.errors[0].details["as_of_date"] == "2023-12-31"
    assert result.errors[0].details["as_of_source"] == "derived_from_tax_year"


def test_defaulted_today_as_of_date_feeds_applicability_check(tmp_path):
    engine = _engine(_write_pack(tmp_path, applicable_from="2999-01-01"))

    result = _calculate(engine)

    assert result.status == "error"
    assert result.errors[0].code == "calculator_not_applicable"
    assert result.errors[0].details["as_of_date"] == date.today().isoformat()
    assert result.errors[0].details["as_of_source"] == "defaulted_to_today"


def test_live_catalog_unbounded_calculation_still_succeeds():
    result = live_engine.calculate(
        CalculationRequest(
            calculator_id="business.invoice_total",
            inputs={"net_amount": 1000, "vat_rate": 0.22},
        )
    )

    assert result.status == "success"
    assert result.result["total"] == "1220.00"
