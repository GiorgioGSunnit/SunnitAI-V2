from datetime import date

from app.core.engine import CalculationEngine
from app.resolvers.parameter_store import ParameterStore
from app.schemas.calculation_request import CalculationRequest
from app.schemas.calculator_definition import CalculatorDefinition, ParameterRef
from scripts.verify_citations import verify_parameters


class SingleDefinitionRegistry:
    def __init__(self, definition):
        self.definition = definition

    def get(self, calculator_id):
        return self.definition

    def is_released(self, calculator_id):
        return True


def _write_parameter_file(tmp_path, last_verified_at=None):
    verified_line = f"    last_verified_at: {last_verified_at}\n" if last_verified_at else ""
    parameters_dir = tmp_path / "parameters"
    parameters_dir.mkdir()
    (parameters_dir / "rates.yml").write_text(
        "values:\n"
        "  - parameter_id: test.official_rate\n"
        "    value: 0.10\n"
        "    unit: rate\n"
        "    effective_from: 2026-01-01\n"
        "    effective_to: null\n"
        "    source: Test law\n"
        "    official: true\n"
        f"{verified_line}"
        "    citations:\n"
        "      - reference: Test citation\n"
        "        url: https://example.test/law\n"
        "        official: true\n",
        encoding="utf-8",
    )
    return parameters_dir


def _engine_for(parameters_dir, stale_after_days=365):
    definition = CalculatorDefinition(
        id="test.calc",
        name="Test calculator",
        category="test",
        strategy="expression",
        parameters=[ParameterRef(name="rate", parameter_id="test.official_rate")],
        formula={"expression": "rate * 100"},
        output={"name": "amount", "round_to": 2},
    )
    return CalculationEngine(
        SingleDefinitionRegistry(definition),
        ParameterStore(parameters_dir),
        parameter_verification_stale_after_days=stale_after_days,
    )


def test_verify_parameters_stamps_reachable_url_entries(tmp_path):
    parameters_dir = _write_parameter_file(tmp_path)

    def fetcher(url, timeout):
        return True, "200 HEAD"

    stats = verify_parameters(parameters_dir, verified_at="2026-07-13", fetcher=fetcher)

    assert stats.checked_entries == 1
    assert stats.checked_urls == 1
    assert stats.verified_entries == 1
    assert "last_verified_at: 2026-07-13" in (parameters_dir / "rates.yml").read_text(encoding="utf-8")


def test_verify_parameters_does_not_stamp_failed_entries(tmp_path):
    parameters_dir = _write_parameter_file(tmp_path)

    def fetcher(url, timeout):
        return False, "HTTP 500"

    stats = verify_parameters(parameters_dir, verified_at="2026-07-13", fetcher=fetcher)

    assert stats.failed_entries == 1
    assert "last_verified_at" not in (parameters_dir / "rates.yml").read_text(encoding="utf-8")


def test_engine_warns_when_official_parameter_has_no_verification_stamp(tmp_path):
    engine = _engine_for(_write_parameter_file(tmp_path))
    request = CalculationRequest(
        calculator_id="test.calc",
        inputs={},
        as_of_date=date(2026, 7, 13),
    )

    result = engine.calculate(request)

    assert result.status == "success"
    assert result.parameters_used["rate"]["last_verified_at"] is None
    assert any(w.code == "parameter_verification_missing" for w in result.warnings)


def test_engine_warns_when_official_parameter_verification_is_stale(tmp_path):
    engine = _engine_for(_write_parameter_file(tmp_path, last_verified_at="2000-01-01"), stale_after_days=30)
    request = CalculationRequest(
        calculator_id="test.calc",
        inputs={},
        as_of_date=date(2026, 7, 13),
    )

    result = engine.calculate(request)

    assert result.status == "success"
    assert result.parameters_used["rate"]["last_verified_at"] == "2000-01-01"
    assert any(w.code == "parameter_verification_stale" for w in result.warnings)
