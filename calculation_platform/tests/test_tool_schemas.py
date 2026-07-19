import re

from fastapi.testclient import TestClient

from app.api.tool_schemas import build_all_tool_schemas, build_tool_schema
from app.main import app, engine


client = TestClient(app)


def _schema(calculator_id):
    return build_tool_schema(engine.registry.get(calculator_id))


def test_every_registered_calculator_has_unique_valid_tool_name():
    schemas = build_all_tool_schemas(engine.registry)
    assert len(schemas) == len(engine.registry.definitions())
    names = [schema["name"] for schema in schemas]
    assert len(names) == len(set(names))
    assert all(re.fullmatch(r"[a-zA-Z0-9_-]+", name) for name in names)


def test_irpef_required_decimal_input():
    schema = _schema("legal_it.irpef")["input_schema"]
    assert "taxable_income" in schema["required"]
    assert schema["properties"]["taxable_income"]["type"] == ["number", "string"]


def test_legal_interest_requires_period_with_date_fields():
    schema = _schema("legal_it.legal_interest")["input_schema"]
    assert "period" in schema["required"]
    period = schema["properties"]["period"]
    assert period["required"] == ["start_date", "end_date"]
    assert period["properties"]["start_date"]["format"] == "date"
    assert period["properties"]["end_date"]["format"] == "date"


def test_compensi_dm55_string_list_and_optional_input():
    schema = _schema("legal_it.compensi_dm55")["input_schema"]
    fasi = schema["properties"]["fasi"]
    assert fasi["type"] == "array"
    assert fasi["items"] == {"type": "string"}
    assert fasi["minItems"] == 1
    assert "aumento_pct" not in schema["required"]


def test_ravvedimento_date_inputs_have_date_format():
    properties = _schema("legal_it.ravvedimento_operoso")["input_schema"]["properties"]
    assert properties["scadenza_originaria"]["format"] == "date"
    assert properties["data_pagamento"]["format"] == "date"


def test_tool_schema_endpoints():
    all_response = client.get("/tool-schemas")
    assert all_response.status_code == 200
    assert len(all_response.json()) == len(engine.registry.definitions())

    one_response = client.get("/calculators/legal_it.irpef/tool-schema")
    assert one_response.status_code == 200
    assert one_response.json() == _schema("legal_it.irpef")

    missing_response = client.get("/calculators/does.not.exist/tool-schema")
    assert missing_response.status_code == 404
