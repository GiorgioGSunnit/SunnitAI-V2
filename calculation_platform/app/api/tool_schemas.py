"""LLM function-calling schemas generated from calculator definitions."""

import re
from typing import Any, Dict, List

from ..schemas.calculator_definition import CalculatorDefinition, InputSpec


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _input_schema(spec: InputSpec) -> Dict[str, Any]:
    type_schemas: Dict[str, Dict[str, Any]] = {
        "decimal": {"type": ["number", "string"]},
        "integer": {"type": "integer"},
        "boolean": {"type": "boolean"},
        "string": {"type": "string"},
        "date": {"type": "string", "format": "date"},
        "string_list": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        },
    }
    try:
        schema = dict(type_schemas[spec.type])
    except KeyError as exc:
        raise ValueError(f"Unsupported input type: {spec.type!r}") from exc

    if spec.description:
        description = _clean(spec.description)
        if spec.type == "date":
            description = f"{description} (YYYY-MM-DD)"
        schema["description"] = description
    elif spec.type == "date":
        schema["description"] = "Data in formato YYYY-MM-DD"

    if spec.min_value is not None:
        schema["minimum"] = spec.min_value
    if spec.max_value is not None:
        schema["maximum"] = spec.max_value
    if not spec.required and spec.default is not None:
        schema["default"] = spec.default
    return schema


def build_tool_schema(definition: CalculatorDefinition) -> dict:
    """Build an Anthropic-style tool definition for one calculator."""
    description_parts = []
    if definition.description:
        description_parts.append(_clean(definition.description))
    if definition.required_context:
        context = ", ".join(_clean(item) for item in definition.required_context)
        description_parts.append(f"Contesto necessario: {context}")
    if definition.ambiguity_notes:
        description_parts.append(f"Nota: {_clean(definition.ambiguity_notes)}")

    properties = {spec.name: _input_schema(spec) for spec in definition.inputs}
    properties["tax_year"] = {
        "type": "integer",
        "description": "Anno d'imposta",
    }
    properties["as_of_date"] = {
        "type": "string",
        "format": "date",
        "description": "Data di riferimento (YYYY-MM-DD)",
    }

    required = [spec.name for spec in definition.inputs if spec.required]
    if definition.requires_period:
        properties["period"] = {
            "type": "object",
            "properties": {
                "start_date": {"type": "string", "format": "date"},
                "end_date": {"type": "string", "format": "date"},
            },
            "required": ["start_date", "end_date"],
            "additionalProperties": False,
        }
        required.append("period")

    return {
        "name": definition.id.replace(".", "__"),
        "calculator_id": definition.id,
        "description": " ".join(description_parts),
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    }


def build_all_tool_schemas(registry) -> List[dict]:
    """Build tool definitions for every calculator in registry order."""
    return [build_tool_schema(definition) for definition in registry.definitions()]
