"""Renders the calculator catalog as the text block a future LLM system
prompt would embed.

This is the platform-side artifact of the planned integration ("the LLM
chooses from a catalog, extracts named inputs, calls /calculate once"):
everything the model needs to pick a calculator and name its inputs
correctly, in one deterministic, regenerable string. Nothing in the
platform consumes this — it exists so the integration contract can be
seen and tested before any real LLM code is written.
"""

from typing import Iterable

from app.schemas.calculator_definition import CalculatorDefinition


def render_catalog(definitions: Iterable[CalculatorDefinition]) -> str:
    blocks = []
    for d in definitions:
        lines = [f"### {d.id} — {d.name}"]
        if d.description:
            lines.append(d.description.strip())
        if d.keywords or d.aliases:
            lines.append("Riconoscibile da: " + ", ".join([*d.keywords, *d.aliases]))

        input_descriptions = []
        for spec in d.inputs:
            parts = [spec.type]
            if spec.unit:
                parts.append(spec.unit)
            if not spec.required:
                parts.append(f"opzionale, default {spec.default}")
            desc = f" — {spec.description}" if spec.description else ""
            input_descriptions.append(f"  - {spec.name} ({', '.join(parts)}){desc}")
        if input_descriptions:
            lines.append("Input:")
            lines.extend(input_descriptions)

        requirements = []
        if d.regime_selector:
            requirements.append(f"richiede {d.regime_selector.get('by', 'tax_year')} per selezionare il regime corretto")
        if d.requires_period:
            requirements.append("richiede period.start_date e period.end_date (YYYY-MM-DD)")
        if requirements:
            lines.append("Requisiti: " + "; ".join(requirements))
        if d.exclusions:
            lines.append("Non copre: " + "; ".join(d.exclusions))

        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)
