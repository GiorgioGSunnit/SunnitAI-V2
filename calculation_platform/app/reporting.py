import html
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .schemas.calculator_definition import CalculatorDefinition
from .schemas.stored_calculation import StoredCalculation


def render_report_html(
    stored: StoredCalculation,
    definition: Optional[CalculatorDefinition],
) -> str:
    result = stored.result or {}
    request = stored.request or {}
    calculator_name = definition.name if definition else stored.calculator_id
    formula_version = result.get("formula_version") or (definition.version if definition else None)

    return "\n".join(
        [
            "<!DOCTYPE html>",
            '<html lang="it">',
            "<head>",
            '<meta charset="utf-8">',
            f"<title>{_e(calculator_name)} - report { _e(stored.request_id) }</title>",
            "<style>",
            _css(),
            "</style>",
            "</head>",
            "<body>",
            "<main>",
            "<header>",
            "<p class=\"eyebrow\">Report calcolo</p>",
            f"<h1>{_e(calculator_name)}</h1>",
            "<dl class=\"meta\">",
            _term("ID calcolatore", stored.calculator_id),
            _term("formula_version", formula_version or "non registrata"),
            _term("request_id", stored.request_id),
            _term("created_at", stored.created_at),
            _term("Risoluzione data", _date_resolution_line(result.get("date_resolution"))),
            "</dl>",
            "</header>",
            _section("Esito", _render_outcome(result)),
            _section("Dati inseriti", _render_inputs(result)),
            _section("Parametri applicati", _render_parameters(result.get("parameters_used", {}))),
            _section("Svolgimento", _render_steps(result.get("steps", []))),
            _section("Avvertenze / Assunzioni", _render_notices(result)),
            _section("Non incluso", _render_exclusions(result, definition)),
            _section("Fonti", _render_citations(result.get("citations", []))),
            "<footer>",
            "<p>Documento generato da un calcolatore deterministico secondo i dati, i parametri e le assunzioni indicati; non costituisce parere legale.</p>",
            f"<p>Generato il {_e(_now_iso())}</p>",
            f"<p class=\"small\">Richiesta archiviata: {_json_block(request)}</p>",
            "</footer>",
            "</main>",
            "</body>",
            "</html>",
        ]
    )


def _css() -> str:
    return """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #1b1f24; margin: 0; background: #f6f8fa; }
main { max-width: 980px; margin: 0 auto; padding: 32px 24px 48px; background: #fff; }
header { border-bottom: 2px solid #1b1f24; padding-bottom: 18px; margin-bottom: 22px; }
.eyebrow { text-transform: uppercase; font-size: 12px; letter-spacing: .08em; color: #57606a; margin: 0 0 6px; }
h1 { margin: 0 0 14px; font-size: 30px; }
h2 { margin: 26px 0 10px; font-size: 19px; border-bottom: 1px solid #d0d7de; padding-bottom: 6px; }
h3 { margin: 16px 0 8px; font-size: 15px; }
.meta { display: grid; grid-template-columns: 180px 1fr; gap: 6px 16px; margin: 0; }
dt { font-weight: 700; color: #24292f; }
dd { margin: 0; overflow-wrap: anywhere; }
table { width: 100%; border-collapse: collapse; margin: 8px 0 12px; }
th, td { border: 1px solid #d0d7de; padding: 7px 9px; vertical-align: top; text-align: left; }
th { background: #f6f8fa; font-weight: 700; }
.split { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
pre { white-space: pre-wrap; overflow-wrap: anywhere; background: #f6f8fa; border: 1px solid #d0d7de; padding: 10px; margin: 0; }
ul { margin-top: 8px; padding-left: 22px; }
li { margin: 5px 0; }
.small { color: #57606a; font-size: 12px; }
footer { border-top: 1px solid #d0d7de; margin-top: 28px; padding-top: 14px; font-size: 13px; }
@media print {
  body { background: #fff; color: #000; }
  main { max-width: none; padding: 0; }
  a, button { display: none; }
  th { background: #fff; }
  header, h2, footer { break-inside: avoid; }
}
"""


def _section(title: str, body: str) -> str:
    return f"<section><h2>{_e(title)}</h2>{body}</section>"


def _term(label: str, value: Any) -> str:
    return f"<dt>{_e(label)}</dt><dd>{_e(value)}</dd>"


def _render_outcome(result: Dict[str, Any]) -> str:
    if result.get("status") == "error":
        return _table_from_mapping("Errore", {"errors": result.get("errors", [])})
    return _table_from_mapping("Risultato", result.get("result", {}))


def _render_inputs(result: Dict[str, Any]) -> str:
    return (
        '<div class="split">'
        "<div><h3>Dati ricevuti</h3>"
        + _table_from_mapping("raw_inputs", result.get("raw_inputs", {}))
        + "</div><div><h3>Dati usati</h3>"
        + _table_from_mapping("inputs_used", result.get("inputs_used", {}))
        + "</div></div>"
    )


def _render_parameters(parameters: Dict[str, Any]) -> str:
    if not parameters:
        return "<p>Nessun parametro registrato.</p>"

    rows = []
    for name, parameter in parameters.items():
        if isinstance(parameter, dict):
            citations = parameter.get("citations") or []
            effective_from = parameter.get("effective_from") or ""
            effective_to = parameter.get("effective_to") or ""
            effective_range = f"{effective_from} - {effective_to or 'aperto'}".strip()
            rows.append(
                "<tr>"
                f"<td>{_e(name)}</td>"
                f"<td>{_json_inline(parameter.get('value'))}</td>"
                f"<td>{_e(parameter.get('origin', ''))}</td>"
                f"<td>{_e(effective_range)}</td>"
                f"<td>{_e(parameter.get('source', ''))}</td>"
                f"<td>{_e(_citations_inline(citations))}</td>"
                f"<td>{_e(parameter.get('last_verified_at') or 'non registrata')}</td>"
                "</tr>"
            )
        else:
            rows.append(
                "<tr>"
                f"<td>{_e(name)}</td>"
                f"<td>{_json_inline(parameter)}</td>"
                "<td></td><td></td><td></td><td></td><td></td>"
                "</tr>"
            )
    return (
        "<table><thead><tr>"
        "<th>Parametro</th><th>Valore</th><th>Origine</th><th>Intervallo efficace</th>"
        "<th>Fonte</th><th>Citazioni</th><th>Verifica</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _render_steps(steps: List[Dict[str, Any]]) -> str:
    if not steps:
        return "<p>Nessun passaggio registrato.</p>"
    rows = []
    for index, step in enumerate(steps, start=1):
        step_number = step.get("step", index) if isinstance(step, dict) else index
        step_type = step.get("type", "") if isinstance(step, dict) else ""
        remaining = {
            key: value
            for key, value in (step.items() if isinstance(step, dict) else [])
            if key not in {"step", "type"}
        }
        rows.append(
            "<tr>"
            f"<td>{_e(step_number)}</td>"
            f"<td>{_e(step_type)}</td>"
            f"<td>{_json_block(remaining)}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Step</th><th>Tipo</th><th>Dettagli</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _render_notices(result: Dict[str, Any]) -> str:
    notices = []
    for key in ("warnings", "assumptions"):
        for item in result.get(key, []) or []:
            if isinstance(item, dict):
                code = item.get("code", key)
                message = item.get("message", item)
            else:
                code = key
                message = item
            notices.append(f"<li><strong>{_e(code)}</strong>: {_e(message)}</li>")
    if not notices:
        return "<p>Nessuna avvertenza o assunzione registrata.</p>"
    return "<ul>" + "".join(notices) + "</ul>"


def _render_exclusions(
    result: Dict[str, Any], definition: Optional[CalculatorDefinition]
) -> str:
    """What the calculation explicitly leaves out, as its own section.

    Read from the stored result first so an archived report keeps showing
    the exclusions that were in force when it ran; the live definition is
    only a fallback for calculations stored before results carried them.
    Never folded into the generic warnings list: "does not include VAT" is
    a scope boundary, not a caveat about the number's reliability.
    """
    # `is None` and not falsiness: a stored empty list is a real statement
    # ("this calculator excluded nothing when it ran") and must not be
    # overwritten with whatever the definition happens to declare today.
    exclusions = result.get("exclusions")
    if exclusions is None and definition is not None:
        exclusions = definition.exclusions
    if not exclusions:
        return "<p>Nessuna esclusione dichiarata da questo calcolatore.</p>"
    return "<ul>" + "".join(f"<li>{_e(item)}</li>" for item in exclusions) + "</ul>"


def _render_citations(citations: List[Dict[str, Any]]) -> str:
    if not citations:
        return "<p>Nessuna fonte registrata.</p>"
    rows = []
    for citation in citations:
        rows.append(
            "<tr>"
            f"<td>{_e(citation.get('reference', ''))}</td>"
            f"<td>{_e(citation.get('source_name', ''))}</td>"
            f"<td>{_e(citation.get('publisher', ''))}</td>"
            f"<td>{_e(citation.get('publication_date', ''))}</td>"
            f"<td>{_e(citation.get('url', ''))}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Riferimento</th><th>Fonte</th><th>Editore</th>"
        "<th>Data</th><th>URL</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _table_from_mapping(label: str, mapping: Any) -> str:
    if not mapping:
        return f"<p>{_e(label)} non registrati.</p>"
    if not isinstance(mapping, dict):
        return f"<pre>{_json_block(mapping)}</pre>"
    rows = [
        f"<tr><th>{_e(key)}</th><td>{_json_inline(value)}</td></tr>"
        for key, value in mapping.items()
    ]
    return "<table><tbody>" + "".join(rows) + "</tbody></table>"


def _date_resolution_line(date_resolution: Any) -> str:
    if not isinstance(date_resolution, dict) or not date_resolution:
        return "Data di riferimento non applicabile o non registrata."
    as_of_date = date_resolution.get("as_of_date", "non registrata")
    source = date_resolution.get("source", "origine non registrata")
    extras = {
        key: value
        for key, value in date_resolution.items()
        if key not in {"as_of_date", "source"}
    }
    if extras:
        return f"as_of_date {as_of_date}; origine {source}; dettagli {json.dumps(extras, ensure_ascii=False, default=str)}"
    return f"as_of_date {as_of_date}; origine {source}"


def _citations_inline(citations: List[Dict[str, Any]]) -> str:
    parts = []
    for citation in citations:
        reference = citation.get("reference", "")
        source_name = citation.get("source_name", "")
        parts.append(" - ".join(part for part in (reference, source_name) if part))
    return "; ".join(parts)


def _json_inline(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return _json_block(value)
    return _e(value)


def _json_block(value: Any) -> str:
    return _e(json.dumps(value, ensure_ascii=False, default=str, indent=2))


def _e(value: Any) -> str:
    if value is None:
        return ""
    return html.escape(str(value), quote=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
