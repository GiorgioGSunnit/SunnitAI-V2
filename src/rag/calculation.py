"""Deterministic HTTP bridge from the RAG graph to the calculation platform."""

from __future__ import annotations

import copy
import json
import logging
import os
import re
import threading
import time
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

from . import normalization

logger = logging.getLogger(__name__)

MATCH_AUTO_ROUTE_MIN_SCORE = 3
_MAX_CLARIFICATION_ROUNDS = 3
_PLATFORM_UNAVAILABLE = {
    "status": "no_match",
    "candidates": [],
    "platform_unavailable": True,
}

# Cache successful calculator tool schemas for five minutes per process.
_TOOL_SCHEMA_CACHE: Dict[tuple[str, str], tuple[Dict[str, Any], float]] = {}
_TOOL_SCHEMA_CACHE_LOCK = threading.Lock()
_TOOL_SCHEMA_CACHE_TTL = 300


class PlatformClient:
    """Small fail-safe HTTP client for the separate calculation service."""

    def __init__(self, base_url: Optional[str] = None, timeout: float = 2.0):
        self.base_url = (
            base_url or os.getenv("CALC_PLATFORM_URL", "http://localhost:8802")
        ).rstrip("/")
        self.timeout = timeout

    def match(self, query: str) -> Dict[str, Any]:
        return self._post("/match", {"query": query})

    def calculate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._post("/calculate", payload)

    def tool_schema(self, calculator_id: str) -> Dict[str, Any]:
        cache_key = (self.base_url, calculator_id)
        with _TOOL_SCHEMA_CACHE_LOCK:
            cached = _TOOL_SCHEMA_CACHE.get(cache_key)
            if cached and (time.time() - cached[1]) < _TOOL_SCHEMA_CACHE_TTL:
                return cached[0]

        schema = self._get(f"/calculators/{calculator_id}/tool-schema")
        if not schema.get("platform_unavailable"):
            with _TOOL_SCHEMA_CACHE_LOCK:
                _TOOL_SCHEMA_CACHE[cache_key] = (schema, time.time())
        return schema

    def _get(self, path: str) -> Dict[str, Any]:
        try:
            response = requests.get(f"{self.base_url}{path}", timeout=self.timeout)
            response.raise_for_status()
            body = response.json()
            if not isinstance(body, dict):
                raise ValueError("calculation platform returned a non-object response")
            return body
        except Exception as exc:
            logger.warning("Calculation platform request failed for %s: %s", path, exc)
            return dict(_PLATFORM_UNAVAILABLE)

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            response = requests.post(
                f"{self.base_url}{path}", json=payload, timeout=self.timeout
            )
            response.raise_for_status()
            body = response.json()
            if not isinstance(body, dict):
                raise ValueError("calculation platform returned a non-object response")
            return body
        except requests.HTTPError as exc:
            response = exc.response
            if response is not None and 400 <= response.status_code < 500:
                message = _short_response_body(response)
                logger.warning(
                    "Calculation platform rejected request for %s: %s", path, message
                )
                return {
                    "status": "error",
                    "errors": [{"code": "request_invalid", "message": message}],
                }
            logger.warning("Calculation platform request failed for %s: %s", path, exc)
            return dict(_PLATFORM_UNAVAILABLE)
        except Exception as exc:
            logger.warning("Calculation platform request failed for %s: %s", path, exc)
            return dict(_PLATFORM_UNAVAILABLE)


def _short_response_body(response: requests.Response, limit: int = 500) -> str:
    """Return a compact, bounded error message from an HTTP response body."""
    text = ""
    try:
        body = response.json()
    except (TypeError, ValueError):
        body = None

    if isinstance(body, dict):
        detail = next(
            (body.get(key) for key in ("detail", "message", "error") if body.get(key)),
            body,
        )
        text = (
            detail
            if isinstance(detail, str)
            else json.dumps(detail, ensure_ascii=False, sort_keys=True)
        )
    elif body is not None:
        text = (
            body
            if isinstance(body, str)
            else json.dumps(body, ensure_ascii=False, sort_keys=True)
        )
    else:
        text = str(getattr(response, "text", ""))

    text = " ".join(text.split())
    if not text:
        text = f"HTTP {getattr(response, 'status_code', 'request error')}"
    if len(text) > limit:
        return f"{text[: limit - 3]}..."
    return text


def _tied_top_candidates(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Candidates sharing the top score of an ambiguous match."""
    candidates = [c for c in response.get("candidates") or [] if isinstance(c, dict)]
    if len(candidates) < 2:
        return []
    top_score = candidates[0].get("score", 0)
    return [c for c in candidates if c.get("score") == top_score]


def calculation_gate(state: Dict[str, Any]) -> Dict[str, Any]:
    """Route a clear, high-scoring platform match to calculation.

    An ambiguous match is escalated to the user ONLY when every tied
    candidate would have auto-routed on its own. That is the case where
    silence is the worse failure: the request is unmistakably a
    calculation, the platform simply cannot tell which one, and dropping
    back to document retrieval answers a question nobody asked. A weak tie
    (two calculators scraping one incidental token each) stays on the
    normal route — prompting there would turn every passing mention of a
    legal topic into a menu.

    A retrieval-only turn (`skip_calculation`) never reaches the platform at
    all. The entry router already bypasses this node, so this check is the
    second line of that defence: it keeps the guarantee inside the node that
    would otherwise do the intercepting, rather than resting on the wiring
    alone.
    """
    try:
        if state.get("skip_calculation"):
            logger.info("calc_gate: route=normal reason=skip_calculation")
            return {"calc_route": "normal"}
        response = PlatformClient().match(state.get("query", ""))
        candidates = response.get("candidates") or []
        top = candidates[0] if candidates and isinstance(candidates[0], dict) else None
        score = top.get("score", 0) if top else 0
        strong = isinstance(score, (int, float)) and score >= MATCH_AUTO_ROUTE_MIN_SCORE
        if response.get("status") == "matched" and strong:
            logger.info(
                "calc_gate: route=calculate calculator=%s score=%s status=%s",
                top.get("calculator_id"), score, response.get("status"),
            )
            return {"calc_route": "calculate", "calculation_match": top}
        if response.get("status") == "ambiguous" and strong:
            tied = _tied_top_candidates(response)
            if len(tied) > 1:
                logger.info(
                    "calc_gate: route=choose candidates=%s score=%s",
                    [c.get("calculator_id") for c in tied], score,
                )
                return {
                    "calc_route": "calculate",
                    "calculation_match": None,
                    "calculation_choices": tied,
                }
        logger.info(
            "calc_gate: route=normal top=%s score=%s status=%s",
            top.get("calculator_id") if top else None,
            score,
            response.get("status"),
        )
    except Exception:
        logger.exception("Calculation gate failed; continuing through the normal RAG route")
    return {"calc_route": "normal"}


def route_after_gate(state: Dict[str, Any]) -> str:
    try:
        return "calculate" if state.get("calc_route") == "calculate" else "normal"
    except Exception:
        logger.exception("Calculation gate router failed; using the normal RAG route")
        return "normal"


def route_after_calculation(state: Dict[str, Any]) -> str:
    """Continue through normal RAG only when calculation explicitly falls back."""
    try:
        return "fallback" if state.get("calc_route") == "normal" else "end"
    except Exception:
        logger.exception("Calculation result router failed; ending safely")
        return "end"


# Offline/failure fallback tier when LLM argument extraction is unavailable.
_DATE_RE = re.compile(r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}[-/]\d{1,2}[-/]\d{4})\b")
_NUMBER_RE = re.compile(
    r"(?<![\w])[-+]?(?:\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+(?:[.,]\d+)?)\s*%?"
)


def _normalize_date(raw: str) -> str:
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
        return datetime.strptime(raw, "%Y-%m-%d").date().isoformat()
    separator = "/" if "/" in raw else "-"
    return datetime.strptime(raw, f"%d{separator}%m{separator}%Y").date().isoformat()


def _normalize_number(raw: str, spec: Dict[str, Any]) -> Any:
    token = raw.strip()
    is_percentage = token.endswith("%")
    token = token.rstrip("%").strip()
    if "," in token:
        token = token.replace(".", "").replace(",", ".")
    elif token.count(".") > 1:
        token = token.replace(".", "")

    value = Decimal(token)
    name = str(spec.get("name", ""))
    description = str(spec.get("description", "")).lower()
    if is_percentage and not name.endswith("_pct"):
        if spec.get("unit") == "rate" or "fraction" in description or "aliquota" in name:
            value /= Decimal("100")
    if spec.get("type") == "integer":
        return int(value)
    return format(value, "f")


def _extract_values(
    query: str,
    specs: Iterable[Dict[str, Any]],
    *,
    supports_tax_year: bool = False,
) -> Dict[str, Any]:
    """Extract positional typed values for the supplied platform input specs."""
    specs = [spec for spec in specs if isinstance(spec, dict) and spec.get("name")]
    values: Dict[str, Any] = {}
    masked = list(query)
    dates: List[str] = []
    for match in _DATE_RE.finditer(query):
        try:
            dates.append(_normalize_date(match.group(0)))
        except ValueError:
            continue
        masked[match.start():match.end()] = " " * (match.end() - match.start())

    date_specs = [spec for spec in specs if spec.get("type") == "date"]
    period_spec = next((spec for spec in specs if spec.get("type") == "period"), None)
    if period_spec and len(dates) >= 2:
        values[period_spec["name"]] = {
            "start_date": dates.pop(0),
            "end_date": dates.pop(0),
        }
    for spec, value in zip(date_specs, dates):
        values[spec["name"]] = value

    number_tokens = [match.group(0) for match in _NUMBER_RE.finditer("".join(masked))]
    if supports_tax_year:
        for index, raw in enumerate(number_tokens):
            year_text = raw.strip().rstrip("%").strip()
            if re.fullmatch(r"(?:19|20)\d{2}", year_text):
                values["tax_year"] = int(year_text)
                number_tokens.pop(index)
                break

    number_specs = [
        spec for spec in specs if spec.get("type") in {"decimal", "integer"}
    ]
    for spec, raw in zip(number_specs, number_tokens):
        try:
            values[spec["name"]] = _normalize_number(raw, spec)
        except (InvalidOperation, ValueError):
            logger.debug("Ignored invalid extracted number %r", raw)
    return values


def _extraction_messages(
    query: str,
    input_schema: Dict[str, Any],
    *,
    missing_specs: Optional[Iterable[Dict[str, Any]]] = None,
    prior_inputs: Optional[Dict[str, Any]] = None,
    frequency_note: str = "",
):
    from langchain_core.messages import HumanMessage, SystemMessage

    today = datetime.now().date().isoformat()
    system_prompt = (
        "Extract calculator arguments from the user's message. Extract ONLY values "
        "the user explicitly stated; never invent or guess a value, and omit unknown "
        "arguments entirely. Normalize Italian numbers (for example 1.200,50 becomes "
        "1200.50), and normalize percentages for rate-like fields as the property's "
        "schema description implies. Return dates as YYYY-MM-DD. Today's date is "
        f"{today}; use it only to resolve explicit relative references such as "
        "'quest'anno'. If the message contains no extractable values, use an empty "
        "arguments object. Your arguments must conform to this JSON schema:\n"
        f"{json.dumps(input_schema, ensure_ascii=False, sort_keys=True)}"
    )
    if frequency_note:
        system_prompt = f"{system_prompt}\n\n{frequency_note}"
    context = []
    if prior_inputs is not None:
        context.append(
            "Already collected: "
            + json.dumps(prior_inputs, ensure_ascii=False, sort_keys=True)
        )
    if missing_specs is not None:
        context.append(
            "Still missing: "
            + json.dumps(list(missing_specs), ensure_ascii=False, sort_keys=True)
        )
    human_prompt = query
    if context:
        human_prompt = f"{'\n'.join(context)}\n\nUser message: {query}"
    return [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]


def _relax_schema_node(node: Any) -> None:
    """Drop `required` in place, at every depth of a JSON schema."""
    if not isinstance(node, dict):
        return
    node.pop("required", None)
    properties = node.get("properties")
    if isinstance(properties, dict):
        for child_schema in properties.values():
            _relax_schema_node(child_schema)
    _relax_schema_node(node.get("items"))


def _relaxed_extraction_schema(input_schema: Dict[str, Any]) -> Dict[str, Any]:
    """Copy the platform schema while allowing partial LLM extraction output.

    `required` is dropped at every depth, not only at the top level and on
    `period`: an object_list (comparator candidates) carries its own required
    list inside `items`, and leaving that in place forces the model to invent
    fields for an offer the user described only partially — the opposite of
    the "extract ONLY explicit values" instruction it is given. The unrelaxed
    schema still drives cleaning, so half-filled objects are pruned there.
    """
    extraction_schema = copy.deepcopy(input_schema)
    _relax_schema_node(extraction_schema)
    return extraction_schema


_DROP_EXTRACTED_VALUE = object()


def _clean_extracted_value(value: Any, schema: Dict[str, Any]) -> Any:
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    if value is None:
        return _DROP_EXTRACTED_VALUE

    schema_type = schema.get("type")
    is_array = schema_type == "array" or (
        isinstance(schema_type, list) and "array" in schema_type
    )
    if is_array and isinstance(value, list):
        return _clean_extracted_list(value, schema)

    is_object = schema_type == "object" or (
        isinstance(schema_type, list) and "object" in schema_type
    )
    if not is_object or not isinstance(value, dict):
        return value

    nested_properties = schema.get("properties")
    if not isinstance(nested_properties, dict):
        return value

    cleaned = {}
    for name, nested_value in value.items():
        nested_schema = nested_properties.get(name)
        if not isinstance(nested_schema, dict):
            continue
        nested_cleaned = _clean_extracted_value(nested_value, nested_schema)
        if nested_cleaned is not _DROP_EXTRACTED_VALUE:
            cleaned[name] = nested_cleaned

    required = schema.get("required") or []
    if any(name not in cleaned for name in required):
        return _DROP_EXTRACTED_VALUE
    return cleaned


def _clean_extracted_list(value: List[Any], schema: Dict[str, Any]) -> Any:
    """Prune candidate objects the model could not fill completely.

    Each item of an object_list (comparator candidates) is validated against
    the platform's own item schema, whose `required` list is intact because
    cleaning runs on the unrelaxed schema. Dropping a half-extracted offer
    here turns what would be a confusing per-field rejection from /calculate
    into the ordinary missing-input clarification for the list itself. Arrays
    of non-objects (string_list) are left exactly as the model returned them.
    """
    items_schema = schema.get("items")
    if not isinstance(items_schema, dict):
        return value
    item_type = items_schema.get("type")
    is_object_item = item_type == "object" or (
        isinstance(item_type, list) and "object" in item_type
    )
    if not is_object_item:
        return value

    cleaned_items = []
    for item in value:
        cleaned_item = _clean_extracted_value(item, items_schema)
        # A non-dict survivor means the model put a bare scalar where an
        # offer object belongs — unusable to the comparator, so drop it too.
        if cleaned_item is _DROP_EXTRACTED_VALUE or not isinstance(cleaned_item, dict):
            continue
        cleaned_items.append(cleaned_item)
    if not cleaned_items:
        return _DROP_EXTRACTED_VALUE
    return cleaned_items


def _clean_extracted_values(
    arguments: Any, properties: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    if hasattr(arguments, "model_dump"):
        arguments = arguments.model_dump()
    if not isinstance(arguments, dict):
        return None
    cleaned = {}
    for name, value in arguments.items():
        property_schema = properties.get(name)
        if not isinstance(property_schema, dict):
            continue
        cleaned_value = _clean_extracted_value(value, property_schema)
        if cleaned_value is not _DROP_EXTRACTED_VALUE:
            cleaned[name] = cleaned_value
    return cleaned


def _extract_values_llm(
    query: str,
    calculator_id: str,
    missing_specs: Optional[Iterable[Dict[str, Any]]] = None,
    prior_inputs: Optional[Dict[str, Any]] = None,
    keep_partial_items: bool = False,
) -> Optional[Dict[str, Any]]:
    """Extract explicit calculator arguments through tools, then JSON mode.

    `keep_partial_items` turns off the pruning of half-filled object_list
    entries. Pruning is right for a one-shot call, where an incomplete
    offer would only produce a confusing per-field rejection from
    /calculate — but wrong during incremental candidate collection, where
    a partially described offer is exactly what the next question is
    about, and throwing it away would make the user repeat what they just
    said.
    """
    try:
        tool_schema = PlatformClient().tool_schema(calculator_id)
        input_schema = tool_schema.get("input_schema")
        properties = input_schema.get("properties") if isinstance(input_schema, dict) else None
        tool_name = tool_schema.get("name")
        if not isinstance(properties, dict) or not isinstance(tool_name, str):
            logger.warning("No usable tool schema for calculator %s", calculator_id)
            return None
        extraction_schema = _relaxed_extraction_schema(input_schema)
        if keep_partial_items:
            properties = extraction_schema.get("properties") or properties

        tool = {
            "name": tool_name,
            "description": str(tool_schema.get("description") or "Extract calculator inputs"),
            "parameters": extraction_schema,
        }
        frequency_fields = normalization.frequency_sensitive_fields(calculator_id)
        messages = _extraction_messages(
            query,
            extraction_schema,
            missing_specs=missing_specs,
            prior_inputs=prior_inputs,
            frequency_note=(
                normalization.extraction_prompt_note(frequency_fields)
                if frequency_fields
                else ""
            ),
        )
    except Exception as exc:
        logger.warning("Could not prepare LLM extraction for %s: %s", calculator_id, exc)
        return None

    try:
        from .ai_chat import _call_chat_with_tools

        response = _call_chat_with_tools(
            messages,
            [tool],
            tool_choice=tool_name,
            max_tokens=1000,
        )
        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            raise ValueError("LLM returned no tool calls")
        matching_calls = [
            call
            for call in tool_calls
            if isinstance(call, dict) and call.get("name") == tool_name
        ]
        if not matching_calls:
            raise ValueError(f"LLM returned no {tool_name!r} tool call")
        arguments = {}
        for call in matching_calls:
            call_arguments = call.get("args")
            if isinstance(call_arguments, str):
                call_arguments = json.loads(call_arguments)
            if hasattr(call_arguments, "model_dump"):
                call_arguments = call_arguments.model_dump()
            if not isinstance(call_arguments, dict):
                raise ValueError("LLM returned non-object tool arguments")
            arguments.update(call_arguments)
        cleaned = _clean_extracted_values(arguments, properties)
        if cleaned is None:
            raise ValueError("LLM returned non-object tool arguments")
        return cleaned
    except Exception as exc:
        logger.warning(
            "Tool-call extraction failed for %s; retrying with JSON mode: %s",
            calculator_id,
            exc,
        )

    try:
        from .ai_chat import chat_model

        structured_model = chat_model.bind(max_tokens=1000).with_structured_output(
            extraction_schema,
            method="json_mode",
        )
        arguments = structured_model.invoke(messages)
        cleaned = _clean_extracted_values(arguments, properties)
        if cleaned is None:
            raise ValueError("LLM returned non-object JSON arguments")
        return cleaned
    except Exception as exc:
        logger.warning("JSON-mode extraction failed for %s: %s", calculator_id, exc)
        return None


_COPY = {
    "it": {
        "result": "Risultato",
        "sources": "Fonti",
        "no_sources": "nessuna fonte indicata dalla piattaforma",
        "clarify": "Per completare il calcolo, puoi indicarmi {items}?",
        "failure": "Non riesco a completare il calcolo in questo momento. Riprova tra poco.",
        "round_limit": "Mi dispiace, non sono riuscito a raccogliere tutti i dati necessari per il calcolo.",
        "warnings": "Avvisi",
        "assumptions": "Assunzioni",
        "exclusions": "Non incluso",
        "defaults": "Valori assunti per default",
        "methodology": "Come e stato calcolato",
        "disclaimer": "Stima indicativa: non sostituisce la verifica di un professionista.",
        "conversions": "Conversioni applicate",
        "ask_frequency": (
            "L'importo di {amount} che hai indicato per {field} e mensile o annuo? "
            "Puoi rispondere 'mensile' oppure 'annuo'."
        ),
        "ask_currency": (
            "Questo calcolo e in euro e non applico conversioni di valuta: "
            "l'importo di {amount} che hai indicato per {field} e in {stated}. "
            "Puoi indicarmi il canone in euro?"
        ),
        "ask_currency_ambiguous": (
            "Non ho capito in quale valuta e espresso il {field}: questo calcolo "
            "e in euro e non applico conversioni di valuta. "
            "Puoi indicarmi il {field} in euro?"
        ),
        "ask_normalization_failed": (
            "Non riesco a interpretare con certezza l'importo del {field}. "
            "Puoi indicarmi il {field} annuo in euro?"
        ),
        "month": "mese",
        "year": "anno",
    },
    "es": {
        "result": "Resultado",
        "sources": "Fuentes",
        "no_sources": "ninguna fuente indicada por la plataforma",
        "clarify": "Para completar el cálculo, ¿puedes indicarme {items}?",
        "failure": "No puedo completar el cálculo en este momento. Inténtalo de nuevo más tarde.",
        "round_limit": "Lo siento, no he podido recopilar todos los datos necesarios para el cálculo.",
        "warnings": "Avisos",
        "assumptions": "Supuestos",
        "exclusions": "No incluido",
        "defaults": "Valores asumidos por defecto",
        "methodology": "Como se ha calculado",
        "disclaimer": "Estimación indicativa: no sustituye la verificación de un profesional.",
        "conversions": "Conversiones aplicadas",
        "ask_frequency": (
            "El importe de {amount} que has indicado para {field} es mensual o anual? "
            "Puedes responder 'mensual' o 'anual'."
        ),
        "ask_currency": (
            "Este cálculo es en euros y no aplico conversiones de divisa: "
            "el importe de {amount} que has indicado para {field} está en {stated}. "
            "¿Puedes indicarme el importe en euros?"
        ),
        "ask_currency_ambiguous": (
            "No he entendido en qué divisa está expresado el {field}: este cálculo "
            "es en euros y no aplico conversiones de divisa. "
            "¿Puedes indicarme el {field} en euros?"
        ),
        "ask_normalization_failed": (
            "No consigo interpretar con certeza el importe del {field}. "
            "¿Puedes indicarme el {field} anual en euros?"
        ),
        "month": "mes",
        "year": "año",
    },
    "en": {
        "result": "Result",
        "sources": "Sources",
        "no_sources": "no sources supplied by the platform",
        "clarify": "To complete the calculation, could you provide {items}?",
        "failure": "I cannot complete the calculation right now. Please try again shortly.",
        "round_limit": "Sorry, I could not collect all the information needed for the calculation.",
        "warnings": "Warnings",
        "assumptions": "Assumptions",
        "exclusions": "Not included",
        "defaults": "Values assumed by default",
        "methodology": "How it was computed",
        "disclaimer": "Indicative estimate: it does not replace a professional's review.",
        "conversions": "Conversions applied",
        "ask_frequency": (
            "Is the {amount} you gave for {field} monthly or annual? "
            "You can reply 'monthly' or 'annual'."
        ),
        "ask_currency": (
            "This calculation is in euro and I do not apply currency conversion: "
            "the {amount} you gave for {field} is in {stated}. "
            "Could you give me the amount in euro?"
        ),
        "ask_currency_ambiguous": (
            "I could not tell which currency the {field} is in: this calculation "
            "is in euro and I do not apply currency conversion. "
            "Could you give me the {field} in euro?"
        ),
        "ask_normalization_failed": (
            "I cannot reliably interpret the {field} amount. "
            "Could you give me the annual {field} in euro?"
        ),
        "month": "month",
        "year": "year",
    },
}

# Copy for the comparison flow: candidate collection, the review/confirm
# steps, and the verdict itself. Kept aligned across the three languages
# production supports — a user collecting offers in Spanish must not drop
# into Italian at the one point where the answer says whether there is a
# winner at all.
_COMPARISON_COPY = {
    "it": {
        "choose": "La richiesta può corrispondere a più calcoli. Quale intendi?",
        "choose_hint": "Rispondi col numero, oppure col nome del calcolo.",
        "choose_unclear": "Non ho capito quale calcolo intendi. Rispondi col numero dell'opzione.",
        "ask_first": "Confronto '{name}'. Descrivimi la prima offerta in un solo messaggio, indicando almeno: {fields}.",
        "ask_next": "Registrate {count} offerte. Descrivimi la prossima, oppure scrivi 'confronta' per procedere.",
        "ask_next_min": "Registrate {count} offerte su almeno {min_items}. Descrivimi la prossima offerta.",
        "recorded": "Registrata l'offerta {label}: {summary}.",
        "updated": "Aggiornata l'offerta {label}: {summary}.",
        "removed": "Rimossa l'offerta {label}. Offerte rimaste: {count}.",
        "not_found": "Non trovo un'offerta chiamata '{label}'. Offerte registrate: {labels}.",
        "incomplete": "Per registrare questa offerta manca ancora: {fields}. Puoi indicarlo?",
        "pending_draft": "Resta in sospeso l'offerta incompleta {label}: manca ancora {fields}.",
        "need_more": "Per un confronto servono almeno {min_items} offerte, finora ne ho {count}. Descrivimi la prossima.",
        "too_many": "Posso confrontare al massimo {max_items} offerte per volta; ne hai già indicate {count}. Scrivi 'confronta' per procedere, oppure rimuovine una ('rimuovi <nome>').",
        "structured_form": "Non riesco a interpretare l'offerta da testo libero in questo momento. Indicala nel formato 'campo: valore', separando i campi con una virgola. Campi disponibili: {fields}.",
        "review_header": "Ecco i dati che userò per il confronto. Confermi?",
        "review_shared": "Dati comuni",
        "review_candidates": "Offerte",
        "review_prompt": "Scrivi 'confermo' per calcolare, oppure correggi un'offerta ripetendola col suo nome ('rimuovi <nome>' per eliminarla).",
        "confirm_defaults": "Alcuni dati che incidono sul punteggio non li hai indicati e sono stati assunti per default:",
        "confirm_prompt": "Scrivi 'confermo' per accettare queste assunzioni e vedere il risultato, oppure indicami i valori corretti.",
        "clear_winner": "Vincitore chiaro secondo il modello di punteggio configurato: {winner} (distacco {gap} punti su 100).",
        "effective_tie": "Sostanziale parità tra {winners}: nessuna differenza materiale con il modello di punteggio attuale (distacco {gap} punti, entro la tolleranza di {tolerance}). Non indico un'offerta come migliore.",
        "provisional": "Risultato PROVVISORIO, non definitivo: {count} campi che incidono sul punteggio sono stati assunti per default (completezza dei dati {completeness}).",
        "ranking": "Classifica",
        "cost": "costo stimato",
        "score": "punteggio",
        "components": "componenti",
        "relative_note": "Il punteggio 0-100 è relativo alle sole offerte confrontate e ai pesi configurati in questo calcolatore: non è una misura oggettiva del mercato.",
    },
    "es": {
        "choose": "La consulta puede corresponder a varios cálculos. ¿Cuál quieres?",
        "choose_hint": "Responde con el número o con el nombre del cálculo.",
        "choose_unclear": "No he entendido qué cálculo quieres. Responde con el número de la opción.",
        "ask_first": "Comparación '{name}'. Descríbeme la primera oferta en un solo mensaje, indicando al menos: {fields}.",
        "ask_next": "He registrado {count} ofertas. Descríbeme la siguiente, o escribe 'comparar' para continuar.",
        "ask_next_min": "He registrado {count} ofertas de al menos {min_items}. Descríbeme la siguiente.",
        "recorded": "Registrada la oferta {label}: {summary}.",
        "updated": "Actualizada la oferta {label}: {summary}.",
        "removed": "Eliminada la oferta {label}. Ofertas restantes: {count}.",
        "not_found": "No encuentro una oferta llamada '{label}'. Ofertas registradas: {labels}.",
        "incomplete": "Para registrar esta oferta todavía falta: {fields}. ¿Puedes indicarlo?",
        "pending_draft": "Queda pendiente la oferta incompleta {label}: todavía falta {fields}.",
        "need_more": "Para comparar hacen falta al menos {min_items} ofertas y por ahora tengo {count}. Descríbeme la siguiente.",
        "too_many": "Puedo comparar como máximo {max_items} ofertas a la vez y ya has indicado {count}. Escribe 'comparar' para continuar, o elimina una ('eliminar <nombre>').",
        "structured_form": "Ahora mismo no puedo interpretar la oferta en texto libre. Indícala con el formato 'campo: valor', separando los campos con una coma. Campos disponibles: {fields}.",
        "review_header": "Estos son los datos que usaré para la comparación. ¿Los confirmas?",
        "review_shared": "Datos comunes",
        "review_candidates": "Ofertas",
        "review_prompt": "Escribe 'confirmo' para calcular, o corrige una oferta repitiéndola con su nombre ('eliminar <nombre>' para quitarla).",
        "confirm_defaults": "Algunos datos que influyen en la puntuación no los has indicado y se han asumido por defecto:",
        "confirm_prompt": "Escribe 'confirmo' para aceptar estos supuestos y ver el resultado, o indícame los valores correctos.",
        "clear_winner": "Ganador claro según el modelo de puntuación configurado: {winner} (diferencia de {gap} puntos sobre 100).",
        "effective_tie": "Empate sustancial entre {winners}: no hay diferencia material con el modelo de puntuación actual (diferencia de {gap} puntos, dentro de la tolerancia de {tolerance}). No señalo ninguna oferta como la mejor.",
        "provisional": "Resultado PROVISIONAL, no definitivo: {count} campos que influyen en la puntuación se han asumido por defecto (integridad de los datos {completeness}).",
        "ranking": "Clasificación",
        "cost": "coste estimado",
        "score": "puntuación",
        "components": "componentes",
        "relative_note": "La puntuación 0-100 es relativa solo a las ofertas comparadas y a los pesos configurados en esta calculadora: no es una medida objetiva del mercado.",
    },
    "en": {
        "choose": "This request could match more than one calculation. Which one do you mean?",
        "choose_hint": "Reply with the number, or with the name of the calculation.",
        "choose_unclear": "I could not tell which calculation you meant. Reply with the option number.",
        "ask_first": "Comparison '{name}'. Describe the first offer in a single message, stating at least: {fields}.",
        "ask_next": "{count} offers recorded. Describe the next one, or write 'compare' to proceed.",
        "ask_next_min": "{count} of at least {min_items} offers recorded. Describe the next one.",
        "recorded": "Recorded offer {label}: {summary}.",
        "updated": "Updated offer {label}: {summary}.",
        "removed": "Removed offer {label}. Offers left: {count}.",
        "not_found": "I cannot find an offer named '{label}'. Recorded offers: {labels}.",
        "incomplete": "This offer is still missing: {fields}. Could you provide it?",
        "pending_draft": "The incomplete offer {label} is still pending: it is missing {fields}.",
        "need_more": "A comparison needs at least {min_items} offers and I have {count} so far. Describe the next one.",
        "too_many": "I can compare at most {max_items} offers at a time and you have already given {count}. Write 'compare' to proceed, or remove one ('remove <name>').",
        "structured_form": "I cannot read a free-text offer right now. Please state it as 'field: value', separating fields with a comma. Available fields: {fields}.",
        "review_header": "Here is the data I will use for the comparison. Shall I go ahead?",
        "review_shared": "Shared data",
        "review_candidates": "Offers",
        "review_prompt": "Write 'confirm' to calculate, or correct an offer by restating it with its name ('remove <name>' to drop it).",
        "confirm_defaults": "Some fields that affect the score were not provided and have been assumed by default:",
        "confirm_prompt": "Write 'confirm' to accept these assumptions and see the result, or give me the correct values.",
        "clear_winner": "Clear winner under the configured scoring model: {winner} ({gap} points ahead out of 100).",
        "effective_tie": "Effective tie between {winners}: no material difference under the current scoring model ({gap} points apart, within the {tolerance} tolerance). I am not calling any offer the best one.",
        "provisional": "PROVISIONAL result, not final: {count} fields that affect the score were assumed by default (data completeness {completeness}).",
        "ranking": "Ranking",
        "cost": "estimated cost",
        "score": "score",
        "components": "components",
        "relative_note": "The 0-100 score is relative to the compared offers and to the weights configured in this calculator only: it is not an objective measure of the market.",
    },
}

# Control words that steer candidate collection, in every language the
# chatbot supports. Matched on the whole normalized message so an offer
# whose name happens to contain "fine" is not mistaken for a command.
_FINISH_WORDS = frozenset({
    "confronta", "confronto", "compara", "calcola", "procedi",
    "basta", "fine", "finito", "vai",
    "compare", "calculate", "finish", "done", "go ahead", "that's all",
    "comparar", "calcular", "terminar", "listo", "ya esta", "adelante",
})
_CONFIRM_WORDS = frozenset({
    "si", "sì", "ok", "okay", "va bene", "conferma", "confermo", "confermato",
    "yes", "confirm", "confirmed", "correct", "go", "proceed",
    "sí", "confirmo", "confirmado", "de acuerdo", "vale", "correcto",
})
_REMOVE_PREFIXES = (
    "rimuovi", "togli", "elimina", "cancella", "scarta",
    "remove", "delete", "drop", "discard",
    "eliminar", "quitar", "borrar",
)
# Above this a "comparison" stops being a comparison and starts being a
# denial-of-service on the collection loop (and on the LLM context).
_MAX_CANDIDATES = 20


def _session_lang(state: Dict[str, Any]) -> str:
    lang = str(state.get("session_language") or "it").strip().lower()[:2]
    return lang if lang in _COPY else "it"


def _calculation_payload(calculator_id: str, values: Dict[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"calculator_id": calculator_id, "inputs": {}}
    for name, value in values.items():
        if name in {"period", "tax_year", "as_of_date"}:
            payload[name] = value
        else:
            payload["inputs"][name] = value
    return payload


def _missing_specs(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    if response.get("status") != "error":
        return []
    for error in response.get("errors") or []:
        if not isinstance(error, dict) or error.get("code") != "input_invalid":
            continue
        details = error.get("details") or {}
        missing = details.get("missing") or []
        if missing:
            return [spec for spec in missing if isinstance(spec, dict)]
        return [
            {"name": name, "type": "string", "required": True}
            for name in details.get("missing_inputs") or []
        ]
    return []


def _clarification_question(lang: str, specs: List[Dict[str, Any]]) -> str:
    labels = []
    for spec in specs:
        name = str(spec.get("name", "dato richiesto"))
        description = spec.get("description")
        labels.append(f"{name} ({description})" if description else name)
    return _COPY[lang]["clarify"].format(items="; ".join(labels))


def _normalize_frequency_inputs(
    calculator_id: str, values: Dict[str, Any], text: str
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Resolve declared frequency-sensitive inputs, never raising.

    Fails CLOSED. Returning the extracted values on error would hand
    /calculate the very number this layer exists to intercept — an
    unconverted monthly rent, which for the lease calculator lands on the legal
    minimum and so reads as a plausible tax. A crash must cost the user a
    question, never a wrong figure. Inputs outside this layer's scope are
    passed through untouched: they were never in doubt.
    """
    try:
        return normalization.normalize_inputs(calculator_id, values, text)
    except Exception:
        logger.exception("Frequency normalization failed for %s", calculator_id)
        safe, unresolved = normalization.failure_unresolved(calculator_id, values)
        return safe, [], unresolved


def _frequency_question(lang: str, unresolved: List[Dict[str, Any]]) -> str:
    copy = _COPY[lang]
    questions = []
    for entry in unresolved:
        amount = normalization.format_amount(entry.get("raw_value"), lang)
        # The reader gets the field's own word for itself, never `annual_rent`.
        field = normalization.field_label(entry, lang)
        reason = entry.get("reason")
        if reason == normalization.REASON_CURRENCY_UNSUPPORTED:
            questions.append(copy["ask_currency"].format(
                amount=amount, field=field, stated=entry.get("stated_currency"),
            ))
        elif reason == normalization.REASON_CURRENCY_AMBIGUOUS:
            questions.append(copy["ask_currency_ambiguous"].format(field=field))
        elif reason == normalization.REASON_NORMALIZATION_FAILED:
            questions.append(copy["ask_normalization_failed"].format(field=field))
        else:
            questions.append(copy["ask_frequency"].format(
                amount=amount, field=field,
            ))
    return " ".join(questions)


def _unresolved_frequency_update(
    lang: str,
    *,
    calculator_id: str,
    inputs_so_far: Dict[str, Any],
    unresolved: List[Dict[str, Any]],
    conversions: List[Dict[str, Any]],
    specs: Iterable[Dict[str, Any]],
    current_round: int,
) -> Dict[str, Any]:
    """Ask what an amount means, instead of calculating with a guess.

    The unresolved field stays OUT of `inputs_so_far` and is listed as missing,
    so a full restatement ("500 euro al mese") flows through the ordinary
    continuation path. `pending_frequency` additionally holds the amount the
    user already gave, so a bare "mensile" is enough on its own.
    """
    # Bounded like every other clarification: a user who cannot pin the amount
    # down must not be asked forever, and an extractor stuck on an ambiguous
    # message must not be able to mint pending rounds without end.
    next_round = current_round + 1
    if next_round > _MAX_CLARIFICATION_ROUNDS:
        return _failure_update(lang, round_limit=True)

    names = {entry["field"] for entry in unresolved}
    missing = [
        spec for spec in specs or []
        if isinstance(spec, dict) and spec.get("name") in names
    ] or [{"name": name, "type": "decimal", "required": True} for name in sorted(names)]

    pending: Dict[str, Any] = {
        "calculator_id": calculator_id,
        "inputs_so_far": inputs_so_far,
        "round": next_round,
        "missing_inputs": missing,
        "pending_frequency": normalization.pending_frequency_state(unresolved),
    }
    if conversions:
        pending["conversions"] = list(conversions)
    return _answered_update(
        _frequency_question(lang, unresolved),
        pending_calculation=pending,
        retrieval_quality_ok=True,
    )


def _display_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _notice_lines(items: Any) -> List[str]:
    """Extract message strings from a list of {code, message} notices."""
    lines: List[str] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        message = item.get("message")
        if message:
            lines.append(str(message))
    return lines


def _comparison_lines(lang: str, result: Dict[str, Any]) -> List[str]:
    """Render a comparison result as prose a reader can act on.

    Verdict first, and only ever the verdict the platform actually reached:
    `clear_winner` names a leader "under the configured model", an
    effective tie names nobody. Then the money, then the synthetic score —
    a 0-100 number printed first would be read as a grade rather than as
    the weighted opinion it is.
    """
    copy = _COMPARISON_COPY[lang]
    comparison = result.get("comparison") or {}
    ranking = [entry for entry in result.get("ranking") or [] if isinstance(entry, dict)]
    lines: List[str] = []

    winners = comparison.get("best_candidates") or ([result["best"]] if result.get("best") else [])
    if comparison.get("decision_status") == "effective_tie":
        lines.append(copy["effective_tie"].format(
            winners=", ".join(winners),
            gap=comparison.get("score_gap"),
            tolerance=comparison.get("tie_tolerance"),
        ))
    elif winners:
        lines.append(copy["clear_winner"].format(
            winner=winners[0], gap=comparison.get("score_gap"),
        ))

    if comparison.get("provisional"):
        lines.append(copy["provisional"].format(
            count=len(comparison.get("scoring_defaults_applied") or []),
            completeness=comparison.get("scoring_completeness"),
        ))

    cost_variable = (comparison.get("cost_basis") or {}).get("variable")
    if ranking:
        lines.append("")
        lines.append(f"{copy['ranking']}:")
        for entry in ranking:
            derived = entry.get("derived") or {}
            parts = []
            if cost_variable and derived.get(cost_variable) is not None:
                parts.append(f"{copy['cost']} {derived[cost_variable]}")
            parts.append(f"{copy['score']} {entry.get('total_score')}/100")
            lines.append(f"{entry.get('rank')}. {entry.get('label')} — {'; '.join(parts)}")
            scores = entry.get("scores") or {}
            if scores:
                detail = ", ".join(f"{name} {value}" for name, value in scores.items())
                lines.append(f"   {copy['components']}: {detail}")

    lines.append("")
    lines.append(copy["relative_note"])
    return lines


def _conversion_lines(lang: str, conversions: List[Dict[str, Any]]) -> List[str]:
    words = {"month": _COPY[lang]["month"], "year": _COPY[lang]["year"]}
    return [
        normalization.format_conversion(record, lang, words)
        for record in conversions or []
        if isinstance(record, dict)
    ]


def _success_answer(
    lang: str,
    response: Dict[str, Any],
    conversions: Optional[List[Dict[str, Any]]] = None,
) -> str:
    result = response.get("result") or {}
    if isinstance(result.get("comparison"), dict):
        rendered = "\n" + "\n".join(_comparison_lines(lang, result))
    else:
        rendered = "; ".join(
            f"{name}: {_display_value(value)}" for name, value in sorted(result.items())
        )
    citations = []
    for citation in response.get("citations") or []:
        if not isinstance(citation, dict):
            continue
        reference = citation.get("reference") or citation.get("source_name")
        if not reference:
            continue
        source_name = citation.get("source_name")
        if source_name and source_name != reference:
            reference = f"{reference} ({source_name})"
        if citation.get("url"):
            reference = f"{reference} — {citation['url']}"
        citations.append(reference)
    sources = "; ".join(citations) or _COPY[lang]["no_sources"]

    sections = [f"{_COPY[lang]['result']}: {rendered}"]
    # Immediately after the figure, never further down: the result IS the
    # conversion's output, so a reader who stops after the number must already
    # have seen the arithmetic that produced its input.
    conversion_lines = _conversion_lines(lang, conversions)
    if conversion_lines:
        sections.append(
            f"{_COPY[lang]['conversions']}:\n"
            + "\n".join(f"- {line}" for line in conversion_lines)
        )
    # Surface the platform's assumptions and warnings — never drop them: a
    # staleness or "gross only" warning changes how the number must be read.
    assumptions = _notice_lines(response.get("assumptions"))
    if assumptions:
        sections.append(
            f"{_COPY[lang]['assumptions']}:\n"
            + "\n".join(f"- {line}" for line in assumptions)
        )
    defaults = response.get("defaults_applied") or []
    if defaults:
        sections.append(
            f"{_COPY[lang]['defaults']}:\n"
            + "\n".join(
                f"- {entry.get('path')} = {_display_value(entry.get('value'))}"
                for entry in defaults
                if isinstance(entry, dict)
            )
        )
    warnings = _notice_lines(response.get("warnings"))
    if warnings:
        sections.append(
            f"{_COPY[lang]['warnings']}:\n"
            + "\n".join(f"- {line}" for line in warnings)
        )
    # Its own labelled section, never flattened into the warnings: what the
    # calculator does not cover is a boundary on the answer, and a reader
    # who skims warnings as boilerplate must still meet it.
    exclusions = [str(item) for item in response.get("exclusions") or [] if item]
    if exclusions:
        sections.append(
            f"{_COPY[lang]['exclusions']}:\n"
            + "\n".join(f"- {line}" for line in exclusions)
        )
    methodology = response.get("methodology")
    explanation = []
    if not isinstance(result.get("comparison"), dict):
        explanation = [str(line) for line in response.get("explanation") or [] if line]
    how_lines = []
    if methodology:
        how_lines.append(str(methodology))
    how_lines += [f"- {line}" for line in explanation]
    if how_lines:
        sections.append(
            f"{_COPY[lang]['methodology']}:\n" + "\n".join(how_lines)
        )
    sections.append(f"{_COPY[lang]['sources']}: {sources}")
    sections.append(_COPY[lang]["disclaimer"])
    return "\n\n".join(sections)


def _answered_update(answer: str, **updates: Any) -> Dict[str, Any]:
    """Build a calculation answer that also clears stale legal clarification state."""
    return {
        "answer": answer,
        "awaiting_clarification": False,
        "pending_sections": [],
        **updates,
    }


def _platform_error_message(
    response: Dict[str, Any], *, include_validation: bool = False
) -> Optional[str]:
    """Return a genuine platform failure message.

    Input-validation errors are excluded by default because the missing-input
    clarification path states them better. They are included only where that
    path has nothing to ask for — an object_list holding fewer candidates than
    the pack's minimum reports no missing field, so suppressing its message
    would leave the user with a failure that explains nothing.
    """
    for error in response.get("errors") or []:
        if not isinstance(error, dict):
            continue
        if error.get("code") == "input_invalid" and not include_validation:
            continue
        message = error.get("message")
        if message:
            return str(message)
    return None


def _failure_update(
    lang: str,
    *,
    round_limit: bool = False,
    platform_message: Optional[str] = None,
    pending_calculation: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    key = "round_limit" if round_limit else "failure"
    answer = _COPY[lang][key]
    if platform_message:
        answer = f"{answer} {platform_message}"
    return _answered_update(
        answer,
        pending_calculation=pending_calculation,
        retrieval_quality_ok=True,
    )


def _handle_response(
    response: Dict[str, Any],
    *,
    lang: str,
    calculator_id: str,
    inputs_so_far: Dict[str, Any],
    current_round: int,
    expected_specs: Optional[Iterable[Dict[str, Any]]] = None,
    conversions: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    if response.get("status") == "success":
        logger.info(
            "calc_node: outcome=success calculator=%s conversions=%s",
            calculator_id,
            [record.get("rule_id") for record in conversions or []],
        )
        return _answered_update(
            _success_answer(lang, response, conversions),
            calculation_result=response.get("result") or {},
            calculation_conversions=list(conversions or []),
            pending_calculation=None,
            retrieval_quality_ok=True,
        )

    missing = _missing_specs(response)
    if missing:
        logger.info(
            "calc_node: outcome=clarify calculator=%s missing=%s",
            calculator_id, [spec.get("name") for spec in missing],
        )
        missing_names = {spec.get("name") for spec in missing}
        for spec in expected_specs or []:
            if (
                isinstance(spec, dict)
                and spec.get("required")
                and spec.get("name") not in inputs_so_far
                and spec.get("name") not in missing_names
            ):
                missing.append(spec)
                missing_names.add(spec.get("name"))
        next_round = current_round + 1
        if next_round > _MAX_CLARIFICATION_ROUNDS:
            return _failure_update(lang, round_limit=True)
        pending: Dict[str, Any] = {
            "calculator_id": calculator_id,
            "inputs_so_far": inputs_so_far,
            "round": next_round,
            "missing_inputs": missing,
        }
        # Carried only when there is something to carry: a conversion already
        # performed has to reach the turn that finally shows the number, or the
        # arithmetic behind an input collected two turns ago goes unseen. When
        # nothing was converted the key stays absent rather than shipping an
        # empty list into every session record.
        if conversions:
            pending["conversions"] = list(conversions)
        return _answered_update(
            _clarification_question(lang, missing),
            pending_calculation=pending,
            retrieval_quality_ok=True,
        )
    # No field to ask for, so a validation message is all the user can act on.
    return _failure_update(
        lang,
        platform_message=_platform_error_message(response, include_validation=True),
    )


# ---------------------------------------------------------------------------
# Incremental candidate collection for object_list (comparator) calculators
#
# A comparator takes a whole array of offers. Asking an LLM to rebuild that
# array from scratch on every turn is how offers silently disappear: the
# model only sees the newest message, so anything the user said three turns
# ago has to be re-derived, and any field it fails to re-derive is simply
# gone from the request. Instead the array is state, owned here: each turn
# contributes at most one candidate, everything already collected is kept
# verbatim, and the model is only ever asked to read the sentence in front
# of it. The phases below are the whole protocol; every one of them is
# JSON-safe so the session store round-trips them unchanged.
# ---------------------------------------------------------------------------

_PHASE_SHARED = "collect_shared_inputs"
_PHASE_CANDIDATES = "collect_candidates"
_PHASE_REVIEW = "review"
_PHASE_CONFIRM = "confirm"
_PHASE_CHOOSE = "choose_calculator"

_FIELD_HEAD = re.compile(r"([A-Za-z_]\w*)\s*[:=]\s*")
_TRUE_FORM_VALUES = {"true", "1", "si", "sì", "sí", "yes", "y", "x"}
_FALSE_FORM_VALUES = {"false", "0", "no", "n"}


def _field_assignments(text: str, names: Iterable[str]):
    """Yield (field, value) for every `field: value` in a structured message.

    A value runs to the next DECLARED field name, not to the next comma:
    Italian writes decimals with a comma, so `prezzo: 0,25, gas: 1,10` has
    to yield "0,25" and "1,10". Cutting at the first comma turned 0,25 into
    0 — and since zero is a valid price, that silently invented a free
    offer and could hand it the comparison. Only declared names open a new
    field, so a colon inside a value ("nome: Alfa: Plus") does not split it.
    """
    declared = set(names)
    body = str(text or "")
    heads = [m for m in _FIELD_HEAD.finditer(body) if m.group(1) in declared]
    for index, head in enumerate(heads):
        end = heads[index + 1].start() if index + 1 < len(heads) else len(body)
        value = body[head.end():end].strip().rstrip(";,").strip()
        yield head.group(1), value


def _normalize_command(message: str) -> str:
    return " ".join(str(message or "").strip().lower().rstrip("!.?").split())


def _candidate_descriptor(spec: Any) -> Optional[Dict[str, Any]]:
    """Describe an object_list input well enough to collect it one item at a
    time, or None when the spec does not carry its item fields.

    Without item fields there is nothing to validate a candidate against and
    no label to correct one by, so the caller must fall back to the ordinary
    single-shot clarification rather than improvise a collection loop.
    """
    if not isinstance(spec, dict) or spec.get("type") != "object_list":
        return None
    fields = [f for f in spec.get("item_fields") or [] if isinstance(f, dict) and f.get("name")]
    if not fields:
        return None
    label_field = next(
        (f["name"] for f in fields if f.get("type") == "string" and f.get("required")),
        next((f["name"] for f in fields if f.get("type") == "string"), None),
    )
    return {
        "name": spec["name"],
        "item_fields": fields,
        "required_fields": [f["name"] for f in fields if f.get("required")],
        "min_items": int(spec.get("min_items") or 2),
        "label_field": label_field,
    }


def _descriptor_from_specs(specs: Iterable[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for spec in specs or []:
        descriptor = _candidate_descriptor(spec)
        if descriptor is not None:
            return descriptor
    return None


def _shared_specs(specs: Iterable[Dict[str, Any]], candidate_field: str) -> List[Dict[str, Any]]:
    return [
        spec
        for spec in specs or []
        if isinstance(spec, dict) and spec.get("name") and spec.get("name") != candidate_field
    ]


def _candidate_label(candidate: Dict[str, Any], descriptor: Dict[str, Any], index: int) -> str:
    label_field = descriptor.get("label_field")
    if label_field and candidate.get(label_field):
        return str(candidate[label_field])
    return f"#{index + 1}"


def _candidate_labels(candidates: List[Dict[str, Any]], descriptor: Dict[str, Any]) -> List[str]:
    return [_candidate_label(c, descriptor, i) for i, c in enumerate(candidates)]


def _find_candidate_exact(
    candidates: List[Dict[str, Any]], descriptor: Dict[str, Any], text: str
) -> Optional[int]:
    """Locate a collected candidate whose label is exactly `text`.

    Used when deciding whether an extracted offer CORRECTS an existing one.
    Substring matching is wrong here: "Alfa Plus" contains "Alfa", so a
    genuinely new product was silently swallowed into the base one and the
    comparison lost a candidate without saying so. A real correction restates
    the same name.
    """
    needle = _normalize_command(text)
    if not needle:
        return None
    labels = [_normalize_command(label) for label in _candidate_labels(candidates, descriptor)]
    return labels.index(needle) if needle in labels else None


def _find_candidate(
    candidates: List[Dict[str, Any]], descriptor: Dict[str, Any], text: str
) -> Optional[int]:
    """Locate a candidate the user named in an explicit command.

    Exact match first, then a containment match, and only when exactly one
    candidate matches — an ambiguous reference must not silently drop the
    wrong offer. Looser than _find_candidate_exact on purpose: here the user
    typed the name themselves, so "rimuovi alfa" should find "Alfa Plus"
    when that is the only candidate it can mean.
    """
    exact = _find_candidate_exact(candidates, descriptor, text)
    if exact is not None:
        return exact
    needle = _normalize_command(text)
    if not needle:
        return None
    labels = [_normalize_command(label) for label in _candidate_labels(candidates, descriptor)]
    partial = [i for i, label in enumerate(labels) if needle in label or label in needle]
    return partial[0] if len(partial) == 1 else None


_REMOVE_RE = re.compile(
    r"^\s*(?:" + "|".join(_REMOVE_PREFIXES) + r")\s+(.+?)\s*[.!?]*$", re.IGNORECASE
)


def _removal_target(message: str) -> Optional[str]:
    """The offer name in a "remove X" command, in the user's own casing —
    echoing back a lowercased name in a "no such offer" message reads like
    the system mangled it."""
    match = _REMOVE_RE.match(str(message or ""))
    return match.group(1).strip() if match else None


def _missing_candidate_fields(candidate: Dict[str, Any], descriptor: Dict[str, Any]) -> List[str]:
    return [name for name in descriptor["required_fields"] if candidate.get(name) in (None, "")]


def _field_label(spec: Dict[str, Any]) -> str:
    description = spec.get("description")
    return f"{spec['name']} ({description})" if description else str(spec["name"])


def _required_field_labels(descriptor: Dict[str, Any]) -> str:
    required = set(descriptor["required_fields"])
    return "; ".join(
        _field_label(spec) for spec in descriptor["item_fields"] if spec["name"] in required
    )


def _candidate_summary(candidate: Dict[str, Any]) -> str:
    return ", ".join(f"{k}={_display_value(v)}" for k, v in candidate.items())


def _parse_structured_candidate(
    message: str, descriptor: Dict[str, Any]
) -> Dict[str, Any]:
    """Deterministic 'field: value' fallback used when LLM extraction is
    unavailable.

    Deliberately literal: only assignments naming a declared field are read,
    so a sentence full of bare numbers yields nothing at all rather than an
    offer assembled from whatever digits happened to be in it. Guessing here
    would put an invented premium in front of a user as if they had said it.
    """
    specs = {spec["name"]: spec for spec in descriptor["item_fields"]}
    return _parse_structured_fields(message, specs)


def _parse_structured_scalars(
    message: str, specs: Iterable[Dict[str, Any]]
) -> Dict[str, Any]:
    """The same literal `field: value` reading for shared scalar inputs.

    Shared inputs used to fall back to the positional number extractor when
    the LLM was unavailable. That binds by ORDER, not by name, so "cosa dice
    l'articolo 40 del codice?" was read as a 40-year-old driver: a question
    silently became an answer. Nothing is inferred here either.
    """
    return _parse_structured_fields(message, {spec["name"]: spec for spec in specs})


def _parse_structured_fields(
    message: str, specs: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}
    for name, raw in _field_assignments(message, specs):
        value = _coerce_form_value(raw, specs[name])
        if value is not None:
            parsed[name] = value
    return parsed


def _coerce_form_value(raw: str, spec: Dict[str, Any]) -> Any:
    if not raw:
        return None
    kind = spec.get("type")
    if kind == "boolean":
        lowered = raw.lower()
        if lowered in _TRUE_FORM_VALUES:
            return True
        if lowered in _FALSE_FORM_VALUES:
            return False
        return None
    if kind in {"decimal", "integer"}:
        try:
            return _normalize_number(raw, spec)
        except (InvalidOperation, ValueError):
            return None
    return raw


def _pending_comparison(
    calculator_id: str,
    descriptor: Dict[str, Any],
    *,
    phase: str,
    inputs_so_far: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    candidate_draft: Optional[Dict[str, Any]] = None,
    missing_inputs: Optional[List[Dict[str, Any]]] = None,
    shared_specs: Optional[List[Dict[str, Any]]] = None,
    rounds: int = 0,
) -> Dict[str, Any]:
    return {
        "calculator_id": calculator_id,
        "phase": phase,
        "inputs_so_far": inputs_so_far,
        "candidate_field": descriptor["name"],
        "candidate_descriptor": descriptor,
        "candidates": candidates,
        "candidate_draft": candidate_draft or {},
        "missing_inputs": missing_inputs or [],
        "shared_specs": shared_specs or [],
        # Candidate-collection turns never touch this. The three-round limit
        # exists to stop a calculator badgering a user who cannot supply a
        # missing figure; describing a fifth offer is the feature working,
        # not a failed clarification.
        "round": rounds,
    }


def _comparison_reply(text: str, pending: Dict[str, Any]) -> Dict[str, Any]:
    return _answered_update(text, pending_calculation=pending, retrieval_quality_ok=True)


def _missing_shared(
    shared_specs: List[Dict[str, Any]], inputs_so_far: Dict[str, Any]
) -> List[Dict[str, Any]]:
    return [
        spec
        for spec in shared_specs
        if spec.get("required") and spec.get("name") not in inputs_so_far
    ]


def _ask_next(lang: str, pending: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Ask for the next thing the comparison needs, given its phase."""
    copy = _COMPARISON_COPY[lang]
    descriptor = pending["candidate_descriptor"]
    # A complete offer held back because the list was full gets its slot as
    # soon as one frees up, instead of the user having to restate it.
    draft = pending.get("candidate_draft") or {}
    if (
        draft
        and not _missing_candidate_fields(draft, descriptor)
        and len(pending["candidates"]) < _MAX_CANDIDATES
    ):
        pending["candidates"].append(draft)
        pending["candidate_draft"] = {}
    # An offer that is still half-described must not sit in silent limbo:
    # say it is pending, so the user knows it has not been counted.
    leftover = pending.get("candidate_draft") or {}
    if leftover:
        specs = {spec["name"]: spec for spec in descriptor["item_fields"]}
        prefix += copy["pending_draft"].format(
            label=_candidate_label(leftover, descriptor, len(pending["candidates"])),
            fields="; ".join(
                _field_label(specs[name])
                for name in _missing_candidate_fields(leftover, descriptor)
            ),
        ) + "\n"
    missing = _missing_shared(pending["shared_specs"], pending["inputs_so_far"])
    if missing:
        pending["phase"] = _PHASE_SHARED
        pending["missing_inputs"] = missing
        question = _clarification_question(lang, missing)
        return _comparison_reply(f"{prefix}{question}".strip(), pending)

    pending["phase"] = _PHASE_CANDIDATES
    pending["missing_inputs"] = []
    count = len(pending["candidates"])
    if count == 0:
        text = copy["ask_first"].format(
            name=pending.get("calculator_name") or pending["calculator_id"],
            fields=_required_field_labels(descriptor),
        )
    elif count < descriptor["min_items"]:
        text = copy["ask_next_min"].format(count=count, min_items=descriptor["min_items"])
    else:
        text = copy["ask_next"].format(count=count)
    return _comparison_reply(f"{prefix}{text}".strip(), pending)


def _review_text(lang: str, pending: Dict[str, Any]) -> str:
    copy = _COMPARISON_COPY[lang]
    descriptor = pending["candidate_descriptor"]
    lines = [copy["review_header"], ""]
    if pending["inputs_so_far"]:
        lines.append(f"{copy['review_shared']}:")
        lines += [
            f"- {name}: {_display_value(value)}"
            for name, value in sorted(pending["inputs_so_far"].items())
        ]
        lines.append("")
    lines.append(f"{copy['review_candidates']}:")
    for index, candidate in enumerate(pending["candidates"]):
        label = _candidate_label(candidate, descriptor, index)
        lines.append(f"{index + 1}. {label} — {_candidate_summary(candidate)}")
    lines.append("")
    lines.append(copy["review_prompt"])
    return "\n".join(lines)


def _start_comparison(
    lang: str,
    calculator_id: str,
    descriptor: Dict[str, Any],
    specs: List[Dict[str, Any]],
    raw_query: str,
    calculator_name: str = "",
) -> Dict[str, Any]:
    """Open a comparison, seeding it with whatever the opening sentence
    already contains (a one-shot "compare A at 420 and B at 510" must not be
    thrown away just because collection is incremental)."""
    pending = _pending_comparison(
        calculator_id,
        descriptor,
        phase=_PHASE_SHARED,
        inputs_so_far={},
        candidates=[],
        shared_specs=_shared_specs(specs, descriptor["name"]),
    )
    pending["calculator_name"] = calculator_name or calculator_id

    extracted = _extract_values_llm(raw_query, calculator_id, keep_partial_items=True)
    if extracted:
        _absorb_shared(pending, extracted)
        for item in extracted.get(descriptor["name"]) or []:
            if isinstance(item, dict):
                _absorb_candidate(pending, item)
    return _ask_next(lang, pending)


def _resume_comparison(
    client: "PlatformClient", lang: str, pending: Dict[str, Any], raw_query: str
) -> Dict[str, Any]:
    """Continue a comparison on a deep copy of the persisted state.

    The pending payload is the session's own stored dict; mutating it in
    place would rewrite the previous turn's record of what had been
    collected, so an abandoned turn could not be reconstructed.
    """
    return _collect_comparison(client, lang, copy.deepcopy(pending), raw_query)


def _absorb_shared(pending: Dict[str, Any], extracted: Dict[str, Any]) -> int:
    """Copy any declared shared scalar out of an extraction result.

    The request-level fields go through too: _calculation_payload lifts them
    out of `inputs` again, and dropping them here would silently discard the
    tax year of a comparator whose parameters are date-versioned.
    """
    names = {spec["name"] for spec in pending["shared_specs"]}
    names |= {"tax_year", "as_of_date", "period"}
    absorbed = 0
    for name, value in extracted.items():
        if name in names and value is not None:
            pending["inputs_so_far"][name] = value
            absorbed += 1
    return absorbed


def _absorb_candidate(
    pending: Dict[str, Any], item: Dict[str, Any]
) -> Tuple[Optional[str], Optional[int]]:
    """Merge one extracted candidate into the collected set.

    Returns (outcome, index) where outcome is "added", "updated",
    "drafted", "full" or None (nothing usable). An item whose label matches
    a candidate already collected is treated as a correction of that offer,
    not as a new one — restating an offer is how a user fixes it.
    """
    descriptor = pending["candidate_descriptor"]
    item = {k: v for k, v in item.items() if v is not None}
    if not item:
        return None, None

    label_field = descriptor.get("label_field")
    item_label = str(item[label_field]) if label_field and item.get(label_field) else None
    if item_label is not None:
        existing = _find_candidate_exact(pending["candidates"], descriptor, item_label)
        if existing is not None:
            pending["candidates"][existing].update(item)
            # The draft belongs to a DIFFERENT, still-unfinished offer;
            # correcting an already-recorded one says nothing about it.
            return "updated", existing

    draft = dict(pending.get("candidate_draft") or {})
    draft_label = str(draft[label_field]) if label_field and draft.get(label_field) else None
    # Merge into the draft only when the two can be the same offer. A
    # complete "Beta" arriving while "Delta" is half-described used to
    # overwrite Delta field by field, so Delta vanished without a word.
    same_offer = (
        not draft
        or draft_label is None
        or item_label is None
        or _normalize_command(draft_label) == _normalize_command(item_label)
    )
    merged = {**draft, **item} if same_offer else dict(item)

    if _missing_candidate_fields(merged, descriptor):
        pending["candidate_draft"] = merged
        return "drafted", None
    if len(pending["candidates"]) >= _MAX_CANDIDATES:
        # Keep the draft rather than discard a complete offer the user
        # just described: removing one of the 20 frees a slot and
        # _ask_next flushes it then.
        pending["candidate_draft"] = merged
        return "full", None
    pending["candidates"].append(merged)
    if same_offer:
        pending["candidate_draft"] = {}
    return "added", len(pending["candidates"]) - 1


def _collect_comparison(
    client: "PlatformClient", lang: str, pending: Dict[str, Any], raw_query: str
) -> Dict[str, Any]:
    """One turn of an in-progress comparison."""
    copy = _COMPARISON_COPY[lang]
    descriptor = pending.get("candidate_descriptor")
    if not isinstance(descriptor, dict) or not descriptor.get("item_fields"):
        return _failure_update(lang)
    pending.setdefault("candidates", [])
    pending.setdefault("candidate_draft", {})
    pending.setdefault("inputs_so_far", {})
    pending.setdefault("shared_specs", [])

    command = _normalize_command(raw_query)
    phase = pending.get("phase")

    # --- explicit removal, valid in every collecting phase ----------------
    target = _removal_target(raw_query)
    if target and phase in (_PHASE_CANDIDATES, _PHASE_REVIEW, _PHASE_CONFIRM):
        index = _find_candidate(pending["candidates"], descriptor, target)
        if index is None:
            labels = ", ".join(_candidate_labels(pending["candidates"], descriptor)) or "-"
            return _comparison_reply(
                copy["not_found"].format(label=target, labels=labels), pending
            )
        removed = _candidate_label(pending["candidates"].pop(index), descriptor, index)
        notice = copy["removed"].format(label=removed, count=len(pending["candidates"]))
        return _ask_next(lang, pending, prefix=f"{notice}\n")

    # --- review / confirm gates -------------------------------------------
    # A finish word past the review is a confirmation, not a request to see
    # the review again: someone who has just read the summary and types
    # "calcola" means go, and re-printing the same block would loop.
    go_ahead = command in _CONFIRM_WORDS or command in _FINISH_WORDS
    if phase == _PHASE_REVIEW and go_ahead:
        return _run_comparison(client, lang, pending, confirm=False)
    if phase == _PHASE_CONFIRM and go_ahead:
        return _run_comparison(client, lang, pending, confirm=True)

    # --- finish word ------------------------------------------------------
    if command in _FINISH_WORDS and phase in (_PHASE_SHARED, _PHASE_CANDIDATES):
        # Shared facts first: "confronta" while the applicant's age is still
        # missing is a request to proceed, not a change of subject, so it
        # must ask for the age rather than escape to ordinary retrieval.
        if _missing_shared(pending["shared_specs"], pending["inputs_so_far"]):
            return _ask_next(lang, pending)
        if len(pending["candidates"]) < descriptor["min_items"]:
            return _comparison_reply(
                copy["need_more"].format(
                    min_items=descriptor["min_items"], count=len(pending["candidates"])
                ),
                pending,
            )
        pending["phase"] = _PHASE_REVIEW
        return _comparison_reply(_review_text(lang, pending), pending)

    # --- ordinary content turn -------------------------------------------
    extracted = _extract_values_llm(
        raw_query,
        pending["calculator_id"],
        missing_specs=pending.get("missing_inputs") or None,
        prior_inputs={"collected_offers": pending["candidates"]},
        keep_partial_items=True,
    )
    if extracted is None:
        # No LLM. Read ONLY explicit `field: value` assignments, for shared
        # scalars as well as offers. The positional number extractor used to
        # cover shared scalars, but it binds by position, not by name: while
        # waiting for the driver's age it read "cosa dice l'articolo 40 del
        # codice?" as a 40-year-old driver, turning a question into an
        # answer. Nothing here is inferred from loose numbers.
        form = _parse_structured_candidate(raw_query, descriptor)
        shared = _parse_structured_scalars(raw_query, pending["shared_specs"])
        extracted = dict(shared)
        if form:
            extracted[descriptor["name"]] = [form]
        elif not shared:
            hints = int(pending.get("form_hints") or 0)
            mentions_a_field = any(
                spec["name"] in (raw_query or "")
                for spec in list(descriptor["item_fields"]) + list(pending["shared_specs"])
            )
            # Explain the form once — twice if they are clearly trying to
            # give an offer — then stop. Repeating it forever would trap a
            # user who has simply moved on to another question.
            if mentions_a_field or hints < 1:
                pending["form_hints"] = hints + 1
                return _comparison_reply(
                    copy["structured_form"].format(
                        fields=", ".join(spec["name"] for spec in descriptor["item_fields"])
                    ),
                    pending,
                )

    absorbed = _absorb_shared(pending, extracted)
    outcome, touched = None, None
    for item in extracted.get(descriptor["name"]) or []:
        if isinstance(item, dict):
            item_outcome, item_index = _absorb_candidate(pending, item)
            if item_outcome is not None:
                outcome, touched = item_outcome, item_index

    if not absorbed and outcome is None:
        # Nothing in this message belongs to the comparison: the user has
        # moved on. Escape to ordinary RAG rather than keep asking about
        # offers they are no longer talking about.
        return {
            "calc_route": "normal",
            "pending_calculation": None,
            "awaiting_clarification": False,
            "pending_sections": [],
        }

    if outcome == "full":
        return _comparison_reply(
            copy["too_many"].format(
                max_items=_MAX_CANDIDATES, count=len(pending["candidates"])
            ),
            pending,
        )
    if outcome == "drafted":
        missing = _missing_candidate_fields(pending["candidate_draft"], descriptor)
        specs = {spec["name"]: spec for spec in descriptor["item_fields"]}
        return _comparison_reply(
            copy["incomplete"].format(
                fields="; ".join(_field_label(specs[name]) for name in missing)
            ),
            pending,
        )

    prefix = ""
    if outcome in ("added", "updated") and touched is not None:
        candidate = pending["candidates"][touched]
        label = _candidate_label(candidate, descriptor, touched)
        key = "recorded" if outcome == "added" else "updated"
        prefix = copy[key].format(label=label, summary=_candidate_summary(candidate)) + "\n"
    return _ask_next(lang, pending, prefix=prefix)


def _run_comparison(
    client: "PlatformClient", lang: str, pending: Dict[str, Any], *, confirm: bool
) -> Dict[str, Any]:
    """Send the collected comparison to the platform.

    Runs twice at most: once to see whether scoring defaults were applied,
    and — only after the user acknowledges them — once more carrying
    `confirm_assumptions`. Confirmation never removes an assumption from the
    payload; it records that the user saw it.
    """
    values = dict(pending["inputs_so_far"])
    values[pending["candidate_field"]] = pending["candidates"]
    payload = _calculation_payload(pending["calculator_id"], values)
    if confirm:
        payload["confirm_assumptions"] = True

    response = client.calculate(payload)
    if response.get("platform_unavailable"):
        return _failure_update(lang, pending_calculation=pending)

    if response.get("status") != "success":
        missing = _missing_specs(response)
        if missing:
            pending["phase"] = _PHASE_CANDIDATES
            return _comparison_reply(_clarification_question(lang, missing), pending)
        return _failure_update(
            lang, platform_message=_platform_error_message(response, include_validation=True)
        )

    comparison = (response.get("result") or {}).get("comparison") or {}
    if not confirm and comparison.get("provisional"):
        pending["phase"] = _PHASE_CONFIRM
        return _comparison_reply(_confirmation_text(lang, response), pending)

    logger.info(
        "calc_node: outcome=comparison calculator=%s status=%s candidates=%s",
        pending["calculator_id"],
        comparison.get("decision_status"),
        len(pending["candidates"]),
    )
    return _answered_update(
        _success_answer(lang, response),
        calculation_result=response.get("result") or {},
        pending_calculation=None,
        retrieval_quality_ok=True,
    )


def _confirmation_text(lang: str, response: Dict[str, Any]) -> str:
    copy = _COMPARISON_COPY[lang]
    comparison = (response.get("result") or {}).get("comparison") or {}
    lines = [copy["confirm_defaults"]]
    for entry in comparison.get("scoring_defaults_applied") or []:
        lines.append(f"- {entry.get('path')} = {_display_value(entry.get('value'))}")
    lines.append("")
    lines.append(copy["confirm_prompt"])
    return "\n".join(lines)


def _choice_text(lang: str, choices: List[Dict[str, Any]]) -> str:
    copy = _COMPARISON_COPY[lang]
    lines = [copy["choose"]]
    for index, choice in enumerate(choices, start=1):
        name = choice.get("name") or choice.get("calculator_id")
        lines.append(f"{index}) {name}")
    lines.append(copy["choose_hint"])
    return "\n".join(lines)


def _resolve_choice(choices: List[Dict[str, Any]], message: str) -> Optional[Dict[str, Any]]:
    """Accept a disambiguation answer by position, calculator id, or name."""
    normalized = _normalize_command(message)
    if not normalized:
        return None
    # The number has to BE the answer, not merely start it: "1Password" and
    # "1. no, tell me about something else" are not selections of option 1.
    digits = re.fullmatch(r"(\d+)\s*[.)\]]?", normalized)
    if digits:
        index = int(digits.group(1)) - 1
        if 0 <= index < len(choices):
            return choices[index]
    for choice in choices:
        if _normalize_command(choice.get("calculator_id", "")) == normalized:
            return choice
    named = [
        choice
        for choice in choices
        if _normalize_command(choice.get("name", "")) == normalized
        or (normalized and normalized in _normalize_command(choice.get("name", "")))
    ]
    return named[0] if len(named) == 1 else None


def _match_specs(match: Dict[str, Any]) -> List[Dict[str, Any]]:
    specs = [
        *[
            {**spec, "required": True}
            for spec in match.get("required_inputs") or []
            if isinstance(spec, dict)
        ],
        *[
            {**spec, "required": False}
            for spec in match.get("optional_inputs") or []
            if isinstance(spec, dict)
        ],
    ]
    if match.get("requires_period") and not any(
        spec.get("type") == "period" for spec in specs if isinstance(spec, dict)
    ):
        specs.append({"name": "period", "type": "period", "required": True})
    return specs


def _start_calculation(
    client: "PlatformClient", lang: str, match: Dict[str, Any], raw_query: str
) -> Dict[str, Any]:
    """Begin a fresh calculation from a resolved calculator match."""
    calculator_id = match.get("calculator_id")
    if not calculator_id:
        return _failure_update(lang)
    specs = _match_specs(match)

    descriptor = _descriptor_from_specs(specs)
    if descriptor is not None:
        return _start_comparison(
            lang, calculator_id, descriptor, specs, raw_query,
            calculator_name=match.get("name") or "",
        )

    inputs_so_far = _extract_values_llm(raw_query, calculator_id)
    if inputs_so_far is None:
        inputs_so_far = _extract_values(
            raw_query,
            specs,
            supports_tax_year=bool(match.get("supports_tax_year")),
        )
    # Between extraction and arithmetic: the LLM reported what the user wrote,
    # this turns it into what the calculator means. Runs on the regex tier too,
    # where an unconverted monthly rent would otherwise be the likeliest input.
    inputs_so_far, conversions, unresolved = _normalize_frequency_inputs(
        calculator_id, inputs_so_far, raw_query
    )
    if unresolved:
        return _unresolved_frequency_update(
            lang,
            calculator_id=calculator_id,
            inputs_so_far=inputs_so_far,
            unresolved=unresolved,
            conversions=conversions,
            specs=specs,
            current_round=0,
        )
    response = client.calculate(_calculation_payload(calculator_id, inputs_so_far))
    if response.get("platform_unavailable"):
        return {
            "calc_route": "normal",
            "calculation_match": None,
            "pending_calculation": None,
        }
    return _handle_response(
        response,
        lang=lang,
        calculator_id=calculator_id,
        inputs_so_far=inputs_so_far,
        current_round=0,
        expected_specs=specs,
        conversions=conversions,
    )


def calculation_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Run a fresh or continued deterministic calculation, never raising."""
    lang = "it"
    try:
        lang = _session_lang(state)
        client = PlatformClient()
        pending = state.get("pending_calculation")
        raw_query = state.get("raw_query") or state.get("query", "")
        if pending and pending.get("phase") == _PHASE_CHOOSE:
            choices = [c for c in pending.get("choices") or [] if isinstance(c, dict)]
            chosen = _resolve_choice(choices, raw_query)
            if chosen is None:
                # Do not strand the user on the menu: an answer that names
                # no option is a change of subject, so hand the turn back to
                # ordinary retrieval rather than re-asking forever.
                return {
                    "calc_route": "normal",
                    "pending_calculation": None,
                    "awaiting_clarification": False,
                    "pending_sections": [],
                }
            return _start_calculation(client, lang, chosen, pending.get("raw_query") or raw_query)

        if pending and pending.get("phase"):
            return _resume_comparison(client, lang, pending, raw_query)

        if pending:
            calculator_id = pending.get("calculator_id")
            if not calculator_id:
                return _failure_update(lang)
            inputs_so_far = dict(pending.get("inputs_so_far") or {})
            current_round = int(pending.get("round") or 0)
            specs = pending.get("missing_inputs") or []
            conversions = [
                record for record in pending.get("conversions") or []
                if isinstance(record, dict)
            ]

            # A bare "mensile" answering the frequency question: the amount is
            # already held, so resolve it here rather than sending a message
            # with no number through extraction, which would find nothing and
            # read the turn as a change of subject.
            held = pending.get("pending_frequency") or {}
            if held:
                (
                    resolved,
                    new_conversions,
                    held_unresolved,
                ) = normalization.resolve_pending_frequency(raw_query, held)
                # A frequency arrived but the amount still cannot be used —
                # its currency is wrong or ambiguous. Ask again rather than
                # annualize a figure denominated in something this calculator
                # does not accept.
                if held_unresolved:
                    return _unresolved_frequency_update(
                        lang,
                        calculator_id=calculator_id,
                        inputs_so_far=inputs_so_far,
                        unresolved=held_unresolved,
                        conversions=conversions,
                        specs=specs,
                        current_round=current_round,
                    )
                if resolved:
                    inputs_so_far.update(resolved)
                    conversions += new_conversions
                    response = client.calculate(
                        _calculation_payload(calculator_id, inputs_so_far)
                    )
                    if response.get("platform_unavailable"):
                        return _failure_update(lang, pending_calculation=pending)
                    return _handle_response(
                        response,
                        lang=lang,
                        calculator_id=calculator_id,
                        inputs_so_far=inputs_so_far,
                        current_round=current_round,
                        conversions=conversions,
                    )

            # Backward-compatible recovery for a pending payload without specs.
            if not specs:
                probe = client.calculate(_calculation_payload(calculator_id, inputs_so_far))
                if probe.get("platform_unavailable"):
                    return _failure_update(lang, pending_calculation=pending)
                if probe.get("status") == "success":
                    return _handle_response(
                        probe,
                        lang=lang,
                        calculator_id=calculator_id,
                        inputs_so_far=inputs_so_far,
                        current_round=current_round,
                    )
                specs = _missing_specs(probe)
                if not specs:
                    return _failure_update(
                        lang,
                        platform_message=_platform_error_message(
                            probe, include_validation=True
                        ),
                    )

            extracted = _extract_values_llm(
                raw_query,
                calculator_id,
                missing_specs=specs,
                prior_inputs=inputs_so_far,
            )
            if extracted is None:
                extracted = _extract_values(raw_query, specs)
            # Only accept values that fill a currently-missing field. A follow-up
            # that yields nothing relevant — e.g. an unrelated question that
            # merely contains a stray number — is a topic change, not a slot
            # answer: escape to normal RAG instead of mis-binding the number.
            # Restricting to missing fields also prevents silently overwriting a
            # value confirmed in an earlier turn.
            missing_names = {
                spec.get("name")
                for spec in specs
                if isinstance(spec, dict) and spec.get("name")
            }
            if missing_names:
                extracted = {
                    name: value
                    for name, value in (extracted or {}).items()
                    if name in missing_names
                }
            if not extracted:
                return {
                    "calc_route": "normal",
                    "pending_calculation": None,
                    "awaiting_clarification": False,
                    "pending_sections": [],
                }
            # The follow-up gets the same treatment as the opening message: a
            # rent restated as "400 euro al mese" three turns in must convert
            # exactly as it would have on turn one.
            extracted, new_conversions, unresolved = _normalize_frequency_inputs(
                calculator_id, extracted, raw_query
            )
            if unresolved:
                return _unresolved_frequency_update(
                    lang,
                    calculator_id=calculator_id,
                    inputs_so_far=inputs_so_far,
                    unresolved=unresolved,
                    conversions=conversions,
                    specs=specs,
                    current_round=current_round,
                )
            conversions += new_conversions
            inputs_so_far.update(extracted)
            response = client.calculate(_calculation_payload(calculator_id, inputs_so_far))
            if response.get("platform_unavailable"):
                return _failure_update(lang, pending_calculation=pending)
            return _handle_response(
                response,
                lang=lang,
                calculator_id=calculator_id,
                inputs_so_far=inputs_so_far,
                current_round=current_round,
                conversions=conversions,
            )

        choices = [c for c in state.get("calculation_choices") or [] if isinstance(c, dict)]
        if choices:
            return _answered_update(
                _choice_text(lang, choices),
                pending_calculation={
                    "phase": _PHASE_CHOOSE,
                    "calculator_id": None,
                    "choices": choices,
                    "raw_query": raw_query,
                    "round": 0,
                },
                retrieval_quality_ok=True,
            )

        return _start_calculation(client, lang, state.get("calculation_match") or {}, raw_query)
    except Exception:
        logger.exception("Unexpected calculation node failure")
        return _failure_update(lang)
