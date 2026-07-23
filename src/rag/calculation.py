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
from typing import Any, Dict, Iterable, List, Optional

import requests

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
            base_url or os.getenv("CALC_PLATFORM_URL", "http://localhost:8000")
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


def calculation_gate(state: Dict[str, Any]) -> Dict[str, Any]:
    """Route only a clear, high-scoring platform match to calculation."""
    try:
        response = PlatformClient().match(state.get("query", ""))
        candidates = response.get("candidates") or []
        top = candidates[0] if candidates and isinstance(candidates[0], dict) else None
        score = top.get("score", 0) if top else 0
        if (
            response.get("status") == "matched"
            and isinstance(score, (int, float))
            and score >= MATCH_AUTO_ROUTE_MIN_SCORE
        ):
            return {"calc_route": "calculate", "calculation_match": top}
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


def _relaxed_extraction_schema(input_schema: Dict[str, Any]) -> Dict[str, Any]:
    """Copy the platform schema while allowing partial LLM extraction output."""
    extraction_schema = copy.deepcopy(input_schema)
    extraction_schema.pop("required", None)
    properties = extraction_schema.get("properties")
    if isinstance(properties, dict):
        period_schema = properties.get("period")
        if isinstance(period_schema, dict):
            period_schema.pop("required", None)
    return extraction_schema


_DROP_EXTRACTED_VALUE = object()


def _clean_extracted_value(value: Any, schema: Dict[str, Any]) -> Any:
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    if value is None:
        return _DROP_EXTRACTED_VALUE

    schema_type = schema.get("type")
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
) -> Optional[Dict[str, Any]]:
    """Extract explicit calculator arguments through tools, then JSON mode."""
    try:
        tool_schema = PlatformClient().tool_schema(calculator_id)
        input_schema = tool_schema.get("input_schema")
        properties = input_schema.get("properties") if isinstance(input_schema, dict) else None
        tool_name = tool_schema.get("name")
        if not isinstance(properties, dict) or not isinstance(tool_name, str):
            logger.warning("No usable tool schema for calculator %s", calculator_id)
            return None
        extraction_schema = _relaxed_extraction_schema(input_schema)

        tool = {
            "name": tool_name,
            "description": str(tool_schema.get("description") or "Extract calculator inputs"),
            "parameters": extraction_schema,
        }
        messages = _extraction_messages(
            query,
            extraction_schema,
            missing_specs=missing_specs,
            prior_inputs=prior_inputs,
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
    },
    "es": {
        "result": "Resultado",
        "sources": "Fuentes",
        "no_sources": "ninguna fuente indicada por la plataforma",
        "clarify": "Para completar el cálculo, ¿puedes indicarme {items}?",
        "failure": "No puedo completar el cálculo en este momento. Inténtalo de nuevo más tarde.",
        "round_limit": "Lo siento, no he podido recopilar todos los datos necesarios para el cálculo.",
    },
    "en": {
        "result": "Result",
        "sources": "Sources",
        "no_sources": "no sources supplied by the platform",
        "clarify": "To complete the calculation, could you provide {items}?",
        "failure": "I cannot complete the calculation right now. Please try again shortly.",
        "round_limit": "Sorry, I could not collect all the information needed for the calculation.",
    },
}


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


def _display_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _success_answer(lang: str, response: Dict[str, Any]) -> str:
    result = response.get("result") or {}
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
    return f"{_COPY[lang]['result']}: {rendered}\n\n{_COPY[lang]['sources']}: {sources}"


def _answered_update(answer: str, **updates: Any) -> Dict[str, Any]:
    """Build a calculation answer that also clears stale legal clarification state."""
    return {
        "answer": answer,
        "awaiting_clarification": False,
        "pending_sections": [],
        **updates,
    }


def _platform_error_message(response: Dict[str, Any]) -> Optional[str]:
    """Return a genuine platform failure message, excluding input validation."""
    for error in response.get("errors") or []:
        if not isinstance(error, dict) or error.get("code") == "input_invalid":
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
) -> Dict[str, Any]:
    if response.get("status") == "success":
        return _answered_update(
            _success_answer(lang, response),
            calculation_result=response.get("result") or {},
            pending_calculation=None,
            retrieval_quality_ok=True,
        )

    missing = _missing_specs(response)
    if missing:
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
        return _answered_update(
            _clarification_question(lang, missing),
            pending_calculation={
                "calculator_id": calculator_id,
                "inputs_so_far": inputs_so_far,
                "round": next_round,
                "missing_inputs": missing,
            },
            retrieval_quality_ok=True,
        )
    return _failure_update(
        lang,
        platform_message=_platform_error_message(response),
    )


def calculation_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Run a fresh or continued deterministic calculation, never raising."""
    lang = "it"
    try:
        lang = _session_lang(state)
        client = PlatformClient()
        pending = state.get("pending_calculation")
        raw_query = state.get("raw_query") or state.get("query", "")
        if pending:
            calculator_id = pending.get("calculator_id")
            if not calculator_id:
                return _failure_update(lang)
            inputs_so_far = dict(pending.get("inputs_so_far") or {})
            current_round = int(pending.get("round") or 0)
            specs = pending.get("missing_inputs") or []

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
                        platform_message=_platform_error_message(probe),
                    )

            extracted = _extract_values_llm(
                raw_query,
                calculator_id,
                missing_specs=specs,
                prior_inputs=inputs_so_far,
            )
            if extracted is None:
                extracted = _extract_values(raw_query, specs)
            if not extracted:
                return {
                    "calc_route": "normal",
                    "pending_calculation": None,
                    "awaiting_clarification": False,
                    "pending_sections": [],
                }
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
            )

        match = state.get("calculation_match") or {}
        calculator_id = match.get("calculator_id")
        if not calculator_id:
            return _failure_update(lang)
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
        inputs_so_far = _extract_values_llm(raw_query, calculator_id)
        if inputs_so_far is None:
            inputs_so_far = _extract_values(
                raw_query,
                specs,
                supports_tax_year=bool(match.get("supports_tax_year")),
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
        )
    except Exception:
        logger.exception("Unexpected calculation node failure")
        return _failure_update(lang)
