"""Deterministic HTTP bridge from the RAG graph to the calculation platform."""

from __future__ import annotations

import json
import logging
import os
import re
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
        except Exception as exc:
            logger.warning("Calculation platform request failed for %s: %s", path, exc)
            return dict(_PLATFORM_UNAVAILABLE)


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


# SLICE-1 STAND-IN: replaced by real LLM tool-calling in Slice 2
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
