"""Locale-aware numeric corroboration.

`1.200` is 1200 to an Italian writer and 1.2 to an English one. The first
version of corroboration accepted BOTH readings, so an extracted 1.2 was
vouched for by Italian text that plainly said 1200 — and 1.2 annualized to
14.40 instead of 14400.

The session language is already resolved by the time normalization runs, so the
separator does not have to be guessed. These tests pin that the resolved
language decides, that punctuation alone is never used to infer a locale when
the language is known, and that an unknown language falls back to accepting only
tokens every convention reads the same way.
"""

import os
from decimal import Decimal

import pytest
import requests

os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test-password")

import src.rag.calculation as calculation
from src.rag import normalization
from src.rag.calculation import calculation_node

LEASE = "legal_it.registration_tax_leases"


class _Response:
    def __init__(self, body, status_code=200):
        self._body = body
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)

    def json(self):
        return self._body


def _mock_http(monkeypatch, *responses):
    queued = iter(responses)
    calls = []

    def fake_post(url, **kwargs):
        calls.append({"url": url, **kwargs})
        response = next(queued)
        if isinstance(response, Exception):
            raise response
        return response if isinstance(response, _Response) else _Response(response)

    monkeypatch.setattr("src.rag.calculation.requests.post", fake_post)
    return calls


def _lease_match():
    return {
        "calculator_id": LEASE,
        "required_inputs": [
            {"name": "annual_rent", "type": "decimal", "required": True, "unit": "EUR"},
            {"name": "years", "type": "decimal", "required": True},
            {"name": "first_registration", "type": "boolean", "required": True},
        ],
        "optional_inputs": [],
    }


def _run(monkeypatch, query, lang, extracted, *responses):
    monkeypatch.setattr(
        calculation,
        "_extract_values_llm",
        lambda *a, **k: {"annual_rent": extracted, "years": "4",
                         "first_registration": True},
    )
    calls = _mock_http(monkeypatch, *responses)
    update = calculation_node(
        {"query": query, "session_language": lang,
         "calculation_match": _lease_match()}
    )
    return update, calls


def _success():
    return {"calculator_id": LEASE, "status": "success",
            "result": {"tax_due": "1152.00"}}


# --- Italian: dot groups, comma decimals -----------------------------------

def test_italian_grouped_thousands_corroborate_the_grouped_value(monkeypatch):
    _, calls = _run(
        monkeypatch, "canone 1.200 euro al mese", "it", "1200", _success()
    )

    assert calls[0]["json"]["inputs"]["annual_rent"] == "14400"


def test_italian_grouped_thousands_reject_the_english_reading(monkeypatch):
    """In Italian text, 1.200 is not 1.2 — so an extracted 1.2 is a different
    number and must not be corroborated, let alone annualized to 14.40."""
    update, calls = _run(monkeypatch, "canone 1.200 euro al mese", "it", "1.2")

    assert calls == []
    assert "14.40" not in update["answer"]
    assert "14400" not in update["answer"]


# --- English: comma groups, dot decimals -----------------------------------

def test_english_grouped_thousands_corroborate_the_grouped_value(monkeypatch):
    _, calls = _run(
        monkeypatch, "rent 1,200 euro per month", "en", "1200", _success()
    )

    assert calls[0]["json"]["inputs"]["annual_rent"] == "14400"


def test_english_grouped_thousands_reject_the_italian_reading(monkeypatch):
    update, calls = _run(monkeypatch, "rent 1,200 euro per month", "en", "1.2")

    assert calls == []
    assert "14400" not in update["answer"]


# --- Spanish shares the Italian convention ---------------------------------

def test_spanish_grouped_and_decimal_forms(monkeypatch):
    _, calls = _run(
        monkeypatch, "alquiler 1.200 euros al mes", "es", "1200", _success()
    )
    assert calls[0]["json"]["inputs"]["annual_rent"] == "14400"


def test_spanish_decimal_comma_is_a_decimal(monkeypatch):
    _, calls = _run(
        monkeypatch, "alquiler 1.200,50 euros al mes", "es", "1200.50", _success()
    )
    assert Decimal(calls[0]["json"]["inputs"]["annual_rent"]) == Decimal("14406")


def test_spanish_rejects_the_english_reading(monkeypatch):
    _, calls = _run(monkeypatch, "alquiler 1.200 euros al mes", "es", "1.2")
    assert calls == []


# --- The token reader, per locale -----------------------------------------

@pytest.mark.parametrize(
    "lang,token,expected",
    [
        # Italian / Spanish: '.' groups, ',' decimals
        ("it", "1.200", ["1200"]),
        ("it", "1.200,50", ["1200.50"]),
        ("it", "1,20", ["1.20"]),
        ("it", "400", ["400"]),
        ("it", "400,50", ["400.50"]),
        ("es", "1.200", ["1200"]),
        ("es", "1.200,50", ["1200.50"]),
        ("es", "1,20", ["1.20"]),
        # A token that is malformed for the locale corroborates nothing.
        ("it", "1,200.50", []),
        ("en", "1.200,50", []),
        # English: ',' groups, '.' decimals
        ("en", "1,200", ["1200"]),
        ("en", "1,200.50", ["1200.50"]),
        ("en", "1.20", ["1.20"]),
        ("en", "400", ["400"]),
        ("en", "400.50", ["400.50"]),
    ],
)
def test_token_values_follow_the_session_language(lang, token, expected):
    assert normalization._token_values(token, lang) == {
        Decimal(value) for value in expected
    }


@pytest.mark.parametrize(
    "token,expected",
    [
        # Unambiguous under every convention.
        ("400", ["400"]),
        ("400,50", ["400.50"]),   # ',50' is too short to be a thousands group
        ("400.50", ["400.50"]),
        ("1.200,50", ["1200.50"]),
        ("1,200.50", ["1200.50"]),
        # Genuinely ambiguous with no language to settle it: unresolved rather
        # than accepting either reading.
        ("1.200", []),
        ("1,200", []),
    ],
)
def test_unknown_language_accepts_only_unambiguous_tokens(token, expected):
    assert normalization._token_values(token, None) == {
        Decimal(value) for value in expected
    }


def test_unknown_language_leaves_an_ambiguous_amount_unresolved():
    frequency, _ = normalization.read_frequency(
        "canone 1.200 euro al mese", "1200", None
    )
    assert frequency == normalization.FREQUENCY_UNKNOWN


# --- The previously-green cases stay green --------------------------------

@pytest.mark.parametrize(
    "lang,text,value,corroborated",
    [
        ("it", "canone di 400 euro al mese", "400", True),
        ("it", "canone di 400,50 euro al mese", "400.50", True),
        ("it", "canone di 400,50 euro al mese", "400.99", False),
        ("it", "canone di 400,50 euro al mese", "400", False),
        ("it", "canone di 1.200,50 euro al mese", "1200.50", True),
        ("it", "canone di 1.200,50 euro al mese", "1200", False),
        ("it", "canone di 4.800 euro all'anno", "4800", True),
        ("en", "rent of 1,200.50 per month", "1200.50", True),
        ("en", "rent of 400 per month", "400", True),
    ],
)
def test_existing_corroboration_cases_are_unchanged(lang, text, value, corroborated):
    frequency, _ = normalization.read_frequency(text, value, lang)
    assert (frequency != normalization.FREQUENCY_UNKNOWN) is corroborated


def test_locale_is_not_inferred_from_punctuation_when_the_language_is_known():
    """The guarantee the requirement asks for, stated directly.

    An English-formatted token inside an Italian session is malformed for that
    session's convention. It must corroborate nothing, rather than being read
    with whichever convention its punctuation happens to suit.
    """
    assert normalization._token_values("1,200.50", "it") == set()
    assert normalization._token_values("1.200,50", "en") == set()
