"""Phase 2A: deterministic monthly-to-annual normalization for `annual_rent`.

`legal_it.registration_tax_leases` requires `annual_rent`, but people state
rents monthly ("un canone di 400 euro al mese"). Production told the LLM to
extract only explicitly stated values and had no conversion layer behind it, so
the three available outcomes were all wrong: omit the rent, or pass 400 as if it
were annual, or let the positional fallback bind the first number it saw. 400
instead of 4800 understates the tax by a factor of twelve, silently.

The contract these tests pin:

  * the LLM identifies the amount AS WRITTEN; every arithmetic step is done
    here, in Decimal, under the named rule `monthly_to_annual_x12`;
  * a conversion is never silent — the answer shows the multiplication;
  * an amount with no stated frequency is a QUESTION, never a guess;
  * a foreign currency is refused outright: no exchange rate is invented;
  * only fields declared frequency-sensitive are affected. Every other
    monetary input keeps its existing behaviour.
"""

import json
import os
from decimal import Decimal

import pytest
import requests

os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test-password")

import src.rag.calculation as calculation
from src.rag.calculation import calculation_node

LEASE = "legal_it.registration_tax_leases"
MONTHLY_RULE = "monthly_to_annual_x12"


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
    """The lease calculator as /match reports it."""
    return {
        "calculator_id": LEASE,
        "name": "Imposta di registro sui contratti di locazione",
        "required_inputs": [
            {
                "name": "annual_rent",
                "type": "decimal",
                "required": True,
                "unit": "EUR",
                "description": "Canone di locazione (affitto) annuo pattuito",
            },
            {"name": "years", "type": "decimal", "required": True, "unit": "years"},
            {"name": "first_registration", "type": "boolean", "required": True},
        ],
        "optional_inputs": [],
    }


def _success(tax="768.00"):
    return {
        "calculator_id": LEASE,
        "status": "success",
        "result": {"tax_due": tax},
        "citations": [{"reference": "Art. 5, D.P.R. 131/1986"}],
    }


def _extraction(monkeypatch, values):
    """Stand in for the LLM, which reports the amount exactly AS WRITTEN.

    Returning 400 for "400 euro al mese" is the contract, not a bug in the
    fixture: converting is this layer's job, never the model's.
    """
    monkeypatch.setattr(
        calculation, "_extract_values_llm", lambda *args, **kwargs: dict(values)
    )


def _fresh(query, monkeypatch, extracted, *responses):
    _extraction(monkeypatch, extracted)
    calls = _mock_http(monkeypatch, *responses)
    update = calculation_node(
        {"query": query, "session_language": "it", "calculation_match": _lease_match()}
    )
    return update, calls


# --- 1. Monthly is converted, deterministically ----------------------------

def test_monthly_rent_is_converted_to_annual_before_calculating(monkeypatch):
    update, calls = _fresh(
        "Calcola l'imposta di registro per un canone di 400 euro al mese",
        monkeypatch,
        {"annual_rent": "400", "years": "4", "first_registration": True},
        _success(),
    )

    assert len(calls) == 1
    payload = calls[0]["json"]
    assert payload["inputs"]["annual_rent"] == "4800"
    # The number the user said must never reach the platform as an annual rent.
    assert payload["inputs"]["annual_rent"] != "400"
    assert "768.00" in update["answer"]


def test_conversion_is_shown_to_the_user_never_silent(monkeypatch):
    update, _ = _fresh(
        "Calcola l'imposta di registro per un canone di 400 euro al mese",
        monkeypatch,
        {"annual_rent": "400", "years": "4", "first_registration": True},
        _success(),
    )

    answer = update["answer"]
    assert "400" in answer and "4.800" in answer
    assert "12" in answer
    assert "mese" in answer and "anno" in answer


def test_conversion_audit_record_is_json_serializable_and_complete(monkeypatch):
    update, _ = _fresh(
        "Calcola l'imposta di registro per un canone di 400 euro al mese",
        monkeypatch,
        {"annual_rent": "400", "years": "4", "first_registration": True},
        _success(),
    )

    conversions = update["calculation_conversions"]
    assert len(conversions) == 1
    record = conversions[0]
    assert record["field"] == "annual_rent"
    assert record["raw_value"] == "400"
    assert record["source_frequency"] == "monthly"
    assert record["canonical_value"] == "4800"
    assert record["rule_id"] == MONTHLY_RULE
    # It rides along in session state, so it has to survive a JSON round trip.
    assert json.loads(json.dumps(record)) == record


@pytest.mark.parametrize(
    "query",
    [
        "Calcola l'imposta di registro per un canone di 400 euro al mese",
        "Calcola l'imposta di registro, canone mensile di 400 euro",
        "Imposta di registro su un canone di 400 euro/mese",
    ],
)
def test_monthly_phrasings_all_convert(monkeypatch, query):
    _, calls = _fresh(
        query,
        monkeypatch,
        {"annual_rent": "400", "years": "4", "first_registration": True},
        _success(),
    )

    assert calls[0]["json"]["inputs"]["annual_rent"] == "4800"


# --- 2. An explicitly annual amount is passed through untouched ------------

@pytest.mark.parametrize(
    "query",
    [
        "Calcola l'imposta di registro per un canone annuo di 4800 euro",
        "Calcola l'imposta di registro per un canone di 4.800 euro all'anno",
    ],
)
def test_annual_rent_is_not_converted(monkeypatch, query):
    update, calls = _fresh(
        query,
        monkeypatch,
        {"annual_rent": "4800", "years": "4", "first_registration": True},
        _success(),
    )

    assert calls[0]["json"]["inputs"]["annual_rent"] == "4800"
    # Nothing was converted, so nothing may be reported as converted.
    assert update.get("calculation_conversions") in (None, [])


# --- 3. No stated frequency is a question, not a guess --------------------

def test_rent_without_a_frequency_asks_instead_of_calculating(monkeypatch):
    _extraction(monkeypatch, {"annual_rent": "400", "years": "4",
                              "first_registration": True})
    calls = _mock_http(monkeypatch)  # nothing queued: /calculate must not run

    update = calculation_node(
        {
            "query": "Calcola l'imposta di registro per un canone di 400 euro",
            "session_language": "it",
            "calculation_match": _lease_match(),
        }
    )

    assert calls == [], "an unresolved frequency must never reach /calculate"
    assert "mensile" in update["answer"].lower()
    assert "annuo" in update["answer"].lower()
    # 400 must NOT have been quietly accepted as the annual rent.
    pending = update["pending_calculation"]
    assert "annual_rent" not in (pending.get("inputs_so_far") or {})


def test_frequency_question_can_be_answered_with_a_bare_frequency(monkeypatch):
    """The question has to be answerable, or it is a dead end.

    The amount the user already gave is held in the pending payload so a reply
    of just "mensile" completes the calculation without making them repeat it.
    """
    _extraction(monkeypatch, {"annual_rent": "400", "years": "4",
                              "first_registration": True})
    _mock_http(monkeypatch)
    first = calculation_node(
        {
            "query": "Calcola l'imposta di registro per un canone di 400 euro",
            "session_language": "it",
            "calculation_match": _lease_match(),
        }
    )

    # Second turn: a bare frequency word, no number at all.
    _extraction(monkeypatch, {})
    calls = _mock_http(monkeypatch, _success())
    second = calculation_node(
        {
            "query": "mensile",
            "raw_query": "mensile",
            "session_language": "it",
            "pending_calculation": first["pending_calculation"],
        }
    )

    assert calls[0]["json"]["inputs"]["annual_rent"] == "4800"
    assert "768.00" in second["answer"]
    assert "4.800" in second["answer"]


# --- 4. Foreign currency is refused, never converted ---------------------

@pytest.mark.parametrize(
    "query",
    [
        "Calcola l'imposta di registro per un canone di 400 USD al mese",
        "Calcola l'imposta di registro for a rent of $400 per month",
    ],
)
def test_foreign_currency_asks_for_euro_and_never_invents_a_rate(monkeypatch, query):
    _extraction(monkeypatch, {"annual_rent": "400", "years": "4",
                              "first_registration": True})
    calls = _mock_http(monkeypatch)  # /calculate must not run

    update = calculation_node(
        {"query": query, "session_language": "it",
         "calculation_match": _lease_match()}
    )

    assert calls == [], "a currency mismatch must never reach /calculate"
    answer = update["answer"].lower()
    assert "euro" in answer
    # No exchange rate may appear anywhere: not the converted amount, and not
    # a rate-shaped number invented to get there.
    assert "4800" not in update["answer"] and "4.800" not in update["answer"]
    assert update.get("calculation_conversions") in (None, [])
    pending = update["pending_calculation"]
    assert "annual_rent" not in (pending.get("inputs_so_far") or {})


# --- 5. The same rules inside a pending-calculation follow-up -------------

def _pending_missing_rent():
    return {
        "calculator_id": LEASE,
        "inputs_so_far": {"years": "4", "first_registration": True},
        "round": 1,
        "missing_inputs": [
            {
                "name": "annual_rent",
                "type": "decimal",
                "required": True,
                "unit": "EUR",
                "description": "Canone annuo",
            }
        ],
    }


def test_follow_up_monthly_rent_is_converted(monkeypatch):
    _extraction(monkeypatch, {"annual_rent": "400"})
    calls = _mock_http(monkeypatch, _success())

    update = calculation_node(
        {
            "query": "400 euro al mese",
            "raw_query": "400 euro al mese",
            "session_language": "it",
            "pending_calculation": _pending_missing_rent(),
        }
    )

    assert calls[0]["json"]["inputs"]["annual_rent"] == "4800"
    assert "4.800" in update["answer"]


def test_follow_up_annual_rent_is_not_converted(monkeypatch):
    _extraction(monkeypatch, {"annual_rent": "4800"})
    calls = _mock_http(monkeypatch, _success())

    update = calculation_node(
        {
            "query": "4800 euro annui",
            "raw_query": "4800 euro annui",
            "session_language": "it",
            "pending_calculation": _pending_missing_rent(),
        }
    )

    assert calls[0]["json"]["inputs"]["annual_rent"] == "4800"
    assert update.get("calculation_conversions") in (None, [])


def test_follow_up_without_a_frequency_asks_again(monkeypatch):
    _extraction(monkeypatch, {"annual_rent": "400"})
    calls = _mock_http(monkeypatch)

    update = calculation_node(
        {
            "query": "400 euro",
            "raw_query": "400 euro",
            "session_language": "it",
            "pending_calculation": _pending_missing_rent(),
        }
    )

    assert calls == []
    assert "mensile" in update["answer"].lower()


def test_follow_up_in_foreign_currency_is_refused(monkeypatch):
    _extraction(monkeypatch, {"annual_rent": "400"})
    calls = _mock_http(monkeypatch)

    update = calculation_node(
        {
            "query": "400 USD al mese",
            "raw_query": "400 USD al mese",
            "session_language": "it",
            "pending_calculation": _pending_missing_rent(),
        }
    )

    assert calls == []
    assert "euro" in update["answer"].lower()
    assert "4800" not in update["answer"]


# --- 6. Scope: only declared frequency-sensitive fields ------------------

def test_other_calculators_are_untouched_even_with_a_frequency_in_the_text(monkeypatch):
    """A monthly cue near a number must not convert a field that never asked.

    `capital` on the legal-interest calculator has no frequency in its meaning;
    inventing one would multiply an unrelated input by twelve.
    """
    monkeypatch.setattr(
        calculation,
        "_extract_values_llm",
        lambda *args, **kwargs: {"capital": "8500"},
    )
    calls = _mock_http(
        monkeypatch,
        {
            "calculator_id": "legal_it.legal_interest",
            "status": "success",
            "result": {"interest": "100.00"},
        },
    )

    update = calculation_node(
        {
            "query": "Calcola gli interessi legali su 8500 euro versati ogni mese",
            "session_language": "it",
            "calculation_match": {
                "calculator_id": "legal_it.legal_interest",
                "required_inputs": [{"name": "capital", "type": "decimal"}],
            },
        }
    )

    assert calls[0]["json"]["inputs"]["capital"] == "8500"
    assert update.get("calculation_conversions") in (None, [])


def test_lease_fields_other_than_the_rent_are_not_converted(monkeypatch):
    """`years` sits next to a monthly cue but is not frequency-sensitive."""
    _, calls = _fresh(
        "Imposta di registro: canone 400 euro al mese, durata 4 anni",
        monkeypatch,
        {"annual_rent": "400", "years": "4", "first_registration": True},
        _success(),
    )

    assert calls[0]["json"]["inputs"]["years"] == "4"
    assert calls[0]["json"]["inputs"]["annual_rent"] == "4800"


# --- 7. Arithmetic stays deterministic and exact -------------------------

def test_conversion_uses_exact_decimal_arithmetic():
    from src.rag import normalization

    # A value float arithmetic would corrupt (0.1-style representation error).
    assert normalization.to_annual(Decimal("416.67")) == Decimal("5000.04")
    assert normalization.to_annual(Decimal("400")) == Decimal("4800")
    assert normalization.MONTHLY_TO_ANNUAL_RULE == MONTHLY_RULE


# --- 7b. The corroboration guard: never convert twice --------------------
# A conversion fires only when the extracted amount is found in the user's own
# words next to a frequency cue. These are the cases that guard depends on.


def test_a_model_that_pre_converted_is_not_converted_again(monkeypatch):
    """The most dangerous failure this design has to exclude.

    If the model ignores its instruction and returns the already-annualized
    4800 for "400 euro al mese", multiplying again would bill 57.600 as the
    annual rent. 4800 does not appear in the text, so the frequency reads as
    unknown and the layer asks instead of touching the number.
    """
    _extraction(monkeypatch, {"annual_rent": "4800", "years": "4",
                              "first_registration": True})
    calls = _mock_http(monkeypatch)

    update = calculation_node(
        {
            "query": "Calcola l'imposta di registro per un canone di 400 euro al mese",
            "session_language": "it",
            "calculation_match": _lease_match(),
        }
    )

    assert calls == []
    assert "57600" not in update["answer"] and "57.600" not in update["answer"]
    assert update.get("calculation_conversions") in (None, [])


def test_a_frequency_word_far_from_the_amount_does_not_label_it():
    """"annualità successiva" is about the tax year, not about the rent.

    Reading any frequency word anywhere in the sentence would make this phrase
    assert that 400 is an annual rent — the silent-400 bug with extra steps.
    """
    from src.rag import normalization

    frequency, _ = normalization.read_frequency(
        "Imposta di registro per annualita successiva, canone 400 euro", "400"
    )
    assert frequency == normalization.FREQUENCY_UNKNOWN


def test_conflicting_cues_for_the_same_amount_are_ambiguous():
    from src.rag import normalization

    frequency, _ = normalization.read_frequency(
        "400 euro al mese il primo anno, poi 400 euro all'anno", "400"
    )
    assert frequency == normalization.FREQUENCY_UNKNOWN


@pytest.mark.parametrize(
    "text,expected",
    [
        ("400 euro mensili", "monthly"),
        ("400 euro/mese", "monthly"),
        ("400 € al mese", "monthly"),
        ("4800 euro annuali", "annual"),
        ("4800 euro l'anno", "annual"),
        ("4800 euro all'anno", "annual"),
        ("4800 euro per anno", "annual"),
        ("rent of 4800 EUR per year", "annual"),
    ],
)
def test_real_phrasings_are_recognized(text, expected):
    from src.rag import normalization

    amount = "400" if expected == "monthly" else "4800"
    assert normalization.read_frequency(text, amount)[0] == expected


# --- 8. Document-generation routing is unaffected ------------------------

def test_generation_routing_is_unchanged_by_normalization(monkeypatch):
    """A drafting request carrying the same monthly rent must still not be
    calculated at all — normalization must not create a reason to route."""
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("EMBEDDING_API_KEY", "test-key")
    import src.chatbot.api as api

    monkeypatch.setattr(api, "_call_chat", lambda *args, **kwargs: "rag")
    message = "Redigi un contratto di locazione con un canone di 400 euro al mese."

    assert (
        api._classify_top_level_intent(message, "it") == "generate"
        or api.is_generation_request(message)
    )
    # And the gate still refuses to act in retrieval-only mode.
    assert calculation.calculation_gate(
        {"query": message, "skip_calculation": True}
    ) == {"calc_route": "normal"}
