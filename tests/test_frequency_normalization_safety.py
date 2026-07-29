"""Safety properties of the Phase 2A frequency layer.

Phase 2A established that a monthly rent must be converted deterministically and
never guessed. These tests cover the ways that guarantee could still be lost:

  1. normalization raising, and failing OPEN — the original 400 going to
     /calculate is exactly the bug the layer exists to prevent;
  2. corroboration matching on integer digits only, so text saying 400,50 would
     vouch for an extracted 400.99;
  3. a bare frequency reply carrying a foreign currency ("mensile in dollari")
     being read as EUR;
  4. frequency clarification looping without a round limit;
  5. the audit record being dropped when the session is saved;
  6. Italian "annualità" (a yearly INSTALMENT) read as an annual frequency;
  7. the internal field name leaking into what the user reads.
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
        "name": "Imposta di registro sui contratti di locazione",
        "required_inputs": [
            {"name": "annual_rent", "type": "decimal", "required": True, "unit": "EUR",
             "description": "Canone di locazione annuo pattuito"},
            {"name": "years", "type": "decimal", "required": True, "unit": "years"},
            {"name": "first_registration", "type": "boolean", "required": True},
        ],
        "optional_inputs": [],
    }


def _success(tax="384.00"):
    return {"calculator_id": LEASE, "status": "success", "result": {"tax_due": tax}}


def _extraction(monkeypatch, values):
    monkeypatch.setattr(
        calculation, "_extract_values_llm", lambda *a, **k: dict(values)
    )


def _pending_missing_rent(**overrides):
    pending = {
        "calculator_id": LEASE,
        "inputs_so_far": {"years": "4", "first_registration": True},
        "round": 1,
        "missing_inputs": [
            {"name": "annual_rent", "type": "decimal", "required": True, "unit": "EUR"}
        ],
    }
    pending.update(overrides)
    return pending


# === 1. Normalization must fail CLOSED =====================================

def test_normalization_failure_never_sends_the_raw_amount_to_calculate(monkeypatch):
    """The whole point of the layer, under its own failure.

    Failing open returns the extracted values untouched, which for a monthly
    rent means annual_rent=400 reaching /calculate — the silent 12x
    understatement Phase 2A exists to stop. A crash here must cost the user a
    question, never a wrong number.
    """
    _extraction(monkeypatch, {"annual_rent": "400", "years": "4",
                              "first_registration": True})

    def boom(*args, **kwargs):
        raise RuntimeError("normalization is broken")

    monkeypatch.setattr(normalization, "normalize_inputs", boom)
    calls = _mock_http(monkeypatch)

    update = calculation_node(
        {"query": "canone di 400 euro al mese", "session_language": "it",
         "calculation_match": _lease_match()}
    )

    assert calls == [], "/calculate must not be called when normalization fails"
    assert update["answer"]
    pending = update.get("pending_calculation") or {}
    assert "annual_rent" not in (pending.get("inputs_so_far") or {})


def test_normalization_failure_preserves_unrelated_inputs(monkeypatch):
    """Only the frequency-sensitive field is in doubt.

    `years` and `first_registration` were never touched by this layer, so
    discarding them would make the user restate facts that are not in question.
    """
    _extraction(monkeypatch, {"annual_rent": "400", "years": "4",
                              "first_registration": True})
    monkeypatch.setattr(
        normalization,
        "normalize_inputs",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("broken")),
    )
    _mock_http(monkeypatch)

    update = calculation_node(
        {"query": "canone di 400 euro al mese per 4 anni", "session_language": "it",
         "calculation_match": _lease_match()}
    )

    inputs = (update.get("pending_calculation") or {}).get("inputs_so_far") or {}
    assert inputs.get("years") == "4"
    assert inputs.get("first_registration") is True


def test_normalization_failure_in_a_follow_up_also_fails_closed(monkeypatch):
    _extraction(monkeypatch, {"annual_rent": "400"})
    monkeypatch.setattr(
        normalization,
        "normalize_inputs",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("broken")),
    )
    calls = _mock_http(monkeypatch)

    update = calculation_node(
        {"query": "400 euro al mese", "raw_query": "400 euro al mese",
         "session_language": "it", "pending_calculation": _pending_missing_rent()}
    )

    assert calls == []
    assert update["answer"]
    assert "annual_rent" not in (
        (update.get("pending_calculation") or {}).get("inputs_so_far") or {}
    )


# === 2. Corroboration must be an EXACT numeric match =======================

def test_decimal_amount_is_corroborated_and_converted_exactly(monkeypatch):
    _extraction(monkeypatch, {"annual_rent": "400.50", "years": "4",
                              "first_registration": True})
    calls = _mock_http(monkeypatch, _success())

    calculation_node(
        {"query": "canone di 400,50 euro al mese", "session_language": "it",
         "calculation_match": _lease_match()}
    )

    # 400.50 x 12 = 4806 exactly. An integral product is rendered without a
    # trailing scale, the same way 400 is sent as "400" and not "400.00".
    sent = calls[0]["json"]["inputs"]["annual_rent"]
    assert Decimal(sent) == Decimal("4806")
    assert sent == "4806"


def test_a_different_decimal_is_not_corroborated_by_an_integer_prefix(monkeypatch):
    """400,50 in the text must not vouch for an extracted 400.99.

    Matching on integer digits alone made every amount sharing an integer part
    interchangeable, so a misread cent value would be converted as if the user
    had written it.
    """
    _extraction(monkeypatch, {"annual_rent": "400.99", "years": "4",
                              "first_registration": True})
    calls = _mock_http(monkeypatch)

    update = calculation_node(
        {"query": "canone di 400,50 euro al mese", "session_language": "it",
         "calculation_match": _lease_match()}
    )

    assert calls == [], "an uncorroborated amount must never be calculated"
    assert "4806" not in update["answer"]
    assert "4811" not in update["answer"]  # 400.99 x 12, had it been trusted


def test_grouped_decimal_amount_converts_exactly(monkeypatch):
    _extraction(monkeypatch, {"annual_rent": "1200.50", "years": "4",
                              "first_registration": True})
    calls = _mock_http(monkeypatch, _success())

    calculation_node(
        {"query": "canone di 1.200,50 euro al mese", "session_language": "it",
         "calculation_match": _lease_match()}
    )

    sent = calls[0]["json"]["inputs"]["annual_rent"]
    assert Decimal(sent) == Decimal("14406")
    assert sent == "14406"


@pytest.mark.parametrize(
    "text,value,corroborated",
    [
        ("canone di 400,50 euro al mese", "400.50", True),
        ("canone di 400,50 euro al mese", "400.99", False),
        # An integer prefix of a decimal the user wrote is a DIFFERENT number.
        ("canone di 400,50 euro al mese", "400", False),
        ("canone di 1.200,50 euro al mese", "1200.50", True),
        ("canone di 1.200,50 euro al mese", "1200", False),
        ("canone di 1.200 euro al mese", "1200", True),
        ("rent of 1,200.50 per month", "1200.50", True),
        ("canone di 400 euro al mese", "400", True),
        ("canone di 4.800 euro all'anno", "4800", True),
    ],
)
def test_exact_amount_corroboration(text, value, corroborated):
    frequency, _ = normalization.read_frequency(text, value)
    assert (frequency != normalization.FREQUENCY_UNKNOWN) is corroborated


def test_conversion_of_a_decimal_amount_is_exact_decimal_arithmetic():
    assert normalization.to_annual(Decimal("400.50")) == Decimal("4806.00")
    assert normalization.to_annual(Decimal("1200.50")) == Decimal("14406.00")


# === 3. A bare frequency reply must carry a valid currency =================

@pytest.mark.parametrize(
    "reply", ["mensile in dollari", "monthly USD", "mensile in $"]
)
def test_foreign_currency_in_a_bare_frequency_reply_is_refused(monkeypatch, reply):
    held = _pending_missing_rent(
        pending_frequency={
            "annual_rent": {"raw_value": "400", "reason": "frequency_unknown",
                            "currency": "EUR"}
        }
    )
    _extraction(monkeypatch, {})
    calls = _mock_http(monkeypatch)

    update = calculation_node(
        {"query": reply, "raw_query": reply, "session_language": "it",
         "pending_calculation": held}
    )

    assert calls == [], "a foreign-currency reply must never be calculated"
    assert "euro" in update["answer"].lower()
    assert "4800" not in update["answer"] and "4.800" not in update["answer"]
    # The user must still be able to correct it without restating the amount.
    assert (update.get("pending_calculation") or {}).get("pending_frequency")


def test_euro_in_a_bare_frequency_reply_still_resolves(monkeypatch):
    held = _pending_missing_rent(
        pending_frequency={
            "annual_rent": {"raw_value": "400", "reason": "frequency_unknown",
                            "currency": "EUR"}
        }
    )
    _extraction(monkeypatch, {})
    calls = _mock_http(monkeypatch, _success())

    update = calculation_node(
        {"query": "mensile in euro", "raw_query": "mensile in euro",
         "session_language": "it", "pending_calculation": held}
    )

    assert calls[0]["json"]["inputs"]["annual_rent"] == "4800"
    assert "384.00" in update["answer"]


def test_conflicting_currency_cues_are_ambiguous_not_unspecified():
    """Two currencies named at once is ambiguity, and ambiguity must stop the
    calculation. Reading it as "no currency stated" would silently accept the
    calculator's own EUR."""
    assert normalization.read_currency("in euro e in dollari") == (
        normalization.CURRENCY_AMBIGUOUS
    )
    assert normalization.read_currency("in euro") == "EUR"
    assert normalization.read_currency("mensile") is None


def test_conflicting_currency_next_to_an_amount_is_not_calculated(monkeypatch):
    _extraction(monkeypatch, {"annual_rent": "400", "years": "4",
                              "first_registration": True})
    calls = _mock_http(monkeypatch)

    update = calculation_node(
        {"query": "canone di 400 euro o dollari al mese", "session_language": "it",
         "calculation_match": _lease_match()}
    )

    assert calls == []
    assert update["answer"]


# === 4. Frequency clarification obeys the round limit =====================

def test_final_allowed_frequency_round_still_asks(monkeypatch):
    _extraction(monkeypatch, {"annual_rent": "400"})
    _mock_http(monkeypatch)

    update = calculation_node(
        {"query": "400 euro", "raw_query": "400 euro", "session_language": "it",
         "pending_calculation": _pending_missing_rent(
             round=calculation._MAX_CLARIFICATION_ROUNDS - 1
         )}
    )

    pending = update["pending_calculation"]
    assert pending["round"] == calculation._MAX_CLARIFICATION_ROUNDS
    assert "mensile" in update["answer"].lower()


def test_frequency_clarification_stops_at_the_round_limit(monkeypatch):
    _extraction(monkeypatch, {"annual_rent": "400"})
    calls = _mock_http(monkeypatch)

    update = calculation_node(
        {"query": "400 euro", "raw_query": "400 euro", "session_language": "it",
         "pending_calculation": _pending_missing_rent(
             round=calculation._MAX_CLARIFICATION_ROUNDS
         )}
    )

    assert calls == []
    assert update["answer"] == calculation._COPY["it"]["round_limit"]
    assert update["pending_calculation"] is None


# === 5. The audit record must survive being saved =========================

def test_conversions_are_persisted_on_the_assistant_message(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("EMBEDDING_API_KEY", "test-key")
    from src.chatbot.session import ChatSession, Message

    record = {
        "field": "annual_rent",
        "raw_value": "400",
        "source_frequency": "monthly",
        "canonical_value": "4800",
        "rule_id": normalization.MONTHLY_TO_ANNUAL_RULE,
    }
    session = ChatSession(session_id="audit")
    session.messages = [
        Message(role="assistant", content="Risultato",
                metadata={"calculation_conversions": [record]})
    ]

    serialized = session.to_dict()
    assert serialized["messages"][0]["calculation_conversions"] == [record]
    restored = ChatSession.from_dict(json.loads(json.dumps(serialized)))
    assert restored.messages[0].metadata["calculation_conversions"] == [record]


def test_ordinary_calculations_do_not_store_an_empty_conversions_key(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("EMBEDDING_API_KEY", "test-key")
    from src.chatbot.session import ChatSession, Message

    session = ChatSession(session_id="no-audit")
    session.messages = [
        Message(role="assistant", content="Risultato", metadata={"citations": []})
    ]

    assert "calculation_conversions" not in session.to_dict()["messages"][0]


def test_chat_stores_conversions_from_the_graph_result(monkeypatch):
    """The value has to travel from graph output into message metadata."""
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("EMBEDDING_API_KEY", "test-key")
    import threading

    from src.chatbot.session import ChatBot, ChatSession, Message

    record = {"field": "annual_rent", "raw_value": "400",
              "source_frequency": "monthly", "canonical_value": "4800",
              "rule_id": normalization.MONTHLY_TO_ANNUAL_RULE}
    session = ChatSession(
        session_id="s",
        messages=[Message(role="user", content="ciao"),
                  Message(role="assistant", content="ciao")],
        _language_fixed_from_first_turn=True,
    )
    bot = ChatBot.__new__(ChatBot)
    bot._sessions = {session.session_id: session}
    bot._lock = threading.Lock()
    bot._save_sessions = lambda: None

    monkeypatch.setattr(
        "src.chatbot.session._rewrite_query_with_context", lambda *a: "q"
    )
    monkeypatch.setattr(
        "src.chatbot.session.get_user_settings_by_email",
        lambda *a: {"tone": 2, "standing": 2, "response_length": 2},
    )
    monkeypatch.setattr(
        "src.chatbot.session.rag_run",
        lambda query, **kwargs: {
            "answer": "Risultato: tax_due: 384.00",
            "retrieval_quality_ok": True,
            "calculation_conversions": [record],
        },
    )

    bot.chat(session.session_id, "canone 400 euro al mese")

    assistant = [m for m in session.messages if m.role == "assistant"][-1]
    assert (assistant.metadata or {}).get("calculation_conversions") == [record]


# === 6. Language cues ====================================================

@pytest.mark.parametrize(
    "text,expected",
    [
        # "annualita" is a yearly INSTALMENT, not a statement about the rent's
        # frequency — being close to the amount must not make it one.
        ("canone 400 euro, annualita successiva", normalization.FREQUENCY_UNKNOWN),
        ("annualita di 400 euro", normalization.FREQUENCY_UNKNOWN),
        # Spanish
        ("alquiler de 4800 euros anual", normalization.FREQUENCY_ANNUAL),
        ("alquiler de 4800 euros anuales", normalization.FREQUENCY_ANNUAL),
        ("alquiler de 400 euros al mes", normalization.FREQUENCY_MONTHLY),
        ("alquiler de 400 euros mensuales", normalization.FREQUENCY_MONTHLY),
        # Italian and English forms that must keep working
        ("canone di 4800 euro annuo", normalization.FREQUENCY_ANNUAL),
        ("canone di 4800 euro annui", normalization.FREQUENCY_ANNUAL),
        ("canone di 4800 euro annuale", normalization.FREQUENCY_ANNUAL),
        ("canone di 4800 euro annuali", normalization.FREQUENCY_ANNUAL),
        ("canone di 4800 euro all'anno", normalization.FREQUENCY_ANNUAL),
        ("rent of 4800 per year", normalization.FREQUENCY_ANNUAL),
        ("rent of 4800 annually", normalization.FREQUENCY_ANNUAL),
        ("canone di 400 euro al mese", normalization.FREQUENCY_MONTHLY),
        ("canone di 400 euro mensili", normalization.FREQUENCY_MONTHLY),
        ("rent of 400 per month", normalization.FREQUENCY_MONTHLY),
        ("rent of 400 monthly", normalization.FREQUENCY_MONTHLY),
    ],
)
def test_frequency_cues_by_language(text, expected):
    amount = "400" if "400" in text else "4800"
    assert normalization.read_frequency(text, amount)[0] == expected


def test_annualita_is_not_annual_end_to_end(monkeypatch):
    """The full path, not just the cue reader: an instalment question about a
    400 euro rent must ask, not silently treat 400 as annual."""
    _extraction(monkeypatch, {"annual_rent": "400", "years": "4",
                              "first_registration": True})
    calls = _mock_http(monkeypatch)

    update = calculation_node(
        {"query": "imposta di registro per annualita successiva, canone 400 euro",
         "session_language": "it", "calculation_match": _lease_match()}
    )

    assert calls == []
    assert "mensile" in update["answer"].lower()


# === 7. No internal field names in user-facing text ======================

@pytest.mark.parametrize("lang,label", [("it", "canone"), ("es", "alquiler"),
                                        ("en", "rent")])
def test_frequency_question_uses_a_localized_label(monkeypatch, lang, label):
    _extraction(monkeypatch, {"annual_rent": "400"})
    _mock_http(monkeypatch)

    update = calculation_node(
        {"query": "400 euro", "raw_query": "400 euro", "session_language": lang,
         "pending_calculation": _pending_missing_rent()}
    )

    assert "annual_rent" not in update["answer"]
    assert label in update["answer"].lower()


def test_currency_question_uses_a_localized_label(monkeypatch):
    _extraction(monkeypatch, {"annual_rent": "400"})
    _mock_http(monkeypatch)

    update = calculation_node(
        {"query": "400 USD al mese", "raw_query": "400 USD al mese",
         "session_language": "it", "pending_calculation": _pending_missing_rent()}
    )

    assert "annual_rent" not in update["answer"]
    assert "canone" in update["answer"].lower()
