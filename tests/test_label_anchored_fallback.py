"""Phase 2B: the offline fallback binds numbers by LABEL, not by position.

When LLM extraction is unavailable, production paired numeric fields with
numbers using `zip(number_specs, number_tokens)` — by ORDER. Order is not
evidence. "ho lavorato 11 anni e 7 mesi" produced an indemnity of 11 x 7 = 77,
and "cosa dice l'articolo 40 del codice?" produced a 40-year-old driver: a
question silently became an answer.

A number is now bound only when it is anchored to a DISTINCTIVE cue for that
field — a token from its name or description, or an explicit `field: value`
assignment. Bare units are never distinctive: `mesi_preavviso` must be cued by
"preavviso", because "mesi" appears in every sentence about time.

One narrow exception survives, because it is the shape of a real answer: a lone
number for a lone remaining field, and only when the message is either a compact
value reply ("42000 euro") or an explicit calculation request. Prose and legal
questions never qualify — that is what keeps case 3 above from returning.

Every test here forces the LLM tier to be unavailable, which is the production
path whenever credentials are missing or the model call fails.
"""

import os

import pytest
import requests

os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test-password")

import src.rag.calculation as calculation
from src.rag.calculation import calculation_node


@pytest.fixture(autouse=True)
def offline_llm(monkeypatch):
    """No LLM: every test here exercises the deterministic fallback tier."""
    monkeypatch.setattr(calculation, "_extract_values_llm", lambda *a, **k: None)


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


# --- Spec fixtures mirroring the real packs --------------------------------

def _notice_specs():
    return [
        {"name": "retribuzione_mensile_globale", "type": "decimal", "required": True,
         "unit": "EUR",
         "description": "Retribuzione mensile globale di fatto (inclusi elementi "
                        "continuativi ex art. 2121 c.c.)"},
        {"name": "mesi_preavviso", "type": "decimal", "required": True,
         "unit": "months",
         "description": "Mesi di preavviso non lavorati secondo il CCNL applicabile"},
    ]


def _irpef_specs():
    return [
        {"name": "taxable_income", "type": "decimal", "required": True, "unit": "EUR",
         "description": "Reddito imponibile IRPEF"},
    ]


def _age_specs():
    return [
        {"name": "eta_conducente", "type": "integer", "required": True,
         "description": "Eta del conducente"},
    ]


def _extract(query, specs, **kwargs):
    return calculation._extract_values(query, specs, **kwargs)


# === 1. Label-anchored binding ============================================

def test_two_named_fields_both_bind_to_their_own_labels():
    values = _extract(
        "retribuzione mensile 2500 euro, preavviso 3 mesi", _notice_specs()
    )

    assert values["retribuzione_mensile_globale"] == "2500"
    assert values["mesi_preavviso"] == "3"


def test_unlabeled_numbers_stay_unbound_while_a_labeled_one_binds():
    """The 11 x 7 = 77 case.

    "11 anni e 7 mesi" names a period of service, not a notice period. Only the
    salary is labelled, so only the salary binds and the notice period stays a
    question.
    """
    values = _extract(
        "ho lavorato 11 anni e 7 mesi, retribuzione mensile 2500 euro",
        _notice_specs(),
    )

    assert values["retribuzione_mensile_globale"] == "2500"
    assert "mesi_preavviso" not in values


def test_a_bare_unit_is_not_a_distinctive_cue():
    """`mesi_preavviso` must be cued by "preavviso", never by "mesi" alone."""
    values = _extract("ho lavorato 7 mesi", _notice_specs())

    assert values == {}


@pytest.mark.parametrize(
    "generic",
    ["euro", "importo", "valore", "mese", "mesi", "anno", "anni", "giorni",
     "data", "tasso", "aliquota", "percentuale"],
)
def test_generic_words_are_never_distinctive_cues(generic):
    from src.rag import label_anchoring

    assert generic in label_anchoring.GENERIC_CUE_TOKENS


def test_explicit_assignment_binds_regardless_of_distance():
    values = _extract(
        "dati: mesi_preavviso = 3, retribuzione_mensile_globale = 2500",
        _notice_specs(),
    )

    assert values["mesi_preavviso"] == "3"
    assert values["retribuzione_mensile_globale"] == "2500"


def test_numbers_are_never_paired_with_fields_by_order():
    """Reversing the labels must reverse the binding, not preserve position."""
    values = _extract("preavviso 3 mesi, retribuzione mensile 2500 euro",
                      _notice_specs())
    reversed_values = _extract("retribuzione mensile 2500 euro, preavviso 3 mesi",
                               _notice_specs())

    assert values == reversed_values
    assert values["mesi_preavviso"] == "3"
    assert values["retribuzione_mensile_globale"] == "2500"


# === 2. A legal question is not a set of inputs ===========================

def test_a_legal_question_does_not_become_an_input():
    """The article number must not become the driver's age.

    Both narrow-fallback conditions hold here — one field, one number — so the
    message shape is the only thing standing between a question and an answer.
    """
    assert _extract("cosa dice l'articolo 40 del codice?", _age_specs()) == {}


@pytest.mark.parametrize(
    "question",
    [
        "cosa dice l'articolo 40 del codice?",
        "mi spieghi l'art. 2043 c.c.?",
        "quali sono i requisiti previsti dalla legge 300 del 1970?",
        "il mio assistito ha 3 figli, cosa prevede la successione?",
    ],
)
def test_prose_and_legal_questions_never_trigger_the_narrow_fallback(question):
    assert _extract(question, _irpef_specs()) == {}
    assert _extract(question, _age_specs()) == {}


def test_two_unlabeled_numbers_for_two_fields_stay_unbound():
    assert _extract("11 e 7", _notice_specs()) == {}


def test_two_unlabeled_numbers_do_not_fill_one_remaining_field():
    """Ambiguity must not be resolved by picking one."""
    assert _extract("ho lavorato 11 anni e 7 mesi", _irpef_specs()) == {}


# === 3. The narrow single-value fallback ==================================

@pytest.mark.parametrize(
    "reply,expected",
    [
        ("42000 euro", "42000"),
        ("42.000,00 euro", "42000.00"),
        ("42000", "42000"),
        ("sono 42000 euro", "42000"),
    ],
)
def test_a_compact_value_reply_fills_the_sole_missing_field(reply, expected):
    values = _extract(reply, _irpef_specs())

    assert values["taxable_income"] == expected


def test_an_explicit_calculation_request_still_binds_its_single_value():
    values = _extract("Calcola l'IRPEF su 42000 euro", _irpef_specs())

    assert values["taxable_income"] == "42000"


@pytest.mark.parametrize(
    "query",
    ["Calcola l'IRPEF su 42000 euro", "quanto pago di IRPEF su 42000 euro",
     "calculate the tax on 42000 euro"],
)
def test_calculation_requests_are_recognized_in_each_language(query):
    assert _extract(query, _irpef_specs())["taxable_income"] == "42000"


def test_known_gap_single_dot_thousands_are_not_yet_locale_parsed():
    """Documents a PRE-EXISTING gap in _normalize_number, not new behaviour.

    A lone dot with three trailing digits and no comma is a thousands separator
    in Italian, but _normalize_number only strips dots when a comma is present
    or when there are several dots. So Italian "42.000" reads as 42.0 in the
    offline tier — a 1000x understatement. Pinned here so the gap is visible and
    cannot regress further; fixing it needs the session language threaded into
    _normalize_number, which is deliberately out of this phase's scope.
    """
    spec = {"name": "taxable_income", "type": "decimal"}

    assert calculation._normalize_number("42.000", spec) == "42.000"
    # The forms that already work, for contrast.
    assert calculation._normalize_number("42.000,00", spec) == "42000.00"
    assert calculation._normalize_number("42000", spec) == "42000"
    assert calculation._normalize_number("1.200.300", spec) == "1200300"


def test_the_narrow_fallback_needs_exactly_one_candidate_field():
    """Two missing numeric fields and one number is still ambiguous."""
    assert _extract("42000 euro", _notice_specs()) == {}


# === 4. Dates and tax years are not ordinary numbers ======================

def test_dates_become_a_period_and_are_not_reused_as_numbers():
    specs = [
        {"name": "capital", "type": "decimal", "required": True},
        {"name": "period", "type": "period", "required": True},
    ]

    values = _extract("8500 euro dal 01/01/2025 al 31/12/2025", specs)

    assert values["capital"] == "8500"
    assert values["period"] == {"start_date": "2025-01-01", "end_date": "2025-12-31"}


def test_a_tax_year_is_not_reused_as_an_ordinary_input():
    values = _extract(
        "Calcola l'IRPEF 2026 su 42.000,00 euro", _irpef_specs(),
        supports_tax_year=True,
    )

    assert values["tax_year"] == 2026
    assert values["taxable_income"] == "42000.00"


def test_a_bare_year_does_not_fill_a_numeric_field_without_tax_year_support():
    """Without tax-year support a stray 2026 is just an unlabeled number."""
    assert _extract("nel 2026 quanto pago di IRPEF?", _irpef_specs()) == {}


# === 5. Percentages keep their existing behaviour =========================

def test_percentage_normalization_is_unchanged():
    specs = [
        {"name": "vat_rate", "type": "decimal", "required": True, "unit": "rate",
         "description": "Aliquota IVA come frazione (es. 0,22 per 22%)"},
    ]

    values = _extract("vat_rate = 22%", specs)

    assert values["vat_rate"] == "0.22"


def test_percentage_helper_is_untouched():
    spec = {"name": "vat_rate", "type": "decimal", "unit": "rate"}
    assert calculation._normalize_number("22%", spec) == "0.22"
    assert calculation._normalize_number("1.200,50", {"name": "x",
                                                      "type": "decimal"}) == "1200.50"


# === 6. The fallback still feeds the frequency normalizer =================

def test_offline_monthly_rent_still_becomes_an_annual_one(monkeypatch):
    """The two deterministic layers have to compose.

    Label anchoring binds 400 to annual_rent via "canone"; the frequency layer
    then converts it. Without the anchor the number would not arrive at all, and
    without the conversion it would arrive twelve times too small.
    """
    calls = _mock_http(
        monkeypatch,
        {"calculator_id": "legal_it.registration_tax_leases", "status": "success",
         "result": {"tax_due": "384.00"}},
    )

    update = calculation_node(
        {
            "query": "Calcola l'imposta di registro, canone 400 euro al mese, 4 anni",
            "session_language": "it",
            "calculation_match": {
                "calculator_id": "legal_it.registration_tax_leases",
                "required_inputs": [
                    {"name": "annual_rent", "type": "decimal", "required": True,
                     "unit": "EUR",
                     "description": "Canone di locazione (affitto) annuo pattuito"},
                ],
                "optional_inputs": [],
            },
        }
    )

    assert calls[0]["json"]["inputs"]["annual_rent"] == "4800"
    assert "4.800" in update["answer"]


# === 7. Pending-calculation topic changes still escape ====================

def test_a_topic_change_during_a_pending_calculation_escapes_to_rag(monkeypatch):
    calls = _mock_http(monkeypatch)

    update = calculation_node(
        {
            "query": "e l'articolo 2043 del codice civile cosa prevede?",
            "raw_query": "e l'articolo 2043 del codice civile cosa prevede?",
            "session_language": "it",
            "pending_calculation": {
                "calculator_id": "legal_it.legal_interest",
                "inputs_so_far": {"capital": "8500"},
                "round": 1,
                "missing_inputs": [
                    {"name": "period", "type": "period", "required": True}
                ],
            },
        }
    )

    assert calls == [], "an incidental number must not be consumed as an input"
    assert update["calc_route"] == "normal"
    assert update["pending_calculation"] is None


def test_a_compact_follow_up_still_answers_a_pending_question(monkeypatch):
    calls = _mock_http(
        monkeypatch,
        {"calculator_id": "legal_it.irpef", "status": "success",
         "result": {"gross_tax": "11060.00"}},
    )

    calculation_node(
        {
            "query": "42000 euro",
            "raw_query": "42000 euro",
            "session_language": "it",
            "pending_calculation": {
                "calculator_id": "legal_it.irpef",
                "inputs_so_far": {},
                "round": 1,
                "missing_inputs": _irpef_specs(),
            },
        }
    )

    assert calls[0]["json"]["inputs"]["taxable_income"] == "42000"


# === 8. Production code must not import the simulation ===================

def test_production_does_not_import_the_simulation_package():
    """The simulation is a design reference, never a production dependency.

    Matches import statements specifically: a docstring may name scripted_llm.py
    as the origin of the doctrine, which is documentation, not coupling.
    """
    import re as _re
    from pathlib import Path

    pattern = _re.compile(
        r"^\s*(?:from|import)\s+\S*(?:simulation|scripted_llm)", _re.MULTILINE
    )
    root = Path(calculation.__file__).resolve().parents[2] / "src"
    offenders = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*.py")
        if pattern.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []
