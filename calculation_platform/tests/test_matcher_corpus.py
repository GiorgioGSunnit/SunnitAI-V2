"""Behavioural regression corpus for the free-text calculator matcher.

Derived from a UX study of how Italian users (lawyers, accountants, and
citizens) phrase calculation requests. Rather than pinning exact scores —
which would be brittle against legitimate vocabulary tuning — these tests
assert the *routing band* under the three-band consumption policy:

  AUTO   score >= 3 on the top candidate  -> auto-route to the calculator
  OFFER  score 1-2                         -> answer normally, append an offer
  NONE   no_match / no candidates          -> ordinary pipeline, no offer

The safety-critical invariant is that pure doctrine questions ("cosa dice
l'art. ...", "come funziona ...") must never reach AUTO: a legal-assistance
bot silently hijacking a doctrine question into a calculator is the single
highest-impact routing failure. Genuine, explicit calculation requests must
reach at least OFFER so the feature is discoverable.
"""

from pathlib import Path

import pytest

from app.core.matcher import match_query
from app.core.registry import CalculatorRegistry

_FORMULA_PACKS = Path(__file__).resolve().parents[1] / "formula_packs"


def _band(response) -> str:
    if response.status == "no_match" or not response.candidates:
        return "NONE"
    return "AUTO" if response.candidates[0].score >= 3 else "OFFER"


@pytest.fixture(scope="module")
def definitions():
    return list(CalculatorRegistry(_FORMULA_PACKS).definitions())


# --- Safety invariant: doctrine questions must never auto-route ------------
# Each is a request for legal information, not a calculation. Auto-routing any
# of these into a calculator is a user-trust failure; OFFER or NONE is fine.
DOCTRINE_QUESTIONS = [
    "Cosa dice l'art. 1284 c.c. sugli interessi legali?",
    "Come funziona il ravvedimento operoso?",
    "Quali redditi concorrono alla formazione dell'IRPEF?",
    "In quanto tempo si prescrive il TFR?",
    "La prima casa e sempre esente da IMU?",
    "I contributi INPS sono deducibili?",
    "Chi e tenuto al pagamento del contributo unificato?",
    "Il giudice puo discostarsi dai parametri del DM 55?",
    "Qual e la differenza tra furto e rapina?",
    "Qual e la pena edittale dell'omicidio colposo?",
]


@pytest.mark.parametrize("query", DOCTRINE_QUESTIONS)
def test_doctrine_questions_never_auto_route(query, definitions):
    assert _band(match_query(query, definitions)) != "AUTO", (
        f"Doctrine question auto-routed to a calculator: {query!r}"
    )


# --- Explicit calculation requests must at least be offered ----------------
# (query, expected_calculator_id). These must reach AUTO or OFFER, and when
# they produce a candidate it must be the right calculator.
EXPLICIT_REQUESTS = [
    ("rata mutuo 200k 25 anni 3,2%", "business.loan_payment"),
    ("interessi legali 10.000 dal 2021", "legal_it.legal_interest"),
    ("IMU seconda casa Roma", "legal_it.imu"),
    ("Puoi calcolare il ravvedimento operoso per un F24 scaduto il 16 giugno?",
     "legal_it.ravvedimento_operoso"),
    ("Calcola gli interessi legali su 8.500 euro dal 3 marzo 2022.",
     "legal_it.legal_interest"),
    ("A quanto ammontano i contributi INPS per la Gestione Separata?",
     "legal_it.inps_contributions"),
    ("Quanto devo pagare di contributo unificato per una causa da 35.000 euro?",
     "legal_it.contributo_unificato_civile"),
    ("Rivaluta 750 euro da gennaio 2020 a oggi usando l'indice ISTAT.",
     "legal_it.rivalutazione_istat"),
    ("Liquidazione compensi ex DM 55/2014, scaglione 52.001-260.000.",
     "legal_it.compensi_dm55"),
    ("Imposta di registro per annualita successiva di locazione commerciale.",
     "legal_it.registration_tax_leases"),
]


@pytest.mark.parametrize("query,expected", EXPLICIT_REQUESTS)
def test_explicit_requests_are_at_least_offered(query, expected, definitions):
    response = match_query(query, definitions)
    assert _band(response) in ("AUTO", "OFFER"), (
        f"Explicit calculation request was dropped entirely: {query!r}"
    )
    assert response.candidates[0].calculator_id == expected, (
        f"{query!r} routed to {response.candidates[0].calculator_id}, "
        f"expected {expected}"
    )


# --- Regression guards for the specific vocabulary-leak bugs fixed ----------
def test_unit_tokens_do_not_leak_across_calculators(definitions):
    """A 25-year mortgage query must not pick up the homicide-penalty
    calculator via the shared bare unit 'anni' (21-24 anni penalty range)."""
    response = match_query("rata mutuo 200k 25 anni 3,2%", definitions)
    ids = {c.calculator_id for c in response.candidates}
    assert "legal_it.omicidio_pena_draft" not in ids


def test_elided_article_does_not_become_vocabulary(definitions):
    """'dell'IRPEF' must tokenize to 'irpef', not leak the fragment 'dell'
    as vocabulary shared with other calculators (e.g. IMU)."""
    response = match_query(
        "Quali redditi concorrono alla formazione dell'IRPEF?", definitions
    )
    for candidate in response.candidates:
        assert "dell" not in candidate.matched_terms
