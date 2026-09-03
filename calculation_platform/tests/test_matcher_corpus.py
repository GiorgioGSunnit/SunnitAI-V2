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
    ("Confronta queste due offerte gas e luce.", "business.confronto_gas_luce"),
    ("Mi confronti queste offerte di luce e gas?", "business.confronto_gas_luce"),
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


# --- Genuine comparison requests -------------------------------------------
# Phrasings a user actually types when they hold two concrete offers and want
# them ranked. These must AUTO-route (a comparison the platform can run and
# then declines to offer is a silent feature outage) AND must land on the
# right domain pack. This is a hand-written behavioural corpus, not a sample:
# it says nothing about production accuracy rates.
COMPARISON_REQUESTS = [
    ("Confronta queste due offerte gas e luce.", "business.confronto_gas_luce"),
    ("Mi confronti queste offerte di luce e gas?", "business.confronto_gas_luce"),
    ("Quale fornitore di luce e gas conviene tra questi?",
     "business.confronto_gas_luce"),
    ("Confronto fornitori energia: chi costa meno?", "business.confronto_gas_luce"),
    ("Confronta queste offerte energia e calcola il costo annuo bolletta.",
     "business.confronto_gas_luce"),
    ("Comparatore bollette: due offerte a confronto.", "business.confronto_gas_luce"),
    ("Qual e l'offerta gas e luce migliore tra queste?",
     "business.confronto_gas_luce"),
    ("Energy offer comparison for these two suppliers.",
     "business.confronto_gas_luce"),
    ("Ho due offerte per la fornitura di luce e gas, quale scelgo?",
     "business.confronto_gas_luce"),
]


@pytest.mark.parametrize("query,expected", COMPARISON_REQUESTS)
def test_genuine_comparison_requests_auto_route_to_the_right_pack(query, expected, definitions):
    response = match_query(query, definitions)
    assert _band(response) == "AUTO", (
        f"Genuine comparison request did not auto-route: {query!r} "
        f"(status={response.status}, "
        f"top={response.candidates[0].calculator_id if response.candidates else None})"
    )
    assert response.candidates[0].calculator_id == expected, (
        f"{query!r} routed to {response.candidates[0].calculator_id}, expected {expected}"
    )


# --- Comparison lookalikes: the word is there, the request is not ----------
# "confronto"/"differenza" carry no comparison intent in these sentences. The
# comparator packs are the newest routing vocabulary and the easiest to
# over-trigger, so they get the same never-AUTO guarantee as doctrine.
COMPARISON_LOOKALIKES = [
    "Resto disponibile ad ogni confronto tra le parti.",
    "Qual e la differenza tra polizza vita e polizza infortuni?",
    "Come funziona il confronto tra offerte nel mercato tutelato?",
    "Quanto costa in media un'assicurazione auto?",
    "Quanto costa in media la bolletta della luce?",
    "Cosa copre una polizza kasko?",
    "Come si disdice un contratto di fornitura di energia elettrica?",
    "Quali sono gli obblighi informativi precontrattuali dell'assicuratore?",
    "Il confronto tra le due sentenze mostra un orientamento diverso.",
    "Che differenza c'e tra mercato libero e maggior tutela?",
    "Come funziona il diritto di recesso in una polizza assicurativa?",
    # Document comparison is a different feature; a calculator claiming it
    # would answer a question about two uploaded files with a price ranking.
    "Confronta questi due documenti che ti ho caricato.",
    "Confronta questi contratti di locazione che ti ho mandato.",
]


@pytest.mark.parametrize("query", COMPARISON_LOOKALIKES)
def test_comparison_lookalikes_never_auto_route(query, definitions):
    assert _band(match_query(query, definitions)) != "AUTO", (
        f"Sentence merely containing comparison vocabulary auto-routed: {query!r}"
    )


INSURANCE_COMPARISON_REQUESTS = [
    "Confronta queste polizze auto: Alfa 420 euro, Beta 510 euro.",
    "Qual e la polizza migliore tra queste offerte assicurative?",
    "Confronto tra le polizze rc auto che ho ricevuto.",
    "Insurance comparison between these two policies.",
]


@pytest.mark.parametrize("query", INSURANCE_COMPARISON_REQUESTS)
def test_insurance_comparison_is_not_exposed_by_the_calculator(query, definitions):
    response = match_query(query, definitions)
    assert "business.confronto_polizze" not in {
        candidate.calculator_id for candidate in response.candidates
    }


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


# --- Document-drafting requests: a limitation, not an invariant -------------
# A drafting request and a calculation request about the same subject share
# their whole vocabulary. "Redigi un contratto di locazione con un canone di
# 400 euro al mese" is a lease to be WRITTEN; the matcher sees the phrase
# "contratto di locazione" (+3) and reports the registration-tax calculator at
# AUTO strength, exactly as it does for "calcola l'imposta di registro ...".
# The verb carries the entire distinction and the matcher cannot see it: it
# scores topic overlap, and no threshold on topic overlap can separate the two.
#
# So these are NOT in the never-AUTO invariant above — they would fail. This
# records the blind spot and names the compensating controls, both outside the
# matcher:
#   * /api/chat classifies generation intent before the RAG graph runs, so a
#     drafting request never reaches the calculation gate on the chat path;
#   * the sources lookup inside the generation branch runs with
#     skip_calculation=True (tests/test_document_intent_routing.py).
#
# When a real intent classifier lands, move these into DOCTRINE_QUESTIONS and
# delete this test. Until then, a change here is a signal, not a pass/fail:
# if one of these stops matching, the matcher got better and the expectation
# below is what needs updating.
DOCUMENT_REQUESTS = [
    ("Redigi un contratto di locazione con un canone di 400 euro al mese.",
     "legal_it.registration_tax_leases"),
    ("Preparami una fattura con importo netto di 1000 euro e IVA al 22%.",
     "business.invoice_total"),
]


@pytest.mark.parametrize("query,shadowing_calculator", DOCUMENT_REQUESTS)
def test_document_requests_are_still_lexically_indistinguishable(
    query, shadowing_calculator, definitions
):
    response = match_query(query, definitions)
    assert _band(response) == "AUTO" and (
        response.candidates[0].calculator_id == shadowing_calculator
    ), (
        f"{query!r} no longer shadows {shadowing_calculator}. If the matcher "
        "now distinguishes drafting from calculating, move this case into "
        "DOCTRINE_QUESTIONS and drop this test."
    )
