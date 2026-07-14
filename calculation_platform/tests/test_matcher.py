from app.core.matcher import match_query
from app.main import engine


def _definitions():
    return engine.registry.definitions()


def test_italian_colloquial_tax_query_matches_irpef():
    response = match_query("quanto pago di tasse su un reddito di 42000 euro", _definitions())
    assert response.status in ("matched", "ambiguous")
    assert response.candidates[0].calculator_id == "legal_it.irpef"


def test_short_keyword_query_matches_loan_payment():
    response = match_query("rata mutuo 150000 euro", _definitions())
    assert response.candidates[0].calculator_id == "business.loan_payment"


def test_accented_and_case_insensitive_matching():
    response = match_query("CALCOLO IVA su una fattura però con sconto", _definitions())
    assert response.candidates[0].calculator_id == "business.invoice_total"


def test_english_query_matches_registration_tax():
    response = match_query("registration tax on a rental contract", _definitions())
    assert response.candidates[0].calculator_id == "legal_it.registration_tax_leases"


def test_gibberish_returns_no_match():
    response = match_query("xylophone zebra quantum", _definitions())
    assert response.status == "no_match"
    assert response.candidates == []


def test_matched_terms_are_reported_for_explainability():
    response = match_query("calcolo irpef", _definitions())
    top = response.candidates[0]
    assert top.calculator_id == "legal_it.irpef"
    assert top.matched_terms  # the caller can see WHY it matched


def test_candidates_report_required_inputs_for_clarification():
    response = match_query("interessi legali su un capitale", _definitions())
    top = response.candidates[0]
    assert top.calculator_id == "legal_it.legal_interest"
    required_names = {i["name"] for i in top.required_inputs}
    assert "capital" in required_names
    assert top.requires_period is True


def test_irpef_candidate_reports_tax_year_support():
    response = match_query("calcolo irpef", _definitions())
    assert response.candidates[0].supports_tax_year is True


def test_phrase_hit_outscores_single_token_overlap():
    # "rata mutuo" is a whole keyword phrase for loan_payment; a query
    # containing it must rank loan_payment above calculators that only
    # share an incidental token.
    response = match_query("rata mutuo", _definitions())
    assert response.candidates[0].calculator_id == "business.loan_payment"
    if len(response.candidates) > 1:
        assert response.candidates[0].score > response.candidates[1].score
