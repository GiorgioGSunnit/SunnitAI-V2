"""Eligibility and ranking.

The rule is deliberately narrow: find the cheapest quote that actually
satisfies what the customer said they need. There is no composite score, no
weighting of "value" against price, and nothing is ever labelled the best
policy — only the lowest-priced one that meets the stated requirements.

Eligibility is decided before price is looked at. A quote that is missing the
information needed to prove it meets a requirement is *not* given the benefit
of the doubt: it is excluded with a reason, because recommending a policy whose
deductible we do not know would be worse than showing the customer a gap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Iterable, Sequence

from ..schemas.profile import CoveragePreferenceData
from ..schemas.quotes import NormalizedQuoteData


class Reason:
    """Machine-readable ineligibility codes."""

    MISSING_PREMIUM = "missing_premium"
    MISSING_COVERAGE_INFO = "missing_coverage_info"
    LIABILITY_BELOW_MINIMUM = "liability_below_minimum"
    DEDUCTIBLE_ABOVE_MAXIMUM = "deductible_above_maximum"
    DRIVING_FORMULA_MISMATCH = "driving_formula_mismatch"
    REQUIRED_COVER_NOT_INCLUDED = "required_cover_not_included"
    BLACK_BOX_REFUSED = "black_box_refused"
    REPAIR_NETWORK_REFUSED = "repair_network_refused"
    DUPLICATE = "duplicate"


_REASON_TEXT_IT = {
    Reason.MISSING_PREMIUM: "Il preventivo non riporta un premio annuo.",
    Reason.MISSING_COVERAGE_INFO: "Mancano informazioni essenziali sulla copertura.",
    Reason.LIABILITY_BELOW_MINIMUM: "Il massimale è inferiore al minimo richiesto.",
    Reason.DEDUCTIBLE_ABOVE_MAXIMUM: "La franchigia supera il massimo accettato.",
    Reason.DRIVING_FORMULA_MISMATCH: "La formula di guida non corrisponde a quella richiesta.",
    Reason.REQUIRED_COVER_NOT_INCLUDED: "Una garanzia richiesta non è inclusa nel premio.",
    Reason.BLACK_BOX_REFUSED: "Il preventivo richiede la scatola nera, non accettata dal cliente.",
    Reason.REPAIR_NETWORK_REFUSED: (
        "Il preventivo obbliga all'uso delle carrozzerie convenzionate, non accettate dal cliente."
    ),
    Reason.DUPLICATE: "Offerta già presente tramite un altro canale.",
}


@dataclass(frozen=True)
class Ineligibility:
    code: str
    message: str
    #: The specific requirement that failed, when there is one.
    detail: str | None = None


@dataclass
class Evaluation:
    quote_id: str
    quote: NormalizedQuoteData
    eligible: bool
    reasons: list[Ineligibility] = field(default_factory=list)


@dataclass
class Recommendation:
    recommended_quote_id: str | None
    explanation: str
    explanation_code: str
    alternatives: list[str] = field(default_factory=list)
    evaluations: list[Evaluation] = field(default_factory=list)

    @property
    def eligible_ids(self) -> list[str]:
        return [e.quote_id for e in self.evaluations if e.eligible]


def _is_included(quote: NormalizedQuoteData, code: str) -> bool:
    return any(c.code == code and c.included for c in quote.coverages)


def evaluate(
    quote_id: str,
    quote: NormalizedQuoteData,
    preferences: CoveragePreferenceData,
    *,
    is_duplicate: bool = False,
) -> Evaluation:
    """Decide whether one quote satisfies every mandatory requirement."""
    reasons: list[Ineligibility] = []

    def fail(code: str, detail: str | None = None) -> None:
        reasons.append(Ineligibility(code, _REASON_TEXT_IT[code], detail))

    if is_duplicate:
        fail(Reason.DUPLICATE)

    if quote.annual_total_premium is None:
        fail(Reason.MISSING_PREMIUM)

    # -- liability limits ----------------------------------------------------
    if preferences.min_liability_limit_people is not None:
        if quote.liability_limit_people is None:
            fail(Reason.MISSING_COVERAGE_INFO, "massimale danni a persone non indicato")
        elif quote.liability_limit_people < preferences.min_liability_limit_people:
            fail(
                Reason.LIABILITY_BELOW_MINIMUM,
                f"danni a persone {quote.liability_limit_people} < "
                f"{preferences.min_liability_limit_people}",
            )

    if preferences.min_liability_limit_property is not None:
        if quote.liability_limit_property is None:
            fail(Reason.MISSING_COVERAGE_INFO, "massimale danni a cose non indicato")
        elif quote.liability_limit_property < preferences.min_liability_limit_property:
            fail(
                Reason.LIABILITY_BELOW_MINIMUM,
                f"danni a cose {quote.liability_limit_property} < "
                f"{preferences.min_liability_limit_property}",
            )

    # -- deductible ----------------------------------------------------------
    if preferences.max_acceptable_deductible is not None:
        if quote.deductible is None:
            fail(Reason.MISSING_COVERAGE_INFO, "franchigia non indicata")
        elif quote.deductible > preferences.max_acceptable_deductible:
            fail(
                Reason.DEDUCTIBLE_ABOVE_MAXIMUM,
                f"franchigia {quote.deductible} > {preferences.max_acceptable_deductible}",
            )

    # -- driving formula -----------------------------------------------------
    if preferences.driving_formula:
        if not quote.driving_formula:
            fail(Reason.MISSING_COVERAGE_INFO, "formula di guida non indicata")
        elif quote.driving_formula != preferences.driving_formula:
            fail(
                Reason.DRIVING_FORMULA_MISMATCH,
                f"{quote.driving_formula} ≠ {preferences.driving_formula}",
            )

    # -- required guarantees --------------------------------------------------
    for code in preferences.required_optional_covers or []:
        if not _is_included(quote, code):
            fail(Reason.REQUIRED_COVER_NOT_INCLUDED, code)

    # -- restrictions the customer refused ------------------------------------
    if preferences.accepts_black_box is False:
        if quote.requires_black_box is None:
            fail(Reason.MISSING_COVERAGE_INFO, "obbligo di scatola nera non indicato")
        elif quote.requires_black_box:
            fail(Reason.BLACK_BOX_REFUSED)

    if preferences.accepts_approved_repair_network is False:
        if quote.requires_approved_repair_network is None:
            fail(Reason.MISSING_COVERAGE_INFO, "obbligo di carrozzerie convenzionate non indicato")
        elif quote.requires_approved_repair_network:
            fail(Reason.REPAIR_NETWORK_REFUSED)

    return Evaluation(quote_id=quote_id, quote=quote, eligible=not reasons, reasons=reasons)


def _format_money(amount: Decimal, currency: str) -> str:
    """Italian formatting — 1.234,56 € — so the explanation matches the UI.

    Done here rather than in the frontend because the explanation is a single
    server-rendered sentence, and a raw ``296.14 EUR`` in the middle of an
    Italian paragraph reads as a bug.
    """
    quantized = amount.quantize(Decimal("0.01"))
    whole, _, cents = f"{quantized:.2f}".partition(".")
    grouped = f"{int(whole):,}".replace(",", ".")
    symbol = "€" if currency == "EUR" else currency
    return f"{grouped},{cents} {symbol}"


def _restriction_count(quote: NormalizedQuoteData) -> int:
    return int(bool(quote.requires_black_box)) + int(bool(quote.requires_approved_repair_network))


def _sort_key(evaluation: Evaluation) -> tuple:
    """Cheapest first, then the tie-breakers, in the order the business wants.

    Liability limits are negated so that *higher* limits sort earlier while the
    rest of the key still sorts ascending.
    """
    q = evaluation.quote
    premium = q.annual_total_premium if q.annual_total_premium is not None else Decimal("Infinity")
    deductible = q.deductible if q.deductible is not None else Decimal("Infinity")
    limit_people = q.liability_limit_people or Decimal("0")
    limit_property = q.liability_limit_property or Decimal("0")
    return (
        premium,
        deductible,
        -limit_people,
        -limit_property,
        _restriction_count(q),
        # Final tie-break so ordering is stable and reproducible.
        q.insurer_name.lower(),
    )


def recommend(
    quotes: Sequence[tuple[str, NormalizedQuoteData]],
    preferences: CoveragePreferenceData,
    *,
    duplicate_ids: Iterable[str] = (),
) -> Recommendation:
    """Rank quotes and pick the cheapest eligible one."""
    duplicates = set(duplicate_ids)
    evaluations = [
        evaluate(qid, quote, preferences, is_duplicate=qid in duplicates)
        for qid, quote in quotes
    ]

    eligible = sorted((e for e in evaluations if e.eligible), key=_sort_key)

    if not eligible:
        return Recommendation(
            recommended_quote_id=None,
            explanation=(
                "Nessun preventivo soddisfa tutti i requisiti selezionati. "
                "Rivedi i requisiti del cliente oppure riprova i provider non disponibili."
            ),
            explanation_code="no_eligible_quote",
            evaluations=evaluations,
        )

    best = eligible[0]
    others = [e.quote_id for e in eligible[1:]]

    premium = best.quote.annual_total_premium
    premium_text = f" a {_format_money(premium, best.quote.currency)}/anno" if premium else ""
    explanation = (
        "Preventivo più economico tra quelli conformi ai requisiti selezionati: "
        f"{best.quote.insurer_name}{premium_text}"
    )
    if others:
        explanation += f" — confrontato con altri {len(others)} preventivi idonei."
    else:
        explanation += "."

    return Recommendation(
        recommended_quote_id=best.quote_id,
        explanation=explanation,
        explanation_code="lowest_price_matching_requirements",
        alternatives=others,
        evaluations=evaluations,
    )
