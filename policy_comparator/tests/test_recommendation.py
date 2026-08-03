"""Eligibility, ranking and deduplication."""

from __future__ import annotations

from decimal import Decimal

from policy_comparator.schemas.profile import CoveragePreferenceData
from policy_comparator.schemas.quotes import CoverageData, NormalizedQuoteData
from policy_comparator.services import deduplication, recommendation
from policy_comparator.services.recommendation import Reason


def quote(
    quote_id: str,
    *,
    premium: str | None = "400.00",
    insurer: str = "Zurich",
    provider_id: str = "zurich",
    channel: str = "direct",
    deductible: str | None = "300",
    people: str | None = "6450000",
    property_limit: str | None = "1300000",
    formula: str | None = "free",
    black_box: bool | None = False,
    repair_network: bool | None = False,
    covers: tuple[tuple[str, bool], ...] = (),
    product: str = "Prodotto",
    reference: str | None = None,
) -> tuple[str, NormalizedQuoteData]:
    return quote_id, NormalizedQuoteData(
        provider_id=provider_id,
        insurer_name=insurer,
        source_channel=channel,
        product_name=product,
        provider_quote_reference=reference or f"REF-{quote_id}",
        annual_total_premium=premium,
        deductible=deductible,
        liability_limit_people=people,
        liability_limit_property=property_limit,
        driving_formula=formula,
        requires_black_box=black_box,
        requires_approved_repair_network=repair_network,
        coverages=[
            CoverageData(code=code, label=code, included=included) for code, included in covers
        ],
    )


NO_REQUIREMENTS = CoveragePreferenceData()


class TestRanking:
    def test_cheapest_eligible_quote_wins(self):
        quotes = [
            quote("a", premium="520.00"),
            quote("b", premium="410.00"),
            quote("c", premium="480.00"),
        ]
        result = recommendation.recommend(quotes, NO_REQUIREMENTS)

        assert result.recommended_quote_id == "b"
        assert result.alternatives == ["c", "a"]
        assert result.explanation_code == "lowest_price_matching_requirements"

    def test_explanation_never_calls_a_quote_the_best(self):
        result = recommendation.recommend([quote("a")], NO_REQUIREMENTS)
        lowered = result.explanation.lower()
        assert "migliore" not in lowered and "best" not in lowered
        assert "più economico" in lowered

    def test_tie_on_price_breaks_on_lower_deductible(self):
        quotes = [
            quote("high", premium="400.00", deductible="500"),
            quote("low", premium="400.00", deductible="200"),
        ]
        assert recommendation.recommend(quotes, NO_REQUIREMENTS).recommended_quote_id == "low"

    def test_tie_on_price_and_deductible_breaks_on_higher_limits(self):
        quotes = [
            quote("small", premium="400.00", deductible="300", people="6450000"),
            quote("big", premium="400.00", deductible="300", people="10000000"),
        ]
        assert recommendation.recommend(quotes, NO_REQUIREMENTS).recommended_quote_id == "big"

    def test_final_tie_breaks_on_fewer_restrictions(self):
        quotes = [
            quote("restricted", premium="400.00", black_box=True),
            quote("free", premium="400.00", black_box=False),
        ]
        assert recommendation.recommend(quotes, NO_REQUIREMENTS).recommended_quote_id == "free"

    def test_ranking_is_stable_regardless_of_input_order(self):
        a = quote("a", premium="400.00", insurer="Alpha")
        b = quote("b", premium="400.00", insurer="Beta")
        forward = recommendation.recommend([a, b], NO_REQUIREMENTS).recommended_quote_id
        backward = recommendation.recommend([b, a], NO_REQUIREMENTS).recommended_quote_id
        assert forward == backward


class TestEligibility:
    def test_black_box_requirement_excludes_a_cheaper_quote(self):
        """The cheapest offer must lose when the customer refuses a telematics box."""
        quotes = [
            quote("cheap_boxed", premium="300.00", black_box=True),
            quote("compliant", premium="450.00", black_box=False),
        ]
        preferences = CoveragePreferenceData(accepts_black_box=False)
        result = recommendation.recommend(quotes, preferences)

        assert result.recommended_quote_id == "compliant"
        reasons = [r.code for e in result.evaluations if e.quote_id == "cheap_boxed" for r in e.reasons]
        assert Reason.BLACK_BOX_REFUSED in reasons

    def test_repair_network_requirement_is_enforced(self):
        quotes = [quote("a", premium="300.00", repair_network=True)]
        result = recommendation.recommend(
            quotes, CoveragePreferenceData(accepts_approved_repair_network=False)
        )
        assert result.recommended_quote_id is None

    def test_deductible_above_the_maximum_is_excluded(self):
        quotes = [
            quote("over", premium="300.00", deductible="800"),
            quote("under", premium="500.00", deductible="200"),
        ]
        preferences = CoveragePreferenceData(max_acceptable_deductible=Decimal("300"))
        result = recommendation.recommend(quotes, preferences)
        assert result.recommended_quote_id == "under"

    def test_liability_below_the_minimum_is_excluded(self):
        quotes = [quote("low", people="5000000")]
        preferences = CoveragePreferenceData(min_liability_limit_people=Decimal("6450000"))
        result = recommendation.recommend(quotes, preferences)

        assert result.recommended_quote_id is None
        assert result.evaluations[0].reasons[0].code == Reason.LIABILITY_BELOW_MINIMUM

    def test_driving_formula_mismatch_is_excluded(self):
        quotes = [quote("free", formula="free"), quote("exclusive", formula="exclusive")]
        preferences = CoveragePreferenceData(driving_formula="exclusive")
        assert recommendation.recommend(quotes, preferences).recommended_quote_id == "exclusive"

    def test_a_required_cover_must_actually_be_included(self):
        quotes = [
            quote("optional_only", premium="300.00", covers=(("furto_incendio", False),)),
            quote("included", premium="500.00", covers=(("furto_incendio", True),)),
        ]
        preferences = CoveragePreferenceData(required_optional_covers=["furto_incendio"])
        result = recommendation.recommend(quotes, preferences)

        assert result.recommended_quote_id == "included"
        reasons = [
            r.code for e in result.evaluations if e.quote_id == "optional_only" for r in e.reasons
        ]
        assert Reason.REQUIRED_COVER_NOT_INCLUDED in reasons


class TestUnknownsAreNotRecommended:
    """A gap in the data is never resolved in the quote's favour."""

    def test_missing_premium_is_never_recommended(self):
        result = recommendation.recommend([quote("a", premium=None)], NO_REQUIREMENTS)
        assert result.recommended_quote_id is None
        assert result.evaluations[0].reasons[0].code == Reason.MISSING_PREMIUM

    def test_unknown_deductible_fails_a_deductible_requirement(self):
        preferences = CoveragePreferenceData(max_acceptable_deductible=Decimal("300"))
        result = recommendation.recommend([quote("a", deductible=None)], preferences)

        assert result.recommended_quote_id is None
        assert result.evaluations[0].reasons[0].code == Reason.MISSING_COVERAGE_INFO

    def test_unknown_black_box_flag_fails_a_black_box_refusal(self):
        preferences = CoveragePreferenceData(accepts_black_box=False)
        result = recommendation.recommend([quote("a", black_box=None)], preferences)
        assert result.recommended_quote_id is None

    def test_unknown_fields_are_fine_when_nothing_requires_them(self):
        result = recommendation.recommend([quote("a", deductible=None)], NO_REQUIREMENTS)
        assert result.recommended_quote_id == "a"


class TestNoEligibleQuote:
    def test_reports_every_exclusion_rather_than_an_empty_screen(self):
        quotes = [quote("a", black_box=True), quote("b", black_box=True)]
        result = recommendation.recommend(quotes, CoveragePreferenceData(accepts_black_box=False))

        assert result.recommended_quote_id is None
        assert result.explanation_code == "no_eligible_quote"
        assert len(result.evaluations) == 2
        assert all(not e.eligible for e in result.evaluations)


class TestDeduplication:
    def test_the_same_offer_via_two_channels_collapses_to_one(self):
        direct = quote("direct", insurer="Zurich", provider_id="zurich", reference="REF-1")
        relayed = quote(
            "relayed",
            insurer="Zurich",
            provider_id="cercassicurazioni",
            channel="aggregator",
            reference="REF-1",
        )
        result = deduplication.deduplicate([direct, relayed])

        assert result.duplicate_to_primary == {"relayed": "direct"}
        assert set(result.channels_by_primary["direct"]) == {"direct", "aggregator"}

    def test_the_direct_copy_is_kept_as_primary(self):
        """The direct quote carries the link the customer will actually use."""
        relayed = quote(
            "relayed",
            provider_id="cercassicurazioni",
            channel="aggregator",
            reference="REF-1",
        )
        direct = quote("direct", provider_id="zurich", reference="REF-1")

        # Aggregator listed first, to prove ordering does not decide it.
        result = deduplication.deduplicate([relayed, direct])
        assert result.duplicate_to_primary == {"relayed": "direct"}

    def test_different_insurers_are_not_merged(self):
        quotes = [
            quote("a", insurer="Zurich", reference="REF-1"),
            quote("b", insurer="Allianz", reference="REF-1"),
        ]
        assert deduplication.deduplicate(quotes).duplicate_to_primary == {}

    def test_different_prices_from_one_insurer_are_not_merged(self):
        quotes = [
            quote("a", insurer="Zurich", premium="400.00", reference="REF-1"),
            quote("b", insurer="Zurich", premium="450.00", reference="REF-2"),
        ]
        assert deduplication.deduplicate(quotes).duplicate_to_primary == {}

    def test_a_duplicate_is_never_recommended(self):
        direct = quote("direct", premium="400.00", reference="REF-1")
        relayed = quote(
            "relayed",
            premium="400.00",
            provider_id="cercassicurazioni",
            channel="aggregator",
            reference="REF-1",
        )
        dedupe = deduplication.deduplicate([direct, relayed])
        result = recommendation.recommend(
            [direct, relayed], NO_REQUIREMENTS, duplicate_ids=dedupe.duplicate_ids
        )

        assert result.recommended_quote_id == "direct"
        assert "relayed" not in result.eligible_ids
