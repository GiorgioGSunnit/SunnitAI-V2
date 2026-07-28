"""Comparator strategy: ranks N candidate offers with normalized 0-100
component scores. End-to-end coverage runs through the real engine and the
business.confronto_* packs (whose worked examples are also verified by
test_examples.py); unit tests pin the strategy mechanics (labels, clamps,
per-unit rules, tie-breaking) and the load-time definition validation.
"""

from decimal import Decimal

import pytest

from app.core.definition_validator import validate_definition
from app.core.errors import DefinitionValidationError
from app.main import engine
from app.resolvers.parameter_store import ParameterStore
from app.schemas.calculation_request import CalculationRequest
from app.schemas.calculator_definition import CalculatorDefinition, InputSpec
from app.strategies.comparator import ComparatorStrategy

from pathlib import Path

PARAMETERS_DIR = Path(__file__).resolve().parent.parent / "parameters"


def _strategy():
    return ComparatorStrategy(ParameterStore(PARAMETERS_DIR))


def _definition(**overrides):
    """A minimal two-component comparator over price + a boolean flag."""
    base = dict(
        id="test.comparator", name="comparator", category="test", strategy="comparator",
        inputs=[InputSpec(
            name="items", type="object_list", required=True, min_items=2,
            item_fields=[
                InputSpec(name="nome", type="string", required=False),
                InputSpec(name="prezzo", type="decimal", required=True),
                InputSpec(name="bonus", type="boolean", required=False, default=False),
            ],
        )],
        formula={
            "candidates_input": "items",
            "label_field": "nome",
            "aggregates": {"prezzo_max": {"function": "max", "over": "prezzo"}},
            "components": [
                {"name": "costo", "weight": "0.7",
                 "expression": "100 - prezzo / prezzo_max * 100",
                 "clamp": {"min": 0, "max": 100}},
                {"name": "extra", "weight": "0.3",
                 "points": [{"field": "bonus", "points": 10}], "scale_max": 10},
            ],
        },
        output={"name": "ranking", "round_to": 2},
    )
    base.update(overrides)
    return CalculatorDefinition(**base)


def _run(definition, items):
    return _strategy().run(
        definition, {"items": items}, CalculationRequest(calculator_id=definition.id)
    )


# ---------------------------------------------------------------- engine e2e


def test_polizze_pack_ranks_and_reports_breakdown():
    result = engine.calculate(CalculationRequest(
        calculator_id="business.confronto_polizze",
        inputs={"eta_conducente": 40, "polizze": [
            {"nome": "Economica", "premio_annuo": 300},
            {"nome": "Completa", "premio_annuo": 600,
             "copertura_kasko": True, "copertura_infortuni": True,
             "copertura_cristalli": True, "assistenza_stradale": True},
        ]},
    ))
    assert result.status == "success", result.errors
    ranking = result.result["ranking"]
    assert [entry["rank"] for entry in ranking] == [1, 2]
    assert result.result["best"] == ranking[0]["label"]
    # every component score stays on the 0-100 scale
    for entry in ranking:
        for score in entry["scores"].values():
            assert Decimal("0") <= Decimal(score) <= Decimal("100")


def test_single_candidate_is_rejected():
    result = engine.calculate(CalculationRequest(
        calculator_id="business.confronto_polizze",
        inputs={"eta_conducente": 40, "polizze": [{"nome": "Sola", "premio_annuo": 300}]},
    ))
    assert result.status == "error"
    assert "at least 2" in result.errors[0].message


def test_missing_item_field_error_names_the_item():
    result = engine.calculate(CalculationRequest(
        calculator_id="business.confronto_polizze",
        inputs={"eta_conducente": 40, "polizze": [
            {"nome": "Ok", "premio_annuo": 300},
            {"nome": "Senza premio"},
        ]},
    ))
    assert result.status == "error"
    assert "polizze[1]" in result.errors[0].message
    assert "premio_annuo" in result.errors[0].message


def test_unknown_candidate_field_is_rejected_not_silently_dropped():
    # A typo'd field (kaskoo) must not silently score the policy as uncovered.
    result = engine.calculate(CalculationRequest(
        calculator_id="business.confronto_polizze",
        inputs={"eta_conducente": 40, "polizze": [
            {"nome": "Ok", "premio_annuo": 300},
            {"nome": "Typo", "premio_annuo": 400, "copertura_kaskoo": True},
        ]},
    ))
    assert result.status == "error"
    assert "copertura_kaskoo" in result.errors[0].message


def test_nan_input_is_rejected_as_structured_error():
    result = engine.calculate(CalculationRequest(
        calculator_id="business.confronto_polizze",
        inputs={"eta_conducente": 40, "polizze": [
            {"nome": "A", "premio_annuo": "NaN"},
            {"nome": "B", "premio_annuo": 300},
        ]},
    ))
    assert result.status == "error"
    assert "finite" in result.errors[0].message


def test_all_zero_costs_score_every_candidate_full_marks():
    # Both offers free and zero consumption. The old `100 - cost/max*100`
    # divided 0/0 and surfaced a CalculationError; ratio_to_best resolves the
    # degenerate set instead — with nothing to tell the offers apart on cost,
    # penalising either of them would be an invention.
    result = engine.calculate(CalculationRequest(
        calculator_id="business.confronto_gas_luce",
        inputs={
            "consumo_annuo_kwh": 0,
            "consumo_annuo_smc": 0,
            "offerte": [
                {"fornitore": "Gratis A", "prezzo_kwh_luce": 0, "prezzo_smc_gas": 0},
                {"fornitore": "Gratis B", "prezzo_kwh_luce": 0, "prezzo_smc_gas": 0},
            ],
        },
    ))
    assert result.status == "success", result.errors
    for entry in result.result["ranking"]:
        assert entry["scores"]["punteggio_costo"] == "100.00"
    assert result.result["comparison"]["decision_status"] == "effective_tie"


def test_one_free_offer_beats_every_priced_one_outright():
    # min is zero with positives present: the ratio against zero is
    # undefined, so free takes the whole component and priced takes none.
    result = engine.calculate(CalculationRequest(
        calculator_id="business.confronto_gas_luce",
        inputs={
            "consumo_annuo_kwh": 1000,
            "consumo_annuo_smc": 0,
            "offerte": [
                {"fornitore": "Gratis", "prezzo_kwh_luce": 0, "prezzo_smc_gas": 0},
                {"fornitore": "A pagamento", "prezzo_kwh_luce": "0.20", "prezzo_smc_gas": "1.00"},
            ],
        },
    ))
    assert result.status == "success", result.errors
    by_label = {e["label"]: e for e in result.result["ranking"]}
    assert by_label["Gratis"]["scores"]["punteggio_costo"] == "100.00"
    assert by_label["A pagamento"]["scores"]["punteggio_costo"] == "0.00"


def test_stacked_discounts_cannot_go_below_zero_cost():
    result = engine.calculate(CalculationRequest(
        calculator_id="business.confronto_gas_luce",
        inputs={
            "consumo_annuo_kwh": 1000,
            "consumo_annuo_smc": 0,
            "offerte": [
                {"fornitore": "Scontatissima", "prezzo_kwh_luce": "0.20", "prezzo_smc_gas": "1.00",
                 "sconto_primo_anno": 0.9, "sconto_pagamento_rid": 0.9},
                {"fornitore": "Normale", "prezzo_kwh_luce": "0.20", "prezzo_smc_gas": "1.00"},
            ],
        },
    ))
    assert result.status == "success", result.errors
    by_label = {e["label"]: e for e in result.result["ranking"]}
    assert by_label["Scontatissima"]["derived"]["costo_annuo_scontato"] == "0.00"


def test_gas_luce_shared_consumption_feeds_every_offer():
    result = engine.calculate(CalculationRequest(
        calculator_id="business.confronto_gas_luce",
        inputs={
            "consumo_annuo_kwh": 1000,
            "consumo_annuo_smc": 0,
            "offerte": [
                {"fornitore": "Caro", "prezzo_kwh_luce": "0.30", "prezzo_smc_gas": "1.00"},
                {"fornitore": "Conveniente", "prezzo_kwh_luce": "0.20", "prezzo_smc_gas": "1.00"},
            ],
        },
    ))
    assert result.status == "success", result.errors
    by_label = {e["label"]: e for e in result.result["ranking"]}
    assert by_label["Caro"]["derived"]["costo_annuo_lordo"] == "300.00"
    assert by_label["Conveniente"]["derived"]["costo_annuo_lordo"] == "200.00"
    assert result.result["best"] == "Conveniente"


# ------------------------------------------------------------ strategy units


def test_label_falls_back_to_position_when_field_missing():
    outcome = _run(_definition(), [
        {"prezzo": Decimal("10")},
        {"prezzo": Decimal("20")},
    ])
    labels = {e["label"] for e in outcome.result["ranking"]}
    assert labels == {"#1", "#2"}


def test_tie_breaks_alphabetically_on_label():
    outcome = _run(_definition(), [
        {"nome": "Beta", "prezzo": Decimal("10")},
        {"nome": "Alfa", "prezzo": Decimal("10")},
    ])
    assert [e["label"] for e in outcome.result["ranking"]] == ["Alfa", "Beta"]
    assert outcome.result["best"] == "Alfa"


def test_aggregates_are_exposed_as_derived_values():
    outcome = _run(_definition(), [
        {"nome": "A", "prezzo": Decimal("10")},
        {"nome": "B", "prezzo": Decimal("40")},
    ])
    assert outcome.derived_values["prezzo_max"] == Decimal("40")


def test_points_component_normalizes_to_100():
    outcome = _run(_definition(), [
        {"nome": "ConBonus", "prezzo": Decimal("10"), "bonus": True},
        {"nome": "SenzaBonus", "prezzo": Decimal("10"), "bonus": False},
    ])
    by_label = {e["label"]: e for e in outcome.result["ranking"]}
    assert by_label["ConBonus"]["scores"]["extra"] == Decimal("100.00")
    assert by_label["SenzaBonus"]["scores"]["extra"] == Decimal("0.00")


def test_ranking_uses_exact_totals_not_rounded_display_values():
    # Z and A display the same rounded total (50.00) but Z's exact score is
    # higher; alphabetical tie-breaking must NOT overrule the real ordering.
    definition = _definition(formula={
        "candidates_input": "items",
        "label_field": "nome",
        "components": [{"name": "valore", "weight": "1", "expression": "prezzo"}],
    })
    outcome = _run(definition, [
        {"nome": "A", "prezzo": Decimal("50.003")},
        {"nome": "Z", "prezzo": Decimal("50.004")},
    ])
    assert [e["label"] for e in outcome.result["ranking"]] == ["Z", "A"]
    assert [e["total_score"] for e in outcome.result["ranking"]] == [Decimal("50.00"), Decimal("50.00")]


def test_duplicate_labels_are_disambiguated():
    outcome = _run(_definition(), [
        {"nome": "Same", "prezzo": Decimal("10")},
        {"nome": "Same", "prezzo": Decimal("20")},
    ])
    labels = {e["label"] for e in outcome.result["ranking"]}
    assert labels == {"Same", "Same #2"}


def test_equals_rule_matches_fractional_numbers():
    # Decimal("0.5") field vs YAML float 0.5 — must match by numeric value.
    definition = _definition(formula={
        "candidates_input": "items",
        "label_field": "nome",
        "components": [
            {"name": "malus", "weight": "1", "base": 100,
             "rules": [{"when": {"field": "prezzo", "equals": 0.5}, "points": -50}]},
        ],
    })
    outcome = _run(definition, [
        {"nome": "Mezzo", "prezzo": Decimal("0.5")},
        {"nome": "Uno", "prezzo": Decimal("1")},
    ])
    by_label = {e["label"]: e for e in outcome.result["ranking"]}
    assert by_label["Mezzo"]["scores"]["malus"] == Decimal("50.00")
    assert by_label["Uno"]["scores"]["malus"] == Decimal("100.00")


def test_component_scores_are_clamped_to_100_even_without_declared_clamp():
    definition = _definition(formula={
        "candidates_input": "items",
        "label_field": "nome",
        "components": [
            {"name": "bonus", "weight": "1", "base": 100,
             "rules": [{"when": {"field": "prezzo", "at_least": 0}, "points": 50}]},
        ],
    })
    outcome = _run(definition, [
        {"nome": "A", "prezzo": Decimal("1")},
        {"nome": "B", "prezzo": Decimal("2")},
    ])
    for entry in outcome.result["ranking"]:
        assert entry["scores"]["bonus"] == Decimal("100.00")
        assert entry["total_score"] == Decimal("100.00")


def test_rules_component_applies_per_unit_points_and_clamp():
    definition = _definition(formula={
        "candidates_input": "items",
        "label_field": "nome",
        "components": [
            {"name": "malus", "weight": "1",
             "base": 100,
             "rules": [{"when": {"field": "prezzo", "greater_than": 0},
                        "points_per_unit": -30}],
             "clamp": {"min": 0, "max": 100}},
        ],
    })
    outcome = _run(definition, [
        {"nome": "Lieve", "prezzo": Decimal("2")},   # 100 - 60 = 40
        {"nome": "Oltre", "prezzo": Decimal("5")},   # 100 - 150 -> clamped to 0
    ])
    by_label = {e["label"]: e for e in outcome.result["ranking"]}
    assert by_label["Lieve"]["scores"]["malus"] == Decimal("40.00")
    assert by_label["Oltre"]["scores"]["malus"] == Decimal("0.00")


# ------------------------------------------------------- definition validation


def _expect_invalid(definition, fragment):
    with pytest.raises(DefinitionValidationError) as exc:
        validate_definition(definition, "inline-test")
    assert fragment in str(exc.value)


def test_weights_must_sum_to_one():
    definition = _definition()
    definition.formula["components"][0]["weight"] = "0.5"
    _expect_invalid(definition, "weights must sum to 1")


def test_points_field_must_be_boolean():
    definition = _definition()
    definition.formula["components"][1]["points"] = [{"field": "prezzo", "points": 10}]
    _expect_invalid(definition, "not a declared boolean item field")


def test_candidates_input_must_be_object_list():
    definition = _definition()
    definition.formula["candidates_input"] = "missing"
    _expect_invalid(definition, "does not reference a declared object_list input")


def test_expression_referencing_unknown_variable_fails_at_load():
    definition = _definition()
    definition.formula["components"][0]["expression"] = "100 - sconosciuto"
    _expect_invalid(definition, "sconosciuto")


def test_object_list_requires_item_fields():
    definition = _definition()
    definition.inputs[0].item_fields = None
    _expect_invalid(definition, "declares no item_fields")


def test_negative_weight_is_rejected_even_if_sum_is_one():
    definition = _definition()
    definition.formula["components"][0]["weight"] = "-1"
    definition.formula["components"][1]["weight"] = "2"
    _expect_invalid(definition, "weight must be a finite number in [0, 1]")


def test_comparator_requires_min_items_of_two():
    definition = _definition()
    definition.inputs[0].min_items = None
    _expect_invalid(definition, "min_items >= 2")


def test_output_name_best_is_reserved():
    definition = _definition(output={"name": "best", "round_to": 2})
    _expect_invalid(definition, "reserved")


def test_aggregate_over_optional_field_without_default_is_rejected():
    definition = _definition()
    definition.inputs[0].item_fields[1].required = False  # prezzo now optional, no default
    _expect_invalid(definition, "numeric always-present")


def test_numeric_rule_on_boolean_field_is_rejected():
    definition = _definition()
    definition.formula["components"][1] = {
        "name": "extra", "weight": "0.3", "base": 100,
        "rules": [{"when": {"field": "bonus", "greater_than": 0}, "points": -10}],
    }
    _expect_invalid(definition, "not numeric-and-always-present")


def test_candidate_field_colliding_with_scalar_input_is_rejected():
    definition = _definition()
    definition.inputs.append(InputSpec(name="prezzo", type="decimal", required=True))
    _expect_invalid(definition, "collides")


def test_clamp_with_min_above_max_is_rejected():
    definition = _definition()
    definition.formula["components"][0]["clamp"] = {"min": 90, "max": 10}
    _expect_invalid(definition, "clamp")


def _relative_definition(**component_overrides):
    component = {
        "name": "costo", "weight": "1",
        "relative_expression": "prezzo",
        "direction": "lower_is_better",
        "normalization": "ratio_to_best",
    }
    component.update(component_overrides)
    return _definition(formula={
        "candidates_input": "items", "label_field": "nome", "components": [component],
    })


def test_relative_component_definition_passes():
    validate_definition(_relative_definition(), "inline-test")


def test_relative_component_rejects_an_unknown_direction():
    _expect_invalid(_relative_definition(direction="cheaper"), "direction must be one of")


def test_relative_component_rejects_a_missing_direction():
    definition = _relative_definition()
    del definition.formula["components"][0]["direction"]
    _expect_invalid(definition, "direction must be one of")


def test_relative_component_rejects_an_unknown_normalization():
    _expect_invalid(_relative_definition(normalization="z_score"), "normalization must be one of")


def test_relative_expression_referencing_an_undeclared_variable_fails_at_load():
    _expect_invalid(_relative_definition(relative_expression="prezzo / sconosciuto"), "sconosciuto")


def test_a_component_may_declare_only_one_kind():
    definition = _relative_definition()
    definition.formula["components"][0]["expression"] = "prezzo"
    _expect_invalid(definition, "must declare exactly one of")


def test_direction_on_a_non_relative_component_is_rejected():
    # Silently ignoring it would let an author believe a plain expression is
    # being normalized when nothing normalizes it.
    definition = _definition(formula={
        "candidates_input": "items", "label_field": "nome",
        "components": [{
            "name": "costo", "weight": "1", "expression": "prezzo",
            "direction": "lower_is_better",
        }],
    })
    _expect_invalid(definition, "those apply to relative_expression only")


@pytest.mark.parametrize("tolerance", ["-1", "abc", "NaN", "Infinity"])
def test_invalid_tie_tolerance_fails_at_load(tolerance):
    definition = _definition()
    definition.formula["tie_tolerance"] = tolerance
    _expect_invalid(definition, "tie_tolerance must be a finite non-negative number")


def test_zero_tie_tolerance_is_allowed():
    definition = _definition()
    definition.formula["tie_tolerance"] = "0"
    validate_definition(definition, "inline-test")


def test_output_name_comparison_is_reserved():
    _expect_invalid(_definition(output={"name": "comparison", "round_to": 2}), "reserved")


def test_valid_definition_passes():
    validate_definition(_definition(), "inline-test")


# ------------------------------------------------------- weight-agnostic props

def _dominance_offers():
    """Two ordinary offers plus one strictly better than both on every scored
    component: cheapest premium, every coverage, no risk penalties, the largest
    discounts, both services, the best rating. `massimale` is deliberately left
    out — it feeds no component today, and whether it should is still open."""
    return [
        {"nome": "Alfa", "premio_annuo": 600, "franchigia": 400,
         "guida_esclusiva": False, "sconto_no_sinistri": "0.05",
         "sconto_fedelta": "0.05", "voto_medio_utenti": "3.8"},
        {"nome": "Beta", "premio_annuo": 500, "franchigia": 350,
         "copertura_kasko": True, "guida_esclusiva": False,
         "sconto_no_sinistri": "0.05", "sconto_fedelta": "0.05",
         "telemedicina": True, "voto_medio_utenti": "4.0"},
        {"nome": "Omega", "premio_annuo": 400, "franchigia": 200,
         "copertura_kasko": True, "copertura_infortuni": True,
         "copertura_cristalli": True, "assistenza_stradale": True,
         "guida_esclusiva": True, "sconto_no_sinistri": "0.15",
         "sconto_fedelta": "0.10", "telemedicina": True, "app_gestione": True,
         "voto_medio_utenti": "4.8"},
    ]


def _rank_dominance_offers():
    result = engine.calculate(CalculationRequest(
        calculator_id="business.confronto_polizze",
        inputs={"eta_conducente": 40, "storico_sinistri": 0,
                "polizze": _dominance_offers()},
    ))
    assert result.status == "success", result.errors
    return result.result["ranking"]


def test_a_strictly_dominating_offer_wins_under_any_weighting():
    """Sign/scale regression guard that survives the weights being retuned.

    Pinning exact totals breaks the moment a weight changes; dominance holds
    for every non-negative weighting, so this keeps catching the bug class the
    packs were built to correct (a component scored so that worse reads as
    better) without needing an update when the numbers are settled.
    """
    ranking = _rank_dominance_offers()

    assert ranking[0]["label"] == "Omega"
    # Strict, not just first: a flipped component would still let Omega tie.
    assert Decimal(ranking[0]["total_score"]) > Decimal(ranking[1]["total_score"])
    for component, score in ranking[0]["scores"].items():
        for other in ranking[1:]:
            assert Decimal(score) >= Decimal(other["scores"][component]), (
                f"dominating offer scored lower on {component!r} "
                f"than {other['label']!r}"
            )


def test_ranking_is_a_permutation_of_the_offers_given():
    """Nothing silently dropped, duplicated, or invented between input and
    ranking — the scoring passes build their own lists per candidate."""
    ranking = _rank_dominance_offers()

    assert [entry["rank"] for entry in ranking] == [1, 2, 3]
    assert sorted(entry["label"] for entry in ranking) == ["Alfa", "Beta", "Omega"]


# ------------------------------------------------ relative cost normalization


def _polizze(offers, **shared):
    inputs = {"eta_conducente": 40, "storico_sinistri": 0, **shared, "polizze": offers}
    result = engine.calculate(CalculationRequest(
        calculator_id="business.confronto_polizze", inputs=inputs,
    ))
    assert result.status == "success", result.errors
    return result


def _cost_scores(result):
    return {e["label"]: e["scores"]["punteggio_costo"] for e in result.result["ranking"]}


_PAIR = [
    {"nome": "Alfa", "premio_annuo": 400},
    {"nome": "Beta", "premio_annuo": 500},
]


def test_cheapest_offer_takes_the_full_cost_score():
    # The whole point of ratio_to_best: under the old `100 - cost/worst*100`
    # the cheapest offer scored 20 here and the dearest a flat 0.
    scores = _cost_scores(_polizze(_PAIR))
    assert scores["Alfa"] == "100.00"
    assert scores["Beta"] == "80.00"  # 400/500


def test_adding_a_dominated_expensive_offer_leaves_existing_scores_untouched():
    """The regression the normalization exists to prevent.

    With the reference point at the WORST candidate, quoting a third
    ludicrous offer silently rescored the two the user actually cared about
    — and could reorder them. Anchoring on the best makes the added offer
    irrelevant to everyone else by construction.
    """
    before = _polizze(_PAIR)
    after = _polizze(_PAIR + [{"nome": "Carissima", "premio_annuo": 5000}])

    for label in ("Alfa", "Beta"):
        assert _cost_scores(after)[label] == _cost_scores(before)[label]
    order_before = [e["label"] for e in before.result["ranking"]]
    order_after = [e["label"] for e in after.result["ranking"] if e["label"] != "Carissima"]
    assert order_after == order_before


def test_candidate_order_does_not_change_any_score_or_the_ranking():
    forward = _polizze(_PAIR)
    reversed_ = _polizze(list(reversed(_PAIR)))

    def _by_label(result):
        return {e["label"]: (e["total_score"], e["scores"]) for e in result.result["ranking"]}

    assert _by_label(forward) == _by_label(reversed_)
    assert [e["label"] for e in forward.result["ranking"]] == [
        e["label"] for e in reversed_.result["ranking"]
    ]


def test_identical_offers_are_an_effective_tie_with_no_single_best():
    result = _polizze([
        {"nome": "Alfa", "premio_annuo": 400},
        {"nome": "Beta", "premio_annuo": 400},
    ])
    comparison = result.result["comparison"]
    assert comparison["decision_status"] == "effective_tie"
    assert sorted(comparison["best_candidates"]) == ["Alfa", "Beta"]
    assert comparison["score_gap"] == "0.00"
    # `best` still exists for backward compatibility, but the tie warning
    # must be present so no renderer can quote it as a recommendation.
    assert result.result["best"] in comparison["best_candidates"]
    messages = [w.message for w in result.warnings]
    assert any("Nessuna differenza sostanziale" in m for m in messages)
    assert any("non c'e un vincitore netto" in m.lower() for m in messages)


def test_tie_determination_uses_exact_totals_not_the_rounded_gap():
    """A gap of 0.5000001 displays as "0.50" at round_to=2, which would read
    as exactly on the 0.50 tolerance. The decision must come from the exact
    totals, so this is a clear winner despite what the display says."""
    definition = _definition(formula={
        "candidates_input": "items",
        "label_field": "nome",
        "tie_tolerance": "0.50",
        "components": [{"name": "valore", "weight": "1", "expression": "prezzo"}],
    })
    outcome = _run(definition, [
        {"nome": "A", "prezzo": Decimal("50")},
        {"nome": "B", "prezzo": Decimal("49.4999999")},
    ])
    comparison = outcome.result["comparison"]
    assert comparison["score_gap"] == Decimal("0.50")  # display says "tie"
    assert comparison["decision_status"] == "clear_winner"
    assert comparison["best_candidates"] == ["A"]


def test_tie_tolerance_is_configurable_per_pack():
    definition = _definition(formula={
        "candidates_input": "items",
        "label_field": "nome",
        "tie_tolerance": "5",
        "components": [{"name": "valore", "weight": "1", "expression": "prezzo"}],
    })
    outcome = _run(definition, [
        {"nome": "A", "prezzo": Decimal("50")},
        {"nome": "B", "prezzo": Decimal("47")},
    ])
    assert outcome.result["comparison"]["decision_status"] == "effective_tie"
    assert outcome.result["comparison"]["tie_tolerance"] == Decimal("5")


def test_higher_is_better_scores_against_the_maximum():
    definition = _definition(formula={
        "candidates_input": "items",
        "label_field": "nome",
        "components": [{
            "name": "valore", "weight": "1",
            "relative_expression": "prezzo",
            "direction": "higher_is_better",
            "normalization": "ratio_to_best",
        }],
    })
    outcome = _run(definition, [
        {"nome": "A", "prezzo": Decimal("80")},
        {"nome": "B", "prezzo": Decimal("100")},
    ])
    by_label = {e["label"]: e for e in outcome.result["ranking"]}
    assert by_label["B"]["scores"]["valore"] == Decimal("100.00")
    assert by_label["A"]["scores"]["valore"] == Decimal("80.00")


# ------------------------------------------------------- insurance modelling


def test_net_premium_drives_the_cost_score_and_discounts_count_once():
    """A 20%-discounted 500 EUR policy and an undiscounted 400 EUR policy
    cost the same. They must score the same on cost — and there must be no
    second component paying the first one again for having a discount."""
    result = _polizze([
        {"nome": "Scontata", "premio_annuo": 500, "sconto_no_sinistri": "0.20"},
        {"nome": "Netta", "premio_annuo": 400},
    ])
    by_label = {e["label"]: e for e in result.result["ranking"]}
    assert by_label["Scontata"]["derived"]["premio_netto"] == "400.00"
    assert by_label["Scontata"]["scores"]["punteggio_costo"] == "100.00"
    assert by_label["Netta"]["scores"]["punteggio_costo"] == "100.00"
    # The discount reaches the score exactly once, through the cost.
    assert "punteggio_sconti" not in by_label["Scontata"]["scores"]
    assert by_label["Scontata"]["total_score"] == by_label["Netta"]["total_score"]


def test_full_discount_floors_the_net_premium_at_zero():
    result = _polizze([
        {"nome": "Regalata", "premio_annuo": 500,
         "sconto_no_sinistri": "0.60", "sconto_fedelta": "0.60"},
        {"nome": "Normale", "premio_annuo": 400},
    ])
    by_label = {e["label"]: e for e in result.result["ranking"]}
    assert by_label["Regalata"]["derived"]["premio_netto"] == "0.00"


def test_shared_applicant_facts_never_reorder_otherwise_identical_offers():
    """Age and claims history are the same for every quote by construction.
    Scoring them subtracted the same points from everyone — informative
    noise at best, and it made the comparison look better informed than it
    was. They must not move a single number."""
    offers = [
        {"nome": "Alfa", "premio_annuo": 400, "copertura_kasko": True},
        {"nome": "Beta", "premio_annuo": 500},
    ]
    young = _polizze(offers, eta_conducente=19, storico_sinistri=4)
    old = _polizze(offers, eta_conducente=70, storico_sinistri=0)

    def _shape(result):
        return [(e["rank"], e["label"], e["total_score"], e["scores"])
                for e in result.result["ranking"]]

    assert _shape(young) == _shape(old)
    # Still collected, validated and auditable — just not scored.
    assert young.inputs_used["eta_conducente"] == 19
    assert any(step.get("type") == "shared_inputs" for step in young.steps)


def test_weights_still_sum_to_exactly_one_in_both_shipped_packs():
    for calculator_id in ("business.confronto_polizze", "business.confronto_gas_luce"):
        definition = engine.registry.get(calculator_id)
        total = sum(
            Decimal(str(component["weight"]))
            for component in definition.formula["components"]
        )
        assert total == Decimal("1"), f"{calculator_id} weights sum to {total}"


def test_massimale_is_collected_but_never_reduces_scoring_completeness():
    """A declared field no component reads says nothing about the quality of
    the ranking; counting it would make every comparison look incomplete for
    a reason the user cannot act on."""
    result = _polizze([
        {"nome": "Alfa", "premio_annuo": 400, "massimale": 1000000},
        {"nome": "Beta", "premio_annuo": 500},
    ])
    comparison = result.result["comparison"]
    assert "massimale" not in comparison["scored_fields"]
    by_label = {e["label"]: e for e in result.result["ranking"]}
    assert by_label["Beta"]["data_quality"]["unknown_fields"] == ["massimale"]
    assert "massimale" not in by_label["Alfa"]["data_quality"]["unknown_fields"]


# --------------------------------------------------------- data-quality metadata


def test_explicit_false_and_zero_are_not_the_same_as_omitted_or_defaulted():
    """Three states the metadata must keep apart: the user said no, the
    platform assumed no, and nobody knows. Flattening them lets a renderer
    claim a coverage was ruled out when it was only never mentioned."""
    result = _polizze([
        {"nome": "Esplicita", "premio_annuo": 400,
         "copertura_kasko": False, "franchigia": 0, "massimale": 500000},
        {"nome": "Omessa", "premio_annuo": 500},
    ])
    by_label = {e["label"]: e["data_quality"] for e in result.result["ranking"]}

    explicit = by_label["Esplicita"]
    assert "copertura_kasko" in explicit["provided_fields"]
    assert "franchigia" in explicit["provided_fields"]
    assert "copertura_kasko" not in explicit["assumed_fields"]
    assert explicit["unknown_fields"] == []

    omitted = by_label["Omessa"]
    assert "copertura_kasko" in omitted["assumed_fields"]
    assert "copertura_kasko" not in omitted["provided_fields"]
    assert "massimale" in omitted["unknown_fields"]

    # Same value in the payload, different provenance.
    assert result.inputs_used["polizze"][0]["copertura_kasko"] is False
    assert result.inputs_used["polizze"][1]["copertura_kasko"] is False


def test_structured_defaults_name_the_nested_candidate_path():
    result = _polizze(_PAIR)
    paths = {entry["path"] for entry in result.defaults_applied}
    assert "polizze[0].franchigia" in paths
    assert {"path": "polizze[0].franchigia", "value": "0"} in result.defaults_applied
    # A defaulted boolean stays a boolean, so "assumed false" is readable as
    # such instead of as the string "False".
    assert {"path": "polizze[0].copertura_kasko", "value": False} in result.defaults_applied


def test_a_fully_specified_comparison_is_not_provisional():
    complete = {
        "premio_annuo": 400, "franchigia": 100, "copertura_kasko": True,
        "copertura_cristalli": True, "copertura_infortuni": True,
        "assistenza_stradale": True, "guida_esclusiva": True,
        "sconto_no_sinistri": "0.10", "sconto_fedelta": "0.05",
        "telemedicina": True, "app_gestione": True, "voto_medio_utenti": "4.5",
    }
    result = _polizze([
        {"nome": "Alfa", **complete},
        {"nome": "Beta", **{**complete, "premio_annuo": 500}},
    ])
    comparison = result.result["comparison"]
    assert comparison["provisional"] is False
    assert comparison["provisional_status"] == "none"
    assert comparison["scoring_completeness"] == "1.0000"
    assert comparison["scoring_defaults_applied"] == []


def test_confirmation_records_acknowledgement_without_dropping_assumptions():
    inputs = {"eta_conducente": 40, "polizze": _PAIR}
    plain = engine.calculate(CalculationRequest(
        calculator_id="business.confronto_polizze", inputs=inputs))
    confirmed = engine.calculate(CalculationRequest(
        calculator_id="business.confronto_polizze", inputs=inputs,
        confirm_assumptions=True))

    assert plain.result["comparison"]["provisional_status"] == "provisional_unconfirmed"
    assert confirmed.result["comparison"]["provisional_status"] == "confirmed_with_assumptions"
    # Confirmation is an acknowledgement, not a retraction: still
    # provisional, and every assumption and default is still reported.
    assert confirmed.result["comparison"]["provisional"] is True
    assert [a.message for a in confirmed.assumptions] == [a.message for a in plain.assumptions]
    assert confirmed.defaults_applied == plain.defaults_applied
    # And the numbers are untouched.
    assert confirmed.result["ranking"] == plain.result["ranking"]


# ---------------------------------------------------------- energy omissions


def test_gas_luce_omissions_stay_visible_in_machine_readable_output():
    result = engine.calculate(CalculationRequest(
        calculator_id="business.confronto_gas_luce",
        inputs={
            "consumo_annuo_kwh": 2000, "consumo_annuo_smc": 800,
            "offerte": [
                {"fornitore": "A", "prezzo_kwh_luce": "0.25", "prezzo_smc_gas": "1.10"},
                {"fornitore": "B", "prezzo_kwh_luce": "0.22", "prezzo_smc_gas": "1.05"},
            ],
        },
    ))
    assert result.status == "success", result.errors
    joined = " ".join(result.exclusions).lower()
    for omission in ("iva", "accise", "oneri di sistema", "f1/f2/f3", "primo anno"):
        assert omission in joined, f"undisclosed omission: {omission}"
    # And as a warning too, so a renderer that only prints warnings still
    # tells the reader this is not the bill.
    assert any("bolletta" in w.message.lower() for w in result.warnings)


def test_gas_luce_inputs_are_rejected_by_the_polizze_pack():
    """The two packs name their candidates differently (offerte vs polizze) and
    share no item fields. Feeding one the other's shape must fail outright, not
    score a partial ranking off whatever happens to line up."""
    result = engine.calculate(CalculationRequest(
        calculator_id="business.confronto_polizze",
        inputs={"consumo_annuo_kwh": 2700, "offerte": [
            {"fornitore": "Alfa", "prezzo_kwh_luce": "0.12"},
            {"fornitore": "Beta", "prezzo_kwh_luce": "0.11"},
        ]},
    ))

    assert result.status == "error"
    assert result.result == {}
    message = result.errors[0].message
    assert "eta_conducente" in message and "polizze" in message


# --- Regressions found by external review ---------------------------------


def test_an_absurd_but_legal_value_is_a_structured_error_not_a_crash():
    """A price of 1E+100 is legal input; the final display quantize then
    raises decimal.InvalidOperation from inside otherwise correct code. The
    engine promises callers never see a raw Python exception."""
    result = engine.calculate(CalculationRequest(
        calculator_id="business.confronto_gas_luce",
        inputs={
            "consumo_annuo_kwh": 1, "consumo_annuo_smc": 0,
            "offerte": [
                {"fornitore": "Assurda", "prezzo_kwh_luce": "1E+100", "prezzo_smc_gas": 0},
                {"fornitore": "Normale", "prezzo_kwh_luce": "1", "prezzo_smc_gas": 0},
            ],
        },
    ))
    assert result.status == "error"
    assert result.errors[0].code
    assert result.result == {}


def test_a_defaulted_candidate_list_makes_the_whole_comparison_provisional():
    """If the offers themselves came from a default, nothing about the
    ranking is caller-supplied — it must not report full completeness."""
    definition = _definition(
        inputs=[InputSpec(
            name="items", type="object_list", required=False, min_items=2,
            default=[{"nome": "A", "prezzo": "10"}, {"nome": "B", "prezzo": "20"}],
            item_fields=[
                InputSpec(name="nome", type="string", required=False),
                InputSpec(name="prezzo", type="decimal", required=True),
                InputSpec(name="bonus", type="boolean", required=False, default=False),
            ],
        )],
    )
    validate_definition(definition, "inline-test")

    from app.core.validators import validate_inputs

    validated = validate_inputs(definition, {})
    strategy = _strategy()
    strategy.validated_inputs = validated
    outcome = strategy.run(
        definition, validated.values, CalculationRequest(calculator_id=definition.id)
    )

    comparison = outcome.result["comparison"]
    assert comparison["provisional"] is True
    assert comparison["scoring_completeness"] == Decimal("0.0000")
    for entry in outcome.result["ranking"]:
        assert entry["data_quality"]["provided_fields"] == []
        assert entry["data_quality"]["assumed_fields"]
