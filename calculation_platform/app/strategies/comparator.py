"""Scores and ranks N candidate offers against each other in one request.

Shape: a single `object_list` input holds the candidates (each an object
validated against the pack's declared item_fields); the pack declares
per-candidate derived variables, set-level aggregates (e.g. the maximum
premium across all candidates), and a list of weighted scoring
components. Every component is clamped to the 0-100 range at runtime and
the weights are validated to be in [0, 1] and sum to exactly 1, so the
weighted total is 0-100 by construction — the invariant is enforced here,
not merely assumed of pack authors.

Component kinds (exactly one per component):
- `relative_expression`: a numeric quantity (typically a cost) scored
  RELATIVE to the best candidate — see `_ratio_to_best` below;
- `expression`: arithmetic over candidate fields + derived + aggregates
  + top-level scalar inputs, via safe_eval;
- `points`: additive weights on boolean fields, normalized by `scale_max`
  (defaults to the sum of all positive points); an absent optional field
  counts as false;
- `rules`: a `base` score adjusted by ordered condition rules (reusing the
  decision_table comparators); a rule adds flat `points` or
  `points_per_unit` multiplied by the FULL value of its condition field
  (not the excess over the threshold).
Any component may declare `clamp: {min, max}` applied before the
platform's own 0-100 clamp.

Ranking sorts on the unrounded weighted total, with the label as a
deterministic tie-break; rounded values are for display only, so two
candidates that display the same total still rank by their true scores.
"Unrounded" means no DISPLAY rounding: a relative score is a Decimal
division and therefore carries the arithmetic context's precision (28
significant digits by default), so two candidates whose costs differ
beyond that are genuinely indistinguishable here and fall back to the
label tie-break. That is far below any monetary resolution, but it is not
infinite precision and the code does not pretend otherwise.

Beyond the ranking the strategy reports how much to TRUST it, because a
0-100 synthetic score reads far more authoritative than it is:

- `comparison.decision_status` is `effective_tie` whenever the runners-up
  sit within `formula.tie_tolerance` (default 0.50) of the top exact
  total. A 0.2-point lead is arithmetic noise from configured weights,
  not a real-world recommendation, and renderers must not call it a
  winner.
- `comparison.provisional` is true when any default was applied to a
  field that actually feeds a component. The arithmetic is still exact;
  what is uncertain is the input it ran on.
- per-candidate `data_quality` separates fields the caller PROVIDED from
  fields the platform ASSUMED from a default and fields still UNKNOWN —
  three states a single "value present" check would flatten (an explicit
  `false` and a defaulted `false` are not the same claim).
"""

import re
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from ..core.audit import AuditTrail
from ..core.errors import StrategyExecutionError
from ..core.result_builder import round_decimal, round_output
from ..core.safe_evaluator import UnsafeExpressionError, extract_variable_names, safe_eval
from ..resolvers.date_parameter_resolver import resolve_parameters
from .base import CalculationStrategy, StrategyOutcome
from .decision_table import _COMPARATORS

_AGGREGATE_FUNCTIONS = {
    "max": max,
    "min": min,
    "sum": lambda values: sum(values, Decimal("0")),
    "mean": lambda values: sum(values, Decimal("0")) / Decimal(len(values)),
}

_ZERO = Decimal("0")
_HUNDRED = Decimal("100")
_ONE = Decimal("1")

# Declarative vocabulary for a `relative_expression` component. Kept here
# next to the implementation so definition_validator imports the single
# source of truth instead of restating it.
DIRECTIONS = frozenset({"lower_is_better", "higher_is_better"})
NORMALIZATIONS = frozenset({"ratio_to_best"})

# How close two total scores have to be before the comparison refuses to
# name a winner. 0.5 of a 100-point synthetic scale: below that the lead
# is an artefact of the configured weights, not a difference anyone would
# act on. Packs may override it with `formula.tie_tolerance`.
DEFAULT_TIE_TOLERANCE = Decimal("0.50")

# Precision for the reported completeness ratio. It is a data-quality
# fraction, not a monetary output, so it does not follow output.round_to.
_COMPLETENESS_PLACES = 4

_ITEM_PATH = re.compile(r"^(?P<input>\w+)\[(?P<index>\d+)\]\.(?P<field>\w+)$")


def _equality_operands(actual: Any, expected: Any):
    """Make `equals`/`not_equals` compare like-for-like: a Decimal field
    against a YAML float (0.1) must match by numeric value, not by Python's
    exact cross-type comparison (Decimal("0.1") == 0.1 is False)."""
    if isinstance(actual, bool) or isinstance(expected, bool):
        return actual, expected
    if isinstance(actual, Decimal) and isinstance(expected, (int, float, str)):
        try:
            return actual, Decimal(str(expected))
        except InvalidOperation:
            return actual, expected
    return actual, expected


def _matches(condition: Dict[str, Any], env: Dict[str, Any]) -> bool:
    actual = env.get(condition["field"])
    for comparator_name, comparator in _COMPARATORS.items():
        if comparator_name in condition:
            expected = condition[comparator_name]
            if comparator_name in ("equals", "not_equals"):
                actual, expected = _equality_operands(actual, expected)
            return comparator(actual, expected)
    raise StrategyExecutionError(
        f"comparator rule has no recognized comparator: {condition!r}",
        details={"condition": condition, "valid_comparators": sorted(_COMPARATORS)},
    )


def _ratio_to_best(values: List[Decimal], direction: str) -> List[Decimal]:
    """Score a set of numbers against the best of the set, exactly.

    For `lower_is_better` (the cost case) the cheapest candidate scores
    100 and everyone else scores `best / value * 100`, i.e. "how many
    times the winner's price does this cost". That is the property the
    naive `100 - value / worst * 100` lacks: the reference point is the
    BEST candidate, which cannot move when a worse candidate joins the
    set, so adding an expensive offer leaves every existing score and the
    whole relative order untouched. Under the old formula the most
    expensive offer always scored a flat 0 and the cheapest rarely reached
    100, so a third dominated offer silently rescored the other two.

    Degenerate sets are resolved rather than divided by zero:
      - nothing strictly positive (typically all-zero): no candidate can
        be discriminated on this axis, so nobody is penalised — all 100;
      - a zero alongside real costs: free is unbeatable and `best/value`
        is undefined against it, so zero-or-less takes 100 and every
        candidate with a real cost takes 0.
    """
    if direction == "lower_is_better":
        positives = [value for value in values if value > _ZERO]
        if not positives:
            return [_HUNDRED for _ in values]
        if len(positives) < len(values):
            return [_HUNDRED if value <= _ZERO else _ZERO for value in values]
        best = min(positives)
        return [best / value * _HUNDRED for value in values]

    # higher_is_better
    best = max(values) if values else _ZERO
    if best <= _ZERO:
        return [_HUNDRED for _ in values]
    return [max(value, _ZERO) / best * _HUNDRED for value in values]


_NORMALIZERS = {"ratio_to_best": _ratio_to_best}


def component_references(component: Dict[str, Any]) -> Set[str]:
    """Every variable name one component reads directly.

    Shared with definition_validator so "which fields feed the score" is
    computed the same way at load time and at run time.
    """
    if "relative_expression" in component or "expression" in component:
        key = "relative_expression" if "relative_expression" in component else "expression"
        try:
            return set(extract_variable_names(component[key]))
        except UnsafeExpressionError:
            return set()
    if "points" in component:
        return {
            entry["field"]
            for entry in component["points"]
            if isinstance(entry, dict) and entry.get("field")
        }
    if "rules" in component:
        return {
            rule["when"]["field"]
            for rule in component["rules"]
            if isinstance(rule, dict)
            and isinstance(rule.get("when"), dict)
            and rule["when"].get("field")
        }
    return set()


def scoring_variables(formula: Dict[str, Any]) -> Set[str]:
    """Close the components' direct references over derived variables and
    aggregates, so a field read only through `costo_annuo_lordo` still
    counts as feeding the score."""
    derived = formula.get("candidate_derived") or {}
    aggregates = formula.get("aggregates") or {}

    pending: List[str] = []
    for component in formula.get("components") or []:
        if isinstance(component, dict):
            pending.extend(component_references(component))

    seen: Set[str] = set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        if name in derived:
            try:
                pending.extend(extract_variable_names(derived[name]))
            except UnsafeExpressionError:
                continue
        elif name in aggregates:
            over = aggregates[name].get("over") if isinstance(aggregates[name], dict) else None
            if over:
                pending.append(over)
    return seen


class ComparatorStrategy(CalculationStrategy):
    def run(self, definition, inputs: Dict[str, Any], request) -> StrategyOutcome:
        cfg = definition.formula
        candidates_input = cfg["candidates_input"]
        label_field = cfg.get("label_field")
        components = cfg["components"]

        candidates: List[Dict[str, Any]] = inputs[candidates_input]
        scalars = {k: v for k, v in inputs.items() if k != candidates_input}
        resolution = resolve_parameters(definition, self.parameter_store, request)
        scalars.update(resolution.values)

        trail = AuditTrail()
        labels = self._labels(candidates, label_field)

        # The applicant/consumption facts are the same for every candidate
        # by construction, so they can never separate two offers. They are
        # still recorded: they gate validation, they belong in the audit
        # trail, and a reader must be able to see which facts the score was
        # computed under even though none of them differentiates.
        if scalars:
            trail.record(
                "shared_inputs",
                applies_to="every candidate",
                values={name: str(value) for name, value in sorted(scalars.items())},
            )

        # Pass 1 — per-candidate derived variables (candidate fields shadow
        # shared scalars on a name clash).
        envs: List[Dict[str, Any]] = []
        for index, candidate in enumerate(candidates):
            env = dict(scalars)
            env.update(candidate)
            for name, expr in (cfg.get("candidate_derived") or {}).items():
                env[name] = self._eval(expr, env, f"candidate_derived.{name}", labels[index])
                trail.record(
                    "candidate_derived",
                    candidate=labels[index],
                    variable=name,
                    expression=expr,
                    value=str(env[name]),
                )
            envs.append(env)

        # Pass 2 — set-level aggregates across all candidates. Each
        # aggregate is injected into every environment as soon as it is
        # computed, so a later aggregate (or any component) may reference an
        # earlier one.
        aggregates: Dict[str, Decimal] = {}
        for name, agg_cfg in (cfg.get("aggregates") or {}).items():
            function = _AGGREGATE_FUNCTIONS[agg_cfg["function"]]
            over = agg_cfg["over"]
            try:
                values = [Decimal(str(env[over])) for env in envs]
                aggregates[name] = function(values)
            except (KeyError, ArithmeticError, ValueError) as e:
                raise StrategyExecutionError(
                    f"cannot compute aggregate {name!r} over {over!r}: {e}",
                    details={"aggregate": name, "over": over},
                ) from e
            for env in envs:
                env[name] = aggregates[name]
            trail.record("aggregate", variable=name, function=agg_cfg["function"], over=over, value=str(aggregates[name]))

        # Pass 3 — relative components, which are the one kind that cannot
        # be scored from a single candidate's environment: they need the
        # whole set to locate the best value first.
        relative_scores = self._relative_scores(components, envs, labels, trail)

        # Pass 4 — score every candidate on every component, then the
        # weighted total. Exact values drive the total and the ranking;
        # rounding (per output.round_to/output.rounding) is display-only.
        entries = []
        for index, env in enumerate(envs):
            label = labels[index]
            scores: Dict[str, Decimal] = {}
            total_exact = _ZERO
            for component in components:
                name = component["name"]
                if name in relative_scores:
                    value = relative_scores[name][index]
                else:
                    value = self._component_value(component, env, trail, label)
                clamp = component.get("clamp")
                if clamp is not None:
                    if clamp.get("min") is not None:
                        value = max(value, Decimal(str(clamp["min"])))
                    if clamp.get("max") is not None:
                        value = min(value, Decimal(str(clamp["max"])))
                # Platform invariant: every component score is 0-100,
                # whatever the pack's own expressions/rules produce.
                bounded = min(max(value, _ZERO), _HUNDRED)
                if bounded != value:
                    trail.record(
                        "component_clamped_to_scale",
                        candidate=label,
                        component=name,
                        pre_clamp_value=str(value),
                    )
                weight = Decimal(str(component["weight"]))
                contribution = bounded * weight
                total_exact += contribution
                scores[name] = round_output(bounded, definition.output)
                trail.record(
                    "component_scored",
                    candidate=label,
                    component=name,
                    score=str(scores[name]),
                    exact_score=str(bounded),
                    weight=str(weight),
                    weighted_contribution=str(contribution),
                )
            trail.record("total_scored", candidate=label, exact_total=str(total_exact))
            entries.append({
                "label": label,
                "index": index,
                "total_exact": total_exact,
                "total_score": round_output(total_exact, definition.output),
                "scores": scores,
                "derived": {
                    name: round_output(Decimal(str(env[name])), definition.output)
                    for name in (cfg.get("candidate_derived") or {})
                } or None,
            })

        quality = self._data_quality(definition, cfg, candidates, scalars)

        entries.sort(key=lambda e: (-e["total_exact"], e["label"]))
        ranked = []
        for rank, entry in enumerate(entries, start=1):
            trail.record("ranked", rank=rank, candidate=entry["label"], total_score=str(entry["total_score"]))
            ranked.append({
                "rank": rank,
                "label": entry["label"],
                "total_score": entry["total_score"],
                "scores": entry["scores"],
                **({"derived": entry["derived"]} if entry["derived"] is not None else {}),
                "data_quality": quality["per_candidate"][entry["index"]],
            })

        comparison = self._comparison(definition, cfg, entries, quality, request, trail)

        output_name = definition.output.get("name", "ranking")
        return StrategyOutcome(
            result={output_name: ranked, "best": ranked[0]["label"], "comparison": comparison},
            derived_values=dict(aggregates),
            parameters_used=resolution.parameters_used(),
            date_resolution=resolution.date_resolution,
            steps=trail.steps,
            warnings=self._model_warnings(comparison),
        )

    # ------------------------------------------------------------ relative

    def _relative_scores(
        self,
        components: List[Dict[str, Any]],
        envs: List[Dict[str, Any]],
        labels: List[str],
        trail: AuditTrail,
    ) -> Dict[str, List[Decimal]]:
        scores: Dict[str, List[Decimal]] = {}
        for component in components:
            expression = component.get("relative_expression")
            if expression is None:
                continue
            name = component["name"]
            direction = component.get("direction", "lower_is_better")
            normalization = component.get("normalization", "ratio_to_best")
            normalizer = _NORMALIZERS.get(normalization)
            if normalizer is None or direction not in DIRECTIONS:
                # definition_validator rejects these at load time; a miss
                # here means the definition bypassed validation, so fail
                # loudly rather than scoring on a guessed rule.
                raise StrategyExecutionError(
                    f"component {name!r} declares unknown direction/normalization "
                    f"({direction!r}/{normalization!r})",
                    details={"component": name, "direction": direction, "normalization": normalization},
                )
            values = [
                self._eval(expression, env, f"components.{name}.relative_expression", labels[index])
                for index, env in enumerate(envs)
            ]
            scores[name] = normalizer(values, direction)
            reference = (
                min((v for v in values if v > _ZERO), default=_ZERO)
                if direction == "lower_is_better"
                else max(values, default=_ZERO)
            )
            trail.record(
                "relative_reference",
                component=name,
                expression=expression,
                direction=direction,
                normalization=normalization,
                best_value=str(reference),
                values={labels[i]: str(v) for i, v in enumerate(values)},
            )
        return scores

    # -------------------------------------------------------- data quality

    def _data_quality(
        self,
        definition,
        cfg: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        scalars: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Split every declared field into provided / assumed / unknown and
        turn that into a scoring-completeness ratio and a provisional flag.

        Only fields that actually feed a component count towards
        completeness: `massimale` is collected for the record but scored by
        nothing, so leaving it out says nothing about the quality of the
        ranking and must not drag the number down.
        """
        candidates_input = cfg["candidates_input"]
        list_spec = next((s for s in definition.inputs if s.name == candidates_input), None)
        item_names = [f.name for f in (list_spec.item_fields or [])] if list_spec else []
        scalar_names = [s.name for s in definition.inputs if s.name != candidates_input]

        relevant = scoring_variables(cfg)
        scoring_items = [name for name in item_names if name in relevant]
        scoring_scalars = [name for name in scalar_names if name in relevant]

        assumed_items, assumed_scalars = self._applied_defaults(candidates_input)
        # A default applied to the list ITSELF means no offer here came from
        # the caller; every field of every candidate is an assumption.
        whole_list_assumed = any(
            str(entry.get("path")) == candidates_input for entry in self._defaults_applied()
        )

        per_candidate: List[Dict[str, Any]] = []
        provided_total = 0
        for index, candidate in enumerate(candidates):
            if whole_list_assumed:
                assumed = sorted(candidate)
            else:
                assumed = sorted(assumed_items.get(index, set()) & set(item_names))
            provided = sorted(set(candidate) - set(assumed))
            unknown = sorted(set(item_names) - set(candidate))
            provided_here = sum(1 for name in scoring_items if name in provided)
            provided_total += provided_here
            per_candidate.append({
                "provided_fields": provided,
                "assumed_fields": assumed,
                # Declared, not supplied, and with no default to fall back
                # on. Distinct from an assumed field: nothing was put in
                # its place, so no component reads it at all.
                "unknown_fields": unknown,
                "scoring_completeness": _ratio(provided_here, len(scoring_items)),
            })

        scalar_provided = sum(
            1 for name in scoring_scalars if name in scalars and name not in assumed_scalars
        )
        denominator = len(scoring_items) * len(candidates) + len(scoring_scalars)
        overall = _ratio(provided_total + scalar_provided, denominator)

        scoring_defaults = [
            entry
            for entry in self._defaults_applied()
            if _default_affects_scoring(entry, candidates_input, scoring_items, scoring_scalars)
        ]
        return {
            "per_candidate": per_candidate,
            "scoring_completeness": overall,
            "scoring_defaults_applied": scoring_defaults,
            "provisional": bool(scoring_defaults),
            "scored_fields": sorted(scoring_items + scoring_scalars),
        }

    def _defaults_applied(self) -> List[Dict[str, Any]]:
        validated = getattr(self, "validated_inputs", None)
        return list(getattr(validated, "defaults_applied", None) or [])

    def _applied_defaults(self, candidates_input: str) -> Tuple[Dict[int, Set[str]], Set[str]]:
        """Index the engine's structured defaults by candidate position."""
        per_index: Dict[int, Set[str]] = {}
        scalars: Set[str] = set()
        for entry in self._defaults_applied():
            path = str(entry.get("path", ""))
            match = _ITEM_PATH.match(path)
            if match and match.group("input") == candidates_input:
                per_index.setdefault(int(match.group("index")), set()).add(match.group("field"))
            elif "[" not in path and "." not in path:
                scalars.add(path)
        return per_index, scalars

    # ------------------------------------------------------ decision status

    def _comparison(
        self,
        definition,
        cfg: Dict[str, Any],
        entries: List[Dict[str, Any]],
        quality: Dict[str, Any],
        request,
        trail: AuditTrail,
    ) -> Dict[str, Any]:
        tolerance = _tie_tolerance(cfg)
        top = entries[0]["total_exact"]
        # Exact totals decide the tie, never the rounded display values: at
        # round_to=2 two candidates 0.004 apart print identically, and two
        # 0.5001 apart print as a 0.50 gap that would read as within a 0.50
        # tolerance. Both would be the wrong call.
        best_candidates = [e["label"] for e in entries if (top - e["total_exact"]) <= tolerance]
        runner_up = entries[1]["total_exact"] if len(entries) > 1 else top
        gap_exact = top - runner_up
        decision_status = "clear_winner" if len(best_candidates) == 1 else "effective_tie"

        provisional = quality["provisional"]
        confirmed = bool(getattr(request, "confirm_assumptions", False))
        if not provisional:
            provisional_status = "none"
        elif confirmed:
            # The caller has seen the assumptions and accepted them. The
            # assumptions themselves stay on the result untouched — this
            # records acknowledgement, it does not retract anything.
            provisional_status = "confirmed_with_assumptions"
        else:
            provisional_status = "provisional_unconfirmed"

        trail.record(
            "decision_status",
            status=decision_status,
            tie_tolerance=str(tolerance),
            exact_top_total=str(top),
            exact_runner_up_total=str(runner_up),
            exact_gap=str(gap_exact),
            best_candidates=best_candidates,
        )
        return {
            "decision_status": decision_status,
            "best_candidates": best_candidates,
            **({"cost_basis": cost_basis} if (cost_basis := _cost_basis(cfg)) else {}),
            "score_gap": round_output(gap_exact, definition.output),
            "tie_tolerance": tolerance,
            "provisional": provisional,
            "provisional_status": provisional_status,
            "assumptions_confirmed": confirmed,
            "scoring_completeness": quality["scoring_completeness"],
            "scoring_defaults_applied": quality["scoring_defaults_applied"],
            "scored_fields": quality["scored_fields"],
            "candidates_compared": len(entries),
        }

    @staticmethod
    def _model_warnings(comparison: Dict[str, Any]) -> List[str]:
        """Code-emitted caveats a pack author cannot forget to write.

        A renderer that only prints `best` would otherwise present a 0.1
        point lead, or a ranking computed largely on assumed data, as a
        firm recommendation.
        """
        warnings: List[str] = []
        if comparison["decision_status"] == "effective_tie":
            warnings.append(
                "Nessuna differenza sostanziale con il modello di punteggio attuale: "
                f"{', '.join(comparison['best_candidates'])} restano entro la tolleranza "
                f"di {comparison['tie_tolerance']} punti. Non c'e un vincitore netto."
            )
        if comparison["provisional"]:
            assumed = ", ".join(
                str(entry.get("path")) for entry in comparison["scoring_defaults_applied"]
            )
            warnings.append(
                "Risultato PROVVISORIO: alcuni campi che incidono sul punteggio non "
                f"sono stati forniti e sono stati assunti per default ({assumed}). "
                "Confermare o correggere i dati prima di decidere."
            )
        return warnings

    # ------------------------------------------------------------------

    @staticmethod
    def _labels(candidates: List[Dict[str, Any]], label_field: Optional[str]) -> List[str]:
        """One display label per candidate, deduplicated: two offers both
        named "Base" become "Base" and "Base #2", so ranking entries and
        `best` stay unambiguous."""
        labels: List[str] = []
        seen: set = set()
        for index, candidate in enumerate(candidates):
            if label_field and candidate.get(label_field):
                label = str(candidate[label_field])
            else:
                label = f"#{index + 1}"
            if label in seen:
                suffix = 2
                while f"{label} #{suffix}" in seen:
                    suffix += 1
                label = f"{label} #{suffix}"
            seen.add(label)
            labels.append(label)
        return labels

    @staticmethod
    def _eval(expression: str, env: Dict[str, Any], where: str, label: str) -> Decimal:
        """safe_eval with arithmetic failures turned into the platform's
        structured error (a 0/0 from an all-zero comparison set must surface
        as a CalculationError, never a raw decimal exception)."""
        try:
            return safe_eval(expression, env)
        except (ArithmeticError, ValueError, KeyError) as e:
            raise StrategyExecutionError(
                f"cannot evaluate {where} for candidate {label!r}: {e}",
                details={"where": where, "candidate": label, "expression": expression},
            ) from e

    def _component_value(self, component: Dict[str, Any], env: Dict[str, Any], trail: AuditTrail, label: str) -> Decimal:
        name = component["name"]
        if "expression" in component:
            return self._eval(component["expression"], env, f"components.{name}", label)

        if "points" in component:
            earned = _ZERO
            positive_total = _ZERO
            for entry in component["points"]:
                points = Decimal(str(entry["points"]))
                if points > 0:
                    positive_total += points
                if env.get(entry["field"]):
                    earned += points
            scale_max = Decimal(str(component.get("scale_max", positive_total)))
            if scale_max <= 0:
                raise StrategyExecutionError(
                    f"component {name!r} has a non-positive scale_max",
                    details={"component": name, "scale_max": str(scale_max)},
                )
            return earned / scale_max * _HUNDRED

        if "rules" in component:
            value = Decimal(str(component.get("base", 100)))
            for rule in component["rules"]:
                try:
                    matched = _matches(rule["when"], env)
                except (ArithmeticError, ValueError) as e:
                    raise StrategyExecutionError(
                        f"cannot evaluate rule {rule['when']!r} in component {name!r} for {label!r}: {e}",
                        details={"component": name, "candidate": label, "rule": rule["when"]},
                    ) from e
                delta = _ZERO
                if matched:
                    if "points_per_unit" in rule:
                        units = Decimal(str(env.get(rule["when"]["field"], 0)))
                        delta = Decimal(str(rule["points_per_unit"])) * units
                    else:
                        delta = Decimal(str(rule["points"]))
                    value += delta
                trail.record(
                    "rule_evaluated",
                    candidate=label,
                    component=name,
                    condition=rule["when"],
                    matched=matched,
                    delta=str(delta),
                )
            return value

        raise StrategyExecutionError(
            f"component {name!r} declares none of relative_expression/expression/points/rules",
            details={"component": name},
        )


def _tie_tolerance(formula: Dict[str, Any]) -> Decimal:
    raw = formula.get("tie_tolerance")
    if raw is None:
        return DEFAULT_TIE_TOLERANCE
    try:
        value = Decimal(str(raw))
    except (InvalidOperation, ValueError, TypeError) as e:
        raise StrategyExecutionError(
            f"formula.tie_tolerance is not a number: {raw!r}",
            details={"tie_tolerance": raw},
        ) from e
    if not value.is_finite() or value < _ZERO:
        raise StrategyExecutionError(
            f"formula.tie_tolerance must be a finite non-negative number, got {raw!r}",
            details={"tie_tolerance": raw},
        )
    return value


def _cost_basis(formula: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Which derived quantity the cost component actually compares.

    Renderers must be able to lead with the money ("482,50 EUR/anno")
    before the synthetic 0-100 score, and they should not have to guess
    which of a pack's derived variables is the price. Reported only when
    the relative expression is a bare variable name, i.e. exactly when the
    value is already published per candidate under `derived`.
    """
    for component in formula.get("components") or []:
        if not isinstance(component, dict):
            continue
        expression = component.get("relative_expression")
        if not expression or component.get("direction") != "lower_is_better":
            continue
        variable = str(expression).strip()
        if not variable.isidentifier() or variable not in (formula.get("candidate_derived") or {}):
            continue
        return {"component": component["name"], "variable": variable}
    return None


def _ratio(provided: int, total: int) -> Decimal:
    if total <= 0:
        # Nothing is scored on caller-supplied data, so there is no
        # incompleteness to report — 1, not 0.
        return round_decimal(_ONE, _COMPLETENESS_PLACES)
    return round_decimal(Decimal(provided) / Decimal(total), _COMPLETENESS_PLACES)


def _default_affects_scoring(
    entry: Dict[str, Any],
    candidates_input: str,
    scoring_items: Iterable[str],
    scoring_scalars: Iterable[str],
) -> bool:
    path = str(entry.get("path", ""))
    match = _ITEM_PATH.match(path)
    if match and match.group("input") == candidates_input:
        return match.group("field") in set(scoring_items)
    if path == candidates_input:
        # The entire candidate list was defaulted, so every offer being
        # compared is an assumption. Nothing about this ranking is less
        # provisional than that.
        return True
    return path in set(scoring_scalars)
