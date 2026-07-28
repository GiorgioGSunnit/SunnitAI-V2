# Calculation Platform

A deterministic calculation engine controlled by an LLM. The LLM never does
math itself — it sends a `CalculationRequest` naming a calculator and its
inputs; Python validates, resolves date-versioned parameters, runs a
sandboxed calculation strategy, and returns a `CalculationResult` with the
number, every input/parameter used, step-by-step derivation, legal/official
citations, and warnings about what the calculation does *not* cover.

This is the repo's single calculation module: it superseded and absorbed
an earlier prototype (`src/calculator/`, since removed) — all of that
prototype's formulas, verified rates, and legal citations were ported
into the YAML formula packs here. Nothing here depends on `src/`.

The `simulation/` package beside `app/` is a deterministic stand-in for
the future LLM/router (planner, scripted extraction, conversation loop,
demo scenarios) — dev/test tooling only, never a production path.

## Install

This currently runs inside the parent repo's shared virtualenv (`.venv`),
which already has `fastapi`, `pydantic`, and `pyyaml`. To run it fully
standalone instead:

```bash
cd calculation_platform
pip install -e ".[dev]"
```

## Run the tests

From the repo root:

```bash
.venv/bin/python -m pytest calculation_platform/tests -v
```

## Verify parameter citations

Date-versioned parameter values can be stamped with `last_verified_at`
after their URL-backed citations have been fetched successfully:

```bash
.venv/bin/python calculation_platform/scripts/verify_citations.py --dry-run
.venv/bin/python calculation_platform/scripts/verify_citations.py
```

The verifier checks `parameters/**/*.yml` only. Entries whose citations do
not contain a URL are reported as skipped and are not stamped. At runtime,
official parameters resolved from the store produce warnings when they have
no verification stamp or the stamp is older than the engine's staleness
window.

## Start the API

From the repo root:

```bash
.venv/bin/uvicorn calculation_platform.app.main:app --reload --port 8802
```

Then visit `http://localhost:8802/docs` for interactive Swagger docs.

## Example API calls

```bash
curl http://localhost:8802/calculators

curl -X POST http://localhost:8802/calculate \
  -H "Content-Type: application/json" \
  -d '{
        "calculator_id": "legal_it.irpef",
        "inputs": {"taxable_income": 42000},
        "tax_year": 2026
      }'
```

Response (abridged):

```json
{
  "status": "success",
  "result": {"gross_tax": "11060.00"},
  "steps": [
    {"bracket_up_to": "28000", "rate": "0.23", "tax_in_bracket": "6440.00"},
    {"bracket_up_to": "50000", "rate": "0.33", "tax_in_bracket": "4620.00"}
  ],
  "citations": [
    {"reference": "Art. 11 D.P.R. 22 dicembre 1986, n. 917 (TUIR)", "official": true},
    {"reference": "Legge 30 dicembre 2025, n. 199 (Legge di Bilancio 2026)", "official": true}
  ],
  "warnings": [
    {"code": "definition", "message": "This calculates gross national IRPEF only."}
  ]
}
```

## Serialization contract (BREAKING CHANGE)

Every `Decimal` leaves the module as an **exact JSON string** (e.g.
`"11060.00"`, `"0.30"`), never a float — in `POST /calculate` responses,
`GET /calculations/*`, replay payloads, and reports. Dates serialize as
ISO-8601 strings. This preserves the engine's Decimal precision across the
serialization boundary: binary float noise (`0.1 + 0.2 ==
0.30000000000000004`) cannot appear in any machine-readable payload, and a
stored calculation replays to byte-identical values. There is no float
fallback mode; clients that previously parsed numbers must parse the
strings (with a decimal type, not a float). Human-readable report text may
format numbers for display; machine payloads always carry strings. The
conversion happens once, centrally, in
`app/core/result_builder.py::to_jsonable`.

## Architecture

```
request -> CalculatorRegistry.get(calculator_id)   # load YAML definition
        -> validate_inputs(definition, request)     # type-coerce, check required
        -> resolve_parameters(definition, request)   # caller value > date-versioned store > static default
        -> STRATEGIES[definition.strategy].run(...)  # the actual math
        -> CalculationResult                         # result + steps + citations + warnings
```

- **Schemas** (`app/schemas/`) — Pydantic models for the request, result,
  a calculator's YAML-loaded definition, a date-versioned parameter value,
  citations, and warnings.
- **Registry** (`app/core/registry.py`) — loads every `*.yml` under
  `formula_packs/` into a dict of `CalculatorDefinition`s at startup.
- **Parameter store** (`app/resolvers/parameter_store.py`) — loads every
  `*.yml` under `parameters/` into date-versioned values, with lookup by
  exact date, by tax year, or "every value overlapping this period" (used
  to split an interest calculation across a rate change). Also supports
  **monthly series** (entries keyed by `year`/`month` with `value` and
  `source_ref`, e.g. the ISTAT FOI index): `resolve_monthly(id, date)`
  returns the value for that date's calendar month, and
  `monthly_pair(id, a, b)` returns both months plus their ratio as a
  full-precision `Decimal`. A missing month fails loudly (naming parameter
  and month) — never interpolated or defaulted.
- **Safe evaluator** (`app/core/safe_evaluator.py`) — an AST-walking
  restricted arithmetic evaluator. Formulas are YAML data, not trusted
  code, so only numeric literals, variable lookups, arithmetic operators,
  and `min/max/abs/round/pow/sum` are allowed — no attribute access,
  imports, comprehensions, or calls to anything else.
- **Strategies** (`app/strategies/`) — pluggable calculation shapes:
  - `expression` — arbitrary arithmetic (invoice totals, VAT, loan
    payments), with an optional `zero_case` escape hatch for formulas that
    would divide by zero under some input.
  - `progressive_brackets` — tiered/progressive tax brackets (IRPEF).
  - `percentage_of_base` — `base * rate`, with an optional minimum and a
    `zero_if` escape hatch for an alternative regime that replaces the tax
    entirely (e.g. cedolare secca replacing registration tax).
  - `date_split_interest` — simple interest, automatically split into
    segments wherever the rate parameter's effective range changes within
    the requested period.
  - `decision_table` — minimal condition -> value lookup; not used by any
    current calculator, kept small on purpose, extend if a real need
    arises.
  - `table_lookup` — fixed amount from a value-banded table with optional
    exemption, indeterminable-value row, and categorical multiplier
    (contributo unificato).
  - `procedural_deadline` — Italian civil procedural deadline arithmetic:
    art. 155 c.p.c. day counting, giorni liberi, feriale suspension,
    holiday/Saturday rolling, backward terms (termini processuali).
  - `ravvedimento` — delay-tiered reduced penalty plus date-split legal
    interest on an omitted tax payment (ravvedimento operoso).
  - `foi_revaluation` — rivalutazione monetaria on the ISTAT FOI monthly
    series (`legal_it.rivalutazione_istat`). **Final-month convention**:
    coefficient = FOI(month of data_finale) / FOI(month of data_iniziale),
    computed in Decimal at full precision, quantized only at the end —
    verifiable against ISTAT's own calculator (rivaluta.istat.it).
  - `foi_revaluation_interest` — rivalutazione + interessi on debiti di
    valore per **Cass. SS.UU. 26/02/1995 n. 1712**
    (`legal_it.rivalutazione_interessi_1712`): calendar-year slices
    (partial first/last years pro rata by actual days, divisor 365/366),
    capital revalued on the FOI chain slice by slice, each year's legal
    interest computed on the **mean** between the capital at slice start
    and the revalued capital at slice end (criterio della media; a result
    warning names the criterion, since jurisprudential variants exist),
    interest accumulated separately and never compounded into the capital
    (no anatocismo). Outputs `capitale_rivalutato`, `interessi_totali`,
    `totale` as separate fields.
  - `dm55_fees` — compensi forensi per fase from the DM 55/2014 table
    (`legal_it.compensi_dm55`): sum of the requested fasi's valori medi
    for the scaglione of the case value, optional aumento (max 80%) or
    riduzione (max 50%, up to 70% for the sole fase istruttoria — bounds
    validated per art. 4 DM 55/2014), then the fixed chain **+15% rimborso
    spese generali (art. 2 DM 55/2014) → +4% CPA → +22% IVA**, with an
    explicit subtotal at every stage.
  - `comparator` — ranks N candidate offers held in a single `object_list`
    input against each other (`business.confronto_polizze`,
    `business.confronto_gas_luce`). See **Comparisons** below.
- **Engine** (`app/core/engine.py`) — the only file that knows the
  end-to-end flow above; it never contains formula logic itself.
- **API** (`app/api/routes.py`, `app/main.py`) — `GET /health`,
  `GET /calculators`, `POST /calculate`.
- **Simulation** (`simulation/`) — deterministic dev-only stand-in for
  the future LLM layer. Its `PlanResult` keeps raw `extracted_values`
  separate from `normalized_inputs`/`inputs`, so tests can verify messy
  user messages are safely converted into canonical `/calculate` requests
  without allowing the planner to compute final results.

## Comparisons

A comparator takes a list of candidate offers and returns a ranking with a
0-100 score per candidate. **The arithmetic being exact says nothing about
the scoring model being right**: the weights and point values in the two
shipped packs are still demonstrative and pending a business decision. The
result is therefore built to be read with that distinction visible — four
separate questions, answered separately:

| Question | Where the answer lives |
|---|---|
| Is the arithmetic correct? | `steps`, exact Decimal strings throughout |
| Is the input data complete? | `defaults_applied`, per-candidate `data_quality`, `comparison.scoring_completeness` |
| Is the scoring model sound? | `warnings`, `assumptions`, the pack's declared weights |
| Is there actually a winner? | `comparison.decision_status` |

### Relative cost normalization

A cost component is declared as a `relative_expression`, scored against the
**best** candidate rather than the worst:

```yaml
- name: punteggio_costo
  weight: "0.60"
  relative_expression: "costo_annuo_scontato"
  direction: lower_is_better       # or higher_is_better
  normalization: ratio_to_best
```

For `lower_is_better` + `ratio_to_best` the lowest strictly positive value
scores 100 and every other positive value scores `best / value * 100`. The
degenerate sets are resolved rather than divided by zero: if every value is
zero all candidates score 100; if the minimum is zero while others are
positive, the zero-cost candidates score 100 and the rest score 0. Results
are clamped to 0-100 and computed in exact `Decimal`.

This replaced `100 - value / worst * 100`, which anchored on the *worst*
candidate. That formula gave the most expensive offer a flat 0, rarely gave
the cheapest 100, and — the real defect — **rescored every candidate
whenever an irrelevant, more expensive offer was added to the set**, which
could reorder the offers the user actually cared about. Anchoring on the
best makes an added worse candidate provably irrelevant to everyone else.

Direction, normalization and the expression's variable references are all
validated at registry-load time, as is the requirement that a component
declare exactly one kind (`relative_expression` / `expression` / `points` /
`rules`).

### Tie semantics

Two totals within `formula.tie_tolerance` (default `0.50` of the 100-point
scale) are not a recommendation, they are noise from the configured
weights. The result reports:

```json
"comparison": {
  "decision_status": "clear_winner",
  "best_candidates": ["Offerta A"],
  "score_gap": "3.42",
  "tie_tolerance": "0.50",
  "provisional": false
}
```

`decision_status` is `clear_winner` or `effective_tie`; `best_candidates`
holds every candidate within tolerance of the top **exact** total. The tie
decision always uses exact totals — `score_gap` is displayed at the pack's
`round_to`, but a 0.5001 gap that prints as "0.50" is still a clear winner.
When the status is `effective_tie` the platform emits a warning and no
renderer may call any offer the best one. `ranking` and `best` are
unchanged and still present for backward compatibility.

### Provisional results, defaults and confirmation

`CalculationResult.defaults_applied` lists every default the platform
substituted, as machine-readable `{"path", "value"}` entries addressing the
input the way the request spells it (`polizze[0].franchigia`). Each ranking
entry carries a `data_quality` block separating `provided_fields` (the
caller stated it), `assumed_fields` (the platform defaulted it) and
`unknown_fields` (nobody knows) — an explicit `false` and a defaulted
`false` are different claims and are never flattened together.

A comparison is **provisional** when a default was applied to a field that
actually feeds a component. Fields declared but scored by nothing (the
insurance `massimale`) do not count against `scoring_completeness`.

`CalculationRequest.confirm_assumptions` lets a caller record that it has
seen those assumptions. It changes no number and removes no assumption:
`comparison.provisional_status` moves from `provisional_unconfirmed` to
`confirmed_with_assumptions`, and that is all it does.

### Incremental candidate collection

In production (`src/rag/calculation.py`) an `object_list` comparator is
filled one candidate per turn through explicit phases —
`collect_shared_inputs` → `collect_candidates` → `review` → `confirm` —
persisted in `pending_calculation` through `src/chatbot/session.py`. The
array is state owned by the route, not something the LLM rebuilds each
turn: prior candidates are kept verbatim, an offer restated by its label
corrects that offer instead of duplicating it, `remove <name>` drops one,
and at most 20 candidates are accepted. Ordinary candidate turns do not
count against the three-round clarification limit. When LLM extraction is
unavailable the route asks for a deterministic `field: value` form rather
than assembling an offer out of stray numbers.

### Known omissions in the shipped packs

These are declared in each pack's `exclusions` and surfaced in
`/calculate`, stored calculations, replays, HTML reports and both chat
renderers:

- **Insurance** — `massimale` is collected but not scored (no verified
  scale exists to convert it to points). The applicant's age and claims
  history are collected, validated and audited but deliberately not scored:
  identical across every quote, they can only shift all totals equally.
- **Energy** — the yearly cost is the commercial energy quota plus the
  declared fixed fee only. No VAT, no excise duties, no ARERA system
  charges, no transport/distribution/metering, no F1/F2/F3 time bands, and
  no price changes after the first year. None of these were added because
  the platform has no verified parameters for them, and inventing them
  would produce a confident number that is wrong.

## How to add a new calculator

1. If an existing strategy fits (most formulas do), write a new YAML file
   under `formula_packs/<domain>/<name>.yml` declaring `inputs`,
   `parameters`, `derived_variables`/`formula`, `output`, `citations`, and
   `warnings`. No Python changes needed.
2. If it needs date-versioned values (a rate, a bracket table, a
   threshold), add them to a YAML file under `parameters/<domain>/`, each
   entry with `effective_from`/`effective_to` and a `source`.
3. If no existing strategy fits the calculation's *shape* (not just its
   numbers), add a new class in `app/strategies/`, subclassing
   `CalculationStrategy`, and register it in `app/strategies/__init__.py`'s
   `STRATEGIES` dict. Existing calculators and the engine are unaffected.
4. Add a test under `tests/` exercising it through `engine.calculate(...)`.

## Placeholder data and verification

Some parameter tables ship as **schema examples with synthetic placeholder
values** (`placeholder: true`, `verified: false`): the FOI monthly indices
(`parameters/legal_it/foi_indices.yml` — only 4 synthetic months) and the
DM 55 fee table (`parameters/legal_it/dm55_compensi.yml` — one scaglione
populated synthetically, the rest are `valore_medio: null` stubs that fail
loudly). Results computed from placeholders carry an explicit warning and
are not usable operationally. `docs/TO_VERIFY.md` is the single checklist
of every value a human must verify against official sources before real
use, including the golden-test expected values (`verified_against: TODO`).

## Known simplifications (by design, for this MVP)

- No authentication — `tenant_id` exists on the request schema but is not
  enforced anywhere.
- Parameters are loaded from YAML files into memory at startup; swapping
  to PostgreSQL later means implementing the same `ParameterStore`
  interface (`resolve_by_date`, `resolve_by_tax_year`,
  `all_effective_ranges`) against a database instead of YAML files.
- No official-source ingestion/scraping — every parameter value in
  `parameters/` is still hand-entered and cited. The citation verifier only
  checks that URL-backed sources are reachable and stamps
  `last_verified_at`; it does not parse legal text or update rates.
- `decision_table` is intentionally minimal (dict equality conditions
  only); extend it if a calculator needs real branching logic.
