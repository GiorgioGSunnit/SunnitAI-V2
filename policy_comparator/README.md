# Policy Comparator

Internal staff tool for requesting Italian auto-insurance quotations from
several providers at once and comparing what comes back.

A staff member enters four fields, picks the companies to interpellate, and
records the customer's consent. The providers are contacted in parallel; each
one that needs more information says so, the UI asks for the gaps once (not
once per provider), and the affected providers are resumed. The results screen
shows a normalized side-by-side comparison and names the **lowest-priced quote
that satisfies the customer's stated requirements** — never a "best" policy.

> **Out of the box this application is a demonstration.** Every provider runs on
> deterministic mock data and nothing leaves the machine. See
> [Production readiness](#production-readiness) for what is required before any
> real provider is contacted.

This sub-project is standalone: it does not import the parent repo's `src/`
package, owns its own database tables (all prefixed `pc_`), and has its own
Alembic history. It is unrelated to `calculation_platform/`, and it does not
use or re-enable `business.confronto_polizze`.

---

## Quick start

```bash
pip install -e "policy_comparator[dev]"
python -m policy_comparator.cli demo
```

That creates a SQLite database and an admin login
(`staff@example.com` / `demo-password`). Then, in two terminals:

```bash
uvicorn policy_comparator.api.app:app --port 8100
```

```bash
python -m policy_comparator.worker
```

Open <http://localhost:8100>. Both processes are required: the API never
contacts a provider itself, so without the worker a request will sit at
"waiting" forever.

### Running the tests

```bash
pytest policy_comparator
```

212 tests, ~7 seconds, no network access. Tests pin their own environment in
`conftest.py`: a throwaway SQLite file, every provider forced to mock mode, and
`LIVE_PROVIDER_AUTOMATION=false` regardless of your shell.

---

## Architecture

```
policy_comparator/
├── api/                 FastAPI routes, auth dependencies, tenant scoping
├── models/              SQLAlchemy tables (pc_*), enums
├── schemas/             Pydantic contracts (profile, quotes, API bodies)
├── providers/           One module per provider + the adapter contract
├── services/            Orchestration, normalization, ranking, dedup, audit
├── worker/              The background process that actually calls providers
├── frontend/            Dependency-free SPA served by the API
├── migrations/          Alembic
└── tests/
```

**The API never calls a provider.** A quotation request is fanned out into one
row per provider in `pc_quote_jobs`, and a separate worker process claims them.
That is not an optimization: a portal round trip takes tens of seconds, and
browser automation cannot run inside a request thread at all.

**The queue is a database table.** A worker killed mid-flight loses nothing —
its lease expires and the next worker reclaims the job. Each job carries an
idempotency key derived from `(request, provider, attempt round)`, so a replayed
submission is recognisable to the provider as the same one.

**Providers are isolated.** Everything specific to a provider — URLs, selectors,
payload mappings, which insurers it can quote — lives in that provider's module.
An exception inside one adapter becomes that adapter's own failed result; the
others in the batch are unaffected (there is a test for exactly this).

**Money is `Decimal` end to end.** The `Money` column type stores a decimal
string and *raises* if handed a float; the Pydantic schemas reject float input.
Prices never touch binary floating point.

### Request lifecycle

```
draft ──start──▶ running ──▶ awaiting_information ──answers──▶ running
                    │                                             │
                    └──────────▶ completed / partially_completed / failed
```

A provider attempt ends in exactly one outcome: `quoted`,
`missing_information`, `unavailable`, `timed_out`, `manual_action_required`,
`authentication_required`, `configuration_error`, or `failed`. Only the
transient three (`unavailable`, `timed_out`, `failed`) are retried
automatically or count against the circuit breaker — retrying a CAPTCHA or a
bad credential just burns a call.

### Frontend choice

Plain HTML, CSS and JavaScript, served as static files by the same FastAPI app.
No build step, no `npm install`, no bundler — which is what keeps the whole
tool a `pip install` plus two commands, and makes it easy to hand to somebody
who just wants to run it on their laptop. Same-origin serving also means there
is no CORS surface. If this grows into a product with several teams touching
it, a framework becomes worth the toolchain; today it would only add setup.

The interface is an original design and deliberately does not resemble
Facile.it, Segugio.it, CercAssicurazioni or any insurer's visual identity.

---

## Providers

| Provider | Type | Mock mode | Live integration |
|---|---|---|---|
| Zurich | Insurer | ✅ complete | ⚠️ scaffolding only — unverified |
| Allianz | Insurer | ✅ complete | ⚠️ scaffolding only — unverified |
| Generali | Insurer | ✅ complete | ⚠️ scaffolding only — unverified |
| CercAssicurazioni | **Aggregator** | ✅ complete | ⚠️ scaffolding only — unverified |

Initial fast-quote fields, mirroring each provider's public form:

- **Zurich** — plate, owner date of birth
- **Allianz** — plate, owner date of birth, email (mobile requested later)
- **Generali** — plate, plus date of birth *or* tax code (either identifies the
  policyholder; the adapter does not demand both)
- **CercAssicurazioni** — plate, owner date of birth

### The aggregator is not an insurer

CercAssicurazioni relays quotes issued by other companies. Two consequences run
through the whole application:

- a quote's `insurer_name` is the company carrying the risk, and
  `source_channel` records that it arrived via an aggregator;
- the same offer can arrive twice — once direct, once relayed. Duplicates are
  detected by insurer + product + quote reference + price + configuration, and
  are **linked, never deleted**. The direct copy becomes the primary (it carries
  the purchase link the customer will use); the relayed copy stays visible,
  marked as a duplicate, so both channels remain auditable.

---

## How a quote is chosen

Eligibility is decided **before** price is considered. A quote is eligible only
if it satisfies every requirement the staff member recorded: minimum liability
limits, maximum deductible, driving formula, required guarantees actually
included in the premium, and any restriction the customer refused (telematics
box, approved repair network).

A quote missing the information needed to *prove* it meets a requirement is
excluded with a reason, not given the benefit of the doubt. Recommending a
policy whose deductible we do not know would be worse than showing the gap.

Among eligible quotes, in order: **lowest annual total**, then lower
deductible, then higher liability limits, then fewer mandatory restrictions.

The results payload always contains the ineligible quotes with their specific
reasons and every provider that produced nothing. Results are never narrowed to
the providers that happened to work.

There is deliberately no 0–100 composite score.

---

## Why the demonstration prices are what they are

Every premium in mock mode is produced by one function,
`mock_engine.compute_offer_pricing`, which returns the price **and** the trail
that produced it. There is one definition of the formula, so the number and its
published explanation cannot drift apart — a test replays the published steps
and asserts they land on the same premium the user is shown.

The formula:

```
310,00 €  base
  × insurer factor
  × merit-class factor        1,00 + (CU − 1) × 0,055
  × age factor                banded by the owner's age at the policy start
  × vehicle-power factor      banded by kW
  × province factor           a short table, default 1,00
  × claims factor             1,00 + (at-fault claims × 0,11)
  × driving-formula factor    exclusive 0,94 · expert 0,97 · free 1,00
  × demonstration adjustment  ±4%, derived from the plate and the insurer
  = RC base                   rounded to 2 decimals, ROUND_HALF_UP
  + each requested optional guarantee (catalogue price × insurer factor)
  = annual total              rounded to 2 decimals, ROUND_HALF_UP
```

Worked example — Zurich, plate `AB123CD`, born 1985-03-04, 51 kW, province RM,
CU 3, no claims, guida esperta, policy starting 2026-09-01:

| Step | Factor | Running |
|---|---|---|
| Base premium | — | 310,00 |
| Insurer (Zurich) | × 1,00 | 310,00 |
| Merit class CU 3 | × 1,110 | 344,10 |
| Age 41 | × 1,00 | 344,10 |
| Power 51 kW | × 0,92 | 316,57 |
| Province RM | × 1,16 | 367,22 |
| Claims 0 | × 1,00 | 367,22 |
| Guida esperta | × 0,97 | 356,21 |
| Demonstration adjustment | × 0,961 | 342,31 |
| **RC base, rounded** | | **342,31 €** |
| Roadside assistance (bundled) | + 0,00 | 342,31 |
| **Annual total** | | **342,31 €** |

Intermediate values are shown rounded for readability; the arithmetic runs in
exact `Decimal` throughout and rounds only at the two marked points.

The UI exposes this under **“Come è stato calcolato questo importo”** on the
recommended quote, labelled as demonstration logic.

Quotes carry a `calculation_source`:

- `demonstration_formula` — this application computed the price, and the full
  breakdown is attached;
- `provider_supplied` — an insurer quoted the price. **No breakdown is ever
  shown**, and a live payload claiming to carry our formula is discarded rather
  than trusted. Reverse-engineering an underwriting model from one quoted
  number would be fabrication, not transparency.

---

## Why real insurer APIs cannot simply be called

The provider modules contain route paths, request bodies and response mappings.
**They are unverified placeholders.** They were written against no
documentation, they have never been executed against a real endpoint, and they
are marked `UNVERIFIED` in the source. Treat them as a shape to fill in, not as
an integration.

**A public quotation website is not a public API.** That a company operates a
form at a known URL says nothing about whether we are permitted to send it
traffic programmatically. Quoting engines are commercial infrastructure: they
are rate-limited, contractually governed, and priced per call. Using one without
an agreement is unauthorized use of somebody else's system regardless of how
easy it is technically, and it also puts *customer* personal data into a channel
nobody has approved.

Before a single provider is switched out of mock mode, all of the following
must exist:

1. **A commercial agreement** — partner, broker or agency — permitting
   programmatic quotation requests. Nothing else on this list matters without it.
2. **Official API documentation** from the provider.
3. **Authorized credentials** issued to us, delivered through a secret manager,
   never committed.
4. **Approved customer-data usage**: a lawful basis and a consent record
   covering transmission to that specific provider, plus the provider's own
   data-processing terms.
5. **Verified request and response mappings**, checked field by field against
   the provider's spec — not inferred from a sample.
6. **Sandbox certification** where the provider offers one, completed before
   production access.
7. **Contract tests** against the provider's sandbox, run on a schedule, so a
   breaking change is detected by us rather than by a member of staff mid-quote.
8. **Operational monitoring**: success rate, latency, error categories, and an
   alert when the circuit breaker opens.
9. **Legal and privacy review** of the flow, the retention period and the
   consent wording.

Two hard rules:

- **Browser automation must never bypass a protection.** CAPTCHA, MFA, rate
  limits, bot detection and session walls are refusals. The adapter returns
  `manual_action_required` and a human takes over. There is no solver, no
  stealth patching, no retry-until-it-passes.
- **Knowing a provider's website is not a reason to contact it.** The
  application will not send a request to a provider that is not configured,
  authorized and explicitly enabled — see the five-condition gate below.

**Integrate one provider end to end first.** Get a single insurer through
agreement, sandbox, certification, monitoring and a period of real use before
enabling a second. Four half-finished integrations are harder to debug and
easier to get wrong than one that works.

---

## Configuring a provider for live use

Copy `.env.example` to `.env`. Reaching a real provider requires **all five**:

1. the provider is configured (`PC_PROVIDER_<ID>_MODE` = `api` or `browser`,
   with the matching URL);
2. the provider is marked `PC_PROVIDER_<ID>_AUTHORIZED=true` — meaning the
   client actually has an authorized relationship with them;
3. `LIVE_PROVIDER_AUTOMATION=true`;
4. a valid consent record covering that provider exists for the request;
5. a staff user explicitly started the request.

Miss any one and the attempt reports `configuration_error` rather than quietly
falling back to demonstration data that looks like a real quote.

Credentials are referenced by the **name** of the environment variable holding
them (`PC_PROVIDER_ZURICH_API_KEY_ENV=ZURICH_API_KEY`), never by value, so
configuration can be dumped or logged without leaking a secret.

Integration priority is fixed: **official/partner API → authorized broker or
agent portal → browser automation**, and only where no API exists and the
client is authorized to automate that portal.

### Anti-bot protection is never bypassed

If a CAPTCHA, an MFA prompt or an unexpected login wall appears, the attempt
ends as `manual_action_required` and a human takes over. There is no solver, no
stealth patching, no retry-until-it-passes. (During development, the public
sites in scope did serve bot protection — expect this outcome in practice.)

### Adding another provider

1. Create `providers/<name>.py` with a `StandardAutoAdapter` subclass: set
   `provider_id`, `display_name`, `provider_type`, `required_paths`,
   `second_stage_paths`, and either `api_contract()` or `browser_flow()`.
2. Register it in `providers/registry.py` — one line.
3. Add its `PC_PROVIDER_<ID>_*` block to `.env.example`.
4. If it needs a field the catalogue doesn't have, add a `FieldSpec` to
   `services/field_catalog.py`. Do not invent labels in the adapter: shared
   wording in the catalogue is what makes the same question from two providers
   deduplicate into one on the missing-information screen.

Nothing else in the application refers to a provider by name.

---

## Security and privacy

- Every endpoint is authenticated; every query is tenant-scoped in one place
  (`api/deps.py`). Another tenant's request returns **404**, not 403 —
  confirming that an id exists elsewhere is itself a leak.
- Tokens are HS256 JWTs with the same claims the parent platform issues, so a
  token minted there works here. `pc_staff_users` exists so the tool also runs
  standalone.
- Sensitive columns (email, tax code, phone, address, names) are encrypted at
  rest with Fernet. Emails carry an HMAC blind index so a customer can be found
  without decrypting every row.
- **Production refuses to start without `PC_ENCRYPTION_KEY`.** There is no
  production fallback key. Development derives a clearly-labelled local key so
  a fresh checkout works with no setup.
- No personal data in URLs or query strings — it travels in request bodies.
- Application logs are scrubbed (tax codes, emails, phone numbers, plates), and
  audit metadata records field *names*, never the values staff typed.
- Processing consent and marketing consent are separate records. Declining
  marketing never blocks a quotation.
- Failure screenshots are only written when diagnostics are enabled; full page
  HTML requires `PC_STORE_RAW_PAGES=true` explicitly, because a rendered quote
  page contains the customer's data.
- **Outbound links are validated server-side.** Only absolute `https` URLs are
  returned to the browser; `javascript:`, `data:`, `file:` and plain `http:`
  URLs are dropped rather than repaired, so a provider response cannot become a
  script-injection vector through an `href`. Real purchase links open with
  `rel="noopener noreferrer"`.
- Demonstration purchase links point at a non-existent host. The UI intercepts
  the click and explains that no purchase is possible in demo mode rather than
  navigating to a dead page.
- The frontend writes every API-supplied value with `textContent`. Insurer
  names, product names and calculation labels all arrive over the wire, and
  none of them is ever treated as markup.
- Quotation submissions are rate-limited per tenant (in-process; move to Redis
  for a multi-process rollout).

---

## Production readiness

Ready:

- [x] Orchestration, retry, backoff, circuit breaker, restart/resume
- [x] Persistent job queue with idempotency keys
- [x] Progressive profile with staff/provider provenance tracking
- [x] Normalization, deduplication, eligibility and ranking
- [x] Tenant isolation, encryption at rest, audit trail, PII-safe logging
- [x] Alembic migration verified to reproduce the model schema exactly
- [x] SQLite for local work, PostgreSQL-compatible

Required before real quotations:

- [ ] **A commercial agreement with each provider.** Nothing else on this list
      matters without it.
- [ ] **Verify every API mapping.** The request/response mappings in each
      provider module are placeholders marked `UNVERIFIED`, written against no
      documentation. They must be checked against the provider's real API spec.
- [ ] **Verify every portal selector**, and implement the page extractors —
      `_extract_from_page` deliberately returns nothing rather than scraping
      against unverified selectors and producing silently wrong prices.
- [ ] Install the browser extra where portal automation is used:
      `pip install -e ".[browser]" && playwright install chromium`
- [ ] Set `PC_ENCRYPTION_KEY` and a real `PC_JWT_SECRET_KEY`
- [ ] Point `PC_DATABASE_URL` at PostgreSQL and run
      `alembic -c policy_comparator/alembic.ini upgrade head`
- [ ] Move rate limiting to shared storage if running multiple API processes
- [ ] Implement the retention job for `PC_DATA_RETENTION_DAYS` (the setting is
      read but no purge job runs yet)
- [ ] Legal review of consent wording and IVASS/IDD distribution obligations

### Known limitations

- **No live provider integration has been verified.** Mock mode is complete and
  deterministic; every live path is scaffolding.
- The public sites in scope are protected by anti-bot systems, so portal
  automation will realistically return `manual_action_required` until an
  authorized API or partner channel is available.
- Mock premiums are invented. They respond plausibly to merit class, age,
  province, power and claims, but they are not real prices and every quote is
  flagged `is_demonstration` and badged in the UI.
- Progress is polled every two seconds rather than pushed.
- Rate limiting is per-process.
