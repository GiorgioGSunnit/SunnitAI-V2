"""Deterministic demonstration quotes.

Everything here is invented. It exists so the whole application — orchestration,
missing-information round trips, deduplication, ranking, the UI — can be run and
tested end to end without contacting any insurer.

Two properties matter:

*Deterministic.* The same profile always produces the same premium, so tests can
assert on exact amounts and a demo is reproducible.

*Consistent across channels.* An offer is generated from the *insurer*, not from
the adapter that fetched it, so the same underlying offer reached directly and
through the aggregator carries the same reference and price — which is what
makes deduplication observable rather than theoretical.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from ..schemas.profile import QuotationProfile

CENT = Decimal("0.01")

#: Standard Italian minimum RC limits.
_STANDARD_LIMIT_PEOPLE = Decimal("6450000")
_STANDARD_LIMIT_PROPERTY = Decimal("1300000")


@dataclass(frozen=True)
class InsurerProfile:
    """A fictional pricing personality for one insurer."""

    key: str
    name: str
    product_name: str
    multiplier: Decimal
    deductible: Decimal
    limit_people: Decimal = _STANDARD_LIMIT_PEOPLE
    limit_property: Decimal = _STANDARD_LIMIT_PROPERTY
    requires_black_box: bool = False
    requires_approved_repair_network: bool = False
    #: Optional guarantees the insurer bundles into the base premium.
    bundled_covers: tuple[str, ...] = ()
    instalment_count: int = 2
    #: Surcharge applied to the annual premium when paying in instalments.
    instalment_surcharge: Decimal = Decimal("0.04")


INSURERS: dict[str, InsurerProfile] = {
    "zurich": InsurerProfile(
        key="zurich",
        name="Zurich",
        product_name="Zurich Connect Auto",
        multiplier=Decimal("1.00"),
        deductible=Decimal("300"),
        limit_people=Decimal("10000000"),
        limit_property=Decimal("2500000"),
        bundled_covers=("assistenza_stradale",),
    ),
    "allianz": InsurerProfile(
        key="allianz",
        name="Allianz",
        product_name="Allianz Ultra Auto",
        multiplier=Decimal("1.08"),
        deductible=Decimal("250"),
        limit_people=Decimal("10000000"),
        limit_property=Decimal("2500000"),
        bundled_covers=("assistenza_stradale", "tutela_legale"),
        requires_approved_repair_network=True,
    ),
    "generali": InsurerProfile(
        key="generali",
        name="Generali",
        product_name="Immagina Strade Nuove",
        multiplier=Decimal("0.96"),
        deductible=Decimal("400"),
        bundled_covers=("assistenza_stradale",),
    ),
    "genertel": InsurerProfile(
        key="genertel",
        name="Genertel",
        product_name="Genertel Auto Facile",
        multiplier=Decimal("0.91"),
        deductible=Decimal("500"),
    ),
    "conte": InsurerProfile(
        key="conte",
        name="ConTe.it",
        product_name="ConTe.it Auto Smart",
        multiplier=Decimal("0.84"),
        deductible=Decimal("500"),
        # The cheapest offer in the set, but only with a telematics box. If the
        # customer refuses one it becomes ineligible — which is exactly the case
        # the ranking service has to get right.
        requires_black_box=True,
    ),
}

#: Deterministic prices for guarantees sold on top of the base premium.
_OPTIONAL_COVER_CATALOGUE: dict[str, tuple[str, Decimal]] = {
    "furto_incendio": ("Furto e incendio", Decimal("148.00")),
    "kasko": ("Kasko completa", Decimal("395.00")),
    "mini_kasko": ("Mini kasko", Decimal("172.00")),
    "cristalli": ("Cristalli", Decimal("54.00")),
    "atti_vandalici": ("Eventi socio-politici e atti vandalici", Decimal("61.00")),
    "eventi_naturali": ("Eventi naturali", Decimal("58.00")),
    "assistenza_stradale": ("Assistenza stradale", Decimal("39.00")),
    "tutela_legale": ("Tutela legale", Decimal("42.00")),
    "infortuni_conducente": ("Infortuni del conducente", Decimal("67.00")),
}

#: A few higher-risk provinces, for a visible but obviously fictional effect.
_PROVINCE_FACTORS: dict[str, Decimal] = {
    "NA": Decimal("1.34"),
    "CE": Decimal("1.28"),
    "RM": Decimal("1.16"),
    "MI": Decimal("1.11"),
    "TO": Decimal("1.07"),
    "FI": Decimal("1.02"),
    "BO": Decimal("1.00"),
    "AO": Decimal("0.88"),
}


def _seed(*parts: Any) -> int:
    joined = "|".join("" if p is None else str(p) for p in parts)
    return int(hashlib.sha256(joined.encode()).hexdigest()[:12], 16)


def _money(value: Decimal) -> Decimal:
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


def _age_on(dob: date | None, at: date) -> int:
    if dob is None:
        return 45
    years = at.year - dob.year - ((at.month, at.day) < (dob.month, dob.day))
    return max(18, min(99, years))


def _age_factor(age: int) -> Decimal:
    if age < 23:
        return Decimal("1.65")
    if age < 27:
        return Decimal("1.34")
    if age < 35:
        return Decimal("1.08")
    if age < 60:
        return Decimal("1.00")
    if age < 72:
        return Decimal("1.06")
    return Decimal("1.19")


def _merit_factor(cu: int | None) -> Decimal:
    """Bonus/malus. Class 1 is the best, 18 the worst."""
    effective = 14 if cu is None else cu
    return Decimal("1.00") + (Decimal(effective - 1) * Decimal("0.055"))


def _power_factor(power_kw: int | None) -> Decimal:
    if not power_kw:
        return Decimal("1.00")
    if power_kw <= 55:
        return Decimal("0.92")
    if power_kw <= 85:
        return Decimal("1.00")
    if power_kw <= 125:
        return Decimal("1.12")
    return Decimal("1.28")


def quote_reference(insurer_key: str, profile: QuotationProfile) -> str:
    """Stable per (insurer, vehicle, start date), independent of the channel.

    Both the direct adapter and the aggregator derive the reference the same
    way, so the two copies of one offer are recognisably the same offer.
    """
    token = _seed(insurer_key, profile.vehicle.plate, profile.policy_start_date)
    return f"{insurer_key.upper()[:3]}-{token % 10**8:08d}"


#: How the demonstration premium is rounded, stated once and shown to the user.
ROUNDING_NOTE = (
    "I fattori vengono moltiplicati in aritmetica decimale esatta. "
    "L'arrotondamento a 2 decimali (ROUND_HALF_UP) viene applicato una sola volta "
    "al risultato della RC base, e poi al totale annuo."
)

#: Marks a quote whose price this application computed itself.
CALCULATION_SOURCE_DEMONSTRATION = "demonstration_formula"
#: Marks a quote whose price came from the insurer. Never carries a formula.
CALCULATION_SOURCE_PROVIDER = "provider_supplied"


@dataclass(frozen=True)
class Step:
    """One auditable line of the demonstration calculation.

    ``running`` is the exact unrounded value at that point, so a test can
    replay the steps and land on the same premium the customer is shown.
    """

    code: str
    label: str
    #: base | factor | rounding | addition | subtotal | total
    kind: str
    factor: Decimal | None = None
    value: Decimal | None = None
    running: Decimal | None = None
    detail: str | None = None

    def as_payload(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "label": self.label,
            "kind": self.kind,
            "factor": str(self.factor) if self.factor is not None else None,
            "value": str(self.value) if self.value is not None else None,
            "running": str(self.running) if self.running is not None else None,
            "detail": self.detail,
        }


@dataclass
class OfferPricing:
    """The complete priced offer, plus the trail that produced it."""

    base_rc_total: Decimal
    optional_total: Decimal
    annual_total: Decimal
    instalment_count: int
    instalment_amount: Decimal
    instalment_total: Decimal
    coverages: list[dict[str, Any]]
    steps: list[Step]

    def breakdown_payload(self) -> dict[str, Any]:
        return {
            "source": CALCULATION_SOURCE_DEMONSTRATION,
            "currency": "EUR",
            "rounding": ROUNDING_NOTE,
            "annual_total": str(self.annual_total),
            "steps": [step.as_payload() for step in self.steps],
        }


def compute_offer_pricing(
    insurer: InsurerProfile, profile: QuotationProfile
) -> OfferPricing:
    """Price one demonstration offer and record every step that produced it.

    This is the single definition of the demonstration formula: the premium and
    its published breakdown come from the same arithmetic, so the two can never
    drift apart.
    """
    steps: list[Step] = []
    running = Decimal("310.00")
    steps.append(
        Step(
            code="base_premium",
            label="Premio base dimostrativo",
            kind="base",
            value=running,
            running=running,
            detail="Punto di partenza convenzionale della formula dimostrativa.",
        )
    )

    def apply(code: str, label: str, factor: Decimal, detail: str | None = None) -> None:
        nonlocal running
        running = running * factor
        steps.append(
            Step(code=code, label=label, kind="factor", factor=factor, running=running, detail=detail)
        )

    age = _age_on(profile.customer.owner_date_of_birth, profile.policy_start_date)
    province = (profile.customer.province or "").upper()
    merit_class = profile.history.universal_merit_class
    claims = profile.history.claims_full_responsibility or 0
    formula = profile.preferences.driving_formula

    apply(
        "insurer_multiplier",
        f"Coefficiente compagnia ({insurer.name})",
        insurer.multiplier,
    )
    apply(
        "merit_class",
        "Classe di merito universale (CU)",
        _merit_factor(merit_class),
        f"CU {merit_class if merit_class is not None else 14}"
        + ("" if merit_class is not None else " (valore predefinito: non dichiarata)"),
    )
    apply("driver_age", "Età del proprietario", _age_factor(age), f"{age} anni")
    apply(
        "vehicle_power",
        "Potenza del veicolo",
        _power_factor(profile.vehicle.power_kw),
        f"{profile.vehicle.power_kw} kW" if profile.vehicle.power_kw else "Potenza non dichiarata",
    )
    apply(
        "province",
        "Provincia di residenza",
        _PROVINCE_FACTORS.get(province, Decimal("1.00")),
        province or "Provincia non dichiarata",
    )
    apply(
        "claims",
        "Sinistri con responsabilità principale",
        Decimal("1.00") + (Decimal(claims) * Decimal("0.11")),
        f"{claims} sinistri negli ultimi 5 anni",
    )

    formula_factor = {"exclusive": Decimal("0.94"), "expert": Decimal("0.97")}.get(
        formula or "", Decimal("1.00")
    )
    apply(
        "driving_formula",
        "Formula di guida",
        formula_factor,
        {"exclusive": "Guida esclusiva", "expert": "Guida esperta"}.get(
            formula or "", "Guida libera"
        ),
    )

    # Deterministic ±4% spread so premiums are not suspiciously round.
    jitter = Decimal(_seed(insurer.key, profile.vehicle.plate) % 81 - 40) / Decimal("1000")
    apply(
        "demo_adjustment",
        "Scostamento dimostrativo deterministico",
        Decimal("1.00") + jitter,
        "Derivato dalla targa e dalla compagnia: identico a parità di dati, "
        "serve solo a rendere i prezzi dimostrativi non uniformi.",
    )

    base_rc_total = _money(running)
    steps.append(
        Step(
            code="base_rc_total",
            label="RC base arrotondata",
            kind="rounding",
            value=base_rc_total,
            running=base_rc_total,
            detail="Arrotondamento a 2 decimali (ROUND_HALF_UP).",
        )
    )

    coverages, optional_total, optional_steps = _price_coverages(insurer, profile)
    steps.extend(optional_steps)

    annual_total = _money(base_rc_total + optional_total)
    steps.append(
        Step(
            code="annual_total",
            label="Premio annuo totale",
            kind="total",
            value=annual_total,
            running=annual_total,
            detail="RC base più le garanzie accessorie incluse nel premio.",
        )
    )

    instalment_total = _money(annual_total * (Decimal("1.00") + insurer.instalment_surcharge))
    instalment_amount = _money(instalment_total / Decimal(insurer.instalment_count))
    steps.append(
        Step(
            code="instalment_surcharge",
            label="Maggiorazione per pagamento rateale",
            kind="factor",
            factor=Decimal("1.00") + insurer.instalment_surcharge,
            value=instalment_total,
            running=instalment_total,
            detail=f"Costo totale se pagato in {insurer.instalment_count} rate.",
        )
    )
    steps.append(
        Step(
            code="instalment_amount",
            label=f"Importo di ciascuna delle {insurer.instalment_count} rate",
            kind="subtotal",
            value=instalment_amount,
            detail="Totale rateale diviso per il numero di rate.",
        )
    )

    return OfferPricing(
        base_rc_total=base_rc_total,
        optional_total=optional_total,
        annual_total=annual_total,
        instalment_count=insurer.instalment_count,
        instalment_amount=instalment_amount,
        instalment_total=instalment_total,
        coverages=coverages,
        steps=steps,
    )


def _price_coverages(
    insurer: InsurerProfile, profile: QuotationProfile
) -> tuple[list[dict[str, Any]], Decimal, list[Step]]:
    """Price the guarantees and record the ones that add to the premium."""
    requested = list(profile.preferences.required_optional_covers or [])
    coverages: list[dict[str, Any]] = [
        {
            "code": "rc_auto",
            "label": "Responsabilità civile auto",
            "included": True,
            "limit_amount": str(insurer.limit_people),
        }
    ]

    optional_total = Decimal("0")
    steps: list[Step] = []

    for code, (label, price) in _OPTIONAL_COVER_CATALOGUE.items():
        bundled = code in insurer.bundled_covers
        wanted = code in requested
        # Deterministic per-insurer discount on add-ons.
        adjusted = _money(price * insurer.multiplier)

        if bundled or wanted:
            coverages.append(
                {
                    "code": code,
                    "label": label,
                    "included": True,
                    "price": "0.00" if bundled else str(adjusted),
                    "deductible": None if bundled else str(insurer.deductible),
                }
            )
            if bundled:
                steps.append(
                    Step(
                        code=f"cover_{code}",
                        label=f"Garanzia inclusa: {label}",
                        kind="addition",
                        value=Decimal("0.00"),
                        detail="Compresa nel premio base dalla compagnia.",
                    )
                )
            else:
                optional_total += adjusted
                steps.append(
                    Step(
                        code=f"cover_{code}",
                        label=f"Garanzia richiesta: {label}",
                        kind="addition",
                        value=adjusted,
                        detail="Richiesta dal cliente e aggiunta al premio.",
                    )
                )
        else:
            coverages.append(
                {
                    "code": code,
                    "label": label,
                    "included": False,
                    "price": str(adjusted),
                    "deductible": str(insurer.deductible),
                }
            )

    steps.append(
        Step(
            code="optional_subtotal",
            label="Totale garanzie accessorie a pagamento",
            kind="subtotal",
            value=optional_total,
            detail="Somma delle sole garanzie che incidono sul premio.",
        )
    )
    return coverages, optional_total, steps


def annual_premium(insurer: InsurerProfile, profile: QuotationProfile) -> Decimal:
    """The RC base premium, before optional guarantees.

    Thin wrapper over :func:`compute_offer_pricing` so there is exactly one
    definition of the formula.
    """
    return compute_offer_pricing(insurer, profile).base_rc_total


def build_offer(insurer_key: str, profile: QuotationProfile) -> dict[str, Any]:
    """One provider-shaped offer payload.

    The shape mimics what a provider API would return; each adapter's
    ``normalize_result`` maps it onto the common structure. Keeping it dict-
    shaped (rather than already normalized) means the normalization step is
    genuinely exercised in mock mode too.
    """
    insurer = INSURERS[insurer_key]
    pricing = compute_offer_pricing(insurer, profile)

    coverages = pricing.coverages
    total = pricing.annual_total
    instalment_total = pricing.instalment_total
    instalment_amount = pricing.instalment_amount

    expires = datetime.now(timezone.utc) + timedelta(days=60)

    return {
        "calculation": pricing.breakdown_payload(),
        "insurer": insurer.name,
        "insurer_key": insurer.key,
        "product": insurer.product_name,
        "quote_reference": quote_reference(insurer.key, profile),
        "premium": {
            "annual_total": str(total),
            "currency": "EUR",
            "instalments": {
                "count": pricing.instalment_count,
                "amount": str(instalment_amount),
                "total": str(instalment_total),
            },
        },
        "liability": {
            "people": str(insurer.limit_people),
            "property": str(insurer.limit_property),
        },
        "driving_formula": profile.preferences.driving_formula or "free",
        "deductible": str(insurer.deductible),
        "percentage_excess": "10%",
        "conditions": {
            "black_box_required": insurer.requires_black_box,
            "approved_repair_network_required": insurer.requires_approved_repair_network,
        },
        "coverages": coverages,
        "exclusions": [
            "Guida in stato di ebbrezza o sotto effetto di sostanze stupefacenti",
            "Partecipazione a gare o competizioni sportive",
            "Conducente privo di patente valida",
        ],
        "expires_at": expires.isoformat(),
        "links": {
            "purchase": f"https://example-demo.invalid/{insurer.key}/quote/"
            f"{quote_reference(insurer.key, profile)}",
            "product_document": f"https://example-demo.invalid/{insurer.key}/dip.pdf",
            "precontractual_document": f"https://example-demo.invalid/{insurer.key}/dip-aggiuntivo.pdf",
        },
        "status": "QUOTED",
        "demonstration": True,
    }


def forced_outcome(provider_id: str) -> str | None:
    """Fault injection for demos and tests.

    ``PC_MOCK_FORCE_OUTCOME_ZURICH=unavailable`` makes that provider fail on
    demand, so the "a provider is down" path can be shown without waiting for a
    real outage.
    """
    return os.getenv(f"PC_MOCK_FORCE_OUTCOME_{provider_id.upper()}") or None


def mock_latency_seconds() -> float:
    """Simulated provider latency, so the progress screen has something to show."""
    raw = os.getenv("PC_MOCK_LATENCY_MS")
    if raw is None:
        return 0.4
    try:
        return max(0.0, int(raw) / 1000)
    except ValueError:
        return 0.0
