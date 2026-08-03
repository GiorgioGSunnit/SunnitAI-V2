"""One canonical description per profile field.

Two providers asking for the same thing must produce the *same* question, so
the missing-information screen shows it once. That only works if the wording,
input type and choices come from here rather than from each adapter.

Labels are in Italian: this is a tool for Italian insurance staff.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..schemas.quotes import MissingField


@dataclass(frozen=True)
class FieldSpec:
    path: str
    label: str
    input_type: str = "text"
    choices: tuple[dict, ...] = field(default_factory=tuple)
    help_text: str | None = None
    #: UI grouping on the missing-information screen.
    group: str = "customer"


def _c(*pairs: tuple[str, str]) -> tuple[dict, ...]:
    return tuple({"value": v, "label": lbl} for v, lbl in pairs)


_SPECS: dict[str, FieldSpec] = {
    spec.path: spec
    for spec in (
        # -- customer --------------------------------------------------------
        FieldSpec("customer.owner_date_of_birth", "Data di nascita del proprietario", "date"),
        FieldSpec("customer.first_name", "Nome", "text"),
        FieldSpec("customer.last_name", "Cognome", "text"),
        FieldSpec(
            "customer.tax_code",
            "Codice fiscale",
            "text",
            help_text="16 caratteri per le persone fisiche.",
        ),
        FieldSpec(
            "customer.gender",
            "Sesso",
            "choice",
            _c(("M", "Maschile"), ("F", "Femminile")),
        ),
        FieldSpec("customer.mobile_number", "Numero di cellulare", "text"),
        FieldSpec("customer.address_street", "Indirizzo di residenza", "text"),
        FieldSpec("customer.municipality", "Comune di residenza", "text"),
        FieldSpec("customer.province", "Provincia", "text", help_text="Sigla di due lettere."),
        FieldSpec("customer.postcode", "CAP", "text"),
        FieldSpec(
            "customer.subject_type",
            "Tipo di contraente",
            "choice",
            _c(("individual", "Persona fisica"), ("company", "Azienda")),
        ),
        FieldSpec("customer.company_name", "Ragione sociale", "text"),
        FieldSpec("customer.vat_number", "Partita IVA", "text"),
        FieldSpec(
            "customer.policyholder_same_as_owner",
            "Il contraente coincide con il proprietario",
            "boolean",
        ),
        # -- vehicle ---------------------------------------------------------
        FieldSpec("vehicle.plate", "Targa del veicolo", "text", group="vehicle"),
        FieldSpec(
            "vehicle.ownership_status",
            "Titolo di possesso",
            "choice",
            _c(
                ("owner", "Proprietario"),
                ("leasing", "Leasing"),
                ("long_term_rental", "Noleggio a lungo termine"),
                ("usufruct", "Usufrutto"),
            ),
            group="vehicle",
        ),
        FieldSpec(
            "vehicle.first_registration_date",
            "Data di prima immatricolazione",
            "date",
            group="vehicle",
        ),
        FieldSpec("vehicle.purchase_date", "Data di acquisto", "date", group="vehicle"),
        FieldSpec("vehicle.make", "Marca", "text", group="vehicle"),
        FieldSpec("vehicle.model", "Modello", "text", group="vehicle"),
        FieldSpec("vehicle.trim", "Allestimento", "text", group="vehicle"),
        FieldSpec(
            "vehicle.fuel_type",
            "Alimentazione",
            "choice",
            _c(
                ("petrol", "Benzina"),
                ("diesel", "Diesel"),
                ("lpg", "GPL"),
                ("methane", "Metano"),
                ("hybrid", "Ibrida"),
                ("electric", "Elettrica"),
            ),
            group="vehicle",
        ),
        FieldSpec("vehicle.power_kw", "Potenza (kW)", "number", group="vehicle"),
        FieldSpec(
            "vehicle.primary_use",
            "Uso prevalente",
            "choice",
            _c(
                ("private", "Uso privato"),
                ("commute", "Casa-lavoro"),
                ("professional", "Uso professionale"),
            ),
            group="vehicle",
        ),
        FieldSpec("vehicle.annual_kilometres", "Chilometri annui", "number", group="vehicle"),
        FieldSpec(
            "vehicle.overnight_parking",
            "Dove parcheggia di notte",
            "choice",
            _c(
                ("garage", "Box o garage"),
                ("private_area", "Area privata"),
                ("public_road", "Strada pubblica"),
            ),
            group="vehicle",
        ),
        FieldSpec(
            "vehicle.anti_theft_system",
            "Sistema antifurto",
            "choice",
            _c(
                ("none", "Nessuno"),
                ("alarm", "Allarme"),
                ("satellite", "Antifurto satellitare"),
                ("mechanical", "Dispositivo meccanico"),
            ),
            group="vehicle",
        ),
        FieldSpec("vehicle.towing_hook", "Gancio di traino", "boolean", group="vehicle"),
        # -- insurance history -----------------------------------------------
        FieldSpec("history.current_insurer", "Compagnia attuale", "text", group="history"),
        FieldSpec(
            "history.existing_policy_expiry", "Scadenza polizza attuale", "date", group="history"
        ),
        FieldSpec(
            "history.universal_merit_class",
            "Classe di merito universale (CU)",
            "number",
            help_text="Da 1 a 18, come indicato nell'attestato di rischio.",
            group="history",
        ),
        FieldSpec(
            "history.first_insurance",
            "Prima assicurazione",
            "boolean",
            help_text="Il veicolo non è mai stato assicurato a nome del proprietario.",
            group="history",
        ),
        FieldSpec(
            "history.claims_last_5_years", "Sinistri negli ultimi 5 anni", "number", group="history"
        ),
        FieldSpec(
            "history.claims_full_responsibility",
            "Sinistri con responsabilità principale",
            "number",
            group="history",
        ),
        FieldSpec(
            "history.claims_partial_responsibility",
            "Sinistri con responsabilità paritaria",
            "number",
            group="history",
        ),
        FieldSpec(
            "history.bersani_applicable",
            "Applicare la RC Familiare (Legge Bersani)",
            "boolean",
            help_text="Consente di ereditare la classe di merito di un familiare convivente.",
            group="history",
        ),
        FieldSpec(
            "history.bersani_source_plate", "Targa del veicolo di riferimento", "text", group="history"
        ),
        FieldSpec(
            "history.bersani_source_merit_class",
            "Classe di merito del veicolo di riferimento",
            "number",
            group="history",
        ),
        FieldSpec(
            "history.risk_certificate_reference",
            "Riferimento attestato di rischio",
            "text",
            group="history",
        ),
        # -- coverage preferences ---------------------------------------------
        FieldSpec(
            "preferences.driving_formula",
            "Formula di guida",
            "choice",
            _c(
                ("free", "Guida libera"),
                ("expert", "Guida esperta"),
                ("exclusive", "Guida esclusiva"),
            ),
            group="preferences",
        ),
        FieldSpec(
            "preferences.min_liability_limit_people",
            "Massimale minimo danni a persone",
            "number",
            group="preferences",
        ),
        FieldSpec(
            "preferences.min_liability_limit_property",
            "Massimale minimo danni a cose",
            "number",
            group="preferences",
        ),
        FieldSpec(
            "preferences.max_acceptable_deductible",
            "Franchigia massima accettata",
            "number",
            group="preferences",
        ),
        FieldSpec(
            "preferences.accepts_black_box",
            "Accetta la scatola nera",
            "boolean",
            group="preferences",
        ),
        FieldSpec(
            "preferences.accepts_approved_repair_network",
            "Accetta le carrozzerie convenzionate",
            "boolean",
            group="preferences",
        ),
        FieldSpec(
            "preferences.payment_frequency",
            "Modalità di pagamento",
            "choice",
            _c(("annual", "Annuale"), ("instalments", "Rateale")),
            group="preferences",
        ),
    )
}

#: Human-readable names for the groups, used as section headings.
GROUP_LABELS = {
    "customer": "Dati del cliente",
    "vehicle": "Dati del veicolo",
    "history": "Storia assicurativa",
    "preferences": "Preferenze di copertura",
}


def describe(path: str) -> MissingField:
    """Render a profile path as a UI question.

    An unknown path still produces a usable question rather than an error, so a
    provider asking for something the catalogue has not seen yet degrades to a
    plain text input instead of breaking the run.
    """
    spec = _SPECS.get(path)
    if spec is None:
        return MissingField(
            field_path=path,
            label=path.rsplit(".", 1)[-1].replace("_", " ").capitalize(),
            input_type="text",
        )
    return MissingField(
        field_path=spec.path,
        label=spec.label,
        input_type=spec.input_type,
        choices=list(spec.choices) or None,
        help_text=spec.help_text,
    )


def group_for(path: str) -> str:
    spec = _SPECS.get(path)
    return spec.group if spec else path.split(".", 1)[0]


def is_known(path: str) -> bool:
    return path in _SPECS


def all_paths() -> tuple[str, ...]:
    return tuple(_SPECS)
