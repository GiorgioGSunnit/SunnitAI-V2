from decimal import Decimal
from typing import Any, Dict, List

from ..core.audit import AuditTrail
from ..core.errors import InputValidationError, ParameterResolutionError
from ..core.result_builder import round_output
from ..schemas.resolved_parameter import ResolvedParameter
from .base import CalculationStrategy, StrategyOutcome, data_quality_warning

FASI = ("studio", "introduttiva", "istruttoria", "decisionale")

# Adjustment bounds from DM 55/2014 as amended by DM 147/2022, art. 4:
# increases and reductions each up to 50% of the tabular value. (The pre-2022
# text allowed larger swings and a special reduction for the fase istruttoria;
# those no longer apply.)
MAX_AUMENTO_PCT = Decimal("50")
MAX_RIDUZIONE_PCT = Decimal("50")

SPESE_GENERALI_RATE = Decimal("0.15")  # art. 2 DM 55/2014
CPA_RATE = Decimal("0.04")
IVA_RATE = Decimal("0.22")

HUNDRED = Decimal("100")


class Dm55FeesStrategy(CalculationStrategy):
    """Compensi forensi per fase su tabella DM 55/2014: somma dei valori
    medi delle fasi richieste, ± aumento/riduzione percentuale (max 50%
    ciascuno, art. 4 come modificato dal DM 147/2022), poi gli accessori in
    ordine esplicito: +15% rimborso spese generali (sempre), +4% CPA (se
    applica_cpa), +22% IVA (se applica_iva) — con il subtotale esplicitato a
    ogni passaggio. CPA e IVA sono flag espliciti del chiamante, mai dedotti."""

    def run(self, definition, inputs: Dict[str, Any], request) -> StrategyOutcome:
        valore_causa = inputs[definition.formula.get("amount_input", "valore_causa")]
        fasi: List[str] = inputs[definition.formula.get("phases_input", "fasi")]
        aumento_pct = inputs.get("aumento_pct")
        riduzione_pct = inputs.get("riduzione_pct")
        # CPA and IVA are explicit, caller-controlled flags — never inferred.
        # They default to True (the ordinary case and the pre-existing
        # behavior); the validator records that default as an assumption.
        applica_cpa = bool(inputs.get("applica_cpa", True))
        applica_iva = bool(inputs.get("applica_iva", True))
        table_parameter_id = definition.formula["table_parameter_id"]

        self._validate_fasi(fasi)
        self._validate_adjustments(fasi, aumento_pct, riduzione_pct)

        pv, rows = self._scaglione_rows(table_parameter_id, valore_causa, request)

        trail = AuditTrail()
        compenso_base = Decimal("0")
        for fase in fasi:
            valore = rows[fase]
            compenso_base += valore
            trail.record(
                "fase",
                fase=fase,
                valore_medio=str(valore),
                note=f"Fase {fase}: valore medio di scaglione {valore} EUR",
            )
        trail.record(
            "compenso_base",
            compenso=str(compenso_base),
            fasi=list(fasi),
            note=f"Compenso tabellare = somma delle fasi richieste = {compenso_base} EUR",
        )

        compenso = compenso_base
        if aumento_pct:
            compenso = compenso_base * (HUNDRED + aumento_pct) / HUNDRED
            trail.record(
                "aumento",
                aumento_pct=str(aumento_pct),
                compenso_adeguato=str(compenso),
                note=f"Aumento del {aumento_pct}% (max 50%, art. 4 DM 55/2014): {compenso} EUR",
            )
        if riduzione_pct:
            compenso = compenso_base * (HUNDRED - riduzione_pct) / HUNDRED
            trail.record(
                "riduzione",
                riduzione_pct=str(riduzione_pct),
                compenso_adeguato=str(compenso),
                note=f"Riduzione del {riduzione_pct}% (max 50%, art. 4 DM 55/2014): {compenso} EUR",
            )

        # Explicit accessory-charge order: compenso -> +15% spese generali
        # (always) -> +4% CPA (if applica_cpa) -> +22% IVA (if applica_iva).
        spese_generali = compenso * SPESE_GENERALI_RATE
        subtotale_con_spese = compenso + spese_generali
        trail.record(
            "spese_generali",
            aliquota=str(SPESE_GENERALI_RATE),
            spese_generali=str(spese_generali),
            subtotale=str(subtotale_con_spese),
            note=f"+15% rimborso spese generali (art. 2 DM 55/2014): {spese_generali} EUR, subtotale {subtotale_con_spese} EUR",
        )

        cpa = subtotale_con_spese * CPA_RATE if applica_cpa else Decimal("0")
        subtotale_con_cpa = subtotale_con_spese + cpa
        trail.record(
            "cpa",
            cpa_applicata=applica_cpa,
            aliquota=str(CPA_RATE),
            cpa=str(cpa),
            subtotale=str(subtotale_con_cpa),
            note=(
                f"+4% CPA (Cassa Previdenza Avvocati): {cpa} EUR, subtotale {subtotale_con_cpa} EUR"
                if applica_cpa else
                "CPA non applicata (applica_cpa=false): 0 EUR"
            ),
        )

        iva = subtotale_con_cpa * IVA_RATE if applica_iva else Decimal("0")
        totale = subtotale_con_cpa + iva
        trail.record(
            "iva",
            iva_applicata=applica_iva,
            aliquota=str(IVA_RATE),
            iva=str(iva),
            totale=str(totale),
            note=(
                f"+22% IVA: {iva} EUR, totale {totale} EUR"
                if applica_iva else
                "IVA non applicata (applica_iva=false): 0 EUR"
            ),
        )

        parameters_used = {
            "dm55_compensi": ResolvedParameter(
                name="dm55_compensi",
                value={fase: str(rows[fase]) for fase in fasi},
                origin="parameter_store",
                parameter_id=table_parameter_id, source=pv.source,
                effective_from=pv.effective_from.isoformat(),
                effective_to=pv.effective_to.isoformat() if pv.effective_to else None,
                official=pv.official, last_verified_at=pv.last_verified_at,
                citations=pv.citations,
            ).model_dump()
        }

        warnings = []
        quality = data_quality_warning(pv, "Valori medi DM 55/2014")
        if quality:
            warnings.append(quality)

        return StrategyOutcome(
            result={
                "compenso_tabellare": round_output(compenso_base, definition.output),
                "compenso_adeguato": round_output(compenso, definition.output),
                "spese_generali": round_output(spese_generali, definition.output),
                "subtotale_con_spese": round_output(subtotale_con_spese, definition.output),
                "cpa_applicata": applica_cpa,
                "cpa": round_output(cpa, definition.output),
                "subtotale_con_cpa": round_output(subtotale_con_cpa, definition.output),
                "iva_applicata": applica_iva,
                "iva": round_output(iva, definition.output),
                "totale": round_output(totale, definition.output),
            },
            parameters_used=parameters_used,
            steps=trail.steps,
            warnings=warnings,
        )

    def _validate_fasi(self, fasi: List[str]) -> None:
        unknown = [f for f in fasi if f not in FASI]
        if unknown:
            raise InputValidationError(
                f"Fasi non riconosciute: {', '.join(unknown)}. Fasi valide: {', '.join(FASI)}",
                details={"input": "fasi", "unknown": unknown, "valid": list(FASI)},
            )
        if len(set(fasi)) != len(fasi):
            raise InputValidationError(
                "La lista delle fasi contiene duplicati",
                details={"input": "fasi", "value": fasi},
            )

    def _validate_adjustments(self, fasi, aumento_pct, riduzione_pct) -> None:
        if aumento_pct and riduzione_pct:
            raise InputValidationError(
                "Indicare aumento_pct oppure riduzione_pct, non entrambi",
                details={"inputs": ["aumento_pct", "riduzione_pct"]},
            )
        if aumento_pct is not None and not Decimal("0") <= aumento_pct <= MAX_AUMENTO_PCT:
            raise InputValidationError(
                f"aumento_pct deve essere tra 0 e {MAX_AUMENTO_PCT} (aumento massimo "
                "50%, art. 4 DM 55/2014 come modificato dal DM 147/2022)",
                details={"input": "aumento_pct", "value": str(aumento_pct), "max": str(MAX_AUMENTO_PCT)},
            )
        if riduzione_pct is not None and not Decimal("0") <= riduzione_pct <= MAX_RIDUZIONE_PCT:
            raise InputValidationError(
                f"riduzione_pct deve essere tra 0 e {MAX_RIDUZIONE_PCT} (riduzione massima "
                "50%, art. 4 DM 55/2014 come modificato dal DM 147/2022)",
                details={
                    "input": "riduzione_pct", "value": str(riduzione_pct),
                    "max": str(MAX_RIDUZIONE_PCT), "fasi": list(fasi),
                },
            )

    def _scaglione_rows(self, table_parameter_id: str, valore_causa: Decimal, request):
        """The (fase -> valore_medio) map of the scaglione containing
        valore_causa. Schema stubs (valore_medio null) fail loudly — a value
        band that is not yet populated must never resolve to a number."""
        from ..resolvers.date_parameter_resolver import describe_as_of_resolution
        from datetime import date as date_cls

        as_of = date_cls.fromisoformat(describe_as_of_resolution(request)["as_of_date"])
        try:
            pv = self.parameter_store.resolve_by_date(table_parameter_id, as_of)
        except KeyError as e:
            raise ParameterResolutionError(
                str(e), details={"parameter_id": table_parameter_id, "as_of_date": as_of.isoformat()}
            ) from e

        matching = [
            row for row in pv.value
            if Decimal(str(row["scaglione_min"])) <= valore_causa
            and (row["scaglione_max"] is None or valore_causa <= Decimal(str(row["scaglione_max"])))
        ]
        if not matching:
            # Above the top tabular scaglione there is no seventh row: the DM
            # 55/2014 tables stop at 520.000 EUR and art. 6 governs higher
            # values with a progressive "up to 30%" increment. We refuse
            # honestly rather than invent a value (product decision).
            highest_max = max(
                (Decimal(str(row["scaglione_max"])) for row in pv.value
                 if row.get("scaglione_max") is not None),
                default=None,
            )
            if highest_max is not None and valore_causa > highest_max:
                raise ParameterResolutionError(
                    (
                        f"Valore di causa {valore_causa} EUR oltre l'ultimo scaglione "
                        f"tabellare ({highest_max} EUR): non esiste una riga della Tabella 2 "
                        "per questa fascia. L'art. 6 DM 55/2014 prevede un aumento "
                        "progressivo fino al 30%; la liquidazione va determinata dal giudice "
                        "e non e supportata automaticamente."
                    ),
                    details={
                        "parameter_id": table_parameter_id,
                        "valore_causa": str(valore_causa),
                        "highest_scaglione_max": str(highest_max),
                        "unsupported_range": True,
                    },
                )
            raise ParameterResolutionError(
                f"Nessuno scaglione DM 55 copre il valore di causa {valore_causa}",
                details={"parameter_id": table_parameter_id, "valore_causa": str(valore_causa)},
            )
        if any(row["valore_medio"] is None for row in matching):
            scaglione = matching[0]
            raise ParameterResolutionError(
                (
                    f"Lo scaglione DM 55 {scaglione['scaglione_min']}-"
                    f"{scaglione['scaglione_max'] or 'oltre'} non e ancora popolato "
                    "(stub TO_VERIFY): caricare la Tabella 2 ministeriale prima di calcolare"
                ),
                details={
                    "parameter_id": table_parameter_id,
                    "valore_causa": str(valore_causa),
                    "scaglione_min": str(scaglione["scaglione_min"]),
                    "scaglione_max": str(scaglione["scaglione_max"]) if scaglione["scaglione_max"] is not None else None,
                },
            )
        rows = {row["fase"]: Decimal(str(row["valore_medio"])) for row in matching}
        missing = [f for f in FASI if f not in rows]
        if missing:
            raise ParameterResolutionError(
                f"Lo scaglione DM 55 per valore {valore_causa} non copre le fasi: {', '.join(missing)}",
                details={"parameter_id": table_parameter_id, "missing_fasi": missing},
            )
        return pv, rows
