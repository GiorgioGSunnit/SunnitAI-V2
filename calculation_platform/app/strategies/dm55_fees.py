from decimal import Decimal
from typing import Any, Dict, List

from ..core.audit import AuditTrail
from ..core.errors import InputValidationError, ParameterResolutionError
from ..core.result_builder import round_output
from ..schemas.resolved_parameter import ResolvedParameter
from .base import CalculationStrategy, StrategyOutcome

FASI = ("studio", "introduttiva", "istruttoria", "decisionale")

# Adjustment bounds from DM 55/2014 (as amended by DM 147/2022), art. 4:
# increases generally up to 80%; reductions generally up to 50%, up to 70%
# for the fase istruttoria only.
MAX_AUMENTO_PCT = Decimal("80")
MAX_RIDUZIONE_PCT = Decimal("50")
MAX_RIDUZIONE_ISTRUTTORIA_PCT = Decimal("70")

SPESE_GENERALI_RATE = Decimal("0.15")  # art. 2 DM 55/2014
CPA_RATE = Decimal("0.04")
IVA_RATE = Decimal("0.22")

HUNDRED = Decimal("100")


class Dm55FeesStrategy(CalculationStrategy):
    """Compensi forensi per fase su tabella DM 55/2014: somma dei valori
    medi delle fasi richieste, ± aumento/riduzione percentuale (validati
    contro i limiti del decreto), poi la catena obbligatoria
    +15% rimborso spese generali, +4% CPA, +22% IVA — con il subtotale
    esplicitato a ogni passaggio."""

    def run(self, definition, inputs: Dict[str, Any], request) -> StrategyOutcome:
        valore_causa = inputs[definition.formula.get("amount_input", "valore_causa")]
        fasi: List[str] = inputs[definition.formula.get("phases_input", "fasi")]
        aumento_pct = inputs.get("aumento_pct")
        riduzione_pct = inputs.get("riduzione_pct")
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
                note=f"Aumento del {aumento_pct}% (max 80%, art. 4 DM 55/2014): {compenso} EUR",
            )
        if riduzione_pct:
            compenso = compenso_base * (HUNDRED - riduzione_pct) / HUNDRED
            trail.record(
                "riduzione",
                riduzione_pct=str(riduzione_pct),
                compenso_adeguato=str(compenso),
                note=(
                    f"Riduzione del {riduzione_pct}% (max 50%, fino al 70% per la sola fase "
                    f"istruttoria, art. 4 DM 55/2014): {compenso} EUR"
                ),
            )

        spese_generali = compenso * SPESE_GENERALI_RATE
        subtotale_con_spese = compenso + spese_generali
        trail.record(
            "spese_generali",
            spese_generali=str(spese_generali),
            subtotale=str(subtotale_con_spese),
            note=f"+15% rimborso spese generali (art. 2 DM 55/2014): {spese_generali} EUR, subtotale {subtotale_con_spese} EUR",
        )
        cpa = subtotale_con_spese * CPA_RATE
        subtotale_con_cpa = subtotale_con_spese + cpa
        trail.record(
            "cpa",
            cpa=str(cpa),
            subtotale=str(subtotale_con_cpa),
            note=f"+4% CPA (Cassa Previdenza Avvocati): {cpa} EUR, subtotale {subtotale_con_cpa} EUR",
        )
        iva = subtotale_con_cpa * IVA_RATE
        totale = subtotale_con_cpa + iva
        trail.record(
            "iva",
            iva=str(iva),
            totale=str(totale),
            note=f"+22% IVA: {iva} EUR, totale {totale} EUR",
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
        if pv.placeholder or pv.verified is False:
            warnings.append(
                "I valori medi DM 55/2014 usati sono SEGNAPOSTO non verificati contro la "
                "tabella ministeriale; il risultato non e utilizzabile operativamente."
            )

        return StrategyOutcome(
            result={
                "compenso_tabellare": round_output(compenso_base, definition.output),
                "compenso_adeguato": round_output(compenso, definition.output),
                "spese_generali": round_output(spese_generali, definition.output),
                "subtotale_con_spese": round_output(subtotale_con_spese, definition.output),
                "cpa": round_output(cpa, definition.output),
                "subtotale_con_cpa": round_output(subtotale_con_cpa, definition.output),
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
                f"aumento_pct deve essere tra 0 e {MAX_AUMENTO_PCT} (aumento massimo di regola "
                "80%, art. 4 DM 55/2014)",
                details={"input": "aumento_pct", "value": str(aumento_pct), "max": str(MAX_AUMENTO_PCT)},
            )
        if riduzione_pct is not None:
            only_istruttoria = list(fasi) == ["istruttoria"]
            limit = MAX_RIDUZIONE_ISTRUTTORIA_PCT if only_istruttoria else MAX_RIDUZIONE_PCT
            if not Decimal("0") <= riduzione_pct <= limit:
                raise InputValidationError(
                    f"riduzione_pct deve essere tra 0 e {limit} (riduzione massima di regola 50%, "
                    "fino al 70% per la sola fase istruttoria, art. 4 DM 55/2014)",
                    details={
                        "input": "riduzione_pct", "value": str(riduzione_pct),
                        "max": str(limit), "fasi": list(fasi),
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
