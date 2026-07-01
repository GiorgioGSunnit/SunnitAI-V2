"""TFR (Trattamento di Fine Rapporto) complex formula plugin.

This plugin implements the year-by-year accrual and ISTAT revaluation
of TFR according to Art. 2120 c.c. It is the only POC formula that:
  - requires multi-step iteration (one loop per calendar year)
  - reads external data from the DB (istat_coefficients table)

Registration: the @register("tfr") decorator fires on import and stores
this function in the plugin registry (_PLUGINS dict in formulas/__init__.py).

IMPORTANT: ISTAT coefficients in the DB are approximate values seeded by
scripts/seed_formulas.py. Verify against official ISTAT publications before
using results in legal documents.
Reference: https://www.istat.it/it/archivio/rivalutazione-monetaria
"""

from src.calculator.formulas import register
from src.calculator.models import FormulaResult, StepResult


class IstatDataMissingError(Exception):
    """Raised when a required ISTAT coefficient is absent from the DB.

    This is NOT swallowed silently — it propagates to the API caller
    so the user can be informed rather than receiving a wrong result.
    """
    def __init__(self, year: int):
        self.year = year
        super().__init__(f"ISTAT coefficient missing for year {year}")


@register("tfr")
def calcola_tfr(params: dict) -> FormulaResult:
    """Calculate TFR accrued and revalued over the employment period.

    Algorithm (Art. 2120 c.c.):
      For each calendar year from y_start to y_end - 1:
        1. Accrue quota annua = RAL / 13.5
        2. Revalue all prior accrued quotas using the ISTAT coefficient for that year
        3. running_total = revalued_prior + quota_annua

    Parameters expected in params:
      retribuzione_annua_lorda (float): gross annual salary in €
      anno_inizio (int): year employment started (e.g. 2018)
      anno_fine (int): year employment ended (e.g. 2024)

    Returns:
      FormulaResult with one StepResult per year × 2 (quota + rivalutazione)
    """
    ral     = float(params["retribuzione_annua_lorda"])
    y_start = int(params["anno_inizio"])
    y_end   = int(params["anno_fine"])

    # Load ISTAT coefficients from DB for the relevant years.
    # Lazy import keeps the DB dependency out of module-load time —
    # this function is only called at execution time, not at import time.
    from src.db.base import SessionLocal
    from src.db.models import IstatCoefficient

    with SessionLocal() as session:
        coefficients = {
            row.year: float(row.tfr_coeff)
            for row in session.query(IstatCoefficient)
                .filter(IstatCoefficient.year.between(y_start, y_end - 1))
                .all()
        }

    steps = []
    quota_annua = ral / 13.5
    running_total = 0.0

    for year in range(y_start, y_end):
        # Step 1 — accrue quota for this year
        steps.append(StepResult(
            label=f"Quota {year}",
            computation=f"{ral:,.2f} ÷ 13,5",
            result=round(quota_annua, 2)
        ))

        # Step 2 — revalue all prior accrued quotas for this year.
        # Missing coefficient is a hard error: we must not produce a wrong result.
        if year not in coefficients:
            raise IstatDataMissingError(year)
        coeff = coefficients[year]

        revalued = round(running_total * coeff, 2) if running_total > 0 else 0.0
        running_total = round(revalued + quota_annua, 2)

        steps.append(StepResult(
            label=f"Rivalutazione {year}",
            computation=f"× {coeff} (ISTAT)",
            result=running_total
        ))

    return FormulaResult(
        formula_slug="tfr",
        formula_name_it="Trattamento di Fine Rapporto (TFR)",
        input_params=params,
        steps=steps,
        final_result=round(running_total, 2),
        unit="€",
        source_norm="Art. 2120 c.c.",
        warning="Coefficienti ISTAT approssimativi — verificare con dati ufficiali prima dell'uso in documenti legali"
    )
