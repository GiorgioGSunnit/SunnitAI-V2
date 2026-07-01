"""Formula result formatter — Phase 4.

Converts a FormulaResult into the citation JSON block presented to the user.

Number formatting uses Italian locale conventions:
  - Thousands separator: dot (.)
  - Decimal separator:   comma (,)
  Example: 9000.0 → '9.000,00'

The returned dict is passed to synthesize_answer as a structured block.
The LLM reads it but does NOT recompute the numbers — it only formats prose.
"""

from src.calculator.models import FormulaResult


def _fmt_number(value: float) -> str:
    """Format a float using Italian locale conventions.

    Python's :,.2f uses commas as thousands separators and dots as decimals.
    Italian locale is the inverse, so we swap after formatting.

    Examples:
        9000.0   → '9.000,00'
        616.44   → '616,44'
        1720.0   → '1.720,00'
        12.0     → '12,00'
    """
    # Format with Python default (comma=thousands, dot=decimal)
    formatted = f"{value:,.2f}"
    # Split on the decimal point
    integer_part, decimal_part = formatted.split(".")
    # Swap separators: replace commas with dots for the thousands
    integer_part = integer_part.replace(",", ".")
    return f"{integer_part},{decimal_part}"


def format_result(result: FormulaResult) -> dict:
    """Produce the citation JSON block in the format agreed with the client.

    This dict is injected into the synthesize_answer LLM prompt so that the
    model can present the calculation in natural language while the deterministic
    numbers remain untouched.

    Keys:
        formula     : Italian name of the formula
        fonte       : legal norm reference (e.g. 'Art. 2120 c.c.')
        parametri   : dict of input parameters as-is
        ragionamento: ordered list of computation steps in Italian locale
        risultato   : final result as a formatted Italian-locale string
        avvertenze  : optional Italian-language caveat (or None)
    """
    unit_suffix = f" {result.unit}" if result.unit else ""

    ragionamento = [
        f"{step.label}: {step.computation} = {_fmt_number(step.result)}{unit_suffix}"
        for step in result.steps
    ]

    return {
        "formula":      result.formula_name_it,
        "fonte":        result.source_norm,
        "parametri":    dict(result.input_params),
        "ragionamento": ragionamento,
        "risultato":    f"{_fmt_number(result.final_result)}{unit_suffix}",
        "avvertenze":   result.warning,
    }
