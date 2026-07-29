"""Deterministic normalization of frequency-sensitive calculator inputs.

Phase 2A covers exactly one rule and one field family. `annual_rent` on the
lease registration-tax calculator asserts a frequency in its NAME, while people
state rents monthly ("un canone di 400 euro al mese"). The extraction prompt
tells the LLM to report the amount *as written*; every arithmetic step happens
here, in `Decimal`, under a named rule. 400 accepted as an annual rent
understates the tax by a factor of twelve, and does it silently.

Two deliberate limits keep this narrow:

* Scope is a declared table, not a heuristic over field names. A calculator that
  means something else by "annual" must never be silently rewritten, so nothing
  outside FREQUENCY_SENSITIVE_INPUTS is touched.
* A conversion fires only when the amount can be found in the user's own words
  next to a frequency cue. That corroboration is what makes double-conversion
  impossible: if a model ignored its instructions and pre-multiplied, its value
  no longer matches the text and the frequency reads as unknown, so this layer
  asks instead of multiplying again.

Unknown frequency and unsupported currency are outcomes, not failures: they
produce a question. Nothing here guesses, and nothing here converts currency.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple

MONTHLY_TO_ANNUAL_RULE = "monthly_to_annual_x12"
MONTHS_PER_YEAR = Decimal("12")

FREQUENCY_MONTHLY = "monthly"
FREQUENCY_ANNUAL = "annual"
FREQUENCY_UNKNOWN = "unknown"

REASON_FREQUENCY_UNKNOWN = "frequency_unknown"
REASON_CURRENCY_UNSUPPORTED = "currency_unsupported"

# The whole scope of this phase. Keyed by calculator id so a field name alone
# can never opt a different calculator in.
FREQUENCY_SENSITIVE_INPUTS: Dict[str, Dict[str, Dict[str, str]]] = {
    "legal_it.registration_tax_leases": {
        "annual_rent": {
            "canonical_frequency": FREQUENCY_ANNUAL,
            "currency": "EUR",
        },
    },
}

_CURRENCY_SYMBOLS = {"EUR": "€", "USD": "$", "GBP": "£"}

# Frequency cues, in the three languages the chatbot supports. Matched only
# inside a short window around the amount itself (see _cue_window), so an
# "annualità successiva" elsewhere in the sentence cannot label a rent.
_MONTHLY_CUE = re.compile(
    r"(?:al\s+mese|a\s+mese|/\s*mese|mensil\w*|per\s+month|/\s*month|monthly|"
    r"al\s+mes|/\s*mes|mensual\w*)",
    re.IGNORECASE,
)
_ANNUAL_CUE = re.compile(
    r"(?:all\s*['’]\s*anno|l\s*['’]\s*anno|/\s*anno|per\s+anno|annu\w*|"
    r"per\s+year|/\s*year|yearly|annual\w*|al\s+a[nñ]o|por\s+a[nñ]o|/\s*a[nñ]o)",
    re.IGNORECASE,
)

_CURRENCY_CUES = (
    ("USD", re.compile(r"(?:\$|\bUSD\b|\bdollar\w*|\bdollar\b)", re.IGNORECASE)),
    ("GBP", re.compile(r"(?:£|\bGBP\b|\bsterlin\w*|\bpound\w*)", re.IGNORECASE)),
    ("EUR", re.compile(r"(?:€|\bEUR\b|\beuro\w*)", re.IGNORECASE)),
)

# How far around the amount a cue is allowed to sit. A frequency normally
# trails its number ("400 euro al mese"), a currency symbol may lead it
# ("$400"), so the window is asymmetric.
_WINDOW_BEFORE = 14
_WINDOW_AFTER = 32


def frequency_sensitive_fields(calculator_id: Optional[str]) -> Dict[str, Dict[str, str]]:
    """The declared frequency-sensitive inputs of one calculator, or {}."""
    return FREQUENCY_SENSITIVE_INPUTS.get(str(calculator_id or ""), {})


def to_annual(monthly: Decimal) -> Decimal:
    """The single arithmetic operation of this module: `monthly_to_annual_x12`."""
    return monthly * MONTHS_PER_YEAR


def extraction_prompt_note(fields: Dict[str, Dict[str, str]]) -> str:
    """The instruction that stops the model doing this module's job.

    Without it the model faces a contradiction — the field is named
    `annual_rent` but the user said a monthly figure — and resolves it by
    either multiplying (arithmetic it must not do) or omitting the rent
    entirely (a required input silently dropped).
    """
    names = ", ".join(sorted(fields))
    return (
        f"For {names}: report the amount EXACTLY as the user wrote it and do "
        "NOT convert it. If the user gives a monthly figure, report that "
        "monthly figure unchanged; a separate deterministic step performs any "
        "conversion. Never multiply, divide, or annualize a value yourself."
    )


def _decimal(value: Any) -> Optional[Decimal]:
    try:
        return Decimal(str(value).strip())
    except (InvalidOperation, ValueError, AttributeError):
        return None


def _amount_patterns(digits: str):
    """Literal and thousands-grouped spellings of the same integer digits."""
    yield re.escape(digits)
    if len(digits) > 3:
        head = len(digits) % 3 or 3
        parts = [digits[:head]] + [digits[i:i + 3] for i in range(head, len(digits), 3)]
        yield r"[.,\s]".join(map(re.escape, parts))


def _cue_windows(text: str, raw_value: Any) -> List[str]:
    """Every stretch of `text` immediately surrounding the given amount.

    Returning [] means the amount is not in the text at all — the signal that
    the value did not come from the user verbatim, so no cue may be trusted.
    """
    amount = _decimal(raw_value)
    if amount is None:
        return []
    digits = str(abs(amount).to_integral_value(rounding="ROUND_DOWN")).lstrip("-")
    if not digits or not digits.isdigit():
        return []

    pattern = re.compile(
        r"(?<!\d)(?:" + "|".join(_amount_patterns(digits)) + r")(?!\d)"
    )
    return [
        text[max(0, match.start() - _WINDOW_BEFORE):match.end() + _WINDOW_AFTER]
        for match in pattern.finditer(text or "")
    ]


def _sole_signal(windows: List[str], detect) -> Optional[str]:
    """One signal, only when every occurrence of the amount agrees.

    Two mentions of the same figure with different frequencies is ambiguity,
    and ambiguity here has to become a question rather than a coin toss.
    """
    found = {signal for signal in (detect(w) for w in windows) if signal}
    return found.pop() if len(found) == 1 else None


def _detect_frequency(window: str) -> Optional[str]:
    monthly = _MONTHLY_CUE.search(window)
    annual = _ANNUAL_CUE.search(window)
    if monthly and not annual:
        return FREQUENCY_MONTHLY
    if annual and not monthly:
        return FREQUENCY_ANNUAL
    return None


def _detect_currency(window: str) -> Optional[str]:
    for code, cue in _CURRENCY_CUES:
        if cue.search(window):
            return code
    return None


def read_frequency(text: str, raw_value: Any) -> Tuple[str, Optional[str]]:
    """The frequency and currency stated next to `raw_value` in `text`."""
    windows = _cue_windows(text or "", raw_value)
    if not windows:
        return FREQUENCY_UNKNOWN, None
    frequency = _sole_signal(windows, _detect_frequency) or FREQUENCY_UNKNOWN
    return frequency, _sole_signal(windows, _detect_currency)


def conversion_record(
    field: str,
    raw_value: Decimal,
    canonical_value: Decimal,
    currency: str,
) -> Dict[str, Any]:
    """The audit trail for one conversion. Plain JSON types only: this rides
    along in session state and must round-trip unchanged."""
    return {
        "field": field,
        "raw_value": _plain(raw_value),
        "source_frequency": FREQUENCY_MONTHLY,
        "canonical_value": _plain(canonical_value),
        "canonical_frequency": FREQUENCY_ANNUAL,
        "rule_id": MONTHLY_TO_ANNUAL_RULE,
        "factor": _plain(MONTHS_PER_YEAR),
        "currency": currency,
    }


def _plain(value: Decimal) -> str:
    """A Decimal as the platform expects it: exact, no exponent notation."""
    normalized = value.normalize()
    if normalized == normalized.to_integral_value():
        normalized = normalized.quantize(Decimal(1))
    return format(normalized, "f")


def normalize_inputs(
    calculator_id: Optional[str], values: Dict[str, Any], text: str
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Resolve declared frequency-sensitive inputs in `values`.

    Returns (values, conversions, unresolved). An unresolved field is REMOVED
    from the returned values: leaving it in is precisely the silent-400 bug, so
    the caller must ask about it instead of calculating with it.
    """
    fields = frequency_sensitive_fields(calculator_id)
    if not fields or not isinstance(values, dict):
        return values, [], []

    resolved = dict(values)
    conversions: List[Dict[str, Any]] = []
    unresolved: List[Dict[str, Any]] = []

    for field, spec in fields.items():
        if field not in resolved:
            continue
        amount = _decimal(resolved[field])
        if amount is None:
            continue

        frequency, currency = read_frequency(text, resolved[field])
        expected_currency = spec.get("currency", "EUR")

        if currency and currency != expected_currency:
            # No exchange rate exists in this system, and inventing one would
            # put a fabricated number in front of a user as if they had said it.
            del resolved[field]
            unresolved.append({
                "field": field,
                "reason": REASON_CURRENCY_UNSUPPORTED,
                "raw_value": _plain(amount),
                "stated_currency": currency,
                "expected_currency": expected_currency,
            })
            continue

        if frequency == FREQUENCY_MONTHLY:
            canonical = to_annual(amount)
            resolved[field] = _plain(canonical)
            conversions.append(
                conversion_record(field, amount, canonical, expected_currency)
            )
        elif frequency == FREQUENCY_ANNUAL:
            resolved[field] = _plain(amount)
        else:
            del resolved[field]
            unresolved.append({
                "field": field,
                "reason": REASON_FREQUENCY_UNKNOWN,
                "raw_value": _plain(amount),
                "currency": expected_currency,
            })

    return resolved, conversions, unresolved


def pending_frequency_state(unresolved: List[Dict[str, Any]]) -> Dict[str, Any]:
    """What must survive to the next turn so the question is answerable.

    Only the amount the user already stated. Holding it here rather than in
    `inputs_so_far` is the point: it is not an input yet, because nobody has
    said what it means.
    """
    return {
        entry["field"]: {
            "raw_value": entry["raw_value"],
            "reason": entry["reason"],
            "currency": entry.get("currency") or entry.get("expected_currency") or "EUR",
        }
        for entry in unresolved
        if entry.get("reason") == REASON_FREQUENCY_UNKNOWN
    }


def resolve_pending_frequency(
    reply: str, pending: Dict[str, Any]
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Apply a bare frequency reply ("mensile") to a held amount.

    Returns ({}, []) when the reply carries a number: a restatement like "500
    euro al mese" changes the amount too, so it belongs to ordinary extraction
    rather than to this shortcut.
    """
    text = str(reply or "")
    if not pending or any(character.isdigit() for character in text):
        return {}, []

    frequency = _detect_frequency(text)
    if frequency is None:
        return {}, []

    values: Dict[str, Any] = {}
    conversions: List[Dict[str, Any]] = []
    for field, held in pending.items():
        amount = _decimal((held or {}).get("raw_value"))
        if amount is None:
            continue
        currency = (held or {}).get("currency") or "EUR"
        if frequency == FREQUENCY_MONTHLY:
            canonical = to_annual(amount)
            values[field] = _plain(canonical)
            conversions.append(
                conversion_record(field, amount, canonical, currency)
            )
        else:
            values[field] = _plain(amount)
    return values, conversions


def format_amount(value: Any, lang: str) -> str:
    """An amount with the thousands grouping the reader's language uses."""
    amount = _decimal(value)
    if amount is None:
        return str(value)
    if amount == amount.to_integral_value():
        rendered = f"{int(amount):,}"
    else:
        rendered = f"{amount:,.2f}"
    if lang in ("it", "es"):
        rendered = rendered.replace(",", "\x00").replace(".", ",").replace("\x00", ".")
    return rendered


def format_conversion(record: Dict[str, Any], lang: str, words: Dict[str, str]) -> str:
    """Render a conversion as the arithmetic it performed, e.g.
    `€400/mese × 12 = €4.800/anno` — the multiplication has to be visible or
    the result is a number the user cannot check."""
    symbol = _CURRENCY_SYMBOLS.get(record.get("currency", "EUR"), "")
    raw = format_amount(record.get("raw_value"), lang)
    canonical = format_amount(record.get("canonical_value"), lang)
    factor = format_amount(record.get("factor"), lang)
    return (
        f"{symbol}{raw}/{words['month']} × {factor} = "
        f"{symbol}{canonical}/{words['year']}"
    )
