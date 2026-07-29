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
REASON_CURRENCY_AMBIGUOUS = "currency_ambiguous"
REASON_NORMALIZATION_FAILED = "normalization_failed"

# Two currencies named at once is not "no currency stated": reading it that way
# would silently accept the calculator's own, which is how a dollar amount ends
# up taxed as euro.
CURRENCY_AMBIGUOUS = "ambiguous"

# The whole scope of this phase. Keyed by calculator id so a field name alone
# can never opt a different calculator in. `labels` keeps the internal field
# name out of anything the user reads.
FREQUENCY_SENSITIVE_INPUTS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "legal_it.registration_tax_leases": {
        "annual_rent": {
            "canonical_frequency": FREQUENCY_ANNUAL,
            "currency": "EUR",
            "labels": {"it": "canone", "es": "alquiler", "en": "rent"},
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
# Enumerated rather than stemmed with `annu\w*`, which also swallowed the
# Italian noun "annualità" — a yearly INSTALMENT of the tax. That word says
# nothing about the rent's frequency, so letting it assert one turned the
# instalment question "annualità successiva, canone 400 euro" into a claim that
# 400 is the annual rent: the silent-400 bug wearing a different hat. Note
# Italian "annuale" (two n) and Spanish "anual" (one n) cannot collide.
_ANNUAL_CUE = re.compile(
    r"(?:all\s*['’]\s*anno|l\s*['’]\s*anno|/\s*anno|per\s+anno"
    r"|annu[oaie]\b|annual(?:e|i|mente)\b"
    r"|\banual(?:es)?\b"
    r"|annual(?:ly)?\b|per\s+year|/\s*year|yearly"
    r"|al\s+a[nñ]o|por\s+a[nñ]o|/\s*a[nñ]o)",
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


# A whole localized number, separators included: 400 / 400,50 / 1.200,50 /
# 1,200.50. The lookarounds pin both ends to the complete token so a prefix of
# a longer number is never read as a number in its own right.
_NUMBER_TOKEN = re.compile(r"(?<![\d.,])\d+(?:[.,]\d+)*(?![\d.,]*\d)")


def _token_values(token: str) -> set:
    """Every plausible reading of one localized numeric token, as Decimals.

    `1.200` is 1200 to an Italian writer and 1.2 to an English one, and nothing
    in the token says which. Corroboration only asks "could the user have
    written this number", so both readings are produced and any exact match
    counts. What this must NOT do is treat readings as interchangeable by
    integer part: `400,50` yields {400.50} alone, so it can never vouch for an
    extracted 400 or 400.99.
    """
    values = set()
    has_dot, has_comma = "." in token, "," in token

    if has_dot and has_comma:
        # Both present: the rightmost separator is the decimal one.
        split_at = max(token.rfind("."), token.rfind(","))
        whole = re.sub(r"[.,]", "", token[:split_at])
        values.add(_decimal(f"{whole}.{token[split_at + 1:]}"))
    elif has_dot or has_comma:
        parts = token.split("." if has_dot else ",")
        # Reading A: thousands grouping — every group after the first is a
        # strict triple, and the first is at most a triple.
        if len(parts[0]) <= 3 and all(len(part) == 3 for part in parts[1:]):
            values.add(_decimal("".join(parts)))
        # Reading B: a single separator as the decimal point.
        if len(parts) == 2:
            values.add(_decimal(f"{parts[0]}.{parts[1]}"))
    else:
        values.add(_decimal(token))

    return {value for value in values if value is not None}


def _cue_windows(text: str, raw_value: Any) -> List[str]:
    """Every stretch of `text` immediately surrounding the given amount.

    Returning [] means the amount is not in the text at all — the signal that
    the value did not come from the user verbatim, so no cue may be trusted.
    Comparison is exact Decimal equality, never a digit-prefix match.
    """
    amount = _decimal(raw_value)
    if amount is None:
        return []
    return [
        text[max(0, match.start() - _WINDOW_BEFORE):match.end() + _WINDOW_AFTER]
        for match in _NUMBER_TOKEN.finditer(text or "")
        if amount in _token_values(match.group(0))
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


def read_currency(text: str) -> Optional[str]:
    """The currency `text` names: a code, CURRENCY_AMBIGUOUS, or None.

    Three states, not two. Collapsing "euro or dollars" to None would make it
    indistinguishable from a sentence that mentions no currency at all, and the
    caller treats that as "use the calculator's own" — which is precisely how a
    dollar figure gets taxed as euro.
    """
    codes = {code for code, cue in _CURRENCY_CUES if cue.search(text or "")}
    if not codes:
        return None
    return codes.pop() if len(codes) == 1 else CURRENCY_AMBIGUOUS


def _window_currency(windows: List[str]) -> Optional[str]:
    found = {read_currency(window) for window in windows}
    found.discard(None)
    if not found:
        return None
    if CURRENCY_AMBIGUOUS in found or len(found) > 1:
        return CURRENCY_AMBIGUOUS
    return found.pop()


def read_frequency(text: str, raw_value: Any) -> Tuple[str, Optional[str]]:
    """The frequency and currency stated next to `raw_value` in `text`."""
    windows = _cue_windows(text or "", raw_value)
    if not windows:
        return FREQUENCY_UNKNOWN, None
    frequency = _sole_signal(windows, _detect_frequency) or FREQUENCY_UNKNOWN
    return frequency, _window_currency(windows)


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
        labels = spec.get("labels") or {}

        if currency == CURRENCY_AMBIGUOUS or (currency and currency != expected_currency):
            # No exchange rate exists in this system, and inventing one would
            # put a fabricated number in front of a user as if they had said it.
            del resolved[field]
            unresolved.append({
                "field": field,
                "labels": labels,
                "reason": (
                    REASON_CURRENCY_AMBIGUOUS
                    if currency == CURRENCY_AMBIGUOUS
                    else REASON_CURRENCY_UNSUPPORTED
                ),
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
                "labels": labels,
                "reason": REASON_FREQUENCY_UNKNOWN,
                "raw_value": _plain(amount),
                "currency": expected_currency,
            })

    return resolved, conversions, unresolved


def failure_unresolved(
    calculator_id: Optional[str], values: Dict[str, Any]
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """What to withhold when normalization itself could not run.

    Fails CLOSED: every declared frequency-sensitive field present in `values`
    is stripped out and reported unresolved, because the one thing known about
    such a field is that nobody has established what its number means. Inputs
    this layer never touches are returned untouched — they are not in doubt, and
    discarding them would make the user restate facts nobody questioned.
    """
    fields = FREQUENCY_SENSITIVE_INPUTS.get(str(calculator_id or ""), {})
    safe = {
        name: value
        for name, value in (values or {}).items()
        if name not in fields
    }
    unresolved = [
        {
            "field": name,
            "labels": (fields[name].get("labels") or {}),
            "reason": REASON_NORMALIZATION_FAILED,
            "raw_value": str((values or {}).get(name)),
            "currency": fields[name].get("currency", "EUR"),
        }
        for name in fields
        if name in (values or {})
    ]
    return safe, unresolved


def pending_frequency_state(unresolved: List[Dict[str, Any]]) -> Dict[str, Any]:
    """What must survive to the next turn so the question is answerable.

    Every unresolved entry is held, each carrying WHY it is unresolved — the
    reason is what decides on the next turn whether the held number may be
    used. Held here rather than in `inputs_so_far` because it is not an input
    yet: nobody has established what it means.
    """
    return {
        entry["field"]: {
            "raw_value": entry["raw_value"],
            "reason": entry["reason"],
            "labels": entry.get("labels") or {},
            "currency": entry.get("currency") or entry.get("expected_currency") or "EUR",
            **(
                {"stated_currency": entry["stated_currency"]}
                if entry.get("stated_currency")
                else {}
            ),
        }
        for entry in unresolved
    }


def resolve_pending_frequency(
    reply: str, pending: Dict[str, Any]
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Apply a bare frequency reply ("mensile") to a held amount.

    Returns nothing usable when the reply carries a number: a restatement like
    "500 euro al mese" changes the amount too, so it belongs to ordinary
    extraction rather than to this shortcut.

    A held amount is convertible ONLY if the sole thing missing was its
    frequency. An amount held because its currency was wrong or ambiguous stays
    unusable no matter what frequency arrives: the number itself is denominated
    in something this calculator does not accept, so "mensile" would annualize
    a dollar figure into a euro field. Those are re-asked, which also keeps the
    state the user needs in order to correct them.
    """
    text = str(reply or "")
    if not pending or any(character.isdigit() for character in text):
        return {}, [], []

    frequency = _detect_frequency(text)
    if frequency is None:
        return {}, [], []

    stated_currency = read_currency(text)
    values: Dict[str, Any] = {}
    conversions: List[Dict[str, Any]] = []
    unresolved: List[Dict[str, Any]] = []

    for field, held in pending.items():
        held = held or {}
        amount = _decimal(held.get("raw_value"))
        expected_currency = held.get("currency") or "EUR"
        entry = {
            "field": field,
            "labels": held.get("labels") or {},
            "raw_value": held.get("raw_value"),
            "currency": expected_currency,
            "expected_currency": expected_currency,
        }

        if amount is None or held.get("reason") != REASON_FREQUENCY_UNKNOWN:
            unresolved.append({
                **entry,
                "reason": held.get("reason") or REASON_NORMALIZATION_FAILED,
                "stated_currency": held.get("stated_currency"),
            })
            continue

        if stated_currency == CURRENCY_AMBIGUOUS:
            unresolved.append({**entry, "reason": REASON_CURRENCY_AMBIGUOUS})
            continue
        if stated_currency and stated_currency != expected_currency:
            unresolved.append({
                **entry,
                "reason": REASON_CURRENCY_UNSUPPORTED,
                "stated_currency": stated_currency,
            })
            continue

        if frequency == FREQUENCY_MONTHLY:
            canonical = to_annual(amount)
            values[field] = _plain(canonical)
            conversions.append(
                conversion_record(field, amount, canonical, expected_currency)
            )
        else:
            values[field] = _plain(amount)
    return values, conversions, unresolved


def field_label(entry: Dict[str, Any], lang: str) -> str:
    """What to call a field in front of a user.

    Falls back to the internal name only when no label is declared — better a
    leaked identifier than an unanswerable question about nothing.
    """
    labels = entry.get("labels") or {}
    return labels.get(lang) or labels.get("it") or str(entry.get("field") or "")


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
