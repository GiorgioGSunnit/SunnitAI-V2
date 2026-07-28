"""Deterministic value-extraction primitives for the simulated LLM layer.

The real integration will have an LLM read the user's sentence and emit a
structured tool call (calculator_id + named inputs). planner.py mimics that
contract; this module supplies its raw extraction with fully deterministic
heuristics.

Design rule (the important one): this fixture is **label-anchored**, never
positional. A number is bound to a named input only when a cue word for that
field sits next to it in the sentence ("retribuzione ... 2500", "aliquota
10,6 per mille"). Numbers with no nearby field cue are left UNBOUND — the
planner then asks a clarifying question instead of silently mapping unlabeled
numbers by position (which produced results like an indemnity of 11 x 7 = 77).
It also distinguishes per-cent from per-mille, parses Italian natural-language
dates, reads explicit booleans with negation, and extracts comma-separated
string lists.

It is still a deliberately simple stand-in — every real edge case is the work
the production LLM (src/rag/calculation.py) already does — but it errs toward
asking, never toward a confident wrong number. None of this runs in production.
"""

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from app.schemas.calculator_definition import CalculatorDefinition

_NUMBER_RE = re.compile(r"-?\d[\d.]*(?:,\d+)?")
_ISO_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
_YES_WORDS = {"si", "sì", "yes", "true", "vero"}
_NO_WORDS = {"no", "false", "falso"}

# Unit words that are never a field's distinctive cue on their own — a field
# named `mesi_preavviso` is cued by "preavviso", not by the bare unit "mesi"
# ("11 anni e 7 mesi" must not bind 7 to it). A field whose whole name IS a
# unit word (e.g. `giorni`) keeps it, since then it is the only cue available.
_GENERIC_CUE_TOKENS = frozenset({
    "mesi", "mese", "anni", "anno", "giorni", "giorno", "euro", "eur",
    "importo", "valore", "somma", "tasso", "coefficiente", "annua", "annuo",
    "lorda", "lordo", "netto", "netta", "termine", "termini", "data", "date",
})

# Function words dropped from description-derived cues so a field's cue set is
# the meaningful nouns of its label, not connectives.
_CUE_FUNCTION_WORDS = frozenset({
    "della", "dello", "degli", "delle", "dallo", "dalla", "sulla", "sullo",
    "come", "cui", "per", "con", "non", "che", "una", "uno", "dei", "del",
    "sul", "dal", "dai", "nel", "nei", "gli", "gia", "ecc", "art", "suo", "sua",
    "esempio", "formato", "riferimento", "calcolare", "applicare",
    "deliberata", "deliberato", "secondo", "inclusi", "esclusa", "esclusi",
    "quello", "quella", "sono", "essere", "viene",
})

_MONTHS = {
    "gennaio": 1, "febbraio": 2, "marzo": 3, "aprile": 4, "maggio": 5,
    "giugno": 6, "luglio": 7, "agosto": 8, "settembre": 9, "ottobre": 10,
    "novembre": 11, "dicembre": 12,
}
_NL_DATE_RE = re.compile(
    r"\b(\d{1,2})\s+(" + "|".join(_MONTHS) + r")\s+(\d{4})\b",
    re.IGNORECASE,
)

# How close (in characters) a field cue must sit to a number to bind it. Tight
# on purpose: a real label hugs its value ("rendita 850", "reddito di 42000"),
# whereas a topic mention sits further off ("mancato preavviso: ho lavorato
# 11", "Compenso ... DM 55") and must NOT capture the number.
_CUE_WINDOW = 12

# Tokens that look like numbers but are identifiers, not input values:
# cadastral category codes (A/3, A/10, C/2) and legal references (DM 55/2014,
# art. 155, n. 289, L. 742). Blanked (length-preserving) before number scan.
_NON_VALUE_RE = re.compile(
    r"\b[A-F]/\d{1,2}\b"
    r"|\b(?:d\.?\s?m\.?|d\.?\s?lgs\.?|d\.?\s?p\.?r\.?|dpr|artt?\.?|n\.|l\.|reg\.?)\s*\d+(?:/\d+)?",
    re.IGNORECASE,
)


def _blank(match) -> str:
    return " " * (match.end() - match.start())


def _strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
    )


def parse_number(token: str) -> Decimal:
    """'42.000,50' -> 42000.50; '42.000' -> 42000; '0,5' -> 0.5."""
    s = token
    if "," in s:
        s = s.replace(".", "").replace(",", ".")
    elif s.count(".") == 1 and len(s.split(".")[1]) == 3:
        s = s.replace(".", "")
    elif s.count(".") > 1:
        s = s.replace(".", "")
    return Decimal(s)


def _extract_nl_dates(text: str) -> Tuple[List[str], str]:
    """Italian natural-language dates ('28 luglio 2025' -> '2025-07-28'), plus
    the text with those spans blanked so they are not re-read as numbers/years."""
    dates: List[str] = []
    masked = list(text)
    for match in _NL_DATE_RE.finditer(text):
        day = int(match.group(1))
        month = _MONTHS[match.group(2).lower()]
        year = int(match.group(3))
        try:
            dates.append(date(year, month, day).isoformat())
        except ValueError:
            continue
        masked[match.start():match.end()] = " " * (match.end() - match.start())
    return dates, "".join(masked)


def extract_values(text: str) -> Dict[str, Any]:
    """The raw material the binder works with: numbers (with their character
    positions, so binding can be label-anchored), ISO and natural-language
    dates, a tax-year candidate, and yes/no signals."""
    lowered = text.lower()

    nl_dates, denl = _extract_nl_dates(text)
    iso_dates = _ISO_DATE_RE.findall(denl)
    # Blank dates and non-value identifiers before the number scan, preserving
    # length so number positions stay aligned with the original text (cue
    # proximity is measured against the original).
    remainder = _ISO_DATE_RE.sub(_blank, denl)
    remainder = _NON_VALUE_RE.sub(_blank, remainder)
    dates = iso_dates + [d for d in nl_dates if d not in iso_dates]

    number_spans: List[Tuple[Decimal, int, int]] = []
    numbers: List[Decimal] = []
    tax_year: Optional[int] = None
    for token in _NUMBER_RE.finditer(remainder):
        value = parse_number(token.group(0))
        if value == value.to_integral_value() and 2000 <= int(value) <= 2099:
            # A bare four-digit year that is not attached to a field cue reads
            # as a tax-year candidate (the binder can still claim it as a
            # number if a cue is adjacent).
            if tax_year is None:
                tax_year = int(value)
            continue
        number_spans.append((value, token.start(), token.end()))
        numbers.append(value)

    words = set(re.findall(r"[a-zà-ù]+", lowered))
    boolean: Optional[bool] = None
    if words & _YES_WORDS:
        boolean = True
    elif words & _NO_WORDS:
        boolean = False

    period = None
    if len(dates) >= 2:
        start, end = sorted(dates[:2])
        period = {"start_date": start, "end_date": end}

    amount_frequency = None
    if any(m in lowered for m in ("al mese", "mensile", "mensili", "mese")):
        amount_frequency = "monthly"
    elif any(m in lowered for m in ("annuo", "annua", "annuale", "all'anno", "l'anno")):
        amount_frequency = "annual"

    boolean_hints: Dict[str, bool] = {}
    if "prima registrazione" in lowered or "registrazione iniziale" in lowered:
        boolean_hints["first_registration"] = True
    if "cedolare secca" in lowered:
        boolean_hints["cedolare_secca"] = True

    return {
        "text": text,
        "numbers": numbers,
        "number_spans": [[str(v), s, e] for v, s, e in number_spans],
        "dates": dates,
        "tax_year": tax_year,
        "period": period,
        "boolean": boolean,
        "amount_frequency": amount_frequency,
        "boolean_hints": boolean_hints,
    }


@dataclass
class SimulatedToolCall:
    """What the real LLM would emit: one structured /calculate request."""

    calculator_id: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    tax_year: Optional[int] = None
    period: Optional[Dict[str, str]] = None


def _cue_tokens(spec) -> List[str]:
    """A field's cue words: the meaningful tokens of its name AND its
    (Italian) description — the description is essential because field names
    are English (`taxable_income`) while users write Italian ("reddito").

    Bare unit words are dropped from the name when a distinctive token remains,
    so `mesi_preavviso` is cued by "preavviso" not "mesi"; generic and function
    words are dropped from the description. A field whose whole name is a unit
    word (e.g. `giorni`) keeps it as its cue."""
    name_tokens = [t for t in re.split(r"[^a-z0-9]+", _strip_accents(spec.name.lower())) if len(t) >= 3]
    distinctive_name = [t for t in name_tokens if t not in _GENERIC_CUE_TOKENS]
    cues = list(distinctive_name or name_tokens)

    if spec.description:
        for t in re.split(r"[^a-z0-9]+", _strip_accents(spec.description.lower())):
            if len(t) >= 3 and t not in _GENERIC_CUE_TOKENS and t not in _CUE_FUNCTION_WORDS:
                cues.append(t)

    seen: set = set()
    return [t for t in cues if not (t in seen or seen.add(t))]


def _cue_positions(lowered_ascii: str, cues: List[str]) -> List[Tuple[int, int]]:
    spans = []
    for cue in cues:
        for m in re.finditer(r"\b" + re.escape(cue) + r"\b", lowered_ascii):
            spans.append((m.start(), m.end()))
    return spans


def _nearest_cue_gap(cue_spans: List[Tuple[int, int]], nstart: int, nend: int) -> Optional[int]:
    best = None
    for cstart, cend in cue_spans:
        if cend <= nstart:
            gap = nstart - cend
        elif cstart >= nend:
            gap = cstart - nend
        else:
            gap = 0
        best = gap if best is None else min(best, gap)
    return best


def _apply_unit(spec, value: Decimal, lowered: str, nstart: int, nend: int, amount_frequency) -> Decimal:
    if spec.name == "annual_rent" and amount_frequency == "monthly":
        value = value * Decimal("12")
    if spec.unit == "rate":
        trailing = lowered[nend:nend + 14]
        if "per mille" in trailing or "‰" in lowered[nstart:nend + 6] or "permille" in trailing:
            value = value / Decimal("1000")
        elif "%" in trailing or "per cento" in trailing:
            value = value / Decimal("100")
        elif value > 1:
            value = value / Decimal("100")
    return value


def _find_string_list(text: str) -> List[str]:
    """Comma/'e'-separated alphabetic items ('studio, introduttiva, e
    decisionale' -> [...]). Unknown items are filtered downstream by the
    calculator's own validation."""
    body = text.split(":", 1)[1] if ":" in text else text
    parts = re.split(r"[,;]|\be\b", body)
    items = []
    for part in parts:
        token = _strip_accents(part.strip().lower())
        token = re.sub(r"[^a-z\s]", "", token).strip()
        if len(token) >= 4 and " " not in token:
            items.append(token)
    return items


def _find_boolean(cues: List[str], lowered_ascii: str) -> Optional[bool]:
    """True/False for a boolean field, read from a cue word and any negation
    ('non ...') a few words before it."""
    for cue in cues:
        for m in re.finditer(r"\b" + re.escape(cue) + r"\b", lowered_ascii):
            window = lowered_ascii[max(0, m.start() - 20):m.start()]
            negated = bool(re.search(r"\bnon\b|\bsenza\b", window))
            return not negated
    return None


def _bind_specs(
    specs,
    inputs: Dict[str, Any],
    values: Dict[str, Any],
    boolean_cues: Optional[Dict[str, List[str]]] = None,
) -> None:
    """The shared binder core: fill still-missing fields from extracted
    values, LABEL-ANCHORED. `boolean_cues` optionally narrows which cue
    words may flip each boolean field (used for offer parsing, where
    description-derived cues like 'polizza' would misfire on every offer)."""
    text = values.get("text", "")
    lowered_ascii = _strip_accents(text.lower())
    amount_frequency = values.get("amount_frequency")
    dates = list(values.get("dates", []))
    spans = [(Decimal(v), int(s), int(e)) for v, s, e in values.get("number_spans", [])]
    consumed: set = set()

    for spec in specs:
        if spec.name in inputs:
            continue

        if spec.type in ("decimal", "integer"):
            cue_spans = _cue_positions(lowered_ascii, _cue_tokens(spec))
            if not cue_spans:
                continue  # no label for this field -> do not guess
            best_idx, best_gap = None, None
            for idx, (value, nstart, nend) in enumerate(spans):
                if idx in consumed:
                    continue
                gap = _nearest_cue_gap(cue_spans, nstart, nend)
                if gap is None or gap > _CUE_WINDOW:
                    continue
                if best_gap is None or gap < best_gap:
                    best_idx, best_gap = idx, gap
            if best_idx is None:
                continue
            value, nstart, nend = spans[best_idx]
            consumed.add(best_idx)
            value = _apply_unit(spec, value, lowered_ascii, nstart, nend, amount_frequency)
            inputs[spec.name] = int(value) if spec.type == "integer" else value

        elif spec.type == "date":
            # ISO/natural dates in appearance order; only required date fields
            # are auto-filled (an optional deadline stays a deliberate blank).
            if dates and spec.required:
                inputs[spec.name] = dates.pop(0)

        elif spec.type == "boolean":
            if spec.name in values.get("boolean_hints", {}):
                inputs[spec.name] = values["boolean_hints"][spec.name]
                continue
            cues = boolean_cues.get(spec.name, []) if boolean_cues is not None else _cue_tokens(spec)
            resolved = _find_boolean(cues, lowered_ascii)
            if resolved is not None:
                inputs[spec.name] = resolved

        elif spec.type == "string_list":
            items = _find_string_list(text)
            if items:
                inputs[spec.name] = items

    # Unambiguous fallback: if, after label-anchored binding, exactly ONE
    # required numeric field is still missing AND exactly ONE number is still
    # unconsumed, that pairing is unambiguous — bind it. This handles clean
    # single-value answers ("42000 euro") without reopening the door to
    # positional guessing: it never fires when several numbers or several
    # fields compete (e.g. "11 anni e 7 mesi" for two fields stays a question).
    missing_numeric = [
        s for s in specs
        if s.name not in inputs and s.required and s.type in ("decimal", "integer")
    ]
    unconsumed = [i for i in range(len(spans)) if i not in consumed]
    if len(missing_numeric) == 1 and len(unconsumed) == 1:
        spec = missing_numeric[0]
        value, nstart, nend = spans[unconsumed[0]]
        value = _apply_unit(spec, value, lowered_ascii, nstart, nend, amount_frequency)
        inputs[spec.name] = int(value) if spec.type == "integer" else value


def bind_values(definition: CalculatorDefinition, inputs: Dict[str, Any], values: Dict[str, Any]) -> None:
    """Fill still-missing inputs from extracted values, LABEL-ANCHORED: numbers
    bind only next to a field cue; unlabeled numbers are left for a clarifying
    question. Booleans and string lists bind from their own cues/format.
    object_list inputs are never bound here — the conversation collects those
    one offer at a time via bind_offer."""
    _bind_specs(
        [s for s in definition.inputs if s.type != "object_list"], inputs, values
    )


def _distinctive_boolean_cues(item_specs) -> Dict[str, List[str]]:
    """Per boolean field, the name tokens that no OTHER boolean field shares —
    'copertura_kasko' is cued by 'kasko' alone, so 'copertura cristalli sì'
    cannot flip it via the shared token 'copertura'. Description tokens are
    excluded entirely: 'app di gestione della polizza' would otherwise make
    the word 'Polizza' in every offer name activate app_gestione. If every
    token is shared (degenerate), the full name token set is kept."""
    tokenized: Dict[str, List[str]] = {}
    counts: Dict[str, int] = {}
    for spec in item_specs:
        if spec.type != "boolean":
            continue
        tokens = [
            t for t in re.split(r"[^a-z0-9]+", _strip_accents(spec.name.lower()))
            if len(t) >= 3
        ]
        tokenized[spec.name] = tokens
        for token in set(tokens):
            counts[token] = counts.get(token, 0) + 1
    return {
        name: ([t for t in tokens if counts[t] == 1] or tokens)
        for name, tokens in tokenized.items()
    }


def bind_offer(item_specs, text: str) -> Dict[str, Any]:
    """Parse ONE candidate offer from one message, label-anchored like every
    other binding ('premio 450' binds, a bare number does not). A leading
    'Polizza A:' / 'Fornitore X:' segment becomes the offer's name (bound to
    the first free string field); everything else follows the shared rules."""
    inputs: Dict[str, Any] = {}

    label: Optional[str] = None
    binding_text = text
    if ":" in text:
        prefix = text.split(":", 1)[0]
        stripped = prefix.strip()
        if 0 < len(stripped) <= 40 and not any(ch.isdigit() for ch in stripped):
            label = stripped
            # The name is a name, not a field label: blanked (length-preserving)
            # so "Polizza X: 450" cannot bind 450 to premio_annuo via the word
            # "polizza" in that field's description — an unlabeled number must
            # stay unbound and become a question.
            binding_text = " " * (len(prefix) + 1) + text[len(prefix) + 1:]

    values = extract_values(binding_text)
    _bind_specs(item_specs, inputs, values, boolean_cues=_distinctive_boolean_cues(item_specs))

    if label:
        target = next(
            (s for s in item_specs if s.type == "string" and s.name not in inputs), None
        )
        if target is not None:
            inputs[target.name] = label
    return inputs
