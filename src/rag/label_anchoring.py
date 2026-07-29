"""Label-anchored numeric binding for the offline extraction fallback.

When LLM extraction is unavailable, calculator inputs still have to be read out
of a sentence. The rule here is that a number binds to a field only when the
sentence SAYS so — a distinctive word for that field sits next to it, or the
message assigns it by name. Order is never evidence.

The previous fallback paired fields and numbers with `zip()`, and the failures
that produced were not near-misses: "ho lavorato 11 anni e 7 mesi" became an
indemnity of 11 x 7 = 77, and "cosa dice l'articolo 40 del codice?" became a
40-year-old driver. In both cases a question was silently answered with a number
nobody had supplied, which is worse than declining to answer.

Three ideas carry the weight:

* **Distinctive cues only.** A field's cues come from its name and description,
  minus words that appear in every sentence about money or time. `mesi_preavviso`
  is cued by "preavviso"; "mesi" alone must never bind it, or every mention of a
  duration would fill it.
* **Proximity.** A real label hugs its value ("retribuzione mensile 2500"),
  whereas a topic mention sits further off. Cues beyond a short window do not
  count.
* **One narrow exception.** A lone number for a lone remaining field is bound
  only when the message looks like an answer — a compact value reply, or an
  explicit calculation request. Prose and legal questions are excluded, which is
  exactly what keeps the article-number case from returning.

The doctrine is adapted from calculation_platform/simulation/scripted_llm.py,
which pioneered it as a development fixture. This is an independent production
implementation; nothing is imported across that boundary in either direction.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# How close a cue must sit to a number to bind it. Tight on purpose: a label
# hugs its value, a topic mention does not.
CUE_WINDOW = 12

_MIN_CUE_LENGTH = 3

# Words that are never a field's distinctive cue, however close they sit.
#
# Bare units and generic money/time nouns appear in almost every sentence a user
# writes, so treating them as labels means any number near any of them binds.
# A field whose whole name IS such a word keeps it, since then it is the only
# cue there is.
GENERIC_CUE_TOKENS = frozenset({
    # currency and amount
    "euro", "eur", "importo", "importi", "valore", "valori", "somma", "somme",
    "cifra", "amount", "value", "total", "totale",
    # time
    "mese", "mesi", "anno", "anni", "annua", "annuo", "annui", "annue",
    "giorno", "giorni", "settimana", "settimane", "data", "date", "durata",
    "periodo", "termine", "termini", "month", "months", "year", "years",
    "day", "days",
    # rates
    "tasso", "tassi", "aliquota", "aliquote", "percentuale", "percentuali",
    "percento", "rate", "percent", "percentage",
    # generic qualifiers that carry no field identity
    "lordo", "lorda", "netto", "netta", "globale", "complessivo", "totale",
    "gross", "net",
})

# Dropped from description-derived cues so a field's cue set is the meaningful
# nouns of its label rather than the connectives around them.
_CUE_FUNCTION_WORDS = frozenset({
    "della", "dello", "degli", "delle", "dallo", "dalla", "sulla", "sullo",
    "come", "cui", "per", "con", "non", "che", "una", "uno", "dei", "del",
    "sul", "dal", "dai", "nel", "nei", "gli", "gia", "ecc", "art", "artt",
    "suo", "sua", "esempio", "formato", "riferimento", "calcolare", "applicare",
    "secondo", "inclusi", "incluso", "esclusa", "esclusi", "escluso",
    "quello", "quella", "sono", "essere", "viene", "deve", "puo", "ove",
    "the", "and", "for", "with", "from", "this", "that", "its",
    "elementi", "continuativi", "applicabile", "lavorati", "fatto",
})

# An explicit `field: value` / `field = value` assignment. The strongest anchor
# available, so distance does not matter for it.
_ASSIGNMENT = re.compile(r"([A-Za-z_]\w*)\s*[:=]\s*")

# An explicit request to compute something. Enough on its own to read a lone
# number as an input, because the user has said what they want done with it.
_CALCULATION_REQUEST = re.compile(
    r"\b(?:calcola\w*|calcolami|calcolo\s+d|quanto\s+(?:pago|devo|costa|sar|viene)"
    r"|quant['’]\s*e|quantifica\w*"
    r"|calculate|compute|how\s+much"
    r"|calcula\w*|cu[aá]nto\s+(?:pago|debo|cuesta))",
    re.IGNORECASE,
)

# Words that make a message prose rather than a value reply, even a short one.
# A question is never an answer.
_INTERROGATIVE = re.compile(
    r"\b(?:cosa|che\s+cosa|quale|quali|come|perche|perch['’]|dove|quando|chi"
    r"|dice|dicono|prevede|prevedono|spieg\w*|significa|riguarda"
    r"|articolo|art|legge|codice|comma|decreto|sentenza|norma|c\.c\.|c\.p\.c\."
    r"|what|which|why|where|when|who|explain|says|article|law"
    r"|qu[eé]|cu[aá]l|c[oó]mo|d[oó]nde|art[ií]culo|ley)\b",
    re.IGNORECASE,
)

# Beyond this many leftover words, a message is prose carrying a number rather
# than a reply consisting of one.
_MAX_RESIDUAL_WORDS = 3

# A bare four-digit year. Never eligible for the narrow single-value fallback:
# in "nel 2026 quanto pago di IRPEF?" the only number present is a year, and
# reading it as the taxable income invents an answer out of the question's own
# date. An anchored binding may still claim such a number, because there the
# sentence has named the field it belongs to.
_BARE_YEAR = re.compile(r"^(?:19|20)\d{2}$")

# Filler that does not stop a message being a compact value reply.
_REPLY_FILLER = frozenset({
    "sono", "circa", "ammonta", "ammontano", "pari", "totale", "pero", "poi",
    "si", "ok", "ecco", "diciamo", "credo", "penso", "forse",
    "is", "are", "about", "roughly", "around", "approximately", "yes", "total",
    "es", "son", "aproximadamente", "unos", "unas",
}) | GENERIC_CUE_TOKENS


def _strip_accents(text: str) -> str:
    return "".join(
        character
        for character in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(character)
    )


def _normalized(text: str) -> str:
    return _strip_accents(str(text or "").lower())


def _tokens_of(text: str) -> List[str]:
    return [token for token in re.split(r"[^a-z0-9]+", _normalized(text)) if token]


def cue_tokens(spec: Dict[str, Any]) -> List[str]:
    """The words that identify one field, strongest first.

    Name tokens come first and the description supplies the rest, because field
    names are English (`taxable_income`) while users write Italian ("reddito").
    Generic words are dropped from both — unless the name consists of nothing
    else, in which case they are all this field has.
    """
    name_tokens = [
        token for token in _tokens_of(spec.get("name", ""))
        if len(token) >= _MIN_CUE_LENGTH
    ]
    distinctive = [token for token in name_tokens if token not in GENERIC_CUE_TOKENS]
    cues = list(distinctive or name_tokens)

    for token in _tokens_of(spec.get("description", "")):
        if (
            len(token) >= _MIN_CUE_LENGTH
            and token not in GENERIC_CUE_TOKENS
            and token not in _CUE_FUNCTION_WORDS
            and not token.isdigit()
        ):
            cues.append(token)

    seen = set()
    return [cue for cue in cues if not (cue in seen or seen.add(cue))]


def _cue_spans(text: str, cues: Sequence[str]) -> List[Tuple[int, int]]:
    spans = []
    for cue in cues:
        for match in re.finditer(r"\b" + re.escape(cue) + r"\b", text):
            spans.append((match.start(), match.end()))
    return spans


def _nearest_gap(
    spans: Sequence[Tuple[int, int]], start: int, end: int
) -> Optional[int]:
    best = None
    for cue_start, cue_end in spans:
        if cue_end <= start:
            gap = start - cue_end
        elif cue_start >= end:
            gap = cue_start - end
        else:
            gap = 0
        best = gap if best is None else min(best, gap)
    return best


def has_calculation_request(text: str) -> bool:
    """Whether the message explicitly asks for something to be computed."""
    return bool(_CALCULATION_REQUEST.search(_normalized(text)))


def is_prose(text: str) -> bool:
    """Whether the message reads as a question or an explanation request."""
    return bool(_INTERROGATIVE.search(_normalized(text)))


def is_compact_value_reply(
    text: str, number_spans: Sequence[Tuple[str, int, int]]
) -> bool:
    """Whether the message is essentially just the number(s) it contains.

    "42000 euro" and "sono 42000 euro" are answers to a question that has
    already been asked. "cosa dice l'articolo 40 del codice?" is not, and no
    number of it being short changes that.
    """
    if not number_spans or is_prose(text):
        return False
    masked = list(str(text or ""))
    for _raw, start, end in number_spans:
        masked[start:end] = " " * (end - start)
    residual = [
        token for token in _tokens_of("".join(masked))
        if token not in _REPLY_FILLER and not token.isdigit()
    ]
    return len(residual) <= _MAX_RESIDUAL_WORDS


def explicit_assignments(text: str, names: Iterable[str]) -> Dict[str, str]:
    """`field: value` pairs naming a declared field.

    A value runs to the next DECLARED field name rather than to the next comma,
    because Italian writes decimals with a comma: `prezzo: 0,25, gas: 1,10` has
    to yield "0,25" and "1,10", not "0" and "1".
    """
    declared = set(names)
    body = str(text or "")
    heads = [m for m in _ASSIGNMENT.finditer(body) if m.group(1) in declared]
    assignments = {}
    for index, head in enumerate(heads):
        end = heads[index + 1].start() if index + 1 < len(heads) else len(body)
        value = body[head.end():end].strip().rstrip(";,").strip()
        if value:
            assignments[head.group(1)] = value
    return assignments


def bind_numbers(
    text: str,
    number_specs: Sequence[Dict[str, Any]],
    number_spans: Sequence[Tuple[str, int, int]],
    *,
    already_bound: Iterable[str] = (),
) -> Dict[str, str]:
    """Bind numbers in `text` to `number_specs` by label.

    `number_spans` is (raw token, start, end) in `text`, in order of appearance.
    Returns {field: raw token} — the caller owns typing and locale parsing.

    Anchored bindings come first, each field taking the nearest unconsumed
    number within CUE_WINDOW of one of its cues. Only then is the narrow
    single-value fallback considered, and only when everything about it is
    unambiguous: one required field still missing, one number still unclaimed,
    and a message that reads as an answer rather than as a question.
    """
    bound: Dict[str, str] = {}
    consumed: set = set()
    lowered = _normalized(text)
    outstanding = set(already_bound)

    for spec in number_specs:
        name = spec.get("name")
        if not name or name in outstanding:
            continue
        spans = _cue_spans(lowered, cue_tokens(spec))
        if not spans:
            continue  # no label for this field in this sentence: do not guess
        best_index, best_gap = None, None
        for index, (_raw, start, end) in enumerate(number_spans):
            if index in consumed:
                continue
            gap = _nearest_gap(spans, start, end)
            if gap is None or gap > CUE_WINDOW:
                continue
            if best_gap is None or gap < best_gap:
                best_index, best_gap = index, gap
        if best_index is not None:
            consumed.add(best_index)
            bound[name] = number_spans[best_index][0]

    unclaimed = [i for i in range(len(number_spans)) if i not in consumed]
    missing_required = [
        spec for spec in number_specs
        if spec.get("name")
        and spec.get("required")
        and spec["name"] not in bound
        and spec["name"] not in outstanding
    ]
    if len(missing_required) == 1 and len(unclaimed) == 1:
        candidate = number_spans[unclaimed[0]][0]
        looks_like_a_year = bool(_BARE_YEAR.fullmatch(candidate.strip().rstrip("%")))
        if not looks_like_a_year and (
            is_compact_value_reply(text, number_spans)
            or (has_calculation_request(text) and not is_prose(text))
        ):
            bound[missing_required[0]["name"]] = candidate

    return bound
