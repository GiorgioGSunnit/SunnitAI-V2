"""Latin article suffixes (bis, ter, ... 452-quaterdecies), every spelling the
codes use, normalised to one canonical form. One table, two identical copies:
SunnitAI-BE lex_package/parsing_utils/article_suffix.py (the parsers accept
every spelling and write the canonical one into the Section names) and
SunnitAI-V2 src/rag/article_suffix.py (which normalises the numbers it looks
up the same way). Change both together.

Canonical forms are the ones current consolidated texts print (L. 241/1990
"21-nonies", c.c. "2545-octiesdecies", c.p. "452-quaterdecies"). ORDINAL is
each suffix's position in the bis/ter/... sequence (the parser's article-order
check); number_variants() also lists the legacy spellings, for Sections
ingested before normalisation.
"""

import re
from typing import List

# canonical -> other spellings; two-word forms ("ter decies", "vicies semel": Codice
# del Consumo 2026, art. 67-ter decies ... 67-vicies bis) match any whitespace.
_VARIANTS = {
    "nonies": ("novies",),
    "terdecies": ("ter decies",),
    "quaterdecies": ("quattuordecies", "quater decies"),
    "quinquiesdecies": ("quindecies", "quinquies decies"),
    "sexiesdecies": ("sexdecies", "sexies decies"),
    "septiesdecies": ("septendecies", "septies decies"),
    "octiesdecies": ("octodecies", "octies decies"),
    "noviesdecies": ("novendecies", "noniesdecies", "novies decies", "nonies decies"),
    "unvicies": ("vicies semel",),
    "duovicies": ("vicies bis",),
    "tervicies": ("vicies ter",),
}
_PLAIN = ("bis", "ter", "quater", "quinquies", "sexies", "septies", "octies", "decies", "undecies",
          "duodecies", "vicies", "quattuorvicies", "quinvicies", "sexvicies", "septenvicies", "octovicies",
          "novenvicies", "tricies")

CANONICAL = {s: s for s in _PLAIN}
for _canonical, _others in _VARIANTS.items():
    CANONICAL[_canonical] = _canonical
    for _other in _others:
        CANONICAL[_other] = _canonical

# Position in the sequence (no suffix = 1, bis = 2, ...), for the article-order check.
_ORDER = ("bis", "ter", "quater", "quinquies", "sexies", "septies", "octies", "nonies", "decies", "undecies",
          "duodecies", "terdecies", "quaterdecies", "quinquiesdecies", "sexiesdecies", "septiesdecies",
          "octiesdecies", "noviesdecies", "vicies", "unvicies", "duovicies", "tervicies", "quattuorvicies",
          "quinvicies", "sexvicies", "septenvicies", "octovicies", "novenvicies", "tricies")
ORDINAL = {s: _ORDER.index(c) + 2 for s, c in CANONICAL.items()}

# Longest first, so "ter" never wins over "terdecies" nor "quater" over "quaterdecies".
SUFFIX = "(?:" + "|".join(s.replace(" ", r"\s+") for s in sorted(CANONICAL, key=len, reverse=True)) + ")"
_NUMBER_RE = re.compile(rf"^(\d+)\s*-?\s*({SUFFIX})(?![a-z])", re.IGNORECASE)


def canonical_suffix(suffix: str) -> str:
    key = " ".join((suffix or "").lower().split())
    return CANONICAL.get(key, key)


def normalize_number(raw: str) -> str:
    """'415bis' / '415-BIS' / '21-novies' / '2051' -> '415-bis' / '415-bis' / '21-nonies' / '2051'."""
    text = (raw or "").strip().lower()
    m = _NUMBER_RE.match(text)
    if m:
        return f"{m.group(1)}-{canonical_suffix(m.group(2))}"
    m = re.match(r"^\d+", text)
    return m.group(0) if m else text


def number_variants(number: str) -> List[str]:
    """The canonical number first, then the same number with each legacy
    spelling of its suffix (for Sections ingested before normalisation)."""
    canonical = normalize_number(number)
    m = re.match(r"^(\d+)-([a-z]+)$", canonical)
    if not m:
        return [canonical]
    return [canonical] + [f"{m.group(1)}-{s}" for s in _VARIANTS.get(m.group(2), ()) if " " not in s]
