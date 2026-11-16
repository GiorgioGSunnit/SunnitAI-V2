"""v75: abbreviations never end a sentence -- «il Sig. Caio», «Dott. Rossi», «Avv. Bianchi», «Alfa S.p.A. Il giorno»,
«art. 640», a name's initial («Tizio R. Caio»). Every splitter of the case's text (issue_records.sentences, fact_signals.
case_only, intake.sentences) splits as before, then rejoins a piece that ends on one with the next (rejoin)."""

import re
from typing import List, Sequence

ABBREVIATIONS = ("sig", "sigg", "sig.ra", "sig.na", "dott", "dott.ssa", "avv", "prof", "prof.ssa", "ing", "geom", "rag",
                 "arch", "spett", "egr", "gent", "gent.mo", "on", "mons", "art", "artt", "n", "nn", "co", "lett", "cfr",
                 "cass", "sez", "pag", "pagg", "ss", "sent", "ord", "rif", "tel", "vol", "cap", "cod", "proc", "trib",
                 "app", "c.d")
_ABBREV_END_RE = re.compile(r"(?:(?<![\w.])(?:" + "|".join(re.escape(a) for a in ABBREVIATIONS) + r")\.|"
                            r"(?<![\w.])[A-Z]\.|\b(?:[A-Za-z]\.){2,}[A-Za-z]?\.?)$", re.IGNORECASE)


def ends_on_abbreviation(piece: str) -> bool:
    return bool(_ABBREV_END_RE.search((piece or "").rstrip()))


def rejoin(pieces: Sequence[str], sep: str = " ") -> List[str]:
    """The pieces with each one that ends on an abbreviation joined to the next."""
    out: List[str] = []
    carry = ""
    for i, piece in enumerate(pieces):
        piece = f"{carry}{sep}{piece}" if carry else piece
        if ends_on_abbreviation(piece) and i < len(pieces) - 1:
            carry = piece
            continue
        carry = ""
        out.append(piece)
    return out


def split(text: str, pattern: "re.Pattern[str]") -> List[str]:
    """re.split by the caller's own pattern (on one line), then rejoin."""
    return rejoin([x for x in pattern.split(text or "") if x is not None])
