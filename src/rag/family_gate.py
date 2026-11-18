"""v79.9 (Block C): the document-level allowlist per matter family -- every candidate source (retrieval, the first
read's institutions, the rules' anchors, the ranked candidates, the riqualificazione, «altri profili») admits a document
only by its family:

  the codes (c.c., c.p., c.p.c., c.p.p., c.p.a., the Constitution, the preleggi and disposizioni di attuazione)  always
  the defaults (strada, stupefacenti, mediazione; consumo with a professional counterparty, Block B3)            always
  a law of the special-law table (data/special_laws.json) with its family's signal in the case's words           yes
  an active institution module's laws (data/institution_modules.json «laws», data/corpus_expected.json)          yes
  a law the user names (its number / year or its name in the case), an article the user names                   yes
  anything else -- a banking, finance, antiriciclaggio, data-protection, insurance, DORA, Banca d'Italia, schools,
  nautica or public-contracts document without its signal, another special law no module names                  no

  classify(code, doc)        (family, law id or ""): what the document is
  context(case, read_words, institutions)   the case's family context: its words, its modules' laws, the professional
                             counterparty
  admitted(code, doc, ctx)   (admitted, family, the signal or rule that admitted it)
  gate(rows, ctx, named)     (the rows admitted, the rows dropped -- each {article, doc, family}, the admitted documents
                             outside the codes -- each {doc, family, by})

No case-specific logic: the tables hold the laws, their families and signals.
"""

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

CODE = "codice"
OTHER = "altra legge speciale"
DEFAULT_FAMILIES = ("strada", "stupefacenti", "mediazione", "consumo")
_ORDINARY_CODES = ("c.c.", "c.p.", "c.p.c.", "c.p.p.", "c.p.a.")
_CODE_TEXT_RE = re.compile(r"^\s*(?:cost\.?|costituzione|disp\.?\s*(?:att|prel|trans)|preleggi|disposizioni\s+(?:sulla\s+legge"
                           r"\s+in\s+generale|di\s+attuazione|transitorie)|codice\s+(?:civile|penale|di\s+procedura|del\s+processo"
                           r"\s+amministrativo))", re.IGNORECASE)
_STRADA_CODE_RE = re.compile(r"^\s*c\.?\s*d\.?\s*s\.?\s*$|codice\s+della\s+strada", re.IGNORECASE)


def _text(*parts: Any) -> str:
    return " ".join(" ".join(str(p or "").split()) for p in parts if p)


def _by_number(text: str, number: str, year: str) -> bool:
    if not number or not year:
        return False
    low = text.lower()
    return bool(re.search(rf"(?<!\d){re.escape(number)}\s*/\s*{re.escape(year)}(?!\d)", low)) or (
        year in low and bool(re.search(rf"\b(?:n\.?\s*º?\s*|numero\s+){re.escape(number)}\b", low)))


def ordinary_code(code: Any, doc: Any = "") -> bool:
    from .plan_then_write import _code_key
    if _code_key(" ".join(str(code or "").split())) in _ORDINARY_CODES:
        return True
    return any(_CODE_TEXT_RE.search(str(x or "")) for x in (code, doc))


def table_law(text: str) -> Optional[Dict[str, Any]]:
    """The special-law table's law the text names -- its code expression, else its number and year."""
    from . import special_laws
    law = special_laws.law_of(text)
    if law:
        return law
    return next((x for x in special_laws.laws() if _by_number(text, str(x.get("number") or ""),
                                                             str(x.get("year") or ""))), None)


def expected_law(text: str) -> str:
    """The data/corpus_expected.json law the text names (a module's law, mediazione), else ""."""
    from . import corpus_laws
    return next((law for law in (corpus_laws.expected().get("laws") or {}) if corpus_laws.present(law, [text])), "")


def classify(code: Any, doc: Any = "") -> Tuple[str, str]:
    """(family, law): CODE for the codes; a table law's family and id; «modulo» / «mediazione» and the expected law
    for corpus_expected's; OTHER for any other special law."""
    if _STRADA_CODE_RE.search(str(code or "")):
        return "strada", "codice_della_strada"
    if ordinary_code(code, doc):
        return CODE, ""
    text = _text(code, doc)
    law = table_law(text)
    if law:
        return str(law.get("family") or OTHER), str(law["id"])
    exp = expected_law(text)
    if exp:
        return ("mediazione" if exp == "d.lgs. 28/2010" else "modulo"), exp
    return OTHER, ""


def context(case: str, words: str = "", institutions: Sequence[str] = ()) -> Dict[str, Any]:
    """The case's family context: the words the signals are read in (the first read's, else the case's), the active
    modules' laws, the professional counterparty (B3), the case's own text (a law the user names)."""
    from . import lawyer_method, special_laws
    mctx = {"case": case or "", "institutions": list(institutions or [])}
    mods = lawyer_method.active_modules(mctx)
    return {"case": case or "", "words": words or case or "", "signals": special_laws.signals(words or case or ""),
            "module_laws": sorted({x for m in mods for x in m.get("laws") or []}),
            "modules": [m["id"] for m in mods], "professional": lawyer_method.professional_counterparty(mctx)}


def _signal_hit(law_id: str, words: str) -> str:
    """The words of the case that hold the law's signal (the nightly's «admitted by»)."""
    from . import special_laws
    law = next((x for x in special_laws.laws() if x["id"] == law_id), None)
    if not law:
        return ""
    hits = [r.search(words or "") for r in law["_signals"]]
    return ", ".join(f"«{m.group(0)}»" for m in hits if m)


def user_named(family: str, law_id: str, law: str, ctx: Dict[str, Any]) -> bool:
    from . import special_laws
    case = ctx.get("case") or ""
    if law_id:
        entry = next((x for x in special_laws.laws() if x["id"] == law_id), None)
        if entry and (entry["_code"].search(case) or _by_number(case, str(entry.get("number") or ""),
                                                                str(entry.get("year") or ""))):
            return True
    if law:
        from . import corpus_laws
        spec = (corpus_laws.expected().get("laws") or {}).get(law) or {}
        return _by_number(case, str(spec.get("number") or ""), str(spec.get("year") or ""))
    number, year = own_number(ctx.get("_doc") or "")
    return _by_number(case, number, year)


_OWN_NUMBER_RE = re.compile(r"\b(\d{1,4})\s*/\s*((?:19|20)\d{2})\b|\bn\.?\s*º?\s*(\d{1,4})\b")
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")


def own_number(doc: str) -> Tuple[str, str]:
    """A document's own number and year («Legge 4 maggio 1983, n. 184» -> 184, 1983; «D.lgs. 297/1994»)."""
    m = _OWN_NUMBER_RE.search(doc or "")
    if not m:
        return "", ""
    if m.group(1):
        return m.group(1), m.group(2)
    year = _YEAR_RE.search(doc or "")
    return m.group(3), year.group(0) if year else ""


def admitted(code: Any, doc: Any, ctx: Dict[str, Any]) -> Tuple[bool, str, str]:
    """(admitted, family, by): by -- «codice», «default», the module, the signal's words, «nominata dall'utente»."""
    family, law = classify(code, doc)
    if family == CODE:
        return True, family, CODE
    is_table = bool(law) and law not in (corpus_expected_laws())
    if user_named(family, law if is_table else "", "" if is_table else law, {**ctx, "_doc": _text(code, doc)}):
        return True, family, "nominata dall'utente"
    if family in DEFAULT_FAMILIES:
        if family == "consumo" and not ctx.get("professional"):
            return False, family, ""
        return True, family, "default"
    if not is_table and law and law in (ctx.get("module_laws") or []):
        return True, family, "modulo " + ", ".join(ctx.get("modules") or [])
    if is_table and (ctx.get("signals") or {}).get(law):
        return True, family, "segnale " + (_signal_hit(law, ctx.get("words") or "") or "nei fatti")
    if is_table and law_module(law, ctx):
        return True, family, "modulo " + ", ".join(ctx.get("modules") or [])
    return False, family, ""


def corpus_expected_laws() -> List[str]:
    from . import corpus_laws
    return list((corpus_laws.expected().get("laws") or {}).keys())


def law_module(law_id: str, ctx: Dict[str, Any]) -> bool:
    """A table law that is also an active module's law (by number and year)."""
    from . import special_laws
    entry = next((x for x in special_laws.laws() if x["id"] == law_id), None)
    if not entry or not entry.get("number"):
        return False
    tag = f"{entry['number']}/{entry['year']}"
    return any(tag in x for x in ctx.get("module_laws") or [])


def gate(rows: Sequence[Dict[str, Any]], ctx: Dict[str, Any], named: set, source: str = "retrieval"
         ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """(kept, dropped, admitted outside the codes): a row whose document the case's families do not admit goes, unless
    the user names its article (named: keys)."""
    from .interactive_flow import _key, _label
    kept, dropped, outside = [], [], []
    for r in rows or []:
        ref = r.get("_article")
        if not ref or len(ref) != 2:
            kept.append(r)
            continue
        doc = str((r.get("d") or {}).get("name") or "")
        ok, family, by = admitted(ref[1], doc, ctx)
        if not ok and _key(ref) in named:
            ok, by = True, "articolo nominato dall'utente"
        if not ok:
            dropped.append({"article": _label(ref), "doc": doc or str(ref[1]), "family": family, "source": source})
            continue
        if family != CODE:
            item = {"doc": doc or str(ref[1]), "family": family, "by": by}
            if item not in outside:
                outside.append(item)
        kept.append(r)
    return kept, dropped, outside


def label_admitted(label: str, ctx: Dict[str, Any], named: Sequence[str] = ()) -> Tuple[bool, str, str]:
    """A candidate's article label («117 d.lgs. 385/1993», «1453 c.c.») -- admitted as gate() would."""
    if label in (named or ()):
        return True, "", "articolo nominato dall'utente"
    number, _, code = str(label or "").partition(" ")
    code = _LABEL_TAIL_RE.sub("", _LABEL_HEAD_RE.sub("", code))     # «61 n. 11 c.p.», «2051 c.c. co. 2»
    return admitted(code, "", ctx)


_LABEL_HEAD_RE = re.compile(r"^(?:(?:n\.|co\.|comma|lett\.)\s*[\w-]+,?\s*)+", re.IGNORECASE)
_LABEL_TAIL_RE = re.compile(r"\s+co\.\s*\d+.*$")
