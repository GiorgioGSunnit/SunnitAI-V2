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

v79.10 (A5): the safety valve -- a document the family rules would drop is admitted all the same through one of four
override doors, each logged (the gate's overrides list: {doc, article, door, signal, source}; interactive_flow logs
family_override per gate call, fam["overrides"], the counter family_overrides):

  «istituto»     a first-read institution names the document (its number/year, its code expression, an alias)
  «utente»       the user names the law (today's user_named) -- by stays «nominata dall'utente», as in v79.9
  «articolo»     the user names the article -- by stays «articolo nominato dall'utente»
  «nome intero»  the row is a first-read hit by the institution's whole name in its rubrica (_first_read and not
                 topic_groups.keyword_hit) -- the gate only, it needs the row

A row the family rules admit anyway is never an override. label_admitted (the candidates' last check) honours the
«istituto» and «utente» doors, so a candidate on an overridden document is not dropped later -- and (review) every door
the retrieval gate opened (carried_override: the gate's overrides in the method's ctx, the «nome intero» door and an
institution the method's own read_institutions does not hold included). The alarm: the rows the
gate dropped, graded in shadow after the main grading (FAMILY_GATE_SHADOW_GRADE, interactive_flow.family_shadow_grade;
a proxy count with the flag off) -- alarm() lists the documents with FAMILY_ALARM_MIN or more graded-relevant articles,
for the nightly («legge di disciplina tagliata»).

  decide(code, doc, ctx, row)   {ok, family, by, door, signal}: admitted() with the override door that admitted it
  rescued(row, ctx, named, …)   (review) the doors for a row special_laws.gate drops (a table law without its signal)
  institution_texts(items)      the first read's institutions as words (a read_institutions dict: name, rubricated,
                                claim)
  alarm(dropped_graded)         the documents with FAMILY_ALARM_MIN or more relevant (or, proxy, dropped) articles

v79.10 (A4, archive names): the archive's document names carry no law numbers. A document is a data/corpus_expected.json
law -- or a table entry with archive names (special_laws.exact) -- only by its exact name (corpus_laws.law_for: the
law's «archive_names», whitespace-normalised), or a label's code that is the law's own citation as a whole («l.
241/1990») or the sigla its document is labelled with («c.d.s.»): classify, expected_law, law_module (the table entry's
«expected» law among the module's), carried_override and the document side of user_named never read a number, an
expression or a substring of such a name («Legge 7 agosto 1990, n. 241» is not the L. 241/1990). The user's words are
read as before (the law's number/year, the table's expression, the aliases).

v79.10 (A4, review): a document's name is never read as a citation or a sigla -- a row with its document's name is
classified by that name's equality alone (a document named «Legge 604/1966» is no L. 604/1966, a «Codice della strada
commentato 2024» or a «Regolamento di esecuzione del codice della strada» no d.lgs. 285/1992); the citation and the
sigla «c.d.s.» are read on a bare label's code only (corpus_laws.cited: the abbreviated forms, never a cached
document name).
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
# v79.10 (A4, review): the sigla as the whole code only -- «codice della strada» inside a name is no archive name of it
_STRADA_CODE_RE = re.compile(r"^\s*c\.?\s*d\.?\s*s\.?\s*$", re.IGNORECASE)
OVERRIDE = "override: "                     # v79.10 (A5): by, for the «istituto» and «nome intero» doors
DOOR_INSTITUTION, DOOR_USER, DOOR_ARTICLE, DOOR_WHOLE_NAME = "istituto", "utente", "articolo", "nome intero"
FAMILY_ALARM_MIN = 5                        # v79.10 (A5): graded-relevant articles of one dropped document: the alarm


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


def table_law(text: str, name_only: bool = False, exact_entries: bool = True) -> Optional[Dict[str, Any]]:
    """The special-law table's law the text names -- its code expression, else its number and year. v79.10 (A4): an
    exact entry (special_laws.exact) by its archive names or its citation only, never by a number in a name; a number
    and year a corpus_expected law holds never names a document either (corpus_laws.reserved). v79.10 (A4, review):
    name_only -- the text is a document's name (an exact entry by its archive names only); exact_entries False -- a
    row's code beside its document's name (the name alone says whether the row is an exact entry's)."""
    from . import corpus_laws, special_laws
    law = special_laws.law_of(text, name_only=name_only, exact_entries=exact_entries)
    if law:
        return law
    return next((x for x in special_laws.laws() if not special_laws.exact(x)
                 and not corpus_laws.reserved(str(x.get("number") or ""), str(x.get("year") or ""))
                 and _by_number(text, str(x.get("number") or ""), str(x.get("year") or ""))), None)


def expected_law(text: str) -> str:
    """The data/corpus_expected.json law the text is (a module's law, mediazione), else "". v79.10 (A4): by its
    archive names (exact, whitespace-normalised), or the law's own citation as the whole text (corpus_laws.law_for)."""
    from . import corpus_laws
    return corpus_laws.law_for(text)


def _linked(exp: str) -> Optional[Dict[str, Any]]:
    """v79.10 (A4): the table entry that is the corpus_expected law («expected»), None when none."""
    from . import special_laws
    return next((x for x in special_laws.laws() if exp and x.get("expected") == exp), None)


def classify(code: Any, doc: Any = "") -> Tuple[str, str]:
    """(family, law): CODE for the codes; a table law's family and id; «modulo» / «mediazione» and the expected law
    for corpus_expected's; OTHER for any other special law. v79.10 (A4): the code and the document name each read on
    its own -- a corpus_expected law by its exact archive name (its table entry's family and id when the table has
    one), before the table's expressions. v79.10 (A4, review): a row with its document's name is that document -- a
    corpus_expected law or an exact table entry by the name's equality only (corpus_laws.law_of_document; never a
    citation or a number read on the name, never the code's sigla or citation beside it: «Legge 604/1966» is not the L.
    604/1966, a «c.d.s.» row of «Codice della strada commentato 2024» not the d.lgs. 285/1992); the code alone (a
    label: its archive name, the sigla «c.d.s.» as the whole code, a citation -- corpus_laws.law_for) only without one."""
    from . import corpus_laws
    code_t, doc_t = _text(code), _text(doc)
    if not doc_t and _STRADA_CODE_RE.search(code_t):
        return "strada", "codice_della_strada"
    if ordinary_code(code, doc):
        return CODE, ""
    exp = corpus_laws.law_of_document(doc_t) if doc_t else expected_law(code_t)
    if exp:
        law = _linked(exp)
        if law:
            return str(law.get("family") or OTHER), str(law["id"])
        return ("mediazione" if exp == "d.lgs. 28/2010" else "modulo"), exp
    if doc_t:       # the code beside the name: the table's other entries; the name: an exact entry by itself only
        tries = ([(code_t, False, False)] if code_t and code_t != doc_t else []) + [(doc_t, True, True)] + (
            [(_text(code, doc), True, False)] if code_t and code_t != doc_t else [])
    else:
        tries = [(code_t, False, True)] if code_t else []
    for part, name_only, exact_entries in tries:
        law = table_law(part, name_only=name_only, exact_entries=exact_entries)
        if law:
            return str(law.get("family") or OTHER), str(law["id"])
    return OTHER, ""


def institution_texts(items: Sequence[Any]) -> List[str]:
    """v79.10 (A5): the first read's institutions as words -- a string as it is; a read_institutions dict (interactive_
    flow) by its name, its rubricated name and its claim, each once."""
    out: List[str] = []
    for x in items or []:
        parts = [x.get(k) for k in ("name", "rubricated", "claim")] if isinstance(x, dict) else [x]
        for part in parts:
            text = " ".join(str(part or "").split())
            if text and text not in out:
                out.append(text)
    return out


def context(case: str, words: str = "", institutions: Sequence[Any] = ()) -> Dict[str, Any]:
    """The case's family context: the words the signals are read in (the first read's, else the case's), the active
    modules' laws, the professional counterparty (B3), the case's own text (a law the user names). v79.10 (A5): the
    first read's institutions as words («institutions»: the «istituto» override door; a read_institutions dict is read
    by its name, rubricated name and claim -- v79.9 joined the dicts themselves)."""
    from . import lawyer_method, special_laws
    insts = institution_texts(institutions)
    mctx = {"case": case or "", "institutions": insts}
    mods = lawyer_method.active_modules(mctx)
    return {"case": case or "", "words": words or case or "", "signals": special_laws.signals(words or case or ""),
            "module_laws": sorted({x for m in mods for x in m.get("laws") or []}),
            "modules": [m["id"] for m in mods], "professional": lawyer_method.professional_counterparty(mctx),
            "institutions": insts}


def _signal_hit(law_id: str, words: str) -> str:
    """The words of the case that hold the law's signal (the nightly's «admitted by»)."""
    from . import special_laws
    law = next((x for x in special_laws.laws() if x["id"] == law_id), None)
    if not law:
        return ""
    hits = [m.group(0) for m in (r.search(words or "") for r in law["_signals"]) if m]
    if not (law["_signals"] and len(hits) == len(law["_signals"])):       # v79.10 (A4): a family's signal
        hits = next((got for got in (special_laws.family_signal(f, words or "") for f in law.get("_families") or [])
                     if got), [])
    return ", ".join(f"«{h}»" for h in hits)


def user_named(family: str, law_id: str, law: str, ctx: Dict[str, Any]) -> bool:
    return bool(naming(ctx.get("case") or "", law_id, law, ctx.get("_doc") or ""))


def naming(text: str, law_id: str, law: str, doc: str, aliases: bool = False) -> str:
    """v79.10 (A5): the words of the text that name the document, "" when none -- a table law by its code expression or
    its number/year; a corpus_expected law by its number/year; any other by the document's own number/year. aliases:
    a corpus_expected alias too (the law's, or the table law's of the same number/year) -- the «istituto» door."""
    from . import special_laws
    if not text:
        return ""
    number = year = ""
    if law_id:
        entry = next((x for x in special_laws.laws() if x["id"] == law_id), None)
        if entry:
            m = entry["_code"].search(text)
            if m:
                return m.group(0)
            number, year = str(entry.get("number") or ""), str(entry.get("year") or "")
            if _by_number(text, number, year):
                return f"{number}/{year}"
    if law:
        from . import corpus_laws
        spec = (corpus_laws.expected().get("laws") or {}).get(law) or {}
        number, year = str(spec.get("number") or ""), str(spec.get("year") or "")
        if _by_number(text, number, year):
            return f"{number}/{year}"
    else:                                   # v79.9: a table law not named by its entry -- the document's own number
        from . import corpus_laws           # v79.10 (A4): never a corpus_expected law's number in another name
        own, own_year = own_number(doc)
        if corpus_laws.reserved(own, own_year):
            own = own_year = ""
        if _by_number(text, own, own_year):
            return f"{own}/{own_year}"
        number, year = (number, year) if number else (own, own_year)
    if aliases and number and year:
        low = text.lower()
        for alias in _aliases(law, number, year):
            if alias and alias in low:
                return alias
    return ""


def _aliases(law: str, number: str, year: str) -> List[str]:
    """The corpus_expected aliases of the law (or of the expected law with the same number and year), lower case."""
    from . import corpus_laws
    laws = corpus_laws.expected().get("laws") or {}
    specs = [laws[law]] if law in laws else [s for s in laws.values()
                                            if str(s.get("number") or "") == number and str(s.get("year") or "") == year]
    return [str(a).lower() for s in specs for a in s.get("aliases") or []]


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
    """(admitted, family, by): by -- «codice», «default», the module, the signal's words, «nominata dall'utente»;
    v79.10 (A5): «override: …» for a first-read institution naming the document (decide)."""
    d = decide(code, doc, ctx)
    return d["ok"], d["family"], d["by"]


def decide(code: Any, doc: Any, ctx: Dict[str, Any], row: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """v79.10 (A5): {ok, family, by, door, signal} -- the family rules (the codes, the defaults, a module's law, a table
    law's signal); a document they refuse admitted through an override door: the user naming the law («utente», by
    «nominata dall'utente» as in v79.9 -- which also wins the by when the rules admit it, then no override), a first-read
    institution naming it («istituto»: ctx["institutions"]), the row a first-read whole-name hit («nome intero»: row).
    door "" when the family rules admitted it (or nothing did)."""
    family, law = classify(code, doc)
    out = {"ok": True, "family": family, "by": CODE, "door": "", "signal": ""}
    if family == CODE:
        return out
    is_table = bool(law) and law not in (corpus_expected_laws())
    law_id, exp_law, text = (law if is_table else ""), ("" if is_table else law), _text(code, doc)
    rule = _family_rule(family, law, is_table, ctx)
    said = naming(ctx.get("case") or "", law_id, exp_law, text)
    if said:
        return {**out, "by": "nominata dall'utente", "door": "" if rule else DOOR_USER, "signal": "" if rule else said}
    if rule:
        return {**out, "by": rule}
    for inst in ctx.get("institutions") or []:
        got = naming(str(inst), law_id, exp_law, text, aliases=True)
        if got:
            signal = f"«{got}» in «{inst}»" if got != inst else f"«{inst}»"
            return {**out, "by": OVERRIDE + signal, "door": DOOR_INSTITUTION, "signal": signal}
    if row is not None and row.get("_first_read"):
        from . import topic_groups
        if not topic_groups.keyword_hit(row):
            signal = f"«{row['_first_read']}»"
            return {**out, "by": OVERRIDE + signal, "door": DOOR_WHOLE_NAME, "signal": signal}
    return {**out, "ok": False, "by": ""}


def _family_rule(family: str, law: str, is_table: bool, ctx: Dict[str, Any]) -> str:
    """The family rule admitting the document (its by), "" when none: a default family (consumo with a professional
    counterparty), an active module's law, a table law's signal or its number/year among the module's laws."""
    if family in DEFAULT_FAMILIES:
        if family == "consumo" and not ctx.get("professional"):
            return ""
        return "default"
    if not is_table and law and law in (ctx.get("module_laws") or []):
        return "modulo " + ", ".join(ctx.get("modules") or [])
    if is_table and (ctx.get("signals") or {}).get(law):
        return "segnale " + (_signal_hit(law, ctx.get("words") or "") or "nei fatti")
    if is_table and law_module(law, ctx):
        return "modulo " + ", ".join(ctx.get("modules") or [])
    return ""


def corpus_expected_laws() -> List[str]:
    from . import corpus_laws
    return list((corpus_laws.expected().get("laws") or {}).keys())


def law_module(law_id: str, ctx: Dict[str, Any]) -> bool:
    """A table law that is also an active module's law. v79.10 (A4): the entry's «expected» law (the corpus_expected
    law it is) among the module's laws -- never by number and year."""
    from . import special_laws
    entry = next((x for x in special_laws.laws() if x["id"] == law_id), None)
    return bool(entry and entry.get("expected")) and entry["expected"] in (ctx.get("module_laws") or [])


def gate(rows: Sequence[Dict[str, Any]], ctx: Dict[str, Any], named: set, source: str = "retrieval",
         overrides: Optional[List[Dict[str, Any]]] = None
         ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """(kept, dropped, admitted outside the codes): a row whose document the case's families do not admit goes, unless
    the user names its article (named: keys). v79.10 (A5): or an override door admits it (decide; the article door
    after the user's and the institution's, the whole-name door last) -- each appended to overrides {doc, article,
    door, signal, source}; a row the family rules admit is never one."""
    from .interactive_flow import _key, _label
    kept, dropped, outside = [], [], []
    for r in rows or []:
        ref = r.get("_article")
        if not ref or len(ref) != 2:
            kept.append(r)
            continue
        doc = str((r.get("d") or {}).get("name") or "")
        d = decide(ref[1], doc, ctx, None if _key(ref) in named else r)
        ok, family, by = d["ok"], d["family"], d["by"]
        door, signal = d["door"], d["signal"]
        if not ok and _key(ref) in named:
            ok, by, door, signal = True, "articolo nominato dall'utente", DOOR_ARTICLE, f"«{_label(ref)}»"
        if ok and door and overrides is not None:
            item = {"doc": doc or str(ref[1]), "article": _label(ref), "door": door, "signal": signal, "source": source}
            if item not in overrides:
                overrides.append(item)
        if not ok:
            dropped.append({"article": _label(ref), "doc": doc or str(ref[1]), "family": family, "source": source})
            continue
        if family != CODE:
            item = {"doc": doc or str(ref[1]), "family": family, "by": by}
            if item not in outside:
                outside.append(item)
        kept.append(r)
    return kept, dropped, outside


def rescued(row: Dict[str, Any], ctx: Dict[str, Any], named: set, source: str = "special_law_gate",
            overrides: Optional[List[Dict[str, Any]]] = None) -> bool:
    """v79.10 (A5, review): a row another gate drops (special_laws.gate: a table law without its signal, which runs
    before this gate) kept all the same when an override door admits its document (gate(): decide's doors) -- appended
    to overrides like gate()'s, source the other gate's. A row only the family rules would admit is not rescued: the
    special-law gate's own rule stands, the safety valve is the doors."""
    got: List[Dict[str, Any]] = []
    kept, _, _ = gate([row], ctx, named, source, got)
    if not (kept and got):
        return False
    if overrides is not None:
        overrides += [x for x in got if x not in overrides]
    return True


def label_admitted(label: str, ctx: Dict[str, Any], named: Sequence[str] = ()) -> Tuple[bool, str, str]:
    """A candidate's article label («117 d.lgs. 385/1993», «1453 c.c.») -- admitted as gate() would. v79.10 (A5): the
    «utente» and «istituto» override doors included (admitted -> decide): a candidate on an overridden document stays."""
    if label in (named or ()):
        return True, "", "articolo nominato dall'utente"
    number, _, code = str(label or "").partition(" ")
    code = _LABEL_TAIL_RE.sub("", _LABEL_HEAD_RE.sub("", code))     # «61 n. 11 c.p.», «2051 c.c. co. 2»
    got = admitted(code, "", ctx)
    if not got[0]:                                                  # v79.10 (A5, review): the retrieval's doors
        carried = carried_override(number, code, ctx)
        if carried:
            return True, got[1], carried
    return got


def carried_override(number: str, code: str, ctx: Dict[str, Any]) -> str:
    """v79.10 (A5, review): the retrieval gate's override that admitted the label's document (ctx["overrides"]: the
    gate's overrides list, state["family_gate"]["overrides"], carried into the method's ctx by lawyer_method.
    family_context) -- its by, "" when none. A candidate on an article the retrieval admitted through a door is not out
    of family: the «nome intero» door (it needs the row, label_admitted has none) and an institution only interactive_
    flow.read_institutions reads (the alternatives, the rubricated names). Matched by the document's code; the «articolo»
    door by the article too (the user named that article only)."""
    from . import corpus_laws
    from .topic_step import _code
    want, own = _code(code), own_number(code)
    law = corpus_laws.law_for(code)         # v79.10 (A4): a corpus_expected law -- by its archive names only
    if law or corpus_laws.reserved(*own):
        own = ("", "")
    for o in ctx.get("overrides") or []:
        if not want or not isinstance(o, dict):
            continue
        num, _, got = str(o.get("article") or "").partition(" ")
        same = _code(got) == want or (all(own) and own_number(str(o.get("doc") or got)) == own)   # «l. 184/1983»
        if not same and law:            # v79.10 (A4, review): the override's document by its name only
            doc = corpus_laws.norm(o.get("doc"))
            same = law == (corpus_laws.law_of_document(doc) if doc else corpus_laws.law_for(got))
        if not same or (o.get("door") == DOOR_ARTICLE and num != str(number)):
            continue
        return {DOOR_USER: "nominata dall'utente", DOOR_ARTICLE: "articolo nominato dall'utente"}.get(
            str(o.get("door") or ""), OVERRIDE + str(o.get("signal") or ""))
    return ""


_LABEL_HEAD_RE = re.compile(r"^(?:(?:n\.|co\.|comma|lett\.)\s*[\w-]+,?\s*)+", re.IGNORECASE)
_LABEL_TAIL_RE = re.compile(r"\s+co\.\s*\d+.*$")


def alarm(dropped_graded: Dict[str, Any], minimum: int = FAMILY_ALARM_MIN) -> List[Dict[str, Any]]:
    """v79.10 (A5): the documents the gate dropped that held minimum or more graded-relevant articles (grade >= 2, the
    shadow grading) -- «we just cut the governing law»; a proxy entry (FAMILY_GATE_SHADOW_GRADE off: the distinct
    dropped articles, never graded) counts its dropped articles, marked proxy. Each {doc, n, articles, proxy}, the
    largest first."""
    out = []
    for doc, x in (dropped_graded or {}).items():
        if not isinstance(x, dict):
            continue
        proxy = bool(x.get("proxy"))
        n = int((x.get("dropped_articles") if proxy else x.get("graded_relevant")) or 0)
        if n >= minimum:
            out.append({"doc": doc, "n": n, "articles": list(x.get("articles") or []), "proxy": proxy})
    return sorted(out, key=lambda a: -a["n"])
