"""v79.9 (B2, B4): the special laws each institution module needs, checked against the graph's primary documents.

data/corpus_expected.json lists, per module, the laws expected (the same list as the module's «laws» in
data/institution_modules.json). The names are read once (warm(), at the API's startup) and cached for the process; a
case whose module has a law missing gets one fixed line in §1 and §9 (lawyer_method.MISSING_LAWS_LINE). When the graph
could not be read, nothing is said to be missing (missing() is None).

v79.10 (A4, archive names): the archive's document names carry no law numbers («Legge sul Procedimento Amministrativo
2026»). Each law maps to «archive_names», the exact names of its primary documents (a list: an edition changes the
name); a document is that law only when its name equals one of them, whitespace-normalised -- never by number,
expression or substring («Legge 7 agosto 1990, n. 241» is no archive name of the L. 241/1990). A law with
«archive_names» [] is unmapped: always missing, never guessed. «codes»: the codes a module names by sigla (the c.p.a.),
mapped the same way. «number», «year» and «aliases» only read the user's own words (family_gate.naming).

  norm(name)               a name whitespace-normalised (the only normalisation: case and punctuation stay)
  archive_names(key)       a law's (or a code's) archive names, normalised; [] when unmapped
  present(key, names)      one of the names is one of its archive names
  law_of_name(name)        the law (or code) whose archive names hold the name, "" when none
  law_of_document(name)    a document's law -- its exact archive name only (never a sigla, never a citation)
  law_for(text)            a label's code: law_of_name, the sigla, else the law the text cites as a whole («l.
                           241/1990», «d.P.R. 309/1990»), "" when none
  status(key, names)       «present» / «NOT IN ARCHIVE» (mapped, none of its names read) / «UNMAPPED»
  citation_of(name)        v79.12.5 (item 8): a document's legal citation -- the law (or code) whose archive names hold
                           its name exactly («Testo Unico sugli Stupefacenti 2026» -> «d.P.R. 309/1990»), "" when none
  cite_titles(text)        v79.12.5 (item 8): the text with every mapped document title -- its archive name, or the
                           bracketed name without its year article_codes.law_label prints -- in its legal citation

scripts/corpus_gaps.py --env .env prints, per module and for the defaults, each law -> its archive names -> its status.

v79.10 (A4, review): a citation is read on a label's code only (an action, a candidate's label) -- in the abbreviated
forms the modules and the pipeline write («l.», «d.lgs.», «d.l.», «d.P.R.»; «Legge 604/1966» is a name's shape, not
one) and never on a text that is a primary document's name (the cached names): a document is matched by its name's
equality only, so a document named like a citation is never a law by its number -- an unmapped law is never guessed.
"""

import json
import logging
import os
import re
import itertools
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

EXPECTED_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "corpus_expected.json")
NAMES_QUERY = "MATCH (d:Document) WHERE d.document_type = 'primary' RETURN d.name AS name"
PRESENT, NOT_IN_ARCHIVE, UNMAPPED = "present", "NOT IN ARCHIVE", "UNMAPPED"
# v79.10 (A4, archive names): a citation of a law as the whole text («L. 241/1990», «d.lgs. 23/2015», «d.P.R. 309/1990»)
# v79.10 (A4, review): the abbreviated forms only -- «Legge 604/1966», «Decreto legislativo n. 23/2015» are a name's shape
_CITATION_RE = re.compile(r"^(?:l\.?|d\.?\s*lgs\.?|d\.?\s*l\.?|d\.?\s*p\.?\s*r\.?)\s*(?:n\.?\s*)?(\d{1,4})\s*/\s*"
                          r"((?:19|20)\d{2})$", re.IGNORECASE)

_EXPECTED: Dict[str, Any] = {}
_NAMES: Dict[str, Any] = {}
_FOR: Dict[Tuple[int, int, str], str] = {}
_LOCK = threading.Lock()
_VERSIONS = itertools.count(1)          # v79.10 (A4, review): law_for's cache key -- the names cache it was read under


def expected() -> Dict[str, Any]:
    """data/corpus_expected.json (read once; {} when missing, logged)."""
    if "d" not in _EXPECTED:
        try:
            with open(EXPECTED_FILE, encoding="utf-8") as f:
                _EXPECTED["d"] = json.load(f)
        except (OSError, ValueError) as exc:
            logger.warning("corpus_laws: %s not read: %s", EXPECTED_FILE, exc)
            _EXPECTED["d"] = {}
    return _EXPECTED["d"]


def norm(name: Any) -> str:
    """v79.10 (A4): a document name whitespace-normalised -- the only normalisation of the exact match."""
    return " ".join(str(name or "").split())


def _spec(key: str) -> Dict[str, Any]:
    exp = expected()
    return (exp.get("laws") or {}).get(key) or (exp.get("codes") or {}).get(key) or {}


def archive_names(key: str) -> List[str]:
    """v79.10 (A4): the law's (or the code's) archive names, normalised; [] when unmapped."""
    return [n for n in (norm(x) for x in _spec(key).get("archive_names") or []) if n]


def present(law: str, names: Sequence[str]) -> bool:
    """A law is in the graph: one of the names is one of its archive names (v79.10: exact, whitespace-normalised)."""
    want = set(archive_names(law))
    return bool(want) and any(norm(n) in want for n in names or [])


def law_of_name(name: Any) -> str:
    """v79.10 (A4): the corpus_expected law -- else code -- whose archive names hold the name, "" when none."""
    text = norm(name)
    if not text:
        return ""
    exp = expected()
    for table in (exp.get("laws") or {}, exp.get("codes") or {}):
        for key in table:
            if text in archive_names(key):
                return key
    return ""


def cited(text: Any) -> Tuple[str, str]:
    """v79.10 (A4): (number, year) of a law cited as the whole text («L. 241/1990»), ("", "") otherwise -- a document's
    name is never one («Legge 7 agosto 1990, n. 241» is not). v79.10 (A4, review): the abbreviated forms only, and
    never a primary document's name (document_name: «L. 604/1966» as the name of a document read by the graph)."""
    text = norm(text)
    m = _CITATION_RE.match(text)
    return (m.group(1), m.group(2)) if m and not document_name(text) else ("", "")


def document_name(text: Any) -> bool:
    """v79.10 (A4, review): the text is a primary document's name -- among the cached names (never read here: no cache,
    no name known)."""
    return norm(text) in (_NAMES.get("set") or ())


def law_of_document(name: Any) -> str:
    """v79.10 (A4, review): the corpus_expected law a document is -- its name one of the law's archive names, "" when
    none (a document's name is never read as a sigla or a citation)."""
    key = law_of_name(name)
    return key if key in (expected().get("laws") or {}) else ""


def sigla_of(text: Any, names: Sequence[str]) -> bool:
    """v79.10 (A4): the text is the sigla the pipeline labels one of these archive documents with (plan_then_write.
    _code_key: «c.d.s.» for «Codice della strada 2026») -- the label of an article of that document."""
    from .plan_then_write import _code_key
    text = norm(text)
    return bool(text) and any(_code_key(n) == text != n for n in names or [])


def law_for(text: Any) -> str:
    """v79.10 (A4): the corpus_expected law a label's code is -- an archive name of it (law_of_name, laws only) or the
    sigla such a document is labelled with («c.d.s.»), else the law cited as the whole text («l. 241/1990»: a
    citation, not a document); "" when none. Cached per text (the table read once)."""
    exp = expected()
    key = (id(exp), _NAMES.get("version", 0), norm(text))
    if key not in _FOR:
        if len(_FOR) > 4096:
            _FOR.clear()
        _FOR[key] = _law_for(text)
    return _FOR[key]


def _law_for(text: Any) -> str:
    key = law_of_name(text)
    laws = expected().get("laws") or {}
    if key in laws:
        return key
    key = next((k for k in laws if sigla_of(text, archive_names(k))), "")
    if key:
        return key
    number, year = cited(text)
    if not number:
        return ""
    return next((k for k, s in laws.items() if str(s.get("number") or "") == number
                 and str(s.get("year") or "") == year), "")


def citation_of(name: Any) -> str:
    """v79.12.5 (item 8): how an article of this document is cited in the text -- the corpus_expected law (or code)
    whose archive names hold the name exactly (law_of_name: «Testo Unico sugli Stupefacenti 2026» -> «d.P.R.
    309/1990», «Riforma del Lavoro Fornero 2026» -> «L. 92/2012», «Codice del processo amministrativo 2025» -> «c.p.a.»);
    "" for a name no entry maps (never guessed from a number or an expression). Cached per name (the table read
    once; issue_spotting reads it for every document it searches)."""
    key = (id(expected()), norm(name))
    if key not in _CITED:
        if len(_CITED) > 4096:
            _CITED.clear()
        _CITED[key] = law_of_name(name)
    return _CITED[key]


_CITED: Dict[Tuple[int, str], str] = {}


def cited_label(label: Any) -> str:
    """v79.12.5 (item 8): an article label («73 Testo Unico sugli Stupefacenti 2026», «2 Codice del Consumo 2026 co.
    1») with its document title in its legal citation («73 d.P.R. 309/1990»); unchanged when the code is no mapped
    title."""
    text = norm(label)
    number, _, rest = text.partition(" ")
    tail = re.search(r"\s+co(?:mma|\.)\s*\d+.*$", rest, re.IGNORECASE)
    code = rest[:tail.start()] if tail else rest
    cited = citation_of(code)
    return f"{number} {cited}{tail.group(0) if tail else ''}" if cited else text


_TITLES: Dict[int, List[Tuple["re.Pattern", str]]] = {}


def _title_patterns() -> List[Tuple["re.Pattern", str]]:
    """Every mapped archive name -- and its bracketed form without the trailing edition year («(Testo Unico sugli
    Stupefacenti)», article_codes.law_label's) -- with its citation, longest first; any run of blanks between words."""
    exp = expected()
    if id(exp) not in _TITLES:
        pairs: List[Tuple[str, str, bool]] = []
        for table in (exp.get("laws") or {}, exp.get("codes") or {}):
            for key in table:
                for name in archive_names(key):
                    pairs.append((name, key, False))
                    bare = re.sub(r"\s+(?:Edizione\s+)?(?:19|20)\d{2}$", "", name)
                    if bare and bare != name:
                        pairs.append((bare, key, True))
                        if len(bare.split()) >= 2:      # review: the title without its year, unbracketed too
                            pairs.append((bare, key, False))
        out = []
        # longest first; at one length the bracketed form first (its parentheses go with it). Review fix: any case
        # («Testo unico sugli stupefacenti 2026») -- the writer reads the archive titles and may print them so
        for name, key, bracket in sorted(pairs, key=lambda x: (-len(x[0]), not x[2])):
            words = r"\s+".join(re.escape(w) for w in name.split())
            rx = rf"\(\s*{words}\s*\)" if bracket else rf"(?<![\w]){words}(?![\w])"
            out.append((re.compile(rx, re.IGNORECASE), key))
        _TITLES.clear()
        _TITLES[id(exp)] = out
    return _TITLES[id(exp)]


def cite_titles(text: str, log: Optional[List[Dict[str, str]]] = None) -> str:
    """v79.12.5 (item 8): the text with every mapped document title in its legal citation (_title_patterns: the
    archive name exactly, whitespace aside, or the bracketed name without its year) -- never a document title in the
    printed text; each replacement into log ({"title", "citation"})."""
    out = text or ""
    for rx, key in _title_patterns():
        def sub(m: "re.Match", key: str = key) -> str:
            if log is not None:
                log.append({"title": " ".join(m.group(0).split()), "citation": key})
            return key
        out = rx.sub(sub, out)
        # review: a title the text gave beside its own citation («d.lgs. 206/2005 (Codice del Consumo)») -- once
        k = r"\s+".join(re.escape(w) for w in key.split())
        out = re.sub(rf"(?<![\w.]){k}(?:\s*,)?\s+(?:\(\s*{k}(?![\w/])\s*\)|{k}(?![\w/]))", key, out)
    return out


def title_forms(name: Any) -> List[str]:
    """v79.12.5 (item 8, review): a document title's printed forms -- the name, and the name without its trailing
    edition year («Testo Unico sugli Stupefacenti»)."""
    name = " ".join(str(name or "").split())
    bare = re.sub(r"\s+(?:Edizione\s+)?(?:19|20)\d{2}$", "", name)
    return [x for x in dict.fromkeys([name, bare]) if x and (x == name or len(x.split()) >= 2)]


def reserved(number: str, year: str) -> bool:
    """v79.10 (A4): a corpus_expected law has that number and year -- its documents are matched by their archive names
    only, never by the number a name may hold."""
    return bool(number and year) and any(str(s.get("number") or "") == number and str(s.get("year") or "") == year
                                         for s in (expected().get("laws") or {}).values())


def status(key: str, names: Sequence[str]) -> str:
    """v79.10 (A4): «present», «NOT IN ARCHIVE» (mapped, none of its archive names among the names) or «UNMAPPED»."""
    if not archive_names(key):
        return UNMAPPED
    return PRESENT if present(key, names) else NOT_IN_ARCHIVE


def fetch_names(session) -> List[str]:
    """The primary documents' names, from a read session (v79.10: the names only -- the archive names are names)."""
    out: List[str] = []
    for r in session.run(NAMES_QUERY):
        name = norm(r["name"])
        if name and name not in out:
            out.append(name)
    return out


def set_names(names: Optional[Sequence[str]]) -> None:
    """The cache (None: the graph could not be read -- nothing is missing)."""
    with _LOCK:
        _NAMES["names"] = list(names) if names is not None else None
        _NAMES["set"] = frozenset(norm(n) for n in names or [] if norm(n))       # v79.10 (A4, review): cited()
        _NAMES["version"] = next(_VERSIONS)


def names() -> Optional[List[str]]:
    """The cached names, read on first use when warm() did not run; None when the graph could not be read."""
    if "names" not in _NAMES:
        warm()
    return _NAMES.get("names")


def warm() -> Optional[List[str]]:
    """Read the names once (the API's startup); logged, never raises."""
    try:
        import neo4j
        from .main import NEO4J_DATABASE, driver
        with driver.session(database=NEO4J_DATABASE, default_access_mode=neo4j.READ_ACCESS) as session:
            got = fetch_names(session)
        set_names(got)
        logger.info("corpus_laws: %d primary document names cached", len(got))
    except Exception as exc:        # noqa: BLE001 -- unknown: no line is printed
        logger.warning("corpus_laws: the primary documents' names not read: %s", exc)
        set_names(None)
    return _NAMES.get("names")


def missing(laws: Sequence[str], known: Optional[Sequence[str]] = None) -> Optional[List[str]]:
    """The laws of the list absent from the graph (unmapped or not among the names); None when the names are unknown."""
    known = names() if known is None else known
    if known is None:
        return None
    return [x for x in laws or [] if not present(x, known)]


def gaps(known: Sequence[str], table_names: Sequence[str] = ()) -> Dict[str, Any]:
    """Per module and for the defaults, the expected laws missing from the names given (corpus_gaps.py). v79.10 (A4):
    «status» -- each law and code -> «present» / «NOT IN ARCHIVE» / «UNMAPPED»; «unmatched_modules» -- the module laws
    not present (corpus_gaps exits 1 on any); «names_unmapped» -- the names read that no law, code or special-law table
    entry maps (table_names: the table's archive names, special_laws.archive_names())."""
    exp = expected()
    mods = {m: [x for x in laws if not present(x, known)] for m, laws in (exp.get("modules") or {}).items()}
    keys = list(exp.get("laws") or {}) + list(exp.get("codes") or {})
    mapped = {n for k in keys for n in archive_names(k)} | {norm(n) for n in table_names or []}
    return {"modules": mods, "defaults": [x for x in exp.get("defaults") or [] if not present(x, known)],
            "names_read": len(known), "status": {k: status(k, known) for k in keys},
            "unmatched_modules": sorted({x for v in mods.values() for x in v}),
            "names_unmapped": sorted({norm(n) for n in known if norm(n) and norm(n) not in mapped})}
