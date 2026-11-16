"""v63: the «cenni» subjects anchored on their articles -- what a short institution paragraph rests on, and what then
gets an «applicazione» paragraph in the norms section.

  resolve(subjects, rows, codes, fetch=, index=, search=)
      {subject: [rows]}, in this order and only with confidence:
        1. a general concept (data/general_concepts.json: the subject matches a concept's pattern, the concept's code
           among the case's) -> its articles (c.p. Libro I: nesso causale 40-41, elemento soggettivo 42-43,
           preterintenzione 43);
        2. else the code's institutions index (fact_resolver.index_channel), its articles kept only when general
           (issue_records.counted is False: c.p. Libro I, procedural, general provisions, definitional);
        3. an article whose rubrica holds every content stem of the subject -- the subject names it («omicidio
           preterintenzionale» -> 584 c.p.): the case's rows first, then the rubrica search; for a general concept
           (1 or 2) it joins the general articles, for any other subject it is the anchor alone.
      A special-part article is never the anchor of a general concept by any other road (a rubrica that merely shares
      a word, the article texts); nothing found with confidence: [] -- the cenni cite no article.

  v64: a general-part article (c.p. Libro I, general_part) is never a grading candidate (general_part_gate: the
  user's named ones aside); it enters the norms section only as an «applicazione» paragraph, through the concepts
  table -- a cenni subject, or the kind of a charged article's element (element_refs: «cagiona» -> nesso causale,
  40-41; «con atti diretti a», «per colpa» -> elemento soggettivo, 42-43) --, through the posture table's sanction
  articles for an imputato (data/posture_acts.json "sanctions"), or as an article the user names.

No case-specific logic: the table is doctrine (the codes' general part), the rest reads the codes' own rubriche.
"""

import json
import logging
import os
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

CONCEPTS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data",
                             "general_concepts.json")
ARTICLES_MAX = 3            # a subject's articles: the general ones first, then the one its name designates
NAMED_MAX = 1               # the articles a subject's own name designates (rule 3)

_CONCEPTS: Dict[str, Any] = {}

GENERAL, INDEX, NAMED, NONE = ("concetto di parte generale", "indice degli istituti", "rubrica dell'articolo",
                               "nessuna norma")


def concepts() -> List[Dict[str, Any]]:
    """data/general_concepts.json's concepts, each with its compiled patterns (read once; [] when missing, logged)."""
    if "list" not in _CONCEPTS:
        out = []
        try:
            with open(CONCEPTS_FILE, encoding="utf-8") as f:
                raw = json.load(f).get("concepts") or []
        except (OSError, ValueError) as exc:
            logger.warning("cenni: %s not read: %s", CONCEPTS_FILE, exc)
            raw = []
        for c in raw:
            try:
                out.append({**c, "_res": [re.compile(p, re.IGNORECASE) for p in c.get("patterns") or []],
                            "_elements": [re.compile(p, re.IGNORECASE) for p in c.get("elements") or []]})
            except re.error as exc:
                logger.warning("cenni: concept %r skipped: %s", c.get("name"), exc)
        _CONCEPTS["list"] = out
    return _CONCEPTS["list"]


def general_refs(subject: str, codes: Sequence[str]) -> List[Tuple[str, str]]:
    """The general-part articles of every concept the subject names (its code among codes), in the table's order, once
    each."""
    from .plan_then_write import _code_key
    have = {_code_key(c) for c in codes or []}
    low = " ".join(str(subject or "").lower().split())
    out: List[Tuple[str, str]] = []
    for c in concepts():
        if _code_key(str(c.get("code") or "")) in have and any(r.search(low) for r in c["_res"]):
            out += [(str(a), str(c["code"])) for a in c.get("articles") or [] if (str(a), str(c["code"])) not in out]
    return out


def procedural_refs(name: str) -> Tuple[str, List[Tuple[str, str]]]:
    """v75: (the procedural concept the name asks for, its articles) -- the table's «procedural» concepts (querela,
    sequestro, restituzione, parte civile), any code; ("", []) when the name is no procedural ask."""
    low = " ".join(str(name or "").lower().split())
    for c in concepts():
        if c.get("procedural") and any(r.search(low) for r in c["_res"]):
            return str(c["name"]), [(str(a), str(c["code"])) for a in c.get("articles") or []]
    return "", []


def element_refs(element: str, codes: Sequence[str]) -> List[Tuple[str, str]]:
    """v64: the general-part articles of the concept whose kind a charged article's element is (its "elements"
    patterns: «cagiona la morte» -> nesso causale -> 40, 41 c.p.), its code among codes; [] for an element of no kind."""
    from .plan_then_write import _code_key
    have = {_code_key(c) for c in codes or []}
    low = " ".join(str(element or "").lower().split())
    out: List[Tuple[str, str]] = []
    for c in concepts():
        if _code_key(str(c.get("code") or "")) in have and any(r.search(low) for r in c["_elements"]):
            out += [(str(a), str(c["code"])) for a in c.get("articles") or [] if (str(a), str(c["code"])) not in out]
    return out


OFFENCES_FROM = 241         # c.p. Libro II opens at art. 241: below it, the general part


def general_part(ref: Sequence[str]) -> bool:
    """v64: a c.p. Libro I article (artt. 1-240): the general part, never a grading candidate."""
    from .plan_then_write import _code_key
    if not ref or len(ref) < 2:
        return False
    m = re.match(r"\d+", str(ref[0]))
    return bool(m) and _code_key(str(ref[1])) == "c.p." and int(m.group(0)) < OFFENCES_FROM


def general_part_gate(rows: Sequence[Dict[str, Any]], named: set) -> Tuple[List[Dict[str, Any]], List[str]]:
    """v64: (the rows without the general-part articles, their labels) -- an article the user names (named: keys)
    stays."""
    from .interactive_flow import _key
    kept, dropped = [], []
    for r in rows or []:
        if r.get("_article") and general_part(r["_article"]) and _key(r["_article"]) not in named:
            label = _label(r)
            if label not in dropped:
                dropped.append(label)
            continue
        kept.append(r)
    return kept, dropped


def _label(row: Dict[str, Any]) -> str:
    from .interactive_flow import _key, _label as label
    return label(_key(row["_article"]))


def rubrica(row: Dict[str, Any]) -> str:
    from .interactive_flow import _row_rubrica
    return _row_rubrica(row)


def general_article(row: Dict[str, Any]) -> bool:
    """A general article: one whose requisites are not counted (c.p. Libro I, procedural, general, definitional)."""
    from . import issue_records
    return not issue_records.counted(_label(row), rubrica(row))


def named_by(subject: str, rubrica_text: str) -> bool:
    """The rubrica holds every content stem of the subject: the subject names the article."""
    from .topic_step import _content_stems
    stems = set(_content_stems(subject))
    return bool(stems) and stems <= set(_content_stems(rubrica_text))


def resolve(subjects: Sequence[str], rows: Sequence[Dict[str, Any]], codes: Sequence[str],
            fetch: Optional[Callable[[List[Tuple[str, str]]], List[Dict[str, Any]]]] = None,
            index: Optional[Callable[[str, str], List[Dict[str, Any]]]] = None,
            search: Optional[Callable[[List[str]], List[Dict[str, Any]]]] = None
            ) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Dict[str, Any]]]:
    """({subject: [rows]}, {subject: {"articles", "how"}}) -- the module's doc. fetch(refs) reads articles by number,
    index(subject, code) the institutions index, search([subject]) the rubrica search; each may be None (skipped)."""
    from .interactive_flow import _key
    pool = [r for r in rows or [] if r.get("_article")]
    out: Dict[str, List[Dict[str, Any]]] = {}
    log: Dict[str, Dict[str, Any]] = {}
    for subject in subjects:
        got: List[Dict[str, Any]] = []
        how: List[str] = []
        refs = general_refs(subject, codes)
        if refs:
            have = {_key(r["_article"]): r for r in pool}
            missing = [ref for ref in refs if _key(ref) not in have]
            for r in (fetch(missing) or []) if missing and fetch else []:
                if r.get("_article"):
                    have.setdefault(_key(r["_article"]), r)
            got = [have[_key(ref)] for ref in refs if _key(ref) in have]
            how += [GENERAL] if got else []
        if not got and index:
            for code in codes:
                got += [r for r in index(subject, code) or [] if r.get("_article") and general_article(r)]
            how += [INDEX] if got else []
        general = bool(got)
        named = [r for r in pool if named_by(subject, rubrica(r))]
        if not named and search:
            named = [r for r in search([subject]) or [] if r.get("_article") and named_by(subject, rubrica(r))]
        seen = {_key(r["_article"]) for r in got}
        named = [r for r in named if _key(r["_article"]) not in seen][:NAMED_MAX]
        how += [NAMED] if named else []
        chosen: List[Dict[str, Any]] = []
        for r in got + named:
            if _key(r["_article"]) not in {_key(x["_article"]) for x in chosen}:
                chosen.append(r)
        out[subject] = chosen[:ARTICLES_MAX]
        log[subject] = {"articles": [_label(r) for r in out[subject]], "how": " + ".join(how) or NONE,
                        "general": general}
    return out, log


def graph_tools(documents: Callable[[], List[Dict[str, Any]]]):
    """(fetch, index) on the graph, each opening its own read session -- and reading the documents (documents()) --
    only when called (the index's only when the code has an index file)."""
    def session_do(job):
        import neo4j
        from .main import NEO4J_DATABASE, driver
        with driver.session(database=NEO4J_DATABASE, default_access_mode=neo4j.READ_ACCESS) as session:
            return job(session)

    def fetch(refs: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        from . import article_text, defensive_generation as dg

        def job(session):
            got = dg._fetch_any(session, documents(), list(refs)) or []
            article_text.fill_texts(session, got)
            return got
        return session_do(job) if refs else []

    def index(subject: str, code: str) -> List[Dict[str, Any]]:
        from . import article_text, fact_resolver
        if fact_resolver.load_index(code) is None:
            return []

        def job(session):
            got = fact_resolver.index_channel(session, documents(), subject, code)
            found = (got[0] if isinstance(got, tuple) else got) or []
            article_text.fill_texts(session, found)
            return found
        return session_do(job)
    return fetch, index
