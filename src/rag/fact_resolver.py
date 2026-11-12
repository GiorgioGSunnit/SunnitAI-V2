"""FACT_ISSUES_RESOLVER_V2_ENABLED (default false): a fact's institution (fact_issues, "fonte": "fatto") resolved to its
articles by their headings, within the named code.

fact_loss_report on the live run aw-20261030-1821: of 119 expected articles lost at "article resolved", 106 had an
institution named from their fact; 46% of them resolved to other articles ("Responsabilità medica" -> 2462/2407 c.c.,
company law; "Decisione amministrativa" -> art. 17 CAD), 43% to nothing ("Risarcimento danni" for unpaid rent), and a
right one only to its core article ("Garanzia per i vizi della cosa venduta" -> 1490, 1495 lost). resolve():

  1. The institution's specific words: its content words without the generic ones (dg's generic heading words,
     GENERIC_WORDS: "responsabilità", "danni", "risarcimento" ...). With none, nothing is resolved: never through a
     single generic word.
  2. Candidates within the named code's document: the top TOP_K of the heading embedding index
     (section_heading_embeddings), of the full-text index on headings (section_title_fulltext) and of the full-text
     index on the text (section_fulltext) -- in that order, each article once.
  3. Each candidate refused outside the institution's chapter when the chapter is known (answer_invariants.CHAPTERS);
     a "core" one when its heading holds at least half of the specific words and never fewer than two (the matching
     words: the specific ones without the broad qualifiers, fact_issues.BROAD_QUALIFIERS -- "civile", "pubblico",
     "contrattuale" name the relationship, not the institute); a "related" one when, beside a core article (within
     RELATED_SPAN numbers), its text holds one of them (1495 beside 1490: "denunzia i vizi"); a text-only one when
     its text holds at least half of them and two or more. At most MAX_KEPT: the cores first.
  4. Only a core article is the institution's (_fact_institution: T1, its own plan issue). A related or text one is
     an ordinary candidate (_fact_related, T2): the graded check judges it like any other (smoke test c1: "Responsabilità
     civile dell'ente pubblico per lesioni permanenti" reached art. 406 c.c., guardianship, by two words of its text,
     and was placed in the plan as T1).

v60n: before the indexes, the rubrica itself (rubrica_matches over the code's rubriche, read once per document): a
rubrica equal to the name once footnote markers and punctuation are stripped («Truffa» -> 640 c.p.), else the rubriche
beginning with it («Corruzione» -> 318, 319, 319-ter, 320 c.p.), else those whose first words have its words' roots
(«Furti» -> «Furto»); what it finds are the core articles, the related and text-only ones then judged as before. A
single-word name is never generic unless its word is (generic_stems): with no rubrica match, a heading holding its one
word is a core.

Logged per institution (the caller's case_analysis_fact_institution): the candidates (article, heading, how found),
the kept (core / related / text) and the refused with the reason.
"""

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

TOP_K = 5
MAX_KEPT = 3
RELATED_SPAN = 15
GENERIC_WORDS = ("responsabilità responsabile risarcimento risarcimenti danno danni tutela tutele azione azioni "
                 "decisione decisioni provvedimento provvedimenti obbligazione obbligazioni rapporto rapporti "
                 "violazione violazioni illecito diligenza")


def _stems(text: str) -> set:
    from . import defensive_generation as dg
    return dg._stems_of(text)


def generic_stems() -> set:
    from . import defensive_generation as dg
    return dg._strict_generic_stems() | _stems(GENERIC_WORDS)


def specific_words(name: str) -> set:
    return _stems(name) - generic_stems()


def matching_words(name: str) -> set:
    """The words a heading or a text must hold: the specific ones without the broad qualifiers."""
    from . import fact_issues
    return specific_words(name) - _stems(fact_issues.BROAD_QUALIFIERS)


def _document(documents: List[Dict[str, Any]], codice: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    from . import defensive_generation as dg
    from .issue_spotting import resolve_code
    abbr = dg._code_of(codice) if codice else None
    if abbr:
        prefix = dg._CODE_PREFIXES[abbr]
        return next((d for d in documents if (d.get("name") or "").startswith(prefix)), None), abbr
    return (resolve_code(codice, documents) if codice else None), None


def _number(name: str) -> Optional[str]:
    from . import defensive_generation as dg
    m = dg._ARTICLE_SECTION_RE.match((name or "").strip())
    return dg._article_num(m.group(1) + (m.group(2) or "")) if m else None


def candidates(session, doc_id: str, name: str) -> List[Dict[str, Any]]:
    """[{"number", "heading", "via"}]: the top TOP_K of the heading index, the heading full text and the text full
    text, within doc_id, each article once; a channel that fails (an index missing) gives nothing."""
    from . import defensive_generation as dg
    from .issue_spotting import _TITLE_FULLTEXT_INDEX
    out: Dict[str, Dict[str, Any]] = {}

    def add(rows, via):
        n = 0
        for row in rows or []:
            number = _number(row.get("name") or "")
            if not number:
                continue
            n += 1
            if number not in out:
                out[number] = {"number": number, "heading": " ".join(str(row.get("heading") or "").split()),
                               "via": via}
            if n >= TOP_K:
                break
    try:
        embedding = list(dg.embedding_model.embed_query(name))
        add(session.run(
            "CALL db.index.vector.queryNodes($index, $k, $embedding) YIELD node, score "
            "MATCH (d:Document)-[:CONTAINS]->(node) WHERE d.id = $doc_id "
            "RETURN node.name AS name, coalesce(node.titolo_articolo, node.title) AS heading, score "
            "ORDER BY score DESC LIMIT 25",
            index=dg._HEADING_VECTOR_INDEX, k=dg._HEADING_INDEX_K, embedding=embedding, doc_id=doc_id).data(),
            "heading index")
    except Exception as exc:
        logger.info("fact resolver: heading index unavailable: %s", exc)
    words = sorted(specific_words(name))
    if words:
        try:
            add(session.run(
                "CALL db.index.fulltext.queryNodes($index, $lucene) YIELD node, score "
                "MATCH (d:Document)-[:CONTAINS]->(node) WHERE d.id = $doc_id "
                "RETURN node.name AS name, coalesce(node.titolo_articolo, node.title) AS heading, score "
                "ORDER BY score DESC LIMIT 25",
                index=_TITLE_FULLTEXT_INDEX, lucene=" ".join(f"titolo_articolo:{w}* title:{w}*" for w in words),
                doc_id=doc_id).data(), "heading full text")
        except Exception as exc:
            logger.info("fact resolver: heading full text unavailable: %s", exc)
    terms = dg._lucene_query(name)
    if terms:
        try:
            add(session.run(
                "CALL db.index.fulltext.queryNodes('section_fulltext', $t) YIELD node, score "
                "MATCH (d:Document)-[:CONTAINS]->(node) WHERE d.id = $doc_id "
                "RETURN node.name AS name, coalesce(node.titolo_articolo, node.title) AS heading, score "
                "ORDER BY score DESC LIMIT 25", t=terms, doc_id=doc_id).data(), "text full text")
        except Exception as exc:
            logger.info("fact resolver: text full text failed: %s", exc)
    return list(out.values())


# --- v60n: the rubrica first -- exact, then prefix, then root ---------------------------------------------------------
RUBRICHE_QUERY = ("MATCH (d:Document)-[:CONTAINS]->(s:Section) WHERE d.id = $doc_id AND s.titolo_articolo IS NOT NULL "
                  "RETURN s.name AS name, s.titolo_articolo AS rubrica")
PREFIX_KEPT = 5                 # the rubriche a name begins (prefix, root): at most these, in article order
EXACT_KEPT = 8                  # v60o: identical rubriche («Confisca», seven in c.p.) are all the institution's
_RUBRICHE: Dict[str, List[Tuple[str, str]]] = {}
_LEADING_ARTICLES = {"il", "lo", "la", "i", "gli", "le", "l", "un", "uno", "una"}
_ART_PREFIX_RE = re.compile(r"^\s*art(?:icolo)?\.?\s*\d+(?:[-\s]?[a-z]+)?\s*[-–—.:]?\s*", re.IGNORECASE)


def rubrica_key(text: Any) -> str:
    """A rubrica (or a name) as rubrica_matches compares it: footnote markers out («(1)», «[202]», «*»), an «Art. N –»
    lead out, lower case, every punctuation mark a space, a leading article out."""
    from .rule_quote import clean_rubrica
    t = _ART_PREFIX_RE.sub("", clean_rubrica(text))
    words = re.sub(r"[^\w\s]|_|\d", " ", t.lower()).split()
    while words and words[0] in _LEADING_ARTICLES:
        words = words[1:]
    return " ".join(words)


def _root(word: str) -> str:
    """A word's root for the inflection-tolerant prefix: its last letter off, six letters at most («furto», «furti» ->
    «furt»; «minaccia», «minacce» -> «minacc»)."""
    return word[:-1][:6] if len(word) > 4 else word


def _order(number: str) -> tuple:
    m = re.match(r"(\d+)(?:[-.]?([a-z]+))?", str(number or ""))
    return (int(m.group(1)), m.group(2) or "") if m else (10 ** 9, str(number))


def code_rubriche(session, doc_id: str) -> List[Tuple[str, str]]:
    """[(number, rubrica)] of the document's articles, in article order -- read once per document (RUBRICHE_QUERY);
    [] when the read fails."""
    if doc_id not in _RUBRICHE:
        try:
            rows = session.run(RUBRICHE_QUERY, doc_id=doc_id).data()
        except Exception as exc:            # noqa: BLE001 -- no list: the indexes alone, as before
            logger.info("fact resolver: rubriche of %s not read: %s", doc_id, exc)
            return []
        out: Dict[str, str] = {}
        for r in rows or []:
            number = _number(r.get("name") or "")
            if number and r.get("rubrica") and number not in out:
                out[number] = str(r["rubrica"])
        _RUBRICHE[doc_id] = sorted(out.items(), key=lambda x: _order(x[0]))
    return _RUBRICHE[doc_id]


def rubrica_matches(rubriche: List[Tuple[str, str]], name: str) -> Tuple[str, List[str]]:
    """(how, numbers) of the rubriche matching the name, in article order: «esatta» -- equal to it (rubrica_key);
    else «prefisso» -- beginning with it, whole words; else «radice» -- their first words having its words' roots
    (_root: a word of three letters or more); ("", []) when none."""
    key = rubrica_key(name)
    if not key:
        return "", []
    keyed = [(n, rubrica_key(r)) for n, r in rubriche]
    exact = [n for n, k in keyed if k == key]
    if exact:
        return "esatta", exact
    prefix = [n for n, k in keyed if k.startswith(key + " ")]
    if prefix:
        return "prefisso", prefix
    roots = [_root(w) for w in key.split() if len(w) > 2]
    if not roots:
        return "", []
    rooted = [n for n, k in keyed if [_root(w) for w in k.split() if len(w) > 2][:len(roots)] == roots]
    return ("radice", rooted) if rooted else ("", [])


def _text(rows: List[Dict[str, Any]]) -> str:
    return " ".join(str((r.get("s") or {}).get("plain_text") or (r.get("s") or {}).get("abstract") or "")
                    for r in rows or [])


def _as_int(number: str) -> Optional[int]:
    m = re.match(r"\d+", number or "")
    return int(m.group(0)) if m else None


def resolve(session, documents: List[Dict[str, Any]], istituto: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict]:
    """(rows of the kept articles, log) for one fact's institution: see the module docstring. Each row tagged
    _article; a core one _fact_institution (its name) and _heading_match, a related or text one _fact_related and
    tier T2."""
    from . import answer_invariants
    from . import defensive_generation as dg
    from .issue_spotting import fetch_article
    name = " ".join(str(istituto.get("nome") or "").split())
    codice = str(istituto.get("codice") or "")
    log: Dict[str, Any] = {"institution": name, "codice": codice, "candidates": [], "kept": [], "refused": []}
    specific = matching_words(name)
    if not specific:
        log["refused"].append({"article": None, "why": "generic institution name: no specific word"})
        return [], log
    doc, abbr = _document(documents, codice)
    if not doc:
        log["refused"].append({"article": None, "why": f"code {codice!r} not resolved to a document"})
        return [], log
    code = abbr or doc.get("name")
    chapter = answer_invariants.chapter(name, abbr or "")
    log["chapter"] = chapter
    how, by_rubrica = rubrica_matches(code_rubriche(session, doc["id"]), name)        # v60n: the rubrica first
    by_rubrica = [n for n in by_rubrica if not chapter or answer_invariants.in_ranges(n, chapter)]
    by_rubrica = by_rubrica[:EXACT_KEPT if how == "esatta" else PREFIX_KEPT]
    if by_rubrica:
        log["rubrica"] = {"how": how, "articles": [f"{n} {code}" for n in by_rubrica]}
    found = candidates(session, doc["id"], name)
    have = {c["number"] for c in found}
    found = [{"number": n, "heading": dict(code_rubriche(session, doc["id"])).get(n, ""), "via": f"rubrica {how}"}
             for n in by_rubrica if n not in have] + found
    log["candidates"] = [{"article": f"{c['number']} {code}", "heading": c["heading"], "via": c["via"]} for c in found]
    judged = []          # (kind, head_shared, order, candidate, rows)
    # v60n: a single-word name's heading match is its word's root among the heading's words (_root), never a bare
    # five-letter stem («complicità» never meets «Completamento»)
    single = [w for w in rubrica_key(name).split() if len(w) > 2 and _stems(w) & specific] if len(specific) == 1 else []
    root = _root(single[0]) if single else ""
    for order, c in enumerate(found):
        label = f"{c['number']} {code}"
        if chapter and not answer_invariants.in_ranges(c["number"], chapter):
            log["refused"].append({"article": label, "why": "outside the institution's chapter"})
            continue
        rows = fetch_article(session, doc["id"], c["number"])
        if not rows:
            log["refused"].append({"article": label, "why": "no text"})
            continue
        heading = c["heading"] or dg._rows_heading(rows)
        head = (1 if root in {_root(w) for w in rubrica_key(heading).split()} else 0) if root \
            else len(specific & _stems(heading))
        text = len(specific & _stems(_text(rows)))
        judged.append((head, text, order, c, rows, label))
    # a heading sharing one word only is never the core -- v60n: unless the name has one word (not a generic one)
    need = 1 if len(specific) == 1 else max(2, (len(specific) + 1) // 2)
    if by_rubrica:              # v60n: the rubriche the name matches are the cores, in article order
        cores = [j for n in by_rubrica for j in judged if j[3]["number"] == n]
    else:
        cores = sorted([j for j in judged if j[0] >= need], key=lambda j: (-j[0], j[2]))
    core_numbers = [_as_int(j[3]["number"]) for j in cores]
    kept: List[Tuple[str, Any]] = [("core", j) for j in cores]
    for j in judged:
        if j in cores:
            continue
        head, text, _, c, _, label = j
        n = _as_int(c["number"])
        if text and any(n is not None and k is not None and abs(n - k) <= RELATED_SPAN for k in core_numbers):
            kept.append(("related", j))
        elif text >= max(2, need):
            kept.append(("text", j))
        elif head or text:
            why = (("related to no core article" if cores else "no core article")
                   + f": heading {head}, text {text} of {len(specific)} specific word(s)")
            log["refused"].append({"article": label, "why": why})
        else:
            log["refused"].append({"article": label, "why": "heading and text share no specific word"})
    kept.sort(key=lambda kj: ({"core": 0, "related": 1, "text": 2}[kj[0]],
                              cores.index(kj[1]) if kj[0] == "core" else kj[1][2]))
    limit = max(MAX_KEPT, len(cores))          # v60n: every core a rubrica match found (PREFIX_KEPT at most)
    out: List[Dict[str, Any]] = []
    for k, (kind, (_, _, _, c, rows, label)) in enumerate(kept):
        if k >= limit:
            log["refused"].append({"article": label, "why": f"beyond the {limit} kept"})
            continue
        from . import evidence_tiers
        for row in rows:
            row["_article"] = (c["number"], code)
            if kind == "core":
                row["_fact_institution"] = name
                row["_heading_match"] = True
            else:
                row["_fact_related"] = name
                row["_tier"], row["_tier_why"] = evidence_tiers.T2, f"{kind} match of a fact institution"
        out.extend(rows)
        log["kept"].append({"article": label, "as": kind, "heading": c["heading"]})
    return out, log


# --- v60h (4): the first read's second and third channels --------------------------------------------------------------
# After the rubrica stem match (resolve) misses: (2) the institution's rubricated name and nouns against the articles'
# TEXT, full text within the code (text_channel), an article kept when its text holds the specific words (two, one when
# the name has one); (3) both missing, the code's institutions index (data/institutions_<code>.json, built once from the
# store's Titolo / Capo / Sezione structure and rubriche by scripts/build_institutions.py): the unit whose name holds
# the institution's words, its articles whose rubrica shares one (else its first). Each row tagged as resolve's are not:
# an ordinary candidate, graded like any other.
TEXT_TOP = 8
TEXT_KEPT = 3
INDEX_KEPT = 3
INDEX_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
_INDEX: Dict[str, Optional[List[Dict[str, Any]]]] = {}
_LEVEL_RANK = {"ARTICOLO": 0, "PARAGRAFO": 1, "SEZIONE": 2, "CAPO": 3, "TITOLO": 4, "PARTE": 5, "LIBRO": 6}


def code_slug(code: str) -> str:
    """«c.c.» -> «cc», «c.p.p.» -> «cpp»; another code's name, its letters and digits."""
    return re.sub(r"[^a-z0-9]+", "", str(code or "").lower())[:40] or "code"


def index_path(code: str) -> str:
    return os.path.join(INDEX_DIR, f"institutions_{code_slug(code)}.json")


def load_index(code: str) -> Optional[List[Dict[str, Any]]]:
    """The code's institutions index (None when not built)."""
    import json
    key = code_slug(code)
    if key not in _INDEX:
        try:
            with open(index_path(code), encoding="utf-8") as fh:
                data = json.load(fh)
            _INDEX[key] = list(data.get("institutions") or []) if isinstance(data, dict) else None
        except (OSError, ValueError):
            _INDEX[key] = None
    return _INDEX[key]


def _words_of(name: str, nouns: str = "") -> set:
    return matching_words(name) | (matching_words(nouns) if nouns else set())


def _need(words: set) -> int:
    return min(2, len(words))


def _rows_for(session, doc: Dict[str, Any], code: str, numbers: List[str]) -> List[Dict[str, Any]]:
    from .issue_spotting import fetch_article
    out = []
    for number in numbers:
        for row in fetch_article(session, doc["id"], number) or []:
            row["_article"] = (number, code)
            out.append(row)
    return out


def text_channel(session, documents: List[Dict[str, Any]], name: str, codice: str, nouns: str = ""
                 ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """(rows, log): channel 2 -- the name's (and the nouns') specific words against the code's article texts (the
    section_fulltext index, TEXT_TOP), an article kept when its text holds _need of them; at most TEXT_KEPT."""
    from . import defensive_generation as dg
    words = _words_of(name, nouns)
    log: Dict[str, Any] = {"channel": "testo", "query": name, "words": sorted(words), "kept": []}
    if not words:
        log["why"] = "no specific word"
        return [], log
    doc, abbr = _document(documents, codice)
    if not doc:
        log["why"] = f"code {codice!r} not resolved"
        return [], log
    code = abbr or doc.get("name")
    terms = dg._lucene_query(" ".join([name] + ([nouns] if nouns else [])))
    try:
        found = session.run(
            "CALL db.index.fulltext.queryNodes('section_fulltext', $t) YIELD node, score "
            "MATCH (d:Document)-[:CONTAINS]->(node) WHERE d.id = $doc_id "
            "RETURN node.name AS name, score ORDER BY score DESC LIMIT $k", t=terms, doc_id=doc["id"], k=TEXT_TOP).data()
    except Exception as exc:            # noqa: BLE001 -- the index missing: no channel
        log["why"] = f"text full text failed: {exc}"[:200]
        return [], log
    numbers = list(dict.fromkeys(n for n in (_number(r.get("name") or "") for r in found) if n))
    out = []
    for number in numbers:
        rows = _rows_for(session, doc, code, [number])
        if rows and len(words & _stems(_text(rows))) >= _need(words):
            out += rows
            log["kept"].append(f"{number} {code}")
            if len(log["kept"]) >= TEXT_KEPT:
                break
    return out, log


def index_channel(session, documents: List[Dict[str, Any]], name: str, codice: str, nouns: str = "",
                  index: Optional[List[Dict[str, Any]]] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """(rows, log): channel 3 -- the code's institutions index: the unit whose name holds _need of the name's specific
    words (the most, then the narrowest unit), its articles whose rubrica shares one of them, else its first; at most
    INDEX_KEPT."""
    words = _words_of(name, nouns)
    log: Dict[str, Any] = {"channel": "indice", "query": name, "words": sorted(words), "kept": []}
    doc, abbr = _document(documents, codice)
    code = (abbr or (doc or {}).get("name") or codice)
    entries = index if index is not None else load_index(abbr or codice)
    if entries is None:
        log["why"] = f"no index for {code} ({index_path(abbr or codice)})"
        return [], log
    if not words or not doc:
        log["why"] = "no specific word" if not words else f"code {codice!r} not resolved"
        return [], log
    best = None
    for e in entries:
        shared = len(words & _stems(" ".join([str(e.get("name") or ""), str(e.get("rubricated") or "")])))
        if shared >= max(1, _need(words)):
            key = (-shared, _LEVEL_RANK.get(str(e.get("level") or ""), 9), len(e.get("articles") or []))
            if best is None or key < best[0]:
                best = (key, e)
    if best is None:
        log["why"] = "no unit holds the words"
        return [], log
    entry = best[1]
    rubriche = entry.get("rubriche") or {}
    articles = [str(a) for a in entry.get("articles") or []]
    named = [a for a in articles if words & _stems(rubriche.get(a) or "")]
    numbers = (named or articles)[:INDEX_KEPT]
    log.update(unit=entry.get("rubricated") or entry.get("name"), level=entry.get("level"),
               kept=[f"{n} {code}" for n in numbers])
    return _rows_for(session, doc, code, numbers), log
