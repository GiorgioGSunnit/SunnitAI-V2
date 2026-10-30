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
     a "core" one when its heading holds at least half of the specific words; a "related" one when, beside a core
     article (within RELATED_SPAN numbers), its text holds one of them (1495 beside 1490: "denunzia i vizi"); a
     text-only one when its text holds at least half of them and two or more. At most MAX_KEPT: the cores first.

Logged per institution (the caller's case_analysis_fact_institution): the candidates (article, heading, how found),
the kept (core / related / text) and the refused with the reason.
"""

import logging
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


def _text(rows: List[Dict[str, Any]]) -> str:
    return " ".join(str((r.get("s") or {}).get("plain_text") or (r.get("s") or {}).get("abstract") or "")
                    for r in rows or [])


def _as_int(number: str) -> Optional[int]:
    m = re.match(r"\d+", number or "")
    return int(m.group(0)) if m else None


def resolve(session, documents: List[Dict[str, Any]], istituto: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict]:
    """(rows of the kept articles, log) for one fact's institution: see the module docstring. Each row tagged
    _article, _fact_institution (its name), _heading_match (a core article)."""
    from . import answer_invariants
    from . import defensive_generation as dg
    from .issue_spotting import fetch_article
    name = " ".join(str(istituto.get("nome") or "").split())
    codice = str(istituto.get("codice") or "")
    log: Dict[str, Any] = {"institution": name, "codice": codice, "candidates": [], "kept": [], "refused": []}
    specific = specific_words(name)
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
    found = candidates(session, doc["id"], name)
    log["candidates"] = [{"article": f"{c['number']} {code}", "heading": c["heading"], "via": c["via"]} for c in found]
    judged = []          # (kind, head_shared, order, candidate, rows)
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
        head = len(specific & _stems(heading))
        text = len(specific & _stems(_text(rows)))
        judged.append((head, text, order, c, rows, label))
    need = max(1, (len(specific) + 1) // 2)
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
            why = ("related to no core article" if cores else "no core article") + f": shares {head + text} word(s)"
            log["refused"].append({"article": label, "why": why})
        else:
            log["refused"].append({"article": label, "why": "heading and text share no specific word"})
    kept.sort(key=lambda kj: ({"core": 0, "related": 1, "text": 2}[kj[0]], kj[1][2]))
    out: List[Dict[str, Any]] = []
    for k, (kind, (_, _, _, c, rows, label)) in enumerate(kept):
        if k >= MAX_KEPT:
            log["refused"].append({"article": label, "why": f"beyond the {MAX_KEPT} kept"})
            continue
        for row in rows:
            row["_article"] = (c["number"], code)
            row["_fact_institution"] = name
            if kind == "core":
                row["_heading_match"] = True
        out.extend(rows)
        log["kept"].append({"article": label, "as": kind, "heading": c["heading"]})
    return out, log
