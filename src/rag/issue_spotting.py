"""Issue spotting for narrative questions (round 3, item 2).

ISSUE_SPOTTING_ENABLED (default false). For a narrative / fact-pattern
question, one model call names the legal institutions at stake, each with the
article and the code or law that governs it; each code is resolved against the
real document names in Neo4j (not a fixed list, so "codice del consumo" or
"legge 392/1978" work when those documents exist), and each article is fetched
exactly by number in that document, whatever the law hint. The fetched articles
then go through the applicability check (defensive_generation, item 3), and
the survivors are added tagged ISSUE_SPOTTING_SOURCE, protected like the
keyword (bm25) and cross-reference rows.

Used by the chat graph (node `issue_spotting`, after context_retrieval on
article_router's "pass" branch) and by the strategy pipeline (_select_articles),
whose own case analysis only knows six codes. Everything fails open: any error
adds nothing.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from . import feature_flags
from .ai_chat import _call_chat
from .verbose_logger import vlog

logger = logging.getLogger(__name__)

ISSUE_SPOTTING_FLAG = "ISSUE_SPOTTING_ENABLED"
NARRATIVE_MIN_WORDS = 20
MAX_PROPOSALS = 8

_SUFFIX = r"(?:bis|ter|quater|quinquies|sexies|septies|octies|novies|decies)"
_ARTICLE = re.compile(rf"^\s*(?:art(?:icolo)?\.?\s*)?(\d+)(?:[\s-]*({_SUFFIX}))?\s*$", re.IGNORECASE)

_SPOT_SYSTEM = (
    "Sei un avvocato italiano. Leggi il caso e individua gli istituti giuridici in gioco: per ciascuno "
    "l'articolo che lo disciplina e il codice o la legge che lo contiene. Includi anche le qualificazioni "
    "alternative e le norme utili alla difesa. Rispondi SOLO con un array JSON, al massimo "
    f"{MAX_PROPOSALS} elementi, nessun altro testo:\n"
    '[{"institution": "<istituto>", "article": "<numero, con bis/ter se c\'è>", '
    '"code": "<nome del codice o della legge, come si scrive in italiano>"}]'
)

# Abbreviations a model writes for the codes, as the words the document names use.
_ABBREVIATIONS = [
    (r"^c\.?\s*p\.?\s*p\.?$|^cod\.?\s*proc\.?\s*pen\.?$", "codice di procedura penale"),
    (r"^c\.?\s*p\.?\s*c\.?$|^cod\.?\s*proc\.?\s*civ\.?$", "codice di procedura civile"),
    (r"^c\.?\s*p\.?$|^cod\.?\s*pen\.?$", "codice penale"),
    (r"^c\.?\s*c\.?$|^cod\.?\s*civ\.?$", "codice civile"),
    (r"^c\.?\s*d\.?\s*s\.?$", "codice della strada"),
    (r"^cod\.?\s*cons\.?$|^c\.?\s*cons\.?$", "codice del consumo"),
]
# Not codes or laws: commentaries and collective agreements are never the target.
_NOT_A_LAW = re.compile(r"commentat|^\W*ccnl|fiscalit", re.IGNORECASE)
_STOP = {"del", "della", "dello", "dei", "delle", "di", "il", "la", "lo", "le", "gli", "n", "art", "nr"}


def is_narrative(text: str) -> bool:
    return len((text or "").split()) >= NARRATIVE_MIN_WORDS


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^\w/]+", " ", (text or "").lower()).split())


def _article(value) -> Optional[str]:
    m = _ARTICLE.match(str(value or ""))
    if not m:
        return None
    return f"{m.group(1)}-{m.group(2).lower()}" if m.group(2) else m.group(1)


def spot_issues(text: str) -> List[Dict[str, str]]:
    """[{institution, article, code}] proposed by the model; [] on any failure."""
    try:
        raw = _call_chat([SystemMessage(content=_SPOT_SYSTEM), HumanMessage(content=text[:6000])], max_tokens=600)
        proposals = json.loads(re.sub(r"```(?:json)?\s*", "", raw or "").strip().rstrip("`").strip())
    except Exception as exc:
        logger.warning("issue spotting: proposal call failed: %s", exc)
        return []
    out, seen = [], set()
    for p in proposals if isinstance(proposals, list) else []:
        if not isinstance(p, dict):
            continue
        number, code = _article(p.get("article")), str(p.get("code") or "").strip()
        if number and code and (number, code.lower()) not in seen:
            seen.add((number, code.lower()))
            out.append({"institution": str(p.get("institution") or "").strip(), "article": number, "code": code})
    return out[:MAX_PROPOSALS]


def resolve_code(code: str, documents: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The document a code or law name refers to, from the real document names
    (and their aliases); None when nothing matches clearly.

    "c.p." / "codice penale" -> the document whose name contains "codice
    penale" (not "codice di procedura penale", not a commentary); "legge
    392/1978" / "L. n. 392 del 1978" -> a document naming both 392 and 1978.
    """
    phrase = _norm(code)
    for pattern, words in _ABBREVIATIONS:
        if re.match(pattern, code.strip(), re.IGNORECASE):
            phrase = words
            break
    laws = [d for d in documents if d.get("name") and not _NOT_A_LAW.search(d["name"])]

    def searchable(d):
        return _norm(" ".join([d["name"]] + list(d.get("aliases") or [])))

    number = re.search(r"\b(\d{1,4})\s*(?:/|del\s+|del\b|,\s*)\s*((?:19|20)\d{2})\b", phrase)
    if number:   # a law's number and year; if no name carries them, try the words
        n, year = number.group(1), number.group(2)
        hits = [d for d in laws if re.search(rf"\b{n}\b", searchable(d)) and re.search(rf"\b{year}\b", searchable(d))]
        if hits:
            return min(hits, key=lambda d: len(d["name"]))

    hits = [d for d in laws if phrase and phrase in searchable(d)]
    if hits:
        starts = [d for d in hits if _norm(d["name"]).startswith(phrase)]
        return min(starts or hits, key=lambda d: len(d["name"]))
    words = {w for w in phrase.split() if w.isalpha() and w not in _STOP and len(w) > 2 and w != "lgs"}
    if len(words) < 2:
        return None
    scored = [(len(words & set(searchable(d).split())), d) for d in laws]
    best = max((s for s, _ in scored), default=0)
    if best < max(2, round(0.75 * len(words))):
        return None
    return min((d for s, d in scored if s == best), key=lambda d: len(d["name"]))


def fetch_article(session, doc_id: str, number: str, visibility: str = "", **params) -> List[Dict[str, Any]]:
    """The article's sections in that one document: its canonical name
    ("N.0.0" or "N"), else its dot-delimited fragments ("N.1", "N.2"), never
    another article that merely starts with the same digits."""
    rows = session.run(
        "MATCH (d:Document)-[:CONTAINS]->(s:Section) "
        "WHERE d.id = $doc_id AND s.name IN [$zero_zero, $num] " + (f"AND {visibility} " if visibility else "")
        + "WITH d, s.name AS name, head(collect(s)) AS s RETURN d, s, name",
        doc_id=doc_id, num=number, zero_zero=f"{number}.0.0", **params,
    ).data()
    if rows:
        rows.sort(key=lambda r: 0 if r.get("name") == f"{number}.0.0" else 1)
        return [{"d": r["d"], "s": r["s"]} for r in rows[:1]]
    return [{"d": r["d"], "s": r["s"]} for r in session.run(
        "MATCH (d:Document)-[:CONTAINS]->(s:Section) "
        "WHERE d.id = $doc_id AND s.name STARTS WITH $num + '.' " + (f"AND {visibility} " if visibility else "")
        + "WITH d, s.name AS name, head(collect(s)) AS s RETURN d, s ORDER BY name LIMIT 3",
        doc_id=doc_id, num=number, **params,
    ).data()]


def _article_tag(number: str, doc_name: str) -> tuple:
    """(number, code) as the strategy pipeline tags rows: "c.p." for the
    Codice Penale, the document's own name for any other law."""
    from .defensive_generation import _CODE_PREFIXES
    for abbr, prefix in sorted(_CODE_PREFIXES.items(), key=lambda kv: -len(kv[1])):
        if doc_name.lower().startswith(prefix.lower()):
            return (number, abbr)
    return (number, doc_name)


def spotted_rows(session, documents: List[Dict[str, Any]], text: str, existing: List[Dict[str, Any]],
                 visibility: str = "", **params) -> List[Dict[str, Any]]:
    """The proposed articles that exist, pass the applicability check and are
    not already among `existing` (same document and section name)."""
    from .defensive_generation import _filter_applicable_articles
    from .nodes.cypher import ISSUE_SPOTTING_SOURCE

    proposals = spot_issues(text)
    have = {((r.get("d") or {}).get("id"), (r.get("s") or {}).get("name")) for r in existing}
    unresolved, missing, fetched = [], [], []
    for p in proposals:
        doc = resolve_code(p["code"], documents)
        if not doc:
            unresolved.append(f"art. {p['article']} {p['code']}")
            continue
        rows = fetch_article(session, doc["id"], p["article"], visibility, **params)
        if not rows:
            missing.append(f"art. {p['article']} {doc['name']}")
        for row in rows:
            key = ((row.get("d") or {}).get("id"), (row.get("s") or {}).get("name"))
            if key in have:
                continue
            have.add(key)
            row["_source"] = ISSUE_SPOTTING_SOURCE
            row["_article"] = _article_tag(p["article"], doc["name"])
            row["_institution"] = p["institution"]
            fetched.append(row)
    checked = _filter_applicable_articles(text, "", fetched) if fetched else []
    vlog("issue_spotting", {
        "proposals": [f"{p['institution']}: art. {p['article']} {p['code']}" for p in proposals],
        "unresolved": unresolved, "not_found": missing,
        "fetched": [f"art. {r['_article'][0]} {r['_article'][1]}" for r in fetched],
        "dropped_by_check": [f"art. {r['_article'][0]} {r['_article'][1]}" for r in fetched if r not in checked],
        "added": len(checked),
    })
    return checked


def issue_spotting(state: Dict[str, Any], driver, database: str) -> Dict[str, Any]:
    """Graph node: adds the spotted articles to raw_result. A no-op unless
    ISSUE_SPOTTING_ENABLED, for comparisons and for short questions."""
    if not feature_flags.enabled(ISSUE_SPOTTING_FLAG) or state.get("is_comparison"):
        return {}
    query = state.get("query") or ""
    if not is_narrative(query):
        return {}
    try:
        from .answer_processing import _visibility_filter
        from .doc_lookup import _fetch_doc_names
        documents = _fetch_doc_names(driver, database)
        existing = list(state.get("raw_result") or [])
        with driver.session(database=database) as session:
            added = spotted_rows(session, documents, query, existing, _visibility_filter(),
                                 user_id=state.get("user_id") or "", tenant_id=state.get("tenant_id") or "")
    except Exception as exc:
        logger.warning("issue spotting failed, nothing added: %s", exc)
        return {}
    return {"raw_result": existing + added} if added else {}
