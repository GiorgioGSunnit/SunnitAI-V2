"""Source tiers (26 Oct): the AUTHORITY tier is the primary documents (codes and laws), the INTERPRETATION tier
the rulings, always filtered by what the authority tier found.

  PRIMARY_FIRST_ENABLED (the chat's direct questions, src/rag/main.py): every article-finding node of the graph
      keeps only rows of primary documents (and special ones, the books the chat cites as doctrine); each node's
      hits are logged by document type, and a check after the filter asserts none is left. Rulings come only as
      support: those whose Sections CITE an article the answer is built on (Cassation boilerplate excluded),
      ranked by similarity to the question -- no vote.

The CITES edges are (:Section)-[:CITES {code, article}]->(:Section), scripts/link_ruling_citations.py, for the
six codes and the Constitution.
"""

import logging
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from . import feature_flags
from .verbose_logger import vlog

logger = logging.getLogger(__name__)

PRIMARY_FIRST_FLAG = "PRIMARY_FIRST_ENABLED"
DIRECT_TYPES = ("primary", "special")
SUPPORT_SOURCE = "ruling_support"


def boilerplate() -> set:
    from .vote_pool import BOILERPLATE
    return BOILERPLATE


# --- document types -------------------------------------------------------------------------------------

def row_doc(row: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """(document id, document type) of a graph row: its "d" node, else a Document-like value, else the
    Section's own id ("DOCUMENT_SECTION::<doc>::<name>"); the type is None when the row does not carry it."""
    d = row.get("d")
    if isinstance(d, dict) and (d.get("id") or d.get("document_id")):
        return d.get("id") or d.get("document_id"), d.get("document_type")
    for key, value in row.items():
        if isinstance(value, dict) and "document_type" in value and (value.get("id") or value.get("document_id")):
            return value.get("id") or value.get("document_id"), value.get("document_type")
    s = row.get("s")
    if isinstance(s, dict):
        parts = str(s.get("id") or "").split("::")
        if len(parts) >= 3:
            return "LEGAL_DOC::" + parts[1], None
        if s.get("document_id"):
            return s["document_id"], None
    return None, None


def doc_types(session, rows: Sequence[Dict[str, Any]]) -> List[Optional[str]]:
    """Each row's document type ("unknown" for a row with no document at all), looking up the ones its row
    does not carry in one query."""
    found = [row_doc(r) for r in rows]
    missing = sorted({doc for doc, kind in found if doc and kind is None})
    looked: Dict[str, Optional[str]] = {}
    if missing and session is not None:
        try:
            looked = {r["id"]: r["type"] for r in session.run(
                "MATCH (d:Document) WHERE d.id IN $ids RETURN d.id AS id, d.document_type AS type",
                ids=missing).data()}
        except Exception as exc:
            logger.warning("source tiers: document types not read: %s", exc)
    out = []
    for doc, kind in found:
        if doc is None:
            out.append("unknown")
        else:
            out.append(kind if kind is not None else looked.get(doc) or "null")
    return out


def keep_types(session, rows: List[Dict[str, Any]], allowed: Iterable[str], stage: str) -> List[Dict[str, Any]]:
    """The rows of an allowed document type; rows with no document at all are kept (they carry no source).
    Logs the hits by type before and the non-allowed ones left after ("source_tier_hits", which must be 0)."""
    allowed = set(allowed)
    if not rows:
        return rows
    types = doc_types(session, rows)
    by_type: Dict[str, int] = {}
    for kind in types:
        by_type[kind] = by_type.get(kind, 0) + 1
    kept = [r for r, kind in zip(rows, types) if kind in allowed or kind == "unknown"]
    # The check on the filter's own output, row by row as it leaves (a row whose type its own "d" now
    # contradicts counts too).
    left = sum(1 for r in kept if (row_doc(r)[1] or "primary") not in allowed and row_doc(r)[0] is not None)
    vlog("source_tier_hits", {"stage": stage, "by_type": by_type, "dropped": len(rows) - len(kept),
                              "non_primary_after": left})
    if left:
        logger.error("source tiers: %d non-%s rows left after the filter at %s", left, "/".join(sorted(allowed)),
                     stage)
    return kept


# --- rulings citing articles ----------------------------------------------------------------------------

def _code_prefixes() -> Dict[str, str]:
    from .defensive_generation import _CODE_PREFIXES
    return {**_CODE_PREFIXES, "Cost.": "Costituzione"}


def refs_of(rows: Iterable[Dict[str, Any]]) -> List[tuple]:
    """(article, code) of article rows: their own "_article" tag, else read from the document's name; only the
    codes the CITES edges know (six codes and the Constitution)."""
    from .defensive_generation import _row_to_fusion_ref
    prefixes = _code_prefixes()
    out: List[tuple] = []
    for row in rows:
        ref = row.get("_article") or _row_to_fusion_ref(row)
        if ref and ref[1] not in prefixes and str(ref[1]).startswith(prefixes["Cost."]):
            ref = (ref[0], "Cost.")
        if ref and ref[0] and ref[1] in prefixes and tuple(ref) not in out:
            out.append(tuple(ref))
    return out


def citing_rulings(session, refs: Iterable[tuple], limit: int = 300) -> Dict[str, set]:
    """{ruling document id: the refs among `refs` its Sections CITE}, Cassation boilerplate excluded, read from
    each article's own Section in a primary document of its code."""
    prefixes = _code_prefixes()
    wanted = [{"article": str(a), "code": c, "prefix": prefixes[c]} for a, c in dict.fromkeys(refs)
              if c in prefixes and (str(a), c) not in boilerplate()]
    if not wanted:
        return {}
    rows = session.run(
        "UNWIND $refs AS ref "
        "MATCH (pd:Document)-[:CONTAINS]->(t:Section) WHERE pd.document_type = 'primary' "
        "AND pd.name STARTS WITH ref.prefix AND (t.name = ref.article + '.0.0' OR t.name = ref.article "
        "OR t.name STARTS WITH ref.article + '.') "
        "MATCH (d:Document)-[:CONTAINS]->(:Section)-[c:CITES]->(t) WHERE d.document_type = 'interpretation' "
        "RETURN DISTINCT d.id AS ruling, ref.article AS article, ref.code AS code LIMIT $limit",
        refs=wanted, limit=limit).data()
    out: Dict[str, set] = {}
    for r in rows:
        out.setdefault(r["ruling"], set()).add((r["article"], r["code"]))
    return out


def ruling_citations(session, ruling_ids: Sequence[str]) -> Dict[str, set]:
    """{ruling id: every (article, code) its Sections CITE}."""
    out: Dict[str, set] = {}
    if not ruling_ids:
        return out
    for r in session.run(
            "UNWIND $ids AS id MATCH (d:Document {id: id})-[:CONTAINS]->(:Section)-[c:CITES]->() "
            "RETURN DISTINCT d.id AS ruling, c.article AS article, c.code AS code", ids=list(ruling_ids)).data():
        if r["article"] and r["code"]:
            out.setdefault(r["ruling"], set()).add((str(r["article"]), r["code"]))
    return out


def rank_by_similarity(session, ruling_ids: Sequence[str], text: str) -> Dict[str, Dict[str, Any]]:
    """{ruling id: {"sim": its best Section's cosine to the text, "d": the ruling, "s": that Section}}."""
    from . import embedding_config
    from .ai_chat import embedding_model
    if not ruling_ids or not (text or "").strip():
        return {}
    prop = embedding_config.VECTOR_PROPERTY
    embedding = list(embedding_model.embed_query(text))
    rows = session.run(
        "UNWIND $ids AS id MATCH (d:Document {id: id})-[:CONTAINS]->(s:Section) "
        f"WHERE s.{prop} IS NOT NULL "
        f"WITH d, s, vector.similarity.cosine(s.{prop}, $embedding) AS sim ORDER BY sim DESC "
        "WITH d, head(collect({s: s, sim: sim})) AS best "
        "RETURN d, best.s AS s, best.sim AS sim",
        ids=list(ruling_ids), embedding=embedding).data()
    return {r["d"]["id"]: {"sim": float(r["sim"] or 0), "d": r["d"], "s": r["s"]} for r in rows if r.get("d")}


def support_rulings(session, article_refs: Sequence[tuple], text: str, k: int,
                    weight_shared: bool = False, stage: str = "") -> List[Dict[str, Any]]:
    """The rulings citing at least one of the articles (boilerplate excluded), best first: by similarity to
    the text, times the number of articles shared when weight_shared (the narrative path); each row
    {"d", "s", "_source": SUPPORT_SOURCE, "_cites": [...], "_support_score"}. [] on any failure."""
    t0 = time.time()
    info: Dict[str, Any] = {"stage": stage, "articles": [f"{a} {c}" for a, c in article_refs]}
    try:
        cites = citing_rulings(session, article_refs)
        info["citing"] = len(cites)
        if not cites:
            return []
        if len(cites) > 150:     # the most-sharing ones are scored
            cites = dict(sorted(cites.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:150])
        sims = rank_by_similarity(session, list(cites), text)
        scored = []
        for ruling, refs in cites.items():
            if ruling not in sims:
                continue
            score = sims[ruling]["sim"] * (len(refs) if weight_shared else 1)
            scored.append((score, ruling))
        scored.sort(key=lambda x: (-x[0], x[1]))
        out = []
        for score, ruling in scored[:k]:
            hit = sims[ruling]
            out.append({"d": hit["d"], "s": hit["s"], "_source": SUPPORT_SOURCE, "_support_score": round(score, 4),
                        "_cites": sorted(f"{a} {c}" for a, c in cites[ruling])})
        info["kept"] = [{"ruling": (r["d"] or {}).get("name"), "score": r["_support_score"], "cites": r["_cites"]}
                        for r in out]
        return out
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        logger.warning("source tiers: support rulings failed: %s", exc)
        return []
    finally:
        vlog("source_tier_support", info, duration_ms=(time.time() - t0) * 1000)


def primary_first_enabled(route: Optional[str]) -> bool:
    """PRIMARY_FIRST_ENABLED, for a request routed direct."""
    return feature_flags.enabled(PRIMARY_FIRST_FLAG) and route == "direct"


# --- the chat's graph (PRIMARY_FIRST_ENABLED) -------------------------------------------------------------

DIRECT_SUPPORT_K = 4


def graph_route(state: Dict[str, Any]) -> str:
    """The route of a graph turn: the state's own, else the router's (ROUTER_ENABLED), else "direct" -- without
    the router, today's routing sent every graph turn the direct way."""
    if state.get("route"):
        return state["route"]
    from . import query_router
    if not query_router.enabled():
        return "direct"
    return query_router.classify(state.get("raw_query") or state.get("query") or "")["route"]


def primary_node(fn, stage: str, driver, database: str):
    """The graph node `fn`, its "raw_result" kept to primary and special documents on a direct turn with
    PRIMARY_FIRST_ENABLED (keep_types logs the hits by type); unchanged otherwise or on a failed check."""
    def node(state):
        out = fn(state)
        if not isinstance(out, dict) or not out.get("raw_result") or not primary_first_enabled(graph_route(state)):
            return out
        try:
            import neo4j
            with driver.session(database=database, default_access_mode=neo4j.READ_ACCESS) as session:
                return {**out, "raw_result": keep_types(session, list(out["raw_result"]), DIRECT_TYPES, stage)}
        except Exception as exc:
            logger.warning("source tiers: %s rows not checked: %s", stage, exc)
            return out
    node.__name__ = getattr(fn, "__name__", stage)
    return node


def support_node(fn, driver, database: str):
    """The synthesis node `fn`, given on a direct turn with PRIMARY_FIRST_ENABLED the rulings that CITE the
    articles it is about to answer from (support_rulings, by similarity to the question, boilerplate
    excluded): the interpretation tier, filtered by what the authority tier found."""
    def node(state):
        if not primary_first_enabled(graph_route(state)):
            return fn(state)
        rows = [r for r in state.get("raw_result") or [] if r.get("_source") != SUPPORT_SOURCE]
        try:
            import neo4j
            with driver.session(database=database, default_access_mode=neo4j.READ_ACCESS) as session:
                types = doc_types(session, rows)
                refs = refs_of(r for r, kind in zip(rows, types) if kind == "primary")
                support = support_rulings(session, refs, state.get("raw_query") or state.get("query") or "",
                                          DIRECT_SUPPORT_K, stage="direct")
        except Exception as exc:
            logger.warning("source tiers: no support rulings: %s", exc)
            support = []
        state = {**state, "raw_result": rows + support}
        out = fn(state)
        if isinstance(out, dict) and "raw_result" not in out:
            out = {**out, "raw_result": state["raw_result"]}
        return out
    node.__name__ = getattr(fn, "__name__", "synthesize_answer")
    return node


# --- measuring (tests/eval_direct.py, scripts/diagnose_long.py) ---------------------------------------------

def ruling_answer_fit(cites_by_ruling: Dict[str, Iterable[tuple]], answer_refs: Iterable[tuple]) -> Dict[str, Any]:
    """For the rulings an answer cites, against the articles it names: {"rulings": n, "citing_answer_article":
    those whose CITES reach one of them, "boilerplate_only": those reaching only boilerplate ones (target 0),
    "unrelated": those reaching none, "per_ruling": {id: [shared refs]}}."""
    answer = {(str(a), c) for a, c in answer_refs}
    plate = boilerplate()
    per_ruling, citing, plate_only, unrelated = {}, 0, 0, 0
    for ruling, refs in cites_by_ruling.items():
        shared = sorted({(str(a), c) for a, c in refs} & answer)
        per_ruling[ruling] = [f"{a} {c}" for a, c in shared]
        if not shared:
            unrelated += 1
        elif all(ref in plate for ref in shared):
            plate_only += 1
        else:
            citing += 1
    n = len(cites_by_ruling)
    return {"rulings": n, "citing_answer_article": citing + plate_only, "boilerplate_only": plate_only,
            "unrelated": unrelated, "share_citing": round((citing + plate_only) / n, 3) if n else None,
            "per_ruling": per_ruling}
