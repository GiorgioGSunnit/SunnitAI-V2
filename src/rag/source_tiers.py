"""Source tiers (26 Oct): the AUTHORITY tier is the primary documents (codes and laws), the INTERPRETATION tier
the rulings, always filtered by what the authority tier found.

  PRIMARY_FIRST_ENABLED (the chat's direct questions, src/rag/main.py): every article-finding node of the graph
      keeps only rows of primary documents (and special ones, the books the chat cites as doctrine), failing
      open: a row whose type is on neither the row nor its Document is kept, only a known other type is dropped;
      each node's hits are logged by document type, and a check after the filter asserts none is left. Rulings come only as
      support: those whose Sections CITE an article the answer is built on (Cassation boilerplate excluded),
      ranked by similarity to the question -- no vote.
  HYBRID_NARRATIVE_ENABLED (the strategy pipeline, defensive_generation): the rulings similar to the facts act
      first as a MAP -- the issues and articles they suggest are hypotheses only, never candidates by themselves
      (ruling_map); the article channels search primary documents only (keep_types, "authority"); a hypothesis becomes
      a candidate when the article exists in the primary corpus and a primary channel finds it, or three or more
      of the top similar rulings of the case's area apply it (confirmed_hypotheses); the rulings the writer gets
      cite at least one selected article (support_rulings, by fact similarity x shared articles).

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
HYBRID_FLAG = "HYBRID_NARRATIVE_ENABLED"
AUTHORITY_TYPES = ("primary",)
DIRECT_TYPES = ("primary", "special")
SUPPORT_SOURCE = "ruling_support"
# A hypothesis no primary channel found is a candidate only when this many of the top similar rulings apply it.
MAP_MIN_RULINGS = 3
MAP_TOP_RULINGS = 20


def boilerplate() -> set:
    from .vote_pool import BOILERPLATE
    return BOILERPLATE


# --- document types -------------------------------------------------------------------------------------

def row_doc(row: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """(document id, document type) of a graph row: its "d" node, else a Document-like value, else the
    Section's own id ("DOCUMENT_SECTION::<doc>::<name>"); the type is None when the row does not carry it."""
    doc, kind, _ = row_doc_source(row)
    return doc, kind


def row_doc_source(row: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], str]:
    """row_doc, plus where the row's document came from: "d.document_type" (the key the type was read
    from), "d (no document_type)", "<key>.document_type", "s.id", "s.document_id" or "none"."""
    d = row.get("d")
    if isinstance(d, dict) and (d.get("id") or d.get("document_id")):
        kind = d.get("document_type")
        return d.get("id") or d.get("document_id"), kind, "d.document_type" if kind else "d (no document_type)"
    for key, value in row.items():
        if isinstance(value, dict) and value.get("document_type") and (value.get("id") or value.get("document_id")):
            return value.get("id") or value.get("document_id"), value["document_type"], f"{key}.document_type"
    s = row.get("s")
    if isinstance(s, dict):
        parts = str(s.get("id") or "").split("::")
        if len(parts) >= 3:
            return "LEGAL_DOC::" + parts[1], None, "s.id"
        if s.get("document_id"):
            return s["document_id"], None, "s.document_id"
    return None, None, "none"


def _lookup_types(session, ids: Sequence[str]) -> Dict[str, Optional[str]]:
    """{document id: its document_type} for the ids, read from their Documents in one query; {} on failure."""
    if not ids or session is None:
        return {}
    try:
        return {r["id"]: r["type"] for r in session.run(
            "MATCH (d:Document) WHERE d.id IN $ids RETURN d.id AS id, d.document_type AS type",
            ids=list(ids)).data()}
    except Exception as exc:
        logger.warning("source tiers: document types not read: %s", exc)
        return {}


def doc_types(session, rows: Sequence[Dict[str, Any]]) -> List[str]:
    """Each row's document type, looking up from its Document the ones its row does not carry (one query);
    "unknown" for a row whose type cannot be told -- no document, no such Document, or a Document with no
    document_type."""
    return [kind for kind, _ in _typed(session, rows)]


def _typed(session, rows: Sequence[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """(type, where it was read) per row: "<key>.document_type", "Document lookup", or why it is unknown."""
    found = [row_doc_source(r) for r in rows]
    looked = _lookup_types(session, sorted({doc for doc, kind, _ in found if doc and kind is None}))
    out = []
    for doc, kind, source in found:
        if kind:
            out.append((kind, source))
        elif doc is None:
            out.append(("unknown", "no document on the row"))
        elif looked.get(doc):
            out.append((looked[doc], f"Document lookup ({source} -> {doc})"))
        elif doc in looked:
            out.append(("unknown", f"Document {doc} has no document_type"))
        else:
            out.append(("unknown", f"no Document {doc} found ({source})"))
    return out


# tests/eval_direct.py --debug: when a list, every wrapped node appends its rows and the filter's decision on
# each (primary_node), with or without PRIMARY_FIRST_ENABLED. None in the app.
DEBUG_ROWS: Optional[List[Dict[str, Any]]] = None


def _decisions(session, rows: Sequence[Dict[str, Any]], allowed: set) -> List[Dict[str, Any]]:
    """Per row: its document, type, where the type was read, kept or dropped and why. Fails open: only a row
    whose type is known and not allowed is dropped; a row with no type, even after the lookup, is kept."""
    out = []
    for row, (kind, source) in zip(rows, _typed(session, rows)):
        doc = row_doc(row)[0]
        if kind == "unknown":
            keep, why = True, f"type unknown ({source}): kept, fail open"
        elif kind in allowed:
            keep, why = True, f"{kind} allowed"
        else:
            keep, why = False, f"{kind} not in {'/'.join(sorted(allowed))}"
        out.append({"doc": doc, "name": (row.get("d") or {}).get("name") if isinstance(row.get("d"), dict) else None,
                    "section": (row.get("s") or {}).get("name") if isinstance(row.get("s"), dict) else None,
                    "type": kind, "type_from": source, "kept": keep, "why": why})
    return out


def keep_types(session, rows: List[Dict[str, Any]], allowed: Iterable[str], stage: str) -> List[Dict[str, Any]]:
    """The rows of an allowed document type -- failing open: a row whose type is missing (not on the row,
    not on its Document, no Document at all) is kept, only a known other type is dropped. Logs the hits by
    type, kept and dropped ("source_tier_hits"), and the non-allowed ones left after (must be 0)."""
    allowed = set(allowed)
    if not rows:
        return rows
    decisions = _decisions(session, rows, allowed)
    by_type: Dict[str, int] = {}
    for item in decisions:
        by_type[item["type"]] = by_type.get(item["type"], 0) + 1
    kept = [r for r, item in zip(rows, decisions) if item["kept"]]
    # The check on the filter's own output, row by row as it leaves (a row whose type its own "d" now
    # contradicts counts too).
    left = sum(1 for r in kept if (row_doc(r)[1] or "primary") not in allowed and row_doc(r)[0] is not None)
    dropped = len(rows) - len(kept)
    vlog("source_tier_hits", {"stage": stage, "rows": len(rows), "kept": len(kept), "by_type": by_type,
                              "dropped": dropped, "unknown_kept": by_type.get("unknown", 0),
                              "looked_up": sum(1 for d in decisions if d["type_from"].startswith("Document lookup")),
                              "non_primary_after": left})
    logger.info("source tiers: %s %d rows, kept %d, dropped %d, by type %s", stage, len(rows), len(kept), dropped,
                by_type)
    if DEBUG_ROWS is not None:
        DEBUG_ROWS.append({"stage": stage, "filter": True, "decisions": decisions})
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


# --- the narrative path's map (HYBRID_NARRATIVE_ENABLED) -------------------------------------------------

def ruling_map(session, driver, database: str, case_text: str, index: str,
               analysis: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The rulings most similar to the facts, read as a map -- hypotheses, never candidates by themselves:
      "articles": [(article, code)] they apply (cards' applied articles where carded, else CITES), IDF-weighted,
                  boilerplate and other areas' codes dropped (vote_pool.filter_refs), best first;
      "stats":    {ref: {"rulings": the top similar rulings applying it, ...}} (vote_pool.vote_stats);
      "issues":   the issues the carded ones decide (parole_chiave_questioni), most frequent first;
      "top":      the similar rulings' ids, best first.
    Empty on any failure."""
    from . import ruling_cards, vote_pool
    t0 = time.time()
    out: Dict[str, Any] = {"articles": [], "stats": {}, "issues": [], "top": []}
    info: Dict[str, Any] = {}
    try:
        table = vote_pool.idf_table(driver, database)
        if table is None:
            info["skipped"] = "IDF counts not ready"
            return out
        idf, default_idf = table
        k = vote_pool.DEFAULT_K
        vector, fulltext = vote_pool._similar_rulings(session, case_text, index, k)
        sims = vote_pool.ruling_sims(vector, fulltext)
        top = [r for r, _ in sorted(sims.items(), key=lambda kv: (-kv[1], kv[0]))]
        cites = vote_pool._citations(session, top)
        cards = vote_pool._card_refs()
        areas = vote_pool.filter_areas(analysis)
        posture = str(((analysis or {}).get("frame") or {}).get("posture_type") or "")
        votes, dropped = vote_pool.filter_refs({r: cards[r] if r in cards else cites.get(r, ()) for r in top},
                                               areas, posture, case_text)
        scores = vote_pool.vote_scores(sims, votes, idf, default_idf)
        out["articles"] = sorted((r for r in scores if scores[r] > 0), key=lambda r: (-scores[r], r))
        head = top[:MAP_TOP_RULINGS]
        out["stats"] = vote_pool.vote_stats({r: votes.get(r, []) for r in head}, {r for r in head if r in cards})
        out["top"] = top
        store = ruling_cards.store()
        issues: Dict[str, int] = {}
        if store is not None:
            by_id = {c["id"]: c for c in store.cards}
            for ruling in head:
                for issue in (by_id.get(ruling) or {}).get("issues") or []:
                    issues[issue] = issues.get(issue, 0) + 1
        out["issues"] = [i for i, _ in sorted(issues.items(), key=lambda kv: (-kv[1], kv[0]))][:6]
        info.update(rulings=len(top), carded=sum(r in cards for r in head), areas=areas,
                    dropped={reason: [f"{a} {c}" for a, c in refs[:20]] for reason, refs in dropped.items()},
                    hypotheses=[f"{a} {c} {scores[(a, c)]:.2f}" for a, c in out["articles"][:15]],
                    issues=out["issues"])
        return out
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        logger.warning("ruling map failed, no hypotheses: %s", exc)
        return {"articles": [], "stats": {}, "issues": [], "top": []}
    finally:
        vlog("hybrid_map", info, duration_ms=(time.time() - t0) * 1000)


def confirmed_hypotheses(hypotheses: Sequence[tuple], channel_refs: Iterable[tuple],
                         stats: Dict[tuple, Dict[str, int]], min_rulings: int = MAP_MIN_RULINGS
                         ) -> Tuple[List[tuple], Dict[str, List[str]]]:
    """(the hypotheses that may become candidates, why each did or did not): found by a primary channel, or
    applied by min_rulings or more of the top similar rulings of the case's area (the map already dropped
    other areas' codes). Whether the article exists in the primary corpus is the fetch's to say."""
    found = set(channel_refs)
    kept, why = [], {"channel": [], "rulings": [], "rejected": []}
    for ref in hypotheses:
        if ref in found:
            kept.append(ref)
            why["channel"].append(f"{ref[0]} {ref[1]}")
        elif (stats.get(ref) or {}).get("rulings", 0) >= min_rulings:
            kept.append(ref)
            why["rulings"].append(f"{ref[0]} {ref[1]}")
        else:
            why["rejected"].append(f"{ref[0]} {ref[1]}")
    return kept, why


def hybrid_enabled(route: Optional[str]) -> bool:
    """HYBRID_NARRATIVE_ENABLED, for a request routed narrative (query_router.route_for_pipeline)."""
    return feature_flags.enabled(HYBRID_FLAG) and route == "narrative"


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


PRIMARY_RERUN_K = 10
# The index is asked for this many nodes, so the primary-document filter inside the query still leaves k when
# rulings fill the index's own top ranks.
PRIMARY_RERUN_FETCH = 400
PRIMARY_RERUN_SOURCE = "primary_rerun"


def _rerun_texts(state: Dict[str, Any]) -> Tuple[str, str]:
    """(fulltext query, vector text) for a channel's second pass: the turn's keywords, else its query."""
    query = state.get("generalized_query") or state.get("raw_query") or state.get("query") or ""
    return " ".join(state.get("retrieval_keywords") or []) or query, query


def primary_search(session, fulltext_text: str, vector_text: str, k: int = PRIMARY_RERUN_K,
                   user_id: str = "", tenant_id: str = "") -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """The top-k Sections of primary documents for the texts, fulltext and vector, the document_type filter
    inside each query (over-fetched from the index, PRIMARY_RERUN_FETCH): rows {"d", "s", "_source",
    "_score"}, fulltext first, one per (document, section name). ([], info) on failure."""
    from . import embedding_config
    from .answer_processing import _visibility_filter
    from .lookups import VECTOR_INDEX_SETTINGS, _cached_embed_query, _index_config, _sanitize_fulltext_query
    where = f"d.document_type = 'primary' AND {_visibility_filter()}"
    info: Dict[str, Any] = {"fulltext_query": fulltext_text, "fulltext": 0, "vector": 0}
    fulltext: List[Dict[str, Any]] = []
    q = _sanitize_fulltext_query(fulltext_text or "")
    if q:
        try:
            fulltext = session.run(
                "CALL db.index.fulltext.queryNodes('section_fulltext', $q, {limit: $fetch}) YIELD node, score "
                f"MATCH (d:Document)-[:CONTAINS]->(node) WHERE {where} "
                "RETURN d, node AS s, score ORDER BY score DESC LIMIT $k",
                q=q, fetch=PRIMARY_RERUN_FETCH, k=k, user_id=user_id, tenant_id=tenant_id).data()
        except Exception as exc:
            info["fulltext_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            logger.warning("source tiers: primary fulltext pass failed: %s", exc)
    vector: List[Dict[str, Any]] = []
    # The live search's own index, query embedding and threshold (lookups.vector_lookup): the configured
    # suffix (EMBEDDING_INDEX_SUFFIX, "section_embeddings_v2"), the query mode's instruction
    # (EMBEDDING_QUERY_MODE), the text-mode min_score.
    base = "section_embeddings"
    index = embedding_config.vector_index_name(base)
    min_score = _index_config(base, VECTOR_INDEX_SETTINGS).get("min_score")
    info.update(vector_index=index, vector_property=embedding_config.VECTOR_PROPERTY,
                query_mode=embedding_config.QUERY_MODE, min_score=min_score)
    if (vector_text or "").strip() and embedding_config.searched(base):
        try:
            vector = session.run(
                "CALL db.index.vector.queryNodes($index, $fetch, $embedding) YIELD node, score "
                "WHERE 'Section' IN labels(node) AND node.plain_text IS NOT NULL AND node.plain_text <> '' "
                "AND score >= $min_score "
                f"MATCH (d:Document)-[:CONTAINS]->(node) WHERE {where} "
                "RETURN d, node AS s, score ORDER BY score DESC LIMIT $k",
                index=index, fetch=PRIMARY_RERUN_FETCH, k=k, min_score=-1.0 if min_score is None else min_score,
                embedding=list(_cached_embed_query(vector_text)), user_id=user_id, tenant_id=tenant_id).data()
        except Exception as exc:
            info["vector_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            logger.warning("source tiers: primary vector pass failed: %s", exc)
    info["fulltext"], info["vector"] = len(fulltext), len(vector)
    rows, seen = [], set()
    for channel, found in (("fulltext", fulltext), ("vector", vector)):
        for r in found:
            d, s = r.get("d") or {}, r.get("s") or {}
            key = (d.get("id"), s.get("name") or s.get("id"))
            if key in seen:
                continue
            seen.add(key)
            rows.append({"d": d, "s": s, "_source": f"{PRIMARY_RERUN_SOURCE}_{channel}", "_score": r.get("score")})
    info["rows"] = len(rows)
    info["kept"] = [f"{(r['d'] or {}).get('name')} {(r['s'] or {}).get('name')}" for r in rows[:2 * k]]
    return rows, info


def primary_node(fn, stage: str, driver, database: str):
    """The graph node `fn`, its "raw_result" kept to primary and special documents on a direct turn with
    PRIMARY_FIRST_ENABLED (keep_types logs the hits by type); unchanged otherwise or on a failed check.
    When the filter leaves none of the node's rows, the channel runs again restricted to primary documents
    (primary_search) so it gives its top-k primary rows instead of nothing; both passes are logged
    ("source_tier_hits", then "source_tier_rerun")."""
    def node(state):
        out = fn(state)
        if not isinstance(out, dict) or not out.get("raw_result"):
            return out
        route = graph_route(state)
        if not primary_first_enabled(route):
            if DEBUG_ROWS is not None:
                DEBUG_ROWS.append({"stage": stage, "filter": False, "route": route,
                                   "rows": [dict(zip(("doc", "type", "type_from"), row_doc_source(r)),
                                                 section=(r.get("s") or {}).get("name")
                                                 if isinstance(r.get("s"), dict) else None)
                                            for r in out["raw_result"]]})
            return out
        try:
            import neo4j
            with driver.session(database=database, default_access_mode=neo4j.READ_ACCESS) as session:
                rows = list(out["raw_result"])
                kept = keep_types(session, rows, DIRECT_TYPES, stage)
                if not kept:
                    kept = _rerun(session, state, stage, rows)
                return {**out, "raw_result": kept}
        except Exception as exc:
            logger.warning("source tiers: %s rows not checked: %s", stage, exc)
            return out
    node.__name__ = getattr(fn, "__name__", stage)
    return node


def _rerun(session, state: Dict[str, Any], stage: str, first: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The second pass of a channel whose every row the filter dropped: primary_search on the turn's texts."""
    t0 = time.time()
    fulltext_text, vector_text = _rerun_texts(state)
    rows, info = primary_search(session, fulltext_text, vector_text, PRIMARY_RERUN_K,
                                state.get("user_id") or "", state.get("tenant_id") or "")
    vlog("source_tier_rerun", {"stage": stage, "first_pass": {"rows": len(first), "kept": 0},
                               "second_pass": info}, duration_ms=(time.time() - t0) * 1000)
    logger.info("source tiers: %s kept none of %d rows; primary pass gave %d (fulltext %d, vector %d on %s)",
                stage, len(first), len(rows), info["fulltext"], info["vector"], info.get("vector_index"))
    if DEBUG_ROWS is not None:
        DEBUG_ROWS.append({"stage": f"{stage} (primary pass)", "filter": False, "route": "direct",
                           "rows": [dict(zip(("doc", "type", "type_from"), row_doc_source(r)),
                                         section=(r.get("s") or {}).get("name")) for r in rows]})
    return rows


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
