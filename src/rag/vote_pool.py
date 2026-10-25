"""VOTE_POOL_ENABLED: the articles that the rulings most similar to the case's facts cite, added to
the fusion pool (_select_articles_fusion).

scripts/exp_distill.py --vote (prod, 24 Oct, 32 cases): the vote alone recalled about as much as
FUSION_CLEAN, but other articles -- 12 expected articles FUSION_CLEAN misses entirely and 22 it
ranks below its top 20 (1158/1079, 43/40, 476/481, 1218, 1453, 1117, 56, 61, 378, 1668, 1140 ...),
at similar noise. The cards' applied articles voted about as the citations did.

  1. Rulings by FACTS, never by the pool's articles: the K ruling Sections most similar to the
     case text by vector and the K by fulltext (rulings only), grouped by ruling, each channel's
     scores divided by its best, a ruling's best section kept (ruling_sims).
  2. Each ruling votes for the articles its (:Section)-[:CITES]->(:Section) edges point to
     (scripts/link_ruling_citations.py) or, when it has a card (ruling_cards.py's files), for
     the card's articoli_applicati: score = sum of sim x log(N rulings / rulings citing it) (vote).
  3. The vote joins the pool as VOTE_POOL_MODE says: "append:N", the first N voted articles
     not already in the capped pool are added after it; "rrf:W_POOL:W_VOTE", the pool's ranking
     and the vote's merged by weighted reciprocal rank before the cap. scripts/vote_pool_combine.py
     measures both offline from exp_distill's vote_results.json.

Before scoring (2c dry run, 24 Oct: the top 15 carried 360/112/115 c.p.c., 606/611 c.p.p., 2697 c.c.,
111 Cost., and c.p.c. articles in a criminal case, all judged "applicable"), each ruling's votes are
filtered (filter_refs): a main code outside the case's areas (criminal c.p./c.p.p., administrative
c.p.a., any other area c.c./c.p.c.) is dropped unless the user's text names it; Cassation-procedure
boilerplate (BOILERPLATE) is dropped unless the case is an appeal to the Cassation.

The voted articles that make the cut are tagged _vote: the applicability check still judges them;
the draft prompt keeps them like multi-signal ones; the plan is told to weigh them and its check
never removes them (plan_then_write.py).

IDF: a JSON file (VOTE_POOL_IDF_FILE: {"n": N, "df": [[code, article, rulings], ...]}, or
exp_distill's vote_cache.json, whose "_idf" is that), read again when it changes; without one, the
counts are made once per process in the background (one pass over the CITES edges), and until they
are ready the vote is skipped. Any failure leaves the pool as it was.
"""

import json
import logging
import math
import os
import re
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from . import feature_flags
from .verbose_logger import vlog

logger = logging.getLogger(__name__)

VOTE_POOL_FLAG = "VOTE_POOL_ENABLED"
MODE_ENV, K_ENV, IDF_FILE_ENV = "VOTE_POOL_MODE", "VOTE_POOL_K", "VOTE_POOL_IDF_FILE"
DEFAULT_MODE = "append:6"
DEFAULT_K = 60
RRF_K0 = 60
# With "rrf", the voted articles among the vote's top TAG_TOP that make the cut are tagged _vote.
TAG_TOP = 20
_OVERFETCH = 5   # the indexes are queried for K x this, so K rulings' sections survive the filter


def enabled() -> bool:
    return feature_flags.enabled(VOTE_POOL_FLAG)


# --- pure: which votes count for the case --------------------------------------------------------

# The main codes, by the case frame's areas (method_candidates.AREA_ARTICLES' keys); any area not
# listed is civil. A code outside these (Cost., c.d.s.) is never dropped for its area.
AREA_CODES = {"penale": {"c.p.", "c.p.p."}, "amministrativo": {"c.p.a."}}
CIVIL_CODES = {"c.c.", "c.p.c."}
MAIN_CODES = {"c.p.", "c.p.p.", "c.c.", "c.p.c.", "c.p.a."}
# What a Cassation ruling cites for its own procedure, not for the case's law.
BOILERPLATE = {(n, "c.p.c.") for n in ("360", "366", "369", "380-bis", "391")} \
    | {(n, "c.p.p.") for n in ("606", "610", "611", "616")} | {("2697", "c.c."), ("2909", "c.c."), ("111", "Cost.")}
# The frame's postures of a case already at the Cassation (defensive_generation._POSTURE_TYPES).
CASSATION_POSTURES = {"sentenza_civile_appello", "sentenza_penale_appello"}
# A code the user's text names: with or without dots ("art. 584 cp"), or in full.
_NAMED_CODE_RES = {
    "c.p.p.": re.compile(r"\bc\.?\s?p\.?\s?p\b|codice di procedura penale|cod\.\s*proc\.\s*pen", re.IGNORECASE),
    "c.p.c.": re.compile(r"\bc\.?\s?p\.?\s?c\b|codice di procedura civile|cod\.\s*proc\.\s*civ", re.IGNORECASE),
    "c.p.a.": re.compile(r"\bc\.?\s?p\.?\s?a\b|codice del processo amministrativo|cod\.\s*proc\.\s*amm",
                         re.IGNORECASE),
    "c.p.": re.compile(r"\bc\.?\s?p\b(?!\.?\s?[cpa]\b)|codice penale|cod\.\s*pen\b", re.IGNORECASE),
    "c.c.": re.compile(r"\bc\.?\s?c\b(?!\.?\s?[p]\b)|codice civile|cod\.\s*civ\b", re.IGNORECASE),
}


def named_codes(text: str) -> set:
    """The main codes the user's text names."""
    return {code for code, pattern in _NAMED_CODE_RES.items() if pattern.search(text or "")}


def case_areas(analysis: Optional[Dict[str, Any]]) -> List[str]:
    """The case frame's areas, else the analysis' own area ("penale", "civile" ...); [] when unknown."""
    from . import method_candidates
    analysis = analysis or {}
    area = str(analysis.get("area") or "").strip().lower()
    return method_candidates.areas_for((analysis.get("frame") or {}).get("areas") or [], area) or ([area] if area else [])


def filter_areas(analysis: Optional[Dict[str, Any]]) -> List[str]:
    """The areas the area filter goes by: case_areas, or none at all (no filter) when
    FRAME_SANITY_ENABLED found the frame's areas uncertain (defensive_generation._sane_frame)."""
    frame = (analysis or {}).get("frame") or {}
    if feature_flags.enabled("FRAME_SANITY_ENABLED") and frame.get("areas_uncertain"):
        return []
    return case_areas(analysis)


def allowed_codes(areas: Iterable[str], named: Iterable[str]) -> Optional[set]:
    """The main codes a vote may name for these areas, plus those the user named; None (no
    filter) when the areas are unknown."""
    areas = list(areas)
    if not areas:
        return None
    codes = set(named)
    for area in areas:
        codes |= AREA_CODES.get(area, CIVIL_CODES)
    return codes


def drop_reason(ref: tuple, allowed: Optional[set], posture: str) -> Optional[str]:
    """Why a voted (article, code) does not count for the case, or None."""
    if ref in BOILERPLATE and posture not in CASSATION_POSTURES:
        return "boilerplate"
    if allowed is not None and ref[1] in MAIN_CODES and ref[1] not in allowed:
        return "other_area"
    return None


def filter_refs(refs_by_ruling: Dict[str, Iterable[tuple]], areas: Iterable[str], posture: str,
                text: str) -> Tuple[Dict[str, List[tuple]], Dict[str, List[tuple]]]:
    """({ruling: the refs that count}, {reason: the refs dropped, once each})."""
    allowed = allowed_codes(areas, named_codes(text))
    kept: Dict[str, List[tuple]] = {}
    dropped: Dict[str, List[tuple]] = {}
    for ruling, refs in refs_by_ruling.items():
        kept[ruling] = []
        for ref in refs:
            reason = drop_reason(ref, allowed, posture or "")
            if reason is None:
                kept[ruling].append(ref)
            elif ref not in dropped.setdefault(reason, []):
                dropped[reason].append(ref)
    return kept, dropped


# --- pure: the vote and how it joins the pool (scripts/vote_pool_combine.py uses these too) --------

def _parse(text: str) -> Optional[Tuple]:
    parts = [p.strip() for p in text.lower().split(":")]
    try:
        if parts[0] == "append" and len(parts) == 2 and int(parts[1]) >= 0:
            return ("append", int(parts[1]))
        if parts[0] == "rrf" and len(parts) == 3 and float(parts[1]) >= 0 and float(parts[2]) >= 0:
            return ("rrf", float(parts[1]), float(parts[2]))
    except ValueError:
        pass
    return None


def parse_mode(text: Optional[str]) -> Tuple:
    """("append", N) or ("rrf", w_pool, w_vote); DEFAULT_MODE for anything else."""
    parsed = _parse(text) if text and text.strip() else None
    if parsed is None and text and text.strip():
        logger.warning("%s=%r is not append:N or rrf:W_POOL:W_VOTE: using %s", MODE_ENV, text, DEFAULT_MODE)
    return parsed or _parse(DEFAULT_MODE)


def mode() -> Tuple:
    return parse_mode(os.getenv(MODE_ENV))


def ruling_sims(vector_rows: Iterable[Dict[str, Any]], fulltext_rows: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    """{ruling id: similarity}: each channel's scores divided by its best, so both run 0..1
    (a cosine and a Lucene score are not comparable), a ruling's best section over both."""
    sims: Dict[str, float] = {}
    for rows in (list(vector_rows), list(fulltext_rows)):
        top = max((r.get("score") or 0 for r in rows), default=0)
        if top <= 0:
            continue
        for r in rows:
            if r.get("doc_id"):
                sims[r["doc_id"]] = max(sims.get(r["doc_id"], 0.0), (r.get("score") or 0) / top)
    return sims


def vote_scores(sims: Dict[str, float], refs_by_ruling: Dict[str, Iterable[tuple]], idf: Dict[tuple, float],
                default_idf: float) -> Dict[tuple, float]:
    """{article: sum over the rulings voting for it of sim x IDF}; an article missing from the
    IDF table (a card's article no CITES edge has) counts as cited once."""
    scores: Dict[tuple, float] = {}
    for ruling, sim in sims.items():
        for ref in refs_by_ruling.get(ruling) or ():
            scores[ref] = scores.get(ref, 0.0) + sim * idf.get(ref, default_idf)
    return scores


def vote(sims, refs_by_ruling, idf, default_idf) -> List[tuple]:
    """The voted articles, best first (ties by the article itself, so a rerun orders them alike)."""
    scores = vote_scores(sims, refs_by_ruling, idf, default_idf)
    return sorted((r for r in scores if scores[r] > 0), key=lambda r: (-scores[r], r))


def weighted_rrf(pool: Sequence[tuple], voted: Sequence[tuple], w_pool: float, w_vote: float,
                 k0: int = RRF_K0) -> List[tuple]:
    """pool and voted merged by w/(k0 + rank) summed over the lists holding each article;
    ties keep the pool's order, then the vote's."""
    scores: Dict[tuple, float] = {}
    for weight, ranked in ((w_pool, pool), (w_vote, voted)):
        for rank, ref in enumerate(ranked, start=1):
            scores[ref] = scores.get(ref, 0.0) + weight / (k0 + rank)
    order = {ref: i for i, ref in enumerate(list(dict.fromkeys(list(pool) + list(voted))))}
    return sorted(scores, key=lambda r: (-scores[r], order[r]))


def new_votes(voted: Sequence[tuple], present: Iterable[tuple], n: int) -> List[tuple]:
    """The first n voted articles not already `present`."""
    have = set(present)
    return [ref for ref in voted if ref not in have][:n]


def append_votes(pool: Sequence[tuple], voted: Sequence[tuple], n: int, head: int) -> List[tuple]:
    """The pool's first `head`, then the first n voted articles not among them, then the rest of
    the pool: what "append:N" does to a pool capped at `head`."""
    added = new_votes(voted, pool[:head], n)
    return list(pool[:head]) + added + [ref for ref in pool[head:] if ref not in added]


def combine(pool: Sequence[tuple], voted: Sequence[tuple], how: Tuple, head: int) -> List[tuple]:
    """The pool with the vote joined as `how` (parse_mode) says."""
    if how[0] == "append":
        return append_votes(pool, voted, how[1], head)
    return weighted_rrf(pool, voted, how[1], how[2])


def tagged(voted: Sequence[tuple], selected: Iterable[tuple], added: Iterable[tuple], how: Tuple) -> set:
    """The articles of `selected` tagged _vote: those the vote added, and the selected ones among
    the vote's top (its first N with "append:N", TAG_TOP with "rrf")."""
    top = how[1] if how[0] == "append" else TAG_TOP
    return set(added) | (set(selected) & set(voted[:top]))


def idf_from(n_rulings: int, df_rows: Iterable[Sequence]) -> Dict[tuple, float]:
    """{(article, code): log(N / rulings citing it)} from [[code, article, rulings], ...]."""
    return {(str(a), c): math.log(n_rulings / df) for c, a, df in df_rows if c and a and df}


# --- the counts ----------------------------------------------------------------------------------

_lock = threading.Lock()
_file_cache: Dict[str, Any] = {"key": None, "table": None}
_computed: Dict[str, Any] = {"table": None, "started": False}


def _load_file(path: str) -> Optional[Tuple[Dict[tuple, float], float]]:
    key = (path, os.path.getmtime(path), os.path.getsize(path))
    if _file_cache["key"] != key:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        data = data.get("_idf", data)
        n = int(data["n"])
        _file_cache.update(key=key, table=(idf_from(n, data["df"]), math.log(max(n, 2))))
        logger.info("vote pool: IDF of %d articles over %d rulings from %s", len(_file_cache["table"][0]), n, path)
    return _file_cache["table"]


def _count(driver, database: str) -> None:
    import neo4j
    t0 = time.time()
    try:
        with driver.session(database=database, default_access_mode=neo4j.READ_ACCESS) as session:
            n = session.run("MATCH (d:Document {document_type: 'interpretation'}) RETURN count(d) AS n").single()["n"]
            rows = session.run(
                "MATCH (d:Document)-[:CONTAINS]->(:Section)-[r:CITES]->() "
                "WITH r.code AS code, r.article AS article, count(DISTINCT d) AS df "
                "RETURN code, article, df").data()
        _computed["table"] = (idf_from(n, [[r["code"], r["article"], r["df"]] for r in rows]), math.log(max(n, 2)))
        logger.info("vote pool: IDF of %d articles over %d rulings counted in %.0fs", len(rows), n, time.time() - t0)
    except Exception as exc:
        logger.warning("vote pool: counting the CITES edges failed, the vote stays off: %s", exc)
    finally:
        vlog("case_pool_vote_idf", {"ready": _computed["table"] is not None}, duration_ms=(time.time() - t0) * 1000)


def idf_table(driver, database: str) -> Optional[Tuple[Dict[tuple, float], float]]:
    """(IDF table, default IDF), or None while the counts are not ready."""
    path = os.getenv(IDF_FILE_ENV, "").strip()
    if path and os.path.isfile(path):
        try:
            with _lock:
                return _load_file(path)
        except Exception as exc:
            logger.warning("vote pool: %s=%s unreadable (%s): counting instead", IDF_FILE_ENV, path, exc)
    with _lock:
        if _computed["table"] is None and not _computed["started"]:
            _computed["started"] = True
            threading.Thread(target=_count, args=(driver, database), daemon=True, name="vote-pool-idf").start()
    return _computed["table"]


# --- the vote for a case ---------------------------------------------------------------------------

_cards_cache: Dict[str, Any] = {"store": None, "refs": {}}


def _card_refs() -> Dict[str, List[tuple]]:
    """{ruling id: [(article, code)]} from the ruling cards' applied articles ("584 c.p."), {} without cards."""
    from . import ruling_cards
    store = ruling_cards.store()
    if store is None:
        return {}
    if _cards_cache["store"] is not store:
        refs = {}
        for card in store.cards:
            items = [str(a).partition(" ") for a in card.get("applied") or ()]
            refs[card["id"]] = [(n, c.strip()) for n, _, c in items if n and c.strip()]
        _cards_cache.update(store=store, refs=refs)
    return _cards_cache["refs"]


def _similar_rulings(session, text: str, index: str, k: int) -> Tuple[list, list]:
    """(vector rows, fulltext rows) of ruling Sections, best first, k each: {doc_id, score}."""
    from .ai_chat import embedding_model
    from .lookups import _sanitize_fulltext_query
    vector = session.run(
        "CALL db.index.vector.queryNodes($index, $fetch, $embedding) YIELD node, score "
        "WHERE 'Section' IN labels(node) AND node.plain_text IS NOT NULL AND node.plain_text <> '' "
        "MATCH (d:Document)-[:CONTAINS]->(node) WHERE d.document_type = 'interpretation' "
        "RETURN d.id AS doc_id, score ORDER BY score DESC LIMIT $k",
        index=index, fetch=k * _OVERFETCH, k=k, embedding=list(embedding_model.embed_query(text))).data()
    q = _sanitize_fulltext_query(text)
    fulltext = session.run(
        "CALL db.index.fulltext.queryNodes('section_fulltext', $q, {limit: $fetch}) YIELD node, score "
        "MATCH (d:Document)-[:CONTAINS]->(node) WHERE d.document_type = 'interpretation' "
        "RETURN d.id AS doc_id, score ORDER BY score DESC LIMIT $k",
        q=q, fetch=k * _OVERFETCH, k=k).data() if q else []
    return vector, fulltext


def _citations(session, ruling_ids: List[str]) -> Dict[str, set]:
    out: Dict[str, set] = {}
    for r in session.run(
            "UNWIND $ids AS id MATCH (d:Document {id: id})-[:CONTAINS]->(:Section)-[r:CITES]->() "
            "RETURN DISTINCT d.id AS ruling, r.code AS code, r.article AS article", ids=ruling_ids).data():
        if r["code"] and r["article"]:
            out.setdefault(r["ruling"], set()).add((str(r["article"]), r["code"]))
    return out


def ruling_vote(session, driver, database: str, case_text: str, index: str,
                analysis: Optional[Dict[str, Any]] = None) -> List[tuple]:
    """The case's voted articles [(article, code)], best first, without other areas' codes and
    Cassation boilerplate (filter_refs); [] when the counts are not ready or on any failure."""
    t0 = time.time()
    info: Dict[str, Any] = {"mode": ":".join(str(p) for p in mode())}
    try:
        table = idf_table(driver, database)
        if table is None:
            info["skipped"] = "IDF counts not ready"
            return []
        idf, default_idf = table
        k = int(os.getenv(K_ENV) or DEFAULT_K)
        vector, fulltext = _similar_rulings(session, case_text, index, k)
        sims = ruling_sims(vector, fulltext)
        cites = _citations(session, list(sims))
        cards = _card_refs()
        areas = filter_areas(analysis)
        posture = str(((analysis or {}).get("frame") or {}).get("posture_type") or "")
        votes, dropped = filter_refs({r: cards[r] if r in cards else cites.get(r, ()) for r in sims},
                                     areas, posture, case_text)
        scores = vote_scores(sims, votes, idf, default_idf)
        ranked = sorted((r for r in scores if scores[r] > 0), key=lambda r: (-scores[r], r))
        info.update(k=k, rulings=len(sims), carded=sum(r in cards for r in sims), cited=len(cites),
                    areas=areas, posture=posture,
                    dropped={reason: [f"{a} {c}" for a, c in refs[:20]] for reason, refs in dropped.items()},
                    top=[f"{a} {c} {scores[(a, c)]:.2f}" for a, c in ranked[:15]])
        return ranked
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        logger.warning("vote pool failed, the pool is the fusion's own: %s", exc)
        return []
    finally:
        vlog("case_pool_vote", info, duration_ms=(time.time() - t0) * 1000)
