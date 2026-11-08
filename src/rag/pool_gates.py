"""POOL_GATE_AREA and POOL_GATE_LINK (v50, default false; dedup runs with either): filters on the candidate pool before
the applicability check, measured first on the v46-v48 dumps (scripts/pool_gates_sim.py).

  dedup  one article per (code, article): a second document holding the same article (c.p.a. and "Codice del processo
         amministrativo 2025"), and a Section repeating an earlier one's text (an "_1" ingestion duplicate), go; their
         tags join the row kept. The kept document's other Sections of the article (its fragments) stay.
  area   (POOL_GATE_AREA) a candidate of a main code (c.c., c.p.c., c.p., c.p.p., c.p.a.) the case's areas do not allow
         goes, unless the user's text names the article or two or more channels found it. The areas: every frame area
         (else the analysis' own), each by AREA_CODES -- every area the frame and the analysis can emit but "altro",
         which allows nothing of its own ("lavoro" and "tributario" were dropped by vote_pool.case_areas: n3,
         lavoro/amministrativo, allowed c.p.a. only); plus the codes the user's text names; none (no gate) when the
         areas are unknown or FRAME_SANITY found them uncertain. Only the five codes are ever gated: the labour laws (l.
         300/1970, d.lgs. 81/2015), the CCNL documents and any other law always pass.
  link   (POOL_GATE_LINK) a candidate found by one channel goes when no article found by two or more (or named by the
         user) is in its Capo (an article of a Titolo without Capi: its Titolo -- 2043-2059 c.c.) or linked to it by
         REFERS_TO either way; a REFERS_TO link with a hub at either end (HUB_INBOUND or more articles citing it) does
         not count. Never: the user's articles, a heading match, a cross-reference, a regime candidate, a procedural
         tool, or a row the understanding named (the frame or distill channel: UNDERSTANDING).
Channels: the fusion signals a row carries (current, baseline, distill, frame, ...), the vote, and the channels that
add rows of their own (cross-reference, method list, regime candidate, issue spotting). The Capo and the links are
scripts/article_links.py --write's Unit and REFERS_TO graph; a failed lookup skips the link gate only.

The rows gated out are given back so EDGE_EXPAND does not bring them in again. Logged as case_analysis_pool_gates
(per gate the articles removed, the rows merged by dedup, the codes allowed)."""

import logging
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from . import feature_flags

logger = logging.getLogger(__name__)

AREA_FLAG = "POOL_GATE_AREA"
LINK_FLAG = "POOL_GATE_LINK"
EVENT = "case_analysis_pool_gates"
HUB_INBOUND = 100          # an article this many articles cite is a hub: its REFERS_TO links do not count
GATES = ("dedup", "area", "link")
# the channels that add rows of their own (row["_source"]); search sources are already fusion signals
_CHANNEL_SOURCES = {"broaden_article_crossref": "crossref", "method_candidates": "method_list",
                    "regime_candidate": "regime_candidate", "issue_spotting": "issue_spotting"}
# rows the link gate never removes: by source, and by the understanding's channels
LINK_EXEMPT_SOURCES = {"broaden_article_crossref", "regime_candidate", "procedural_tool"}
UNDERSTANDING = {"frame", "distill"}

_CIVIL = frozenset({"c.c.", "c.p.c."})
# The main codes each area allows: the frame's areas (defensive_generation._FRAME_AREAS) and the analysis' own
# ("civile", "lavoro", "tributario", ...); "altro" none of its own. Labour: c.c. (Libro V) and c.p.c. (artt. 409 ff.);
# the gate works on whole codes, and l. 300/1970, d.lgs. 81/2015 and the CCNL are never gated.
AREA_CODES = {"penale": frozenset({"c.p.", "c.p.p."}), "amministrativo": frozenset({"c.p.a."}),
              "lavoro": _CIVIL, "tributario": _CIVIL, "civile": _CIVIL, "contratti": _CIVIL,
              "responsabilita_civile": _CIVIL, "locazione": _CIVIL, "successioni": _CIVIL, "diritti_reali": _CIVIL,
              "famiglia": _CIVIL, "consumo": _CIVIL}
NO_CODES = {"altro"}

_NAME_MATCH = "(s.name = a.num OR s.name STARTS WITH a.num + '.' OR s.name STARTS WITH a.num + '_')"
_ROWS = "UNWIND $rows AS a MATCH (d:Document {id: a.doc})-[:CONTAINS]->(s:Section) WHERE " + _NAME_MATCH
# the article's deepest unit and the Capo and Titolo above it (or it)
QUERY_CAPO = (_ROWS + " MATCH (u:Unit)-[:HAS_ARTICLE]->(s) "
              "OPTIONAL MATCH (u)-[:CHILD_OF*0..4]->(c:Unit {level: 'CAPO'}) "
              "OPTIONAL MATCH (u)-[:CHILD_OF*0..4]->(t:Unit {level: 'TITOLO'}) "
              "RETURN DISTINCT a.key AS key, u.code AS code, u.level AS deepest, c.path AS capo, t.path AS titolo")
QUERY_INBOUND = (_ROWS + " OPTIONAL MATCH (src:Section)-[:REFERS_TO]->(s) "
                 "RETURN a.key AS key, count(DISTINCT src) AS inbound")
QUERY_LINKS = (_ROWS + " MATCH (s)-[:REFERS_TO]-(t:Section)<-[:CONTAINS]-(td:Document) WHERE td.id IN $docs "
               "RETURN DISTINCT a.key AS key, td.name AS doc, t.name AS section")


def area_enabled() -> bool:
    return feature_flags.enabled(AREA_FLAG)


def link_enabled() -> bool:
    return feature_flags.enabled(LINK_FLAG)


def enabled() -> bool:
    return area_enabled() or link_enabled()


def gate_areas(analysis: Optional[Dict[str, Any]]) -> List[str]:
    """The areas the area gate goes by: the frame's (else the analysis' own) that AREA_CODES or NO_CODES knows; []
    when none, or when FRAME_SANITY_ENABLED found the frame's areas uncertain."""
    analysis = analysis or {}
    frame = analysis.get("frame") or {}
    if feature_flags.enabled("FRAME_SANITY_ENABLED") and frame.get("areas_uncertain"):
        return []
    known = set(AREA_CODES) | NO_CODES
    areas = [a for a in (str(x).strip().lower() for x in frame.get("areas") or []) if a in known]
    own = str(analysis.get("area") or "").strip().lower()
    return areas or ([own] if own in known else [])


def allowed_codes(areas: Iterable[str], named: Iterable[str]) -> Optional[Set[str]]:
    """The main codes the areas allow, plus those the user's text names; None (no gate) when no area allows any."""
    codes: Set[str] = set()
    for area in areas:
        codes |= AREA_CODES.get(area, set())
    return (codes | set(named)) if codes else None


def key(ref: Sequence[str]) -> Tuple[str, str]:
    """(number, code) with a code's document name read as its abbreviation ("Codice del processo amministrativo 2025"
    -> c.p.a.)."""
    from . import edge_expand
    number, code = edge_expand.key(ref)
    return number, edge_expand.code_of_document(code) if code else code


def label(k: Sequence[str]) -> str:
    return f"{k[0]} {k[1]}"


# --- pure: the gates on candidates ---------------------------------------------------------------------------------

def channels(row: Dict[str, Any]) -> Set[str]:
    """The channels that found a pool row."""
    out = {str(s) for s in row.get("_signals") or [] if not str(s).startswith("edge")}
    if row.get("_vote"):
        out.add("vote")
    source = _CHANNEL_SOURCES.get(str(row.get("_source") or ""))
    if source:
        out.add(source)
    return out


def candidate(row: Dict[str, Any], named: Iterable[Sequence[str]]) -> Dict[str, Any]:
    """The gates' view of a row: {"key", "channels", "named", "exempt"}."""
    from .nodes.cypher import CROSSREF_SOURCE
    k = key(row["_article"])
    return {"key": k, "channels": channels(row), "named": k in {key(n) for n in named},
            "exempt": bool(row.get("_heading_match")) or row.get("_source") in LINK_EXEMPT_SOURCES | {CROSSREF_SOURCE}}


def merge(cands: List[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """One candidate per article, its channels and protections the union of its rows'."""
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for c in cands:
        got = out.setdefault(c["key"], {"key": c["key"], "channels": set(), "named": False, "exempt": False})
        got["channels"] |= set(c["channels"])
        got["named"] = got["named"] or c["named"]
        got["exempt"] = got["exempt"] or c["exempt"]
    return out


def strong(c: Dict[str, Any]) -> bool:
    return c["named"] or len(c["channels"]) >= 2


def area_out(cands: Dict[Tuple[str, str], Dict[str, Any]], allowed: Optional[Set[str]]) -> List[Tuple[str, str]]:
    """The articles of a main code the case's areas do not allow, neither named nor found by two channels."""
    from . import vote_pool
    if allowed is None:
        return []
    return [k for k, c in cands.items()
            if k[1] in vote_pool.MAIN_CODES and k[1] not in allowed and not strong(c)]


def link_out(cands: Dict[Tuple[str, str], Dict[str, Any]], lookup, hub_inbound: int = HUB_INBOUND
             ) -> List[Tuple[str, str]]:
    """The single-channel articles (not named, not exempt, not found by the understanding's channels) with no strong
    article in their Capo or linked by REFERS_TO (no hub at either end). lookup: capo(k) -> unit or None, links(k) ->
    keys, inbound(k) -> int."""
    anchors = [k for k, c in cands.items() if strong(c)]
    capos = {lookup.capo(k) for k in anchors} - {None}
    hubs = {k for k in cands if lookup.inbound(k) >= hub_inbound}
    linkable = set(anchors) - hubs
    out = []
    for k, c in cands.items():
        if strong(c) or c["exempt"] or set(c["channels"]) & UNDERSTANDING:
            continue
        if lookup.capo(k) is not None and lookup.capo(k) in capos:
            continue
        if k not in hubs and (set(lookup.links(k)) & linkable) - {k}:
            continue
        out.append(k)
    return out


def plan_gates(cands: Dict[Tuple[str, str], Dict[str, Any]], allowed: Optional[Set[str]], lookup,
               hub_inbound: int = HUB_INBOUND, area: bool = True) -> Dict[str, List[Tuple[str, str]]]:
    """{"area": keys, "link": keys}: the area gate first (area False: none), the link gate on what it leaves (lookup
    None: none)."""
    area = area_out(cands, allowed) if area else []
    rest = {k: c for k, c in cands.items() if k not in set(area)}
    return {"area": area, "link": link_out(rest, lookup, hub_inbound) if lookup is not None else []}


# --- the rows -------------------------------------------------------------------------------------------------------

def _text(row: Dict[str, Any]) -> str:
    s = row.get("s") or {}
    return " ".join(str(s.get("plain_text") or s.get("abstract") or "").split())


_TAGS = ("_multi_signal", "_heading_match", "_vote", "_institution", "_central", "_map", "_regime")


def dedup(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """(rows, merged): per (code, article) the first document's rows; another document's rows, and a row repeating the
    text of a kept one of its article, merged into the article's first row (tags, signals, a cross-reference source)."""
    from .nodes.cypher import CROSSREF_SOURCE
    first: Dict[Tuple[str, str], Dict[str, Any]] = {}
    doc_of: Dict[Tuple[str, str], Any] = {}
    texts: Dict[Tuple[str, str], Set[str]] = {}
    kept, merged = [], []
    for row in rows:
        if not row.get("_article"):
            kept.append(row)
            continue
        k = key(row["_article"])
        doc = (row.get("d") or {}).get("id")
        text = _text(row)
        if k not in first:
            first[k], doc_of[k], texts[k] = row, doc, {text}
            kept.append(row)
            continue
        if doc == doc_of[k] and text not in texts[k]:
            texts[k].add(text)
            kept.append(row)                     # a fragment of the same article: its own text
            continue
        into = first[k]
        for tag in _TAGS:
            if row.get(tag) and not into.get(tag):
                into[tag] = row[tag]
        if row.get("_signals"):
            into["_signals"] = list(dict.fromkeys(list(into.get("_signals") or []) + list(row["_signals"])))
        if row.get("_source") == CROSSREF_SOURCE and into.get("_source") != CROSSREF_SOURCE:
            into["_source"] = CROSSREF_SOURCE
        merged.append(row)
    return kept, merged


def _number(section: str) -> str:
    from . import defensive_generation as dg
    return dg._article_num(section or "")


class GraphLinks:
    """The candidates' Capo, inbound REFERS_TO count and links to the strong candidates' documents, in three queries."""

    def __init__(self, session, rows: List[Dict[str, Any]], cands: Dict[Tuple[str, str], Dict[str, Any]],
                 hub_inbound: int = HUB_INBOUND):
        params, docs_of = [], {}
        for row in rows:
            doc = (row.get("d") or {}).get("id")
            if row.get("_article") and doc:
                k = key(row["_article"])
                docs_of.setdefault(k, set()).add(doc)
                params.append({"key": "|".join(k), "doc": doc, "num": str(row["_article"][0])})
        params = list({(p["key"], p["doc"]): p for p in params}.values())
        self._capo: Dict[Tuple[str, str], tuple] = {}
        for r in session.run(QUERY_CAPO, rows=params).data() if params else []:
            unit = ((r.get("code"), "CAPO", r["capo"]) if r.get("capo") else
                    (r.get("code"), "TITOLO", r["titolo"]) if r.get("titolo") and r.get("deepest") == "TITOLO" else None)
            if unit:
                self._capo.setdefault(tuple(r["key"].split("|", 1)), unit)
        self._inbound: Dict[Tuple[str, str], int] = {}
        for r in session.run(QUERY_INBOUND, rows=params).data() if params else []:
            k = tuple(r["key"].split("|", 1))
            self._inbound[k] = max(self._inbound.get(k, 0), int(r.get("inbound") or 0))
        singles = {k for k, c in cands.items() if not strong(c) and not c["exempt"]
                   and not set(c["channels"]) & UNDERSTANDING and self._inbound.get(k, 0) < hub_inbound}
        docs = sorted({d for k, c in cands.items() if strong(c) for d in docs_of.get(k, ())})
        wanted = [p for p in params if tuple(p["key"].split("|", 1)) in singles]
        self._links: Dict[Tuple[str, str], Set[Tuple[str, str]]] = {}
        from . import edge_expand
        for r in session.run(QUERY_LINKS, rows=wanted, docs=docs).data() if wanted and docs else []:
            number = _number(r.get("section"))
            if number:
                self._links.setdefault(tuple(r["key"].split("|", 1)), set()).add(
                    key((number, edge_expand.code_of_document(r.get("doc")))))

    def capo(self, k):
        return self._capo.get(tuple(k))

    def links(self, k):
        return self._links.get(tuple(k), set())

    def inbound(self, k):
        return self._inbound.get(tuple(k), 0)


class CsvLinks:
    """The same from article_links' units.csv and refers_to.csv (scripts/pool_gates_sim.py): inbound is how many
    distinct articles cite it."""

    def __init__(self, units_csv: str, refers_csv: str):
        from . import edge_expand
        self._csv = edge_expand.CsvLookup(units_csv, refers_csv)

    def capo(self, k):
        from . import edge_expand
        path = self._csv.paths.get(edge_expand.key(k)) or {}
        if "CAPO" in path:
            return self._csv._unit(k, "CAPO")
        return self._csv._unit(k, "TITOLO") if "TITOLO" in path and "SEZIONE" not in path else None

    def links(self, k):
        return {key(r) for r in self._csv.refers_out(k) | self._csv.refers_in(k)}

    def inbound(self, k):
        return self._csv.inbound(k)


def apply(session, rows: List[Dict[str, Any]], analysis: Dict[str, Any], case_text: str, lookup=None,
          area: Optional[bool] = None, link: Optional[bool] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """(rows kept, rows gated out): dedup, then the area gate (POOL_GATE_AREA), then the link gate (POOL_GATE_LINK);
    logged. area / link: None, as the flags say."""
    from . import defensive_generation as dg, vote_pool
    t0 = time.time()
    area = area_enabled() if area is None else area
    link = link_enabled() if link is None else link
    named = dg._refs_in(case_text)
    kept, merged = dedup(rows)
    cands = merge([candidate(r, named) for r in kept if r.get("_article")])
    areas = gate_areas(analysis)
    allowed = allowed_codes(areas, vote_pool.named_codes(case_text))
    lookup_note = "graph" if link else "off"
    if not link:
        lookup = None
    elif lookup is None:
        try:
            lookup = GraphLinks(session, kept, cands)
        except Exception as exc:
            dg._log_exception("pool_gates", exc)
            lookup, lookup_note = None, f"failed: {type(exc).__name__}"
    out = plan_gates(cands, allowed, lookup, area=area)
    gone = set(out["area"]) | set(out["link"])
    final = [r for r in kept if not (r.get("_article") and key(r["_article"]) in gone)]
    removed = merged + [r for r in kept if r.get("_article") and key(r["_article"]) in gone]
    dg.vlog(EVENT, {"rows_before": len(rows), "rows_after": len(final),
                    "articles_before": len(cands), "articles_after": len(cands) - len(gone),
                    "dedup": {"rows": len(merged), "articles": sorted({label(key(r["_article"])) for r in merged})},
                    "area": [label(k) for k in out["area"]], "link": [label(k) for k in out["link"]],
                    "gates": [g for g, on in (("area", area), ("link", link)) if on], "areas": areas,
                    "allowed": sorted(allowed) if allowed is not None else None, "hub_inbound": HUB_INBOUND,
                    "lookup": lookup_note}, duration_ms=(time.time() - t0) * 1000)
    return final, removed
