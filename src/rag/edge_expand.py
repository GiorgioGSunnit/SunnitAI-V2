"""EDGE_EXPAND (v48, default false; link_reach on v47: 23 of 29 missed expected norms one hop from an article of their
case's prompt -- h3's 1158 in 1159's Sezione, h5's 593 c.p.p. cited by 580): once the prompt is fitted, a few one-hop
neighbours of its articles in the graph scripts/article_links.py --write built, then the prompt fitted again.

Anchors: the articles in the fitted prompt, and the rows graded 3, in order of grade (one exempt from the check -- named
by the user, a cross-reference, a heading match -- counts as 3, one past its cap as 0), then of how many retrieval
signals they carry.
Per anchor, in that order, until MAX_ADDED (60) rows have been added:
  edge:sezione    the SEZIONE_N (2) articles of its Sezione nearest by number;
  edge:refers_to  the REFERS_N (2) articles it cites (REFERS_TO out of it) that the fewest articles of the whole corpus
                  cite (the most specific; a hub such as art. 1 c.p. comes last);
  edge:cited_by   the REFERS_N articles citing it (REFERS_TO into it), by the same rule;
never an anchor, never past one hop, never an article the check already rejected. A neighbour already in the rows and
graded (or exempt) is only tagged and does not count; the others -- new ones, read with
defensive_generation._fetch_any, and rows the check never graded (past its cap) -- count against MAX_ADDED, are tagged
and graded by the applicability check like any row. The rank is the order of choice.
Tagged: its channels, anchors and rank, and the retrieval signal "edge", plus "edge:2+ anchors" when two or more anchors
reach it (a second signal); PLAN_RULE_INCLUDE counts both, and two signals make the row multi-signal (kept whole by the
prompt's trimming like any). Logged as case_analysis_edge_expand (anchors, tagged, graded with their rank, added,
where the cap stopped, the grading calls) and, at the end, case_analysis_edge_reach (how many reached the prompt, the
plan, the answer).

The policy (P3) was chosen on the v46 and v47 dumps (scripts/edge_policy.py) for the most misses reached per row added,
among P1 (no edge:cited_by, cap 40), P2 (no edge:cited_by, cap 60), P3 (as above) and P4 (P1 with anchors only graded
>= 2 or carrying a signal): 10 of the 57 misses reached for 59 rows a case, against 8 (P2), 6 (P1), 5 (P4); the others
stay as parameters (REFERS_SOURCES, MAX_ADDED, ANCHOR_MIN_GRADE)."""

import logging
import re
import threading
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from . import feature_flags

logger = logging.getLogger(__name__)

FLAG = "EDGE_EXPAND"
SEZIONE_N = 2              # same-Sezione neighbours per anchor, nearest by number
REFERS_N = 2               # REFERS_TO targets per anchor, fewest inbound links first
REFERS_SOURCES = True      # also the REFERS_N articles citing the anchor, fewest inbound links first (P3)
ANCHOR_MIN_GRADE = None    # P4: only anchors graded >= this or carrying a retrieval signal
MAX_ADDED = 60             # rows per case sent to the check (new, or never graded)
SIGNAL = "edge"
SIGNAL_ANCHORS = "edge:2+ anchors"     # reached by two or more anchors: a second signal
SOURCE = "edge_expand"
CHANNELS = ("edge:sezione", "edge:refers_to", "edge:cited_by")

_NAME_MATCH = "(s.name = a.num OR s.name STARTS WITH a.num + '.' OR s.name STARTS WITH a.num + '_')"
_ANCHOR = "UNWIND $anchors AS a MATCH (d:Document {id: a.doc})-[:CONTAINS]->(s:Section) WHERE " + _NAME_MATCH
# a neighbour with how many Sections of the corpus cite it (its specificity)
QUERY_REFERS_OUT = (_ANCHOR + " MATCH (s)-[:REFERS_TO]->(t:Section)<-[:CONTAINS]-(td:Document) "
                    "OPTIONAL MATCH (src:Section)-[:REFERS_TO]->(t) "
                    "RETURN a.key AS anchor, td.name AS doc, t.name AS section, count(DISTINCT src) AS inbound")
QUERY_REFERS_IN = (_ANCHOR + " MATCH (s)<-[:REFERS_TO]-(t:Section)<-[:CONTAINS]-(td:Document) "
                   "OPTIONAL MATCH (src:Section)-[:REFERS_TO]->(t) "
                   "RETURN a.key AS anchor, td.name AS doc, t.name AS section, count(DISTINCT src) AS inbound")
QUERY_SEZIONE = (_ANCHOR + " MATCH (u:Unit {level: 'SEZIONE'})-[:HAS_ARTICLE]->(s) "
                 "MATCH (u)-[:HAS_ARTICLE]->(t:Section)<-[:CONTAINS]-(td:Document) "
                 "RETURN DISTINCT a.key AS anchor, td.name AS doc, t.name AS section, 0 AS inbound")

_local = threading.local()


def enabled() -> bool:
    return feature_flags.enabled(FLAG)


def key(ref: Sequence[str]) -> Tuple[str, str]:
    return (re.sub(r"[\s-]+", "-", str(ref[0]).strip().lower()), str(ref[1]).strip())


def _number(article: str) -> float:
    """An article number for distances: "1218" 1218, "1159-bis" 1159.2."""
    m = re.match(r"(\d+)(?:-(\w+))?", str(article))
    if not m:
        return float("inf")
    order = ("bis", "ter", "quater", "quinquies", "sexies", "septies", "octies", "novies", "decies")
    return int(m.group(1)) + ((order.index(m.group(2)) + 1) / 10 if m.group(2) in order else 0)


def code_of_document(name: str) -> str:
    """The code a row's _article carries for a document: the six codes' abbreviation, else its name."""
    from . import defensive_generation as dg
    return next((abbr for abbr, prefix in dg._CODE_PREFIXES.items() if (name or "").startswith(prefix)), name or "")


# --- the anchors and their neighbours -----------------------------------------------------------------------------

def exempt(row: Dict[str, Any]) -> bool:
    """A row the applicability check does not grade: a cross-reference, a heading match (the user's own articles are in
    the prompt, anchors)."""
    from .nodes.cypher import CROSSREF_SOURCE
    return bool(row.get("_heading_match")) or row.get("_source") == CROSSREF_SOURCE


def _anchor_grade(row: Dict[str, Any]) -> int:
    """The row's grade; an exempt one as 3 (in the prompt without a grade: the user named it), one past the check's cap
    as 0."""
    if isinstance(row.get("_grade"), int):
        return row["_grade"]
    return 3 if exempt(row) or row.get("_applicability") != "non valutato" else 0


def anchors(rows: Iterable[Dict[str, Any]], prompt: str = "", min_grade: Optional[int] = None) -> List[Dict[str, Any]]:
    """The rows the expansion starts from, once per article, with why -- in the prompt as fitted ("[art. N code]" in its
    article block), or graded 3 -- in order of grade, then of signal count. min_grade (P4): only those graded at least
    that or carrying a signal."""
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in rows:
        ref = row.get("_article")
        if not ref or key(ref) in out:
            continue
        why = ("in the prompt" if f"[art. {ref[0]} {ref[1]}]" in (prompt or "") else
               "grade 3" if row.get("_grade") == 3 else "")
        if why and (min_grade is None or _anchor_grade(row) >= min_grade or row.get("_signals")):
            out[key(ref)] = {"ref": tuple(ref), "row": row, "why": why}
    return sorted(out.values(), key=lambda a: (-_anchor_grade(a["row"]), -len(set(a["row"].get("_signals") or []))))


def neighbours(anchor_refs: Sequence[Tuple[str, str]], lookup, status: Optional[Callable] = None,
               cap: Optional[int] = None) -> Tuple[List[Dict[str, Any]], Optional[int]]:
    """([{"ref", "key", "channels", "anchors", "action", "rank"}] in the order chosen, the index of the anchor the cap
    stopped at or None). Per anchor in order: the SEZIONE_N same-Sezione articles nearest by number, the REFERS_N
    articles it cites with the fewest inbound links in the corpus (and, REFERS_SOURCES, the REFERS_N citing it), each
    one status(key) does not "skip"; one "grade" (to be graded) counts against cap (MAX_ADDED), the first past it stops
    the expansion; "tag" (in the rows, graded) does not count. One reached again gets the anchor added."""
    status = status or (lambda k: "grade")
    cap = MAX_ADDED if cap is None else cap
    anchor_keys = {key(a) for a in anchor_refs}
    found: Dict[Tuple[str, str], Dict[str, Any]] = {}
    order: List[Dict[str, Any]] = []
    added = 0

    def eligible(refs):
        return {key(r) for r in refs} - anchor_keys - {k for k in {key(r) for r in refs} if status(k) == "skip"}

    def nearest(ka):
        return lambda k: (0 if k[1] == ka[1] else 1, abs(_number(k[0]) - _number(ka[0])) if k[1] == ka[1] else 0,
                          k[1], _number(k[0]))
    for i, anchor in enumerate(anchor_refs):
        ka = key(anchor)
        picks = [("edge:sezione", sorted((k for k in eligible(lookup.sezione(anchor)) if k[1] == ka[1]),
                                         key=nearest(ka))[:SEZIONE_N]),
                 ("edge:refers_to", sorted(eligible(lookup.refers_out(anchor)),
                                           key=lambda k: (lookup.inbound(k), nearest(ka)(k)))[:REFERS_N])]
        if REFERS_SOURCES:
            picks.append(("edge:cited_by", sorted(eligible(lookup.refers_in(anchor)),
                                                  key=lambda k: (lookup.inbound(k), nearest(ka)(k)))[:REFERS_N]))
        label = f"{anchor[0]} {anchor[1]}"
        for channel, ks in picks:
            for k in ks:
                entry = found.get(k)
                if entry is None:
                    action = "tag" if status(k) == "tag" else "grade"
                    if action == "grade":
                        if added >= cap:
                            return order, i
                        added += 1
                    entry = found[k] = {"ref": k, "key": k, "channels": [], "anchors": [], "action": action,
                                        "rank": len(order) + 1}
                    order.append(entry)
                if channel not in entry["channels"]:
                    entry["channels"].append(channel)
                if label not in entry["anchors"]:
                    entry["anchors"].append(label)
    return order, None


def _article_rows(rows, per):
    """{anchor key: {(number, code): inbound}} from query rows, an article's inbound its Sections' highest."""
    from . import defensive_generation as dg
    for row in rows:
        number = dg._article_num(row.get("section") or "")
        if number:
            ref = (number, code_of_document(row.get("doc")))
            got = per.setdefault(row["anchor"], {})
            got[ref] = max(got.get(ref, 0), int(row.get("inbound") or 0))


class GraphLookup:
    """The neighbours of a set of anchors read in three queries (REFERS_TO out, in, the Sezione) on the graph, with
    each REFERS_TO neighbour's inbound count."""

    def __init__(self, session, anchor_rows: Sequence[Dict[str, Any]]):       # anchors() entries
        params = []
        for a in anchor_rows:
            doc = ((a["row"].get("d") or {}).get("id"))
            if doc:
                params.append({"key": f"{a['ref'][0]}|{a['ref'][1]}", "doc": doc, "num": str(a["ref"][0])})
        self._by: Dict[str, Dict[str, Dict[Tuple[str, str], int]]] = {"out": {}, "in": {}, "sezione": {}}
        queries = [("out", QUERY_REFERS_OUT), ("sezione", QUERY_SEZIONE)] + ([("in", QUERY_REFERS_IN)]
                                                                              if REFERS_SOURCES else [])
        for kind, query in queries:
            _article_rows(session.run(query, anchors=params).data() if params else [], self._by[kind])
        self._inbound: Dict[Tuple[str, str], int] = {}
        for kind in ("out", "in"):
            for got in self._by[kind].values():
                for ref, n in got.items():
                    self._inbound[key(ref)] = max(self._inbound.get(key(ref), 0), n)

    def _get(self, kind, ref):
        return set(self._by[kind].get(f"{ref[0]}|{ref[1]}", {}))

    def refers_out(self, ref):
        return self._get("out", ref)

    def refers_in(self, ref):
        return self._get("in", ref)

    def sezione(self, ref):
        return self._get("sezione", ref)

    def inbound(self, k):
        return self._inbound.get(key(k), 0)


class CsvLookup:
    """The same neighbours from article_links' units.csv and refers_to.csv (offline checks on the dumps); an article's
    inbound count is how many distinct articles cite it."""

    LEVELS = ("LIBRO", "PARTE", "TITOLO", "CAPO", "SEZIONE")

    def __init__(self, units_csv: str, refers_csv: str):
        import csv
        self.paths: Dict[Tuple[str, str], Dict[str, str]] = {}
        with open(units_csv, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                self.paths.setdefault(key((row["article"], row["code"])), {})[row["level"]] = row["unit name"]
        self.out: Dict[Tuple[str, str], Set[Tuple[str, str]]] = {}
        self.into: Dict[Tuple[str, str], Set[Tuple[str, str]]] = {}
        with open(refers_csv, encoding="utf-8") as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for row in reader:
                a, b = _label_ref(row[0]), _label_ref(row[1])
                if a and b and a != b:
                    self.out.setdefault(a, set()).add(b)
                    self.into.setdefault(b, set()).add(a)
        self._units: Dict[tuple, Set[Tuple[str, str]]] = {}
        for k in self.paths:
            unit = self._unit(k, "SEZIONE")
            if unit:
                self._units.setdefault(unit, set()).add(k)

    def _unit(self, ref, level):
        path = self.paths.get(key(ref)) or {}
        if level not in path:
            return None
        return (key(ref)[1],) + tuple(path.get(l, "") for l in self.LEVELS[:self.LEVELS.index(level) + 1])

    def refers_out(self, ref):
        return self.out.get(key(ref), set())

    def refers_in(self, ref):
        return self.into.get(key(ref), set())

    def sezione(self, ref):
        return set(self._units.get(self._unit(ref, "SEZIONE"), set()))

    def inbound(self, k):
        return len(self.into.get(key(k), set()))


def _label_ref(label: str) -> Optional[Tuple[str, str]]:
    m = re.match(r"^(\d+(?:[\s-]*(?:bis|ter|quater|quinquies|sexies|septies|octies|novies|decies)\b)?)\s+(.+)$",
                 (label or "").strip(), re.IGNORECASE)
    return key((m.group(1), m.group(2))) if m else None


# --- the expansion ----------------------------------------------------------------------------------------------------

def tag(row: Dict[str, Any], entry: Dict[str, Any]) -> None:
    """A neighbour's channels, anchors and rank, and its signals, in place: "edge", and "edge:2+ anchors" when two or
    more anchors reach it (a second signal); with another signal it is multi-signal."""
    row["_edge_channels"] = list(dict.fromkeys(list(row.get("_edge_channels") or []) + entry["channels"]))
    row["_edge_anchors"] = list(dict.fromkeys(list(row.get("_edge_anchors") or []) + entry["anchors"]))
    row["_edge_rank"] = entry.get("rank")
    added = [SIGNAL] + ([SIGNAL_ANCHORS] if len(row["_edge_anchors"]) >= 2 else [])
    row["_signals"] = list(dict.fromkeys(list(row.get("_signals") or []) + added))
    if len(set(row["_signals"])) >= 2:
        row["_multi_signal"] = True


def plan_expansion(checked: List[Dict[str, Any]], before_check: Iterable[Dict[str, Any]], lookup, prompt: str = ""
                   ) -> Dict[str, Any]:
    """{"anchors", "ranked" (every neighbour chosen, in order), "tag": {key: entry} (in the rows, graded: tagged only),
    "grade": [entry] (new ones and rows the check never graded, at most MAX_ADDED), "stopped_at": the anchor the cap
    stopped at (None: every anchor expanded), "rejected": [labels] (left out: the check took them out)} -- no reading,
    no grading here."""
    found = anchors(checked, prompt, ANCHOR_MIN_GRADE)
    rows = {key(r["_article"]): r for r in checked if r.get("_article")}
    seen = {key(r["_article"]) for r in before_check if r.get("_article")}
    rejected: List[str] = []

    def status(k):
        row = rows.get(k)
        if row is not None and (row.get("_grade") is not None or exempt(row)):
            return "tag"
        if row is not None or k not in seen:
            return "grade"
        label = f"{k[0]} {k[1]}"
        if label not in rejected:
            rejected.append(label)
        return "skip"
    ranked, stopped = neighbours([a["ref"] for a in found], lookup, status)
    return {"anchors": found, "ranked": ranked, "tag": {e["key"]: e for e in ranked if e["action"] == "tag"},
            "grade": [e for e in ranked if e["action"] == "grade"],
            "stopped_at": None if stopped is None else f"{found[stopped]['ref'][0]} {found[stopped]['ref'][1]}",
            "anchors_expanded": len(found) if stopped is None else stopped, "rejected": rejected}


def grading_calls(n: int) -> int:
    from . import defensive_generation as dg
    return -(-n // dg._APPLICABILITY_BATCH) if n else 0


def expand(session, documents: List[Dict[str, Any]], checked: List[Dict[str, Any]], before_check: List[Dict[str, Any]],
           prompt: str, grade, store=None, lookup=None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """(the rows with the anchors' neighbours, the rows new to them): the ones there and graded tagged; up to MAX_ADDED
    in the order chosen -- new ones read, and ones the check never graded -- tagged and graded (grade: the
    applicability check on a list of rows), the ones it keeps in. Logged with each neighbour's rank and the grading
    calls it took."""
    from . import defensive_generation as dg
    t0 = time.time()
    lookup = lookup or GraphLookup(session, anchors(checked, prompt, ANCHOR_MIN_GRADE))
    plan = plan_expansion(checked, before_check, lookup, prompt)
    by_key = {key(r["_article"]): r for r in checked if r.get("_article")}
    for k, entry in plan["tag"].items():
        tag(by_key[k], entry)
    again = [by_key[e["key"]] for e in plan["grade"] if e["key"] in by_key]
    wanted = {e["key"]: e for e in plan["grade"] if e["key"] not in by_key}
    new_rows = dg._fetch_any(session, documents, [e["ref"] for e in wanted.values()], store) if wanted else []
    entries = {e["key"]: e for e in plan["grade"]}
    for row in again + new_rows:
        entry = entries.get(key(row["_article"])) if row.get("_article") else None
        if entry:
            tag(row, entry)
    for row in new_rows:
        row["_source"] = SOURCE
    to_grade = again + new_rows
    graded = grade(to_grade) if to_grade else []
    kept_again = [r for r in graded if any(r is a for a in again)]
    kept_new = [r for r in graded if not any(r is a for a in again)]
    rows = [r for r in checked if not any(r is a for a in again)] + kept_again + kept_new

    def item(e, row=None):
        out = {"article": f"{e['key'][0]} {e['key'][1]}", "rank": e["rank"], "channels": e["channels"],
               "anchors": e["anchors"]}
        if row is not None:
            out["grade"] = row.get("_grade")
        return out
    graded_by = {key(r["_article"]): r for r in graded if r.get("_article")}
    dg.vlog("case_analysis_edge_expand", {
        "anchors": [{"article": f"{a['ref'][0]} {a['ref'][1]}", "why": a["why"]} for a in plan["anchors"]],
        "neighbours": len(plan["ranked"]),
        "tagged": [item(e) for e in plan["tag"].values()],
        "graded": [item(e, graded_by.get(e["key"])) for e in plan["grade"]],
        "added": [item(entries[key(r["_article"])], r) for r in kept_new],
        "cap": MAX_ADDED, "stopped_at": plan["stopped_at"], "anchors_expanded": plan["anchors_expanded"],
        "read": len(new_rows), "sent_to_grading": len(to_grade), "grading_calls": grading_calls(len(to_grade)),
        "graded_out": len(to_grade) - len(graded), "rejected_before": plan["rejected"]},
        duration_ms=(time.time() - t0) * 1000)
    return rows, kept_new


def note_plan(plan: Optional[Dict[str, Any]]) -> None:
    """The plan's articles, for the reach log (plan_then_write.make_plan)."""
    _local.plan = {key(r) for q in (plan or {}).get("questioni") or [] for r in q.get("norme") or []}


def log_reach(rows: List[Dict[str, Any]], prompt: str, draft: str) -> Optional[Dict[str, Any]]:
    """How many edge rows (added or tagged) reached the prompt, the plan and the answer; None without any."""
    from . import citation_ledger, defensive_generation as dg
    edged = [r for r in rows if r.get("_edge_channels") and r.get("_article")]
    if not edged:
        return None
    cited = {citation_ledger._key(a) for a in citation_ledger.cited(draft or "")}
    plan = getattr(_local, "plan", set()) or set()
    out = {"edge_rows": len({key(r["_article"]) for r in edged}),
           "added": len({key(r["_article"]) for r in edged if r.get("_source") == SOURCE}), "in_prompt": [], "in_plan": [],
           "in_answer": []}
    for r in edged:
        label = f"{r['_article'][0]} {r['_article'][1]}"
        if f"[art. {label}]" in (prompt or "") and label not in out["in_prompt"]:
            out["in_prompt"].append(label)
        if key(r["_article"]) in plan and label not in out["in_plan"]:
            out["in_plan"].append(label)
        if citation_ledger._key(label) in cited and label not in out["in_answer"]:
            out["in_answer"].append(label)
    dg.vlog("case_analysis_edge_reach", out)
    return out
