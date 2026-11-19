"""INTERACTIVE_CASE_FIRST (B): the topics of round 1 from what retrieval found and the grader kept, not from our rules.

Every pool row graded 2 or more (defensive_generation's graded check, "_grade") becomes a topic or joins one: rows are
grouped by their Sezione (fallback their Capo, else the row alone), and a group is named by the Sezione's title, or --
when that title is a category any part of the codes uses (topic_step's generic-word rules) -- by the leading rubrica.
Then:
  - the rules' and tables' topics (topic_step.build) attach their questions, their «da verificare» lines and their
    anchors to the group holding one of their articles; they decide no topic;
  - each hypothesis of the first read (case_read) attaches to the group whose rows anchor one of its institutions (its
    content words -- two, or its only one -- among the group's title, unit and rubriche); one no group anchors is "da
    verificare" (unanchored), shown
    as such in round 1, never asserted in an answer;
  - the rows graded 1 are the "altri profili", grouped by Capo;
  - order: the group holding the main-line hypothesis first, then by the number of graded rows, then the best grade
    (the angle the user states comes first afterwards: interactive_flow.apply_stated); the first is what "procedi"
    takes.
"""

import os
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from . import topic_step

TOPICS_MAX = topic_step.TOPICS_MAX
TOPIC_ARTICLES = 8
STRONG = 2                      # a grade from which a row makes or joins a topic
_UNIT_PREFIX_RE = re.compile(r"^\s*(?:sezione|capo|titolo|libro|parte|paragrafo|§)\s+[ivxlcdm\d]+(?:[-\s]*(?:bis|ter|quater))?"
                             r"\s*[–—:.-]*\s*", re.IGNORECASE)

UNITS_QUERY = ("UNWIND $items AS it "
               "MATCH (d:Document {id: it.doc})-[:CONTAINS]->(s:Section) "
               "WHERE s.name = it.num OR s.name STARTS WITH it.num + '.' "
               "MATCH (u:Unit)-[:HAS_ARTICLE]->(s) WHERE u.level IN ['SEZIONE', 'CAPO'] "
               "RETURN it.doc AS doc, it.num AS num, u.level AS level, u.name AS name")


def _ref(row: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    ref = row.get("_article")
    return (str(ref[0]), topic_step._code(ref[1])) if ref else None


def _label(ref: Sequence[str]) -> str:
    return f"{ref[0]} {ref[1]}"


def _rubrica(row: Dict[str, Any]) -> str:
    from .rule_quote import clean_rubrica
    s = row.get("s") or {}
    return clean_rubrica(s.get("titolo_articolo") or s.get("title") or "").strip(" .")


def unit_key(row: Dict[str, Any]) -> Tuple[str, str]:
    """(doc id, article number) as the units query reads them."""
    from .article_text import article_number
    s, d = row.get("s") or {}, row.get("d") or {}
    number = article_number(s.get("name") or "") or (str(row["_article"][0]) if row.get("_article") else "")
    return str(d.get("id") or ""), number


def units(session, rows: Iterable[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, str]]:
    """{(doc id, number): {"SEZIONE": name, "CAPO": name}} for the rows, one query."""
    items = sorted({unit_key(r) for r in rows if all(unit_key(r))})
    if not items:
        return {}
    out: Dict[Tuple[str, str], Dict[str, str]] = {}
    for rec in session.run(UNITS_QUERY, items=[{"doc": d, "num": n} for d, n in items]).data():
        out.setdefault((rec["doc"], rec["num"]), {}).setdefault(rec["level"], str(rec["name"] or ""))
    return out


def unit_title(name: str) -> str:
    """«SEZIONE II – Della legittima difesa» -> «Della legittima difesa» -> «Legittima difesa»."""
    from .rule_quote import clean_rubrica
    text = _UNIT_PREFIX_RE.sub("", clean_rubrica(" ".join(str(name or "").split())))     # v60e: no «[1025]»
    if not re.search(r"[^\W\d_]{3}", text):           # a marker, a number: no title
        return ""
    text = re.sub(r"^(?:(?:dei|degli|delle|della|dello|del|di)\s+|dell['’]\s*)", "", text, flags=re.IGNORECASE)
    return topic_step._title(text.lower() if text.isupper() else text)


def generic(name: str) -> bool:
    """A name any part of the codes uses: a GENERIC_RUBRICHE entry, or no content word outside GENERIC_STEMS."""
    if not str(name or "").strip():
        return True
    if tuple(topic_step.tokens(name)) in topic_step.GENERIC_RUBRICHE:
        return True
    return not (topic_step._content_stems(name) - topic_step.GENERIC_STEMS)


def _order_key(row: Dict[str, Any]) -> tuple:
    rank = row.get("_fused_rank")
    return (-int(row.get("_grade") or 0), rank if isinstance(rank, (int, float)) else 10 ** 6)


def _best_rows(rows: Iterable[Dict[str, Any]], keep) -> List[Dict[str, Any]]:
    """One row per article (its best grade), those keep() accepts, best first."""
    from .article_text import row_abrogated, row_misaligned
    best: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in rows:
        ref = _ref(row)
        if ref is None or not isinstance(row.get("_grade"), int) or not keep(row["_grade"]):
            continue
        if row_abrogated(row) or row_misaligned(row):   # v60c/v60e: abrogated, or its rubrica not its text: never a topic's
            continue
        if ref not in best or _order_key(row) < _order_key(best[ref]):
            best[ref] = row
    return sorted(best.values(), key=_order_key)


def _group(rows: List[Dict[str, Any]], unit_of: Dict[Tuple[str, str], Dict[str, str]], levels: Sequence[str]
           ) -> List[Dict[str, Any]]:
    groups: Dict[tuple, Dict[str, Any]] = {}
    for row in rows:
        names = unit_of.get(unit_key(row)) or {}
        level = next((lv for lv in levels if names.get(lv)), "")
        key = (unit_key(row)[0], level, names.get(level, "")) if level else ("", "row", _label(_ref(row)))
        groups.setdefault(key, {"level": level.lower() or "rubrica", "unit": names.get(level, "") if level else "",
                                "rows": []})["rows"].append(row)
    return list(groups.values())


def _name(group: Dict[str, Any]) -> str:
    """The Sezione's (Capo's) title, else -- generic -- the leading rubrica that is not, else the unit's title."""
    title = unit_title(group["unit"]) if group["unit"] else ""
    if title and not generic(title):
        return title
    for row in group["rows"]:
        rubrica = _rubrica(row)
        if rubrica and not generic(rubrica) and re.search(r"[^\W\d_]{3}", rubrica):
            return topic_step._title(rubrica)
    fallback = topic_step._title(_rubrica(group["rows"][0]))
    return title or (fallback if re.search(r"[^\W\d_]{3}", fallback) else "") or f"Art. {_label(_ref(group['rows'][0]))}"


# v60c: the primary codes -- their groups graded 2+ are never capped; a special code's group is (TOPICS_MAX in all) and
# comes after them unless one of its rows is graded 3
PRIMARY_CODES = ("c.c.", "c.p.", "c.p.c.", "c.p.p.", "c.p.a.")


def _topic(group: Dict[str, Any]) -> Dict[str, Any]:
    """A topic of graded rows: its writable articles are its graded rows only (articles[:planned], matched); what
    attaches later -- a rule's or a table's anchors -- is listed after them, never planned or written."""
    rows = group["rows"]
    labels = [_label(_ref(r)) for r in rows]
    code = _ref(rows[0])[1]
    return {"title": _name(group), "anchor": labels[0], "how": "graded", "rubrica": _rubrica(rows[0]),
            "articles": labels, "planned": min(len(labels), TOPIC_ARTICLES), "facts": [], "merged": [],
            "source": group["level"], "unit": group["unit"], "matched": list(labels), "graded": len(rows),
            "grades": {_label(_ref(r)): r["_grade"] for r in rows}, "best_grade": max(r["_grade"] for r in rows),
            "hypotheses": [], "questions": [], "code": code, "primary": code in PRIMARY_CODES, "offences": [],
            "institutions": sorted({str(r["_first_read"]) for r in rows if r.get("_first_read")})}


def _topic_words(topic: Dict[str, Any], rubriche: Dict[str, str]) -> Set[str]:
    words = topic_step._content_stems(topic["title"]) | topic_step._content_stems(topic.get("unit"))
    for label in topic["articles"]:
        words |= topic_step._content_stems(rubriche.get(label, ""))
    return words - topic_step.GENERIC_STEMS


def attach_rules(topics: List[Dict[str, Any]], rule_topics: Sequence[Dict[str, Any]]) -> List[str]:
    """Each rule's or table's topic (topic_step.build) to the group holding its anchor or one of its own articles: its
    questions, its «da verificare» lines, its anchors (matched, after the group's). The titles attached nowhere."""
    lost = []
    for rt in rule_topics or []:
        own = [rt.get("anchor") or ""] + list(rt.get("matched") or [])
        own = [a for a in dict.fromkeys(own) if a]
        held = next((t for t in topics if set(own) & set(t["articles"])), None)
        if held is None:
            lost.append(str(rt.get("title") or ""))
            continue
        held["questions"] = list(dict.fromkeys(list(held.get("questions") or []) + list(rt.get("questions") or [])))
        held["facts"] = list(dict.fromkeys(list(held["facts"]) + [f for f in rt.get("facts") or []]))
        # v60c: a rule's or table's anchor not graded is listed, never planned nor written (the graded rows are)
        held["articles"] = list(held["articles"]) + [a for a in own if a not in held["articles"]]
        held["merged_sources"] = list(dict.fromkeys(list(held.get("merged_sources") or []) + [rt.get("source")]))
        if rt.get("title") and rt["title"] != held["title"]:
            held["merged"] = list(dict.fromkeys(list(held["merged"]) + [rt["title"]]))
    return lost


def attach_hypotheses(topics: List[Dict[str, Any]], hypotheses: Sequence[Dict[str, Any]],
                      rubriche: Dict[str, str], procedural_names: Iterable[str] = ()) -> List[Dict[str, Any]]:
    """Each hypothesis to the topic one of its institutions anchors best (module docstring); the unanchored ones. v60n:
    only where it may (may_anchor)."""
    unanchored = []
    words = [(_topic_words(t, rubriche), t) for t in topics]
    procedural_names = list(procedural_names or ())
    for h in hypotheses or []:
        named = [inst for inst in h.get("institutions") or []
                 if topic_step._content_stems(inst) - topic_step.GENERIC_STEMS]

        def score(w, t):        # an institution anchors when all its content words, two at most needed, are the topic's
            stems = [topic_step._content_stems(inst) - topic_step.GENERIC_STEMS for inst in named
                     if may_anchor(inst, t, procedural_names)]
            return max((len(x & w) for x in stems if len(x & w) >= min(2, len(x))), default=0)
        scored = [(score(w, t), -i, t) for i, (w, t) in enumerate(words)]
        best = max(scored, key=lambda x: x[:2], default=None)
        if best and best[0] >= 1:
            best[2]["hypotheses"].append(h)
        else:
            unanchored.append(h)
    return unanchored


def weight(topic: Dict[str, Any]) -> int:
    """0: a primary code's group, or a special code's with a row graded 3; 1: another special code's."""
    return 0 if topic.get("primary", True) or int(topic.get("best_grade") or 0) >= 3 else 1


def order(topics: List[Dict[str, Any]], angle: Sequence[int] = ()) -> None:
    """In place (v60c): code weight (weight()), then the number of graded rows, then the main-line hypothesis, then the
    user's angle (angle: the positions, 1-based, of the topics it matches), then the best grade. v60d: then, for the
    same institution, the area's code first (area_first)."""
    position = {id(t): i for i, t in enumerate(topics)}
    hits = {id(topics[i - 1]) for i in angle if 0 < i <= len(topics)}
    topics.sort(key=lambda t: (weight(t), -int(t.get("graded") or 0),
                               not any(h.get("line") == "principale" for h in t.get("hypotheses") or []),
                               id(t) not in hits, -int(t.get("best_grade") or 0), position[id(t)]))
    area_first(topics)


def mark_area(topics: Iterable[Dict[str, Any]], codes: Sequence[str]) -> None:
    """v60d: each topic's in_area -- its code among the codes of the case's area (interactive_flow.area_codes)."""
    for t in topics:
        t["in_area"] = (t.get("code") or "") in set(codes)


def institutions_of(topic: Dict[str, Any]) -> Set[str]:
    """The first read's institutions a topic answers: its rows' first-read searches, and the institution anchoring a
    hypothesis in it."""
    out = {str(i).lower() for i in topic.get("institutions") or []}
    out |= {str(h.get("anchor_institution")).lower() for h in topic.get("hypotheses") or [] if h.get("anchor_institution")}
    return out


def area_first(topics: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """In place (v60d): a topic of the case's area's code (in_area) goes just before an earlier topic of another code
    that answers the same institution (institutions_of); topics without in_area are left. The moves (title, before)."""
    moves = []
    i = 0
    while i < len(topics):
        t = topics[i]
        if t.get("in_area") is True and institutions_of(t):
            mine = institutions_of(t)
            j = next((k for k in range(i) if topics[k].get("in_area") is False and mine & institutions_of(topics[k])),
                     None)
            if j is not None:
                topics.insert(j, topics.pop(i))
                moves.append((t.get("title"), topics[j + 1].get("title")))
        i += 1
    return moves


def capped(topics: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """(kept, spilled), in order: every primary code's group; the special codes' while TOPICS_MAX is not reached."""
    primary = sum(1 for t in topics if t.get("primary", True))
    room = max(0, TOPICS_MAX - primary)
    kept, spilled = [], []
    for t in topics:
        if t.get("primary", True):
            kept.append(t)
        elif room > 0:
            kept.append(t)
            room -= 1
        else:
            spilled.append(t)
    return kept, spilled


def rubriche_of(rows: Iterable[Dict[str, Any]]) -> Dict[str, str]:
    return {_label(_ref(r)): _rubrica(r) for r in rows if _ref(r)}


def build(rows: Iterable[Dict[str, Any]], unit_of: Dict[Tuple[str, str], Dict[str, str]],
          hypotheses: Sequence[Dict[str, Any]] = (), rule_topics: Sequence[Dict[str, Any]] = ()) -> Dict[str, Any]:
    """{"topics", "others", "unanchored", "lost_rules"} (module docstring)."""
    rows = list(rows)
    strong = _best_rows(rows, lambda g: g >= STRONG)
    weak = _best_rows(rows, lambda g: g == 1)
    rubriche = {_label(_ref(r)): _rubrica(r) for r in strong}
    topics = [_topic(g) for g in _group(strong, unit_of, ("SEZIONE", "CAPO"))]
    lost = attach_rules(topics, rule_topics)
    unanchored = attach_hypotheses(topics, hypotheses, rubriche)

    order(topics)
    kept, spilled = capped(topics)
    others = []
    for g in _group(weak, unit_of, ("CAPO",)):
        t = _topic(g)
        others.append({**t, "capo": g["unit"], "why": "grado 1"})
    for t in spilled:
        others.append({**t, "capo": t.get("unit") if t["source"] == "capo" else "", "why": "oltre il numero di argomenti"})
    return {"topics": kept, "others": others, "unanchored": unanchored, "lost_rules": lost}


def outside_profiles(rows: Iterable[Dict[str, Any]], unit_of: Dict[Tuple[str, str], Dict[str, str]], why: str
                     ) -> List[Dict[str, Any]]:
    """v60q: the articles the code-area gate keeps out (interactive_flow.area_gate) as «altri profili» -- one per Capo
    (else per Sezione, else per article), never graded nor planned; why their reason."""
    best: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in rows:
        ref = _ref(row)
        if ref and ref not in best:
            best[ref] = {**row, "_grade": 0}
    out = []
    for g in _group(list(best.values()), unit_of, ("CAPO", "SEZIONE")):
        t = _topic(g)
        out.append({**t, "how": "area", "planned": 0, "matched": [], "graded": 0, "grades": {}, "best_grade": None,
                    "capo": g["unit"] if g["level"] == "capo" else "", "why": why})
    return out


# --- v60q: the Capo-scope gate of Libro IV Titolo III c.c. («Dei singoli contratti») ------------------------------------
# Each Capo there is one contract type: its article is a candidate, a topic's article or an anchor only when the case's
# contract type is that Capo's -- the extraction's contracts (fact_signals facts["contracts"]: type, parties, clauses), or
# the frame's area «locazione» -- or when the user's text names the article. Titolo II (the general part, artt.
# 1321-1469) and every article outside Titolo III: always. The gate drops only when the extraction yields at least one
# contract type (a contract's type); none, it admits everything (logged «off: no contract type»). A closed list: (Capo,
# contract, first article, last article, the words of a contract type that make it that Capo's).
CONTRACT_CAPI: Tuple[Tuple[str, str, int, int, str], ...] = (
    ("I", "vendita", 1470, 1547, r"vendit|compravendit|acquist|cessione d\w* (?:immobil|bene|quot)"),
    ("II", "riporto", 1548, 1551, r"riporto"),
    ("III", "permuta", 1552, 1555, r"permut"),
    ("IV", "contratto estimatorio", 1556, 1558, r"estimatori|conto vendita"),
    ("V", "somministrazione", 1559, 1570, r"somministr|fornitur"),
    ("VI", "locazione", 1571, 1654, r"locazion|locat|affitt|canone|sublocaz"),
    ("VII", "appalto", 1655, 1677, r"appalt|ristruttur|costruzion|lavori"),
    ("VIII", "trasporto", 1678, 1702, r"trasport|vettor"),
    ("IX", "mandato", 1703, 1741, r"mandat|commission|spedizion"),
    ("X", "agenzia", 1742, 1753, r"agenzi|agente"),
    ("XI", "mediazione", 1754, 1765, r"mediazion\w* (?:immobiliar|creditiz|commercial)|mediator|agenzia immobiliar|"
                                    r"provvigion"),
    ("XII", "deposito", 1766, 1797, r"deposit|custodi|parcheggi|albergator|albergo"),
    ("XIII", "sequestro convenzionale", 1798, 1802, r"sequestro convenzional"),
    ("XIV", "comodato", 1803, 1812, r"comodat"),
    ("XV", "mutuo", 1813, 1822, r"mutu|prestit|finanziament"),
    ("XVI", "conto corrente", 1823, 1833, r"conto corrent"),
    ("XVII", "contratti bancari", 1834, 1860, r"banc|conto corrent|apertura di credito|anticipazion|sconto|cassett"),
    ("XVIII", "rendita perpetua", 1861, 1871, r"rendita perpetu"),
    ("XIX", "rendita vitalizia", 1872, 1881, r"vitalizi"),
    ("XX", "assicurazione", 1882, 1932, r"assicur|polizz"),
    ("XXI", "giuoco e scommessa", 1933, 1935, r"giuoc|gioc|scommess"),
    ("XXII", "fideiussione", 1936, 1957, r"fideiuss"),
    ("XXIII", "mandato di credito", 1958, 1959, r"mandato di credito"),
    ("XXIV", "anticresi", 1960, 1964, r"anticresi"),
    ("XXV", "transazione", 1965, 1976, r"transazion|transatti"),
    ("XXVI", "cessione dei beni ai creditori", 1977, 1986, r"cessione dei beni ai creditori"),
)
CONTRACT_AREAS = {"locazione": "locazione"}         # a frame area that is a Capo's contract type


def capo_of(ref: Sequence[str]) -> Optional[Tuple[str, str]]:
    """(Capo, contract) of a c.c. article of Titolo III; None for any other article."""
    if not ref or len(ref) != 2 or topic_step._code(ref[1]) != "c.c.":
        return None
    m = re.match(r"\d+", str(ref[0]))
    n = int(m.group(0)) if m else 0
    return next(((capo, name) for capo, name, lo, hi, _ in CONTRACT_CAPI if lo <= n <= hi), None)


def contract_types(facts: Any) -> List[str]:
    """The contract types the extraction names (each contract's "type"; a contract given as a string, the string)."""
    out = []
    for c in (facts or {}).get("contracts") or [] if isinstance(facts, dict) else []:
        value = c.get("type") if isinstance(c, dict) else c
        if str(value or "").strip():
            out.append(" ".join(str(value).split()))
    return out


def contract_scope(analysis: Dict[str, Any], case: str = "") -> Dict[str, Any]:
    """The gate's reading of the case: {"on" (the extraction yields a contract type), "off" (why not), "contracts" (the
    extraction's contract types), "areas", "admitted" (the Capi whose contract type the case's is: its contracts' words
    -- type, parties, clauses -- and the contract areas), "named" (the labels the user's text names)}."""
    from . import defensive_generation as dg, fact_signals
    signals = (analysis or {}).get("fact_signals") or {}
    facts = signals.get("facts")
    areas = [str(a) for a in ((analysis or {}).get("frame") or {}).get("areas") or signals.get("areas") or []]
    types = contract_types(facts)
    words = "\n".join(types + [fact_signals._texts(facts, ("contracts",))] + [CONTRACT_AREAS[a] for a in areas
                                                                              if a in CONTRACT_AREAS]) if types else ""
    admitted = [capo for capo, _, _, _, pattern in CONTRACT_CAPI if words and re.search(pattern, words, re.IGNORECASE)]
    named = {_label((str(r[0]), topic_step._code(r[1]))) for r in dg._refs_in(case or "") if len(r) == 2}
    return {"on": bool(types), **({} if types else {"off": "no contract type"}), "contracts": types, "areas": areas,
            "admitted": admitted, "named": sorted(named)}


def out_of_scope(ref: Sequence[str], scope: Dict[str, Any]) -> Optional[str]:
    """The reason the gate drops the article ("Capo VI – locazione: nessun contratto del caso"), None when it stays."""
    capo = capo_of(ref)
    if not capo or not scope.get("on") or capo[0] in (scope.get("admitted") or ()):
        return None
    if _label((str(ref[0]), topic_step._code(ref[1]))) in set(scope.get("named") or ()):
        return None
    return f"Capo {capo[0]} – {capo[1]}: il contratto del caso non è questo"


# --- v79.6 (item 6): the matter of an article, every book -- a keyword hit outside the case's matters is no candidate --------

MATTER_SPANS = {   # (from, to, matter): the c.c. by Libro, Libri IV and VI by Titolo; the c.p. by Libro
    "c.c.": ((1, 455, "c.c. I persone e famiglia"), (456, 809, "c.c. II successioni e donazioni"),
             (810, 1172, "c.c. III proprietà"), (1173, 1320, "c.c. IV obbligazioni in generale"),
             (1321, 1469, "c.c. IV contratti in generale"), (1470, 1986, "c.c. IV singoli contratti"),
             (1987, 2059, "c.c. IV altre fonti e fatti illeciti"), (2060, 2642, "c.c. V lavoro e impresa"),
             (2643, 2696, "c.c. VI trascrizione"), (2697, 2739, "c.c. VI prove"),
             (2740, 2899, "c.c. VI responsabilità patrimoniale, privilegi e ipoteche"),
             (2900, 2933, "c.c. VI tutela giurisdizionale dei diritti"), (2934, 2969, "c.c. VI prescrizione")),
    "c.p.": ((1, 240, "c.p. I parte generale"), (241, 649, "c.p. II delitti"), (650, 734, "c.p. III contravvenzioni")),
}


def matter_of(ref: Sequence[str]) -> str:
    """The matter an article belongs to (MATTER_SPANS); "" for a code the table does not split (procedure, special
    laws): never gated."""
    if not ref or len(ref) != 2:
        return ""
    spans = MATTER_SPANS.get(topic_step._code(ref[1]))
    m = re.match(r"\d+", str(ref[0]))
    if not spans or not m:
        return ""
    n = int(m.group(0))
    return next((name for lo, hi, name in spans if lo <= n <= hi), "")


def keyword_hit(row: Dict[str, Any]) -> bool:
    """A row the first read's institution search reached by a word of its rubrica, not by the institution's whole name:
    the rubrica lacks one of the institution's content words («riduzione della pena» -> «Riduzione delle ipoteche»)."""
    from .posture import _stems
    inst = str(row.get("_first_read") or "")
    if not inst:
        return False
    want = _stems(inst)
    return bool(want) and not want <= _stems(_rubrica(row))


def matter_gate(rows: Sequence[Dict[str, Any]], keep: Sequence[str] = (), log: Optional[Dict[str, Any]] = None,
                every: bool = False) -> List[Dict[str, Any]]:
    """v79.6 (item 6): the rows without the keyword hits (keyword_hit) whose matter is none of the case's -- the case's
    matters are those of the rows reached by two signals or more, of the first read's whole-name hits and of the
    articles kept by name or rule (keep: labels). With no matter known, nothing is dropped. log["dropped"]: each with
    its matter. v79.8 (A1): every -- any retrieval hit outside the case's matters goes (a vector hit as a keyword one:
    another code's institution, another Libro's or Titolo's article), before the applicability grading."""
    keep = set(keep or ())
    matters = set()
    for r in rows or []:
        ref = _ref(r)
        if not ref:
            continue
        if r.get("_multi_signal") or _label(ref) in keep or (r.get("_first_read") and not keyword_hit(r)):
            matters.add(matter_of(ref))
    matters.discard("")
    if any(m.startswith("c.p. ") for m in matters):        # the general part serves every c.p. matter
        matters.add("c.p. I parte generale")
    out = []
    for r in rows or []:
        ref = _ref(r)
        m = matter_of(ref) if ref else ""
        if matters and ref and (every or keyword_hit(r)) and m and m not in matters and _label(ref) not in keep \
                and not r.get("_multi_signal") and not (r.get("_first_read") and not keyword_hit(r)):
            if log is not None:
                log.setdefault("dropped", []).append({"article": _label(ref), "matter": m,
                                                      "institution": str(r.get("_first_read") or ""),
                                                      "hit": "keyword" if keyword_hit(r) else "vector"})
            continue
        out.append(r)
    if log is not None:
        log["matters"] = sorted(matters)
    return out


# --- v79.12.1: the Capo-scope gate on every book of the codes ------------------------------------------------------------
# Libro IV Titolo III c.c. keeps its own gate (CONTRACT_CAPI, the case's contract type). Every other article of the c.c.
# and the c.p. belongs to a Titolo (a Capo where a Titolo holds several institutions: c.p. Titoli XII and XIII, Libro
# III); within the codes an article whose unit is none of the case's institutions' enters grading only through an
# institution match or the first read -- 2873 c.c. (Riduzione delle ipoteche) for a penalty clause, 890 / 892 c.c. (the
# distances) in an administrative case, 699 / 518-bis / 610 c.p. in a homicide. The general parts serve every
# institution and are never gated (CODE_GENERAL); the procedural codes and the special laws neither. A unit: (first
# number, suffixed, last number, suffixed, name) -- «suffixed» 1 takes the -bis … articles of that number in (518 is
# Titolo VIII, 518-bis Titolo VIII-bis).
CODE_GENERAL: Dict[str, Tuple[Tuple[int, int], ...]] = {
    "c.c.": ((1173, 1469), (2043, 2059), (2697, 2739), (2907, 2969)),   # obbligazioni e contratti in generale, fatti
    "c.p.": ((1, 240),)}                                                # illeciti, prove, tutela e prescrizione; c.p. I
CODE_UNITS: Dict[str, Tuple[Tuple[int, int, int, int, str], ...]] = {
    "c.c.": (
        (1, 0, 10, 1, "Libro I, Titolo I – persone fisiche"),
        (11, 0, 42, 1, "Libro I, Titolo II – persone giuridiche"),
        (43, 0, 73, 1, "Libro I, Titoli III-IV – domicilio, residenza, assenza e morte presunta"),
        (74, 0, 342, 1, "Libro I, Titoli V-IX-bis – famiglia: parentela, matrimonio, filiazione, adozione, "
                        "responsabilità genitoriale"),
        (343, 0, 432, 1, "Libro I, Titoli X-XII – tutela, emancipazione, misure di protezione"),
        (433, 0, 448, 1, "Libro I, Titolo XIII – alimenti"),
        (449, 0, 455, 1, "Libro I, Titolo XIV – atti dello stato civile"),
        (456, 0, 768, 1, "Libro II, Titoli I-IV – successioni e divisione"),
        (769, 0, 809, 1, "Libro II, Titolo V – donazioni"),
        (810, 0, 831, 1, "Libro III, Titolo I – beni"),
        (832, 0, 951, 1, "Libro III, Titolo II – proprietà"),
        (952, 0, 956, 1, "Libro III, Titolo III – superficie"),
        (957, 0, 977, 1, "Libro III, Titolo IV – enfiteusi"),
        (978, 0, 1026, 1, "Libro III, Titolo V – usufrutto, uso e abitazione"),
        (1027, 0, 1099, 1, "Libro III, Titolo VI – servitù prediali"),
        (1100, 0, 1139, 1, "Libro III, Titolo VII – comunione e condominio"),
        (1140, 0, 1172, 1, "Libro III, Titolo VIII – possesso"),
        (1987, 0, 1991, 1, "Libro IV, Titolo IV – promesse unilaterali"),
        (1992, 0, 2027, 1, "Libro IV, Titolo V – titoli di credito"),
        (2028, 0, 2032, 1, "Libro IV, Titolo VI – gestione di affari"),
        (2033, 0, 2040, 1, "Libro IV, Titolo VII – pagamento dell'indebito"),
        (2041, 0, 2042, 1, "Libro IV, Titolo VIII – arricchimento senza causa"),
        (2060, 0, 2081, 1, "Libro V, Titolo I – attività professionali"),
        (2082, 0, 2221, 1, "Libro V, Titolo II – lavoro nell'impresa"),
        (2222, 0, 2238, 1, "Libro V, Titolo III – lavoro autonomo"),
        (2239, 0, 2246, 1, "Libro V, Titolo IV – lavoro subordinato in particolari rapporti"),
        (2247, 0, 2510, 1, "Libro V, Titolo V – società"),
        (2511, 0, 2548, 1, "Libro V, Titolo VI – imprese cooperative e mutue assicuratrici"),
        (2549, 0, 2554, 1, "Libro V, Titolo VII – associazione in partecipazione"),
        (2555, 0, 2574, 1, "Libro V, Titolo VIII – azienda"),
        (2575, 0, 2594, 1, "Libro V, Titolo IX – opere dell'ingegno e invenzioni industriali"),
        (2595, 0, 2620, 1, "Libro V, Titolo X – concorrenza e consorzi"),
        (2621, 0, 2642, 1, "Libro V, Titolo XI – disposizioni penali in materia di società"),
        (2643, 0, 2696, 1, "Libro VI, Titolo I – trascrizione"),
        (2740, 0, 2906, 1, "Libro VI, Titolo III – responsabilità patrimoniale, cause di prelazione, conservazione "
                           "della garanzia")),
    "c.p.": (
        (241, 0, 313, 1, "Libro II, Titolo I – delitti contro la personalità dello Stato"),
        (314, 0, 360, 1, "Libro II, Titolo II – delitti contro la pubblica amministrazione"),
        (361, 0, 401, 1, "Libro II, Titolo III – delitti contro l'amministrazione della giustizia"),
        (402, 0, 413, 1, "Libro II, Titolo IV – delitti contro il sentimento religioso e la pietà dei defunti"),
        (414, 0, 421, 1, "Libro II, Titolo V – delitti contro l'ordine pubblico"),
        (422, 0, 452, 0, "Libro II, Titolo VI – delitti contro l'incolumità pubblica"),
        (452, 1, 452, 1, "Libro II, Titolo VI-bis – delitti contro l'ambiente"),
        (453, 0, 498, 1, "Libro II, Titolo VII – delitti contro la fede pubblica"),
        (499, 0, 518, 0, "Libro II, Titolo VIII – delitti contro l'economia pubblica, l'industria e il commercio"),
        (518, 1, 518, 1, "Libro II, Titolo VIII-bis – delitti contro il patrimonio culturale"),
        (519, 0, 544, 0, "Libro II, Titolo IX – delitti contro la moralità pubblica e il buon costume"),
        (544, 1, 544, 1, "Libro II, Titolo IX-bis – delitti contro il sentimento per gli animali"),
        (545, 0, 555, 1, "Libro II, Titolo X – delitti contro l'integrità e la sanità della stirpe"),
        (556, 0, 574, 1, "Libro II, Titolo XI – delitti contro la famiglia"),
        (575, 0, 593, 1, "Libro II, Titolo XII, Capo I – delitti contro la vita e l'incolumità individuale"),
        (594, 0, 599, 1, "Libro II, Titolo XII, Capo II – delitti contro l'onore"),
        (600, 0, 623, 1, "Libro II, Titolo XII, Capo III – delitti contro la libertà individuale"),
        (624, 0, 639, 1, "Libro II, Titolo XIII, Capo I – delitti contro il patrimonio mediante violenza"),
        (640, 0, 649, 1, "Libro II, Titolo XIII, Capo II – delitti contro il patrimonio mediante frode"),
        (650, 0, 705, 1, "Libro III, Titolo I, Capo I – contravvenzioni di polizia di sicurezza"),
        (706, 0, 730, 1, "Libro III, Titolo I, Capo II – contravvenzioni di polizia amministrativa sociale"),
        (731, 0, 734, 1, "Libro III, Titolo II – contravvenzioni concernenti l'attività sociale della pubblica "
                         "amministrazione")),
}


def _number_key(number: Any) -> Optional[Tuple[int, int]]:
    """(the article's number, 1 when it carries a suffix -- 518-bis -- else 0); None without a number."""
    m = re.match(r"\s*(\d+)\s*(?:-?\s*([a-z]+))?", str(number or ""), re.IGNORECASE)
    return (int(m.group(1)), 1 if m.group(2) else 0) if m else None


def code_unit(ref: Sequence[str]) -> Optional[str]:
    """v79.12.1: «c.p. Libro II, Titolo XII, Capo I – …», the Titolo or Capo of a c.c. / c.p. article (CODE_UNITS); None
    for a general part (CODE_GENERAL), Libro IV Titolo III c.c. (its own Capo gate, capo_of), another code or a number
    outside the table -- never gated."""
    if not ref or len(ref) != 2:
        return None
    code = topic_step._code(ref[1])
    units = CODE_UNITS.get(code)
    key = _number_key(ref[0])
    if not units or key is None or capo_of((str(ref[0]), code)):
        return None
    if any(lo <= key[0] <= hi for lo, hi in CODE_GENERAL.get(code, ())):
        return None
    return next((f"{code} {name}" for lo, los, hi, his, name in units if (lo, los) <= key <= (hi, his)), None)


def institution_match(row: Dict[str, Any], names: Sequence[str]) -> str:
    """v79.12.1: the institution (a first-read name or rubricated name) whose every content word the row's rubrica holds,
    "" when none."""
    from .posture import _stems
    rubrica = _stems(_rubrica(row))
    if not rubrica:
        return ""
    for name in names or []:
        want = _stems(name)
        if want and want <= rubrica:
            return str(name)
    return ""


def code_scope_gate(rows: Sequence[Dict[str, Any]], institutions: Sequence[str], keep: Sequence[str] = (),
                    pass_only: Sequence[str] = (), log: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """v79.12.1: the rows without the code articles of a Titolo / Capo (code_unit) outside the case's institutions. The
    case's units: those of the rows in scope -- an institution match (institution_match on the first read's names and
    rubricated names), a first-read whole-name hit (never a keyword hit: keyword_hit), an article in keep (the user's,
    an active module's action) -- and, for an institution none of whose rows is a match, its first-read rows whose
    rubrica names nothing the institution does not (review: every content word of the rubrica is the institution's --
    «Truffa» for «truffa aggravata», «Riduzione della penale» for «riduzione equitativa della penale»; one shared word is
    a keyword hit, never scope: «Riduzione delle ipoteche», «Porto abusivo di armi»). A row outside them stays only in scope itself or in pass_only (the rules'
    and tables' anchors: kept, never widening the scope). No first-read institution: off, nothing dropped (log «off»).
    log: «units» (the case's), «dropped» ({article, unit, hit}), «kept_by» ({article: why} for a row kept outside the
    units)."""
    from .posture import _stems
    log = log if log is not None else {}
    names = [str(n) for n in institutions or [] if str(n or "").strip()]
    if not names:
        log["off"] = "nessun istituto dalla prima lettura"
        return list(rows or [])
    keep_set, passing = set(keep or ()), set(pass_only or ())

    def in_scope(r: Dict[str, Any]) -> str:
        ref = _ref(r)
        if not ref:
            return ""
        if _label(ref) in keep_set:
            return "articolo dell'utente o del modulo"
        inst = institution_match(r, names)
        if inst:
            return f"istituto «{inst}»"
        if r.get("_first_read") and not keyword_hit(r):
            return f"prima lettura «{r['_first_read']}»"
        return ""
    reasons = {id(r): in_scope(r) for r in rows or []}
    units = {code_unit(_ref(r)) for r in rows or [] if reasons[id(r)] and _ref(r)}
    matched = {str(r.get("_first_read")) for r in rows or [] if reasons[id(r)] and r.get("_first_read")}
    for r in rows or []:          # an institution with no matching row: its first-read rows whose rubrica it covers
        inst = str(r.get("_first_read") or "")
        ref = _ref(r)
        rubrica = _stems(_rubrica(r))
        if inst and ref and inst not in matched and rubrica and rubrica <= _stems(inst):
            units.add(code_unit(ref))
            reasons[id(r)] = reasons[id(r)] or f"prima lettura «{inst}» (rubrica nell'istituto)"
    units.discard(None)
    out, dropped, kept_by = [], {}, {}
    for r in rows or []:
        ref = _ref(r)
        unit = code_unit(ref) if ref else None
        if unit is None or unit in units:
            out.append(r)
            continue
        why = reasons[id(r)] or ("ancora di una regola" if _label(ref) in passing else "")
        if why:
            kept_by[_label(ref)] = why
            out.append(r)
            continue
        hit = ("prima lettura, per parola" if r.get("_first_read") else
               "più segnali" if r.get("_multi_signal") else "recupero")
        dropped.setdefault(_label(ref), {"article": _label(ref), "unit": unit, "hit": hit})
    log.update(units=sorted(units), dropped=list(dropped.values()), kept_by=kept_by)
    return out


def gate_rows(rows: Iterable[Dict[str, Any]], scope: Dict[str, Any], log: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The rows whose article the gate keeps; each dropped one in log["rows"] with its Capo."""
    out = []
    for row in rows:
        ref = _ref(row)
        why = out_of_scope(ref, scope) if ref else None
        if why:
            log.setdefault("rows", []).append({"article": _label(ref), "why": why})
            continue
        out.append(row)
    return out


def gate_topics(topics: Sequence[Dict[str, Any]], scope: Dict[str, Any], log: Dict[str, Any], key: str = "topics"
                ) -> List[Dict[str, Any]]:
    """The topics without the articles the gate drops (articles, matched, planned; the anchor the first kept); a topic
    left with none is gone. log[key]: each topic changed or dropped."""
    out = []
    for t in topics:
        arts = list(t.get("articles") or [])
        gone = [a for a in arts + [t.get("anchor") or ""] if a and _parse(a) and out_of_scope(_parse(a), scope)]
        if not gone:
            out.append(t)
            continue
        planned = int(t.get("planned") or len(arts))
        kept = [a for a in arts if a not in gone]
        log.setdefault(key, []).append({"title": t.get("title"), "dropped": list(dict.fromkeys(gone)),
                                        "gone": not kept})
        if not kept:
            continue
        t = {**t, "articles": kept, "planned": len([a for a in arts[:planned] if a in kept]) or 1,
             "matched": [a for a in t.get("matched") or [] if a in kept]}
        if t.get("anchor") in gone or t.get("anchor") not in kept:
            t["anchor"] = kept[0]
        if isinstance(t.get("grades"), dict):
            t["grades"] = {a: g for a, g in t["grades"].items() if a not in gone}
        out.append(t)
    return out


def _parse(label: str) -> Optional[Tuple[str, str]]:
    m = re.match(r"\s*(\d+(?:-[a-z]+)?)\s+(.+?)\s*$", str(label or ""))
    return (m.group(1), m.group(2)) if m else None


# --- v60n: area-code priority -- a procedural code's article anchors only a procedural institution ----------------------
# For a substantive institution the area's substantive code (c.p. in criminal, c.c. in civil) wins over any procedural
# code's hit: a c.p.p. / c.p.c. article never anchors it (v6x sims: p2 «appropriazione indebita» -> 381 c.p.p., the
# arrest article listing the offence). An institution is procedural when its name says so (_PROCEDURAL_RE, a closed list
# erring towards procedural: a wrong «procedural» only keeps today's behaviour) or when the store rubricates it only in a
# procedural code (interactive_flow.first_read_search: its rubrica hits).
PROCEDURAL_CODES = ("c.p.p.", "c.p.c.")
_PROCEDURAL_RE = re.compile(
    r"\b(?:arrest[oi]|ferm[oi]|convalid\w*|misur\w* cautelar\w*|cautelar\w*|custodia|domiciliar\w*|riesame|"
    r"impugnazion\w*|appell[oi]|ricors[oi]|cassazione|revisione|rimessione|competenz\w*|giurisdizion\w*|"
    r"notific\w*|perquisizion\w*|ispezion\w*|intercettazion\w*|sequestr\w* (?:preventiv|probatori|conservativ|"
    r"giudiziari)\w*|incidente probatorio|indagin\w*|udienz\w*|archiviazion\w*|rinvio a giudizio|patteggiament\w*|"
    r"applicazione della pena su richiesta|giudizio|rito|decreto penale|decreto ingiuntivo|ingiunzion\w*|opposizion\w*|"
    r"precetto|pignorament\w*|esecuzione forzata|espropriazione forzata|sfratt\w*|procediment\w*|process[oi]|"
    r"processual\w*|litisconsorzi\w*|chiamata in causa|intervento|provvediment\w* d'urgenza|testimon\w*|"
    r"mezzi di prova|prova|prove|querel\w*|citazione|costituzione in giudizio|contumacia|domanda giudiziale|"
    r"legittimazione|interesse ad agire|inammissibilit\w*|improcedibilit\w*|procedibilit\w*|revocazione|"
    r"spese di lite|gratuito patrocinio|difensore|sentenz\w*|giudicato|termine per (?:impugnare|proporre))\b",
    re.IGNORECASE)


def procedural(name: Any) -> bool:
    """v60n: an institution named as a procedural one (_PROCEDURAL_RE)."""
    return bool(_PROCEDURAL_RE.search(" ".join(str(name or "").split())))


def may_anchor(institution: str, topic: Dict[str, Any], procedural_names: Iterable[str] = ()) -> bool:
    """v60n: whether the topic may hold the institution -- never a procedural code's topic for a substantive one."""
    if (topic.get("code") or "") not in PROCEDURAL_CODES:
        return True
    low = {str(x).lower() for x in procedural_names or ()}
    return procedural(institution) or str(institution or "").lower() in low


def substantive_codes(codes: Iterable[str]) -> List[str]:
    """v60n: the area's substantive codes (c.p. in criminal, c.c. in civil), in the areas' order."""
    return [c for c in codes or () if c not in PROCEDURAL_CODES]


# --- v60c: hypotheses anchored -- the graded rows' rubriche by stems first, then the Sezione / Capo titles by similarity -
# An institution of the hypothesis anchors in a topic when its content words (two, or its only one) are in the rubrica
# of one of the topic's graded rows ("rubrica"); else when its embedding is at ANCHOR_MIN or more from the topic's Sezione
# or Capo title ("titolo", env HYPOTHESIS_ANCHOR_MIN; one embedding call per case, titles and names together). Each
# hypothesis' path and score are logged; the others stay «Ipotesi da verificare».
ANCHOR_MIN_ENV = "HYPOTHESIS_ANCHOR_MIN"
ANCHOR_MIN = 0.75


def anchor_min() -> float:
    try:
        return float(os.getenv(ANCHOR_MIN_ENV) or ANCHOR_MIN)
    except ValueError:
        return ANCHOR_MIN


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    den = (sum(x * x for x in a) ** 0.5) * (sum(y * y for y in b) ** 0.5)
    return num / den if den else 0.0


def _embedder() -> Callable[[List[str]], List[List[float]]]:
    from .ai_chat import embedding_model
    return lambda texts: [list(v) for v in embedding_model.embed_documents(texts)]


def topic_vectors(session, topics: Sequence[Dict[str, Any]], rows: Iterable[Dict[str, Any]] = (),
                  embed=None) -> Dict[str, Any]:
    """The titles each topic is matched on by similarity: its Sezione or Capo title. No graph read, nothing embedded
    here. {"topics": [[(what, title)] per topic]}."""
    return {"topics": [[("sezione" if t.get("source") == "sezione" else "capo", unit_title(t["unit"]))]
                       if t.get("unit") else [] for t in topics]}


def _stem_anchor(institution: str, topics: Sequence[Dict[str, Any]], rubriche: Dict[str, str],
                 codes: Sequence[str] = (), procedural_names: Iterable[str] = ()) -> Tuple[int, Optional[int], str]:
    """(shared words, topic index, the matched rubrica) of the best rubrica match of the topics' graded rows. v60n:
    never a topic the institution may not anchor in (may_anchor); on a tie, the area's substantive code's topic."""
    wanted = topic_step._content_stems(institution) - topic_step.GENERIC_STEMS
    best = (0, None, "")
    if not wanted:
        return best
    mine = substantive_codes(codes)
    best_key = (0, False)
    for i, t in enumerate(topics):
        if not may_anchor(institution, t, procedural_names):
            continue
        for label in (t.get("grades") or {}):
            shared = len(wanted & (topic_step._content_stems(rubriche.get(label, "")) - topic_step.GENERIC_STEMS))
            key = (shared, (t.get("code") or "") in mine)
            if shared >= min(2, len(wanted)) and key > best_key:
                best, best_key = (shared, i, f"art. {label}: {rubriche.get(label, '')}"), key
    return best


def anchor_by_similarity(topics: List[Dict[str, Any]], vectors: Dict[str, Any], hypotheses: Sequence[Dict[str, Any]],
                         embed: Optional[Callable[[List[str]], List[List[float]]]] = None,
                         threshold: Optional[float] = None, rubriche: Optional[Dict[str, str]] = None,
                         codes: Sequence[str] = (), procedural_names: Iterable[str] = ()
                         ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """(the unanchored hypotheses, one log entry per hypothesis: the path that attached it -- "rubrica" or "titolo" --
    or none, its institution, topic, matched text and score). v60n: an institution anchors only where it may
    (may_anchor: a procedural code's topic for a procedural institution only), the area's substantive code first
    (codes: the case's area codes)."""
    procedural_names = list(procedural_names or ())
    threshold = anchor_min() if threshold is None else threshold
    rubriche = rubriche or {}
    by_stem = {}

    def names_of(h):            # v60e: the code's own name for an institution first
        return list(dict.fromkeys([x for x in list(h.get("rubricated") or []) + list(h.get("institutions") or []) if x]))
    for h in hypotheses or []:
        for inst in names_of(h):
            shared, i, match = _stem_anchor(inst, topics, rubriche, codes, procedural_names)
            if i is not None and shared > by_stem.get(id(h), (0,))[0]:
                by_stem[id(h)] = (shared, i, inst, match)
    rest = [h for h in hypotheses or [] if id(h) not in by_stem]
    names = list(dict.fromkeys(i for h in rest for i in names_of(h)))
    titles = list(dict.fromkeys(txt for mine in vectors.get("topics") or [] for _, txt in mine if txt))
    got: Dict[str, List[float]] = {}
    if names and titles:
        texts = names + [t for t in titles if t not in names]
        got = dict(zip(texts, (embed or _embedder())(texts)))
    unanchored, log = [], []
    for h in hypotheses or []:
        if id(h) in by_stem:
            shared, i, inst, match = by_stem[id(h)]
            topics[i]["hypotheses"].append({**h, "anchor_path": "rubrica", "anchored_by": match,
                                            "anchor_institution": inst})
            log.append({"line": h.get("line"), "path": "rubrica", "institution": inst, "topic": topics[i]["title"],
                        "match": match, "score": shared, "attached": True})
            continue
        best = (-1.0, None, "", "")
        for inst in names_of(h):
            if inst not in got:
                continue
            for i, mine in enumerate(vectors.get("topics") or []):
                if i < len(topics) and not may_anchor(inst, topics[i], procedural_names):
                    continue
                for what, title in mine:
                    if title in got:
                        score = _cosine(got[inst], got[title])
                        if score > best[0]:
                            best = (score, i, inst, f"{what}: {title}")
        attached = best[1] is not None and best[0] >= threshold and best[1] < len(topics)
        if attached:
            topics[best[1]]["hypotheses"].append({**h, "anchor_path": "titolo", "anchor_score": round(best[0], 3),
                                                  "anchored_by": best[3], "anchor_institution": best[2]})
        else:
            unanchored.append({**h, "anchor_score": round(best[0], 3) if best[1] is not None else None})
        log.append({"line": h.get("line"), "path": "titolo" if attached else "none", "institution": best[2],
                    "topic": topics[best[1]]["title"] if best[1] is not None and best[1] < len(topics) else "",
                    "match": best[3], "score": round(best[0], 3), "attached": attached, "min": threshold})
    return unanchored, log


# --- v60c: the offences, and the immediate answer ------------------------------------------------------------------------

def offence_rows(rows: Iterable[Dict[str, Any]], hypotheses: Sequence[Dict[str, Any]] = ()) -> List[Dict[str, Any]]:
    """The offences the procedural store identifies among the graded rows (a c.p. offence whose text gives its
    procedural facts: procedural_facts.from_row): every one graded 2 or more, and one graded 1 a first-read institution
    names (its content words in the rubrica, two or its only one). Best grade first."""
    from . import procedural_facts, rule_quote
    from .article_text import row_abrogated
    names = [topic_step._content_stems(i) - topic_step.GENERIC_STEMS
             for h in hypotheses or [] for i in h.get("institutions") or []]
    out = []
    for row in _best_rows(rows, lambda g: g >= 1):
        ref = _ref(row)
        if ref is None or ref[1] != "c.p." or not rule_quote.offence(ref) or row_abrogated(row):
            continue
        if not procedural_facts.facts_of(row).get("procedibilita"):
            continue
        rubrica = topic_step._content_stems(_rubrica(row))
        if row["_grade"] >= STRONG or any(n and len(n & rubrica) >= min(2, len(n)) for n in names):
            out.append(row)
    return out


def ensure_offences(topics: List[Dict[str, Any]], others: List[Dict[str, Any]], rows: Iterable[Dict[str, Any]],
                    unit_of: Dict[Tuple[str, str], Dict[str, str]], hypotheses: Sequence[Dict[str, Any]] = ()
                    ) -> List[str]:
    """Every identified offence (offence_rows) a writable article of a topic: of the one listing it, else of its
    Sezione's or Capo's, else of a topic of its own; marked in topic["offences"], out of the altri profili. The labels."""
    labels = []
    for row in offence_rows(rows, hypotheses):
        label = _label(_ref(row))
        labels.append(label)
        home = next((t for t in topics if label in (t.get("articles") or [])[:int(t.get("planned") or 0)]), None)
        if home is None:
            names = set((unit_of.get(unit_key(row)) or {}).values())
            home = next((t for t in topics if label in (t.get("articles") or [])), None) or \
                next((t for t in topics if t.get("unit") and t["unit"] in names), None)
            if home is None:
                home = _topic({"level": "rubrica", "unit": "", "rows": [row]})
                topics.append(home)
            else:
                arts = [a for a in home["articles"] if a != label]
                planned = int(home.get("planned") or 0)
                home["articles"] = arts[:planned] + [label] + arts[planned:]
                home["planned"] = planned + 1
                home["matched"] = list(dict.fromkeys(list(home.get("matched") or []) + [label]))
                home.setdefault("grades", {})[label] = row["_grade"]
                home["graded"] = len(home["grades"])
        home["offences"] = list(dict.fromkeys(list(home.get("offences") or []) + [label]))
        for t in others:
            if label in (t.get("articles") or []):
                t["articles"] = [a for a in t["articles"] if a != label]
    others[:] = [t for t in others if t.get("articles")]
    return labels


def immediate(topics: Sequence[Dict[str, Any]], cap: int = TOPIC_ARTICLES) -> List[int]:
    """The topics (1-based) the immediate answer takes: the topics holding an identified offence first (their articles
    are written first: interactive_flow.capped_refs follows this order), the first topic always, then the next ones in
    order while their writable articles fit the cap."""
    if not topics:
        return []
    chosen = [i for i, t in enumerate(topics, 1) if t.get("offences")]
    chosen += [] if 1 in chosen else [1]
    used = sum(int(topics[i - 1].get("planned") or 1) for i in chosen)
    for i, t in enumerate(topics, 1):
        need = int(t.get("planned") or 1)
        if i not in chosen and used + need <= cap:
            chosen.append(i)
            used += need
    return chosen
