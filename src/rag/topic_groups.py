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
                      rubriche: Dict[str, str]) -> List[Dict[str, Any]]:
    """Each hypothesis to the topic one of its institutions anchors best (module docstring); the unanchored ones."""
    unanchored = []
    words = [(_topic_words(t, rubriche), t) for t in topics]
    for h in hypotheses or []:
        stems = [topic_step._content_stems(inst) - topic_step.GENERIC_STEMS for inst in h.get("institutions") or []]
        stems = [x for x in stems if x]

        def score(w):           # an institution anchors when all its content words, two at most needed, are the topic's
            return max((len(x & w) for x in stems if len(x & w) >= min(2, len(x))), default=0)
        scored = [(score(w), -i, t) for i, (w, t) in enumerate(words)]
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


def _stem_anchor(institution: str, topics: Sequence[Dict[str, Any]], rubriche: Dict[str, str]
                 ) -> Tuple[int, Optional[int], str]:
    """(shared words, topic index, the matched rubrica) of the best rubrica match of the topics' graded rows."""
    wanted = topic_step._content_stems(institution) - topic_step.GENERIC_STEMS
    best = (0, None, "")
    if not wanted:
        return best
    for i, t in enumerate(topics):
        for label in (t.get("grades") or {}):
            shared = len(wanted & (topic_step._content_stems(rubriche.get(label, "")) - topic_step.GENERIC_STEMS))
            if shared >= min(2, len(wanted)) and shared > best[0]:
                best = (shared, i, f"art. {label}: {rubriche.get(label, '')}")
    return best


def anchor_by_similarity(topics: List[Dict[str, Any]], vectors: Dict[str, Any], hypotheses: Sequence[Dict[str, Any]],
                         embed: Optional[Callable[[List[str]], List[List[float]]]] = None,
                         threshold: Optional[float] = None, rubriche: Optional[Dict[str, str]] = None
                         ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """(the unanchored hypotheses, one log entry per hypothesis: the path that attached it -- "rubrica" or "titolo" --
    or none, its institution, topic, matched text and score)."""
    threshold = anchor_min() if threshold is None else threshold
    rubriche = rubriche or {}
    by_stem = {}

    def names_of(h):            # v60e: the code's own name for an institution first
        return list(dict.fromkeys([x for x in list(h.get("rubricated") or []) + list(h.get("institutions") or []) if x]))
    for h in hypotheses or []:
        for inst in names_of(h):
            shared, i, match = _stem_anchor(inst, topics, rubriche)
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
        if not (row.get("_proc") or procedural_facts.from_row(row)).get("procedibilita"):
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
