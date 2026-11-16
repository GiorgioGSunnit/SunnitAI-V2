"""v68: the plain route's grounding on a «cenni» or a «quesito» (nodes/synthesis.synthesize_answer) -- the parere has its
citation discipline, the plain answer had none.

  ground(answer, rows, log)   every article the answer cites is among the retrieved sections (article_codes.
                              codes_by_article: their numbers and codes) -- else its number goes and the sentence stays,
                              generic (anchored_writing.drop_citations; a bare «art. N» with its «dell'»); a sentence on
                              case law («la giurisprudenza ha …», «la Cassazione ha …»: anchored_writing._CASE_LAW_RE)
                              stays only when a ruling is in the retrieved context (rulings_in) -- else it goes
  applies(deliverable)        the intake's deliverable is one of DELIVERABLES
  store_lines(rows)           v69: a quesito's procedural articles among the rows, their terms as the store reads them
                              (procedural_facts.term_lines, the one accessor) -- the term and its article, no date
  cenni_area(subject)         v69: the code and the unit (the Titolo first) a cenni subject names in the codes'
                              institutions index («prescrizione civile» -> c.c., «Della prescrizione e della decadenza»,
                              2934 ss.); None when no unit, or two codes, name it
  cenni_restrict(...)         v69: the cenni's rows inside that area (its articles read when the retrieval missed them)
                              and the nota dottrinale's from the same code family only
  no_closing_question(answer) v69: a cenni never closes with a question

Lawyers forgive a thin answer; they don't forgive invented case law. No case-specific logic: the rows decide.
"""

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

DELIVERABLES = ("cenni", "quesito")
# the codes whose label the rows and the citations write alike -- another law's label may be written otherwise: its
# citation goes only when no row has its number at all
CODE_LABELS = ("c.c.", "c.p.", "c.p.c.", "c.p.p.", "c.p.a.", "Cost.")
# v68 (my draft): the synthesis prompt's line on a cenni or a quesito whose retrieved context holds no ruling
NO_RULINGS_NOTE = ("The retrieved data contains no court ruling: do not mention case law (giurisprudenza, Cassazione, "
                   "sentenze, orientamenti) and cite only articles that appear in the data above.")
_BARE_RE = re.compile(r"(?P<intro>\s*\(?\s*(?:(?:ai sensi|a norma|in base)\s+)?(?:dell['’]|all['’]|nell['’]|dall['’]|"
                      r"sull['’]|l['’])\s*)?\b(?:art\.|articolo)\s*(?P<num>\d+(?:-[a-z]+)?)\b\)?", re.IGNORECASE)


STORE_CODES = ("c.p.c.", "c.p.p.", "c.p.a.")
# v69 (my draft): the synthesis prompt's block on a quesito -- the terms the code read from the retrieved articles
STORE_TERMS_NOTE = ("Terms read by the code from the retrieved articles' text -- state the one the question asks for "
                    "with its article, never compute a date:\n{lines}")


def store_lines(rows: Iterable[Dict[str, Any]]) -> List[str]:
    """v69: «- art. 166 c.p.c.: Costituzione: almeno settanta giorni prima dell'udienza … (art. 166, …)» for each
    procedural article (STORE_CODES) among the rows, from its own text -- the store's reading, never a model's."""
    from . import article_codes, procedural_facts
    from .defensive_generation import _article_num
    out, seen = [], set()
    for row in rows or []:
        d = row.get("d") if isinstance(row.get("d"), dict) else {}
        s = row.get("s") if isinstance(row.get("s"), dict) else {}
        if d.get("document_type") in ("interpretation", "special") or row.get("_source") == "ruling_support":
            continue
        number = _article_num(str(s.get("name") or "")) if s.get("name") else ""
        code = article_codes._label(str(d.get("name") or "")) if d.get("name") else ""
        if not number or code not in STORE_CODES or (number, code) in seen:
            continue
        seen.add((number, code))
        facts = procedural_facts.from_row({**row, "_article": (number, code)})
        lines, _ = procedural_facts.term_lines(facts, f"{number} {code}")
        out += [f"- art. {number} {code}: {x}" for x in lines]
    return out


# --- v69: the plain cenni, inside the area its subject names --------------------------------------------------------------

CENNI_CODES = ("c.c.", "c.p.", "c.p.c.", "c.p.p.")
# a family word in the subject narrows the codes («prescrizione civile», «del reato»); its stem never matches a unit
CENNI_FAMILIES = (("civil", ("c.c.", "c.p.c.")), ("penal", ("c.p.", "c.p.p.")), ("reato", ("c.p.", "c.p.p.")),
                  ("reati", ("c.p.", "c.p.p.")))
# the unit a cenni rests on: the Titolo first, then the narrower ones, the broadest last
_UNIT_ORDER = {"TITOLO": 0, "CAPO": 1, "SEZIONE": 2, "PARAGRAFO": 3, "PARTE": 4, "LIBRO": 5, "ARTICOLO": 6}
CENNI_MIN_ROWS = 3              # fewer of the area's articles retrieved: its first ones read
CENNI_FETCH = 4
FAMILY_WORDS = {"c.c.": "civil", "c.p.c.": "civil", "c.p.": "penal", "c.p.p.": "penal"}


def cenni_area(subject: str, index_of=None) -> Optional[Dict[str, Any]]:
    """v69: {"code", "unit", "level", "articles"} -- the unit of a code's institutions index (fact_resolver.load_index)
    sharing the most content stems with the subject (the family words aside -- topic_step's GENERIC_STEMS count here:
    «prescrizione» is the subject), the Titolo before a narrower unit; None when none shares one, or the best are in
    two codes."""
    from . import fact_resolver
    from .topic_step import _content_stems
    stems = set(_content_stems(subject))
    codes: Sequence[str] = CENNI_CODES
    for key, family in CENNI_FAMILIES:          # the raw words: «reato» is no content stem
        if re.search(rf"\b{key}", (subject or "").lower()):
            codes = family
    stems = {s for s in stems if not s.startswith(tuple(k for k, _ in CENNI_FAMILIES))}
    found = []
    for code in codes:
        for e in (index_of or fact_resolver.load_index)(code) or []:
            if not e.get("articles"):
                continue
            shared = stems & set(_content_stems(f"{e.get('name') or ''} {e.get('rubricated') or ''}"))
            if shared:
                found.append((len(shared), -_UNIT_ORDER.get(str(e.get("level") or ""), 9), len(e["articles"]), code, e))
    if not found:
        return None
    most = max(f[0] for f in found)
    if len({f[3] for f in found if f[0] == most}) > 1:      # two codes name it alike: no area, never a guess
        return None
    best = max((f for f in found if f[0] == most), key=lambda f: f[:3])
    e = best[4]
    return {"code": best[3], "unit": str(e.get("name") or ""), "level": str(e.get("level") or ""),
            "articles": [str(a) for a in e.get("articles") or []]}


def _ref_of(row: Dict[str, Any]) -> Optional[tuple]:
    from . import article_codes
    from .article_suffix import normalize_number
    from .defensive_generation import _article_num
    tagged = row.get("_article")
    if tagged and len(tagged) == 2:
        return normalize_number(str(tagged[0])), str(tagged[1])
    d = row.get("d") if isinstance(row.get("d"), dict) else {}
    s = row.get("s") if isinstance(row.get("s"), dict) else {}
    number = _article_num(str(s.get("name") or "")) if s.get("name") else ""
    code = article_codes._label(str(d.get("name") or "")) if d.get("name") else ""
    return (normalize_number(str(number)), code) if number and code else None


def cenni_restrict(subject: str, rows: Sequence[Dict[str, Any]], notes: Sequence[Dict[str, Any]],
                   log: Dict[str, Any], index_of=None, fetch=None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """v69: (the rows, the nota dottrinale's rows) of a plain cenni -- with its subject's area (cenni_area): only the
    rows of the area's articles, its first articles read (fetch(refs)) when fewer than CENNI_MIN_ROWS were retrieved;
    the notes only from a commentary of the same code family (its name: «civil», «penal»). No area: as they came."""
    from .article_suffix import normalize_number
    area = cenni_area(subject, index_of)
    log["area"] = area and {k: area[k] for k in ("code", "unit", "level")}
    if not area:
        return list(rows or []), list(notes or [])
    inside = {(normalize_number(a), area["code"]) for a in area["articles"]}
    kept = [r for r in rows or [] if _ref_of(r) in inside]
    log["dropped"] = [" ".join(_ref_of(r) or ("?",)) for r in rows or [] if _ref_of(r) not in inside][:20]
    if len(kept) < CENNI_MIN_ROWS and fetch is not None:
        have = {_ref_of(r) for r in kept}
        missing = [(a, area["code"]) for a in area["articles"] if (normalize_number(a), area["code"]) not in have]
        try:
            got = [r for r in fetch(missing[:CENNI_FETCH]) or [] if _ref_of(r) in inside]
        except Exception as exc:            # noqa: BLE001 -- the graph unread: the retrieved rows inside the area
            log["fetch_failed"] = f"{type(exc).__name__}: {exc}"[:200]
            got = []
        kept += got
        log["read"] = [" ".join(_ref_of(r) or ("?",)) for r in got]
    word = FAMILY_WORDS.get(area["code"], "")
    same = [n for n in notes or [] if word and word in str(((n.get("d") or {}) if isinstance(n.get("d"), dict) else {})
                                                           .get("name") or "").lower()]
    log["notes_dropped"] = len(list(notes or [])) - len(same)
    return kept, same


def graph_fetch(refs: List[tuple]) -> List[Dict[str, Any]]:
    """v69: the area's articles read from the graph by number (cenni.graph_tools' fetch)."""
    from . import cenni
    from .doc_lookup import _fetch_doc_names
    from .main import NEO4J_DATABASE, driver
    return cenni.graph_tools(lambda: _fetch_doc_names(driver, NEO4J_DATABASE))[0](list(refs))


_CLOSING_QUESTION_RE = re.compile(r"(?:^|(?<=[.!?:])\s+|\n+)[^.!?\n]*\?\s*$")


def no_closing_question(answer: str) -> str:
    """v69: a cenni never closes with a question -- its last sentence asking one, gone (again, while one is left)."""
    text = (answer or "").rstrip()
    while text.endswith("?"):
        m = _CLOSING_QUESTION_RE.search(text)
        if not m or m.start() == 0:
            break
        text = text[:m.start()].rstrip()
    return text


def applies(deliverable: Any) -> bool:
    return str(deliverable or "") in DELIVERABLES


def rulings_in(rows: Iterable[Dict[str, Any]]) -> bool:
    """A ruling (a massima) is in the retrieved context: an «interpretation» document or a ruling_support row."""
    for row in rows or []:
        d = row.get("d") if isinstance(row.get("d"), dict) else {}
        if d.get("document_type") == "interpretation" or row.get("_source") == "ruling_support":
            return True
    return False


def _documents(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out, seen = [], set()
    for row in rows or []:
        d = row.get("d") if isinstance(row.get("d"), dict) else {}
        if d.get("name") and d["name"] not in seen:
            seen.add(d["name"])
            out.append({"id": d.get("id"), "name": d["name"]})
    return out


def _known(number: str, code: str, table: Dict[str, set]) -> bool:
    from .article_suffix import normalize_number
    codes = table.get(normalize_number(str(number))) or set()
    if not codes:
        return False
    return code not in CODE_LABELS or code in codes


def ground(answer: str, rows: Sequence[Dict[str, Any]], log: Dict[str, Any]) -> str:
    """The answer with no citation of an article the rows do not hold (its number dropped, its sentence kept) and no
    case-law sentence without a ruling among the rows. log: {"dropped_citations", "dropped_case_law", "rulings"}."""
    from . import anchored_writing, article_codes, plan_then_write as ptw
    if not (answer or "").strip():
        return answer
    table = article_codes.codes_by_article(rows)
    documents = _documents(rows)
    rulings = rulings_in(rows)
    log.setdefault("dropped_citations", [])
    log.setdefault("dropped_case_law", [])
    log["rulings"] = rulings
    edits = []
    for s, e in ptw._sentence_spans(answer):
        sentence = answer[s:e]
        if not rulings and anchored_writing._CASE_LAW_RE.search(sentence):
            edits.append((s, e, ""))
            log["dropped_case_law"].append(" ".join(sentence.split())[:200])
            continue
        bad = [r for _, _, refs in ptw._citations(sentence, documents) for r in refs if not _known(r[0], r[1], table)]
        now = anchored_writing.drop_citations(sentence, bad, documents) if bad else sentence
        for m in reversed(list(_BARE_RE.finditer(now))):          # «art. 2948» with no code: its number decides
            if article_codes._has_code(now, m.start("num") - 4, m.end("num")) or _known(m.group("num"), "", table):
                continue
            bad.append((m.group("num"), ""))
            now = now[:m.start()] + now[m.end():]
        if bad:
            now = re.sub(r"\(\s*\)", "", re.sub(r"[ \t]{2,}", " ", now)).replace(" ,", ",").replace(" .", ".").strip()
            if sentence.rstrip().endswith((".", "!", "?")) and not now.endswith((".", "!", "?", ":", ";")):
                now = now.rstrip(" ,;") + "."              # «… ai sensi dell'art. 2947 c.c.»: its period was the code's
            edits.append((s, e, now))
            log["dropped_citations"].append({"articles": [" ".join(x for x in r if x) for r in bad],
                                             "sentence": " ".join(sentence.split())[:200]})
    for s, e, now in reversed(edits):
        answer = answer[:s] + now + answer[e:]
    answer = re.sub(r"(?m)^[ \t]*-[ \t]*$\n?", "", answer)          # a bullet left empty
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+\n", "\n", re.sub(r"[ \t]{2,}", " ", answer))).strip()
