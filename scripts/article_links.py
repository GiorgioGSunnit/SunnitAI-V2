"""The structure and the cross-references of the primary codes, read off their Sections' text; --write puts them in the
graph.

    cd /root/v38 && venv/bin/python scripts/article_links.py --env .env.v45 --census [--only "Codice Civile"] [--codes c.c.,c.p.c.]
    cd /root/v38 && venv/bin/python scripts/article_links.py --env .env.v45 --dry-run OUTDIR [--codes ...]
    cd /root/v38 && venv/bin/python scripts/article_links.py --env .env.v45 --write OUTDIR [--codes ...]
    ... --dry-run OUTDIR --compare OLD/refers_to.csv     # what changed against a previous dry run (changes.csv)
    ... --write OUTDIR --prune                           # the stale edges deleted first, then the merge
    ... --unitless                                       # v60h: the normative documents no Unit holds yet (read-only)
    ... --dry-run OUTDIR --doc LEGAL_DOC::cba1e3…        # v60h: one document by its store id, whatever its type

v60h: --doc ID (repeatable) reads documents by their store id, whatever their document_type («Codice del processo
amministrativo 2025»); --types primary,special widens the listing; --unitless lists the normative documents (their name
names an act: citations.NORMATIVE_NAME_RE) of --types (default primary, special) whose articles no Unit holds
(HAS_ARTICLE), with their articles, and stops. A document whose texts hold no heading marker in capitals is read with its
markers in title case («Titolo I – …», «Capo I», «Libro primo»), each accepted only where a heading stands -- at the
start of the text, after a sentence's end, or right after another heading -- never after a preposition or an article
(«di cui al Capo I», «del Titolo II»: a reference); the census says which («markers: capitals | title case»). The census
gives the rows --write would send: units by level, CHILD_OF, HAS_ARTICLE, REFERS_TO by resolution. --write is additive:
MERGE only, every node and edge it creates stamped source: 'article_links' and computed_at, nothing that exists is changed
(an edge's resolution set only where it has none), nothing deleted unless --prune.

Target: every Document with document_type = 'primary', the six codes first (Codice Civile, c.p.c., c.p., c.p.p., c.p.a.,
Codice della strada), then the others by name; --only NAME (repeatable) keeps the documents whose name starts with it,
--codes the ones whose code label (or name prefix) is listed. One session: read-only for --census and --dry-run (MATCH
/ RETURN only), a write session for --write.

Every Section is known by its element id, never by its number. Articles are ordered by the number in their Section name
(bis / ter / quater ... after the plain number), an article's whole node before its fragments ("4", then "4.0.0" ...
"4.4.0": one article), a "_1" duplicate after its original; the NEXT chain only breaks ties and places the Sections with
no article number (after the numbered one before them on the chain). The numbering runs are the name prefixes: "prel-"
the disposizioni sulla legge in generale, "att-" the disposizioni di attuazione, each its own unit tree, never merged
with the code; their articles are labelled "2 [disposizioni di attuazione]" and a reference from one of them resolves in
its own run first, then in the code's numbering. A Section name with a "_N" suffix next to the same name without it
(att-38 / att-38_1) is counted per document as an ingestion defect.

--write OUTDIR writes the dry run's very CSVs to OUTDIR, then MERGEs, idempotently, in sequential batches of 500 via
UNWIND (no parallelism): (:Unit {code, level, path, name}) (path: the unit's names from the top of its tree, " > "
between, a prel- / att- run's name first; an index on code, level, path), (:Unit)-[:CHILD_OF]->(:Unit),
(:Unit)-[:HAS_ARTICLE]->(:Section) from an article's deepest unit to each of its Sections, and
(:Section)-[:REFERS_TO {pattern, text, source: 'article_links', computed_at}]->(:Section) from the Section whose text
holds the reference to the target article's first Section (computed_at set when created). It prints what was created and
what was already there.

Placement. The loader leaves a code's headings inside the Sections' text, mostly at the end of the article before the
unit they open ("... per usucapione o per destinazione del padre di famiglia. CAPO II – DELLE SERVITU’ COATTIVE", art.
1031 c.c.), sometimes at the start of the article they open. The markers LIBRO, PARTE, TITOLO, CAPO and SEZIONE (in
capitals, with a roman or ordinal number, then the unit's name after a dash) are read in the articles' order; a marker
opens its unit from the next article on, or from its own article when nothing precedes it in the Section's text; a
marker closes the units below it (a new CAPO closes the SEZIONE). The articles before the first marker in the text
(c.p.c. 1-5: art. 1 is the article itself, its headings in no article's text) are in the opening units:
one below the first marker seen later at each level, a level counting only if no marker above it came first (c.p.c.:
SEZIONE II at art. 6, CAPO II, TITOLO II, LIBRO SECONDO later -- LIBRO I, TITOLO I, CAPO I, SEZIONE I), named after the
document's heading-only Sections (no article number, the text a marker line) of the same level and ordinal, else
"SEZIONE I (intestazione non nel testo)"; the units above stay open past the first marker (art. 6: CAPO I, SEZIONE II).
An article is placed when it has at least one unit; unplaced, with why: before the first marker (only with no opening
unit), a heading-only Section, or a Section name with no article number. The ingestion repo's build_heading_index.py is
not reachable from here (SunnitAI-V2's scripts/build_heading_index.py only embeds rubricas), so the markers are parsed
here, on the line-start keywords lex_package/parsing_utils uses (parser_regolamento.py, parser_banca.py).

v53 (2044 c.c. pointed to c.c. 52, 55 and 7: "articolo 52, commi secondo, terzo e quarto, del codice penale", "art. 7,
comma 1, L. 26 aprile 2019, n. 36"): after an article's number its comma clause ("comma 1", "commi secondo, terzo e
quarto", "secondo comma", "lettera b)", "n. 3") and the line breaks are skipped before the code or act is read
(qualifier); an act that is not one of the codes read is "other act", unresolved, never the reference's own code. Every
reference, and every REFERS_TO edge (e.resolution), says how its target was read: explicit_code (a code named, its own
included: "del presente codice", "c.c."), same_code (nothing named) or relative ("articolo precedente"). --dry-run also
counts the stale edges (this source's, from the documents read, whose (from, to) the new rows no longer hold;
stale_edges.csv, read-only); --write --prune deletes them by id before the merge; --compare OLD_CSV lists the
references whose target changed or that no longer resolve (changes.csv, 20 printed).

Cross-references, by pattern: "articoli N, M e K" (list), "articolo N" (word), "art. N" (abbreviation), "articolo
precedente / seguente / successivo" (relative), "del presente codice" (on a reference it follows), and a reference
naming another code or act (other code; resolved only when that code is one of the documents read). One resolves when
the article it names is a Section of the document it points to (relative: the previous / next article in order).

--census prints per document: articles (by numbering run and number); placed / unplaced (with the first 10 unplaced and
why); the _N-suffixed duplicate Section names (an ingestion defect); the heading-only Sections (how many, how many
matched to a unit by level and ordinal, how many named an opening unit) and the opening units; references by pattern and
how many resolve. --dry-run OUTDIR writes units.csv (code, level, unit name, article), refers_to.csv (from article, to
article, pattern, matched text; the resolved ones) and unresolved_sample.csv (20 unresolved references), and prints the
census as well.
"""

import argparse
import csv
import os
import re
import sys
from collections import Counter, OrderedDict, defaultdict
from typing import Any, Dict, List, Optional, Tuple

SIX_CODES = OrderedDict([("c.c.", "Codice Civile"), ("c.p.c.", "Codice di procedura civile"), ("c.p.", "Codice Penale"),
                         ("c.p.p.", "Codice di procedura penale"), ("c.p.a.", "Codice del processo amministrativo"),
                         ("c.d.s.", "Codice della strada")])
LEVELS = ("LIBRO", "PARTE", "TITOLO", "CAPO", "SEZIONE")
UNPLACED_SHOWN = 10
SAMPLE = 20

_ORDINALS = (r"PRIM[OA]|SECOND[OA]|TERZ[OA]|QUART[OA]|QUINT[OA]|SEST[OA]|SETTIM[OA]|OTTAV[OA]|NON[OA]|DECIM[OA]|"
             r"UNIC[OA]|PRELIMINARE")
_NUMBER = rf"(?:[IVXLCDM]+|{_ORDINALS}|\d+)(?:[\s-]*(?:BIS|TER|QUATER|QUINQUIES|SEXIES|bis|ter|quater))?"
MARKER_RE = re.compile(rf"(?<![\w’'])(?P<level>{'|'.join(LEVELS)})\s+(?P<number>{_NUMBER})\b\s*(?:[–—\-:.]\s*)?")
# v60h: the same markers in title case («Titolo I», «Capo II», «Libro primo»), for a document with none in capitals
_ORDINALS_T = (r"(?:[Pp]rim|[Ss]econd|[Tt]erz|[Qq]uart|[Qq]uint|[Ss]est|[Ss]ettim|[Oo]ttav|[Nn]on|[Dd]ecim)[oa]|[Uu]nic[oa]|"
               r"[Pp]reliminare")
_NUMBER_T = rf"(?:[IVXLCDM]+|{_ORDINALS_T}|\d+)(?:[\s-]*(?:bis|ter|quater|quinquies|sexies|BIS|TER|QUATER))?"
MARKER_TITLE_RE = re.compile(rf"(?<![\w’'])(?P<level>Libro|Parte|Titolo|Capo|Sezione)\s+(?P<number>{_NUMBER_T})\b"
                             rf"\s*(?:[–—\-:.]\s*)?")
# a title-case marker after one of these is a reference to a unit («di cui al Capo I», «del Titolo II»), no heading
_REFERENCE_BEFORE_RE = re.compile(r"(?:\b(?:del|dal|al|nel|sul|il|lo|la|dello|della|dei|degli|delle|nei|negli|nelle|ai|"
                                  r"agli|alle|dai|dagli|dalle|presente|medesimo|stesso|citato|predetto|suddetto|cui|e|ed|"
                                  r"o|ovvero|di|a|da|in|su|per|con|tra|fra)|[dlnst]?ell['’]|all['’]|dall['’])\s*$",
                                  re.IGNORECASE)
HEADING_GAP = 300               # a title-case marker right after another heading: within this many characters, no sentence end
_NAME_END_RE = re.compile(r"\s(?=(?:Art\.|Articolo)\s*\d)|(?<=[a-zà-ù])\.\s+(?=[A-ZÀ-Ü][a-zà-ù])|\s(?=\d{1,3}\.\s+[A-ZÀ-Ü])")

_SUFFIX = r"(?:[\s-]*(?:bis|ter|quater|quinquies|sexies|septies|octies|novies|decies)\b)?"
_NUM = rf"\d+{_SUFFIX}"
REFERENCE_RE = re.compile(
    rf"(?P<list>\bartt?(?:icoli|\.)\s+{_NUM}(?:\s*(?:,|e|ed|o)\s*{_NUM})+)"
    rf"|(?P<relative>\barticol[oi]\s+(?:precedent[ei]|seguent[ei]|successiv[oi]|anteriore))"
    rf"|(?P<word>\barticolo\s+{_NUM})"
    rf"|(?P<abbr>\bart\.\s*{_NUM})", re.IGNORECASE)
_NUMBERS_RE = re.compile(_NUM, re.IGNORECASE)
_THIS_CODE_RE = re.compile(r"^[^.;]{0,40}?\bdel(?:l[ao])?\s+presente\s+codice\b", re.IGNORECASE)
# A reference followed by another code's or act's name points there (what follows within 60 characters, before a stop)
_OTHER_CODES = [("c.p.c.", r"codice di procedura civile|c\.\s?p\.\s?c\.|cod\.\s*proc\.\s*civ"),
                ("c.p.p.", r"codice di procedura penale|c\.\s?p\.\s?p\.|cod\.\s*proc\.\s*pen"),
                ("c.p.a.", r"codice del processo amministrativo|c\.\s?p\.\s?a\."),
                ("c.d.s.", r"codice della strada|c\.\s?d\.\s?s\."),
                ("c.c.", r"codice civile|c\.\s?c\.(?!\w)|cod\.\s*civ"),
                ("c.p.", r"codice penale|c\.\s?p\.(?![\w.])|cod\.\s*pen")]
_OTHER_ACT_RE = re.compile(r"\b(?:legge|l\.\s*n?\.?\s*\d|d\.\s?lgs|decreto|d\.\s?p\.\s?r|d\.\s?l\.|r\.\s?d\.|t\.\s?u\.|testo unico|"
                           r"regolamento|disp(?:osizioni)?\.?\s+(?:att|di attuazione)|costituzione|trattato|direttiva)",
                           re.IGNORECASE)
# v53 (2044 c.c.: "articolo 52, commi secondo, terzo e quarto, del codice penale" read as 52 c.c.): the comma clause between
# an article's number and its code or act -- "comma 1", "commi secondo, terzo e quarto", "secondo comma", "lettera b)",
# "n. 3" -- skipped token by token; only a run holding one of its keywords counts as a clause
_ORDINAL_WORDS = (r"(?:prim|second|terz|quart|quint|sest|settim|ottav|non|decim|undicesim|dodicesim|ultim|penultim)[oia]"
                  r"(?:[\s-]*(?:bis|ter|quater))?")
_CLAUSE_KEY = r"comma|commi|co\.|lett(?:era|ere|\.)|periodo|periodi|alinea|numer[oi]|nn?\."
_CLAUSE_TOKEN_RE = re.compile(
    rf"\s*(?:(?P<key>(?:{_CLAUSE_KEY})(?!\w))|,|\b(?:e|ed)\b|\b{_ORDINAL_WORDS}\b|\b\d+(?:[\s-]*(?:bis|ter|quater|quinquies|"
    rf"sexies))?\b|\b[a-z]\)|\b(?:del|al|nel)\s+(?=(?:comma|commi)\b))", re.IGNORECASE)
_PREPOSITION_RE = re.compile(r"\s*,?\s*(?:(?:del(?:l[aoe'’]|gli|lo)?|dei|delle)\s*)?(?:(?:citat|predett|suddett|richiamat)\w*\s+)?",
                             re.IGNORECASE)
_SAME_CODE_RE = re.compile(r"(?:presente|medesim\w*|stess\w*)\s+codice\b", re.IGNORECASE)
RESOLUTIONS = ("explicit_code", "same_code", "relative")

QUERY_DOCS = "MATCH (d:Document) WHERE d.document_type = 'primary' RETURN d.id AS id, d.name AS name ORDER BY d.name"
QUERY_DOCS_TYPES = ("MATCH (d:Document) WHERE d.document_type IN $types RETURN d.id AS id, d.name AS name, "
                    "d.document_type AS type ORDER BY d.name")
QUERY_DOC_IDS = ("MATCH (d:Document) WHERE d.id IN $ids RETURN d.id AS id, d.name AS name, d.document_type AS type "
                 "ORDER BY d.name")
# v60h: the documents of the types given whose article Sections no Unit holds (read-only)
QUERY_UNITLESS = ("MATCH (d:Document)-[:CONTAINS]->(s:Section) WHERE d.document_type IN $types "
                  "AND s.titolo_articolo IS NOT NULL OPTIONAL MATCH (u:Unit)-[:HAS_ARTICLE]->(s) "
                  "WITH d, count(DISTINCT s) AS articles, count(DISTINCT u) AS units WHERE units = 0 "
                  "RETURN d.id AS id, d.name AS name, d.document_type AS type, articles ORDER BY articles DESC, d.name")
QUERY_SECTIONS = ("MATCH (d:Document {id: $doc})-[:CONTAINS]->(s:Section) "
                  "OPTIONAL MATCH (s)-[:NEXT]->(n:Section)<-[:CONTAINS]-(d) "
                  "RETURN elementId(s) AS id, s.name AS name, s.plain_text AS text, elementId(n) AS next")


# --- documents --------------------------------------------------------------------------------------------------------

def code_of(name: str) -> str:
    """A document's code label: one of the six codes', else its own name."""
    for code, prefix in SIX_CODES.items():
        if (name or "").lower().startswith(prefix.lower()):
            return code
    return name or "?"


def ordered(docs: List[Dict[str, str]], only: List[str] = ()) -> List[Dict[str, str]]:
    """The six codes first (in SIX_CODES order), then the others by name; --only prefixes when given."""
    if only:
        docs = [d for d in docs if any((d["name"] or "").lower().startswith(o.lower()) for o in only)]
    rank = {code: i for i, code in enumerate(SIX_CODES)}
    return sorted(docs, key=lambda d: (rank.get(code_of(d["name"]), len(rank)), d["name"] or ""))


SUFFIXES = ("bis", "ter", "quater", "quinquies", "sexies", "septies", "octies", "novies", "nonies", "decies", "undecies",
            "duodecies", "terdecies", "quaterdecies", "quinquiesdecies", "sexiesdecies", "septiesdecies", "octiesdecies")
_SECTION_NAME_RE = re.compile(r"^(?:(?P<run>att|prel)-)?(?P<num>\d+)(?:[\s_-]?(?P<suf>[a-z]+))?(?P<rest>.*)$", re.IGNORECASE)
# the numbering runs, by Section-name prefix: (run index, name, its place in the document's order)
RUNS = {"": 0, "prel": 1, "att": 2}
RUN_NAMES = {0: "", 1: "disposizioni sulla legge in generale", 2: "disposizioni di attuazione"}
RUN_ORDER = {1: 0, 0: 1, 2: 2}


def parse_name(section_name: str) -> Optional[Dict[str, Any]]:
    """"att-38_1" -> {"run": 2, "article": "38", "number": 38, "suffix": 0, "fragment": None, "duplicate": 1};
    "1159-bis.2.0" -> {"run": 0, "article": "1159-bis", "number": 1159, "suffix": 1, "fragment": (2, 0), ...}; None
    without an article number."""
    m = _SECTION_NAME_RE.match((section_name or "").strip())
    if not m:
        return None
    suf, rest = (m.group("suf") or "").lower(), m.group("rest") or ""
    if suf and suf not in SUFFIXES:              # "38abc": no suffix, the letters are the rest
        rest, suf = (m.group(0)[m.end("num"):]), ""
    dup = re.search(r"_(\d+)$", rest)
    frag = re.match(r"\.(\d+(?:\.\d+)*)", rest)
    return {"run": RUNS[(m.group("run") or "").lower()], "article": m.group("num") + (f"-{suf}" if suf else ""),
            "number": int(m.group("num")), "suffix": SUFFIXES.index(suf) + 1 if suf else 0,
            "fragment": tuple(int(x) for x in frag.group(1).split(".")) if frag else None,
            "duplicate": int(dup.group(1)) if dup else 0}


def article_of(section_name: str) -> Optional[str]:
    """The article number of a Section name, without its run prefix ("att-38_1" -> "38", "1159bis.1.0" -> "1159-bis")."""
    parsed = parse_name(section_name)
    return parsed["article"] if parsed else None


def chain(sections: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """(the Sections in NEXT order from the document's first, those off the chain). The first: a Section no other's
    NEXT points to; with several, the longest chain wins."""
    by_id = {s["id"]: s for s in sections}
    pointed = {s["next"] for s in sections if s.get("next")}
    best: List[Dict[str, Any]] = []
    for head in (s for s in sections if s["id"] not in pointed):
        walk, seen, cur = [], set(), head
        while cur is not None and cur["id"] not in seen:
            walk.append(cur)
            seen.add(cur["id"])
            cur = by_id.get(cur.get("next"))
        if len(walk) > len(best):
            best = walk
    on = {s["id"] for s in best}
    return best, [s for s in sections if s["id"] not in on]


# --- the heading markers ----------------------------------------------------------------------------------------------

def _caps(word: str) -> bool:
    letters = [c for c in word if c.isalpha()]
    return not letters or all(c.isupper() for c in letters)


def _title_markers(text: str) -> list:
    """The title-case markers that stand where a heading does (v60h): at the text's start, after a sentence's end, or
    within HEADING_GAP characters of the heading before with no sentence end between; never after a preposition or an
    article (a reference to a unit)."""
    out = []
    for m in MARKER_TITLE_RE.finditer(text or ""):
        before = text[:m.start()]
        if _REFERENCE_BEFORE_RE.search(before[-24:]):
            continue
        tail = before.rstrip()
        near = bool(out) and m.start() - out[-1].start() <= HEADING_GAP and not re.search(
            r"[.;]\s+[A-ZÀ-Ü][a-zà-ù]", text[out[-1].end():m.start()])
        if not tail or tail[-1] in ".:;)»\"”" or "\n" in before[len(tail):] or near:
            out.append(m)
    return out


def marker_style(texts) -> str:
    """«capitals» when the texts hold a marker in capitals, else «title case» when they hold one so, else «none»."""
    texts = [t for t in texts if t]
    if any(MARKER_RE.search(t) for t in texts):
        return "capitals"
    return "title case" if any(_title_markers(t) for t in texts) else "none"


def markers(text: str, title: bool = False) -> List[Dict[str, Any]]:
    """[{"level", "number", "name", "start", "lead"}] of the heading markers in a Section's text, in order; "lead": only
    blanks, other markers or a heading in capitals ("DISPOSIZIONI SULLA LEGGE IN GENERALE CAPO I") precede it (it opens
    its own article's unit). title (v60h): the markers in title case (_title_markers)."""
    text = text or ""
    found = _title_markers(text) if title else list(MARKER_RE.finditer(text))
    out = []
    for i, m in enumerate(found):
        end = found[i + 1].start() if i + 1 < len(found) else len(text)
        name = text[m.end():end].strip()
        cut = _NAME_END_RE.search(name)
        if cut:
            name = name[:cut.start()]
        words = name.split()
        if words and _caps(words[0]):           # a name in capitals ends where they do ("CAPO I – DEL GIUDICE La ...")
            k = next((i for i, w in enumerate(words) if not _caps(w)), len(words))
            name = " ".join(words[:k])
        name = " ".join(name.split()).strip(" –—-:.")[:200]
        before = text[:m.start()]
        lead = (not before.strip() or (out and out[-1]["lead"] and not text[out[-1]["end"]:m.start()].strip())
                or (not out and _caps(before) and any(c.isalpha() for c in before)))     # after a heading in capitals
        out.append({"level": m.group("level").upper(), "number": " ".join(m.group("number").split()).upper(),
                    "name": name, "start": m.start(), "end": m.start() + len(m.group(0)) + len(name), "lead": bool(lead)})
    return out


def arrange(sections: List[Dict[str, Any]]
            ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[int], Dict[int, str]]:
    """(the walk, the Sections left out of it, the run of each Section of the walk, {run: its name}). The walk: the
    numbered Sections ordered by run (prel- first, the code, att- last), article number, its bis/ter/... suffix, the
    whole-article node before its fragments (4, then 4.0.0 ... 4.4.0), a _N duplicate after its original; the NEXT
    chain's order only breaks ties. A Section with no article number on the chain goes right after the numbered Section
    before it on the chain (its markers stay where the loader put them); off the chain, it is left out. The runs are the
    name prefixes (att- disposizioni di attuazione, prel- disposizioni sulla legge in generale): each its own unit tree,
    never merged with the code."""
    on_chain, _ = chain(sections)
    position = {sec["id"]: i for i, sec in enumerate(on_chain)}
    numbered = [(parse_name(sec.get("name")), sec) for sec in sections]
    walk_keyed = sorted(((p, sec) for p, sec in numbered if p),
                        key=lambda ps: (RUN_ORDER[ps[0]["run"]], ps[0]["number"], ps[0]["suffix"],
                                        ps[0]["fragment"] is not None, ps[0]["fragment"] or (), ps[0]["duplicate"],
                                        position.get(ps[1]["id"], len(position)), str(ps[1]["id"])))
    walk = [sec for _, sec in walk_keyed]
    runs = [p["run"] for p, _ in walk_keyed]
    after: Dict[Optional[str], List[Dict[str, Any]]] = {}
    last_numbered: Optional[str] = None
    for sec in on_chain:
        if parse_name(sec.get("name")):
            last_numbered = sec["id"]
        else:
            after.setdefault(last_numbered, []).append(sec)
    out, out_runs = list(after.get(None, [])), [runs[0] if runs else 0] * len(after.get(None, []))
    for sec, run in zip(walk, runs):
        out.append(sec)
        out_runs.append(run)
        for extra in after.get(sec["id"], []):
            out.append(extra)
            out_runs.append(run)
    placed = {sec["id"] for sec in out}
    return out, [sec for sec in sections if sec["id"] not in placed], out_runs, dict(RUN_NAMES)


def duplicates(sections: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """[(name, its _N duplicate)] of a document's Section names (att-38 / att-38_1): an ingestion defect."""
    names = {str(sec.get("name") or "") for sec in sections}
    pairs = [(re.sub(r"_\d+$", "", n), n) for n in names if re.search(r"_\d+$", n)]
    return sorted((pair for pair in pairs if pair[0] in names),
                  key=lambda pair: (RUN_ORDER[(parse_name(pair[0]) or {"run": 0})["run"]],
                                    (parse_name(pair[0]) or {}).get("number", 0), pair))


# --- the opening units: articles before the first marker in the text ---------------------------------------------------

_ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
_WORDS = ("PRIM", "SECOND", "TERZ", "QUART", "QUINT", "SEST", "SETTIM", "OTTAV", "NON", "DECIM")
HEADING_ONLY_CHARS = 400


def ordinal(number: str) -> Optional[int]:
    """"II" 2, "SECONDO" 2, "UNICO" 1, "3" 3; None for PRELIMINARE or a number with bis / ter."""
    n = (number or "").strip().upper()
    if not n or re.search(r"[\s-]", n):
        return None
    if n.isdigit():
        return int(n)
    if n[:-1] in _WORDS and n[-1] in "OA":
        return _WORDS.index(n[:-1]) + 1
    if n in ("UNICO", "UNICA"):
        return 1
    if all(c in _ROMAN for c in n):
        total = 0
        for i, c in enumerate(n):
            v = _ROMAN[c]
            total += -v if i + 1 < len(n) and _ROMAN[n[i + 1]] > v else v
        return total
    return None


def ordinal_text(n: int, like: str) -> str:
    """n written as `like` is (a word ordinal with its ending, else roman)."""
    like = (like or "").upper()
    if like[:-1] in _WORDS and like[-1:] in ("O", "A") and 1 <= n <= len(_WORDS):
        return _WORDS[n - 1] + like[-1]
    out = ""
    for value, digits in ((1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"), (50, "L"),
                          (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")):
        while n >= value:
            out, n = out + digits, n - value
    return out


def heading_only(sections: List[Dict[str, Any]], title: bool = False) -> List[Dict[str, Any]]:
    """[{"id", "level", "number", "ordinal", "unit"}] of the Sections with no article number whose text is a marker line
    (it starts with a marker, at most HEADING_ONLY_CHARS characters), one entry per marker, in the order given."""
    out = []
    for s in sections:
        text = " ".join(str(s.get("text") or "").split())
        if article_of(s.get("name")) is not None or not text or len(text) > HEADING_ONLY_CHARS:
            continue
        found = markers(text, title)
        if not found or not found[0]["lead"]:
            continue
        for m in found:
            out.append({"id": s["id"], "level": m["level"], "number": m["number"], "ordinal": ordinal(m["number"]),
                        "unit": f"{m['level']} {m['number']}" + (f" – {m['name']}" if m["name"] else "")})
    return out


def opening_units(walk: List[Dict[str, Any]], runs: List[int], run: int, headings: List[Dict[str, Any]],
                  title: bool = False) -> Tuple[Dict[str, str], List[str]]:
    """({level: unit name}, [ids of the heading-only Sections used]) of a run's opening units: the units its articles
    before the first marker in the text belong to, one below the first marker seen at each level (SEZIONE II first:
    SEZIONE I) -- a level counts only if no marker above it came first. Named after a heading-only Section of the
    same level and ordinal, else "SEZIONE I (intestazione non nel testo)"."""
    first: Dict[str, Dict[str, Any]] = {}
    closed: set = set()
    for section, r in zip(walk, runs):
        if r != run:
            continue
        for m in markers(section.get("text") or "", title):
            if m["level"] not in first and m["level"] not in closed:
                first[m["level"]] = m
            closed |= {lower for lower in LEVELS[LEVELS.index(m["level"]) + 1:] if lower not in first}
    out, used = {}, []
    for level in LEVELS:
        m = first.get(level)
        n = ordinal(m["number"]) if m else None
        if not n or n < 2:
            continue
        match = next((h for h in headings if h["level"] == level and h["ordinal"] == n - 1), None)
        if match:
            out[level] = match["unit"]
            used.append(match["id"])
        else:
            out[level] = f"{level} {ordinal_text(n - 1, m['number'])} (intestazione non nel testo)"
    return out, used


def label_of(article: str, run: int, names: Dict[int, str]) -> str:
    """An article's label within its document: its number, with its run's name past the first ("2 [disposizioni di
    attuazione]")."""
    return article if not run else f"{article} [{names[run]}]"


def place(walk: List[Dict[str, Any]], runs: Optional[List[int]] = None, names: Optional[Dict[int, str]] = None,
          headings: Optional[List[Dict[str, Any]]] = None, title: bool = False
          ) -> Tuple[Dict[str, Dict[str, str]], List[Tuple[str, str]]]:
    """({article label: {level: "CAPO V – DELL'ESERCIZIO ..."}}, [(article label, why unplaced)]) along the chain; each
    numbering run its own tree, starting in its opening units (opening_units; headings: heading_only's entries)."""
    runs = runs if runs is not None else [0] * len(walk)
    names = names or {0: ""}
    heads = headings or []
    head_ids = {h["id"] for h in heads}
    current: Dict[str, str] = {}
    pending: List[Dict[str, Any]] = []
    units: Dict[str, Dict[str, str]] = {}
    unplaced: List[Tuple[str, str]] = []
    run_now = None

    def apply(marker):
        level = marker["level"]
        for lower in LEVELS[LEVELS.index(level):]:
            current.pop(lower, None)
        current[level] = f"{level} {marker['number']}" + (f" – {marker['name']}" if marker["name"] else "")
    for section, run in zip(walk, runs):
        if run != run_now:          # a numbering: its own tree, from its opening units
            run_now, current = run, dict(opening_units(walk, runs, run, heads, title)[0])
        found = markers(section.get("text") or "", title)
        for m in pending:                        # opened at the end of the article before
            apply(m)
        pending = []
        for m in found:
            if m["lead"]:
                apply(m)
            else:
                pending.append(m)
        article = article_of(section.get("name"))
        if article is None:
            unplaced.append((section.get("name") or "?", "heading-only Section" if section["id"] in head_ids
                             else "Section name with no article number"))
            continue
        label = label_of(article, run, names)
        if label in units:
            continue                             # a later piece of the same article
        if current:
            units[label] = dict(current)         # a later piece placed: the article is (its first piece was not)
            unplaced = [(a, w) for a, w in unplaced if a != label]
        elif not any(a == label for a, _ in unplaced):
            unplaced.append((label, "before the first heading marker"))
    return units, unplaced


# --- the cross-references ---------------------------------------------------------------------------------------------

def skip_clause(after: str) -> str:
    """after without the comma clause it opens with ("commi secondo, terzo e quarto, del codice penale" -> "del codice
    penale"): its tokens consumed while they are commas, "e", ordinals, numbers, letters "b)" or the clause's keywords
    (_CLAUSE_KEY); none of the keywords among them, nothing is skipped."""
    pos, keyed = 0, False
    while True:
        m = _CLAUSE_TOKEN_RE.match(after, pos)
        if not m or m.end() == pos:
            break
        keyed = keyed or bool(m.group("key"))
        pos = m.end()
    return after[pos:] if keyed else after


def qualifier(after: str, own: str) -> Tuple[Optional[str], str]:
    """(the code or act a reference names right after its number, how): its comma clause and the line breaks skipped
    (skip_clause), then a code -- (None, "explicit_code") when it is the reference's own, as "del presente codice" --, an
    act ("other act": not one of the codes read, unresolved, never the reference's own code), or (None, "same_code")
    when nothing names one."""
    text = skip_clause(" ".join((after or "")[:300].split()))
    lead = _PREPOSITION_RE.match(text)
    rest = text[lead.end():] if lead else text
    for code, pattern in _OTHER_CODES:
        if re.match(rf"(?:{pattern})", rest, re.IGNORECASE):
            return (None if code == own else code), "explicit_code"
    if _SAME_CODE_RE.match(rest):
        return None, "explicit_code"
    if _OTHER_ACT_RE.match(rest):
        return "other act", "explicit_code"
    return None, "same_code"


def _other_code(after: str, own: str) -> Optional[str]:
    """The code or act a reference names right after it (qualifier): a code label, "other act", or None (its own)."""
    return qualifier(after, own)[0]


def _heading_at(text: str, m) -> bool:
    """A capitalised "Art. N" / "Articolo N" at the start of the text, after a sentence or clause end, or after a heading
    in capitals: an article's heading, not a reference."""
    if not m.group(0)[:1].isupper() or m.lastgroup in ("list", "relative"):
        return False
    before = text[:m.start()].rstrip()
    if not before or before[-1] in ".:;–—-)":
        return True
    tail = re.findall(r"[^\W\d_]+", before[-40:])
    return len(tail) >= 2 and all(w.isupper() for w in tail[-2:])


def references(text: str, own: str) -> List[Dict[str, Any]]:
    """[{"pattern", "matched", "numbers", "to_code", "this_code"}] of the cross-references in an article's text."""
    out = []
    for m in REFERENCE_RE.finditer(text or ""):
        kind = m.lastgroup
        if _heading_at(text, m):
            continue                    # an article's own heading the loader left in the text ("... TITOLO I Art. 1173")
        after = text[m.end():]
        to, how = qualifier(after, own)
        entry = {"pattern": "other code" if to else kind, "kind": kind, "matched": " ".join(m.group(0).split()),
                 "resolution": "relative" if kind == "relative" else how,
                 "numbers": [] if kind == "relative" else [" ".join(n.split()).lower().replace(" ", "-")
                                                          for n in _NUMBERS_RE.findall(m.group(0))],
                 "to_code": to or own, "this_code": bool(_THIS_CODE_RE.match(after))}
        if kind == "relative":
            entry["direction"] = -1 if re.search(r"precedent|anterior", m.group(0), re.IGNORECASE) else 1
            count = re.search(r"\b(due|tre|quattro|cinque)\s+$", text[max(0, m.start() - 12):m.start()], re.IGNORECASE)
            entry["count"] = _COUNTS[count.group(1).lower()] if count else 1      # "dai due articoli precedenti"
            if count:
                entry["matched"] = f"{count.group(1)} {entry['matched']}"
        out.append(entry)
    return out


_COUNTS = {"due": 2, "tre": 3, "quattro": 4, "cinque": 5}


def _norm(number: str) -> str:
    return re.sub(r"[\s-]+", "-", number.lower())


# --- one document -----------------------------------------------------------------------------------------------------

def read_document(session, doc: Dict[str, str]) -> Dict[str, Any]:
    sections = session.run(QUERY_SECTIONS, doc=doc["id"]).data()
    walk, off, runs, names = arrange(sections)
    return {"doc": doc, "code": code_of(doc["name"]), "sections": sections, "walk": walk, "off": off,
            "runs": runs, "run_names": names, "markers": marker_style(s.get("text") for s in sections)}


def analyse(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per document: units and unplaced by article label, the Sections of each article (element ids, every piece),
    references (with "resolved", "to_article" and the Sections they join) -- every document read first, so a reference
    to another code read here resolves against it. A Section is known by its element id only; an article by its
    numbering run and number. A reference resolves in its own run first, then in the document's own numbering."""
    index: Dict[str, Dict[Tuple[int, str], List[str]]] = {}
    for d in docs:
        pieces: Dict[Tuple[int, str], List[str]] = {}
        runs = d.get("runs") or [0] * len(d["walk"])
        for s, run in list(zip(d["walk"], runs)) + [(s, 0) for s in d["off"]]:
            a = article_of(s["name"])
            if a:
                pieces.setdefault((run, _norm(a)), []).append(s["id"])
        index[d["code"]] = pieces
    out = []
    for d in docs:
        runs = d.get("runs") or [0] * len(d["walk"])
        names = d.get("run_names") or {0: ""}
        title = d.get("markers") == "title case"         # v60h: a document whose headings are in title case
        heads = heading_only(list(d["walk"]) + list(d["off"]), title)
        units, unplaced = place(d["walk"], runs, names, heads, title)
        opening = {r: opening_units(d["walk"], runs, r, heads, title) for r in sorted(set(runs))}
        used = {i for _, ids in opening.values() for i in ids}
        placed_levels = {(level, ordinal(name.split()[1]) if len(name.split()) > 1 else None)
                         for u in units.values() for level, name in u.items()}
        heading_census = {"sections": len({h["id"] for h in heads}),
                          "matched": len({h["id"] for h in heads if (h["level"], h["ordinal"]) in placed_levels}),
                          "named_opening": len(used),
                          "opening": {label_of("", r, names).strip() or "own numbering": o for r, (o, _) in opening.items()
                                      if o}}
        placed_or_not = set(units) | {a for a, _ in unplaced}
        head_ids = {h["id"] for h in heads}
        unplaced += [(article_of(s["name"]) or s["name"] or "?",
                      "heading-only Section" if s["id"] in head_ids else
                      "Section name with no article number (off the NEXT chain)") for s in d["off"]
                     if (article_of(s["name"]) or s["name"]) not in placed_or_not]
        order: Dict[int, List[str]] = {}
        for s, run in zip(d["walk"], runs):
            a = article_of(s["name"])
            if a and (not order.get(run) or order[run][-1] != a) and a not in order.get(run, []):
                order.setdefault(run, []).append(a)
        refs = []
        for s, run in list(zip(d["walk"], runs)) + [(s, 0) for s in d["off"]]:
            article = article_of(s["name"])
            if article is None:
                continue
            for r in references(s.get("text") or "", d["code"]):
                if r["kind"] != "relative" and r["to_code"] == d["code"] and [_norm(n) for n in r["numbers"]] == [_norm(article)]:
                    continue                    # the article's own heading ("Art. 1051 – ..."), no reference
                to_run = run
                if r["kind"] == "relative":
                    seq = order.get(run, [])
                    at = seq.index(article) if article in seq else None
                    steps = range(1, r.get("count", 1) + 1)
                    targets = [seq[at + r["direction"] * k] for k in steps
                               if at is not None and 0 <= at + r["direction"] * k < len(seq)]
                    resolved = len(targets) == r.get("count", 1)
                elif r["to_code"] == "other act":
                    targets, resolved = r["numbers"], False
                else:
                    have = index.get(r["to_code"])
                    targets = r["numbers"]
                    if r["to_code"] != d["code"]:
                        to_run = 0
                    elif have is not None and not all((run, _norm(n)) in have for n in targets):
                        to_run = 0                  # not in its own run: the document's own numbering
                    resolved = have is not None and bool(targets) and all((to_run, _norm(n)) in have for n in targets)
                for t in targets or [""]:
                    to_ids = (index.get(r["to_code"]) or {}).get((to_run, _norm(t)), []) if resolved and t else []
                    refs.append({**r, "from": article, "from_label": label_of(article, run, names), "from_id": s["id"],
                                 "to_article": t, "to_label": label_of(t, to_run, names) if r["to_code"] == d["code"] else t,
                                 "to_id": to_ids[0] if to_ids else None, "resolved": resolved})
        articles = {label_of(a, run, names): ids for (run, a), ids in ((k, v) for k, v in index[d["code"]].items())}
        out.append({"doc": d["doc"], "code": d["code"], "articles": len(index[d["code"]]), "units": units,
                    "unplaced": unplaced, "references": refs, "pieces": articles, "run_names": names,
                    "headings": heading_census, "duplicates": duplicates(d["sections"]),
                    "markers": d.get("markers") or "capitals"})
    return out


# --- output -----------------------------------------------------------------------------------------------------------

def census_lines(result: Dict[str, Any]) -> List[str]:
    refs = result["references"]
    by_pattern = Counter(r["pattern"] for r in refs)
    resolved = Counter(r["pattern"] for r in refs if r["resolved"])
    why = Counter(w for _, w in result["unplaced"])
    levels = Counter(level for u in result["units"].values() for level in u)
    lines = [f"## {result['doc']['name']} ({result['code']}, {result['doc'].get('id', '')})", "",
             f"- markers: {result.get('markers') or 'capitals'}",
             f"- articles: {result['articles']}",
             f"- placed in a Libro/Parte/Titolo/Capo/Sezione path: {len(result['units'])}"
             + (f" (articles with a unit at each level: " + ", ".join(f"{l} {levels[l]}" for l in LEVELS if levels[l]) + ")"
                if levels else ""),
             f"- not placed: {len(result['unplaced'])}" + (" (" + "; ".join(f"{w}: {n}" for w, n in why.most_common()) + ")"
                                                          if why else "")]
    dups = result.get("duplicates") or []
    lines.append(f"- ingestion defect: Section names duplicated with a _N suffix: {len(dups)}"
                 + (" (" + ", ".join(f"{a} / {b}" for a, b in dups[:UNPLACED_SHOWN])
                    + (f", +{len(dups) - UNPLACED_SHOWN}" if len(dups) > UNPLACED_SHOWN else "") + ")" if dups else ""))
    heads = result.get("headings") or {}
    lines.append(f"- heading-only Sections (no article number, a marker line): {heads.get('sections', 0)}, matched to "
                 f"a unit (level and ordinal): {heads.get('matched', 0)}, naming an opening unit: "
                 f"{heads.get('named_opening', 0)}")
    for run, units in (heads.get("opening") or {}).items():
        lines.append(f"  opening units ({run}): " + " > ".join(units[level] for level in LEVELS if level in units))
    if result["unplaced"]:
        lines.append("  first unplaced: " + ", ".join(f"{a} ({w})" for a, w in result["unplaced"][:UNPLACED_SHOWN]))
    lines += [f"- cross-references: {len(refs)}, resolving to an existing Section: {sum(resolved.values())}",
              "", "| pattern | references | resolved |", "|---|---:|---:|"]
    for pattern in ("list", "word", "abbr", "relative", "other code"):
        if by_pattern[pattern]:
            lines.append(f"| {pattern} | {by_pattern[pattern]} | {resolved[pattern]} |")
    this_code = [r for r in refs if r["this_code"]]
    lines.append(f"| of them «del presente codice» | {len(this_code)} | {sum(1 for r in this_code if r['resolved'])} |")
    others = Counter(r["to_code"] for r in refs if r["pattern"] == "other code")
    if others:
        lines.append("")
        lines.append("other codes or acts named: " + ", ".join(f"{c} {n}" for c, n in others.most_common()))
    return lines + [""]


def rows_census(rows: Dict[str, List[Dict[str, Any]]]) -> List[str]:
    """v60h: what --write would send for these results -- units by level, CHILD_OF, HAS_ARTICLE, REFERS_TO by
    resolution."""
    levels = Counter(u["level"] for u in rows["units"])
    how = Counter(r["resolution"] for r in rows["refers_to"])
    return [f"- graph rows: units {len(rows['units'])} ("
            + ", ".join(f"{level} {levels[level]}" for level in LEVELS if levels[level]) + ")"
            + f"; CHILD_OF {len(rows['child_of'])}; HAS_ARTICLE {len(rows['has_article'])}; REFERS_TO "
            f"{len(rows['refers_to'])} (" + ", ".join(f"{k} {how.get(k, 0)}" for k in RESOLUTIONS) + ")", ""]


def art_code(label: str, code: str) -> str:
    """"2 c.p.c." / "2 c.p.c. [disposizioni di attuazione]" from an article label and its code."""
    m = re.match(r"^(\S+) \[(.+)\]$", label or "")
    return f"{m.group(1)} {code} [{m.group(2)}]" if m else f"{label} {code}".strip()


def _run_of(label: str) -> str:
    m = re.match(r"^\S+ \[(.+)\]$", label or "")
    return m.group(1) if m else ""


def write_csvs(results: List[Dict[str, Any]], outdir: str) -> Dict[str, str]:
    os.makedirs(outdir, exist_ok=True)
    paths = {name: os.path.join(outdir, name) for name in ("units.csv", "refers_to.csv", "unresolved_sample.csv")}
    with open(paths["units.csv"], "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["code", "level", "unit name", "article"])
        for r in results:
            for article, units in r["units"].items():
                for level in LEVELS:
                    if level in units:
                        w.writerow([r["code"], level, units[level], article])
    unresolved = []
    with open(paths["refers_to.csv"], "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["from article", "to article", "pattern", "matched text"])
        for r in results:
            for ref in r["references"]:
                if ref["resolved"]:
                    w.writerow([art_code(ref.get("from_label", ref["from"]), r["code"]),
                                art_code(ref.get("to_label", ref["to_article"]), ref["to_code"]), ref["pattern"],
                                ref["matched"]])
                else:
                    unresolved.append((r["code"], ref))
    with open(paths["unresolved_sample.csv"], "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["from article", "to article", "pattern", "matched text"])
        step = max(1, len(unresolved) // SAMPLE)
        for code, ref in unresolved[::step][:SAMPLE]:
            w.writerow([art_code(ref.get("from_label", ref["from"]), code),
                        f"{ref['to_article']} {ref['to_code']}".strip(), ref["pattern"], ref["matched"]])
    return paths


# --- --write: the graph -----------------------------------------------------------------------------------------------

BATCH = 500
SOURCE = "article_links"
UNIT_INDEX = "CREATE INDEX unit_key IF NOT EXISTS FOR (u:Unit) ON (u.code, u.level, u.path)"
# v60h: additive -- what is created is stamped (source, computed_at); what exists is left as it is (a REFERS_TO edge's
# resolution set only where it has none)
WRITE_UNITS = ("UNWIND $rows AS r MERGE (u:Unit {code: r.code, level: r.level, path: r.path}) "
               "ON CREATE SET u.name = r.name, u.source = $source, u.computed_at = $computed_at")
WRITE_CHILD_OF = ("UNWIND $rows AS r MATCH (c:Unit {code: r.code, level: r.level, path: r.path}) "
                  "MATCH (p:Unit {code: r.code, level: r.parent_level, path: r.parent_path}) MERGE (c)-[e:CHILD_OF]->(p) "
                  "ON CREATE SET e.source = $source, e.computed_at = $computed_at")
WRITE_HAS_ARTICLE = ("UNWIND $rows AS r MATCH (u:Unit {code: r.code, level: r.level, path: r.path}) "
                     "MATCH (s:Section) WHERE elementId(s) = r.section MERGE (u)-[e:HAS_ARTICLE]->(s) "
                     "ON CREATE SET e.source = $source, e.computed_at = $computed_at")
WRITE_REFERS_TO = ("UNWIND $rows AS r MATCH (a:Section) WHERE elementId(a) = r.from MATCH (b:Section) WHERE elementId(b) = r.to "
                   "MERGE (a)-[e:REFERS_TO {pattern: r.pattern, text: r.text, source: $source}]->(b) "
                   "ON CREATE SET e.computed_at = $computed_at, e.resolution = r.resolution "
                   "ON MATCH SET e.resolution = coalesce(e.resolution, r.resolution)")
# v53: before the corrected edges are merged, the stale ones go -- an edge of this source from a Section of the documents
# read whose (from, to) the new rows no longer hold. Found read-only (also by --dry-run), deleted by id in batches.
FIND_STALE = ("MATCH (d:Document)-[:CONTAINS]->(a:Section)-[e:REFERS_TO {source: $source}]->(b:Section) WHERE d.id IN $docs "
              "RETURN elementId(e) AS edge, elementId(a) AS from, elementId(b) AS to, e.pattern AS pattern, e.text AS text")
DELETE_STALE = "UNWIND $edges AS id MATCH ()-[e:REFERS_TO {source: $source}]->() WHERE elementId(e) = id DELETE e"


def graph_rows(results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """The rows --write sends, each once: units {code, level, path, name} (path: the unit's names from the top of its
    tree, a second numbering run's name first, " > " between); child_of; has_article (the article's deepest unit to each
    of its Sections); refers_to (Section to Section, by element id: the Section whose text holds the reference to the
    target article's first Section)."""
    units, child, has, refers = {}, {}, {}, {}
    for r in results:
        code = r["code"]
        for label, levels in r["units"].items():
            path, parent = [_run_of(label)] if _run_of(label) else [], None
            for level in LEVELS:
                if level not in levels:
                    continue
                path = path + [levels[level]]
                key = (code, level, " > ".join(path))
                units.setdefault(key, {"code": code, "level": level, "path": key[2], "name": levels[level]})
                if parent:
                    child.setdefault((key, parent), {"code": code, "level": level, "path": key[2],
                                                     "parent_level": parent[1], "parent_path": parent[2]})
                parent = key
            if parent:
                for section in r["pieces"].get(label) or []:
                    has.setdefault((parent, section), {"code": code, "level": parent[1], "path": parent[2],
                                                       "section": section})
        for ref in r["references"]:
            if ref["resolved"] and ref.get("to_id") and ref["from_id"] != ref["to_id"]:
                key = (ref["from_id"], ref["to_id"], ref["pattern"], ref["matched"])
                refers.setdefault(key, {"from": ref["from_id"], "to": ref["to_id"], "pattern": ref["pattern"],
                                        "text": ref["matched"], "resolution": ref.get("resolution", "same_code")})
    return {"units": list(units.values()), "child_of": list(child.values()), "has_article": list(has.values()),
            "refers_to": list(refers.values())}


def write_graph(session, rows: Dict[str, List[Dict[str, Any]]], computed_at: str) -> Dict[str, Dict[str, int]]:
    """MERGE every row, in sequential batches of BATCH (one session, no parallelism): {kind: {"rows", "created",
    "present"}} -- created from the server's counters, present the rows that were there already."""
    session.run(UNIT_INDEX).consume()
    summary = {}
    for kind, query, counter in (("units", WRITE_UNITS, "nodes_created"),
                                 ("child_of", WRITE_CHILD_OF, "relationships_created"),
                                 ("has_article", WRITE_HAS_ARTICLE, "relationships_created"),
                                 ("refers_to", WRITE_REFERS_TO, "relationships_created")):
        items, created = rows[kind], 0
        for start in range(0, len(items), BATCH):
            counters = session.run(query, rows=items[start:start + BATCH], source=SOURCE,
                                   computed_at=computed_at).consume().counters
            created += getattr(counters, counter, 0)
        summary[kind] = {"rows": len(items), "created": created, "present": len(items) - created}
    return summary


def stale_edges(session, doc_ids: List[str], rows: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """The graph's REFERS_TO edges of this source from the documents' Sections whose (from, to) the new rows do not hold
    (read-only)."""
    keep = {(r["from"], r["to"]) for r in rows["refers_to"]}
    found = session.run(FIND_STALE, docs=list(doc_ids), source=SOURCE).data()
    return [e for e in found if (e["from"], e["to"]) not in keep]


def delete_stale(session, edges: List[Dict[str, Any]]) -> int:
    """The stale edges deleted by element id, in sequential batches of BATCH; how many the server deleted."""
    ids, deleted = [e["edge"] for e in edges], 0
    for start in range(0, len(ids), BATCH):
        counters = session.run(DELETE_STALE, edges=ids[start:start + BATCH], source=SOURCE).consume().counters
        deleted += getattr(counters, "relationships_deleted", 0)
    return deleted


def compare(results: List[Dict[str, Any]], old_csv: str, outdir: str) -> Dict[str, Any]:
    """Against a previous dry run's refers_to.csv: per (from article, matched text), the targets that changed and the ones
    no longer resolved; written to changes.csv, {"changed", "unresolved", "new", "samples"}."""
    read = {r["code"] for r in results}
    old: Dict[Tuple[str, str], set] = defaultdict(set)
    with open(old_csv, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):          # only the references from the codes read now (--codes)
            if re.sub(r"\s*\[.*\]$", "", row["from article"]).partition(" ")[2] in read:
                old[(row["from article"], row["matched text"])].add(row["to article"])
    new: Dict[Tuple[str, str], set] = defaultdict(set)
    how: Dict[Tuple[str, str], str] = {}
    for r in results:
        for ref in r["references"]:
            key = (art_code(ref.get("from_label", ref["from"]), r["code"]), ref["matched"])
            how[key] = ref.get("resolution", "")
            if ref["resolved"]:
                new[key].add(art_code(ref.get("to_label", ref["to_article"]), ref["to_code"]))
            else:
                new.setdefault(key, set())
    rows = []
    for key, targets in old.items():
        now = new.get(key, set())
        for target in sorted(targets - now):
            kind = "changed target" if now - targets else "now unresolved"
            rows.append({"kind": kind, "from": key[0], "matched": key[1], "old": target,
                         "new": "; ".join(sorted(now - targets)) or "-", "resolution": how.get(key, "")})
    added = sum(len(t - old.get(k, set())) for k, t in new.items())
    path = os.path.join(outdir, "changes.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["kind", "from article", "matched text", "old target", "new target", "resolution"])
        for row in rows:
            w.writerow([row["kind"], row["from"], row["matched"], row["old"], row["new"], row["resolution"]])
    step = max(1, len(rows) // SAMPLE)
    return {"changed": sum(r["kind"] == "changed target" for r in rows),
            "unresolved": sum(r["kind"] == "now unresolved" for r in rows), "new": added,
            "samples": rows[::step][:SAMPLE], "path": path}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--env", default=os.path.join(os.getcwd(), ".env"))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--census", action="store_true")
    mode.add_argument("--unitless", action="store_true",
                      help="list the normative documents of --types whose articles no Unit holds, and stop (read-only)")
    mode.add_argument("--dry-run", metavar="OUTDIR")
    mode.add_argument("--write", metavar="OUTDIR", help="the dry run's CSVs to OUTDIR, then MERGE them into the graph")
    parser.add_argument("--only", action="append", default=[], metavar="NAME",
                        help="documents whose name starts with NAME (repeatable)")
    parser.add_argument("--codes", default="", help='code labels or name prefixes, comma-separated: "c.c.,c.p.c."')
    parser.add_argument("--doc", action="append", default=[], metavar="ID",
                        help="a document by its store id, whatever its type (repeatable)")
    parser.add_argument("--types", default="",
                        help="document types to read, comma-separated (default: primary; --unitless: primary,special)")
    parser.add_argument("--compare", metavar="OLD_REFERS_CSV",
                        help="with --dry-run: the references whose target changed or that no longer resolve, against a "
                             "previous dry run's refers_to.csv (changes.csv, 20 samples printed)")
    parser.add_argument("--prune", action="store_true",
                        help="with --write: first delete this source's REFERS_TO edges from the documents read whose "
                             "(from, to) the new rows no longer hold, then merge (--dry-run always counts them)")
    args = parser.parse_args()
    from dotenv import load_dotenv
    load_dotenv(args.env)
    uri = os.getenv("NEO4J_URI")
    if not uri:
        sys.exit(f"NEO4J_URI is not set ({args.env})")
    import datetime as dt
    import neo4j
    codes = [c.strip() for c in args.codes.split(",") if c.strip()]
    access = neo4j.WRITE_ACCESS if args.write else neo4j.READ_ACCESS
    with neo4j.GraphDatabase.driver(uri, auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))) as driver:
        with driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j"), default_access_mode=access) as session:
            types = [t.strip() for t in args.types.split(",") if t.strip()]
            if args.unitless:
                print_unitless(unitless(session, types or ["primary", "special"]))
                return
            if args.doc:
                found = session.run(QUERY_DOC_IDS, ids=args.doc).data()
                missing = sorted(set(args.doc) - {d["id"] for d in found})
                if missing:
                    sys.exit(f"no Document with id {', '.join(missing)}")
                docs = ordered(found)
            elif types and types != ["primary"]:
                docs = select_codes(ordered(session.run(QUERY_DOCS_TYPES, types=types).data(), args.only), codes)
            else:
                docs = select_codes(ordered(session.run(QUERY_DOCS).data(), args.only), codes)
            read = [read_document(session, d) for d in docs]
            results = analyse(read)
            print(f"# article_links census: {len(results)} documents"
                  + (" (writing)" if args.write else " (read-only)") + "\n")
            for r in results:
                print("\n".join(census_lines(r) + rows_census(graph_rows([r]))))
            outdir = args.dry_run or args.write
            if outdir:
                paths = write_csvs(results, outdir)
                print("Files: " + ", ".join(os.path.abspath(p) for p in paths.values()))
            rows = graph_rows(results)
            if args.compare and args.dry_run:
                diff = compare(results, args.compare, outdir)
                print(f"\n## Against {args.compare}\n\n- target changed: {diff['changed']}\n- no longer resolved: "
                      f"{diff['unresolved']}\n- new (from, matched text, target): {diff['new']}\n- all in {diff['path']}\n")
                print("| kind | from | matched | old target | new target | resolution |\n|---|---|---|---|---|---|")
                for row in diff["samples"]:
                    print(f"| {row['kind']} | {row['from']} | {row['matched']} | {row['old']} | {row['new']} | "
                          f"{row['resolution']} |")
            stale = stale_edges(session, [d["id"] for d in docs], rows)
            by_how = Counter(r["resolution"] for r in rows["refers_to"])
            print(f"\n## Edges\n\n- new REFERS_TO rows: {len(rows['refers_to'])} ("
                  + ", ".join(f"{k} {by_how.get(k, 0)}" for k in RESOLUTIONS) + ")"
                  + f"\n- stale edges in the graph (source {SOURCE}, (from, to) not in the new rows): {len(stale)}")
            if outdir:
                with open(os.path.join(outdir, "stale_edges.csv"), "w", newline="", encoding="utf-8") as fh:
                    w = csv.writer(fh, lineterminator="\n")
                    w.writerow(["edge", "from", "to", "pattern", "text"])
                    for e in stale:
                        w.writerow([e["edge"], e["from"], e["to"], e["pattern"], e["text"]])
            if args.write and args.prune:
                print(f"- deleted: {delete_stale(session, stale)} stale edges (before the merge)")
            if args.write:
                summary = write_graph(session, rows, dt.datetime.now(dt.timezone.utc).isoformat())
                print("\n## Written (MERGE: created / already present)\n")
                for kind, n in summary.items():
                    print(f"- {kind}: {n['rows']} rows, {n['created']} created, {n['present']} already present")


def unitless(session, types: List[str]) -> List[Dict[str, Any]]:
    """v60h: the normative documents of these types (their name names an act) whose articles no Unit holds, with their
    articles (read-only)."""
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from src.rag.citations import normative_name
    return [d for d in session.run(QUERY_UNITLESS, types=types).data() if normative_name(d.get("name"))]


def print_unitless(docs: List[Dict[str, Any]]) -> None:
    print(f"# Normative documents no Unit holds: {len(docs)} (read-only)\n")
    print("| id | name | type | articles | code |\n|---|---|---|---:|---|")
    for d in docs:
        print(f"| {d['id']} | {d['name']} | {d.get('type') or ''} | {d.get('articles', 0)} | {code_of(d['name'])} |")


def select_codes(docs: List[Dict[str, str]], codes: List[str]) -> List[Dict[str, str]]:
    """--codes: the documents whose code label is one of them or whose name starts with one."""
    if not codes:
        return docs
    low = [c.lower() for c in codes]
    return [d for d in docs if code_of(d["name"]).lower() in low or any((d["name"] or "").lower().startswith(c) for c in low)]


if __name__ == "__main__":
    main()
