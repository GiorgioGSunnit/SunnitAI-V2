"""PROTECTED_PARAPHRASE_ENABLED (default false): protected sources are never reproduced, on every answer path.

Which sources are protected (reason(), mark()):
  - document_type 'special' (the doctrine books) and 'ccnl' (the collective agreements);
  - the documents named in PROTECTED_DOCUMENT_NAMES (';' or newline separated) or in the file
    PROTECTED_DOCUMENTS_FILE (one name per line, '#' a comment), by exact name -- case, runs of spaces and a
    leading "[TAG]" do not count: books, manuals, "Principî di diritto ..." collections, seminar material, also
    when they are mislabelled 'primary' on prod (scripts/protected_candidates.py lists the candidates);
  - a document whose text carries a reproduction notice ("Riproduzione Riservata", "è vietata la
    riproduzione", "tutti i diritti riservati", "proprietà letteraria riservata", a copyright sign, a text "di
    proprietà di" a publisher), unless it is a 'primary' document: primary law keeps its text, its notices
    are only stripped. A notice seen on one row protects the document's other rows (by document id).
  Rows are marked at retrieval time (row["_protected"] = the reason) on every channel: the chat graph's nodes
  (main.build_graph), the strategy pipeline's articles and rulings, the CCNL path's clauses.

The writer (TAG, with_rule()): a protected passage reaches the writer with TAG in its label, and every prompt
holding one gets WRITER_RULE: understand it and paraphrase, cite it by reference (title, article or section,
page if known), never quote it; short figures (amounts, days, percentages, dates) may appear as data.

The post-check (enforce()), on every answer: each sentence compared with the texts of the protected rows used.
A sentence sharing VERBATIM_WORDS (10) or more consecutive normalised words with one, or a near-duplicate of a
protected passage (Jaccard of SHINGLE-word shingles >= JACCARD, 0.5, with the window of the protected text of the
sentence's length its shared shingles align it to, for sentences of MIN_WORDS or more) is
rewritten once as a paraphrase, in one call for all of them; a rewrite that loses a number of the sentence (a
citation, an amount) or is still too close is refused, and the sentence is dropped. Reproduction notices are
stripped from the answer.

The sources shown (redact()): a protected document's sections keep only their reference (name and title, cut
to fewer than VERBATIM_WORDS words), never their text -- the citations of every path, the chat response, the
saved session, the section endpoint. Primary law keeps its text.

Measured with the flag off too, so a live run compares: enforce() logs "protected_paraphrase" (found /
rewritten / dropped / remaining; with the flag off nothing is rewritten and remaining = found) and redact()
logs "protected_sources_shown" (excerpts shown after it, target 0). measure() and excerpts_shown() read an
answer and its citations after the fact (tests/eval_narrative.py --direct / --ccnl, scripts/diagnose_long.py,
scripts/dryrun_strategy.py).
"""

import json
import logging
import os
import re
import threading
import unicodedata
from collections import Counter
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from . import feature_flags
from .verbose_logger import vlog

logger = logging.getLogger(__name__)

FLAG = "PROTECTED_PARAPHRASE_ENABLED"
NAMES_ENV = "PROTECTED_DOCUMENT_NAMES"
NAMES_FILE_ENV = "PROTECTED_DOCUMENTS_FILE"
PROTECTED_TYPES = ("special", "ccnl")
KEY = "_protected"

VERBATIM_WORDS = 10
SHINGLE = 3
JACCARD = 0.5
MIN_WORDS = 8


def enabled() -> bool:
    return feature_flags.enabled(FLAG)


# --- text ---------------------------------------------------------------------------------------------------

def fold(text: Optional[str]) -> str:
    """Lowercase, accents removed, anything but letters and digits a space."""
    text = unicodedata.normalize("NFKD", (text or "").lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


_SPLIT_RE = re.compile(r"((?<=[.!?;:])\s+(?=[A-ZÀ-Ü«\"“*(\-\d])|\n)")


def sentences(text: str) -> List[str]:
    """The text's sentences, each with the space or newline after it (joined back they give the text); a full
    stop splits only before a capital or a digit, so "art. 16" and "c.c. e" stay whole."""
    parts = _SPLIT_RE.split(text or "")
    return [parts[i] + (parts[i + 1] if i + 1 < len(parts) else "") for i in range(0, len(parts), 2)
            if parts[i] or (i + 1 < len(parts) and parts[i + 1])]


# --- reproduction notices -----------------------------------------------------------------------------------

# Detected AND stripped. The weak forms ("proprietà di", "vietata la riproduzione") only in a notice's wording:
# "il fondo di proprietà di Tizio" or a ruling on copyright is not a notice.
_NOTICE_RES = [
    re.compile(r"[-–—]?\s*Il\s?CCNL\s*\|\s*S\.?\s?I\.?\s?A\.?\s*S\.?r\.?l\.?\s*\|\s*Riproduzione\s+Riservata\s*[-–—]?",
               re.IGNORECASE),
    re.compile(r"\briproduzione\s+(?:\w+\s+)?riservata\b[.!]?", re.IGNORECASE),
    re.compile(r"\b(?:è\s+|e'\s+)?(?:severamente\s+)?vietat[ao]\s+(?:la\s+|ogni\s+)?riproduzion\w*"
               r"(?=[\s,]+(?:anche\s+parziale|totale|parziale|non\s+autorizzat|con\s+qualsiasi|in\s+qualsiasi|"
               r"dell[ae]?\s+(?:present|opera|testo|document|material|dispens))|\s*[.!]|\s*$)[^.\n]{0,160}[.!]?",
               re.IGNORECASE | re.MULTILINE),
    re.compile(r"\btutti\s+i\s+diritti\s+(?:sono\s+)?riservati\b[.!]?", re.IGNORECASE),
    re.compile(r"\bproprietà\s+(?:letteraria|intellettuale|artistica)\s+riservata\b[.!]?", re.IGNORECASE),
    re.compile(r"\b(?:testo|opera|documento|contenuto|materiale|dispensa|pubblicazione|volume)\s+(?:è\s+)?"
               r"(?:di\s+)?(?:esclusiva\s+)?proprietà\s+(?:esclusiva\s+)?di\s+[^.\n]{1,80}[.!]?", re.IGNORECASE),
    re.compile(r"\bdi\s+proprietà\s+(?:esclusiva\s+)?di\s+[^.\n]{0,60}?\b(?:editore|edizioni|casa\s+editrice)\b"
               r"[^.\n]{0,40}[.!]?", re.IGNORECASE),
    re.compile(r"(?:©|\(c\)\s+(?=\d{4}|[A-Z])|\bcopyright\b)[^\n]{0,100}", re.IGNORECASE),
    re.compile(r"\bwww\.ilccnl\.it\S*", re.IGNORECASE),
]


def has_notice(text: Optional[str]) -> bool:
    return bool(text) and any(p.search(text) for p in _NOTICE_RES)


def strip_notices(text: Optional[str]) -> str:
    """The text without reproduction notices and watermarks; spaces collapsed where one was cut, newlines kept."""
    if not text or not has_notice(text):
        return text or ""
    out = text
    for pattern in _NOTICE_RES:
        out = pattern.sub(" ", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r" +\n", "\n", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip() if text == text.strip() else out


# --- which documents ----------------------------------------------------------------------------------------

def norm_name(name: Optional[str]) -> str:
    name = re.sub(r"^\s*﻿?\[[A-Z]+\]\s*", "", name or "")
    return " ".join(unicodedata.normalize("NFC", name).casefold().split())


_NAMES_CACHE: Dict[str, Any] = {"key": None, "names": frozenset()}
_NAMES_LOCK = threading.Lock()


def protected_names() -> frozenset:
    """The configured names, normalised (norm_name); read again when the variable or the file changes."""
    raw = os.getenv(NAMES_ENV) or ""
    path = os.getenv(NAMES_FILE_ENV) or ""
    try:
        mtime = os.path.getmtime(path) if path else None
    except OSError:
        mtime = None
    key = (raw, path, mtime)
    with _NAMES_LOCK:
        if _NAMES_CACHE["key"] == key:
            return _NAMES_CACHE["names"]
        names = [n for n in re.split(r"[;\n]", raw)]
        if path and mtime is not None:
            try:
                with open(path, encoding="utf-8") as fh:
                    names += [line for line in fh.read().splitlines() if not line.strip().startswith("#")]
            except OSError as exc:
                logger.warning("protected sources: %s not read: %s", path, exc)
        _NAMES_CACHE.update(key=key, names=frozenset(norm_name(n) for n in names if n.strip()))
        return _NAMES_CACHE["names"]


# Documents seen with a reproduction notice in this process (by id): their other rows are protected too.
_NOTICE_DOCS: set = set()
_NOTICE_DOCS_MAX = 5000


def _remember_notice(doc_id: Optional[str]) -> None:
    if doc_id and len(_NOTICE_DOCS) < _NOTICE_DOCS_MAX:
        _NOTICE_DOCS.add(doc_id)


def classify(doc_id: Optional[str], doc_type: Optional[str], doc_name: Optional[str],
             texts: Iterable[Optional[str]] = ()) -> Optional[str]:
    """Why a document is protected ("special", "ccnl", "by name", "notice"), or None."""
    if doc_type in PROTECTED_TYPES:
        return doc_type
    if doc_name and norm_name(doc_name) in protected_names():
        return "by name"
    if doc_type == "primary":
        return None
    if doc_id and doc_id in _NOTICE_DOCS:
        return "notice"
    if any(has_notice(t) for t in texts):
        _remember_notice(doc_id)
        return "notice"
    return None


def _row_doc(row: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """(document id, type, name) of a row, from what the row carries."""
    from .source_tiers import row_doc
    doc_id, kind = row_doc(row)
    d = row.get("d") if isinstance(row.get("d"), dict) else {}
    name = d.get("name") or d.get("document_name") or d.get("document_title") or row.get("document_name")
    kind = kind or row.get("document_type")
    return doc_id or row.get("document_id"), kind, name


_TEXT_KEYS = ("plain_text", "text", "abstract")


def row_text(row: Dict[str, Any]) -> str:
    """The text a row gives the writer: its chosen passage, its Section's text or abstract, or its own text."""
    if row.get("_passage"):
        return row["_passage"]
    s = row.get("s") if isinstance(row.get("s"), dict) else {}
    for key in _TEXT_KEYS:
        if s.get(key):
            return s[key]
    for key in _TEXT_KEYS:
        if isinstance(row.get(key), str) and row[key]:
            return row[key]
    return ""


def reason(row: Dict[str, Any]) -> Optional[str]:
    """The row's mark, else why it is protected (computed), else None."""
    if not isinstance(row, dict):
        return None
    if row.get(KEY):
        return row[KEY]
    doc_id, kind, name = _row_doc(row)
    return classify(doc_id, kind, name, [row_text(row)])


def _lookup(session, ids: Sequence[str]) -> Dict[str, Tuple[Optional[str], Optional[str]]]:
    if not ids or session is None:
        return {}
    try:
        return {r["id"]: (r["type"], r["name"]) for r in session.run(
            "MATCH (d:Document) WHERE d.id IN $ids RETURN d.id AS id, d.document_type AS type, d.name AS name",
            ids=list(ids)).data()}
    except Exception as exc:
        logger.warning("protected sources: document types not read: %s", exc)
        return {}


def _strip_row(row: Dict[str, Any]) -> None:
    if row.get("_passage"):
        row["_passage"] = strip_notices(row["_passage"])
    s = row.get("s")
    if isinstance(s, dict):
        for key in _TEXT_KEYS:
            if isinstance(s.get(key), str) and has_notice(s[key]):
                s[key] = strip_notices(s[key])
    for key in _TEXT_KEYS:
        if isinstance(row.get(key), str) and has_notice(row[key]):
            row[key] = strip_notices(row[key])


def mark(rows: Optional[List[Dict[str, Any]]], session=None, stage: str = "") -> Optional[List[Dict[str, Any]]]:
    """With the flag: each protected row marked (row["_protected"] = why), notices stripped from every row's
    text, the documents a row's text does not type or name looked up in one query (when a session is given).
    Logged as protected_rows. The same list back; nothing changes with the flag off."""
    if not rows or not enabled():
        return rows
    try:
        found = [(r, *_row_doc(r)) for r in rows if isinstance(r, dict)]
        missing = sorted({doc for _, doc, kind, name in found if doc and (kind is None or name is None)})
        looked = _lookup(session, missing)
        # A notice on one row of a document marks all of them (this batch and later ones).
        for row, doc, kind, _ in found:
            kind = kind or (looked.get(doc) or (None, None))[0]
            if kind != "primary" and has_notice(row_text(row)):
                _remember_notice(doc)
        counts: Counter = Counter()
        for row, doc, kind, name in found:
            if doc in looked:
                kind, name = kind or looked[doc][0], name or looked[doc][1]
            why = row.get(KEY) or classify(doc, kind, name, [row_text(row)])
            _strip_row(row)
            if why:
                row[KEY] = why
                counts[why] += 1
        if counts:
            vlog("protected_rows", {"stage": stage, "rows": len(found), "protected": sum(counts.values()),
                                    "by_reason": dict(counts)})
    except Exception as exc:
        logger.warning("protected sources: %s rows not marked: %s", stage, exc)
    return rows


def node(fn: Callable, stage: str, driver=None, database: Optional[str] = None) -> Callable:
    """The graph node `fn` with its rows ("raw_result", "special_dottrina_rows") marked (mark); unchanged
    with the flag off."""
    def wrapped(state):
        out = fn(state)
        if not enabled() or not isinstance(out, dict):
            return out
        keys = [k for k in ("raw_result", "special_dottrina_rows") if out.get(k)]
        if not keys:
            return out
        try:
            if driver is not None:
                import neo4j
                with driver.session(database=database, default_access_mode=neo4j.READ_ACCESS) as session:
                    for k in keys:
                        mark(out[k], session, stage)
            else:
                for k in keys:
                    mark(out[k], None, stage)
        except Exception as exc:
            logger.warning("protected sources: %s rows not marked: %s", stage, exc)
        return out
    wrapped.__name__ = getattr(fn, "__name__", stage)
    return wrapped


def protected_texts(rows: Optional[Iterable[Dict[str, Any]]]) -> List[str]:
    """The texts of the protected rows among `rows` (marked, or classified now), notices stripped."""
    out = []
    for row in rows or []:
        if isinstance(row, dict) and reason(row):
            text = strip_notices(row_text(row))
            if text.strip():
                out.append(text)
    return out


# --- the writer ---------------------------------------------------------------------------------------------

TAG = "[FONTE RISERVATA: da parafrasare e citare solo per riferimento, mai riportarne il testo]"
WRITER_RULE = (
    "FONTI RISERVATE: i passaggi contrassegnati [FONTE RISERVATA] (o con il campo \"_protected\") provengono da "
    "opere protette (dottrina, manuali, raccolte di massime, contratti collettivi). Usali per comprendere e "
    "spiegare con parole tue, con una struttura della frase diversa; citali solo per riferimento (titolo "
    "dell'opera o del contratto, articolo o sezione, pagina se indicata). Non riportarne mai il testo, nemmeno "
    "tra virgolette o in parte, e non seguirne la formulazione frase per frase. Cifre brevi (importi, giorni, "
    "percentuali, date) possono comparire come dati. Il testo delle norme di legge può essere citato."
)


def tag(row: Dict[str, Any]) -> str:
    """" " + TAG for a protected row with the flag on, else ""."""
    return f" {TAG}" if enabled() and reason(row) else ""


def with_rule(text: str) -> str:
    """The prompt text with WRITER_RULE appended when it holds a protected passage (TAG or a "_protected" row)
    and the flag is on."""
    if enabled() and text and (TAG in text or '"_protected"' in text) and WRITER_RULE not in text:
        return f"{text}\n\n{WRITER_RULE}"
    return text


def _citation_reason(c: Dict[str, Any]) -> Optional[str]:
    if c.get("protected"):
        return c.get("protected") if isinstance(c.get("protected"), str) else "marked"
    texts = [s.get(k) for s in c.get("sections") or [] if isinstance(s, dict) for k in _TEXT_KEYS]
    return classify(c.get("document_id"), c.get("document_type"), c.get("document_name"), texts)


def citation_tag(c: Dict[str, Any]) -> str:
    """" " + TAG for a protected citation with the flag on, else ""."""
    return f" {TAG}" if enabled() and _citation_reason(c) else ""


def citation_lines(citations: Optional[List[Dict[str, Any]]], chars: int, docs: int = 5, sections: int = 2) -> str:
    """The "- document, section: text" lines a writer prompt lists its sources with (the format every caller
    used), each protected one with TAG when the flag is on, notices stripped."""
    on = enabled()
    lines = []
    for c in (citations or [])[:docs]:
        label = TAG if on and _citation_reason(c) else ""
        for s in (c.get("sections") or [])[:sections]:
            text = s.get("plain_text", "") or ""
            if on:
                text = strip_notices(text)
            lines.append(f"- {c['document_name']}, {s['name']}{' ' + label if label else ''}: {text[:chars]}")
    return "\n".join(lines)


def citation_texts(citations: Optional[List[Dict[str, Any]]]) -> List[str]:
    """The texts of the protected citations' sections (for enforce on a path that only has citations)."""
    out = []
    for c in citations or []:
        if _citation_reason(c):
            for s in c.get("sections") or []:
                text = strip_notices(next((s.get(k) for k in _TEXT_KEYS if s.get(k)), "") or "")
                if text.strip():
                    out.append(text)
    return out


# --- the post-check -----------------------------------------------------------------------------------------

def _grams(words: Sequence[str], n: int) -> set:
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


class _Index:
    """The protected texts' VERBATIM_WORDS-grams, and their SHINGLE-grams by position: a sentence is compared
    with the window of a protected text of its own length that its shared shingles align it to."""

    def __init__(self, texts: Iterable[str]):
        self.runs: set = set()
        self.words: List[List[str]] = []
        self.postings: Dict[str, List[Tuple[int, int]]] = {}
        for tid, text in enumerate(texts):
            words = fold(text).split()
            self.words.append(words)
            self.runs |= _grams(words, VERBATIM_WORDS)
            for i in range(len(words) - SHINGLE + 1):
                self.postings.setdefault(" ".join(words[i:i + SHINGLE]), []).append((tid, i))

    def __bool__(self):
        return bool(self.postings)

    def why(self, sentence: str) -> Optional[str]:
        """"verbatim" (a run of VERBATIM_WORDS words), "near-duplicate" (Jaccard >= JACCARD with the aligned
        window), or None."""
        words = fold(sentence).split()
        if len(words) >= VERBATIM_WORDS and _grams(words, VERBATIM_WORDS) & self.runs:
            return "verbatim"
        if len(words) < MIN_WORDS:
            return None
        ordered = [" ".join(words[i:i + SHINGLE]) for i in range(len(words) - SHINGLE + 1)]
        grams = set(ordered)
        votes: Counter = Counter()
        for offset, gram in enumerate(ordered):
            for tid, at in self.postings.get(gram, ()):
                votes[(tid, at - offset)] += 1
        n = len(words)
        for (tid, start), _ in votes.most_common(5):
            for st in range(max(0, start - 2), start + 3):      # a word or two inserted or removed
                window = _grams(self.words[tid][st:st + n], SHINGLE)
                shared = len(grams & window)
                if window and shared / (len(grams) + len(window) - shared) >= JACCARD:
                    return "near-duplicate"
        return None


def overlaps(answer: str, texts: Iterable[str], index: Optional[_Index] = None) -> List[Tuple[str, str]]:
    """(sentence, why) for each sentence of the answer too close to a protected text."""
    index = index if index is not None else _Index(texts)
    if not index:
        return []
    out = []
    for sentence in sentences(answer or ""):
        s = sentence.strip()
        if s:
            why = index.why(s)
            if why:
                out.append((s, why))
    return out


_NUMBER_RE = re.compile(r"\d+(?:[.,/\-]\d+)*")


def _keeps_numbers(old: str, new: str) -> bool:
    """Every number of the old sentence (citation numbers, article numbers, amounts, dates) is in the new one."""
    have = Counter(_NUMBER_RE.findall(new))
    need = Counter(_NUMBER_RE.findall(old))
    return all(have[k] >= v for k, v in need.items())


_REWRITE_SYSTEM = (
    "Riscrivi ciascuna delle frasi date come parafrasi: stesso significato, parole tue e una struttura della "
    "frase diversa, perché riproducono troppo da vicino il testo di un'opera protetta. Mantieni ogni numero "
    "così com'è (numeri di citazione come [1], articoli, sentenze, importi, giorni, percentuali, date), ogni "
    "riferimento e la formattazione iniziale (elenco puntato, grassetto). Non aggiungere né togliere contenuti. "
    "Rispondi SOLO con un array JSON di stringhe, una per frase, nello stesso ordine."
)


def _rewrite(found: List[str]) -> List[Optional[str]]:
    """One call: a paraphrase per sentence (None where the reply has none)."""
    from langchain_core.messages import HumanMessage, SystemMessage
    from .ai_chat import _call_chat
    reply = _call_chat([SystemMessage(content=_REWRITE_SYSTEM),
                        HumanMessage(content=json.dumps(found, ensure_ascii=False))],
                       max_tokens=min(4000, 200 + 120 * len(found)), temperature=0.3)
    text = re.sub(r"```(?:json)?\s*", "", reply or "").strip().rstrip("`").strip()
    start, end = text.find("["), text.rfind("]")
    parsed = json.loads(text[start:end + 1]) if start != -1 and end > start else []
    out = [p if isinstance(p, str) and p.strip() else None for p in parsed][:len(found)]
    return out + [None] * (len(found) - len(out))


_EMPTY_LINE_RE = re.compile(r"(?m)^[ \t]*(?:[-*•]|\d+[.)])?[ \t]*(?:\*\*)?[ \t]*$\n?")


def _drop(answer: str, sentence: str) -> str:
    out = answer.replace(sentence, "", 1)
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = _EMPTY_LINE_RE.sub(lambda m: "\n" if m.group(0).endswith("\n") else "", out)
    return re.sub(r"\n{3,}", "\n\n", out)


def enforce(answer: str, rows: Optional[Iterable[Dict[str, Any]]] = None, texts: Optional[Iterable[str]] = None,
            stage: str = "", llm: bool = True, rewrite: Optional[Callable[[List[str]], List[Optional[str]]]] = None
            ) -> Tuple[str, Dict[str, Any]]:
    """(the answer, its stats) -- see the module's docstring. With the flag off the answer comes back unchanged
    and only the overlaps are counted (remaining = found). Logged as protected_paraphrase. Never raises: on a
    failure the answer is the one given, notices stripped."""
    stats: Dict[str, Any] = {"stage": stage, "enabled": enabled(), "protected_texts": 0, "found": 0,
                             "rewritten": 0, "dropped": 0, "remaining": 0}
    if not answer:
        return answer, stats
    on = stats["enabled"]
    original = answer
    try:
        sources = protected_texts(rows) + [strip_notices(t) for t in (texts or []) if t and t.strip()]
        stats["protected_texts"] = len(sources)
        if on:
            answer = strip_notices(answer)
            stats["notices_stripped"] = answer != original
        index = _Index(sources)
        found = overlaps(answer, sources, index) if index else []
        stats["found"] = len(found)
        stats["kinds"] = dict(Counter(why for _, why in found))
        if not on:
            stats["remaining"] = len(found)
            if found:
                vlog("protected_paraphrase", stats)
            return original, stats
        if found and llm:
            flagged = [s for s, _ in found]
            try:
                rewrites = (rewrite or _rewrite)(flagged)
            except Exception as exc:
                logger.warning("protected sources: %s rewrite failed, the sentences go: %s", stage, exc)
                rewrites = [None] * len(flagged)
            for old, new in zip(flagged, rewrites):
                new = strip_notices((new or "").strip())
                if new and new != old and _keeps_numbers(old, new) and not index.why(new) and old in answer:
                    answer = answer.replace(old, new, 1)
                    stats["rewritten"] += 1
        still = overlaps(answer, sources, index) if found else []
        for sentence, _ in still:
            answer = _drop(answer, sentence)
        stats["dropped"] = len(still)
        stats["remaining"] = len(overlaps(answer, sources, index)) if found else 0
        if found or stats.get("notices_stripped"):
            vlog("protected_paraphrase", stats)
            logger.info("protected sources: %s found %d, rewritten %d, dropped %d, remaining %d", stage,
                        stats["found"], stats["rewritten"], stats["dropped"], stats["remaining"])
        return answer, stats
    except Exception as exc:
        logger.warning("protected sources: %s post-check failed, answer kept: %s", stage, exc)
        stats["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        return (strip_notices(original) if on else original), stats


# --- the sources shown --------------------------------------------------------------------------------------

def _reference_only(value: Optional[str]) -> Optional[str]:
    """A section's name or title cut below VERBATIM_WORDS words (a commentary's name is often a text fragment)."""
    if not value:
        return value
    value = strip_notices(str(value))
    words = value.split()
    limit = VERBATIM_WORDS - 1
    return value if len(words) <= limit else " ".join(words[:limit - 1]) + " …"


def redact_section(section: Dict[str, Any]) -> Dict[str, Any]:
    out = {k: v for k, v in section.items() if k not in _TEXT_KEYS}
    out.update(plain_text="", protected=True)
    for key in ("name", "title"):
        if key in out:
            out[key] = _reference_only(out[key])
    return out


def redact(citations: Optional[List[Dict[str, Any]]], stage: str = "") -> List[Dict[str, Any]]:
    """With the flag: every protected citation's sections as references only (redact_section; the citation
    carries "protected"), notices stripped from the others' text. Logged as protected_sources_shown (also with
    the flag off: what is shown). New dicts; the input is not changed."""
    citations = list(citations or [])
    if not citations:
        return citations
    on = enabled()
    out, redacted = [], 0
    for c in citations:
        if not isinstance(c, dict):
            out.append(c)
            continue
        why = _citation_reason(c)
        if on and why:
            sections = [redact_section(s) if isinstance(s, dict) else s for s in c.get("sections") or []]
            redacted += sum(1 for s in c.get("sections") or [] if isinstance(s, dict)
                            and any(s.get(k) for k in _TEXT_KEYS))
            out.append({**c, "sections": sections, "protected": why})
        elif on:
            out.append({**c, "sections": [{**s, **{k: strip_notices(s[k]) for k in _TEXT_KEYS
                                                   if isinstance(s.get(k), str)}} if isinstance(s, dict) else s
                                          for s in c.get("sections") or []]})
        else:
            out.append(c)
    shown = excerpts_shown(out)
    if redacted or shown:
        vlog("protected_sources_shown", {"stage": stage, "enabled": on, "redacted": redacted, "shown": shown})
    return out


def redact_flat(sections: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """The flat {document_name, document_id, document_type, name, plain_text, ...} entries (pending_sections)
    with a protected one's text removed, for showing; unchanged with the flag off."""
    if not sections or not enabled():
        return list(sections or [])
    out = []
    for s in sections:
        if isinstance(s, dict) and classify(s.get("document_id"), s.get("document_type"), s.get("document_name"),
                                            [s.get(k) for k in _TEXT_KEYS]):
            out.append(redact_section(s))
        else:
            out.append(s)
    return out


def excerpts_shown(citations: Optional[List[Dict[str, Any]]]) -> int:
    """How many sections of protected documents the citations show with text (target 0); flag-independent."""
    n = 0
    for c in citations or []:
        if isinstance(c, dict) and _citation_reason(c):
            n += sum(1 for s in c.get("sections") or [] if isinstance(s, dict)
                     and any(str(s.get(k) or "").strip() for k in _TEXT_KEYS))
    return n


def measure(answer: str, rows: Optional[Iterable[Dict[str, Any]]] = None,
            citations: Optional[List[Dict[str, Any]]] = None, texts: Optional[Iterable[str]] = None) -> Dict[str, int]:
    """{"overlaps": the answer's sentences too close to a protected text of `rows` / `texts`, "excerpts_shown":
    excerpts_shown(citations)}, read after the fact; flag-independent."""
    sources = protected_texts(rows) + [t for t in (texts or []) if t and t.strip()]
    return {"overlaps": len(overlaps(answer or "", sources)), "excerpts_shown": excerpts_shown(citations)}


# --- measuring (tests/eval_narrative.py --direct / --ccnl, scripts/diagnose_long.py, scripts/dryrun_strategy.py) --

COUNTS = ("found", "rewritten", "dropped", "remaining", "excerpts_shown")


def totals(events: Iterable[Sequence[Any]]) -> Dict[str, int]:
    """found / rewritten / dropped / remaining summed over a run's protected_paraphrase events ((event, data)
    or (event, data, ms) items)."""
    out = {"found": 0, "rewritten": 0, "dropped": 0, "remaining": 0, "checks": 0}
    for item in events:
        event, data = item[0], item[1]
        if event == "protected_paraphrase" and isinstance(data, dict):
            out["checks"] += 1
            for k in ("found", "rewritten", "dropped", "remaining"):
                out[k] += int(data.get(k) or 0)
    return out


def run_record(events: Iterable[Sequence[Any]], answer: str, rows: Optional[Iterable[Dict[str, Any]]] = None,
               citations: Optional[List[Dict[str, Any]]] = None, texts: Optional[Iterable[str]] = None
               ) -> Dict[str, Any]:
    """One case's figures: found / rewritten / dropped from its post-checks' events, remaining read off the
    final answer against the protected rows / texts given (the larger of that and the events' own count, so a
    step after the check that copies text is counted too), excerpts_shown off its citations."""
    logged = totals(events)
    measured = measure(answer, rows, citations, texts)
    remaining = max(measured["overlaps"], logged["remaining"])
    return {"enabled": enabled(), "found": max(logged["found"], remaining), "rewritten": logged["rewritten"],
            "dropped": logged["dropped"], "remaining": remaining, "excerpts_shown": measured["excerpts_shown"]}


def sum_records(records: Iterable[Optional[Dict[str, Any]]]) -> Dict[str, int]:
    out = {k: 0 for k in COUNTS}
    for r in records:
        for k in COUNTS:
            out[k] += int((r or {}).get(k) or 0)
    return out


def report_line(total: Dict[str, int]) -> str:
    return (f"verbatim overlaps with protected sources: found {total.get('found', 0)} / rewritten "
            f"{total.get('rewritten', 0)} / dropped {total.get('dropped', 0)} / remaining "
            f"{total.get('remaining', 0)} (target 0); protected excerpts shown in sources: "
            f"{total.get('excerpts_shown', 0)} (target 0)")
