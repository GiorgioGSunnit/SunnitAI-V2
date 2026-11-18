"""v79.11 (A): ELEMENT_SCHEMAS (default false) -- per-article element schemas: what an article requires, element by
element, with the logic that joins them, its function (tipo), who it binds and who may invoke it, made once from the
article's text by scripts/article_cards_bulk.py --schemas (SCHEMA_SYSTEM, user_message, parse_reply, make_line live
here, so the generator and the tests share them) and reviewed by the lawyer (scripts/schema_show.py --edit).

Files (read at call time, cached on (path, size, mtime), re-read when they change; a missing file is empty):
- data/article_schemas.jsonl (ARTICLE_SCHEMAS_FILE): generated, one line per article, keyed (doc, article); the last
  line of a key wins. Each line carries the source_hash (article_cards.text_hash of the RAW article text) and the
  checks run on it (check); a line with a blocking check is never used (usable). v79.12.2: only an invalid schema
  and a logic tree with dangling refs block (BLOCKING); the element checks (ELEMENT_CHECKS: ungrounded on Italian
  stems, fragments, branches without an alternative) fix or drop the offending element and keep the schema (repair,
  applied when the line is read; "dropped" logs it); a logic node with no args is removed, an empty tree is no logic.
- data/article_schemas_reviewed.jsonl (ARTICLE_SCHEMAS_REVIEWED_FILE): the lawyer's overlay, keyed by label («582
  c.p.»): a reviewed schema (reviewed: true) wins over the generated one while its source_hash is the article's; the
  lawyer's notes (note_di_inquadramento) are merged into whichever line is used. Never written by the generator.

for_row / for_label give (line, reason): "schema", "reviewed", or why the text path runs ("off", "no_schema",
"hash_mismatch", "unusable:<check>", "reviewed_stale"). Flag off: nothing here is consulted -- (None, "off").
requisites(...) turns a line into issue_records._requisites' element dicts; summary(...) is what the record keeps;
gate / decide / hard_excluded serve the general-part gate (an unreviewed schema never excludes); shadow_admission the
shadow admission (never authoritative); quality the measures against the lawyer's corrections.
"""

import hashlib
import json
import logging
import os
import re
import threading
import unicodedata
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

FLAG = "ELEMENT_SCHEMAS"
FILE_ENV, REVIEWED_ENV = "ARTICLE_SCHEMAS_FILE", "ARTICLE_SCHEMAS_REVIEWED_FILE"
SCHEMA_VERSION = 1
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_FILE = os.path.join(ROOT, "data", "article_schemas.jsonl")
DEFAULT_REVIEWED_FILE = os.path.join(ROOT, "data", "article_schemas_reviewed.jsonl")
MODULES_FILE = os.path.join(ROOT, "data", "institution_modules.json")

TIPI = ("claim-conferring", "defence", "scriminante", "limiting", "definitional", "procedural", "sanction", "other")
KINDS = ("condotta", "evento", "elemento soggettivo", "condizione", "termine", "qualità del soggetto", "oggetto",
         "nesso_causale", "quantificazione", "negazione", "elemento_finalistico", "competenza", "onere_della_prova",
         "other")
ROLES = ("attore", "convenuto", "imputato", "indagato", "persona offesa", "parte civile", "PA", "terzo", "chiunque",
         "creditore", "debitore", "lavoratore", "datore di lavoro", "consumatore", "professionista", "altro")
GROUNDING = ("explicit", "inferred")
LOGIC_OPS = ("all_of", "any_of", "not", "conditional")
CONNECTIVES = ("e", "o", None)
SPECIAL_KINDS = ("presunzione", "aggravamento", "attenuazione", "equiparazione", "caso speciale", "scriminante",
                 "procedibilità", "other")
CHECK_KEYS = ("invalid", "ungrounded", "fragments", "branches_without_alternative", "logic_errors",
              "logic_incomplete", "fonte_unlocated", "inferred_unexplained", "comma_unknown", "kind_other_unexplained")
# v79.12.2: only an invalid schema and a logic tree with dangling references (logic_errors) make a schema unusable;
# the element checks (ELEMENT_CHECKS) fix or drop the offending element (repair) and keep the schema
BLOCKING = ("failed", "cut", "invalid", "logic_errors")
ELEMENT_CHECKS = ("ungrounded", "fragments", "branches_without_alternative")
NON_BLOCKING = ELEMENT_CHECKS + ("logic_incomplete", "fonte_unlocated", "inferred_unexplained", "comma_unknown",
                                 "kind_other_unexplained")
BLOCKING_79_12_1 = ("failed", "cut", "invalid", "ungrounded", "fragments", "branches_without_alternative",
                    "logic_errors")     # the rule before v79.12.2 -- schema_show --recheck's «before» on old lines
STRUCTURAL = ("invalid", "logic_errors")        # what even a reviewed schema must pass to be read safely
NOT_REQUISITE_KINDS = ("quantificazione", "competenza", "onere_della_prova")   # kept in the summary, never requisites
DECOMPOSABLE_TIPI = ("sanction", "claim-conferring", "defence", "scriminante")
KIND_RULE = {"elemento soggettivo": "mental", "termine": "termine", "condizione": "condition"}   # others -> "plain"
REASONS = ("schema", "reviewed", "off", "no_schema", "hash_mismatch", "unusable", "reviewed_stale")


def enabled() -> bool:
    from . import feature_flags
    return feature_flags.enabled(FLAG)


def text_hash(text: Optional[str]) -> str:
    """v79.11 (A): article_cards.text_hash -- sha1 of the whitespace-normalised text, 16 hex."""
    from .article_cards import text_hash as cards_hash
    return cards_hash(text)


def _resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def schema_file() -> str:
    return _resolve(os.getenv(FILE_ENV) or DEFAULT_FILE)


def reviewed_file() -> str:
    return _resolve(os.getenv(REVIEWED_ENV) or DEFAULT_REVIEWED_FILE)


# --- labels -----------------------------------------------------------------------------------------------------------

def _code(code: Any) -> str:
    from .topic_step import _code as topic_code
    return topic_code(code)


def label_of(number: Any, code: Any) -> str:
    """v79.11 (A): «582 c.p.» -- issue_records._label's form (a code's document name read as its sigla)."""
    return f"{str(number).strip()} {_code(code)}"


def norm_label(label: Any) -> str:
    """v79.11 (A): a label as the overlay keys it -- the number lowercased, the code by its sigla; a special comma's
    («52 c.p. co. 2») and a numero's («62 n. 2 c.p.») label read as their article's."""
    m = re.match(r"^\s*(\S+)\s+(.*?)\s*$", str(label or ""))
    if not m:
        return str(label or "").strip().lower()
    rest = re.sub(r"\s+co\.\s*\S+\s*$", "", m.group(2))
    rest = re.sub(r"^n\.\s*\d+\s+", "", rest)
    return f"{m.group(1).lower()} {_code(rest)}"


def _key(doc: Any, number: Any) -> Tuple[str, str]:
    return (str(doc or ""), str(number).strip().lower())


def _line_label(line: Dict[str, Any]) -> str:
    if line.get("label"):
        return norm_label(line["label"])
    if line.get("article") is not None and (line.get("code") or line.get("doc")):
        return norm_label(label_of(line.get("article"), line.get("code") or line.get("doc")))
    return ""


# --- the two files ----------------------------------------------------------------------------------------------------

_lock = threading.Lock()
_FILES: Dict[str, Dict[str, Any]] = {}       # path -> {"sig", "lines", <index name>: index}


def _entry(path: str) -> Optional[Dict[str, Any]]:
    """The file's lines (bad JSON lines skipped), re-read when (path, size, mtime) changes; None when missing."""
    try:
        if not os.path.isfile(path):
            return None
        sig = (path, os.path.getsize(path), os.path.getmtime(path))
    except OSError:
        return None
    with _lock:
        cached = _FILES.get(path)
        if cached and cached["sig"] == sig:
            return cached
        lines: List[Dict[str, Any]] = []
        try:
            with open(path, encoding="utf-8") as fh:
                for raw in fh:
                    try:
                        record = json.loads(raw)
                    except ValueError:
                        continue
                    if isinstance(record, dict):
                        lines.append(record)
        except OSError as exc:
            logger.warning("element schemas: %s not readable: %s", path, exc)
            return None
        entry = {"sig": sig, "lines": lines}
        _FILES[path] = entry
        logger.info("element schemas: %d lines from %s", len(lines), path)
        return entry


def _index(path: str, name: str, build) -> Dict[Any, Dict[str, Any]]:
    entry = _entry(path)
    if entry is None:
        return {}
    with _lock:
        if name not in entry:
            entry[name] = build(entry["lines"])
        return entry[name]


def generated() -> Dict[Tuple[str, str], Dict[str, Any]]:
    """v79.11 (A): {(doc, article_lower): line} of the generated file, the last line of each key."""
    return _index(schema_file(), "by_key", lambda lines: {_key(x.get("doc"), x.get("article")): x for x in lines})


def generated_by_label() -> Dict[str, Dict[str, Any]]:
    """{label: line} of the generated file (the last line of a label, whatever its document)."""
    def build(lines):
        out: Dict[str, Dict[str, Any]] = {}
        for x in lines:
            label = _line_label(x)
            if label:
                out[label] = x
        return out
    return _index(schema_file(), "by_label", build)


def reviewed() -> Dict[str, Dict[str, Any]]:
    """v79.11 (A): {label: overlay line} of the lawyer's file, the last line of each label."""
    def build(lines):
        out: Dict[str, Dict[str, Any]] = {}
        for x in lines:
            label = _line_label(x)
            if label:
                out[label] = x
        return out
    return _index(reviewed_file(), "by_label", build)


# --- text helpers -----------------------------------------------------------------------------------------------------

def _fold(text: Any) -> str:
    t = unicodedata.normalize("NFKD", str(text or "").lower())
    return "".join(c for c in t if not unicodedata.combining(c))


_STOP = frozenset(("gli", "uno", "una", "del", "dei", "della", "dello", "delle", "degli", "dal", "dai", "dalla",
                   "dallo", "dalle", "dagli", "nel", "nei", "nella", "nello", "nelle", "negli", "sul", "sui", "sulla",
                   "sullo", "sulle", "sugli", "col", "coi", "coll", "alla", "allo", "alle", "agli", "all", "dell",
                   "dall", "nell", "sull", "per", "tra", "fra", "con"))


def content_words(text: Any) -> List[str]:
    """v79.11 (A): the words of 3 letters or more, lowercased, accent-folded, articles and prepositions left out."""
    return [w for w in re.findall(r"[^\W\d_]+", _fold(text)) if len(w) >= 3 and w not in _STOP]


# v79.12.2: the function words grounding ignores (accent-folded) -- articles and prepositions (_STOP), pronouns,
# determiners, conjunctions, auxiliaries and modals: an element is grounded on its content words only. Never a word
# that carries the element's polarity or direction (non, senza, salvo, più/meno, molto/poco, prima/dopo/entro,
# sopra/sotto): an element saying the opposite of the text («senza il consenso» against «con il consenso») stays
# ungrounded, as under v79.12.1
FUNCTION_WORDS = _STOP | frozenset((
    "che", "chi", "cui", "nonche", "anche", "come", "quale", "quali", "quanto", "quanta", "quanti", "quante",
    "questo", "questa", "questi", "queste", "quello", "quella", "quelli", "quelle", "quel", "quei", "quegli",
    "suo", "sua", "suoi", "sue", "loro", "esso", "essa", "essi", "esse", "egli", "ella", "lui", "lei", "nei", "ogni",
    "alcuno", "alcuna", "alcuni", "alcune", "taluno", "taluna", "altro", "altra", "altri", "altre", "tale", "tali",
    "stesso", "stessa", "stessi", "stesse", "medesimo", "medesima", "medesimi", "medesime", "proprio", "propria",
    "propri", "proprie", "ovvero", "oppure", "ossia", "ovverosia", "ancorche", "sebbene", "purche", "qualora",
    "quando", "mentre", "dove", "ove", "verso", "presso", "contro", "mediante", "tutto", "tutta", "tutti", "tutte",
    "sia", "siano", "essere", "sono", "era", "erano", "fosse", "fossero", "stato", "stata", "stati", "state",
    "avere", "aver", "abbia", "abbiano", "avesse", "avessero", "avuto", "viene", "vengono", "venga", "vengano", "puo",
    "possono", "possa", "possano", "deve", "devono", "debba", "debbano"))

_ACCENTED = "àèéìíòóùú"
_STEM_SUFFIXES = (re.compile(r"(?:are|ere|ire)$"),        # the infinitive: the verb stem
                  re.compile(r"[aiu]t[aeio]$"),            # the participle -ato/-ito/-uto and its -a/-i/-e
                  re.compile(r"[aeio]$"))                  # the inflection -e/-i/-o/-a
_VERB_FORM = re.compile(r"(?:[aiu]t[aeio]|are|ere|ire)$")  # a participle or an infinitive


def _accented(word: Any) -> bool:
    return unicodedata.normalize("NFC", str(word or "").strip().lower())[-1:] in _ACCENTED


def stem(word: Any, accented: bool = False) -> str:
    """v79.12.2: an Italian word's stem for grounding -- accent-folded, lowercased; the infinitive's -are/-ere/-ire,
    else a participle's -ato/-ito/-uto (-a/-i/-e), else the final -e/-i/-o/-a stripped, each only when 3 letters stay
    («cagionata», «cagiona», «cagionare» -> «cagion»; «lesioni», «lesione» -> «lesion»). A word ending accented
    («qualità», «città», or «qualita'» -- accented=True) keeps its last letter: no inflection there («autorità» is
    no «autore»). A short verb's participle meets its verb by verb_base."""
    w = _fold(word)
    if accented or _accented(word):
        return w
    for suffix in _STEM_SUFFIXES:
        m = suffix.search(w)
        if m and m.start() >= 3:
            return w[:m.start()]
    return w


def verb_base(word: Any, accented: bool = False) -> str:
    """v79.12.2: a participle's or an infinitive's verb base, down to 2 letters («usata», «usato», «usare» -> «us»);
    "" for any other word. It meets another verb form's base or a plain word less its final vowel («usa», «uso»)."""
    w = _fold(word)
    if accented or _accented(word):
        return ""
    m = _VERB_FORM.search(w)
    return w[:m.start()] if m and m.start() >= 2 else ""


def _bare(word: str, accented: bool) -> str:
    """A word less its final -e/-i/-o/-a (2 letters at least), matched only against a verb_base («usa» -> «us»)."""
    return word[:-1] if not accented and word[-1:] in "aeio" and len(word) >= 3 else ""


def _tokens(text: Any) -> List[Tuple[str, bool]]:
    """(the accent-folded word, whether it ends accented) of each word of 3 letters or more not in _STOP --
    content_words with the accent kept in view: «qualità» and «qualita'» end accented, «dell'atto»'s «dell» not."""
    s = unicodedata.normalize("NFC", str(text or "").lower())
    out: List[Tuple[str, bool]] = []
    for m in re.finditer(r"[^\W\d_]+", s):
        w = _fold(m.group())
        if len(w) < 3 or w in _STOP:
            continue
        after = s[m.end():m.end() + 2]
        out.append((w, m.group()[-1] in _ACCENTED or (after[:1] in ("'", "’") and not after[1:2].isalpha())))
    return out


class _Lexicon:
    """A text's words for grounding: every word as such; the content words (FUNCTION_WORDS never ground by stem:
    «dove» is no «dovere») by stem, by verb base and less their final vowel."""

    def __init__(self, tokens: Iterable[Tuple[str, bool]]):
        tokens = list(tokens)
        content = [(w, a) for w, a in tokens if w not in FUNCTION_WORDS]
        self.words = {w for w, _ in tokens}
        self.stems = {stem(w, a) for w, a in content}
        self.verbs = {verb_base(w, a) for w, a in content} - {""}
        self.bares = {_bare(w, a) for w, a in content} - {""}

    def has(self, word: str, accented: bool = False) -> bool:
        if word in self.words or stem(word, accented) in self.stems:
            return True
        vb = verb_base(word, accented)
        if vb and (vb in self.verbs or vb in self.bares):
            return True
        bare = _bare(word, accented)
        return bool(bare) and bare in self.verbs


def lexicon(text: Any) -> _Lexicon:
    """v79.12.2: the grounding lexicon of an article's text (accents read from the text itself)."""
    return _Lexicon(_tokens(text))


def grounding_words(text: Any) -> List[str]:
    """v79.12.2: the words grounding compares -- content_words without FUNCTION_WORDS."""
    return [w for w in content_words(text) if w not in FUNCTION_WORDS]


def missing_words(passage: Any, text_words: Any) -> List[str]:
    """v79.12.2: the passage's grounding words found in the text neither as such, nor by stem, nor a participle by its
    verb (usata / usa); [] when every content word matches -- the element is explicit. text_words: the text (a
    string, read by lexicon), its lexicon, or its content_words."""
    if isinstance(text_words, _Lexicon):
        lex = text_words
    elif isinstance(text_words, str):
        lex = lexicon(text_words)
    else:
        lex = _Lexicon((w, False) for w in text_words)
    seen: Dict[str, bool] = {}
    for w, a in _tokens(passage):
        if w not in FUNCTION_WORDS and w not in seen:
            seen[w] = a
    return [w for w, a in seen.items() if not lex.has(w, a)]


def _norm_passage(text: Any) -> str:
    return " ".join(str(text or "").replace("’", "'").split()).strip(" ,;:.")


def locate(text: Any, passage: Any) -> Tuple[Optional[int], Optional[int]]:
    """v79.11 (A): (start, end) of the passage in the article's text -- whitespace-normalised, case-insensitive (its
    leading and trailing punctuation left out); (None, None) when it is not there. The offsets are the code's, never
    the model's."""
    needle = _norm_passage(passage).lower()
    if not needle or not text:
        return None, None
    chars: List[str] = []
    where: List[int] = []
    for m in re.finditer(r"\S+", str(text)):
        if chars:
            chars.append(" ")
            where.append(m.start() - 1)
        for i, ch in enumerate(m.group()):
            low = ch.replace("’", "'").lower()
            chars.append(low if len(low) == 1 else ch)
            where.append(m.start() + i)
    pos = "".join(chars).find(needle)
    if pos < 0:
        return None, None
    return where[pos], where[pos + len(needle) - 1] + 1


# v79.11 (A): an alternative in the source passage -- rule_quote's _DISJ (o, od, ovvero, oppure) and ossia/ovverosia,
# «sia … sia», «anche solo»; a list item's «o)» is no disjunction
ALTERNATIVE_RE = re.compile(r"(?<![\w'’])(?:o|od|oppure|ovvero(?:\s+sia)?|ovverosia|ossia)(?![\w'’)])|"
                            r"\bsia\b.+?\bsia\b|\banche\s+solo\b", re.IGNORECASE | re.DOTALL)


def has_alternative(passage: Any) -> bool:
    return bool(ALTERNATIVE_RE.search(" ".join(str(passage or "").split())))


def norm_comma(value: Any) -> str:
    """«1», «co. 2», «comma 2», «1 bis», «1bis» -> "1", "2", "2", "1-bis", "1-bis"; "" for none."""
    if value is None:
        return ""
    t = str(value).strip().lower()
    t = re.sub(r"^(?:co\.|comma)\s*", "", t)
    m = re.match(r"^(\d+)\s*-?\s*([a-z]+)?\.?$", t)
    if m:
        return m.group(1) + (f"-{m.group(2)}" if m.group(2) else "")
    return t


def commi(text: Any) -> Dict[str, str]:
    """v79.11 (A): {comma label: its text} of the article (rule_quote.commi_of, suffixed: «1-bis» its own); an article
    with no numbered commi has only comma "1"."""
    out: Dict[str, str] = {}
    try:
        from .rule_quote import commi_of
        for k, t in commi_of(str(text or ""), suffixed=True):
            out[norm_comma(k) if k is not None else "1"] = t
    except Exception:           # noqa: BLE001 -- the heads' regex below still reads them
        out = {}
    for m in re.finditer(r"(?m)^\s*(\d{1,3}(?:\s*-\s*[a-z]+)?)\.\s", str(text or "")):   # a short comma merged above
        out.setdefault(norm_comma(m.group(1)), "")
    if not out:
        return {"1": " ".join(str(text or "").split())}
    return dict(sorted(out.items(), key=lambda kv: _comma_order(kv[0])))


def _comma_order(label: str) -> Tuple[int, int, str]:
    from .comma_material import SUFFIXES
    m = re.match(r"^(\d+)(?:-([a-z]+))?$", label)
    if not m:
        return (10 ** 6, 0, label)
    suffix = m.group(2) or ""
    return (int(m.group(1)), SUFFIXES.index(suffix) + 1 if suffix in SUFFIXES else (0 if not suffix else 99), label)


def _canon(value: Any, vocabulary: Sequence[Optional[str]]) -> Optional[str]:
    """The vocabulary's own spelling of a value («nesso causale», «Nesso_Causale» -> "nesso_causale"); None if none."""
    if not isinstance(value, str):
        return None
    k = " ".join(value.replace("_", " ").lower().split())
    return next((v for v in vocabulary if v is not None and " ".join(v.replace("_", " ").lower().split()) == k), None)


def canon_kind(kind: Any) -> Optional[str]:
    return _canon(kind, KINDS)


def canon_role(role: Any) -> Optional[str]:
    return _canon(role, ROLES)


def _roles(entries: Any) -> List[Dict[str, Any]]:
    """Role entries as dicts ({"ruolo", "condizione"}); a bare string read as a role with no condition."""
    out = []
    for e in entries or []:
        if isinstance(e, str):
            out.append({"ruolo": e, "condizione": None})
        elif isinstance(e, dict):
            out.append(e)
    return out


# --- checks -----------------------------------------------------------------------------------------------------------

def empty_checks() -> Dict[str, Any]:
    return {"failed": "", "cut": False, **{k: [] for k in CHECK_KEYS}}


class _Text:
    def __init__(self, text: str):
        self.text = str(text or "")
        self.lex = lexicon(self.text)
        self.commi = commi(self.text)

    def span(self, passage: Any) -> Tuple[Optional[int], Optional[int]]:
        return locate(self.text, passage)


def _refs(node: Any, out: List[str], depth: int = 0) -> None:
    if not isinstance(node, dict) or depth > 60:
        return
    if "ref" in node:
        out.append(str(node.get("ref")))
        return
    for key in ("args",):
        for a in node.get(key) or []:
            _refs(a, out, depth + 1)
    for key in ("arg", "if", "then", "else"):
        if node.get(key) is not None:
            _refs(node.get(key), out, depth + 1)


def _logic(node: Any, ids: Sequence[str], where: str, out: Dict[str, Any], any_nodes: List[Dict[str, Any]],
           depth: int = 0) -> None:
    """v79.12.2: a ref to no element is a logic error (blocking: the tree dangles); a node that is no node or an
    unknown op is invalid; a node with no args (all_of / any_of without args, not without arg) is logic_incomplete --
    repair removes it, recursively, and an empty tree is no logic --, as are an any_of of one arg (read as its arg)
    and a conditional without if or then."""
    if depth > 60:
        out["invalid"].append(f"{where}: albero troppo profondo")
        return
    if not isinstance(node, dict):
        out["invalid"].append(f"{where}: nodo non valido")
        return
    if "ref" in node and "op" not in node:
        if str(node.get("ref")) not in ids:
            out["logic_errors"].append(f"{where}: ref {node.get('ref')} senza elemento")
        return
    op = node.get("op")
    if op not in LOGIC_OPS:
        out["invalid"].append(f"{where}: op sconosciuto {op!r}")
        return
    if op in ("all_of", "any_of"):
        args = node.get("args")
        if args is not None and not isinstance(args, list):
            out["invalid"].append(f"{where}: {op} con args non in lista")
            return
        if not args:
            out["logic_incomplete"].append(f"{where}: {op} senza args")
            return
        if op == "any_of":
            if len(args) < 2:
                out["logic_incomplete"].append(f"{where}: any_of con meno di due args")
            else:
                any_nodes.append(node)
        for i, a in enumerate(args):
            _logic(a, ids, f"{where}.{op}[{i}]", out, any_nodes, depth + 1)
    elif op == "not":
        if node.get("arg") is None:
            out["logic_incomplete"].append(f"{where}: not senza arg")
            return
        _logic(node["arg"], ids, f"{where}.not", out, any_nodes, depth + 1)
    else:
        if node.get("if") is None or node.get("then") is None:
            out["logic_incomplete"].append(f"{where}: conditional senza if/then")
        for key in ("if", "then", "else"):
            if node.get(key) is not None:
                _logic(node[key], ids, f"{where}.{key}", out, any_nodes, depth + 1)


def _grounded(name: str, item: Dict[str, Any], ctx: _Text, out: Dict[str, Any]) -> None:
    """An explicit item: its content words found in the text, as such or by stem (missing_words: ungrounded -- repair
    drops it); its fonte.testo found there (fonte_unlocated, reported: repair takes the item's own words when the text
    holds them). An inferred one: a spiegazione and a fonte.comma (never rejected for its wording)."""
    fonte = item.get("fonte") if isinstance(item.get("fonte"), dict) else {}
    if item.get("grounding") == "explicit":
        missing = missing_words(item.get("testo"), ctx.lex)
        if missing:
            out["ungrounded"].append(f"{name}: parole assenti dal testo ({', '.join(missing[:6])})")
        if not str(fonte.get("testo") or "").strip():
            out["fonte_unlocated"].append(f"{name}: fonte.testo mancante")
        elif ctx.span(fonte.get("testo"))[0] is None:
            out["fonte_unlocated"].append(f"{name}: fonte.testo non trovato nel testo")
    elif item.get("grounding") == "inferred":
        if not str(item.get("spiegazione") or "").strip():
            out["inferred_unexplained"].append(f"{name}: spiegazione mancante")
        if not norm_comma(fonte.get("comma")):
            out["inferred_unexplained"].append(f"{name}: fonte.comma mancante")


def _fragment_why(testo: Any) -> str:
    """v79.10.4 (D4): v79.9's fragment kinds (rule_quote.fragment_kind, wider=False) -- what check() reports and
    repair drops; requisites() still drops D4's wider ones alone."""
    if not isinstance(testo, str) or not testo.strip():
        return ""
    from .rule_quote import FRAGMENT_WHY, fragment_kind
    kind = fragment_kind(testo, wider=False)
    return FRAGMENT_WHY[kind] if kind else ""


def _branches_bad(el: Dict[str, Any]) -> bool:
    """Two or more branches and no «o» in the element's passage (its fonte.testo, else its testo)."""
    branches = el.get("branches")
    if not isinstance(branches, list) or len([b for b in branches if str(b or "").strip()]) < 2:
        return False
    fonte = el.get("fonte") if isinstance(el.get("fonte"), dict) else {}
    return not has_alternative(fonte.get("testo") or el.get("testo"))


def _comma_known(name: str, value: Any, ctx: _Text, out: Dict[str, Any]) -> None:
    k = norm_comma(value)
    if not k:
        out["comma_unknown"].append(f"{name}: comma mancante")
    elif k not in ctx.commi:
        out["comma_unknown"].append(f"{name}: comma {k} non tra {', '.join(ctx.commi)}")


def _any_span(node: Dict[str, Any], by_id: Dict[str, Dict[str, Any]], ctx: _Text) -> str:
    """The passage an any_of node chooses within: from its first located element to its last; when one is not
    located (an inferred one), the text of the commi its elements belong to; else the whole text."""
    refs: List[str] = []
    _refs(node, refs)
    els = [by_id[r] for r in refs if r in by_id]
    spans = []
    for el in els:
        fonte = el.get("fonte") if isinstance(el.get("fonte"), dict) else {}
        s, e = ctx.span(fonte.get("testo") or el.get("testo"))
        if s is None:
            s, e = ctx.span(el.get("testo"))
        spans.append((s, e))
    if spans and all(s is not None for s, _ in spans):
        return ctx.text[min(s for s, _ in spans):max(e for _, e in spans)]
    ks = {norm_comma(el.get("comma")) for el in els}
    parts = [ctx.commi.get(k) or "" for k in ks if ctx.commi.get(k)]
    return " ".join(parts) if parts else ctx.text


def _check_one(schema: Dict[str, Any], ctx: _Text, out: Dict[str, Any], prefix: str, top: bool) -> None:
    p = prefix
    tipo = schema.get("tipo")
    if not isinstance(tipo, list) or not tipo:
        out["invalid"].append(f"{p}tipo mancante")
    else:
        bad = [t for t in tipo if t not in TIPI]
        if bad:
            out["invalid"].append(f"{p}tipo sconosciuto {bad}")
    for field in ("destinatario", "legittimato", "soggetto_passivo"):
        value = schema.get(field)
        if value is None:
            continue
        if not isinstance(value, list):
            out["invalid"].append(f"{p}{field} non è una lista")
            continue
        for e in value:
            role = e.get("ruolo") if isinstance(e, dict) else e
            if canon_role(role) is None:
                out["invalid"].append(f"{p}{field}: ruolo sconosciuto {role!r}")
    if schema.get("rapporto") is not None and not isinstance(schema.get("rapporto"), str):
        out["invalid"].append(f"{p}rapporto non è un testo")
    if schema.get("collegati") is not None and (not isinstance(schema.get("collegati"), list) or
                                                not all(isinstance(x, str) for x in schema["collegati"])):
        out["invalid"].append(f"{p}collegati non è una lista di articoli")
    if not top and schema.get("commi_speciali"):
        out["invalid"].append(f"{p}commi_speciali dentro un comma speciale")
    elementi = schema.get("elementi")
    if elementi is None:
        elementi = []
    if not isinstance(elementi, list):
        out["invalid"].append(f"{p}elementi non è una lista")
        elementi = []
    ids: List[str] = []
    by_id: Dict[str, Dict[str, Any]] = {}
    for i, el in enumerate(elementi):
        if not isinstance(el, dict):
            out["invalid"].append(f"{p}elementi[{i}]: non è un oggetto")
            continue
        eid = str(el.get("id") or "").strip()
        name = f"{p}{eid or f'elementi[{i}]'}"
        if not eid:
            out["invalid"].append(f"{name}: id mancante")
        elif eid in by_id:
            out["invalid"].append(f"{name}: id duplicato")
        else:
            ids.append(eid)
            by_id[eid] = el
        if not isinstance(el.get("testo"), str) or not el["testo"].strip():
            out["invalid"].append(f"{name}: testo mancante")
        kind = canon_kind(el.get("kind"))
        if kind is None:
            out["invalid"].append(f"{name}: kind sconosciuto {el.get('kind')!r}")
        elif kind == "other" and not str(el.get("kind_note") or "").strip():
            out["kind_other_unexplained"].append(f"{name}: kind other senza kind_note")
        if el.get("grounding") not in GROUNDING:
            out["invalid"].append(f"{name}: grounding {el.get('grounding')!r} non valido")
        if el.get("connective") not in CONNECTIVES:
            out["invalid"].append(f"{name}: connective {el.get('connective')!r} non valido")
        branches = el.get("branches")
        if branches is not None and (not isinstance(branches, list) or not all(isinstance(b, str) for b in branches)):
            out["invalid"].append(f"{name}: branches non è una lista di testi")
            branches = []
        parti = el.get("parti")
        if parti is not None and (not isinstance(parti, list) or not all(isinstance(x, str) for x in parti)):
            out["invalid"].append(f"{name}: parti non è una lista di testi")
        if el.get("fonte") is not None and not isinstance(el.get("fonte"), dict):
            out["invalid"].append(f"{name}: fonte non è un oggetto")
        _comma_known(name, el.get("comma"), ctx, out)
        _grounded(name, el, ctx, out)
        why = _fragment_why(el.get("testo"))        # v79.12.2: reported; repair drops the element, the schema stands
        if why:
            out["fragments"].append(f"{name}: {why}")
        if branches and _branches_bad(el):
            out["branches_without_alternative"].append(f"{name}: alternative senza «o» nel testo")
    logic = schema.get("logic")
    any_nodes: List[Dict[str, Any]] = []
    if logic is None:
        if ids:
            out["logic_incomplete"].append(f"{p}logic mancante")
    else:
        _logic(logic, ids, f"{p}logic", out, any_nodes)
        referenced: List[str] = []
        _refs(logic, referenced)
        for eid in ids:
            if eid not in referenced:
                out["logic_incomplete"].append(f"{p}{eid}: mai citato nella logic")
    for node in any_nodes:
        if not has_alternative(_any_span(node, by_id, ctx)):
            refs: List[str] = []
            _refs(node, refs)
            out["branches_without_alternative"].append(f"{p}any_of({', '.join(refs)}): nessuna alternativa nel testo")
    eccezioni = schema.get("eccezioni")
    if eccezioni is not None and not isinstance(eccezioni, list):
        out["invalid"].append(f"{p}eccezioni non è una lista")
        eccezioni = []
    for i, ex in enumerate(eccezioni or []):
        name = f"{p}eccezione {i + 1}"
        if not isinstance(ex, dict):
            out["invalid"].append(f"{name}: non è un oggetto")
            continue
        if not isinstance(ex.get("testo"), str) or not ex["testo"].strip():
            out["invalid"].append(f"{name}: testo mancante")
        if ex.get("grounding") not in GROUNDING:
            out["invalid"].append(f"{name}: grounding {ex.get('grounding')!r} non valido")
        _comma_known(name, ex.get("comma"), ctx, out)
        _grounded(name, ex, ctx, out)
    if schema.get("conseguenza") is not None and not isinstance(schema.get("conseguenza"), dict):
        out["invalid"].append(f"{p}conseguenza non è un oggetto")


def check(schema: Any, text: str) -> Dict[str, Any]:
    """v79.11 (A): the checks of a schema against its article's text (spec §4), over the schema and every
    commi_speciali sub-schema; each problem a short string naming the id. failed/cut are the generator's. v79.12.2:
    BLOCKING (invalid, logic_errors = dangling refs) make it unusable; ELEMENT_CHECKS are what repair() fixes or
    drops; the rest is reported."""
    out = empty_checks()
    if not isinstance(schema, dict):
        out["invalid"].append("schema non è un oggetto")
        return out
    ctx = _Text(text)
    _check_one(schema, ctx, out, "", True)
    special = schema.get("commi_speciali")
    if special is not None and not isinstance(special, list):
        out["invalid"].append("commi_speciali non è una lista")
        special = []
    for i, cs in enumerate(special or []):
        if not isinstance(cs, dict):
            out["invalid"].append(f"commi_speciali[{i}]: non è un oggetto")
            continue
        k = norm_comma(cs.get("comma"))
        name = f"co.{k or i}"
        if cs.get("kind") not in SPECIAL_KINDS:
            out["invalid"].append(f"{name}: kind speciale {cs.get('kind')!r} non valido")
        _comma_known(name, cs.get("comma"), ctx, out)
        if not isinstance(cs.get("schema"), dict):
            out["invalid"].append(f"{name}: schema mancante")
            continue
        _check_one(cs["schema"], ctx, out, f"{name}/", False)
    return out


# --- the repair (v79.12.2) --------------------------------------------------------------------------------------------

def prune_logic(node: Any, dropped: Iterable[str] = (), depth: int = 0) -> Any:
    """v79.12.2: the logic tree without the refs to the dropped ids and without the nodes left with no args, removed
    recursively (an all_of / any_of with no args, a not with no arg, a conditional with no if, then or else left -- a
    pruned branch's key left out, a live else kept with its node); an any_of
    of one arg read as its arg. None when nothing is left -- no logic, not an error. A node that is no node, an unknown
    op or a dangling ref is left as it is (check() says so)."""
    gone = dropped if isinstance(dropped, (set, frozenset)) else set(dropped)
    if depth > 60 or not isinstance(node, dict):
        return node
    if "ref" in node and "op" not in node:
        return None if str(node.get("ref")) in gone else node
    op = node.get("op")
    if op in ("all_of", "any_of"):
        args = node.get("args")
        if args is not None and not isinstance(args, list):
            return node
        kept = [x for x in (prune_logic(a, gone, depth + 1) for a in args or []) if x is not None]
        if not kept:
            return None
        if op == "any_of" and len(kept) == 1:
            return kept[0]
        return {**node, "args": kept}
    if op == "not":
        arg = prune_logic(node.get("arg"), gone, depth + 1) if node.get("arg") is not None else None
        return None if arg is None else {**node, "arg": arg}
    if op == "conditional":
        out = dict(node)
        for key in ("if", "then", "else"):
            if node.get(key) is not None:
                out[key] = prune_logic(node[key], gone, depth + 1)
            if out.get(key) is None:
                out.pop(key, None)              # a branch pruned away is left out, never kept as null
        if not any(k in out for k in ("if", "then", "else")):
            return None                         # no branch left: the node has no args
        return out                              # a live branch kept (a missing if/then: logic_incomplete)
    return node


def _any_to_all(node: Any, by_id: Dict[str, Dict[str, Any]], ctx: _Text, scope: str,
                actions: List[Dict[str, Any]], depth: int = 0) -> Any:
    """An any_of whose passage holds no alternative (has_alternative over _any_span) read as an all_of, logged."""
    if depth > 60 or not isinstance(node, dict) or "op" not in node:
        return node
    out = dict(node)
    if isinstance(node.get("args"), list):
        out["args"] = [_any_to_all(a, by_id, ctx, scope, actions, depth + 1) for a in node["args"]]
    for key in ("arg", "if", "then", "else"):
        if isinstance(node.get(key), dict):
            out[key] = _any_to_all(node[key], by_id, ctx, scope, actions, depth + 1)
    if out.get("op") == "any_of" and len(out.get("args") or []) >= 2 and not has_alternative(_any_span(out, by_id, ctx)):
        refs: List[str] = []
        _refs(out, refs)
        out["op"] = "all_of"
        actions.append({"scope": scope, "id": f"any_of({', '.join(refs)})", "check": "branches_without_alternative",
                        "action": "fixed", "why": "nessuna alternativa nel testo: letto come all_of", "testo": "",
                        "comma": scope})
    return out


def _repair_one(sc: Dict[str, Any], ctx: _Text, scope: str, actions: List[Dict[str, Any]]) -> None:
    def act(item: Dict[str, Any], eid: str, check_key: str, action: str, why: str) -> None:
        actions.append({"scope": scope, "id": eid, "check": check_key, "action": action, "why": why,
                        "testo": " ".join(str(item.get("testo") or "").split()),
                        "comma": norm_comma(item.get("comma")) or scope or "1"})
    elementi = sc.get("elementi")
    dropped: set = set()
    if isinstance(elementi, list):
        kept = []
        for i, el in enumerate(elementi):
            if not isinstance(el, dict):
                kept.append(el)                 # invalid: check() says so, nothing to repair
                continue
            eid = str(el.get("id") or "").strip() or f"elementi[{i}]"
            why = _fragment_why(el.get("testo"))
            if why:
                act(el, eid, "fragments", "dropped", why)
                dropped.add(eid)
                continue
            if el.get("grounding") == "explicit":
                missing = missing_words(el.get("testo"), ctx.lex)
                if missing:
                    act(el, eid, "ungrounded", "dropped", f"parole assenti dal testo ({', '.join(missing[:6])})")
                    dropped.add(eid)
                    continue
                fonte = el.get("fonte")
                if fonte is None or isinstance(fonte, dict):
                    passage = str((fonte or {}).get("testo") or "")
                    if not passage.strip() or ctx.span(passage)[0] is None:
                        s, e = ctx.span(el.get("testo"))
                        if s is not None:
                            el["fonte"] = {**(fonte or {}), "comma": (fonte or {}).get("comma") or el.get("comma"),
                                           "testo": " ".join(str(el.get("testo")).split()), "start": s, "end": e}
                            act(el, eid, "fonte_unlocated", "fixed", "fonte.testo preso dal testo dell'elemento")
            if isinstance(el.get("branches"), list) and _branches_bad(el):
                el["branches"] = []
                if el.get("connective") == "o":
                    el["connective"] = None
                act(el, eid, "branches_without_alternative", "fixed", "alternative tolte: nessuna «o» nel testo")
            kept.append(el)
        sc["elementi"] = kept
    eccezioni = sc.get("eccezioni")
    if isinstance(eccezioni, list):
        kept_ex = []
        for i, ex in enumerate(eccezioni):
            if isinstance(ex, dict) and ex.get("grounding") == "explicit":
                missing = missing_words(ex.get("testo"), ctx.lex)
                if missing:
                    act(ex, f"eccezione {i + 1}", "ungrounded", "dropped",
                        f"parole assenti dal testo ({', '.join(missing[:6])})")
                    continue
            kept_ex.append(ex)
        sc["eccezioni"] = kept_ex
    logic = sc.get("logic")
    if logic is not None:
        pruned = prune_logic(logic, dropped)
        if pruned != logic:
            actions.append({"scope": scope, "id": "logic", "check": "logic_incomplete", "action": "fixed",
                            "why": "albero vuoto: nessuna logica" if pruned is None else
                            "nodi vuoti o riferimenti a elementi tolti rimossi", "testo": "", "comma": scope})
        by_id = {str(e.get("id")): e for e in sc.get("elementi") or [] if isinstance(e, dict)}
        sc["logic"] = _any_to_all(pruned, by_id, ctx, scope, actions)


def repair(schema: Any, text: str) -> Tuple[Any, List[Dict[str, Any]]]:
    """v79.12.2: (a copy of the schema with each offending element fixed or dropped, what was done) -- over the schema
    and every commi_speciali sub-schema: a fragment (fragments) and an explicit element or eccezione whose content words
    are not all in the text by stem (ungrounded) dropped; branches with no «o» in the passage cleared, an any_of over no
    alternative read as an all_of (branches_without_alternative); an explicit element's unlocated fonte.testo replaced
    by its own words when the text holds them (fonte_unlocated); the logic pruned of the dropped elements' refs and of
    the nodes left empty (prune_logic). Each action {"scope", "id", "check", "action": "dropped" | "fixed", "why",
    "testo", "comma"}. The schema is kept; only invalid and a dangling ref make it unusable. Never the reviewed
    schemas' (the lawyer's word)."""
    if not isinstance(schema, dict):
        return schema, []
    out = json.loads(json.dumps(schema, ensure_ascii=False))
    ctx = _Text(text)
    actions: List[Dict[str, Any]] = []
    _repair_one(out, ctx, "", actions)
    for cs in out.get("commi_speciali") if isinstance(out.get("commi_speciali"), list) else []:
        if isinstance(cs, dict) and isinstance(cs.get("schema"), dict):
            _repair_one(cs["schema"], ctx, norm_comma(cs.get("comma")) or "?", actions)
    return out, actions


def raised(checks: Optional[Dict[str, Any]], names: Sequence[str] = BLOCKING) -> List[str]:
    """The names of the checks (of `names`) that raised."""
    return [k for k in names if (checks or {}).get(k)]


def blocking(line: Optional[Dict[str, Any]]) -> List[str]:
    """v79.11 (A): why a line is unusable -- its blocking checks raised; "schema" when it holds none, "schema_version"
    for another version's; [] for a usable one."""
    if not isinstance(line, dict) or not isinstance(line.get("schema"), dict):
        return ["schema"]
    if line.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
        return ["schema_version"]
    return raised(line.get("checks"))


def usable(line: Optional[Dict[str, Any]]) -> bool:
    """v79.11 (A): a schema and no blocking check raised."""
    return not blocking(line)


def ground(schema: Any, text: str) -> Any:
    """v79.11 (A): a copy of the schema with every fonte's start/end computed from the text (locate) -- the model's
    offsets never kept."""
    if not isinstance(schema, dict):
        return schema
    out = json.loads(json.dumps(schema, ensure_ascii=False))

    def fix(items):
        for item in items or []:
            if isinstance(item, dict) and isinstance(item.get("fonte"), dict):
                s, e = locate(text, item["fonte"].get("testo"))
                item["fonte"]["start"], item["fonte"]["end"] = s, e

    def one(sc):
        if isinstance(sc, dict):
            fix(sc.get("elementi") if isinstance(sc.get("elementi"), list) else [])
            fix(sc.get("eccezioni") if isinstance(sc.get("eccezioni"), list) else [])
    one(out)
    for cs in out.get("commi_speciali") if isinstance(out.get("commi_speciali"), list) else []:
        if isinstance(cs, dict):
            one(cs.get("schema"))
    return out


# --- the lookup -------------------------------------------------------------------------------------------------------

_FRESH: Dict[Tuple[str, str], Dict[str, Any]] = {}
_REPAIRED: Dict[Tuple[str, str], Tuple[Any, List[Dict[str, Any]]]] = {}


def _memo_key(schema: Any, text: str) -> Tuple[str, str]:
    return (hashlib.sha1(json.dumps(schema, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest(),
            text_hash(text))


def _fresh(schema: Dict[str, Any], text: str) -> Dict[str, Any]:
    """check() re-run on the line as read (memoised): the checks of today's rules apply, whatever the line was made
    with -- a stricter one blocks it, a laxer one (v79.12.2's element checks) no longer does."""
    key = _memo_key(schema, text)
    if key not in _FRESH:
        if len(_FRESH) > 4096:
            _FRESH.clear()
        _FRESH[key] = check(schema, text)
    return _FRESH[key]


TEXT_FREE = ("invalid", "logic_errors", "logic_incomplete")     # the checks that read no article text


def current_checks(schema: Any, stored: Optional[Dict[str, Any]], text: Optional[str] = None) -> Dict[str, Any]:
    """v79.12.2: a line's checks by today's rules -- re-run on its text when there is one; without it, the recorded
    ones with the checks that need no text (TEXT_FREE) re-run, so an older line's «mai citato» or «any_of con meno di
    due args», once filed under logic_errors, no longer blocks it. failed/cut always the generator's, as recorded."""
    stored = stored if isinstance(stored, dict) else {}
    if not isinstance(schema, dict):
        out = {**empty_checks(), **stored}
    elif str(text or "").strip():
        out = dict(_fresh(schema, str(text)))
    else:
        out = {**empty_checks(), **stored}
        fresh = _fresh(schema, "")
        for k in TEXT_FREE:
            out[k] = list(fresh[k])
    out["failed"], out["cut"] = str(stored.get("failed") or ""), bool(stored.get("cut"))
    return out


def _repaired(schema: Dict[str, Any], text: str, label: str = "") -> Tuple[Any, List[Dict[str, Any]]]:
    """repair() memoised; what it dropped or fixed logged once."""
    key = _memo_key(schema, text)
    if key not in _REPAIRED:
        if len(_REPAIRED) > 4096:
            _REPAIRED.clear()
        _REPAIRED[key] = repair(schema, text)
        actions = _REPAIRED[key][1]
        if actions:
            logger.info("element schemas: %s -- %s", label or "?", "; ".join(
                f"{a['check']} {a['action']} {('co' + a['scope'] + '.') if a['scope'] else ''}{a['id']}"
                for a in actions))
    return _REPAIRED[key]


def _notes(*lines: Optional[Dict[str, Any]]) -> List[Any]:
    out: List[Any] = []
    seen = set()
    for line in lines:
        for n in (line or {}).get("note_di_inquadramento") or []:
            k = json.dumps(n, sort_keys=True, ensure_ascii=False)
            if k not in seen:
                seen.add(k)
                out.append(n)
    return out


def _is_reviewed(line: Optional[Dict[str, Any]]) -> bool:
    return bool(line) and line.get("reviewed") is True and isinstance(line.get("schema"), dict)


def _pick(label: str, gen: Optional[Dict[str, Any]], text: Optional[str]) -> Tuple[Optional[Dict[str, Any]], str]:
    """The line and the path reason for an article: the reviewed overlay while its source is the article's, else the
    generated line when usable and of this text; else None and why."""
    overlay = reviewed().get(label) if label else None
    known = text is not None
    h = text_hash(text) if known else None
    if _is_reviewed(overlay):
        base = h if known else ((gen or {}).get("current_source_hash") or  # v79.11: the generator records the text
                                (gen or {}).get("source_hash", overlay.get("source_hash")))  # now beside a reviewed label
        if base and overlay.get("source_hash") == base:
            bad = raised(current_checks(overlay["schema"], overlay.get("checks"), text), STRUCTURAL)
            if bad:
                return None, f"unusable:{bad[0]}"
            return {**overlay, "source": "reviewed", "note_di_inquadramento": _notes(overlay)}, "reviewed"
        return None, "reviewed_stale"
    if gen is None:
        return None, "no_schema"
    if known and gen.get("source_hash") != h:
        return None, "hash_mismatch"
    bad = blocking(gen)
    if bad not in (["schema"], ["schema_version"]):
        # v79.12.2: the generator's own (failed, cut) as recorded; the rest re-checked with today's rules -- on the
        # text (the row's, else the line's own), without one the checks that need no text (current_checks)
        bad = raised(current_checks(gen["schema"], gen.get("checks"), text if known else gen.get("text")))
    if bad:
        return None, f"unusable:{bad[0]}"
    base = text if known else str(gen.get("text") or "")
    schema, dropped = _repaired(gen["schema"], base, label) if base.strip() else (gen["schema"], [])
    return {**gen, "schema": schema, "dropped": dropped, "base_comma": base_comma(gen["schema"]), "source": "schema",
            "reviewed": False, "note_di_inquadramento": _notes(gen, overlay)}, "schema"


def for_row(row: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
    """v79.11 (A): (the schema-bearing line, path reason) of an article row -- keyed (row["d"]["name"], the article
    number), else by its label; matched against the RAW text's hash (row["s"]["plain_text"]). Reasons: "schema",
    "reviewed", "off", "no_schema", "hash_mismatch", "unusable:<check>", "reviewed_stale"."""
    if not enabled():
        return None, "off"
    ref = row.get("_article") if isinstance(row, dict) else None
    if not ref:
        return None, "no_schema"
    raw = str(((row.get("s") or {}).get("plain_text")) or "")
    label = norm_label(label_of(ref[0], ref[1]))
    gen = generated().get(_key((row.get("d") or {}).get("name"), ref[0])) or generated_by_label().get(label)
    if not raw.strip():
        return None, "hash_mismatch" if (gen or _is_reviewed(reviewed().get(label))) else "no_schema"
    return _pick(label, gen, raw)


def for_label(label: str, text: Optional[str] = None) -> Tuple[Optional[dict], str]:
    """v79.11 (A): for_row by label (code articles); with the text, its hash is checked; without it, a reviewed
    overlay is stale when the generated line's source_hash differs from its own."""
    if not enabled():
        return None, "off"
    key = norm_label(label)
    return _pick(key, generated_by_label().get(key), text)


# --- what the pipeline reads -------------------------------------------------------------------------------------------

def _int_comma(value: Any) -> int:
    """issue_records._comma's reading: an int, a non-digit label as 1."""
    k = norm_comma(value)
    return int(k) if k.isdigit() else 1


def base_comma(schema: Any) -> str:
    """The article's base comma: the comma of the top schema's first element ("1" when none)."""
    elementi = schema.get("elementi") if isinstance(schema, dict) else None
    if not isinstance(elementi, list):
        return "1"
    return next((norm_comma(e.get("comma")) for e in elementi if isinstance(e, dict)), "") or "1"


def _schemas(line: Dict[str, Any]) -> List[Tuple[Dict[str, Any], str, str]]:
    """[(schema, special kind, scope)] -- the top schema ("" kind) and every commi_speciali sub-schema."""
    schema = line.get("schema") if isinstance(line, dict) else None
    if not isinstance(schema, dict):
        return []
    out = [(schema, "", "")]
    for cs in schema.get("commi_speciali") or []:
        if isinstance(cs, dict) and isinstance(cs.get("schema"), dict):
            out.append((cs["schema"], str(cs.get("kind") or "other"), norm_comma(cs.get("comma"))))
    return out


def requisites(row: Dict[str, Any], line: Dict[str, Any], analysis: Optional[Dict[str, Any]],
               fragments: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, Any]]:
    """v79.11 (A): the line's elements as issue_records._requisites' element dicts (kind "requisito", comma an int,
    the facts' status, "split": "schema", branches when two or more, "special": the commi_speciali kind for a special
    comma's elements), plus schema_id, schema_kind, schema_connective, schema_parti, schema_grounding. A fragment
    (rule_quote.fragment_why) is dropped, kept in `fragments` and logged as today; kinds quantificazione, competenza,
    onere_della_prova are no requisites (summary keeps them); at most issue_records.REQUISITES_PER_ARTICLE."""
    from . import rule_quote
    from .interactive_flow import facts_status
    from .issue_records import REQUISITES_PER_ARTICLE, _log_fragment
    parts = _schemas(line)
    if not parts:
        return []
    for a in line.get("dropped") or []:           # v79.12.2: what repair dropped, logged as the fragments always were
        if not isinstance(a, dict) or a.get("action") != "dropped":
            continue
        if a.get("check") == "fragments":
            k = _int_comma(a.get("comma"))
            if fragments is not None:
                fragments.append({"comma": str(k), "text": str(a.get("testo") or "")[:120], "why": a.get("why") or ""})
            _log_fragment(row, str(a.get("testo") or ""), str(a.get("why") or ""))
        else:
            _log_dropped(row, a)
    # v79.12.2: the base comma as the model's schema had it -- a repair that drops the top's comma-1 elements never
    # makes a special comma (an aggravamento) the base and its elements plain requisites
    first = norm_comma(line.get("base_comma")) or base_comma(parts[0][0])
    special_of = {scope: kind for _, kind, scope in parts[1:] if scope and scope != first}
    out: List[Dict[str, Any]] = []
    seen = set()
    for schema, kind, scope in parts:
        if scope and scope == first:           # the base comma is never special
            kind = ""
        for el in schema.get("elementi") or []:
            if not isinstance(el, dict):
                continue
            el_kind = canon_kind(el.get("kind"))
            if el_kind in NOT_REQUISITE_KINDS:
                continue
            text = " ".join(str(el.get("testo") or "").split())
            comma = norm_comma(el.get("comma")) or scope or "1"
            if not text or (comma, text.lower()) in seen or len(out) >= REQUISITES_PER_ARTICLE:
                continue
            seen.add((comma, text.lower()))
            k = _int_comma(comma)
            why = rule_quote.fragment_why(text)
            if why:
                if fragments is not None:
                    fragments.append({"comma": str(k), "text": text[:120], "why": why})
                _log_fragment(row, text, why)
                continue
            special = kind if scope else special_of.get(comma, "")
            status = facts_status(text, analysis or {})
            alternatives = [" ".join(str(b).split()) for b in el.get("branches") or [] if str(b or "").strip()]
            eid = str(el.get("id") or "")
            out.append({"id": "", "comma": k, "text": text, "kind": "requisito", "facts_status": status,
                        "status": status, "why": "", "supporting": [], "contradicting": [], "missing": "",
                        "question": "", "split": "schema",
                        **({"branches": alternatives} if len(alternatives) >= 2 else {}),
                        **({"special": special} if special else {}),
                        "schema_id": f"co{scope}.{eid}" if scope else eid, "schema_kind": el_kind,
                        "schema_connective": el.get("connective"),
                        "schema_parti": [str(x) for x in el.get("parti") or [] if str(x or "").strip()],
                        "schema_grounding": el.get("grounding")})
    return out


def _log_dropped(row: Dict[str, Any], action: Dict[str, Any]) -> None:
    """v79.12.2: an element the repair dropped (ungrounded), logged; never fails the decomposition."""
    try:
        from .interactive_flow import _log
        _log("schema_element_dropped", {"article": " ".join(str(x) for x in (row or {}).get("_article") or ()),
                                         "id": action.get("id"), "check": action.get("check"),
                                         "text": str(action.get("testo") or "")[:120], "why": action.get("why")})
    except Exception:           # noqa: BLE001
        pass


def summary(line: Dict[str, Any]) -> Dict[str, Any]:
    """v79.11 (A): what the record keeps of its schema (rec["schema"])."""
    schema = (line or {}).get("schema") or {}
    is_rev = _is_reviewed(line) or (line or {}).get("source") == "reviewed"
    others = []
    for sc, _, scope in _schemas(line or {}):
        for el in sc.get("elementi") or []:
            if isinstance(el, dict) and canon_kind(el.get("kind")) in NOT_REQUISITE_KINDS:
                others.append({"id": f"co{scope}.{el.get('id')}" if scope else el.get("id"),
                               "testo": el.get("testo"), "kind": canon_kind(el.get("kind")), "comma": el.get("comma")})
    return {"source": "reviewed" if is_rev else "schema", "tipo": list(schema.get("tipo") or []),
            "legittimato": _roles(schema.get("legittimato")), "destinatario": _roles(schema.get("destinatario")),
            "soggetto_passivo": _roles(schema.get("soggetto_passivo")), "rapporto": schema.get("rapporto"),
            "logic": schema.get("logic"), "reviewed": is_rev,
            "schema_version": (line or {}).get("schema_version", SCHEMA_VERSION),
            "eccezioni": len(schema.get("eccezioni") or []),
            "note_di_inquadramento": list((line or {}).get("note_di_inquadramento") or []),
            "commi_speciali": [{"comma": norm_comma(cs.get("comma")), "kind": cs.get("kind")}
                               for cs in schema.get("commi_speciali") or [] if isinstance(cs, dict)],
            "altri_elementi": others,
            **({"dropped": [{"id": f"co{a.get('scope')}.{a.get('id')}" if a.get("scope") else a.get("id"),
                             "check": a.get("check"), "action": a.get("action")}
                            for a in (line or {}).get("dropped") or [] if isinstance(a, dict)]}
               if (line or {}).get("dropped") else {})}


def kind_rule(kind: Any) -> str:
    """v79.11 (A): the grading rule a schema kind selects (KIND_RULE), "plain" for the others."""
    return KIND_RULE.get(canon_kind(kind) or "", "plain")


# --- roles and admission (shadow) -------------------------------------------------------------------------------------

# the intake role -> the schema roles it is (a match) and those it may be (a conditional match)
_ROLE_SAME = {"attore": ("attore",), "convenuto": ("convenuto",), "imputato": ("imputato", "indagato"),
              "indagato": ("indagato", "imputato"), "persona_offesa": ("persona offesa", "parte civile"),
              "parte_civile": ("parte civile", "persona offesa"), "terzo": ("terzo",)}
_ROLE_MAYBE = {"attore": ("persona offesa", "parte civile"), "persona_offesa": ("attore",),
               "parte_civile": ("attore",)}
_QUALITIES = ("creditore", "debitore", "lavoratore", "datore di lavoro", "consumatore", "professionista", "PA",
              "altro")


def _intake_role(role: Any) -> str:
    return "_".join(str(role or "").strip().lower().split()) or "sconosciuto"


def roles_match(client_role: str, entries: Sequence[Any]) -> Tuple[bool, bool]:
    """v79.11 (A): (match, conditional) of the client's intake role against a schema's role entries -- «chiunque»
    matches anyone; imputato/indagato and persona offesa/parte civile are one another's; a substantive quality
    (creditore, consumatore, PA, altro …) or a role with a condizione is a conditional match; an unknown client role
    is never excluded (conditional)."""
    role = _intake_role(client_role)
    sure = maybe = False
    for e in _roles(entries):
        r = canon_role(e.get("ruolo"))
        if r is None:
            continue
        cond = bool(str(e.get("condizione") or "").strip())
        if r == "chiunque" or r in _ROLE_SAME.get(role, ()):
            hit, conditional = True, cond
        elif role == "sconosciuto" or r in _QUALITIES or r in _ROLE_MAYBE.get(role, ()):
            hit, conditional = True, True
        else:
            hit, conditional = False, False
        if hit and not conditional:
            sure = True
        elif hit:
            maybe = True
    return (True, False) if sure else ((True, True) if maybe else (False, False))


def _names(entries: Any) -> str:
    return ", ".join(str(e.get("ruolo")) for e in _roles(entries)) or "nessuno"


def shadow_admission(label: str, line: Optional[Dict[str, Any]], role: str, side: Optional[str]) -> Dict[str, Any]:
    """v79.11 (A, §7): what the schema would admit for the client -- claim (claim-conferring, the client legittimato,
    the client's own norm), defence (defence/scriminante, the client legittimato or accused), contest (the
    counterparty's sanction or claim, or a limiting norm for the convenuto); each refusal with its reason. Shadow
    only: the 79.8 admission decides."""
    if not isinstance(line, dict) or not isinstance(line.get("schema"), dict):
        return {"article": label, "schema": False, "tipo": [], "legittimato": [], "claim": None, "defence": None,
                "contest": None, "conditional": False, "why": ["no_schema"]}
    schema = line["schema"]
    tipo = [t for t in schema.get("tipo") or [] if isinstance(t, str)]
    leg = _roles(schema.get("legittimato"))
    r = _intake_role(role)
    match, conditional = roles_match(r, leg)
    threat = side == "threat"
    why: List[str] = []
    tipi = ", ".join(tipo) or "nessuno"
    if "claim-conferring" not in tipo:
        claim = False
        why.append(f"claim: tipo {tipi} non conferisce azione")
    elif not match:
        claim = False
        why.append(f"claim: legittimato {_names(leg)} non comprende {r}")
    elif threat:
        claim = False
        why.append("claim: norma della controparte")
    else:
        claim = True
    accused = r in ("imputato", "indagato", "convenuto") and any(
        canon_role(e.get("ruolo")) in ("imputato", "convenuto") for e in leg)
    if not ({"defence", "scriminante"} & set(tipo)):
        defence = False
        why.append(f"defence: tipo {tipi} non è difesa né scriminante")
    elif not (match or accused):
        defence = False
        why.append(f"defence: legittimato {_names(leg)} non comprende {r}")
    else:
        defence = True
    if threat and ({"sanction", "claim-conferring"} & set(tipo)):
        contest = True
    elif "limiting" in tipo and r == "convenuto":
        contest = True
    else:
        contest = False
        if not threat and "limiting" not in tipo:
            why.append("contest: norma del cliente, non della controparte")
        elif threat:
            why.append(f"contest: tipo {tipi} non è sanzione né azione della controparte")
        else:
            why.append(f"contest: norma limitativa, ma il cliente è {r}, non convenuto")
    return {"article": label, "schema": True, "tipo": tipo, "legittimato": leg, "rapporto": schema.get("rapporto"),
            "reviewed": _is_reviewed(line) or line.get("source") == "reviewed", "claim": claim, "defence": defence,
            "contest": contest, "conditional": bool(match and conditional and (claim or defence)), "why": why}


# --- the general-part gate ---------------------------------------------------------------------------------------------

def opinion(line: Optional[Dict[str, Any]]) -> Optional[bool]:
    """The schema's own view: True for a decomposable tipo (sanction, claim-conferring, defence, scriminante), False
    for one only limiting/definitional/procedural/other, None without a tipo."""
    tipo = ((line or {}).get("schema") or {}).get("tipo") if isinstance(line, dict) else None
    if not isinstance(tipo, list) or not tipo:
        return None
    return bool(set(tipo) & set(DECOMPOSABLE_TIPI))


def gate(label: str, line: Optional[Dict[str, Any]]) -> Optional[bool]:
    """v79.11 (A, §6): the schema's gate -- True for a decomposable tipo; False only for a REVIEWED schema whose tipo is
    only limiting/definitional/procedural/other (an unreviewed one never gates out); None for no opinion (flag off,
    no schema, an unreviewed non-decomposable one). The hard invariants (hard_excluded) are the caller's."""
    if not enabled() or not isinstance(line, dict):
        return None
    view = opinion(line)
    if view is None:
        return None
    if view:
        return True
    return False if (_is_reviewed(line) or line.get("source") == "reviewed") else None


def hard_excluded(label: str) -> bool:
    """v79.11 (A): the v64 invariants no schema overrides -- a procedural code's article (issue_records.
    PROCEDURAL_CODES), the general provisions (GENERAL_CODES), c.p. Libro I (below OFFENCES_FROM) other than the
    justifications (50-54)."""
    from . import issue_records
    parts = str(label or "").split(" ", 1)
    if len(parts) < 2:
        return False
    code = _code(parts[1])
    if code in issue_records.PROCEDURAL_CODES or code in issue_records.GENERAL_CODES:
        return True
    m = re.match(r"\d+", parts[0])
    return code == "c.p." and bool(m) and int(m.group(0)) < issue_records.OFFENCES_FROM and \
        not issue_records.justification(label)


def decide(label: str, line: Optional[Dict[str, Any]], current: bool) -> Tuple[bool, str, bool]:
    """v79.11 (A): (decomposed, by, disagree) -- the gate with a schema: "schema" when gate() decides, "invariant" when
    a hard invariant keeps the current gate's answer over the schema's True, "current" when the schema has no opinion
    (or agrees); disagree when the schema's view differs from the current gate's."""
    g = gate(label, line)
    view = opinion(line) if isinstance(line, dict) and enabled() else None
    disagree = view is not None and view != bool(current)
    if g is None:
        return bool(current), "current", disagree
    if g and hard_excluded(label):
        return bool(current), "invariant", disagree
    return g, "schema", disagree


# --- quality (§9) -----------------------------------------------------------------------------------------------------

MATCH_JACCARD = 0.6


def _ntext(text: Any) -> str:
    return " ".join(_fold(text).replace("’", "'").split()).strip(" ,;:.")


def _similar(a: Any, b: Any) -> float:
    if _ntext(a) and _ntext(a) == _ntext(b):
        return 1.0
    x, y = set(content_words(a)), set(content_words(b))
    return len(x & y) / len(x | y) if x and y else 0.0


def _match(gold: Sequence[Any], pred: Sequence[Any], text) -> List[Tuple[int, int]]:
    """One-to-one (gold index, predicted index) pairs: same normalised text or content-word Jaccard >= 0.6, best first."""
    pairs = sorted(((s, i, j) for i, g in enumerate(gold) for j, p in enumerate(pred)
                    for s in [_similar(text(g), text(p))] if s >= MATCH_JACCARD), key=lambda x: (-x[0], x[1], x[2]))
    used_g, used_p, out = set(), set(), []
    for _, i, j in pairs:
        if i not in used_g and j not in used_p:
            used_g.add(i)
            used_p.add(j)
            out.append((i, j))
    return out


def _flat(schema: Any) -> List[Tuple[str, Dict[str, Any]]]:
    return [(scope, el) for sc, _, scope in _schemas({"schema": schema}) for el in sc.get("elementi") or []
            if isinstance(el, dict)]


def canonical(node: Any, names: Dict[Tuple[str, str], str], scope: str = "", depth: int = 0) -> Any:
    """A logic tree's canonical form: refs renamed (names), all_of/any_of flattened and ordered, a one-argument
    all_of/any_of read as its argument."""
    if depth > 60 or not isinstance(node, dict):
        return ("?",)
    if "ref" in node and "op" not in node:
        rid = str(node.get("ref"))
        return ("ref", names.get((scope, rid), f"?{scope}:{rid}"))
    op = node.get("op")
    if op in ("all_of", "any_of"):
        args = []
        for a in node.get("args") or []:
            c = canonical(a, names, scope, depth + 1)
            args.extend(c[1] if c[0] == op else [c])
        if len(args) == 1:
            return args[0]
        return (op, tuple(sorted(args, key=repr)))
    if op == "not":
        return ("not", canonical(node.get("arg"), names, scope, depth + 1))
    if op == "conditional":
        return ("conditional", canonical(node.get("if"), names, scope, depth + 1),
                canonical(node.get("then"), names, scope, depth + 1),
                canonical(node.get("else"), names, scope, depth + 1) if node.get("else") is not None else None)
    return ("?", str(op))


def _logic_equal(gold: Dict[str, Any], pred: Dict[str, Any], mapping: Dict[Tuple[str, str], str]) -> bool:
    gnames = {(scope, str(el.get("id"))): f"{scope}:{el.get('id')}" for scope, el in _flat(gold)}
    gold_trees = {scope: canonical(sc.get("logic"), gnames, scope) for sc, _, scope in _schemas({"schema": gold})}
    pred_trees = {scope: canonical(sc.get("logic"), mapping, scope) for sc, _, scope in _schemas({"schema": pred})}
    return gold_trees == pred_trees


def quality(generated_schema: Any, reviewed_schema: Any) -> Dict[str, Any]:
    """v79.11 (A, §9): the model's schema (prediction) against the lawyer's (gold), one article -- counts and ratios:
    element_precision, element_recall, logic_accuracy (1/0), exception_recall, comma_accuracy, grounding_accuracy."""
    gold = reviewed_schema if isinstance(reviewed_schema, dict) else {}
    pred = generated_schema if isinstance(generated_schema, dict) else {}
    g_el, p_el = _flat(gold), _flat(pred)
    pairs = _match(g_el, p_el, lambda x: x[1].get("testo"))
    mapping = {(p_el[j][0], str(p_el[j][1].get("id"))): f"{g_el[i][0]}:{g_el[i][1].get('id')}" for i, j in pairs}
    comma_same = sum(1 for i, j in pairs if norm_comma(g_el[i][1].get("comma")) == norm_comma(p_el[j][1].get("comma")))
    ground_same = sum(1 for i, j in pairs if g_el[i][1].get("grounding") == p_el[j][1].get("grounding"))
    g_ex = [x for x in gold.get("eccezioni") or [] if isinstance(x, dict)]
    p_ex = [x for x in pred.get("eccezioni") or [] if isinstance(x, dict)]
    ex_pairs = _match(g_ex, p_ex, lambda x: x.get("testo"))
    logic_ok = _logic_equal(gold, pred, mapping)
    counts = {"predicted": len(p_el), "gold": len(g_el), "matched": len(pairs), "comma_same": comma_same,
              "grounding_same": ground_same, "exceptions_gold": len(g_ex), "exceptions_matched": len(ex_pairs),
              "logic_equal": int(logic_ok)}
    return {**counts, **_ratios(counts, 1)}


def _ratio(a: int, b: int) -> Optional[float]:
    return round(a / b, 3) if b else None


def _ratios(c: Dict[str, int], schemas: int) -> Dict[str, Optional[float]]:
    return {"element_precision": _ratio(c["matched"], c["predicted"]),
            "element_recall": _ratio(c["matched"], c["gold"]),
            "logic_accuracy": _ratio(c["logic_equal"], schemas),
            "exception_recall": _ratio(c["exceptions_matched"], c["exceptions_gold"]),
            "comma_accuracy": _ratio(c["comma_same"], c["matched"]),
            "grounding_accuracy": _ratio(c["grounding_same"], c["matched"])}


def grounding_rate(lines: Iterable[Dict[str, Any]]) -> Tuple[Optional[float], int]:
    """(share of explicit elements passing the ungrounded check, explicit elements) over generated lines -- re-checked
    on the line's own text when it carries one, else read from its stored checks."""
    total = passed = 0
    for line in lines:
        schema = (line or {}).get("schema")
        if not isinstance(schema, dict):
            continue
        checks = check(schema, line["text"]) if line.get("text") else (line.get("checks") or {})
        flagged = {str(m).split(":", 1)[0] for m in checks.get("ungrounded") or []}
        for scope, el in _flat(schema):
            if el.get("grounding") != "explicit":
                continue
            total += 1
            name = f"co.{scope}/{el.get('id')}" if scope else str(el.get("id"))
            passed += name not in flagged
    return _ratio(passed, total), total


def quality_report(lines: Optional[Iterable[Dict[str, Any]]] = None,
                   generated_lines: Optional[Iterable[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """v79.11 (A, §9): the measures over the reviewed lines (default: the overlay's reviewed ones; gold their schema,
    prediction their generated_schema), micro-averaged, plus correction_rate, n, and the automatic grounding rate over
    the generated lines (default: the generated file). No reviewed line: every gold measure None, «nessuna revisione»."""
    rev = [x for x in (reviewed().values() if lines is None else lines) if _is_reviewed(x)]
    gen = list(generated().values() if generated_lines is None else generated_lines)
    rate, explicit = grounding_rate(gen)
    pairs = [x for x in rev if isinstance(x.get("generated_schema"), dict)]
    totals = {k: 0 for k in ("predicted", "gold", "matched", "comma_same", "grounding_same", "exceptions_gold",
                             "exceptions_matched", "logic_equal")}
    for x in pairs:
        q = quality(x["generated_schema"], x["schema"])
        for k in totals:
            totals[k] += q[k]
    measures = _ratios(totals, len(pairs)) if pairs else {k: None for k in _ratios(totals, 1)}
    corrected = sum(1 for x in rev if x.get("corrections"))
    return {"n": len(pairs), "reviewed": len(rev), **measures,
            "correction_rate": _ratio(corrected, len(rev)) if rev else None, "grounding_rate_auto": rate,
            "explicit_elements": explicit, "generated_lines": len(gen), "counts": totals,
            "note": "" if pairs else "nessuna revisione"}


# --- the generator's prompt, reply and line ----------------------------------------------------------------------------

def module_ids() -> List[str]:
    """The ids of data/institution_modules.json's modules ([] when the file is missing)."""
    try:
        with open(MODULES_FILE, encoding="utf-8") as fh:
            return [str(m.get("id")) for m in json.load(fh).get("modules") or [] if isinstance(m, dict) and m.get("id")]
    except (OSError, ValueError, AttributeError) as exc:
        logger.warning("element schemas: %s not read: %s", MODULES_FILE, exc)
        return []


def _quoted(values: Iterable[Any]) -> str:
    return ", ".join(f'"{v}"' for v in values)


SCHEMA_SYSTEM = f"""Sei un giurista italiano. Ricevi il testo ufficiale di UN articolo di legge e ne scrivi lo SCHEMA \
DEGLI ELEMENTI: la norma scomposta nei requisiti che la sua applicazione richiede, con la logica che li lega, ricavato \
SOLO dal testo dato (mai da conoscenze esterne, giurisprudenza o dottrina).

Rispondi con UN SOLO oggetto JSON valido, senza testo prima o dopo, con esattamente queste chiavi:
{{"tipo": [...], "destinatario": [...], "legittimato": [...], "soggetto_passivo": [...], "rapporto": "...", \
"elementi": [...], "logic": {{...}}, "eccezioni": [...], "conseguenza": {{...}}, "commi_speciali": [...], \
"collegati": [...]}}

REGOLE
1. tipo: lista non vuota di una o più funzioni della norma tra {_quoted(TIPI)}. Una norma incriminatrice è \
"sanction"; un'azione civile, un diritto o un rimedio è "claim-conferring"; una causa di giustificazione è \
"scriminante"; un'eccezione o difesa della parte è "defence"; una norma che limita o esclude una pretesa è "limiting". \
Più funzioni sono ammesse.
2. destinatario (chi deve tenere o non tenere la condotta), legittimato (chi può far valere la norma: agire, querelare, \
eccepire), soggetto_passivo (chi subisce la condotta): liste di {{"ruolo": R, "condizione": null}} con R tra \
{_quoted(ROLES)}; "condizione" è una frase («se …») quando il ruolo vale solo a una condizione. Lista vuota quando il \
testo non lo dice.
3. rapporto: l'id del rapporto regolato scelto tra {_quoted(module_ids())} quando uno corrisponde; altrimenti una breve \
descrizione libera (es. "lesioni personali").
4. elementi: OGNI comma del testo va coperto. Un elemento è un requisito che deve sussistere (o mancare) perché la \
norma si applichi: {{"id": "E1", "testo": "...", "kind": "...", "kind_note": null, "comma": "1", "branches": [], \
"connective": null, "parti": [], "grounding": "explicit", "fonte": {{"comma": "1", "testo": "..."}}, \
"spiegazione": null}}.
   - kind tra {_quoted(KINDS)}; "other" solo con kind_note che dice di che si tratta.
   - Mai un elemento fatto solo del soggetto («Chiunque», «il debitore»), di un connettivo («con cui si conviene \
che»), di una clausola di salvezza («salvo che il fatto costituisca più grave reato»), di una facoltà senza condizione \
(«può chiedere»), di un oggetto nudo («il danno») o di una clausola di ambito («ai sensi del …»): il soggetto \
qualificato sta nell'elemento cui appartiene, le clausole di salvezza tra le eccezioni.
   - grounding "explicit": l'elemento è detto dal testo; fonte.testo è il passo del testo COPIATO ALLA LETTERA (stesse \
parole, stesso ordine) e "testo" usa le parole del testo. grounding "inferred": l'elemento è implicito nel testo (per \
esempio il dolo di un delitto che non prevede la colpa); fonte.comma è il comma da cui si ricava, fonte.testo può essere \
null, "spiegazione" dice in una frase perché è richiesto.
   - comma: l'etichetta del comma come nel testo ("1", "2", "1-bis"); un articolo senza commi numerati ha solo il comma \
"1".
   - branches: le alternative dell'elemento SOLO quando il testo le unisce con «o», «od», «oppure», «ovvero», «ossia», \
«sia … sia» o un elenco «a) … b)» chiuso da «o»; altrimenti []. connective: "e" quando il testo dell'elemento unisce \
parti cumulative (elencate in "parti"), "o" quando unisce alternative, altrimenti null.
   - Non indicare posizioni o offset: li calcola il programma.
5. logic: l'albero che lega gli elementi. Un nodo è {{"ref": "E1"}}, oppure {{"op": "all_of", "args": [...]}} (tutti), \
{{"op": "any_of", "args": [...]}} (almeno uno: almeno due argomenti, solo dove il testo pone un'alternativa con «o» o \
«ovvero»), {{"op": "not", "arg": ...}} (deve mancare), {{"op": "conditional", "if": ..., "then": ..., "else": ...}} \
(se … allora …, "else" facoltativo). L'albero cita OGNI id degli elementi almeno una volta, e solo id esistenti.
6. eccezioni: le clausole che escludono o limitano l'applicazione («salvo che …», «non è punibile …», «non si applica \
…»): {{"testo": "...", "comma": "...", "grounding": "explicit", "fonte": {{"comma": "...", "testo": "..."}}, \
"spiegazione": null}}, con le stesse regole di grounding degli elementi.
7. conseguenza: {{"testo": l'effetto (la pena, l'obbligo, il diritto, la nullità …), "tipo": "pena" | "risarcimento" | \
"obbligo" | "diritto" | "invalidità" | "altro", "comma": "..."}}.
8. commi_speciali: per ogni comma che non pone il caso base ma un caso speciale: {{"comma": "2", "kind": K, "schema": \
{{lo stesso formato, senza commi_speciali}}}} con K tra {_quoted(SPECIAL_KINDS)}.
9. collegati: gli articoli che il testo richiama espressamente, nella forma "583 c.p.".
10. Il testo abrogato (tra parentesi quadre o segnato «(abrogato)») non è mai vigente: nessun elemento, eccezione o \
conseguenza ne deriva.
11. Nessun elemento, eccezione o articolo che il testo non dica o non implichi necessariamente."""

SHORTER_NOTE = ("La risposta precedente è stata troncata: sii più conciso (testi brevi, spiegazioni di una riga, nessun "
                "campo superfluo), ma copri ogni comma.")
JSON_NOTE = "La risposta precedente non era JSON valido: rispondi solo con JSON valido, un unico oggetto."


def prompt_hash() -> str:
    return hashlib.sha1(SCHEMA_SYSTEM.encode("utf-8")).hexdigest()[:12]


def generation_hash(source_hash: str, model: str, prompt: Optional[str] = None) -> str:
    """sha1(f"{schema_version}|{prompt_hash}|{model}|{source_hash}")[:16] -- a change of any of them makes the line due."""
    raw = f"{SCHEMA_VERSION}|{prompt or prompt_hash()}|{model}|{source_hash}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def user_message(article: Dict[str, Any], note: str = "") -> str:
    """The user turn: the article's reference and rubrica, its commi labels, its official text (and a retry's note)."""
    number, code = article.get("number") or article.get("article") or "", article.get("code") or ""
    rubrica = " ".join(str(article.get("rubrica") or "").split())
    text = str(article.get("text") or "")
    head = f"ARTICOLO: art. {number} {code}".rstrip() + (f" - {rubrica}" if rubrica else "")
    out = f"{head}\nCOMMI DEL TESTO: {', '.join(commi(text))}\n\nTESTO UFFICIALE:\n{text}"
    return f"{out}\n\nNOTA: {note}" if note else out


def parse_reply(text: Any) -> Dict[str, Any]:
    """v79.11 (A): the reply's JSON object -- a <think> block dropped, a fenced ```json block or the first object amid
    prose; ValueError when there is none or it is not valid JSON."""
    t = re.sub(r"<think>.*?</think>", "", str(text or ""), flags=re.DOTALL | re.IGNORECASE)
    fence = re.search(r"```(?:json|JSON)?\s*\n?(.*?)```", t, flags=re.DOTALL)
    candidates = [fence.group(1)] if fence else []
    candidates.append(t)
    decoder = json.JSONDecoder()
    error: Optional[Exception] = None
    for c in candidates:
        start = c.find("{")
        while start >= 0:
            try:
                obj, _ = decoder.raw_decode(c, start)
            except ValueError as exc:
                error = error or exc
                start = c.find("{", start + 1)
                continue
            if isinstance(obj, dict):
                return obj
            start = c.find("{", start + 1)
    raise ValueError(f"nessun oggetto JSON valido nella risposta: {error or 'nessuna graffa'}")


def make_line(article: Dict[str, Any], endpoint_model: str, schema: Optional[Dict[str, Any]], failed: str = "",
              cut: bool = False, seconds: float = 0.0, usage: Optional[Dict[str, Any]] = None,
              doc: Optional[str] = None) -> Dict[str, Any]:
    """v79.11 (A): the generated file's line (spec §2) for one article: its keys and hashes, the schema with the
    fonte offsets computed by the code (ground) -- the model's, kept --, the checks (check; failed/cut the
    generator's), "dropped" (v79.12.2: what repair drops or fixes when the line is read) and "text" (the raw article
    text, for the review tool)."""
    text = str(article.get("text") or "")
    number = str(article.get("number") or article.get("article") or "").strip()
    code = str(article.get("code") or "")
    source = text_hash(text)
    grounded = ground(schema, text) if isinstance(schema, dict) else None
    checks = check(grounded, text) if isinstance(grounded, dict) else empty_checks()
    dropped = repair(grounded, text)[1] if isinstance(grounded, dict) else []
    checks["failed"] = str(failed or "") or ("" if isinstance(grounded, dict) else "nessuno schema")
    checks["cut"] = bool(cut)
    usage = usage or {}
    ph = prompt_hash()
    return {"doc": doc if doc is not None else str(article.get("document") or article.get("doc") or ""),
            "article": number, "code": code, "label": label_of(number, code) if code else number,
            "rubrica": " ".join(str(article.get("rubrica") or "").split()), "schema_version": SCHEMA_VERSION,
            "source_hash": source, "text_hash": source, "prompt_hash": ph,
            "generation_hash": generation_hash(source, endpoint_model, ph), "model": endpoint_model,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "seconds": round(float(seconds or 0), 1),
            "usage": {"prompt_tokens": int(usage.get("prompt_tokens") or 0),
                      "completion_tokens": int(usage.get("completion_tokens") or 0)},
            "schema": grounded, "checks": checks, "dropped": dropped, "text": text}
