"""v79 block B (79.1): grounded rule statements.

A rule statement is a sentence asserting what a norm requires, permits, excludes, punishes or when it applies -- its verb
one of RULE_VERBS («richiede», «si configura», «è integrato», «non è necessario», «prevede», «punisce», «si applica»,
«esclude», «decorre» and their forms). In every written section of the parere, §6's cells and §7's objections and
replies included, a rule statement stays only when it traces to one of:

  - an element of the record of an article it names (records and circumstances: the element's words in the sentence);
  - a store field of an article it names: procedibilità, termini, pena, competenza (procedural_facts, the store's or
    computed from the article's text) -- the sentence on that field's subject;
  - a quoted passage of a registry ruling (rulings.build's kept) on an article it names -- the passage's words in it.

A sentence naming no article traces to any record element, store field or passage of the answer. Otherwise it is
removed, and logged with its reason (removed()). Code-written text (the code's argued theses, the fallback lines, the
tables) is not checked: it is built from those sources.

check(text, sources, log) is the pass compose runs; unsourced(text, sources) is the nightly's output check (B3): the
rule statements of the final text with no source -- zero in every cell.
"""

import re
from typing import Any, Dict, List, Optional, Sequence, Set

# the verbs of a rule statement -- what a norm requires, permits, excludes, punishes, or when it applies
RULE_VERBS = (
    r"richied(?:e|ono)|si\s+richied(?:e|ono)|è\s+richiest[oa]|sono\s+richiest[ie]",
    r"si\s+configura(?:no)?|è\s+configurabile|configura(?:no)?\b",
    r"(?:è|sono|risulta|risultano)\s+integrat[oaie]|integra(?:no)?\s+(?:il|la|lo|l['’]|i|gli|le)\b",
    r"(?:non\s+)?(?:è|sono)\s+necessari[oaie]|occorr(?:e|ono)|(?:è|sono)\s+sufficient[ei]|basta(?:no)?\s+(?:che|il|la)",
    r"prevede|prevedono|è\s+previst[oa]|sono\s+previst[ie]",
    r"punisce|puniscono|(?:è|sono)\s+punit[oaie]|si\s+punisce",
    r"si\s+applica(?:no)?|(?:è|sono)\s+applicabil[ei]|trova(?:no)?\s+applicazione",
    r"esclude|escludono|(?:è|sono)\s+esclus[oaie](?!\s+(?:dai|dal|nei|in\s+base\s+ai)\s+fatt)",     # «escluso dai fatti»: the record's, not a rule
    r"decorr(?:e|ono)|(?:il\s+termine|i\s+termini)\s+(?:è|sono)\s+di",
    r"consente|consentono|(?:è|sono)\s+consentit[oaie]|(?:è|sono)\s+ammess[oaie]|vieta|vietano|(?:è|sono)\s+vietat[oaie]",
    r"presuppone|presuppongono|(?:è|sono)\s+tenut[oaie]\s+a",
)
RULE_RE = re.compile(r"\b(?:" + "|".join(RULE_VERBS) + r")", re.IGNORECASE)

# the store fields, each with the words of a sentence on its subject
STORE_FIELDS = {
    "procedibilita": re.compile(r"\bquerela|\bd['’]ufficio\b|\bprocedibil\w*|\bremission\w*", re.IGNORECASE),
    "termini": re.compile(r"\btermin[ei]\b|\bentro\s+\w+\s+(?:giorni|mesi|anni)|\bdecorr\w+|\bgiorni\b|\bmesi\b",
                          re.IGNORECASE),
    "pena": re.compile(r"\bpen[ae]\b|\breclusione|\bmulta\b|\barresto\b|\bammenda|\bpunit\w*|\bpunisc\w*|\bergastolo",
                       re.IGNORECASE),
    "competenza": re.compile(r"\bcompeten\w+|\bgiudice\s+di\s+pace|\bcorte\s+d['’]assise|\btribunale\s+(?:in\s+composizione\s+)?"
                             r"(?:collegiale|monocratic\w+)", re.IGNORECASE),
}
_FACT_FIELD = {"procedibilita": "procedibilita", "querela": "procedibilita", "querela_termine": "termini",
               "remissione": "procedibilita", "termini": "termini", "termine": "termini", "decorrenze": "termini",
               "pena": "pena", "competenza": "competenza"}

# words that tie a sentence to no source: the rule's own vocabulary, the code's, the articles'
_STOP = {
    "della", "delle", "dello", "degli", "dell", "nella", "nelle", "nello", "negli", "sulla", "sulle", "dalla", "dalle",
    "alla", "alle", "allo", "agli", "quale", "quali", "quando", "anche", "sono", "essere", "stato", "stata", "questo",
    "questa", "questi", "queste", "quello", "quella", "come", "dove", "ogni", "altro", "altra", "altri", "altre", "tale",
    "tali", "solo", "sola", "soltanto", "perché", "poiché", "mentre", "oppure", "ovvero", "sempre", "fatto", "fatti",
    "caso", "casi", "norma", "norme", "articolo", "articoli", "comma", "commi", "codice", "legge", "reato", "reati",
    "fattispecie", "elemento", "elementi", "requisito", "requisiti", "richiede", "richiedono", "configura",
    "configurano", "integrato", "integrata", "integra", "necessario", "necessaria", "prevede", "prevedono", "previsto",
    "prevista", "punisce", "puniscono", "punito", "punita", "applica", "applicano", "applicabile", "esclude",
    "escludono", "escluso", "esclusa", "decorre", "decorrono", "occorre", "sufficiente", "consente", "presuppone",
    "chiunque", "soggetto", "condotta", "parte", "parti", "cliente", "nostro", "nostra", "deve", "devono", "può",
    "possono", "affinché", "fine", "senza", "secondo", "presente", "infatti", "inoltre", "dunque", "quindi",
}
_STEM = 5


def stems(text: str) -> Set[str]:
    """The text's content words, cut to their first five letters («custode», «custodia» -> «custo»)."""
    words = re.findall(r"[a-zà-öø-ÿ]{4,}", str(text or "").lower())
    return {w[:_STEM] for w in words if w not in _STOP}


def is_rule(sentence: str) -> bool:
    return bool(RULE_RE.search(sentence or ""))


def _labels(sentence: str) -> List[str]:
    from . import citations
    return list(dict.fromkeys(citations.label(r) for r in citations.article_refs(sentence or "")))


def _overlap(a: Set[str], b: Set[str]) -> bool:
    """Enough words shared: two, or every one of a source holding fewer than two."""
    shared = len(a & b)
    return shared >= 2 or (shared >= 1 and len(b) <= 1)


def sources(ctx: Dict[str, Any], rows: Sequence[Dict[str, Any]] = ()) -> Dict[str, Any]:
    """The answer's sources, by article label: {"elements": {label: [stems]}, "store": {label: {fields}}, "passages":
    {label: [stems]}} -- the records' and the circumstances' live elements, the store's fields of every article the
    rows hold (ctx["store_fields"] when prepare computed them), the registry's quoted passages."""
    from .lawyer_method import _live, _norm_label
    elements: Dict[str, List[Set[str]]] = {}
    for label, rec in list((ctx.get("records") or {}).items()) + list((ctx.get("circumstances") or {}).items()):
        for r in _live(rec):
            s = stems(r.get("text"))
            if s:
                elements.setdefault(_norm_label(label), []).append(s)
    store = {_norm_label(k): set(v) for k, v in (ctx.get("store_fields") or store_fields(rows)).items()}
    passages: Dict[str, List[Set[str]]] = {}
    for r in (ctx.get("rulings_record") or {}).get("kept") or []:
        if r.get("passage"):
            passages.setdefault(_norm_label(r.get("article") or ""), []).append(stems(r["passage"]))
    return {"elements": elements, "store": store, "passages": passages}


def store_fields(rows: Sequence[Dict[str, Any]]) -> Dict[str, List[str]]:
    """{label: [field]} -- the store fields (procedibilita, termini, pena, competenza) each article of the rows holds."""
    from . import citations, procedural_facts
    out: Dict[str, List[str]] = {}
    for code in procedural_facts.CODES:
        for number, facts in procedural_facts.by_number(rows or [], code).items():
            fields = sorted({_FACT_FIELD[k] for k, v in (facts or {}).items() if k in _FACT_FIELD and v not in (None, "", [])})
            if fields:
                out[citations.label((number, code))] = fields
    return out


def source_of(sentence: str, src: Dict[str, Any]) -> str:
    """The source a rule statement traces to -- «element:<label>», «store:<field>:<label>», «passage:<label>» -- or ""."""
    words = stems(sentence)
    named = _labels(sentence)
    scope = named or sorted(set(src["elements"]) | set(src["store"]) | set(src["passages"]))
    for label in scope:
        if any(_overlap(words, e) for e in src["elements"].get(label) or []):
            return f"element:{label}"
    for label in scope:
        for field in sorted(src["store"].get(label) or ()):
            if STORE_FIELDS[field].search(sentence):
                return f"store:{field}:{label}"
    for label in scope:
        if any(_overlap(words, p) for p in src["passages"].get(label) or []):
            return f"passage:{label}"
    return ""


def reason(sentence: str) -> str:
    named = _labels(sentence)
    return ("regola senza fonte: nessun elemento del registro, campo dell'archivio o passaggio del registro delle "
            "pronunce " + (f"per {', '.join(named)}" if named else "nella risposta"))


def _kept(sentence: str, keep: Sequence[str]) -> bool:
    """v79.12.5 (item 10): the sentence is part of a code-written line (keep: the stage table's closed list)."""
    s = " ".join(sentence.split()).lstrip("-•* ").strip()
    return bool(s) and any(s in " ".join(str(k).split()) for k in keep or ())


def check(text: str, src: Dict[str, Any], log: Optional[List[Dict[str, str]]] = None, keep: Sequence[str] = (),
          kept_log: Optional[List[str]] = None) -> str:
    """The text without its rule statements that trace to no source; each removal logged {sentence, reason}. Table rows
    and headings are left as they are. v79.12.5 (item 10): a sentence of a keep line (the stage table's closed list,
    code-written: render_computed.stage_blocks) is never removed -- the ATP and 700 lines of the claimant's pre-filing
    list are conditions, not rule statements the record could ground; each so kept appended to kept_log."""
    from .lawyer_method import _by_sentence
    if not text:
        return text or ""

    def review(sentence: str) -> str:
        s = sentence.strip()
        if s.startswith("|") or s.startswith("#") or not is_rule(s) or source_of(s, src):
            return sentence
        if _kept(s, keep):
            if kept_log is not None:
                kept_log.append(s[:300])
            return sentence
        if log is not None:
            log.append({"sentence": s[:300], "reason": reason(s)})
        return ""
    out = "\n".join(_by_sentence(line, review) if not line.lstrip().startswith("|") else line
                    for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", out).strip() if out != text else text


# v79.12.1 (review): a sentence declaring a norm applicable to the case -- «si applica», «è (inoltre) applicabile»,
# «trova applicazione», «è (altresì) configurabile», «si configura», «ricorre la fattispecie», «è integrato il reato»,
# «integra il reato» (up to two words between the verb and its predicate)
_GAP = r"(?:[\w'’]+\s+){0,2}?"
APPLIES_RE = re.compile(
    r"\bsi\s+applica(?:no)?\b|\bsi\s+configura(?:no)?\b"
    rf"|\b(?:è|sono|risulta|risultano|appare|appaiono)\s+{_GAP}(?:applicabil[ei]|configurabil[ei]|integrat[oaie])\b"
    rf"|\btrova(?:no)?\s+{_GAP}applicazione\b"
    rf"|\bricorr(?:e|ono)\s+{_GAP}(?:la|le|il|i|gli|lo|l['’]|un[ao]?)\s*(?:fattispecie|reat[oi]|delitt[oi]|"
    r"contravvenzion[ei]|ipotesi)\b"
    rf"|\bintegra(?:no)?\s+{_GAP}(?:il|la|lo|gli|i|le|l['’]|un[ao]?)\s*(?:reat[oi]|delitt[oi]|fattispecie|"
    r"contravvenzion[ei]|ipotesi)\b",
    re.IGNORECASE)


def declares_applicable(sentence: str) -> bool:
    return bool(APPLIES_RE.search(sentence or ""))


def check_declared(text: str, src: Dict[str, Any], own: Sequence[str],
                   log: Optional[List[Dict[str, str]]] = None) -> str:
    """v79.12.1 (review): the check for a plain opinion -- the method stepped aside, so its decisive article has no
    record elements and Block B's full check would strip every rule on the case's own norms. Only a sentence declaring
    another norm applicable (declares_applicable) and naming an article outside own (the case's: its records, the
    article the method stepped aside on, the user's) is checked: it stays when it traces to an element of that article's
    record or to a quoted registry passage on it (never a store field: a penalty or a term grounds no applicability);
    otherwise removed and logged {sentence, reason}. A sentence naming no article is left as it is."""
    from .lawyer_method import _by_sentence, _norm_label
    if not text:
        return text or ""
    mine = {_norm_label(x) for x in own}

    def review(sentence: str) -> str:
        s = sentence.strip()
        if s.startswith("|") or s.startswith("#") or not declares_applicable(s):
            return sentence
        others = [x for x in (_norm_label(y) for y in _labels(s)) if x not in mine]
        if not others:
            return sentence
        words = stems(s)
        if any(_overlap(words, e) for x in others
               for e in list(src["elements"].get(x) or []) + list(src["passages"].get(x) or [])):
            return sentence
        if log is not None:
            log.append({"sentence": s[:300], "reason": "norma dichiarata applicabile senza fonte: nessun elemento del "
                        f"registro né passaggio del registro delle pronunce per {', '.join(others)}"})
        return ""
    out = "\n".join(_by_sentence(line, review) if not line.lstrip().startswith("|") else line
                    for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", out).strip() if out != text else text


def unsourced(text: str, src: Dict[str, Any]) -> List[str]:
    """B3: the rule statements of a text that trace to no source (the nightly's output check: none)."""
    log: List[Dict[str, str]] = []
    check(text, src, log)
    return [x["sentence"] for x in log]


def serializable(src: Dict[str, Any]) -> Dict[str, Any]:
    """The sources as JSON (the state, the nightly): sets as sorted lists."""
    return {"elements": {k: [sorted(x) for x in v] for k, v in src["elements"].items()},
            "store": {k: sorted(v) for k, v in src["store"].items()},
            "passages": {k: [sorted(x) for x in v] for k, v in src["passages"].items()}}


def from_serializable(data: Dict[str, Any]) -> Dict[str, Any]:
    return {"elements": {k: [set(x) for x in v] for k, v in (data.get("elements") or {}).items()},
            "store": {k: set(v) for k, v in (data.get("store") or {}).items()},
            "passages": {k: [set(x) for x in v] for k, v in (data.get("passages") or {}).items()}}
