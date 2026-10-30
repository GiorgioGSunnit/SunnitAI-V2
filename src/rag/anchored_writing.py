"""ANCHORED_WRITING_ENABLED (default false): every rule the parere states is written from a passage named by an anchor.

fetch_audit's provenance by section: the sections that receive only the plan (Risposte ai quesiti, Conclusioni)
state rules from memory, and the others state rules their texts do not hold. With the flag:

  1. The plan names, for each issue, the passages its argument rests on ("passi": an article's comma, "art. 2051
     c.c. co. 1", or a ruling's reference), read by clean_plan. Once the plan is final, build() numbers them [A1],
     [A2], ...: each "passo" whose article is still among the issue's norms, then, for a norm the plan left without
     one, the comma that shares most words with the issue (its title, argument, objection and reply), and each
     ruling of the issue with its passage. The procedural section's own texts (its procedural articles and the legal
     terms' articles) get anchors of their own.
  2. Every section gets the passages it may rest on (block()): the plan's for every section -- so Risposte ai
     quesiti and Conclusioni, which got only the plan, now get the commi the plan cites for each issue, compact --
     the procedural ones too for the procedural section, and the plan's text shows each issue's anchors. The rule
     (ANCHOR_RULE): each sentence stating a rule, an institution, a procedural step or a holding ends with the anchor
     of its passage.
  3. After the sections, verify(), on two kinds of sentence (gr-off: "holding" sentences 71% not in material, most
     in Applicazione al caso, mostly the answer's reasoning on the facts):
     - a statement of the law or of case law in general (classify: "law" / "case_law"; the rewrite call decides the
       sentences code cannot tell) passes when every anchor exists and was given to its section, every article or
       ruling it cites is its anchors' own, at least MIN_OVERLAP of its content words are in the anchors' passages
       or RUBRICA_SHARE of an anchored article's rubrica words are in it, and -- case law -- one anchor is a ruling
       passage of its section; one citing without an anchor fails. A failing statement (aw-on: 165 removals cost
       MENTIONED 10):
         citing an article of the checked material (graded 2 or more, named in the case, a tool's or a term's):
           rewritten from that article's passage and anchored there; the rewrite failing, kept -- never removed;
         case law without an anchor to a ruling passage of its section: restated as the rule of its anchored
           article, else removed;
         every citation outside the checked material: the citations dropped, the sentence kept if it still reads
           as one, else removed;
         otherwise rewritten once from its section's passages, else kept.
     - an application of a rule to this case's facts ("application": a party or a fact of the case is its subject)
       needs no anchor and is never removed; an article it cites that neither its section's material nor the
       checked material holds loses only the citation (drop_citations).
     Then every anchor is stripped.
  4. The procedural section gets its stage's procedure vocabulary (practical_checklist.PROCEDURE_VOCABULARY) and the
     rule to name no procedural institution outside it.

Logged as anchored_writing: the passages, and per section the sentences anchored / citing without an anchor /
failed (by reason, the overlap failures by their overlap) / rewritten / kept / removed, each entry with the
articles it cited (diagnose_long: the expected ones lost with it). Off, nothing changes.
"""

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from . import feature_flags
from .verbose_logger import vlog

logger = logging.getLogger(__name__)

FLAG = "ANCHORED_WRITING_ENABLED"
MIN_OVERLAP = 0.15        # share of a sentence's content words that must be in its anchors' passages, or
RUBRICA_SHARE = 0.5       # the share of an anchored article's rubrica words the sentence must hold instead
OVERLAP_BUCKETS = (0.05, 0.10, 0.15, 0.25)   # the logged distribution of the overlap failures
PASSAGE_CHARS = 900
REWRITE_MARK = "FRASI DA RISCRIVERE:"   # fetch_audit cuts a rewrite call's material here (as grounded_review's)

ANCHOR_RE = re.compile(r"\[\s*(A\d+(?:\s*[,;]\s*A\d+)*)\s*\]")
_STRIP_RE = re.compile(r"[ \t]*\[\s*A\d+(?:\s*[,;]\s*A\d+)*\s*\]")
_COMMA_RE = re.compile(r"\bco(?:mma)?\.?\s*(\d+)", re.IGNORECASE)
_RULING_NUMBER_RE = re.compile(r"(?<![\d/])(\d{1,6}/\d{4})(?![\d/])")

BLOCK_TITLE = "PASSI CON ANCORA"
ANCHOR_RULE = (
    " Ogni frase che enuncia una regola, un istituto, un passaggio processuale o ciò che una pronuncia afferma termina "
    "con l'ancora del passo di PASSI CON ANCORA su cui si fonda, tra parentesi quadre ([A3], o [A3, A5]): cita la "
    "norma o la pronuncia di quel passo e ne riprende il contenuto. Una regola che nessun passo sostiene non si "
    "scrive.")
PLAN_RULE = (
    "- Per ogni questione indica in \"passi\" i commi delle sue norme e le pronunce su cui si fonda l'argomento: "
    "l'identificativo della norma con il comma (\"art. 2051 c.c. co. 1\") o il riferimento della pronuncia.\n")
PLAN_KEY = '"passi": ["art. N codice co. K", "riferimento"], '

REWRITE_SYSTEM = (
    "Sei un avvocato italiano e correggi un parere. Ricevi alcuni PASSI di norme e pronunce, ciascuno con la sua "
    "ancora, e alcune FRASI del parere, numerate, ciascuna con le ancore dei passi da cui va riscritta.\n"
    "Per una frase segnata [prima il tipo] decidi prima il tipo: \"applicazione\" se applica una regola ai fatti di "
    "questo caso (il soggetto è una parte o un fatto del caso: \"la reazione di Tizio era proporzionata\", \"il "
    "ritardo è imputabile al venditore\"): non va riscritta, lascia la frase vuota; \"generale\" se enuncia la legge "
    "o la giurisprudenza in generale (ciò che una norma prevede, ciò che la giurisprudenza afferma, una definizione, "
    "dei requisiti). Le altre frasi sono \"generale\".\n"
    "Riscrivi ogni frase \"generale\" dicendo solo ciò che i suoi passi dicono, con le loro parole o "
    "parafrasandoli, citando la norma o la pronuncia di quei passi (numero e codice, o riferimento) e nessun'altra, "
    "nello stesso punto del ragionamento; una frase segnata [come regola della norma, senza giurisprudenza] diventa "
    "la regola della norma del suo passo, senza nominare la giurisprudenza né una pronuncia. Se i passi non "
    "sostengono la frase, o non ce ne sono, la frase è vuota. Non usare ciò che sai. Rispondi SOLO con un array "
    "JSON: [{\"i\": numero, \"tipo\": \"generale\"|\"applicazione\", \"frase\": \"la frase riscritta, o vuota\"}].")


def enabled() -> bool:
    return feature_flags.enabled(FLAG)


# --- helpers ------------------------------------------------------------------------------------------------------

_STOP = set("""della delle dello degli nella nelle nello negli sulla sulle dalla dalle alla alle allo agli come anche
quando quale quali questo questa sono essere viene deve devono può possono ogni altro altra loro tale tali articolo
comma codice norma legge ovvero nonché caso parte""".split())


def stems(text: str) -> set:
    return {w[:6] for w in re.findall(r"[a-zà-ù]{4,}", (text or "").lower()) if w not in _STOP}


def strip_anchors(text: str) -> str:
    """text without its anchors, and without the brackets an anchor leaves empty ("destinata ([A3])" -> "destinata")."""
    from . import text_tidy
    stripped = _STRIP_RE.sub("", text or "")
    return text_tidy.empty_brackets(stripped) if stripped != (text or "") else stripped


def _row_text(row: Dict[str, Any]) -> str:
    s = row.get("s") or {}
    return row.get("_passage") or " ".join(str(s.get("plain_text") or s.get("abstract") or "").split())


def _commi(row: Dict[str, Any]) -> Tuple[List[str], bool]:
    """(the article's commi, numbered): COMMA_MATERIAL's when the row holds them, else its text split; numbered False
    for a sentence split (its parts are not commi)."""
    from . import comma_material
    if row.get(comma_material.COMMI_KEY):
        return list(row[comma_material.COMMI_KEY]), row.get(comma_material.FROM_KEY) != "sentences"
    parts, how = comma_material.split_commi(_row_text(row))
    return parts, how not in ("sentences", "none")


def best_comma(commi: Sequence[str], focus: str) -> int:
    """The index of the comma sharing most content words with focus (the first on a tie or none)."""
    words = stems(focus)
    best, score = 0, -1
    for i, comma in enumerate(commi):
        shared = len(words & stems(comma))
        if shared > score:
            best, score = i, shared
    return best


def parse_passi(value: Any, known: set, references: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """The plan issue's "passi": [{"ref", "comma"}] for an article among `known` (its comma or None), [{"ruling"}]
    for a reference among `references`; anything else left out."""
    from . import plan_then_write as ptw
    out: List[Dict[str, Any]] = []
    for item in ptw._texts(value):
        refs = [r for r in ptw._refs_of(item, known) if r in known]
        if refs:
            m = _COMMA_RE.search(item)
            entry = {"ref": refs[0], "comma": int(m.group(1)) if m else None}
        else:
            reference = ptw._matching_reference(item, references)
            entry = {"ruling": reference} if reference else None
        if entry and entry not in out:
            out.append(entry)
    return out


# --- the passages ---------------------------------------------------------------------------------------------------

class Catalogue:
    """The passages, by anchor id: {"id", "kind" (article / ruling), "ref" or "ruling", "label", "text"}; the plan's
    ids (each issue's in plan["questioni"][i]["ancore"]) and the procedural section's."""

    def __init__(self):
        self.passages: Dict[str, Dict[str, Any]] = {}
        self.plan_ids: List[str] = []
        self.procedural_ids: List[str] = []
        self.section_ids: Dict[str, List[str]] = {}      # the passages each section was given (its block)
        self.checked: List[tuple] = []     # the checked material's articles (grounded_review.strict_refs)
        self._by_key: Dict[tuple, str] = {}

    def add(self, key: tuple, kind: str, label: str, text: str, **extra) -> Optional[str]:
        text = " ".join((text or "").split())[:PASSAGE_CHARS]
        if not text:
            return None
        if key in self._by_key:
            return self._by_key[key]
        pid = f"A{len(self.passages) + 1}"
        self.passages[pid] = {"id": pid, "kind": kind, "label": label, "text": text, **extra}
        self._by_key[key] = pid
        return pid

    def of_ref(self, ref: tuple) -> List[str]:
        from . import plan_then_write as ptw
        return [pid for pid, p in self.passages.items() if p.get("ref") and ptw._same_ref(p["ref"], ref)]

    def of_ruling(self, number: str) -> List[str]:
        return [pid for pid, p in self.passages.items() if p.get("ruling") and number in _RULING_NUMBER_RE.findall(p["ruling"])]

    def rulings(self, ids: Iterable[str]) -> List[str]:
        return [i for i in ids if (self.passages.get(i) or {}).get("kind") == "ruling"]

    def block(self, ids: Iterable[str]) -> str:
        lines = [f"[{pid}] {self.passages[pid]['label']}: {self.passages[pid]['text']}"
                 for pid in dict.fromkeys(ids) if pid in self.passages]
        return f"{BLOCK_TITLE}:\n" + "\n".join(lines) if lines else ""

    def log(self) -> List[Dict[str, Any]]:
        return [{"id": p["id"], "label": p["label"], "chars": len(p["text"])} for p in self.passages.values()]


def _rubrica(row: Dict[str, Any]) -> str:
    s = row.get("s") or {}
    return " ".join(str(s.get("titolo_articolo") or s.get("title") or "").split())


def _article_passage(cat: Catalogue, row: Dict[str, Any], comma: Optional[int], focus: str) -> Optional[str]:
    from . import plan_then_write as ptw
    ref = row["_article"]
    commi, numbered = _commi(row)
    if not commi:
        return None
    k = comma - 1 if comma and numbered and 1 <= comma <= len(commi) else best_comma(commi, focus)
    label = f"art. {ref[0]} {ptw._code_key(ref[1])}" + (f", co. {k + 1}" if numbered and len(commi) > 1 else "")
    return cat.add(("article", ref, k), "article", label, commi[k], ref=ref, comma=k + 1 if numbered else None,
                   rubrica=_rubrica(row))


def build(plan: Dict[str, Any], article_rows: List[Dict[str, Any]], references: Dict[str, List[Dict[str, Any]]],
          procedural: Sequence[tuple] = (), terms: Sequence[Dict[str, str]] = ()) -> Catalogue:
    """The passages of the final plan, each issue's anchors set in plan["questioni"][i]["ancore"] (see the module
    docstring), then the procedural articles' and the terms' (`terms`: [{"ref", "text"}])."""
    from . import plan_then_write as ptw
    cat = Catalogue()
    from . import comma_material
    rows: Dict[tuple, Dict[str, Any]] = {}
    for row in article_rows:
        ref = row.get("_article")
        if ref and ref not in rows and (_row_text(row) or row.get(comma_material.COMMI_KEY)):
            rows[ref] = row
    for q in plan.get("questioni") or []:
        focus = " ".join(str(q.get(k) or "") for k in ("titolo", "argomento", "obiezione", "replica"))
        ids: List[str] = []
        planned = {}
        for p in q.get("passi") or []:
            if p.get("ref"):
                planned.setdefault(p["ref"], []).append(p.get("comma"))
        for ref in q.get("norme") or []:
            row = rows.get(ref) or next((r for k, r in rows.items() if ptw._same_ref(k, ref)), None)
            if row is None:
                continue
            for comma in planned.get(ref) or [None]:
                pid = _article_passage(cat, row, comma, focus)
                if pid:
                    ids.append(pid)
        for reference in list(q.get("pronunce") or []) + list(q.get("pronunce_contrarie") or []):
            hits = references.get(reference) or []
            if hits:
                pid = cat.add(("ruling", reference), "ruling", reference, _row_text(hits[0]), ruling=reference)
                if pid:
                    ids.append(pid)
        q["ancore"] = list(dict.fromkeys(ids))
        cat.plan_ids += [i for i in q["ancore"] if i not in cat.plan_ids]
    for ref in procedural:
        row = rows.get(ref) or next((r for k, r in rows.items() if ptw._same_ref(k, ref)), None)
        if row is not None:
            commi, _ = _commi(row)
            pid = cat.add(("procedural", ref), "article", f"art. {ref[0]} {ptw._code_key(ref[1])}",
                          " ".join(commi) or _row_text(row), ref=ref, rubrica=_rubrica(row))
            if pid:
                cat.procedural_ids.append(pid)
    for term in terms:
        refs = [r for _, _, rs in ptw._citations(f"art. {term.get('ref', '')}", []) for r in rs]
        if refs and term.get("text"):
            pid = cat.add(("term", refs[0]), "article", f"art. {refs[0][0]} {ptw._code_key(refs[0][1])}",
                          term["text"], ref=refs[0])
            if pid and pid not in cat.procedural_ids:
                cat.procedural_ids.append(pid)
    return cat


# --- the two kinds of sentence --------------------------------------------------------------------------------------
# gr-off's provenance by section: "holding" sentences were 71% not in material (280/394), most in Applicazione al
# caso (158/212) -- most of them the answer's own reasoning on the case's facts, not a statement of the law. Two kinds:
#   "law" / "case_law"  a statement of the law or of case law in general (what an article provides, what the case law
#                       holds, a definition, a requirement): it needs an anchor -- case law an anchor to a ruling passage
#                       given to its section -- and is rewritten from its passages or removed;
#   "application"       a rule applied to this case's facts (its subject a party or a fact of the case): no anchor
#                       needed, never removed; an article it cites that the section's material lacks loses only the
#                       citation.
# Code reads the clear ones (classify); the rest, when they fail the anchor check, the rewrite call classifies.
_CASE_LAW_RE = re.compile(
    r"\b(?:la|una|costante|consolidata|prevalente|recente|ormai)\s+giurisprudenza\b|\bin\s+giurisprudenza\b|"
    r"\b(?:la|le)\s+(?:suprema\s+)?(?:corte|cassazione|sezioni\s+unite)\b|\bsezioni\s+unite\b|\bgiudici\s+di\s+legittimità\b|"
    r"\bsecondo\s+(?:un|l['’]|il)\s*orientamento\b|\b(?:è|risulta)\s+(?:ormai\s+)?(?:consolidat|pacific)\w*|"
    r"\b(?:cass\.|cassazione)\b|\bn\.\s*\d{1,6}/\d{4}\b|\bsent(?:enza)?\.?\s+n\.", re.IGNORECASE)
_LAW_RE = re.compile(
    r"^\W*(?:(?:l['’]|gli|all['’]|dall['’]|nell['’])\s*artt?(?:icol[oi]|\.)|(?:la|le|questa|tale)\s+(?:norma|norme|"
    r"disposizion\w+|legge|disciplina)|il\s+codice|il\s+legislatore|(?:ai\s+sensi|a\s+norma|in\s+base)\s+"
    r"(?:dell?['’]|degli|agli|all['’])\s*artt?\.[^,]{0,80},\s*(?:il|la|lo|l['’]|i|gli|le|chi|è|sono|si)\b)"
    r"|\bsi\s+intende\s+per\b|\b(?:requisit|presuppost)\w*\s+(?:del|della|dell['’]|per|di)\b|\bè\s+definit\w*\b"
    r"|\b(?:prevede|stabilisce|dispone|sancisce|punisce|impone|consente)\s+che\b", re.IGNORECASE)
_PARTY_WORDS = re.compile(
    r"\b(?:il\s+nostro\s+assistito|la\s+parte\s+assistita|l['’]\s*assistit\w+|il\s+cliente|la\s+cliente|"
    r"(?:la\s+)?controparte|il\s+convenuto|la\s+convenuta|l['’]\s*attor\w+|l['’]\s*imputat\w+|l['’]\s*indagat\w+|"
    r"la\s+persona\s+offesa|la\s+vittima|il\s+danneggiat\w+|il\s+venditore|l['’]\s*acquirente|il\s+conduttore|"
    r"il\s+locatore|il\s+datore|il\s+lavoratore|nel\s+(?:nostro\s+|presente\s+)?caso|nella\s+(?:specie|fattispecie\s+"
    r"in\s+esame|vicenda)|in\s+questo\s+caso|nel\s+caso\s+(?:di\s+specie|in\s+esame)|qui\b)", re.IGNORECASE)
_LEGAL_CAPITALS = {"cassazione", "corte", "tribunale", "codice", "costituzione", "sezioni", "unite", "stato", "repubblica",
                   "consiglio", "legge", "decreto", "giudice", "ministero", "procura", "tar", "inps", "inail", "ue"}


def _names(case_text: str) -> set:
    """The case's proper names: capitalised words of the case not at a sentence's start, without legal ones."""
    out = set()
    for m in re.finditer(r"(?<![.!?]\s)(?<!^)\b([A-ZÀ-Ü][a-zà-ù]{2,})\b", case_text or ""):
        w = m.group(1).lower()
        if w not in _LEGAL_CAPITALS and w not in _STOP:
            out.add(w)
    return out


def classify(sentence: str, case_text: str = "", names: Optional[set] = None) -> Optional[str]:
    """"case_law", "law", "application" or None (not clear to code): case-law words anywhere first; a sentence
    opening on the law ("L'art. 2051 c.c. prevede...") or stating a definition or requirements; one naming a party or
    the case ("Tizio", "il convenuto", "nel caso di specie"), or made mostly of the case's words, applies a rule."""
    body = strip_anchors(sentence)
    if _CASE_LAW_RE.search(body):
        return "case_law"
    if _LAW_RE.search(body):
        return "law"
    names = _names(case_text) if names is None else names
    words = re.findall(r"[A-Za-zÀ-ü]+", body)
    if _PARTY_WORDS.search(body) or any(w.lower() in names for w in words[1:] if w[:1].isupper()) or \
            (words and words[0].lower() in names):
        return "application"
    own, case = stems(body), stems(case_text)
    if case_text and len(own) >= 4 and len(own & case) / len(own) >= 0.6:
        return "application"
    return None


# --- the check ------------------------------------------------------------------------------------------------------

def overlap(sentence: str, passages: Sequence[Dict[str, Any]]) -> Tuple[float, bool]:
    """(the share of the sentence's content words in the passages, whether it holds RUBRICA_SHARE of an article
    passage's rubrica words): a paraphrase of a comma shares few of its words but names its institution."""
    words = stems(strip_anchors(sentence))         # numbers and code sigle carry no stem
    if not words:
        return 1.0, True
    ratio = len(words & stems(" ".join(p["text"] for p in passages))) / len(words)
    rubrica = False
    for p in passages:
        r = stems(p.get("rubrica") or "")
        if r and len(r & words) / len(r) >= RUBRICA_SHARE:
            rubrica = True
    return ratio, rubrica


def check(sentence: str, ids: Sequence[str], cat: Catalogue, documents: Optional[List[Dict[str, Any]]] = None,
          known: Optional[set] = None, allowed: Optional[Sequence[str]] = None,
          case_law: bool = False) -> Optional[str]:
    """Why a statement of the law fails, or None: "unknown anchor", "anchor not given to the section" (not among
    `allowed`, the section's own passages), "case law without a ruling anchor" (case_law and no ruling passage among
    its anchors), "citation not its anchor's", "overlap" (fewer than MIN_OVERLAP of its content words in its anchors'
    passages and less than RUBRICA_SHARE of an anchored article's rubrica words: overlap()), "unanchored citation"
    (no anchor, cites an article or a ruling)."""
    from . import plan_then_write as ptw
    body = strip_anchors(sentence)
    refs = [r for _, _, rs in ptw._citations(body, documents or [], known) for r in rs]
    rulings = _RULING_NUMBER_RE.findall(body)
    if not ids:
        if case_law:
            return "case law without a ruling anchor"
        return "unanchored citation" if refs or rulings else None
    if any(i not in cat.passages for i in ids):
        return "unknown anchor"
    if allowed is not None and any(i not in allowed for i in ids):
        return "anchor not given to the section"
    if case_law and not cat.rulings(ids):
        return "case law without a ruling anchor"
    passages = [cat.passages[i] for i in ids]
    for ref in refs:
        if not any(p.get("ref") and ptw._same_ref(p["ref"], ref) for p in passages):
            return "citation not its anchor's"
    for number in rulings:
        if not any(p.get("ruling") and number in _RULING_NUMBER_RE.findall(p["ruling"]) for p in passages):
            return "citation not its anchor's"
    ratio, rubrica = overlap(body, passages)
    return "overlap" if ratio < MIN_OVERLAP and not rubrica else None


def _bucket(ratio: float) -> str:
    low = 0.0
    for high in OVERLAP_BUCKETS:
        if ratio < high:
            return f"{round(low * 100)}-{round(high * 100)}%"
        low = high
    return f">={round(low * 100)}%"


# --- dropping a citation only (an application citing an article its section lacks) ---------------------------------

_CODE_AFTER_RE = re.compile(
    r"^\s*(?:,?\s*(?:co(?:mma)?\.?|primo|secondo|terzo|quarto)\s*\d*\s*(?:comma)?\s*,?\s*)?(?:del\s+|della\s+|cod\.\s*)?"
    r"(?:c\.\s?p\.\s?c\.|c\.\s?p\.\s?p\.|c\.\s?p\.\s?a\.|c\.\s?d\.\s?s\.|c\.\s?c\.|c\.\s?p\.|cost\.|"
    r"(?:codice|d\.\s?lgs\.|l\.|legge|d\.\s?p\.\s?r\.)[^,.;()]{0,60}?(?=[,.;()]|$))", re.IGNORECASE)
_COMMA_AFTER_RE = re.compile(r"^\s*,?\s*(?:comma|co\.)\s*\d+(?:-\w+)?(?:\s*(?:e|,)\s*\d+)*", re.IGNORECASE)
_INTRO_BEFORE_RE = re.compile(
    r"(?:,\s*)?(?:\b(?:ai\s+sensi|a\s+norma|in\s+base|come\s+previsto|come\s+prevede|previst[oaie]|di\s+cui|secondo|"
    r"ex|in\s+forza|per\s+effetto|in\s+applicazione)\s+)?(?:dell?['’]\s*|degli\s+|agli\s+|all['’]\s*|dall['’]\s*|"
    r"dagli\s+|dal\s+|l['’]\s*|gli\s+)?$", re.IGNORECASE)


def drop_citations(sentence: str, drop: Sequence[tuple], documents: Optional[List[Dict[str, Any]]] = None,
                   known: Optional[set] = None) -> str:
    """The sentence without the citations whose every article is among `drop` (with the code and the comma after
    them, "c.c., comma 1", "comma 1 del Codice Civile", and the words introducing them, "ai sensi dell'"; a
    parenthesis left empty goes; a sentence that started with the citation starts again with a capital); a citation
    also naming a kept article stays."""
    from . import plan_then_write as ptw
    from . import text_tidy
    spans = []
    for start, end, refs in ptw._citations(sentence, documents or [], known):
        if not refs or not all(any(ptw._same_ref(r, d) for d in drop) for r in refs):
            continue
        m = _CODE_AFTER_RE.match(sentence[end:])
        end += m.end() if m else 0
        m = _COMMA_AFTER_RE.match(sentence[end:])
        end += m.end() if m else 0
        intro = _INTRO_BEFORE_RE.search(sentence[:start])
        start = intro.start() if intro else start
        spans.append((start, end))
    for start, end in sorted(spans, reverse=True):
        sentence = sentence[:start] + sentence[end:]
    sentence = re.sub(r"\(\s*[,;]?\s*\)", "", sentence)
    sentence = re.sub(r"\s+([,.;:)])", r"\1", re.sub(r"\(\s+", "(", sentence))
    sentence = re.sub(r"[ \t]{2,}", " ", re.sub(r",\s*,", ",", text_tidy.empty_brackets(sentence))).strip()
    if spans and min(s for s, _ in spans) == 0 or re.match(r"[,;:]", sentence):
        sentence = text_tidy.capitalise(re.sub(r"^[\s,;:]+", "", sentence))
    return sentence


# --- the rewrite call -----------------------------------------------------------------------------------------------

def _anchor_rewrite(cat: Catalogue, items: List[Tuple[str, List[str], str]]) -> Dict[int, Dict[str, str]]:
    """{index: {"tipo", "frase"}} for one section's failing sentences: (sentence, its passages, how: "classify" (code
    could not tell its kind), "law", or "case_law" (rewritten as the rule of its article, no case law)); {} on a
    failure."""
    from . import defensive_generation as dg
    ids = list(dict.fromkeys(i for _, anchors, _ in items for i in anchors))
    notes = {"case_law": " [come regola della norma, senza giurisprudenza]", "law": "", "classify": " [prima il tipo]"}
    human = (f"PASSI:\n" + ("\n".join(f"[{i}] {cat.passages[i]['label']}: {cat.passages[i]['text']}" for i in ids)
                            or "(nessuno)")
             + f"\n\n{REWRITE_MARK}\n"
             + "\n".join(f"{n}. {s} [passi: {', '.join(a) or 'nessuno'}]{notes[how]}"
                         for n, (s, a, how) in enumerate(items, 1)))
    try:
        raw = dg._chat_json([SystemMessage(content=REWRITE_SYSTEM), HumanMessage(content=human)],
                            max_tokens=200 + 150 * len(items), temperature=0)
    except Exception as exc:
        logger.warning("anchored writing: rewrite failed, the failing statements kept or removed by their rule: %s", exc)
        return {}
    out: Dict[int, Dict[str, str]] = {}
    for item in raw if isinstance(raw, list) else []:
        try:
            k = int(item.get("i")) - 1
        except (TypeError, ValueError, AttributeError):
            continue
        if 0 <= k < len(items) and k not in out:
            out[k] = {"tipo": str(item.get("tipo") or "").strip().lower(),
                      "frase": " ".join(str(item.get("frase") or "").split())}
    return out


def _source_ids(sentence: str, ids: List[str], cat: Catalogue, documents, known,
                allowed: Optional[Sequence[str]] = None, articles_only: bool = False) -> List[str]:
    """The passages a failing statement is rewritten from, among the section's own (`allowed`): its anchors, else
    those of what it cites; articles_only (case law to be restated as the rule of its article): article passages."""
    from . import plan_then_write as ptw
    ok = (lambda i: i in cat.passages and (allowed is None or i in allowed)
          and (not articles_only or cat.passages[i]["kind"] == "article"))
    own = [i for i in ids if ok(i)]
    if own:
        return own
    body = strip_anchors(sentence)
    out = [pid for _, _, rs in ptw._citations(body, documents or [], known) for r in rs for pid in cat.of_ref(r)]
    if not articles_only:
        out += [pid for n in _RULING_NUMBER_RE.findall(body) for pid in cat.of_ruling(n)]
    return [i for i in dict.fromkeys(out) if ok(i)]


# --- the verification -------------------------------------------------------------------------------------------------
# aw-on (fi-off vs aw-on: MENTIONED 71 -> 61, 165 statements removed): the removals took expected articles with them.
# What a failing statement becomes:
#   cites an article of the checked material (Catalogue.checked: graded 2 or more, named in the case, a tool's or a
#     term's)           rewritten from THAT article's passage and anchored there; the rewrite failing, kept as it is --
#                       never removed, its citation never dropped;
#   case law without a ruling passage of its section among its anchors   restated as the rule of its anchored
#                       article, else removed;
#   every citation outside the checked material     the citations dropped; the sentence kept if it still reads as one
#                                                   (_reads_whole), else removed;
#   anything else       rewritten from its passages, else kept.
# An application never loses the citation of a checked article either.

_DANGLING_END_RE = re.compile(
    r"(?:\b(?:di|del|della|dello|dei|degli|delle|dell['’]|a|al|alla|ai|agli|all['’]|da|dal|dalla|in|nel|nella|con|per|"
    r"su|tra|fra|e|o|che|ex|ai\s+sensi|a\s+norma|in\s+base|secondo|previsto|prevista|cui)\s*|[,(:])$", re.IGNORECASE)


def _reads_whole(sentence: str) -> bool:
    """A sentence left after its citations went still reads as one: three content words at least, not ending on a
    preposition, a conjunction or an open parenthesis."""
    body = sentence.strip().rstrip(".!?;:").strip()
    return len(stems(body)) >= 3 and not _DANGLING_END_RE.search(body)


def _is_checked(ref: tuple, checked: Sequence[tuple]) -> bool:
    from . import plan_then_write as ptw
    return any(ptw._same_ref(ref, c) for c in checked)


def _checked_ids(refs: Sequence[tuple], cat: Catalogue, rows: Dict[tuple, Dict[str, Any]], focus: str,
                 allowed: Optional[Sequence[str]]) -> List[str]:
    """The passages of the checked articles a statement cites: its section's first, else the catalogue's, else one
    made now from the article's row (the comma closest to the sentence)."""
    from . import plan_then_write as ptw
    out: List[str] = []
    for ref in refs:
        mine = cat.of_ref(ref)
        given = [i for i in mine if allowed is None or i in allowed]
        if given or mine:
            out += given or mine
            continue
        row = rows.get(ref) or next((r for k, r in rows.items() if ptw._same_ref(k, ref)), None)
        pid = _article_passage(cat, row, None, focus) if row is not None else None
        if pid:
            out.append(pid)
    return list(dict.fromkeys(out))


def _label(ref: tuple) -> str:
    return f"{ref[0]} {ref[1]}"


def verify(sections: List[Tuple[str, str]], cat: Catalogue, documents: Optional[List[Dict[str, Any]]] = None,
           article_rows: Optional[List[Dict[str, Any]]] = None, exempt: Sequence[str] = (),
           case_text: str = "", materials: Optional[Dict[str, str]] = None) -> List[Tuple[str, str]]:
    """The sections with every statement of the law or case law checked against its anchors (check) and a failing
    one treated by what it cites (the rules above); every application to the facts kept, an article it cites that
    its section's material (`materials`, by title) lacks -- and that is not checked -- losing only its citation; every
    anchor stripped. Sentences of `exempt` texts (the standard blocks written in code) are not checked."""
    from . import plan_then_write as ptw
    known = {r["_article"] for r in article_rows or [] if r.get("_article")}
    rows: Dict[tuple, Dict[str, Any]] = {}
    for row in article_rows or []:
        if row.get("_article") and row["_article"] not in rows:
            rows[row["_article"]] = row
    checked = list(cat.checked or [])
    exempt_text = " ".join(" ".join(t.split()) for t in exempt)
    names = _names(case_text)
    plans = []
    for title, text in sections:
        allowed = cat.section_ids.get(title)
        material_refs = ptw._material_refs((materials or {}).get(title, ""), documents or []) \
            if materials and title in materials else None
        work = []           # (start, end, sentence, kind, why, action, src, refs, ratio)
        stats = {"section": title, "sentences": 0, "anchored": 0, "law": 0, "case_law": 0, "application": 0,
                 "unclear": 0, "unanchored_citing": 0, "failed": {}, "overlap_failures": {}, "rewritten": 0,
                 "removed": 0, "removed_case_law": 0, "removed_without_citation": 0, "kept_checked": 0,
                 "kept_failing": 0, "citations_dropped": 0, "kept_as_application": 0}
        for s, e in ptw._sentence_spans(text):
            sentence = text[s:e]
            stats["sentences"] += 1
            if exempt_text and " ".join(strip_anchors(sentence).split()) in exempt_text:
                continue
            ids = [i.strip() for m in ANCHOR_RE.finditer(sentence) for i in re.split(r"[,;]", m.group(1))]
            stats["anchored"] += bool(ids)
            kind = classify(sentence, case_text, names)
            stats[kind or "unclear"] += 1
            body = strip_anchors(sentence)
            refs = [r for _, _, rs in ptw._citations(body, documents or [], known) for r in rs]
            if kind == "application":
                work.append((s, e, sentence, kind, None, "application", [], refs, None))
                continue
            why = check(sentence, ids, cat, documents, known, allowed, case_law=kind == "case_law")
            stats["unanchored_citing"] += why == "unanchored citation"
            if not why:
                continue
            stats["failed"][why] = stats["failed"].get(why, 0) + 1
            ratio = None
            if why == "overlap":
                ratio = overlap(body, [cat.passages[i] for i in ids])[0]
                b = _bucket(ratio)
                stats["overlap_failures"][b] = stats["overlap_failures"].get(b, 0) + 1
            on_checked = [r for r in refs if _is_checked(r, checked)]
            if on_checked:
                action = "checked"
                src = _checked_ids(on_checked, cat, rows, body, allowed)
            elif kind == "case_law" and not [i for i in cat.rulings(ids) if allowed is None or i in allowed]:
                action = "case_law"
                src = _source_ids(sentence, ids, cat, documents, known, allowed, articles_only=True)
            elif refs:
                action, src = "outside", []
            else:
                action = "rewrite"
                src = _source_ids(sentence, ids, cat, documents, known, allowed)
            work.append((s, e, sentence, kind, why, action, src, refs, ratio))
        plans.append((title, text, work, stats, material_refs))

    def asks(item) -> bool:
        return bool(item[4]) and item[5] in ("checked", "case_law", "rewrite") and bool(item[6])

    def rewrite(plan):
        items = [(strip_anchors(w[2]).strip(), w[6],
                  "case_law" if w[3] == "case_law" and w[5] in ("case_law", "checked")
                  else "classify" if w[3] is None else "law")
                 for w in plan[2] if asks(w)]
        return _anchor_rewrite(cat, items) if items else {}
    with ThreadPoolExecutor(max_workers=max(1, min(6, len(plans)))) as pool:
        replies = list(pool.map(rewrite, plans))
    out, entries, stats_out = [], [], []
    for (title, text, work, stats, material_refs), reply in zip(plans, replies):
        changes: Dict[Tuple[int, int], Optional[str]] = {}
        n = 0
        allowed = cat.section_ids.get(title)
        for w in work:
            s, e, sentence, kind, why, action, src, refs, ratio = w
            body = strip_anchors(sentence)
            got = {}
            if asks(w):
                got = reply.get(n) or {}
                n += 1
            entry = {"section": title, "kind": kind or "law (by the call)", "why": why, "action": action,
                     "sentence": body[:400], "cited": [_label(r) for r in refs],
                     **({"overlap": round(ratio, 3)} if ratio is not None else {})}
            if action == "application" or (kind is None and got.get("tipo") == "applicazione"):
                # the answer's reasoning: kept; a citation neither its section's material nor the checked one holds
                # goes alone
                if material_refs is not None:
                    drop = [r for r in refs if not any(ptw._same_ref(r, m) for m in material_refs)
                            and not _is_checked(r, checked)]
                    if drop:
                        new = drop_citations(body, drop, documents, known)
                        changes[(s, e)] = new
                        stats["citations_dropped"] += len(drop)
                        entries.append({**entry, "kind": "application", "how": "citation dropped", "new": new[:400],
                                        "dropped": [_label(r) for r in drop]})
                if kind is None:
                    stats["kept_as_application"] += 1
                continue
            if action == "outside":
                new = drop_citations(body, refs, documents, known)
                stats["citations_dropped"] += len(refs)
                if _reads_whole(new):
                    how = "citation dropped"
                    if sentence.rstrip()[-1:] in ".!?;:" and new[-1:] not in ".!?;:":
                        new += sentence.rstrip()[-1]
                else:
                    how, new = "removed", None
                    stats["removed"] += 1
                    stats["removed_without_citation"] += 1
                changes[(s, e)] = new
                entries.append({**entry, "how": how, "dropped": entry["cited"], **({"new": new[:400]} if new else {})})
                continue
            new = got.get("frase") or None
            if new and (check(strip_anchors(new), src, cat, documents, known, None if action == "checked" else allowed)
                        or (kind == "case_law" and action in ("case_law", "checked") and _CASE_LAW_RE.search(new))):
                new = None
            if new:
                how = "rewritten"
                stats["rewritten"] += 1
                if sentence.rstrip()[-1:] in ".!?;:" and new[-1:] not in ".!?;:":
                    new += sentence.rstrip()[-1]
                changes[(s, e)] = new
            elif action == "case_law":
                how = "removed"
                stats["removed"] += 1
                stats["removed_case_law"] += 1
                changes[(s, e)] = None
            else:        # a checked article's statement, or one with nothing to remove it for: kept as written
                how = "kept"
                stats["kept_checked" if action == "checked" else "kept_failing"] += 1
            entries.append({**entry, "how": how, "from": src, **({"new": new[:400]} if new else {})})
        if changes:
            from . import grounded_review
            text = grounded_review.apply(text, changes)
            if stats["removed"]:
                text, _ = ptw.drop_dangling(text, title)
        out.append((title, strip_anchors(text).strip() or strip_anchors(text)))
        stats_out.append(stats)
    keys = ("sentences", "anchored", "law", "case_law", "application", "unclear", "unanchored_citing", "rewritten",
            "removed", "removed_case_law", "removed_without_citation", "kept_checked", "kept_failing",
            "citations_dropped", "kept_as_application")
    totals: Dict[str, Any] = {k: sum(st[k] for st in stats_out) for k in keys}
    totals["overlap_failures"] = {}
    for st in stats_out:
        for b, v in st["overlap_failures"].items():
            totals["overlap_failures"][b] = totals["overlap_failures"].get(b, 0) + v
    vlog("anchored_writing", {"passages": cat.log(), "sections": stats_out, "entries": entries, "totals": totals})
    return out
