"""v60 (E): the read-back pass on the finished interactive answer, after every existing check.

1. A claim and its negation on the same article («l'art. 52 c.p. si applica» / «l'art. 52 c.p. non si applica»): the
   one the requisites' statuses support stays (a requisite «non risulta»: the negation; every one «provato»: the claim),
   else the first; the other sentence is dropped. Conditional sentences («se», «qualora», «ove», «potrebbe») are no
   claim.
2. A fact asserted that is in neither the case (its text, the uploaded text, the answered questions: the sources) nor
   the articles' texts or the computed block -- a date, a year, an amount, a percentage, a person's name -- marks its
   sentence «(non risulta dal caso)».
3. Then the output lint (scripts/answer_lint.py, when present), logged.
v60c: before them, a sentence stating what an article provides that shares no stem with the article's text is dropped
(ungrounded); claims are also compared across consecutive paragraphs (contradictions_across); numbers inside citations
are no fact; an amount the case's amounts give by one operation is «(calcolato)».
The code-rendered parts (headings, status lines, the procedural section) are not read.
"""

import importlib.util
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

NOT_IN_CASE = " (non risulta dal caso)"
_AFFIRM = (r"si applica|è applicabile|sussiste|ricorre|è configurabile|è integrat[oa]|è fondat[oa]|spetta|"
           r"è dovut[oa]|è responsabile|è punibile")
_CLAIM_RE = re.compile(rf"\b(?:{_AFFIRM})\b", re.IGNORECASE)
_NEGATED_RE = re.compile(rf"\bnon (?:(?:più|ancora|mai) )?(?:{_AFFIRM})\b", re.IGNORECASE)
_CONDITIONAL_RE = re.compile(r"\b(?:se|qualora|ove|laddove|potrebbe|potrebbero|potrà|ipotesi|eventual\w*|occorre "
                             r"verificare|da verificare|da provare)\b", re.IGNORECASE)
_ARTICLE_RE = re.compile(r"\bart(?:icolo|t?\.)\s*(\d+(?:-[a-z]+)?)(?:,\s*co(?:mma|\.)\s*\d+,?)?\s*"
                         r"(c\.\s*c\.|c\.\s*p\.\s*p\.|c\.\s*p\.\s*c\.|c\.\s*p\.\s*a\.|c\.\s*p\.)", re.IGNORECASE)
_SENTENCE_RE = re.compile(r"[^.!?\n]+(?:[.!?]+(?=\s|$)|$)")
_MONTHS = r"gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre"
_ATOMS = (
    ("data", re.compile(rf"\b\d{{1,2}}(?:°)?\s+(?:{_MONTHS})\s+\d{{4}}\b|\b\d{{1,2}}[/.]\d{{1,2}}[/.]\d{{2,4}}\b",
                        re.IGNORECASE)),
    ("anno", re.compile(r"(?<![\d/.-])(?:19|20)\d{2}(?![\d/])")),
    ("importo", re.compile(r"(?:€|euro)\s*\d+(?:[.,]\d+)*|\d+(?:[.,]\d+)*\s*(?:€|euro)\b", re.IGNORECASE)),
    ("percentuale", re.compile(r"\b\d+(?:[.,]\d+)?\s*%")),
)
_NAME_RE = re.compile(r"(?<![.\w«\"(])(?<!art\. )\b([A-ZÀ-Ü][a-zà-ü]{2,})\b")
_NOT_NAMES = {
    "art", "artt", "codice", "corte", "cassazione", "tribunale", "giudice", "pace", "assise", "stato", "repubblica",
    "costituzione", "italia", "comune", "regione", "provincia", "consiglio", "procura", "pubblico", "ministero", "tar",
    "legge", "decreto", "sezione", "capo", "titolo", "libro", "parte", "cliente", "controparte", "avvocato", "parere",
    "norme", "norma", "obiezioni", "conclusioni", "risposta", "strategia", "profili", "procedibilità", "competenza",
    "mediazione", "termini", "quanto", "inoltre", "tuttavia", "pertanto", "infatti", "dunque", "quindi", "ai", "nel",
    "la", "il", "lo", "le", "gli", "un", "una", "per", "con", "in", "se", "non", "ciò", "questo", "questa", "tale",
    "euro", "unione", "europea", "cedu", "inps", "inail", "asl", "srl", "spa", "snc", "sas", "codice civile",
    # v60c: places of offices (TAR Lazio) and centuries are no person
    "abruzzo", "basilicata", "calabria", "campania", "emilia", "romagna", "friuli", "venezia", "giulia", "lazio",
    "liguria", "lombardia", "marche", "molise", "piemonte", "puglia", "sardegna", "sicilia", "toscana", "trentino",
    "adige", "umbria", "valle", "aosta", "veneto", "ottocento", "novecento", "settecento", "duemila",
}
_SKIP_LINE_RE = re.compile(r"^\s*(?:\*\*|- |\d+[.)]\s|#|>|artt?\.\s*\d|Altre norme del tema|Norme considerate e non applicabili)")


def _plain(text: str) -> str:
    return " ".join(str(text or "").lower().split())


def _norm_code(code: str) -> str:
    return re.sub(r"\s+", "", code.lower())


# --- 1. a claim and its negation -----------------------------------------------------------------------------------

def _polarity(sentence: str) -> Optional[bool]:
    if _CONDITIONAL_RE.search(sentence):
        return None
    if _NEGATED_RE.search(sentence):
        return False
    if _CLAIM_RE.search(sentence):
        return True
    return None


def _cited(sentence: str, documents: Optional[Sequence[Dict[str, Any]]] = None) -> List[str]:
    """v60e: the articles a sentence cites, every form (citations: a code named with spaces too), as «N code» keys."""
    from . import citations
    return list(dict.fromkeys(citations.compact(r) for r in citations.refs_in(sentence, documents)))


def contradictions(sections: List[Tuple[str, str]], statuses: Dict[str, List[str]], log: List[Dict[str, Any]],
                   skip: Sequence[str] = (), documents: Optional[Sequence[Dict[str, Any]]] = None
                   ) -> List[Tuple[str, str]]:
    """statuses: {"52 c.p.": [the article's requisites' statuses]}. Over the whole answer, by article."""
    from .plan_then_write import _sentence_spans
    seen: Dict[str, Tuple[bool, str]] = {}
    drop: Dict[int, List[Tuple[int, int]]] = {}
    for i, (title, text) in enumerate(sections):
        if title in skip:
            continue
        for start, end in _sentence_spans(text or ""):
            sentence = text[start:end]
            if _SKIP_LINE_RE.match(text[text.rfind("\n", 0, start) + 1:]):
                continue
            cited = set(_cited(sentence, documents))
            if len(cited) != 1:
                continue
            polarity = _polarity(sentence)
            if polarity is None:
                continue
            label = next(iter(cited))
            if label not in seen:
                seen[label] = (polarity, sentence.strip())
                continue
            first, said = seen[label]
            if first == polarity:
                continue
            states = statuses.get(label) or []
            keep = (False if "non risulta" in states else True if states and all(s == "provato" for s in states)
                    else first)
            if keep == polarity:        # the later one is right: the first goes (only when still in this answer)
                log.append({"article": label, "kept": sentence.strip()[:200], "dropped": said[:200]})
                seen[label] = (polarity, sentence.strip())
                for j, (_, other) in enumerate(sections):
                    at = (other or "").find(said)
                    if at >= 0:
                        drop.setdefault(j, []).append((at, at + len(said)))
                        break
            else:
                log.append({"article": label, "kept": said[:200], "dropped": sentence.strip()[:200]})
                drop.setdefault(i, []).append((start, end))
    out = []
    for i, (title, text) in enumerate(sections):
        for s, e in sorted(drop.get(i, []), reverse=True):
            text = text[:s] + text[e:]
        out.append((title, re.sub(r"[ \t]{2,}", " ", text).replace(" \n", "\n") if i in drop else text))
    return out


# --- 2. a fact in no source ------------------------------------------------------------------------------------------

# v60c: the numbers of a citation (art. 2043 c.c., co. 3, d.lgs. 28/2010, l. 241/1990, n. 12) are no fact of the case
_CITATION_RE = re.compile(
    r"\bartt?\.\s*\d+(?:-[a-z]+)?(?:\s*(?:,|e)\s*\d+(?:-[a-z]+)?)*(?:\s*,?\s*(?:co\.|comma|commi)\s*\d+(?:\s*(?:,|e)\s*\d+)*)?"
    r"(?:\s+(?:(?:[A-ZÀ-Ü][\w'’.]*|sul|sulla|sui|del|della|dei|delle|di|e|in|per|d\.\s*lgs\.)\s*)+(?:\d{4})?)?"
    r"|\b(?:co\.|comma|commi)\s*\d+(?:\s*(?:,|e|-)\s*\d+)*"
    r"|\b(?:d\.\s*lgs\.|d\.\s*l\.|d\.\s*p\.\s*r\.|l\.|legge|dpr|n\.|reg\.|dir\.)\s*(?:n\.\s*)?\d+(?:\s*/\s*\d+)?"
    r"(?:\s+del\s+\d{4})?"
    r"|\b\d+\s*/\s*(?:19|20)\d{2}\b", re.IGNORECASE)


def _without_citations(sentence: str) -> str:
    return _CITATION_RE.sub(lambda m: " " * len(m.group(0)), sentence)


def _atoms(sentence: str) -> List[Tuple[str, str]]:
    sentence = _without_citations(sentence)
    found = []
    taken = []
    for kind, pattern in _ATOMS:
        for m in pattern.finditer(sentence):
            if any(s <= m.start() < e for s, e in taken):
                continue
            taken.append((m.start(), m.end()))
            found.append((kind, m.group(0)))
    for m in _NAME_RE.finditer(sentence):
        word = m.group(1)
        before = sentence[:m.start()]
        after = sentence[m.end():]
        prev = re.search(r"(\S+)\s+$", before)
        # a person's name stands mid-clause after a lower-case word, alone: not a sentence's or a clause's first word,
        # not part of a run of capitalised words (an office, a law, «Codice del Consumo»), not an adverb in -mente
        if not prev or not re.fullmatch(r"[a-zà-ü'’]+", prev.group(1)) or word.lower() in _NOT_NAMES \
                or word.lower().endswith("mente") or re.match(r"\s+(?:[A-ZÀ-Ü]|(?:del|della|di|dei|delle)\s+[A-ZÀ-Ü])", after) \
                or re.search(r"[A-ZÀ-Ü][\w'’]*\s+(?:del|della|di|dei|delle|sul|sulla)\s+$", before):
            continue
        found.append(("nome", word))
    return found


def _in_source(kind: str, atom: str, source: str) -> bool:
    if kind == "importo":
        digits = re.sub(r"[^\d]", "", atom)
        return bool(digits) and digits in re.sub(r"[^\d]", "", source)
    if kind == "nome":
        return atom.lower()[:5] in source
    if kind == "data":
        return _plain(atom) in source or all(p in source for p in re.findall(r"\d+", atom))
    return _plain(atom) in source


CALCULATED = " (calcolato)"
_NUMBER_RE = re.compile(r"\d+(?:\.\d{3})*(?:,\d+)?|\d+(?:[.,]\d+)?")
_SMALL = range(2, 13)


def _number(text: str) -> Optional[float]:
    """«12.000» / «12.000,50» / «1500» / «7,5» -> float."""
    t = text.strip()
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+(?:,\d+)?", t):
        t = t.replace(".", "").replace(",", ".")
    else:
        t = t.replace(",", ".")
    try:
        return float(t)
    except ValueError:
        return None


def case_amounts(case: str) -> Tuple[List[float], List[float]]:
    """(the case's amounts -- the numbers beside € / euro --, its percentages)."""
    plain = _without_citations(case or "")
    amounts = [_number(m.group(1) or m.group(2)) for m in re.finditer(
        r"(?:€|euro)\s*(\d+(?:[.,]\d+)*)|(\d+(?:[.,]\d+)*)\s*(?:€|euro)\b", plain, re.IGNORECASE)]
    rates = [_number(m.group(1)) for m in re.finditer(r"(\d+(?:[.,]\d+)?)\s*(?:%|per\s*cento)", plain, re.IGNORECASE)]
    return [a for a in amounts if a], [r for r in rates if r]


def derivable(value: float, amounts: Sequence[float], rates: Sequence[float]) -> bool:
    """An amount the case's amounts give by one operation: a sum or a difference of two, a multiple or a fraction by
    2-12 (months, instalments), a percentage the case states."""
    near = lambda x: abs(x - value) <= max(0.01, 0.005 * value)        # noqa: E731
    for i, a in enumerate(amounts):
        if any(near(a * k) or near(a / k) for k in _SMALL):
            return True
        if any(near(a * r / 100) or near(a * (1 + r / 100)) for r in rates):
            return True
        for b in amounts[i + 1:]:
            if near(a + b) or near(abs(a - b)):
                return True
    return False


def unsupported_facts(sections: List[Tuple[str, str]], sources: Sequence[str], log: List[Dict[str, Any]],
                      skip: Sequence[str] = (), case: str = "") -> List[Tuple[str, str]]:
    """Each prose sentence asserting an atom (a date, a year, an amount, a percentage, a name) in no source gets
    NOT_IN_CASE, once; v60c: an amount the case's own amounts give by one operation gets CALCULATED instead; numbers
    inside a citation are no atom."""
    from .plan_then_write import _sentence_spans
    source = _plain(" ".join(str(s or "") for s in sources))
    amounts, rates = case_amounts(case or " ".join(str(s or "") for s in sources[:1]))
    out = []
    for title, text in sections:
        if title in skip or not text:
            out.append((title, text))
            continue
        pieces = []
        cited = set(re.findall(r"\d+", " ".join(m.group(0) for m in _CITATION_RE.finditer(text))))
        for start, end in _sentence_spans(text):
            sentence = text[start:end]
            if _SKIP_LINE_RE.match(text[text.rfind("\n", 0, start) + 1:]) or NOT_IN_CASE.strip() in sentence \
                    or "«" in sentence or "»" in sentence or re.match(r"\s*\(co\.", sentence):
                continue            # the article's own words are no fact of the case
            atoms = [(kind, a) for kind, a in _atoms(sentence) if not _in_source(kind, a, source)
                     and not (kind == "anno" and a in cited)]
            if not atoms or CALCULATED.strip() in sentence:
                continue
            computed = [a for kind, a in atoms if kind == "importo" and (lambda v: v is not None and derivable(
                v, amounts, rates))(_number(re.sub(r"[^\d.,]", "", a).strip(".,")))]
            missing = [a for _, a in atoms if a not in computed]
            mark = NOT_IN_CASE if missing else CALCULATED
            stop = re.search(r"[.!?]+\s*$", sentence)
            marked = (sentence[:stop.start()] + mark + sentence[stop.start():]) if stop else sentence.rstrip() + mark
            pieces.append((start, end, marked))
            log.append({"section": title, "not_in_case": missing, "calculated": computed,
                        "sentence": sentence.strip()[:200]})
        for start, end, now in sorted(pieces, reverse=True):
            text = text[:start] + now + text[end:]
        out.append((title, text))
    return out


# --- v60c: a claim and its negation across consecutive paragraphs -------------------------------------------------------

_NEGATION_RE = re.compile(r"\b(?:non|né|nessun[oa]?|mai)\b", re.IGNORECASE)
CLAIM_OVERLAP = 0.6


def _claim_stems(sentence: str) -> set:
    plain = _NEGATION_RE.sub(" ", _without_citations(sentence).lower())
    return {w[:6] for w in re.findall(r"[a-zà-ü]{4,}", plain)} - {"quindi", "pertan", "inoltr", "tuttav", "infatt",
                                                                   "questo", "questa", "risult", "caso"}


def _paragraph_spans(text: str) -> List[Tuple[int, int]]:
    return [(m.start(), m.end()) for m in re.finditer(r"[^\n]+(?:\n(?!\s*\n)[^\n]+)*", text or "")]


def contradictions_across(sections: List[Tuple[str, str]], log: List[Dict[str, Any]],
                          skip: Sequence[str] = ()) -> List[Tuple[str, str]]:
    """A sentence asserting what a sentence of the same or the next prose paragraph denies (no article needed: their
    content words, the negation aside, overlap by CLAIM_OVERLAP of all their words -- Jaccard --, one negated and the
    other not, neither conditional nor a quotation of the article): the later one is dropped."""
    from .plan_then_write import _sentence_spans
    paragraphs = []                 # (section index, start, end, [(s, e, polarity, stems)])
    for i, (title, text) in enumerate(sections):
        if title in skip:
            continue
        for ps, pe in _paragraph_spans(text):
            if _SKIP_LINE_RE.match(text[ps:pe]):
                continue
            sentences = []
            for s, e in _sentence_spans(text[ps:pe]):
                sentence = text[ps + s:ps + e]
                if _CONDITIONAL_RE.search(sentence) or "«" in sentence or re.match(r"\s*\(co\.", sentence):
                    continue            # a conditional, or the article's own words: no claim of the answer's
                stems = _claim_stems(sentence)
                if len(stems) >= 4:
                    sentences.append((ps + s, ps + e, bool(_NEGATION_RE.search(sentence)), stems, sentence))
            paragraphs.append((i, sentences))
    drop: Dict[int, List[Tuple[int, int]]] = {}
    for k, (i, mine) in enumerate(paragraphs):
        window = mine + (paragraphs[k + 1][1] if k + 1 < len(paragraphs) else [])
        owner = [(i, x) for x in mine] + ([(paragraphs[k + 1][0], x) for x in paragraphs[k + 1][1]]
                                          if k + 1 < len(paragraphs) else [])
        for a in range(len(window)):
            for b in range(a + 1, len(window)):
                sa, sb = window[a], window[b]
                if sa[2] == sb[2]:
                    continue
                overlap = len(sa[3] & sb[3]) / max(1, len(sa[3] | sb[3]))
                j, later = owner[b]
                if overlap >= CLAIM_OVERLAP and (later[0], later[1]) not in drop.get(j, []):
                    drop.setdefault(j, []).append((later[0], later[1]))
                    log.append({"kept": sa[4].strip()[:200], "dropped": sb[4].strip()[:200], "overlap": round(overlap, 2)})
    out = []
    for i, (title, text) in enumerate(sections):
        for s, e in sorted(set(drop.get(i, [])), reverse=True):
            text = text[:s] + text[e:]
        out.append((title, re.sub(r"[ \t]{2,}", " ", text).replace(" \n", "\n") if i in drop else text))
    return out


# --- v60c: what an article provides, grounded in its text ---------------------------------------------------------------

_PROVIDES_RE = re.compile(r"\b(?:prevede|prevedono|dispone|dispongono|stabilisce|stabiliscono|sancisce|disciplina|"
                          r"regola|attribuisce|impone|consente|costituisce|costituiscono|configura|punisce|richiede|"
                          r"riconosce|prescrive|vieta|definisce|comporta)\b", re.IGNORECASE)
_HEADING_RE = re.compile(r"^\*\*Art\.\s+(\S+)\s+(c\.\s*c\.|c\.\s*p\.\s*p\.|c\.\s*p\.\s*c\.|c\.\s*p\.\s*a\.|c\.\s*p\.)",
                         re.IGNORECASE)


_GROUND_STOP = {"caso", "fatt", "part", "norm", "arti", "comm", "sens", "quan", "ques", "tale", "dell", "dall", "nell",
                "sull", "alla", "alle", "agli", "dagl", "degl", "sono", "esse", "ogni", "anch", "come", "quel", "qual",
                "stat", "vien", "dove", "cosi", "tutt", "altr", "dett", "prev", "disp", "stab", "cost", "rego"}


def _stems5(text: str) -> set:
    """Content stems: the first four letters of each word of four or more, small common words left out."""
    words = re.findall(r"[a-zà-ü]{4,}", _without_citations(str(text or "")).lower())
    return {w[:4] for w in words} - _GROUND_STOP


_SUBJECT_ARTICLE_RE = re.compile(r"^\W*(?:l['’]\s*|il\s+)?(?:art(?:icolo|\.)|artt\.)\s*\d", re.IGNORECASE)
_BY_ARTICLE_RE = re.compile(r"^\W*(?:ai sensi|a norma|secondo|in base a|come (?:prevede|dispone|stabilisce))\b"
                            r"[^.;:]{0,30}?\bart(?:icolo|\.)\s*\d[^,]{0,40},", re.IGNORECASE)


def ungrounded(sections: List[Tuple[str, str]], articles: Dict[str, str], log: List[Dict[str, Any]],
               skip: Sequence[str] = (), rubriche: Optional[Dict[str, str]] = None,
               documents: Optional[Sequence[Dict[str, Any]]] = None) -> List[Tuple[str, str]]:
    """A sentence stating what an article provides, whose words after its provision verb («prevede», «costituisce»,
    «dispone», …) share no stem with that article's text and rubrica (articles: {"120 c.p.": rubrica + text}), is
    dropped. Stating what an article provides: the article is its subject («L'art. 120 c.p. prevede …»); or it opens on
    it («Ai sensi dell'art. 120 c.p., …»); or, in the article's §2 block with no other citation, its subject (the
    words before the verb) is the article's institution -- every content stem of it in the rubrica (rubriche) («La
    querela costituisce …» under art. 120 c.p., «Diritto di querela»). A sentence applying the article to the facts is none of these."""
    from .plan_then_write import _sentence_spans
    rubriche = rubriche or {}
    out = []
    for title, text in sections:
        if title in skip or not text:
            out.append((title, text))
            continue
        cuts = []
        from . import citations          # v60e: a heading's code with spaces too
        heads, at = [], 0
        for line in text.split("\n"):
            h = citations.heading(line)
            if h:
                heads.append((at, citations.compact((h[0], h[1]))))
            at += len(line) + 1
        for s, e in _sentence_spans(text):
            sentence = text[s:e]
            if _SKIP_LINE_RE.match(text[text.rfind("\n", 0, s) + 1:]) or "«" in sentence:
                continue
            verb = _PROVIDES_RE.search(sentence)
            if not verb or _CONDITIONAL_RE.search(sentence[:verb.start()]):
                continue
            cited = _cited(sentence, documents)
            if len(cited) > 1:
                continue
            subject = sentence[:verb.start()]
            if cited:
                label = cited[0]
                about = (_SUBJECT_ARTICLE_RE.match(sentence) and len(subject) <= 80) or _BY_ARTICLE_RE.match(sentence)
            else:
                label = next((lab for at, lab in reversed(heads) if at < s), "")
                mine = _stems5(subject)
                about = bool(label) and bool(mine) and mine <= _stems5(rubriche.get(label, ""))
            if not about or label not in articles:
                continue
            said = _stems5(sentence[verb.end():])
            if said and not said & _stems5(articles[label]):
                cuts.append((s, e))
                log.append({"article": label, "dropped": sentence.strip()[:200]})
        for s, e in sorted(cuts, reverse=True):
            text = text[:s] + text[e:]
        out.append((title, re.sub(r"[ \t]{2,}", " ", text).replace(" \n", "\n") if cuts else text))
    return out


# --- v60g (2): a party's act or event, mapped to a sentence of the case --------------------------------------------------

UNVERIFIED = " (da verificare)"
ACT_SHARE = 0.25
_ROLE_SUBJECT_RE = re.compile(
    r"\b(?:il|la|lo|l['’]|i|gli|le|del|della|dello|dei|degli|delle|al|alla|allo|ai|agli|alle|dal|dalla|dai|dagli|"
    r"dalle)\s*(?:client\w*|assistit\w*|controparte|convenut\w*|attor\w*|attric\w*|ricorrent\w*|resistent\w*|"
    r"imputat\w*|indagat\w*|persona offesa|vittima|querelant\w*|denunciant\w*|conduttor\w*|locator\w*|"
    r"venditor\w*|acquirent\w*|compratore|committent\w*|appaltator\w*|datore di lavoro|lavorator\w*|"
    r"debitor\w*|creditor\w*|proprietari\w*|vicin\w*|condomin\w*|amministrator\w*|assicurator\w*|"
    r"danneggiat\w*|danneggiant\w*|erede|eredi|coniuge|ex coniuge|mutuatari\w*|mutuant\w*|comodatari\w*|"
    r"inquilin\w*|dipendent\w*|società|ditta|impresa|banca|compagnia)\b", re.IGNORECASE)
_ACT_VERB_RE = re.compile(
    r"\b(?:ha|hanno|aveva|avevano|avrebbe|avrebbero|è|sono|era|erano|fu|furono)\s+(?:già\s+|poi\s+|anche\s+|mai\s+|"
    r"più\s+|non\s+)?(?:stat[oaie]\s+)?(?!diritt|tutt|interess|obblig|titol|onere|cura\b)"
    r"[a-zà-ü]{1,}(?:ato|ata|ati|ate|uto|uta|uti|ute|ito|ita|iti|ite|esso|essa|"
    r"otto|otta|erto|erta|olto|olta|ritto|ritta|eso|esa|uso|usa|osto|osta|asto|asta|tto|tta|tti|tte)\b"
    r"|\b[a-zà-ü]{3,}(?:ò|ì|rono|aveva|evano)\b", re.IGNORECASE)
_RULE_TALK_RE = re.compile(r"\b(?:prevede|dispone|stabilisce|punisce|richiede|occorre|bisogna|deve|devono|dovrà|"
                           r"potrà|potrebbe|potrebbero|va provat|da provare|da verificare|da accertare|si applica|"
                           r"sussiste|ricorre|configura|è tenut\w*|è obbligat\w*|è legittimat\w*|"
                           r"responsabil\w*)\b", re.IGNORECASE)
_ACT_STOP = {"stato", "stata", "stati", "state", "avere", "essere", "quanto", "fatto", "fatti", "caso", "parte",
             "cliente", "controparte", "infatti", "quindi", "pertanto", "inoltre", "tuttavia", "questo", "questa",
             "quello", "quella", "anche", "ancora", "sempre", "dopo", "prima", "senza", "presso", "della", "delle",
             "dello", "degli", "nella", "nelle", "sulla", "sulle", "dalla", "dalle", "alla", "alle"}
_ROLE_STEMS = {"clien", "assis", "contr", "conve", "attor", "attri", "ricor", "resis", "imput", "indag", "offes", "vitti",
               "quere", "denun", "condu", "locat", "vendi", "acqui", "compr", "commi", "appal", "dator", "lavor", "debit",
               "credi", "vicin", "condo", "ammin", "assic", "danne", "erede", "eredi", "coniu", "mutua", "comod", "inqui",
               "dipen", "socie", "ditta", "impre", "banca", "compa"}


def _act_stems(text: str, names: Sequence[str] = ()) -> set:
    own = {w.lower()[:5] for n in names for w in re.findall(r"[^\W\d_]{3,}", str(n))}
    words = re.findall(r"[a-zà-ü]{4,}", _without_citations(str(text or "")).lower())
    return {w[:5] for w in words if w not in _ACT_STOP} - own - _ROLE_STEMS


def _party_subject(sentence: str, parties: Sequence[str]) -> bool:
    if _ROLE_SUBJECT_RE.search(sentence):
        return True
    for name in parties or []:
        words = [w for w in re.findall(r"[A-ZÀ-Ü][\w'’]{2,}", str(name))]
        if any(re.search(rf"\b{re.escape(w)}\b", sentence) for w in words):
            return True
    return False


def party_acts(sections: List[Tuple[str, str]], sentences: Sequence[Dict[str, str]], parties: Sequence[str],
               log: List[Dict[str, Any]], skip: Sequence[str] = ()) -> List[Tuple[str, str]]:
    """A prose sentence asserting an act or an event of a party -- a party named (its name or its role) and a finite
    past verb, not conditional, no rule talk, no quotation -- maps to the case sentence (sentences: issue_records'
    tagged ones) sharing the most of its content stems (ACT_SHARE of them, two at least), logged with that sentence's
    id and certainty; none does: the sentence is marked UNVERIFIED (once)."""
    from .plan_then_write import _sentence_spans
    if not sentences:
        return sections
    case = [(x, _act_stems(x.get("text"), parties)) for x in sentences]
    out = []
    for title, text in sections:
        if title in skip or not text:
            out.append((title, text))
            continue
        pieces = []
        for start, end in _sentence_spans(text):
            sentence = text[start:end]
            if _SKIP_LINE_RE.match(text[text.rfind("\n", 0, start) + 1:]) or "«" in sentence or "(da verificare" in sentence \
                    or NOT_IN_CASE.strip() in sentence or _CONDITIONAL_RE.search(sentence) or _RULE_TALK_RE.search(sentence):
                continue
            if not _party_subject(sentence, parties) or not _ACT_VERB_RE.search(sentence):
                continue
            mine = _act_stems(sentence, parties)
            if len(mine) < 2:
                continue
            best, share = None, 0.0
            for x, theirs in case:
                hit = len(mine & theirs)
                if hit >= 2 and hit / len(mine) > share:
                    best, share = x, hit / len(mine)
            if best is not None and share >= ACT_SHARE:
                log.append({"section": title, "mapped": best["id"], "certainty": best.get("certainty"),
                            "share": round(share, 2), "sentence": sentence.strip()[:200]})
                continue
            stop = re.search(r"[.!?]+\s*$", sentence)
            marked = (sentence[:stop.start()] + UNVERIFIED + sentence[stop.start():]) if stop else sentence.rstrip() + UNVERIFIED
            pieces.append((start, end, marked))
            log.append({"section": title, "marked": True, "share": round(share, 2), "sentence": sentence.strip()[:200]})
        for start, end, now in sorted(pieces, reverse=True):
            text = text[:start] + now + text[end:]
        out.append((title, text))
    return out


# --- 3. the lint ------------------------------------------------------------------------------------------------------

_LINT = None


def _lint_module():
    global _LINT
    if _LINT is None:
        path = Path(__file__).resolve().parents[2] / "scripts" / "answer_lint.py"
        if not path.is_file():
            _LINT = False
            return None
        import sys
        scripts = str(path.parent)
        if scripts not in sys.path:
            sys.path.insert(0, scripts)
        spec = importlib.util.spec_from_file_location("answer_lint", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _LINT = module
    return _LINT or None


def lint(round1: str, answer: str, choice: str) -> List[Tuple[str, str, str]]:
    """[(section, rule, excerpt)] of the output lint on this answer; [] when the lint is not installed."""
    module = _lint_module()
    if module is None:
        return []
    return module.lint_case({"round1": round1 or "", "round2": "", "answer": answer or "", "choice": choice or ""})


# --- the pass -----------------------------------------------------------------------------------------------------------

def _sections(draft: str, titles: Sequence[str]) -> List[Tuple[str, str]]:
    """The draft split at its numbered bold section titles («**2. Norme applicate ai fatti**»): [(title, text)], the
    title line kept in the text; the part before the first, titled ""."""
    out: List[Tuple[str, str]] = []
    marks = [m for m in re.finditer(r"(?m)^\*\*\d+\.\s+([^*\n]+)\*\*\s*$", draft or "")]
    if not marks:
        return [("", draft or "")]
    if marks[0].start():
        out.append(("", draft[:marks[0].start()]))
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(draft)
        out.append((m.group(1).strip(), draft[m.start():end]))
    return out


def read_back(draft: str, sources: Sequence[str], statuses: Dict[str, List[str]], skip: Sequence[str],
              log: Dict[str, Any], round1: str = "", choice: str = "", articles: Optional[Dict[str, str]] = None,
              case: str = "", rubriche: Optional[Dict[str, str]] = None,
              documents: Optional[Sequence[Dict[str, Any]]] = None, sentences: Optional[Sequence[Dict[str, str]]] = None,
              parties: Sequence[str] = ()) -> str:
    """The pass (module docstring); log gets "ungrounded", "contradictions" (by article, then across consecutive
    paragraphs), "not_in_case", "party_acts" (v60g), "lint". articles: {"N code": rubrica and text} for the grounding
    check; case: the case's own text, whose amounts make a computed one «(calcolato)»; sentences / parties (v60g 2): the
    case's tagged sentences and the parties' names, for the acts of a party (party_acts)."""
    for key in ("ungrounded", "contradictions", "not_in_case", "party_acts"):
        log.setdefault(key, [])
    sections = _sections(draft, ())
    if articles:
        sections = ungrounded(sections, articles, log["ungrounded"], skip, rubriche, documents)
    sections = contradictions(sections, statuses, log["contradictions"], skip, documents)
    sections = contradictions_across(sections, log["contradictions"], skip)
    sections = unsupported_facts(sections, sources, log["not_in_case"], skip, case)
    if sentences:
        sections = party_acts(sections, sentences, parties, log["party_acts"], skip)
    text = "".join(t for _, t in sections)
    try:
        log["lint"] = [list(x) for x in lint(round1, text, choice)]
    except Exception as exc:
        logger.warning("readback: the lint failed: %s", exc)
        log["lint"] = []
    return text
