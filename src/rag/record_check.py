"""v60g (e): the one check of the interactive answer, run last, against the record.

Before it, no step deletes: with a deferring ledger (answer_ledger, defer=True) every step's removals are put back and
recorded as proposals; the steps' rewrites and markers stand. This check reads every sentence of the answer -- the
procedural section (code-written), the «Altri profili» list and (v60q) §2's closing lines aside -- against the
per-article record and the case,
and is the only one that removes; each removal goes into the ledger with its rule and reason (and the steps that had
proposed it), so the record says why every sentence the writer wrote is not in the answer.

Removed (each rule a reference, never a heuristic of its own):
  fuori_dal_record          the sentence cites an article neither in the record (the topic's planned articles) nor in
                            the computed material (terms, tools, the procedural section), nor named by the case
  esito_contraddetto        it says an article does not apply while its record's verdict is not «non applicabile», or
                            that it applies while the verdict is «non applicabile»; v60i (2): of an «argomento del
                            cliente» (the counterparty's article whose elements fail) saying it does not apply is the
                            argument itself, saying it applies contradicts it
  norma_non_applicabile     (v60h 5) a §2 paragraph under an article whose record's verdict is «non applicabile»: the
                            article is one item of «Norme considerate e non applicabili», never a block
  norma_senza_testo         a §2 paragraph on an article whose text was not read (the record has no requisite)
  applicazione_non_chiesta  a §2 paragraph on an article the writer was not asked to apply, whose record has no fact for it
                            (v60i 1: no block due -- issue_records.block_due)
  virgolette_fuori_testo    a quotation, attributed to an article, its text and rubrica do not hold (rule_quote's measure)
  comma_inesistente         it cites a comma the article does not have (v65: or a number or a letter: phantom_item)
  contro_assistito          it proposes an action, a complaint or a claim against the client (the state's client)
  ripetizione               it repeats an earlier sentence of its section (repetition_guard's measure: four content
                            stems at least -- «Questo requisito è provato.» twice is two requisites)
  frammento                 a fragment: no subject (headless; a list's leading dash aside) or a cut citation's tail
                            (text_tidy.lead_fragment)
  testo_della_norma         (v60q, single shot) it reproduces an article's text -- its quotations of the text take half
                            its characters or more, or, unquoted, VERBATIM_SHARE of its word 4-grams are in one article's
                            text (VERBATIM_WORDS words at least): an article is cited as a reference, its text goes to the
                            response's article citations, never into the body
  regola_fuori_testo        (v60q, single shot) a rule sentence rule_quote judged outside its comma (it proposed it, never
                            replacing it with the comma's quotation)
  stato_soggettivo          (v60q, single shot) it gives a counterparty's state of mind (deliberato, intenzionale,
                            consapevole, in mala fede, dolo: interactive_flow.state_of_mind) as a fact the case does not
                            state -- the parere argues it from the facts; the «il caso non dice …» clause is cut, the
                            sentence removed only when nothing is left of it (logged as rewritten otherwise)
  requisito_non_registrato  (v64) in §1 or the conclusions (ctx["asserting"]), a sentence stating what a threat
                            article (the charge, the claim: ctx["threats"]) «richiede», «presuppone», «è necessario
                            che» … -- cited, or named by its rubrica -- whose requirement names none of its recorded
                            elements. v65: the writer review's semantic verdict (ctx["requirement_review"]: the
                            sentences it flagged, plan_then_write.REVIEW_REQUIREMENT) where its call answered for the
                            section; the word test (_unrecorded_requirement: half of the clause's content words at
                            least must be the elements') only where it failed or did not run
  alternativa_senza_fatti   (v65) in §1 (ctx["lead"]), a sentence citing an alternative qualification no fact of the
                            case holds (the first read's verdict: ctx["unfounded"]) -- it belongs to the alternatives
                            section, as a NO
  domanda_nel_parere        (v60k, single shot: ctx["single_shot"]) a question in the parere's body -- a fact the case
                            does not state is «da verificare: il caso non dice …» where it counts (v60q: no
                            «Punti da precisare» by default, OPEN_POINTS_MAX 0)
Marked, never removed:
  istituto_rubrica          it names an institution for an article («… (art. N)», «… ex art. N», «art. N (…)») that no
                            source clears (v60j, in order, the first to clear logged: the article's rubrica; the code's
                            institutions index -- a unit listing the article whose name shares a stem with the
                            institution, its head noun counted: «L'azione confessoria ex art. 1079 c.c.» and «Delle
                            azioni a difesa delle servitù» --; the article's own text; the names the record gives it,
                            its topic and the first read's institution): «(da verificare: art. N, rubrica «…»)»
  contraddizione_interna    it denies what the sentence before it in its paragraph asserts (readback's measure)
A proposal no rule confirms is rejected: the sentence stays (logged with the steps that proposed it).
"""

import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

REMOVE_RULES = ("fuori_dal_record", "esito_contraddetto", "norma_non_applicabile", "norma_senza_testo",
                "applicazione_non_chiesta",
                "virgolette_fuori_testo", "comma_inesistente", "contro_assistito", "ripetizione", "frammento",
                "testo_della_norma", "regola_fuori_testo", "stato_soggettivo", "domanda_nel_parere",
                "requisito_non_registrato", "alternativa_senza_fatti")
MARK_RULES = ("istituto_rubrica", "contraddizione_interna")
QUOTE_MIN = 25
_QUOTE_RE = re.compile(r"«([^»]{%d,})»" % QUOTE_MIN)
_AGAINST_RE = re.compile(r"\b(?:quereli?\w*|denunci\w*|agire|agisca|citare|convenire|convenuto in giudizio|"
                         r"chiedere (?:il )?risarciment\w*|domanda(?:re)? di risarcimento|azione)\b[^.;]{0,60}?"
                         r"\b(?:contro|nei confronti d\w*|a carico d\w*)\s+(?:il |la |lo |l['’]\s*|del |della )?"
                         r"(?:sig(?:\.|nor[ae]?)\s+|dott\.(?:ssa)?\s+)?", re.IGNORECASE)
# v60g (3): an institution named for an article -- the words just before «(art. N …)», «ex art. N», «ai sensi
# dell'art. N», «di cui all'art. N», or inside «art. N … (…)» after it
# v72: «art. N», «art N», «art.N», «articolo N» alike (the shared parser's forms, citations.article_refs)
_ART = r"(?:artt?\.?|articol[oi])\s*"
_BEFORE_RE = re.compile(r"(?:\(\s*|\b(?:ex|di cui all['’]|previst[oa] dall['’]|disciplinat[oa] dall['’])\s*)"
                        + _ART + r"(?P<num>\d+(?:-[a-z]+)?)", re.IGNORECASE)
_AFTER_RE = re.compile(r"\b" + _ART + r"(?P<num>\d+(?:-[a-z]+)?)[^(),;.]{0,12}\(\s*(?P<inst>[a-zà-ü'’ ]{6,60})\)",
                       re.IGNORECASE)
# v68: «lesioni personali ai sensi dell'art. 575 c.p.», «a norma dell'art.», «in base all'art.» -- the words before it an
# institution only when some article's rubrica names it (institution_article), never «Tizio ha agito ai sensi …»
_LOOSE_BEFORE_RE = re.compile(r"\b(?:ai sensi dell['’]|ai sensi degli|a norma dell['’]|in base all['’])\s*"
                              + _ART + r"(?P<num>\d+(?:-[a-z]+)?)", re.IGNORECASE)
_LOOSE_INTRO_RE = re.compile(r"\b(?:ai sensi dell['’]|ai sensi degli|a norma dell['’]|in base all['’])\s*$", re.IGNORECASE)
_VERB_END_RE = re.compile(r"(?:are|ere|ire|ato|ata|ati|ate|ito|ita|iti|ite|uto|uta|ando|endo|ebbe|ono|isce|iscono)$")
INSTITUTION_WORDS = 3
_INSTITUTION_STOP = {"della", "delle", "dello", "degli", "dalla", "dalle", "nella", "nelle", "sulla", "sulle", "alla",
                     "alle", "questo", "questa", "quello", "quella", "norma", "articolo", "comma", "codice", "sensi",
                     "previsto", "prevista", "disciplinato", "disciplinata", "secondo", "anche", "inoltre", "quindi",
                     "pertanto", "infatti", "applica", "applicabile", "richiamo", "base", "materia", "tema", "ipotesi",
                     "caso", "fatti", "fatto", "parte", "cliente",
                     # v60g: the words of any application, never an institution's name
                     "fattispecie", "requisiti", "requisito", "sussistenza", "presupposti", "presupposto", "disciplina",
                     "applicazione", "termine", "termini", "rientra", "rientrano", "integra", "integrano", "prescrive",
                     "dispone", "prevede", "stabilisce", "effetti", "effetto", "conseguenze", "ambito", "profilo",
                     "profili", "verifica", "decorrenza", "natura", "condotta", "violazione", "rispetto", "tutela",
                     "azione", "domanda", "richiesta", "giudizio", "reato", "delitto", "norme", "regola", "principio"}


def _stems(text: str) -> Set[str]:
    return {w[:5] for w in re.findall(r"[a-zà-ü]{5,}", str(text or "").lower()) if w not in _INSTITUTION_STOP}


def _plain(text: str) -> str:
    return " ".join(re.findall(r"[a-zà-ü0-9]+", str(text or "").lower()))


def _label_of(ref: Sequence[str]) -> str:
    from . import citations
    return citations.label(ref)


# --- the rules ------------------------------------------------------------------------------------------------------

def _outside(cited: Sequence[Tuple[str, str]], allowed: Set[str]) -> List[str]:
    return [_label_of(r) for r in cited if _label_of(r) not in allowed]


def _verdict_contradicted(sentence: str, label: str, records: Dict[str, Dict[str, Any]]) -> str:
    """The reason when the sentence says of the article the opposite of its record's verdict; "" otherwise."""
    from . import issue_records, readback
    from .interactive_flow import declares_inapplicable
    rec = records.get(label)
    if not rec or not rec.get("requisites"):
        return ""
    verdict = rec.get("verdict") or issue_records.verdict(rec)
    failing = (issue_records.NOT_APPLICABLE, issue_records.CLIENT_ARGUMENT)
    if verdict not in failing and declares_inapplicable(sentence):
        return f"dichiara non applicabile l'art. {label}, che il record dà «{verdict}»"
    if verdict == issue_records.CLIENT_ARGUMENT and _ATTRIBUTED_RE.search(sentence):
        return ""               # v60i (2): the other side's reliance, stated as theirs, is the argument's first half
    if verdict in failing and readback._polarity(sentence) is True:
        return f"dice applicabile l'art. {label}, che il record dà «{verdict}» ({issue_records.verdict_reason(rec)})"
    return ""


_QUESTION_END_RE = re.compile(r"\?\s*[»\"”)]*\s*$")
_ATTRIBUTED_RE = re.compile(r"\bcontroparte\b|\b(?:invoc|sostien|sosterr|eccepi|eccepisc|pretend|afferm|richiam|si\s+fond)"
                            r"\w*", re.IGNORECASE)


def _quote_outside(sentence: str, labels: Sequence[str], texts: Dict[str, str], rubriche: Dict[str, str]) -> str:
    """A quotation (QUOTE_MIN characters or more) none of the sentence's articles' texts or rubriche holds, by
    rule_quote's own measure (rule_quote._holds: its words, or nine tenths of its content words); a quotation cut by
    an ellipsis is held piece by piece."""
    from .rule_quote import _holds
    own = [lab for lab in labels if texts.get(lab)]
    if not own or len(own) < len(set(labels)):
        return ""               # an article whose text was not read: nothing to hold the quotation against
    sentence = re.sub(r"\[DA VERIFICARE:[^\]]*\]|\(da verificare[^)]*\)", " ", sentence, flags=re.IGNORECASE)
    for quote in _QUOTE_RE.findall(sentence):
        pieces = [p for p in re.split(r"\s*(?:\[\s*(?:…|\.\.\.)\s*\]|\(\s*(?:…|\.\.\.)\s*\)|…|\.\.\.)\s*", quote)
                  if len(p.strip()) >= 12]
        ground = [f"{rubriche.get(lab, '')} {texts[lab]}" for lab in own]
        if pieces and not any(all(_holds(p, g) for p in pieces) for g in ground):
            return f"virgolette «{quote[:60]}…» che non sono nel testo dell'art. {own[0]}"
    return ""


# v60q (single shot): the body cites articles, it never reproduces their text
REPRODUCED_SHARE = 0.5      # a sentence's characters its quotations of an article's text take
VERBATIM_SHARE = 0.75       # a sentence's word 4-grams one article's text holds
VERBATIM_WORDS = 6
_TAG_RE = re.compile(r"\(\s*co\.\s*\d+[^)]{0,12}\)|«\s*co\.\s*\d+[^»]{0,12}»|\[\d{1,2}\]|"
                     r"\bartt?\.\s*\d+(?:-[a-z]+)?(?:\s*,?\s*(?:co\.|comma)\s*\d+)?,?(?:\s+(?:[a-z]{1,4}\.\s?){1,4})?",
                     re.IGNORECASE)
_ANY_QUOTE_RE = re.compile(r"«([^»]{20,})»|“([^”]{20,})”")


def _grams(words: Sequence[str], n: int = 4) -> Set[Tuple[str, ...]]:
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def _reproduces(sentence: str, texts: Dict[str, str], memo: Dict[str, Set[Tuple[str, ...]]]) -> str:
    """The reason when the sentence reproduces an article's text (testo_della_norma); "" otherwise."""
    from .rule_quote import _holds
    body = _TAG_RE.sub(" ", sentence)
    chars = len(re.sub(r"[\s«»“”.,;:()\-–—]", "", body))
    if not chars:
        return ""
    held = 0
    for a, b in _ANY_QUOTE_RE.findall(body):
        quote = a or b
        label = next((lab for lab, t in texts.items() if t and _holds(quote, t)), None)
        if label:
            held += len(re.sub(r"[\s«»“”.,;:()\-–—]", "", quote))
            if held >= REPRODUCED_SHARE * chars:
                return f"riporta tra virgolette il testo dell'art. {label}"
    words = _plain(_ANY_QUOTE_RE.sub(lambda m: m.group(0).strip("«»“”"), body)).split()
    if len(words) < VERBATIM_WORDS:
        return ""
    mine = _grams(words)
    for label, text in texts.items():
        if not text:
            continue
        if label not in memo:
            memo[label] = _grams(_plain(text).split())
        if mine and len(mine & memo[label]) >= VERBATIM_SHARE * len(mine):
            return f"riproduce il testo dell'art. {label}"
    return ""


def _state_of_mind_cut(sentence: str, client: str = "", parties: Sequence[str] = ()) -> Optional[str]:
    """v60q: a sentence giving a counterparty's state of mind as what the case does not state -- the sentence without that
    clause («…, ma da verificare: il caso non dice se agì consapevolmente.» -> «….»), "" when nothing is left of it;
    None when it gives none (the client's own state of mind is a fact the client can state: interactive_flow.state_of_mind)."""
    from .interactive_flow import state_of_mind
    m = _MISSING_RE.search(sentence)
    if not m or not state_of_mind(sentence[m.start():], client, parties):
        return None
    head = re.sub(r"(?:[\s,;:(–—-]|\b(?:ma|e|però|tuttavia|anche se)\b)*$", "", sentence[:m.start()],
                  flags=re.IGNORECASE).rstrip()
    if len(re.findall(r"[^\W\d_]+", head)) < 4:
        return ""
    tail = sentence[len(sentence.rstrip()):]
    return head + ("" if head.endswith((".", "!", "?")) else ".") + tail


# the open-point lead in the writer's prose: «da verificare: il caso non dice …», «il caso non indica …»
_MISSING_RE = re.compile(r"(?:\(?\s*da\s+verificare\s*:\s*)?\b(?:il\s+caso|la\s+narrazione|i\s+fatti)\s+non\s+"
                         r"(?:dice|indica|precisa|chiarisce|specifica|riferisce)\b", re.IGNORECASE)


# v65: an article's numbered and lettered items («1) …», «a) …»), and a sentence citing one («art. 61, n. 4»,
# «lettera b) dell'art. 7»)
_ITEM_NUMBER_RE = re.compile(r"(?:^|[\s;:,.(«])(\d{1,2})\)\s")
_ITEM_LETTER_RE = re.compile(r"(?:^|[\s;:,.(«])([a-z])\)\s")
_ITEM_KIND = r"(n(?:umero|\.|°)|lett(?:era|\.)?)"
_ITEM_VALUE = r"(\d{1,2}|[a-z])\b\)?"
_ART_ITEM_RE = re.compile(r"\bart(?:icolo|\.)\s*(\d+(?:-[a-z]+)?)\s*(?:[a-z.]{2,8}\s*)?,?\s*(?:(?:comma|co\.)\s*\d{1,2}"
                          r"\s*,?\s*)?" + _ITEM_KIND + r"\s*" + _ITEM_VALUE, re.IGNORECASE)
_ITEM_OF_RE = re.compile(_ITEM_KIND + r"\s*" + _ITEM_VALUE + r"\s*,?\s*(?:del(?:l['’])?\s*(?:comma|co\.)\s*\d{1,2}\s*,?\s*)?"
                         r"(?:dell['’]\s*|dello\s+|del\s+)?art(?:icolo|\.)\s*(\d+(?:-[a-z]+)?)", re.IGNORECASE)
TEXT_CUT_MARGIN = 50


def article_items(text: str, cut: int = 0) -> Dict[str, Any]:
    """v65: {"numbers", "letters", "cut"} of an article's text -- its numbered and lettered items; cut: the text may be
    cut short (its length within TEXT_CUT_MARGIN of the cut given), so an item past its last one is unknown, not
    missing."""
    text = " ".join(str(text or "").split())
    return {"numbers": {int(n) for n in _ITEM_NUMBER_RE.findall(text)},
            "letters": set(_ITEM_LETTER_RE.findall(text)),
            "cut": bool(cut) and len(text) >= cut - TEXT_CUT_MARGIN, "read": bool(text)}


def phantom_item(sentence: str, documents, items: Dict[str, Dict[str, Any]]) -> str:
    """v65: why the sentence cites a number or a letter its article does not have («art. 61, n. 22 c.p.», «lettera f)
    dell'art. 7») -- "" when none, or when the article's text is not read or may be cut before it."""
    from . import citations
    found = [(m.group(1), m.group(2), m.group(3), m) for m in _ART_ITEM_RE.finditer(sentence or "")]
    found += [(m.group(3), m.group(1), m.group(2), m) for m in _ITEM_OF_RE.finditer(sentence or "")]
    for number, kind, value, m in found:
        labels = [_label_of(r) for r in citations.refs_in(sentence[m.start():m.end() + 30], documents)
                  if str(r[0]) == str(number)]
        for lab in labels:
            have = items.get(lab)
            if not have or not have.get("read"):
                continue
            letter = kind.lower().startswith("lett")
            pool = have["letters"] if letter else have["numbers"]
            value_k = value.lower() if letter else int(value) if value.isdigit() else None
            if value_k is None or value_k in pool:
                continue
            if have.get("cut") and (not pool or value_k > max(pool)):
                continue
            what = f"la lettera {value_k})" if letter else f"il numero {value_k})"
            has = ", ".join(str(x) for x in sorted(pool)) or "nessuno"
            return f"cita {what} dell'art. {lab}, che non l'ha (ha: {has})"
    return ""


def _phantom_comma(sentence: str, documents, commi: Dict[str, Set[int]]) -> str:
    from .interactive_flow import _ART_COMMA_RE, _COMMA_OF_RE
    from . import citations
    for pattern in (_ART_COMMA_RE, _COMMA_OF_RE):
        for m in pattern.finditer(sentence):
            number, comma = (m.group(1), m.group(2)) if pattern is _ART_COMMA_RE else (m.group(2), m.group(1))
            labels = [_label_of(r) for r in citations.refs_in(sentence[m.start():m.end() + 30], documents)
                      if str(r[0]) == str(number)]
            for lab in labels:
                have = commi.get(lab)
                if have and int(comma) not in have:
                    return f"cita il comma {comma} dell'art. {lab}, che ha i commi {sorted(have)}"
    return ""


_ACT_ON_RE = re.compile(r"\b(?:querelar\w*|denunciar\w*|citar\w*|convenir\w*|diffidar\w*|chiamar\w* in giudizio)\s+"
                        r"(?:anche\s+)?(?:il |la |lo |l['’]\s*)?(?:sig(?:\.|nor[ae]?)\s+|dott\.(?:ssa)?\s+)?",
                        re.IGNORECASE)


def _against_client(sentence: str, client: str) -> str:
    if not client:
        return ""
    names = [w for w in re.findall(r"[A-ZÀ-Ü][\w'’]{2,}", client)] or [client]
    for pattern in (_AGAINST_RE, _ACT_ON_RE):
        for m in pattern.finditer(sentence):
            rest = sentence[m.end():m.end() + 60]
            if any(re.match(rf"\s*{re.escape(n)}\b", rest) for n in names):
                return f"propone un'azione contro la parte assistita ({client})"
    return ""


def _nouns(words: Sequence[str]) -> List[str]:
    """The content nouns among words: five letters or more, no function word, no verb form."""
    return [w for w in words if len(w) >= 5 and w.lower() not in _INSTITUTION_STOP and not _VERB_END_RE.search(w.lower())]


# v60j (2): the sources that clear an institution named for an article, in the order they are tried
CLEAR_SOURCES = ("rubrica", "indice", "testo", "record")
# the words that name nothing in an index unit's heading -- the institution's head noun («azione», «domanda») counts
_FUNCTION_STOP = {"della", "delle", "dello", "degli", "dalla", "dalle", "dagli", "nella", "nelle", "negli", "sulla",
                  "sulle", "alla", "alle", "agli", "questo", "questa", "quello", "quella", "norma", "articolo", "comma",
                  "codice", "sensi", "anche", "inoltre", "quindi", "pertanto", "infatti", "secondo", "previsto",
                  "prevista", "disciplinato", "disciplinata"}


def _head_stems(text: str) -> Set[str]:
    return {w[:4] for w in re.findall(r"[a-zà-ü]{4,}", str(text or "").lower()) if w not in _FUNCTION_STOP}


def _index_units(label: str, index=None) -> Optional[str]:
    """The names of the code's institutions index units that list the article («Delle azioni a difesa delle servitù»
    for 1079 c.c.); "" when none lists it, None when the code has no index (fact_resolver.load_index)."""
    from . import fact_resolver
    number, _, code = str(label).partition(" ")
    entries = (index or fact_resolver.load_index)(code)
    if entries is None:
        return None
    return " ".join(f"{e.get('name') or ''} {e.get('rubricated') or ''}" for e in entries      # v60n: units only,
                    if e.get("level") != "ARTICOLO" and number in [str(a) for a in e.get("articles") or []])  # no leaf


def _cleared(label: str, nouns: Sequence[str], words: Sequence[str], rubriche: Dict[str, str], texts: Dict[str, str],
             names: Optional[Dict[str, str]] = None, index=None) -> Tuple[str, bool]:
    """(the first source that clears the institution for the article -- CLEAR_SOURCES --, or ""; whether the code has
    an index). nouns: the institution's content nouns; words: all its words, the head noun with them (the index)."""
    mine = _stems4(" ".join(nouns))
    if mine & _stems4(rubriche.get(label, "")):
        return "rubrica", True
    units = _index_units(label, index)
    if units and _head_stems(" ".join(words)) & _head_stems(units):
        return "indice", True
    if mine & _stems4(texts.get(label, "")):
        return "testo", units is not None
    if mine & _stems4((names or {}).get(label, "")):
        return "record", units is not None
    return "", units is not None


def _institution(sentence: str, documents, rubriche: Dict[str, str], texts: Dict[str, str],
                 names: Optional[Dict[str, str]] = None, cleared: Optional[List[Dict[str, Any]]] = None,
                 index=None, records: Optional[Dict[str, Dict[str, Any]]] = None
                 ) -> Optional[Tuple[str, str, bool, Tuple[int, int], Optional[str], bool]]:
    """(label, institution, whether its code has an index) when the sentence names an institution for an article -- the
    nouns among the
    INSTITUTION_WORDS words before its citation, or in the parenthesis after it -- that no source clears (_cleared:
    the article's rubrica, the code's institutions index, its text, the names the record gives it: its topic's title,
    the first read's institution that found it). cleared (v60j): each institution a source cleared, with the source.
    v68: (…, the cited number's span, the article whose rubrica names the institution -- institution_article --, whether
    the form is a loose one): «ai sensi dell'art. N» and its kind count only when such an article exists."""
    from . import citations
    found = []
    for m in _BEFORE_RE.finditer(sentence):
        head = re.sub(r"\b(?:ex|di cui all['’]|previst[oa] dall['’]|disciplinat[oa] dall['’])\s*$", "",
                      sentence[:m.start()], flags=re.IGNORECASE)
        words = re.findall(r"[a-zà-ü'’]+", head.lower())[-INSTITUTION_WORDS:]
        found.append((m, _nouns(words), words, False))
    for m in _AFTER_RE.finditer(sentence):
        words = re.findall(r"[a-zà-ü'’]+", m.group("inst").lower())
        found.append((m, _nouns(words), words, False))
    for m in _LOOSE_BEFORE_RE.finditer(sentence):           # v68
        head = _LOOSE_INTRO_RE.sub("", sentence[:m.end() - len(m.group(0))])
        words = re.findall(r"[a-zà-ü'’]+", head.lower())[-INSTITUTION_WORDS:]
        found.append((m, _nouns(words), words, True))
    for m, nouns, words, loose in found:
        if not _stems4(" ".join(nouns)):
            continue
        snippet = sentence[m.start():m.end() + 30]
        labels = [_label_of(r) for r in citations.refs_in(snippet, documents) if str(r[0]) == m.group("num")] or \
            [_label_of(r) for r in citations.article_refs(snippet) if str(r[0]) == m.group("num")]     # v72
        for lab in labels:
            # v69: the record's rubrica when the rows hold none (an article a road brought in: the posture's table)
            rubrica = rubriche.get(lab) or str(((records or {}).get(lab) or {}).get("rubrica") or "")
            if not rubrica:
                continue                # without the article's own words, nothing to check against
            by, indexed = _cleared(lab, nouns, words, {**rubriche, lab: rubrica}, texts, names, index)
            if by:
                if cleared is not None:
                    cleared.append({"article": lab, "institution": " ".join(nouns), "by": by})
                continue
            target = institution_article(nouns, lab, records or {}, rubriche)
            if loose and not target:
                continue                # v68: the words name no article of the material -- no institution
            return lab, " ".join(nouns), indexed, m.span("num"), target, loose
    return None


def _cited_instead(sentence: str, hit: Tuple[Any, ...], documents) -> str:
    """v68: the sentence with the mis-cited article replaced by the institution's (hit[4]: the number only, the code
    as written), or -- none -- without the citation (anchored_writing.drop_citations: its code, its «ai sensi dell'»)."""
    from . import anchored_writing
    (start, end), target = hit[3], hit[4]
    if target:
        return sentence[:start] + target.split(" ", 1)[0] + sentence[end:]
    number, _, code = hit[0].partition(" ")
    return anchored_writing.drop_citations(sentence, [(number, code)], documents)


def institution_article(nouns: Sequence[str], cited: str, records: Dict[str, Dict[str, Any]],
                        rubriche: Dict[str, str]) -> Optional[str]:
    """v68: the article of the institution a sentence names for another one -- the record's article whose rubrica shares
    the most stems with it (the records first, then the material's rows), in the cited article's code; None when none
    does, or two tie."""
    mine = _stems4(" ".join(nouns))
    code = cited.split(" ", 1)[1] if " " in cited else ""
    for pool in ({k: str((r or {}).get("rubrica") or "") for k, r in records.items()}, rubriche):
        scored = sorted(((len(mine & _stems4(rub)), lab) for lab, rub in pool.items()
                         if lab != cited and rub and lab.split(" ", 1)[-1] == code and mine & _stems4(rub)), reverse=True)
        if scored and (len(scored) == 1 or scored[0][0] > scored[1][0]):
            return scored[0][1]
        if scored:
            return None
    return None


def _stems4(text: str) -> Set[str]:
    """Four-letter stems of the words of four letters or more («vizio» and «vizi» one)."""
    return {w[:4] for w in re.findall(r"[a-zà-ü]{4,}", str(text or "").lower()) if w not in _INSTITUTION_STOP}


def _repeats(sentence: str, earlier: List[Tuple[set, set, set, str]]) -> str:
    from . import repetition_guard as rg
    mine = rg.stems(sentence)
    if len(mine) < rg.MIN_STEMS:
        return ""
    refs, numbers = rg._refs(sentence), rg._numbers(sentence)
    for other, other_refs, other_numbers, said in earlier:
        if len(mine & other) >= rg.OVERLAP * len(mine) and refs <= other_refs and numbers <= other_numbers:
            return f"ripete «{said[:80]}»"
    earlier.append((mine, refs, numbers, sentence))
    return ""


def _fragment(sentence: str) -> str:
    from . import text_tidy
    from .interactive_flow import _headless
    sentence = re.sub(r"^\s*[–—-]\s+(?=[A-ZÀ-Ü«(\"“])", "", sentence)       # «— Il venditore …»: a list's dash
    if _headless(sentence):
        return "frammento senza soggetto"
    if not text_tidy.lead_fragment(sentence.strip()):
        return "frammento di una citazione tagliata"
    return ""


def _denies_previous(sentence: str, previous: str) -> bool:
    from . import readback
    if not previous or readback._CONDITIONAL_RE.search(sentence) or readback._CONDITIONAL_RE.search(previous):
        return False
    a, b = readback._claim_stems(previous), readback._claim_stems(sentence)
    if len(a) < 4 or len(b) < 4:
        return False
    if bool(readback._NEGATION_RE.search(previous)) == bool(readback._NEGATION_RE.search(sentence)):
        return False
    return len(a & b) / max(1, len(a | b)) >= readback.CLAIM_OVERLAP


_ABBREVIATION_END_RE = re.compile(r"\b(?:[a-zA-Z]{1,5}\.){2,}$|\b(?:art|artt|co|n|lett|ss)\.$")


def _mark(sentence: str, mark: str) -> str:
    """The sentence with the mark before its full stop -- v60s: a sentence ending on an abbreviation («… c.c.») keeps it
    whole and takes its stop after the mark («… c.c. (da …).»)."""
    body = sentence.rstrip()
    if _ABBREVIATION_END_RE.search(body):
        return body + mark + "." + sentence[len(body):]
    stop = re.search(r"[.!?]+$", body)
    if not stop:
        return body + mark + sentence[len(body):]
    return body[:stop.start()] + mark + stop.group(0) + sentence[len(body):]


# --- the check ------------------------------------------------------------------------------------------------------

def check(sections: List[Tuple[str, str]], ctx: Dict[str, Any], ledger=None) -> Tuple[List[Tuple[str, str]], Dict[str, Any]]:
    """(sections, log). ctx: records, allowed (labels), glossed (labels), texts / rubriche / names ({label: …}), commi
    ({label: {numbers}}), documents, client, skip (titles), norms_title, numbering (the writer's «[N]» refs)."""
    from . import answer_ledger, citations
    from .plan_then_write import _sentence_spans
    records = ctx.get("records") or {}
    allowed = set(ctx.get("allowed") or ())
    glossed = set(ctx.get("glossed") or ())
    texts, rubriche, commi = ctx.get("texts") or {}, ctx.get("rubriche") or {}, ctx.get("commi") or {}
    documents = ctx.get("documents") or []
    skip = set(ctx.get("skip") or ())
    log: Dict[str, Any] = {"removed": [], "marked": [], "rejected": [], "rules": {}, "cleared": [], "rewritten": []}
    single = bool(ctx.get("single_shot"))
    grams: Dict[str, Set[Tuple[str, ...]]] = {}
    if ledger is not None:
        ledger.steps.append(answer_ledger.CHECK_STEP)
    out = []
    removed_norms: Set[str] = set()
    for title, text in sections:
        if title in skip or not (text or "").strip():
            out.append((title, text))
            continue
        units = answer_ledger.tracked(text, ctx.get("numbering") or (), documents)
        cuts: List[Tuple[int, int]] = []
        marks: Dict[Tuple[int, int], str] = {}
        earlier: List[Tuple[set, set, set, str]] = []
        previous = ""
        for u in units:
            s, e = u["start"], u["end"]
            sentence = text[s:e]
            if _code_line(text, s):     # v60q: the code's closing lines of §2 are no writer's sentence
                continue
            cited = citations.refs_in(sentence, documents)
            labels = [_label_of(r) for r in cited] or ([u["heading"]] if u["heading"] else [])
            in_norms = title == ctx.get("norms_title") and u["heading"]
            rule, why = "", ""
            outside = _outside(cited, allowed)
            if outside:
                rule, why = ("fuori_dal_record", f"cita art. {', '.join(outside)}: non è tra le norme degli argomenti scelti "
                             "né nel materiale calcolato")
            if not rule:
                for lab in list(dict.fromkeys(labels))[:3]:
                    why = _verdict_contradicted(sentence, lab, records)
                    if why:
                        rule = "esito_contraddetto"
                        break
            if not rule and in_norms:
                from . import issue_records
                rec = records.get(u["heading"])
                if rec is not None and (rec.get("verdict") or issue_records.verdict(rec)) == issue_records.NOT_APPLICABLE:
                    rule, why = ("norma_non_applicabile", f"paragrafo sull'art. {u['heading']}: esito del record «non "
                                 "applicabile», una riga in «Norme considerate e non applicabili»")
                elif rec is not None and not rec.get("requisites") and not texts.get(u["heading"]):
                    rule, why = "norma_senza_testo", f"paragrafo sull'art. {u['heading']}, il cui testo non è stato letto"
                elif rec is not None and glossed and u["heading"] not in glossed:
                    from . import issue_records
                    by_id = {x["id"]: x for x in ctx.get("sentences") or [] if x.get("id")}
                    if not issue_records.block_due(rec, by_id or None, ctx.get("open_questions")):
                        rule, why = ("applicazione_non_chiesta", f"paragrafo sull'art. {u['heading']}: applicazione "
                                     "non chiesta e nessun fatto a sostegno nel record")
            if not rule:
                why = _quote_outside(sentence, labels, texts, rubriche)
                rule = "virgolette_fuori_testo" if why else ""
            if not rule and single:     # v60q: an article is a reference, its text is the response's citations
                why = _reproduces(sentence, texts, grams)
                rule = "testo_della_norma" if why else ""
            if not rule and single and ledger is not None and "rule_quote" in ledger.proposed_by(u["norm"]):
                rule, why = ("regola_fuori_testo", "enuncia la regola di una norma con parole che il suo testo non "
                             "contiene (rule_quote), e il parere non ne riporta il testo")
            if not rule:
                why = _phantom_comma(sentence, documents, commi) or \
                    phantom_item(sentence, documents, ctx.get("items") or {})        # v65: letters and numbers too
                rule = "comma_inesistente" if why else ""
            if not rule:
                why = _against_client(sentence, ctx.get("client") or "")
                rule = "contro_assistito" if why else ""
            if not rule:
                why = _repeats(sentence, earlier)
                rule = "ripetizione" if why else ""
            if not rule:
                why = _fragment(sentence)
                rule = "frammento" if why else ""
            unfounded = set(ctx.get("unfounded") or ()) & set(labels)
            if not rule and unfounded and title == ctx.get("lead"):        # v65: no fact holds it: not in §1
                rule, why = ("alternativa_senza_fatti", f"propone in §1 l'art. {sorted(unfounded)[0]}, qualificazione "
                             "alternativa che nessun fatto del caso sostiene (sezione delle qualificazioni alternative)")
            if not rule and title in set(ctx.get("asserting") or ()):     # v64: only the threat's recorded elements
                review = ctx.get("requirement_review") or {}
                if title in (review.get("ran") or ()):      # v65: the writer review answered -- its verdict decides
                    why = (REQUIREMENT_FLAGGED if _flagged(sentence, review.get("flagged") or ()) else "")
                else:                                       # its call failed or did not run: the word test
                    why = _unrecorded_requirement(sentence, ctx.get("threats") or {}, documents,
                                                  [ctx.get("client") or ""] + list(ctx.get("parties") or ()))
                rule = "requisito_non_registrato" if why else ""
            cut = _state_of_mind_cut(sentence, ctx.get("client") or "", ctx.get("parties") or ()) \
                if single and not rule else None
            if cut == "":
                rule, why = ("stato_soggettivo", "dà lo stato soggettivo della controparte per un fatto che il caso non "
                             "dice: il parere lo argomenta dai fatti")
            if not rule and ctx.get("single_shot") and _QUESTION_END_RE.search(sentence):
                rule, why = ("domanda_nel_parere", "una domanda nel corpo del parere: il fatto mancante è «da verificare: "
                             "il caso non dice …» dove conta")
            proposers = ledger.proposed_by(u["norm"]) if ledger is not None else []
            if rule:
                cuts.append((s, e))
                removed_norms.add(u["norm"])
                log["rules"][rule] = log["rules"].get(rule, 0) + 1
                log["removed"].append({"section": title, "rule": rule, "reason": why, "sentence": u["text"][:200],
                                       "proposed_by": proposers})
                if ledger is not None:
                    ledger.remove(why, title, sentence, u["articles"], rule, proposers)
                continue
            if cut:                     # v60q: the «il caso non dice …» clause on a state of mind cut, the rest kept
                marks[(s, e)] = cut
                log["rewritten"].append({"section": title, "rule": "stato_soggettivo", "sentence": u["text"][:200],
                                         "now": " ".join(cut.split())[:200]})
                log["rules"]["stato_soggettivo"] = log["rules"].get("stato_soggettivo", 0) + 1
                previous = cut if "\n" not in text[e:e + 2] else ""
                continue
            mark = ""
            hit = _institution(sentence, documents, rubriche, texts, ctx.get("names"), log["cleared"],
                               ctx.get("index"), ctx.get("records"))
            if hit:                     # v68: the citation replaced by the institution's article, else dropped
                now = _cited_instead(sentence, hit, documents)
                marks[(s, e)] = now
                log["rewritten"].append({"section": title, "rule": "istituto_rubrica", "article": hit[0],
                                         "institution": hit[1], "now_article": hit[4] or "",
                                         "missed": [x for x in CLEAR_SOURCES if hit[2] or x != "indice"],
                                         "index": hit[2], "sentence": u["text"][:200], "now": " ".join(now.split())[:200]})
                log["rules"]["istituto_rubrica"] = log["rules"].get("istituto_rubrica", 0) + 1
                previous = now if "\n" not in text[e:e + 2] else ""
                continue
            elif _denies_previous(sentence, previous):
                mark = " (da verificare: contraddice la frase precedente)"
                log["marked"].append({"section": title, "rule": "contraddizione_interna", "sentence": u["text"][:200]})
            if mark and "(da verificare" not in sentence:
                marks[(s, e)] = _mark(sentence, mark)
                log["rules"][log["marked"][-1]["rule"]] = log["rules"].get(log["marked"][-1]["rule"], 0) + 1
            if proposers:
                log["rejected"].append({"section": title, "proposed_by": proposers, "sentence": u["text"][:200]})
            previous = sentence if "\n" not in text[e:e + 2] else ""
        lead, trail = text[:len(text) - len(text.lstrip("\n"))], text[len(text.rstrip("\n")):]
        for s, e in sorted(set(cuts) | set(marks), reverse=True):
            text = text[:s] + marks[(s, e)] + text[e:] if (s, e) in marks else text[:s] + text[e:]
        if cuts:
            text = lead + _tidy(text) + trail
        out.append((title, text))
    return out, log


# v64: a sentence stating what a norm requires, and the words of a requirement clause that name nothing of it
_REQUIRES_RE = re.compile(r"\b(?:richied\w*|presuppon\w*|(?:è|sono|risulta|risultano)\s+necessari\w*|occorr\w*\s+che|"
                          r"esig\w*|postul\w*|(?:deve|devono)\s+(?:sussistere|ricorrere|esserci))\b", re.IGNORECASE)
_NEUTRAL_WORDS = ("condotta", "agente", "soggetto", "autore", "fatto", "fatti", "caso", "norma", "articolo", "codice",
                  "comma", "fattispecie", "configurabilità", "configurazione", "sussistenza", "presenza", "prova",
                  "necessario", "necessaria", "imputato", "reato", "delitto", "elemento", "elementi", "requisito",
                  "requisiti", "integrazione", "integrare", "richiede", "presuppone", "occorre", "affinché", "inoltre",
                  "anche", "sempre", "solo", "soltanto", "parte")
REQUIREMENT_COVERED = 0.5
_OTHER_SUBJECT_RE = re.compile(r"\b(?:difesa|accusa|pm|pubblico\s+ministero|giudice|tribunale|corte|cliente|parte\s+"
                               r"civile|controparte|imputat\w*|persona\s+offesa)\b", re.IGNORECASE)
_REQUIREMENT_STOP = {"dell", "nell", "sull", "dall", "quell", "dagl", "degl", "negl", "sugl", "dove", "come", "cioè",
                     "ossia", "ovvero", "oppure", "sia", "siano", "abbia", "abbiano", "fosse", "fossero", "essere",
                     "stato", "stata", "sono", "dalla", "tale", "tali", "quale", "quali", "loro", "suoi", "sue", "suo",
                     "propri", "propria", "proprio", "altro", "altra", "altri", "ogni", "parte", "della", "delle"}


def _unrecorded_requirement(sentence: str, threats: Dict[str, Dict[str, Any]], documents,
                            names: Sequence[str] = ()) -> str:
    """v64: the reason when the sentence states what a threat article requires and the requirement names none of its
    recorded elements -- the clause after «richiede», «presuppone», «è necessario che» …: its content stems (the
    article's rubrica, the parties' names and neutral words aside) less than REQUIREMENT_COVERED the elements'; "" else."""
    from . import citations

    def stems(text: str) -> Set[str]:       # four letters and up: «dolo», «uomo» count; function words never
        return {w[:6] for w in re.findall(r"[a-zà-ü]{4,}", str(text or "").lower())
                if w not in _FUNCTION_STOP and w not in _REQUIREMENT_STOP}
    m = _REQUIRES_RE.search(sentence or "")
    if not m or not threats:
        return ""
    before = sentence[:m.start()]           # the norm is the verb's subject: named before it, no one else between
    cited = {_label_of(r) for r in citations.refs_in(before, documents)}
    words = stems(before)
    about = [label for label, t in threats.items() if label in cited
             or (stems(t.get("rubrica") or "") and stems(t.get("rubrica") or "") <= words)]
    if not about:
        return ""
    tail = re.split(r"\bartt?\.\s*\d+[\w-]*(?:\s+c\.[a-z.]+)?", before)[-1].lower()
    if _OTHER_SUBJECT_RE.search(tail) or any(n and len(n) > 2 and n.lower() in tail for n in names or ()):
        return ""
    clause = re.split(r"[;]", sentence[m.end():], maxsplit=1)[0]
    clause = re.sub(r"\bartt?\.\s*\d+[\w-]*(?:\s+(?:c\.p\.p\.|c\.p\.c\.|c\.p\.|c\.c\.))?", " ", clause)
    asked = stems(clause) - stems(" ".join(_NEUTRAL_WORDS)) - stems(" ".join(names or ()))
    for label in about:
        asked_here = asked - stems(threats[label].get("rubrica") or "")
        own = stems(" ".join(threats[label].get("elements") or []))
        if not asked_here or len(asked_here & own) >= REQUIREMENT_COVERED * len(asked_here):
            return ""
    label = about[0]
    elements = "; ".join(f"«{e[:60]}»" for e in threats[label].get("elements") or []) or "nessuno"
    return f"enuncia un requisito dell'art. {label} che il registro non elenca (elementi registrati: {elements})"


# v74: the negation check's word test (the review call failed) -- a negation, and most of an element's content stems or its
# id, in the sentence; an element the sentence's thesis contests is never one
_NEGATION_RE = re.compile(r"\b(?:non|né|manc\w*|assen\w*|esclu\w*|insussist\w*|difett\w*|carenz\w*|priv[oaie]|"
                          r"senza|nega\w*|neg[ao])\b", re.IGNORECASE)
NEGATION_COVERED = 0.6


def negated_elements(sentence: str, elements: Dict[str, Dict[str, Any]], allowed: Sequence[str] = ()) -> List[str]:
    """v74: the ids of the record's elements the sentence negates and may not -- a negation word in it, and the
    element's id or at least NEGATION_COVERED of its content stems (two at least); [] else."""
    if not _NEGATION_RE.search(sentence or ""):
        return []

    def stems(text: str) -> Set[str]:
        return {w[:6] for w in re.findall(r"[a-zà-ü]{4,}", str(text or "").lower()) if w not in _FUNCTION_STOP}
    words = stems(sentence)
    out = []
    for eid, e in (elements or {}).items():
        if eid in allowed:
            continue
        own = stems(e.get("text") or "")
        if re.search(rf"\b{re.escape(eid)}\b", sentence) or \
                (len(own) >= 2 and len(own & words) >= NEGATION_COVERED * len(own)):
            out.append(eid)
    return out


REQUIREMENT_FLAGGED = ("il riesame segnala un requisito di una norma dell'addebito o della domanda che non è tra i suoi "
                       "elementi registrati")


def _words_of(text: str) -> str:
    return " ".join(re.findall(r"[a-zà-ü0-9]+", str(text or "").lower()))


def _flagged(sentence: str, flagged: Sequence[str]) -> bool:
    """v65: the sentence is one the writer review flagged (word for word, one holding the other: later passes may have
    added a marker or cut a clause)."""
    mine = _words_of(sentence)
    return bool(mine) and any(x and (x in mine or mine in x) for x in (_words_of(f) for f in flagged or ()))


def _code_line(text: str, at: int) -> bool:
    """v60q: the sentence at `at` is on one of §2's closing lines the code writes («Altre norme del tema non trattate: …»,
    «Norme considerate e non applicabili: …») -- their «tema «…»» reasons are no quotation, their items no repetition."""
    from . import interaction_copy as C
    line = text[text.rfind("\n", 0, at) + 1:].lstrip()
    heads = [str(t).split("{", 1)[0] for t in (C.OTHER_NORMS_LINE, C.NOT_APPLICABLE_LINE) if t]
    return any(h and line.startswith(h) for h in heads)


def _tidy(text: str) -> str:
    """After the cuts: no double spaces, no line left with a mark or punctuation alone, no triple breaks."""
    lines = [re.sub(r"^ (?=[^\s-])", "", ln.rstrip()) for ln in text.split("\n")]     # v60q: a paragraph's first cut
    lines = [ln for ln in lines if not re.fullmatch(r"\s*(?:\[\d{1,2}\])?\s*[.;:,]?\s*", ln) or not ln.strip()]
    text = "\n".join(lines)
    text = re.sub(r"(?<=\S)[ \t]{2,}", " ", text)
    text = re.sub(r"[ \t]+([.;:,])", r"\1", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip("\n")
