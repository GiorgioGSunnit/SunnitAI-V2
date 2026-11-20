"""v79.13.0: the applicability judgment -- between retrieval and decomposition, for every admitted candidate norm the
model judges whether the provision governs the case's facts; the code verifies every word of that judgment against the
article's text and the facts before it has any effect. The model decides fit; it adds nothing.

Behind APPLICABILITY_FIT (default off: nothing here runs, no output changes). The tables, the word lists and the
prompt are data/applicability.json (DATA_FILE); the prompt is hashed (prompt_hash) and the hash logged with every run.

The call (call_batch): the norms in batches of at most «batch_size» (8) per call, the calls run at once up to
«concurrency», thinking off (chat_template_kwargs enable_thinking=False, the body ai_chat sends for LLM_THINKING
false/per_call), temperature 0, a JSON reply. Each norm goes with its full text and rubrica, its commi numbered, and --
with ELEMENT_SCHEMAS on -- its schema's tipo and rapporto labelled as context, never the verdict; once per call the
facts (each with its id, labelled «certo» or «allegazione») and the client's role. The reply, per norm: fit
(si / no / dipende), perche (one sentence quoting the article's passage it rests on and naming the facts by id),
condizione (with dipende only: the question to the user), ambito (fattispecie / scriminante / circostanza / rimedio /
procedura / definizione). A failed call, a reply that is no JSON, a norm the reply leaves out or answers out of the
schema: that norm unjudged (fit_unverified, logged), never a crash. A batch not back by «deadline_s» likewise
(fit_timeouts): the step never holds the answer past its budget.

The verification (verify -- pure code, no model):
  (a) the perche's quoted passage(s) («…», “…”, "…") occur in the article's text word by word in order (_align: a
      content word by the schema checker's stem grounding -- element_schemas' lexicon: stem, verb base --; a short
      word, a number, a function word, a polarity preposition and «non» exactly; the text's articles and simple
      prepositions may be left out) and never just after a negation of the text (_negated_before) -- else discarded;
  (b) every fact id named exists among the facts given; a perche naming none is discarded;
  (c) the perche's other content words (the quotes, the fact ids and the norm's own «art. N» out; function words and
      the data's connective/judgment words aside) each in the passage, the named facts' text or the norm's reviewed
      note (the lawyer's note, a reviewed schema), by the same stem grounding; a number only as one of theirs -- else
      discarded and counted fit_added_content: the model may not add legal content; (review) a «no» or a «dipende»
      rests on its passage (_rests_on_passage: no count or restriction the passage does not set -- fit_added_content
      --, a requirement beyond the generic words, for a «no» one the facts lack, what the facts are, a negation); the
      perche never contradicts a named fact (contradicted);
  (d) a dipende's condizione is a question (ends with «?») whose content words are in the article's text (any comma),
      the facts or the data's question words, a quantity word only as the quoted passage's -- else the dipende is
      discarded, fit_added_content;
  (e) the elements (the norm's «elements», else its commi) a verified condizione touches, by shared grounding words --
      what the pipeline forces «da provare».
  A judgment that fails any check is no judgment: the norm keeps the current path (fit_unverified), the reason logged
  (log «applicability_fit_discard»).
  v79.14.2 (A): the verification by verdict type. A «si» or a «dipende» as above (its quote of the article, verified on
  the article's text). A «no» quotes the fact sentence(s) it rests on, «nei fatti» (verify_no): every quote an exact
  substring of the case text (fact_quote_at: case_frame.found_at's normalisation -- the quote's marks and edge
  punctuation stripped, then _full_tokens: NFC, lowercased, «’» as «'», accent-folded with its mark kept -- matched as
  a contiguous word sequence with no stem, no omitted word, never just after a negation of the text) or, beside one,
  a passage of the article verified as (a); no article quote required; a quote in neither, or none of the case text:
  unverified (fit_no_unverified_quote). (review) A fact quote holds at least FACT_QUOTE_MIN_WORDS content words and
  never crosses a sentence break of the case text. Its other words as (c), grounded on its quotes, the named and
  quoted facts, the article's rubrica and the reviewed note -- never the article's unquoted text (an invented count,
  rule or claim still discarded, fit_added_content) --, a named fact id existing, one sentence, never contradicting a
  fact. The case text: judge's case_text (the flow's interactive_flow.fact_source: the user's own words, never the
  pipeline's labels), else the facts' text.
  v79.14.3 (item 2): a dedicated fact-quote field. The reply's schema carries «fatto_citato» (FACT_FIELD: the exact
  words of one fact sentence; "" only with «si»). A «no» and a «dipende» are verified on it (verify_no): non-empty and
  found by fact_quote_at (the same normaliser, an exact word run inside one sentence, at least FACT_QUOTE_MIN_WORDS
  content words) -- else unverified, the norm kept (a «no»: fit_no_unverified_quote); the perché is no longer parsed for
  the quote (a quote of the article there, found in its text, is kept as the printed passage and grounds its words;
  _rests_on_passage no longer applies); a «dipende»'s condizione as (d), (e). A «si» as (a)-(c), unchanged.
  v79.15.0 (item 3): the fit verified by ids -- the string matcher of 79.14.2/79.14.3 (fact_quote_at, verify_no,
  «fatto_citato», the article-passage quote of a «si») removed with its prompt text. The article goes to the model as
  its numbered periods (article_ids: «A<token>.<n>», the analysis step's own numbering -- one splitter); a «si» cites
  them (cita_norma: at least one id of this article), a «no» and a «dipende» the facts (cita_fatti: at least one S-id
  given; cita_norma optional). A missing or unknown id: unverified (fit_unverified_ids; a «no»: fit_no_unverified_ids),
  the norm kept. The printed passage is the cited periods' text, the fact quotes the cited facts' text. The perché's
  checks (c) stay on the cited periods, the cited and named facts, the rubrica and the note (an invented count, rule
  or claim: fit_added_content); a «dipende»'s condizione (d), (e). The flow never removes the frame's verified
  addebito/domanda article nor an article the user names (fit_protected; counter fit_protected_kept).

  judge(norms, facts, role, call=None)   the step: {"judgments", "by_article", "counters", "timing", "prompt_hash",
                                          "log"}; call(messages, kwargs) -> reply text replaces the model (tests)
  verify(item, norm, facts)              one reply item checked: the judgment record
  reviewed_note(label)                   the norm's reviewed note text (article_notes + a reviewed schema)
  request_kwargs()                       the call's bind kwargs: temperature 0, max_tokens, thinking off
  batches(items, size)                   the items in balanced batches of at most size

Counters: fit_judged (norms sent to the model) = fit_si + fit_no + fit_dipende + fit_unverified; fit_added_content
(judgments rejected for content not in the passage, the facts or the note -- none of them ever has an effect);
fit_skipped (norms with no text: not sent); fit_calls, fit_failed_calls, fit_timeouts. v79.14.2 (A): fit_no_verified
(the verified «no», = fit_no); the flow's fit_removed (the norms a verified «no» removed, each logged
«applicability_fit_discard» with the reason REMOVED_REASON). v79.15.0 (item 3): fit_unverified_ids (a judgment whose
cited ids are missing or unknown), fit_no_unverified_ids (the «no» among them -- 79.14.2's fit_no_unverified_quote
removed with the matcher); the flow's fit_protected_kept (a verified «no» on a protected article, kept).
"""

import contextvars
import hashlib
import json
import logging
import os
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, wait as _wait
from functools import lru_cache
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .feature_flags import enabled as _flag_enabled

logger = logging.getLogger(__name__)

FLAG = "APPLICABILITY_FIT"
DATA_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data",
                         "applicability.json")
COUNTERS = ("fit_judged", "fit_si", "fit_no", "fit_dipende", "fit_unverified", "fit_added_content", "fit_skipped",
            "fit_calls", "fit_failed_calls", "fit_timeouts",
            "fit_no_verified",                                     # v79.14.2 (A)
            "fit_unverified_ids", "fit_no_unverified_ids")         # v79.15.0 (item 3)
REMOVED_REASON = "removed: fit no (verified)"         # v79.14.2 (A): the removal's «applicability_fit_discard» reason
SI, NO, DIPENDE = "si", "no", "dipende"
ASSUMED = "assunto"     # v79.15.0 (item 3): issue_records.ASSUNTO -- the system's assumption, never a fact cited


def enabled() -> bool:
    return _flag_enabled(FLAG)


@lru_cache(maxsize=1)
def config() -> Dict[str, Any]:
    with open(DATA_FILE, encoding="utf-8") as f:
        return json.load(f)


def prompt_system() -> str:
    return "\n".join(config()["prompt_system"])


def prompt_hash() -> str:
    return hashlib.sha256(prompt_system().encode("utf-8")).hexdigest()[:12]


def _log(event: str, data: Dict[str, Any]) -> None:
    try:
        from .verbose_logger import vlog
        vlog(event, data)
    except Exception:           # noqa: BLE001 -- the log never fails the step
        pass


# --- the words --------------------------------------------------------------------------------------------------------

def _es():
    from . import element_schemas
    return element_schemas


@lru_cache(maxsize=1)
def _connectives():
    es = _es()
    return es.lexicon(" ".join(config().get("connective_words") or []))


@lru_cache(maxsize=1)
def _question_words():
    es = _es()
    return es.lexicon(" ".join((config().get("connective_words") or []) + (config().get("question_words") or [])))


def _numbers(text: Any) -> List[str]:
    return [n.replace(".", "").replace(",", "") for n in re.findall(r"\d+(?:[.,]\d+)*", str(text or ""))]


def quotes(text: Any) -> List[str]:
    """The quoted passages of a perche, in order («…», “…”, "…")."""
    out: List[Tuple[int, str]] = []
    for pattern in config().get("quote_patterns") or []:
        for m in re.finditer(pattern, str(text or "")):
            if m.group(1).strip():
                out.append((m.start(), m.group(1).strip()))
    return [q for _, q in sorted(out)]


def _unquoted(text: Any) -> str:
    s = str(text or "")
    for pattern in config().get("quote_patterns") or []:
        s = re.sub(pattern, " ", s)
    return s


def fact_ids(text: Any, known: Iterable[str] = ()) -> List[str]:
    """The fact ids a text names: the data's id pattern, and any known id standing as a word."""
    s = str(text or "")
    found = [(m.start(), m.group()) for m in re.finditer(config()["fact_id_pattern"], s)]
    for k in known:
        k = str(k)
        if k:
            found += [(m.start(), k) for m in re.finditer(rf"(?<![\w-]){re.escape(k)}(?![\w-])", s)]
    out: List[str] = []
    for _, x in sorted(found):
        if x not in out:
            out.append(x)
    return out


def _strip_ids(text: str, ids: Iterable[str]) -> str:
    for k in sorted(set(ids), key=len, reverse=True):
        text = re.sub(rf"(?<![\w-]){re.escape(k)}(?![\w-])", " ", text)
    return text


def _code_key(code: Any) -> str:
    """A code's spelling compacted («c.p.», «c. p.», «cp» -> «cp»; «codice civile» -> «cc»)."""
    c = re.sub(r"[\s.]", "", str(code or "")).lower()
    return {"codicecivile": "cc", "codicepenale": "cp", "codicediprocedurapenale": "cpp",
            "codicediproceduracivile": "cpc", "codpen": "cp", "codciv": "cc"}.get(c, c)


def _strip_own_reference(text: str, label: str) -> str:
    """The norm's own «art. N codice» out of a perche -- its own code only, or none (another norm's number stays, and
    fails the number check: «art. 646 c.c.» on 646 c.p. is another norm)."""
    m = re.match(r"^\s*(\S+)\s*(.*)$", str(label or ""))
    if not m:
        return text
    number, own = m.group(1).strip(), _code_key(re.sub(r"\s+co\.\s*\S+\s*$", "", m.group(2)))
    pattern = config()["own_reference_pattern"].replace("{number}", re.escape(number))

    def strip(found) -> str:
        code = re.sub(r"^.*?\b" + re.escape(number) + r"\b(?:\s*,?\s*(?:co\.|comma)\s*\d+)?", "", found.group(0),
                      flags=re.IGNORECASE)
        return " " if not code.strip() or _code_key(code) == own else found.group(0)
    return re.sub(pattern, strip, text, flags=re.IGNORECASE)


_WORD_RE = re.compile(r"[^\W_]+")


def _full_tokens(text: Any) -> List[Tuple[str, str, bool]]:
    """(the word lowercased, accent-folded, whether it ends accented) of every word of a text -- the short ones and the
    numbers too («o», «da», «sé», «2», «000»): what a quoted passage is matched on, word by word."""
    es = _es()
    s = unicodedata.normalize("NFC", str(text or "").replace("’", "'").lower())
    out: List[Tuple[str, str, bool]] = []
    for m in _WORD_RE.finditer(s):
        raw = m.group()
        after = s[m.end():m.end() + 2]
        accented = raw[-1] in es._ACCENTED or (after[:1] == "'" and not after[1:2].isalpha())
        out.append((raw, es._fold(raw), accented))
    return out


def _skey(tok: Tuple[str, str, bool]) -> str:
    """A word as matched exactly: folded, its accent kept as a mark («è» and «e'» alike, never «e»; «sé» never «se»)."""
    return tok[1] + ("'" if tok[2] else "")


@lru_cache(maxsize=1)
def _word_sets() -> Dict[str, frozenset]:
    """The data's word lists, as _skey reads them (negations, the words a negation stands behind, the omittable and
    the article words, the quantity words)."""
    cfg = config()

    def keys(name: str) -> frozenset:
        return frozenset(_skey(t) for w in cfg.get(name) or [] for t in _full_tokens(w))
    return {name: keys(name) for name in ("negation_words", "negation_aux", "omittable_words", "article_words",
                                          "quantity_words", "generic_norm_words")}


def _strict(tok: Tuple[str, str, bool]) -> bool:
    """A word matched only as itself: a short one, a number, a function word, a polarity preposition, «non»."""
    es = _es()
    w = tok[1]
    return len(w) < 3 or any(c.isdigit() for c in w) or w in es.FUNCTION_WORDS or \
        w in es.POLARITY_PREPOSITIONS or w == "non"


def _tok_match(p: Tuple[str, str, bool], t: Tuple[str, str, bool]) -> bool:
    if _strict(p) or _strict(t):
        return _skey(p) == _skey(t)
    return _es()._Lexicon([(t[1], t[2])]).has(p[1], p[2])


def _align(ptoks: Sequence[Tuple[str, str, bool]], ttoks: Sequence[Tuple[str, str, bool]], start: int
           ) -> Optional[int]:
    """The passage matched word by word from the text's word `start`: each word as the schema checker grounds it (a
    short word, a number, a function word exactly); the text's articles and simple prepositions may be left out of
    the quote, the quote's own articles may be added -- nothing else («o» for «e», «da» for «a», a number changed,
    a word skipped: no match). The index of the first text word matched, None when it does not match."""
    sets = _word_sets()
    j, k, first = 0, start, None
    while j < len(ptoks):
        if k < len(ttoks) and _tok_match(ptoks[j], ttoks[k]):
            first = k if first is None else first
            j, k = j + 1, k + 1
        elif first is not None and k < len(ttoks) and _skey(ttoks[k]) in sets["omittable_words"]:
            k += 1
        elif _skey(ptoks[j]) in sets["article_words"]:
            j += 1
        else:
            return None
    return first


def _negated_before(ttoks: Sequence[Tuple[str, str, bool]], first: int) -> bool:
    """A negation stands just before the text's word `first` (behind at most two of the data's «negation_aux»: «non è
    punibile»): a quote starting there says the opposite of the text."""
    sets = _word_sets()
    k = first - 1
    for _ in range(3):
        if k < 0:
            return False
        key = _skey(ttoks[k])
        if key in sets["negation_words"]:
            return True
        if key not in sets["negation_aux"]:
            return False
        k -= 1
    return False


def passage_found(text: str, passage: str, lex=None) -> bool:
    """(a) The passage occurs in the article's text: word by word in order (_align: each content word as the schema
    checker grounds it -- element_schemas' stem and verb base --, a short word, a number, a function word, a polarity
    preposition and «non» exactly; the text's articles and simple prepositions may be left out), and no negation of
    the text just before it (_negated_before: «è punibile» out of «non è punibile» is not the text). A passage with
    no content word is never found. lex: unused (kept for the callers)."""
    return passage_at(text, passage) is not None


def passage_at(text: str, passage: str) -> Optional[int]:
    """v79.14.0: passage_found's match, as the index of the text's first word matched (None when not found) -- the
    case frame (src/rag/case_frame.py) orders the questions it verifies by it."""
    es = _es()
    if not str(passage or "").strip() or not str(text or "").strip():
        return None
    if not [w for w, _ in es._tokens(passage) if w not in es.FUNCTION_WORDS]:
        return None
    ptoks, ttoks = _full_tokens(passage), _full_tokens(text)
    if not ptoks:
        return None
    starts_negated = _skey(ptoks[0]) in _word_sets()["negation_words"]
    for i in range(len(ttoks)):
        first = _align(ptoks, ttoks, i)
        if first is not None and (starts_negated or not _negated_before(ttoks, first)):
            return first
    return None


# v79.15.0 (item 3): the article's numbered periods -- what a «si» cites (cita_norma) and what the analysis step
# (src/rag/analysis_step.py) cites, numbered the same way by both (this one splitter). The split: the article's commi
# (element_schemas.commi, in their order; a comma with no text skipped), each comma's text (its spaces collapsed) cut
# into its periods at a full stop followed by a space and a capital letter or an opening mark (« " “ ( ) -- an
# abbreviation never ends one (sentence_split.split: «art.», «Sig.», «c.p.»); every period numbered 1..n across the
# whole article. Its id: «A» + the article's token (its label lowercased, accents folded, every character but a-z and
# 0-9 out: «640 c.p.» -> «640cp», «62-bis c.p.» -> «62biscp», «2043 c.c.» -> «2043cc») + «.» + n: «A640cp.1». The
# code is in the token, so two articles with the same number in two codes never share an id.
_PERIOD_RE = re.compile(r"(?<=\.)\s+(?=[A-ZÀ-Ü«\"“(])")
ARTICLE_ID_RE = re.compile(r"\bA[0-9a-z]+\.\d+\b")


def article_token(label: Any) -> str:
    """v79.15.0: an article label as its ids spell it («640 c.p.» -> «640cp»)."""
    folded = unicodedata.normalize("NFKD", str(label or "").lower())
    return re.sub(r"[^0-9a-z]", "", "".join(c for c in folded if not unicodedata.combining(c)))


def article_ids(label: Any, text: Any) -> List[Dict[str, str]]:
    """v79.15.0 (item 3): the article's text as its numbered periods -- [{"id": «A<token>.<n>», "comma", "text"}] (the
    split above); [] for an article with no text."""
    from .sentence_split import split
    token = article_token(label)
    out: List[Dict[str, str]] = []
    if not token or not str(text or "").strip():
        return out
    for comma, body in _es().commi(text).items():
        body = " ".join(str(body or "").split())
        if not body:
            continue
        for piece in split(body, _PERIOD_RE):
            piece = piece.strip()
            if re.search(r"\w", piece):
                out.append({"id": f"A{token}.{len(out) + 1}", "comma": str(comma), "text": piece})
    return out


def id_list(value: Any) -> List[str]:
    """v79.15.0: a reply's id list (cita_fatti, cita_norma) as strings, each once, in order -- a list of ids, or one
    string of them (commas, semicolons or spaces between); anything else, none."""
    if isinstance(value, str):
        value = re.split(r"[\s,;]+", value)
    if not isinstance(value, (list, tuple)):
        return []
    out: List[str] = []
    for x in value:
        x = str(x or "").strip().strip("[]()«»\"'.")
        if x and x not in out:
            out.append(x)
    return out



def _clauses(text: str) -> List[List[Tuple[str, str, bool]]]:
    """A text's clauses (split at a quote, a punctuation mark or a parenthesis), each as its words."""
    s = str(text or "")
    for pattern in config().get("quote_patterns") or []:
        s = re.sub(pattern, " | ", s)
    return [_full_tokens(part) for part in re.split(r"[|,;:.!?()\[\]\n]", s) if part.strip()]


def _negated_words(text: str) -> List[Tuple[Tuple[str, str, bool], bool]]:
    """Each word of a text with whether a negation of the data («non», «né», «senza», …) stands before it within three
    words of its clause («non ha versato»: versato negated; «non si applica: il prestito»: prestito not)."""
    neg = _word_sets()["negation_words"]
    out = []
    for words in _clauses(text):
        for i, tok in enumerate(words):
            out.append((tok, any(_skey(w) in neg for w in words[max(0, i - 3):i])))
    return out


def contradicted(text: str, facts: Sequence[str]) -> List[str]:
    """The content words of a perché (outside its quotes) that the named facts hold only with the other polarity: the
    perché negates what the fact states («non ha versato» against «ha versato»), or states what the fact negates. A word
    the facts lack, a connective, a function word: not read here."""
    es = _es()
    fact_words = [w for f in facts for w in _negated_words(f)]
    out: List[str] = []
    for tok, negated in _negated_words(text):
        if _strict(tok) or _connectives().has(tok[1], tok[2]) or _skey(tok) in _word_sets()["negation_words"]:
            continue
        seen = [n for (t, n) in fact_words if not _strict(t) and es._Lexicon([(t[1], t[2])]).has(tok[1], tok[2])]
        if seen and negated not in seen and tok[0] not in out:
            out.append(tok[0])
    return out


def _ungrounded(text: str, ground: str, allowed) -> List[str]:
    """The content words of text found neither in ground (missing_words: stem, verb base, polarity) nor among the
    allowed words; and every number not among ground's."""
    es = _es()
    out = []
    for w in es.missing_words(text, ground):
        if not allowed.has(w.split(" ")[-1]):          # a polarity pair «senza X»: X allowed, the pair is too
            out.append(w)
    numbers = set(_numbers(ground))
    out += [n for n in _numbers(text) if n not in numbers]
    return out


def _sentences(text: str) -> int:
    parts = [p for p in re.split(r"(?<=[.!?;])\s+(?=[A-ZÀ-Ü])", " ".join(text.split())) if re.search(r"\w", p)]
    return len(parts)


def touches(condizione: str, norm: Dict[str, Any]) -> Dict[str, List[str]]:
    """(e) The norm's elements (its «elements»: [{id, testo}]) and commi the condizione touches: those sharing the most
    grounding words with it (connective and question words aside); none when it shares none."""
    es = _es()
    words = [w for w in es.grounding_words(condizione) if not _question_words().has(w)]

    def best(items: Sequence[Tuple[str, str]]) -> List[str]:
        scored = []
        for key, body in items:
            lex = es.lexicon(body)
            scored.append((sum(1 for w in set(words) if lex.has(w)), key))
        top = max((s for s, _ in scored), default=0)
        return [k for s, k in scored if top and s == top]
    elements = [(str(e.get("id") or ""), str(e.get("testo") or e.get("text") or ""))
                for e in norm.get("elements") or [] if isinstance(e, dict)]
    return {"elements": best([e for e in elements if e[0]]),
            "commi": best(list(es.commi(norm.get("text") or "").items()))}


# --- the reviewed note ------------------------------------------------------------------------------------------------

def reviewed_note(label: str, rows: Iterable[Dict[str, Any]] = ()) -> str:
    """The norm's reviewed note: the lawyer's note (article_notes.for_label) and, from the lawyer's reviewed schema
    file, its notes and -- when reviewed -- its elements' text. "" when none; never fails."""
    parts: List[str] = []
    try:
        from . import article_notes
        parts.append(article_notes.for_label(label, rows)[0])
    except Exception as exc:    # noqa: BLE001
        logger.info("applicability: note for %s not read: %s", label, exc)
    try:
        es = _es()
        line = es.reviewed().get(es.norm_label(label)) or {}
        for n in line.get("note_di_inquadramento") or []:
            if isinstance(n, dict):
                parts.append(str(n.get("testo") or ""))
        if es._is_reviewed(line):
            for e in (line.get("schema") or {}).get("elementi") or []:
                if isinstance(e, dict):
                    parts.append(str(e.get("testo") or ""))
    except Exception as exc:    # noqa: BLE001
        logger.info("applicability: reviewed schema for %s not read: %s", label, exc)
    return " ".join(p for p in parts if p).strip()


# --- verification -----------------------------------------------------------------------------------------------------

def _record(norm: Dict[str, Any], item: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    item = item or {}
    return {"article": norm.get("article"), "fit": None, "perche": str(item.get("perche") or item.get("perché") or ""),
            "passage": "", "comma": "", "facts": [], "condizione": "", "ambito": None, "verified": False,
            "reason": "", "added_content": False, "touches": {"elements": [], "commi": []},
            "fact_quotes": [], "cita_fatti": [], "cita_norma": [], "ids_failed": False}


def _canon_fit(value: Any) -> Optional[str]:
    v = str(value or "").strip().lower()
    v = (config().get("fit_synonyms") or {}).get(v, v)
    return v if v in config()["fit_values"] else None


def _quantities(text: Any) -> List[str]:
    """The data's quantity words (a count, a threshold, a restriction: «due», «almeno», «solo») a text holds."""
    q = _word_sets()["quantity_words"]
    return list(dict.fromkeys(t[1] for t in _full_tokens(text) if _skey(t) in q))


def verify(item: Any, norm: Dict[str, Any], facts: Sequence[Dict[str, Any]], case_text: str = "") -> Dict[str, Any]:
    """One reply item checked against the norm's numbered periods and the facts: the judgment record, verified or with
    the reason it was discarded. Never raises on a malformed item. v79.15.0 (item 3): verified by ids, no string
    matcher -- a «si» cites the article's periods it rests on (cita_norma: ids of article_ids, of this article only, at
    least one), a «no» and a «dipende» the facts (cita_fatti: S-ids of the facts given, at least one; cita_norma
    optional, verified the same way when given; a system's assumption -- certainty «assunto» -- is no fact to cite);
    a missing or unknown id: unverified (rec["ids_failed"]: fit_unverified_ids;
    a «no»: fit_no_unverified_ids), the norm kept on its current path. Then, as 79.13 (c)-(e): the perché one sentence,
    every fact id it names existing, its content words grounded on the cited periods, the cited and named facts, the
    rubrica and the reviewed note (an invented count, rule or claim: fit_added_content), for a «no» or a «dipende» a
    count or a restriction only as the cited periods' own, never contradicting a cited fact; a «dipende»'s condizione
    a grounded question and the elements it touches. rec["passage"]: the cited periods' text (the article's own words,
    what the contest prints); rec["fact_quotes"]: the cited facts' text; rec["facts"]: the cited, then the named ids.
    case_text: unused (kept for the callers)."""
    rec = _record(norm, item if isinstance(item, dict) else None)
    if not isinstance(item, dict):
        rec["reason"] = "giudizio non è un oggetto"
        return rec
    if item.get(CONFLICT):
        rec["reason"] = f"giudizi contrastanti per la stessa norma: {', '.join(str(f) for f in item.get('fits') or [])}"
        return rec
    fit = _canon_fit(item.get("fit"))
    ambito = str(item.get("ambito") or "").strip().lower()
    rec["fit"], rec["ambito"] = fit or (str(item.get("fit")) if item.get("fit") is not None else None), ambito or None
    if fit is None:
        rec["reason"] = f"fit fuori schema: {item.get('fit')!r}"
        return rec
    if ambito not in config()["ambiti"]:
        rec["reason"] = f"ambito fuori schema: {item.get('ambito')!r}"
        return rec
    perche = rec["perche"].strip()
    if not perche:
        rec["reason"] = "perché mancante"
        return rec
    label = str(norm.get("article") or "")
    text = str(norm.get("text") or "")
    by_id = {str(f.get("id")): f for f in facts if isinstance(f, dict) and f.get("id")}
    periods = {p["id"]: p for p in article_ids(label, text)}
    cited_f, cited_n = id_list(item.get("cita_fatti")), id_list(item.get("cita_norma"))
    rec["cita_fatti"], rec["cita_norma"] = cited_f, cited_n
    bad_f = [i for i in cited_f if i not in by_id]
    bad_n = [i for i in cited_n if i not in periods]
    if bad_f:
        rec["reason"], rec["ids_failed"] = f"cita_fatti: fatto inesistente: {', '.join(bad_f[:6])}", True
        return rec
    assumed = [i for i in cited_f if by_id[i].get("certainty") == ASSUMED]     # the system's assumption: no fact
    if assumed:
        rec["reason"], rec["ids_failed"] = f"cita_fatti: assunzione del sistema, non un fatto: {', '.join(assumed)}", True
        return rec
    if bad_n:
        rec["reason"], rec["ids_failed"] = f"cita_norma: periodo non dell'articolo: {', '.join(bad_n[:6])}", True
        return rec
    if fit == SI and not cited_n:
        rec["reason"], rec["ids_failed"] = "cita_norma mancante", True
        return rec
    if fit in (NO, DIPENDE) and not cited_f:
        rec["reason"], rec["ids_failed"] = "cita_fatti mancante", True
        return rec
    passage = [periods[i]["text"] for i in cited_n]
    rec["passage"] = " … ".join(passage)
    rec["comma"] = periods[cited_n[0]]["comma"] if cited_n else ""
    rec["fact_quotes"] = [str(by_id[i].get("text") or "") for i in cited_f]
    if _sentences(_strip_own_reference(_unquoted(perche), label)) > 1:     # «art. 1819 c.c. Il …»
        rec["reason"] = "perché di più frasi"
        return rec
    named = fact_ids(perche, by_id)
    unknown = [i for i in named if i not in by_id]
    if unknown:
        rec["reason"] = f"fatto inesistente: {', '.join(unknown)}"
        return rec
    rec["facts"] = list(dict.fromkeys(cited_f + named))
    fact_texts = [str(by_id[i].get("text") or "") for i in rec["facts"]]
    ground = " ".join(passage + fact_texts + [str(norm.get("rubrica") or ""), str(norm.get("note") or "")])
    ids = by_id.keys() | set(named) | set(periods)
    plain = _strip_ids(_strip_own_reference(perche, label), ids)
    for pattern in config().get("quote_patterns") or []:            # a quote's words are words like any other
        plain = re.sub(pattern, lambda m: f" {m.group(1)} ", plain)
    extra = _ungrounded(plain, ground, _connectives())                          # (c)
    if fit in (NO, DIPENDE):        # a count or a restriction only as the cited periods' own
        have = set(_quantities(" ".join(passage)))
        extra += [w for w in _quantities(plain) if w not in have and w not in extra]
    if extra:
        rec["added_content"] = True
        rec["reason"] = f"contenuto non nei periodi citati, nei fatti o nella nota: {', '.join(extra[:6])}"
        return rec
    against = contradicted(plain, fact_texts)                                   # (c) the facts' polarity
    if against:
        rec["reason"] = f"contraddice il fatto nominato: {', '.join(against[:6])}"
        return rec
    if fit == DIPENDE:                                                          # (d), as 79.13
        cond = " ".join(str(item.get("condizione") or "").split())
        rec["condizione"] = cond
        if not cond:
            rec["reason"] = "dipende senza condizione"
            return rec
        if not cond.endswith("?"):
            rec["reason"] = "condizione non è una domanda"
            return rec
        cond_ids = fact_ids(cond, by_id)
        if [i for i in cond_ids if i not in by_id]:
            rec["reason"] = f"condizione con fatto inesistente: {', '.join(i for i in cond_ids if i not in by_id)}"
            return rec
        cond_ground = " ".join([text] + [str(f.get("text") or "") for f in by_id.values()])
        cond_words = _strip_ids(_strip_own_reference(cond, label), by_id.keys() | set(periods))
        extra = _ungrounded(cond_words, cond_ground, _question_words())
        extra += [w for w in _quantities(cond_words) if w not in _quantities(" ".join(passage)) and w not in extra]
        if extra:
            rec["added_content"] = True
            rec["reason"] = f"condizione con contenuto non nel testo o nei fatti: {', '.join(extra[:6])}"
            return rec
        rec["touches"] = touches(cond, norm)                                    # (e)
    rec["verified"] = True
    return rec



# --- the call ---------------------------------------------------------------------------------------------------------

def request_kwargs() -> Dict[str, Any]:
    """The call's bind kwargs: temperature 0, the reply's budget, thinking off (Qwen3's chat template:
    enable_thinking=False -- ai_chat's no-reasoning body)."""
    return {"temperature": 0, "max_tokens": int(config().get("max_tokens") or 1800),
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}


def _default_call(messages: List[Any], kwargs: Dict[str, Any]) -> str:
    """The repo's chat model (ai_chat.chat_model, the env's LLM_* config) bound to the kwargs; the reply without its
    reasoning (ai_chat._reply_text), the call on the request's stage meter."""
    from . import ai_chat
    ai_chat._stage_count()
    started = time.time()
    response = ai_chat.chat_model.bind(**kwargs).invoke(messages)
    ai_chat._stage_call("applicability_fit", started, response)
    return ai_chat._reply_text(response, kwargs.get("max_tokens"))


def batches(items: Sequence[Any], size: int) -> List[List[Any]]:
    """The items in ceil(n / size) batches of near-equal length (30 by 8: 8, 8, 7, 7), order kept."""
    items = list(items)
    if not items:
        return []
    size = max(1, int(size))
    k = -(-len(items) // size)
    base, extra = divmod(len(items), k)
    out, i = [], 0
    for j in range(k):
        n = base + (1 if j < extra else 0)
        out.append(items[i:i + n])
        i += n
    return out


def fact_label(fact: Dict[str, Any]) -> str:
    labels = config().get("fact_labels") or {}
    return labels.get(str(fact.get("certainty") or ""), labels.get("_default", "allegazione"))


def _norm_block(norm: Dict[str, Any], with_schema: bool) -> str:
    text = str(norm.get("text") or "")
    cap = int(config().get("max_article_chars") or 6000)
    lines = [f"ARTICOLO: {norm.get('article')}", f"Rubrica: {norm.get('rubrica') or '(nessuna)'}",
             "Testo (periodi numerati):"]
    for period in article_ids(norm.get("article"), text):          # v79.15.0 (item 3): what cita_norma cites
        lines.append(f"[{period['id']}] {period['text']}")
    block = "\n".join(lines)
    if len(block) > cap:
        logger.info("applicability: %s text cut at %d characters in the prompt", norm.get("article"), cap)
        block = block[:cap] + " […]"
    schema = norm.get("schema") if with_schema else None
    if isinstance(schema, dict) and (schema.get("tipo") or schema.get("rapporto")):
        tipo = schema.get("tipo")
        tipo = ", ".join(str(t) for t in tipo) if isinstance(tipo, (list, tuple)) else str(tipo or "")
        block += (f"\nContesto dello schema (solo contesto, non il giudizio): tipo: {tipo or '-'}; "
                  f"rapporto: {schema.get('rapporto') or '-'}")
    return block


def build_messages(norms: Sequence[Dict[str, Any]], facts: Sequence[Dict[str, Any]], role: str,
                   frame_context: str = "") -> List[Any]:
    """[system, human] for one batch: the norms, the facts (id, label, text), the role. v79.14.0: frame_context -- the
    verified case frame as context (case_frame.fit_context, CASE_FRAME on), first in the human message; "" adds
    nothing (the message is 79.13.0's)."""
    from langchain_core.messages import HumanMessage, SystemMessage
    with_schema = _es().enabled()
    facts_txt = "\n".join(f"{f.get('id')} [{fact_label(f)}] {' '.join(str(f.get('text') or '').split())}"
                          for f in facts)
    human = (f"RUOLO DEL CLIENTE: {role or 'non indicato'}\n\nFATTI:\n{facts_txt}\n\nNORME CANDIDATE:\n\n"
             + "\n\n".join(_norm_block(n, with_schema) for n in norms))
    if frame_context:
        human = f"{frame_context}\n\n{human}"
    return [SystemMessage(content=prompt_system()), HumanMessage(content=human)]


def parse_reply(raw: Any) -> List[Dict[str, Any]]:
    """The reply's judgment items; ValueError when it holds no JSON or no list of them."""
    from .llm_reply import parse_json_reply
    data = parse_json_reply(raw if isinstance(raw, str) else str(raw or ""))
    if isinstance(data, dict):
        data = data.get("giudizi", data.get("judgments"))
    if not isinstance(data, list):
        raise ValueError("no list of judgments in the reply")
    return data


def article_key(raw: Any) -> Tuple[str, str]:
    """(number, compact code) of an «articolo» as a model may echo it: «Art. 1819 c.c.», «articolo 1819 c.c.»,
    «1819 cc», «1819 c.c. (Restituzione rateale)» -> ("1819", "cc"); a bare «1819» -> ("1819", "")."""
    s = " ".join(str(raw or "").split())
    s = re.sub(r"\([^)]*\)", " ", s)
    s = re.sub(r"^\s*(?:art(?:icolo|t)?\.?)\s*", "", s, flags=re.IGNORECASE).strip(" ,;:-–—")
    m = re.match(r"^(\S+?)[.,]?(?:\s+(.*))?$", s)
    if not m:
        return "", ""
    return m.group(1).lower(), _code_key(re.sub(r"\s+co\.\s*\S+\s*$", "", m.group(2) or ""))


CONFLICT = "_conflict"


def _match(items: List[Any], norms: Sequence[Dict[str, Any]]) -> Dict[int, Any]:
    """{norm index: its item} by the item's «articolo»: the norm's label (as element_schemas.norm_label reads both), or
    its number and code however spelt (article_key: «Art. 1819 c.c.», «1819 cc», a rubrica in brackets), or its number
    alone when one norm of the batch has it. A second item for the same norm with the same fit ignored; with another
    fit, the norm's judgments contradict each other: {CONFLICT: True} (the norm unjudged, logged)."""
    es = _es()
    keys: Dict[Any, int] = {}
    numbers: Dict[str, List[int]] = {}
    for i, n in enumerate(norms):
        keys.setdefault(es.norm_label(n.get("article")), i)
        keys.setdefault(" ".join(str(n.get("article") or "").lower().split()), i)
        num, code = article_key(n.get("article"))
        keys.setdefault((num, code), i)
        numbers.setdefault(num, []).append(i)
    out: Dict[int, Any] = {}
    for it in items:
        if not isinstance(it, dict):
            continue
        raw = it.get("articolo", it.get("article"))
        i = keys.get(es.norm_label(raw), keys.get(" ".join(str(raw or "").lower().split())))
        if i is None:
            num, code = article_key(raw)
            i = keys.get((num, code))
            if i is None and len(numbers.get(num) or []) == 1:
                only = numbers[num][0]
                if not code or code == article_key(norms[only].get("article"))[1] or \
                        not article_key(norms[only].get("article"))[1]:
                    i = only
                elif not re.fullmatch(r"c[cp]{1,2}", code):     # a code the batch's norm spells otherwise
                    i = only
        if i is None:
            continue
        if i not in out:
            out[i] = it
        elif isinstance(out[i], dict) and not out[i].get(CONFLICT) and \
                _canon_fit(out[i].get("fit")) != _canon_fit(it.get("fit")):
            out[i] = {CONFLICT: True, "articolo": raw, "fits": [out[i].get("fit"), it.get("fit")]}
    return out


def call_batch(norms: Sequence[Dict[str, Any]], facts: Sequence[Dict[str, Any]], role: str,
               call: Optional[Callable[[List[Any], Dict[str, Any]], str]] = None,
               frame_context: str = "") -> Dict[str, Any]:
    """One call: {"status": ok | failed | malformed, "seconds", "items": {norm index: item}, "error"}."""
    started = time.time()
    try:
        reply = (call or _default_call)(build_messages(norms, facts, role, frame_context), request_kwargs())
    except Exception as exc:    # noqa: BLE001 -- the batch's norms keep the current path
        return {"status": "failed", "seconds": time.time() - started, "items": {},
                "error": f"{type(exc).__name__}: {exc}"[:300]}
    try:
        items = parse_reply(reply)
    except Exception as exc:    # noqa: BLE001
        return {"status": "malformed", "seconds": time.time() - started, "items": {}, "error": str(exc)[:300]}
    return {"status": "ok", "seconds": time.time() - started, "items": _match(items, norms), "error": ""}


# --- the step ---------------------------------------------------------------------------------------------------------

def empty_counters() -> Dict[str, int]:
    return {k: 0 for k in COUNTERS}


def judge(norms: Sequence[Dict[str, Any]], facts: Sequence[Dict[str, Any]], role: str = "",
          call: Optional[Callable[[List[Any], Dict[str, Any]], str]] = None,
          started: Optional[float] = None, frame_context: str = "", case_text: str = "") -> Dict[str, Any]:
    """The step over the admitted candidates. norms: [{article, text, rubrica, note?, elements?, schema?}]; facts:
    [{id, text, certainty}]; role: the client's. Every norm gets a record (judgments, input order; by_article); only a
    verified one may have an effect -- the caller's. No facts: nothing is sent (every record «nessun fatto»).
    started: when the caller's step began (build B: the flow's, its inputs read) -- the deadline counts from then.
    frame_context: v79.14.0 -- the verified case frame as the calls' context (case_frame.fit_context); "" none.
    case_text: unused since v79.15.0 (item 3: the fit verified by ids; kept for the callers)."""
    t0 = started if started is not None else time.time()
    cfg = config()
    counters = empty_counters()
    log: List[Dict[str, Any]] = []
    records: List[Optional[Dict[str, Any]]] = [None] * len(norms)
    facts = [f for f in facts or [] if isinstance(f, dict) and f.get("id") and str(f.get("text") or "").strip()]
    sendable = []
    for i, n in enumerate(norms):
        if not str(n.get("text") or "").strip() or not facts:
            rec = _record(n)
            rec["reason"] = "testo dell'articolo mancante" if facts else "nessun fatto"
            records[i] = rec
            counters["fit_skipped"] += 1
        else:
            sendable.append(i)
    calls: List[Dict[str, Any]] = []
    groups = batches(sendable, cfg.get("batch_size") or 8)
    counters["fit_judged"] = len(sendable)
    if groups:
        deadline = float(cfg.get("deadline_s") or 20)
        pool = ThreadPoolExecutor(max_workers=max(1, min(int(cfg.get("concurrency") or 4), len(groups))),
                                  thread_name_prefix="applicability")
        try:
            futures = {pool.submit(contextvars.copy_context().run, call_batch, [norms[i] for i in g], facts, role, call,
                                   frame_context): j
                       for j, g in enumerate(groups)}
            done, _ = _wait(list(futures), timeout=max(0.0, deadline - (time.time() - t0)))
            for fut, j in sorted(futures.items(), key=lambda kv: kv[1]):
                g = groups[j]
                counters["fit_calls"] += 1
                if fut not in done:
                    res = {"status": "timeout", "seconds": time.time() - t0, "items": {}, "error": f"oltre {deadline} s"}
                    counters["fit_timeouts"] += 1
                else:
                    res = fut.result()
                if res["status"] in ("failed", "timeout"):
                    counters["fit_failed_calls"] += 1
                if res["status"] != "ok":
                    logger.warning("applicability: batch %d (%d norms) %s: %s", j, len(g), res["status"], res["error"])
                calls.append({"batch": j, "norms": len(g), "seconds": round(res["seconds"], 3),
                              "status": res["status"], "error": res["error"]})
                for k, i in enumerate(g):
                    item = res["items"].get(k)
                    if item is None:
                        rec = _record(norms[i])
                        rec["reason"] = {"ok": "giudizio mancante nella risposta", "malformed": "risposta non JSON",
                                         "failed": "chiamata fallita", "timeout": "chiamata oltre il tempo"}[res["status"]]
                    else:
                        rec = verify(item, norms[i], facts, case_text)
                    records[i] = rec
        finally:
            pool.shutdown(wait=False, cancel_futures=True)     # a batch still queued never calls the model
    sent = set(sendable)
    for i, rec in enumerate(records):
        if i not in sent:
            log.append({"article": rec["article"], "reason": rec["reason"]})
            continue
        if rec["verified"]:
            counters[f"fit_{rec['fit']}"] += 1
            counters["fit_no_verified"] += int(rec["fit"] == NO)           # v79.14.2 (A)
        else:
            counters["fit_unverified"] += 1
            if rec["added_content"]:
                counters["fit_added_content"] += 1
            if rec.get("ids_failed"):                                         # v79.15.0 (item 3)
                counters["fit_unverified_ids"] += 1
                counters["fit_no_unverified_ids"] += int(_canon_fit(rec.get("fit")) == NO)
            log.append({"article": rec["article"], "reason": rec["reason"]})
            logger.info("applicability: %s unverified: %s", rec["article"], rec["reason"])
            _log("applicability_fit_discard", {"article": rec["article"], "reason": rec["reason"],
                                               "fit": rec["fit"], "perche": rec["perche"][:300]})
    total = time.time() - t0
    result = {"prompt_hash": prompt_hash(), "judgments": records,
              "by_article": {str(r["article"]): r for r in records if r is not None},
              "counters": counters, "timing": {"total_s": round(total, 3), "calls": calls}, "log": log}
    _log("applicability_fit", {"prompt_hash": result["prompt_hash"], "counters": counters,
                               "total_s": result["timing"]["total_s"], "calls": calls})
    return result


# --- build B: the step's inputs and effects in the flow ---------------------------------------------------------------
# What a verified judgment does (interactive_flow.applicability_fit runs the step, once per answer, on the admitted
# candidates, after the family gate and the cap and before decomposition):
#   no       the norm out of the records, the topics (hence the candidates, §4, the writer's rows, the anchors), the
#            decomposition's routes (never back: state["fit_removed"], read as dead by every later step and answer) --
#            unless the user names it or the text charges it: then kept, the judgment the defence's contest;
#   si       nothing;
#   dipende  the norm kept, the elements the condizione touches marked «fit_open» (issue_records.settle: «da provare»
#            whatever any later grading says), the condizione printed in §10 as the user's question;
#   an unverified judgment: nothing.

# my drafts (v79.13.0): the reader's wording of the judgment's effects
FIT_CONTEST_TESI = "L'art. {art} non governa i fatti narrati: {perche}"
FIT_CONTEST_ARGUMENT = {
    "pro": "Si fonda sul testo della norma ({passage}) letto sui fatti narrati",
    "contro": "{who} sosterrà che la norma governa i fatti narrati",
    "argomento": "Il giudizio di applicabilità sul testo della norma: {perche}"}
FIT_OBJECTION = ("{who} potrà sostenere che l'art. {art} non governa i fatti narrati: {perche}",
                 "Il punto si decide sul testo della norma ({passage}) e sui fatti che restano da provare")
# v79.14.2 (A), my drafts: a «no» resting on the facts it quotes, with no passage of the article quoted
FIT_CONTEST_PRO_FACTS = "Si fonda sui fatti narrati ({facts}) letti sul testo della norma"
FIT_OBJECTION_ANSWER_FACTS = "Il punto si decide sui fatti narrati ({facts}) e su quelli che restano da provare"
FIT_QUESTION = "Art. {art}: {question}"
FIT_PROSE_TITLE = "Norme di cui si contesta l'applicabilità ai fatti"     # v79.13.0 (review): the prose answer's block
FIT_OPEN_ROW = "dipende dalla risposta a: {question}"
FIT_OPEN_WHY = "resta da provare: dipende dalla risposta a «{question}»"
FIT_REMOVED_TRACE = "giudicato non applicabile ai fatti (giudizio di applicabilità)"   # the nightly's ranked trace
FLOW_COUNTERS = ("fit_judged", "fit_si", "fit_no", "fit_dipende", "fit_unverified", "fit_added_content",
                 "fit_skipped", "fit_calls", "fit_failed_calls", "fit_timeouts", "norms_removed_by_fit",
                 "fit_kept_named", "fit_added_content_in_effects", "fit_seconds",
                 "fit_no_verified", "fit_removed",                                 # v79.14.2 (A)
                 "fit_unverified_ids", "fit_no_unverified_ids", "fit_protected_kept")   # v79.15.0 (item 3)


def _quoted(parts: Iterable[Any]) -> str:
    return "; ".join(f"«{' '.join(str(x).split())}»" for x in parts or [] if str(x or "").strip())


def contest_pro(item: Dict[str, Any]) -> str:
    """The kept norm's contest, its «pro»: on the article's quoted passage (FIT_CONTEST_ARGUMENT «pro»), else -- v79.14.2
    (A), a «no» quoting only the facts -- on the facts it quotes (FIT_CONTEST_PRO_FACTS)."""
    if item.get("passage") or not item.get("fact_quotes"):
        return FIT_CONTEST_ARGUMENT["pro"].format(passage=f"«{item.get('passage') or ''}»")
    return FIT_CONTEST_PRO_FACTS.format(facts=_quoted(item.get("fact_quotes")))


def objection_answer(item: Dict[str, Any]) -> str:
    """The other side's objection, its answer: on the article's quoted passage (FIT_OBJECTION's), else -- v79.14.2 (A)
    -- on the facts the «no» quotes (FIT_OBJECTION_ANSWER_FACTS)."""
    if item.get("passage") or not item.get("fact_quotes"):
        return FIT_OBJECTION[1].format(passage=f"«{item.get('passage') or ''}»")
    return FIT_OBJECTION_ANSWER_FACTS.format(facts=_quoted(item.get("fact_quotes")))


def printable(text: Any, known: Iterable[str] = ()) -> str:
    """A perché or a condizione as the reader reads it: a parenthesis of fact ids alone dropped («(S1, S3)»), a bare
    id «il fatto narrato», the spaces tidied. The quoted passages are kept as they are."""
    s = " ".join(str(text or "").split())
    ids = r"(?:{p})".format(p=config()["fact_id_pattern"].replace("\\b", ""))
    for k in sorted({str(x) for x in known if x}, key=len, reverse=True):
        ids += rf"|(?:{re.escape(k)})"
    s = re.sub(rf"\s*\(\s*(?:fatt[oi]\s+)?(?:{ids})(?:\s*(?:,|e|ed|;)\s*(?:{ids}))*\s*\)", "", s)
    s = re.sub(rf"(?<![\w«-])(?:fatt[oi]\s+)?(?:{ids})(?![\w»-])", "il fatto narrato", s)
    s = re.sub(r"\s+([.,;:?])", r"\1", s)
    return s.strip()


_LEAD_RE = re.compile(r"^(L['’]|Un['’]|(?:Il|Lo|La|I|Gli|Le|Un|Uno|Una|Non|Nel|Nella|Nei|Negli|Nelle|Per|Poiché|"
                      r"Perché|Secondo|Dal|Dalla|Dai|Del|Della|Dei|Questo|Questa|Qui|Nessun|Nessuna|Mentre|Anche)"
                      r"(?=\W))")


def clause(text: Any) -> str:
    """A printable perché after a colon: its leading article or connective lowercased («Il prestito …» -> «il prestito
    …»), a name or a quote left as it is; the final full stop out."""
    s = str(text or "").strip().rstrip(".")
    return _LEAD_RE.sub(lambda m: m.group(1).lower(), s, count=1)


def norm_input(label: str, row: Optional[Dict[str, Any]], rec: Optional[Dict[str, Any]] = None,
               rows: Iterable[Dict[str, Any]] = ()) -> Dict[str, Any]:
    """One candidate as the step reads it: {article, text (the row's, footnotes out), rubrica, note (reviewed_note),
    elements (the record's requisites: id, testo), schema (the record's schema summary, else the row's -- ELEMENT_SCHEMAS
    on only: context, never the verdict)}."""
    rec = rec or {}
    sec = (row or {}).get("s") or {}
    text = str(sec.get("plain_text") or "")
    try:
        from . import rule_quote
        text = rule_quote.clean_text(text) if text else text
        rubrica = rule_quote.clean_rubrica(str(sec.get("titolo_articolo") or "")) if sec.get("titolo_articolo") \
            else str(rec.get("rubrica") or "")
    except Exception:           # noqa: BLE001 -- the raw text then
        rubrica = str(sec.get("titolo_articolo") or rec.get("rubrica") or "")
    schema = rec.get("schema") if isinstance(rec.get("schema"), dict) else None
    if schema is None and row is not None and _es().enabled():
        try:
            from . import issue_records
            schema = issue_records.schema_summary(issue_records.schema_line(row)[0])
        except Exception:       # noqa: BLE001
            schema = None
    return {"article": label, "text": text, "rubrica": rubrica, "note": reviewed_note(label, rows),
            "elements": [{"id": r.get("id"), "testo": r.get("text")} for r in rec.get("requisites") or []
                         if isinstance(r, dict) and r.get("id")],
            **({"schema": schema} if schema else {})}


def facts_input(sentences: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The case's sentences as the step's facts: {id, text, certainty}."""
    return [{"id": s.get("id"), "text": s.get("text"), "certainty": s.get("certainty")}
            for s in sentences or [] if isinstance(s, dict) and s.get("id") and str(s.get("text") or "").strip()]


def effects(result: Dict[str, Any], protected: Iterable[str]) -> Dict[str, List[Dict[str, Any]]]:
    """What the verified judgments do: {"removed": the «no» of a norm no one protects, "kept_named": the «no» of a
    norm the user names or the text charges (kept: the defence's contest), "open": the «dipende»}; each entry {article,
    fit, perche, passage, comma, facts, condizione, ambito, touches}. An unverified judgment is in none."""
    es = _es()
    guard = {es.norm_label(x) for x in protected or []} | {" ".join(str(x).lower().split()) for x in protected or []}
    out: Dict[str, List[Dict[str, Any]]] = {"removed": [], "kept_named": [], "open": []}
    for rec in result.get("judgments") or []:
        if not rec or not rec.get("verified"):
            continue
        entry = {k: rec.get(k) for k in ("article", "fit", "perche", "passage", "comma", "facts", "condizione",
                                         "ambito", "touches", "added_content", "fact_quotes", "cita_fatti",
                                         "cita_norma")}
        label = str(rec.get("article") or "")
        if rec["fit"] == NO:
            named = es.norm_label(label) in guard or " ".join(label.lower().split()) in guard
            out["kept_named" if named else "removed"].append(entry)
        elif rec["fit"] == DIPENDE:
            out["open"].append(entry)
    return out


def facts_signature(facts: Iterable[Dict[str, Any]]) -> str:
    """The facts a judgment was given, as one hash (id, text, certainty): a removal holds while they are the same."""
    rows = sorted((str(f.get("id") or ""), " ".join(str(f.get("text") or "").split()), str(f.get("certainty") or ""))
                  for f in facts or [] if isinstance(f, dict))
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False).encode("utf-8")).hexdigest()[:12]


_TOPIC_FIELDS = ("articles", "matched", "offences")


def strip_topics(topics: List[Dict[str, Any]], keys: set, key_of: Callable[[str], Any],
                 drop_empty: bool = False, marks: Optional[Dict[Any, List[Dict[str, Any]]]] = None
                 ) -> List[Dict[str, Any]]:
    """The topics with the removed norms out of their lists (articles, matched, offences, grades), in place; a topic's
    anchor removed -- its next article the anchor. A topic left with no article stays in the list (the topics'
    numbers are the user's choices) unless drop_empty. marks: {key: [{"i", "before", "dropped"}]} -- each touched
    topic as it was (restore_topics puts a readmitted norm back where it stood). The topics."""
    import copy
    out = []
    for i, t in enumerate(topics or []):
        if not isinstance(t, dict):
            out.append(t)
            continue
        gone = lambda x: key_of(str(x)) in keys
        before = copy.deepcopy(t) if marks is not None else None
        for field in _TOPIC_FIELDS:
            if isinstance(t.get(field), list):
                t[field] = [x for x in t[field] if not gone(x)]
        if isinstance(t.get("grades"), dict):
            t["grades"] = {k: v for k, v in t["grades"].items() if not gone(k)}
        if t.get("anchor") and gone(t["anchor"]):
            t["anchor"] = (t.get("articles") or [""])[0]
        if isinstance(t.get("planned"), int):
            t["planned"] = min(t["planned"], len(t.get("articles") or []))
        kept = bool(t.get("articles") or "articles" not in t or not drop_empty)
        if kept:
            out.append(t)
        if before is not None:
            hit = {key_of(str(x)) for f in _TOPIC_FIELDS for x in before.get(f) or [] if gone(x)}
            hit |= {key_of(str(x)) for x in (before.get("grades") or {}) if gone(x)}
            if before.get("anchor") and gone(before["anchor"]):
                hit.add(key_of(str(before["anchor"])))
            for k in hit:
                marks.setdefault(k, []).append({"i": i, "before": before, "dropped": not kept})
    return out


def restore_topics(topics: List[Dict[str, Any]], key: Any, marks: Sequence[Dict[str, Any]],
                   key_of: Callable[[str], Any]) -> List[Dict[str, Any]]:
    """A readmitted norm back in the topics where it stood (strip_topics' marks): in each list at its place, its grade,
    the anchor it was, the planned count; a topic dropped for it back at its place. The topics."""
    import copy
    topics = list(topics or [])
    for m in marks or []:
        before = m.get("before") or {}
        if m.get("dropped"):
            topics.insert(min(int(m.get("i") or 0), len(topics)), copy.deepcopy(before))
            continue
        i = int(m.get("i") or 0)
        if i >= len(topics) or not isinstance(topics[i], dict):
            continue
        t = topics[i]
        for field in _TOPIC_FIELDS:
            if not isinstance(before.get(field), list) or not isinstance(t.get(field), list):
                continue
            for pos, x in enumerate(before[field]):
                if key_of(str(x)) == key and x not in t[field]:
                    t[field].insert(min(pos, len(t[field])), x)
        for x, v in (before.get("grades") or {}).items():
            if key_of(str(x)) == key and isinstance(t.get("grades"), dict):
                t["grades"] = {k: (before["grades"][k] if k in before["grades"] else t["grades"][k])
                               for k in [k for k in before["grades"] if k in t["grades"] or key_of(str(k)) == key]
                               + [k for k in t["grades"] if k not in before["grades"]]}
        if before.get("anchor") and key_of(str(before["anchor"])) == key:
            t["anchor"] = before["anchor"]
        if isinstance(before.get("planned"), int):
            t["planned"] = min(before["planned"], len(t.get("articles") or [])) if "articles" in t else \
                before["planned"]
    return topics


def force_open(records: Dict[str, Dict[str, Any]], opened: Iterable[Dict[str, Any]],
               texts: Optional[Dict[str, str]] = None, known: Iterable[str] = ()) -> Dict[str, List[str]]:
    """The «dipende» judgments on the records, in place: every earlier mark cleared, then the elements each condizione
    touches (touches() against the record's requisites now -- the decomposition may have renumbered them; none touched:
    the requisites of the quoted passage's comma) marked «fit_open» with the question (issue_records.settle keeps them
    «da provare» whatever the link or a later grading says). {article: [the element ids marked]}."""
    for rec in (records or {}).values():
        for r in list(rec.get("requisites") or []) + list(rec.get("procedibilita") or []):
            if isinstance(r, dict):
                r.pop("fit_open", None)
    done: Dict[str, List[str]] = {}
    for entry in opened or []:
        label = str(entry.get("article") or "")
        rec = (records or {}).get(label)
        if not rec:
            continue
        reqs = [r for r in rec.get("requisites") or [] if isinstance(r, dict) and r.get("id")]
        question = printable(entry.get("condizione"), known)
        hit = touches(str(entry.get("condizione") or ""), {
            "elements": [{"id": r["id"], "testo": r.get("text")} for r in reqs],
            "text": (texts or {}).get(label, "")})
        ids = list(hit.get("elements") or [])
        if not ids:
            commi = [str(c) for c in hit.get("commi") or []] or ([str(entry["comma"])] if entry.get("comma") else [])
            ids = [r["id"] for r in reqs if str(r.get("comma") or "") in commi]
        for r in reqs:
            if r["id"] in ids:
                r["fit_open"] = question
        done[label] = ids
    return done


def questions(entries: Iterable[Dict[str, Any]], labels: Iterable[str], known: Iterable[str] = ()) -> List[str]:
    """§10's points: each «dipende» condizione of a norm still in the answer (labels), as the user's question."""
    have = set(labels or [])
    out = []
    for e in entries or []:
        q = printable(e.get("condizione"), known)
        if q and str(e.get("article") or "") in have:
            line = FIT_QUESTION.format(art=e["article"], question=q)
            if line not in out:
                out.append(line)
    return out


def removed_cited(text: str, keys: set, key_of: Callable[[Sequence[str]], Any]) -> List[str]:
    """The removed norms (their keys) the text still cites, each once."""
    from . import citations
    return list(dict.fromkeys(citations.label(r) for r in citations.article_refs(text or "") if key_of(r) in keys))


def removed_out(text: str, keys: set, key_of: Callable[[Sequence[str]], Any], log: Optional[List[str]] = None) -> str:
    """The last pass: every sentence citing a removed norm dropped (logged)."""
    from . import citations, lawyer_method
    if not text or not keys:
        return text or ""

    def review(sentence: str) -> str:
        hit = [citations.label(r) for r in citations.article_refs(sentence) if key_of(r) in keys]
        if hit and log is not None:
            log.append(f"{', '.join(dict.fromkeys(hit))}: {sentence[:120]}")
        return "" if hit else sentence
    return lawyer_method._by_sentence(text, review)
