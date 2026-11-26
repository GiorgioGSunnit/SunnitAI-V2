"""v79.14.0: the case frame -- one model call at the very top of the pipeline, before intake's rules run, reads the case
frame from the request's full text; the code verifies every field against the text and never overrides a verified
frame with a regex. Intake's and posture's rules are the fallback for the fields the model leaves empty or that fail
verification.

Behind CASE_FRAME (default off: nothing here runs, no output changes). The tables, the vocabularies and the prompt are
data/case_frame.json (DATA_FILE); the prompt is hashed (prompt_hash) and the hash logged with every run. (Not the older
CASE_FRAME_ENABLED of defensive_generation._case_frame: another path, another flag.)

The call (read): one call, thinking off (chat_template_kwargs enable_thinking=False, as applicability.request_kwargs),
temperature 0, a JSON reply, the request's full text. The reply:
  parti               [{nome, ruolo_nel_fatto, lato ∈ cliente | controparte | terzo | autorità}]
  cliente             {nome, perche} -- the party the questions are written for, its perché quoting the sentence
  ruolo               {valore ∈ attore | convenuto | imputato/indagato | persona_offesa | ricorrente | resistente |
                       sconosciuto, perche}
  rito                {valore ∈ intake's riti, perche}
  stadio              {valore ∈ pre_filing | indagini | primo_grado | appello | cassazione | esecuzione | sconosciuto,
                       perche}
  addebito_o_domanda  {articolo or null, descrizione, frase}
  istituto            {testo, frase}
  quesiti             [the questions verbatim, in order]
Each value is mapped onto intake's own vocabulary by the data (ruolo «imputato/indagato» -> intake «imputato»,
«ricorrente» -> «attore», «resistente» -> «convenuto»; stadio «appello» / «cassazione» -> «impugnazione»): every
verified value carries both («valore», «intake»). A failed call, a reply that is no JSON object, a call not back by
«deadline_s» (15 s): the whole frame empty, every field to the fallback, logged -- never a crash.

The verification (verify -- pure code, no model). Every quote is matched as the 79.13 fit checker matches a passage
(applicability.passage_at: word by word, each content word by its stem, short words, numbers and «non» exactly, never
just after a negation of the text):
  - a perché / frase: its quoted passages («…», “…”, "…") each in the text -- or, with no quote marks, the whole of it;
  - a name (a party's, the client's): exactly, never by stem (name_found: «Mario Rossi» is not «Maria Rossi»; titles
    aside, the company forms written one way: «Orsa Srl» is «Orsa S.r.l.»);
  - parti: each party's name in the text and its lato in the vocabulary, else the party dropped (an off-schema lato
    saying controparte still bars it as the client); its ruolo_nel_fatto blanked when a content word of it is not in
    the text;
  - cliente: a quote found (no quote: never accepted) that shows it (more than its bare name; a quote naming another
    party and not it: refused), its name in the text, one of the verified parties with lato cliente, never a party
    the reply lists as controparte, never another listed party than the user's «assisto X», never named only «a
    carico di» in its quote unless the text says the lawyer assists or defends it; one client only (a list of more:
    dropped); refused on its sides, it takes the reply's ruolo with it;
  - ruolo / rito / stadio: a value of the vocabulary and its quote found (no quote: never accepted); «sconosciuto» with
    its quote found is a verified value (the honesty line, neither side's wording: role_unknown) -- with no quote, the
    field is empty and falls back;
  - addebito_o_domanda: its frase found; its descrizione blanked when a content word of it is not in the text; in a
    penal rito (the frame's -- «sconosciuto» is none --, else rito_hint, the fallback's) its articolo, if any, a penal
    provision (is_penal: the c.p., the civil code's reati societari, a penal law's penal articles -- data
    «penal_provisions.laws», by range) -- else the article is dropped and counted charge_rito_dropped; a bare number
    takes the code the text cites it with, else the article is dropped (frame_unverified.addebito_o_domanda.articolo);
  - istituto: its frase found and its testo in the lawyer's words (every content word in the text);
  - quesiti: each a whole question of the text (question_units), else dropped; the kept ones in the text's order,
    once each.
A field that fails is dropped and logged (log «case_frame_unverified»; counter frame_unverified.<field>).

  read(text, call=None, rito_hint="")   the step: {"frame", "status", "error", "seconds", "prompt_hash", "verified",
                                         "unverified", "empty", "fallback", "counters", "log"}; call(messages,
                                         kwargs) -> reply text replaces the model (tests, the timing script)
  verify(raw, text, rito_hint="")       the reply checked: the same dict without the call's keys
  fallback_fields(result)               the fields the rules fill (empty or unverified; all when the call failed)
  count_fallback(counters, field)       build B: a rule filled the field -- frame_fallback.<field> += 1
  intake_view(result)                   the verified frame in intake's words: {client, role, rito, stage, charge,
                                         institution, questions, role_unknown} (None where the rules decide)
  role_unknown(result)                  a verified «sconosciuto» role: the honesty line, neither side's wording
  fit_context(result)                   the verified frame as the 79.13 applicability judgment's context ("" none)

Counters: frame_fields_verified (the fields verified non-empty, of the eight), frame_unverified.<field> (a field
dropped by verification; a party or a question dropped counts one each; a part blanked counts as
frame_unverified.<field>.<part>), charge_rito_dropped (a non-penal article as a penal rito's charge, dropped -- the
reply's error, never in the frame), frame_failed (the call failed, its reply was no JSON object, or it ran out of
time), frame_seconds.

The binding (build B). The step runs in intake.request_intake, before intake's rules (once per request: the API, the
case-first flow, the sims all read the intake there; a follow-up reuses the state's frame, a new case reads its own),
the result in the intake («case_frame») and the state (state["case_frame"], interactive_flow.take_intake). The verified
frame is the single source of client, role, rito, stage, charge or claim, institution and questions: intake takes them
first (confidence «frame»: its rules, the model and derive_role only fill what the frame leaves); every later step that
re-derives one defers (defer: a different value is logged frame_conflict with the step's name and loses) --
intake.resolve, posture.of, posture.choose_charge (and the decomposition's charge), interactive_flow.stated's client,
settle_client, client_of, clients_of (the rules' several clients only around the frame's own), the plan's client and
position and the extraction's position (position), the first read's client and institution
(the frame's institution the first one module activation reads: interactive_flow.read_institutions), the method's
«Risposte ai quesiti» (intake's questions). A verified «sconosciuto» role: the honesty line, neither side's wording.
  held(holder) / bound(holder, field) / defer(holder, field, step, proposed) / conflict(...)
  audit(state)       the counters on the delivered answer: frame_fallback.<field> (the rules filled it),
                     frame_conflict (a bound field delivered changed: must be 0), frame_conflict_attempts (the steps
                     that disagreed and lost), role_flip (intake's role not the method's, or the counterparty taken as
                     client: must be 0), charge_rito_mismatch (a penal rito's held charge no penal provision: must be 0)
"""

import contextvars
import hashlib
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _Timeout
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .feature_flags import enabled as _flag_enabled

logger = logging.getLogger(__name__)

FLAG = "CASE_FRAME"
DATA_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data",
                         "case_frame.json")
FIELDS = ("parti", "cliente", "ruolo", "rito", "stadio", "addebito_o_domanda", "istituto", "quesiti")
VALUE_FIELDS = {"ruolo": "ruoli", "rito": "riti", "stadio": "stadi"}
PENAL_RITI = ("penale", "penale_gdp")
COUNTERS = ("frame_fields_verified",) + tuple(f"frame_unverified.{f}" for f in FIELDS) + \
    tuple(f"frame_fallback.{f}" for f in FIELDS) + ("charge_rito_dropped", "frame_failed", "frame_seconds") + \
    ("frame_corrections_applied", "frame_pretese_count", "frame_article_disputed", "frame_role_deduced") + \
    ("frame_hypothesis_confirmed", "frame_hypothesis_dropped", "frame_article_uncited")   # v79.19.0; .1 (1); .2 (2)
OK, FAILED, MALFORMED, TIMEOUT, NO_TEXT = "ok", "failed", "malformed", "timeout", "no_text"
# v79.19.0 (item 3): the frame may read several charges / claims (addebito_o_domanda a list of 1..MAX_PRETESE); the
# first is the charge the principale line is built on (frame «addebito_o_domanda»), every verified one is in frame
# «addebiti» (the others: alternative lines, their own §4 schede). (item 4) a role the text does not state but the
# facts allow to deduce: «fonte»: «dedotto» (frame ruolo «dedotto»: the banner says so). (item 5) the article the facts
# point to when it differs from the one the text names: «articolo_dai_fatti» (frame addebito «articolo_fatti»,
# «disputed»: both go to the fit, the judgment decides -- apply_fit).
MAX_PRETESE = 3
DEDUCED = "dedotto"
SETTLED = "settled"             # state["frame"]: the corrected frame, written once (settle), read by every later step
CORRECTION_EVENT = "frame_correction"


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


def _ap():
    from . import applicability
    return applicability


def _es():
    from . import element_schemas
    return element_schemas


def _flat(value: Any) -> str:
    return " ".join(str(value or "").split())


# --- the words --------------------------------------------------------------------------------------------------------

def found_at(text: str, quote: str) -> Optional[int]:
    """Where a quote occurs in the text -- the fit checker's own match (applicability.passage_at: word by word, stem
    tolerant, never just after a negation); None when it does not."""
    return _ap().passage_at(text, _flat(quote).strip(" «»“”\"'.,;:"))


def quote_check(text: str, said: Any) -> Tuple[bool, List[str], str]:
    """(found, the quoted passages, why not) for a perché or a frase: its quoted passages («…», “…”, "…") each in the
    text; with no quote marks, the whole of it. An empty one: not found («senza citazione»)."""
    s = _flat(said)
    if not s:
        return False, [], "senza citazione"
    quoted = _ap().quotes(s) or [s.strip(" «»“”\"'")]
    for q in quoted:
        if found_at(text, q) is None:
            return False, quoted, f"citazione non nel testo: «{q[:160]}»"
    return True, quoted, ""


def ungrounded(words: Any, text: str) -> List[str]:
    """The content words of words not in the text (element_schemas.missing_words: stem, verb base, polarity)."""
    return _es().missing_words(_flat(words), text) if _flat(words) else []


def _fold(value: Any) -> str:
    return _flat(value).lower()


def canon(value: Any, table: str) -> Optional[str]:
    """A value of the data's table (ruoli, riti, stadi, lati) however spelt: as such, its spaces as underscores or the
    reverse, or by the table's «…_sinonimi»; None outside it."""
    cfg = config()
    values = cfg.get(table) or {}
    keys = list(values) if isinstance(values, dict) else list(values)
    synonyms = cfg.get(f"{table}_sinonimi") or {}
    v = _fold(value).strip(" .;:")
    for cand in (v, v.replace(" ", "_"), v.replace("_", " "), v.replace(" / ", "/")):
        if cand in keys:
            return cand
        if cand in synonyms:
            return synonyms[cand]
    return None


def _name_words(name: Any) -> frozenset:
    es = _es()
    return frozenset(t[1] for t in _ap()._full_tokens(name) if len(t[1]) >= 2 and t[1] not in es.FUNCTION_WORDS
                     and not any(c.isdigit() for c in t[1]))


def name_tokens(value: Any) -> List[str]:
    """A name's (or a passage's) words for the name match: accent-folded, lowercased, the company forms written one way
    (data «company_forms»: «S.r.l.», «Srl», «s.r.l» -> «srl»), every other mark a word boundary."""
    import unicodedata
    s = unicodedata.normalize("NFKD", _flat(value).replace("’", "'").lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    for pattern, form in _company_forms():
        s = pattern.sub(f" {form} ", s)
    return re.findall(r"[a-z0-9]+", s)


@lru_cache(maxsize=1)
def _company_forms() -> Tuple[Tuple[Any, str], ...]:
    out = []
    for form, pattern in (config().get("company_forms") or {}).items():
        try:
            out.append((re.compile(pattern), form))
        except re.error as exc:
            logger.warning("case_frame: company form %r skipped: %s", form, exc)
    return tuple(out)


def _bare_name(name: Any) -> List[str]:
    """The name's words without its leading titles and articles (data «name_titles»: «il signor», «la società», «Sig.»,
    «dott.»)."""
    words = name_tokens(name)
    titles = set(config().get("name_titles") or [])
    while words and words[0] in titles:
        words = words[1:]
    return words


def name_found(text: str, name: Any) -> bool:
    """The name occurs in the text as written -- its words (titles aside, company forms normalised) one after the
    other, each exactly: never by stem («Mario Rossi» is not «Maria Rossi»)."""
    words, hay = _bare_name(name), name_tokens(text)
    n = len(words)
    return bool(n) and any(hay[i:i + n] == words for i in range(len(hay) - n + 1))


def _distinctive(name: Any) -> frozenset:
    """The words that tell a party apart in a passage: the name's own (no title, no company form, no function word)."""
    forms = set(config().get("company_forms") or {})
    es = _es()
    return frozenset(w for w in _bare_name(name) if len(w) >= 2 and w not in forms and w not in es.FUNCTION_WORDS
                     and not w.isdigit())


def names_in(passage: Any, name: Any) -> bool:
    """The passage names the party: one of its distinctive words among the passage's («Orsa» names «Orsa S.r.l.»)."""
    own = _distinctive(name)
    return bool(own) and bool(own & set(name_tokens(passage)))


def same_party(a: Any, b: Any) -> bool:
    """Two names of one party: the content words of one all among the other's («Mario Rossi», «Rossi»)."""
    x, y = _name_words(a), _name_words(b)
    return bool(x and y and (x <= y or y <= x))


# --- the charge's article ---------------------------------------------------------------------------------------------

# where an article's code ends: a word that goes on with the sentence, a stop before a capital, a punctuation mark
_CODE_END_RE = re.compile(r"^(.*?)(?=\s+(?:e|ed|o|che|per|nel|nella|nei|in|con|a|ai|al|alla|il|la|lo|le|gli|perché|"
                          r"poiché|quando|se|ma)\s|[,;:()?!«»]|(?<=\.)\s+[A-ZÀ-Ü]|$)", re.DOTALL)


def parse_article(raw: Any) -> Dict[str, str]:
    """{number (normalised: «640-bis»), base (the digits), code (compact: «cp», «cc», «dpr309/1990», "" bare), rest (the
    code as written)} of an «articolo» as a model may write it («art. 640 c.p.», «640-bis c.p.», «73, comma 5, d.P.R.
    309/1990», «2621 c.c.»); number "" when none."""
    from .article_suffix import SUFFIX, normalize_number
    s = _flat(raw)
    s = re.sub(r"\([^)]*\)", " ", s)
    s = re.sub(r"^\s*(?:art(?:icolo|t)?\.?)\s*", "", s, flags=re.IGNORECASE)
    m = re.match(rf"^(\d+)(?:\s*-?\s*({SUFFIX})(?![a-z]))?", s, re.IGNORECASE)
    if not m:
        return {"number": "", "base": "", "code": "", "rest": s}
    number = normalize_number(m.group(0).replace(" ", ""))
    rest = s[m.end():]
    rest = re.sub(r"^\s*[,;]?\s*(?:(?:comm[ai]|co\.|lett(?:era|\.)?|n\.|numero)\s*[\w\-]+\)?\s*,?\s*)*", "", rest,
                  flags=re.IGNORECASE)
    rest = re.sub(r"^\s*(?:del|della|dello|dei|delle|di|ex)\s+", "", rest, flags=re.IGNORECASE).strip(" ,;:-")
    rest = code_or_law(rest)        # v79.19.1 (item 4): the code or the law alone -- «c.c. inadempimento» -> «c.c.»
    return {"number": number, "base": m.group(1), "code": _ap()._code_key(rest) if rest else "", "rest": rest}


# v79.19.1 (item 4): where a law's name ends -- its kind («d.P.R.», «d.lgs.», «l.», «legge», «r.d.», «d.l.», «t.u.»)
# with its number and year («309/1990», «n. 300 del 1970»); the words after it are the charge's description, never the
# law's name
_LAW_RE = re.compile(
    r"^(?:d\.?\s?p\.?\s?r\.?|d\.?\s?lgs?\.?|d\.?\s?l\.?|d\.?\s?m\.?|r\.?\s?d\.?|l\.?|legge|decreto[\s-]legge|"
    r"decreto\s+legislativo|decreto\s+del\s+presidente\s+della\s+repubblica|regio\s+decreto|t\.?\s?u\.?[a-z.]*|"
    r"testo\s+unico(?:\s+[a-zà-ü]+){0,3}?)\s*(?:n\.?\s*)?\d+(?:[/-]\d{2,4}|\s+del\s+\d{4}|\s*\(\d{4}\))?", re.IGNORECASE)


def code_or_law(rest: Any) -> str:
    """v79.19.1 (item 4): an article's code or law alone, the words after it cut -- a code the lead-code table names
    («c.c.», «c.p.p.», «codice civile»: _LEAD_CODE_RE), else a law's kind, number and year (_LAW_RE), else the older
    cut at a sentence word or a punctuation mark (_CODE_END_RE). «c.c. inadempimento del contratto» -> «c.c.»,
    «d.P.R. 309/1990 detenzione» -> «d.P.R. 309/1990», «l. 300/1970 reintegrazione» -> «l. 300/1970»."""
    text = _flat(rest).strip(" ,;:-")
    if not text:
        return ""
    m = _LEAD_CODE_RE.match(text)
    if m:
        return m.group(0).strip(" ,;:-")
    m = _LAW_RE.match(text)
    if m and not re.match(r"[A-Za-zÀ-ü]", text[m.end():m.end() + 1] or ""):
        return m.group(0).strip(" ,;:-")
    return _CODE_END_RE.match(text).group(1).strip(" ,;:-")


# a code written first in the article's code («c.c.», «c.p.p.», «codice civile» ...): that code is the article's, whatever
# follows («1439 c.c. ex d.lgs. 74/2000» is the civil code's)
_LEAD_CODE_RE = re.compile(
    r"^(?:(?P<cpc>c\.\s?p\.\s?c\.?|cpc|codice di procedura civile)"
    r"|(?P<cpp>c\.\s?p\.\s?p\.?|cpp|codice di procedura penale)"
    r"|(?P<cpa>c\.\s?p\.\s?a\.?|cpa|codice del processo amministrativo)"
    r"|(?P<cc>c\.\s?c\.?|cc|codice civile|cod\.\s?civ\.?)"
    r"|(?P<cp>c\.\s?p\.?|cp|codice penale|cod\.\s?pen\.?)"
    r"|(?P<cost>cost\.?|costituzione))(?=\s|$|[,;:])", re.IGNORECASE)


def lead_code(rest: Any) -> str:
    """The code an article's code starts with («cc», «cp», «cpc», «cpp», «cpa», «cost»), "" for none (a law)."""
    m = _LEAD_CODE_RE.match(_flat(rest))
    return next((k for k, v in m.groupdict().items() if v), "") if m else ""


@lru_cache(maxsize=1)
def _penal_laws() -> Tuple[Dict[str, Any], ...]:
    """The penal laws of data «penal_provisions.laws», each with its compiled code expression (its own «expression»,
    else data/special_laws.json's entry «special_law_id») and its penal articles («articles»: ranges of the base
    number; «numbers»: exact numbers; neither: every article)."""
    out = []
    laws_by_id: Dict[str, Any] = {}
    try:
        from . import special_laws
        laws_by_id = {law.get("id"): law for law in special_laws.laws()}
    except Exception as exc:    # noqa: BLE001
        logger.warning("case_frame: special laws not read: %s", exc)
    for entry in (config().get("penal_provisions") or {}).get("laws") or []:
        try:
            if entry.get("expression"):
                rx = re.compile(entry["expression"], re.IGNORECASE)
            elif entry.get("special_law_id") in laws_by_id:
                rx = laws_by_id[entry["special_law_id"]]["_code"]
            else:
                logger.warning("case_frame: penal law %r has no expression", entry.get("name"))
                continue
        except re.error as exc:
            logger.warning("case_frame: penal law %r skipped: %s", entry.get("name"), exc)
            continue
        out.append({**entry, "_rx": rx})
    return tuple(out)


def _in_articles(article: Dict[str, str], entry: Dict[str, Any]) -> bool:
    ranges, numbers = entry.get("articles"), entry.get("numbers")
    if ranges is None and numbers is None:
        return True
    if article.get("number") in (numbers or []):
        return True
    base = int(article["base"]) if str(article.get("base") or "").isdigit() else None
    return base is not None and any(lo <= base <= hi for lo, hi in ranges or [])


def is_penal(article: Dict[str, str]) -> bool:
    """A parsed article is a penal provision: the c.p.'s; a code's penal range (data «code_ranges»: the civil code's
    reati societari, 2621-2642); a penal law's penal articles (data «laws»: d.lgs. 74/2000 artt. 2-11, the legge
    fallimentare's 216-237, d.P.R. 380/2001 art. 44 ...). An article led by another code (c.c., c.p.c., c.p.p.,
    c.p.a., Cost.) is never a penal law's, whatever follows it; a law's article outside its penal ones is not penal
    («117 TUB», «67 l. fall.», «142 codice della strada»)."""
    p = config().get("penal_provisions") or {}
    if not article.get("number"):
        return False
    rest = article.get("rest") or ""
    lead = lead_code(rest) or (article.get("code") if article.get("code") in ("cp", "cc", "cpc", "cpp", "cpa", "cost")
                               else "")
    if lead:
        if lead in (p.get("codes") or []):
            return True
        return any(lo <= int(article["base"]) <= hi for lo, hi in (p.get("code_ranges") or {}).get(lead) or []
                   if str(article.get("base") or "").isdigit())
    return any(_in_articles(article, entry) for entry in _penal_laws() if entry["_rx"].search(rest))


def _cited_code(text: str, number: str) -> Optional[Dict[str, str]]:
    """A bare number's code as the text cites it («art. 640 c.p.» somewhere in the text): the parsed article, or None
    when the text cites it with no code or not at all."""
    base = re.escape(number.split("-")[0])
    for m in re.finditer(rf"\bart(?:icol[oi]|t)?\.?\s*{base}\b[^\n]{{0,60}}", text or "", re.IGNORECASE):
        got = parse_article(m.group(0))
        if got["number"] == number and got["code"]:
            return got
    return None


# --- verification -----------------------------------------------------------------------------------------------------

def empty_counters() -> Dict[str, Any]:
    out: Dict[str, Any] = {k: 0 for k in COUNTERS}
    out["frame_seconds"] = 0.0
    return out


def _empty_frame() -> Dict[str, Any]:
    return {f: ([] if f in ("parti", "quesiti") else None) for f in FIELDS}


def _blank(raw: Any) -> bool:
    if raw is None:
        return True
    if isinstance(raw, str):
        return not raw.strip() or raw.strip().lower() in ("null", "none", "nessuno", "nessuna")
    if isinstance(raw, (list, tuple, dict)):
        return not raw or (isinstance(raw, dict) and not any(_flat(v) for v in raw.values()))
    return False


def _get(item: Dict[str, Any], *keys: str) -> Any:
    for k in keys:
        if k in item:
            return item[k]
    return None


class _Report:
    def __init__(self, text: str):
        self.text = text
        self.frame = _empty_frame()
        self.unverified: Dict[str, List[str]] = {}
        self.empty: List[str] = []
        self.counters = empty_counters()
        self.log: List[Dict[str, Any]] = []
        self.corrections: List[Dict[str, Any]] = []      # v79.19.0 (item 2): the reading's corrections, old -> new

    def fail(self, field: str, reason: str, value: Any = None, count: bool = True) -> None:
        """A field (or a part of it, «field.part») dropped: logged, counted frame_unverified.<field>."""
        key = f"frame_unverified.{field}"
        if count:
            self.counters[key] = int(self.counters.get(key) or 0) + 1
        top = field.split(".")[0]
        self.unverified.setdefault(top, []).append(reason if top == field else f"{field.split('.', 1)[1]}: {reason}")
        entry = {"field": field, "reason": reason, "value": _flat(json.dumps(value, ensure_ascii=False)
                                                                   if not isinstance(value, str) else value)[:300]}
        self.log.append(entry)
        logger.info("case_frame: %s unverified: %s", field, reason)
        _log("case_frame_unverified", entry)


def _verify_value(rep: _Report, field: str, raw: Any) -> None:
    """ruolo / rito / stadio: a vocabulary value with its quote found; «sconosciuto» with its quote found is verified,
    with no quote the field is empty."""
    if _blank(raw):
        rep.empty.append(field)
        return
    if isinstance(raw, dict):
        value, why = _get(raw, "valore", "value"), _get(raw, "perche", "perché", "why")
    else:
        value, why = raw, ""
    table = VALUE_FIELDS[field]
    value_c = canon(value, table)
    if value_c is None:
        if _blank(value) and _blank(why):
            rep.empty.append(field)
            return
        rep.fail(field, f"valore fuori schema: {_flat(value)[:60]!r}", raw)
        return
    unknown = config().get("unknown") or "sconosciuto"
    if value_c == unknown and _blank(why):
        rep.empty.append(field)             # no quote: the text's silence is the rules' to read
        return
    ok, quoted, reason = quote_check(rep.text, why)
    if not ok:
        rep.fail(field, reason, raw)
        return
    rep.frame[field] = {"valore": value_c, "intake": (config().get(table) or {}).get(value_c, value_c),
                        "perche": _flat(why), "citazione": quoted}
    if field == "ruolo" and isinstance(raw, dict) and _deduced(_get(raw, "fonte", "source")) and value_c != unknown:
        rep.frame[field][DEDUCED] = True        # v79.19.0 (item 4): read from the facts, not stated: the banner says so
        rep.counters["frame_role_deduced"] = 1


def _deduced(fonte: Any) -> bool:
    """v79.19.0 (item 4): the reply says the role is deduced from the facts («fonte»: data «deduced_words»)."""
    f = _fold(fonte)
    return bool(f) and any(f == w or f.startswith(w) for w in (config().get("deduced_words") or [DEDUCED]))


def _verify_parti(rep: _Report, raw: Any) -> List[Dict[str, Any]]:
    """The parties kept (name in the text, lato in the vocabulary); every listed one with its lato read (for the
    controparte rule): returned."""
    listed: List[Dict[str, Any]] = []
    if _blank(raw):
        rep.empty.append("parti")
        return listed
    if not isinstance(raw, list):
        rep.fail("parti", "non è una lista", raw)
        return listed
    kept: List[Dict[str, Any]] = []
    for item in raw:
        if isinstance(item, str):
            item = {"nome": item}
        if not isinstance(item, dict):
            rep.fail("parti", "parte non è un oggetto", item)
            continue
        name = _flat(_get(item, "nome", "name"))
        lato = canon(_get(item, "lato", "side"), "lati")
        listed.append({"nome": name, "lato": lato or ("controparte" if _says_counterparty(_get(item, "lato", "side"))
                                                      else None)})
        if not name:
            rep.fail("parti", "parte senza nome", item)
            continue
        if lato is None:
            rep.fail("parti", f"lato fuori schema: {_flat(_get(item, 'lato', 'side'))[:40]!r} ({name})", item)
            continue
        if not name_found(rep.text, name):
            rep.fail("parti", f"nome non nel testo: «{name[:80]}»", item)
            continue
        role = _flat(_get(item, "ruolo_nel_fatto", "ruolo"))
        if role and ungrounded(role, rep.text):
            rep.fail("parti.ruolo_nel_fatto", f"parole non nel testo: {', '.join(ungrounded(role, rep.text)[:6])} "
                                              f"({name})", item)
            role = ""
        party = {"nome": name, "ruolo_nel_fatto": role, "lato": lato}
        if not any(p["nome"] == name and p["lato"] == lato for p in kept):
            kept.append(party)
    rep.frame["parti"] = kept
    if not kept and "parti" not in rep.unverified:
        rep.empty.append("parti")
    return listed


def _says_counterparty(lato: Any) -> bool:
    """An off-schema lato that still says controparte («controparte (creditore)», «parte avversaria»): data
    «controparte_words»."""
    low = " ".join(name_tokens(lato))
    return any(w in low for w in config().get("controparte_words") or [])


def _signalled_client(text: str, name: str) -> bool:
    """The text says the lawyer assists or defends the party: a sentence holding one of its distinctive words and a
    client signal (data «client_signals»: «assisto», «che difendo», «difenderlo», «il mio cliente» ...) within
    «client_signal_window» words of it, no counterparty marker between them («assisto Carla contro Dario» signals
    Carla only)."""
    own = _distinctive(name)
    if not own:
        return False
    signals = config().get("client_signals") or []
    window = int(config().get("client_signal_window") or 8)
    markers = [name_tokens(m) for m in config().get("counterparty_markers") or [] if name_tokens(m)]
    for sentence in re.split(r"[.?!;\n]+", text or ""):
        words = name_tokens(sentence)
        at = [i for i, w in enumerate(words) if w in own]
        sig = [i for i, w in enumerate(words) if any(w.startswith(x) for x in signals)]
        for i in at:
            for j in sig:
                if abs(i - j) > window:
                    continue
                between = words[min(i, j) + 1:max(i, j)]
                if not any(between[k:k + len(m)] == m for m in markers for k in range(len(between) - len(m) + 1)):
                    return True
    return False


def _named_as_counterparty(quote: str, name: str) -> bool:
    """The quote names the client only right after one of the data's «counterparty_markers» («quali reati a carico di
    Tizio»; an article or a title between: «contro la società Alfa»): there the name is whom the questions are about,
    never whom they serve. False when the quote does not name it at all (the quote shows the client otherwise:
    «assisto…», the questions' perspective)."""
    tok = lambda x: [t[1] for t in _ap()._full_tokens(x)]
    head = [w for w in tok(name) if w in _name_words(name)][:1]
    words = tok(quote)
    at = [i for i, w in enumerate(words) if head and w == head[0]]
    if not at:
        return False
    markers = [tok(m) for m in config().get("counterparty_markers") or [] if tok(m)]
    titles = set(config().get("name_titles") or [])

    def after_marker(i: int) -> bool:
        k = i
        while k > 0 and words[k - 1] in titles:
            k -= 1
        return any(len(m) <= k and words[k - len(m):k] == m for m in markers)
    return all(after_marker(i) for i in at)


def _stated_client(text: str) -> str:
    """The client the user names in their own words («assisto X», «sono il difensore di X», interactive_flow.stated);
    "" when none."""
    try:
        from .interactive_flow import stated
        return str(stated(text).get("client") or "")
    except Exception:           # noqa: BLE001 -- no stated client: the other checks decide
        return ""


def _verify_cliente(rep: _Report, raw: Any, listed: Sequence[Dict[str, Any]]) -> bool:
    """One client: its quote found and showing it (more than its bare name; naming it, or no other party), its name in
    the text as written and among the verified parties with lato cliente, never a party listed as controparte, never
    named in its quote only as the subject «a carico» of the questions (unless the text says the lawyer assists or
    defends it), never another listed party than the one the user says they assist («assisto X»). True when the
    client was refused on the sides -- the reply's sides contradict the text, so its role (the client's) goes too."""
    if isinstance(raw, list):
        raw = [x for x in raw if not _blank(x)]
        if len(raw) > 1:
            rep.fail("cliente", f"più di un cliente ({len(raw)}): il quadro ne indica uno", raw)
            return False
        raw = raw[0] if raw else None
    if _blank(raw):
        rep.empty.append("cliente")
        return False
    if not isinstance(raw, dict):
        raw = {"nome": raw}
    name, why = _flat(_get(raw, "nome", "name")), _get(raw, "perche", "perché", "why")
    if not name:
        rep.fail("cliente", "cliente senza nome", raw)
        return False
    ok, quoted, reason = quote_check(rep.text, why)
    if not ok:
        rep.fail("cliente", reason, raw)
        return False
    if not name_found(rep.text, name):
        rep.fail("cliente", f"nome non nel testo: «{name[:80]}»", raw)
        return False
    if any(p["lato"] == "controparte" and same_party(p["nome"], name) for p in listed):
        rep.fail("cliente", f"«{name[:80]}» è indicato tra le parti come controparte", raw)
        return True
    said = _stated_client(rep.text)     # the user's «assisto X», X a party the reply lists: the client is X
    if said and not same_party(said, name) and any(same_party(said, p["nome"]) for p in listed if p.get("nome")):
        rep.fail("cliente", f"il testo dice chi assisti: «{said[:80]}», non «{name[:80]}»", raw)
        return True
    if not any(names_in(q, name) for q in quoted):
        others = [p["nome"] for p in listed if p.get("nome") and not same_party(p["nome"], name)
                  and any(names_in(q, p["nome"]) for q in quoted)]
        if others:
            rep.fail("cliente", f"il passo citato mostra «{others[0][:80]}», non «{name[:80]}»", raw)
            return True
    if quoted and all(_named_as_counterparty(q, name) for q in quoted) and not _signalled_client(rep.text, name):
        rep.fail("cliente", f"nel passo citato «{name[:80]}» è il soggetto «a carico» del quale sono le domande, non "
                            f"il cliente", raw)
        return True
    own = _distinctive(name) | set(_bare_name(name))
    es = _es()
    if not any(w not in own and w not in es.FUNCTION_WORDS and len(w) >= 2 for q in quoted for w in name_tokens(q)):
        rep.fail("cliente", f"il passo citato è solo il nome di «{name[:80]}»: non mostra chi è il cliente", raw)
        return False
    if not [p for p in rep.frame["parti"] if same_party(p["nome"], name)]:
        rep.fail("cliente", f"«{name[:80]}» non è tra le parti verificate", raw)
        return False
    parties = [p for p in rep.frame["parti"] if same_party(p["nome"], name) and p["lato"] == "cliente"]
    if not parties:
        rep.fail("cliente", f"«{name[:80]}» non è tra le parti indicate con lato cliente", raw)
        return True
    rep.frame["cliente"] = {"nome": name, "perche": _flat(why), "citazione": quoted, "lato": parties[0]["lato"]}
    return False


def _verify_addebito(rep: _Report, raw: Any, rito: str) -> None:
    """The charge or claim: its frase found; its descrizione in the text's words; in a penal rito its article a penal
    provision. v79.19.0 (item 3): a list of 1..MAX_PRETESE such objects -- each verified on its own; the first verified
    one is «addebito_o_domanda», all of them «addebiti» (frame_pretese_count); more than MAX_PRETESE: the rest dropped
    and logged."""
    if _blank(raw):
        rep.empty.append("addebito_o_domanda")
        return
    items = [x for x in (raw if isinstance(raw, list) else [raw]) if not _blank(x)]
    if not items:
        rep.empty.append("addebito_o_domanda")
        return
    if len(items) > MAX_PRETESE:
        rep.fail("addebito_o_domanda", f"più di {MAX_PRETESE} addebiti ({len(items)}): letti i primi {MAX_PRETESE}",
                 raw, count=False)
        items = items[:MAX_PRETESE]
    kept: List[Dict[str, Any]] = []
    for item in items:
        got = _verify_one_addebito(rep, item, rito)
        if got is not None and not any(got["frase"] == k["frase"] and got.get("articolo") == k.get("articolo")
                                       for k in kept):
            kept.append(got)
    if not kept:
        return
    rep.frame["addebito_o_domanda"] = kept[0]
    rep.frame["addebiti"] = kept
    rep.counters["frame_pretese_count"] = len(kept)


_FRASE_ART_RE = re.compile(r"\bart(?:icol[oi]|t)?\.?\s*(\d+(?:\s*-?\s*(?:bis|ter|quater|quinquies|sexies|septies|"
                           r"octies|novies|decies))?)\b[^\n«»]{0,40}", re.IGNORECASE)


def _frase_articles(frase: Any) -> List[Dict[str, str]]:
    """v79.19.0 (item 2): the articles a charge sentence cites with their code («contestato l'art. 584 c.p.»), each
    once, in order -- what the text names, against which the reply's number is checked."""
    out: List[Dict[str, str]] = []
    for m in _FRASE_ART_RE.finditer(_flat(frase)):
        got = parse_article(m.group(0))
        if got["number"] and got["code"] and not any(x["number"] == got["number"] and x["code"] == got["code"]
                                                     for x in out):
            out.append(got)
    return out


def _parse_charge_article(rep: _Report, article_raw: Any, rito: str, part: str, raw: Any) -> Optional[Dict[str, str]]:
    """One article of the reply (part: «articolo» or «articolo_fatti»): parsed, a bare number taking the code the text
    cites it with, in a penal rito a penal provision (charge_rito_dropped); None when dropped (logged)."""
    art = parse_article(article_raw)
    penal = rito in PENAL_RITI
    if art["number"] and not art["code"]:
        cited = _cited_code(rep.text, art["number"])
        if cited:
            art = cited
        else:
            rep.fail(f"addebito_o_domanda.{part}", f"articolo senza codice: {_flat(article_raw)[:40]!r}", raw)
            return None
    elif not art["number"]:
        rep.fail(f"addebito_o_domanda.{part}", f"articolo illeggibile: {_flat(article_raw)[:40]!r}", raw)
        return None
    if penal and not is_penal(art):
        rep.counters["charge_rito_dropped"] += 1
        rep.fail(f"addebito_o_domanda.{part}", f"in un rito penale «{_flat(article_raw)[:40]}» non è una norma "
                                               f"penale", raw)
        _log("charge_rito_dropped", {"articolo": _flat(article_raw)[:80], "rito": rito})
        return None
    return art


def _article_text(art: Dict[str, str]) -> str:
    return f"{art['number']} {art['rest']}".strip()


def cited_anywhere(text: str, art: Dict[str, str]) -> bool:
    """v79.19.2 (item 2): the case text cites the article somewhere -- «art. N» with the same code, or with no code at
    all (the _cited_code pattern); a citation with another code is not this article."""
    base = re.escape(str(art.get("base") or art.get("number") or "").split("-")[0])
    if not base:
        return False
    for m in re.finditer(rf"\bart(?:icol[oi]|t)?\.?\s*{base}\b[^\n]{{0,60}}", text or "", re.IGNORECASE):
        got = parse_article(m.group(0))
        if got["number"] == art["number"] and (not got["code"] or got["code"] == art["code"]):
            return True
    return False


def _verify_one_addebito(rep: _Report, raw: Any, rito: str) -> Optional[Dict[str, Any]]:
    """One charge / claim object verified (see _verify_addebito). v79.19.0 (item 2): the reply's article number is
    checked against the article the charge sentence itself cites -- the sentence names one article with its code and
    the reply's number differs: the text's article stands (a correction: «corretto_da», frame_corrections_applied,
    logged frame_correction). (item 5) «articolo_dai_fatti»: the article the facts point to when it differs from the
    one the text names -- kept as «articolo_fatti» with «disputed» (frame_article_disputed; both go to the fit); with
    no article named by the text it is the article itself («da_fatti»)."""
    if not isinstance(raw, dict):
        rep.fail("addebito_o_domanda", "non è un oggetto", raw)
        return None
    frase = _get(raw, "frase", "perche", "perché")
    ok, quoted, reason = quote_check(rep.text, frase)
    if not ok:
        rep.fail("addebito_o_domanda", reason, raw)
        return None
    descr = _flat(_get(raw, "descrizione", "description"))
    if descr and ungrounded(descr, rep.text):
        rep.fail("addebito_o_domanda.descrizione",
                 f"parole non nel testo: {', '.join(ungrounded(descr, rep.text)[:6])}", raw)
        descr = ""
    out: Dict[str, Any] = {"articolo": None, "numero": "", "codice": "", "descrizione": descr, "frase": _flat(frase),
                           "citazione": quoted}
    article_raw = _get(raw, "articolo", "article")
    art = _parse_charge_article(rep, article_raw, rito, "articolo", raw) if not _blank(article_raw) else None
    hypothesis = None           # v79.19.2 (item 2): the reply's article the text never cites -- a hypothesis
    if art:
        named = _frase_articles(" ".join(quoted))
        if not named and not cited_anywhere(rep.text, art):
            # the charge sentence cites no article and the case text cites this number nowhere: the model's article is
            # read from memory -- the hypothesis path (a fit candidate, the charge only on a verified «si» / «dipende»)
            hypothesis, art = art, None
            rep.counters["frame_article_uncited"] = int(rep.counters.get("frame_article_uncited") or 0) + 1
            _log("frame_article_uncited", {"articolo": _article_text(hypothesis), "frase": out["frase"][:160]})
    if art:
        if len(named) == 1 and (named[0]["number"] != art["number"] or named[0]["code"] != art["code"]):
            before = _article_text(art)
            art = named[0] if not (rito in PENAL_RITI and not is_penal(named[0])) else art
            if _article_text(art) != before:
                _correct(rep, "addebito_o_domanda.articolo", before, _article_text(art),
                         "il testo cita un altro articolo nella frase dell'addebito", "read")
                out["corretto_da"] = before
        out.update({"articolo": _article_text(art), "numero": art["number"], "codice": art["code"]})
    facts_raw = _get(raw, "articolo_dai_fatti", "articolo_fatti", "article_from_facts")
    facts = _parse_charge_article(rep, facts_raw, rito, "articolo_fatti", raw) if not _blank(facts_raw) else None
    if hypothesis is not None and not facts:
        facts = hypothesis                  # v79.19.2 (item 2): the uncited article goes the facts' way
    if facts is not None:
        if facts and art and facts["number"] == art["number"] and (facts["code"] == art["code"] or not facts["code"]):
            facts = None                    # the same article: nothing disputed
        if facts and not art:
            # v79.19.1 (item 1): an article read from the facts alone is a hypothesis, never the verified charge -- it
            # goes to the fit as a candidate (alternative_charges) and becomes the charge only on a «si» / «dipende»
            # (apply_fit: frame_hypothesis_confirmed, else frame_hypothesis_dropped); «articolo» stays null
            out.update({"ipotesi": _article_text(facts), "da_fatti": True})
            _log("frame_article_hypothesis", {"ipotesi": out["ipotesi"], "frase": out["frase"][:160]})
        elif facts:
            out["articolo_fatti"] = _article_text(facts)
            out["disputed"] = True
            rep.counters["frame_article_disputed"] = int(rep.counters.get("frame_article_disputed") or 0) + 1
            _log("frame_article_disputed", {"articolo": out["articolo"], "articolo_fatti": out["articolo_fatti"],
                                            "frase": out["frase"][:160]})
    return out


def _correct(rep: _Report, field: str, old: Any, new: Any, why: str, step: str) -> None:
    """v79.19.0 (item 2): a correction of the frame at reading time, logged with the field and old -> new
    (frame_correction) and counted frame_corrections_applied; kept in the result's «corrections»."""
    entry = {"field": field, "old": _flat(old)[:200], "new": _flat(new)[:200], "why": why, "step": step}
    rep.corrections.append(entry)
    rep.counters["frame_corrections_applied"] = int(rep.counters.get("frame_corrections_applied") or 0) + 1
    logger.info("case_frame: %s corrected %r -> %r (%s)", field, entry["old"], entry["new"], why)
    _log(CORRECTION_EVENT, entry)


def _verify_istituto(rep: _Report, raw: Any) -> None:
    if _blank(raw):
        rep.empty.append("istituto")
        return
    if isinstance(raw, str):
        raw = {"testo": raw}
    if not isinstance(raw, dict):
        rep.fail("istituto", "non è un oggetto", raw)
        return
    testo, frase = _flat(_get(raw, "testo", "text", "nome")), _get(raw, "frase", "perche", "perché")
    if not testo:
        rep.fail("istituto", "istituto senza testo", raw)
        return
    ok, quoted, reason = quote_check(rep.text, frase)
    if not ok:
        rep.fail("istituto", reason, raw)
        return
    missing = ungrounded(testo, rep.text)
    if missing:
        rep.fail("istituto", f"non con le parole del testo: {', '.join(missing[:6])}", raw)
        return
    rep.frame["istituto"] = {"testo": testo, "frase": _flat(frase), "citazione": quoted}


def question_units(text: str) -> List[str]:
    """The text's questions as whole units a quesito must be (word for word, stem-tolerant): each «?»-ended clause of a
    sentence (with, and without, the lead-in before a «:»), the questions intake reads (intake.questions: the ones
    phrased as a request or a task too, also as their whole sentence). Never a statement, never a question cut
    short."""
    from . import intake
    try:
        asked = intake.questions(text or "")
    except Exception:           # noqa: BLE001 -- the «?» clauses alone
        asked = []
    out: List[str] = []
    for sentence in intake.sentences(text or ""):
        flat = _flat(sentence)
        for part in flat.split("?")[:-1]:
            part = part.strip()
            if len(part.split()) >= 2:
                out.append(part + "?")
                if ":" in part:
                    out.append(part.split(":")[-1].strip() + "?")
        if "?" not in flat and any(found_at(flat, q) is not None for q in asked):
            out.append(flat)    # a request or a task as the whole sentence («Il candidato illustri …»)
    out += asked
    return list(dict.fromkeys(x for x in out if x.strip()))


def _verify_quesiti(rep: _Report, raw: Any) -> None:
    if _blank(raw):
        rep.empty.append("quesiti")
        return
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        rep.fail("quesiti", "non è una lista", raw)
        return
    kept: List[Tuple[int, int, str]] = []
    units = question_units(rep.text)
    for n, item in enumerate(raw):
        q = _flat(_get(item, "testo", "quesito", "text") if isinstance(item, dict) else item)
        if not q:
            continue
        at = found_at(rep.text, q)
        if at is None:
            rep.fail("quesiti", f"domanda non nel testo: «{q[:160]}»", item)
            continue
        if not any(found_at(unit, q) is not None and found_at(q, unit) is not None for unit in units):
            rep.fail("quesiti", f"non è una domanda intera del testo: «{q[:160]}»", item)
            continue
        if not any(q == k[2] for k in kept):
            kept.append((at, n, q))
    rep.frame["quesiti"] = [q for _, _, q in sorted(kept)]
    if not kept and "quesiti" not in rep.unverified:
        rep.empty.append("quesiti")


def verify(raw: Any, text: str, rito_hint: str = "") -> Dict[str, Any]:
    """The reply checked against the text: {"frame", "verified", "unverified": {field: [reasons]}, "empty",
    "fallback", "counters", "log"}. rito_hint: the fallback's rito (intake's), read for the charge rule when the frame
    has none. Never raises on a malformed reply."""
    rep = _Report(str(text or ""))
    if not isinstance(raw, dict):
        rep.log.append({"field": "*", "reason": "risposta non è un oggetto JSON", "value": _flat(raw)[:300]})
        return _finish(rep, failed=True)
    get = lambda name: _get(raw, name, name.replace("_o_", "/"))
    _verify_value(rep, "rito", get("rito"))
    _verify_value(rep, "stadio", get("stadio"))
    _verify_value(rep, "ruolo", get("ruolo"))
    listed = _verify_parti(rep, get("parti"))
    if _verify_cliente(rep, get("cliente"), listed) and rep.frame["ruolo"] and not role_unknown(rep.frame):
        rep.frame["ruolo"] = None       # the reply's sides contradict the text: its role is the refused client's
        rep.fail("ruolo", "il ruolo è quello del cliente scartato (le parti della risposta contraddicono il testo)",
                 get("ruolo"))
    unknown = config().get("unknown") or "sconosciuto"
    rito = (rep.frame["rito"] or {}).get("intake") or ""
    rito = rito if rito and rito != unknown else str(rito_hint or "")     # «sconosciuto»: the fallback's rito
    _verify_addebito(rep, get("addebito_o_domanda") if get("addebito_o_domanda") is not None else
                     _get(raw, "addebito", "domanda"), rito)
    _verify_istituto(rep, get("istituto"))
    _verify_quesiti(rep, get("quesiti"))
    return _finish(rep)


def _finish(rep: _Report, failed: bool = False) -> Dict[str, Any]:
    verified = [f for f in FIELDS if rep.frame[f]]
    rep.counters["frame_fields_verified"] = len(verified)
    rep.counters["frame_failed"] = int(failed)
    empty = list(FIELDS) if failed else [f for f in FIELDS if f in rep.empty and f not in verified]
    return {"frame": rep.frame, "verified": verified, "unverified": rep.unverified, "empty": empty,
            "fallback": [f for f in FIELDS if f not in verified], "counters": rep.counters, "log": rep.log,
            "corrections": rep.corrections}


# --- the call ---------------------------------------------------------------------------------------------------------

def request_kwargs() -> Dict[str, Any]:
    """The call's bind kwargs: temperature 0, the reply's budget, thinking off (applicability.request_kwargs')."""
    return {"temperature": 0, "max_tokens": int(config().get("max_tokens") or 1500),
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}


def _default_call(messages: List[Any], kwargs: Dict[str, Any]) -> str:
    """The repo's chat model (ai_chat.chat_model, the env's LLM_* config) bound to the kwargs; the reply without its
    reasoning, the call on the request's stage meter."""
    from . import ai_chat
    ai_chat._stage_count()
    started = time.time()
    response = ai_chat.chat_model.bind(**kwargs).invoke(messages)
    ai_chat._stage_call("case_frame", started, response)
    return ai_chat._reply_text(response, kwargs.get("max_tokens"))


def build_messages(text: str) -> List[Any]:
    """[system, human]: the prompt, the request's full text (cut at «max_text_chars», logged)."""
    from langchain_core.messages import HumanMessage, SystemMessage
    body = str(text or "")
    cap = int(config().get("max_text_chars") or 40000)
    if len(body) > cap:
        logger.info("case_frame: the request cut at %d characters in the prompt (%d)", cap, len(body))
        body = body[:cap] + " […]"
    return [SystemMessage(content=prompt_system()), HumanMessage(content=f"TESTO DELLA RICHIESTA:\n{body}")]


def parse_reply(raw: Any) -> Dict[str, Any]:
    """The reply's frame object; ValueError when it holds no JSON object."""
    from .llm_reply import parse_json_reply
    data = parse_json_reply(raw if isinstance(raw, str) else str(raw or ""))
    if isinstance(data, dict) and isinstance(data.get("quadro"), dict):
        data = data["quadro"]
    if not isinstance(data, dict):
        raise ValueError("no JSON object in the reply")
    return data


def read(text: str, call: Optional[Callable[[List[Any], Dict[str, Any]], str]] = None,
         rito_hint: str = "") -> Dict[str, Any]:
    """The step: the call (within «deadline_s»), then verify. A failed call, a reply that is no JSON object, a call not
    back in time: every field empty and to the fallback (frame_failed), logged; never raises."""
    t0 = time.time()
    status, error, raw = OK, "", None
    if not str(text or "").strip():
        status = NO_TEXT
    else:
        deadline = float(config().get("deadline_s") or 15)
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="case_frame")
        try:
            fut = pool.submit(contextvars.copy_context().run, call or _default_call, build_messages(text),
                              request_kwargs())
            reply = fut.result(timeout=deadline)
            raw = parse_reply(reply)
        except _Timeout:
            status, error = TIMEOUT, f"oltre {deadline} s"
        except ValueError as exc:
            status, error = MALFORMED, str(exc)[:300]
        except Exception as exc:    # noqa: BLE001 -- every field falls back
            status, error = FAILED, f"{type(exc).__name__}: {exc}"[:300]
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
    if status == OK:
        result = verify(raw, text, rito_hint)
    else:
        result = verify(None, text, rito_hint)
        result["counters"]["frame_failed"] = int(status != NO_TEXT)
        if status != NO_TEXT:
            logger.warning("case_frame: %s: %s (every field to the fallback)", status, error)
    seconds = round(time.time() - t0, 3)
    result["counters"]["frame_seconds"] = seconds
    result.update({"status": status, "error": error, "seconds": seconds, "prompt_hash": prompt_hash()})
    _log("case_frame_read", {"prompt_hash": result["prompt_hash"], "status": status, "error": error,
                             "seconds": seconds, "verified": result["verified"], "fallback": result["fallback"],
                             "counters": result["counters"]})
    return result


# --- what the pipeline reads (build B) --------------------------------------------------------------------------------

def _frame(result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(result, dict):
        return {}
    return result.get("frame") if isinstance(result.get("frame"), dict) else result


def fallback_fields(result: Optional[Dict[str, Any]]) -> List[str]:
    """The fields the rules fill: those the frame leaves empty or that failed verification (every one with no frame)."""
    if not isinstance(result, dict) or not isinstance(result.get("frame"), dict):
        return list(FIELDS)
    frame = result["frame"]
    return [f for f in FIELDS if not frame.get(f)]


def count_fallback(counters: Dict[str, Any], field: str) -> None:
    """A rule filled the field the frame left to it: frame_fallback.<field> += 1 (build B, where the rule runs)."""
    if field in FIELDS:
        key = f"frame_fallback.{field}"
        counters[key] = int(counters.get(key) or 0) + 1


def role_unknown(result: Optional[Dict[str, Any]]) -> bool:
    """The frame's verified role is «sconosciuto»: the honesty line prints and neither side's wording is used."""
    role = _frame(result).get("ruolo") or {}
    return bool(role) and role.get("valore") == (config().get("unknown") or "sconosciuto")


def intake_view(result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The verified frame in intake's words -- {client, role, rito, stage, charge: {articolo, numero, codice,
    descrizione, frase}, institution, questions, role_unknown}; None (questions: []) where the rules decide."""
    f = _frame(result)
    pick = lambda k: (f.get(k) or {}).get("intake")
    return {"client": (f.get("cliente") or {}).get("nome"), "role": pick("ruolo"), "rito": pick("rito"),
            "stage": pick("stadio"), "charge": dict(f["addebito_o_domanda"]) if f.get("addebito_o_domanda") else None,
            "institution": (f.get("istituto") or {}).get("testo"), "questions": list(f.get("quesiti") or []),
            "role_unknown": role_unknown(result)}


def fit_context(result: Optional[Dict[str, Any]]) -> str:
    """The verified frame as the 79.13 applicability judgment's context (labelled «solo contesto»); "" when there is
    no verified field to give."""
    f = _frame(result)
    labels = config().get("fit_context") or {}
    lines = []
    if f.get("cliente"):
        lines.append(f"{labels.get('cliente', 'Cliente')}: {f['cliente']['nome']}")
    for k in ("ruolo", "rito", "stadio"):
        if f.get(k):
            lines.append(f"{labels.get(k, k)}: {f[k]['valore']}")
    if f.get("addebito_o_domanda"):
        a = f["addebito_o_domanda"]
        what = "; ".join(x for x in (a.get("articolo") or "", a.get("descrizione") or "") if x) or a.get("frase")
        lines.append(f"{labels.get('addebito_o_domanda', 'Addebito o domanda')}: {what}")
    if f.get("istituto"):
        lines.append(f"{labels.get('istituto', 'Istituto')}: {f['istituto']['testo']}")
    if not lines:
        return ""
    return "\n".join([labels.get("title", "QUADRO DEL CASO:")] + lines)


def printable(result: Optional[Dict[str, Any]]) -> List[str]:
    """The frame as the nightly prints it, one line per field (verified value, else «—» with why it fell back)."""
    f = _frame(result)
    unverified = (result or {}).get("unverified") or {}
    out = []
    for k in FIELDS:
        v = f.get(k)
        if k == "parti":
            shown = "; ".join(f"{p['nome']} ({p['lato']}" + (f", {p['ruolo_nel_fatto']}" if p.get("ruolo_nel_fatto")
                                                                  else "") + ")" for p in v or [])
        elif k == "cliente":
            shown = f"{v['nome']} — {v['perche']}" if v else ""
        elif k in VALUE_FIELDS:
            shown = f"{v['valore']} (intake {v['intake']}) — {v['perche']}" if v else ""
        elif k == "addebito_o_domanda":
            shown = (f"{v.get('articolo') or 'articolo —'}; {v.get('descrizione') or '—'} — {v['frase']}") if v else ""
        elif k == "istituto":
            shown = f"{v['testo']} — {v['frase']}" if v else ""
        else:
            shown = " | ".join(v or [])
        why = "; ".join(unverified.get(k) or [])
        out.append(f"{k}: {shown or '—'}" + (f" [scartato: {why}]" if why else ""))
    return out


# --- the binding (build B): the verified frame, the single source of its fields for the whole pipeline -----------------

FRAME_HOW = "frame"             # intake's confidence for a field the frame decided (beside rule | model | default)
NO_CLIENT_HOW = "nessuno (quadro del caso)"     # state["client"]["how"]: the frame leaves no client to assume
BOUND_FIELDS = ("cliente", "ruolo", "rito", "stadio", "addebito_o_domanda", "istituto", "quesiti")
AUDIT_COUNTERS = ("frame_conflict", "frame_conflict_attempts", "role_flip", "charge_rito_mismatch")


def held(holder: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The request's frame the holder carries -- v79.19.0 (item 2): the settled one first (state["frame"]: the read's
    result with the corrections applied, written once by settle -- every later step reads it and nothing else), else
    the read's result (state["case_frame"], else its intake's, or an intake result's «case_frame») -- with CASE_FRAME
    on; None otherwise (the flag off: nothing binds)."""
    if not isinstance(holder, dict) or not enabled():
        return None
    settled = holder.get("frame")
    if isinstance(settled, dict) and settled.get(SETTLED) and isinstance(settled.get("frame"), dict):
        return settled
    for got in (holder.get("case_frame"), (holder.get("intake") or {}).get("case_frame")
                if isinstance(holder.get("intake"), dict) else None):
        if isinstance(got, dict) and isinstance(got.get("frame"), dict):
            return got
    return None


# --- v79.19.0 (item 2): one source of truth -- state["frame"], the corrected frame written once ------------------------
# After the frame's read (verify: its own corrections, «corrections») and the correction passes the pipeline applies
# (the rules' finer role word -- intake._refine_frame_role --, a penal posture refusing a non-penal charge --
# settle_posture --, the applicability judgment settling a disputed article -- apply_fit), the corrected values are in
# state["frame"] (settle: a copy of the read's result, SETTLED); held returns it first, so every consumer (bound,
# charge_label, intake_view, the line choice, the decomposition, the rites, the procedibilità, §8, the writer) reads
# the same values. Each correction: {field, old, new, why, step}, logged frame_correction, counted
# frame_corrections_applied (audit).

def settle(holder: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """state["frame"] from state["case_frame"] (the read's result): a deep copy, SETTLED, its corrections those of the
    reading plus the role refinement intake recorded («refined_from»); each charge / claim gets its records' label
    («label», charge_label). Idempotent: an already settled frame is returned as it is. None with no frame."""
    import copy
    if not isinstance(holder, dict) or not enabled():
        return None
    have = holder.get("frame")
    if isinstance(have, dict) and have.get(SETTLED) and isinstance(have.get("frame"), dict):
        return have
    got = holder.get("case_frame")
    if not isinstance(got, dict) or not isinstance(got.get("frame"), dict):
        return None
    out = copy.deepcopy(got)
    out[SETTLED] = True
    out["corrections"] = list(got.get("corrections") or [])
    f = out["frame"]
    role = f.get("ruolo") or {}
    if role.get("refined_from") and role.get("refined_from") != role.get("intake"):
        _settle_correction(out, "ruolo.intake", role["refined_from"], role["intake"],
                           "la parola più precisa delle regole per lo stesso lato", "intake")
    for a in _addebiti(f):
        a["label"] = charge_label({"frame": {"addebito_o_domanda": a}}) if a.get("articolo") else ""
        if a.get("articolo_fatti"):
            got_f = _ap_label(a["articolo_fatti"])
            a["label_fatti"] = got_f
        if a.get("ipotesi"):            # v79.19.1 (item 1): the facts' hypothesis, a fit candidate
            a["label_ipotesi"] = _ap_label(a["ipotesi"])
    holder["frame"] = out
    _log("frame_settled", {"corrections": out["corrections"], "addebiti": [a.get("label") for a in _addebiti(f)],
                           "disputed": [a.get("label_fatti") for a in _addebiti(f) if a.get("disputed")]})
    return out


def _ap_label(articolo: Any) -> str:
    from . import posture
    got = posture.sentence_articles(f"art. {_flat(articolo)}")
    return got[0] if got else ""


def _addebiti(f: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The frame's charges / claims in order (v79.19.0 item 3: «addebiti», else the one «addebito_o_domanda»)."""
    many = f.get("addebiti")
    if isinstance(many, list) and many:
        return [a for a in many if isinstance(a, dict)]
    one = f.get("addebito_o_domanda")
    return [one] if isinstance(one, dict) else []


def _settle_correction(out: Dict[str, Any], field: str, old: Any, new: Any, why: str, step: str) -> Dict[str, Any]:
    entry = {"field": field, "old": _flat(old)[:200], "new": _flat(new)[:200], "why": why, "step": step}
    out.setdefault("corrections", []).append(entry)
    counters = out.setdefault("counters", {})
    counters["frame_corrections_applied"] = int(counters.get("frame_corrections_applied") or 0) + 1
    logger.info("case_frame: %s corrected %r -> %r (%s, %s)", field, entry["old"], entry["new"], why, step)
    _log(CORRECTION_EVENT, entry)
    return entry


def settle_posture(holder: Dict[str, Any], posture: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The posture's correction pass, once: in a penal posture a charge whose article is no penal provision loses its
    article (the frame had no rito to check it against; charge_rito_dropped, holder «frame_charge_refused» for the
    audit) -- a correction old -> "". The corrections made."""
    out = settle(holder)
    if out is None or out.get("posture_checked"):
        return []
    out["posture_checked"] = True
    made = []
    if not _penal_posture(posture if posture is not None else holder.get("posture")):
        return made
    for a in _addebiti(out["frame"]):
        for key, label_key in (("articolo", "label"), ("articolo_fatti", "label_fatti"), ("ipotesi", "label_ipotesi")):
            label = a.get(label_key) or ""
            if label and not is_penal(parse_article(label)):
                made.append(_settle_correction(out, f"addebito_o_domanda.{key}", label, "",
                                               "in una posizione penale non è una norma penale", "posture"))
                a[f"refused_{key}"] = label
                a[key], a[label_key] = None, ""
                if key == "ipotesi":            # v79.19.1 (item 1): a non-penal hypothesis in a penal posture: dropped
                    a["ipotesi_resolved"] = "scartata"
                    counters = out.setdefault("counters", {})
                    counters["frame_hypothesis_dropped"] = int(counters.get("frame_hypothesis_dropped") or 0) + 1
                elif key == "articolo":
                    a["numero"], a["codice"] = "", ""
                    if holder.get("frame_charge_refused") != label:     # the audit counts it charge_rito_dropped
                        holder["frame_charge_refused"] = label
                    _log("charge_rito_dropped", {"articolo": label, "rito": "penale (posture)", "where": "settle"})
                else:
                    a.pop("disputed", None)
    return made


def apply_fit(holder: Dict[str, Any], judgments: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """v79.19.0 (item 5): the applicability judgment settles a disputed article, once -- the text's article judged «no»
    (verified) and the facts' one «si» or «dipende»: the facts' article is the charge (a correction old -> new, why
    the judgment); the text's article judged «si» / «dipende»: it stands, the facts' one stays an alternative
    (alternative_charges). Neither verified: nothing changes (the text's article stands, logged). The corrections
    made."""
    out = settle(holder)
    if out is None or out.get("fit_checked"):
        return []
    out["fit_checked"] = True
    verdict: Dict[str, str] = {}
    for j in judgments or []:
        if isinstance(j, dict) and j.get("verified") and j.get("article") and j.get("fit"):
            verdict[_norm(j["article"])] = str(j["fit"])
    made = []
    for a in _addebiti(out["frame"]):
        if not a.get("disputed") or not a.get("label_fatti"):
            continue
        text_v, facts_v = verdict.get(_norm(a.get("label") or "")), verdict.get(_norm(a["label_fatti"]))
        if text_v == "no" and facts_v in ("si", "dipende"):
            made.append(_settle_correction(out, "addebito_o_domanda.articolo", a.get("label"), a["label_fatti"],
                                           f"giudizio di applicabilità: l'articolo citato nel testo non governa i "
                                           f"fatti («no»), quello letto dai fatti «{facts_v}»", "fit"))
            a["disputed_resolved"] = "fatti"
            a["articolo_testo"], a["label_testo"] = a.get("articolo"), a.get("label")
            a["articolo"], a["label"] = a["articolo_fatti"], a["label_fatti"]
            parsed = parse_article(a["articolo"])
            a["numero"], a["codice"] = parsed["number"], parsed["code"]
        else:
            a["disputed_resolved"] = "testo"
            _log("frame_article_disputed_kept", {"articolo": a.get("label"), "articolo_fatti": a["label_fatti"],
                                                 "fit_testo": text_v, "fit_fatti": facts_v})
    # v79.19.1 (item 1): a hypothesis (an article read from the facts, none named by the text) becomes the charge
    # only on a verified «si» / «dipende» (a correction "" -> the article, frame_hypothesis_confirmed); anything else
    # -- «no», unverified, not judged -- drops it (frame_hypothesis_dropped): never the charge from memory
    counters = out.setdefault("counters", {})
    for a in _addebiti(out["frame"]):
        if not a.get("ipotesi") or not a.get("label_ipotesi") or a.get("ipotesi_resolved"):
            continue
        hyp_v = verdict.get(_norm(a["label_ipotesi"]))
        if hyp_v in ("si", "dipende"):
            made.append(_settle_correction(out, "addebito_o_domanda.articolo", "", a["label_ipotesi"],
                                           f"giudizio di applicabilità: nessun articolo nel testo, quello letto dai "
                                           f"fatti «{hyp_v}»", "fit"))
            a["ipotesi_resolved"] = "confermata"
            a["articolo"], a["label"] = a["ipotesi"], a["label_ipotesi"]
            parsed = parse_article(a["articolo"])
            a["numero"], a["codice"] = parsed["number"], parsed["code"]
            counters["frame_hypothesis_confirmed"] = int(counters.get("frame_hypothesis_confirmed") or 0) + 1
        else:
            a["ipotesi_resolved"] = "scartata"
            counters["frame_hypothesis_dropped"] = int(counters.get("frame_hypothesis_dropped") or 0) + 1
            _log("frame_hypothesis_dropped", {"ipotesi": a.get("label_ipotesi"), "fit": hyp_v,
                                              "frase": str(a.get("frase") or "")[:160]})
    return made


def _norm(label: Any) -> str:
    try:
        return _es().norm_label(str(label or ""))
    except Exception:           # noqa: BLE001
        return " ".join(str(label or "").lower().split())


def charges(holder: Optional[Dict[str, Any]]) -> List[str]:
    """v79.19.0 (item 3): the frame's charge / claim labels in order (the first the principale line's), "" entries
    out; [] with no frame or no article."""
    got = held(holder)
    if got is None:
        return []
    out = []
    for a in _addebiti(got["frame"]):
        label = a.get("label") if "label" in a else (charge_label({"frame": {"addebito_o_domanda": a}})
                                                    if a.get("articolo") else "")
        if label and label not in out:
            out.append(label)
    return out


def alternative_charges(holder: Optional[Dict[str, Any]]) -> List[str]:
    """v79.19.0 (items 3, 5): the labels that become alternative lines -- the charges after the first, and a disputed
    article not chosen as the charge (the facts' one while the text's stands, the text's once the judgment chose the
    facts')."""
    got = held(holder)
    if got is None:
        return []
    out: List[str] = []
    first = charges(holder)[:1]
    for n, a in enumerate(_addebiti(got["frame"])):
        label = a.get("label") or ""
        if n and label and label not in first and label not in out:
            out.append(label)
        other = a.get("label_testo") if a.get("disputed_resolved") == "fatti" else a.get("label_fatti")
        if a.get("disputed") and other and other not in first and other not in out:
            out.append(other)
        hyp = a.get("label_ipotesi") or ""          # v79.19.1 (item 1): the facts' hypothesis, a fit candidate until
        if hyp and not a.get("ipotesi_resolved") and hyp not in first and hyp not in out:      # the judgment decides
            out.append(hyp)
    return out


def hypotheses(holder: Optional[Dict[str, Any]]) -> List[Dict[str, str]]:
    """v79.19.1 (item 1): the frame's articles read from the facts alone -- [{label, stato: aperta | confermata |
    scartata}]; [] with no frame or none."""
    got = held(holder)
    if got is None:
        return []
    return [{"label": a.get("label_ipotesi") or _ap_label(a.get("ipotesi") or a.get("refused_ipotesi")),
             "stato": a.get("ipotesi_resolved") or "aperta"}
            for a in _addebiti(got["frame"]) if a.get("ipotesi") or a.get("refused_ipotesi")]


def disputed(holder: Optional[Dict[str, Any]]) -> bool:
    """v79.19.0 (item 5): the frame's charge carries a disputed article the judgment has not settled yet."""
    got = held(holder)
    if got is None:
        return False
    return any(a.get("disputed") and not a.get("disputed_resolved") for a in _addebiti(got["frame"]))


def role_deduced(holder: Optional[Dict[str, Any]]) -> bool:
    """v79.19.0 (item 4): the held frame's role was deduced from the facts, not stated."""
    got = held(holder)
    return got is not None and bool((got["frame"].get("ruolo") or {}).get(DEDUCED))


def corrections(holder: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    got = held(holder)
    return list((got or {}).get("corrections") or [])


def charge_label(result: Optional[Dict[str, Any]]) -> str:
    """The frame's charge article as the records label it («640 c.p.», «2634 c.c.», «73 d.P.R. 309/1990»); "" without
    one (no charge, or its article dropped)."""
    charge = _frame(result).get("addebito_o_domanda") or {}
    if not charge.get("articolo"):
        return ""
    from . import posture
    got = posture.sentence_articles(f"art. {charge['articolo']}")
    return got[0] if got else ""


def _penal_posture(posture: Optional[Dict[str, Any]]) -> bool:
    p = posture if isinstance(posture, dict) else {}
    return p.get("rito") in PENAL_RITI or p.get("intake_rito") in PENAL_RITI or p.get("family") == "penale"


def bound(holder: Optional[Dict[str, Any]], field: str, posture: Optional[Dict[str, Any]] = None) -> Any:
    """The frame's value of a field in the pipeline's words, None where the rules decide (no frame, the field empty or
    dropped): cliente -> the name; ruolo -> intake's role («sconosciuto» too: the honesty line); rito / stadio ->
    intake's value (a verified «sconosciuto» binds nothing: the text's silence, the rules read it); addebito_o_domanda
    -> the charge's label (None without an article; None too when the posture -- posture, else the holder's -- is
    penal and the article is no penal provision: the frame had no rito to check it against, the posture's decides --
    refused, logged charge_rito_dropped, the rules' charge fills it); istituto -> its words; quesiti -> the list."""
    got = held(holder)
    if got is None:
        return None
    f = got["frame"]
    view = intake_view(got)
    unknown = config().get("unknown") or "sconosciuto"
    if field == "cliente":
        return view["client"] or None
    if field == "ruolo":
        return view["role"] or None
    if field in ("rito", "stadio"):
        v = view["rito" if field == "rito" else "stage"]
        return v if v and v != unknown else None
    if field == "addebito_o_domanda":
        # v79.19.0 (item 2): the penal posture's refusal is a correction pass (settle_posture), applied once into the
        # settled frame -- the first read with a posture at hand applies it; an unsettled frame's read only refuses
        # (no side effect)
        if got.get(SETTLED) and not got.get("posture_checked") and (posture is not None or holder.get("posture")):
            settle_posture(holder, posture)
            got = held(holder) or got
        label = charge_label(got)
        if label and not got.get(SETTLED) and _penal_posture(posture if posture is not None
                                                             else holder.get("posture")) and \
                not is_penal(parse_article(label)):
            return None
        return label or None
    if field == "istituto":
        return view["institution"] or None
    if field == "quesiti":
        return list(f.get("quesiti") or []) or None
    return None


def same_value(field: str, a: Any, b: Any) -> bool:
    """Two values of a field agree: one party by its name's words (cliente), the same text (case and spaces aside)."""
    if field == "cliente":
        return same_party(a, b) or _fold(a) == _fold(b)
    if field == "quesiti":
        return [_fold(x) for x in a or []] == [_fold(x) for x in b or []]
    return _fold(a) == _fold(b)


def conflict(holder: Dict[str, Any], field: str, step: str, proposed: Any, value: Any) -> None:
    """A later step disagreeing with a frame field: logged (frame_conflict, with the step's name), kept once per field,
    step and value in holder["frame_conflicts"] -- the step loses."""
    entry = {"field": field, "step": step, "proposed": _flat(proposed)[:200], "frame": _flat(value)[:200]}
    seen = holder.setdefault("frame_conflicts", []) if isinstance(holder, dict) else []
    if entry in seen:
        return
    seen.append(entry)
    logger.info("case_frame: %s disagrees on %s (%r, the frame %r): the frame wins", step, field, entry["proposed"],
                entry["frame"])
    _log("frame_conflict", entry)


def defer(holder: Optional[Dict[str, Any]], field: str, step: str, proposed: Any,
          log_to: Optional[Dict[str, Any]] = None) -> Any:
    """A later step's value for a frame field: the frame's whenever it holds one -- a different, non-empty proposal is
    logged frame_conflict (the step's name; into log_to, else the holder) and loses; the step's own otherwise."""
    value = bound(holder, field)
    if value is None:
        return proposed
    if proposed not in (None, "", []) and not same_value(field, proposed, value):
        conflict(log_to if isinstance(log_to, dict) else holder, field, step, proposed, value)
    return value


def why_of(result: Optional[Dict[str, Any]], field: str) -> str:
    """intake's «why» for a field the frame decided: «quadro del caso: «the quote»»."""
    v = _frame(result).get(field) or {}
    quote = (v.get("citazione") or [""])[0] if isinstance(v, dict) else ""
    return f"quadro del caso: «{_flat(quote or (v.get('perche') if isinstance(v, dict) else ''))[:160]}»"


def side_of_role(role: Any) -> str:
    """A role's side in fact_signals.client_side's words (intake.side_of: accused | victim | terzo | "")."""
    from . import intake
    return intake.side_of(str(role or ""))


def position(holder: Optional[Dict[str, Any]], posizione: Any, step: str) -> Optional[str]:
    """The client's «posizione» (the extraction's, the plan's) under the frame: None when the frame binds no role (the
    step's own stands); "" for a verified «sconosciuto» role (neither side's wording); the frame role's label when the
    position reads the other side (logged frame_conflict with the step's name: it loses); the position itself when it
    agrees or reads no side."""
    role = bound(holder, "ruolo")
    if role is None:
        return None
    if role_unknown_held(holder):
        if _flat(posizione):
            conflict(holder, "ruolo", step, posizione, role)
        return ""
    from . import fact_signals, posture
    have = fact_signals.client_side(_flat(posizione)) if _flat(posizione) else ""
    want = side_of_role(role)
    if not have or have == want:
        return _flat(posizione) if _flat(posizione) else posture.role_label(role, bound(holder, "stadio") or "")
    conflict(holder, "ruolo", step, posizione, role)
    return posture.role_label(role, bound(holder, "stadio") or "") or role


def no_client(holder: Optional[Dict[str, Any]]) -> bool:
    """The frame left no client to assume (settle_client's honesty line under the frame: NO_CLIENT_HOW)."""
    return held(holder) is not None and (holder.get("client") or {}).get("how") == NO_CLIENT_HOW


def role_unknown_held(holder: Optional[Dict[str, Any]]) -> bool:
    """The held frame's verified role is «sconosciuto» (role_unknown): the honesty line, neither side's wording."""
    got = held(holder)
    return got is not None and role_unknown(got)


def counterparty_names(result: Optional[Dict[str, Any]]) -> List[str]:
    """The verified parties the frame lists as controparte."""
    return [p["nome"] for p in _frame(result).get("parti") or [] if p.get("lato") == "controparte"]


def _delivered(state: Dict[str, Any]) -> Dict[str, Any]:
    """What the answer carried for each bound field: the settled client, the posture's role, rito and stage, the held
    charge, the first read's frame institution (module activation's first institution), the intake's questions."""
    posture = state.get("posture") or {}
    got = state.get("intake") or {}
    client = (state.get("client") or {}).get("name") or (state.get("answered") or {}).get("parte_assistita") or \
        (got.get("client") or {}).get("name") or ""
    read = state.get("read") if isinstance(state.get("read"), dict) else {}
    institution = ((read or {}).get("frame_institution") or {}).get("testo") or ""
    if not institution and read:
        from .interactive_flow import read_institutions
        first = read_institutions(read)
        institution = first[0]["name"] if first else ""
    return {"cliente": client, "ruolo": posture.get("role") or got.get("role") or "",
            "rito": posture.get("intake_rito") or posture.get("rito") or got.get("rito") or "",
            "stadio": posture.get("stage") or got.get("stage") or "",
            "addebito_o_domanda": (state.get("charge") or {}).get("article") or "",
            "istituto": institution, "quesiti": list(got.get("questions") or []), "parti": _parties(state)}


def _parties(state: Dict[str, Any]) -> List[str]:
    """The parties the analysis extracted (the rules' fallback for «parti»)."""
    try:
        from .interactive_flow import _facts, _restored_analysis
        facts = _facts(_restored_analysis(state["analysis"])) if isinstance(state.get("analysis"), dict) else {}
        return [str(p["name"]) for p in facts.get("parties") or [] if isinstance(p, dict) and p.get("name")]
    except Exception:           # noqa: BLE001 -- a counter never fails the answer
        return []


def audit(state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The frame's counters on the delivered answer ({} with no frame held): the read's (frame_fields_verified,
    frame_unverified.*, frame_failed, frame_seconds, charge_rito_dropped), and --
      frame_fallback.<field>   1 where the frame binds nothing for the field (empty, dropped, a charge with its
                               article dropped, a rito or stage «sconosciuto») and the rules filled it;
      frame_conflict           the bound fields the delivered answer carries changed (must be 0);
      frame_conflict_attempts  the later steps that disagreed and lost (logged with their names);
      role_flip                intake's (the frame's) role is not the method's (the posture's), or the client the
                               answer serves is a party the frame lists as controparte (must be 0); v79.14.1: with
                               item 1's reasons (state["role_flip_reasons"]) as one set -- each distinct flip once;
      charge_rito_mismatch     a penal rito whose held charge is no penal provision (must be 0).
    «filled»: each fallback field with the value the rules gave it (the nightly prints it)."""
    got = held(state)
    if got is None:
        return {}
    out: Dict[str, Any] = dict(got.get("counters") or {})
    delivered = _delivered(state)
    changed, filled = [], {}
    for field in BOUND_FIELDS:
        value = bound(state, field)
        have = delivered.get(field)
        if value is not None:
            if field == "addebito_o_domanda" and "charge" not in state:
                continue                # no charge step ran (no records): nothing delivered, nothing changed
            if field == "istituto" and not state.get("read"):
                continue                # no first read: no institution delivered
            if not same_value(field, have, value):
                changed.append({"field": field, "frame": value, "delivered": have})
    left = [f for f in FIELDS if (f in fallback_fields(got) if f == "parti" else bound(state, f) is None)]
    for field in left:                  # the rules decide: empty or dropped, an article dropped, a «sconosciuto»
        have = delivered.get(field)     # rito or stage (binding nothing)
        out[f"frame_fallback.{field}"] = int(bool(have))
        if have:
            filled[field] = have
    conflicts = []
    for c in list(got.get("frame_conflicts") or []) + list(state.get("frame_conflicts") or []) + \
            list((state.get("intake") or {}).get("frame_conflicts") or []):
        if c not in conflicts:
            conflicts.append(c)
    if state.get("frame_charge_refused"):     # the frame's charge refused at binding (a penal posture)
        out["charge_rito_dropped"] = int(out.get("charge_rito_dropped") or 0) + 1
    out["frame_conflict"] = len(changed)
    out["frame_conflict_attempts"] = len({(c.get("field"), c.get("step"), c.get("proposed")) for c in conflicts})
    known = lambda r: r not in ("", None, config().get("unknown") or "sconosciuto")
    first = bound(state, "ruolo") or (state.get("intake") or {}).get("role") or ""
    method = (state.get("posture") or {}).get("role") or ""
    try:                                # what mark_sides and the sentence sides read (client_of)
        from .interactive_flow import client_of
        read_by = client_of(state)
    except Exception:                   # noqa: BLE001 -- a counter never fails the answer
        read_by = ""
    served = [delivered["cliente"], read_by, str(state.get("plan_client") or ""),       # the plan's (frame_plan)
              str((state.get("analysis") or {}).get("parte_assistita") or "")]
    taken_names = [n for n in counterparty_names(got) if any(x and same_party(x, n) for x in served)]
    taken = bool(taken_names)
    # v79.14.1: ONE role_flip -- this audit's reasons and item 1's (interactive_flow.role_flip_reasons, kept in
    # state["role_flip_reasons"]: the method's role not the intake's, a client's in multi-client, the counterparty
    # taken as client) as one set of distinct flips: the same flip read by both definitions counts once
    from .interactive_flow import ROLE_FLIP_COUNTERPARTY, ROLE_FLIP_ROLE
    reasons = list(state.get("role_flip_reasons") or [])

    def of_client(reason: str, kind: str) -> str:      # «ruolo:<client>» / «controparte:<client>» -> the client
        return reason.split(":", 1)[1] if reason.startswith(kind + ":") else ""
    if known(first) and known(method) and first != method:
        # the posture's flip carried to a client of a multi-client answer (its method role the posture's): one flip
        roles = {str(m.get("client")): str(m.get("role")) for m in state.get("method_roles") or []}
        reasons = [r for r in reasons if not (of_client(r, ROLE_FLIP_ROLE) and
                                              roles.get(of_client(r, ROLE_FLIP_ROLE)) == method)]
        reasons.append(ROLE_FLIP_ROLE)
    for name in taken_names:            # a counterparty served: once, item 1's per-client reason when it names it
        if not any(same_party(of_client(r, ROLE_FLIP_COUNTERPARTY), name) for r in reasons
                   if of_client(r, ROLE_FLIP_COUNTERPARTY)):
            reasons.append(ROLE_FLIP_COUNTERPARTY)
    reasons = list(dict.fromkeys(reasons))
    out["role_flip"] = len(reasons)
    posture = state.get("posture") or {}
    rito = bound(state, "rito") or posture.get("intake_rito") or posture.get("rito") or ""
    charge = delivered["addebito_o_domanda"]
    out["charge_rito_mismatch"] = int(bool(charge) and (rito in PENAL_RITI or posture.get("family") == "penale")
                                      and not is_penal(parse_article(charge)))
    # v79.19.0: the corrections applied to the settled frame (item 2), the charges read (item 3), the disputed
    # article (item 5), the deduced role (item 4)
    made = list(got.get("corrections") or [])
    out["frame_corrections_applied"] = len(made)
    out["frame_pretese_count"] = len(_addebiti(got["frame"]))
    out["frame_article_disputed"] = int(any(a.get("disputed") for a in _addebiti(got["frame"])))
    out["frame_role_deduced"] = int(role_deduced(state))
    out["frame_audit"] = {"changed": changed, "filled": filled, "conflicts": conflicts,
                          "role": {"frame_or_intake": first, "method": method, "client_is_counterparty": taken,
                                   "role_flip_reasons": reasons},
                          "corrections": made, "charges": charges(state), "alternative_charges":
                          alternative_charges(state)}
    return out


def printable_delivered(result: Optional[Dict[str, Any]], filled: Optional[Dict[str, Any]] = None) -> List[str]:
    """printable, each field the frame left to the rules with the value they filled it with («fallback: …»)."""
    lines = printable(result)
    filled = filled or {}
    out = []
    for k, line in zip(FIELDS, lines):
        if k in filled:
            v = filled[k]
            shown = " | ".join(str(x) for x in v) if isinstance(v, list) else str(v)
            if line.startswith(f"{k}: —"):
                line = line.replace(f"{k}: —", f"{k}: — (fallback: {shown[:200]})", 1)
            elif k == "addebito_o_domanda" and line.startswith(f"{k}: articolo —"):     # the article alone dropped
                line = line.replace("articolo —", f"articolo — (fallback: {shown[:200]})", 1)
        out.append(line)
    return out
