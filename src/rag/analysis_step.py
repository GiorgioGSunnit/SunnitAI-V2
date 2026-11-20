"""v79.15.0 (item 1): the verified analysis step -- one model call between the case frame and rendering decides the
leading and the subordinate line and every schema element's esito; the code verifies every citation before anything
counts. The old engine (lawyer_method's schema matcher -> thesis generator -> score) no longer decides when the flag
is on: build B (the rendering section at the end of this module; interactive_flow.analysis_step_run,
lawyer_method.compose / answers_block / legacy_plan) prints §1, §4's element tables, §6, §9, every «x su y» count and
the answers' line from the verified analysis only, and logs the old engine's plan («legacy_plan») without printing it.

Behind ANALYSIS_STEP (default off: run() returns {"status": "off"}, nothing is called, no output changes). The prompt
and the word lists are data/analysis_step.json (DATA_FILE); the remedy / general-part set is data/remedy_articles.json
(REMEDY_FILE). The prompt is hashed (prompt_hash) and the hash logged with every run.

The input (make_input; inputs(state) reads it from the flow's state):
  - the verified frame -- only its verified fields (case_frame.held's «frame»: a field that failed verification is not
    in it): cliente, ruolo, rito, stadio, addebito_o_domanda (articolo, descrizione), istituto, parti (nome, lato,
    ruolo_nel_fatto);
  - the fitted norms -- the records left after applicability_fit (not dropped, not removed by a verified «no»), each
    with its rubrica, its schema elements (the record's requisites: what §4 tables; ELEMENT_SCHEMAS on, the schema's
    own) and its text as numbered periods (applicability.article_ids -- the fit's own numbering, one splitter: the
    commi, each cut at a full stop before a capital or an opening mark, abbreviations never ending one, numbered
    1..n across the article: «A640cp.1»); a norm of the remedy set marked so in the prompt;
  - the case's sentences numbered S1..Sn exactly as the flow numbers them (issue_records.sentences, the state's
    «sentences»: the case, then the client's precisazioni, then the system's assumptions) -- the system's assumptions
    (source «sistema») never sent and never valid ids (an assumption is no fact); their numbers unchanged;
  - the user's questions (the frame's quesiti, else interactive_flow.asked_questions).

The call (call_once): thinking off, temperature 0, a JSON reply (applicability.request_kwargs' body, this step's
max_tokens), on a worker thread waited at most the step's remaining budget -- «deadline_s» (40 s) from the step's
start, the re-ask included (the applicability batching/deadline pattern: a call not back in time is abandoned, its
pool shut down without waiting). At most 2 calls: the first, and one re-ask when the first's verification rejects a
line (any reason below), leaves out the leading line, or misses the frame's charge -- the one re-ask carries every
reason, one per line («reask»). The first call failed, not back in time or no JSON object: status failed / timeout /
malformed, analysis None -- nothing from the step is printed. The re-ask failing: the first reply's verification
stands, with the fallbacks.

The verification (verify, pure code, no model):
  - every S-id exists among the sentences sent; every A-id exists among the periods of the article it is cited for
    (a line's: its own article; a law statement's: any fitted article); a claim citing an id that does not exist is
    void (a line rejected, an element's verdict «da_provare», a law statement dropped) -- counted analysis_bad_sid /
    analysis_bad_aid;
  - elemento (and a line's elemento_in_gioco) matches one of the article's schema elements verbatim: equal after only
    the whitespace collapsed and the quote marks normalised (’ ‘ ` as ', “ ” « » „ as ", one enclosing pair of " out)
    -- case and every other character as written (verbatim_key); else the verdict is dropped
    (analysis_element_not_verbatim) or the line rejected;
  - esito «provato» needs at least one cited S-id whose sentence is not a reported assertion of a party other than
    the client (reported_by): parte_che_afferma names a party that is not the client, or the sentence carries 79.14.3's
    alleged_by (interactive_flow.alleging_party: another party's allegation), or ANY of its asserting phrases
    (_subject_spans: the words right before an assertion verb -- data «assertion_verbs», whole words, any case -- in the
    same clause; the words after «secondo» / «a detta di» … -- data «speaker_prefixes» -- or after «per» at the
    sentence's start -- «speaker_prefixes_initial»; a passive's agent, «accusato da …») names as its speaker (speaker:
    a subject's last mention, a phrase after a marker its first) a non-client party of frame.parti (its name -- the
    client never by a word he shares with another party, a tie read as the other party -- or a content word of its
    ruolo_nel_fatto) or an authority (data «authority_words»: accusa, Procura, PM, investigatori), whatever the
    client says in the same sentence. The client's own statement (every asserting phrase naming the client, his side
    -- cliente, assistito, difesa -- or his procedural role: interactive_flow.client_roles) is never one. Only such sentences -> «contestato» with the party
    (analysis_provato_downgraded); no valid S-id -> «da_provare» (analysis_provato_unsupported);
  - esito «escluso» needs an S-id (else «da_provare», analysis_escluso_unsupported); «contestato» likewise
    (analysis_contestato_unsupported); an esito outside the four -> «da_provare» (analysis_esito_invalid); an element
    of a line's article the reply leaves out -> «da_provare» (analysis_element_missing): every schema element of a line's
    article gets one verdict, in the schema's order;
  - a line (linea_principale, linea_subordinata) is rejected (analysis_line_rejected, the reason re-asked) when its
    articolo is not a fitted norm, is in the remedy / general-part set (data/remedy_articles.json; 2043 c.c. only when
    the verified frame is contractual -- is_contractual; the frame's own verified addebito/domanda article never), its
    elemento_in_gioco is no verbatim schema element of it, its tesi is empty, it cites no id, or an id it cites does not
    exist. After the re-ask: a rejected or missing leading line is «da determinare» (LINE_UNDETERMINED,
    analysis_line_undetermined) -- the old engine's line is never its substitute; a rejected subordinate line is none;
  - the frame's verified addebito/domanda article (case_frame.charge_label) must be the leading or the subordinate
    line's article: else, after the re-ask (none when the article is not among the fitted norms -- the model cannot
    cite it), analysis_frame_charge_missing = 1 and the frame's article is the leading line (stato
    «addebito_del_quadro», no tesi: nothing generated) with every schema element «da_provare»; the model's verified
    leading line (else its subordinate one) becomes the subordinate line;
  - affermazioni_di_diritto: each kept only with at least one A-id, every A-id existing, every article its text names
    a fitted norm with one of its own periods cited (law_statements_off_article), and its tokens grounded in the cited
    periods (data «law_statements.overlap», law_statements_ungrounded) (law_statements_kept); else dropped, never
    printed (law_statements_dropped).
Nothing unverified is in the result; no text is generated by the code beyond the labels.

  run(state, call=None, inp=None, started=None)   the step: {"status", "error", "analysis", "counters", "prompt_hash",
                                                  "seconds", "calls", "reasons", "rejected"}; logs «analysis_step»
  analyse(inp, call=None, started=None)           the same on a ready input (tests; build B)
  make_input(frame, norms, sentences, questions, client, client_roles)   the input
  verify(reply, inp)                              one reply checked (pure)
  finish(checked, inp)                            the verified analysis, the fallbacks applied

Counters: analysis_calls (<= 2), analysis_reasked, analysis_failed (the first call failed, timed out or answered no
JSON), analysis_timeouts, analysis_line_rejected, analysis_line_undetermined, analysis_frame_charge_missing,
analysis_bad_sid, analysis_bad_aid, analysis_element_not_verbatim, analysis_element_missing, analysis_esito_invalid,
analysis_provato_downgraded, analysis_provato_unsupported, analysis_escluso_unsupported,
analysis_contestato_unsupported, law_statements_kept, law_statements_dropped, law_statements_off_article,
law_statements_ungrounded, analysis_seconds.
"""

import contextvars
import hashlib
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, wait as _wait
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .feature_flags import enabled as _flag_enabled

logger = logging.getLogger(__name__)

FLAG = "ANALYSIS_STEP"
_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
DATA_FILE = os.path.join(_DATA, "analysis_step.json")
REMEDY_FILE = os.path.join(_DATA, "remedy_articles.json")
COUNTERS = ("analysis_calls", "analysis_reasked", "analysis_failed", "analysis_timeouts", "analysis_line_rejected",
            "analysis_line_undetermined", "analysis_frame_charge_missing", "analysis_bad_sid", "analysis_bad_aid",
            "analysis_element_not_verbatim", "analysis_element_missing", "analysis_esito_invalid",
            "analysis_provato_downgraded", "analysis_provato_unsupported", "analysis_escluso_unsupported",
            "analysis_contestato_unsupported", "law_statements_kept", "law_statements_dropped",
            "law_statements_off_article", "law_statements_ungrounded", "analysis_seconds")
PROVATO, CONTESTATO, DA_PROVARE, ESCLUSO = "provato", "contestato", "da_provare", "escluso"
OK, FAILED, MALFORMED, TIMEOUT, NO_INPUT, OFF = "ok", "failed", "malformed", "timeout", "no_input", "off"
LINE_KINDS = ("linea_principale", "linea_subordinata")
VERIFIED, FRAME_LINE, UNDETERMINED = "verificata", "addebito_del_quadro", "da_determinare"     # a line's «stato»
# my drafts (v79.15.0): the labels build B prints -- the leading line the step could not verify (§1), never a text
LINE_UNDETERMINED = "da determinare"
LINE_UNVERIFIED_LABEL = "linea non verificata"


def enabled() -> bool:
    return _flag_enabled(FLAG)


@lru_cache(maxsize=1)
def config() -> Dict[str, Any]:
    with open(DATA_FILE, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def remedy_config() -> Dict[str, Any]:
    with open(REMEDY_FILE, encoding="utf-8") as f:
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


def _cf():
    from . import case_frame
    return case_frame


def _flat(value: Any) -> str:
    return " ".join(str(value or "").split())


def empty_counters() -> Dict[str, Any]:
    return {k: 0 for k in COUNTERS}


# --- articles: labels, the remedy set, the contractual frame ----------------------------------------------------------

def same_article(a: Any, b: Any) -> bool:
    """Two labels of one article: the schema's label (element_schemas.norm_label), else (number, code) as
    applicability.article_key reads them («Art. 640 c.p.», «640 cp»)."""
    if not str(a or "").strip() or not str(b or "").strip():
        return False
    es, ap = _es(), _ap()
    if es.norm_label(a) == es.norm_label(b):
        return True
    ka, kb = ap.article_key(a), ap.article_key(b)
    return bool(ka[0]) and ka == kb


def resolve(raw: Any, labels: Sequence[str]) -> Optional[str]:
    """The fitted norm's label an «articolo» of the reply names (same_article; a bare number when one norm has it);
    None when it names none."""
    for label in labels:
        if same_article(raw, label):
            return label
    num, code = _ap().article_key(raw)
    if num and not code:
        hits = [x for x in labels if _ap().article_key(x)[0] == num]
        if len(hits) == 1:
            return hits[0]
    return None


def _number_key(number: Any) -> Optional[Tuple[int, int]]:
    """(the number, its suffix's rank: none 0, bis 1, ter 2 …) of «62-bis», «62 bis», «62bis»; None if not one."""
    from .comma_material import SUFFIXES
    m = re.match(r"^\s*(\d+)\s*(?:-\s*|\s+)?([a-z]+)?\s*$", str(number or "").lower())
    if not m:
        return None
    suffix = m.group(2) or ""
    if suffix and suffix not in SUFFIXES:
        return None
    return int(m.group(1)), (SUFFIXES.index(suffix) + 1 if suffix else 0)


def _in_list(number: str, entries: Sequence[Any]) -> bool:
    key = _number_key(number)
    if key is None:
        return False
    for e in entries or []:
        if isinstance(e, dict):
            lo, hi = _number_key(e.get("from")), _number_key(e.get("to"))
            if lo is not None and hi is not None and lo <= key <= hi:
                return True
        elif _number_key(e) == key:
            return True
    return False


def is_contractual(frame: Optional[Dict[str, Any]]) -> bool:
    """The verified frame says the relationship is contractual: its rito not penal (case_frame.PENAL_RITI, the frame's
    value or its intake value) and either its addebito_o_domanda article a c.c. article inside the data's «cc_ranges»
    (obligations and contracts, 1173-2042) or a word of its addebito_o_domanda (descrizione, frase) or istituto (testo)
    beginning with one of the data's «stems» (contratt-, inadempi-, …). No frame, or a penal one: not contractual."""
    f = frame if isinstance(frame, dict) else {}
    if not f:
        return False
    rito = f.get("rito") or {}
    if rito.get("valore") in _cf().PENAL_RITI or rito.get("intake") in _cf().PENAL_RITI:
        return False
    cfg = remedy_config().get("contractual") or {}
    charge = f.get("addebito_o_domanda") or {}
    if charge.get("articolo"):
        num, code = _ap().article_key(charge["articolo"])
        if code == "cc" and _in_list(num, cfg.get("cc_ranges") or []):
            return True
    words = " ".join(_flat(x) for x in (charge.get("descrizione"), charge.get("frase"),
                                       (f.get("istituto") or {}).get("testo"))).lower()
    return any(re.search(rf"\b{re.escape(stem)}", words) for stem in cfg.get("stems") or [])


def is_remedy(label: Any, contractual: bool = False) -> bool:
    """The article is in the remedy / general-part set (data/remedy_articles.json «articles»; «when_contractual» only
    when the frame is contractual)."""
    num, code = _ap().article_key(label)
    cfg = remedy_config()
    if _in_list(num, (cfg.get("articles") or {}).get(code) or []):
        return True
    return contractual and _in_list(num, (cfg.get("when_contractual") or {}).get(code) or [])


# --- the input --------------------------------------------------------------------------------------------------------

_QUOTE_MAP = str.maketrans({"’": "'", "‘": "'", "`": "'", "“": '"', "”": '"', "«": '"', "»": '"', "„": '"'})


def verbatim_key(text: Any) -> str:
    """An element's text as compared verbatim: whitespace collapsed, quote marks normalised (’ ‘ ` -> ', “ ” « » „ ->
    ") and one enclosing pair of " out; nothing else changed (case, accents, punctuation as written)."""
    s = _flat(str(text or "").translate(_QUOTE_MAP))
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1].strip()
    return s


def make_input(frame: Optional[Dict[str, Any]], norms: Sequence[Dict[str, Any]], sentences: Sequence[Dict[str, Any]],
               questions: Sequence[str] = (), client: str = "", client_roles: Sequence[str] = ()) -> Dict[str, Any]:
    """The step's input. frame: the verified frame's fields (only verified ones; {} none); norms: [{article, text,
    rubrica, elements: [{id, testo}]}] -- the fitted norms; sentences: [{id, text, certainty, source, alleged_by?}]
    numbered as the flow numbers them (a «sistema» one left out, the numbers kept); questions: the user's; client: the
    client's name (the frame's, else the flow's); client_roles: his procedural role words."""
    f = dict(frame or {})
    contractual = is_contractual(f)
    charge = _cf().charge_label({"frame": f}) if (f.get("addebito_o_domanda") or {}).get("articolo") else ""
    out_norms = []
    for n in norms or []:
        label = str(n.get("article") or "").strip()
        if not label:
            continue
        elements = [{"id": str(e.get("id") or ""), "testo": _flat(e.get("testo") or e.get("text"))}
                    for e in n.get("elements") or [] if isinstance(e, dict) and _flat(e.get("testo") or e.get("text"))]
        out_norms.append({"article": label, "rubrica": _flat(n.get("rubrica")), "text": str(n.get("text") or ""),
                          "elements": elements, "periods": _ap().article_ids(label, n.get("text")),
                          "remedy": is_remedy(label, contractual) and not (charge and same_article(label, charge))})
    sents = [{"id": str(s.get("id")), "text": _flat(s.get("text")), "certainty": s.get("certainty"),
              "source": s.get("source"), "alleged_by": str(s.get("alleged_by") or "")}
             for s in sentences or [] if isinstance(s, dict) and s.get("id") and _flat(s.get("text"))
             and s.get("source") != "sistema"]
    client = _flat(client or (f.get("cliente") or {}).get("nome"))
    return {"frame": f, "norms": out_norms, "sentences": sents, "questions": [_flat(q) for q in questions or [] if
                                                                              _flat(q)],
            "client": client, "client_roles": [str(r) for r in client_roles or [] if str(r or "").strip()],
            "charge": charge, "contractual": contractual}


def inputs(state: Dict[str, Any]) -> Dict[str, Any]:
    """The input as the flow's state holds it, after applicability_fit: the held frame's verified fields
    (case_frame.held -- CASE_FRAME on), the records not dropped and not removed by a verified «no»
    (interactive_flow.fit_removed_keys) with their row's text (applicability.norm_input), the state's sentences
    (interactive_flow.case_sentences when none), the frame's quesiti (else asked_questions), the client (the frame's,
    else the state's) and his role words (client_roles)."""
    from . import interactive_flow as flow
    cf, ap = _cf(), _ap()
    held = cf.held(state)
    frame = cf._frame(held) if held else {}
    rows = flow._restored_rows(state.get("rows") or [])
    have = {flow._key(r["_article"]): r for r in rows if r.get("_article")}
    removed = flow.fit_removed_keys(state)
    norms = []
    for label, rec in (state.get("records") or {}).items():
        if not isinstance(rec, dict) or rec.get("dropped"):
            continue
        key = flow._key(flow._parse_label(label))
        if key in removed:
            continue
        norms.append(ap.norm_input(label, have.get(key), rec, rows))
    sents = state.get("sentences") or flow.case_sentences(state)
    questions = list(frame.get("quesiti") or []) or flow.asked_questions(state)
    client = (frame.get("cliente") or {}).get("nome") or (state.get("client") or {}).get("name") or ""
    try:
        roles = flow.client_roles(state)
    except Exception:           # noqa: BLE001 -- no role read: the client's name and side words only
        roles = []
    return make_input(frame, norms, sents, questions, client, roles)


# --- the prompt -------------------------------------------------------------------------------------------------------

def _frame_lines(f: Dict[str, Any]) -> List[str]:
    labels = config()["labels"]
    lines = []
    if f.get("cliente"):
        lines.append(f"{labels['cliente']}: {f['cliente'].get('nome')}")
    for k in ("ruolo", "rito", "stadio"):
        if f.get(k):
            lines.append(f"{labels[k]}: {f[k].get('valore')}")
    if f.get("addebito_o_domanda"):
        a = f["addebito_o_domanda"]
        what = "; ".join(x for x in (a.get("articolo") or "", a.get("descrizione") or "") if x) or _flat(a.get("frase"))
        lines.append(f"{labels['addebito_o_domanda']}: {what}")
    if f.get("istituto"):
        lines.append(f"{labels['istituto']}: {f['istituto'].get('testo')}")
    parti = [p for p in f.get("parti") or [] if isinstance(p, dict) and p.get("nome")]
    if parti:
        lines.append(f"{labels['parti']}:")
        lines += [f"- {p['nome']} ({p.get('lato') or '-'}" + (f"; {p['ruolo_nel_fatto']}" if p.get("ruolo_nel_fatto")
                                                             else "") + ")" for p in parti]
    return lines


def _norm_block(n: Dict[str, Any]) -> str:
    labels = config()["labels"]
    head = [f"ARTICOLO: {n['article']}" + (f" ({config()['remedy_mark']})" if n.get("remedy") else ""),
            f"Rubrica: {n.get('rubrica') or labels['none']}", f"{labels['elements']}:"]
    head += [f"- {e['testo']}" for e in n.get("elements") or []] or [f"- {labels['none']}"]
    head.append(f"{labels['periods']}:")
    body = "\n".join(f"[{p['id']}] {p['text']}" for p in n.get("periods") or [])
    cap = int(config().get("max_article_chars") or 6000)
    if len(body) > cap:
        logger.info("analysis_step: %s periods cut at %d characters in the prompt", n["article"], cap)
        body = body[:cap] + " […]"
    return "\n".join(head) + "\n" + body


def build_messages(inp: Dict[str, Any]) -> List[Any]:
    """[system, human]: the verified frame's fields, the facts (S-id, label, text), the fitted norms (elements,
    numbered periods, the remedy mark), the user's questions."""
    from langchain_core.messages import HumanMessage, SystemMessage
    labels = config()["labels"]
    ap = _ap()
    parts = []
    frame = _frame_lines(inp.get("frame") or {})
    if frame:
        parts.append("\n".join([labels["frame"] + ":"] + frame))
    parts.append(f"{labels['facts']}:\n" + "\n".join(f"{s['id']} [{ap.fact_label(s)}] {s['text']}"
                                                      for s in inp.get("sentences") or []))
    parts.append(f"{labels['norms']}:\n\n" + "\n\n".join(_norm_block(n) for n in inp.get("norms") or []))
    qs = inp.get("questions") or []
    parts.append(f"{labels['questions']}:\n" + ("\n".join(f"- {q}" for q in qs) if qs else labels["none"]))
    return [SystemMessage(content=prompt_system()), HumanMessage(content="\n\n".join(parts))]


def reask_messages(messages: List[Any], reply: str, reasons: Sequence[str]) -> List[Any]:
    """The re-ask: the first exchange, then the rejection reasons (data «reask»)."""
    from langchain_core.messages import AIMessage, HumanMessage
    text = "\n".join(config()["reask"]).replace("{reasons}", "\n".join(f"- {r}" for r in reasons))
    return list(messages) + [AIMessage(content=str(reply or "")), HumanMessage(content=text)]


def request_kwargs() -> Dict[str, Any]:
    """The call's bind kwargs: the fit's (temperature 0, thinking off -- applicability.request_kwargs) with this
    step's reply budget."""
    out = dict(_ap().request_kwargs())
    out["max_tokens"] = int(config().get("max_tokens") or 3000)
    return out


def _default_call(messages: List[Any], kwargs: Dict[str, Any]) -> str:
    """The repo's chat model (ai_chat.chat_model) bound to the kwargs; the reply without its reasoning, the call on the
    request's stage meter -- as applicability._default_call."""
    from . import ai_chat
    ai_chat._stage_count()
    started = time.time()
    response = ai_chat.chat_model.bind(**kwargs).invoke(messages)
    ai_chat._stage_call("analysis_step", started, response)
    return ai_chat._reply_text(response, kwargs.get("max_tokens"))


def parse_reply(raw: Any) -> Dict[str, Any]:
    """The reply's object; ValueError when it holds no JSON object."""
    from .llm_reply import parse_json_reply
    data = parse_json_reply(raw if isinstance(raw, str) else str(raw or ""))
    if not isinstance(data, dict):
        raise ValueError("no JSON object in the reply")
    return data


def call_once(messages: List[Any], call: Optional[Callable[[List[Any], Dict[str, Any]], str]],
              budget: float) -> Dict[str, Any]:
    """One call on a worker thread, waited at most budget seconds: {"status": ok | failed | timeout | malformed,
    "reply" (the text), "data" (the parsed object), "seconds", "error"}. A call not back in time is abandoned (its pool
    shut down without waiting); never raises."""
    started = time.time()
    if budget <= 0:
        return {"status": TIMEOUT, "reply": "", "data": None, "seconds": 0.0, "error": "nessun tempo residuo"}
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="analysis_step")
    try:
        fut = pool.submit(contextvars.copy_context().run, call or _default_call, messages, request_kwargs())
        done, _ = _wait([fut], timeout=budget)
        if fut not in done:
            return {"status": TIMEOUT, "reply": "", "data": None, "seconds": round(time.time() - started, 3),
                    "error": f"oltre {round(budget, 1)} s"}
        try:
            reply = fut.result()
        except Exception as exc:    # noqa: BLE001 -- nothing from the step
            return {"status": FAILED, "reply": "", "data": None, "seconds": round(time.time() - started, 3),
                    "error": f"{type(exc).__name__}: {exc}"[:300]}
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    try:
        data = parse_reply(reply)
    except Exception as exc:        # noqa: BLE001
        return {"status": MALFORMED, "reply": str(reply or "")[:2000], "data": None,
                "seconds": round(time.time() - started, 3), "error": str(exc)[:300]}
    return {"status": OK, "reply": str(reply or ""), "data": data, "seconds": round(time.time() - started, 3),
            "error": ""}


# --- who asserts a sentence -------------------------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _verb_re() -> "re.Pattern[str]":
    verbs = "|".join(f"(?:{v})" for v in config().get("assertion_verbs") or [])
    return re.compile(rf"(?<![\w'’])(?:{verbs})(?!\w)", re.IGNORECASE)


@lru_cache(maxsize=1)
def _prefix_re() -> "re.Pattern[str]":
    """A speaker prefix: data «speaker_prefixes» anywhere after a space or a mark, «speaker_prefixes_initial» («Per la
    Procura, …») only at the sentence's start or after , ; : -- the phrase after it, to the next mark."""
    def alt(key: str) -> str:
        return "|".join(re.escape(p).replace(r"\ ", r"\s+") for p in config().get(key) or []) or r"(?!x)x"
    return re.compile(rf"(?:(?:^|[\s,;:(])(?:{alt('speaker_prefixes')})|(?:^|[,;:])\s*(?:{alt('speaker_prefixes_initial')}))"
                      rf"\s+([^,;:.()]+)", re.IGNORECASE)


_CLAUSE_CUT_RE = re.compile(r"[,;:()«»\"“”]|\b(?:che|mentre|ma|però|perché|poiché|quando|dove|cui|se)\b",
                            re.IGNORECASE)
_SUBJECT_WORDS = 8
_TRAIL = {"ha", "hanno", "aveva", "avevano", "avrebbe", "avrebbero", "abbia", "abbiano", "è", "sono", "era", "erano",
          "non", "anche", "pure", "inoltre", "invece", "poi", "già", "sempre", "infatti", "ora", "oggi", "si"}
# a passive's agent: an assertion verb's participle followed by «da» («è accusato da Giulia Neri di …»); the agent's
# phrase ends at the next mark or at the first «di» / «per» / «a» / «in» / «con» that follows its first word
_PARTICIPLE_RE = re.compile(r"(?:at|ut|it|tt|st)[oaie]$", re.IGNORECASE)
_AGENT_RE = re.compile(r"^\s+(dall['’]|(?:dallo|dalla|dalle|dagli|dai|dal|da)\s)\s*([^,;:.()]+)", re.IGNORECASE)
_AGENT_ARTICLE = {"dal": "il ", "dallo": "lo ", "dalla": "la ", "dai": "i ", "dagli": "gli ", "dalle": "le ",
                  "dall'": "l'", "dall’": "l’"}         # «dalla Procura» -> «la Procura»
_AGENT_END_RE = re.compile(r"\s(?:di|per|a|ai|in|con|nei|nel|nella)\s", re.IGNORECASE)
SUBJECT, AFTER = "subject", "after"     # a phrase before its verb (the speaker its last mention) / after a marker (first)
CLIENT, OTHER = "client", "other"       # who a mention names: the client / another party or an authority


def _words(text: str) -> List[str]:
    return re.findall(r"[^\W\d_]+", str(text or "").lower())


def _subject_spans(text: str) -> List[Tuple[int, str, str]]:
    """(position, phrase, SUBJECT | AFTER) of every phrase the sentence names as asserting something, in order: after a
    speaker prefix («Secondo la società, …», AFTER); the words right before each assertion verb in its clause (SUBJECT:
    a clause boundary -- a punctuation mark or «che», «mentre», «ma», … -- cuts them; an inciso between two commas
    skipped, «X, con una lettera, sostiene»; a verb coordinated with the assertion verb, «ammette e dichiara», and a
    trailing auxiliary or adverb left out; at most _SUBJECT_WORDS words); a passive's agent after the verb's participle
    («accusato da Giulia Neri», AFTER)."""
    s = str(text or "")
    found: List[Tuple[int, str, str]] = [(m.start(), m.group(1), AFTER) for m in _prefix_re().finditer(s)]
    for m in _verb_re().finditer(s):
        parts = _CLAUSE_CUT_RE.split(s[:m.start()])
        tail = parts[-1]
        if not tail.strip() and len(parts) >= 3:     # «X, <inciso>, sostiene»: the subject is before the inciso
            tail = parts[-3]
        words = tail.split()
        if len(words) >= 2 and words[-1].lower() in ("e", "ed"):      # «ammette e dichiara»
            words = words[:-2]
        while words and words[-1].lower().strip(".,") in _TRAIL:
            words = words[:-1]
        phrase = " ".join(words[-_SUBJECT_WORDS:]).strip()
        if re.search(r"[^\W\d_]{2,}", phrase):
            found.append((m.start(), phrase, SUBJECT))
        agent = _AGENT_RE.match(s[m.end():]) if _PARTICIPLE_RE.search(m.group(0)) else None
        if agent:
            phrase = _AGENT_ARTICLE.get(agent.group(1).strip().lower(), "") + \
                _AGENT_END_RE.split(" " + agent.group(2).strip() + " ", maxsplit=1)[0].strip()
            phrase = " ".join(phrase.split()[:_SUBJECT_WORDS])
            if re.search(r"[^\W\d_]{2,}", phrase):
                found.append((m.end(), phrase, AFTER))
    return sorted(found)


def subjects(text: str) -> List[str]:
    """The phrases a sentence names as asserting it, in order (_subject_spans)."""
    return [p for _, p, _k in _subject_spans(text)]


def _parties(inp: Dict[str, Any]) -> List[Tuple[str, bool]]:
    """(name, is the client) of frame.parti, the input's client added when no party is his."""
    cf = _cf()
    out = [(str(p["nome"]), p.get("lato") == "cliente") for p in (inp.get("frame") or {}).get("parti") or []
           if isinstance(p, dict) and _flat(p.get("nome"))]
    client = _flat(inp.get("client"))
    if client and not any(cf.same_party(client, n) for n, _ in out):
        out.append((client, True))
    elif client:
        out = [(n, c or cf.same_party(client, n)) for n, c in out]
    return out


def _mentions(phrase: str, inp: Dict[str, Any]) -> List[Tuple[int, str, str]]:
    """(position, CLIENT | OTHER, the label) of every party or authority a phrase names, in order:
      - an authority word (data «authority_words») -> OTHER, labelled by the word as written with the article before it
        («la Procura», «Gli investigatori»);
      - a run of consecutive name words (case_frame name tokens: accent-folded, lowercased) -> the parties whose
        distinctive words (case_frame._distinctive) are among the run's, a party whose matched words are a strict subset
        of another's left out («Giulia Sarti» names Giulia Sarti, not Paolo Sarti); the client alone -> CLIENT;
        any other party left -> OTHER, its name (a tie between the client and another party -- «Sarti» with both
        named Sarti -- is OTHER: never read as the client's own statement);
      - a content word (4 letters or more) of a non-client party's ruolo_nel_fatto not in the client's -> OTHER;
      - the client's side (interactive_flow.CLIENT_SIDE_STEMS: cliente, assistito, difesa) or procedural role
        (_speaks_as_role on client_roles) -> CLIENT."""
    from . import interactive_flow as flow
    cf = _cf()
    raw = str(phrase or "")
    toks = [(m.start(), m.group(0)) for m in re.finditer(r"[^\W\d_]+", raw)]
    folded = [(pos, (cf.name_tokens(w) or [""])[0]) for pos, w in toks]
    out: List[Tuple[int, str, str]] = []
    low = raw.lower()
    for w in config().get("authority_words") or []:
        w = str(w).lower()
        pat = rf"(?<![\w.]){re.escape(w)}(?![\w])" if "." in w else rf"(?<![\w]){re.escape(w).replace(r'\ ', r'\s+')}\b"
        for m in re.finditer(pat, low):
            det = re.search(r"(?:^|\s)((?:il|lo|la|i|gli|le)\s+|l['’])$", raw[:m.start()], re.IGNORECASE)
            start = m.start() - len(det.group(1)) if det else m.start()
            out.append((start, OTHER, raw[start:m.end()].strip()))
    parties = _parties(inp)
    own = [(name, client, cf._distinctive(name)) for name, client in parties]
    name_words = set().union(*[d for _, _, d in own]) if own else set()
    i = 0
    while i < len(folded):
        if folded[i][1] not in name_words:
            i += 1
            continue
        j = i
        while j < len(folded) and folded[j][1] in name_words:
            j += 1
        run = {w for _, w in folded[i:j]}
        hit = [(name, client, d & run) for name, client, d in own if d & run]
        hit = [h for h in hit if not any(h[2] < o[2] for o in hit)]
        others = [h for h in hit if not h[1]]
        out.append((folded[i][0], OTHER, others[0][0]) if others else (folded[i][0], CLIENT, hit[0][0]))
        i = j
    client_roles = {w for name, client in parties if client for p in (inp.get("frame") or {}).get("parti") or []
                    if isinstance(p, dict) and p.get("nome") == name for w in _words(p.get("ruolo_nel_fatto"))}
    for p in (inp.get("frame") or {}).get("parti") or []:
        if not isinstance(p, dict) or not p.get("nome") or p.get("lato") == "cliente":
            continue
        role = {w for w in _words(p.get("ruolo_nel_fatto")) if len(w) >= 4} - client_roles
        out += [(pos, OTHER, str(p["nome"])) for pos, w in toks if w.lower() in role]
    for pos, w in toks:
        if any(w.lower().startswith(stem) for stem in flow.CLIENT_SIDE_STEMS) or (
                inp.get("client_roles") and flow._speaks_as_role(w, inp["client_roles"])):
            out.append((pos, CLIENT, w))
    if inp.get("client_roles") and flow._speaks_as_role(raw, inp["client_roles"]) and \
            not any(k == CLIENT for _, k, _l in out):          # a role of several words: where its phrase starts
        out.append((0, CLIENT, raw))
    return sorted(out, key=lambda x: x[0])


def speaker(phrase: str, inp: Dict[str, Any], last: bool = True) -> Tuple[str, str]:
    """(CLIENT | OTHER | "", the label) of who a phrase names as speaking: its last mention (a subject before its verb:
    «La difesa di Giulia Neri» is Giulia Neri, «Il difensore di Paolo Sarti» the client) or, last=False, its first (a
    phrase after a marker: «Per la Procura Paolo Sarti ha …» is the Procura); ("", "") none."""
    found = _mentions(phrase, inp)
    if not found:
        return "", ""
    _pos, kind, label = found[-1] if last else found[0]
    return kind, label


def _is_client(phrase: str, inp: Dict[str, Any]) -> bool:
    """The phrase names the client as the one speaking (speaker: CLIENT)."""
    return speaker(phrase, inp)[0] == CLIENT


def _other_party(phrase: str, inp: Dict[str, Any]) -> str:
    """The non-client party or authority the phrase names as speaking (speaker: OTHER), "" none."""
    kind, label = speaker(phrase, inp)
    return label if kind == OTHER else ""


def asserting_party(text: str, inp: Dict[str, Any]) -> str:
    """The non-client party or authority a sentence reports as asserting something: the first of its asserting phrases
    (_subject_spans) whose speaker is not the client -- one is enough, whatever the client says in the same sentence
    («Paolo Sarti nega, ma la Procura sostiene …» is the Procura's assertion); "" when none."""
    for _pos, phrase, kind in _subject_spans(text):
        who, label = speaker(phrase, inp, last=(kind == SUBJECT))
        if who == OTHER:
            return label
    return ""


def reported_by(sentence: Dict[str, Any], inp: Dict[str, Any], parte: str = "") -> str:
    """The party other than the client whose reported assertion the sentence is, "" when it is not one: parte (the
    reply's parte_che_afferma) naming a non-client party; 79.14.3's mark (alleged_by: interactive_flow.alleging_party);
    the sentence's asserting subject (asserting_party)."""
    if _flat(parte) and not _is_client(parte, inp):
        return _flat(parte)
    if sentence.get("alleged_by"):
        return str(sentence["alleged_by"])
    return asserting_party(str(sentence.get("text") or ""), inp)


# --- the verification -------------------------------------------------------------------------------------------------

def _ids(value: Any) -> List[str]:
    return _ap().id_list(value)


def _maps(inp: Dict[str, Any]) -> Dict[str, Any]:
    norms = {n["article"]: n for n in inp.get("norms") or []}
    return {"norms": norms, "labels": list(norms), "sents": {s["id"]: s for s in inp.get("sentences") or []},
            "periods": {label: {p["id"]: p for p in n.get("periods") or []} for label, n in norms.items()},
            "all_periods": {p["id"]: label for label, n in norms.items() for p in n.get("periods") or []}}


def _element_of(norm: Dict[str, Any], text: Any) -> Optional[Dict[str, str]]:
    key = verbatim_key(text)
    if not key:
        return None
    return next((e for e in norm.get("elements") or [] if verbatim_key(e["testo"]) == key), None)


def _verify_line(raw: Any, inp: Dict[str, Any], m: Dict[str, Any], counters: Dict[str, Any]
                 ) -> Tuple[Optional[Dict[str, Any]], str]:
    """(the verified line, "") or (None, why it is rejected)."""
    if not isinstance(raw, dict):
        return None, "non è un oggetto"
    label = resolve(raw.get("articolo"), m["labels"])
    if label is None:
        return None, f"articolo non tra le norme applicabili: {_flat(raw.get('articolo'))[:60] or '(vuoto)'}"
    norm = m["norms"][label]
    if norm.get("remedy"):
        return None, f"art. {label}: rimedio o parte generale, mai una linea"
    element = _element_of(norm, raw.get("elemento_in_gioco"))
    if element is None:
        return None, f"art. {label}: elemento_in_gioco non è il testo di un elemento dello schema"
    sf, sn = _ids(raw.get("cita_fatti")), _ids(raw.get("cita_norma"))
    bad_f = [i for i in sf if i not in m["sents"]]
    bad_n = [i for i in sn if i not in m["periods"][label]]
    counters["analysis_bad_sid"] += len(bad_f)
    counters["analysis_bad_aid"] += len(bad_n)
    if bad_f:
        return None, f"art. {label}: cita_fatti inesistenti: {', '.join(bad_f[:6])}"
    if bad_n:
        return None, f"art. {label}: cita_norma non dell'articolo: {', '.join(bad_n[:6])}"
    if not sf and not sn:
        return None, f"art. {label}: nessun id citato"
    tesi = _flat(raw.get("tesi"))
    if not tesi:
        return None, f"art. {label}: tesi mancante"
    return {"articolo": label, "elemento_in_gioco": element["testo"], "elemento_id": element["id"], "tesi": tesi,
            "cita_fatti": sf, "cita_norma": sn, "stato": VERIFIED}, ""


def verify(reply: Dict[str, Any], inp: Dict[str, Any]) -> Dict[str, Any]:
    """One reply checked (pure): {"lines": {kind: verified line or None}, "rejected": [{kind, articolo, why}],
    "reasons": [the re-ask's], "charge_missing", "reply": the reply, "counters"}. The elements and the law statements
    are verified in finish (on the final lines' articles)."""
    counters = empty_counters()
    m = _maps(inp)
    lines: Dict[str, Optional[Dict[str, Any]]] = {}
    rejected: List[Dict[str, Any]] = []
    reasons: List[str] = []
    for kind in LINE_KINDS:
        raw = reply.get(kind) if isinstance(reply, dict) else None
        if raw in (None, "", {}, []):
            lines[kind] = None
            if kind == "linea_principale":
                reasons.append("linea_principale mancante")
            continue
        line, why = _verify_line(raw, inp, m, counters)
        lines[kind] = line
        if why:
            counters["analysis_line_rejected"] += 1
            rejected.append({"kind": kind, "articolo": _flat((raw or {}).get("articolo") if isinstance(raw, dict)
                                                             else ""), "why": why})
            reasons.append(f"{kind}: {why}")
    charge = inp.get("charge") or ""
    missing = bool(charge) and not any(x and same_article(x["articolo"], charge) for x in lines.values())
    if missing:
        reasons.append(f"l'addebito o la domanda del quadro (art. {charge}) deve essere la linea principale o la "
                       "subordinata")
    return {"lines": lines, "rejected": rejected, "reasons": reasons, "charge_missing": missing,
            "reply": reply if isinstance(reply, dict) else {}, "counters": counters}


def _reply_elements(reply: Dict[str, Any]) -> List[Tuple[Any, Dict[str, Any]]]:
    """The reply's element verdicts as (the «articolo» it names or None, the item): a list, or an object keyed by
    article."""
    raw = reply.get("elementi")
    out: List[Tuple[Any, Dict[str, Any]]] = []
    if isinstance(raw, dict):
        for art, items in raw.items():
            for it in items if isinstance(items, list) else []:
                if isinstance(it, dict):
                    out.append((it.get("articolo") or art, it))
    elif isinstance(raw, list):
        out = [(it.get("articolo"), it) for it in raw if isinstance(it, dict)]
    return out


def _canon_esito(value: Any) -> Optional[str]:
    v = _flat(value).lower()
    v = (config().get("esiti_synonyms") or {}).get(v, v)
    return v if v in config()["esiti"] else None


def _verify_element(item: Dict[str, Any], element: Dict[str, str], inp: Dict[str, Any], m: Dict[str, Any],
                    counters: Dict[str, Any]) -> Dict[str, Any]:
    out = {"elemento": element["testo"], "id": element["id"], "esito": DA_PROVARE, "cita_fatti": [],
           "parte_che_afferma": None, "origine": "modello"}
    esito = _canon_esito(item.get("esito"))
    if esito is None:
        counters["analysis_esito_invalid"] += 1
        return out
    ids = _ids(item.get("cita_fatti"))
    bad = [i for i in ids if i not in m["sents"]]
    if bad:                         # a verdict citing a fact that does not exist is void
        counters["analysis_bad_sid"] += len(bad)
        out["scartato"] = f"cita_fatti inesistenti: {', '.join(bad[:6])}"
        return out
    parte = _flat(item.get("parte_che_afferma")) if isinstance(item.get("parte_che_afferma"), str) else ""
    out["cita_fatti"] = ids
    if parte:
        out["parte_che_afferma"] = parte
    if esito == PROVATO:
        if not ids:
            counters["analysis_provato_unsupported"] += 1
            return out
        by = [reported_by(m["sents"][i], inp, parte) for i in ids]
        if all(by):
            counters["analysis_provato_downgraded"] += 1
            out["esito"], out["parte_che_afferma"] = CONTESTATO, by[0]
            out["declassato"] = "provato su un'allegazione"
            return out
        out["esito"] = PROVATO
        return out
    if esito in (ESCLUSO, CONTESTATO) and not ids:
        counters["analysis_escluso_unsupported" if esito == ESCLUSO else "analysis_contestato_unsupported"] += 1
        return out
    if esito == CONTESTATO and not parte:
        who = [w for w in (reported_by(m["sents"][i], inp) for i in ids) if w]
        out["parte_che_afferma"] = who[0] if who else None
    out["esito"] = esito
    return out


def _elements_for(label: str, inp: Dict[str, Any], m: Dict[str, Any], items: List[Tuple[Any, Dict[str, Any]]],
                  counters: Dict[str, Any], all_open: bool = False) -> List[Dict[str, Any]]:
    """Every schema element of the article, in the schema's order, with its verified verdict (the reply's first item
    for this article -- or naming none -- that matches it verbatim; none: «da_provare», analysis_element_missing;
    all_open: every one «da_provare», the reply's verdicts not read)."""
    norm = m["norms"].get(label) or {}
    got: Dict[str, Dict[str, Any]] = {}
    if not all_open:
        for art, it in items:
            target = resolve(art, m["labels"]) if art else None
            if art and target != label:
                continue
            element = _element_of(norm, it.get("elemento"))
            if element is None or element["testo"] in got:
                continue
            got[element["testo"]] = _verify_element(it, element, inp, m, counters)
    out = []
    for e in norm.get("elements") or []:
        if e["testo"] in got:
            out.append(got[e["testo"]])
            continue
        if not all_open:
            counters["analysis_element_missing"] += 1
        out.append({"elemento": e["testo"], "id": e["id"], "esito": DA_PROVARE, "cita_fatti": [],
                    "parte_che_afferma": None, "origine": "quadro" if all_open else "mancante"})
    return out


def _not_verbatim(items: List[Tuple[Any, Dict[str, Any]]], m: Dict[str, Any], line_labels: Sequence[str]) -> int:
    """The reply's element verdicts for a line's article (or naming no article) that match none of its schema elements
    verbatim -- dropped, never read."""
    n = 0
    for art, it in items:
        target = resolve(art, m["labels"]) if art else None
        if art and target not in line_labels:
            continue
        where = [target] if target else list(line_labels)
        if not any(_element_of(m["norms"].get(x) or {}, it.get("elemento")) for x in where):
            n += 1
    return n


def _law_statements(reply: Dict[str, Any], m: Dict[str, Any], counters: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The verified affermazioni_di_diritto. One is kept only when (in this order, the first failure dropping it,
    law_statements_dropped, logged with the reason):
      - it has a text and at least one A-id, and every A-id exists among the fitted norms' periods (analysis_bad_aid);
      - every article its text names (citations.article_refs) is a fitted norm and has at least one of its own periods
        among the cited ones -- a statement about one article never verified by another's text
        (law_statements_off_article);
      - its tokens (tokens(): the Q&A filter's own) are grounded in the cited periods' text: at least the data's
        «law_statements.overlap» (0.6) of the statement's tokens among the periods' tokens, the denominator the
        statement's tokens (law_statements_ungrounded) -- a statement adding what the cited text does not say is not
        verified by it."""
    from .citations import article_refs
    need = float((config().get("law_statements") or {}).get("overlap") or 0.6)
    out = []
    raw = reply.get("affermazioni_di_diritto")
    for it in raw if isinstance(raw, list) else []:
        testo = _flat(it.get("testo")) if isinstance(it, dict) else ""
        ids = _ids(it.get("cita_norma")) if isinstance(it, dict) else []
        bad = [i for i in ids if i not in m["all_periods"]]
        counters["analysis_bad_aid"] += len(bad)
        why = "vuota" if not testo else "nessun id" if not ids else "id inesistente" if bad else ""
        if not why:
            for num, code in article_refs(testo):
                label = resolve(f"{num} {code}".strip(), m["labels"])
                if label is None or not any(m["all_periods"][i] == label for i in ids):
                    why = f"art. {num} {code}".strip() + (" non tra le norme applicabili" if label is None else
                                                          ": nessun suo periodo citato")
                    counters["law_statements_off_article"] += 1
                    break
        if not why:
            cited = " ".join(str((m["periods"][m["all_periods"][i]].get(i) or {}).get("text") or "") for i in ids)
            if overlap(testo, cited) < need:
                why = f"non fondata sui periodi citati ({round(overlap(testo, cited), 2)})"
                counters["law_statements_ungrounded"] += 1
        if why:
            counters["law_statements_dropped"] += 1
            logger.info("analysis_step: law statement dropped (%s): %s", why, testo[:120])
            continue
        counters["law_statements_kept"] += 1
        out.append({"testo": testo, "cita_norma": ids,
                    "articoli": list(dict.fromkeys(m["all_periods"][i] for i in ids))})
    return out


def finish(checked: Dict[str, Any], inp: Dict[str, Any]) -> Dict[str, Any]:
    """The verified analysis from a checked reply, the fallbacks applied: the frame's charge missing -> its article the
    leading line (FRAME_LINE, every element «da_provare»), the model's verified leading line (else subordinate) the
    subordinate one; no verified leading line (and no frame charge) -> «da determinare» (UNDETERMINED); the elements of
    the lines' articles and the law statements verified. Its counters added to checked["counters"]."""
    counters = checked["counters"]
    m = _maps(inp)
    lines = dict(checked["lines"])
    principale, subordinata = lines.get("linea_principale"), lines.get("linea_subordinata")
    charge = inp.get("charge") or ""
    frame_label = ""
    if checked.get("charge_missing") and charge:
        counters["analysis_frame_charge_missing"] = 1
        frame_label = resolve(charge, m["labels"]) or charge
        subordinata = principale or subordinata
        principale = {"articolo": frame_label, "elemento_in_gioco": None, "elemento_id": None, "tesi": "",
                      "cita_fatti": [], "cita_norma": [], "stato": FRAME_LINE}
    elif principale is None:
        counters["analysis_line_undetermined"] = 1
        principale = {"articolo": None, "elemento_in_gioco": None, "elemento_id": None, "tesi": "", "cita_fatti": [],
                      "cita_norma": [], "stato": UNDETERMINED, "label": LINE_UNDETERMINED,
                      "label_sintesi": LINE_UNVERIFIED_LABEL}
    if subordinata is not None and principale.get("articolo") and subordinata["articolo"] == principale["articolo"] \
            and subordinata.get("elemento_in_gioco") == principale.get("elemento_in_gioco"):
        subordinata = None          # the same line twice: once
    items = _reply_elements(checked.get("reply") or {})
    elements: Dict[str, List[Dict[str, Any]]] = {}
    for x in (principale, subordinata):
        if not x or not x.get("articolo") or x["articolo"] in elements:
            continue
        elements[x["articolo"]] = _elements_for(x["articolo"], inp, m, items, counters,
                                                all_open=(x.get("stato") == FRAME_LINE))
    read = [x["articolo"] for x in (principale, subordinata) if x and x.get("articolo") and x.get("stato") != FRAME_LINE]
    counters["analysis_element_not_verbatim"] += _not_verbatim(items, m, read)
    return {"linea_principale": principale, "linea_subordinata": subordinata, "elementi": elements,
            "affermazioni_di_diritto": _law_statements(checked.get("reply") or {}, m, counters),
            "rimedi": [n["article"] for n in inp.get("norms") or [] if n.get("remedy")],
            "frame_charge": charge, "frame_charge_missing": bool(frame_label), "contractual": inp.get("contractual"),
            "rejected": checked.get("rejected") or []}


# --- the step ---------------------------------------------------------------------------------------------------------

def _line_log(analysis: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for kind in LINE_KINDS:
        x = (analysis or {}).get(kind)
        if x:
            out.append({"kind": kind, "articolo": x.get("articolo"), "elemento_in_gioco": x.get("elemento_in_gioco"),
                        "stato": x.get("stato"), "cita_fatti": x.get("cita_fatti"), "cita_norma": x.get("cita_norma")})
    return out


def analyse(inp: Dict[str, Any], call: Optional[Callable[[List[Any], Dict[str, Any]], str]] = None,
            started: Optional[float] = None) -> Dict[str, Any]:
    """The step on a ready input: at most 2 calls within «deadline_s» from started (default: now); the result
    {"status", "error", "analysis" (None unless ok), "counters", "prompt_hash", "seconds", "calls", "reasons",
    "rejected"}; logged «analysis_step»."""
    t0 = started if started is not None else time.time()
    deadline = float(config().get("deadline_s") or 40)
    counters = empty_counters()
    calls: List[Dict[str, Any]] = []
    status, error, analysis, reasons, rejected = NO_INPUT, "", None, [], []
    if inp.get("norms") and inp.get("sentences"):
        messages = build_messages(inp)
        first = call_once(messages, call, deadline - (time.time() - t0))
        calls.append({"attempt": 1, "status": first["status"], "seconds": first["seconds"], "error": first["error"]})
        status, error = first["status"], first["error"]
        if first["status"] == OK:
            checked = verify(first["data"], inp)
            reasons = list(checked["reasons"])
            rejected = [dict(r, attempt=1) for r in checked["rejected"]]
            futile = checked["reasons"] == [r for r in checked["reasons"] if r.startswith("l'addebito")] and \
                resolve(inp.get("charge"), [n["article"] for n in inp["norms"]]) is None
            if checked["reasons"] and not futile:
                second = call_once(reask_messages(messages, first["reply"], checked["reasons"]), call,
                                   deadline - (time.time() - t0))
                calls.append({"attempt": 2, "status": second["status"], "seconds": second["seconds"],
                              "error": second["error"]})
                counters["analysis_reasked"] = 1
                if second["status"] == OK:
                    again = verify(second["data"], inp)
                    again["counters"]["analysis_line_rejected"] += checked["counters"]["analysis_line_rejected"]
                    rejected += [dict(r, attempt=2) for r in again["rejected"]]
                    reasons += [r for r in again["reasons"] if r not in reasons]
                    checked = again
                else:
                    logger.warning("analysis_step: re-ask %s: %s (the first reply's verification stands)",
                                   second["status"], second["error"])
            analysis = finish(checked, inp)
            analysis["rejected"] = rejected
            for k, v in checked["counters"].items():
                counters[k] = counters.get(k, 0) + v
        else:
            counters["analysis_failed"] = 1
            counters["analysis_timeouts"] = int(first["status"] == TIMEOUT)
            logger.warning("analysis_step: %s: %s (nothing from the step)", first["status"], first["error"])
    counters["analysis_calls"] = len(calls)
    seconds = round(time.time() - t0, 3)
    counters["analysis_seconds"] = seconds
    result = {"status": status, "error": error, "analysis": analysis if status == OK else None, "counters": counters,
              "prompt_hash": prompt_hash(), "seconds": seconds, "calls": calls, "reasons": reasons,
              "rejected": rejected}
    _log("analysis_step", {"prompt_hash": result["prompt_hash"], "status": status, "error": error,
                           "seconds": seconds, "calls": calls, "counters": counters,
                           "lines": _line_log(result["analysis"]), "rejected": rejected})
    return result


def run(state: Dict[str, Any], call: Optional[Callable[[List[Any], Dict[str, Any]], str]] = None,
        inp: Optional[Dict[str, Any]] = None, started: Optional[float] = None) -> Dict[str, Any]:
    """The step on the flow's state (after applicability_fit): ANALYSIS_STEP off -> {"status": "off"} (nothing read,
    nothing called); else analyse(inputs(state)). Never raises: an input that cannot be read is status failed."""
    if not enabled():
        return {"status": OFF, "error": "", "analysis": None, "counters": {}, "prompt_hash": prompt_hash(),
                "seconds": 0.0, "calls": [], "reasons": [], "rejected": []}
    t0 = started if started is not None else time.time()
    if inp is None:
        try:
            inp = inputs(state)
        except Exception as exc:    # noqa: BLE001 -- the step never fails the answer
            logger.warning("analysis_step: input not read: %s", exc)
            counters = empty_counters()
            counters["analysis_failed"] = 1
            result = {"status": FAILED, "error": f"input: {type(exc).__name__}: {exc}"[:300], "analysis": None,
                      "counters": counters, "prompt_hash": prompt_hash(), "seconds": round(time.time() - t0, 3),
                      "calls": [], "reasons": [], "rejected": []}
            _log("analysis_step", {k: result[k] for k in ("prompt_hash", "status", "error", "seconds", "calls",
                                                          "counters")} | {"lines": [], "rejected": []})
            return result
    return analyse(inp, call=call, started=t0)


# --- build B: rendering with the flag on (brief item 2) ----------------------------------------------------------------
# With ANALYSIS_STEP on, the method's §1, §4's element tables (the Esito column), §6, §9, every «x su y» count and the
# «Risposte ai quesiti» strategy line come from the verified analysis only (view): the code prints the verified lines'
# article, elemento_in_gioco and tesi, the verified verdicts and the cited sentences of the case as they are -- nothing
# else but the labels below. The step not run, failed or answering nothing: the leading line «da determinare» (§1:
# LINE_UNVERIFIED_LABEL), no table, no count -- never the old engine's line in its place.

# my drafts (v79.15.0 build B): the labels -- never a text
LABEL_MAIN = "Linea principale"
LABEL_SUB = "Linea subordinata"
LABEL_ELEMENT = "elemento in gioco"
LABEL_FRAME_LINE = "addebito del quadro, linea non verificata"
LABEL_REMEDIES = "Rimedi e parte generale collegati alla linea"
LABEL_ESITI = "Esiti dell'art. {art}"
LABEL_TO_PROVE = "Da provare"
LABEL_CONTESTED = "Contestati"
LABEL_NO_TABLE = "Nessuna linea verificata: nessuna tabella degli elementi."
LABEL_NO_FACT = "nessun fatto citato"
LABEL_QA_MAIN = "La linea principale è: {line}."
LAW_STATEMENTS_TITLE = "**Affermazioni di diritto verificate**"
ESITO_LABELS = {PROVATO: "provato", CONTESTATO: "contestato", DA_PROVARE: "da provare", ESCLUSO: "escluso"}
CONTESTED_BY = "contestato (allegazione {di_parte})"
TABLE_HEAD = ("| Elemento | Fatti citati | Esito |", "|---|---|---|")
SOLUTIONS_HEAD = ("| Linea | Articolo | Elemento in gioco | Tesi | Fatti citati |", "|---|---|---|---|---|")
RENDER_COUNTERS = ("qa_sentences_dropped", "norms_sentences_dropped", "legacy_disagrees_line", "legacy_disagrees_esito")


def undetermined_line() -> Dict[str, Any]:
    return {"articolo": None, "elemento_in_gioco": None, "elemento_id": None, "tesi": "", "cita_fatti": [],
            "cita_norma": [], "stato": UNDETERMINED, "label": LINE_UNDETERMINED, "label_sintesi": LINE_UNVERIFIED_LABEL}


def view(result: Optional[Dict[str, Any]], inp: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """What the rendering reads (flag on): {"status", "analysis" (the verified one; the step not ok: the leading line
    «da determinare», nothing else), "sentences" {S-id: text}, "rubriche" {label: rubrica}, "remedies" [labels of the
    fitted norms in the remedy / general-part set], "statements" (the verified affermazioni_di_diritto)}."""
    result = result or {}
    inp = inp or {}
    analysis = result.get("analysis") if result.get("status") == OK else None
    if not isinstance(analysis, dict):
        analysis = {"linea_principale": undetermined_line(), "linea_subordinata": None, "elementi": {},
                    "affermazioni_di_diritto": [], "rimedi": [n["article"] for n in inp.get("norms") or []
                                                              if n.get("remedy")]}
    return {"status": result.get("status") or NO_INPUT, "analysis": analysis,
            "sentences": {s["id"]: s["text"] for s in inp.get("sentences") or []},
            "rubriche": {n["article"]: n.get("rubrica") or "" for n in inp.get("norms") or []},
            "remedies": list(analysis.get("rimedi") or []),
            "statements": list(analysis.get("affermazioni_di_diritto") or [])}


def _lines(v: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    a = v.get("analysis") or {}
    out = [(LABEL_MAIN, a.get("linea_principale") or undetermined_line())]
    if a.get("linea_subordinata"):
        out.append((LABEL_SUB, a["linea_subordinata"]))
    return out


def _stop(text: Any) -> str:
    return _flat(text).rstrip(" .;:")


def line_text(line: Optional[Dict[str, Any]]) -> str:
    """One line as printed: «art. X — <tesi> (elemento in gioco: «…»)»; the frame's article kept by the fallback
    «art. X (addebito del quadro, linea non verificata)»; undetermined «da determinare (linea non verificata)»."""
    x = line or undetermined_line()
    if x.get("stato") == UNDETERMINED or not x.get("articolo"):
        return f"{LINE_UNDETERMINED} ({LINE_UNVERIFIED_LABEL})"
    if x.get("stato") == FRAME_LINE:
        return f"art. {x['articolo']} ({LABEL_FRAME_LINE})"
    out = f"art. {x['articolo']} — {_stop(x.get('tesi'))}"
    if x.get("elemento_in_gioco"):
        out += f" ({LABEL_ELEMENT}: «{x['elemento_in_gioco']}»)"
    return out


def main_tesi(v: Dict[str, Any]) -> str:
    """The leading line's printed text (the main-line-repeated check reads it)."""
    return line_text((v.get("analysis") or {}).get("linea_principale"))


def summary_text(v: Dict[str, Any]) -> str:
    """§1: the leading line and the subordinate one, labels and the verified text only."""
    return "\n\n".join(f"{label}: {line_text(x)}." for label, x in _lines(v))


def esito_text(e: Dict[str, Any]) -> str:
    if e.get("esito") == CONTESTATO and e.get("parte_che_afferma"):
        from . import issue_records
        return CONTESTED_BY.format(di_parte=issue_records.of_party(e["parte_che_afferma"]))
    return ESITO_LABELS.get(e.get("esito"), ESITO_LABELS[DA_PROVARE])


def _cell(text: Any) -> str:
    return _flat(text).replace("|", "/")


def _facts_cell(ids: Sequence[str], v: Dict[str, Any]) -> str:
    texts = [v["sentences"][i] for i in ids or [] if i in v.get("sentences", {})]
    return " / ".join(texts) if texts else LABEL_NO_FACT


def line_articles(v: Dict[str, Any]) -> List[str]:
    """The lines' articles, once each, the leading one first (none for an undetermined line)."""
    out: List[str] = []
    for _, x in _lines(v):
        if x.get("articolo") and x["articolo"] not in out:
            out.append(x["articolo"])
    return out


def counts(v: Dict[str, Any], label: str) -> Dict[str, int]:
    """{esito: n, "totale": n} of the article's verified verdicts -- the only «x su y» the answer prints."""
    els = ((v.get("analysis") or {}).get("elementi") or {}).get(label) or []
    out = {k: sum(1 for e in els if e.get("esito") == k) for k in ESITO_LABELS}
    out["totale"] = len(els)
    return out


def counts_text(v: Dict[str, Any], label: str) -> str:
    c = counts(v, label)
    if not c["totale"]:
        return ""
    return LABEL_ESITI.format(art=label) + ": " + "; ".join(f"{ESITO_LABELS[k]} {c[k]} su {c['totale']}"
                                                            for k in ESITO_LABELS) + "."


def norms_tables(v: Dict[str, Any], extra_remedies: Sequence[str] = ()) -> str:
    """§4's «Analisi per elementi»: one table per line article (Elemento | Fatti citati | Esito, the verified verdicts),
    its count; then the remedy / general-part articles attached to the line -- named, no table, no count."""
    parts: List[str] = []
    for label in line_articles(v):
        els = ((v.get("analysis") or {}).get("elementi") or {}).get(label) or []
        if not els:
            continue
        rubrica = _flat((v.get("rubriche") or {}).get(label))
        rows = [f"**Art. {label}**" + (f" — {rubrica}" if rubrica else ""), "", *TABLE_HEAD]
        rows += [f"| {_cell(e['elemento'])} | {_cell(_facts_cell(e.get('cita_fatti'), v))} | {_cell(esito_text(e))} |"
                 for e in els]
        parts.append("\n".join(rows))
        parts.append(counts_text(v, label))
    if not parts:
        parts.append(LABEL_NO_TABLE)
    remedies = list(dict.fromkeys(list(v.get("remedies") or []) + [x for x in extra_remedies or [] if is_remedy(x)]))
    remedies = [x for x in remedies if not any(same_article(x, y) for y in line_articles(v))]
    if remedies:
        named = [f"art. {x}" + (f" ({_flat(v['rubriche'][x])})" if _flat((v.get("rubriche") or {}).get(x)) else "")
                 for x in remedies]
        parts.append(f"**{LABEL_REMEDIES}**: " + "; ".join(named) + ".")
    return "\n\n".join(p for p in parts if p)


def solutions_text(v: Dict[str, Any]) -> str:
    """§6: the verified lines only (an undetermined leading line said as such) -- never a containment line or any
    other the analysis did not return."""
    rows = list(SOLUTIONS_HEAD)
    for label, x in _lines(v):
        if x.get("stato") == VERIFIED:
            rows.append(f"| {label} | art. {_cell(x['articolo'])} | {_cell(x.get('elemento_in_gioco'))} | "
                        f"{_cell(_stop(x.get('tesi')))} | {_cell(_facts_cell(x.get('cita_fatti'), v))} |")
        else:
            rows.append(f"| {label} | {_cell(line_text(x))} | — | — | — |")
    return "\n".join(rows)


def conclusions_text(v: Dict[str, Any]) -> str:
    """§9: the lines, the leading article's counts and its open and contested elements -- verified, labels only."""
    out = [f"**{label}**: {line_text(x)}." for label, x in _lines(v)]
    for label in line_articles(v)[:1]:
        c = counts_text(v, label)
        if c:
            head, _, rest = c.partition(": ")
            out.append(f"**{head}**: {rest}")
        els = ((v.get("analysis") or {}).get("elementi") or {}).get(label) or []
        open_ = [f"«{e['elemento']}»" for e in els if e.get("esito") == DA_PROVARE]
        contested = [f"«{e['elemento']}» ({esito_text(e)})" for e in els if e.get("esito") == CONTESTATO]
        if open_:
            out.append(f"**{LABEL_TO_PROVE}**: " + ", ".join(open_) + ".")
        if contested:
            out.append(f"**{LABEL_CONTESTED}**: " + ", ".join(contested) + ".")
    return "\n".join(out)


def answer_strategy(v: Dict[str, Any]) -> List[str]:
    """The «Risposte ai quesiti» strategy sentence: the verified leading line (labels only)."""
    return [LABEL_QA_MAIN.format(line=line_text((v.get("analysis") or {}).get("linea_principale")))]


def statements_block(v: Dict[str, Any]) -> str:
    """The verified affermazioni_di_diritto, each with its articles -- printed in «Risposte ai quesiti»."""
    items = [f"- {_stop(s['testo'])} (art. {', '.join(s.get('articoli') or [])})." for s in v.get("statements") or []
             if _flat(s.get("testo"))]
    return f"{LAW_STATEMENTS_TITLE}\n\n" + "\n".join(items) if items else ""


# --- the Q&A prose: a statement of what a norm provides with no verified counterpart, removed ---------------------------

@lru_cache(maxsize=1)
def _qa_verb_re() -> "re.Pattern[str]":
    verbs = "|".join(f"(?:{x})" for x in (config().get("qa_statements") or {}).get("verbs") or [])
    return re.compile(rf"(?<![\w'’])(?:{verbs})(?!\w)", re.IGNORECASE)


def tokens(text: Any) -> set:
    """A sentence's tokens for the Q&A match: its words (letters only) of «token_min_len» letters or more, lowercased,
    the data's «stopwords» out."""
    cfg = config().get("qa_statements") or {}
    n = int(cfg.get("token_min_len") or 3)
    stop = {str(w).lower() for w in cfg.get("stopwords") or []}
    return {w for w in re.findall(r"[^\W\d_]+", str(text or "").lower()) if len(w) >= n and w not in stop}


def overlap(sentence: Any, statement: Any) -> float:
    """|tokens(sentence) ∩ tokens(statement)| / |tokens(sentence)| -- the denominator the Q&A sentence's tokens (0.0 for
    a sentence with none)."""
    s = tokens(sentence)
    return len(s & tokens(statement)) / len(s) if s else 0.0


def _numbers(refs: Sequence[Tuple[str, str]]) -> set:
    return {str(n).lower().replace(" ", "") for n, _ in refs or [] if n}


@lru_cache(maxsize=1)
def _qa_referent_re() -> "re.Pattern[str]":
    words = "|".join(f"(?:{x})" for x in (config().get("qa_statements") or {}).get("referents") or []) or r"(?!x)x"
    return re.compile(rf"(?<![\w'’])(?:{words})(?!\w)", re.IGNORECASE)


def norm_statement(sentence: str) -> Optional[List[Tuple[str, str]]]:
    """A sentence stating what a norm provides: it holds one of the data's «verbs» (the verbs and phrasings of a rule:
    «prevede», «punisce», «commina», «deve», «la pena», «reclusione», …) and either cites an article -> the articles it
    cites, or names a norm without a number (data «referents»: «la norma», «la disposizione», «l'articolo», …) -> []
    (a statement on whichever article); None when it is not one."""
    from .citations import article_refs
    if not _qa_verb_re().search(sentence or ""):
        return None
    refs = article_refs(sentence)
    if refs:
        return refs
    return [] if _qa_referent_re().search(sentence or "") else None


def counterpart(sentence: str, v: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The verified affermazione_di_diritto the sentence matches: overlap() at least the data's «overlap» (0.6) and,
    when the sentence cites articles, one of the statement's articles' numbers among them (a sentence naming «la
    norma» without a number: any verified statement); None."""
    refs = norm_statement(sentence)
    nums = _numbers(refs or [])
    need = float((config().get("qa_statements") or {}).get("overlap") or 0.6)
    for s in v.get("statements") or []:
        theirs = {_ap().article_key(a)[0].replace(" ", "") for a in s.get("articoli") or []}
        if (not nums or nums & theirs) and overlap(sentence, s.get("testo")) >= need:
            return s
    return None


def qa_filter(sentences: Sequence[str], v: Dict[str, Any], log: Optional[List[Dict[str, Any]]] = None) -> List[str]:
    """The Q&A sentences kept: a statement of what a norm provides (norm_statement) with no verified counterpart is
    removed (into log: the sentence, its articles, the best overlap)."""
    out = []
    for x in sentences or []:
        refs = norm_statement(x)
        if refs is not None and counterpart(x, v) is None:
            if log is not None:
                best = max([overlap(x, s.get("testo")) for s in v.get("statements") or []] or [0.0])
                log.append({"testo": _flat(x)[:200], "articoli": [f"{n} {c}".strip() for n, c in refs],
                            "overlap": round(best, 2)})
            continue
        out.append(x)
    return out


# --- the old engine, logged and compared (never rendered with the flag on) ----------------------------------------------

LEGACY_ESITO = {"provato": PROVATO, "da provare": DA_PROVARE, "non risulta": ESCLUSO}


def legacy_esito(r: Dict[str, Any]) -> str:
    from . import issue_records
    if issue_records.contested(r):
        return CONTESTATO
    return LEGACY_ESITO.get(r.get("status") or "da provare", DA_PROVARE)


def compare(v: Dict[str, Any], legacy: Dict[str, Any], records: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """The old engine against the verified analysis: legacy_disagrees_line (1 when the old main line's article is not the
    analysis's leading article -- an undetermined one included), legacy_disagrees_esito (the line articles' elements
    whose old verdict differs: matched verbatim on the record's requisites; an element the record lacks is not
    compared); the old verdicts per element."""
    lead = ((v.get("analysis") or {}).get("linea_principale") or {}).get("articolo") or ""
    old = (legacy.get("linea_principale") or {}).get("articolo") or ""
    disagrees_line = int(not (lead and old and same_article(lead, old)) and bool(lead or old))
    diffs, olds = 0, {}
    for label in line_articles(v):
        rec = next((r for k, r in (records or {}).items() if same_article(k, label)), None) or {}
        by_text = {verbatim_key(r.get("text")): r for r in rec.get("requisites") or [] if isinstance(r, dict)}
        rows = []
        for e in ((v.get("analysis") or {}).get("elementi") or {}).get(label) or []:
            r = by_text.get(verbatim_key(e["elemento"]))
            if r is None:
                continue
            was = legacy_esito(r)
            rows.append({"elemento": e["elemento"], "legacy": was, "verificato": e.get("esito")})
            diffs += int(was != e.get("esito"))
        olds[label] = rows
    return {"legacy_disagrees_line": disagrees_line, "legacy_disagrees_esito": diffs, "elementi": olds}


def open_facts(v: Dict[str, Any], split: Dict[str, Any], records: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """§2's «Fatti mancanti» with the verified analysis: only the entries on an element the analysis holds «da_provare»
    (the record's element matched verbatim to the line article's verdict) -- the old engine's other open elements
    never listed."""
    want = {(label, verbatim_key(e["elemento"])) for label in line_articles(v)
            for e in ((v.get("analysis") or {}).get("elementi") or {}).get(label) or [] if e.get("esito") == DA_PROVARE}
    kept = []
    for m in (split or {}).get("mancanti") or []:
        label = next((x for x, _ in want if same_article(x, m.get("article"))), None)
        rec = next((r for k, r in (records or {}).items() if same_article(k, m.get("article"))), None) or {}
        text = next((r.get("text") for r in rec.get("requisites") or [] if isinstance(r, dict)
                     and str(r.get("id")) == str(m.get("elemento"))), None)
        if label and text is not None and (label, verbatim_key(text)) in want:
            kept.append(m)
    return {**(split or {}), "mancanti": kept}
