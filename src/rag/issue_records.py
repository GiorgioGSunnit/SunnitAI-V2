"""v60f: the per-article record -- the intermediate representation between round 1 and the answer.

Built at the end of round 1 (build: no model call), kept in the session (state["records"], state["sentences"]):

  state["sentences"]: the case as numbered sentences, each {id "S1", text, source ("caso" | "precisazione"),
      certainty: "accertato" (narrated as a fact) | "allegato" (attributed to someone: «sostiene», «secondo», a reported
      conditional) | "ipotizzato" (hedged: «forse», «sembra») | "assunto" (the system's: an assumed client, a deduced
      field)}; precisazioni added by replies (sentences()).
  state["records"]: {label "N code": {
      article, rubrica, topic (title), grade, issue (the grader's), kept [reasons], dropped (reason | None),
      requisites: [{id "R1", comma, text, kind ("requisito" | "comma"), facts_status (requisite_status: the extracted
                    facts), status (the record's: settle), why (its ground), supporting [S ids], contradicting [S ids],
                    missing (what is not known), question (the open question asking it, its id)}],
      verdict («applicabile» | «da provare» | «non applicabile»: verdict), linked (the fact<->element call made: its
      sentences' fingerprint), answer {bucket, reason, removals}}}

The fact<->element link (link): one bounded model call per chosen topic, the topics in parallel, at the answer step:
for each requisite of the topic's articles, the case sentences that support it, those that contradict it, and what is
missing. v60g (d): the record's status settles both (settle): the facts' first; else a sentence of the case narrated as
a fact (accertato) contradicting it makes it «non risulta», one supporting it «provato»; an allegation or a hypothesis
alone leaves it «da provare». While an answer is written (statuses_from), every status the answer reads -- the §2 status
lines, the writer's rules block, the checks, the read-back -- is the record's (recorded_status): one status source.

v60h: (5) «provato» by the link needs a supporting sentence holding the element's own nouns (names_element), else «da
provare» (v60t: gone -- a grounded supporting sentence, the case's own words, is enough: settle, grounded); the requisite that is round 1's distinctive element is marked "distinctive" (seed), and its «non risulta»
makes the verdict «non applicabile» -- one line in «Norme considerate e non applicabili», never a block; round 1's
demotion runs in smaller calls with a longer bound and logs what it left unassessed. (1) A querela offence's d'ufficio
conditions are record elements of their own (rec["procedibilita"], kind "procedibilita", ids P1…): linked with the
topic's requisites, settled like them, never part of the verdict; §4's line takes their statuses.

v60i: (1) a §2 block only for an article whose verdict is «applicabile», or «da provare» with at least one sentence of
the case supporting one of its requisites, or (v60j) with a missing fact an open round-1 question asks -- the lawyer
asked for exactly that «da verificare» (block_due); every other article is one line -- «Norme considerate e non
applicabili» or «Altre norme del tema non trattate» -- with its reason (line_reason). (2) Role-aware verdicts: an
article the counterparty relies on (rec["relied_by"], set at the answer step: the other side's line of the first read,
or the action gate) whose elements fail is the client's argument -- verdict «argomento del cliente», a block written as
such --, never «non applicabile».

v60m: round 1 demotes an article to grade 1 only when its distinctive element is excluded by the extracted facts or
contradicted by a sentence of the case narrated as a fact (demote_contradicted); an element the case is silent on never
demotes -- the article keeps its grade and the earned-block rule (block_due) decides whether it is written.
"""

import contextlib
import contextvars

import hashlib
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

SENTENCES_MAX = 80
SENTENCE_CHARS = 260
COMMI_MAX = 6
REQUISITES_PER_ARTICLE = 12
LINK_TOKENS = 1100
LINK_SECONDS = 30.0
LINK_REQUISITES_MAX = 36

ACCERTATO, ALLEGATO, IPOTIZZATO, ASSUNTO = "accertato", "allegato", "ipotizzato", "assunto"
COUNTERPARTY = "controparte"        # v60t: a sentence's side when it is attributed to a party other than the client
# v78: «sostenendo», «sosteneva», the forms of «negare», «apprendeva che» -- a reported claim («il difensore apprendeva
# che Tizio aveva negato…, sostenendo che…») was read as narrated fact and made an element «non risulta» (v60t forbids it)
# v79.8.5 (C1, the PO's line): only an attributed sentence -- reported speech, a party's version -- is an allegazione; a
# narrated fact stays certain even with one of these words («Luca lamenta forti dolori», «ha sporto denuncia», «secondo il
# contratto», «accusa un malore», «il medico ha dichiarato una prognosi»): each verb counts only as it reports content
_PARTY = (r"(?:client\w*|controparte|difes\w*|ricorrent\w*|attor\w*|convenut\w*|imputat\w*|indagat\w*|testimon\w*|"
          r"test[ei]\b|denunciant\w*|querelant\w*|vittim\w*|persona\s+offesa|perit\w*|consulent\w*|accusa\b)")
_ALLEGED_RE = re.compile(r"\b(?:sostien\w*|sosteng\w*|sosten(?:endo|eva|evano|ne|nero)\b|sosten(?:ut[oa]|ere)\s+che|afferm\w*|"
                         r"lament\w*\s+(?:che|di\s+(?:aver|essere|non))|contest\w*\s+(?:che|di\s+aver|a\s+(?-i:[A-ZÀ-Ü])\w+)|"
                         r"dichiar(?:a|ano|ava|avano|ò|ato|ata)\s+(?:che|di)\b|riferisc\w*\s+che|riferit\w*\s+che|"
                         r"asseris\w*|asserit\w*|denunci\w*\s+che|accus\w*\s+(?:\w+\s+){0,2}?di\b|ritien\w*\s+(?:che|di)\b|"
                         r"pretend\w*\s+che|eccepi\w*|"
                         r"neg(?:a|ano|ava|avano|ato|ata|ati|ando|herebbe|ò|arono)|apprend\w*\s+che|appres[oa]\s+che|"
                         r"venut[oa]\s+a\s+sapere|a\s+sua\s+versione|nella\s+sua\s+versione|"
                         r"secondo\s+(?:quanto\s+(?:riferit|dichiarat|sostenut|affermat)\w*|la\s+(?:sua|loro)\s+versione|"
                         r"(?-i:[A-ZÀ-Ü])\w+|(?:il|la|lo|i|le|gli|l['’])\s*" + _PARTY + r")|"
                         r"a (?:suo|loro) dire|a detta d\w*)\b",
                         re.IGNORECASE)
# v79.9 (D1): an allegazione only on an attribution verb or «secondo <parte>» -- the conditional («avrebbero dovuto
# concludersi», «sarebbe stato restituito»), the tense and «presunto» no longer make one; a sentence with no attribution
# is a fatto certo whatever its grammar (a hedge -- «forse», «potrebbe», «se fosse» -- stays ipotizzato, as v60t)
_HEDGED_RE = re.compile(r"\b(?:forse|probabilmente|sembra\w*|pare(?:va)?|potrebbe\w*|presumibilmente|non è chiaro|"
                        r"non si sa|non risulta chiaro|verosimilmente|ipotizz\w*|"
                        r"(?:se|qualora|ove) (?:fosse|fossero|avesse|avessero)\b)\b", re.IGNORECASE)   # v60t: «se fosse»


# v79.8 (C1): the client's own claim letter («con diffida del 3 marzo Tizio chiedeva …») is an allegazione, never a fact
_CLAIM_LETTER_RE = re.compile(r"\b(?:lettera|diffida|raccomandata|pec|messa\s+in\s+mora|missiva|reclamo|atto\s+di\s+"
                              r"citazione|ricorso)\b[^.;]{0,120}?\b(?:chied|richied|intim|contest|lament|diffid|invit|"
                              r"sollecit|pretend|domand)\w*", re.IGNORECASE)


def certainty(sentence: str) -> str:
    """How the case states it: ipotizzato (hedged or hypothetical -- v60t: first, an attributed hypothesis is one too),
    allegato (attributed or reported; v79.8: a claim letter's content too), else accertato. v79.12.7 (item 7): a
    procedural act narrated (the PM's contestazione, an avviso, a notifica, the rinvio a giudizio, a decreto --
    data/text_hygiene.json) is a certain fact: only what is left once the act's own words are out is read for a hedge or
    an attribution («il PM ha contestato a Gamma l'omicidio» is accertato; «Gamma sostiene di non aver ricevuto la
    notifica» stays allegato)."""
    from . import text_hygiene
    act = text_hygiene.procedural_act(sentence or "")
    text = text_hygiene.act_remainder(sentence, act) if act else (sentence or "")
    if _HEDGED_RE.search(text):
        return IPOTIZZATO
    if _ALLEGED_RE.search(text) or _CLAIM_LETTER_RE.search(text):
        return ALLEGATO
    return ACCERTATO


def stated_certain(sentence: Dict[str, Any]) -> bool:
    """v79.12.7 (item 7): narrated as a fatto certo -- grounded and accertato, a procedural act's sentence included:
    §2's «fatti certi»."""
    return grounded(sentence) and sentence.get("certainty") == ACCERTATO


def act_bears(sentence: Dict[str, Any], element: str = "") -> bool:
    """v79.12.7 (item 7): the sentence may decide the element -- any sentence but a procedural act's, which proves the
    act was done and nothing it charges: only an element naming the act (text_hygiene.act_named)."""
    act = sentence.get("act")
    if not act:
        return True
    from . import text_hygiene
    return text_hygiene.act_named(element, act)


def certain_fact(sentence: Dict[str, Any], element: str = "") -> bool:
    """v79.8 (C1): a fatto certo -- the case's own words (grounded), stated as accertato: the one source of every
    «provato» in §4. An allegazione, a counterparty's statement, a claim letter: never. v79.12.7 (item 7): a procedural
    act's sentence only for an element naming the act (element: the element's text; none, never) -- §2 reads
    stated_certain."""
    return stated_certain(sentence) and act_bears(sentence, element)


_SENTENCE_RE = re.compile(r"(?<=[.;!?])\s+(?=[A-ZÀ-Ü«\"(])")


def _split(text: str) -> List[str]:
    from .sentence_split import split          # v75: «il Sig. Caio» is one sentence
    out = []
    for piece in [x for line in re.split(r"\n+", text or "") for x in split(line, _SENTENCE_RE)]:
        piece = " ".join(piece.split())
        if len(piece) >= 12 and re.search(r"[a-zà-ü]{3}", piece):
            out.append(piece[:SENTENCE_CHARS])
    return out


def _asked(text: str, asks: Sequence[str]) -> bool:
    """v75: the sentence is one of the requested blocks (a listed quesito) or holds one."""
    key = " ".join(re.findall(r"[a-zà-ü0-9]+", text.lower()))
    for a in asks or []:
        ask = " ".join(re.findall(r"[a-zà-ü0-9]+", str(a or "").lower()))
        if len(ask.split()) >= 2 and (ask in key or key in ask):
            return True
    return False


def sentences(case: str, additions: Sequence[str] = (), assumed: Sequence[str] = (),
              asks: Sequence[str] = ()) -> List[Dict[str, str]]:
    """The case's sentences, then the client's precisazioni, then the system's assumptions, numbered S1… v60i (4): never
    the user's own message (fact_signals.case_only: the request over an upload, a question, «Sei l'avvocato di …», an
    instruction) -- no element is supported or contradicted by it. v75: nor the requested blocks (asks: the quesiti
    intake listed) -- the ask is never a fact."""
    from .fact_signals import case_only, user_message
    from . import text_hygiene
    out: List[Dict[str, str]] = []
    seen = set()

    def add(text, source, how=None, origin=""):
        key = text.lower()
        if key in seen or len(out) >= SENTENCES_MAX or source != "sistema" and _asked(text, asks):
            return
        seen.add(key)
        out.append({"id": f"S{len(out) + 1}", "text": text, "source": source, "certainty": how or certainty(text),
                    "verbatim": verbatim(text, origin)})                # v60t (1): grounding
        act = text_hygiene.procedural_act(text) if source != "sistema" else ""
        if act:                         # v79.12.7 (item 7): the act it narrates (certain_fact reads it)
            out[-1]["act"] = act
    for s in _split(case_only(case)):
        add(s, "caso", origin=case)
    for a in additions or []:
        for s in _split(a):
            if not user_message(s):
                add(s, "precisazione", origin=a)
    for a in assumed or []:
        add(" ".join(str(a).split())[:SENTENCE_CHARS], "sistema", ASSUNTO)
    return out


def verbatim(sentence: str, origin: str) -> bool:
    """v60t (1): the sentence is the case's own words -- its text (spaces aside) inside the text it was read from."""
    norm = lambda t: " ".join(str(t or "").split())          # noqa: E731
    return bool(sentence) and norm(sentence) in norm(origin)


def grounded(sentence: Dict[str, Any]) -> bool:
    """v60t (1): a supporting sentence that grounds «provato» -- the case's or the client's own words, verbatim; never the
    system's (an assumption)."""
    return sentence.get("source") != "sistema" and bool(sentence.get("verbatim", True)) and bool(sentence.get("text"))


def fingerprint(sents: Sequence[Dict[str, str]]) -> str:
    return hashlib.sha1("\n".join(s["text"] for s in sents).encode("utf-8")).hexdigest()[:12]


# --- the record ---------------------------------------------------------------------------------------------------------

def _label(ref: Sequence[str]) -> str:
    from .topic_step import _code
    return f"{ref[0]} {_code(ref[1])}"


def _requisites(row: Dict[str, Any], analysis: Dict[str, Any], fragments: Optional[List[Dict[str, str]]] = None
                ) -> List[Dict[str, Any]]:
    """The writer's own reading of the article (rule_quote.article_rules, no plan): per selected comma its requisites,
    a comma with none as one element; each with the facts' status (facts_status). v60g: the row's text cleaned as the
    answer step cleans it (rule_quote.clean_text), so the requisites' words are the answer's."""
    from . import rule_quote
    from .interactive_flow import facts_status, DA_PROVARE
    schema_reqs = _schema_requisites(row, analysis, fragments)          # v79.11 (6): ELEMENT_SCHEMAS only
    if schema_reqs is not None:
        return schema_reqs
    sec = row.get("s") or {}
    if sec.get("plain_text"):
        row = {**row, "s": {**sec, "plain_text": rule_quote.clean_text(sec.get("plain_text"))}}
    rules = rule_quote.article_rules({}, [row], analysis, [tuple(row["_article"])])
    if not rules:
        return []
    rule = rules[0]
    split_by = "clausole penali" if rule.get("how") == "offence_clauses" else ""      # v76: logged
    texts = dict(rule.get("commi") or [])
    out: List[Dict[str, Any]] = []
    selected = (rule.get("selected") or [])[:COMMI_MAX]
    alt_commi = alternative_commi(texts, selected)      # v79.19.4 (rule 1): a comma offering the option of the one before
    for k in selected:
        reqs = (rule.get("requisites") or {}).get(k)
        special = special_comma(texts.get(k) or "") if selected and k != selected[0] else ""     # v79.8 (D1)
        if not reqs and not rule_quote.offence(tuple(row["_article"])) and texts.get(k):
            one = rule_quote.civil_split(texts[k], single=True)     # v76b: a record's one condition is an element
            reqs = one if len(one) == 1 else reqs
        items = reqs if reqs else [" ".join(str(texts.get(k) or "").split())[:300]]
        for text in items:
            if not text or len(out) >= REQUISITES_PER_ARTICLE:
                continue
            why = rule_quote.fragment_why(text) if reqs else ""       # v79.8 (C2): a fragment is no element
            if why:
                if fragments is not None:
                    fragments.append({"comma": str(k), "text": text[:120], "why": why})
                _log_fragment(row, text, why)
                continue
            status = facts_status(text, analysis) if reqs else DA_PROVARE
            alternatives = rule_quote.branches(text) if reqs else []     # v79.6: one element, its branches
            out.append({"id": "", "comma": k, "text": text, "kind": "requisito" if reqs else "comma",
                        "facts_status": status, "status": status, "why": "", "supporting": [], "contradicting": [],
                        "missing": "", "question": "", **({"split": split_by} if split_by and reqs else {}),
                        **({"branches": alternatives} if alternatives else {}),
                        **({"alt_commi": alt_commi[k]} if k in alt_commi else {}),
                        **({"special": special} if special else {})})
    for r in out:               # v79.19.4 (rule 1): each requisite of an alternative comma knows the other's requisites
        if r.get("alt_commi"):
            r["alternatives"] = [x["text"] for x in out if str(x.get("comma")) in {str(c) for c in r["alt_commi"]}
                                 and x.get("kind") == "requisito"]
    return out


# v79.19.4 (rule 1): a selected comma that opens on a disjunction («In alternativa …», «Ovvero …», «Oppure …», «O, …»)
# offers the option of the comma before it: their requisites are alternatives, never required together. The
# disjunctions inside one element are rule_quote.branches (v79.6); the two sets feed demote_contradicted, the record's
# verdict (unmet_in_anchor) and the elements the reader gets (element_status).
_ALT_COMMA_RE = re.compile(r"^\W*(?:in\s+alternativa|ovvero|oppure|o,)\b", re.IGNORECASE)


def alternative_commi(texts: Dict[Any, str], selected: Sequence[Any]) -> Dict[Any, List[Any]]:
    """{comma: [the commi whose requisites are its alternatives]} among the selected commi -- a comma opening on a
    disjunction marker and the one selected before it; {} with none."""
    out: Dict[Any, List[Any]] = {}
    for prev, k in zip(list(selected), list(selected)[1:]):
        if _ALT_COMMA_RE.match(" ".join(str(texts.get(k) or "").split())):
            out.setdefault(k, []).append(prev)
            out.setdefault(prev, []).append(k)
    return out


def alternative_open(rec: Dict[str, Any], req: Dict[str, Any]) -> bool:
    """v79.19.4 (rule 1): the requisite is one alternative of a paragraph (alt_commi) and the other comma's requisites
    are not every one «non risulta» -- its own exclusion excludes nothing."""
    from .interactive_flow import NON_RISULTA
    commi = {str(c) for c in req.get("alt_commi") or []}
    others = [r for r in rec.get("requisites") or [] if r.get("kind") == "requisito" and str(r.get("comma")) in commi
              and r is not req]
    return bool(others) and any(r.get("status") != NON_RISULTA for r in others)


def alternative_satisfied(rec: Dict[str, Any], req: Dict[str, Any]) -> bool:
    """v79.19.4 (rule 1, row 80): the other alternative of the paragraph is «provato» -- this one is not a missing fact
    of a line built on that other."""
    from .interactive_flow import PROVATO
    commi = {str(c) for c in req.get("alt_commi") or []}
    return any(r.get("kind") == "requisito" and str(r.get("comma")) in commi and r.get("status") == PROVATO
               for r in rec.get("requisites") or [])


def schema_line(row: Optional[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], str]:
    """v79.11 (6): (the article's schema line, why) -- element_schemas.for_row; (None, "off") with ELEMENT_SCHEMAS
    off, nothing read."""
    from . import element_schemas
    if not element_schemas.enabled():
        return None, "off"
    if not row:
        return None, "no_schema"
    return element_schemas.for_row(row)


def _schema_requisites(row: Dict[str, Any], analysis: Dict[str, Any],
                       fragments: Optional[List[Dict[str, str]]] = None) -> Optional[List[Dict[str, Any]]]:
    """v79.11 (6): ELEMENT_SCHEMAS -- the article's elements from its schema (element_schemas.requisites) when for_row
    gives a line; None -- the text path, unchanged -- with the flag off, no usable schema, or a schema that yields no
    element (its fragments then left to the text path's own)."""
    line, _ = schema_line(row)
    if line is None:
        return None
    from . import element_schemas
    own: List[Dict[str, str]] = []
    reqs = element_schemas.requisites(row, line, analysis, own)
    if not reqs:
        return None
    if fragments is not None:
        fragments.extend(own)
    return reqs


def element_path(row: Optional[Dict[str, Any]], reqs: Sequence[Dict[str, Any]] = (),
                 line: Optional[Dict[str, Any]] = None, why: str = "") -> Optional[Dict[str, str]]:
    """v79.11 (6): how the record's elements were read -- {"path": "schema" | "reviewed" | "text", "why": for_row's
    reason ("schema_empty" when the schema gave no element)}; None with ELEMENT_SCHEMAS off. line/why: for_row's, when
    the caller has them."""
    from . import element_schemas
    if not element_schemas.enabled():
        return None
    if not why:
        line, why = schema_line(row)
    if line is not None:
        if any(r.get("split") == "schema" for r in reqs or []):
            return {"path": "reviewed" if why == "reviewed" else "schema", "why": why}
        return {"path": "text", "why": "schema_empty"}
    return {"path": "text", "why": why}


def schema_summary(line: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """v79.11 (6): what the record keeps of its schema (element_schemas.summary); None without one."""
    if not isinstance(line, dict):
        return None
    from . import element_schemas
    return element_schemas.summary(line)


def _log_event(event: str, data: Dict[str, Any]) -> None:
    """v79.11: an interactive_<event> log line; never fails the caller."""
    try:
        from .interactive_flow import _log
        _log(event, data)
    except Exception:           # noqa: BLE001
        pass


def log_element_path(label: str, path: Optional[Dict[str, str]]) -> None:
    """v79.11 (6): one element_path event per record {article, path, why} (ELEMENT_SCHEMAS only: path is None off)."""
    if path:
        _log_event("element_path", {"article": label, **path})


# v79.8 (D1): a comma that states a special case, a presumption or an aggravation of the base rule -- its own record
_SPECIAL_COMMA_RE = re.compile(r"^\W*(?:\d+\.\s*)?(?:nei\s+casi\s+previst\w*|nel\s+caso\s+previst\w*|nei\s+casi\s+di\b|"
                               r"(?:la|le)\s+(?:pena|pene)\s+(?:è|sono)\s+(?:aumentat|diminuit|ridott)\w*|"
                               r"(?:la|le)\s+(?:stessa|stesse)\s+pen[ae]\b|alla\s+stessa\s+pena\b|"
                               r"se\s+il\s+fatto\s+è\s+commess\w*|agli\s+effetti\s+del)|"
                               r"\bsi\s+(?:consider\w*|presum\w*|reput\w*|ha\s+per)\b|\b(?:è|sono)\s+presunt\w*\b|"
                               r"\bla\s+pena\s+è\s+aumentat\w*", re.IGNORECASE)
SPECIAL_KINDS = (("presunzione", re.compile(r"\bsi\s+(?:presum|reput|ha\s+per)|\bpresunt", re.I)),
                 ("aggravamento", re.compile(r"\bpen[ae]\b[^.]{0,40}\b(?:aumentat|diminuit|ridott)|stess[ae]\s+pen", re.I)),
                 ("equiparazione", re.compile(r"\bsi\s+consider|\bagli\s+effetti\s+del", re.I)),
                 ("caso speciale", re.compile(r".", re.S)))


def special_comma(text: str) -> str:
    """v79.8 (D1): the kind of a comma that states a special case («nei casi previsti …»), a presumption («si
    presume …», «si considera …»), an aggravation («la pena è aumentata …») -- "" for the base rule's own commi."""
    t = " ".join(str(text or "").split())
    if not t or not _SPECIAL_COMMA_RE.search(t):
        return ""
    return next(kind for kind, rx in SPECIAL_KINDS if rx.search(t))


def comma_label(label: str, comma: Any) -> str:
    """v79.8 (D1): a special comma's record label -- «2051 c.c. co. 2»."""
    return f"{label} co. {comma}"


def split_special(label: str, rec: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """v79.8 (D1): {label: record} -- the base rule's record with its own elements only, and one record per special
    comma (its elements, labelled with its comma, «comma_of» the base): a special comma never adds elements to the
    base rule's record, nor its open ones to the base's missing facts. A record with no special comma: as it is."""
    reqs = rec.get("requisites") or []
    by_comma: Dict[Any, List[Dict[str, Any]]] = {}
    for r in reqs:
        if r.get("special"):
            by_comma.setdefault(r.get("comma"), []).append(r)
    if not by_comma:
        return {label: rec}
    out = {label: {**rec, "requisites": [r for r in reqs if not r.get("special")]}}
    for k, own in by_comma.items():
        base = {x: v for x, v in rec.items() if x not in ("charged", "named_due", "anchor_due", "requisites",
                                                         "procedibilita", "alternative", "alternative_facts")}
        out[comma_label(label, k)] = {**base, "requisites": own, "comma_of": label, "comma": k,
                                      "special": own[0]["special"],
                                      "rubrica": f"{rec.get('rubrica') or ''} (co. {k}: {own[0]['special']})".strip()}
    return out


def _log_fragment(row: Dict[str, Any], text: str, why: str) -> None:
    """v79.8 (C2): the fragment dropped at decomposition, logged with its reason; never fails the decomposition."""
    try:
        from .interactive_flow import _log
        _log("element_fragment", {"article": " ".join(str(x) for x in row.get("_article") or ()), "text": text[:120],
                                  "why": why})
    except Exception:           # noqa: BLE001
        pass


def _question_for(requisite: Dict[str, Any], questions: Sequence[Dict[str, Any]]) -> str:
    """The open question asking what the requisite needs: its fact family's, else a first-read question sharing two of
    its words."""
    from .interactive_flow import _FAMILY, _names_in, _stems
    families = {_FAMILY.get(n) for n in _names_in(requisite["text"])} - {None}
    for q in questions or []:
        if q.get("family") in families:
            return str(q.get("id"))
    mine = _stems(requisite["text"])
    for q in questions or []:
        if q.get("kind") == "free" and len(mine & _stems(q.get("point") or q.get("text"))) >= 2:
            return str(q.get("id"))
    return ""


def own_rubrica(label: str, row: Dict[str, Any], topic: Dict[str, Any],
               rubriche: Optional[Dict[str, str]] = None) -> str:
    """v79.12.1: a record's rubrica is the article's own heading -- its row's (the store's «titolo_articolo»), else the
    store's heading read for it elsewhere (rubriche: {label: heading}, the pool's and the rows'), and the topic's rubrica
    only for the topic's own anchor (it is the anchor's heading); never the heading of the topic the article is filed
    under (52 and 55 c.p. filed under «Omicidio colposo», 61 c.p. under «Furto»). "" when none is known."""
    s = (row or {}).get("s") or {}
    own = str(s.get("titolo_articolo") or "").strip()
    if own:
        return own
    stored = str((rubriche or {}).get(label) or "").strip()
    if stored:
        return stored
    return str(topic.get("rubrica") or "") if label == str(topic.get("anchor") or "") else ""


def build(topics: Sequence[Dict[str, Any]], rows: Sequence[Dict[str, Any]], analysis: Dict[str, Any],
          questions: Sequence[Dict[str, Any]] = (), others: Sequence[Dict[str, Any]] = (),
          grading: Optional[Dict[str, Any]] = None, rubriche: Optional[Dict[str, str]] = None
          ) -> Dict[str, Dict[str, Any]]:
    """{label: record} for every article of the listed topics (kept: why), the altri profili (dropped: grado 1 / beyond
    the topics' number) and the prefilter's drops (dropped: prefiltro). v79.12.1: each record's rubrica is its own
    (own_rubrica; rubriche: the store's headings by label), never its topic's."""
    from .interactive_flow import _key, _parse_label, _topic_refs
    by_key = {_key(r["_article"]): r for r in rows or [] if r.get("_article")}
    out: Dict[str, Dict[str, Any]] = {}
    from . import comma_material
    by_grade: List[str] = []            # v79.19.9 (A, row 107): the records writable by their grade alone
    for t in topics or []:
        grades = t.get("grades") or {}
        matched = set(t.get("matched") or [])
        # v60s (2): a comma the topic names for one of its articles («Legittima difesa (art. 52, co. 2, c.p.)»)
        named = _named_commi(" ".join(str(t.get(k) or "") for k in ("title", "table_name", "rubrica")), comma_material)
        for ref in _topic_refs(t, all_=True):
            label = _label(ref)
            if label in out:
                continue
            row = by_key.get(_key(ref)) or {}
            s = row.get("s") or {}
            kept = []
            grade = grades.get(label, row.get("_grade"))
            if isinstance(grade, int):
                kept.append(f"grado {grade}")
            if label == t.get("anchor"):
                kept.append("ancora del tema")
            if label in matched:
                kept.append("articolo del tema")
            if label in (t.get("offences") or []):
                kept.append("reato identificato")
            if row.get("_first_read"):
                kept.append(f"primo esame: {row['_first_read']}")
            # v79.19.9 (A, row 107): a listed article graded 2 or more is writable whatever its place in the topic's
            # list (79.8's «non scrivibile» kept 21 key articles out); the writer's refs already rank every graded
            # article (capped_refs), the link, the plan issues and the fallback plan read them too (_topic_refs all_)
            planned = ref in _topic_refs(t)
            writable = planned or (isinstance(grade, int) and grade >= 2)
            if writable and not planned:
                by_grade.append(label)
            rubrica = own_rubrica(label, row, t, rubriche)          # v79.12.1: never the topic's heading
            line, line_why = schema_line(row)                       # v79.11 (6): (None, "off") with the flag off
            gated = not decomposed_by_gate(label, rubrica,          # v74: never a fattispecie record, any route
                                           **({"schema": line} if line is not None else {}))
            frags: List[Dict[str, str]] = []                        # v79.8 (C2): the fragments dropped
            split = bool(row and writable and not gated)
            out[label] = {"article": label, "rubrica": rubrica,
                          "topic": t.get("title") or "", "grade": grade if isinstance(grade, int) else None,
                          "issue": str(row.get("_grade_issue") or ""), "kept": kept or ["elencato nel tema"],
                          "dropped": None if writable else "elencato, non scrivibile (né graduato né del tema)",
                          "requisites": _requisites(row, analysis, frags) if split else [],
                          "linked": "", "answer": {}}
            if frags:
                out[label]["fragments"] = frags
            path = element_path(row, out[label]["requisites"], line, line_why) if split else None
            if path:                                                # v79.11 (6): ELEMENT_SCHEMAS only
                out[label]["element_path"] = path
                log_element_path(label, path)
            if line is not None:
                out[label]["schema"] = schema_summary(line)
            if gated:
                out[label]["not_fattispecie"] = NOT_FATTISPECIE
            seed(out[label], row)
            out[label]["procedibilita"] = _conditions(row) if row and writable else []
            if named.get(_key(ref)):
                out[label]["anchor_comma"] = min(named[_key(ref)])
    for o in others or []:
        for a in o.get("articles") or []:
            label = _label(_parse_label(a))
            row = by_key.get(_key(_parse_label(a))) or {}
            out.setdefault(label, {"article": label, "rubrica": "", "topic": o.get("title") or "", "grade": 1 if
                                   o.get("why") == "grado 1" else None, "issue": str(row.get("_grade_issue") or ""),
                                   "kept": [], "dropped": (row.get("_grade_issue") if row.get("_demoted") else None)
                                   or o.get("why") or "altri profili", "requisites": [], "linked": "", "answer": {}})
    for label in (grading or {}).get("prefilter_dropped") or []:
        label = _label(_parse_label(label))
        out.setdefault(label, {"article": label, "rubrica": "", "topic": "", "grade": None, "issue": "", "kept": [],
                               "dropped": "prefiltro (fuori dai primi N per somiglianza)", "requisites": [],
                               "linked": "", "answer": {}})
    if by_grade:                        # v79.19.9 (A): logged interactive_writable_by_grade {articles}
        _log_event("writable_by_grade", {"articles": by_grade})
    number(out)
    refresh(out, analysis, questions)
    return out


def _named_commi(text: str, comma_material) -> Dict[Tuple[str, str], set]:
    """{article key: the commi the text names for it} (comma_material.named_commi, keys as interactive_flow._key)."""
    from .interactive_flow import _key
    try:
        return {_key(ref): set(ks) for ref, ks in comma_material.named_commi(text).items() if ks}
    except Exception:                       # a title the patterns cannot read names no comma
        return {}


def _conditions(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """v60h (1): a querela offence's d'ufficio conditions (procedural_facts: the store's, else its text) as record
    elements of their own (kind "procedibilita": linked and settled like a requisite, never a §2 requisite nor part of
    the verdict)."""
    from . import procedural_facts
    from .interactive_flow import DA_PROVARE
    facts = procedural_facts.facts_of(row)          # v79.12.3: with the schema's procedibilità block (ELEMENT_SCHEMAS)
    if not (facts and facts.get("querela")):
        return []
    conds = facts.get("ufficio_condizioni")
    if conds is None:
        conds = procedural_facts.from_row(row).get("ufficio_condizioni") or []
    return [{"id": "", "comma": c.get("comma"), "text": c.get("text") or "", "opener": c.get("opener") or "",
             "kind": "procedibilita", "facts_status": DA_PROVARE, "status": DA_PROVARE, "why": "", "supporting": [],
             "contradicting": [], "missing": "", "question": "", **({"rejoined": True} if c.get("rejoined") else {})}
            for c in rejoin_conditions(conds) if c.get("text")]


# v79.8 (C3): a condition the store cut at its connective («… e», «quando …») is one condition again; a condition joined
# by «quando / se / qualora / purché / e» holds only when every part does
_DANGLING_HEAD_RE = re.compile(r"^\s*(?:e|ed|quando|qualora|se|purch[ée]|ove|nonch[ée])\b", re.IGNORECASE)
_DANGLING_TAIL_RE = re.compile(r"\b(?:e|ed|quando|qualora|se|purch[ée]|ove|nonch[ée]|che)\s*[,;]?\s*$", re.IGNORECASE)
_PART_SPLIT_RE = re.compile(r",?\s+(?:quando|qualora|purch[ée]|ove)\s+|,?\s+(?:e|ed)\s+(?=(?:se|quando|qualora|purch|il|"
                            r"la|lo|l['’]|i|gli|le|ricorr\w*|sussist\w*|vi|ci|si|sia|siano|abbia|abbiano|risult\w*)\b)|"
                            r",?\s+se\s+", re.IGNORECASE)


def dangling(text: str) -> bool:
    """v79.8 (C3): a condition's text that begins or ends on a connective -- half of a condition."""
    return bool(_DANGLING_HEAD_RE.match(text or "") or _DANGLING_TAIL_RE.search(text or ""))


def rejoin_conditions(conds: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """v79.8 (C3): the store's conditions with every half joined to its other half (same comma, the first ending on a
    connective or the second beginning with one) -- marked «rejoined»."""
    out: List[Dict[str, Any]] = []
    for c in conds or []:
        c = dict(c)
        text = " ".join(str(c.get("text") or "").split())
        prev = out[-1] if out else None
        if prev is not None and prev.get("comma") == c.get("comma") and \
                (_DANGLING_TAIL_RE.search(prev["text"]) or _DANGLING_HEAD_RE.match(text)):
            prev["text"] = f"{prev['text'].rstrip(' ,;')} {text}".strip()
            prev["rejoined"] = True
            continue
        out.append({**c, "text": text})
    return out


def condition_parts(text: str) -> List[str]:
    """v79.8 (C3): the parts a connective joins in a condition (each holding a noun of its own); [] for one part."""
    parts = [p.strip(" ,;") for p in _PART_SPLIT_RE.split(text or "") if p and element_nouns(p)]
    return parts if len(parts) >= 2 else []


def elements(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The record's linked elements: its requisites, then its d'ufficio conditions (v60h)."""
    return list(rec.get("requisites") or []) + list(rec.get("procedibilita") or [])


def number(records: Dict[str, Dict[str, Any]]) -> None:
    n = p = 0
    for rec in records.values():
        for req in rec.get("requisites") or []:
            n += 1
            req["id"] = f"R{n}"
        for cond in rec.get("procedibilita") or []:
            p += 1
            cond["id"] = f"P{p}"


def refresh(records: Dict[str, Dict[str, Any]], analysis: Dict[str, Any], questions: Sequence[Dict[str, Any]] = (),
            sents: Sequence[Dict[str, str]] = ()) -> None:
    """The facts' statuses again (after a reply), each requisite settled with its link (settle), each open requisite's
    question, each article's verdict."""
    from .interactive_flow import DA_PROVARE, facts_status
    by_id = {s["id"]: s for s in sents or []}
    for rec in records.values():
        for req in rec.get("requisites") or []:
            if req.get("kind") == "requisito":
                req["facts_status"] = facts_status(req["text"], analysis)
            settle(req, by_id)
            req["question"] = _question_for(req, questions) if req["status"] == DA_PROVARE else ""
        for cond in rec.get("procedibilita") or []:
            settle(cond, by_id)
            # v60k: the open question asking a d'ufficio condition (single shot: a point that changes procedibilità)
            cond["question"] = _question_for(cond, questions) if cond["status"] == DA_PROVARE else ""
        rec["verdict"] = verdict(rec)


# v60g (d): the record's status of a requisite -- the facts' when they decide it, else the link's sentences (v60t: a
# contradiction narrated as fact makes it «non risulta»; any grounded support «provato»); a whole comma (kind "comma")
# keeps the facts' (a sentence never proves a comma's every element)
def _trigger(req: Dict[str, Any], by_id: Dict[str, Dict[str, str]]) -> Optional[Dict[str, str]]:
    """v76: the case sentence a fact-derived status rests on -- the case's or a precisazione's own words (grounded,
    narrated or attributed, never the system's), sharing the element's nouns (two, or its only one; a precisazione --
    the user's answer to a question -- one); None when no sentence does: the status stays «da provare»."""
    own = element_nouns(req.get("text") or "")
    if not own:
        return None
    for s in sorted(by_id.values(), key=lambda x: x.get("source") != "precisazione"):
        if s.get("source") == "sistema" or not certain_fact(s, req.get("text") or ""):      # v79.8 (C1): a fatto certo
            continue
        need = 1 if s.get("source") == "precisazione" else min(2, len(own))
        if len(own & element_nouns(s.get("text") or "")) >= need:
            return s
    return None


def settle(req: Dict[str, Any], by_id: Dict[str, Dict[str, str]]) -> None:
    from .interactive_flow import DA_PROVARE, NON_RISULTA, PROVATO
    facts = req.get("facts_status") or req.get("status") or DA_PROVARE
    req.setdefault("facts_status", facts)
    linked = req.get("link") or {}
    base_sup = list(linked.get("supporting", req.get("supporting") or []))
    base_con = list(linked.get("contradicting", req.get("contradicting") or []))
    # v60t: a sentence attributed to the counterparty is its version -- never support: the contradiction path
    theirs = [i for i in base_sup if (by_id.get(i) or {}).get("side") == COUNTERPARTY]
    req["supporting"] = [i for i in base_sup if i not in theirs]
    req["contradicting"] = base_con + [i for i in theirs if i not in base_con]
    sup = [by_id[i] for i in req["supporting"] if i in by_id]
    con = [by_id[i] for i in req["contradicting"] if i in by_id]
    sure_con = [s for s in con if s.get("certainty") == ACCERTATO and act_bears(s, req.get("text") or "")]
    # v60t (1): «provato» needs a supporting sentence that is the case's own words (grounded); a hedged or hypothetical
    # one never. v79.8 (C1): a fatto certo only -- an allegazione, a counterparty's statement, a claim letter leave the
    # element «da provare» (certain_fact: §2's «fatti certi» are the same sentences)
    ground = [s for s in sup if certain_fact(s, req.get("text") or "")]       # v79.12.7 (item 7): an act's, its own
    alleged = [s for s in sup if grounded(s) and s.get("certainty") == ALLEGATO]
    # v76: a status needs a sentence -- «provato» only with a case sentence that holds it (the link's, grounded), «non
    # risulta» only with one that says the contrary; what the extracted facts or a signal read with no such sentence
    # stays «da provare» (its why says so)
    if facts == PROVATO and not ground and not sure_con:
        trigger = _trigger(req, by_id)                  # the sentence that brought the fact (a precisazione's too)
        if trigger:
            req["supporting"] = list(req["supporting"]) + [trigger["id"]]
            status, why = PROVATO, f"lo dice il caso ({trigger['id']}: la frase da cui viene il fatto)"
        else:
            status, why = DA_PROVARE, "i fatti estratti lo indicano, ma nessuna frase del caso lo sostiene"
    elif facts == NON_RISULTA and not sure_con:
        status, why = DA_PROVARE, "i fatti estratti lo escludono, ma nessuna frase del caso lo dice"
    elif req.get("kind") not in ("requisito", "procedibilita"):
        status, why = (PROVATO, f"lo dice il caso ({ground[0]['id']})") if facts == PROVATO and ground else (
            (NON_RISULTA, f"il caso dice il contrario ({sure_con[0]['id']})") if facts == NON_RISULTA and sure_con
            else (DA_PROVARE, "comma intero: nessuna frase del caso lo decide"))
    elif sure_con:
        status, why = NON_RISULTA, f"il caso dice il contrario ({sure_con[0]['id']})"
    elif ground:
        status, why = PROVATO, f"lo dice il caso ({ground[0]['id']}, {ground[0].get('certainty') or ''})".replace(", )", ")")
    elif alleged:
        status, why = DA_PROVARE, f"lo afferma solo un'allegazione ({alleged[0]['id']}): non è un fatto certo"
    elif sup:
        how = sup[0].get("source") if sup[0].get("source") == "sistema" else sup[0].get("certainty")
        status, why = DA_PROVARE, f"nessuna frase del caso a sostegno, solo {how} ({sup[0]['id']})"
    else:
        status, why = DA_PROVARE, "nessun fatto del caso" + (f"; manca: {req['missing']}" if req.get("missing") else "")
    before = status                     # v79.11 (6): the status before the element's rule (grading_rule)
    if status == PROVATO:               # v79.8 (C3): every part of the condition (v79.11: element_rule's condition)
        status, why = condition_check(req, by_id, status, why)
    req["status"], req["why"] = status, why
    if req.get("branches"):             # v79.6 (item 4): one element with branches
        settle_branches(req, sup, sure_con)
        if req.get("schema_kind") and req["status"] == PROVATO:     # v79.11: a branch never overturns the C3 check
            req["status"], req["why"] = condition_check(req, by_id, req["status"], req["why"])
    polarity(req, by_id, sure_con)      # v79.9 (A1, A2): a mental element, a duty element; v79.11: a termine
    negated_check(req, by_id)           # v79.12.1: an absence-phrased element only on a fact stating the absence
    if req.get("schema_kind"):          # v79.11 (6): a schema element -- its rule recorded and logged
        rule, source = element_rule(req)
        req["grading_rule"] = {"rule": rule, "source": source, "before": before, "after": req["status"]}
        _log_rule(req, rule, source, before)
    contested_mark(req, by_id, base_sup)    # v79.14.3 (item 1): an element only another party's allegation supports
    from . import applicability                 # v79.13.0: a «dipende» the applicability judgment marked -- «da
    if req.get("fit_open") and applicability.enabled():         # provare» until the user answers the question,
        from .applicability import FIT_OPEN_WHY                 # whatever the link says (review: the flag on only)
        req["status"], req["why"] = DA_PROVARE, FIT_OPEN_WHY.format(question=req["fit_open"])


# v79.14.3 (item 1): allegations never integrate an element. An element whose supporting sentences (the link's, before
# v60t moved a named counterparty's to the contradiction path, and the ones settle added) are ALL allegations of a party
# other than the client (§2's «allegazione» with its speaker: interactive_flow.alleging_party, «alleged_by») is
# «contestato»: its status «da provare» (never «provato»: it counts as no proven element anywhere -- §1/§9's «x elementi
# su y», the thesis scoring -- and is contestable as every «da provare» one), req["contested_by"] the party,
# req["contested_ids"] the allegations. A certain fact bearing on it among them (mixed support), a contradiction that
# excludes it («non risulta»), an allegation of the client's own (his statement, even against himself) or one with no
# speaker: unchanged.
CONTESTATO = "contestato"
CONTESTED_WHY = "lo afferma solo un'allegazione {parte} ({ids}): contestato"     # my draft


def contested(req: Optional[Dict[str, Any]]) -> bool:
    """v79.14.3 (item 1): the element is «contestato» -- «da provare», resting only on another party's allegation."""
    from .interactive_flow import DA_PROVARE
    return isinstance(req, dict) and bool(req.get("contested_by")) and (req.get("status") or DA_PROVARE) == DA_PROVARE


def of_party(party: str) -> str:
    """«di <parte>» with its articled preposition («della società», «di Carla Tasso»)."""
    from . import party_names
    who = " ".join(str(party or "").split())
    if _ART_LEAD_RE.match(who):
        who = who[:1].lower() + who[1:]
    return party_names.contract(f"di {who}").strip()


_ART_LEAD_RE = re.compile(r"^(?:Il|Lo|La|I|Gli|Le|L['’])(?:\s|(?<=['’]))")


def contested_mark(req: Dict[str, Any], by_id: Dict[str, Dict[str, Any]], base_sup: Sequence[str] = ()) -> None:
    """v79.14.3 (item 1), in place at the end of settle: the «contestato» mark (see CONTESTATO) set or cleared."""
    from .interactive_flow import DA_PROVARE, NON_RISULTA
    req.pop("contested_by", None)
    req.pop("contested_ids", None)
    if req.get("kind", "requisito") != "requisito" or req.get("status") == NON_RISULTA:
        return
    ids = list(dict.fromkeys([str(i) for i in base_sup or []] + [str(i) for i in req.get("supporting") or []]))
    sup = [by_id[i] for i in ids if i in by_id]
    if not sup or any(certain_fact(s, req.get("text") or "") for s in sup):
        return
    if not all(s.get("alleged_by") for s in sup):
        return
    req["status"] = DA_PROVARE
    req["contested_by"] = str(sup[0]["alleged_by"])
    req["contested_ids"] = [s["id"] for s in sup]
    req["why"] = CONTESTED_WHY.format(parte=of_party(req["contested_by"]), ids=", ".join(req["contested_ids"]))


def condition_check(req: Dict[str, Any], by_id: Dict[str, Dict[str, Any]], status: str, why: str) -> Tuple[str, str]:
    """v79.8 (C3): a «provato» condition stays so only when a supporting fatto certo holds every part (condition_rule's
    parts; req["parts"] records which); else «da provare». v79.11: also run after settle_branches on a schema element,
    so a branch found never raises a condition with a missing part back to «provato». Only lowers a status."""
    from .interactive_flow import DA_PROVARE
    parts = condition_rule(req)[0]
    if not parts:
        return status, why
    facts = [by_id[i] for i in req.get("supporting") or [] if i in by_id and certain_fact(by_id[i], req.get("text") or "")]
    held = {p: any(element_nouns(p) & element_nouns(x.get("text") or "") for x in facts) for p in parts}
    req["parts"] = held
    missing = [p for p, ok in held.items() if not ok]
    if missing:
        return DA_PROVARE, f"provata solo una parte della condizione: manca «{missing[0][:80]}»"
    return status, why


# v79.9 (A1): a culpability or state-of-mind element -- integrato only by a fact that states the state of mind or its
# recognised indicators (an admission, a warning ignored, a false statement knowingly made, a precautionary rule broken);
# a conduct fact («spinge», «colpisce», «vende») leaves it «da provare», never integrato, never escluso
_MENTAL_RE = re.compile(r"\b(?:per\s+colpa|colpos[oaie]|con\s+dolo|dolo\b|dolos[oaie]|mala\s+fede|malafede|buona\s+fede|"
                        r"negligen\w*|imprudenz\w*|imprudent\w*|imperizi\w*|consapevol\w*|intenzion\w*|scientemente)\b",
                        re.IGNORECASE)
_MIND_FACT_RE = re.compile(r"\b(?:ammett\w*|ammess[oa]\s+di|ha\s+ammesso|ammissione|confess\w*|riconosc\w*\s+di|"
                           r"riconosciuto\s+di|sapev\w*|sapendo|consapevol\w*|a\s+conoscenza|conoscev\w*|"
                           r"intenzional\w*|di\s+proposito|deliberatamente|volontariamente|apposta|con\s+l['’]intento|"
                           r"allo\s+scopo\s+di|premeditat\w*|avvertit[oaie]|avvisat[oaie]|diffidat[oaie]|ignorando|"
                           r"nonostante\s+(?:\w+\s+){0,2}?(?:avvis|avvert|diffid|segnalaz|richiam|divieto|divieti)\w*|"
                           r"malgrado\s+(?:\w+\s+){0,2}?(?:avvis|avvert|diffid|segnalaz|richiam|divieto)\w*|falsamente|"
                           r"mentendo|ha\s+mentito|il\s+falso|pur\s+sapendo|ben\s+sapendo|in\s+violazione|violando|"
                           r"senza\s+rispettare|col\s+rosso|con\s+il\s+rosso|semaforo\s+rosso|eccesso\s+di\s+velocit\w*|"
                           r"distratt\w*|disattent\w*|ubriac\w*|in\s+stato\s+di\s+ebbrezza|ignorav\w*|non\s+sapev\w*|"
                           r"in\s+buona\s+fede|in\s+mala\s+fede|involontariamente|accidentalmente|inavvertitamente|"
                           r"per\s+errore|per\s+sbaglio)\b", re.IGNORECASE)
# v79.9 (A2): a duty-phrased element -- graded on its breach: the fact that breaches the duty integrates it (the
# claimant's cause of action), never excludes it; a fact denying the duty itself (no contract, no sale) still can
_DUTY_RE = re.compile(r"\b(?:sia|siano)\s+(?:immun\w*|esent\w*|priv[oaie]|conform\w*|idone\w*)\b|\bdev(?:e|ono)\s+\w+|"
                      r"\b(?:è|sono)\s+tenut[oaie]\b|\bobbligat[oaie]\s+a\b|\bha\s+l['’]obbligo\b|\bgarantire\s+che\b",
                      re.IGNORECASE)
_NO_DUTY_RE = re.compile(r"\b(?:nessun\w*\s+(?:contratto|accordo|vendita|obbligo|incarico)|non\s+(?:è|era|fu)\s+(?:mai\s+)?"
                         r"(?:stat[oa]\s+)?(?:stipulat|conclus|firmat|sottoscritt)\w*|non\s+esiste\w*)", re.IGNORECASE)
MENTAL_CONDUCT_WHY = "elemento psicologico: i fatti narrano una condotta, non lo stato soggettivo"
MENTAL_OPEN_WHY = "elemento psicologico: nessun fatto ne dice lo stato soggettivo"
DUTY_BREACH_WHY = "l'obbligo risulta violato: la violazione lo integra"


def mental_element(text: str) -> bool:
    return bool(_MENTAL_RE.search(text or ""))


def states_mind(sentence: Dict[str, Any]) -> bool:
    return bool(_MIND_FACT_RE.search(sentence.get("text") or ""))


def duty_element(text: str) -> bool:
    return bool(_DUTY_RE.search(text or "")) and not mental_element(text)


# v79.11 (6): the rule grading an element -- one selector (element_rule) for settle, polarity and the counters
RULES = ("mental", "duty", "condition", "termine", "branches", "plain")
RULE_SOURCES = ("schema", "regex", "both")
# v79.10.4 (D1): a time condition graded on the durations the facts give (termine_check) -- it replaces v79.11's stand-in
TERMINE_OPEN_WHY = "termine: nessuna durata accertata nei fatti"
TERMINE_INSIDE_WHY = "termine rispettato: {span} nei fatti ({sid}), nel termine «{term}»"
TERMINE_BEYOND_WHY = "termine superato: {span} nei fatti ({sid}), oltre il termine «{term}»"
TERMINE_UNREAD_WHY = "termine: la sua durata non si legge nel testo dell'elemento"
INSIDE, BEYOND = "inside", "beyond"
# v79.11 (6): a fact stating a date or a duration -- digits, giorni/mesi/anni, a month, «entro», «dal … al …»; v79.10.4
# (D1): read only for a schema «termine» whose own text states no duration
_DATED_RE = re.compile(r"\d|\b(?:giorn[oi]|mes[ei]|ann[oi]|settiman[ae]|gennaio|febbraio|marzo|aprile|maggio|giugno|"
                       r"luglio|agosto|settembre|ottobre|novembre|dicembre|entro)\b|\bdal\b.{1,80}?\bal\b",
                       re.IGNORECASE)
_MONTHS = ("gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno", "luglio", "agosto", "settembre", "ottobre",
           "novembre", "dicembre")
_UNIT_DAYS = {"giorn": 1.0, "settiman": 7.0, "mes": 30.4375, "ann": 365.25}
_UNIT = r"(giorn|settiman|mes|ann)\w*"


_NUMBER_CACHE: Dict[str, Any] = {}


def _number_words() -> Dict[str, int]:
    """The I7 durations' words («otto», «sessanta», «vent'») and every number word from one to centonovantanove
    («tredici», «cinquanta», «venticinque», «settantadue»)."""
    if "words" not in _NUMBER_CACHE:
        from .answer_invariants import _NUMBERS
        units = ("", "uno", "due", "tre", "quattro", "cinque", "sei", "sette", "otto", "nove")
        teens = ("dieci", "undici", "dodici", "tredici", "quattordici", "quindici", "sedici", "diciassette", "diciotto",
                 "diciannove")
        tens = {20: "venti", 30: "trenta", 40: "quaranta", 50: "cinquanta", 60: "sessanta", 70: "settanta",
                80: "ottanta", 90: "novanta"}
        base: Dict[str, int] = {w: i for i, w in enumerate(units) if w}
        base.update({w: 10 + i for i, w in enumerate(teens)})
        for t, word in tens.items():
            base[word] = t
            for u in range(1, 10):
                head = word[:-1] if u in (1, 8) else word           # ventuno, ventotto
                base[head + units[u]] = t + u
                if u == 3:
                    base[head + "tré"] = t + u
        out = dict(base)
        out["cento"] = 100
        for w, n in base.items():
            out[("cent" if w.startswith("o") else "cento") + w] = 100 + n
        out.update(_NUMBERS)
        _NUMBER_CACHE["words"] = out
        _NUMBER_CACHE["re"] = r"(\d+|" + "|".join(sorted((re.escape(w) for w in out), key=len, reverse=True)) + r")"
    return _NUMBER_CACHE["words"]


def _num_re() -> str:
    _number_words()
    return _NUMBER_CACHE["re"]


def _number(word: str) -> Optional[int]:
    w = str(word or "").lower()
    return int(w) if w.isdigit() else _number_words().get(w)


# the term an element states: «entro dieci anni», «entro due anni dalla consegna», «entro il termine di sessanta giorni
# dalla scoperta», «nel corso di dieci anni dal compimento», «non oltre trenta giorni»; its anchor the noun after «dal»
_TERM_OPEN = r"\b(?:entro|nel\s+corso\s+di|non\s+oltre|nel\s+termine\s+(?:perentorio\s+)?di)\s+(?:il\s+termine\s+di\s+)?"
_ANCHOR_RE = re.compile(r"^\s*,?\s*(?:a\s+decorrere\s+)?(?:dal|dalla|dallo|dall['’]|dai|dalle|dagli|da)\s*"
                        r"(?:(?:giorno|momento|data)\s+(?:della|del|dello|dell['’]|dei|delle|in\s+cui)\s*)?"
                        r"(?:(?:sua|suo|loro)\s+)?([a-zà-ù]{4,})", re.IGNORECASE)


def term_of(text: str) -> Optional[Dict[str, Any]]:
    """v79.10.4 (D1): the term an element states -- {n, unit, days, anchor (the stem of the event it runs from, "" for
    none), phrase} -- or None: «entro dieci anni», «entro due anni dalla consegna», «entro sessanta giorni dalla
    scoperta»; «entro un anno» is one year."""
    m = re.search(_TERM_OPEN + _num_re() + r"\s+" + _UNIT, text or "", re.IGNORECASE)
    if not m:
        return None
    n = _number(m.group(1))
    if not n:
        return None
    unit = m.group(2).lower()
    a = _ANCHOR_RE.match((text or "")[m.end():])
    anchor = a.group(1).lower()[:5] if a else ""
    phrase = " ".join(((text or "")[m.start():m.end()] + (a.group(0) if a else "")).split()).strip(" ,")
    return {"n": n, "unit": unit, "days": n * _UNIT_DAYS[unit], "anchor": anchor, "phrase": phrase,
            "negated": _negated((text or "")[:m.start()]), "before": (text or "")[:m.start()]}


# the act the term governs negated: «(se) non denunzia i vizi entro otto giorni» -- the decadence is integrated by the
# late act; never «se non entro» (only within), nor a «non» of another clause or another verb before the term
_TERM_NOT_RE = re.compile(r"\bnon\s+((?:[^\s,;:]+\s+){0,8})$", re.IGNORECASE)
_TERM_NOT_STOP_RE = re.compile(r"\b(?:che|cui|chi|quando|qualora|se|deve|devono|va|vanno|può|possono|è\s+tenut\w*|"
                               r"sono\s+tenut\w*)\b", re.IGNORECASE)


def _negated(before: str) -> bool:
    m = _TERM_NOT_RE.search(before or "")
    return bool(m) and bool(m.group(1).strip()) and not _TERM_NOT_STOP_RE.search(m.group(1))


def termine_rule(req: Dict[str, Any]) -> str:
    """v79.10.4 (D1): the termine rule's selection -- a schema element by its kind alone («termine»: "schema"), any other
    element by its text stating a term (term_of: "regex"); "" for neither."""
    if req.get("schema_kind"):
        return "schema" if req.get("schema_kind") == "termine" else ""
    return "regex" if term_of(req.get("text") or "") else ""


# the durations the facts give: a stated span («oltre sei anni», «dopo tre mesi», «da otto anni», «due mesi fa») and the
# dated events («il 3 marzo 2019», «03/03/2019», «marzo 2019», «nel 2015»)
_SPAN_BEFORE = (r"\b(oltre|più\s+di|almeno|ben|dopo(?:\s+circa)?|a\s+distanza\s+di|circa|quasi|meno\s+di|nel\s+giro\s+di|"
                r"entro|da|per|trascors[oiae]|decors[oiae]|passat[oiae]|(?:sono|erano|siano)\s+(?:passat|trascors|"
                r"decors)\w*)\s+(?:(?:circa|ben|oltre|più\s+di|quasi)\s+)?")
_SPAN_AFTER = r"\s+(fa|dopo|più\s+tardi)\b"
_LOWER = ("oltre", "più di", "almeno", "ben")
_UPPER = ("meno di", "quasi", "nel giro di", "entro")
_STRICT = ("oltre", "più di")
_STATE_WORDS = ("da", "per")         # «in attività da quindici anni», «per tre anni»: how long a state lasts
_AGO_RE = re.compile(r"\s+fa\b", re.IGNORECASE)     # v79.10.7: «oltre un anno fa» -- a before-word span counted to today
# a sentence stipulating a deadline («il contratto prevedeva il compimento entro due anni»): its «entro» is no time elapsed
_STIPULATION_RE = re.compile(r"\b(?:prevede\w*|prevista|previst[oie]|dovev\w*|dovr[àaeo]\w*|deve|devono|pattuit\w*|"
                             r"stabilit\w*|concordat\w*|convenut\w*|impegn\w*|termine|scadenz\w*|obblig\w*)\b",
                             re.IGNORECASE)


def spans_of(text: str) -> List[Dict[str, Any]]:
    """v79.10.4 (D1): the spans a sentence states -- [{days, bound ("exact" | "lower" | "upper"), strict, phrase,
    word, ago}]; a number with a unit and no span word around it («un bambino di dieci anni») is no span, nor an
    «entro» in a sentence stipulating a deadline («il contratto prevedeva il compimento entro due anni»). v79.10.7:
    «ago» -- a span counted to today («tre mesi fa», «oltre un anno fa»): it dates an event, never the time between
    two (term_durations)."""
    out: List[Dict[str, Any]] = []
    num = _num_re()
    stipulated = bool(_STIPULATION_RE.search(text or ""))
    taken: List[Tuple[int, int]] = []
    for m in re.finditer(_SPAN_BEFORE + num + r"\s+" + _UNIT, text or "", re.IGNORECASE):
        n = _number(m.group(2))
        if not n:
            continue
        word = " ".join(m.group(1).lower().split())
        if word == "entro" and stipulated:
            continue
        bound = "lower" if word in _LOWER else ("upper" if word in _UPPER else "exact")
        ago = bool(_AGO_RE.match((text or "")[m.end():]))
        out.append({"days": n * _UNIT_DAYS[m.group(3).lower()], "bound": bound, "strict": word in _STRICT,
                    "phrase": " ".join(m.group(0).split()), "word": word, "ago": ago, "unit": m.group(3).lower()})
        taken.append((m.start(2), m.end()))
    for m in re.finditer(r"\b" + num + r"\s+" + _UNIT + _SPAN_AFTER, text or "", re.IGNORECASE):
        n = _number(m.group(1))
        if any(lo < m.end(2) and m.start(1) < hi for lo, hi in taken):     # v79.10.7: «oltre un anno fa» is one span
            continue
        if n and not any(m.group(0) in x["phrase"] or x["phrase"] in m.group(0) for x in out):
            out.append({"days": n * _UNIT_DAYS[m.group(2).lower()], "bound": "exact", "strict": False,
                        "phrase": " ".join(m.group(0).split()), "word": m.group(3).lower(),
                        "ago": m.group(3).lower() == "fa", "unit": m.group(2).lower()})
    return out


_FULL_DATE_RE = re.compile(r"\b(\d{1,2})(?:°|º)?\s+(" + "|".join(_MONTHS) + r")\s+((?:19|20)\d{2})\b", re.IGNORECASE)
_NUM_DATE_RE = re.compile(r"\b(\d{1,2})[/.-](\d{1,2})[/.-]((?:19|20)\d{2})\b")
_MONTH_DATE_RE = re.compile(r"\b(" + "|".join(_MONTHS) + r")\s+(?:del\s+)?((?:19|20)\d{2})\b", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(?:nel|nell['’]\w+\s+del|del|dal|al|fino\s+al|entro\s+il|anno)\s+((?:19|20)\d{2})\b",
                      re.IGNORECASE)


def dates_of(text: str) -> List[Tuple[Any, Any]]:
    """v79.10.4 (D1): the dates a sentence states, each as its (earliest, latest) day: a full date one day, «marzo 2019»
    its month, «nel 2015» its year; a number in an article or a ruling («n. 92/2012») is no date."""
    import calendar
    from datetime import date
    out: List[Tuple[Any, Any]] = []
    rest = text or ""

    def blank(m):
        return " " * (m.end() - m.start())
    for m in _FULL_DATE_RE.finditer(rest):
        with contextlib.suppress(ValueError):
            d = date(int(m.group(3)), _MONTHS.index(m.group(2).lower()) + 1, int(m.group(1)))
            out.append((d, d))
    rest = _FULL_DATE_RE.sub(blank, rest)
    for m in _NUM_DATE_RE.finditer(rest):
        with contextlib.suppress(ValueError):
            d = date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
            out.append((d, d))
    rest = _NUM_DATE_RE.sub(blank, rest)
    for m in _MONTH_DATE_RE.finditer(rest):
        y, mo = int(m.group(2)), _MONTHS.index(m.group(1).lower()) + 1
        out.append((date(y, mo, 1), date(y, mo, calendar.monthrange(y, mo)[1])))
    rest = _MONTH_DATE_RE.sub(blank, rest)
    for m in _YEAR_RE.finditer(rest):
        y = int(m.group(1))
        out.append((date(y, 1, 1), date(y, 12, 31)))
    return out


def _after(start: Any, n: int, unit: str) -> Any:
    """The day a term of n units from start ends (calendar months and years; the day clamped to the month's last)."""
    import calendar
    from datetime import timedelta
    if unit in ("giorn", "settiman"):
        return start + timedelta(days=n * int(_UNIT_DAYS[unit]))
    months = n * (12 if unit == "ann" else 1)
    y, m = divmod(start.month - 1 + months, 12)
    year, month = start.year + y, m + 1
    return start.replace(year=year, month=month, day=min(start.day, calendar.monthrange(year, month)[1]))


def _span_verdict(span: Dict[str, Any], term: Dict[str, Any]) -> str:
    """inside / beyond / "" -- a stated span against the term: exact by its figure; a lower bound («oltre sei anni»)
    beyond only from the term up (the stated figure inside the term is inside); an upper bound inside only within it."""
    d, t = span["days"], term["days"]
    if span["bound"] == "lower":
        return BEYOND if d > t or (d >= t and span["strict"]) else INSIDE
    if span["bound"] == "upper":
        return INSIDE if d <= t else ""
    return INSIDE if d <= t else BEYOND


def _has_stem(text: str, stem: str) -> bool:
    return bool(stem) and bool(re.search(r"\b" + re.escape(stem), text or "", re.IGNORECASE))


def term_durations(term: Dict[str, Any], req: Dict[str, Any], by_id: Dict[str, Dict[str, Any]]
                   ) -> List[Dict[str, Any]]:
    """v79.10.4 (D1): the durations of the facts read against the term -- [{verdict, span, sid}]: the spans stated in
    the element's own linked sentences, and in the certain facts naming the term's anchor (the term with none: the
    certain facts naming the act it governs -- every one for a bare term --, never a state's length «da/per N anni»
    outside the linked ones); the dated events from the anchor's date (the term with none: the earliest) to the date of
    the act (the anchor's own sentence, a linked fact, a fact naming the act): an unrelated later date never measures
    it. A negated act («non denunzia …
    entro otto giorni») turns the verdict (verdict inside: the element integrated). Only fatti certi count. v79.10.7: a
    span counted to today («tre mesi fa») is read only by ago_durations; with the act's date not found the dated
    events give nothing (an unrelated later date measures no term: «da provare»)."""
    facts = [s for s in by_id.values() if certain_fact(s, req.get("text") or "")]
    linked = set(req.get("supporting") or []) | set(req.get("contradicting") or []) | \
        set((req.get("link") or {}).get("supporting") or []) | set((req.get("link") or {}).get("contradicting") or [])
    anchor = term.get("anchor") or ""
    # the act the term governs, by the element's own nouns («denunzia», «azione»), the term's words aside
    act = element_nouns(req.get("text") or "") - element_nouns(term.get("phrase") or "") - {anchor}
    flip = {INSIDE: BEYOND, BEYOND: INSIDE} if term.get("negated") else {}
    out: List[Dict[str, Any]] = []
    for s in facts:
        text = s.get("text") or ""
        if anchor:
            read = s["id"] in linked or _has_stem(text, anchor)
        else:       # no anchor: the linked facts, and those of the act (any, for a bare term) -- never a state's length
            read = s["id"] in linked or (bool(act & element_nouns(text)) if act else True)
        if not read:
            continue
        for sp in spans_of(text):
            if sp.get("ago"):           # v79.10.7: counted to today -- read below (ago_durations), never here
                continue
            if not anchor and s["id"] not in linked and sp.get("word") in _STATE_WORDS:
                continue
            verdict = _span_verdict(sp, term)
            if verdict:
                out.append({"verdict": flip.get(verdict, verdict), "span": sp["phrase"], "sid": s["id"]})
    out += [{**x, "verdict": flip.get(x["verdict"], x["verdict"])} for x in ago_durations(term, facts)]
    dated = [(lo, hi, s["id"]) for s in facts for lo, hi in dates_of(s.get("text") or "")]
    if len(dated) >= 2:
        starts = [d for d in dated if _has_stem(by_id[d[2]].get("text") or "", anchor)] if anchor else \
            [min(dated, key=lambda d: d[0])]
        for lo, hi, sid in starts:
            later = [d for d in dated if d[0] > hi]
            if not later:                       # the anchor's date is the latest: no span after it
                continue
            # the act's dates: the start's own sentence, the linked facts, the facts naming the act (its nouns the
            # start's sentence does not share); found, each is measured. v79.10.7 (the user's ruling on D1, review):
            # the act's date not found, no later date measures the term -- neither «beyond» nor «inside» (an unrelated
            # later date is no act): the element stays «da provare»
            own = act - element_nouns(by_id[sid].get("text") or "")
            ends = [d for d in later if d[2] == sid or d[2] in linked
                    or (own and own & element_nouns(by_id[d[2]].get("text") or ""))]
            for e_lo, e_hi, e_sid in ends:
                span = f"dal {lo.strftime('%d/%m/%Y') if lo == hi else lo.year} al " \
                       f"{e_hi.strftime('%d/%m/%Y') if e_lo == e_hi else e_hi.year}"
                if e_hi <= _after(lo, term["n"], term["unit"]):
                    out.append({"verdict": flip.get(INSIDE, INSIDE), "span": span, "sid": e_sid})
                elif e_lo > _after(hi, term["n"], term["unit"]):
                    out.append({"verdict": flip.get(BEYOND, BEYOND), "span": span, "sid": e_sid})
    return out



# v79.10.7: a sentence's clauses for the spans counted to today -- «scoperto tre mesi fa e denunciato un mese fa»
_AGO_CLAUSE_SPLIT = re.compile(r"[,;:]|\s+(?:e|ed|ma|mentre|poi|quindi)\s+", re.IGNORECASE)
# v79.10.7: the stems that never name the act a term governs -- the parties by their role («il compratore deve
# denunziare»: the act is «denunziare», never «compratore») and the verbs of duty or doing («deve», «va fatta»)
_NOT_ACT_STEMS = frozenset({"compr", "acqui", "vendi", "alien", "condu", "inqui", "locat", "commi", "appal", "lavor",
                            "dator", "credi", "debit", "clien", "attor", "conve", "imput", "danne", "offes",
                            "parte", "titol", "propr", "mutua", "assic", "deve", "devon", "dovrà", "fatta",
                            "fatto", "fatte", "fatti", "esser", "stato", "stata", "tenut", "pena", "decad"})
# v79.10.7 (review): the duty or passive word before the act's verb («deve denunziare», «va proposto», «si esercita»,
# «è denunziato»)
_ACT_MODAL_RE = re.compile(r"\b(?:è\s+tenut\w*\s+a|sono\s+tenut\w*\s+a|deve|devono|dovr\w*|va|vanno|può|possono|si|è|"
                           r"sono|viene|vengono|venga|vengano|sia|siano)\b", re.IGNORECASE)
# v79.10.7 (review): the nouns that name an act a term governs («la denunzia va fatta», «il ricorso va proposto»)
_ACT_NOUN_STEMS = frozenset({"denun", "ricor", "azion", "oppos", "impug", "quere", "recla", "istan", "richi", "doman",
                             "comun", "disde", "reces", "dichi", "eccez", "appel", "conte", "diffi", "notif", "costi"})
# v79.10.7 (review): a clause where the act is denied, not yet done or only intended -- it dates no act
_ACT_NOT_DONE_RE = re.compile(r"\b(?:non|mai|ancora|vuole|vogliono|vorrebbe|vorrebbero|intende\w*|intenzione|"
                              r"potrebbe|potrebbero|dovrebbe|dovrebbero|dovr\w*|deve|devono|può|possono)\b",
                              re.IGNORECASE)
# v79.10.7 (review): a clause stating the act as done -- a compound past («ha denunciato», «è stato depositato») or a
# passato remoto («denunciò») -- never a state («la denuncia è pronta») nor an intention
_ACT_DONE_RE = re.compile(r"\b(?:ha|hanno|abbia|abbiano|aveva|avevano|è|sono|era|erano|fu|furono|venne|vennero|"
                          r"stat[oaie])\s+(?:\w+\s+){0,2}?\w+(?:at[oaie]|ut[oaie]|it[oaie]|ost[oaie]|ess[oaie]|"
                          r"ett[oaie])\b|"
                          r"^\s*(?:\w+\s+){0,2}?\w+(?:at[oaie]|ut[oaie]|it[oaie]|ost[oaie])\b|\b\w{3,}[òì]\b",
                          re.IGNORECASE)
# v79.10.7 (review): the days a unit can span on the calendar -- a span to today in months or years is approximate
_UNIT_RANGE = {"giorn": (1.0, 1.0), "settiman": (7.0, 7.0), "mes": (28.0, 31.0), "ann": (365.0, 366.0)}


def _word_stems(text: str) -> set:
    """v79.10.7: the five-letter stems of a text's words of four letters or more, verbs included, stopwords aside."""
    from .topic_step import _STOPWORDS
    return {w[:5] for w in re.findall(r"[a-zà-ü]{4,}", str(text or "").lower()) if w not in _STOPWORDS}


def _content_stems(text: str) -> List[str]:
    from .topic_step import _STOPWORDS
    return [w[:5] for w in re.findall(r"[a-zà-ü]{4,}", str(text or "").lower())
            if w not in _STOPWORDS and w[:5] not in _NOT_ACT_STEMS]


def act_stems(before: str, anchor: str = "") -> set:
    """v79.10.7 (review): the act a term governs, from the element's words before the term (term_of's «before») -- the
    verb after the duty or passive word («il compratore deve denunziare il vizio», «il vizio è denunziato»:
    «denunziare») and a noun naming an act (_ACT_NOUN_STEMS: «la denunzia va fatta», «il ricorso va proposto»); with
    neither, the first word that is no party's role nor a verb of duty («il compratore denunzia il vizio»). The act's
    object («il vizio») never names it: an event about the vizio is no denunzia."""
    words = [x for x in _content_stems(before) if x != anchor]
    out = {x for x in words if x in _ACT_NOUN_STEMS}
    m = _ACT_MODAL_RE.search(before or "")
    if m:
        out |= {x for x in _content_stems((before or "")[m.end():])[:1] if x != anchor}
    return out or set(words[:1])


def _span_range(sp: Dict[str, Any]) -> Tuple[float, float]:
    """The days a span counted to today can mean on the calendar: «tre mesi fa» 84-93 days, «dieci giorni fa» 10."""
    unit = sp.get("unit") or ""
    if unit not in _UNIT_RANGE:
        return sp["days"], sp["days"]
    n = sp["days"] / _UNIT_DAYS[unit]
    return n * _UNIT_RANGE[unit][0], n * _UNIT_RANGE[unit][1]


def ago_durations(term: Dict[str, Any], facts: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """v79.10.7 (the user's ruling on D1): a span counted to today («scoperto tre mesi fa») dates the event of its
    clause, not the time between the term's starting event and the act the term governs -- so it establishes a duration
    only between those two. A clause naming the term's anchor («la scoperta è avvenuta tre mesi fa») dates the start; a
    clause names the act (act_stems: its verb or its noun, never its object -- an event «del vizio» is no denunzia) only
    when it states it as done (_ACT_DONE_RE: «ha denunciato», «è stato depositato») and neither denied, nor not yet
    done, nor only intended (_ACT_NOT_DONE_RE: «non ha mai denunciato», «vuole denunciare»).
      - Both dated to today («denunciato un mese fa»): the gap between the two read on the calendar (_span_range: a
        month 28-31 days) -- «inside» when even its longest reading is within the term, «beyond» when even its
        shortest exceeds it, nothing in between («tre mesi fa – un mese fa» against sixty days).
      - The act stated as done but undated: it lies between the start and today -- «inside» when even the start's
        longest reading is within the term (an upper bound); nothing otherwise, never «beyond».
      - The act not named as done (only the start dated, an event of the object, a denied or intended act): nothing --
        the element stays «da provare», never «escluso».
    A term with no anchor: nothing (its starting event is not named). Only fatti certi (facts); [{verdict, span, sid}]
    unflipped (term_durations turns them for a negated act)."""
    anchor = term.get("anchor") or ""
    if not anchor:
        return []
    clauses = [(c, s["id"]) for s in facts for c in _AGO_CLAUSE_SPLIT.split(s.get("text") or "") if c.strip()]
    starts = [(sp, sid, c) for c, sid in clauses if _has_stem(c, anchor) for sp in spans_of(c)
              if sp.get("ago") and sp["bound"] == "exact"]
    out: List[Dict[str, Any]] = []
    act = act_stems(term.get("before") or "", anchor)
    for sp, sid, start_clause in starts:
        shared = {x[:4] for x in _word_stems(start_clause)}
        own = {x for x in act if x[:4] not in shared}
        named = [(c, csid) for c, csid in clauses if not _has_stem(c, anchor) and own & _word_stems(c)
                 and _ACT_DONE_RE.search(c) and not _ACT_NOT_DONE_RE.search(c)]
        s_lo, s_hi = _span_range(sp)
        if named and s_hi <= term["days"]:      # the act done by today: within the start's span (an upper bound)
            out.append({"verdict": INSIDE, "span": sp["phrase"], "sid": sid})
        for c, csid in named:
            for ap in spans_of(c):
                if not ap.get("ago") or ap["bound"] != "exact":
                    continue
                a_lo, a_hi = _span_range(ap)
                gap_lo, gap_hi = s_lo - a_hi, s_hi - a_lo
                if gap_hi < 0:                  # the act before the start: no reading of the term
                    continue
                verdict = INSIDE if gap_hi <= term["days"] else (BEYOND if gap_lo > term["days"] else "")
                if verdict:
                    out.append({"verdict": verdict, "span": f"{sp['phrase']} – {ap['phrase']}", "sid": csid})
    return out


def _source(by_schema: bool, by_regex: bool) -> str:
    return "both" if by_schema and by_regex else ("schema" if by_schema else ("regex" if by_regex else ""))


def mental_rule(req: Dict[str, Any]) -> str:
    """v79.11 (6): the 79.9-A1 check's selection -- "schema" (kind «elemento soggettivo»), "regex" (mental_element),
    "both", "" for neither: the union (a schema adds the check, never removes it)."""
    return _source(req.get("schema_kind") == "elemento soggettivo", mental_element(req.get("text") or ""))


def condition_rule(req: Dict[str, Any]) -> Tuple[List[str], str]:
    """v79.11 (6): the 79.8-C3 per-part check's selection -- (the parts, source): a schema «condizione» joined by «e»
    with two parts or more (schema_parti, else condition_parts of its text), or (as today) a d'ufficio condition whose
    text condition_parts splits; ([], "") for neither. The schema's parts first when both select it."""
    text = req.get("text") or ""
    schema: List[str] = []
    if req.get("schema_kind") == "condizione" and req.get("schema_connective") == "e":
        schema = [" ".join(str(p).split()) for p in req.get("schema_parti") or [] if str(p or "").strip()]
        if len(schema) < 2:
            schema = condition_parts(text)
    regex = condition_parts(text) if req.get("kind") == "procedibilita" else []
    return (schema or regex), _source(bool(schema), bool(regex))


def element_rule(req: Dict[str, Any]) -> Tuple[str, str]:
    """v79.11 (6): (the rule grading the element, its source) -- rule in RULES: "mental" (schema kind «elemento
    soggettivo» or mental_element), "duty" (duty_element, regex as v79.9), "condition" (condition_rule), "termine"
    (termine_rule), "branches" (req["branches"]), else "plain"; source "schema" / "regex" / "both". With no schema key
    on the element it selects exactly what v79.9's regexes do, a term aside. The kind only chooses which check runs:
    every check keeps its own validation, and none sets a status by itself -- the termine's durations aside. v79.10.4
    (D1): "termine" right after "mental", for a schema element of kind «termine» ("schema") and for any element without
    a schema kind whose text states a term (term_of: "regex") -- a term that is also a duty or a condition is graded on
    its durations."""
    mental = mental_rule(req)
    if mental:
        return "mental", mental
    termine = termine_rule(req)
    if termine:
        return "termine", termine
    if duty_element(req.get("text") or ""):
        return "duty", "regex"
    parts, source = condition_rule(req)
    if parts:
        return "condition", source
    schema = bool(req.get("schema_kind"))
    if req.get("branches"):
        return "branches", "schema" if schema else "regex"
    return "plain", "schema" if schema else "regex"


def termine_check(req: Dict[str, Any], by_id: Dict[str, Dict[str, Any]]) -> None:
    """v79.10.4 (D1, replacing v79.11's stand-in): an element stating a term (termine_rule) graded on the durations of
    the facts (term_durations) -- one inside the term «provato» (integrato), else one beyond it «non risulta» (escluso),
    none «da provare»: never «non risulta» while a fact gives a duration inside the term. The fact behind it joins the
    element's supporting (inside) or contradicting (beyond) sentences; req["termine"] {term, verdict, span, sid}. With no
    duration the computed terms keep a «provato» they decided (facts_status). A schema «termine» whose text states no
    duration keeps v79.11's check: «provato» only on a dated supporting fatto certo."""
    from .interactive_flow import DA_PROVARE, NON_RISULTA, PROVATO
    term = term_of(req.get("text") or "") or term_of(" ".join(str(p) for p in req.get("schema_parti") or []))
    if not term:
        if req.get("status") == PROVATO and req.get("facts_status") != PROVATO and not any(
                i in by_id and certain_fact(by_id[i], req.get("text") or "") and _DATED_RE.search(by_id[i].get("text") or "")
                for i in req.get("supporting") or []):
            req["status"], req["why"] = DA_PROVARE, TERMINE_UNREAD_WHY
        return
    found = term_durations(term, req, by_id)
    hit = next((x for x in found if x["verdict"] == INSIDE), None) or \
        next((x for x in found if x["verdict"] == BEYOND), None)
    req["termine"] = {"term": term["phrase"], "verdict": hit["verdict"] if hit else "", "span": hit["span"] if hit else "",
                      "sid": hit["sid"] if hit else ""}
    sup, con = list(req.get("supporting") or []), list(req.get("contradicting") or [])
    inside_why, beyond_why = (TERMINE_BEYOND_WHY, TERMINE_INSIDE_WHY) if term.get("negated") else \
        (TERMINE_INSIDE_WHY, TERMINE_BEYOND_WHY)        # a negated act («non denunzia … entro»): late integrates it
    if hit and hit["verdict"] == INSIDE:
        req["supporting"] = list(dict.fromkeys(sup + [hit["sid"]]))
        req["contradicting"] = [i for i in con if i != hit["sid"]]
        req["status"], req["why"] = PROVATO, inside_why.format(span=hit["span"], sid=hit["sid"], term=term["phrase"])
    elif hit:
        req["contradicting"] = list(dict.fromkeys(con + [hit["sid"]]))
        req["supporting"] = [i for i in sup if i != hit["sid"]]
        req["status"], req["why"] = NON_RISULTA, beyond_why.format(span=hit["span"], sid=hit["sid"],
                                                                   term=term["phrase"])
    elif not (req.get("status") == PROVATO and req.get("facts_status") == PROVATO):
        req["status"], req["why"] = DA_PROVARE, TERMINE_OPEN_WHY


def termine_counters(records: Dict[str, Dict[str, Any]], by_id: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """v79.10.4 (D1): the elements graded by the termine rule, by outcome {integrato, escluso, da provare}, and
    termine_excluded_inside -- one «non risulta» while a fact gives a duration inside its term (must be 0)."""
    from .interactive_flow import NON_RISULTA, PROVATO
    graded = {"integrato": 0, "escluso": 0, "da provare": 0}
    wrong = 0
    for rec in (records or {}).values():
        for r in list(rec.get("requisites") or []) + list(rec.get("procedibilita") or []):
            if element_rule(r)[0] != "termine":
                continue
            graded[{PROVATO: "integrato", NON_RISULTA: "escluso"}.get(r.get("status"), "da provare")] += 1
            term = term_of(r.get("text") or "")
            if term and r.get("status") == NON_RISULTA and any(
                    x["verdict"] == INSIDE for x in term_durations(term, r, by_id)):
                wrong += 1
    return {"termini_graded": graded, "termine_excluded_inside": wrong}


def _log_rule(req: Dict[str, Any], rule: str, source: str, before: str) -> None:
    """v79.11 (6): the element_rule event, ELEMENT_SCHEMAS only; never fails the grading."""
    try:
        from . import element_schemas
        if not element_schemas.enabled():
            return
        _log_event("element_rule", {"id": req.get("id"), "text": (req.get("text") or "")[:120], "rule": rule,
                                    "source": source, "before": before, "status": req.get("status"),
                                    "why": req.get("why")})
    except Exception:           # noqa: BLE001
        pass


def polarity(req: Dict[str, Any], by_id: Dict[str, Dict[str, Any]], sure_con: Sequence[Dict[str, Any]]) -> None:
    """v79.9 (A1, A2), in place after settle: a mental element «provato» only on a fatto certo stating the state of mind
    (else «da provare», marked mental_conduct when a conduct fact was its support), «non risulta» only on one stating the
    contrary; a duty element «non risulta» on a breach is «provato» on that breach (the sentences move to its support).
    v79.11 (6): the element's rule is element_rule's (a schema's «elemento soggettivo» is a mental element too).
    v79.10.4 (D1): a term graded on the facts' durations (termine_check). (D3): on the client's record (req["binds"],
    client_bindings) an element the counterparty must satisfy is never «non risulta» on a breach -- the breach integrates
    it, whatever its wording (a prohibition too) --; one the client must satisfy keeps the contradiction (no v79.9-A2
    conversion: the client's own duty contradicted is not met)."""
    from .interactive_flow import DA_PROVARE, NON_RISULTA, PROVATO
    if req.get("kind") not in ("requisito", "procedibilita"):
        return
    rule = element_rule(req)[0]
    if rule == "termine":               # v79.10.4 (D1): replaces v79.11's stand-in
        termine_check(req, by_id)
        return
    if rule == "mental":
        req["mental"] = True
        mind = [by_id[i] for i in req.get("supporting") or [] if i in by_id and certain_fact(by_id[i], req.get("text") or "")
                and states_mind(by_id[i])]
        if req["status"] == PROVATO and not mind:
            req["status"], req["why"], req["mental_conduct"] = DA_PROVARE, MENTAL_CONDUCT_WHY, True
        elif req["status"] == NON_RISULTA and not any(states_mind(s) for s in sure_con):
            req["status"], req["why"] = DA_PROVARE, MENTAL_OPEN_WHY
        return
    bound = req.get("binds")            # v79.10.4 (D3): whose element it is on the client's record
    if req["status"] == NON_RISULTA and (bound == COUNTERPARTY_BINDS or rule == "duty" and bound != CLIENT_BINDS):
        breach = [s for s in sure_con if not _NO_DUTY_RE.search(s.get("text") or "")]
        if breach:
            ids = [s["id"] for s in breach]
            req["supporting"] = list(dict.fromkeys(list(req.get("supporting") or []) + ids))
            req["contradicting"] = [i for i in req.get("contradicting") or [] if i not in ids]
            req["status"], req["why"], req["duty_breach"] = PROVATO, f"{DUTY_BREACH_WHY} ({ids[0]})", True
            if bound == COUNTERPARTY_BINDS:
                req["counterparty_breach"] = True


# v79.10.4 (D3): whose element it is on the client's record -- a duty or a prohibition (deontic) binds the party its
# subject names: the client when the subject is one of the client's own role words, else the counterparty (the norm the
# client invokes binds the other side); a term (D1) and a mental element are no one's here
COUNTERPARTY_BINDS, CLIENT_BINDS = "controparte", "cliente"
_PROHIBIT_RE = re.compile(r"\b(?:è\s+vietat\w*|è\s+fatto\s+divieto|è\s+proibit\w*|non\s+(?:può|possono|deve|devono)\b|"
                          r"è\s+tenut\w*\s+a\s+non\b|si\s+astien\w*|deve\s+astenersi)", re.IGNORECASE)
_DEONTIC_RE = re.compile(_DUTY_RE.pattern + "|" + _PROHIBIT_RE.pattern, re.IGNORECASE)
_NOT_ROLE = {"clien", "parte", "attor", "conve", "ricor", "resis", "assis", "imput", "indag", "perso", "offes", "civil",
             "avvoc", "sogge"}


def deontic(text: str) -> bool:
    """v79.10.4 (D3): a duty (v79.9's duty_element) or a prohibition («è vietato», «non può», «non deve»)."""
    return (duty_element(text) or bool(_PROHIBIT_RE.search(text or ""))) and not mental_element(text)


# the code's names for the same party («acquirente» is the «compratore» of art. 1470 ss.): a role word brings its group
_ROLE_SYNONYMS = ({"acqui", "compr"}, {"vendi", "alien"}, {"condu", "inqui"})


def role_stems(words: Sequence[str]) -> set:
    """v79.10.4 (D3): the client's role words as five-letter stems («acquirente», «compratore»), never the procedural
    ones (cliente, attore, convenuto, imputato …); each with its synonyms in the code (_ROLE_SYNONYMS: the «acquirente»
    is the «compratore», the «inquilino» the «conduttore»)."""
    out = set()
    for w in words or ():
        out |= {x for x in element_nouns(str(w or "")) if x not in _NOT_ROLE}
    for group in _ROLE_SYNONYMS:
        if out & group:
            out |= group
    return out


def binds(req: Dict[str, Any], client_roles: Sequence[str] = ()) -> str:
    """v79.10.4 (D3): on the client's record, who must satisfy the element -- CLIENT_BINDS when its deontic verb's
    subject (the words before it) names one of the client's role stems (role_stems), COUNTERPARTY_BINDS for another
    deontic element, "" for the rest (a term, a mental element, a descriptive one: as before) and for every element
    while the client's role is not known (no role stems: who must satisfy it cannot be told -- as before)."""
    text = req.get("text") or ""
    if not client_roles or req.get("kind", "requisito") != "requisito" or not deontic(text) \
            or element_rule(req)[0] in ("termine", "mental"):
        return ""
    m = _DEONTIC_RE.search(text)
    subject = text[:m.start()] if m else ""
    if set(client_roles or ()) & element_nouns(subject):
        return CLIENT_BINDS
    return COUNTERPARTY_BINDS


ACCUSED_PENAL = ("imputato", "indagato")


def client_bindings(records: Dict[str, Dict[str, Any]], client_roles: Sequence[str] = (),
                    by_id: Optional[Dict[str, Dict[str, Any]]] = None, role: str = "") -> Dict[str, Any]:
    """v79.10.4 (D3), in place once the sides are marked (posture.mark_sides): each element of a client's record (side
    «weapon») gets req["binds"] (binds); a threat's elements none, nor any record of an accused client (role imputato or
    indagato: his weapons are his own alternatives, norms he would violate, never the counterparty's). An element whose
    binding changed is settled again on by_id (none given: only marked) and its record's verdict recomputed. The counts
    {counterparty_elements,
    client_elements, counterparty_breaches, rebound}."""
    out = {"counterparty_elements": 0, "client_elements": 0, "counterparty_breaches": 0, "rebound": []}
    for label, rec in (records or {}).items():
        mine = rec.get("side") == "weapon" and role not in ACCUSED_PENAL
        changed = False
        for r in rec.get("requisites") or []:
            b = binds(r, client_roles) if mine else ""
            if b != (r.get("binds") or ""):
                changed = True
                if b:
                    r["binds"] = b
                else:
                    r.pop("binds", None)
                r.pop("counterparty_breach", None)
                if by_id and (r.get("link") or r.get("supporting") or r.get("contradicting")):
                    settle(r, by_id)
            out["counterparty_elements"] += b == COUNTERPARTY_BINDS
            out["client_elements"] += b == CLIENT_BINDS
            out["counterparty_breaches"] += bool(r.get("counterparty_breach"))
        if changed:
            rec["verdict"] = verdict(rec)
            out["rebound"].append(label)
    return out


# v79.12.1: a negated element -- one phrased as an absence («non vi è desistenza», «non ha autorizzato», «senza il
# consenso», «in mancanza del titolo») -- is integrato only by a fatto certo stating that absence; a fact stating the
# presence of the thing (the noun the trigger shares: «ha desistito», «ha autorizzato») or no fact leaves it «da
# provare». A prohibition («non può», «non deve»: deontic, v79.10.4) and a mental or a termine element keep their rules.
_ARTS = r"(?:il|lo|la|l['’]\s*|i|gli|le|un|una|uno|alcun\w*|previ[oaie]|su[oaie]|propri[oaie]|dett[oaie])"
_NEGATED_ELEMENT_RE = re.compile(
    r"\bnon\s+(?:vi|v['’]|ci|c['’])\s*(?:è|e|sia|siano|sono|era|fosse|sia\s+stat[oaie]|è\s+stat[oaie])\b|"
    r"\bnon\s+(?:ha|abbia|hanno|abbiano|aveva|avevano|avesse|avessero)\s+(?:(?:mai|previamente|espressamente)\s+)?"
    r"[a-zà-ü]+(?:at|ut|it|ess|ost)[oaie]\b|"
    r"\bnon\s+(?:sia|siano|è|sono|fosse|risult\w*)\s+(?:stat[oaie]\s+)?(?:autorizzat|consentit|prestat|rilasciat|"
    r"concess|dat|richiest|voluti?|dovut|giustificat|ricevut)\w*|"
    r"\bnon\s+(?:autorizzat|consentit|voluti?\b|richiest|dovut|giustificat|accettat|approvat)\w*|"
    r"\bsenza\s+(?:" + _ARTS + r"\s*)*[a-zà-ü]{4,}|"
    r"\b(?:in\s+)?(?:assenza|mancanza|difetto)\s+d(?:i|el|ello|ella|ell['’]|ei|egli|elle)\b", re.IGNORECASE)
_ABSENCE_RE = re.compile(r"\b(?:non|senza|mai|nessun\w*|né|alcun\w*|privo|priva|privi|prive|assenza|mancanza|"
                         r"mancav\w*|mancat\w*|assente|assenti|negat\w*|rifiutat\w*|rifiut\w*|dinieg\w*)\b",
                         re.IGNORECASE)
_THING_STOP = {"non", "senza", "vi", "ci", "è", "sia", "siano", "sono", "era", "fosse", "stato", "stata", "stati",
               "state", "ha", "abbia", "hanno", "abbiano", "aveva", "avevano", "avesse", "avessero", "mai", "il", "lo",
               "la", "i", "gli", "le", "un", "una", "uno", "di", "del", "dello", "della", "dei", "degli", "delle",
               "in", "assenza", "mancanza", "difetto", "previo", "previa", "suo", "sua", "suoi", "sue", "proprio",
               "propria", "alcun", "alcuno", "alcuna", "che", "per", "con", "da", "dal", "dalla", "nel", "nella",
               "titolare", "parte", "fatto", "previamente", "espressamente"}
# review (v79.12.1): an auxiliary or a light verb is never the thing («senza essere autorizzato» -> the authorisation,
# «non ha prestato il consenso» -> the consent); the same words may sit between a negation and the thing it governs
_LIGHT = {"essere", "esser", "avere", "aver", "averne", "avendo", "essendo", "avuto", "prestato", "prestata",
          "prestare", "dato", "data", "dare", "rilasciato", "rilasciata", "rilasciare", "concesso", "concessa",
          "concedere", "fornito", "fornita", "fornire", "manifestato", "manifestata", "espresso", "espressa",
          "ottenuto", "ottenuta", "ottenere", "ricevuto", "ricevuta", "ricevere", "richiesto", "richiesta",
          "richiedere", "chiesto", "chiesta", "chiedere"}
_FILLER = _LIGHT | {"ha", "ho", "hai", "abbiamo", "hanno", "abbia", "abbiano", "aveva", "avevano", "avesse",
                    "avessero", "ebbe", "è", "era", "erano", "fosse", "fossero", "sia", "siano", "sono", "fu", "furono",
                    "sarà", "sarebbe", "venne", "viene", "stato", "stata", "stati", "state", "vi", "ci", "v", "c", "si",
                    "lo", "la", "li", "le", "gli", "ne", "mi", "ti", "l", "il", "i", "un", "una", "uno", "alcun",
                    "alcuna", "alcuno", "più", "affatto", "ancora", "previamente", "espressamente", "preventivamente",
                    "formalmente", "validamente", "qualsiasi", "qualunque", "di", "del", "della", "dello", "dell", "dei", "degli", "delle", "a",
                    "ad", "in", "modo", "suo", "sua", "suoi", "sue", "proprio", "propria", "loro", "previo", "previa"}
_GENITIVE = {"di", "del", "della", "dello", "dell", "dei", "degli", "delle"}
_BREAK = {"e", "ed", "ma", "o", "od", "oppure", "però", "perché", "poiché", "quando", "mentre", "che", "se", "dopo",
          "prima", "anche", "pur", "benché", "sebbene", "quindi"}
_NEG_STRONG = {"non", "senza", "né"}
_NEG_CONCORD_RE = re.compile(r"(?:mai|nessun\w*|niente|nulla)")
_NEG_WORD_RE = re.compile(r"(?:privo|priva|privi|prive|assenza|mancanza|difetto|manca\w*|mancò|assente|assenti|"
                          r"neg(?:a|ò|are|ava|avano|ando|arono|at[oaie])|rifiut\w*|dinieg\w*|revoc\w*)")
_NOT_A_NEGATION = {"appena", "solo", "soltanto", "sempre"}      # «non appena», «non solo»: no negation of the thing
_TOKEN_RE = re.compile(r"[a-zà-ü]+|[,;:.()!?«»\"\[\]–-]")
# the same thing in other words (prefixes): the consent, the authorisation, the performance (adempiere: pagare,
# versare, saldare, corrispondere), the restitution (restituire, riconsegnare)
_THING_GROUPS = (("consen", "accons", "assens"), ("autoriz", "permes", "licenz"),
                 ("adempi", "adempu", "paga", "pagò", "versat", "saldat", "corrispos"),
                 ("restitu", "riconsegn", "ridat"))
NEGATED_PRESENCE_WHY = "elemento negativo: il fatto ne afferma la presenza, non l'assenza ({sid})"
NEGATED_SILENT_WHY = "elemento negativo: nessun fatto ne afferma l'assenza"


def _thing_stems(words: Sequence[str]) -> set:
    """The thing's prefixes: each word's first six letters (the whole word when shorter), with its group's
    (_THING_GROUPS) when it belongs to one."""
    out = set()
    for w in words:
        if len(w) < 4 or w in _THING_STOP:
            continue
        out.add(w[:6])
        for group in _THING_GROUPS:
            if any(w.startswith(g) for g in group):
                out |= set(group)
    return out


def negated_element(text: str) -> Optional[set]:
    """v79.12.1: the prefixes of the thing an absence-phrased element says is missing -- its first content word
    (never an auxiliary or a light verb, _LIGHT: «senza essere autorizzato» -> «autori…», «non ha prestato il
    consenso» -> «consen…»), with its other words (_THING_GROUPS); None when the element is not phrased as an absence
    (or is a prohibition, a mental or a termine element)."""
    text = str(text or "")
    m = _NEGATED_ELEMENT_RE.search(text)
    if not m or _PROHIBIT_RE.search(text) or mental_element(text) or term_of(text):
        return None
    words = re.findall(r"[a-zà-ü]+", (m.group(0) + " " + text[m.end():]).lower())[:8]
    first = next((w for w in words if len(w) >= 4 and w not in _THING_STOP and w not in _LIGHT), "") or \
        next((w for w in words if len(w) >= 4 and w not in _THING_STOP), "")
    return _thing_stems([first]) if first else set()


def _is_thing(word: str, thing: set) -> bool:
    return len(word) >= 4 and any(word.startswith(p) for p in thing)


def _negation_parity(tokens: Sequence[str], i: int, step: int) -> int:
    """The negations governing tokens[i] on one side (step -1: before it, +1: after it): the chain of negations,
    auxiliaries, clitics, articles and light verbs (_FILLER) reaching it, stopped by any other word, a punctuation
    mark or a conjunction (after the thing, a «di …» complement of up to two words is crossed). A strong negation
    («non», «senza», «né») and a negating word («negato», «rifiutato», «revocato», «mancava», «privo») count one
    each; «mai», «nessun» count one only with no strong negation (negative concord: «non … mai» is one negation);
    after the thing only a «non» before an auxiliary or a light verb counts («il consenso non è stato dato», never
    «… non opponendosi», never «il consenso senza riserve»)."""
    strong = verbs = concord = 0
    gen = 0
    j = i + step
    while 0 <= j < len(tokens):
        t = tokens[j]
        nxt = tokens[j + 1] if j + 1 < len(tokens) else ""
        if not t[0].isalpha() or t in _BREAK:
            break
        if t in _NEG_STRONG:
            if t == "non" and nxt in _NOT_A_NEGATION:
                break
            if step > 0 and (t != "non" or nxt not in _FILLER):     # after the thing: «senza …» governs what follows
                break
            strong += 1
        elif _NEG_CONCORD_RE.fullmatch(t):
            if step > 0 and nxt not in _FILLER:
                break
            concord += 1
        elif _NEG_WORD_RE.fullmatch(t):
            verbs += 1
        elif t in _FILLER:
            if step > 0 and t in _GENITIVE:
                gen = 2
        elif step > 0 and gen > 0:
            gen -= 1
        else:
            break
        j += step
    return strong + verbs + (1 if concord and not strong else 0)


def thing_mentions(text: str, thing: set) -> List[bool]:
    """review (v79.12.1): each mention of the thing in the sentence, True when a negation governs it (an odd number of
    negations in the chain before it, or after it: «non ha autorizzato», «senza il suo consenso», «nessun consenso»,
    «il consenso non fu dato», «ha rifiutato di restituire»; «non ha rifiutato il consenso» is two -- the presence).
    A «non» or «senza» elsewhere in the sentence («ha autorizzato l'accesso e non ha sollevato obiezioni», «ha dato il
    consenso senza riserve») governs nothing."""
    tokens = _TOKEN_RE.findall(str(text or "").lower())
    return [bool(_negation_parity(tokens, i, -1) % 2 or _negation_parity(tokens, i, 1) % 2)
            for i, w in enumerate(tokens) if w[0].isalpha() and _is_thing(w, thing)]


def states_absence(sentence: Dict[str, Any], thing: set) -> bool:
    """v79.12.1: the sentence states the absence of the thing -- review: a negation governs one of its mentions
    (thing_mentions), never an absence word merely near it (with no thing read, any absence word)."""
    text = str(sentence.get("text") or "")
    if not thing:
        return any(_ABSENCE_RE.fullmatch(w) for w in re.findall(r"[a-zà-ü]+", text.lower()))
    return any(thing_mentions(text, thing))


def states_presence(sentence: Dict[str, Any], thing: set) -> bool:
    """review (v79.12.1): the sentence names the thing and no negation governs any of its mentions."""
    marks = thing_mentions(str(sentence.get("text") or ""), thing) if thing else []
    return bool(marks) and not any(marks)


def negated_check(req: Dict[str, Any], by_id: Dict[str, Dict[str, Any]]) -> None:
    """v79.12.1, in place after polarity: a negated element (negated_element) «provato» stays so only when a supporting
    fatto certo states the absence (states_absence); else «da provare» -- the presence of the thing, or no such fact
    (req["negated"] marks it; req["negated_presence"] the fact that stated the presence)."""
    from .interactive_flow import DA_PROVARE, PROVATO
    if req.get("kind") not in ("requisito", "procedibilita") or element_rule(req)[0] in ("mental", "termine"):
        return
    thing = negated_element(req.get("text") or "")
    if thing is None:
        return
    req["negated"] = True
    if req.get("status") != PROVATO:
        return
    facts = [by_id[i] for i in req.get("supporting") or [] if i in by_id and certain_fact(by_id[i], req.get("text") or "")]
    if any(states_absence(s, thing) for s in facts):
        return
    presence = next((s for s in facts if states_presence(s, thing)), None)
    if presence:
        req["negated_presence"] = presence.get("id")
        req["status"], req["why"] = DA_PROVARE, NEGATED_PRESENCE_WHY.format(sid=presence.get("id"))
    else:
        req["status"], req["why"] = DA_PROVARE, NEGATED_SILENT_WHY


def negated_counters(records: Dict[str, Dict[str, Any]], by_id: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
    """v79.12.1: negated_integrated_by_presence -- a negated element «provato» with no supporting fatto certo stating
    the absence (must be 0); negated_elements -- the negated elements graded."""
    from .interactive_flow import PROVATO
    wrong = seen = 0
    for rec in (records or {}).values():
        for r in list(rec.get("requisites") or []) + list(rec.get("procedibilita") or []):
            if r.get("kind", "requisito") not in ("requisito", "procedibilita") or \
                    element_rule(r)[0] in ("mental", "termine"):
                continue
            thing = negated_element(r.get("text") or "")
            if thing is None:
                continue
            seen += 1
            if r.get("status") == PROVATO and not any(
                    certain_fact(by_id[i], r.get("text") or "") and states_absence(by_id[i], thing)
                    for i in r.get("supporting") or [] if i in by_id):
                wrong += 1
    return {"negated_integrated_by_presence": wrong, "negated_elements": seen}


def polarity_counters(records: Dict[str, Dict[str, Any]], by_id: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
    """v79.9 (A4): a mental element «provato» with no supporting fatto certo stating the state of mind; a duty element
    «non risulta» on a certain contradicting fact that is a breach (not a denial of the duty). Both must be 0. v79.11
    (6): the elements selected by element_rule, as polarity grades them. v79.10.4 (D3): a duty the client must satisfy
    (req["binds"] «cliente») is contradicted by its breach -- not counted."""
    from .interactive_flow import NON_RISULTA, PROVATO
    mental = duty = 0
    for rec in (records or {}).values():
        for r in list(rec.get("requisites") or []) + list(rec.get("procedibilita") or []):
            rule = element_rule(r)[0]           # v79.11 (6): the grading's own selector
            if rule == "mental" and r.get("status") == PROVATO and not any(
                    certain_fact(by_id[i], r.get("text") or "") and states_mind(by_id[i]) for i in r.get("supporting") or [] if i in by_id):
                mental += 1
            if rule == "duty" and r.get("binds") != CLIENT_BINDS and r.get("status") == NON_RISULTA and any(
                    certain_fact(by_id[i], r.get("text") or "") and not _NO_DUTY_RE.search(by_id[i].get("text") or "")
                    for i in r.get("contradicting") or [] if i in by_id):
                duty += 1
    return {"mental_integrated_by_conduct": mental, "duty_polarity_errors": duty}


def branch_combine(statuses: Sequence[str]) -> str:
    """v79.6: an element with branches -- «integrato» (provato) if any branch is, «escluso» (non risulta) only if every
    branch is, else «da provare»."""
    from .interactive_flow import DA_PROVARE, NON_RISULTA, PROVATO
    statuses = list(statuses)
    if any(x == PROVATO for x in statuses):
        return PROVATO
    if statuses and all(x == NON_RISULTA for x in statuses):
        return NON_RISULTA
    return DA_PROVARE


def branch_hit(branch: str, sentence: Dict[str, Any]) -> bool:
    """A sentence speaks of a branch: it holds one of the branch's own nouns."""
    own = element_nouns(branch)
    return bool(own) and bool(own & element_nouns(sentence.get("text") or ""))


def sibling_hit(sibling: str, element: str, sentence: Dict[str, Any]) -> bool:
    """v79.19.4 (rule 1): a sentence speaks of an alternative comma's requisite (sibling) -- it holds one of the
    sibling's own nouns that the element itself does not share (a noun common to both, «cosa», tells nothing)."""
    own = element_nouns(sibling) - element_nouns(element)
    return bool(own) and bool(own & element_nouns(sentence.get("text") or ""))


def settle_branches(req: Dict[str, Any], sup: Sequence[Dict[str, Any]], sure_con: Sequence[Dict[str, Any]]) -> None:
    """v79.6 (item 4): the element's branches graded on the sentences the link found -- a branch a grounded supporting
    sentence speaks of «provato», one a sure contradicting sentence speaks of (and no support) «non risulta», the others
    «da provare» --, the element's status their combination (branch_combine) whenever a sentence speaks of a branch: a
    contradiction of one branch never excludes the element while another holds or stays open. req["branch"]: the branch
    that applies (the first «provato»)."""
    from .interactive_flow import DA_PROVARE, NON_RISULTA, PROVATO
    ground = [x for x in sup if certain_fact(x, req.get("text") or "")]           # v79.8 (C1)
    out = {}
    for b in req.get("branches") or []:
        if any(branch_hit(b, x) for x in ground):
            out[b] = PROVATO
        elif any(branch_hit(b, x) for x in sure_con):
            out[b] = NON_RISULTA
        else:
            out[b] = DA_PROVARE
    req["branch_status"] = out
    spoken = any(v != DA_PROVARE for v in out.values())
    if not spoken:
        return
    combined = branch_combine(out.values())
    if req["status"] == NON_RISULTA and combined != NON_RISULTA:
        req["status"] = combined
        req["why"] = "un'alternativa è esclusa, un'altra resta" + (" ed è provata" if combined == PROVATO else
                                                                   " da provare")
    elif req["status"] == DA_PROVARE and combined == PROVATO:
        req["status"], req["why"] = PROVATO, "un'alternativa dell'elemento è provata"
    req["branch"] = next((b for b, v in out.items() if v == PROVATO), "")


# v60h (5): an element's own nouns -- its content words (no function word, no verb form, none of the generic legal
# ones) as five-letter stems (v60t: no longer a condition of «provato»; the open questions' words read them)
_NOUN_STOP = {"chiun", "quals", "qualu", "quand", "ovver", "nonch", "sempr", "tutti", "tutte", "altri", "altro", "stess",
              "medes", "quest", "quell", "esser", "fatto", "fatti", "perso", "caso", "const", "attua"}
_VERB_END_RE = re.compile(r"(?:are|ere|ire|ano|ono|ava|eva|iva|isce|iscono|ando|endo|ebbe|rebbe|asse|esse|isse)$")
_WORD_STOP = {"mediante", "taluno", "taluna", "altrui", "chiunque", "ovvero", "nonché", "qualora", "quale", "quali",
              "quando", "senza", "presso", "verso", "contro", "secondo", "durante", "oltre", "salvo", "ogni", "tale",
              "tali", "stesso", "stessa", "ciascuno", "ciascuna", "nessuno", "nessuna", "proprio", "propria"}


def element_nouns(text: str) -> set:
    from .topic_step import GENERIC_STEMS, _STEM_STOP, _STOPWORDS
    out = set()
    for w in re.findall(r"[a-zà-ü]{4,}", str(text or "").lower()):
        if w in _STOPWORDS or w in _STEM_STOP or w in _WORD_STOP or _VERB_END_RE.search(w) or w[:6] in GENERIC_STEMS:
            continue
        if w[:5] not in _NOUN_STOP:
            out.add(w[:5])
    return out


APPLICABLE, TO_PROVE, NOT_APPLICABLE = "applicabile", "da provare", "non applicabile"
CLIENT_ARGUMENT = "argomento del cliente"          # v60i (2): the counterparty's article whose elements fail
NOT_RELEVANT = "non pertinente"                    # v60r (3): a comma's status -- an unmet element outside the anchor comma


def anchor_comma(rec: Dict[str, Any]) -> int:
    """v60s (2): the comma an article's verdict hangs on -- co. 1, unless its topic names another (rec["anchor_comma"],
    set by build from the topic's own words); the distinctive element never moves it. An article whose text has no
    numbered commi is its co. 1."""
    named = rec.get("anchor_comma")
    return int(named) if isinstance(named, int) or str(named or "").isdigit() else 1


def _comma(req: Dict[str, Any]) -> int:
    k = req.get("comma")
    return int(k) if isinstance(k, int) or str(k or "").isdigit() else 1


def unmet_in_anchor(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """v60r (3): the article's unmet elements («non risulta»: the facts exclude it or a sentence of the case contradicts
    it) of its anchor comma -- the only ones that exclude."""
    from .interactive_flow import NON_RISULTA
    k = anchor_comma(rec)
    return [r for r in rec.get("requisites") or [] if r.get("status") == NON_RISULTA and _comma(r) == k
            and not alternative_open(rec, r)]       # v79.19.4 (rule 1): an excluded alternative with another open


# --- v62: what is counted, and the threats read from the defence -------------------------------------------------------

THREAT, WEAPON = "threat", "weapon"           # rec["side"]: the article against the client (the charge, the claim) or its own
APPLICATION = "applicazione"                  # the citations' status of an article whose requisites are not counted
PROCEDURAL_CODES = ("c.p.p.", "c.p.c.", "c.p.a.")
GENERAL_CODES = ("disp. prel.", "preleggi", "disp. att.")
OFFENCES_FROM = 241                           # c.p. Libro II and III (the norme incriminatrici); Libro I before it
_DEFINITIONAL_RE = re.compile(r"^\W*(?:nozion\w*|definizion\w*|ambito di applicazion\w*|oggetto\b|finalit\w*|"
                              r"principi generali|disposizioni (?:generali|preliminari|comuni)|"
                              # v76: a rule on time limits or proof confers no remedy -- a procedural line, never a claim
                              r"termin[ei]\b|onere della prova|prescrizion\w*|decadenz\w*|"
                              # v76b: what modifies, extends or administers other articles is no fattispecie either
                              r"circostanz\w*|pen[ae]\s+accessori\w*|sanzion\w*\s+accessori\w*|confisc\w*|"
                              r"cas[io]\s+di\s+(?:non\s+punibilit|procedibilit)\w*|caus[ae]\s+di\s+non\s+punibilit\w*|"
                              r"giurisdizion\w*|competenz\w*|procedibilit\w*|pubblicazion\w*\s+della\s+sentenza|"
                              r"disposizion\w*\s+(?:finali|transitori\w*|di\s+attuazione|di\s+coordinamento)|"
                              r"norm[ae]\s+(?:finali|transitori\w*)|entrata\s+in\s+vigore|abrogazion\w*)", re.IGNORECASE)
ELEMENT_CHARS = 140


def counted(label: str, rubrica: str = "") -> bool:
    """v62: the article's requisites are counted («requisiti provati: N su M») -- a fattispecie: a norma incriminatrice
    (c.p. Libro II and III) or a civil fattispecie. Never c.p. Libro I, a procedural code (c.p.p., c.p.c., c.p.a.), the
    general provisions, or a definitional article (its rubrica: «Nozione», «Definizioni», «Ambito di applicazione»,
    «Disposizioni generali» …): those get an «applicazione» paragraph."""
    from .plan_then_write import _code_key
    parts = str(label or "").split(" ", 1)
    if len(parts) < 2:
        return True
    code = _code_key(parts[1])
    if code in PROCEDURAL_CODES or code in GENERAL_CODES:
        return False
    m = re.match(r"\d+", parts[0])
    if code == "c.p." and m and int(m.group(0)) < OFFENCES_FROM:
        return False
    if prescription(label, rubrica):        # v79 (E4): never a claim, never a candidate
        return False
    return not _DEFINITIONAL_RE.match(str(rubrica or ""))


# v79 (E4): prescription and decadenza -- c.c. Libro VI, Titolo V (artt. 2934-2969) and any rubrica on them («Decorrenza
# della prescrizione», «Cause che impediscono la decadenza») -- are never a claim or a candidate
PRESCRIPTION_SPAN = (2934, 2969)
_PRESCRIPTION_RE = re.compile(r"\bprescrizion\w*|\bdecadenz\w*", re.IGNORECASE)


def prescription(label: str, rubrica: str = "") -> bool:
    from .plan_then_write import _code_key
    parts = str(label or "").split(" ", 1)
    m = re.match(r"\d+", parts[0]) if parts else None
    if len(parts) == 2 and _code_key(parts[1]) == "c.c." and m and \
            PRESCRIPTION_SPAN[0] <= int(m.group(0)) <= PRESCRIPTION_SPAN[1]:
        return True
    return bool(_PRESCRIPTION_RE.search(str(rubrica or "")))


JUSTIFICATIONS = ("50", "51", "52", "53", "54")    # c.p. cause di giustificazione: graded, never a fattispecie
NOT_FATTISPECIE = "parte generale o norma processuale: non è una fattispecie"     # my draft (v74), logged


def justification(label: str) -> bool:
    """v74: a c.p. cause di giustificazione (artt. 50-54): Libro I, never a fattispecie record (never the charge, a
    threat, an alternative, a claim, a «sussunzione» table), but its requisites graded -- as a circumstance's are --,
    the scriminante line's."""
    parts = str(label or "").split(" ", 1)
    from .plan_then_write import _code_key
    return len(parts) == 2 and _code_key(parts[1]) == "c.p." and parts[0] in JUSTIFICATIONS


# v79 (D1): the general-part articles that can be a defence line -- causation (40, 41), fault (43), the scriminanti
# (50-54), excess (55), attempt (56), aberratio delicti (83), particolare tenuità (131-bis)
DEFENCE_ARTICLES = ("40", "41", "43", "50", "51", "52", "53", "54", "55", "56", "83", "131-bis")


def defence_article(label: str) -> bool:
    parts = str(label or "").split(" ", 1)
    from .plan_then_write import _code_key
    return len(parts) == 2 and _code_key(parts[1]) == "c.p." and parts[0] in DEFENCE_ARTICLES


DEFENCE_WHY = "difesa di parte generale"      # v79 (D1): the route a defence article enters by (decompose_records)


def decomposed_by_gate(label: str, rubrica: str = "", defence: bool = False,
                       schema: Optional[Dict[str, Any]] = None) -> bool:
    """v74: the gate before any decomposition, every route -- a fattispecie (counted) or a cause di giustificazione
    gets elements; a procedural code's article, c.p. Libro I's, the general provisions, a definitional one never do
    (they stay «applicazione» paragraphs and procedural lines). v79 (D1): a general-part defence article
    (DEFENCE_ARTICLES) gets elements too when it entered by its route (defence: the first read names its concept, or
    the retrieval ranks it among the top) -- never a fattispecie (counted stays False), a defence line's.
    v79.11 (6): schema -- the article's schema line (or a record's rec["schema"] summary), ELEMENT_SCHEMAS only
    (element_schemas.decide): a sanction / claim-conferring / defence / scriminante tipo decomposes it, except where the
    hard invariants hold (a procedural code's article, the general provisions, c.p. Libro I but 50-54: the current gate
    stays); a tipo only limiting / definitional / procedural / other gates it out only when the schema is REVIEWED (an
    unreviewed one: the current gate decides). A disagreement is logged (schema_gate_disagree)."""
    current = counted(label, rubrica) or justification(label) or (defence and defence_article(label))
    if schema is None:
        return current
    from . import element_schemas
    if not element_schemas.enabled():
        return current
    line = gate_line(schema)
    decided, by, disagree = element_schemas.decide(label, line, bool(current))
    if disagree:
        _log_event("schema_gate_disagree", {"article": label, "current": bool(current), "schema":
                                            element_schemas.opinion(line), "tipo": (line.get("schema") or {}).get("tipo"),
                                            "reviewed": bool(line.get("reviewed")), "by": by, "decided": decided})
    return decided


def gate_line(schema: Dict[str, Any]) -> Dict[str, Any]:
    """v79.11 (6): what element_schemas.gate reads -- a schema line as it is; a record's summary (rec["schema"]) as a
    line holding its tipo and whether it is reviewed."""
    if not isinstance(schema, dict) or isinstance(schema.get("schema"), dict):
        return schema if isinstance(schema, dict) else {}
    return {"schema": {"tipo": list(schema.get("tipo") or [])}, "reviewed": schema.get("reviewed") is True,
            "source": schema.get("source") or ""}


def decomposed(rec: Dict[str, Any]) -> bool:
    """v74: the record is split into elements -- at least one requisite read from the article (kind «requisito»); the
    whole-comma fallback (kind «comma») is a text, not an element."""
    return any(r.get("kind") == "requisito" for r in rec.get("requisites") or [])


def applied(label: str, rec: Dict[str, Any]) -> bool:
    """v63: the article's paragraph is an application, no requisite counted -- not counted (counted), or brought in by
    a «cenni» subject or (v64) another general road (rec["applicazione"]: interactive_flow.general_rows) with no
    requisite of its own."""
    return bool(rec.get("applicazione")) and not rec.get("requisites") or not counted(label, rec.get("rubrica") or "")


def _element(req: Dict[str, Any]) -> str:
    text = " ".join(str(req.get("text") or "").split()).strip(" .;,")
    return text if len(text) <= ELEMENT_CHARS else text[:ELEMENT_CHARS - 1].rstrip() + "…"


def defence_items(rec: Dict[str, Any], penal: bool = True) -> List[str]:
    """v62: a threat article's elements read from the defence -- a proven one «elemento da contestare: …» (its own
    text, the point to contest), a silent one «elemento non provato dall'accusa» («dalla controparte» on a civil claim),
    a contradicted one «elemento escluso dai fatti del caso»; an element «non pertinente» (another comma's) left out."""
    from . import interaction_copy as C
    from .interactive_flow import NON_RISULTA, PROVATO
    out = []
    for r in rec.get("requisites") or []:
        status = element_status(rec, r)
        if status == NOT_RELEVANT or not _element(r):
            continue
        if status == PROVATO:
            template = C.THREAT_CONTEST
        elif status == NON_RISULTA:
            template = C.THREAT_EXCLUDED
        else:
            template = C.THREAT_UNPROVEN["penale" if penal else "civile"]
        out.append(C.fill(template, element=_element(r)))
    return [x for x in out if x]


def contradicted(rec: Dict[str, Any]) -> bool:
    """v65: an element of the record the facts contradict -- «non risulta» in its anchor comma (element_status), or a
    sentence of the case against an element not proven. v79.10.4 (D3): on the client's record a sentence against an
    element the counterparty must satisfy (req["binds"] «controparte») is its breach, never a contradiction: only
    «non risulta» counts for it (the duty itself denied: no contract, no sale)."""
    from .interactive_flow import NON_RISULTA, PROVATO
    return any(element_status(rec, r) == NON_RISULTA or (r.get("contradicting") and r.get("status") != PROVATO
                                                         and r.get("binds") != COUNTERPARTY_BINDS)
               for r in rec.get("requisites") or [])


def counterparty_contradicted(records: Dict[str, Dict[str, Any]], by_id: Dict[str, Dict[str, Any]]) -> int:
    """v79.10.4 (D3): elements the counterparty must satisfy on the client's record left «non risulta» on a certain
    contradicting fact that is a breach (not a denial of the duty) -- must be 0."""
    from .interactive_flow import NON_RISULTA
    return sum(1 for rec in (records or {}).values() for r in rec.get("requisites") or []
               if r.get("binds") == COUNTERPARTY_BINDS and r.get("status") == NON_RISULTA and any(
                   certain_fact(by_id[i], r.get("text") or "") and not _NO_DUTY_RE.search(by_id[i].get("text") or "")
                   for i in r.get("contradicting") or [] if i in by_id))


def threat_elements(rec: Dict[str, Any]) -> List[str]:
    """v63: a threat article's elements by name -- its record's (an element «non pertinente» left out): all §1 and the
    conclusions may contest."""
    return [_element(r) for r in rec.get("requisites") or [] if element_status(rec, r) != NOT_RELEVANT and _element(r)]


def threat_status(rec: Dict[str, Any], penal: bool = True) -> Optional[str]:
    """v62: a threat article's one status sentence, from the defence (defence_items) -- never «applicabile, N su N»."""
    from . import interaction_copy as C
    items = defence_items(rec, penal)
    return C.fill(C.THREAT_STATUS, items=C.THREAT_ITEMS_JOIN.join(items)) if items else None


def element_status(rec: Dict[str, Any], req: Dict[str, Any]) -> str:
    """v60r (3): an element's status as the reader gets it -- an unmet one outside the anchor comma is «non pertinente»
    (it excludes nothing, it proves nothing); else the record's."""
    from .interactive_flow import DA_PROVARE, NON_RISULTA
    if req.get("status") == NON_RISULTA and req not in unmet_in_anchor(rec):
        return NOT_RELEVANT
    if req.get("status") == DA_PROVARE and req.get("alt_commi") and alternative_satisfied(rec, req):
        return NOT_RELEVANT         # v79.19.4 (rule 1, row 80): the other alternative holds -- no missing fact here
    return str(req.get("status") or "")


def verdict(rec: Dict[str, Any]) -> str:
    """«non applicabile» when every comma with requisites has one that «non risulta» -- v60h (5): or when the article's
    distinctive element (round 1's, "distinctive") «non risulta»; «applicabile» when every requisite is «provato»; else
    «da provare». v60i (2): an article the counterparty relies on (rec["relied_by"]) that would be «non applicabile» is
    «argomento del cliente». v60r (3), every article: «non applicabile» only from an unmet element of its anchor comma
    (anchor_comma -- v60s: co. 1 unless its topic names another); an unmet element of any other comma makes that comma «non pertinente» -- it never excludes and
    never proves: the verdict is the anchor comma's and the other commi's with no unmet element (all «provato»:
    «applicabile», else «da provare»)."""
    base = _verdict(rec)
    return CLIENT_ARGUMENT if base == NOT_APPLICABLE and rec.get("relied_by") else base


def _verdict(rec: Dict[str, Any]) -> str:
    from .interactive_flow import NON_RISULTA, PROVATO
    if unmet_in_anchor(rec):
        return NOT_APPLICABLE
    reqs = [r for r in rec.get("requisites") or [] if r.get("kind") == "requisito"]
    if not reqs:
        return TO_PROVE
    commi: Dict[Any, List[str]] = {}
    for r in reqs:
        commi.setdefault(r.get("comma"), []).append(r.get("status"))
    live = {k: v for k, v in commi.items() if NON_RISULTA not in v}       # a comma with an unmet one: non pertinente
    if live and all(x == PROVATO for v in live.values() for x in v):
        return APPLICABLE
    return TO_PROVE


def verdict_reason(rec: Dict[str, Any]) -> str:
    """The ground of a «non applicabile» verdict: the requisite that does not hold and why (v60r: the anchor comma's)."""
    out = unmet_in_anchor(rec)
    req = next((r for r in out if r.get("distinctive")), None) or (out[0] if out else None)
    what = "elemento distintivo" if req and req.get("distinctive") else "requisito"
    return f"{what} «{req['text'][:120]}» non risulta ({req.get('why') or ''})".replace(" ()", "") if req else ""


def assertable(rec: Dict[str, Any]) -> bool:
    """v60s (4): an article §1 and §4 may assert -- «applicabile», or «da provare» with at least one element «provato»."""
    from .interactive_flow import PROVATO
    v = rec.get("verdict") or verdict(rec)
    return v == APPLICABLE or (v == TO_PROVE and any(r.get("status") == PROVATO for r in rec.get("requisites") or []))


def no_signal(req: Dict[str, Any]) -> bool:
    """v60r (6): an element with no fact signal -- «da provare», no sentence of the case for or against it, the facts
    deciding nothing, no open question asking it: the payload's «da provare», never narrated as missing."""
    from .interactive_flow import DA_PROVARE
    return (req.get("status") == DA_PROVARE and not req.get("supporting") and not req.get("contradicting")
            and (req.get("facts_status") or DA_PROVARE) == DA_PROVARE and not req.get("question"))


def support(rec: Dict[str, Any]) -> int:
    """The element support of an article: requisites proved by the facts or backed by a case sentence."""
    from .interactive_flow import PROVATO
    return sum(1 for r in rec.get("requisites") or [] if r.get("status") == PROVATO or r.get("supporting"))


def supported(rec: Dict[str, Any], by_id: Optional[Dict[str, Dict[str, str]]] = None) -> int:
    """v60i (1): the requisites (never a d'ufficio condition) a sentence of the case supports -- «provato» (by the
    facts the case states, or by a sentence holding its nouns) or with a supporting sentence; by_id given, a sentence
    of the system's (an assumption) supports none."""
    from .interactive_flow import PROVATO
    n = 0
    for r in rec.get("requisites") or []:
        ids = [i for i in r.get("supporting") or [] if by_id is None or (by_id.get(i) or {}).get("source") != "sistema"]
        if r.get("status") == PROVATO or ids:
            n += 1
    return n


def asked(rec: Dict[str, Any], open_questions: Optional[Sequence[str]] = None) -> List[str]:
    """v60j: the open round-1 questions asking a missing fact of the article -- a requisite «da provare» whose question
    (requisite["question"], _question_for) is still open (open_questions given: among them)."""
    from .interactive_flow import DA_PROVARE
    out = []
    for r in rec.get("requisites") or []:
        q = str(r.get("question") or "")
        if q and r.get("status") == DA_PROVARE and (open_questions is None or q in open_questions) and q not in out:
            out.append(q)
    return out


def block_due(rec: Dict[str, Any], by_id: Optional[Dict[str, Dict[str, str]]] = None,
              open_questions: Optional[Sequence[str]] = None) -> bool:
    """v60i (1): the article has a §2 block -- its verdict «applicabile», «argomento del cliente» (2), or «da provare»
    with a sentence of the case supporting one of its requisites (supported) or, v60j, with a missing fact an open
    round-1 question asks (asked: the lawyer asked for exactly that «da verificare»). v60p: the anchor of a covered
    topic always (rec["anchor_due"], set for the answer being written: interactive_flow.anchor_blocks) -- never one
    «non applicabile». v60t (2, single shot: rec["needs_proof"]): «da provare» earns one only with an element «provato»
    or a missing fact an open question asks -- a supporting sentence alone no longer does; the anchor_due one is the
    main line's anchor alone. v63: an article a «cenni» subject rests on (v64: any general road, rec["applicazione"])
    always: its application."""
    if rec.get("applicazione") or rec.get("named_due"):     # v63, v64: a general road's article; v65: the user's
        return True
    v = rec.get("verdict") or verdict(rec)
    if v in (APPLICABLE, CLIENT_ARGUMENT):
        return True
    if rec.get("anchor_due") and v != NOT_APPLICABLE:
        return True
    if v != TO_PROVE:
        return False
    if rec.get("needs_proof"):          # v60t (2, single shot): a proven element, or a missing fact a question asks
        return proven(rec) > 0 or bool(asked(rec, open_questions))
    return supported(rec, by_id) > 0 or bool(asked(rec, open_questions))


def proven(rec: Dict[str, Any]) -> int:
    """v60t (2): the article's elements «provato»."""
    from .interactive_flow import PROVATO
    return sum(1 for r in rec.get("requisites") or [] if r.get("status") == PROVATO)


LINE_NOT_APPLICABLE, LINE_NOT_TREATED = "non applicabile", "non trattata"
ELEMENT_CHARS = 80


def _short(text: str, limit: int = ELEMENT_CHARS) -> str:
    text = " ".join(str(text or "").split()).strip(" .;,")
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut.rstrip(" ,;:") + "…"


def line_reason(rec: Dict[str, Any], by_id: Optional[Dict[str, Dict[str, str]]] = None,
                open_questions: Optional[Sequence[str]] = None) -> Optional[Tuple[str, str]]:
    """v60i (1): (the line, its reason in the reader's words) of an article with no §2 block; None when it has one.
    «non applicabile»: the element that does not result; «non trattata»: no text read, or no fact of the case for it."""
    from . import interaction_copy as C
    if block_due(rec, by_id, open_questions):
        return None
    v = rec.get("verdict") or verdict(rec)
    if v == NOT_APPLICABLE:
        out = unmet_in_anchor(rec)
        req = next((r for r in out if r.get("distinctive")), None) or (out[0] if out else None)
        return LINE_NOT_APPLICABLE, (C.fill(C.LINE_REASON_NOT_RESULTING, element=_short(req["text"])) if req else
                                     C.LINE_REASON_NOT_RESULTING_ANY)
    if not rec.get("requisites"):
        return LINE_NOT_TREATED, C.LINE_REASON_NO_TEXT
    return LINE_NOT_TREATED, C.LINE_REASON_NO_SUPPORT


# --- v60g (d): one status source while an answer is written ------------------------------------------------------------

_STATUSES: "contextvars.ContextVar[Optional[Dict[str, str]]]" = contextvars.ContextVar("record_statuses", default=None)
_KEY_CHARS = 200


def _req_key(text: str) -> str:
    return " ".join(re.sub(r"[«»\"“”]", "", str(text or "")).lower().split())[:_KEY_CHARS]


@contextlib.contextmanager
def statuses_from(records: Optional[Dict[str, Dict[str, Any]]]):
    """While active, recorded_status(text) is the record's status of the requisite (or comma) with these words."""
    table: Dict[str, str] = {}
    for rec in (records or {}).values():
        for r in rec.get("requisites") or []:
            if r.get("text") and r.get("status"):
                table.setdefault(_req_key(r["text"]), r["status"])
    token = _STATUSES.set(table if records else None)
    try:
        yield table
    finally:
        _STATUSES.reset(token)


def recorded_status(text: str) -> Optional[str]:
    table = _STATUSES.get()
    if not table:
        return None
    return table.get(_req_key(text))


# v60k (single shot): while an answer is written, the fact a requisite «da provare» lacks -- the open question asking it
# (requisite["question"]) as «il caso non dice …» reads it -- for its §2 status line
_POINTS: "contextvars.ContextVar[Optional[Dict[str, str]]]" = contextvars.ContextVar("record_points", default=None)
# v60l: the requisites «da provare» whose open matter is a legal characterisation -- §2 «da verificare», no question
_QUALIFIED: "contextvars.ContextVar[Optional[set]]" = contextvars.ContextVar("record_qualified", default=None)


@contextlib.contextmanager
def points_from(records: Optional[Dict[str, Dict[str, Any]]], facts: Dict[str, str],
                qualifications: Sequence[str] = ()):
    """While active, recorded_point(text) is the fact the requisite (or comma) with these words lacks: facts {question
    id: the fact in words} of the open questions. v60l: recorded_qualified(text), a requisite with no such fact sharing
    two of its words with one of the legal characterisations the case leaves open (qualifications: never a point).
    (v60p's silent elements reading «da verificare» are gone: v60r 6, an element no fact touches is «da provare».)"""
    from .interactive_flow import DA_PROVARE, _stems
    table: Dict[str, str] = {}
    qualified: set = set()
    quals = [_stems(x) for x in qualifications or () if str(x).strip()]
    for label, rec in (records or {}).items():
        for r in rec.get("requisites") or []:
            q = str(r.get("question") or "")
            if not (r.get("text") and r.get("status") == DA_PROVARE):
                continue
            if facts.get(q):
                table.setdefault(_req_key(r["text"]), facts[q])
            elif any(len(_stems(r["text"]) & mine) >= 2 for mine in quals):
                qualified.add(_req_key(r["text"]))
    token = _POINTS.set(table if records else None)
    q_token = _QUALIFIED.set(qualified - set(table) if records else None)
    try:
        yield table
    finally:
        _POINTS.reset(token)
        _QUALIFIED.reset(q_token)


def recorded_point(text: str) -> str:
    table = _POINTS.get()
    return (table or {}).get(_req_key(text), "")


def recorded_qualified(text: str) -> bool:
    return _req_key(text) in (_QUALIFIED.get() or set())


# --- the fact <-> element link ----------------------------------------------------------------------------------------

# v60s (3): the element grading standard -- a fact the case narrates as happened supports its element (the record makes it
# «provato»); a fact it narrates against it contradicts it («non risulta»); none, «da provare». No evidentiary proof is
# asked for: the link reads the narrative, never documents, witnesses or rulings.
_LINK_SYSTEM = (
    "Sei un avvocato. Ricevi i FATTI del caso, numerati (S1, S2…), e i REQUISITI delle norme di un tema, numerati (R1, "
    "R2…). Per ciascun requisito indica quali fatti lo affermano (sostiene) e quali lo negano (contraddice), citando "
    "solo i numeri dei fatti. Basta che il caso racconti il fatto come accaduto: non serve una prova (documenti, "
    "testimoni, perizie, sentenze) e non va chiesta. Un fatto sostiene un requisito solo se lo dice davvero; nessun "
    "fatto: lista vuota. Un requisito con alternative («… o …», «… ovvero …») è sostenuto se il caso ne afferma anche "
    "una sola, contraddetto solo se le nega tutte. "
    "In «manca», in poche parole, che cosa il caso non dice di quel requisito (vuoto se lo dice). "
    "Non inventare fatti. Rispondi SOLO con un JSON: "
    "{\"R1\": {\"sostiene\": [\"S3\"], \"contraddice\": [], \"manca\": \"…\"}, …}")


def _link_call(topic: str, reqs: List[Tuple[str, Dict[str, Any]]], sents: Sequence[Dict[str, str]]) -> Dict[str, Any]:
    from langchain_core.messages import HumanMessage, SystemMessage
    from . import defensive_generation as dg
    facts = "\n".join(f"{s['id']}. {s['text']}" for s in sents)
    asked = "\n".join(f"{r['id']}. art. {label}, co. {r.get('comma') or '-'}: {r['text']}" for label, r in reqs)
    human = f"TEMA: {topic}\n\nFATTI:\n{facts}\n\nREQUISITI:\n{asked}"
    raw = dg._chat_json([SystemMessage(content=_LINK_SYSTEM), HumanMessage(content=human)], max_tokens=LINK_TOKENS,
                        temperature=0)
    return raw if isinstance(raw, dict) else {}


def _apply(raw: Dict[str, Any], reqs: List[Tuple[str, Dict[str, Any]]], known: set,
           by_id: Optional[Dict[str, Dict[str, str]]] = None) -> int:
    """The link's answer into the requisites; v60g (d): each settled again (settle) with its sentences."""
    n = 0
    for _, r in reqs:
        got = raw.get(r["id"]) if isinstance(raw.get(r["id"]), dict) else None
        if got is None:
            continue
        ids = lambda v: [str(x) for x in (v if isinstance(v, list) else []) if str(x) in known][:6]
        r["supporting"], r["contradicting"] = ids(got.get("sostiene")), ids(got.get("contraddice"))
        r["link"] = {"supporting": list(r["supporting"]), "contradicting": list(r["contradicting"])}   # v60t: as linked
        r["missing"] = " ".join(str(got.get("manca") or "").split())[:200]
        if by_id is not None:
            settle(r, by_id)
        n += 1
    return n


def link(records: Dict[str, Dict[str, Any]], topics: Sequence[Dict[str, Any]], refs_by_topic: Sequence[Sequence[str]],
         sents: Sequence[Dict[str, str]], call=None, seconds: float = LINK_SECONDS) -> Dict[str, Any]:
    """In place: each chosen topic's requisites linked to the case sentences, one call per topic, the topics in
    parallel, each bounded (seconds); a topic already linked on the same sentences is not asked again. The log:
    {"topics": [{topic, requisites, linked, ms, cached | error}], "total_ms"}."""
    call = call or _link_call
    fp = fingerprint(sents)
    known = {s["id"] for s in sents}
    by_id = {s["id"]: s for s in sents}
    jobs = []
    for t, labels in zip(topics, refs_by_topic):
        reqs = [(label, r) for label in labels for r in (records.get(label) or {}).get("requisites") or []]
        reqs = reqs[:LINK_REQUISITES_MAX] + [(label, c) for label in labels         # v60h (1): beyond the cap
                                             for c in (records.get(label) or {}).get("procedibilita") or []]
        if not reqs:
            continue
        cached = all((records.get(label) or {}).get("linked") == fp for label in labels if records.get(label))
        jobs.append((t, labels, reqs, cached))
    t0 = time.time()
    entries = []
    todo = [j for j in jobs if not j[3]]
    entries += [{"topic": t.get("title"), "requisites": len(reqs), "cached": True} for t, _, reqs, cached in jobs if cached]
    if todo:
        pool = ThreadPoolExecutor(max_workers=len(todo), thread_name_prefix="records_link")
        futures = [(j, time.time(), pool.submit(call, j[0].get("title") or "", j[2], sents)) for j in todo]
        for (t, labels, reqs, _), started, future in futures:
            left = max(0.1, seconds - (time.time() - t0))
            entry = {"topic": t.get("title"), "requisites": len(reqs)}
            try:
                raw = future.result(timeout=left)
                entry["linked"] = _apply(raw or {}, reqs, known, by_id)
                for label in labels:
                    if records.get(label):
                        records[label]["linked"] = fp
                        records[label]["verdict"] = verdict(records[label])
            except FutureTimeout:
                entry["error"] = f"timeout {seconds:.0f}s"
            except Exception as exc:        # noqa: BLE001 -- a failed link leaves the record as it was
                entry["error"] = f"{type(exc).__name__}: {exc}"[:200]
            entry["ms"] = round((time.time() - started) * 1000)
            entries.append(entry)
        pool.shutdown(wait=False)
    return {"topics": entries, "total_ms": round((time.time() - t0) * 1000), "fingerprint": fp}


# --- v60g (1): round 1 -- each graded article's distinctive element, linked before the topics --------------------------

DISTINCTIVE_SECONDS = 45.0         # v60h (5): to completion -- smaller calls, a longer bound (v60g: 12 per call, 20 s)
DISTINCTIVE_PER_CALL = 6
DISTINCTIVE_TOPIC = "elementi distintivi delle norme candidate"


def _element_stems(text: str) -> set:
    from .topic_step import _content_stems
    return set(_content_stems(text))


def distinctive(rows: Sequence[Dict[str, Any]], analysis: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """{label: element} for each row graded 2 or more: among its requisites (its selected commi's; with none, its first
    comma) the one whose content stems the other candidates' elements share the least -- what tells it from them --,
    the longer on a tie."""
    cands: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows or []:
        if not (row.get("_article") and isinstance(row.get("_grade"), int) and row["_grade"] >= 2):
            continue
        reqs = _requisites(row, analysis)
        items = [r for r in reqs if r["kind"] == "requisito"] or reqs[:1]
        if items:
            cands[_label(row["_article"])] = items
    counts: Dict[str, int] = {}
    for items in cands.values():
        for w in set().union(*(_element_stems(r["text"]) for r in items)):
            counts[w] = counts.get(w, 0) + 1
    out = {}
    for label, items in cands.items():
        def shared(r):
            mine = _element_stems(r["text"])
            own = set().union(*(_element_stems(x["text"]) for x in items))
            return sum(1 for w in mine if counts.get(w, 0) - (1 if w in own else 0) > 0), -len(mine)
        best = min(items, key=shared)
        out[label] = {"comma": best["comma"], "text": best["text"], "kind": best["kind"],
                      **({"alternatives": list(best["alternatives"])} if best.get("alternatives") else {})}  # v79.19.4
    return out


DEMOTED_CONTRADICTED = "contraddetto da una frase del caso narrata come fatto"
DEMOTED_EXCLUDED = "escluso dai fatti estratti"


def analysis_client_roles(analysis: Dict[str, Any]) -> set:
    """v79.10.4 (D3): before the sides, the client's role stems (role_stems) from the extraction -- the role in its
    client_side («Mario Rossi (acquirente)») and the parties named as the client --; none for an accused client
    (imputato, indagato: his norms are the ones he would violate) or when no role is given."""
    try:
        from .interactive_flow import _client_names, _facts, _low
        facts = _facts(analysis or {})
        side = str(facts.get("client_side") or "")
        words = re.findall(r"\(([^)]*)\)", side)
        names = [_low(n) for n in _client_names(facts) if n]
        words += [str(p.get("role") or "") for p in facts.get("parties") or [] if isinstance(p, dict)
                  and _low(p.get("name")) and any(n in _low(p.get("name")) or _low(p.get("name")) in n for n in names)]
        if any(re.search(r"\b(?:imputat|indagat)\w*", _low(w)) for w in words):
            return set()
        return role_stems([w for w in words if w.strip()])
    except Exception:           # noqa: BLE001 -- no role: no binding, the demotion as before
        return set()


def demote_contradicted(rows: List[Dict[str, Any]], case: str, analysis: Dict[str, Any], keep: Sequence[str] = (),
                        call=None, seconds: float = DISTINCTIVE_SECONDS) -> Dict[str, Any]:
    """In place, before the topics: each graded (2+) row's distinctive element (distinctive) linked to the case's
    sentences -- calls of DISTINCTIVE_PER_CALL elements, in parallel, bounded (seconds) --; each answered row keeps its
    element's link (_distinctive) for its record. v60m: a row is graded 1 (its grade before and why: _grade_before,
    _grade_issue, _demoted) only when its element is excluded by the extracted facts (facts_status «non risulta») or
    contradicted by a sentence of the case narrated as a fact (accertato) -- as the record settles it (settle: the facts
    first) --; an element the case is silent on never demotes: the article keeps its grade and the earned-block rule
    (block_due) decides whether it is written. keep: labels never demoted (the articles the case names). A call that
    fails or times out demotes nothing by contradiction (the facts still exclude). The log: each demotion with its
    reason and the sentences (and, excluded, the facts) behind it. v79.10.4 (D3): a duty or a prohibition (deontic) the
    counterparty must satisfy (binds, on the client's role words the extraction gives: analysis_client_roles) «contradicted»
    by a certain fact that is its breach (not a denial of the duty) never demotes -- the norm violated is found, not lost
    (log kept_breach); the client's own duty, an accused client's norms and any while the client's role is unknown
    demote as before."""
    from .interactive_flow import NON_RISULTA, PROVATO, excluding_facts, facts_status
    call = call or _link_call
    t0 = time.time()
    sents = sentences(case)
    elements = distinctive(rows, analysis)
    log: Dict[str, Any] = {"elements": len(elements), "sentences": len(sents), "calls": 0, "answered": 0,
                           "demoted": [], "kept_named": [], "silent": 0, "errors": [], "unassessed": [],
                           "complete": True, "ms": 0}
    for row in rows or []:      # v60h (5): each graded row knows its element, linked or not (the record's distinctive)
        if row.get("_article") and _label(row["_article"]) in elements:
            row["_distinctive_text"] = elements[_label(row["_article"])]["text"]
    answered: Dict[str, Dict[str, Any]] = {}
    if elements and sents:
        items = list(elements.items())
        batches = [items[i:i + DISTINCTIVE_PER_CALL] for i in range(0, len(items), DISTINCTIVE_PER_CALL)]
        known = {s["id"] for s in sents}
        pool = ThreadPoolExecutor(max_workers=len(batches), thread_name_prefix="distinctive_link")
        futures = []
        for batch in batches:
            reqs = [(label, {"id": f"R{k}", "comma": e["comma"], "text": e["text"]})
                    for k, (label, e) in enumerate(batch, 1)]
            futures.append((reqs, pool.submit(call, DISTINCTIVE_TOPIC, reqs, sents)))
        log["calls"] = len(futures)
        for reqs, future in futures:
            try:
                raw = future.result(timeout=max(0.1, seconds - (time.time() - t0))) or {}
            except FutureTimeout:
                log["errors"].append(f"timeout {seconds:.0f}s ({len(reqs)} elementi)")
                log["unassessed"] += [label for label, _ in reqs]
                continue
            except Exception as exc:            # noqa: BLE001 -- a failed call demotes nothing by contradiction
                log["errors"].append(f"{type(exc).__name__}: {exc}"[:200])
                log["unassessed"] += [label for label, _ in reqs]
                continue
            for label, r in reqs:
                got = raw.get(r["id"]) if isinstance(raw, dict) and isinstance(raw.get(r["id"]), dict) else None
                if got is None:
                    log["unassessed"].append(label)
                    continue
                ids = lambda v: [str(x) for x in (v if isinstance(v, list) else []) if str(x) in known][:6]
                answered[label] = {**elements[label], "supporting": ids(got.get("sostiene")),
                                   "contradicting": ids(got.get("contraddice")),
                                   "missing": " ".join(str(got.get("manca") or "").split())[:200]}
        pool.shutdown(wait=False)
    log["answered"] = len(answered)
    log["complete"] = not log["unassessed"]
    keep = set(keep or ())
    by_id = {s["id"]: s for s in sents}
    roles = analysis_client_roles(analysis)
    for row in rows or []:
        if not row.get("_article"):
            continue
        label = _label(row["_article"])
        element = elements.get(label)
        if element is None:
            continue
        link_ = answered.get(label)
        if link_ is not None:
            row["_distinctive"] = link_
        if not (isinstance(row.get("_grade"), int) and row["_grade"] >= 2):
            continue
        status = facts_status(element["text"], analysis)
        against = [by_id[i] for i in (link_ or {}).get("contradicting") or [] if i in by_id]
        sure = [x for x in against if x.get("certainty") == ACCERTATO and act_bears(x, element["text"])]
        from . import rule_quote
        # v79.6 (item 4): one branch contradicted while another holds never demotes. v79.19.4 (rule 1): the
        # alternatives are the element's own branches («o», «ovvero», «oppure», «in alternativa») or the requisites of
        # the paragraph's alternative comma (alternatives, alternative_commi); the article is demoted only when every
        # one of them is contradicted (a contradiction attributable to no branch is of the whole element: demotes); the
        # contradicted alternative is marked on the row (_alternative_contradicted, the record's requisite «contraddetta»
        # by seed) and logged (alternative_contradicted). A conjunctive element: as before.
        branches_ = rule_quote.branches(element["text"])
        siblings = [x for x in element.get("alternatives") or [] if x and x != element["text"]]
        alternatives = branches_ or ([element["text"]] + siblings if siblings else [])
        if sure and alternatives:
            if branches_:
                hit = [b for b in alternatives if any(branch_hit(b, x) for x in sure)]
            else:               # the element itself is what the link contradicts; a sibling only on its own nouns
                hit = [element["text"]] + [b for b in siblings if any(sibling_hit(b, element["text"], x) for x in sure)]
            if hit and len(hit) < len(alternatives):     # (a contradiction of no branch is of the whole element: demotes)
                row["_alternative_contradicted"] = list(hit)
                log.setdefault("alternative_contradicted", []).append({
                    "article": label, "alternative": hit[0][:160], "contradicted": [x[:160] for x in hit],
                    "open": [x[:160] for x in alternatives if x not in hit], "kept": row["_grade"]})
                sure = []
        if sure and roles and binds({"kind": "requisito", "text": element["text"]}, roles) == COUNTERPARTY_BINDS \
                and any(not _NO_DUTY_RE.search(x.get("text") or "") for x in sure) and status != NON_RISULTA:
            log.setdefault("kept_breach", []).append(label)     # v79.10.4 (D3): a breach is no contradiction
            continue
        if status == NON_RISULTA:
            reason, cited = DEMOTED_EXCLUDED, against
        elif status != PROVATO and sure:
            reason, cited = DEMOTED_CONTRADICTED, sure
        else:
            log["silent"] += 1 if link_ is not None and not (link_.get("supporting") or against) else 0
            continue
        if label in keep:
            log["kept_named"].append(label)
            continue
        # v79.19.9 (B, row 108): a grade of 3 is not demoted by one contradicted element alone (12 such demotions in
        # the seven runs, 8 of them key articles); the facts' exclusion demotes as before, grade 2 unchanged
        if reason == DEMOTED_CONTRADICTED and row["_grade"] >= 3:
            log.setdefault("kept_grade3", []).append(label)
            continue
        row["_grade_before"] = row["_grade"]
        row["_grade"] = 1
        row["_demoted"] = True
        row["_grade_issue"] = f"elemento distintivo {reason}: «{element['text'][:120]}»"
        entry = {"article": label, "grade": row["_grade_before"], "element": element["text"][:160], "reason": reason,
                 "sentences": [{"id": x["id"], "text": x["text"][:200], "certainty": x.get("certainty")} for x in cited]}
        if reason == DEMOTED_EXCLUDED:
            entry["facts"] = excluding_facts(element["text"], analysis)
        log["demoted"].append(entry)
    log["ms"] = round((time.time() - t0) * 1000)
    return log


def seed(rec: Dict[str, Any], row: Dict[str, Any]) -> None:
    """The record's requisite that is the row's distinctive element takes its round-1 link; v60h (5): and is marked
    "distinctive" (its «non risulta» makes the verdict «non applicabile»)."""
    link_ = row.get("_distinctive") or {}
    element = _req_key(row.get("_distinctive_text") or link_.get("text") or "")
    for req in rec.get("requisites") or []:
        if element and req.get("kind") == "requisito" and _req_key(req["text"]) == element:
            req["distinctive"] = True
            if row.get("_alternative_contradicted"):        # v79.19.4 (rule 1): the alternative the case contradicts
                req["contraddetta"] = list(row["_alternative_contradicted"])
        if link_ and _req_key(req["text"]) == _req_key(link_.get("text")) and not req.get("supporting"):
            req["supporting"], req["contradicting"] = list(link_.get("supporting") or []), \
                list(link_.get("contradicting") or [])
            req["missing"] = req.get("missing") or link_.get("missing") or ""


# --- the plan's view --------------------------------------------------------------------------------------------------

def plan_issues(topics: Sequence[Dict[str, Any]], refs_by_topic: Sequence[Sequence[Tuple[str, str]]],
                records: Dict[str, Dict[str, Any]], sents: Sequence[Dict[str, str]]) -> List[Dict[str, Any]]:
    """The fixed issues for the plan: one per chosen topic, its planned articles, and the block of its records (each
    requisite with its status, the sentences for and against, what is missing)."""
    text_of = {s["id"]: s for s in sents}
    out = []
    for t, refs in zip(topics, refs_by_topic):
        if not refs:
            continue
        lines = []
        for ref in refs:
            rec = records.get(_label(ref)) or {}
            head = f"- art. {_label(ref)}" + (f" — {rec['rubrica']}" if rec.get("rubrica") else "")
            if rec.get("grade") is not None:
                head += f" (grado {rec['grade']}" + (f": {rec['issue']}" if rec.get("issue") else "") + ")"
            lines.append(head)
            for r in rec.get("requisites") or []:
                why = []
                for key, word in (("supporting", "a favore"), ("contradicting", "contro")):
                    if r.get(key):
                        why.append(f"{word}: " + "; ".join(f"{i} «{text_of[i]['text'][:120]}» ({text_of[i]['certainty']})"
                                                         for i in r[key] if i in text_of))
                if r.get("missing"):
                    why.append(f"manca: {r['missing']}")
                lines.append(f"  · {r['id']} co. {r.get('comma') or '-'}: «{r['text'][:160]}» — {r['status']}"
                             + (f" [{' | '.join(why)}]" if why else ""))
        hyps = [h for h in t.get("hypotheses") or [] if h.get("claim")]
        if hyps:
            lines.append("  linee: " + "; ".join(f"{h.get('line')}: {h['claim']}" for h in hyps))
        out.append({"titolo": t.get("title") or "", "norme": [tuple(r) for r in refs], "scheda": "\n".join(lines)})
    return out


# --- v60g (d): the writer's view ----------------------------------------------------------------------------------------

NO_SIGNAL_NOTE = "nessun fatto del caso a favore o contro: non dire che manca"      # v60r (6, my draft)
WRITER_TITLE = "SCHEDE DELLE NORME (dal record: lo stato di ogni requisito, i fatti a favore e contro, che cosa manca)"
FACTS_TITLE = ("FATTI DEL CASO (come il caso li dà: accertato = narrato come fatto; allegato = affermato da una parte; "
               "ipotizzato = incerto; assunto = un'assunzione, non un fatto)")
WRITER_SENTENCE_CHARS = 160


def writer_block(records: Dict[str, Dict[str, Any]], refs: Sequence[Tuple[str, str]], sents: Sequence[Dict[str, str]],
                 questions: Sequence[Dict[str, Any]] = (), penal: bool = True, defending: bool = False) -> str:
    """The SCHEDE for §2, in the writer's numbering (refs: the paragraphs' order): per article its verdict, then each
    requisite with its status, the case's sentences for and against it (their words and certainty, never their ids),
    what is missing and the open question asking it. "" with no record. v62: an article whose requisites are not
    counted (counted) is one line, «applicazione», no requisite; a threat (rec["side"]) is the charge or the claim, its
    elements read from the defence (defence_items) -- the rules for both after the schede (writer_rules). v65: for a
    defending client (defending: imputato, convenuto) an application reads from the defence too
    (APPLICATION_MARK_DEFENCE): what the defence argues, what to verify, never a conclusion against the client."""
    from . import writer_rules
    text_of = {s["id"]: s for s in sents or []}
    asked = {str(q.get("id")): q.get("text") or "" for q in questions or []}
    lines = []
    applications, threats = False, False
    for i, ref in enumerate(refs, 1):
        rec = records.get(_label(ref))
        if not rec:
            continue
        head = f"[{i}] art. {_label(ref)}" + (f" — {rec['rubrica']}" if rec.get("rubrica") else "")
        if applied(_label(ref), rec):        # v62: no count, an application paragraph (v63: a cenni's article too)
            lines.append(f"{head} — {writer_rules.APPLICATION_MARK_DEFENCE if defending else writer_rules.APPLICATION_MARK}")
            applications = True
            continue
        if rec.get("side") == THREAT:                                  # v62: the charge or the claim, from the defence
            lines.append(f"{head} — {writer_rules.THREAT_MARK['penale' if penal else 'civile']}")
            threats = True
            elements = iter(defence_items(rec, penal))
            for r in rec.get("requisites") or []:
                if element_status(rec, r) == NOT_RELEVANT or not _element(r):
                    continue
                parts = [f"  - {next(elements, '')}"]
                for key, word in (("supporting", "fatti dell'accusa" if penal else "fatti della controparte"),
                                  ("contradicting", "fatti a favore della difesa")):
                    said = [f"«{text_of[x]['text'][:WRITER_SENTENCE_CHARS]}» ({text_of[x]['certainty']})"
                            for x in r.get(key) or [] if x in text_of]
                    if said:
                        parts.append(f"{word}: " + "; ".join(said))
                lines.append(" — ".join(parts))
            continue
        v = rec.get("verdict") or verdict(rec)
        why = ""
        if v == NOT_APPLICABLE:
            why = f" ({verdict_reason(rec)})"
        elif v == CLIENT_ARGUMENT:          # v60i (2): written as the client's argument against the other side
            why = f" (la controparte vi si fonda — {rec.get('relied_by')}; ma {verdict_reason(rec)})"
        lines.append(f"{head} — esito: {v}{why}")
        for r in rec.get("requisites") or []:
            where = f"co. {r['comma']}" if r.get("comma") else "testo"
            name = f"requisito «{r['text'][:200]}»" if r.get("kind") == "requisito" else "comma"
            quiet = no_signal(r)        # v60r (6): no fact for or against it, none asked -- never narrated as missing
            parts = [f"  - {where}, {name}: {element_status(rec, r)}" + (f" ({NO_SIGNAL_NOTE})" if quiet else "")]
            for key, word in (("supporting", "a favore"), ("contradicting", "CONTRO")):
                said = [f"«{text_of[x]['text'][:WRITER_SENTENCE_CHARS]}» ({text_of[x]['certainty']})"
                        for x in r.get(key) or [] if x in text_of]
                if said:
                    parts.append(f"{word}: " + "; ".join(said))
            if r.get("status") != "provato" and not quiet:
                if r.get("missing"):
                    parts.append(f"MANCA: {r['missing']}")
                if r.get("question") and asked.get(str(r["question"])):
                    parts.append(f"domanda aperta: {asked[str(r['question'])]}")
            lines.append(" — ".join(parts))
    rules = ([writer_rules.APPLICATION_RULE_DEFENCE if defending else writer_rules.APPLICATION_RULE]
             if applications else []) + \
        ([writer_rules.THREAT_RULE["penale" if penal else "civile"]] if threats else [])
    return f"{WRITER_TITLE}:\n" + "\n".join(lines + rules) if lines else ""


def facts_block(sents: Sequence[Dict[str, str]]) -> str:
    """The case's sentences with their certainty, for the writer (no ids)."""
    lines = [f"- {s['text']} ({s['certainty']})" for s in sents or []]
    return f"{FACTS_TITLE}:\n" + "\n".join(lines) if lines else ""


def fix_plan(plan: Dict[str, Any], issues: Sequence[Dict[str, Any]], log: Optional[List[Dict[str, Any]]] = None
             ) -> Dict[str, Any]:
    """The plan's questioni made the fixed issues, in their order: each takes the argument, objection and reply of the
    plan's issue sharing most of its articles (else its title's words); its norms are its own; a plan issue matching
    none is dropped, an issue no plan issue matches keeps the empty fields (logged)."""
    from .plan_then_write import _same_ref
    from .topic_step import _content_stems
    log = log if log is not None else []
    model = [q for q in (plan or {}).get("questioni") or [] if isinstance(q, dict)]
    used = set()
    out = []
    for issue in issues:
        def score(i_q):
            i, q = i_q
            shared = sum(1 for r in q.get("norme") or [] for n in issue["norme"] if len(r) >= 2 and _same_ref(tuple(r), n))
            words = len(_content_stems(q.get("titolo")) & _content_stems(issue["titolo"]))
            return (shared, words)
        best = max(((i, q) for i, q in enumerate(model) if i not in used), key=score, default=None)
        if best is not None and score(best) > (0, 0):
            used.add(best[0])
            q = best[1]
        else:
            q = {}
            log.append({"issue": issue["titolo"], "plan": "no plan issue matched: the fields left empty"})
        out.append({**{k: v for k, v in q.items() if k not in ("titolo", "norme")}, "titolo": issue["titolo"],
                    "norme": list(issue["norme"]), "pronunce": list(q.get("pronunce") or []),
                    "argomento": q.get("argomento") or "", "obiezione": q.get("obiezione") or "",
                    "replica": q.get("replica") or ""})
    for i, q in enumerate(model):
        if i not in used:
            log.append({"dropped_plan_issue": q.get("titolo"), "norme": [list(r) for r in q.get("norme") or []]})
    return {**(plan or {}), "questioni": out}


def jsonable(records: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    return json.loads(json.dumps(records, ensure_ascii=False, default=str))
