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
_ALLEGED_RE = re.compile(r"\b(?:sostien\w*|afferm\w*|lament\w*|contest\w*|dichiar\w*|riferisc\w*|riferit\w*|asseris\w*|"
                         r"asserit\w*|denunci\w*|accus\w*|ritien\w*|pretend\w*|eccepi\w*|secondo (?:il|la|lo|i|le|gli|"
                         r"[A-ZÀ-Ü])|a (?:suo|loro) dire|a detta d\w*|avrebbe\w*|sarebbe\w* stat\w*|presunt\w*)\b",
                         re.IGNORECASE)
_HEDGED_RE = re.compile(r"\b(?:forse|probabilmente|sembra\w*|pare(?:va)?|potrebbe\w*|presumibilmente|non è chiaro|"
                        r"non si sa|non risulta chiaro|verosimilmente|ipotizz\w*|"
                        r"(?:se|qualora|ove) (?:fosse|fossero|avesse|avessero)\b)\b", re.IGNORECASE)   # v60t: «se fosse»


def certainty(sentence: str) -> str:
    """How the case states it: ipotizzato (hedged or hypothetical -- v60t: first, an attributed hypothesis is one too),
    allegato (attributed or reported), else accertato."""
    if _HEDGED_RE.search(sentence or ""):
        return IPOTIZZATO
    if _ALLEGED_RE.search(sentence or ""):
        return ALLEGATO
    return ACCERTATO


def _split(text: str) -> List[str]:
    out = []
    for piece in re.split(r"(?<=[.;!?])\s+(?=[A-ZÀ-Ü«\"(])|\n+", text or ""):
        piece = " ".join(piece.split())
        if len(piece) >= 12 and re.search(r"[a-zà-ü]{3}", piece):
            out.append(piece[:SENTENCE_CHARS])
    return out


def sentences(case: str, additions: Sequence[str] = (), assumed: Sequence[str] = ()) -> List[Dict[str, str]]:
    """The case's sentences, then the client's precisazioni, then the system's assumptions, numbered S1… v60i (4): never
    the user's own message (fact_signals.case_only: the request over an upload, a question, «Sei l'avvocato di …», an
    instruction) -- no element is supported or contradicted by it."""
    from .fact_signals import case_only, user_message
    out: List[Dict[str, str]] = []
    seen = set()

    def add(text, source, how=None, origin=""):
        key = text.lower()
        if key in seen or len(out) >= SENTENCES_MAX:
            return
        seen.add(key)
        out.append({"id": f"S{len(out) + 1}", "text": text, "source": source, "certainty": how or certainty(text),
                    "verbatim": verbatim(text, origin)})                # v60t (1): grounding
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


def _requisites(row: Dict[str, Any], analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The writer's own reading of the article (rule_quote.article_rules, no plan): per selected comma its requisites,
    a comma with none as one element; each with the facts' status (facts_status). v60g: the row's text cleaned as the
    answer step cleans it (rule_quote.clean_text), so the requisites' words are the answer's."""
    from . import rule_quote
    from .interactive_flow import facts_status, DA_PROVARE
    sec = row.get("s") or {}
    if sec.get("plain_text"):
        row = {**row, "s": {**sec, "plain_text": rule_quote.clean_text(sec.get("plain_text"))}}
    rules = rule_quote.article_rules({}, [row], analysis, [tuple(row["_article"])])
    if not rules:
        return []
    rule = rules[0]
    texts = dict(rule.get("commi") or [])
    out: List[Dict[str, Any]] = []
    for k in (rule.get("selected") or [])[:COMMI_MAX]:
        reqs = (rule.get("requisites") or {}).get(k)
        items = reqs if reqs else [" ".join(str(texts.get(k) or "").split())[:300]]
        for text in items:
            if not text or len(out) >= REQUISITES_PER_ARTICLE:
                continue
            status = facts_status(text, analysis) if reqs else DA_PROVARE
            out.append({"id": "", "comma": k, "text": text, "kind": "requisito" if reqs else "comma",
                        "facts_status": status, "status": status, "why": "", "supporting": [], "contradicting": [],
                        "missing": "", "question": ""})
    return out


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


def build(topics: Sequence[Dict[str, Any]], rows: Sequence[Dict[str, Any]], analysis: Dict[str, Any],
          questions: Sequence[Dict[str, Any]] = (), others: Sequence[Dict[str, Any]] = (),
          grading: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
    """{label: record} for every article of the listed topics (kept: why), the altri profili (dropped: grado 1 / beyond
    the topics' number) and the prefilter's drops (dropped: prefiltro)."""
    from .interactive_flow import _key, _parse_label, _topic_refs
    by_key = {_key(r["_article"]): r for r in rows or [] if r.get("_article")}
    out: Dict[str, Dict[str, Any]] = {}
    from . import comma_material
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
            writable = ref in _topic_refs(t)
            out[label] = {"article": label, "rubrica": str(s.get("titolo_articolo") or t.get("rubrica") or ""),
                          "topic": t.get("title") or "", "grade": grade if isinstance(grade, int) else None,
                          "issue": str(row.get("_grade_issue") or ""), "kept": kept or ["elencato nel tema"],
                          "dropped": None if writable else "elencato, non scrivibile (né graduato né del tema)",
                          "requisites": _requisites(row, analysis) if row and writable else [],
                          "linked": "", "answer": {}}
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
    facts = row.get("_proc") or procedural_facts.from_row(row)
    if not (facts and facts.get("querela")):
        return []
    conds = facts.get("ufficio_condizioni")
    if conds is None:
        conds = procedural_facts.from_row(row).get("ufficio_condizioni") or []
    return [{"id": "", "comma": c.get("comma"), "text": c.get("text") or "", "opener": c.get("opener") or "",
             "kind": "procedibilita", "facts_status": DA_PROVARE, "status": DA_PROVARE, "why": "", "supporting": [],
             "contradicting": [], "missing": "", "question": ""} for c in conds if c.get("text")]


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
    sure_con = [s for s in con if s.get("certainty") == ACCERTATO]
    # v60t (1): «provato» needs a supporting sentence that is the case's own words (grounded) -- the client's account,
    # narrated or attributed (a parere rests on the facts as the client gives them); a hedged or hypothetical one never.
    # v60h's condition (the sentence holding the element's own nouns) is gone
    ground = [s for s in sup if grounded(s) and s.get("certainty") in (ACCERTATO, ALLEGATO)]
    if facts == PROVATO:
        status, why = PROVATO, "dai fatti estratti"
    elif facts == NON_RISULTA:
        status, why = NON_RISULTA, "i fatti estratti lo escludono"
    elif req.get("kind") not in ("requisito", "procedibilita"):
        status, why = facts, "comma intero: dai fatti estratti"
    elif sure_con:
        status, why = NON_RISULTA, f"il caso dice il contrario ({sure_con[0]['id']})"
    elif ground:
        status, why = PROVATO, f"lo dice il caso ({ground[0]['id']}, {ground[0].get('certainty') or ''})".replace(", )", ")")
    elif sup:
        how = sup[0].get("source") if sup[0].get("source") == "sistema" else sup[0].get("certainty")
        status, why = DA_PROVARE, f"nessuna frase del caso a sostegno, solo {how} ({sup[0]['id']})"
    else:
        status, why = DA_PROVARE, "nessun fatto del caso" + (f"; manca: {req['missing']}" if req.get("missing") else "")
    req["status"], req["why"] = status, why


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
    return [r for r in rec.get("requisites") or [] if r.get("status") == NON_RISULTA and _comma(r) == k]


def element_status(rec: Dict[str, Any], req: Dict[str, Any]) -> str:
    """v60r (3): an element's status as the reader gets it -- an unmet one outside the anchor comma is «non pertinente»
    (it excludes nothing, it proves nothing); else the record's."""
    from .interactive_flow import NON_RISULTA
    if req.get("status") == NON_RISULTA and req not in unmet_in_anchor(rec):
        return NOT_RELEVANT
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
    main line's anchor alone."""
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
    "fatto: lista vuota. In «manca», in poche parole, che cosa il caso non dice di quel requisito (vuoto se lo dice). "
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
        out[label] = {"comma": best["comma"], "text": best["text"], "kind": best["kind"]}
    return out


DEMOTED_CONTRADICTED = "contraddetto da una frase del caso narrata come fatto"
DEMOTED_EXCLUDED = "escluso dai fatti estratti"


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
    reason and the sentences (and, excluded, the facts) behind it."""
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
        sure = [x for x in against if x.get("certainty") == ACCERTATO]
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
                 questions: Sequence[Dict[str, Any]] = ()) -> str:
    """The SCHEDE for §2, in the writer's numbering (refs: the paragraphs' order): per article its verdict, then each
    requisite with its status, the case's sentences for and against it (their words and certainty, never their ids),
    what is missing and the open question asking it. "" with no record."""
    text_of = {s["id"]: s for s in sents or []}
    asked = {str(q.get("id")): q.get("text") or "" for q in questions or []}
    lines = []
    for i, ref in enumerate(refs, 1):
        rec = records.get(_label(ref))
        if not rec:
            continue
        head = f"[{i}] art. {_label(ref)}" + (f" — {rec['rubrica']}" if rec.get("rubrica") else "")
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
    return f"{WRITER_TITLE}:\n" + "\n".join(lines) if lines else ""


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
