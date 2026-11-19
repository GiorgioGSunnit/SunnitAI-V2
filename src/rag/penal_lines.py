"""v79.12.7: the penal reasoning's rules read from data/penal_lines.json (DATA_FILE) -- the tables are data, the code
here applies them; lawyer_method calls these at prepare, at the candidates and at the selection.

  (1) jurisprudential_elements  an element the law's text does not state but the lawyer's note di inquadramento
                                (article_notes, rec["note_di_inquadramento"]) or a kept ruling passage (rulings.build's
                                «kept») does (584: «prevedibilità in concreto dell'evento morte») enters the record as
                                inferred, «da provare», its source named (the note, the ruling's estremi); with
                                ELEMENT_SCHEMAS a reviewed schema already carrying it is used as is (tagged, never a
                                second copy). Contesting it is sostenibile, never minoritaria.
  (3) absorb                    a title absorbed by another in the records (584 absorbs 581/582) is out of the records;
                                one admitted only as the riqualificazione alternative (589/590 beside 584) is kept
                                unprinted as an alternative, never a threat; one the facts make inapposite (593 where a
                                soccorso is narrated) is out, the soccorso a favourable element after the fact.
                                absorbed_as_autonomous counts what is still autonomous (must be 0).
  (4) requalify_rule /          a riqualificazione between listed articles (584 -> 589/590) names the element that
      requalify_facts           differs and the facts bearing on it, read in the case's sentences; without them it is
                                not offered.
  (5) causation_cap /           a causation defence (art. 41 c.p.) on an ordinary consequence (a fall, a trauma) is
      denies_causation          capped at realismo basso unless the facts narrate a pathology, a medical error or an
                                abnormal intervening cause; a reply denying the causal link is replaced by one arguing
                                the subjective element.
  (2) circumstance_triggers /   the attenuanti read from the facts («circumstance_triggers»), replacing the fixed
      attenuanti_lines          attenuanti block: 62 n. 2 when a diverbio / provocazione / offesa is narrated (its
                                «da verificare: natura del diverbio» point tied to it), 62 n. 6 tied to the concrete
                                step narrated (trattativa con gli eredi, quantificazione, risarcimento offerto: named),
                                62-bis with record support only, «da verificare» without; the art. 69 c.p. line whenever
                                an aggravante is in the record. Never «esclusa» unless the record excludes it
                                (unsupported_exclusions: a written sentence saying so is rewritten).
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

DATA_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data",
                         "penal_lines.json")
_CACHE: Dict[str, Any] = {}

DA_PROVARE = "da provare"
SOURCE_NOTE = "nota di inquadramento dell'avvocato"          # my draft (v79.12.7): the note as an element's source
SOURCE_SCHEMA = "scheda rivista dall'avvocato"               # my draft (v79.12.7): a reviewed schema's element
# my draft (v79.12.7): why a jurisprudential element is «da provare» -- the record's «why», §4's row
INFERRED_WHY = "elemento di fonte giurisprudenziale ({source}), non scritto nel testo della norma: da provare"
# my drafts (v79.12.7): the lines under the main line (§6, the strategy) -- the source of the element it contests, the
# ruling's passage under it; the favourable element after the fact (a soccorso narrated)
SOURCE_LINE = "Fonte dell'elemento contestato: {source}"
SOURCE_PASSAGE_LINE = "Fonte dell'elemento contestato: {estremi}, passaggio della motivazione «{passage}»"
POST_FACT_LINE = ("Elemento favorevole successivo al fatto: {fact}, condotta susseguente al reato (art. 133, secondo "
                  "comma, n. 3 c.p.) da valere nella commisurazione della pena e per le attenuanti generiche "
                  "(art. 62-bis c.p.)")
# my drafts (v79.12.7): the riqualificazione argued through the element that differs and the facts bearing on it
REQUALIFY_TEXT = ("Riqualificazione nell'art. {art}{rubrica}: l'elemento che la distingue dall'art. {charge} è {alt} "
                  "anziché {base}; i fatti che lo riguardano: {facts}")
# my drafts (v79.12.7): the art. 41 line on an ordinary consequence -- capped; its «contro», and the reply that argues
# the subjective element instead of denying the causal link
CAUSATION_CONTRO = ("La caduta e il trauma sono conseguenze ordinarie della condotta: le concause non escludono il nesso "
                    "(art. 41, primo comma, c.p.) e i fatti non narrano una causa sopravvenuta da sola sufficiente "
                    "(art. 41, secondo comma, c.p.)")
CAUSATION_REPLY = ("Il nesso condizionalistico non è in discussione: la caduta e il trauma sono conseguenze ordinarie "
                   "della condotta (art. 41, primo comma, c.p.). La difesa contesta l'elemento soggettivo: la "
                   "prevedibilità in concreto dell'evento più grave, che resta da provare")

_LABEL_RE = re.compile(r"^\s*(\d+(?:-[a-z]+)?)\s+(.+?)(?:\s+co(?:mma|\.)\s*\d+.*)?\s*$", re.IGNORECASE)


def data() -> Dict[str, Any]:
    """data/penal_lines.json (read once; {} when missing or unreadable, logged)."""
    if "d" not in _CACHE:
        try:
            with open(DATA_FILE, encoding="utf-8") as f:
                _CACHE["d"] = json.load(f)
        except (OSError, ValueError) as exc:
            logger.warning("penal_lines: %s not read: %s", DATA_FILE, exc)
            _CACHE["d"] = {}
    return _CACHE["d"]


def _log(event: str, payload: Any) -> None:
    try:
        from .verbose_logger import vlog
        vlog(f"penal_lines_{event}", payload)
    except Exception:           # noqa: BLE001 -- logging never fails the answer
        pass


def _plain(text: Any) -> str:
    return " ".join(str(text or "").split())


def article(label: str) -> str:
    """«584 c.p.» of «584 c.p.», «584 c.p. co. 2», «584 cp» -- the number and the code key, no comma."""
    m = _LABEL_RE.match(str(label or ""))
    if not m:
        return ""
    from .plan_then_write import _code_key
    return f"{m.group(1).lower()} {_code_key(m.group(2).strip())}"


def same(label: str, spec: str) -> bool:
    a = article(label)
    return bool(a) and a == article(spec)


def listed(label: str, specs: Sequence[str]) -> bool:
    return any(same(label, s) for s in specs or [])


def _search(patterns: Sequence[str], text: str) -> Optional[re.Match]:
    for p in patterns or []:
        try:
            m = re.search(p, text or "", re.IGNORECASE)
        except re.error as exc:
            logger.warning("penal_lines: bad pattern %r: %s", p, exc)
            continue
        if m:
            return m
    return None


def _short(text: Any, limit: int = 160) -> str:
    text = _plain(text)
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "…"


# --- (1) jurisprudential elements --------------------------------------------------------------------------------------

def _rules_for(label: str) -> List[Dict[str, Any]]:
    return [r for r in data().get("jurisprudential_elements") or [] if isinstance(r, dict)
            and listed(label, r.get("articles") or [])]


def max_id(*groups: Optional[Dict[str, Dict[str, Any]]]) -> int:
    """The highest «R<n>» element id of the records given (0 for none)."""
    n = 0
    for group in groups:
        for rec in (group or {}).values():
            for r in (rec or {}).get("requisites") or []:
                m = re.match(r"^R(\d+)$", str(r.get("id") or ""))
                if m:
                    n = max(n, int(m.group(1)))
    return n


def _next_id(records: Dict[str, Dict[str, Any]], circ: Optional[Dict[str, Dict[str, Any]]] = None,
             floor: int = 0) -> str:
    return f"R{max(max_id(records, circ), floor) + 1}"


def _holds(rec: Dict[str, Any], rule: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The record's element already stating the rule's element (its patterns in the element's text), if any."""
    for r in rec.get("requisites") or []:
        if _search(rule.get("patterns") or [], str(r.get("text") or "")):
            return r
    return None


_STATEMENT_END_RE = re.compile(r"[.;!?](?=\s|$)")


def states_element(rule: Dict[str, Any], text: str) -> bool:
    """(review) The text states the rule's element as required: one of its patterns hits where none of the rule's
    «denial_patterns» stands in the statement around it -- from the statement's start (a «.», a «;») to the hit, and from
    the hit to the next comma or the statement's end («non è richiesta la prevedibilità in concreto», «la prevedibilità
    in concreto … non è necessaria»: a passage or a note saying the element is NOT required is no source of it)."""
    text = _plain(text)
    denials = [p for p in rule.get("denial_patterns") or [] if p]
    for p in rule.get("patterns") or []:
        try:
            hits = list(re.finditer(p, text, re.IGNORECASE))
        except re.error as exc:
            logger.warning("penal_lines: bad pattern %r: %s", p, exc)
            continue
        for m in hits:
            ends = [x.end() for x in _STATEMENT_END_RE.finditer(text, 0, m.start())]
            start = ends[-1] if ends else 0
            after = text[m.end():]
            stop = re.search(r"[,.;!?](?=\s|$)", after)
            around = text[start:m.start()] + " " + m.group(0) + " " + (after[:stop.start()] if stop else after)
            if not _search(denials, around):
                return True
    return False


def _sources(label: str, rec: Dict[str, Any], rule: Dict[str, Any], kept: Sequence[Dict[str, Any]]
             ) -> List[Dict[str, Any]]:
    """The note's and the kept passages' statements of the rule's element, for the record's article -- (review) a
    statement denying the element is required is none (states_element)."""
    out = []
    note = _plain(rec.get("note_di_inquadramento"))
    if note and states_element(rule, note):
        out.append({"kind": "note", "source": SOURCE_NOTE})
    for k in kept or []:
        if not isinstance(k, dict) or not k.get("passage") or not same(str(k.get("article") or ""), label):
            continue
        if states_element(rule, str(k.get("passage") or "")):
            out.append({"kind": "ruling", "source": str(k.get("estremi") or ""), "estremi": str(k.get("estremi") or ""),
                        "passage": str(k.get("passage") or ""), "ruling_id": k.get("id")})
    return out


def source_text(sources: Sequence[Dict[str, Any]]) -> str:
    """The sources named in words: the note, the rulings' estremi (each once)."""
    names = []
    for s in sources or []:
        name = _plain(s.get("source"))
        if name and name not in names:
            names.append(name)
    return " e ".join(names)


def jurisprudential_elements(records: Dict[str, Dict[str, Any]], kept: Sequence[Dict[str, Any]] = (),
                             schemas_on: bool = False, circ: Optional[Dict[str, Dict[str, Any]]] = None,
                             log: Optional[Dict[str, Any]] = None, id_floor: int = 0) -> List[Dict[str, Any]]:
    """In place: each record of a listed article gets its jurisprudential element when the note or a kept passage
    states it and the text does not -- inferred, «da provare», its source(s) named. An element already in the record
    (a reviewed schema's, ELEMENT_SCHEMAS on; or added before, from the note) is used as is: tagged, its sources
    extended, never a second copy, never «provato» (the jurisprudence's evaluative element is the judge's, the facts
    narrate none of it). The elements added or tagged, [{article, id, element, sources, how}]."""
    out: List[Dict[str, Any]] = []
    for label, rec in (records or {}).items():
        if not isinstance(rec, dict) or rec.get("comma_of"):
            continue
        for rule in _rules_for(label):
            have = _holds(rec, rule)
            sources = _sources(label, rec, rule, kept)
            reviewed = schemas_on and isinstance(rec.get("schema"), dict) and rec["schema"].get("reviewed") is True
            if have is not None and (have.get("jurisprudential") or (reviewed and have.get("split") == "schema")):
                if not have.get("jurisprudential"):
                    sources = [{"kind": "schema", "source": SOURCE_SCHEMA}] + sources
                old = list(have.get("sources") or [])
                merged = old + [s for s in sources if s.get("source") not in {o.get("source") for o in old}]
                have.update(jurisprudential=True, inferred=True, sources=merged, status=DA_PROVARE,
                            source=source_text(merged), rule=rule.get("id"),
                            why=INFERRED_WHY.format(source=source_text(merged)))
                if merged != old:
                    out.append({"article": label, "id": have.get("id"), "element": have.get("text"),
                                "sources": [s.get("source") for s in merged], "how": "tagged"})
                continue
            if have is not None or not sources:     # the text states it (no inference), or nothing states it
                continue
            el = {"id": _next_id(records, circ, id_floor), "comma": 1, "kind": "requisito", "text": rule["element"],
                  "status": DA_PROVARE, "facts_status": DA_PROVARE, "supporting": [], "contradicting": [],
                  "missing": "", "question": "", "split": "giurisprudenza", "inferred": True,
                  "jurisprudential": True, "rule": rule.get("id"), "sources": sources,
                  "source": source_text(sources), "why": INFERRED_WHY.format(source=source_text(sources))}
            rec.setdefault("requisites", []).append(el)
            out.append({"article": label, "id": el["id"], "element": el["text"],
                        "sources": [s.get("source") for s in sources], "how": "added"})
    if out:
        _log("jurisprudential", out)
    if log is not None:
        log.setdefault("jurisprudential", []).extend(out)
    return out


def element_source_lines(element: Dict[str, Any]) -> List[str]:
    """§6, under the main line: the contested element's sources -- each ruling with its passage (redacted as §5
    prints it), the note or the reviewed schema named."""
    from . import rulings
    lines = []
    for s in element.get("sources") or []:
        if s.get("kind") == "ruling" and s.get("passage"):
            lines.append(SOURCE_PASSAGE_LINE.format(estremi=s.get("estremi") or s.get("source"),
                                                    passage=_short(rulings.printed(s["passage"]), 400).rstrip(". ")))
        elif s.get("source"):
            lines.append(SOURCE_LINE.format(source=s["source"]))
    return list(dict.fromkeys(lines))


# --- (3) absorbed and inapposite titles -------------------------------------------------------------------------------

_CLAUSE_EDGE_RE = re.compile(r"[;:]|,\s|\s(?:e|ed|ma|mentre|poi|quindi|però|finché|dopo\s+che|quando)\s", re.IGNORECASE)


def clause_at(text: str, start: int, end: int) -> Tuple[str, str]:
    """(review) (the clause holding text[start:end], what of it precedes the hit) -- a clause ends at «;», «:», «, » or a
    connective («e», «ma», «mentre» …)."""
    left = 0
    for m in _CLAUSE_EDGE_RE.finditer(text, 0, start):
        left = m.end()
    m = _CLAUSE_EDGE_RE.search(text, end)
    right = m.start() if m else len(text)
    return text[left:right], text[left:start]


def soccorso_narrated(sentences: Sequence[Dict[str, Any]], rule: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The case's sentences narrating the soccorso the offender gave: a pattern hit no negation precedes (within 40
    characters or in its own clause) -- (review) in a clause naming no one else who gave it (the rule's
    «not_by_offender_patterns»: passers-by, a neighbour, a witness; a passive «fu portato al pronto soccorso») and never
    the place «pronto soccorso» alone: help the offender refused, or others gave, is no favourable element of his."""
    out = []
    neg = rule.get("negation_pattern") or ""
    others = [p for p in rule.get("not_by_offender_patterns") or [] if p]
    for s in sentences or []:
        text = str(s.get("text") or "")
        for p in rule.get("narrated_patterns") or []:
            hit = None
            for m in re.finditer(p, text, re.IGNORECASE):
                before = text[max(0, m.start() - 40):m.start()]
                clause, lead = clause_at(text, m.start(), m.end())
                if neg and (re.search(neg, before, re.IGNORECASE) or re.search(neg, lead, re.IGNORECASE)):
                    continue
                if _search(others, clause):
                    continue
                hit = m
                break
            if hit:
                out.append(s)
                break
    return out


def _present(records: Dict[str, Dict[str, Any]], specs: Sequence[str]) -> List[str]:
    """The absorber's records standing as the case's offence -- never one that is itself only an alternative, a ranked
    candidate, unprinted, or a special comma's record."""
    return [x for x in records if listed(x, specs) and not any(records[x].get(k) for k in (
        "alternative", "ranked_candidate", "unprinted", "comma_of"))]


def absorb(records: Dict[str, Dict[str, Any]], sentences: Sequence[Dict[str, Any]] = ()
           ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """(records, report): the records with every absorbed title out (mode «absorbed», «post_fact») or kept only as the
    riqualificazione alternative («alternative_only»: a copy, unprinted, «alternative», never a threat nor charged);
    report {"absorbed": [{article, by, mode, why}], "post_fact": [{text, sid}]}. The records are never mutated."""
    out = dict(records or {})
    report: Dict[str, Any] = {"absorbed": [], "post_fact": []}
    for rule in data().get("absorption") or []:
        if not isinstance(rule, dict):
            continue
        mode = rule.get("mode") or "absorbed"
        targets = [x for x in out if listed(x, rule.get("absorbed") or [])]
        if not targets:
            continue
        if mode == "post_fact":
            hits = soccorso_narrated(sentences, rule)
            if not hits:
                continue
            for x in targets:
                out.pop(x, None)
                report["absorbed"].append({"article": x, "by": "", "mode": mode, "why": rule.get("why") or ""})
            for s in hits:
                if all(p["sid"] != s.get("id") for p in report["post_fact"]):
                    report["post_fact"].append({"text": _plain(s.get("text")), "sid": s.get("id") or ""})
            continue
        by = [x for x in _present(out, rule.get("absorber") or []) if x not in targets]
        if not by:
            continue
        for x in targets:
            if mode == "alternative_only":
                rec = {k: v for k, v in out[x].items() if k not in ("side", "charged", "named_due", "anchor_due",
                                                                     "charge_hypothesis", "exposure")}
                out[x] = {**rec, "alternative": True, "unprinted": True, "alternative_of": by[0]}
            else:
                out.pop(x, None)
            report["absorbed"].append({"article": x, "by": by[0], "mode": mode, "why": rule.get("why") or ""})
    if report["absorbed"]:
        _log("absorbed", report)
    return out, report


_FLOW_DROPPED = ("side", "side_why", "charged", "named_due", "anchor_due", "charge_hypothesis", "exposure")


def absorb_into(records: Optional[Dict[str, Dict[str, Any]]], report: Dict[str, Any]) -> List[str]:
    """(review) In place: the flow's records (state["records"]: what the flow's own paths read -- its riti, the anchor
    blocks, the article citations) made to agree with the absorption report -- an «absorbed» or «post_fact» title out, an
    «alternative_only» one an alternative, never a threat nor charged. The labels changed."""
    if not isinstance(records, dict):
        return []
    done = []
    for a in (report or {}).get("absorbed") or []:
        x = a.get("article")
        if x not in records:
            continue
        if a.get("mode") == "alternative_only":
            rec = records[x]
            if not isinstance(rec, dict):
                continue
            for k in _FLOW_DROPPED:
                rec.pop(k, None)
            rec["alternative"] = True
            rec["alternative_of"] = a.get("by") or ""
        else:
            records.pop(x, None)
        done.append(x)
    return done


def absorbed_as_autonomous(ctx: Dict[str, Any]) -> int:
    """The absorbed or inapposite titles still offered as autonomous in the answer's material (must be 0): a record
    printed or marked threat / charged, or a candidate on it -- an «alternative_only» title's riqualificazione excepted
    -- re-read on the final records and candidates. (review) The flow's records too (ctx["flow_records"]: the state's,
    which the flow's own paths print): an absorbed title still there, or an alternative-only one still a threat or
    charged, counts."""
    n = _autonomous_in(ctx.get("records") or {}, ctx.get("candidates") or [], ctx.get("sentences") or [])
    flow = ctx.get("flow_records")
    if isinstance(flow, dict):
        n += _autonomous_in(flow, [], ctx.get("sentences") or [], flow=True)
    return n


def _autonomous_in(records: Dict[str, Dict[str, Any]], cands: Sequence[Dict[str, Any]],
                   sentences: Sequence[Dict[str, Any]], flow: bool = False) -> int:
    owner = {str(r.get("id")): x for x, rec in records.items() for r in (rec or {}).get("requisites") or []
             if isinstance(rec, dict) and r.get("id")}
    n = 0
    for rule in data().get("absorption") or []:
        if not isinstance(rule, dict):
            continue
        mode = rule.get("mode") or "absorbed"
        if mode == "post_fact":
            if not soccorso_narrated(sentences, rule):
                continue
        elif not _present(records, rule.get("absorber") or []):
            continue
        for x, rec in records.items():
            if not listed(x, rule.get("absorbed") or []):
                continue
            if mode == "alternative_only":
                n += int(bool(rec.get("side") == "threat" or rec.get("charged")
                              or (not flow and not rec.get("unprinted"))))
            else:
                n += 1
        for c in cands:
            arts = [c.get("article") or ""] + [owner.get(str(i), "") for i in list(c.get("contesta") or [])
                                               + list(c.get("poggia_su") or [])]
            if not any(a and listed(a, rule.get("absorbed") or []) for a in arts):
                continue
            if mode == "alternative_only" and c.get("kind") == "requalify":
                continue
            n += 1
    return n


def post_fact_lines(ctx: Dict[str, Any]) -> List[str]:
    """§6: the favourable elements after the fact (a soccorso narrated), the imputato's only."""
    role = str((ctx.get("posture") or {}).get("role") or "")
    if role not in ("imputato", "indagato"):
        return []
    return [POST_FACT_LINE.format(fact=f"«{_short(p['text'], 160).rstrip('. ')}»") for p in ctx.get("post_fact") or []
            if p.get("text")]


# --- (4) the riqualificazione through the element that differs ----------------------------------------------------------

def requalify_rule(charge: str, alternative: str) -> Optional[Dict[str, Any]]:
    """The data's rule for a riqualificazione of charge into alternative, None when none is listed."""
    return next((r for r in data().get("requalify_elements") or [] if isinstance(r, dict)
                 and listed(charge, r.get("from") or []) and listed(alternative, r.get("to") or [])), None)


def requalify_facts(rule: Dict[str, Any], sentences: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The case's sentences bearing on the differing element (a fact pattern in them), each once -- (review) a hit the
    rule's «negation_pattern» precedes (within 40 characters: «nega di aver dato alcuna spinta», «non vi fu alcuna
    spinta») is no fact bearing on it."""
    by_id = {str(s.get("id") or ""): s for s in sentences or []}
    return [{"sid": f["sid"], "text": f["text"], "certainty": (by_id.get(f["sid"]) or {}).get("certainty") or ""}
            for f in _narrated(rule.get("fact_patterns") or [], sentences, str(rule.get("negation_pattern") or ""))]


def requalify_text(rule: Dict[str, Any], art: str, rubrica: str, charge: str, facts: Sequence[Dict[str, Any]]) -> str:
    return REQUALIFY_TEXT.format(art=art, rubrica=rubrica, charge=charge, alt=rule.get("alternative_element") or "",
                                 base=rule.get("charge_element") or "",
                                 facts="; ".join(f"«{_short(f['text'], 140).rstrip('. ')}»" for f in facts[:3]))


# --- (5) art. 41 realism -------------------------------------------------------------------------------------------------

def causation_line(label: str) -> bool:
    rule = data().get("causation_realism") or {}
    return listed(label, rule.get("articles") or [])


def causation_cap(text: str) -> Dict[str, Any]:
    """{"capped", "ordinary", "triggers": {kind: matched words}} of the case's facts: capped when they narrate an
    ordinary consequence and no trigger (a pathology, a medical error, an abnormal intervening cause)."""
    rule = data().get("causation_realism") or {}
    ordinary = _search(rule.get("ordinary_patterns") or [], text or "")
    neg = str(rule.get("trigger_negation_pattern") or "")
    triggers = {}
    for kind, pats in (rule.get("triggers") or {}).items():
        m = _unnegated(pats, text or "", neg)       # (review) «non soffriva di alcuna patologia» lifts nothing
        if m:
            triggers[kind] = m.group(0)
    return {"capped": bool(ordinary) and not triggers, "ordinary": ordinary.group(0) if ordinary else "",
            "triggers": triggers}


def _unnegated(patterns: Sequence[str], text: str, neg: str = "") -> Optional[re.Match]:
    """The first hit of the patterns no negation precedes (within 40 characters, or in its own clause)."""
    for p in patterns or []:
        try:
            found = list(re.finditer(p, text or "", re.IGNORECASE))
        except re.error as exc:
            logger.warning("penal_lines: bad pattern %r: %s", p, exc)
            continue
        for m in found:
            before = text[max(0, m.start() - 40):m.start()]
            _, lead = clause_at(text, m.start(), m.end())
            if neg and (re.search(neg, before, re.IGNORECASE) or re.search(neg, lead, re.IGNORECASE)):
                continue
            return m
    return None


def denies_causation(text: str) -> bool:
    """A sentence denying the causal link (interrupted, excluded, a later sufficient cause)."""
    rule = data().get("causation_realism") or {}
    return bool(_search(rule.get("deny_patterns") or [], text or ""))


# --- (2) attenuanti from the facts -------------------------------------------------------------------------------------

# my drafts (v79.12.7, item 2): the attenuanti lines that replace the fixed block (§6, under the containment or the
# subordinate line) -- a fact narrated, a concrete step named, the point tied to it; «da verificare» without a fact;
# «esclusa» only where the record excludes it (a sentence of the case says the contrary); art. 69's bilanciamento
CIRC_FACT_LINE = "{name} ({article}): i fatti narrano {facts}"
CIRC_STEP_LINE = "{name} ({article}): il passo concreto narrato è {step}, {facts}"
CIRC_POINT = "da verificare: {point}"
CIRC_TO_VERIFY_LINE = "{name} ({article}): da verificare, {to_verify}"
CIRC_EXCLUDED_LINE = "{name} ({article}): esclusa dai fatti, {facts}"
POST_FACT_SUPPORT = "la condotta successiva al fatto {fact}"
BILANCIAMENTO_LINE = ("Bilanciamento delle circostanze ({article}): il registro contiene {aggravanti}; le attenuanti "
                      "riconosciute si bilanciano con le aggravanti: se prevalenti la pena si diminuisce, se equivalenti "
                      "si applica la pena prevista senza circostanze, se subvalenti si applica l'aumento; il calcolo dei "
                      "riti assume le attenuanti prevalenti")
# my draft (v79.12.7, item 2): a written sentence calling a circumstance excluded that the record does not exclude
CIRC_UNEXCLUDED = "{name} ({article}): resta aperta sui fatti del caso{open}."
CIRC_UNEXCLUDED_OPEN = " (da verificare: {open})"
CIRC_NAMES = {"aggravante": "Aggravante", "attenuante": "Attenuante"}
NON_RISULTA, PROVATO = "non risulta", "provato"


def _narrated(patterns: Sequence[str], sentences: Sequence[Dict[str, Any]], neg: str = "",
              exclude: Sequence[str] = ()) -> List[Dict[str, Any]]:
    """The case's sentences holding one of the patterns that no negation precedes (within 40 characters), each once --
    (review) exclude: patterns that, in the hit's own clause, make it no hit (62 n. 6: «la parte civile chiede il
    risarcimento, da quantificare» is the victim's claim, no step of the offender)."""
    out = []
    for s in sentences or []:
        text = str(s.get("text") or "")
        hit = False
        for p in patterns or []:
            try:
                found = list(re.finditer(p, text, re.IGNORECASE))
            except re.error as exc:
                logger.warning("penal_lines: bad pattern %r: %s", p, exc)
                continue
            for m in found:
                before = text[max(0, m.start() - 40):m.start()]
                if neg and re.search(neg, before, re.IGNORECASE):
                    continue
                if exclude and _search(exclude, clause_at(text, m.start(), m.end())[0]):
                    continue
                hit = True
                break
            if hit:
                break
        if hit:
            out.append({"sid": s.get("id") or "", "text": _plain(text)})
    return out


def trigger_rules() -> List[Dict[str, Any]]:
    return [r for r in data().get("circumstance_triggers") or [] if isinstance(r, dict) and r.get("label")]


def circumstance_triggers(sentences: Sequence[Dict[str, Any]], post_fact: Sequence[Dict[str, Any]] = ()
                          ) -> Dict[str, Dict[str, Any]]:
    """{label: {"facts": [{sid, text, post_fact?}], "step": name}} of the attenuanti the facts trigger -- a pattern hit
    (62 n. 2), the first concrete step hit (62 n. 6, the step named), a support pattern hit or, with «post_fact», a
    favourable element after the fact (62-bis); a negated hit never counts. A label nothing triggers is absent."""
    neg = str(data().get("circumstance_negation_pattern") or "")
    out: Dict[str, Dict[str, Any]] = {}
    for rule in trigger_rules():
        facts: List[Dict[str, Any]] = []
        step = ""
        if rule.get("patterns"):
            facts = _narrated(rule["patterns"], sentences, neg)
        for st in rule.get("steps") or []:
            got = _narrated(st.get("patterns") or [], sentences, neg, rule.get("step_exclude_patterns") or [])
            if got:
                facts, step = got, str(st.get("name") or "")
                break
        if rule.get("support_patterns"):
            facts += [f for f in _narrated(rule["support_patterns"], sentences, neg)
                      if all(f["sid"] != x["sid"] for x in facts)]
        if rule.get("post_fact"):       # the soccorso narrated (the «post_fact» absorption rule's patterns), 593 or not
            after = [{"sid": p.get("sid") or "", "text": _plain(p.get("text"))} for p in post_fact or [] if p.get("text")]
            for a in data().get("absorption") or []:
                if isinstance(a, dict) and a.get("mode") == "post_fact":
                    after += [{"sid": x.get("id") or "", "text": _plain(x.get("text"))} for x in soccorso_narrated(
                        sentences, a)]
            for p in after:
                if all(p["sid"] != x["sid"] for x in facts):
                    facts.append({**p, "post_fact": True})
        if facts:
            out[str(rule["label"])] = {"facts": facts, "step": step}
    if out:
        _log("circumstance_triggers", {k: [f["sid"] for f in v["facts"]] for k, v in out.items()})
    return out


def excluded(rec: Optional[Dict[str, Any]]) -> List[str]:
    """The sentence ids by which the record excludes a circumstance -- an element «non risulta» with a contradicting
    sentence of the case ([] when the record does not exclude it)."""
    out: List[str] = []
    for r in (rec or {}).get("requisites") or []:
        if r.get("kind", "requisito") == "requisito" and r.get("status") == NON_RISULTA:
            out += [str(x) for x in r.get("contradicting") or [] if x]
    return list(dict.fromkeys(out))


def supported(rec: Optional[Dict[str, Any]]) -> bool:
    """The record holds the circumstance: an element «provato» or with a supporting sentence, or a trigger fact."""
    if (rec or {}).get("triggered_by"):
        return True
    return any(r.get("status") == PROVATO or r.get("supporting") or (r.get("link") or {}).get("supporting")
               for r in (rec or {}).get("requisites") or [] if r.get("kind", "requisito") == "requisito")


def _quote(facts: Sequence[Dict[str, Any]], limit: int = 2) -> str:
    return "; ".join(f"«{_short(f['text'], 140).rstrip('. ')}»" for f in list(facts)[:limit])


def attenuanti_lines(triggers: Dict[str, Dict[str, Any]], circ: Optional[Dict[str, Dict[str, Any]]],
                     sentences: Sequence[Dict[str, Any]] = (), log: Optional[Dict[str, Any]] = None) -> List[str]:
    """The attenuanti lines from the facts (CIRC_* lines, the table's order), then art. 69's (BILANCIAMENTO_LINE)
    whenever an aggravante is in the record (one the record holds and does not exclude). An attenuante the record
    excludes: «esclusa dai fatti» with the sentence that says so; one the facts trigger: the facts (the step named, the
    point tied to it, the condition); neither: the table's «absent» -- «to_verify» its «da verificare» line, «omit»
    nothing. Never «esclusa» on absent facts. log: {"lines": {label: how}, "aggravanti": [...]}."""
    by_id = {str(s.get("id") or ""): s for s in sentences or []}
    circ = circ or {}
    lines: List[str] = []
    how: Dict[str, str] = {}
    for rule in trigger_rules():
        label = str(rule["label"])
        name, article_ = str(rule.get("name") or label), str(rule.get("article") or f"art. {label}")
        rec = circ.get(label)
        out_ids = excluded(rec)
        got = (triggers or {}).get(label) or {}
        if out_ids:
            facts = [{"text": by_id[i].get("text")} for i in out_ids if i in by_id]
            if not facts:
                continue
            lines.append(CIRC_EXCLUDED_LINE.format(name=name, article=article_, facts=_quote(facts, 1)))
            how[label] = "excluded"
        elif got.get("facts"):
            plain = [f for f in got["facts"] if not f.get("post_fact")]
            after = [f for f in got["facts"] if f.get("post_fact")]
            parts = [_quote(plain)] if plain else []
            parts += [POST_FACT_SUPPORT.format(fact=_quote(after, 1))] if after else []
            facts = "; ".join(p for p in parts if p)
            line = CIRC_STEP_LINE.format(name=name, article=article_, step=got["step"], facts=facts) if got.get("step") \
                else CIRC_FACT_LINE.format(name=name, article=article_, facts=facts)
            if rule.get("point"):
                line += "; " + CIRC_POINT.format(point=rule["point"])
            if rule.get("condition"):
                line += "; " + str(rule["condition"])
            lines.append(line)
            how[label] = "step" if got.get("step") else "facts"
        elif rule.get("absent") == "to_verify" and rule.get("to_verify"):
            lines.append(CIRC_TO_VERIFY_LINE.format(name=name, article=article_, to_verify=rule["to_verify"]))
            how[label] = "to_verify"
    aggr = aggravanti_held(circ)
    if aggr:
        b = data().get("bilanciamento") or {}
        named = (f"l'aggravante dell'art. {aggr[0]}" if len(aggr) == 1 else
                 "le aggravanti degli artt. " + ", ".join(x.replace(" c.p.", "") for x in aggr[:-1]) + " e " + aggr[-1])
        lines.append(BILANCIAMENTO_LINE.format(article=b.get("article") or "art. 69 c.p.", aggravanti=named))
    if log is not None:
        log.update({"lines": how, "aggravanti": aggr})
    return [x.rstrip(". ") + "." for x in lines]


def aggravanti_held(circ: Optional[Dict[str, Dict[str, Any]]]) -> List[str]:
    """The aggravanti in the record -- each aggravante circumstance record the record holds (supported) and does not
    exclude."""
    kinds = set((data().get("bilanciamento") or {}).get("kinds") or ["aggravante"])
    return [str(x) for x, rec in (circ or {}).items() if (rec or {}).get("kind") in kinds and not excluded(rec)
            and supported(rec)]


def _rx(key: str) -> Optional["re.Pattern"]:
    pats = [p for p in data().get(key) or [] if p]
    if not pats:
        return None
    try:
        return re.compile("|".join(f"(?:{p})" for p in pats), re.IGNORECASE)
    except re.error as exc:
        logger.warning("penal_lines: bad %s: %s", key, exc)
        return None


def says_excluded(sentence: str) -> bool:
    """A sentence saying something is excluded (exclusion_patterns), not one saying it cannot be excluded
    (not_exclusion_patterns)."""
    yes, no = _rx("exclusion_patterns"), _rx("not_exclusion_patterns")
    text = sentence or ""
    if not yes or not yes.search(text):
        return False
    if no:
        text = no.sub(" ", text)
    return bool(yes.search(text))


def circumstance_name(label: str, rec: Optional[Dict[str, Any]], table: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
    """(name, article) of a circumstance for CIRC_UNEXCLUDED -- the trigger table's when it has one, else the kind and
    the table's name («Attenuante: provocazione»)."""
    rule = next((r for r in trigger_rules() if str(r["label"]) == label), None)
    if rule:
        return str(rule.get("name") or label), str(rule.get("article") or f"art. {label}")
    kind = str((rec or {}).get("kind") or (table or {}).get("kind") or "")
    nm = str((table or {}).get("name") or "")
    return (f"{CIRC_NAMES.get(kind, 'Circostanza')}{' (' + nm + ')' if nm else ''}", f"art. {label}")
