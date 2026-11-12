"""INTERACTIVE_FLOW (default false): a narrative case answered in rounds instead of one parere. Off, or on a request the
router reads as "direct", nothing changes: the single-shot pipeline (defensive_generation._case_analysis_pipeline) is
neither called differently nor edited; this module only imports its parts.

1. analyse_case(): on a narrative case message, the facts (fact_signals: regimes, terms, mediation, the parties with the
   current-owner rule, the procedural tools, the action gate), the frame, retrieval as the pipeline runs it, the topics
   (topic_step) with their articles' texts read once, and the rendered computed block (render_computed). No plan, no
   writing. The whole object is the session's state (new_state), kept in the session's anchors under STATE_KEY with a
   TTL of TTL_HOURS and a version that every later change raises; a follow-up reads no corpus.
2. Round 1 (round1): "Argomenti individuati" (each topic: title, anchor articles, its «da verificare» line), "Fatti
   accertati" (parties with roles, dates, the computed terms), "Da chiarire" (at most MAX_QUESTIONS questions, from the
   unknown fact fields that change a term or a requisite, ranked by how much they change: questions()), and
   INSTRUCTION. State TOPICS_OFFERED.
3. A reply (parse_reply): numbers -> topics, "procedi / ok / vai / genera" -> confirm, "tutti" -> every topic, without a
   model; anything else one bounded JSON call (_REPLY_SYSTEM): the open questions' answers as fact values, an angle (free
   text matched against the topics), corrections to parties and dates. apply_reply() puts them into the facts and
   recomputes the terms, the rules, the requisites and the computed block (recompute()), every change logged.
4. Round 2 (round2): the frame that will be written, ending with CONFIRM_LINE. State FRAME_RESTATED; a confirmation
   -> FRAME_CONFIRMED and the answer; a correction back to 3. "procedi" at round 1 skips round 2.
5. The topic answer (write_answer): a plan on the chosen topics' articles (at most TOPIC_ARTICLES each) and the
   procedural tools' rows; ANSWER_TITLES: Risposta e strategia; Norme applicate ai fatti (the topic's rules quoted,
   rule_quote, each requisite «provato» / «da provare» from the fact fields only: requisite_status); Obiezioni e
   repliche (objections(): the «da verificare» items, tolleranza / ritardo / cortili e giardini for a real action, at
   most MAX_OBJECTIONS, one reply each); Profili processuali (the rendered block, not written); Conclusioni. Then
   TERM_CONSISTENCY (term_consistency), UNSUPPORTED_OUT (unsupported_out: an article with no requisite the facts support
   goes to OTHER_PROFILES_TITLE -- a chosen topic's anchor excepted), the RULE_QUOTE check with the extended verbs, the repetition guard, the citation check
   against the topic set and the block (citation_check), the rubrica stamp with its text check (the 1061 fix:
   RUBRICA_TEXT_CHECK). It ends with "Posso approfondire: <the other topics>". "approfondisci N" answers another topic
   from the cache; "aggiungi che ..." is a new fact: recomputed, and the answer offered again.

v54 (round-1 quality): the topics as topic_step builds them -- no generic rubrica anchor, the frame's unsupported
institutions as "altri profili" (others), every open term a topic, a decided one a fact, gated actions out of the titles;
the questions per topic from its anchor's requisites unknown in the facts (_REQUISITE_FACTS -> _FAMILY), on the client's
side; the writer's requisite statuses enforced (strict_status) and a requisite «da provare» putting every sentence on it
in the conditional (conditional_on_unknown). Every text the rounds show comes from interaction_copy (no model wording).

v60 (INTERACTIVE_CASE_FIRST; the PO's A-H): the first read beside retrieval (case_read), the topics from the graded rows
(topic_groups), round-1 grading limited to two rounds on the embedding prefilter's top N and the chosen topics' rest graded
in the answer step (grade_chosen); always: the procedural facts from the article's text in §4 and against the writer's
sentences (procedural_facts), the client settled before writing (settle_client), the read-back pass (readback), replies
read for their intent with one clarifying question as the fallback (follow_up), closing lines that dictate no string.

Every step is isolated (_isolated): a failure is logged with its traceback (interactive_step_failed) and the turn goes
on without it. The API response carries `interaction` (interaction()): {schema, version, state, round, topics, facts,
questions, chosen, changes, actions}.
"""

import copy
import json
import logging
import os
import re
import time
import traceback
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import feature_flags
from . import interaction_copy as C
from .topic_step import OPEN_TERM_STATUS

logger = logging.getLogger(__name__)

FLAG = "INTERACTIVE_FLOW"
STATE_KEY = "interactive"
SCHEMA = 1
TTL_HOURS = 24
MAX_QUESTIONS = 5
TOPIC_ARTICLES = 8
MAX_OBJECTIONS = 4
ROW_TEXT_CHARS = 6000           # an article's text kept in the session
NEW_CASE_CHARS = 400            # a narrative this long, in an open interaction, is a new case
_REPLY_TOKENS = 600

TOPICS_OFFERED, FRAME_RESTATED, FRAME_CONFIRMED, ANSWERED = "topics_offered", "frame_restated", "frame_confirmed", "answered"
INSTRUCTION = C.INSTRUCTION
CONFIRM_LINE = C.CONFIRM_LINE
ANSWER_TITLES = C.SECTION_TITLES
OTHER_PROFILES_TITLE = C.OTHER_PROFILES_SECTION
CONCLUSIONS_TOKENS = 1000       # v54: +300, so the conclusions' list completes
RULES_TITLE = "REGOLE DELLE NORME DELL'ARGOMENTO"
RULES_TEXT = ("Enuncia la regola di ciascuna norma riportando il comma tra «…» con il suo numero («co. N») o con parole tue "
              "che restano dentro quel comma. Ogni requisito numerato è già segnato «provato» o «da provare» dai fatti: "
              "riporta quello stato, non cambiarlo, non aggiungere requisiti; per un requisito da provare indica la prova "
              "da acquisire.")
PROVATO, DA_PROVARE, NON_RISULTA = C.STATUS_PROVATO, C.STATUS_DA_PROVARE, C.STATUS_NON_RISULTA
CASE_KEY = "_case_text"         # the case as written, in the analysis: what "the case states" is read against


def enabled() -> bool:
    return feature_flags.enabled(FLAG)


# v60: the case-first method (A first read, B topics from the graded rows, G round-1 grading limits)
CASE_FIRST_FLAG = "INTERACTIVE_CASE_FIRST"
ROUND1_GRADE_ROUNDS = 2         # G: expansion rounds; the answer step grades the chosen topics' rest. v60o: never the
                                # kept set's limit -- round 1 grades every candidate the prefilter keeps (grade_kept)
RESERVE_MAX = 12                # G: ungraded rows kept for the answer step (the chosen topics' Sezione / Capo)


def case_first_enabled() -> bool:
    return feature_flags.enabled(CASE_FIRST_FLAG) or single_shot_enabled()


# v60k: single shot on the case-first pipeline -- one message in, one complete parere out (single_shot); round 1 runs
# inside it, never shown. On, it is the narrative case's path whatever INTERACTIVE_FLOW says (the rounds stay behind it).
SINGLE_SHOT_FLAG = "CASE_FIRST_SINGLE_SHOT"
SINGLE_SHOT_ARTICLES = 15       # the article cap of the single-shot answer (the rounds' TOPIC_ARTICLES: 8)
OPEN_POINTS_ENV = "OPEN_POINTS_MAX"
OPEN_POINTS_MAX = 0             # v60q: no «Punti da precisare» block -- every missing fact «da verificare» where it counts
SINGLE_SHOT_TARGET_S = 240      # the PO's target, logged against (single_shot_timings)
SINGLE_SHOT_TOPICS = 5          # v60n: the topics the single-shot answer covers (covered_topics); the rest, one line each
SINGLE_SHOT_TOPICS_MIN = 3      # v60q: at least this many covered where as many exist (covered_topics)


def single_shot_enabled() -> bool:
    return feature_flags.enabled(SINGLE_SHOT_FLAG)


def turns_on() -> bool:
    """The API's gate: a narrative case goes to this module -- in rounds (INTERACTIVE_FLOW) or single shot."""
    return enabled() or single_shot_enabled()


def single_mode(state: Optional[Dict[str, Any]]) -> bool:
    """The state is a single-shot one (single_shot made it): its answer, its follow-ups."""
    return (state or {}).get("mode") == "single_shot"


# --- logging and isolation -------------------------------------------------------------------------------------------

def _log(event: str, data: Any, duration_ms: Optional[float] = None) -> None:
    try:
        from .verbose_logger import vlog
        vlog(f"interactive_{event}", data, duration_ms)
    except Exception:           # logging never fails a turn
        logger.debug("interactive: %s not logged", event, exc_info=True)


def _isolated(step: str, run: Callable[[], Any], default: Any = None) -> Any:
    """run(), or default on any exception -- logged with its full traceback (interactive_step_failed)."""
    try:
        return run()
    except Exception as exc:
        logger.exception("interactive: step %s failed, going on without it", step)
        _log("step_failed", {"step": step, "error": f"{type(exc).__name__}: {exc}"[:500],
                             "traceback": traceback.format_exc()[-4000:]})
        return default


# --- the state, in the session ---------------------------------------------------------------------------------------

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _jsonable(value: Any) -> Any:
    def default(obj):
        if isinstance(obj, (set, frozenset)):
            return sorted(obj, key=str)
        return str(obj)
    return json.loads(json.dumps(value, ensure_ascii=False, default=default))


def _label(ref: Sequence[str]) -> str:
    return f"{ref[0]} {ref[1]}"


def _key(ref: Sequence[str]) -> Tuple[str, str]:
    from .topic_step import _code
    return str(ref[0]), _code(ref[1])


def _parse_label(label: str) -> Tuple[str, str]:
    number, _, code = str(label).partition(" ")
    return number, code


def _compact_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """A pool row as the session keeps it: its article, its section's name, rubrica and text (cut), its document, and
    its simple tags."""
    s, d = row.get("s") or {}, row.get("d") or {}
    out: Dict[str, Any] = {
        "_article": list(row["_article"]) if row.get("_article") else None,
        "s": {k: (str(s.get(k))[:ROW_TEXT_CHARS] if k in ("plain_text", "abstract") else s.get(k))
              for k in ("name", "titolo_articolo", "title", "plain_text", "abstract") if s.get(k) is not None},
        "d": {k: d.get(k) for k in ("id", "name", "type") if d.get(k) is not None}}
    for k, v in row.items():
        if k.startswith("_") and k != "_article":
            try:
                json.dumps(v)
            except (TypeError, ValueError):
                continue
            out[k] = v
    return out


def _restored_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for row in rows or []:
        row = copy.deepcopy(row)
        if row.get("_article"):
            row["_article"] = tuple(row["_article"])
        out.append(row)
    return out


def _restored_analysis(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """The analysis after JSON: the tuples the code compares again tuples."""
    analysis = copy.deepcopy(analysis or {})
    signals = analysis.get("fact_signals") or {}
    for tool in signals.get("tools") or []:
        if isinstance(tool, dict) and tool.get("refs"):
            tool["refs"] = [tuple(r) for r in tool["refs"]]
    return analysis


def active(anchors: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The session's interactive state, unless missing, of another schema or past its TTL."""
    state = (anchors or {}).get(STATE_KEY)
    if not isinstance(state, dict) or state.get("schema") != SCHEMA:
        return None
    try:
        if datetime.fromisoformat(state["expires"]) < _now():
            return None
    except (KeyError, ValueError, TypeError):
        return None
    return state


def touch(state: Dict[str, Any], why: str) -> Dict[str, Any]:
    """The state one version on, its TTL renewed."""
    state["version"] = int(state.get("version") or 0) + 1
    state["updated"] = _now().isoformat()
    state["expires"] = (_now() + timedelta(hours=TTL_HOURS)).isoformat()
    state.setdefault("history", []).append({"version": state["version"], "why": why, "at": state["updated"]})
    state["history"] = state["history"][-30:]
    return state


def store(anchors: Optional[Dict[str, Any]], state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The session's anchors with the state (or without one). A new case's state replacing an earlier one takes the
    version after it, so the versions of a session only grow."""
    anchors = dict(anchors or {})
    if state is None:
        anchors.pop(STATE_KEY, None)
        return anchors
    old = anchors.get(STATE_KEY) if isinstance(anchors.get(STATE_KEY), dict) else {}
    if old and old.get("created") != state.get("created"):
        state["version"] = max(int(state.get("version") or 0), int(old.get("version") or 0) + 1)
    anchors[STATE_KEY] = _jsonable(state)
    return anchors


# --- routing --------------------------------------------------------------------------------------------------------

def routes_here(message: str) -> bool:
    """A case the router reads as narrative (a "direct" question keeps the old path)."""
    from . import query_router
    try:
        return query_router.classify(message or "", tie_break=False, log=False)["route"] == "narrative"
    except Exception as exc:
        logger.warning("interactive: router failed, the old path: %s", exc)
        return False


def is_new_case(message: str) -> bool:
    return len(message or "") >= NEW_CASE_CHARS and routes_here(message)


# --- 1. the analysis ------------------------------------------------------------------------------------------------

def _signals(analysis: Dict[str, Any]) -> Dict[str, Any]:
    return (analysis or {}).get("fact_signals") or {}


def _facts(analysis: Dict[str, Any]) -> Dict[str, Any]:
    return _signals(analysis).get("facts") or {}


def _areas(analysis: Dict[str, Any]) -> List[str]:
    return (analysis.get("frame") or {}).get("areas") or _signals(analysis).get("areas") or []


def party(analysis: Dict[str, Any]) -> str:
    from . import defensive_generation as dg
    name = str((analysis or {}).get("parte_assistita") or "").strip()
    if not name:
        side = str(_facts(analysis).get("client_side") or "").strip()
        name = re.split(r"\s*[(,]", side)[0] if side else ""
    return dg._stance_party({**(analysis or {}), "parte_assistita": name}) if name else "la parte assistita"


def capped_refs(topics: Sequence[Dict[str, Any]], cap: int = None, records: Optional[Dict[str, Any]] = None,
                dropped: Optional[List[Tuple[str, str]]] = None, available: Optional[set] = None) -> List[Tuple[str, str]]:
    """v57 (C: v55 h5 «2, 3, 5» -> 19 extras): the chosen topics' articles, at most TOPIC_ARTICLES across them -- every
    topic's matched article(s) first, in the topics' order, then the topics' other articles taken in turn. v60f (A4: v59
    c3, 1460 c.c. cut by position): with the records of graded topics, by legal weight instead -- each topic's best
    article first, then the rest by (identified offence, grade, element support: issue_records.support), the position
    only last; dropped gets the articles the cap leaves out."""
    cap = TOPIC_ARTICLES if cap is None else cap
    out: List[Tuple[str, str]] = []
    seen: set = set()

    def put(ref):
        if len(out) < cap and _key(ref) not in seen:
            seen.add(_key(ref))
            out.append(ref)
    if records is not None and any(t.get("grades") for t in topics):
        from . import issue_records

        def score(t, i, ref):
            label = issue_records._label(ref)
            rec = records.get(label) or {}
            grade = rec.get("grade") if rec.get("grade") is not None else (t.get("grades") or {}).get(label)
            return (label in (t.get("offences") or []), int(grade or 0), issue_records.support(rec), -i)
        def candidates(t):          # every graded article of the topic, not only its first TOPIC_ARTICLES
            grades = t.get("grades") or {}
            listed = _topic_refs(t, all_=True) if grades else _topic_refs(t)
            out = [r for r in listed if (not grades or issue_records._label(r) in grades)
                   and (available is None or _key(r) in available or r in _topic_refs(t))]
            return out or _topic_refs(t)
        ranked = [sorted(((score(t, i, r), r) for i, r in enumerate(candidates(t))), key=lambda x: x[0], reverse=True)
                  for t in topics]
        for mine in ranked:
            if mine:
                put(mine[0][1])
        for _, ref in sorted((x for mine in ranked for x in mine[1:]), key=lambda x: x[0], reverse=True):
            put(ref)
        if dropped is not None:
            dropped += [r for mine in ranked for _, r in mine if _key(r) not in seen and r not in dropped]
        return out
    for t in topics:
        for a in [t.get("anchor") or ""] + list(t.get("matched") or []):
            if a and _parse_label(a) in _topic_refs(t, all_=True):
                put(_parse_label(a))
    rest = [[r for r in _topic_refs(t) if _key(r) not in seen] for t in topics]
    for i in range(max((len(x) for x in rest), default=0)):
        for x in rest:
            if i < len(x):
                put(x[i])
    if dropped is not None:
        dropped += [r for t in topics for r in _topic_refs(t) if _key(r) not in seen and r not in dropped]
    return out


def _topic_refs(topic: Dict[str, Any], all_: bool = False) -> List[Tuple[str, str]]:
    """A topic's planned articles (its first TOPIC_ARTICLES; all_: every one listed)."""
    arts = topic.get("articles") or []
    cut = min(TOPIC_ARTICLES, int(topic.get("planned") or len(arts)))       # v57: never the listed-only tail
    return [_parse_label(a) for a in (arts if all_ else arts[:cut])]


# --- v60d: the first read's institutions as searches ------------------------------------------------------------------
# Each institution the first read names is searched in the codes of the case's area (AREA_CODES: the frame's areas'
# families; none known: CIVIL_CODES and the criminal ones) by fact_resolver.resolve -- its rubrica (heading index and
# heading full text) and the code's text full text, within the code. What it finds joins the candidates as an ordinary
# one (no tier, no exemption) and is graded outside the prefilter's cap (grading_limits "prefilter_keep").
AREA_CODES = {"penale": ("c.p.", "c.p.p."), "amministrativo": ("c.p.a.",), "civile": ("c.c.", "c.p.c.")}
FIRST_READ_MAX = 8          # institutions searched, the main line's first
_RESOLVER_TAGS = ("_fact_institution", "_heading_match", "_fact_related", "_tier", "_tier_why")


def area_codes(analysis: Dict[str, Any]) -> List[str]:
    """The codes of the case's areas, in the areas' order (the first area's first)."""
    from . import fact_signals
    out: List[str] = []
    for area in _areas(analysis):
        out += list(AREA_CODES.get(fact_signals.family(area), ()))
    return list(dict.fromkeys(out)) or ["c.c.", "c.p.", "c.p.c.", "c.p.p."]


# v60i (3): the c.p.a. only in an administrative frame -- with c.p.a. units in the graph (article_links) its Titoli and
# Capi reach the index channel and the topic grouping; a case whose frame (or analysis) has no administrative area gets
# no c.p.a. article, unless the user's text names it. Unknown areas: not administrative.
# v60q: the code-area gate, general (area_gate) -- an article of a code outside the frame's area (CODE_FAMILY: c.c.
# outside civile, c.p. and c.p.p. outside penale, c.p.a. outside amministrativo) is a candidate only when a first-read
# institution found it (its row's _first_read), a fired rule or table names it (its topic's anchor or own articles:
# topic_step's "matched") or the user's text names it; otherwise it is one of the «altri profili», never graded nor a
# topic. Each admission logged with its reason. The frame's areas unknown: no gate (logged).
CPA_CODE = "c.p.a."
CODE_FAMILY = {"c.c.": "civile", "c.p.": "penale", "c.p.p.": "penale", "c.p.a.": "amministrativo"}
AREA_OUT_WHY = "fuori dall'area del caso"


def administrative_frame(analysis: Dict[str, Any]) -> bool:
    return "amministrativo" in frame_families(analysis)


def frame_families(analysis: Dict[str, Any]) -> List[str]:
    """The families (civile, penale, amministrativo) of the frame's areas and the analysis' area."""
    from . import fact_signals
    areas = list(_areas(analysis or {})) + [str((analysis or {}).get("area") or "")]
    return sorted({fact_signals.family(a) for a in areas} - {""})


_QUERELA_RE = re.compile(r"\bquerel\w*", re.IGNORECASE)


def frame_party_label(analysis: Dict[str, Any], case: str = "", role: str = "") -> str:
    """v60r (5): the label of the party the action is directed at, from the frame's first area -- penale: «querelato»
    when the case or its facts speak of a querela, else «indagato»; civile: «convenuto»; any other: "". The party's
    extracted role holding the same word gives its ending («convenuta»): never guessed from the name."""
    from . import fact_signals
    areas = list(_areas(analysis or {})) + [str((analysis or {}).get("area") or "")]
    family = next((fact_signals.family(a) for a in areas if fact_signals.family(a)), "")
    if family == "penale":
        said = f"{case or ''}\n{fact_signals._texts(_facts(analysis or {}))}"
        label = C.FRAME_PARTY_LABELS.get("penale_querela" if _QUERELA_RE.search(said) else "penale", "")
    else:
        label = C.FRAME_PARTY_LABELS.get(family, "")
    m = re.search(rf"\b{re.escape(label[:-1])}[oaie]\b", str(role or ""), re.IGNORECASE) if label else None
    return m.group(0).lower() if m else label


def rule_named(rule_topics: Sequence[Dict[str, Any]]) -> Dict[Tuple[str, str], str]:
    """{article key: the rule or table that names it} -- each fired rule's or table's topic's own articles (anchor and
    matched)."""
    out: Dict[Tuple[str, str], str] = {}
    for t in rule_topics or []:
        for label in [t.get("anchor") or ""] + list(t.get("matched") or []):
            if label:
                out.setdefault(_key(_parse_label(label)), str(t.get("source") or t.get("title") or "regola"))
    return out


def area_gate(rows: Sequence[Dict[str, Any]], analysis: Dict[str, Any], case: str,
              named_by_rules: Optional[Dict[Tuple[str, str], str]] = None, log: Optional[Dict[str, Any]] = None
              ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """(the rows admitted, the rows outside the frame's area for the «altri profili»). log: {"families", "admitted":
    [{article, why}] (an outside article and its reason), "altri_profili": [labels], "off"} filled in."""
    from . import defensive_generation as dg
    log = log if log is not None else {}
    families = frame_families(analysis)
    log["families"] = families
    if not families:
        log["off"] = "areas unknown"
        return list(rows or []), []
    named = {_key(r) for r in dg._refs_in(case or "") if len(r) == 2}
    rules = named_by_rules or {}
    kept, outside = [], []
    for row in rows or []:
        ref = row.get("_article")
        key = _key(ref) if ref and len(ref) == 2 else None
        family = CODE_FAMILY.get(key[1]) if key else None
        if not family or family in families:
            kept.append(row)
            continue
        label = _label(key)
        why = ("nominato dall'utente" if key in named else
               f"istituto della prima lettura «{row['_first_read']}»" if row.get("_first_read") else
               f"regola «{rules[key]}»" if key in rules else
               f"regola «{row['_rule_anchor']}»" if row.get("_rule_anchor") else "")
        if why:
            kept.append(row)
            if label not in {x["article"] for x in log.get("admitted") or []}:
                log.setdefault("admitted", []).append({"article": label, "why": why})
        else:
            outside.append(row)
            if label not in (log.get("altri_profili") or []):
                log.setdefault("altri_profili", []).append(label)
    return kept, outside


ALTERNATIVES_MAX = 3        # v60g (7): the first read's alternative qualifications searched, beyond FIRST_READ_MAX


def read_institutions(read: Dict[str, Any], top: int = FIRST_READ_MAX) -> List[Dict[str, str]]:
    """The first read's institutions, each once: the main line's, then the subordinate's, then the other side's; each
    {"name": its own wording, "rubricated": its name as the code rubricates it (v60e; "" when the read gave none),
    "claim": the hypothesis sentence}. v60g (7): then the alternative qualifications of the same facts (their «perché»
    as the claim, "alternative": True), ALTERNATIVES_MAX of them beyond top."""
    alternatives = []
    for a in (read or {}).get("alternatives") or []:
        if a.get("name"):
            alternatives.append({"name": " ".join(str(a["name"]).split()),
                                 "rubricated": " ".join(str(a.get("rubricated") or "").split()),
                                 "claim": str(a.get("why") or ""), "alternative": True})
    lines = _line_institutions(read, top)
    have = {_low(x["name"]) for x in lines}
    return lines + [a for a in alternatives if _low(a["name"]) not in have][:ALTERNATIVES_MAX]


def _line_institutions(read: Dict[str, Any], top: int) -> List[Dict[str, str]]:
    rank = {"principale": 0, "subordinata": 1, "controparte": 2}
    hyps = sorted(read.get("hypotheses") or [], key=lambda h: rank.get(h.get("line"), 3))
    seen: Dict[str, Dict[str, str]] = {}
    for h in hyps:
        rubricated = list(h.get("rubricated") or [])
        for i, inst in enumerate(h.get("institutions") or []):
            key = _low(inst)
            if key and key not in seen:
                seen[key] = {"name": " ".join(str(inst).split()),
                             "rubricated": " ".join(str(rubricated[i]).split()) if i < len(rubricated) else "",
                             "claim": str(h.get("claim") or "")}
    return list(seen.values())[:top]


_NOUN_STOP = set("""della dello delle degli dalla dalle nella nelle sulla sulle alla alle questo questa quello quella
essere avere sono stato stata parte cliente controparte caso fatto fatti quando perché poiché anche ancora sempre
proprio propria dovuto dovuta potrebbe possono deve devono rispetto contro senza""".split())


def claim_nouns(claim: str, top: int = 4) -> str:
    """v60e: the content nouns of a hypothesis sentence (no verb forms, no function words), for a rubrica search."""
    words = [w for w in re.findall(r"[a-zà-ü]{5,}", _low(claim)) if w not in _NOUN_STOP
             and not re.search(r"(?:are|ere|ire|ato|ata|ati|ate|ito|ita|iti|ando|endo|ebbe|ono)$", w)]
    return " ".join(list(dict.fromkeys(words))[:top])


def first_read_search(driver, database: str, documents: List[Dict[str, Any]], institutions: Sequence[Any],
                      codes: Sequence[str], session=None, resolve=None, text_search=None, index_search=None
                      ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """(rows, log): each institution searched in each code (fact_resolver.resolve; jobs on their own graph sessions,
    serially on session without a driver) -- v60e: its rubricated name first, then its own wording, then the nouns of
    its hypothesis sentence, the first that finds rows -- each row tagged _first_read (the institution) and
    _first_read_code, the resolver's tier tags dropped. log: {"institutions", "codes", "hits": {institution: [labels]},
    "tried": {institution: [[query, how, code, n]]}}. v60h (4): the rubrica match missing in a code, (2) the rubricated
    name and the hypothesis' nouns against the code's article texts (fact_resolver.text_channel), (3) both missing, the
    code's institutions index (fact_resolver.index_channel); each row tagged _first_read_channel («rubrica», «testo»,
    «indice»); log "channels": {institution: {label: channel}}. v60n: a substantive institution's hits in a
    procedural code (c.p.p., c.p.c.) are dropped -- its area's substantive code answers it (procedural_institutions);
    log "procedural" (the names) and "area_code" ({institution: {procedural, why, dropped}})."""
    from . import defensive_generation as dg, fact_resolver, topic_groups
    resolve = resolve or fact_resolver.resolve
    text_search = text_search or fact_resolver.text_channel
    index_search = index_search or fact_resolver.index_channel
    searches = [{"name": x, "rubricated": "", "claim": ""} if isinstance(x, str) else dict(x) for x in institutions]
    alternatives = [s["name"] for s in searches if s.get("alternative")]

    def attempts(item):
        out = [(item.get("rubricated"), "rubricato"), (item["name"], "istituto"), (claim_nouns(item.get("claim")), "tesi")]
        seen, kept = set(), []
        for query, how in out:
            if query and _low(query) not in seen:
                seen.add(_low(query))
                kept.append((query, how))
        return kept

    def job_for(item, code):
        def job(own):
            tried = []
            for query, how in attempts(item):
                got = resolve(own, documents, {"nome": query, "codice": code})
                found = (got[0] if isinstance(got, tuple) else got) or []
                tried.append([query, how, code, len(found)])
                if found:
                    return [{**r, "_first_read_channel": "rubrica"} for r in found], tried
            query = item.get("rubricated") or item["name"]          # v60h (4): the second and third channels
            nouns = claim_nouns(item.get("claim"))
            for channel, search in (("testo", text_search), ("indice", index_search)):
                try:
                    got = search(own, documents, query, code, nouns)
                except Exception as exc:        # noqa: BLE001 -- a channel failing is a channel missing
                    logger.info("first read search: channel %s failed: %s", channel, exc)
                    got = []
                found = (got[0] if isinstance(got, tuple) else got) or []
                tried.append([query, channel, code, len(found)])
                if found:
                    return [{**r, "_first_read_channel": channel} for r in found], tried
            return [], tried
        return job
    pairs = [(item, code) for item in searches for code in codes]
    jobs = [job_for(item, code) for item, code in pairs]
    results = dg.graph_parallel(driver, database, jobs) if driver is not None else []
    if not results and session is not None:
        results = []
        for job in jobs:
            try:
                results.append(job(session))
            except Exception as exc:        # noqa: BLE001 -- one search failing is no search
                results.append(exc)
    rows: List[Dict[str, Any]] = []
    hits: Dict[str, List[str]] = {item["name"]: [] for item in searches}
    tried: Dict[str, List[list]] = {item["name"]: [] for item in searches}
    channels: Dict[str, Dict[str, str]] = {item["name"]: {} for item in searches}
    seen = set()
    procedural, area_log = procedural_institutions(searches, pairs, results)          # v60n: area-code priority
    for (item, code), got in zip(pairs, results):
        inst = item["name"]
        if isinstance(got, Exception) or not got:
            continue
        found, attempts_made = got
        tried[inst] += attempts_made
        if code in topic_groups.PROCEDURAL_CODES and inst not in procedural:
            if found:
                area_log.setdefault(inst, {"procedural": False})["dropped"] = [
                    _label(r["_article"]) for r in found if r.get("_article")]
            continue
        for row in found or []:
            if not row.get("_article"):
                continue
            row = {k: v for k, v in row.items() if k not in _RESOLVER_TAGS}
            row["_first_read"], row["_first_read_code"] = inst, code
            row.setdefault("_first_read_channel", "rubrica")
            key = _key(row["_article"])
            if _label(key) not in hits[inst]:
                hits[inst].append(_label(key))
                channels[inst][_label(key)] = row["_first_read_channel"]
            if key not in seen:
                seen.add(key)
                rows.append(row)
    for item in searches:               # the anchoring looks the hits up by either name
        if item.get("rubricated"):
            hits.setdefault(item["rubricated"], hits[item["name"]])
            tried.setdefault(item["rubricated"], tried[item["name"]])
    return rows, {"institutions": [s["name"] for s in searches], "alternatives": alternatives,
                  "rubricated": {s["name"]: s.get("rubricated") for s in searches if s.get("rubricated")},
                  "codes": list(codes), "hits": hits, "tried": tried, "channels": channels,
                  "procedural": sorted(procedural | {s.get("rubricated") for s in searches
                                                     if s["name"] in procedural and s.get("rubricated")}),
                  "area_code": area_log}


def procedural_institutions(searches: Sequence[Dict[str, Any]], pairs: Sequence[tuple], results: Sequence[Any]
                            ) -> Tuple[set, Dict[str, Dict[str, Any]]]:
    """v60n: (the names of the procedural institutions, the log): an institution is procedural when its name or its
    rubricated name says so (topic_groups.procedural), or when its rubrica hits (channel «rubrica») are all in a
    procedural code -- the store rubricates it there only."""
    from . import topic_groups
    rubrica_codes: Dict[str, set] = {}
    for (item, code), got in zip(pairs, results):
        if isinstance(got, Exception) or not got:
            continue
        if any(r.get("_first_read_channel", "rubrica") == "rubrica" for r in got[0] or []):
            rubrica_codes.setdefault(item["name"], set()).add(code)
    out, log = set(), {}
    for item in searches:
        inst = item["name"]
        if topic_groups.procedural(inst) or topic_groups.procedural(item.get("rubricated")):
            out.add(inst)
            log[inst] = {"procedural": True, "why": "nome"}
        elif rubrica_codes.get(inst) and rubrica_codes[inst] <= set(topic_groups.PROCEDURAL_CODES):
            out.add(inst)
            log[inst] = {"procedural": True, "why": f"rubrica solo in {', '.join(sorted(rubrica_codes[inst]))}"}
    return out, log


CUTTING_SHARE = 0.8         # v60d: of the prefilter's N kept, more than this share graded 2+: the prefilter is cutting


def grading_keep(rows: Sequence[Dict[str, Any]], case: str, anchors: Sequence[str], first_read: Sequence[str]
                 ) -> List[str]:
    """v60o: the labels round 1 grades first, past the prefilter -- the candidates the user's text names, then the
    rules' and tables' anchors, then the first read's hits (anchors, first_read: the grader's labels), each once."""
    from . import defensive_generation as dg
    named = {_key(r) for r in dg._refs_in(case or "") if len(r) == 2}
    labels = [f"{r['_article'][0]} {r['_article'][1]}" for r in rows or [] if r.get("_article") and _key(r["_article"]) in named]
    return list(dict.fromkeys(labels + list(anchors or []) + list(first_read or [])))


def grading_invariant(limits: Dict[str, Any]) -> Dict[str, Any]:
    """v60o: the invariant graded == kept of one run, as the check wrote it back (dg._grade_kept): {kept, graded, ok,
    missing: the kept labels with no grade, first: the labels graded first}. No candidate: nothing to grade, ok; the
    check failing before it graded: not ok, said why."""
    inv = limits.get("invariant")
    if inv:
        return {k: inv.get(k) for k in ("kept", "graded", "ok", "missing", "first", "no_text")}
    if not limits.get("candidates_n"):
        return {"kept": 0, "graded": 0, "ok": True, "missing": [], "first": []}
    return {"kept": None, "graded": limits.get("graded_n"), "ok": False, "missing": [],
            "first": [], "why": "the kept set was not graded (the check failed before grading)"}


def grading_breadth(rows: Sequence[Dict[str, Any]], limits: Dict[str, Any]) -> Dict[str, Any]:
    """v60d: candidates / kept by the prefilter / graded 2 or more, for one run. cutting: the prefilter's kept articles
    graded 2+ are more than CUTTING_SHARE of its N (40 of 50) -- relevant articles likely left below the cut."""
    grades: Dict[str, int] = {}
    for r in rows:
        if r.get("_article") and isinstance(r.get("_grade"), int):
            label = f"{r['_article'][0]} {r['_article'][1]}"
            grades[label] = max(grades.get(label, 0), r["_grade"])
    kept = list(limits.get("prefilter_kept") or [])
    n = int(limits.get("prefilter") or 0)
    kept_2 = sum(1 for label in kept if grades.get(label, 0) >= 2)
    return {"candidates": limits.get("candidates_n"), "prefilter_n": n, "prefilter_kept": len(kept),
            "outside_cap": len(limits.get("prefilter_keep") or []), "kept": limits.get("kept_n"),     # v60o
            "graded": limits.get("graded_n"),
            "graded_2plus": sum(1 for g in grades.values() if g >= 2), "kept_graded_2plus": kept_2,
            "cutting": bool(n) and kept_2 > CUTTING_SHARE * n}


def _retrieve(text: str, request: str, lang: str, analysis: Dict[str, Any], pending=None) -> Dict[str, Any]:
    """Retrieval as _case_analysis_pipeline runs it, up to its pool texts, then the topics and their articles' texts:
    {"article_rows", "regime_rows", "topics", "topic_rows"}."""
    import neo4j
    from . import (article_text, comma_material, defensive_generation as dg, pool_gates, query_router, source_tiers,
                   topic_step)
    from .doc_lookup import _fetch_doc_names
    from .main import NEO4J_DATABASE, driver

    from . import stage_meter
    stage_meter.mark("route")
    case = dg._case_text(text, request)
    route = query_router.route_for_pipeline(case, analysis)
    analysis["route"] = route["route"]
    hybrid = source_tiers.hybrid_enabled(route["route"])
    v2 = feature_flags.enabled(dg.CASE_ANALYSIS_FLAG)
    fusion = v2 and feature_flags.enabled(dg._CASE_POOL_FUSION_FLAG)
    store_ = article_text.ArticleStore()
    out: Dict[str, Any] = {"article_rows": [], "regime_rows": [], "topics": [], "others": [], "topic_rows": []}
    stage_meter.mark("doc_names")
    graph = stage_meter.counted(driver)         # v60: every graph query of round 1 counted per stage
    documents = _fetch_doc_names(graph, NEO4J_DATABASE)
    with graph.session(database=NEO4J_DATABASE, default_access_mode=neo4j.READ_ACCESS) as session:
        select = dg._select_articles_fusion if fusion else (dg._select_articles_v2 if v2 else dg._select_articles)
        stage_meter.mark("select")
        picked = select(session, graph, NEO4J_DATABASE, text, request, analysis, store=store_)
        stage_meter.mark("method_candidates")
        picked = dg._add_method_candidates(session, graph, NEO4J_DATABASE, analysis, picked, store=store_)
        stage_meter.mark("regime_candidates")
        picked = dg._add_regime_candidates(session, graph, NEO4J_DATABASE, analysis, picked, store=store_)
        stage_meter.mark("regime_rows")
        regime_rows = dg._regime_rows(session, graph, NEO4J_DATABASE, analysis, store=store_)
        stage_meter.mark("tiers_gates")
        if picked.get("tiers"):
            from . import evidence_tiers
            for row in regime_rows:
                row["_regime"] = True
            evidence_tiers.annotate(picked["article_rows"] + regime_rows,
                                    set(picked.get("named") or []) | set(picked.get("keep") or []),
                                    picked.get("fact_resolved") or [], picked.get("heading_exact") or [])
        article_rows = picked["article_rows"]
        if hybrid:
            article_rows = source_tiers.keep_types(session, article_rows, source_tiers.AUTHORITY_TYPES, "authority")
            regime_rows = source_tiers.keep_types(session, regime_rows, source_tiers.AUTHORITY_TYPES, "authority_regimes")
        if pool_gates.enabled():
            article_rows = _isolated("pool_gates", lambda: pool_gates.apply(session, article_rows, analysis, case)[0],
                                     article_rows)
        case_first = case_first_enabled()
        must: List[str] = []
        search_log: Dict[str, Any] = {}
        if case_first and pending is not None:      # v60d: the first read's institutions searched, before the grading
            stage_meter.mark("first_read_search")
            read = pending.result()
            institutions = read_institutions(read)
            if institutions:
                t_search = time.time()
                found_rows, search_log = _isolated("first_read_search", lambda: first_read_search(
                    graph, NEO4J_DATABASE, documents, institutions, area_codes(analysis), session), ([], {})) or ([], {})
                have = {_key(r["_article"]): r for r in article_rows if r.get("_article")}
                added = []
                for row in found_rows:
                    key = _key(row["_article"])
                    if key in have:         # already a candidate: tagged, and graded outside the cap too
                        have[key].setdefault("_first_read", row["_first_read"])
                        have[key].setdefault("_first_read_code", row["_first_read_code"])
                        have[key].setdefault("_first_read_channel", row.get("_first_read_channel"))
                    else:
                        added.append(row)
                    must.append(f"{row['_article'][0]} {row['_article'][1]}")
                    for r in article_rows:
                        if r.get("_article") and _key(r["_article"]) == key:
                            must.append(f"{r['_article'][0]} {r['_article'][1]}")
                if added:
                    _isolated("first_read_texts", lambda: article_text.fill_texts(session, added, store_))
                article_rows = list(article_rows) + added
                search_log = {**search_log, "new": len(added), "s": round(time.time() - t_search, 1)}
        links = topic_step.GraphLinks(session, documents)
        anchor_keep: List[str] = []
        if case_first:          # v60n: the fired rules' and tables' anchors -- candidates, graded like any other
            stage_meter.mark("rule_anchors")
            pre = _isolated("rule_anchor_topics", lambda: topic_step.build(analysis, article_rows + regime_rows, links,
                                                                           case, others=[]), []) or []
            added, anchor_log = _isolated("rule_anchors", lambda: rule_anchor_rows(
                session, documents, pre, article_rows, regime_rows, store_), ([], {})) or ([], {})
            article_rows = list(article_rows) + list(added)
            anchor_keep = list(anchor_log.get("keep") or [])
            out["rule_anchors"] = anchor_log
            _log("rule_anchors", anchor_log)
        scope: Dict[str, Any] = {}
        gate_log: Dict[str, Any] = {}
        outside: List[Dict[str, Any]] = []
        if case_first:          # v60q: a code outside the frame's area only when the first read, a rule or the user names
            area_log: Dict[str, Any] = {}                                # the article (v60i: the c.p.a. alone, dropped)
            by_rules = rule_named(pre)
            gated = _isolated("area_gate", lambda: (area_gate(article_rows, analysis, case, by_rules, area_log),
                                                    area_gate(regime_rows, analysis, case, by_rules, area_log)), None)
            if gated is not None:
                (article_rows, out_a), (regime_rows, out_r) = gated
                outside = out_a + out_r
                gone = {_key(r["_article"]) for r in outside}
                anchor_keep = [a for a in anchor_keep if _key(_parse_label(a)) not in gone]
                must = [a for a in must if _key(_parse_label(a)) not in gone]
            out["area_gate"] = area_log
            _log("area_gate", area_log)
        if case_first:          # v60q: Titolo III c.c., a Capo only for the case's own contract type
            from . import topic_groups
            scope = _isolated("capo_scope", lambda: topic_groups.contract_scope(analysis, case), {}) or {}
            article_rows = _isolated("capo_gate", lambda: topic_groups.gate_rows(article_rows, scope, gate_log),
                                     article_rows)
            gone = {x["article"] for x in gate_log.get("rows") or []}
            anchor_keep = [a for a in anchor_keep if a not in gone]
            must = [a for a in must if a not in gone]
        stage_meter.mark("grading")
        if case_first:          # v60 (G): round 1's graded check -- the embedding prefilter's top N
            # v60o: every kept candidate graded (grade_kept) -- the user-named articles, the rules' and tables' anchors
            # and the first read's hits first, past the prefilter and in a call of their own; the invariant graded == kept
            limits = {"graded": True, "grade_kept": True, "grade_all": True, "rounds": ROUND1_GRADE_ROUNDS,
                      "prefilter": dg.grade_prefilter_n(),
                      "prefilter_keep": grading_keep(article_rows, case, anchor_keep, must)}
            t_grade = time.time()
            with dg.grading_limits(limits):
                graded = _isolated("grading", lambda: dg._filter_applicable_articles(text, request, article_rows, analysis),
                                   None)
            if graded is not None:
                kept = {tuple(g["_article"]) for g in graded if g.get("_article")}
                have = {tuple(g["_article"]) for g in graded if g.get("_article") and isinstance(g.get("_grade"), int)}
                ungraded = [r for r in graded if r.get("_article") and tuple(r["_article"]) in kept - have]
                article_rows = graded
            else:
                ungraded = []
            out["grading"] = {"s": round(time.time() - t_grade, 1), "rounds": limits.get("rounds_run"),
                              "graded": limits.get("graded_n"), "prefilter_ms": limits.get("prefilter_ms"),
                              "prefilter_dropped": limits.get("prefilter_dropped") or []}
            out["grading"]["breadth"] = grading_breadth(article_rows, limits)
            out["grading"]["invariant"] = grading_invariant(limits)                     # v60o: graded == kept
            out["grading"]["calls"] = limits.get("calls") or {}         # v60p: calls, wall time, the longest call
            _log("grading_breadth", out["grading"]["breadth"])
            _log("grading_invariant", out["grading"]["invariant"])
            _log("grading_calls", {**out["grading"]["calls"], "grading_s": out["grading"]["s"]})
            if search_log:
                grades = {f"{r['_article'][0]} {r['_article'][1]}": r.get("_grade") for r in article_rows
                          if r.get("_article") and r.get("_first_read")}
                search_log["graded"] = {inst: {label: grades.get(label) for label in labels}
                                        for inst, labels in (search_log.get("hits") or {}).items()}
                search_log["graded_2plus"] = sum(1 for g in grades.values() if isinstance(g, int) and g >= 2)
                out["first_read_search"] = search_log
                _log("first_read_search", search_log)
        if case_first and graded is not None:   # v60g (1), v60m: an article whose distinctive element the case
            from . import issue_records        # contradicts or the facts exclude: grade 1 (never one the case is silent on)
            stage_meter.mark("distinctive_link")
            named = [issue_records._label(r) for r in dg._refs_in(case)]
            out["distinctive"] = _isolated("distinctive_link", lambda: issue_records.demote_contradicted(
                article_rows, case, analysis, keep=named), {}) or {}
            out["grading"]["distinctive"] = dict(out["distinctive"])     # v60m: each demotion with why, for the sim
            _log("distinctive_link", out["distinctive"], out["distinctive"].get("ms"))
        out.update(article_rows=article_rows, regime_rows=regime_rows)
        if case_first:          # v60e: a rubrica not its text's, confirmed by a neighbour, is never a topic's
            stage_meter.mark("misaligned")
            _isolated("misaligned", lambda: article_text.confirm_misaligned(
                session, [r for r in article_rows if isinstance(r.get("_grade"), int) and r["_grade"] >= 1]))
        stage_meter.mark("topic_step")
        others: List[Dict[str, Any]] = []
        topics = _isolated("topic_step", lambda: topic_step.build(analysis, article_rows + regime_rows, links, case,
                                                                   others=others), [])
        out["topics"], out["others"] = topics, others
        if case_first:          # v60 (B): the topics from the graded rows; the rules' and tables' attach to them
            from . import topic_groups
            stage_meter.mark("topic_groups")
            t_topics = time.time()
            unit_of = _isolated("units", lambda: topic_groups.units(session, article_rows + ungraded + outside),
                                {}) or {}
            built = _isolated("topic_groups", lambda: topic_groups.build(article_rows, unit_of, (), topics), None)
            out["unit_of"] = unit_of
            out["units"] = {topic_groups._label(topic_groups._ref(r)): sorted(set(unit_of[topic_groups.unit_key(r)]
                                                                                  .values()))
                            for r in article_rows + ungraded + outside
                            if topic_groups._ref(r) and unit_of.get(topic_groups.unit_key(r))}   # v60r (2)
            if built is not None:
                out["topics"], out["others"] = built["topics"], built["others"]
                out["lost_rules"] = built["lost_rules"]
                out["rubriche"] = topic_groups.rubriche_of(article_rows)
                units = {t.get("unit") for t in built["topics"] if t.get("unit")}
                reserve = [r for r in ungraded
                           if (unit_of.get(topic_groups.unit_key(r)) or {}).get("SEZIONE") in units
                           or (unit_of.get(topic_groups.unit_key(r)) or {}).get("CAPO") in units]
                out["reserve"] = [{**r, "_unit": next(u for u in (unit_of.get(topic_groups.unit_key(r)) or {}).values()
                                                       if u in units)} for r in reserve[:RESERVE_MAX]]
            if outside:         # v60q: the code-area gate's articles, «altri profili» (never graded, never a topic)
                out["others"] = list(out["others"]) + (_isolated("area_profiles", lambda: topic_groups.outside_profiles(
                    outside, unit_of, AREA_OUT_WHY), []) or [])
            out["topics"] = _isolated("capo_gate_topics", lambda: topic_groups.gate_topics(    # v60q: the anchors the
                out["topics"], scope, gate_log), out["topics"])                                 # rules attached after
            out["others"] = _isolated("capo_gate_others", lambda: topic_groups.gate_topics(    # the rows' gate
                out["others"], scope, gate_log, "others"), out["others"])
            out["capo_gate"] = {k: v for k, v in scope.items()} | gate_log
            _log("capo_gate", out["capo_gate"])
            out["grading"]["topics_s"] = round(time.time() - t_topics, 1)
            topics = out["topics"]
            stage_meter.mark("anchor_vectors")      # v60: the graded rows' rubrica vectors, the units' titles
            out["vectors"] = _isolated("anchor_vectors", lambda: topic_groups.topic_vectors(session, out["topics"],
                                                                                           article_rows), None)
        stage_meter.mark("topic_texts")
        pool = {}
        for row in article_rows + regime_rows:
            if row.get("_article"):
                pool.setdefault(_key(row["_article"]), row)
        wanted = list(dict.fromkeys(ref for t in topics for ref in _topic_refs(t)))
        missing = [ref for ref in wanted if ref not in pool]
        by_code = {topic_step._code(d.get("name")): d.get("name") for d in documents if d.get("name")}
        fetch = [(n, c if c in dg._CODE_PREFIXES else by_code.get(c, c)) for n, c in missing]
        fetched = _isolated("topic_texts", lambda: dg._fetch_any(session, documents, fetch, store_), []) if fetch else []
        for row in fetched:
            if row.get("_article"):
                pool.setdefault(_key(row["_article"]), row)
        rows = [pool[ref] for ref in wanted if ref in pool]
        reserve_rows = list(out.get("reserve") or [])
        _isolated("fill_texts", lambda: article_text.fill_texts(session, rows + regime_rows + reserve_rows, store_))
        if case_first:          # v60e: the rows read now checked too; the confirmed ones logged per code
            _isolated("misaligned", lambda: article_text.confirm_misaligned(session, rows))
            _isolated("misaligned_log", lambda: log_misaligned(list(article_rows) + rows))
        out["topics"] = prune_abrogated(out["topics"], rows)       # v60c: their texts read, the abrogated ones out
        from . import procedural_facts           # v60 (C): the procedural facts of the topics' criminal / procedural articles
        stage_meter.mark("procedural_facts")
        _isolated("procedural_facts", lambda: procedural_facts.attach(session, rows))
        stage_meter.mark("commi")
        if comma_material.enabled():
            _isolated("commi", lambda: comma_material.attach(session, rows, "strategy", store=store_))
        out["topic_rows"] = rows
    return out


def rule_anchor_rows(session, documents: List[Dict[str, Any]], rule_topics: Sequence[Dict[str, Any]],
                     article_rows: Sequence[Dict[str, Any]], regime_rows: Sequence[Dict[str, Any]], store=None
                     ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """v60n: the fired rules' and tables' anchor articles (topic_step's topics: each its anchor) as candidates, graded
    like any other (the caller keeps them past the prefilter) -- they make a topic only when graded 2 or more
    (topic_groups.build). One already a candidate is tagged _rule_anchor (its topic's source); one among the tools' rows
    is copied (their tags left); the others read. (the rows added, the log {"anchors", "already", "from_tools", "read",
    "unread", "keep": the grader's labels of every anchor now a candidate})."""
    from . import article_text, defensive_generation as dg, topic_step
    have = {_key(r["_article"]): r for r in article_rows or [] if r.get("_article")}
    tools = {_key(r["_article"]): r for r in regime_rows or [] if r.get("_article")}
    anchors: Dict[Tuple[str, str], str] = {}
    for rt in rule_topics or []:
        if rt.get("anchor"):
            anchors.setdefault(_key(_parse_label(rt["anchor"])), str(rt.get("source") or rt.get("title") or ""))
    log: Dict[str, Any] = {"anchors": [_label(k) for k in anchors], "already": [], "from_tools": [], "read": [],
                           "unread": []}
    added, missing = [], []
    for key, source in anchors.items():
        if key in have:
            have[key].setdefault("_rule_anchor", source)
            log["already"].append(_label(key))
        elif key in tools:
            row = {k: v for k, v in tools[key].items() if not k.startswith("_") or k == "_article"}
            added.append({**row, "_rule_anchor": source})
            log["from_tools"].append(_label(key))
        else:
            missing.append((key, source))
    if missing:
        by_code = {topic_step._code(d.get("name")): d.get("name") for d in documents if d.get("name")}
        fetch = [(n, c if c in dg._CODE_PREFIXES else by_code.get(c, c)) for (n, c), _ in missing]
        got: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for row in dg._fetch_any(session, documents, fetch, store) or []:
            if row.get("_article"):
                got.setdefault(_key(row["_article"]), row)
        for key, source in missing:
            if key in got:
                added.append({**got[key], "_rule_anchor": source})
                log["read"].append(_label(key))
            else:
                log["unread"].append(_label(key))
    if added:
        article_text.fill_texts(session, added, store)
    log["keep"] = [f"{r['_article'][0]} {r['_article'][1]}" for r in added] + \
        [f"{have[_key(_parse_label(label))]['_article'][0]} {have[_key(_parse_label(label))]['_article'][1]}"
         for label in log["already"]]
    return added, log


def attach_side(topics: List[Dict[str, Any]], hypotheses: Sequence[Dict[str, Any]], vectors: Optional[Dict[str, Any]],
                rubriche: Optional[Dict[str, str]], search: Optional[Dict[str, Any]] = None, codes: Sequence[str] = ()
                ) -> Tuple[List[Dict[str, Any]], str]:
    """(unanchored, method): the topics' hypotheses replaced by these (v60c: the rubriche by stems, then the titles by
    similarity; the embedding failing, by words). v60e: an unanchored one logs what was searched for it -- its
    institutions, their rubricated names and the first-read search's hits for them. v60n: a procedural code's topic
    anchors only a procedural institution, the area's substantive code first (codes; search["procedural"])."""
    from . import topic_groups
    for t in topics:
        t["hypotheses"] = []
    hypotheses = list(hypotheses or [])
    unanchored: List[Dict[str, Any]] = []
    method = "none"
    procedural = list((search or {}).get("procedural") or [])
    if vectors is not None and hypotheses:
        matched = _isolated("anchor_hypotheses", lambda: topic_groups.anchor_by_similarity(
            topics, vectors, hypotheses, rubriche=rubriche or {}, codes=codes, procedural_names=procedural), None)
        if matched is not None:
            unanchored, entries = matched
            method = "rubrica/titolo"
            hits = (search or {}).get("hits") or {}
            tried = (search or {}).get("tried") or {}
            for entry, h in zip(entries, hypotheses):
                if not entry.get("attached"):
                    names = list(h.get("rubricated") or []) + list(h.get("institutions") or [])
                    entry["searched"] = {"names": names, "tried": {n: tried.get(n) for n in names if n in tried},
                                         "hits": {n: hits.get(n) for n in names if n in hits}}
                entry["party"] = h.get("party")
                _log("hypothesis_anchor", entry)
    if method == "none" and rubriche is not None:      # the embedding failed: by words, as before
        unanchored = topic_groups.attach_hypotheses(topics, hypotheses, rubriche, procedural)
        method = "words"
    elif method == "none":
        unanchored = list(hypotheses)
    return unanchored, method


def _pool_rubriche_labels(state: Dict[str, Any]) -> Dict[str, str]:
    """{"N code": rubrica} of the session's pool and rows, for re-anchoring."""
    from . import topic_groups
    out = {f"{p['_article'][0]} {topic_groups.topic_step._code(p['_article'][1])}": p.get("rubrica") or ""
           for p in state.get("pool") or [] if p.get("_article")}
    out.update(topic_groups.rubriche_of(_restored_rows(state.get("rows"))))
    return out


def reside(state: Dict[str, Any], client: str = "", renumber: Sequence[List[int]] = (), reorder: bool = True
           ) -> Dict[str, Any]:
    """v60e: the client settled (stated, answered, or the one an answer assumes: client), its side drives the case-first
    topics -- its own lines of the first read attached (case_read.side), re-read once for it (bounded, READ_SECONDS)
    when the read has none of its lines, then the topics re-ordered (topic_groups.order), so the immediate answer
    (default_choice) is its. renumber: lists of topic numbers (1-based) remapped in place to the new order, as is
    state["chosen"]. Logged; {} when nothing to do."""
    from . import case_read, topic_groups
    topics = state.get("topics") or []
    if not case_first_enabled() or not state.get("read") or not _case_first_topics(topics):
        return {}
    analysis = _restored_analysis(state["analysis"])
    client = (client or str(analysis.get("parte_assistita") or (state.get("answered") or {}).get("parte_assistita")
                            or "")).strip()
    if not client:
        return {}
    if state.get("read_side") and case_read.same_party(state["read_side"], client):
        _log("reside", {"client": client, "skipped": "lette già per questa parte", "read_side": state["read_side"],
                        "lines": len(case_read.side(state["read"].get("hypotheses") or [], client)), "verified": True})
        return {}
    read = dict(state["read"])
    hyps = list(read.get("hypotheses") or [])
    mine = case_read.side(hyps, client)
    before_parties = sorted({h.get("party") for h in hyps if h.get("party")})
    rerun_ms = None
    if not mine:
        t0 = time.time()
        reread = case_read.start(state.get("text") or "", state.get("request") or "", state.get("lang") or "it",
                                 reader=lambda t, r, lang: case_read.read(t, r, lang, client=client)).result()
        rerun_ms = round((time.time() - t0) * 1000)
        mine = [{**h, "party": client} for h in reread.get("hypotheses") or []]
        read["hypotheses"] = hyps + mine
        read["unknown"] = list(read.get("unknown") or []) or list(reread.get("unknown") or [])
        state["read"] = _jsonable(read)
    before = list(topics)
    unanchored, method = attach_side(topics, mine, topic_groups.topic_vectors(None, topics), _pool_rubriche_labels(state),
                                     {"procedural": state.get("procedural") or []}, area_codes(analysis))
    if reorder:                 # the answer step (topics already chosen) only re-attaches
        topic_groups.order(topics)
    where = {id(t): i for i, t in enumerate(topics, 1)}
    remap = {i: where[id(t)] for i, t in enumerate(before, 1)}
    for numbers in list(renumber) + [state.setdefault("chosen", [])]:
        numbers[:] = [remap.get(n, n) for n in numbers]
    state["unanchored"] = _jsonable(unanchored)
    state["read_side"] = client
    # v60g (4): the re-read verified -- the client's own lines, each its party, attached to the topics
    verified = bool(mine) and all(case_read.same_party(h.get("party"), client) for h in mine)
    entry = {"client": client, "lines": len(mine), "rerun_ms": rerun_ms, "reread": rerun_ms is not None,
             "method": method, "anchored": len(mine) - len(unanchored), "verified": verified,
             "parties_read": before_parties, "attached": sum(len(t.get("hypotheses") or []) for t in topics),
             **({} if verified else {"why": "la rilettura non ha dato linee della parte" if rerun_ms is not None
                                     else "nessuna linea della parte"}),
             "order": [t.get("title") for t in topics],
             "moved": {before[i - 1].get("title"): j for i, j in remap.items() if i != j}}
    _log("reside", entry)
    return entry


def log_misaligned(rows: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, str]]]:
    """v60e: the rows whose rubrica is not their text's (article_text.row_misaligned), per code, with their node ids --
    one journal line for ingestion."""
    from .article_text import row_misaligned
    out: Dict[str, List[Dict[str, str]]] = {}
    seen = set()
    for r in rows:
        bad = row_misaligned(r) if r.get("_article") else None
        if not bad:
            continue
        s, d = r.get("s") or {}, r.get("d") or {}
        node = str(s.get("id") or f"{d.get('id') or ''}::{s.get('name') or ''}")
        if node in seen:
            continue
        seen.add(node)
        out.setdefault(_key(r["_article"])[1], []).append({"article": r["_article"][0], "id": node, **bad})
    if out:
        _log("misaligned", {"by_code": out})
    return out


def anchor_channel(row: Dict[str, Any]) -> str:
    """v60h (4): the channel that brought an article: the first read's («rubrica», «testo», «indice»), else the
    retrieval's own («ricerca»)."""
    if row.get("_first_read"):
        return str(row.get("_first_read_channel") or "rubrica")
    return "ricerca"


def log_anchor_channels(topics: Sequence[Dict[str, Any]], rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """v60h (4): per topic, its anchor and the channel that found it (anchor_channel); logged as anchor_channels."""
    by_label = {}
    for r in rows or []:
        if r.get("_article"):
            by_label.setdefault(_label(_key(r["_article"])), r)
    out = []
    for t in topics or []:
        anchor = str(t.get("anchor") or "")
        row = by_label.get(_label(_key(_parse_label(anchor)))) if anchor else None
        out.append({"topic": t.get("title"), "anchor": anchor,
                    "channel": anchor_channel(row) if row else "fuori dal pool",
                    "institution": (row or {}).get("_first_read") or ""})
    counts: Dict[str, int] = {}
    for x in out:
        counts[x["channel"]] = counts.get(x["channel"], 0) + 1
    _log("anchor_channels", {"anchors": out, "by_channel": counts})
    return out


def _with_texts(found: Dict[str, Any]) -> None:
    """The topics' planned articles a late step added (an offence) among the rows kept with their text."""
    have = {_key(r["_article"]) for r in found.get("topic_rows") or [] if r.get("_article")}
    pool = {_key(r["_article"]): r for r in found.get("article_rows") or [] if r.get("_article")}
    for t in found.get("topics") or []:
        for label in _topic_refs(t):
            key = _key(label)
            if key not in have and key in pool:
                found.setdefault("topic_rows", []).append(pool[key])
                have.add(key)


def _case_branches_early(text: str, request: str):
    """v60b: the pool fusion's case-only branches (its baseline search, its distill call and searches) started beside
    the analysis, whose four calls are a chain (frame and facts at once, then the fact issues, then the analysis): the
    retrieval takes their results (defensive_generation.prefetch_case_branches). Nothing without the fusion."""
    import contextlib
    from . import defensive_generation as dg, stage_meter
    if not (feature_flags.enabled(dg.CASE_ANALYSIS_FLAG) and feature_flags.enabled(dg._CASE_POOL_FUSION_FLAG)):
        return contextlib.nullcontext()
    try:
        from .main import NEO4J_DATABASE, driver
    except Exception as exc:          # no graph: the retrieval runs them itself
        logger.warning("interactive: the case branches not started early: %s", exc)
        return contextlib.nullcontext()
    return dg.prefetch_case_branches(stage_meter.counted(driver), NEO4J_DATABASE, text, request)


def prune_abrogated(topics: List[Dict[str, Any]], rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """v60c: an article whose text is its abrogation (article_text.row_abrogated) is no topic's article, planned,
    matched or anchor (the next article anchors); a topic left with none is gone. Logged."""
    from .article_text import row_abrogated, row_misaligned
    gone = {_label(_key(r["_article"])) for r in rows if r.get("_article") and row_abrogated(r)}
    wrong = {_label(_key(r["_article"])) for r in rows if r.get("_article") and row_misaligned(r)}   # v60e
    gone |= wrong
    if not gone:
        return topics
    out = []
    for t in topics:
        arts = list(t.get("articles") or [])
        planned = int(t.get("planned") or len(arts))
        kept = [a for a in arts if _label(_key(_parse_label(a))) not in gone]
        if not kept:
            continue
        t = {**t, "articles": kept, "planned": len([a for a in arts[:planned] if a in kept]) or 1,
             "matched": [a for a in t.get("matched") or [] if a in kept]}
        if _label(_key(_parse_label(t.get("anchor") or ""))) in gone or t.get("anchor") not in kept:
            t["anchor"] = kept[0]
        out.append(t)
    _log("abrogated_out", {"articles": sorted(gone - wrong), "misaligned": sorted(wrong), "topics_before": len(topics),
                           "topics_after": len(out)})
    return out


def rules_in_area(analysis: Dict[str, Any]) -> List[str]:
    """v60d (INTERACTIVE_CASE_FIRST; a servitù rule on a succession): a fired rule fires only on a case whose areas (the
    frame's) hold the rule's own area -- not merely its family (civile) -- unless the rule names its families. The ids
    taken out (logged); nothing without the case's areas."""
    from . import fact_signals
    signals = _signals(analysis)
    areas = set(_areas(analysis))
    if not case_first_enabled() or not areas or not signals.get("fired"):
        return []
    explicit = {r["id"] for r in _isolated("rules", fact_signals.load_rules, []) or [] if r.get("families")}
    out = [r for r in signals["fired"] if r.get("area") and r["area"] not in areas and r.get("id") not in explicit]
    if out:
        gone = {r["id"] for r in out}
        signals["fired"] = [r for r in signals["fired"] if r.get("id") not in gone]
        _log("rules_area", {"areas": sorted(areas), "not_fired": [f"{r['id']} ({r['area']})" for r in out]})
    return [r["id"] for r in out]


def _analysed_and_retrieved(text: str, request: str, lang: str, analysis: Dict[str, Any], t0: float, meter,
                            pending=None) -> Dict[str, Any]:
    """The analysis' last steps and retrieval, under the request's stage meter; the times as "_t1", "_t2"."""
    from . import defensive_generation as dg
    meter.mark("unstated_rules")
    if case_first_enabled():        # v60d: §4's criminal content only from the store, for the topics' offences
        analysis["store_offences_only"] = True
    _isolated("rules_area", lambda: rules_in_area(analysis))
    analysis[CASE_KEY] = dg._case_text(text, request)
    _isolated("unstated_rules", lambda: mark_unstated(analysis))
    t1 = time.time()
    found = _isolated("retrieval", lambda: _retrieve(text, request, lang, analysis, pending),
                      {"article_rows": [], "regime_rows": [], "topics": [], "others": [], "topic_rows": []})
    meter.stop()
    return {**found, "_t1": t1, "_t2": time.time()}


def analyse_case(text: str, request: str = "", lang: str = "it") -> Dict[str, Any]:
    """Step 1: the analysis object, as a new state (TOPICS_OFFERED is set by round1)."""
    from . import case_read, defensive_generation as dg, render_computed, stage_meter
    t0 = time.time()
    pending = case_read.start(text, request, lang) if case_first_enabled() else None     # v60 (A): beside the rest
    meter = stage_meter.Meter()                 # v60: seconds and calls per stage, one journal line
    with meter.active(), _case_branches_early(text, request):
        meter.mark("analysis")
        analysis = dg._analyse_case_for_pipeline(text, lang, request)
        _isolated("relative_periods", lambda: relative_periods(analysis, text, request))      # v60g (6)
        _isolated("terms_untriggered", lambda: trigger_terms(analysis, text, request))         # v60h (3)
        found = _analysed_and_retrieved(text, request, lang, analysis, t0, meter, pending)
    t1, t2 = found.pop("_t1"), found.pop("_t2")
    round1_stages = meter.as_log()
    _log("retrieval", round1_stages)
    read = pending.result() if pending is not None else {}
    unanchored: List[Dict[str, Any]] = []
    read_side = ""
    if pending is not None:
        from . import topic_groups
        hypotheses = read.get("hypotheses") or []
        # v60e: the client known now (the analysis', else the read's): its lines; unknown: every party's
        known = str(analysis.get("parte_assistita") or (read.get("client") or {}).get("name") or "").strip()
        mine = case_read.side(hypotheses, known) if known else []
        read_side = known if mine else ""
        unanchored, method = attach_side(found["topics"], mine or hypotheses, found.get("vectors"),
                                         found.get("rubriche"), found.get("first_read_search"), area_codes(analysis))
        if found.get("rubriche") is not None:       # v60c: every identified offence in a topic
            found["offences"] = _isolated("offences", lambda: topic_groups.ensure_offences(
                found["topics"], found.setdefault("others", []), found["article_rows"], found.get("unit_of") or {},
                hypotheses), []) or []
            _with_texts(found)
        topic_groups.mark_area(found["topics"], area_codes(analysis))     # v60d: the area's code first, by institution
        topic_groups.order(found["topics"])
        _log("topic_order", {"codes": area_codes(analysis),
                             "topics": [{"title": t.get("title"), "code": t.get("code"), "in_area": t.get("in_area"),
                                         "institutions": sorted(topic_groups.institutions_of(t))}
                                        for t in found["topics"]]})
        _isolated("anchor_channels", lambda: log_anchor_channels(found["topics"], found["article_rows"]))   # v60h (4)
        _log("first_read_parties", {          # v60g (4): every named party's lines while the client is not known
            "client_known": bool(known), "parties": [p.get("name") for p in read.get("parties") or []],
            "with_lines": sorted({h.get("party") for h in hypotheses if h.get("party")}),
            "missing": case_read.missing_parties(read), "topped_up": read.get("topped_up"),
            "alternatives": [a.get("name") for a in read.get("alternatives") or []]})
        _log("hypotheses", {"n": len(hypotheses), "anchored": len(hypotheses) - len(unanchored),
                            "unanchored": len(unanchored), "method": method,
                            "ms": round(pending.ms) if pending.ms is not None else None,
                            "wait_ms": round(pending.wait_ms or 0), "llm": pending.llm, "timed_out": pending.timed_out,
                            "budget_s": case_read.READ_SECONDS})
    rendered = _isolated("render_computed", lambda: render_computed.render(analysis, party(analysis)), "") or ""
    state = {
        "schema": SCHEMA, "version": 0, "state": "", "created": _now().isoformat(), "lang": lang,
        "text": text, "request": request, "additions": [],
        "analysis": _jsonable({**{k: v for k, v in analysis.items() if k != "topics"},
                               CASE_KEY: dg._case_text(text, request)}),
        "topics": _jsonable(found["topics"]), "others": _jsonable(found.get("others") or []),
        "units": _jsonable(found.get("units") or {}),            # v60r (2): each article's Sezione / Capo
        "rows": [_compact_row(r) for r in found["topic_rows"]],
        "tool_rows": [_compact_row(r) for r in found["regime_rows"]],
        "pool": [{"_article": list(r["_article"]), "_multi_signal": bool(r.get("_multi_signal")),
                  "rubrica": str((r.get("s") or {}).get("titolo_articolo") or "")}
                 for r in found["article_rows"] if r.get("_article")],
        "rendered": rendered, "answered": {}, "chosen": [], "angle": "", "changes": [], "answers": {},
        "timings": {"analysis_s": round(t1 - t0, 1), "retrieval_s": round(t2 - t1, 1),
                    "round1_stages": {k: v for k, v in round1_stages.items() if k.endswith("_s")}}}
    if pending is not None:     # v60: the first read, the hypotheses no topic anchors, the rows for the answer step
        state.update(read=_jsonable(read), unanchored=_jsonable(unanchored), read_side=read_side,
                     procedural=list((found.get("first_read_search") or {}).get("procedural") or []),   # v60n
                     reserve=[_compact_row(r) for r in found.get("reserve") or []],
                     grading=_jsonable(found.get("grading") or {}), lost_rules=list(found.get("lost_rules") or []),
                     capo_gate=_jsonable(found.get("capo_gate") or {}),                 # v60q
                     area_gate=_jsonable(found.get("area_gate") or {}))
        state["timings"].update(first_read_s=round(pending.ms / 1000, 1) if pending.ms is not None else None,
                                first_read_wait_s=round((pending.wait_ms or 0) / 1000, 1),
                                grading_s=(found.get("grading") or {}).get("s"),
                                topics_s=(found.get("grading") or {}).get("topics_s"))
    state["questions"] = _isolated("questions", lambda: questions(_restored_analysis(state["analysis"]), (),
                                                                  state["topics"], state["rows"]), [])
    if pending is not None:     # v60c (5): what the case does not say, asked (after the rules' and tables' questions)
        state["read_unknown"] = list(read.get("unknown") or [])     # v60d: after the rules' questions, never before
        dropped: List[Dict[str, Any]] = []
        state["questions"] = with_read_questions(state["questions"], state["read_unknown"], {}, dropped,
                                                 case=case_text(state), names=party_names_of(state),
                                                 client=client_of(state))
        state["qualifications"] = [d["point"] for d in dropped if d.get("dropped") == "qualificazione"]    # v60l
        state["states_of_mind"] = [d["point"] for d in dropped if d.get("dropped") == "stato soggettivo"]  # v60q
        _log("first_read_questions", {"raw": state["read_unknown"], "raw_reply": read.get("unknown_raw"),
                                      "asked": [q["id"] for q in state["questions"] if q["id"].startswith("read_")],
                                      "dropped": dropped})
    if pending is not None:     # v60f: the per-article records, the intermediate representation the answer carries on
        _isolated("records", lambda: build_records(state, list(found["topic_rows"]) + list(found["article_rows"])))
    touch(state, "analysis")
    _log("analysis", {"topics": [t.get("title") for t in state["topics"]], "rows": len(state["rows"]),
                      "questions": [q["id"] for q in state["questions"]], "rendered_chars": len(rendered),
                      "timings": state["timings"]}, (time.time() - t0) * 1000)
    return state


# --- 2c. the questions ----------------------------------------------------------------------------------------------

def _low(value: Any) -> str:
    return " ".join(str(value or "").lower().split())


def _blank(value: Any) -> bool:
    return not str(value or "").strip()


def _present_year(facts: Dict[str, Any]) -> int:
    from . import fact_signals
    now = fact_signals.present(facts)
    if now:
        return now[1][0].year
    latest = fact_signals.latest_fact(facts)
    return latest[1][0].year if latest else _now().year


_DETENTION_RE = r"conduttor|inquilin|locatari|comodatari|affittuari|locazion|comodat|affitt|detenzion"

# a requisite's words -> the fact it needs (requisite_status reads the facts; the questions ask the unknown ones)
_REQUISITE_FACTS: Tuple[Tuple[str, str], ...] = (
    (r"visibil|apparent|permanent", "work_visible"),
    (r"stato dal quale risulta|post[eoa] o lasciat", "work_at_separation"),
    (r"stesso proprietario|unico proprietario|cessarono di appartenere", "common_owner"),
    (r"alienazion|divisione|separa|attualmente divisi|cessarono di appartenere", "separation_act"),
    (r"intercluso|chiuso|uscita sulla via pubblica|accesso alla via pubblica|circondato da fondi", "landlocked"),
    (r"insufficient|inadatt|disagevol|ampliament", "access_insufficient"),
    (r"vent\w* anni|ventenn|possesso continuato|continuat\w* per|per (?:dieci|venti) anni", "possession_years"),
    (r"violen|clandestin|pacific", "possession_peaceful"),
    (r"titolo\b|trascritt|trascrizion", "title_from_non_owner"),
    # v56: art. 52 co. 2 c.p.
    (r"legittimamente detenut", "weapon_lawful"),
    (r"desistenza", "no_desistance"),
    (r"pericolo d['’]\s*aggression", "aggression_danger"),
)
# the fact -> the question family that asks it (a fact with none is never asked)
_FAMILY = {"work_visible": "opera_separazione", "work_at_separation": "opera_separazione",
           "common_owner": "atto_separazione", "separation_act": "atto_separazione",
           "possession_years": "possesso_inizio", "landlocked": "accesso_alternativo",
           "access_insufficient": "accesso_alternativo", "fact_date": "data_fatto", "notice_date": "data_messa_in_mora",
           "act_notice_date": "atto_notifica", "measure_date": "data_misura", "weapon_lawful": "arma_detenuta",
           "no_desistance": "desistenza", "aggression_danger": "pericolo_aggressione", "elusive_act_date": "data_atto_elusivo",
           "judgment_date": "data_giudicato"}
# a table topic's question families (topic_tables "questions") -> the fact each asks
_FAMILY_FACT = {"data_fatto": "fact_date", "data_messa_in_mora": "notice_date", "atto_notifica": "act_notice_date",
                "data_misura": "measure_date", "arma_detenuta": "weapon_lawful", "desistenza": "no_desistance",
                "pericolo_aggressione": "aggression_danger", "data_atto_elusivo": "elusive_act_date",
                "data_giudicato": "judgment_date"}
# v56: the defence's facts (art. 52 co. 2 c.p.), in facts["defence"][0]: fact -> (its field, the question asks it inverted)
_DEFENCE = {"weapon_lawful": ("weapon_lawful", False), "no_desistance": ("no_desistance", True),
            "aggression_danger": ("aggression_danger", False)}
# an open term's topic asks the date its term runs from: the fact (prescription, querela) or the notice (a contract's)
_TERM_FACTS = {"querela": "fact_date", "prescrizione_extracontrattuale": "fact_date", "prescrizione_circolazione": "fact_date",
               "prescrizione_ordinaria": "notice_date", "denuncia_vizi_vendita": "notice_date",
               "denuncia_difformita_appalto": "notice_date", "denuncia_conformita_consumo": "notice_date",
               "denuncia_gravi_difetti": "notice_date", "ricorso_tar": "act_notice_date", "ricorso_tar_appalti": "act_notice_date",
               "nullita_giudicato": "elusive_act_date", "ottemperanza": "judgment_date"}
_TAR_TERMS = ("ricorso_tar", "ricorso_tar_appalti")
_MEASURE_RE = r"ordinanz\w* cautelar|misur\w* cautelar|custodia cautelare|arresti domiciliari|arrest"
_NOTICE_RE = r"diffid|messa in mora|mett\w* in mora|contestaz|reclam|denunci|raccomandat|pec\b"
# an anchor whose text is not in the session: the facts its requisites name, by number (c.c.)
_ANCHOR_FACTS = {"1062": ("work_at_separation", "common_owner", "separation_act"), "1061": ("work_visible",),
                 "1158": ("possession_years",), "1159": ("possession_years", "title_from_non_owner"),
                 "1051": ("landlocked",), "1052": ("access_insufficient",), "1054": ("landlocked", "separation_act")}
# v60d (a servitù question on a succession): each family's area -- asked only when the case's areas (the frame's) hold
# it, and a real-rights family only for a topic whose anchor is in the c.c.'s property, real rights and possession
# (Libro III, artt. 810-1172); a family with no area here (the dates, the client) is every area's
_FAMILY_AREA = {"opera_separazione": "diritti_reali", "atto_separazione": "diritti_reali",
                "possesso_inizio": "diritti_reali", "accesso_alternativo": "diritti_reali",
                "arma_detenuta": "penale", "desistenza": "penale", "pericolo_aggressione": "penale", "data_misura": "penale",
                "atto_notifica": "amministrativo", "data_atto_elusivo": "amministrativo",
                "data_giudicato": "amministrativo"}
_REAL_RIGHTS_RANGE = (810, 1172)


def family_fits(family: str, areas: Sequence[str], anchor: str = "") -> bool:
    """v60d: the family's area among the case's (no areas known: yes), and a real-rights family's topic anchored in
    the c.c.'s Libro III (no anchor: by the areas only)."""
    area = _FAMILY_AREA.get(family)
    if not area:
        return True
    if areas and area not in areas:
        return False
    if area == "diritti_reali" and anchor:
        number, code = _key(_parse_label(anchor))
        m = re.match(r"\d+", number or "")
        if code not in ("c.c.", "") or not m or not _REAL_RIGHTS_RANGE[0] <= int(m.group(0)) <= _REAL_RIGHTS_RANGE[1]:
            return False
    return True


_KIND = {"opera_separazione": "yesno", "atto_separazione": "transfer", "possesso_inizio": "date",
         "accesso_alternativo": "yesno", "data_fatto": "date", "data_messa_in_mora": "date", "atto_notifica": "date",
         "data_misura": "date", "arma_detenuta": "yesno", "desistenza": "yesno", "pericolo_aggressione": "yesno",
         "data_atto_elusivo": "date", "data_giudicato": "date"}


def _names_in(text: str) -> List[str]:
    low = _low(text)
    return list(dict.fromkeys(name for pattern, name in _REQUISITE_FACTS if re.search(pattern, low)))


def _anchor_facts(topic: Dict[str, Any], rows: Sequence[Dict[str, Any]]) -> List[str]:
    """The facts the topic's anchor's requisites name: from its operative commi's requisites (rule_quote.requisites; a
    comma with none, its whole text), else, with no text in the session, _ANCHOR_FACTS."""
    from . import rule_quote
    term = re.match(r"termine (\S+)", str(topic.get("source") or ""))
    extra = [_TERM_FACTS[term.group(1)]] if term and term.group(1) in _TERM_FACTS else []
    number, code = _parse_label(topic.get("anchor") or "")
    if number == "1062" and _key((number, code))[1] in ("c.c.", ""):      # 1062: both its questions, always considered
        extra += ["work_at_separation", "separation_act"]
    extra += [_FAMILY_FACT[f] for f in topic.get("questions") or [] if f in _FAMILY_FACT]     # v55: a table's questions
    return list(dict.fromkeys(_anchor_text_facts(topic, rows) + extra))


def _anchor_text_facts(topic: Dict[str, Any], rows: Sequence[Dict[str, Any]]) -> List[str]:
    from . import rule_quote
    number, code = _parse_label(topic.get("anchor") or "")
    row = next((r for r in rows or [] if r.get("_article") and _key(r["_article"]) == _key((number, code))), None)
    if row is not None:
        names: List[str] = []
        for _, comma in rule_quote.row_commi(row):
            for piece in rule_quote.requisites(comma) or [comma]:
                names += _names_in(piece)
        if names:
            return list(dict.fromkeys(names))
    return list(_ANCHOR_FACTS.get(number, ())) if _key((number, code))[1] in ("c.c.", "") else []


def _client_names(facts: Dict[str, Any]) -> List[str]:
    side = str(facts.get("client_side") or "")
    names = [w for w in re.findall(r"[A-ZÀ-Ü][\w'’-]+", side.split("(")[0])]
    names += [p["name"] for p in facts.get("parties") or [] if p.get("name") and re.search(r"\bcliente\b", _low(p.get("role")))]
    return list(dict.fromkeys(names))


def _client_side(who: str, facts: Dict[str, Any]) -> bool:
    low = _low(who)
    return bool(re.search(r"\bcliente\b|assistit", low)) or any(_low(n) in low for n in _client_names(facts) if n)


def _counterparty_matters(facts: Dict[str, Any], signals: Dict[str, Any]) -> bool:
    """The regime that needs the other side's possession: a client resisting a servitude or claiming ownership (the
    counterparty's usucapione is then the issue)."""
    from . import fact_signals
    try:
        roles = fact_signals.client_roles(facts, signals.get("fired") or [])
    except Exception:
        return False
    return bool({"resists_servitude", "claims_ownership"} & set(roles))


def _family_questions(family: str, facts: Dict[str, Any], signals: Dict[str, Any],
                      case_text: str = "") -> List[Dict[str, Any]]:
    """The questions of one family on the facts' unknown fields, each with its template filled (interaction_copy.
    QUESTIONS); a question whose template is missing or cannot be filled is not asked."""
    from . import fact_signals
    out: List[Dict[str, Any]] = []

    def add(qid, key, target, slots, **extra):
        from . import party_names           # v60h (9): a party named by role reads as a role
        slots = {k: party_names.label(v, case_text) if k in ("who", "client", "counterparty", "owner")
                 and isinstance(v, str) else v for k, v in slots.items()}
        text = C.fill(C.QUESTIONS.get(key), **slots)
        text = party_names.contract(text) if text else text
        if text:
            out.append({"id": qid, "family": family, "template": key, "kind": _KIND[family], "target": target,
                        "text": text, "short": C.QUESTION_SHORT.get(key) or family, **extra})
    if family == "opera_separazione":
        works = [(i, w) for i, w in enumerate(facts.get("works_or_signs") or [])
                 if not re.search(fact_signals._CLOSURE_USE_RE, _low(w.get("what")))]       # a gate is the obstruction
        if fact_state("work_at_separation", facts, signals, case_text) != DA_PROVARE:
            return out          # the case text states it (or a field says no): nothing to ask
        for i, w in works:
            if not any(_low(w.get(f)) in ("no", "false") for f in ("visible", "before_separation")):
                add(f"opera_separazione_{i}", "opera_separazione", ["works_or_signs", i, ["visible", "before_separation"]],
                    {"work": str(w.get("what") or "")})
    elif family == "atto_separazione":
        if fact_state("separation_act", facts, signals, case_text) != DA_PROVARE:
            return out          # stated in the case text: a deduced or filled-in value never silences it
        i, entry = _split_entry(facts)
        owner = str(entry.get("owner") or "")
        add("atto_separazione", "atto_separazione", ["property_history", i, ["ended_by"]], {"owner": owner})
    elif family == "possesso_inizio":
        if fact_signals._duration_stated(facts):
            return out
        banned = fact_signals._not_start_years(facts) if facts.get("_regimes") else set()
        owners = {_low(h.get("owner")) for h in facts.get("property_history") or [] if h.get("owner")}
        owners |= {_low(x.get("name")) for x in facts.get("parties") or [] if re.search(r"proprietar", _low(x.get("role")))}
        other_side = _counterparty_matters(facts, signals)
        for i, p in enumerate(facts.get("possession") or []):
            who = str(p.get("who") or "")
            if re.search(fact_signals._CLOSURE_USE_RE, _low(p.get("use"))) or not who:
                continue
            if not _client_side(who, facts) and not other_side:       # the other side's possession: not asked
                continue
            if _low(who) in owners and not re.search(fact_signals._WAY, _low(f"{p.get('asset')} {p.get('use')}")):
                continue            # an owner's possession of its own land is no usucapione
            if re.search(_DETENTION_RE, _low(f"{who} {p.get('use')} {p.get('how')}")):
                continue            # a tenant detains
            since = str(p.get("since") or "")
            if since.strip() and fact_signals.parse_date(since) and not set(re.findall(r"\b(1[89]\d\d|20\d\d)\b", since)) & banned:
                continue
            client = next((n for n in _client_names(facts) if _low(n) in _low(who)), "") if _client_side(who, facts) else ""
            add(f"possesso_inizio_{i}", "possesso_inizio", ["possession", i, ["since"]],
                {"who": client or who, "what": _with_article(_head_noun(p.get("asset"))),
                 "land": _with_article(_client_land_noun(facts))})
    elif family == "accesso_alternativo":
        access = facts.get("access") or []
        others = [n for n in (p.get("name") for p in facts.get("parties") or []) if n and not _client_side(n, facts)]
        client = (_client_names(facts) or [""])[0]
        for i, a in enumerate(access):
            asset = str(a.get("asset") or "")
            if not _blank(a.get("landlocked")) or any(re.search(rf"\b{re.escape(n)}\b", asset) for n in others):
                continue            # known, or another party's land
            add(f"accesso_alternativo_{i}", "accesso_alternativo", ["access", i, ["landlocked"]], {"client": client},
                invert=True)
        if not access:
            add("accesso_alternativo", "accesso_alternativo", ["access", None, ["landlocked"]], {"client": client},
                invert=True, new_item={"asset": _client_land(facts)})
    elif family in ("arma_detenuta", "desistenza", "pericolo_aggressione"):
        name = _FAMILY_FACT[family]
        if fact_state(name, facts, signals, case_text) != DA_PROVARE:
            return out
        field, invert = _DEFENCE[name]
        client = (_client_names(facts) or [""])[0] or C.DEFENDER
        index = 0 if facts.get("defence") else None
        add(family, family, ["defence", index, [field]], {"client": client}, invert=invert, new_item={})
    elif family in ("atto_notifica", "data_misura", "data_atto_elusivo", "data_giudicato"):
        events = facts.get("events") or []
        if family == "atto_notifica":       # v55: only while a TAR term is open (its start, the act's notification)
            if not any(t.get("id") in _TAR_TERMS and t.get("status") in OPEN_TERM_STATUS for t in signals.get("terms") or []):
                return out
            pattern, item = fact_signals._ACT, {"what": "notifica del provvedimento impugnato", "kind": "notifica"}
        elif family == "data_atto_elusivo":
            pattern, item = fact_signals._REISSUED, {"what": "nuovo provvedimento in contrasto con la sentenza",
                                                     "kind": "provvedimento"}
        elif family == "data_giudicato":
            pattern, item = fact_signals._JUDGMENT, {"what": "sentenza passata in giudicato", "kind": "altro"}
        else:
            pattern, item = _MEASURE_RE, {"what": "esecuzione o notifica dell'ordinanza cautelare", "kind": "notifica"}
        if fact_state(_FAMILY_FACT[family], facts, signals, case_text) != DA_PROVARE:
            return out
        i = next((k for k, e in enumerate(events) if _blank(e.get("date")) and re.search(pattern, _low(e.get("what")))), None)
        add(family, family, ["events", i, ["date"]], {}, new_item=item)
    elif family in ("data_fatto", "data_messa_in_mora"):
        events = facts.get("events") or []
        if family == "data_fatto":
            patterns = [t.start for t in fact_signals.TERMS if t.id in _TERM_FACTS and _TERM_FACTS[t.id] == "fact_date"
                        and t.start]
            i = next((k for k, e in enumerate(events) if _blank(e.get("date"))
                      and any(re.search(pt, _low(e.get("what"))) for pt in patterns)), None)
            add("data_fatto", "data_fatto", ["events", i, ["date"]], {},
                new_item={"what": "il fatto da cui decorrono i termini", "kind": "altro"})
        else:
            i = next((k for k, e in enumerate(events) if _blank(e.get("date"))
                      and (_low(e.get("kind")) in ("diffida", "denuncia/reclamo") or re.search(_NOTICE_RE, _low(e.get("what"))))),
                     None)
            parties = facts.get("parties") or []
            other = next((p.get("name") for p in parties if re.search(r"controparte", _low(p.get("role")))), "") or \
                next((p.get("name") for p in parties if p.get("name") and not _client_side(p["name"], facts)), "")
            from . import party_names           # v60i (7): «all'inquilino», never «a Inquilino»
            add("data_messa_in_mora", "data_messa_in_mora", ["events", i, ["date"]], {"counterparty": other},
                new_item={"what": party_names.contract(f"messa in mora o contestazione a {party_names.label(other)}"),
                          "kind": "diffida"})
    return out


def _head_noun(text: Any) -> str:
    """The noun a phrase opens with, its article and adjectives dropped ("strada sterrata" -> "strada")."""
    words = re.findall(r"[^\W\d_]+", str(text or "").lower())
    words = [w for w in words if w not in ("il", "lo", "la", "i", "gli", "le", "l", "un", "una", "uno")]
    return words[0] if words else ""


_FEMININE_E = {"corte", "parte", "fonte", "rete", "sorgente", "abitazione", "costruzione", "proprietà"}


def _client_land_noun(facts: Dict[str, Any]) -> str:
    """The noun of the client's land as the facts name it (an access, then a property entry naming the client -- "terreni
    di Leonardo" -> "terreni" -- then the client's own entry), else "fondo"."""
    names = [_low(n) for n in _client_names(facts) if n]
    assets = [a.get("asset") for a in facts.get("access") or []] + [h.get("asset") for h in facts.get("property_history") or []]
    for asset in assets:
        if asset and any(re.search(rf"\bdi\s+{re.escape(n)}\b", _low(asset)) for n in names):
            return _head_noun(asset)
    own = next((h.get("asset") for h in facts.get("property_history") or [] if _low(h.get("owner")) in names and h.get("asset")),
               "")
    return _head_noun(own) or "fondo"


def _with_article(noun: str) -> str:
    """The noun with its definite article, by its ending ("strada" -> "la strada", "terreni" -> "i terreni")."""
    if not noun:
        return ""
    vowel = noun[0] in "aeiouàèéìòù"
    if noun.endswith("i"):
        return f"gli {noun}" if vowel or re.match(r"s[^aeiou]|z|gn|ps|x", noun) else f"i {noun}"
    if noun.endswith("a"):
        return f"l’{noun}" if vowel else f"la {noun}"
    if noun.endswith("e") and noun in _FEMININE_E:
        return f"l’{noun}" if vowel else f"la {noun}"
    return f"l’{noun}" if vowel else (f"lo {noun}" if re.match(r"s[^aeiou]|z|gn|ps|x", noun) else f"il {noun}")


def _client_land(facts: Dict[str, Any]) -> str:
    """The client's land as the facts name it ("il terreno di Leonardo"), for a question on its access."""
    names = [_low(n) for n in _client_names(facts)]
    owner = next((h for h in facts.get("property_history") or []
                  if h.get("owner") and _low(h["owner"]) in names and h.get("asset")), None)
    named = next((h["asset"] for h in facts.get("property_history") or []
                  if h.get("asset") and any(re.search(rf"\bdi\s+{re.escape(n)}\b", _low(h["asset"])) for n in names)), "")
    if owner:
        asset = owner["asset"]
        return asset if re.search(r"\bdi\s+[A-ZÀ-Ü]", asset) else f"{asset} di {owner['owner']}"
    name = (_client_names(facts) or [""])[0]
    from . import party_names               # v60i (7): «il fondo del proprietario», never «di il proprietario»
    if named and name:
        return party_names.contract(f"il terreno di {party_names.label(name)}")
    return party_names.contract(f"il fondo di {party_names.label(name)}") if name else "il fondo del cliente"


def _with_answer(facts: Dict[str, Any], question: Dict[str, Any], value: str) -> Dict[str, Any]:
    """facts (changed in place) with the question's field set to value."""
    from . import fact_signals
    key, index, fields = question["target"]
    items = facts.setdefault(key, [])
    if index is None or index >= len(items):
        items.append(dict(question.get("new_item") or {"asset": question.get("asset") or ""}))
        index = len(items) - 1
        question["target"] = [key, index, fields]
    item = items[index]
    if question.get("invert") and value in ("sì", "no"):       # "has it another access?" -- sì: not landlocked
        value = "no" if value == "sì" else "sì"
    for field in fields:
        item[field] = value
        item.pop(f"{field}_how", None)          # stated now, no longer deduced
        item[f"{field}_stated"] = True          # v55: the client said it (stated_in_case)
        if question["kind"] == "date":
            parsed = fact_signals.parse_date(value)
            item[f"{field}_iso"] = parsed[2] if parsed else ""
    return facts


def _derived(facts: Dict[str, Any], areas: List[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """({fired rule: its conclusion}, {term: its status}) of facts, no logging."""
    from . import fact_signals
    terms = fact_signals.compute_terms(facts, areas)
    fired = fact_signals.fire_rules(facts, terms, areas=areas)
    terms = terms + fact_signals._non_use_for_a_servitude(facts, terms, fired) + fact_signals.no_prescription_terms(facts, fired)
    return ({r["id"]: str(r.get("conclusion") or "") for r in fired},
            {t["id"]: str(t.get("status") or "") for t in terms})


def _rule_names() -> Dict[str, str]:
    from . import fact_signals
    try:
        return {r["id"]: (fact_signals._institutions(r) or [r["id"]])[0] for r in fact_signals.load_rules()}
    except Exception:
        return {}


def _changed(base, after, terms: Optional[Sequence[Dict[str, Any]]] = None) -> List[str]:
    """The changes from base to after, in the copy's words (CHANGE_*): the rules by their institution, the terms by
    their label."""
    rules0, terms0 = base
    rules1, terms1 = after
    rule_name = _rule_names()
    label = {t.get("id"): t.get("label") for t in terms or [] if t.get("id")}

    def rname(k):
        return C.capital(str(rule_name.get(k) or k).replace("_", " "))
    out = [C.fill(C.CHANGE_RULE_ON if k in rules1 else C.CHANGE_RULE_OFF, rule=rname(k))
           for k in sorted(set(rules0) ^ set(rules1))]
    out += [C.fill(C.CHANGE_RULE_TEXT, rule=rname(k)) for k in sorted(set(rules0) & set(rules1)) if rules0[k] != rules1[k]]
    out += [C.fill(C.CHANGE_TERM, term=C.capital(str(label.get(k) or k).replace("_", " ")), before=terms0.get(k) or "—",
                   after=terms1.get(k) or "—")
            for k in sorted(set(terms0) | set(terms1)) if terms0.get(k) != terms1.get(k)]
    return [c for c in out if c]


def _trial_values(question: Dict[str, Any], facts: Dict[str, Any]) -> List[str]:
    if question["kind"] == "date":
        year = _present_year(facts)
        return [str(year - 25), str(year - 5)]
    if question["kind"] == "transfer":
        return ["divisione", "compravendita"]
    return ["sì", "no"]


def questions(analysis: Dict[str, Any], answered: Sequence[str] = (), topics: Optional[Sequence[Dict[str, Any]]] = None,
              rows: Optional[Sequence[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """Per topic, from its anchor's requisites unknown in the facts (_anchor_facts, _fact_holds, _FAMILY): the question
    families' questions (_family_questions, client's side only); each once, with "topics" (the anchors it serves) and
    "unlocks" (the rules, terms and requisites it would change: _changed and the topics' requisites flipping, the most
    of the values tried); ranked by how many, at most MAX_QUESTIONS. No topics given: every family is tried."""
    facts, signals = _facts(analysis), _signals(analysis)
    if not facts:
        return []
    case = str(analysis.get(CASE_KEY) or "")
    areas = _areas(analysis)
    rows = _restored_rows(rows or [])
    wanted: Dict[str, List[Dict[str, Any]]] = {}        # family -> the topics asking it
    unfit: Dict[str, str] = {}
    for topic in topics if topics is not None else [{"anchor": ""}]:
        names = _anchor_facts(topic, rows) if topics is not None else list(_FAMILY)
        for name in names:
            family = _FAMILY.get(name)
            if family and not family_fits(family, areas, topic.get("anchor") or ""):
                unfit.setdefault(family, str(topic.get("anchor") or ""))
                continue
            if family and fact_state(name, facts, signals, case) == DA_PROVARE:
                wanted.setdefault(family, []).append(topic)
    if unfit:
        _log("questions_area", {"areas": list(areas), "not_asked": unfit})
    base = _derived(copy.deepcopy(facts), areas)
    ranked = []
    order = 0
    for family, asking in wanted.items():
        reqs = [n for t in asking for n in (_anchor_facts(t, rows) if topics is not None else list(_FAMILY))
                if _FAMILY.get(n) == family]
        for q in _family_questions(family, facts, signals, case):
            order += 1
            if q["id"] in answered:
                continue
            best: List[str] = []
            for value in _trial_values(q, facts):
                trial = _with_answer(copy.deepcopy(facts), copy.deepcopy(q), value)
                changes = _changed(base, _derived(trial, areas), signals.get("terms"))
                trial_signals = {**signals, "facts": trial}
                changes += [f"requisito: {n}" for n in dict.fromkeys(reqs)
                            if fact_state(n, facts, signals, case) != PROVATO and _fact_holds(n, trial, trial_signals)]
                if len(changes) > len(best):
                    best = changes
            ranked.append((generic_question(q), -len(best), order,
                           {**q, "unlocks": best, "topics": list(dict.fromkeys(t.get("anchor") or "" for t in asking))}))
    ranked.sort(key=lambda x: x[:3])            # v60k: a generic template after every specific one
    first = client_question(facts, answered, case)
    return (first + [q for *_, q in ranked])[:MAX_QUESTIONS]


# v60k: the two question defects, both modes -- a question the case already answers is never asked (asked_in_case: the
# nouns of its core in one case sentence, with a date or a number there when it asks a date); a rule question whose
# template names nothing of the case (generic_question: «Quando è avvenuto il fatto?») ranks below the first read's
# specific ones and goes first at the cap
_ASKS_DATE_RE = re.compile(r"\b(?:quando|da quando|in (?:quale|che) (?:data|anno|giorno|mese)|(?:quale|che) (?:data|anno)|"
                           r"da quale anno)\b", re.IGNORECASE)
_VALUE_RE = re.compile(r"\b\d{1,2}[/.-]\d{1,2}(?:[/.-]\d{2,4})?\b|\b(?:1[89]|20)\d{2}\b|\b\d{1,2}\s+(?:gennaio|febbraio|"
                       r"marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre)\b|\b(?:ieri|oggi|"
                       r"stamattina|stamane)\b", re.IGNORECASE)
_QUESTION_STOP = {"quand", "quale", "quali", "quant", "data", "anno", "giorn", "fatto", "fatti", "serve", "decor",
                  "termi", "dipen", "valut", "indic", "altro", "altri", "anche", "quest", "ancor", "esatt", "punto",
                  "preci", "risul"}


def generic_question(q: Dict[str, Any]) -> bool:
    """A rule question whose template has no slot for a fact of the case (a name, a work, a party): «Quando è avvenuto
    il fatto?». The first read's questions and the client's are never generic."""
    if q.get("kind") == "free" or str(q.get("id") or "").startswith("read_") or q.get("id") == "parte_assistita":
        return False
    template = C.QUESTIONS.get(str(q.get("template") or q.get("family") or q.get("id") or ""))
    return bool(template) and "{" not in template


def question_nouns(q: Dict[str, Any], names: Sequence[str] = ()) -> set:
    """The nouns of the question's core (its first clause: what it asks), five-letter stems, no party's name."""
    from . import issue_records
    text = str(q.get("point") or q.get("text") or "")
    core = re.split(r"[?;(]|,\s|\s—\s|\banche se\b", text, maxsplit=1)[0]
    party = {w[:5] for n in names for w in re.findall(r"[a-zà-ü]{4,}", _low(n))}
    return {n for n in issue_records.element_nouns(core) if n not in _QUESTION_STOP and n not in party}


def asked_in_case(q: Dict[str, Any], sentences: Sequence[str], names: Sequence[str] = ()) -> str:
    """The case sentence already answering the question, "" when none: a question asking a date -- one sentence holding
    the nouns of its core (two, or its one) and a date or a number; any other -- one sentence holding every noun of its
    core (two at least). The client's question never."""
    from . import fact_signals
    if q.get("id") == "parte_assistita":
        return ""
    nouns = question_nouns(q, names)
    if not nouns:
        return ""
    date = q.get("kind") == "date" or bool(_ASKS_DATE_RE.search(str(q.get("point") or q.get("text") or "")))
    for sentence in sentences or []:
        have = {w[:5] for w in re.findall(r"[a-zà-ü]{4,}", _low(sentence))}
        if date:
            if len(nouns & have) >= min(2, len(nouns)) and (_VALUE_RE.search(sentence) or fact_signals.relative_date(sentence)):
                return sentence
        elif len(nouns) >= 2 and nouns <= have:
            return sentence
    return ""


# v60l: a point that asks for a legal characterisation, never a fact a client can state (an event, a date, a document, a
# quantity, a party's act or status): the parere's own job -- never a question nor an open point; in §2 «da verificare»
# (C.QUALIFIED_STATUS) without a question. A closed list: the PO's three («la natura giuridica di…», «se si tratta di…»,
# «la qualificazione di…») and their own variants
_QUALIFICATION_RE = re.compile(r"\bnatura\s+giuridica\b|\bqualificazion[ei]\b|\bqualificar[ei]\b|\bqualificabil[ei]\b|"
                               r"\binquadrament[oi]\s+giuridic[oi]\b|\bse\s+si\s+tratt(?:a|i|asse)\s+di\b|"
                               r"^(?:si\s+)?tratt(?:a|i|asse)\s+di\b", re.IGNORECASE)


def qualification(point: Any) -> bool:
    """A point (a question, or its words) asking for a legal characterisation (_QUALIFICATION_RE)."""
    text = point if isinstance(point, str) else str((point or {}).get("point") or (point or {}).get("text") or "")
    return bool(_QUALIFICATION_RE.search(" ".join(text.split())))


# v60q: a party's state of mind -- what the parere argues from the facts it has, never a fact to ask nor one to list as
# missing: the PO's five (deliberato, intenzionale, consapevole, in mala fede, dolo) and their own variants, a closed list
_STATE_OF_MIND_RE = re.compile(
    r"\bdeliberat\w*|\bintenzional\w*|\bintenzion[ei]\b|\bconsapevol\w*|\b(?:mala|buona|cattiva)\s*fede\b|"
    r"\bdolo\b|\bdolos\w*|\bvolontariamente\b|\bdi\s+proposito\b|\bscientemente\b|\bsapendo\s+che\b|"
    r"\b(?:era|fosse|sia)\s+a\s+conoscenza\b|\bconoscev\w+\b|\bsapev\w+\b", re.IGNORECASE)


def state_of_mind(point: Any, client: str = "", names: Sequence[str] = ()) -> bool:
    """v60q: a point (a question, or its words) on a party's state of mind (_STATE_OF_MIND_RE) -- never the client's own:
    a point naming the client and no other party (names) is a fact the client can state."""
    text = point if isinstance(point, str) else str((point or {}).get("point") or (point or {}).get("text") or "")
    text = " ".join(text.split())
    if not _STATE_OF_MIND_RE.search(text):
        return False
    def named(name: str) -> bool:
        words = [w for w in re.findall(r"[^\W\d_]{3,}", str(name or "")) if w[:1].isupper()]
        return any(re.search(rf"\b{re.escape(w)}\b", text) for w in words)
    others = [n for n in names or () if n and n != client]
    return not (client and named(client) and not any(named(n) for n in others))


def client_of(state: Dict[str, Any]) -> str:
    """The client's name: the one the answer settled, else the extraction's."""
    name = str((state.get("client") or {}).get("name") or "")
    return name or (_client_name(_restored_analysis(state["analysis"])) if state.get("analysis") else "")


def _case_first_topics(topics: Sequence[Dict[str, Any]]) -> bool:
    return any("graded" in t for t in topics or [])


def default_choice(state: Dict[str, Any]) -> List[int]:
    """What «procedi» with no topic chosen takes: v60c, case-first topics -- the immediate answer (topic_groups.immediate:
    the identified offences' topics, the first, the next within the 8 articles); else the first topic."""
    topics = state.get("topics") or []
    if _case_first_topics(topics):
        from . import topic_groups
        return topic_groups.immediate(topics) or [1]
    return [1]


READ_QUESTIONS_MIN = 2      # v60d: the room the first read's own questions keep among the MAX_QUESTIONS


def with_read_questions(asked: List[Dict[str, Any]], unknown: Sequence[str], answered: Dict[str, Any],
                        log: Optional[List[Dict[str, Any]]] = None, case: str = "",
                        names: Sequence[str] = (), client: str = "") -> List[Dict[str, Any]]:
    """v60c (5): the first read's «what the case does not say», each a question after the rules' and the tables', not
    one they already ask (half its words in one of theirs), not answered, at most MAX_QUESTIONS in all. v60d: the read's
    questions keep READ_QUESTIONS_MIN places (the client's question first, then the rules' that fit); each point not
    asked is in log with its reason ("empty", "answered", "dedup" with the question it repeats, "cap"). v60k, both
    modes: case given, a question a case sentence already answers is dropped ("nel caso", asked_in_case: never the
    user's own message); a generic rule question (generic_question) comes after the read's, dropped first at the cap.
    v60l, both modes: a point asking a legal characterisation (qualification) is never a question ("qualificazione").
    v60q, both modes: nor one on a party's state of mind, the client's own aside (state_of_mind: "stato soggettivo")."""
    log = log if log is not None else []
    rules = [q for q in asked if not str(q.get("id") or "").startswith("read_")]
    mine_q: List[Dict[str, Any]] = []
    for i, point in enumerate(unknown or [], 1):
        qid = f"read_{i}"
        point = " ".join(str(point or "").split()).rstrip(" .;")
        if not point:
            log.append({"point": point, "dropped": "empty"})
            continue
        if qid in answered:
            log.append({"point": point, "dropped": "answered"})
            continue
        if qualification(point):        # v60l: the parere's own job, never asked
            log.append({"point": point, "dropped": "qualificazione"})
            continue
        if state_of_mind(point, client, names):     # v60q: argued from the facts, never asked
            log.append({"point": point, "dropped": "stato soggettivo"})
            continue
        mine = _stems(point)
        twin = next((q for q in rules + mine_q
                     if mine and len(mine & (_stems(q.get("text")) | _stems(q.get("short")))) * 2 >= len(mine)), None)
        if twin is not None:
            log.append({"point": point, "dropped": "dedup", "as": twin.get("id")})
            continue
        text = C.fill(C.QUESTION_FROM_READ, point=point[:1].lower() + point[1:])
        if not text:
            log.append({"point": point, "dropped": "empty"})
            continue
        mine_q.append({"id": qid, "family": "lettura", "template": "lettura", "kind": "free", "target": None,
                       "text": text, "short": point if len(point) <= 60 else point[:57].rsplit(" ", 1)[0] + "…",
                       "unlocks": [], "topics": [], "point": point})
    if case:                    # v60k: never a fact the case states
        from . import fact_signals
        sentences = fact_signals.case_sentences(case)
        kept = []
        for q in rules + mine_q:
            said = asked_in_case(q, sentences, names)
            if said:
                log.append({"question": q.get("id"), **({"point": q["point"]} if q.get("point") else {}),
                            "dropped": "nel caso", "sentence": said[:160]})
            else:
                kept.append(q)
        rules = [q for q in rules if q in kept]
        mine_q = [q for q in mine_q if q in kept]
    client = [q for q in rules if q.get("id") == "parte_assistita"]
    generic = [q for q in rules if generic_question(q)]
    specific = [q for q in rules if q not in client and q not in generic]
    room = max(0, MAX_QUESTIONS - len(client) - min(READ_QUESTIONS_MIN, len(mine_q)))
    kept_rules = client + specific[:room]
    kept_read = mine_q[:MAX_QUESTIONS - len(kept_rules)]
    kept_generic = generic[:max(0, MAX_QUESTIONS - len(kept_rules) - len(kept_read))]
    for q in mine_q[len(kept_read):]:
        log.append({"point": q["point"], "dropped": "cap"})
    for q in specific[room:]:
        log.append({"question": q.get("id"), "dropped": "cap (room for the first read)"})
    for q in generic[len(kept_generic):]:
        log.append({"question": q.get("id"), "dropped": "cap (generica, dopo le domande specifiche)"})
    return kept_rules + kept_read + kept_generic


def party_names_of(state: Dict[str, Any]) -> List[str]:
    """The parties' names of the state's facts."""
    facts = _facts(_restored_analysis(state["analysis"])) if state.get("analysis") else {}
    return [str(p.get("name")) for p in facts.get("parties") or [] if isinstance(p, dict) and p.get("name")]


def client_question(facts: Dict[str, Any], answered: Sequence[str] = (), case: str = "") -> List[Dict[str, Any]]:
    """v57 (H): the parties the extraction left as possible clients (fact_signals.one_client: client_ambiguous) -- asked
    first, "Chi assisti …?"; v58: also whenever no party is the client (the parties named, no public body); [] when the
    client is known or the question answered."""
    dead = set(deceased(facts, case))           # v60d: the deceased is never a client option
    names = [n for n in facts.get("client_ambiguous") or [] if n and n not in dead]
    parties = [p for p in facts.get("parties") or [] if isinstance(p, dict) and str(p.get("name") or "").strip()
               and p["name"] not in dead]
    if not any("cliente" in _low(p.get("role")).split(", ") for p in parties):
        # v58 (v57-l3, p2, h7: no party tagged cliente, no question): every time, the parties the case names
        names = names or [p["name"] for p in parties if not re.search(r"pubblica amministrazione|autorit", _low(p.get("role")))]
    if not names or "parte_assistita" in answered:
        return []
    from . import party_names               # v60h (9): the options by role where the case names no one
    shown = [party_names.label(n, case) for n in names]
    joined = shown[0] if len(shown) == 1 else C.LIST_COMMA.join(shown[:-1]) + C.CLIENT_JOIN + shown[-1]
    text = C.fill(C.QUESTIONS.get("parte_assistita"), names=joined)
    if not text:
        return []
    return [{"id": "parte_assistita", "family": "parte_assistita", "template": "parte_assistita", "kind": "party",
             "target": ["parties", None, ["role"]], "text": text, "short": C.QUESTION_SHORT.get("parte_assistita") or
             "parte assistita", "unlocks": ["parte assistita"], "topics": [], "names": names}]


# --- 2. round 1 -----------------------------------------------------------------------------------------------------

def carried_terms(terms: Sequence[Dict[str, Any]], topics: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """v59 (v58-l3: «L'usucapione va verificata», the negatoria's and the rivendicazione's lines, no such topic): the
    computed terms a listed topic carries (topic_step.carries_term); with no topic, all."""
    from . import topic_step
    if not topics:
        return list(terms)
    return [t for t in terms if topic_step.carries_term(t, topics)]


def _criminal_topics(topics: Sequence[Dict[str, Any]]) -> bool:
    from .plan_then_write import _code_key
    return any(_code_key(r[1]) in ("c.p.", "c.p.p.") for t in topics for r in _topic_refs(t, all_=True))


def question_wording(question: Dict[str, Any], topics: Sequence[Dict[str, Any]]) -> str:
    """The question's text; v59 (v58-h10 q.3: «… di prescrizione e di querela» in an administrative case): the date of
    the fact asked for the prescription alone when no listed topic is criminal."""
    if question.get("family") == "data_fatto" and topics and not _criminal_topics(topics) \
            and C.QUESTIONS.get("data_fatto_civile"):
        return C.QUESTIONS["data_fatto_civile"]
    return question.get("text") or ""


def _same_item(short: str, item: str) -> bool:
    """A question's label the topic's condition already says: more than half its words, two at least."""
    words = _stems(short)
    shared = len(words & _stems(item))
    return shared >= 2 and shared * 2 > len(words)


def _to_verify(topic: Dict[str, Any], open_questions: Sequence[Dict[str, Any]]) -> List[str]:
    """A topic's «da verificare» items: its rule's conditions to verify, then the open questions it asks (by its anchor),
    each named by QUESTION_REF."""
    out = [f[len("da verificare: "):] for f in topic.get("facts") or [] if str(f).startswith("da verificare: ")]
    asked = [q for q in open_questions if topic.get("anchor") in (q.get("topics") or [])]
    if asked:           # a term's own item says no more than the question that asks for its date
        out = [i for i in out if not re.search(r": termine (?:da verificare|incerto|non verificabile)$", i)]
    for n, q in enumerate(open_questions, 1):
        if topic.get("anchor") not in (q.get("topics") or []):
            continue
        ref = C.fill(C.QUESTION_REF, short=q.get("short") or q.get("family"), n=n)
        if ref and not any(_same_item(q.get("short") or "", item) for item in out):
            out.append(ref)
    return list(dict.fromkeys(out))


def _articles_line(topic: Dict[str, Any]) -> str:
    """The topic's articles, by code: "1051, 1052, 1054 c.c." (several codes: "52, 55 c.p.; 2044 c.c.")."""
    refs = _topic_refs(topic)
    by_code: Dict[str, List[str]] = {}
    for n, c in refs:
        by_code.setdefault(c, []).append(n)
    return "; ".join(f"{', '.join(ns)} {c}" for c, ns in by_code.items())


def _decides(topic: Dict[str, Any]) -> str:
    """What the topic decides, as the data says it: the fact rule's own "why" (the topic's rule, or the rule of its term),
    else its anchor's rubrica when it says more than the title; "" otherwise."""
    from . import fact_signals, topic_step
    m = re.match(r"(?:regola(?: da verificare)?|termine) (\S+)", str(topic.get("source") or ""))
    why = ""
    if m:
        try:            # the rule's why speaks of its first institution: only that topic, or the rule's own term's
            rule = next((r for r in fact_signals.load_rules() if r.get("id") == m.group(1)), None)
            first = (fact_signals._institutions(rule) or [""])[0] if rule else ""
            if rule and (str(topic.get("source")).startswith("termine")
                         or set(topic_step.content_words(first)) & set(topic_step.content_words(topic.get("title")))):
                why = str(rule.get("why") or "")
        except Exception:
            why = ""
    if not why and topic.get("rubrica") and \
            sorted(topic_step.tokens(topic["rubrica"])) != sorted(topic_step.tokens(topic.get("title"))):
        why = str(topic["rubrica"])
    why = " ".join(why.split()).rstrip(" .")
    why = why[:1].lower() + why[1:] if len(why) > 1 and why[1:2].islower() else why
    if why and str(topic.get("source") or "").startswith("regola da verificare"):       # v55: never a fixed conclusion
        return C.fill(C.DECIDES_TO_VERIFY, decides=why) or why
    return why


def _term_sentence(term: Dict[str, Any]) -> str:
    """A computed term in the copy's words (TERM_TEXTS by term and status, every usucapione term as one), else its own
    computed line."""
    tid, status = str(term.get("id") or ""), str(term.get("status") or "")
    key = ("usucapione" if tid.startswith("usucapione") else tid, status)
    years = re.findall(r"\b(1[89]\d\d|20\d\d)\b", str((term.get("from") or {}).get("date") or ""))
    year = str(term.get("expires") or "") if tid == "non_uso_servitu" else (years[0] if years else "")
    text = C.fill(C.TERM_TEXTS.get(key), year=year)
    line = " ".join(str(term.get("text") or "").split()).rstrip(" .")
    return text or C.fill(C.TERM_LINE, text=C.capital(line) + "." if line else "") or ""


def events_sentence(facts_list: Sequence[Dict[str, str]]) -> str:
    """The dated events as one sentence in chronological order (EVENT_IN / EVENT_ON, joined by LIST_JOIN)."""
    parts = []
    for f in facts_list:
        if f["kind"] != "date" or f.get("source") != "event":
            continue
        start = re.search(r"\binizio\s+del(?:l['’])?\s*(\d{4})", str(f["date"]), re.IGNORECASE)
        day = re.search(r"\b\d{1,2}\s+[a-z]+\s+\d{4}\b|\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b", str(f["date"]))
        part = (C.fill(C.EVENT_START, year=start.group(1), what=f["what"]) if start else
                C.fill(C.EVENT_ON if day else C.EVENT_IN, date=f["date"], what=f["what"]))
        if part:
            parts.append(part)
    return C.capital(C.LIST_JOIN.join(parts)) + "." if parts else ""


def _sold(facts: Dict[str, Any], name: str) -> bool:
    """The party's sale or transfer is in the facts (fact_signals._transferred), or its own ownership ended by one."""
    from . import fact_signals
    history = facts.get("property_history") or []
    if fact_signals._transferred(facts, name, history):
        return True
    return any(_low(h.get("owner")) == _low(name) and _low(h.get("ended_by")) in fact_signals._SEPARATING | {"compravendita"}
               and _low(h.get("ended_by")) != "successione" for h in history)


def known_facts(analysis: Dict[str, Any], request: str = "") -> List[Dict[str, str]]:
    """The facts as they stand: parties with roles, the dated events, the computed terms (in the copy's words). v60: an
    event that is the user's request (request: the message's request, the stated angle) is no fact."""
    from .case_read import not_request
    signals, facts = _signals(analysis), _facts(analysis)
    out: List[Dict[str, str]] = []
    client = _low(facts.get("client_side"))
    for p in facts.get("parties") or []:
        if not p.get("name"):
            continue
        role = p.get("role") or ""
        if re.search(r"\bproprietari[oa]\b", role, re.IGNORECASE) and _sold(facts, p["name"]):
            # v56: who sold before the dispute owns nothing now (h3: Sempronio sold to Domitilla in 2003)
            role = re.sub(r"\b(?:ex\s+)?proprietari([oa])\b", lambda m: C.FORMER_OWNER[m.group(1)], role,
                          flags=re.IGNORECASE)
        from . import party_names           # v60h (9): a role reads as a role, never as a name
        shown = party_names.label(p["name"], str(analysis.get(CASE_KEY) or ""))
        text = C.fill(C.PARTY, name=shown, role=role) if role else C.fill(C.PARTY_NO_ROLE, name=shown)
        if client and _low(p["name"]) in client:
            text = C.fill(C.PARTY_CLIENT, party=text) or text
        out.append({"kind": "party", "name": p["name"], "role": role, "text": text or p["name"]})
    dated = [(e.get("date_iso") or "", e.get("date"), e.get("what")) for e in facts.get("events") or []
             if e.get("date") and e.get("what") and (not request or not_request(str(e.get("what")), request))]
    for iso, date, what in sorted(dated, key=lambda x: x[0] or "9999"):
        what = " ".join(str(what).split()).rstrip(" .")
        out.append({"kind": "date", "source": "event", "date": str(date), "what": what, "text": f"{date}: {what}"})
    shown: set = set()
    for t in signals.get("terms") or []:
        text = _term_sentence(t)
        if text and text not in shown:
            shown.add(text)
            out.append({"kind": "term", "id": str(t.get("id") or ""), "status": str(t.get("status") or ""), "text": text,
                        "ref": str(t.get("ref") or ""), "label": str(t.get("label") or "")})
    return out


def interaction(state: Dict[str, Any], round_: int, actions: Sequence[str]) -> Dict[str, Any]:
    """The structured field of the API response (the frontend's)."""
    open_q = state.get("questions") or []
    return {"schema": SCHEMA, "version": state.get("version"), "state": state.get("state"), "round": round_,
            "topics": [{"n": i, "title": t.get("title"), "anchor": t.get("anchor"),
                        "articles": [_label(r) for r in _topic_refs(t)], "to_verify": _to_verify(t, open_q),
                        "answered": str(i) in (state.get("answers") or {})}
                       for i, t in enumerate(state.get("topics") or [], 1)],
            "other_profiles": [{"title": t.get("title"), "articles": [_label(r) for r in _topic_refs(t)],
                                "why": t.get("why") or ""} for t in state.get("others") or []],
            "facts": known_facts(_restored_analysis(state.get("analysis") or {})),
            "questions": [{"id": q["id"], "n": i, "text": q["text"], "kind": q["kind"], "family": q.get("family"),
                           "topics": q.get("topics") or [], "unlocks": len(q.get("unlocks") or [])}
                          for i, q in enumerate(open_q, 1)],
            "chosen": list(state.get("chosen") or []), "changes": list(state.get("changes") or []),
            "actions": list(actions)}


def _capo_label(name: str) -> str:
    """A Capo's name as the corpus writes it ("CAPO II – DELLE SERVITU’ COATTIVE") in sentence case ("Capo II – Delle
    servitu’ coattive")."""
    m = re.match(r"\s*(capo\s+[ivxlcdm\d]+(?:-?[a-z]+)?)\s*[–—-]?\s*(.*)$", str(name or ""), re.IGNORECASE)
    if not m:
        return C.capital(str(name or "").strip().lower())
    head = "Capo " + m.group(1).split(None, 1)[1].upper()
    rest = m.group(2).strip().lower()
    return f"{head} – {C.capital(rest)}" if rest else head


def other_profile_lines(others: Sequence[Dict[str, Any]]) -> List[str]:
    """The «altri profili», one line per anchor Capo (OTHER_PROFILE_LINE: its articles together), else by title."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for t in others:
        groups.setdefault(_capo_label(t["capo"]) if t.get("capo") else "", []).append(t)
    lines = []
    for capo, topics in groups.items():
        if capo:
            refs = list(dict.fromkeys(a for t in topics for a in (t.get("articles") or [])[:TOPIC_ARTICLES]))
            lines.append(C.fill(C.OTHER_PROFILE_LINE, capo=capo, articles=_articles_line({"articles": refs})))
        else:
            lines += [C.fill(C.OTHER_PROFILE_LINE_TITLE, title=t["title"], articles=_articles_line(t)) for t in topics]
    return [x for x in lines if x]


def _client_name(analysis: Dict[str, Any]) -> str:
    facts = _facts(analysis)
    return (_client_names(facts) or [str(analysis.get("parte_assistita") or "").strip()])[0]


def round1(state: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Round 1, every line from interaction_copy."""
    analysis = _restored_analysis(state["analysis"])
    open_q = state.get("questions") or []
    from . import party_names               # v60h (9): roles are not names
    case = str(analysis.get(CASE_KEY) or state.get("text") or "")
    opening = C.fill(C.ROUND1_OPENING, client=party_names.label(_client_name(analysis), case))
    lines = [party_names.contract(opening) if opening else opening, ""]
    if state.get("angle"):          # v60: the question the user asked, the stated focus
        lines += [C.fill(C.ANGLE_STATED_CHOSEN if state.get("angle_first") else C.ANGLE_STATED, angle=state["angle"]), ""]
    lines += [C.TOPICS_TITLE, ""]
    several = len({_low(h.get("party")) for t in state.get("topics") or [] for h in t.get("hypotheses") or []
                   if h.get("party")}) > 1
    if state.get("topics"):
        for i, t in enumerate(state["topics"], 1):
            line = (C.fill(C.TOPIC_LINE, n=i, title=t["title"], articles=_articles_line(t), decides=_decides(t))
                    or C.fill(C.TOPIC_LINE_SHORT, n=i, title=t["title"], articles=_articles_line(t)) or t["title"])
            point = C.fill(C.TOPIC_POINT, point=C.LIST_JOIN.join(_to_verify(t, open_q)))
            lines.append(line + (point or ""))
            lines += [C.fill(C.HYPOTHESIS_LINE_PARTY, line=h.get("line"), party=party_names.label(h.get("party") or "", case),
                             claim=h.get("claim"))
                      if several and h.get("party") else C.fill(C.HYPOTHESIS_LINE, line=h.get("line"), claim=h.get("claim"))
                      for h in t.get("hypotheses") or []]       # v60 (A, B); v60e: whose line, when the client is open
    else:
        lines.append(C.NO_TOPICS)
    if state.get("unanchored"):         # v60 (B): a hypothesis no row found anchors: to verify, never asserted
        lines += ["", C.UNANCHORED_TITLE, ""]
        lines += [C.fill(C.UNANCHORED_LINE, claim=h.get("claim")) for h in state["unanchored"]]
    if state.get("others"):
        lines += ["", C.OTHER_PROFILES_TITLE, ""]
        lines += other_profile_lines(state["others"])
    facts = known_facts(analysis, " ".join(x for x in (state.get("request"), state.get("angle")) if x))
    lines += ["", C.FACTS_INTRO, ""]
    parties = [f["text"] for f in facts if f["kind"] == "party"]
    if parties:
        lines.append(C.fill(C.PARTIES_LINE, parties=C.LIST_JOIN.join(parties)))
    events = events_sentence(facts)
    if events:
        lines.append(events)
    terms = " ".join(f["text"] for f in carried_terms([f for f in facts if f["kind"] == "term"], state.get("topics") or []))
    if terms:
        lines.append(terms)
    if not facts:
        lines.append(C.NO_FACTS)
    if open_q:
        lines += ["", C.QUESTIONS_TITLE, ""]
        lines += [C.fill(C.QUESTION_LINE, n=i, text=question_wording(q, state.get("topics") or []))
                  for i, q in enumerate(open_q, 1)]
    lines += ["", C.INSTRUCTION]
    state["state"] = TOPICS_OFFERED
    text = "\n".join(l for l in lines if l is not None)
    state["round1_text"] = text             # v60 (E): the read-back's lint reads the topics as listed
    return text, interaction(state, 1, ("choose_topic", "answer_questions", "proceed"))


# --- 3. the reply ---------------------------------------------------------------------------------------------------

# v60 (H): the shortcuts read without the model -- the same meanings the interpreter gives them
_CONFIRM_RE = re.compile(r"^\W*(?:procedi|procediamo|ok(?:ay)?|vai|genera|sì|si|confermo|conferma|va bene|d['’]accordo|"
                         r"esatto|perfetto|certo|scrivi(?: (?:pure|il parere|la risposta))?|"
                         r"(?:dammi|scrivimi|fammi) (?:il|un) parere|ok così|vai (?:pure|avanti)|avanti|prosegui)"
                         r"\W*(?:pure|grazie|così)?\W*$", re.IGNORECASE)
_ALL_RE = re.compile(r"^\W*(?:tutti|tutte|tutti gli argomenti|tutto)\W*$", re.IGNORECASE)
_NUMBERS_RE = re.compile(r"^\W*(?:(?:approfondisci|argoment[oi]|tema|punto|n\.?|numer[oi])\s*)?"
                         r"(\d{1,2})((?:\s*(?:,|;|e|ed|\+|/|\s)\s*\d{1,2})*)\W*$", re.IGNORECASE)
_ADD_RE = re.compile(r"^\W*(?:aggiungi|aggiungo|inoltre|preciso|precisazione|correggi|correzione)\b", re.IGNORECASE)

# v60 (H): what a reply inside the interaction can do -- the interpreter's intents
INTENTS = ("scegli", "rispondi", "fatti", "nuovo_argomento", "focus", "procedi", "documento", "domanda_flusso",
           "nuovo_caso")
_REPLY_SYSTEM = (
    "Sei l'assistente di uno studio legale. L'utente risponde, con parole sue, a un messaggio che gli proponeva degli "
    "ARGOMENTI numerati e delle DOMANDE con un id. Riporta la sua risposta in un JSON con ESATTAMENTE queste chiavi:\n"
    "- intents: la lista di ciò che chiede, tra: \"scegli\" (sceglie uno o più argomenti, per numero, per nome o "
    "descrivendoli), \"rispondi\" (risponde a domande), \"fatti\" (aggiunge o corregge fatti), \"nuovo_argomento\" "
    "(chiede un profilo non elencato), \"focus\" (cambia l'angolo o la priorità), \"procedi\" (chiede la risposta o "
    "il parere, con qualunque parola: procedi, vai, scrivi, dammi il parere, ok così…), \"documento\" (chiede il "
    "documento, la bozza, il file), \"domanda_flusso\" (una domanda su come funziona questo scambio), \"nuovo_caso\" "
    "(descrive un caso diverso, da capo);\n"
    "- topics: i numeri degli argomenti che sceglie (lista, anche vuota): se li indica per nome o li descrive, il numero "
    "dell'argomento che corrisponde;\n"
    "- confirm: true se chiede di procedere con la risposta, altrimenti false;\n"
    "- answers: {id della domanda: risposta in breve} solo per le domande a cui risponde davvero; per sì/no scrivi «sì» "
    "o «no»; per una data, la data come la scrive; per l'atto di separazione, il tipo (vendita, divisione, donazione, "
    "successione) e la data se la dice;\n"
    "- angle: il profilo o l'angolo che chiede se non è uno degli argomenti, in poche parole; altrimenti \"\";\n"
    "- corrections: le correzioni ai fatti, ciascuna {kind: \"party\", name, role} o {kind: \"date\", event, date};\n"
    "- statements: i fatti nuovi che afferma, ciascuno in una frase, come li dice;\n"
    "- question: la domanda sul funzionamento dello scambio, se ne fa una; altrimenti \"\".\n"
    "Non dedurre nulla che l'utente non scriva. Rispondi SOLO con il JSON.")


def _numbers(text: str, top: int) -> List[int]:
    m = _NUMBERS_RE.match(text or "")
    if not m:
        return []
    found = [int(n) for n in re.findall(r"\d{1,2}", m.group(0))]
    return [n for n in dict.fromkeys(found) if 1 <= n <= top]


def parse_reply(message: str, state: Dict[str, Any]) -> Dict[str, Any]:
    """{"how": "numbers" | "confirm" | "all" | "model" | "none", "topics", "confirm", "answers", "angle", "corrections",
    "statements"} -- deterministic first, then one bounded JSON call."""
    top = len(state.get("topics") or [])
    empty = {"topics": [], "confirm": False, "answers": {}, "angle": "", "corrections": [], "statements": [],
             "intents": [], "question": ""}
    text = (message or "").strip()
    if _CONFIRM_RE.match(text):
        return {**empty, "how": "confirm", "confirm": True, "intents": ["procedi"]}
    if _ALL_RE.match(text):
        return {**empty, "how": "all", "topics": list(range(1, top + 1)), "intents": ["scegli"]}
    numbers = _numbers(text, top)
    if numbers:
        return {**empty, "how": "numbers", "topics": numbers, "intents": ["scegli"]}
    parsed = _isolated("reply_json", lambda: _reply_call(text, state), None)
    if not isinstance(parsed, dict):
        return {**empty, "how": "none"}
    return {**_clean_reply(parsed, state), "how": "model"}


def _reply_call(text: str, state: Dict[str, Any]) -> Any:
    from langchain_core.messages import HumanMessage, SystemMessage
    from . import defensive_generation as dg
    topics = "\n".join(f"{i}. {t['title']}" for i, t in enumerate(state.get("topics") or [], 1)) or "-"
    asked = "\n".join(f"[{q['id']}] {q['text']}" for q in state.get("questions") or []) or "-"
    human = f"ARGOMENTI:\n{topics}\n\nDOMANDE:\n{asked}\n\nRISPOSTA DELL'UTENTE:\n{text}"
    t0 = time.time()
    try:
        return dg._chat_json([SystemMessage(content=_REPLY_SYSTEM), HumanMessage(content=human)], max_tokens=_REPLY_TOKENS,
                             temperature=0)
    finally:
        _log("reply_llm", {"chars": len(text)}, (time.time() - t0) * 1000)


def _yes_no(value: str) -> str:
    low = _low(value)
    if re.match(r"^(?:s[iì]|yes|vero|certo|esatto|confermo)\b", low):
        return "sì"
    if re.match(r"^(?:no|non|falso|mai)\b", low):
        return "no"
    return ""


def _clean_reply(raw: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """The model's reply kept only where it holds: topics in range, answers to open questions in their kind's form,
    corrections with a name or an event and a parseable date."""
    from . import fact_signals
    top = len(state.get("topics") or [])
    by_id = {q["id"]: q for q in state.get("questions") or []}
    topics = [int(n) for n in raw.get("topics") or [] if str(n).isdigit() and 1 <= int(n) <= top]
    answers: Dict[str, str] = {}
    relative: List[str] = []
    for qid, value in (raw.get("answers") or {}).items() if isinstance(raw.get("answers"), dict) else []:
        q = by_id.get(str(qid))
        value = " ".join(str(value or "").split())
        if not q or not value:
            continue
        if q["kind"] == "date":
            parsed = fact_signals.parse_date(value)
            if parsed:
                answers[q["id"]] = value
            elif fact_signals.relative_date(value):          # v60d: «2 giorni fa», «ieri», «a marzo»
                answers[q["id"]] = fact_signals.relative_date(value)
                relative.append(q["id"])
        elif q["kind"] == "party":          # v60d: the client named -- never dropped as a non yes/no (apply_stated)
            answers[q["id"]] = value[:120]
        elif q["kind"] == "transfer":
            kind = fact_signals._enum(value, fact_signals.TRANSFER_KINDS, fact_signals._TRANSFER_SYNONYMS)
            if kind and kind != "altro" or re.search(r"vend|divis|don|succe|eredit", value, re.IGNORECASE):
                answers[q["id"]] = value
        elif q["kind"] == "free":           # v60c (5): a first-read question, answered in the user's words
            answers[q["id"]] = value[:300]
        elif _yes_no(value):
            answers[q["id"]] = _yes_no(value)
    corrections = []
    for c in raw.get("corrections") or []:
        if not isinstance(c, dict):
            continue
        if c.get("kind") == "party" and str(c.get("name") or "").strip():
            corrections.append({"kind": "party", "name": " ".join(str(c["name"]).split()),
                                "role": " ".join(str(c.get("role") or "").split())})
        elif c.get("kind") == "date" and str(c.get("event") or "").strip() and fact_signals.parse_date(str(c.get("date") or "")):
            corrections.append({"kind": "date", "event": " ".join(str(c["event"]).split()),
                                "date": " ".join(str(c["date"]).split())})
    statements = [" ".join(str(s).split()) for s in raw.get("statements") or [] if str(s or "").strip()][:5]
    intents = [i for i in (raw.get("intents") or []) if isinstance(i, str) and i in INTENTS] \
        if isinstance(raw.get("intents"), list) else []
    confirm = bool(raw.get("confirm")) or "procedi" in intents
    return {"topics": topics, "confirm": confirm, "answers": answers,
            "angle": " ".join(str(raw.get("angle") or "").split())[:200], "corrections": corrections,
            "statements": statements, "intents": intents,
            "question": " ".join(str(raw.get("question") or "").split())[:300], "relative_dates": relative}


def reply_stated(message: str, state: Dict[str, Any], parsed: Dict[str, Any]) -> Dict[str, Any]:
    """v60d: what the reply's own words settle, whatever the interpreter read: the client in the first person or by its
    role («Assisto X», «difendo l'imputato», «il mio cliente è X»: parsed["client"]) and a relative date («2 giorni fa»,
    «ieri») as the answer of the open date-of-the-fact question (parsed["answers"], parsed["relative_dates"]). Logged."""
    from . import fact_signals
    said = stated(message)
    out = dict(parsed)
    if said.get("client") or said.get("client_role"):
        out["client"] = {"client": said.get("client") or "", "client_role": said.get("client_role") or ""}
        out["answers"] = {k: v for k, v in (out.get("answers") or {}).items() if k != "parte_assistita"}
    answered = set(state.get("answered") or {})
    date_q = next((q for q in state.get("questions") or [] if q.get("family") == "data_fatto"
                   and q.get("id") not in answered), None)
    if said.get("date") and date_q is not None:
        current = (out.get("answers") or {}).get(date_q["id"])
        if not current or not fact_signals.parse_date(current):
            out["answers"] = {**(out.get("answers") or {}), date_q["id"]: said["date"]}
            out["relative_dates"] = list(dict.fromkeys(list(out.get("relative_dates") or []) + [date_q["id"]]))
    if out.get("client") or said.get("date"):
        _log("reply_stated", {"client": out.get("client"), "date": said.get("date"),
                              "date_question": date_q["id"] if date_q else None})
    return out


def _stems(text: str) -> set:
    return {w[:6] for w in re.findall(r"[a-zà-ü]{4,}", _low(text))}


def _apply_answer(facts: Dict[str, Any], question: Dict[str, Any], value: str) -> None:
    from . import fact_signals
    if question["kind"] == "free":          # v60c: a first-read question -- the answer joins the case as a precisazione
        return
    if question["kind"] == "party":         # v57: the client the user names becomes the only one
        chosen = next((n for n in question.get("names") or [] if _low(n) and _low(n) in _low(value)), "")
        if chosen:
            for p in facts.get("parties") or []:
                if p.get("name") == chosen and "cliente" not in _low(p.get("role")).split(", "):
                    p["role"] = ", ".join(["cliente"] + [r for r in str(p.get("role") or "").split(", ") if r])
            facts["client_side"] = chosen
            facts.pop("client_ambiguous", None)
            fact_signals.one_client(facts, "")
        return
    if question["kind"] == "transfer":
        kind = fact_signals._enum(value, fact_signals.TRANSFER_KINDS, fact_signals._TRANSFER_SYNONYMS)
        _with_answer(facts, question, kind if kind and kind != "altro" else value)
        parsed = fact_signals.parse_date(value)
        key, index, _ = question["target"]
        if parsed and index is not None and index < len(facts.get(key) or []):
            facts[key][index]["to"] = parsed[2]
            facts[key][index]["to_iso"] = parsed[2]
        return
    _with_answer(facts, question, value)


def _apply_correction(facts: Dict[str, Any], correction: Dict[str, str]) -> str:
    from . import fact_signals
    if correction["kind"] == "party":
        parties = facts.setdefault("parties", [])
        match = next((p for p in parties if _low(p.get("name")) == _low(correction["name"])), None)
        if match is None:
            parties.append({"name": correction["name"], "role": correction["role"], "age": ""})
            return C.fill(C.CHANGE_PARTY_ADDED, name=correction["name"]) or correction["name"]
        before = match.get("role") or ""
        if correction["role"]:
            match["role"] = correction["role"]
        return C.fill(C.CHANGE_PARTY_ROLE, name=match["name"], before=before or "—", after=match.get("role") or "—") or ""
    events = facts.setdefault("events", [])
    wanted = _stems(correction["event"])
    best = max(events, key=lambda e: len(wanted & _stems(e.get("what"))), default=None)
    parsed = fact_signals.parse_date(correction["date"])
    if best is not None and wanted & _stems(best.get("what")):
        before = best.get("date") or "—"
        best["date"], best["date_iso"] = correction["date"], parsed[2] if parsed else ""
        return C.fill(C.CHANGE_DATE, what=best.get("what") or "—", before=before, after=correction["date"]) or ""
    events.append({"date": correction["date"], "date_iso": parsed[2] if parsed else "", "what": correction["event"],
                   "who": "", "kind": "altro"})
    return C.fill(C.CHANGE_EVENT_ADDED, date=correction["date"], what=correction["event"]) or ""


# v60h (8): a yes/no answer restated as the fact it settles, affirmative or negative -- never the question's text. A
# rule's question: its statement in the copy (RESTATE_YES_NO); a first read's point «se il lavoratore abbia firmato …»:
# the clause, its subjunctive made indicative, «non» before its verb (and the verb's clitics) for a «no».
_SUBJUNCTIVE = (("ci fossero", "c'erano"), ("ci fosse", "c'era"), ("ci siano", "ci sono"), ("ci sia", "c'è"),
                ("abbiano", "hanno"), ("abbia", "ha"), ("siano", "sono"), ("sia", "è"), ("fossero", "erano"),
                ("fosse", "era"), ("avessero", "avevano"), ("avesse", "aveva"), ("possano", "possono"), ("possa", "può"),
                ("potesse", "poteva"), ("debba", "deve"), ("dovesse", "doveva"), ("vengano", "vengono"),
                ("venga", "viene"), ("esistano", "esistono"), ("esista", "esiste"), ("esistesse", "esisteva"),
                ("risulti", "risulta"), ("sapesse", "sapeva"), ("conoscesse", "conosceva"), ("facesse", "faceva"),
                ("facessero", "facevano"), ("svolgesse", "svolgeva"), ("ricevesse", "riceveva"))
_FINITE = {"ha", "hanno", "è", "sono", "era", "erano", "aveva", "avevano", "ebbe", "può", "possono", "poteva",
           "potevano", "deve", "devono", "doveva", "dovevano", "viene", "vengono", "veniva", "venivano", "esiste",
           "esistono", "esisteva", "risulta", "risultano", "c'è", "c'era", "c'erano", "fu", "furono", "sa", "sapeva",
           "conosceva", "fa", "faceva", "facevano"}
_CLITICS = {"si", "gli", "le", "lo", "la", "li", "ne", "ci", "vi", "mi", "ti", "l'", "c'"}
_IMPERFECT_RE = re.compile(r"[a-zà-ù]{3,}(?:ava|avano|eva|evano|iva|ivano)$")


def _clause_of(point: str) -> str:
    """«se il lavoratore abbia firmato …» -> «il lavoratore ha firmato …»; "" when the point is no «se» clause."""
    text = " ".join(str(point or "").split()).strip(" .;:?")
    m = re.match(r"^(?:(?:il caso non dice|non è chiaro|non si sa|da chiarire)\s+)?se\s+", text, re.IGNORECASE)
    if not m:
        return ""
    text = text[m.end():]
    for subj, ind in _SUBJUNCTIVE:
        text = re.sub(rf"\b{re.escape(subj)}\b", ind, text)
    return text


def _negated(clause: str) -> str:
    """The clause with «non» before its first finite verb (and that verb's clitics); "" when none is found."""
    words = clause.split(" ")
    for i, w in enumerate(words):
        low = w.lower().strip(",;")
        if i == 0 and low not in _FINITE:
            continue
        if low in _FINITE or _IMPERFECT_RE.match(low):
            j = i
            while j > 0 and words[j - 1].lower() in _CLITICS:
                j -= 1
            if j > 0 and words[j - 1].lower() == "non":
                return clause
            return " ".join(words[:j] + ["non"] + words[j:])
    return ""


def stated_fact(question: Dict[str, Any], value: str) -> str:
    """The fact a yes/no answer to the question settles, in words ("" when the answer is no «sì»/«no» or the question
    gives no statement): RESTATE_YES_NO for a rule's question, the point's clause for a first read's."""
    yes = _yes_no(value)
    if not yes:
        return ""
    pair = C.RESTATE_YES_NO.get(str(question.get("template") or question.get("family") or ""))
    if pair:
        return pair[0] if yes == "sì" else pair[1]
    clause = _clause_of(question.get("point") or "")
    if not clause:
        return ""
    return clause if yes == "sì" else _negated(clause)


def _party_named(value: str, state: Dict[str, Any]) -> str:
    """The party of the facts a free answer names («Rossi», «assisto Mario Rossi»), "" when none."""
    facts = _facts(_restored_analysis(state["analysis"]))
    low = _low(value)
    for p in facts.get("parties") or []:
        name = str((p or {}).get("name") or "") if isinstance(p, dict) else ""
        if name and (_low(name) in low or any(len(w) > 3 and re.search(rf"\b{re.escape(w)}\b", low)
                                              for w in _low(name).split())):
            return name
    return stated(value).get("client") or ""


def restatement(state: Dict[str, Any], phase_answered: bool = False) -> str:
    """v60e: after a reply that answers questions, before any answer -- two prose sentences: the client (RESTATE_CLIENT,
    else RESTATE_NO_CLIENT); what the reply answered, in the answers' words (RESTATE_ANSWERS, each once); then what
    comes next (the topics to treat, or the answer to rewrite). Never the questions' text, arrows or repeats."""
    analysis = _restored_analysis(state["analysis"])
    client = str(analysis.get("parte_assistita") or (state.get("answered") or {}).get("parte_assistita") or "").strip()
    from . import party_names               # v60h (9)
    first = party_names.contract(C.fill(C.RESTATE_CLIENT, client=party_names.label(
        client, str(analysis.get(CASE_KEY) or state.get("text") or ""))) or "") if client \
        else C.RESTATE_NO_CLIENT
    seen, items = set(), []
    for item in state.get("said_items") or []:
        key = _low(item).strip(" .")
        if key and key not in seen and not (client and key.endswith(_low(client))):
            seen.add(key)
            items.append(item.strip().rstrip("."))
    second = C.fill(C.RESTATE_ANSWERS, items=C.LIST_JOIN.join(items)) if items else ""
    numbers = (state.get("pending_regenerate") if phase_answered else state.get("chosen")) or default_choice(state)
    titles = C.LIST_JOIN.join(state["topics"][n - 1]["title"] for n in numbers if 0 < n <= len(state.get("topics") or []))
    nxt = C.fill(C.RESTATE_NEXT_ANSWERED if phase_answered else C.RESTATE_NEXT_TOPICS, titles=titles, topics=titles)
    return " ".join(x for x in (first, second) if x) + ("\n\n" + nxt if nxt else "")


def apply_reply(state: Dict[str, Any], parsed: Dict[str, Any]) -> List[str]:
    """The reply's answers, corrections and statements put into the facts (the case text gains them as the client's
    precisazioni); then recompute(). The changes, in words (also in state["changes"])."""
    said: List[str] = []
    client_note = ""
    by_id = {q["id"]: q for q in state.get("questions") or []}
    client = dict(parsed.get("client") or {})
    party_answer = (parsed.get("answers") or {}).get("parte_assistita")
    if party_answer and by_id.get("parte_assistita") and not client:
        client = {"client": _party_named(party_answer, state) or party_answer.strip(" ."), "client_role": ""}
    if client.get("client") or client.get("client_role"):      # v60d: the client the reply names, as round 1 takes it
        _isolated("reply_client", lambda: apply_stated(state, client, [parsed.get("topics") or []]))
        name = (state.get("answered") or {}).get("parte_assistita")
        if name:
            state["answered_in_reply"] = True
            client_note = C.fill(C.CHANGE_CLIENT, name=name) or name
            said.append(C.fill(C.ADDITION_ANSWER, question=C.QUESTION_SHORT.get("parte_assistita") or "parte assistita",
                               answer=name) or name)
    by_id = {q["id"]: q for q in state.get("questions") or []}
    analysis = _restored_analysis(state["analysis"])
    facts = _facts(analysis)
    before = _derived(copy.deepcopy(facts), _areas(analysis)) if facts else None
    items: List[str] = []           # v60e: what this reply answered, in its words (the restatement)
    for qid, value in (parsed.get("answers") or {}).items():
        q = by_id.get(qid)
        if not q or not facts or qid == "parte_assistita":
            continue
        point = C.QUESTION_SHORT.get(q.get("template") or "") or q.get("short") or q.get("family") or ""
        fact = stated_fact(q, value)        # v60h (8): a yes/no answer as the fact it settles
        items.append(fact or C.fill(C.RESTATE_ITEM, point=point, answer=value) or value)
        _apply_answer(facts, q, value)
        state.setdefault("answered", {})[qid] = value
        state["answered_in_reply"] = True
        said.append(C.capital(fact) + "." if fact else C.fill(C.ADDITION_ANSWER, question=q["text"], answer=value)
                    or value)
    if facts and any(qid in (state.get("answered") or {}) for qid in parsed.get("relative_dates") or []) \
            and not str(facts.get("present_date") or "").strip():
        from . import fact_signals          # v60d: a date the user gave relative to today: today is the case's present
        facts["present_date"] = fact_signals._written_day(fact_signals.reply_today())
    notes = [n for n in (_apply_correction(facts, c) for c in parsed.get("corrections") or [] if facts) if n]
    said += notes
    items += [f"{c['name']}, {c['role']}" if c.get("kind") == "party" and c.get("role") else
              f"{c.get('event')}: {c.get('date')}" if c.get("kind") == "date" else str(c.get("name") or "")
              for c in parsed.get("corrections") or []]
    items += [str(x).rstrip(" .") for x in parsed.get("statements") or []]
    state["said_items"] = list(dict.fromkeys(x for x in items if x.strip()))
    said += list(parsed.get("statements") or [])
    state.setdefault("additions", []).extend(s for s in said if s not in state["additions"])
    if not said:
        return []
    state["analysis"] = _jsonable(analysis)
    changes = ([client_note] if client_note else []) + notes + recompute(state, before)
    state["changes"] = changes
    touch(state, "reply")
    _log("reply_applied", {"answers": parsed.get("answers"), "corrections": parsed.get("corrections"),
                           "statements": parsed.get("statements"), "changes": changes})
    return changes


def recompute(state: Dict[str, Any], before: Optional[Tuple[Dict[str, str], Dict[str, str]]] = None) -> List[str]:
    """The terms, the rules, the tools, the requisites' facts and the computed block again from the facts; the topics'
    «da verificare» lines from the rules; the open questions. No corpus read: a term's article text is the one read
    before. The changes, in words."""
    from . import defensive_generation as dg, fact_signals, render_computed
    analysis = _restored_analysis(state["analysis"])
    old = _signals(analysis)
    facts = old.get("facts")
    if not facts:
        return []
    areas = _areas(analysis)
    before = before or _derived(copy.deepcopy(facts), areas)       # the facts before the reply's edits, when given
    new = _signals_again(analysis, facts, state.get("request") or "", case_text(state))
    _isolated("rules_area", lambda: rules_in_area(analysis))         # v60d
    _isolated("unstated_rules", lambda: mark_unstated(analysis))
    changes = _changed(before, _derived(copy.deepcopy(new.get("facts") or facts), areas), new.get("terms"))
    rendered = _isolated("render_computed", lambda: render_computed.render(analysis, party(analysis)), state.get("rendered"))
    if (rendered or "") != (state.get("rendered") or ""):
        changes.append(C.CHANGE_PROCEDURAL)
    state["rendered"] = rendered or ""
    state["analysis"] = _jsonable(analysis)
    _isolated("refresh_topics", lambda: _refresh_topics(state, analysis))
    state["questions"] = _isolated("questions", lambda: questions(analysis, list(state.get("answered") or {}),
                                                                  state.get("topics") or [], state.get("rows") or []), [])
    state["questions"] = with_read_questions(state["questions"], state.get("read_unknown") or [],
                                             state.get("answered") or {}, case=case_text(state),
                                             names=party_names_of(state))          # v60c (5); v60k: the case's own
    _isolated("records_refresh", lambda: refresh_records(state))                   # v60f
    _log("recomputed", {"changes": changes, "questions": [q["id"] for q in state["questions"]],
                        "terms_untriggered": [d.get("id") for d in new.get("terms_untriggered") or []]})
    return changes


def _signals_again(analysis: Dict[str, Any], facts: Dict[str, Any], request: str, case: str) -> Dict[str, Any]:
    """The terms, the rules, the tools again from the facts (no corpus read), into analysis["fact_signals"]."""
    from . import defensive_generation as dg, fact_signals
    old = _signals(analysis)
    new = fact_signals.signals_from(copy.deepcopy(facts), old.get("problem") or "", _areas(analysis), request=request,
                                    case=case if case_first_enabled() else None)     # v60h (3): terms need their fact
    if fact_signals.regimes_enabled():
        new["posture"] = fact_signals.claim_posture(case, new.get("facts"))
        _isolated("fit_to_role", lambda: fact_signals.fit_to_role(new, dg._stance_party(analysis)))
        have = {t["id"] for t in new.get("tools") or []}
        keyed = _isolated("regime_tools", lambda: [t for t in fact_signals.regime_tools(new, {**analysis, "fact_signals": new})
                                                   if t["id"] not in have], [])
        new["tools"] = list(new.get("tools") or []) + list(keyed or [])
    sources = {t.get("ref"): t.get("sources") for t in old.get("terms") or [] if t.get("sources")}
    for t in new.get("terms") or []:
        if t.get("ref") in sources:
            t.setdefault("sources", sources[t["ref"]])
    for k, v in old.items():
        new.setdefault(k, v)
    analysis["fact_signals"] = new
    return new


def relative_periods(analysis: Dict[str, Any], text: str, request: str = "") -> List[Dict[str, str]]:
    """v60g (6): the case's dates written relative to its present («tre mesi fa», «da circa due anni», «l'estate
    scorsa») as their dates or ranges (fact_signals.resolve_relative: against the case's own present, else the day the
    user writes), and the terms computed again on them -- an uncertain start is a range, its term DECORSO, NON DECORSO
    or INCERTO by both ends. Logged; the resolved fields."""
    from . import defensive_generation as dg, fact_signals
    facts = _signals(analysis).get("facts")
    if not facts:
        return []
    done = fact_signals.resolve_relative(facts)
    if not done:
        return []
    new = _signals_again(analysis, facts, request, dg._case_text(text, request))
    _log("relative_periods", {"resolved": done, "present_date": facts.get("present_date"),
                              "terms": [f"{t.get('id')}: {t.get('status')}" for t in new.get("terms") or []]})
    return done


def trigger_terms(analysis: Dict[str, Any], text: str, request: str = "") -> List[Dict[str, str]]:
    """v60h (3, INTERACTIVE_CASE_FIRST): the terms computed again with each one's triggering fact required in a case
    sentence (fact_signals.TERM_TRIGGERS: a delivered good for the vizi and conformità terms, a notified act for the
    TAR's, a judgment for the appeal's, …): a term without it is neither computed nor rendered. Logged as
    terms_untriggered (the dropped, and the kept with the sentence that triggers each); the dropped."""
    from . import defensive_generation as dg
    facts = _signals(analysis).get("facts")
    if not facts or not case_first_enabled():
        return []
    new = _signals_again(analysis, facts, request, dg._case_text(text, request))
    dropped = list(new.get("terms_untriggered") or [])
    _log("terms_untriggered", {"dropped": dropped,
                               "kept": [{"id": t.get("id"), "trigger": t.get("trigger")} for t in new.get("terms") or []]})
    return dropped


def _refresh_topics(state: Dict[str, Any], analysis: Dict[str, Any]) -> None:
    """A topic of a rule to verify: its «da verificare» lines from the rule now (none once it fires); v55: a fired rule
    marked unstated (mark_unstated) is one to verify, its conditions those that fail without the unstated fields."""
    from . import topic_step
    signals = _signals(analysis)
    fired = {r["id"]: r for r in signals.get("fired") or []}
    pending = {r["id"]: r for r in topic_step.rules_to_verify(signals, _areas(analysis))}
    for topic in state.get("topics") or []:
        m = re.match(r"regola( da verificare)? (\S+)", str(topic.get("source") or ""))
        rule = fired.get(m.group(2)) if m else None
        if not m or (not m.group(1) and not (rule or {}).get("unstated")):
            continue
        m = re.match(r"regola(?: da verificare)? (\S+)", str(topic["source"]))
        rest = [f for f in topic.get("facts") or [] if not str(f).startswith("da verificare: ")]
        if rule is not None and rule.get("unstated"):        # v55: fired on a field the case does not state
            topic["source"] = f"regola da verificare {m.group(1)}"
            topic["facts"] = [f"da verificare: {c}" for c in rule.get("unstated_conditions") or []] + rest
        elif rule is not None:
            topic["source"] = f"regola {m.group(1)}"
            topic["facts"] = rest
        elif m.group(1) in pending:
            topic["facts"] = [f"da verificare: {c}" for c in pending[m.group(1)]["to_verify"]] + rest


def match_angle(state: Dict[str, Any], angle: str) -> List[int]:
    """The topics an angle names (by the stems it shares with the title, the rubrica and the articles' rubriche); none:
    a topic anchored on the pool article whose rubrica shares most of them, its articles read (topic_step.articles)."""
    wanted = _stems(angle)
    if not wanted:
        return []
    rubriche = {_label(_key(r["_article"])): r.get("rubrica") or "" for r in state.get("pool") or []}
    for row in state.get("rows") or []:
        if row.get("_article"):
            rubriche.setdefault(_label(_key(row["_article"])), str((row.get("s") or {}).get("titolo_articolo") or ""))
    scored = []
    for i, t in enumerate(state.get("topics") or [], 1):
        words = _stems(f"{t.get('title')} {t.get('rubrica')} " + " ".join(rubriche.get(a, "") for a in t.get("articles") or []))
        hit = len(wanted & words)
        if hit:
            scored.append((-hit, i))
    if scored:
        best = min(scored)[0]
        return [i for s, i in sorted(scored) if s == best]
    need = 1 if len(wanted) <= 2 else 2
    pool = [(len(wanted & _stems(r)), label) for label, r in rubriche.items() if r]
    pool = [p for p in pool if p[0] >= need]
    if not pool:
        return []
    label = max(pool)[1]
    topic = _isolated("angle_topic", lambda: _angle_topic(state, angle, _parse_label(label)), None)
    if not topic:
        return []
    state["topics"].append(topic)
    return [len(state["topics"])]


def _angle_topic(state: Dict[str, Any], angle: str, ref: Tuple[str, str]) -> Dict[str, Any]:
    import neo4j
    from . import article_text, defensive_generation as dg, topic_step
    from .doc_lookup import _fetch_doc_names
    from .main import NEO4J_DATABASE, driver
    documents = _fetch_doc_names(driver, NEO4J_DATABASE)
    pool = {_key(r["_article"]): {"_article": tuple(r["_article"]), "_multi_signal": r.get("_multi_signal")}
            for r in state.get("pool") or []}
    analysis = _restored_analysis(state["analysis"])
    with driver.session(database=NEO4J_DATABASE, default_access_mode=neo4j.READ_ACCESS) as session:
        links = topic_step.GraphLinks(session, documents)
        refs = topic_step.articles(ref, links, pool, topic_step.allowed(analysis, case_text(state)), angle)
        have = {_key(r["_article"]) for r in state.get("rows") or [] if r.get("_article")}
        by_code = {topic_step._code(d.get("name")): d.get("name") for d in documents if d.get("name")}
        fetch = [(n, c if c in dg._CODE_PREFIXES else by_code.get(c, c)) for n, c in refs[:TOPIC_ARTICLES]
                 if (n, c) not in have]
        rows = dg._fetch_any(session, documents, fetch) if fetch else []
        article_text.fill_texts(session, rows)
    state["rows"] = list(state.get("rows") or []) + [_compact_row(r) for r in rows]
    title = " ".join(angle.split())
    return {"title": title[:1].upper() + title[1:], "anchor": _label(ref), "how": "angle", "rubrica": "",
            "articles": [_label(r) for r in refs], "facts": [], "source": "angolo dell'utente", "merged": []}


# --- 4. round 2 -----------------------------------------------------------------------------------------------------

def _parties_line(analysis: Dict[str, Any]) -> str:
    """The full «Parti da convenire» line as the procedural section renders it (render_computed: the PARTIES_ENFORCED
    line, else each party to sue with why), without its own label."""
    from . import render_computed
    body = next((b for t, b in _isolated("parties_line", lambda: render_computed.blocks(analysis, party(analysis)), [])
                 or [] if t == "Parti da convenire"), "")
    body = re.sub(r"^\s*Parti da convenire(?: in giudizio)?\s*:\s*", "", " ".join(str(body).split()))
    return body.rstrip(" .")


def round2(state: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Round 2: the copy's ROUND2, its sentences with nothing to say left out."""
    analysis = _restored_analysis(state["analysis"])
    chosen = state.get("chosen") or [1]
    topics = [state["topics"][n - 1] for n in chosen if 0 < n <= len(state.get("topics") or [])]
    open_q = state.get("questions") or []
    facts = known_facts(analysis)
    based = [events_sentence(facts).rstrip(".")] + list(dict.fromkeys(state.get("said_items") or []))     # v60e
    verify = list(dict.fromkeys(re.sub(r"\s*\(domanda \d+\)$", "", v) for t in topics for v in _to_verify(t, open_q)))
    text = C.fill_parts(C.ROUND2, topics=C.LIST_JOIN.join(t["title"] for t in topics),
                        facts=C.LIST_JOIN.join(b for b in based if b), points=C.LIST_JOIN.join(verify),
                        parties=_parties_line(analysis))
    state["state"] = FRAME_RESTATED
    return text or C.CONFIRM_LINE, interaction(state, 2, ("confirm", "correct"))


# --- 5. the topic answer --------------------------------------------------------------------------------------------

def case_sentences(state: Dict[str, Any]) -> List[Dict[str, str]]:
    """v60f: the case's numbered sentences with their certainty (issue_records.sentences): the message, the client's
    precisazioni, and the system's assumptions (an assumed client, the facts' deduced fields)."""
    from . import defensive_generation as dg, issue_records
    assumed = []
    client = state.get("client") or {}
    if client.get("how") == "assunta" and client.get("name"):
        assumed.append(f"Parte assistita assunta: {client['name']}")
    facts = _facts(_restored_analysis(state["analysis"])) if state.get("analysis") else {}
    for key, items in facts.items() if isinstance(facts, dict) else []:
        for item in items if isinstance(items, list) else []:
            for field, how in (item.items() if isinstance(item, dict) else []):
                if field.endswith("_how") and how and item.get(field[:-4]):
                    assumed.append(f"{key}: {field[:-4]} = {item.get(field[:-4])} (dedotto: {how})")
    sents = issue_records.sentences(dg._case_text(state.get("text") or "", state.get("request") or ""),
                                    state.get("additions") or [], assumed[:10])
    mark_sides(sents, state)
    return sents


# v60t: an attributed sentence's speaker -- a party named just before an alleging verb («Sempronio sostiene che…») or
# after «secondo» / «a detta di»; one other than the client speaks for the counterparty (issue_records.COUNTERPARTY:
# never support, the contradiction path); the client's own, or no speaker named, is the client's account
_SAYS_RE = (r"(?:sostien|sostenut|afferm|lament|riferisc|riferit|dichiar|ritien|pretend|eccepi|contest|denunci|accus|"
            r"asseris|asserit)\w*")


def attributed_to(sentence: str, names: Sequence[str]) -> str:
    """The party a sentence attributes its content to (names: the case's parties), "" when none is named."""
    for name in sorted({str(n) for n in names or () if n}, key=len, reverse=True):
        for word in [w for w in re.findall(r"[^\W\d_]{3,}", name) if w[:1].isupper()]:
            w = re.escape(word)
            if re.search(rf"\b{w}\b(?:\s+[\w'’]+){{0,3}}?\s+{_SAYS_RE}", sentence) or \
                    re.search(rf"\b(?:secondo|a detta d\w*|stando a)\s+(?:\w+\s+)?{w}\b", sentence, re.IGNORECASE):
                return name
    return ""


def mark_sides(sents: Sequence[Dict[str, Any]], state: Dict[str, Any]) -> None:
    """In place: each attributed sentence's speaker ("by") and side ("side": the counterparty's when the speaker is a
    party other than the client -- client_of: settled, else the extraction's; unknown client: no side)."""
    from . import issue_records
    names, client = party_names_of(state), _low(client_of(state))
    for s in sents:
        by = attributed_to(s.get("text") or "", names) if s.get("certainty") == issue_records.ALLEGATO else ""
        s["by"] = by
        mine = bool(client) and (_low(by) == client or _low(by) in client or client in _low(by))
        s["side"] = issue_records.COUNTERPARTY if by and client and not mine else ("cliente" if by and mine else "")


def build_records(state: Dict[str, Any], rows: Sequence[Dict[str, Any]]) -> None:
    """v60f: the per-article records and the case's sentences, in the state (issue_records.build); timed and logged."""
    from . import issue_records
    t0 = time.time()
    analysis = _restored_analysis(state["analysis"])
    records = issue_records.build(state.get("topics") or [], rows, analysis, state.get("questions") or [],
                                  state.get("others") or [], state.get("grading") or {})
    state["sentences"] = case_sentences(state)
    issue_records.refresh(records, analysis, state.get("questions") or [], state["sentences"])   # v60g: round 1's links
    state["records"] = issue_records.jsonable(records)
    ms = round((time.time() - t0) * 1000)
    state.setdefault("timings", {})["records_s"] = round(ms / 1000, 2)
    _log("records_built", {"articles": len(records), "requisites": sum(len(r["requisites"]) for r in records.values()),
                           "dropped": sum(1 for r in records.values() if r.get("dropped")),
                           "sentences": len(state["sentences"]),
                           "certainty": {k: sum(1 for x in state["sentences"] if x["certainty"] == k)
                                         for k in ("accertato", "allegato", "ipotizzato", "assunto")}}, ms)


def refresh_records(state: Dict[str, Any]) -> None:
    """v60f: after a reply -- the sentences with the new precisazioni, every status and open question again."""
    from . import issue_records
    if not state.get("records"):
        return
    state["sentences"] = case_sentences(state)
    issue_records.refresh(state["records"], _restored_analysis(state["analysis"]), state.get("questions") or [],
                          state["sentences"])


def persist_client(state: Dict[str, Any], client: str, assumption: str = "") -> None:
    """v60f (B8): the client the answer settles, in the state -- state["client"] {name, how: «indicata dall'utente» |
    «dal caso» | «assunta»} and the analysis' parte_assistita -- so the next answer, the plan and the records read it."""
    analysis = state.get("analysis") or {}
    if (state.get("answered") or {}).get("parte_assistita"):
        how = "indicata dall'utente"
    elif str(analysis.get("parte_assistita") or "").strip() and \
            _low(analysis.get("parte_assistita")) == _low(client) and not assumption:
        how = "dal caso"
    else:
        how = "assunta"
    state["client"] = {"name": client, "how": how}
    if isinstance(state.get("analysis"), dict):
        state["analysis"]["parte_assistita"] = client
    _log("client_persisted", state["client"])


def link_records(state: Dict[str, Any], topics: Sequence[Dict[str, Any]], dead: set) -> Optional[Dict[str, Any]]:
    """v60f (b): the chosen topics' requisites linked to the case's sentences (issue_records.link: one bounded call per
    topic, in parallel), in state["records"]; timed and logged."""
    from . import issue_records
    records = state.get("records") or {}
    if not records:
        return None
    sents = state.get("sentences") or case_sentences(state)
    mark_sides(sents, state)                    # v60t: the client is settled now -- whose account each sentence is
    state["sentences"] = sents
    labels = [[issue_records._label(r) for r in _topic_refs(t) if _key(r) not in dead] for t in topics]
    log = issue_records.link(records, topics, labels, sents)
    by_id = {s["id"]: s for s in sents}
    for rec in records.values():                # a topic linked before (cached) settled again on today's sides
        for r in list(rec.get("requisites") or []) + list(rec.get("procedibilita") or []):
            issue_records.settle(r, by_id)
        rec["verdict"] = issue_records.verdict(rec)
    state.setdefault("timings", {})["records_link_s"] = round(log["total_ms"] / 1000, 2)
    _log("records_link", log, log["total_ms"])
    return log


def plan_inputs(state: Dict[str, Any], analysis: Dict[str, Any], topics: Sequence[Dict[str, Any]],
                refs: Sequence[Tuple[str, str]]) -> List[Dict[str, Any]]:
    """v60f (c, M8): into the analysis the plan reads -- the chosen topics as its fixed issues, each with its planned
    articles and their records (ptw.PLAN_ISSUES_KEY), and the user's question (ptw.PLAN_ANGLE_KEY). The issues ([] with
    no records: the legacy topics, the plan as before)."""
    from . import issue_records, plan_then_write as ptw
    angle = str(state.get("angle") or "").strip()
    if angle:
        analysis[ptw.PLAN_ANGLE_KEY] = angle
    records = state.get("records") or {}
    issues: List[Dict[str, Any]] = []
    if records:
        wanted = {_key(r) for r in refs}
        by_topic = [[r for r in _topic_refs(t) if _key(r) in wanted] for t in topics]
        issues = issue_records.plan_issues(topics, by_topic, records, state.get("sentences") or [])
        if issues:
            analysis[ptw.PLAN_ISSUES_KEY] = issues
    _log("plan_inputs", {"issues": [{"titolo": i["titolo"], "norme": [f"{n} {c}" for n, c in i["norme"]]} for i in issues],
                         "angle": angle, "chars": sum(len(i.get("scheda") or "") for i in issues)})
    return issues


def main_line_topic(topics: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """v60t (2): the answer's main line -- the topic holding the first read's main-line hypothesis, else the first."""
    topics = list(topics or [])
    return next((t for t in topics if any(h.get("line") == "principale" for h in t.get("hypotheses") or [])),
                topics[0] if topics else None)


def anchor_blocks(state: Dict[str, Any], topics: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """v60p: the anchor article of each covered topic (the topics this answer covers -- grade 3, the main line, a
    user-named article) always earns its §2 block (issue_records.block_due: rec["anchor_due"]), never one «non
    applicabile» (v60r 6: its silent elements «da provare», no longer «da verificare»). Every
    record's flag set anew for this answer. Logged; the entries. v60t (2), single shot: the main line's anchor alone
    (main_line_topic; state["main_anchor"]) -- every other article's block needs a proven element (rec["needs_proof"])."""
    from . import issue_records
    records = state.get("records") or {}
    single = single_mode(state)
    for rec in records.values():
        rec.pop("anchor_due", None)
        if single:                          # v60t (2): single shot, a block needs a proven element
            rec["needs_proof"] = True
        else:
            rec.pop("needs_proof", None)
    main = main_line_topic(topics)
    state["main_anchor"] = issue_records._label(_parse_label(main["anchor"])) if main and main.get("anchor") else ""
    out = []
    by_id = {s["id"]: s for s in state.get("sentences") or [] if s.get("id")}
    for t in ([main] if single and main else topics or []):     # v60t (2): single shot, the main line's anchor only
        if not t.get("anchor"):
            continue
        label = issue_records._label(_parse_label(t["anchor"]))
        rec = records.get(label)
        if rec is None:
            out.append({"topic": t.get("title"), "anchor": label, "record": False})
            continue
        before = issue_records.block_due(rec, by_id, open_questions(state))
        rec["anchor_due"] = str(t.get("title") or True)
        out.append({"topic": t.get("title"), "anchor": label, "verdict": rec.get("verdict"), "earned_before": before,
                    "block": issue_records.block_due(rec, by_id, open_questions(state)),
                    "silent": sum(1 for r in rec.get("requisites") or [] if r.get("status") == DA_PROVARE
                                  and not r.get("supporting") and not r.get("contradicting"))})
    _log("anchor_blocks", out)
    return out


def offence_issue(topic: Dict[str, Any]) -> bool:
    """v60s (1): an issue titled after an offence unit -- its anchor (whose Sezione / Capo, else rubrica, names it) is an
    article of c.p. Libro II (rule_quote.offence)."""
    from . import rule_quote
    anchor = topic.get("anchor") or (topic.get("articles") or [""])[0]
    return bool(anchor) and rule_quote.offence(_key(_parse_label(anchor)))


def surviving_issues(state: Dict[str, Any], topics: Sequence[Dict[str, Any]], rows: Sequence[Dict[str, Any]],
                     dead: set) -> List[Dict[str, Any]]:
    """v60s (1): after the qualification check (the record's verdicts, link_records), an offence issue (offence_issue)
    keeps at least one offence article -- c.p. Libro II, a pena in the procedural store (its row's "_proc", else read
    from its text: procedural_facts), not abrogated or misaligned (dead), its record not «non applicabile»; otherwise the
    issue goes with every article it lists -- a procedural or ancillary article never survives its removed offence (in
    another surviving issue it stays that issue's). Each dropped issue logged (issue_dropped). The issues kept."""
    from . import issue_records, procedural_facts, rule_quote
    records = state.get("records") or {}
    by_key = {_key(r["_article"]): r for r in rows if r.get("_article")}
    out = []
    for t in topics:
        if not offence_issue(t):
            out.append(t)
            continue
        refs = [_key(r) for r in _topic_refs(t, all_=True)]
        kept, why = [], {}
        for key in refs:
            if not rule_quote.offence(key):
                continue
            row = by_key.get(key) or {}
            pena = (row.get("_proc") or (procedural_facts.from_row(row) if row else {}) or {}).get("pena")
            verdict = (records.get(_label(key)) or {}).get("verdict")
            if key in dead:
                why[_label(key)] = "abrogato o non allineato"
            elif not pena:
                why[_label(key)] = "nessuna pena nel testo"
            elif verdict == issue_records.NOT_APPLICABLE:
                why[_label(key)] = "non applicabile"
            else:
                kept.append(_label(key))
        if kept:
            out.append(t)
            continue
        _log("issue_dropped", {"issue": t.get("title"), "anchor": t.get("anchor"), "offences": why,
                               "articles": [_label(k) for k in refs]})
    return out


def own_analysis(state: Dict[str, Any], topics: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """v60p: the parere's own analysis for §4's triggers (render_computed.own_trigger): the first read's hypothesis lines
    (claim, institutions, their rubricated names) and the covered topics (title, rubrica, unit, their articles'
    rubriche; their articles)."""
    from . import issue_records
    read = state.get("read") or {}
    lines = [" ".join(str(x) for x in [h.get("claim") or ""] + list(h.get("institutions") or [])
                      + list(h.get("rubricated") or []) if x) for h in read.get("hypotheses") or []]
    records = state.get("records") or {}
    out = []
    for t in topics or []:
        arts = [issue_records._label(_parse_label(a)) for a in t.get("articles") or []]
        text = " ".join(str(x) for x in [t.get("title"), t.get("rubrica"), t.get("unit")]
                        + [(records.get(a) or {}).get("rubrica") for a in arts] if x)
        out.append({"title": str(t.get("title") or ""), "text": text, "articles": arts})
    return {"lines": [x for x in lines if x.strip()], "topics": out}


def record_verdicts(state: Dict[str, Any]) -> Dict[str, str]:
    """v60g: {"N code": the record's verdict} (issue_records.verdict)."""
    from . import issue_records
    return {label: rec.get("verdict") or issue_records.verdict(rec) for label, rec in (state.get("records") or {}).items()}


def open_questions(state: Dict[str, Any]) -> List[str]:
    """v60j: the ids of the round-1 questions still open (asked, not answered)."""
    answered = set((state or {}).get("answered") or {})
    return [str(q["id"]) for q in (state or {}).get("questions") or [] if q.get("id") and q["id"] not in answered]


def record_lines(state: Dict[str, Any], sanctions: Optional[Dict[Tuple[str, str], Optional[tuple]]] = None,
                 beyond: Optional[Dict[str, str]] = None, cap: int = 0) -> Dict[str, Tuple[str, str]]:
    """v60i (1): {"N code": (line, reason)} -- each article with no §2 block due (issue_records.block_due: «non
    applicabile», or «da provare» with no sentence of the case for it), its line and reason (line_reason); a sanction
    article whose obligation is not treated, «non trattata» as such."""
    from . import issue_records
    by_id = {s["id"]: s for s in state.get("sentences") or [] if s.get("id")}
    asked = open_questions(state)
    out: Dict[str, Tuple[str, str]] = {}
    for label, rec in (state.get("records") or {}).items():
        got = issue_records.line_reason(rec, by_id, asked)
        if got:
            out[label] = got
    for key, target in (sanctions or {}).items():
        if target is None:
            out[_label(key)] = (issue_records.LINE_NOT_TREATED, C.LINE_REASON_SANCTION)
    for label, topic in (beyond or {}).items():         # v60k (single shot): the cap's, with its topic
        out[label] = (issue_records.LINE_NOT_TREATED, C.fill(C.LINE_REASON_CAP, topic=topic, cap=cap)
                      or C.LINE_REASON_NOT_WRITTEN)
    return out


# v60i (2): the articles the counterparty relies on -- a line of the other side (the client's «controparte» line, or
# another party's own) anchored on the article (its rubrica; else its topic's anchor), and the action gate's adverse
# ones -- none the client's own lines rest on too
_ANCHORED_ART_RE = re.compile(r"^art\.\s+(\S+)\s+(.+?):\s")


def counterparty_articles(state: Dict[str, Any]) -> Dict[str, str]:
    """{"N code": why} of the articles the other side relies on; {} while the client is not known."""
    from . import case_read
    analysis = state.get("analysis") or {}
    client = str((state.get("client") or {}).get("name") or analysis.get("parte_assistita")
                 or (state.get("answered") or {}).get("parte_assistita") or "").strip()
    if not client:
        return {}
    theirs: Dict[str, str] = {}
    mine: set = set()
    for t in state.get("topics") or []:
        for h in t.get("hypotheses") or []:
            party, line = str(h.get("party") or ""), str(h.get("line") or "")
            own = case_read.same_party(party, client) if party else True
            m = _ANCHORED_ART_RE.match(str(h.get("anchored_by") or ""))
            label = f"{m.group(1)} {m.group(2)}" if m else str(t.get("anchor") or "")
            if not label:
                continue
            if (own and line == "controparte") or (not own and line in ("principale", "subordinata")):
                theirs.setdefault(label, " ".join(str(h.get("claim") or h.get("anchor_institution") or
                                                      "linea della controparte").split())[:160])
            elif own and line in ("principale", "subordinata"):
                mine.add(label)
    for label in (analysis.get("action_gate") or {}):
        theirs.setdefault(str(label), "azione che spetta alla controparte")
    return {k: v for k, v in theirs.items() if k not in mine}


def role_verdicts(state: Dict[str, Any]) -> Dict[str, Any]:
    """v60i (2): each record marked with the counterparty's reliance (rec["relied_by"]), its verdict again
    (issue_records.verdict: «argomento del cliente» for such an article whose elements fail). Logged; the log."""
    from . import issue_records
    records = state.get("records") or {}
    relied = counterparty_articles(state)
    for label, rec in records.items():
        if label in relied:
            rec["relied_by"] = relied[label]
        else:
            rec.pop("relied_by", None)
        rec["verdict"] = issue_records.verdict(rec)
    out = {"relied": relied, "client_arguments": [label for label, rec in records.items()
                                                  if rec.get("verdict") == issue_records.CLIENT_ARGUMENT]}
    _log("role_verdicts", out)
    return out


def check_context(state: Dict[str, Any], rows: Sequence[Dict[str, Any]], refs: Sequence[tuple],
                  glossed: Sequence[tuple], allowed: Sequence[tuple], documents: Sequence[Dict[str, Any]], case: str,
                  rules: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """v60g (e): what record_check reads -- the records, the articles the answer may cite (the topic's, the computed
    material's: _allowed, and those the case names), the ones the writer applies, each article's text, rubrica and
    commi, the client."""
    from . import citations, defensive_generation as dg
    texts, rubriche, names = {}, {}, {}
    for r in rows:
        if r.get("_article"):
            s = r.get("s") or {}
            label = citations.label(r["_article"])
            texts.setdefault(label, str(s.get("plain_text") or ""))
            rubriche.setdefault(label, str(s.get("titolo_articolo") or ""))
            if r.get("_first_read"):
                names[label] = f"{names.get(label, '')} {r['_first_read']}".strip()
    for label, rec in (state.get("records") or {}).items():      # the institution its topic is named after
        if rec.get("topic"):
            names[label] = f"{names.get(label, '')} {rec['topic']}".strip()
    commi = {citations.label(rule["ref"]): {k for k, _ in rule.get("commi") or [] if isinstance(k, int)}
             for rule in rules or []}
    allowed_labels = {citations.label(r) for r in list(allowed) + list(refs) + list(dg._refs_in(case or "")) if len(r) == 2}
    return {"records": state.get("records") or {}, "sentences": state.get("sentences") or [],
            "open_questions": open_questions(state), "single_shot": single_mode(state), "allowed": allowed_labels,
            "glossed": {citations.label(r) for r in glossed}, "texts": texts, "rubriche": rubriche, "names": names,
            "commi": commi,
            "documents": list(documents or []) + ref_documents(list(refs) + list(glossed), rows),
            "client": str((state.get("client") or {}).get("name") or ""), "parties": party_names_of(state),   # v60q
            "skip": {ANSWER_TITLES[3], OTHER_PROFILES_TITLE}, "norms_title": ANSWER_TITLES[1],
            "numbering": list(glossed)}


def _template_re(template: str, slots: Dict[str, str]) -> "re.Pattern":
    """A copy template as a line pattern: its words escaped, each slot its own pattern (slots), anchored."""
    out, at = "", 0
    for m in re.finditer(r"\{(\w+)\}", template):
        out += re.escape(template[at:m.start()]) + slots.get(m.group(1), r".+?")
        at = m.end()
    return re.compile("^" + out + re.escape(template[at:]) + "$")


_COMMA_SLOT = r"(?:co\. \S+|" + re.escape(C.COMMA_LABEL_ONLY) + ")"
_ROW_LINE_RES = [_template_re(t, {"comma": _COMMA_SLOT, "requisite": ".+", "status": ".+", "number": r"\S+",
                                  "code": ".+?", "rubrica": ".+"})
                 for t in (C.REQUISITE_LINE, C.COMMA_LINE, C.SANCTION_LINE_RUBRICA, C.SANCTION_LINE) if t]
_SANCTION_ART_RE = re.compile(r"art\.\s*(\S+)\s+(.+?)(?:\s+—\s+.*)?\.$")


ASSERTING_TITLES = (ANSWER_TITLES[0], ANSWER_TITLES[4])     # v60t (3): «Risposta e strategia», «Conclusioni»


def asserted_only(draft: str, state: Dict[str, Any], log: Optional[List[Dict[str, Any]]] = None) -> str:
    """v60s (4): the answer's opening and closing sections (ASSERTING_TITLES -- v60t: by title, the numbers shift when
    «Obiezioni e repliche» is empty; never «Profili processuali», whose lines the code writes) assert only the articles
    whose record allows it (issue_records.assertable: «applicabile», or «da provare» with a proven element); a sentence
    naming another article with a record gets «(da valutare: art. N …)» (TO_EVALUATE_MARK) unless it says so already.
    Outside the rule: an article with no record (the computed material's, the procedure's), an «argomento del cliente»
    (the client's argument against it) and (v60t) the main line's anchor (state["main_anchor"]), always written with its
    own status."""
    from . import answer_ledger, citations, issue_records, record_check
    from .plan_then_write import _sentence_spans
    records = state.get("records") or {}
    if not records:
        return draft
    documents = ref_documents([_parse_label(k) for k in records])
    main = str(state.get("main_anchor") or "")
    out = []
    for title, text in answer_ledger.sections_of(draft):
        if title not in ASSERTING_TITLES or not (text or "").strip():
            out.append((title, text))
            continue
        marks: Dict[Tuple[int, int], str] = {}
        for s, e in _sentence_spans(text):
            sentence = text[s:e]
            if "da valutare" in sentence.lower():
                continue
            labels = list(dict.fromkeys(citations.label(r) for r in citations.refs_in(sentence, documents)))
            weak = [x for x in labels if x in records and x != main and not issue_records.assertable(records[x])
                    and (records[x].get("verdict") or issue_records.verdict(records[x])) != issue_records.CLIENT_ARGUMENT]
            if not weak:
                continue
            items = C.LIST_COMMA.join(C.fill(C.TO_EVALUATE_ITEM, number=_parse_label(x)[0],
                                             code=_parse_label(x)[1]) or x for x in weak)
            marks[(s, e)] = record_check._mark(sentence, C.fill(C.TO_EVALUATE_MARK, articles=items) or "")
            if log is not None:
                log.append({"section": title, "articles": weak, "sentence": " ".join(sentence.split())[:200]})
        for (s, e) in sorted(marks, reverse=True):
            text = text[:s] + marks[(s, e)] + text[e:]
        out.append((title, text))
    return answer_ledger.rejoin(draft, out)


_LABEL_WORDS = {"art", "artt", "articolo", "articoli", "c", "p", "a", "codice", "civile", "penale", "procedura",
                "di", "del", "della", "dei", "co", "comma", "e", "ed"}


def _bare_article_line(line: str, heading: str, records: Dict[str, Dict[str, Any]]) -> bool:
    """v60s (5): a line of a §2 block that is an article's label alone -- «art. 1062 c.c.», «[1] Art. 1062 c.c. —
    Destinazione del padre di famiglia», a «[N]» mark alone: no word beyond the label's, its code's and a rubrica's
    (the block's heading's, the records')."""
    st = line.strip()
    if not st:
        return False
    if re.fullmatch(r"\[\d{1,2}\]", st):
        return True
    if not re.match(r"^\W*(?:\[\d{1,2}\]\s*)?\W*art(?:icol[oi]|t)?\.?\s*\d", st, re.IGNORECASE):
        return False
    rubriche = " ".join([heading.split(" — ", 1)[1] if " — " in heading else ""] +
                        [str(r.get("rubrica") or "") for r in records.values()])
    allowed = _LABEL_WORDS | set(re.findall(r"[a-zà-ü’']+", rubriche.lower()))
    return all(w in allowed for w in re.findall(r"[a-zà-ü’']+", st.lower()))


def ledger_out(draft: str, state: Dict[str, Any], log: Optional[Dict[str, Any]] = None
               ) -> Tuple[str, List[Dict[str, Any]]]:
    """v60r (1, single shot): the record's rows never in the body. §2's status lines (REQUISITE_LINE, COMMA_LINE), its
    sanction lines and its closing lines naming articles («Altre norme del tema non trattate», «Norme considerate e non
    applicabili») leave the text, and so does a block left with its heading alone; each block keeps its heading, ONE
    status sentence (ARTICLE_STATUS, the record's verdict) and the writer's paragraphs. (the text, the rows for the
    article citations: per article one row {article, comma None, requisite None, status (its verdict or its line),
    reason, in_body}, then one per element {article, comma, requisite, status (issue_records.element_status: «da provare»
    for one no fact touches, v60r 6), supporting_sentence, contradicting_sentence}.) log: what moved."""
    from . import answer_ledger, issue_records, rule_quote
    log = log if log is not None else {}
    records = state.get("records") or {}
    said = {x["id"]: x.get("text") or "" for x in state.get("sentences") or [] if x.get("id")}
    sections = answer_ledger.sections_of(draft)
    if ANSWER_TITLES[1] not in dict(sections):
        return draft, []
    articles: Dict[str, Dict[str, Any]] = {}          # label -> {"in_body", "line", "reason"} in order of appearance

    def seen(label: str, **info) -> None:
        articles.setdefault(label, {"in_body": False, "line": "", "reason": ""}).update(
            {k: v for k, v in info.items() if v or k == "in_body"})

    def norms(text: str) -> str:
        lines = text.split("\n")
        pre: List[str] = []
        blocks: List[Dict[str, Any]] = []
        i = 0
        while i < len(lines):
            line, st = lines[i], lines[i].strip()
            h = _NORM_HEADING_RE.match(st)
            if h:
                label = f"{h.group(1)} {_code_label(h.group(2).strip())}"
                seen(label)                     # the articles in the order §2 gives them
                blocks.append({"label": label, "heading": line, "body": [], "rows": 0})
                i += 1
                while i < len(lines) and lines[i].strip() and any(r.match(lines[i].strip()) for r in _ROW_LINE_RES):
                    m = _SANCTION_ART_RE.search(lines[i].strip()) if "Sanzione" in lines[i] else None
                    if m:
                        seen(f"{m.group(1)} {_code_label(m.group(2).strip())}", line="sanzione",
                             reason=f"sanziona l'art. {label}")
                    blocks[-1]["rows"] += 1
                    i += 1
                continue
            closing = _OTHER_NORMS_RE.match(st) or _NOT_APPLICABLE_RE.match(st)
            if closing:
                kind = issue_records.LINE_NOT_TREATED if _OTHER_NORMS_RE.match(st) else issue_records.LINE_NOT_APPLICABLE
                for number, code, reason in _line_items(closing.group(1)):
                    if number:
                        seen(f"{number} {_code_label(code)}", line=kind, reason=reason)
                log.setdefault("closing_lines", []).append(kind)
                i += 1
                continue
            (blocks[-1]["body"] if blocks else pre).append(line)
            i += 1
        out = ["\n".join(pre).strip("\n")] if "\n".join(pre).strip() else []
        main = str(state.get("main_anchor") or "")
        if main and main in records and main not in {b["label"] for b in blocks}:
            # v60t (2): the main line's anchor is always written, with its honest status -- first, as the main line
            rec = records[main]
            number, code = _parse_label(main)
            heading = C.fill(C.NORM_HEADING_RUBRICA, number=number, code=code,
                             rubrica=rule_quote.clean_rubrica(rec.get("rubrica") or "").rstrip(" .")) or \
                C.fill(C.NORM_HEADING, number=number, code=code)
            blocks.insert(0, {"label": main, "heading": heading, "body": [], "rows": 0})
            rest = {k: v for k, v in articles.items() if k != main}         # first in the citations too, its line gone
            articles.clear()
            seen(main)
            articles.update(rest)
            log["main_anchor_added"] = main
        for b in blocks:
            kept = [x for x in b["body"] if not _bare_article_line(x, b["heading"], records)]      # v60s (5)
            if len(kept) < len(b["body"]):
                log.setdefault("bare_lines", {})[b["label"]] = [" ".join(x.split()) for x in b["body"] if x not in kept]
            body = re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip("\n")
            if not body.strip() and b["label"] == main and main in records:      # v60t (2): heading and status only
                seen(main, in_body=True)
                rec = records[main]
                reqs = rec.get("requisites") or []
                status = C.fill(C.ARTICLE_STATUS, verdict=rec.get("verdict") or issue_records.verdict(rec),
                                proved=str(issue_records.proven(rec)), total=str(len(reqs))) if reqs else None
                out.append(b["heading"] + (f"\n{status}" if status else ""))
                continue
            if not body.strip():            # a heading alone: the article is a row of the citations only
                seen(b["label"], line=issue_records.LINE_NOT_TREATED, reason=C.LINE_REASON_NOT_WRITTEN)
                log.setdefault("bare", []).append(b["label"])
                continue
            seen(b["label"], in_body=True)
            rec = records.get(b["label"]) or {}
            reqs = rec.get("requisites") or []
            status = C.fill(C.ARTICLE_STATUS, verdict=rec.get("verdict") or issue_records.verdict(rec),
                            proved=str(sum(1 for r in reqs if r.get("status") == PROVATO)), total=str(len(reqs))) \
                if reqs else None
            out.append(b["heading"] + (f"\n{status}" if status else "") + "\n\n" + body)
            log.setdefault("status_lines", {})[b["label"]] = b["rows"]
        return "\n\n".join(out)
    new = [(t, (x[:len(x) - len(x.lstrip("\n"))] + norms(x.strip("\n")) + x[len(x.rstrip("\n")):])
            if t == ANSWER_TITLES[1] else x) for t, x in sections]
    rows: List[Dict[str, Any]] = []
    for label, info in articles.items():
        rec = records.get(label) or {}
        verdict = (rec.get("verdict") or issue_records.verdict(rec)) if rec else ""
        rows.append({"article": label, "comma": None, "requisite": None,
                     "status": verdict if info["in_body"] else (info["line"] or verdict),
                     "reason": info["reason"] if not info["in_body"] else "", "in_body": info["in_body"]})
        for r in rec.get("requisites") or []:
            sup = next((said[x] for x in r.get("supporting") or [] if said.get(x)), None)
            con = next((said[x] for x in r.get("contradicting") or [] if said.get(x)), None)
            rows.append({"article": label, "comma": r.get("comma"), "requisite": r.get("text") or None,
                         "status": issue_records.element_status(rec, r), "supporting_sentence": sup,
                         "contradicting_sentence": con})
    log["articles"] = {k: ("body" if v["in_body"] else v["line"]) for k, v in articles.items()}
    return answer_ledger.rejoin(draft, new), rows


def folded_norms(draft: str, verdicts: Optional[Dict[str, str]], log: List[Dict[str, Any]],
                 lines: Optional[Dict[str, Tuple[str, str]]] = None) -> str:
    """v60g: after the check, §2's blocks left bare named in its closing lines (norms_final), the section's spacing
    kept. v60i (1): lines, the record's one-line articles with their reasons."""
    from . import answer_ledger
    sections = answer_ledger.sections_of(draft)
    out = []
    for title, text in sections:
        if title == ANSWER_TITLES[1]:
            lead = text[:len(text) - len(text.lstrip("\n"))]
            trail = text[len(text.rstrip("\n")):]
            text = lead + norms_final(text.strip("\n"), log, verdicts, lines) + trail
        out.append((title, text))
    return answer_ledger.rejoin(draft, out)


BUCKET_WRITTEN, BUCKET_NOT_APPLICABLE, BUCKET_NOT_TREATED, BUCKET_CAP = (
    "scritto", "non applicabile", "non trattato", "oltre il limite")
REASON_UNEARNED = "nessuna applicazione chiesta al redattore (earned_refs)"
REASON_UNSUPPORTED = "nessuna applicazione chiesta al redattore: il record non ha fatti a sostegno"
REASON_NOT_WRITTEN = "il redattore non l'ha scritto"
REASON_LOST = "perso senza rimozione registrata"


def bucket_articles(refs: Sequence[Tuple[str, str]], glossed: Sequence[Tuple[str, str]], draft: str, ledger,
                    cap_dropped: Sequence[Tuple[str, str]] = (), rows: Sequence[Dict[str, Any]] = (),
                    records: Optional[Dict[str, Dict[str, Any]]] = None,
                    sanctions: Optional[Dict[Tuple[str, str], Optional[tuple]]] = None,
                    cap: int = TOPIC_ARTICLES) -> Dict[str, Dict[str, Any]]:
    """v60f: each planned article in exactly one bucket -- scritto (its §2 block has a paragraph), non applicabile (the
    «Norme considerate e non applicabili» line), non trattato with the reason (not asked of the writer, not written by
    it, or removed: the first step that removed its sentences, from the ledger); the cap's drops «oltre il limite».
    v60g (records): «non applicabile» is the record's verdict, with its ground; a removal's reason is the check's."""
    from . import answer_ledger, citations, issue_records
    docs = ref_documents(list(refs) + list(glossed), rows)
    norms = dict(answer_ledger.sections_of(draft)).get(ANSWER_TITLES[1], "")
    na_prefix = C.NOT_APPLICABLE_LINE.split("{", 1)[0]
    nt_prefix = C.OTHER_NORMS_LINE.split("{", 1)[0]
    blocks: Dict[str, str] = {}
    na, nt, current = set(), set(), None
    for line in norms.splitlines():
        h = citations.heading(line)
        if h:
            current = f"{h[0]} {h[1]}"
            blocks.setdefault(current, "")
            continue
        if line.startswith(na_prefix):
            na |= {citations.label(r) for r in citations.refs_in(line, docs)}
            current = None
        elif line.startswith(nt_prefix):
            nt |= {citations.label(r) for r in citations.refs_in(line, docs)}
            current = None
        elif current and line.strip() and not line.lstrip().startswith("- "):
            blocks[current] += line + "\n"
    asked = {citations.label(r) for r in glossed}
    out: Dict[str, Dict[str, Any]] = {}
    for ref in refs:
        label = citations.label(ref)
        if label in out:
            continue
        removals = ledger.removals(label) if ledger is not None else []
        rec = (records or {}).get(label) or {}
        verdict = (rec.get("verdict") or issue_records.verdict(rec)) if rec else ""
        if sanctions and _key(ref) in sanctions:      # v60h (6): a line under its obligation's block, or «non trattate»
            target = sanctions[_key(ref)]
            if target and blocks.get(citations.label(target)) is not None:
                entry = {"bucket": BUCKET_WRITTEN, "reason": f"sanzione: riga sotto art. {citations.label(target)}"}
            else:
                entry = {"bucket": BUCKET_NOT_TREATED, "reason": "sanzione: la norma di obbligo che sanziona non è "
                         "trattata", "where": "elenco «non trattate»" if label in nt else "assente dal §2"}
            entry["removed_sentences"] = len(removals)
            out[label] = entry
            continue
        if records is not None and verdict == issue_records.NOT_APPLICABLE:
            entry = {"bucket": BUCKET_NOT_APPLICABLE, "reason": issue_records.verdict_reason(rec) or "esito del record",
                     "written": bool(blocks.get(label, "").strip())}
        elif records is not None and verdict == issue_records.CLIENT_ARGUMENT and blocks.get(label, "").strip():
            entry = {"bucket": BUCKET_WRITTEN, "reason": f"argomento del cliente: la controparte vi si fonda "
                     f"({rec.get('relied_by') or ''}), ma {issue_records.verdict_reason(rec)}"}
        elif records is None and label in na:
            entry = {"bucket": BUCKET_NOT_APPLICABLE, "reason": "la glossa dichiara la norma non applicabile"}
        elif blocks.get(label, "").strip():
            entry = {"bucket": BUCKET_WRITTEN, "reason": ""}
        else:
            where = ("elenco «non trattate»" if label in nt else "solo intestazione e righe di stato" if label in blocks
                     else "assente dal §2")
            if label not in asked:
                reason = REASON_UNEARNED if records is None else REASON_UNSUPPORTED
            elif removals:
                reason = f"rimosso da {removals[0]['step']}" + (f": {removals[0]['reason']}"
                                                                if removals[0].get("reason") and
                                                                removals[0]["reason"] != removals[0]["step"] else "")
            elif ledger is not None and not ledger.written(label):
                reason = REASON_NOT_WRITTEN
            else:
                reason = REASON_LOST
            entry = {"bucket": BUCKET_NOT_TREATED, "reason": reason, "where": where}
        entry["removed_sentences"] = len(removals)
        out[label] = entry
    for ref in cap_dropped:
        label = citations.label(ref)
        out.setdefault(label, {"bucket": BUCKET_CAP, "reason": f"limite di {cap} articoli, ordinati per grado "
                               "e sostegno degli elementi", "removed_sentences": 0})
    return out


def record_answer(state: Dict[str, Any], buckets: Dict[str, Dict[str, Any]], ledger, unaccounted: Sequence[str],
                  numbers: Sequence[int]) -> None:
    """v60f: each article's bucket and removals into its record (record["answer"]), the ledger into the state; logged.
    An article lost with no recorded removal, or a sentence missing with none, is logged as an invariant breach."""
    records = state.setdefault("records", {})
    for label, entry in buckets.items():
        rec = records.setdefault(label, {"article": label, "rubrica": "", "topic": "", "grade": None, "issue": "",
                                         "kept": [], "dropped": None, "requisites": [], "linked": "", "answer": {}})
        rec["answer"] = {**entry, "version": state.get("version"), "topics": list(numbers),
                         "removals": [{k: e[k] for k in ("step", "sentence")} for e in ledger.removals(label)][:5]}
    log = ledger.as_log()
    unreasoned = ledger.unreasoned()
    state["ledger"] = {"topics": list(numbers), "steps": log["steps"], "defer": log["defer"],
                       "removed_by_step": log["removed_by_step"], "proposed_by_step": log["proposed_by_step"],
                       "removed_by_reason": log["removed_by_reason"], "entries": log["entries"][:300],
                       "unaccounted": list(unaccounted), "unreasoned": unreasoned[:20]}
    _log("answer_buckets", {label: [e["bucket"], e.get("reason") or ""] for label, e in buckets.items()})
    _log("answer_ledger", {"defer": log["defer"], "removed_by_step": log["removed_by_step"],
                           "proposed_by_step": log["proposed_by_step"], "removed_by_reason": log["removed_by_reason"],
                           "entries": len(log["entries"]), "unaccounted": list(unaccounted)[:10],
                           "unreasoned": len(unreasoned),
                           "lost": [label for label, e in buckets.items() if e.get("reason") == REASON_LOST]})


def case_text(state: Dict[str, Any]) -> str:
    """The case as the writer reads it: the message, then the client's precisazioni given in the rounds."""
    from . import defensive_generation as dg
    case = dg._case_text(state.get("text") or "", state.get("request") or "")
    if state.get("additions"):
        case += "\n\nPRECISAZIONI DEL CLIENTE (fornite dopo il caso):\n" + "\n".join(f"- {a}" for a in state["additions"])
    return case


def _works(facts: Dict[str, Any]) -> List[Dict[str, Any]]:
    from . import fact_signals
    return [w for w in facts.get("works_or_signs") or [] if not re.search(fact_signals._CLOSURE_USE_RE, _low(w.get("what")))]


def _split_entry(facts: Dict[str, Any]) -> Tuple[Optional[int], Dict[str, Any]]:
    """(index, entry) of the property history's owner whose land was split (fact_signals._the_split), else its first
    owner; (None, {}) without a history."""
    from . import fact_signals
    history = facts.get("property_history") or []
    owner = _low(fact_signals._the_split(facts).get("owner"))
    i = next((k for k, h in enumerate(history) if owner and _low(h.get("owner")) == owner), None)
    if i is None:
        i = next((k for k, h in enumerate(history) if h.get("owner")), None)
    return (i, history[i]) if i is not None else (None, {})


_SEPARATION_WORDS = r"divis|divid|separa|frazion|vend|alien|ced[eu]t|don|succe|eredit|lasci"
_EXISTED_WORDS = r"già|esistev|preesist|c['’]er|present|tracciat"
_YEAR_RE = r"\b(1[5-9]\d\d|20\d\d)\b"


def _case_sentences(case_text: str) -> List[str]:
    return [x for x in re.split(r"(?<=[.;!?])\s+|\n+", case_text or "") if x.strip()]


def stated_in_case(name: str, facts: Dict[str, Any], case_text: str) -> bool:
    """Whether the case text itself states the fact a 1062 question asks (a value deduced from the owners' sequence, or
    filled in by the extraction from other words, never counts): work_at_separation -- a sentence naming a work (or an
    opera) with words of existing and of the separation; separation_act -- a sentence naming the split owner with an act
    of separation and a year."""
    sentences = _case_sentences(case_text)
    if name == "work_at_separation":
        if any(w.get("before_separation_stated") for w in _works(facts)):        # answered in a round
            return True
        works = [w for w in _works(facts) if w.get("what")]
        stems = {x for w in works for x in _stems(w["what"])} | {"opera", "strada", "sentie", "passag"}
        return any(_stems(x) & stems and re.search(_EXISTED_WORDS, x, re.IGNORECASE)
                   and re.search(_SEPARATION_WORDS, x, re.IGNORECASE) for x in sentences)
    if name == "separation_act":
        if _split_entry(facts)[1].get("ended_by_stated"):
            return True
        owner = str(_split_entry(facts)[1].get("owner") or "")
        return bool(owner) and any(re.search(rf"\b{re.escape(owner)}\b", x) and re.search(_SEPARATION_WORDS, x, re.IGNORECASE)
                                   and re.search(_YEAR_RE, x) for x in sentences)
    return True


_TEXT_STATED = ("work_at_separation", "separation_act")      # provato only when the case text states it


def unstated_blanked(facts: Dict[str, Any], case_text: str) -> Tuple[Dict[str, Any], List[str]]:
    """(a copy of facts with the _TEXT_STATED fields the extraction filled but the case text does not state left blank,
    the facts so blanked): the works' before_separation, the split owner's ended_by."""
    from . import fact_signals
    out, names = copy.deepcopy(facts or {}), []
    if not stated_in_case("work_at_separation", facts, case_text):
        for w in _works(out):
            if fact_signals._yes(w.get("before_separation")):
                w["before_separation"] = ""
                names.append("work_at_separation")
    if not stated_in_case("separation_act", facts, case_text):
        i, entry = _split_entry(out)
        if entry and _low(entry.get("ended_by")) in fact_signals._SEPARATING:
            out["property_history"][i]["ended_by"] = ""
            names.append("separation_act")
    return out, list(dict.fromkeys(names))


def mark_unstated(analysis: Dict[str, Any]) -> List[str]:
    """v55: a fired fact rule that does not fire once the fields the case text does not state are blank (unstated_blanked)
    rests on the extraction, not on the case: it gets "unstated" (the question families of those fields) and
    "unstated_conditions" (its conditions that fail without them). It is then a rule to verify -- topic_step's
    institutions, the topic line, §2's statuses, the conclusions (fact_signals.conclusions_block leaves it out; the
    answer's material lists it among CONCLUSIONS_TO_VERIFY) -- never a fixed conclusion; the question for the field is
    asked. Marked again on every recompute (an answer states the field). The ids marked."""
    from . import fact_signals
    signals = _signals(analysis)
    fired, facts = signals.get("fired") or [], signals.get("facts")
    for rule in fired:
        rule.pop("unstated", None)
        rule.pop("unstated_conditions", None)
    if not fired or not facts:
        return []
    blanked, names = unstated_blanked(facts, str(analysis.get(CASE_KEY) or ""))
    if not names:
        return []
    terms, areas = signals.get("terms") or [], _areas(analysis)
    still = {r["id"] for r in fact_signals.fire_rules(blanked, terms, areas=areas)}
    explained = {r["id"]: r for r in fact_signals.explain_rules(blanked, terms, areas=areas)}
    marked = []
    for rule in fired:
        if rule["id"] in still:
            continue
        rule["unstated"] = list(dict.fromkeys(_FAMILY[n] for n in names))
        rule["unstated_conditions"] = [c["text"] for c in (explained.get(rule["id"]) or {}).get("conditions") or []
                                       if not c["matched"]]
        marked.append(rule["id"])
    if marked:
        _log("unstated_rules", {"rules": marked, "fields": names})
    return marked


def _defence(facts: Dict[str, Any]) -> Dict[str, Any]:
    """The defence's answers, every facts["defence"] item merged (each answer may add its own)."""
    out: Dict[str, Any] = {}
    for item in facts.get("defence") or []:
        if isinstance(item, dict):
            out.update({k: v for k, v in item.items() if str(v or "").strip()})
    return out


def _fact_false(name: str, facts: Dict[str, Any], signals: Dict[str, Any]) -> bool:
    """The fact a requisite names is known not to hold (a field saying no, a term known not run)."""
    from . import fact_signals
    works = _works(facts)
    if name == "work_visible":
        return bool(works) and all(_low(w.get("visible")) in ("no", "false") for w in works)
    if name == "work_at_separation":
        return bool(works) and all(_low(w.get("before_separation")) in ("no", "false") for w in works)
    if name == "landlocked":
        access = facts.get("access") or []
        return bool(access) and all(_low(a.get("landlocked")) in ("no", "false") for a in access)
    if name == "possession_years":
        return any(str(t.get("id") or "").startswith("usucapione") and t.get("status") == "NON DECORSO"
                   for t in signals.get("terms") or [])
    if name == "possession_peaceful":
        return any(re.search(r"violent|clandestin", _low(p.get("how"))) for p in facts.get("possession") or [])
    if name in _DEFENCE:
        return _low(_defence(facts).get(_DEFENCE[name][0])) in ("no", "false")
    return False


def fact_state(name: str, facts: Dict[str, Any], signals: Dict[str, Any], case_text: str = "") -> str:
    """PROVATO, NON_RISULTA or DA_PROVARE for the fact a requisite names, from the fact fields (and, for _TEXT_STATED,
    only when the case text states it)."""
    if name in _TEXT_STATED and not stated_in_case(name, facts, case_text):
        return DA_PROVARE
    if _fact_holds(name, facts, signals):
        return PROVATO
    return NON_RISULTA if _fact_false(name, facts, signals) else DA_PROVARE


def _fact_holds(name: str, facts: Dict[str, Any], signals: Dict[str, Any]) -> bool:
    """Whether the fact a requisite names holds on the extracted facts and the computed terms (never on the model)."""
    from . import fact_signals
    terms = signals.get("terms") or []
    possession = facts.get("possession") or []
    if name == "work_visible":
        return any(fact_signals._yes(w.get("visible")) for w in _works(facts))
    if name == "work_at_separation":
        return any(fact_signals._yes(w.get("before_separation")) for w in _works(facts))
    if name == "common_owner":
        return bool(fact_signals._p_common_owner(facts, terms, {}))
    if name == "separation_act":            # stated for the owner whose land was split, never deduced from the sequence
        entry = _split_entry(facts)[1]
        return bool(entry) and _low(entry.get("ended_by")) in fact_signals._SEPARATING and \
            entry.get("ended_by_how") != fact_signals.SEQUENCE_DEDUCED
    if name == "landlocked":
        return bool(fact_signals._p_landlocked(facts, terms, {}))
    if name == "access_insufficient":
        return any(re.search(r"insufficient|inadatt|disagevol", _low(a.get("how_reached"))) for a in facts.get("access") or [])
    if name == "possession_years":
        return any(str(t.get("id") or "").startswith("usucapione") and t.get("status") == "DECORSO" for t in terms)
    if name == "possession_peaceful":
        return bool(possession) and all(_low(p.get("how")) and not re.search(r"violent|clandestin", _low(p.get("how")))
                                        for p in possession)
    if name == "title_from_non_owner":
        return bool(fact_signals._title_from_non_owner(facts))
    if name in _DEFENCE:                  # v56: the client's answer (art. 52 co. 2 c.p.)
        return fact_signals._yes(_defence(facts).get(_DEFENCE[name][0]))
    if name in ("elusive_act_date", "judgment_date"):          # v57: a dated event of its kind
        pattern = fact_signals._REISSUED if name == "elusive_act_date" else fact_signals._JUDGMENT
        return any(not _blank(e.get("date")) and fact_signals.parse_date(str(e.get("date")))
                   and re.search(pattern, _low(e.get("what"))) for e in facts.get("events") or [])
    if name in ("fact_date", "act_notice_date", "measure_date"):     # v55: a dated event of its kind
        if name == "act_notice_date" and any(t.get("id") in _TAR_TERMS and t.get("status") in OPEN_TERM_STATUS
                                             for t in terms):
            return False
        pattern = {"fact_date": fact_signals._OFFENCE + "|" + fact_signals._DAMAGE, "act_notice_date": fact_signals._ACT,
                   "measure_date": _MEASURE_RE}[name]
        return any(not _blank(e.get("date")) and fact_signals.parse_date(str(e.get("date")))
                   and re.search(pattern, _low(e.get("what"))) for e in facts.get("events") or [])
    return False


def requisite_status(requisite: str, analysis: Dict[str, Any]) -> str:
    """The requisite's status. v60g (d): while an answer is written, the record's (issue_records.recorded_status: the
    facts' settled with the case's sentences) -- one status source for the status lines, the writer and the checks;
    else, and for a requisite no record holds, the facts' (facts_status)."""
    from . import issue_records
    recorded = issue_records.recorded_status(requisite)
    return recorded if recorded else facts_status(requisite, analysis)


def shown_status(requisite: str, analysis: Dict[str, Any]) -> str:
    """The status a §2 line shows: requisite_status; v60k (single shot, issue_records.points_from active), a «da provare»
    whose missing fact an open question asks reads «da verificare: il caso non dice …» (C.OPEN_POINT); v60l, one whose
    open matter is a legal characterisation, «da verificare» (C.QUALIFIED_STATUS), no question."""
    from . import issue_records
    status = requisite_status(requisite, analysis)
    if status == DA_PROVARE:
        point = issue_records.recorded_point(requisite)
        if point:
            return C.fill(C.OPEN_POINT, fact=point) or status
        if issue_records.recorded_qualified(requisite):
            return C.QUALIFIED_STATUS
    return status


def facts_status(requisite: str, analysis: Dict[str, Any]) -> str:
    """The requisite's status from the facts the requisite's words name (fact_state, each): NON_RISULTA when one is known
    not to hold, PROVATO when every one holds, else DA_PROVARE -- from the extracted facts, the computed terms and the
    case text only, never from the model."""
    facts, signals = _facts(analysis), _signals(analysis)
    named = list(dict.fromkeys(name for pattern, name in _REQUISITE_FACTS if re.search(pattern, _low(requisite))))
    if not named or not facts:
        return DA_PROVARE
    states = [fact_state(n, facts, signals, str((analysis or {}).get(CASE_KEY) or "")) for n in named]
    return NON_RISULTA if NON_RISULTA in states else PROVATO if all(x == PROVATO for x in states) else DA_PROVARE


def excluding_facts(requisite: str, analysis: Dict[str, Any]) -> List[str]:
    """v60m: the facts the requisite's words name that the extracted facts say do not hold (fact_state «non risulta»)."""
    facts, signals = _facts(analysis), _signals(analysis)
    if not facts:
        return []
    named = list(dict.fromkeys(name for pattern, name in _REQUISITE_FACTS if re.search(pattern, _low(requisite))))
    case = str((analysis or {}).get(CASE_KEY) or "")
    return [n for n in named if fact_state(n, facts, signals, case) == NON_RISULTA]


_REAL_ACTION_RE = re.compile(r"servit|confessori|negatori|rivendic|passagg|usucapi|propriet|possess", re.IGNORECASE)


# v58 (v57-l3: h3's servitù objections in a condominium water-damage case): each objection of the copy, the chosen
# topics' sources (their rule, regime or term: topic_step's "source") it belongs to; no such topic, no objection
OBJECTION_SOURCES = {
    "tolleranza": ("destinazione_padre_famiglia", "usucapione_servitu_apparente", "usucapione_ordinaria", "possesso_vizioso",
                   "possessoria_preclusa", "possessoria_in_termine", "ostruzione_in_corso", "fondo_intercluso",
                   "azione_reale_immobiliare", "usucapione", "passaggio_coattivo", "non_uso_servitu"),
    "ritardo": ("destinazione_padre_famiglia", "usucapione_servitu_apparente", "possessoria_preclusa",
                "possessoria_in_termine", "ostruzione_in_corso", "azione_reale_immobiliare", "non_uso_servitu",
                "fondo_intercluso", "passaggio_coattivo"),
    "cortili_giardini": ("fondo_intercluso", "passaggio_coattivo"),
}
# the questions whose objection a topic of these sources raises (OBJECTIONS_BY_QUESTION)
QUESTION_OBJECTION_SOURCES = {"opera_separazione": ("destinazione_padre_famiglia",),
                              "possesso_inizio": ("usucapione_ordinaria", "usucapione_abbreviata", "usucapione_servitu_apparente",
                                                  "usucapione")}


def _topic_sources(topics: Sequence[Dict[str, Any]]) -> set:
    """The rule, regime, term and table ids of the chosen topics and of the topics merged into them."""
    out = set()
    for t in topics:
        for src in [t.get("source")] + list(t.get("merged_sources") or []):
            m = re.match(r"(?:regola(?: da verificare)?|regime|termine|tabella)\s+(\S+)", str(src or ""))
            if m:
                out.add(m.group(1))
    return out


def objections(state: Dict[str, Any], topics: Sequence[Dict[str, Any]], allowed: Sequence[Tuple[str, str]]) -> List[str]:
    """At most MAX_OBJECTIONS objection titles, each the copy's (one with no template is not raised), from the chosen
    topics only (v58): the open questions' (OBJECTIONS_BY_QUESTION) when a topic of their sources is chosen
    (QUESTION_OBJECTION_SOURCES); then tolleranza, ritardo and cortili e giardini, each when a chosen topic's source
    is one of its (OBJECTION_SOURCES). None: no objections section (never another topic's or case's)."""
    sources = _topic_sources(topics)
    out: List[Optional[str]] = []
    for q in state.get("questions") or []:
        if set(QUESTION_OBJECTION_SOURCES.get(q.get("family"), ())) & sources:
            out.append(C.fill(C.OBJECTIONS_BY_QUESTION.get(q.get("family"))))
    for key in ("tolleranza", "ritardo", "cortili_giardini"):
        if set(OBJECTION_SOURCES[key]) & sources:
            out.append(C.fill(C.OBJECTIONS.get(key)))
    return list(dict.fromkeys(o for o in out if o))[:MAX_OBJECTIONS]


_REPLY_MARK_RE = re.compile(r"(?m)^\s*\[(\d{1,2})\]\s*")


def _marked(text: str, count: int) -> Optional[Dict[int, str]]:
    """{N: the writer's text after "[N]"}; with no mark, the paragraphs one per item when they match in number; None
    otherwise."""
    marks = list(_REPLY_MARK_RE.finditer(text or ""))
    if marks:
        return {int(m.group(1)): text[m.end():(marks[i + 1].start() if i + 1 < len(marks) else len(text))].strip()
                for i, m in enumerate(marks)}
    paragraphs = [x.strip() for x in re.split(r"\n\s*\n", text or "") if x.strip()]
    return {i: x for i, x in enumerate(paragraphs, 1)} if paragraphs and len(paragraphs) == count else None


def section_refs(plan: Dict[str, Any], rows: Sequence[Dict[str, Any]], refs: Sequence[Tuple[str, str]]) -> List[tuple]:
    """v56: §2's articles, one block each, in the writer's numbering: the topic's articles with text (_rule_refs), then
    every other article the plan names among the topic's (a heading of its own even with no text read -- v55-p1: the
    writer's 52 and 56 c.p. under art. 42's heading)."""
    from .plan_then_write import _plan_refs
    own = _rule_refs(rows, refs)
    topic = {_key(r) for r in refs}
    seen = {_key(r) for r in own}
    out = list(own)
    for ref in _plan_refs(plan) if (plan or {}).get("questioni") else []:
        ref = tuple(ref)
        if len(ref) == 2 and _key(ref) in topic and _key(ref) not in seen:
            seen.add(_key(ref))
            out.append(ref)
    return out


_ART_RE = re.compile(r"\bart(?:t)?\.\s*(\d+(?:-[a-z]+)?)\s+([a-z.]+(?:\s?[a-z.]+)?)", re.IGNORECASE)


def earned_refs(topics: Sequence[Dict[str, Any]], analysis: Dict[str, Any], case: str,
                planned: Sequence[tuple]) -> List[tuple]:
    """v57 (B: v55-all's 37 misused, all glossed §2 blocks): the planned articles that earned an application to the
    facts -- a chosen topic's matched article(s) (its anchor, a table's or a regime's own), an article a fired rule rests
    on (its requires, its conclusion's), an article the user's text names -- in plan order."""
    from . import citations
    keys = set()
    documents = ref_documents(planned)          # v60e: a code named with spaces read too
    for t in topics:
        keys |= {_key(_parse_label(a)) for a in [t.get("anchor") or ""] + list(t.get("matched") or []) if a}
    signals = _signals(analysis)
    for rule in signals.get("fired") or []:
        keys |= {_key(tuple(r)) for r in rule.get("requires") or [] if len(r) == 2}
        keys |= {_key(r) for r in citations.refs_in(str(rule.get("conclusion") or ""), documents)}
    # v58 (v57-l3: 1122-ter's «non si applica» paragraph): never a rule institution's resolved articles -- retrieval
    # resolved "uso delle parti comuni" to the videosurveillance article -- only what the rule itself names
    keys |= {_key(r) for r in citations.refs_in(case or "", documents)}
    return [r for r in planned if _key(r) in keys]


def record_mode(state: Dict[str, Any]) -> bool:
    """v60g: the answer reads the per-article records (case-first: state["records"]) -- the writer applies them (d),
    one check against them removes (e)."""
    return bool((state or {}).get("records"))


# v60h (6): a sanction article never gets a block -- its rubrica opens with «Sanzioni», or (outside the criminal codes,
# whose offences carry their own pena) its text is penalties only, each citing the article it sanctions. It is one line
# under the planned obligation article it cites (SANCTION_LINE); none planned: the «non trattate» line.
_SANCTION_RUBRICA_RE = re.compile(r"^\W*sanzion", re.IGNORECASE)
_PENALTY_RE = re.compile(r"\b(?:(?:è|sono) punit[oiea]|si applica(?:no)? (?:la|le) sanzion|sanzion\w* amministrativ|"
                         r"sanzion\w* pecuniari|(?:è|sono) soggett\w* alla sanzion|con l['’]arresto|con l['’]ammenda|"
                         r"con la multa|con la reclusione|si applica la pena|(?:è|sono) punibil)", re.IGNORECASE)
_CITED_RE = re.compile(r"\b(?:articol[oi]|artt?\.)\s*((?:\d+(?:-[a-z]+)?(?:\s*,\s*|\s+e\s+|\s+o\s+)?)+)", re.IGNORECASE)
# the obligation a penalty sanctions -- «per la violazione dell'articolo 18», «che viola l'articolo 21», «inosservanza»
# (an offence of another code, «Fuori dai casi previsti dall'art. 2622 … sono puniti», sanctions no obligation)
_VIOLATION_RE = re.compile(r"\bviolazion\w*|\bviola(?:no)?\b|\binosservanz\w*|\bnon osserva|\bnon ottempera|"
                           r"\bcontravvien\w*|\bin difformità", re.IGNORECASE)
PENALTY_SHARE = 0.8


def _row_of(ref: Sequence[str], rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return next((r for r in rows or [] if r.get("_article") and _key(r["_article"]) == _key(ref)), {})


def is_sanction(row: Dict[str, Any]) -> bool:
    """v60h (6): a sanction article -- its rubrica opens with «Sanzioni», or, outside c.p. / c.p.p., at least
    PENALTY_SHARE of its sentences state a penalty, each of them cites an article and one names the violation."""
    from . import rule_quote
    s = row.get("s") or {}
    if _SANCTION_RUBRICA_RE.match(rule_quote.clean_rubrica(str(s.get("titolo_articolo") or ""))):
        return True
    ref = row.get("_article")
    if not ref or _key(ref)[1] in ("c.p.", "c.p.p."):
        return False
    text = rule_quote.clean_text(str(s.get("plain_text") or ""))
    sentences = [x for x in re.split(r"(?<=[.;])\s+", text) if len(x.split()) >= 4]
    penal = [x for x in sentences if _PENALTY_RE.search(x)]
    return bool(sentences) and len(penal) >= PENALTY_SHARE * len(sentences) and all(_CITED_RE.search(x) for x in penal) \
        and any(_VIOLATION_RE.search(x) for x in penal)


def sanctioned(row: Dict[str, Any]) -> List[str]:
    """The article numbers a sanction article's text cites (its own aside), in order."""
    from . import rule_quote
    text = rule_quote.clean_text(str((row.get("s") or {}).get("plain_text") or ""))
    own = str((row.get("_article") or [""])[0])
    out: List[str] = []
    for m in _CITED_RE.finditer(text):
        for n in re.findall(r"\d+(?:-[a-z]+)?", m.group(1)):
            if n != own and n not in out:
                out.append(n)
    return out


def sanction_map(refs: Sequence[tuple], rows: Sequence[Dict[str, Any]]) -> Dict[Tuple[str, str], Optional[tuple]]:
    """{sanction key: the planned obligation article it sanctions (the first it cites, of its code, planned and no
    sanction itself), or None} for the planned sanction articles."""
    sanctions = {_key(r): _row_of(r, rows) for r in refs if is_sanction(_row_of(r, rows))}
    out: Dict[Tuple[str, str], Optional[tuple]] = {}
    for key, row in sanctions.items():
        planned = {_key(r)[0]: r for r in refs if _key(r)[1] == key[1] and _key(r) not in sanctions}
        out[key] = next((planned[n] for n in sanctioned(row) if n in planned), None)
    return out


def glossed_refs(state: Dict[str, Any], topics: Sequence[Dict[str, Any]], analysis: Dict[str, Any], case: str,
                 planned: Sequence[tuple], rows: Sequence[Dict[str, Any]] = ()) -> List[tuple]:
    """The planned articles the writer applies, in plan order: those that earned an application (earned_refs); v60g
    (d), with records, also every one whose record has a requisite proved or a case sentence for it
    (issue_records.support) -- the record, not the topic's anchor alone, says what is argued. v60h: never a «non
    applicabile» article (5) nor, rows given, a sanction article (6: a line under its obligation's block). v60i (1): with
    records, exactly the articles a §2 block is due to (issue_records.block_due: «applicabile», «argomento del
    cliente», or «da provare» with a sentence of the case for it or, v60j, a missing fact an open round-1 question
    asks); an article with no record, as earned."""
    from . import issue_records
    earned = {_key(r) for r in earned_refs(topics, analysis, case, planned)}
    records = (state or {}).get("records") or {}
    if records:
        by_id = {s["id"]: s for s in (state or {}).get("sentences") or [] if s.get("id")}
        asked = open_questions(state)
        due = {_key(r) for r in planned if records.get(issue_records._label(r))
               and issue_records.block_due(records[issue_records._label(r)], by_id, asked)}
        earned = {k for k in earned if not records.get(_label(k))} | due
    if rows:
        earned -= set(sanction_map(planned, rows))
    return [r for r in planned if _key(r) in earned]


def norm_blocks(rules: List[Dict[str, Any]], analysis: Dict[str, Any], rows: Sequence[Dict[str, Any]],
                refs: Optional[Sequence[tuple]] = None, rubriche: Optional[Dict[Tuple[str, str], str]] = None,
                brief: Optional[set] = None, matched: Optional[set] = None) -> List[str]:
    """Per article of the topic, in order (refs: section_refs; default, the rules'): its heading (NORM_HEADING, with its
    rubrica when read), then each selected comma's status lines -- each requisite with its status (REQUISITE_LINE), or
    a comma with no requisite with the comma's own (COMMA_LINE) -- from the fact fields (requisite_status). An article
    with no text read: its heading only. v60 (v59-p3: 585 c.p., the lesioni table's, «non trattate»): a matched article
    (matched: keys -- a chosen topic's anchor, a table's own) with text and no line gets its first selected comma's."""
    from . import article_text, plan_then_write as ptw, rule_quote
    out = []
    brief = brief or set()          # v57: articles that earned no application (earned_refs): no requisite lines
    by_key = {_key(r["ref"]): r for r in rules}
    for ref in refs if refs is not None else [r["ref"] for r in rules]:
        rule = by_key.get(_key(ref)) or {"ref": tuple(ref), "commi": []}
        number, code = str(rule["ref"][0]), ptw._code_key(rule["ref"][1])
        row = next((r for r in rows if r.get("_article") and _key(r["_article"]) == _key(rule["ref"])), {})
        rubrica = str(article_text.heading(row.get("s")) or "") if row else ""
        rubrica = rule_quote.clean_rubrica(rubrica or (rubriche or {}).get(_key(ref)) or "").rstrip(" .")
        lines = [C.fill(C.NORM_HEADING_RUBRICA, number=number, code=code, rubrica=rubrica)
                 or C.fill(C.NORM_HEADING, number=number, code=code)]
        chosen = [(k, t) for k, t in rule.get("commi") or [] if k in (rule.get("selected") or [])] or rule.get("commi") or []
        if rule.get("how") in ("all", "none") and chosen:
            # v57 (v56-p1 589 c.p.: the road-traffic aggravanti): no comma is the role's -- the first, and those naming a
            # fact the case's questions read (1061 co. 2's visible works)
            chosen = chosen[:1] + [(k, t) for k, t in chosen[1:] if _names_in(t)]
        for k, comma in chosen:
            reqs = (rule.get("requisites") or {}).get(k) or [] if _key(rule["ref"]) not in brief else []
            label = C.fill(C.COMMA_LABEL, k=k) if k else C.COMMA_LABEL_ONLY
            if reqs:
                lines += [C.fill(C.REQUISITE_LINE, comma=label, requisite=r, status=shown_status(r, analysis))
                          for r in reqs]
            elif _names_in(comma) or k in (rule.get("referring") or []):
                # v57: a comma line only where the comma names a fact (never «testo: da provare»); v59 (v58-h7 321): or
                # where it extends other articles' penalties -- its line, never a penalty as a requisite
                lines.append(C.fill(C.COMMA_LINE, comma=label, status=shown_status(comma, analysis)))
        if len(lines) == 1 and chosen and _key(rule["ref"]) in (matched or set()):
            k, comma = chosen[0]
            lines.append(C.fill(C.COMMA_LINE, comma=C.fill(C.COMMA_LABEL, k=k) if k else C.COMMA_LABEL_ONLY,
                                status=shown_status(comma, analysis)))
        out.append("\n".join(x for x in lines if x))
    return out


_QUOTE_RE = re.compile(r"«([^»]{20,})»")
_COMMA_LEAD_RE = re.compile(r"^\s*(?:«\s*co\.\s*(\d+)[^»]{0,12}»|\(co\.\s*(\d+)\))\s*[—–-]?\s*")


def _plain(text: str) -> str:
    return " ".join(re.findall(r"[a-zà-ü0-9]+", str(text or "").lower()))


def _paragraphs(text: str) -> List[Tuple[Optional[int], str]]:
    """The writer's §2 as [(its "[N]" mark -- a paragraph after a marked one, in the same mark, carries it too --,
    paragraph)]."""
    out: List[Tuple[Optional[int], str]] = []
    marks = list(_REPLY_MARK_RE.finditer(text or ""))
    pieces = [(None, (text or "")[:marks[0].start()] if marks else (text or ""))]
    pieces += [(int(m.group(1)), text[m.end():(marks[i + 1].start() if i + 1 < len(marks) else len(text))])
               for i, m in enumerate(marks)]
    for mark, body in pieces:
        out += [(mark, x.strip()) for x in re.split(r"\n\s*\n", body or "") if x.strip()]
    return out


def ref_documents(refs: Sequence[tuple], rows: Sequence[Dict[str, Any]] = ()) -> List[Dict[str, Any]]:
    """v60e: the documents the topic's articles belong to, by name (the codes that are no sigla, the rows' documents):
    what citations reads a «art. N <name>» against."""
    from . import citations
    names = [citations.code_key(r[1]) for r in refs if len(r) == 2]
    names += [str((r.get("d") or {}).get("name") or "") for r in rows or []]
    out, seen = [], set()
    for name in names:
        if name and name not in seen and not re.match(r"^c\.\s?[cpd]", name):
            seen.add(name)
            out.append({"id": name, "name": name, "aliases": []})
    return out


def _paragraph_article(paragraph: str, mark: Optional[int], refs: Sequence[tuple],
                       commi: Dict[int, List[str]], documents: Optional[Sequence[Dict[str, Any]]] = None) -> Optional[int]:
    """The block (1-based) a paragraph belongs to: the block whose own comma it quotes («…», v58: first); else its mark's
    article when it cites it; else the first block article it cites; v57 (v56-p1: "Ai sensi del comma 2, …" under art. 42): else the
    block whose comma holds RULE_QUOTE's share (rule_quote.SHARE) of one of its sentences' content words, the most; else
    its mark; None when nothing says."""
    from . import citations, rule_quote
    index = {_key(r): i for i, r in enumerate(refs, 1)}
    documents = documents if documents is not None else ref_documents(refs)
    # v58 (v57-h7: 320's comma, quoted, under 318 because it cites 318 and 319): the article whose own comma holds the
    # quotation first; the cited numbers only then
    for quote in _QUOTE_RE.findall(paragraph):
        q = _plain(quote)
        hits = [i for i, texts in commi.items() if q and any(q in _plain(t) for t in texts)]
        if hits:
            return mark if mark in hits else hits[0]
    # v60e: every citation form, a code named with spaces too (citations)
    cited = list(dict.fromkeys(index[_key(r)] for r in citations.refs_in(paragraph, documents) if _key(r) in index))
    if mark in cited:
        return mark
    if cited:
        return cited[0]
    sentences = [x for x in re.split(r"(?<=[.;:!?])\s+", paragraph) if len(x.split()) >= 5]
    best = max(((rule_quote.share(x, t), i) for x in sentences for i, texts in commi.items() for t in texts),
               default=(0.0, None))
    if best[1] is not None and best[0] >= rule_quote.SHARE:
        return best[1]
    return mark if mark and 1 <= mark <= len(refs) else None


def assemble_norms(text: str, rules: List[Dict[str, Any]], analysis: Dict[str, Any], rows: Sequence[Dict[str, Any]],
                   log: List[Dict[str, Any]], refs: Optional[Sequence[tuple]] = None,
                   rubriche: Optional[Dict[Tuple[str, str], str]] = None,
                   glossed: Optional[Sequence[tuple]] = None, matched: Optional[Sequence[tuple]] = None,
                   keep: bool = False, sanctions: Optional[Dict[Tuple[str, str], Optional[tuple]]] = None,
                   beyond: Sequence[tuple] = ()) -> str:
    """§2: one block per article (norm_blocks: heading and status lines, the code's, never the writer's), then the
    writer's paragraphs on it. v56: each paragraph goes to the article it cites, or whose comma it quotes, else to its
    "[N]" mark's (_paragraph_article) -- never to the previous heading because it follows it; a paragraph nothing places
    goes with the one before it, and before any placed one after the blocks (logged). No marks and as many paragraphs
    as articles, none citing: one per article, in order. v57 (B): glossed, the articles that earned an application
    (earned_refs), in the writer's numbering ("[N]" is the N-th of them); another article's block is its heading and
    comma line only, a paragraph placed on it is dropped (logged). v60: a matched article (a chosen topic's anchor, a
    table's own) the writer skips keeps its block with its comma line (norm_blocks), never «non trattate». keep (v60g e:
    the answer with records): no paragraph is dropped here -- one on an unearned or a textless article stays in its
    block, a comma no article holds stays where it is; the check run last (record_check) decides. sanctions (v60h 6:
    sanction_map): a sanction article has no block -- one line (SANCTION_LINE) under its obligation's block, a paragraph
    placed on it goes to that block; none planned, it is named in the «non trattate» line. beyond (v60k, single shot):
    the articles the cap left out, named in that line too (their reason, with the topic: record_lines)."""
    from . import rule_quote
    refs = list(refs) if refs is not None else [r["ref"] for r in rules]
    sanctions = sanctions or {}
    earned = {_key(r) for r in glossed} if glossed is not None else {_key(r) for r in refs}
    blocks = norm_blocks(rules, analysis, rows, refs, rubriche, {_key(r) for r in refs} - earned,
                         {_key(tuple(r)) for r in matched or []})
    if not blocks:
        return text
    by_key = {_key(r["ref"]): r for r in rules}
    commi = {i: [t for _, t in (by_key.get(_key(ref)) or {}).get("commi") or []] for i, ref in enumerate(refs, 1)}
    position = {_key(r): i for i, r in enumerate(refs, 1)}
    numbering = [position.get(_key(r)) for r in (glossed if glossed is not None else refs)]
    paragraphs = [((numbering[m - 1] if m and m <= len(numbering) else None), p) for m, p in _paragraphs(text)]
    placed: Dict[int, List[str]] = {}
    orphans: List[str] = []
    documents = ref_documents(refs, rows)
    where = [_paragraph_article(p, m, refs, commi, documents) for m, p in paragraphs]
    if not any(m for m, _ in paragraphs) and not any(where) and len(paragraphs) == len(blocks):
        where = list(range(1, len(blocks) + 1))
    last = None
    moved = []
    for (mark, paragraph), i in zip(paragraphs, where):
        i = i or last
        if i is None:
            orphans.append(paragraph)
            continue
        if mark and i != mark:
            moved.append({"mark": mark, "to": _label(refs[i - 1])})
        placed.setdefault(i, []).append(paragraph)
        last = i
    wrong: List[Dict[str, Any]] = []
    placed = commi_moved(placed, refs, by_key, wrong, keep=keep)
    placed = {i: v for i, v in placed.items() if v}
    unearned = {} if keep else {i: placed.pop(i) for i in list(placed) if _key(refs[i - 1]) not in earned}
    # v58 (v57-h5, l3, h7, p2: bare headings, 3 of the 5 misuses left): an article whose text was not read gets no
    # block -- its paragraphs dropped --, nor one whose block would be its heading alone; both are named in one closing
    # line (OTHER_NORMS_LINE)
    textless = {i for i, ref in enumerate(refs, 1) if not (by_key.get(_key(ref)) or {}).get("commi")}
    no_text = {} if keep else {i: placed.pop(i) for i in list(placed) if i in textless}
    under: Dict[int, List[str]] = {}            # v60h (6): the sanction lines, by their obligation's block
    sanction_at = {i for i, ref in enumerate(refs, 1) if _key(ref) in sanctions}
    lined: set = set()
    for i in sorted(sanction_at):
        ref = refs[i - 1]
        target = sanctions.get(_key(ref))
        j = position.get(_key(target)) if target else None
        if j and j not in sanction_at:
            row = _row_of(ref, rows)
            from . import article_text
            rubrica = rule_quote.clean_rubrica(str(article_text.heading(row.get("s")) or "") if row else ""
                                               or (rubriche or {}).get(_key(ref)) or "").rstrip(" .")
            line = (C.fill(C.SANCTION_LINE_RUBRICA, number=ref[0], code=_code_label(ref[1]), rubrica=rubrica)
                    or C.fill(C.SANCTION_LINE, number=ref[0], code=_code_label(ref[1])))
            under.setdefault(j, []).append(line)
            lined.add(i)
            if placed.get(i):
                placed.setdefault(j, []).extend(placed.pop(i))
        elif placed.get(i):
            orphans.extend(placed.pop(i))
    out, untreated = [], []
    for i, b in enumerate(blocks, 1):
        if i in sanction_at:
            if i not in lined:
                untreated.append(refs[i - 1])
            continue
        if under.get(i):
            b = b + "".join(f"\n{x}" for x in under[i] if x)
        if (i in textless and not placed.get(i)) or (len(b.split("\n")) == 1 and not placed.get(i)):
            untreated.append(refs[i - 1])
            continue
        out.append(b + "".join(f"\n\n{p}" for p in placed.get(i, [])))
    log.append({"norms": len(blocks), "applied": sorted(placed), "moved": moved, "unplaced": len(orphans), "commi": wrong,
                "sanctions": {_label(refs[i - 1]): (_label(sanctions[_key(refs[i - 1])])
                                                    if sanctions.get(_key(refs[i - 1])) else None) for i in sanction_at},
                "dropped_unearned": {_label(refs[i - 1]): len(v) for i, v in unearned.items()},
                "dropped_no_text": {_label(refs[i - 1]): len(v) for i, v in no_text.items()},
                "untreated": [_label(r) for r in untreated]})
    have = {_key(r) for r in list(refs) + list(untreated)}
    untreated += [r for r in beyond or () if _key(r) not in have]
    closing = C.fill(C.OTHER_NORMS_LINE, articles=C.OTHER_NORMS_JOIN.join(
        C.fill(C.OTHER_NORMS_ITEM, number=r[0], code=_code_label(r[1])) or _label(r) for r in untreated).rstrip(".")
    ) if untreated else ""
    return "\n\n".join(out + orphans + ([closing] if closing else []))


def commi_moved(placed: Dict[int, List[str]], refs: Sequence[tuple], by_key: Dict[Tuple[str, str], Dict[str, Any]],
                log: List[Dict[str, Any]], keep: bool = False) -> Dict[int, List[str]]:
    """v57 (I: v56-h10 «co. 4» of 31 c.p.a. under 21-septies, which has two commi): a paragraph opening on a comma
    («co. N») its block's article does not have goes to the article whose comma N holds RULE_QUOTE's share of its first
    sentence, else is dropped; placed: {block (1-based): [paragraph]}; v59: also a line opening on it inside a
    paragraph."""
    from . import rule_quote
    numbers = {i: [k for k, _ in (by_key.get(_key(ref)) or {}).get("commi") or []] for i, ref in enumerate(refs, 1)}
    out = {i: list(v) for i, v in placed.items()}
    for i in list(out):
        for paragraph in list(out[i]):
            if not numbers.get(i) or None in numbers[i]:
                continue
            lines = paragraph.split("\n")
            bad = [n for n, line in enumerate(lines) for m in [_COMMA_LEAD_RE.match(line)]
                   if m and int(m.group(1) or m.group(2)) not in numbers[i]]
            if not bad:
                continue
            first = bad[0]
            m = _COMMA_LEAD_RE.match(lines[first])
            k, body = int(m.group(1) or m.group(2)), "\n".join(lines[first:])[m.end():]
            holder = max(((rule_quote.share(body.split(".")[0], t), j) for j, ref in enumerate(refs, 1)
                          for kk, t in (by_key.get(_key(ref)) or {}).get("commi") or [] if kk == k), default=(0.0, None))
            moved = "\n".join(lines[first:])
            kept = "\n".join(lines[:first]).strip()
            if keep and not (holder[1] is not None and holder[0] >= rule_quote.SHARE):
                log.append({"from": _label(refs[i - 1]), "comma": k, "to": None, "kept": True})
                continue        # v60g: no article holds it -- it stays, the check decides
            at = out[i].index(paragraph)
            if kept:
                out[i][at] = kept
            else:
                out[i].remove(paragraph)
            if holder[1] is not None and holder[0] >= rule_quote.SHARE:
                out.setdefault(holder[1], []).append(moved)
            log.append({"from": _label(refs[i - 1]), "comma": k,
                        "to": _label(refs[holder[1] - 1]) if holder[1] and holder[0] >= rule_quote.SHARE else None})
    return out


def _code_label(code: str) -> str:
    from .plan_then_write import _code_key
    return _code_key(code)


def assemble_objections(text: str, titles: Sequence[str], log: List[Dict[str, Any]]) -> str:
    """The objections section: each objection's title from the copy (OBJECTION_TITLE), then the writer's reply to it
    ("[N]"); an objection with no reply is left out, and with no reply at all the section is empty (never a
    placeholder, never the writer's own wording of an objection)."""
    replies = _marked(text, len(titles)) or {}
    kept = [(t, replies.get(i)) for i, t in enumerate(titles, 1) if replies.get(i)]
    log.append({"objections": len(titles), "replied": len(kept)})
    return "\n\n".join(f"{C.fill(C.OBJECTION_TITLE, n=n, text=t)}\n\n{r}" for n, (t, r) in enumerate(kept, 1))


def _offences(topics: Sequence[Dict[str, Any]], rows: Sequence[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """The chosen topics' matched articles that are delitti (rule_quote.offence: c.p. Libro II), each with its rubrica
    (cleaned), for render_computed's «Procedibilità e competenza»."""
    from . import article_text, rule_quote
    rubrica = {_key(r["_article"]): rule_quote.clean_rubrica(article_text.heading(r.get("s"))) for r in rows
               if r.get("_article")}
    out = []
    for t in topics:
        for a in [t.get("anchor") or ""] + list(t.get("matched") or []):
            ref = _parse_label(a) if a else None
            if ref and rule_quote.offence(ref) and (ref[0], rubrica.get(_key(ref), "")) not in out:
                out.append((ref[0], rubrica.get(_key(ref), "")))
    return out


def _matched_refs(topics: Sequence[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """The chosen topics' anchors and matched articles."""
    return list(dict.fromkeys(_parse_label(a) for t in topics for a in [t.get("anchor") or ""]
                              + list(t.get("matched") or []) if a))


def _pool_rubriche(state: Dict[str, Any]) -> Dict[Tuple[str, str], str]:
    return {_key(r["_article"]): str(r.get("rubrica") or "") for r in state.get("pool") or [] if r.get("_article")}


def _fallback_plan(state: Dict[str, Any], analysis: Dict[str, Any], topics: Sequence[Dict[str, Any]],
                   refs: Sequence[Tuple[str, str]]) -> Dict[str, Any]:
    """A plan with no model call when make_plan gives none: one issue per topic, its articles."""
    name = str(analysis.get("parte_assistita") or "").strip()
    return {"parte_assistita": name, "posizione": str(analysis.get("posizione") or ""), "obiettivo": "",
            "questione_centrale": topics[0]["title"] if topics else "", "risposte": [], "subordinate": [],
            "punti_deboli": [], "questioni": [{"titolo": t["title"], "norme": [r for r in _topic_refs(t) if r in refs],
                                              "pronunce": [], "argomento": "", "obiezione": "", "replica": "",
                                              "ancore": []} for t in topics]}


def _rule_refs(rows: Sequence[Dict[str, Any]], refs: Sequence[Tuple[str, str]]) -> List[tuple]:
    """The topic's articles as the rows name them (their own code spelling), in the topic's order."""
    by_key = {_key(r["_article"]): r["_article"] for r in rows if r.get("_article")}
    return [tuple(by_key[_key(ref)]) for ref in refs if _key(ref) in by_key]


CONCLUSIONS_TO_VERIFY = "CONCLUSIONI DA VERIFICARE (dipendono da un fatto che il caso non dice)"
CONCLUSIONS_TO_VERIFY_RULE = ("Non sono accertate: riportale solo come condizionate al fatto indicato («se …»), mai come "
                              "punti fermi.")


def to_verify_block(analysis: Dict[str, Any]) -> str:
    """v55: the fired rules marked unstated (mark_unstated), each its conclusion with the conditions to verify."""
    lines = [f"- {r['conclusion']} Da verificare: {'; '.join(r.get('unstated_conditions') or []) or 'il fatto indicato'}."
             for r in _signals(analysis).get("fired") or [] if r.get("unstated") and r.get("conclusion")]
    return f"{CONCLUSIONS_TO_VERIFY}:\n" + "\n".join(lines) + f"\n{CONCLUSIONS_TO_VERIFY_RULE}" if lines else ""


def relationships_block(state: Dict[str, Any], analysis: Dict[str, Any]) -> str:
    """v60d: for the writer -- the case's sentences stating a relationship (relationships()), and who died
    (deceased()): never a party, never the client."""
    case = case_text(state)
    lines = [f"- {x}" for x in relationships(case)]
    dead = deceased(_facts(analysis), case)
    if dead:
        lines.append(f"- {', '.join(dead)}: deceduto/a (non è una parte)")
    if not lines:
        return ""
    _log("relationships", {"sentences": len(lines) - (1 if dead else 0), "deceased": dead})
    return f"{RELATIONSHIPS_TITLE}:\n" + "\n".join(lines) + f"\n{RELATIONSHIPS_RULE}"


def sections_note() -> str:
    """v60f (P9): the answer's sections, one number and one set of titles, for the writer's system and its jobs."""
    names = [t.lower() for t in ANSWER_TITLES]
    words = {3: "tre", 4: "quattro", 5: "cinque", 6: "sei"}
    return f"La risposta ha {words.get(len(names), len(names))} sezioni ({'; '.join(names)})"


def _jobs(state, plan, analysis, rows, refs, topics, rendered, documents) -> List[Tuple[str, str, str, int]]:
    """The four sections' jobs. v60g (d), with records: §2 reads the SCHEDE (issue_records.writer_block) and the case's
    sentences with their certainty instead of the flat facts, under writer_rules' instruction and rules text; the
    parties to sue under its rule; every job its scope note (P4)."""
    from . import (defensive_generation as dg, issue_records, plan_then_write as ptw, protected_sources, render_computed,
                   rule_quote, writer_rules)
    records = (state.get("records") or {}) if record_mode(state) else {}
    who = plan.get("parte_assistita") or "la parte assistita"
    case = f"CASO:\n{case_text(state)}"
    titles = "; ".join(t["title"] for t in topics)
    shape = (f" {sections_note()} ({ANSWER_TITLES[3]}: già redatti). Tratta solo l'argomento indicato: {titles}."
             + (writer_rules.SCOPE_NOTE if records else ""))

    def block(title: str, body: str) -> str:
        return f"{title}:\n{body}" if str(body or "").strip() else ""

    def material(*parts: str) -> str:
        return protected_sources.with_rule("\n\n".join(p for p in parts if p))
    fixed = block(ptw._CONCLUSIONS_TITLE, f"{ptw._conclusions(analysis)}\n{ptw._CONCLUSIONS_RULE}"
                  if ptw._conclusions(analysis) else "")
    fixed = "\n\n".join(b for b in (fixed, to_verify_block(analysis), unknown_block(state)) if b)
    to_sue = block(ptw._DEFENDANTS_TITLE, f"{ptw._defendants(analysis)}\n"
                   f"{writer_rules.DEFENDANTS_RULE if records else ptw._DEFENDANTS_RULE}"
                   if ptw._defendants(analysis) else "")
    topic_lines = "\n".join(f"- {t['title']} ({_articles_line(t)})" + "".join(
        f"\n  da verificare: {v}" for v in _to_verify(t, state.get("questions") or [])) for t in topics)
    glossed = glossed_refs(state, topics, analysis, case_text(state), section_refs(plan, rows, refs), rows)
    rule_refs = [r for r in _rule_refs(rows, refs) if _key(r) in {_key(x) for x in glossed}]
    references = single_mode(state)          # v60q: single shot, an article is a reference -- never its quoted text
    rules = rule_quote.material_block(plan, rows, analysis, rule_refs, status=lambda r: requisite_status(r, analysis),
                                      title=RULES_TITLE, rule_text=writer_rules.RULES_TEXT_REFERENCES if references
                                      else writer_rules.RULES_TEXT if records else RULES_TEXT, force=True)
    numbered = "\n".join(f"{i}. art. {n} {ptw._code_key(c)}" for i, (n, c) in enumerate(glossed, 1))
    rules = f"{rules}\n\nNORME NELL'ORDINE DEI PARAGRAFI:\n{numbered}" if rules and numbered else rules
    schede = issue_records.writer_block(records, glossed, state.get("sentences") or [], state.get("questions") or []) \
        if records else ""               # v60g (d): the record, as the writer applies it
    facts = issue_records.facts_block(state.get("sentences") or []) if records and state.get("sentences") else \
        block("FATTI DEL CASO", ptw._facts_list(analysis))
    texts = ptw._with_headings(dg._source_block([r for r in rows if r.get("_article") in rule_refs], dg._ARTICLE_CHARS),
                               ptw._headings(rows))
    read_only = render_computed.read_only_block(rendered or "", ANSWER_TITLES[3])       # v60f (P9): its real name
    objected = objections(state, topics, refs)
    gated = render_computed.objections_block(plan)
    lines_of_case = hypotheses_block(topics)          # v60 (A): the first read's lines the chosen topics anchor
    people = relationships_block(state, analysis)     # v60d: the relationships the case states, the deceased
    no_rulings = " Non citare pronunce."
    main_topic = main_line_topic(topics)          # v60s (4): §1's norms; v60t (3): the main line's anchor among them
    main_anchor = issue_records._label(_parse_label(main_topic["anchor"])) if main_topic and main_topic.get("anchor") \
        else ""
    ok = [label for label, rec in records.items() if issue_records.assertable(rec) or label == main_anchor]
    asserted = (C.fill(writer_rules.ASSERTABLE_RULE, articles=", ".join(f"art. {x}" for x in ok)) if ok else
                writer_rules.NONE_ASSERTABLE_RULE) if records else ""
    budget = writer_budget(len(glossed), len(topics)) if single_mode(state) else {
        "lead": 900, "norms": 1700, "conclusions": CONCLUSIONS_TOKENS}
    jobs = [
        (ANSWER_TITLES[0],
         f"SEZIONE: Risposta e strategia. La risposta diretta a {who} sull'argomento, in due-quattro paragrafi: che cosa "
         "può ottenere, con quale azione e su quale norma, e la strategia consigliata (prima la domanda principale, poi "
         "le subordinate). Non illustrare le norme nel dettaglio (sezione successiva)." + asserted + shape + no_rulings,
         material(case, fixed, people, block("ARGOMENTO", topic_lines), lines_of_case,
                  ptw._plan_text(plan, answers=True, rulings=False), read_only), budget["lead"]),
        (ANSWER_TITLES[1],
         (writer_rules.NORMS_INSTRUCTION_REFERENCES if references else writer_rules.NORMS_INSTRUCTION if records else
          f"SEZIONE: Norme applicate ai fatti. Per ciascuna norma di {RULES_TITLE}, nell'ordine, un paragrafo aperto dal "
          "numero della norma nell'elenco tra parentesi quadre su una riga nuova («[1] ...», «[2] ...»), senza titolo: la "
          "regola, riportando il comma tra «…» con il suo numero o restando dentro le sue parole, e come si applica ai "
          "fatti del caso, con il fatto che prova ciascun requisito o la prova da acquisire. Lo stato di ogni requisito è "
          "già scritto sopra il paragrafo: non ripeterlo e non contraddirlo. Solo le norme dell'elenco: le altre norme "
          "dell'argomento hanno già il loro titolo e non vanno commentate.") + shape + no_rulings,
         material(case, fixed, to_sue, people, facts, schede, rules, block("TESTI DELLE NORME", texts)), budget["norms"]),
        (ANSWER_TITLES[2],
         f"SEZIONE: Obiezioni e repliche. Per ciascuna delle OBIEZIONI DELLA CONTROPARTE DA TRATTARE, nell'ordine e senza "
         f"aggiungerne: solo la replica di {who}, in due-tre frasi, sui fatti del caso e sulle norme fornite, aperta dal "
         "numero dell'obiezione tra parentesi quadre su una riga nuova («[1] ...», «[2] ...»); non riscrivere "
         "l'obiezione. Se la replica dipende da un fatto da verificare, dillo e indica la prova." + (f" Le norme di {render_computed.OBJECTIONS_TITLE} solo come argomento "
                                                  "della controparte." if gated else "") + shape + no_rulings,
         material(case, fixed, block("OBIEZIONI DELLA CONTROPARTE DA TRATTARE",
                                     "\n".join(f"{i}. {o}" for i, o in enumerate(objected, 1))), gated, lines_of_case,
                  block("TESTI DELLE NORME", texts)), 1100),
        (ANSWER_TITLES[4],
         f"SEZIONE: Conclusioni. In due paragrafi: l'esito probabile per {who} su questo argomento; i rischi, cioè i "
         "requisiti da provare e le obiezioni più forti; la condotta consigliata e le prove da acquisire." + asserted
         + shape + no_rulings,
         material(fixed, people, lines_of_case, ptw._plan_text(plan, subordinate=True, weak=True, rulings=False), rules,
                  read_only),
         budget["conclusions"]),
    ]
    return [j for j in jobs if j[0] != ANSWER_TITLES[2] or objected]       # v58: no objection of the topic, no §3


# v60k: the single-shot writer's budget, by the articles §2 writes and the topics the other sections cover
NORMS_TOKENS_BASE, NORMS_TOKENS_PER_ARTICLE, NORMS_TOKENS_MAX = 400, 220, 4000
TOPIC_TOKENS, LEAD_TOKENS_MAX, CONCLUSIONS_TOKENS_MAX = 150, 1600, 1800


def writer_budget(articles: int, topics: int) -> Dict[str, int]:
    """{"lead", "norms", "conclusions"}: never below the rounds' (900, 1700, CONCLUSIONS_TOKENS); §2 by its articles,
    §1 and §5 by the topics beyond the first."""
    more = max(0, topics - 1)
    out = {"lead": min(LEAD_TOKENS_MAX, 900 + TOPIC_TOKENS * more),
           "norms": min(NORMS_TOKENS_MAX, max(1700, NORMS_TOKENS_BASE + NORMS_TOKENS_PER_ARTICLE * articles)),
           "conclusions": min(CONCLUSIONS_TOKENS_MAX, CONCLUSIONS_TOKENS + TOPIC_TOKENS * more)}
    _log("writer_budget", {"articles": articles, "topics": topics, **out})
    return out


def term_consistency(sections: List[Tuple[str, str]], context: Dict[str, Any], analysis: Dict[str, Any],
                     log: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """TERM_CONSISTENCY: every computed term is a fixed conclusion. A sentence contradicting one (plan_then_write's term
    checks) is rewritten: to the conditional form when it asserts usucapione on an unknown start, else to the term's own
    line (once per section); a second one in the same section is dropped."""
    from . import plan_then_write as ptw
    terms = context.get("terms") or []
    lines = {t.get("id"): " ".join(str(t.get("text") or "").split()) for t in _signals(analysis).get("terms") or []}
    if not terms:
        return sections
    out = []
    for title, text in sections:
        changes: Dict[Tuple[int, int], str] = {}
        given: set = set()
        for s, e in ptw._sentence_spans(text):
            sentence = text[s:e]
            why = (ptw._term_conflict(sentence, terms) or ptw._other_term(sentence, terms, context.get("case_durations") or [])
                   or (ptw._case_statement(sentence, context) if "possession's start unknown" in
                       ptw._case_statement(sentence, context) else ""))
            if not why:
                continue
            term = next((t for t in terms if (ptw._term_marker(t) and ptw._term_marker(t).search(sentence))), None)
            if "start" in why or (term and term.get("start_unknown") and re.search(r"usucapi", sentence, re.IGNORECASE)):
                now = ptw.conditional_form(sentence)
            elif term and lines.get(term.get("id")) and term.get("id") not in given:
                given.add(term.get("id"))
                line = lines[term["id"]]
                now = C.fill(C.TERM_FIXED, text=(line[:1].lower() + line[1:]).rstrip(".")) or ""
            else:
                now = ""
            changes[(s, e)] = now
            log.append({"section": title, "why": why, "how": "rewritten" if now else "dropped",
                        "sentence": " ".join(sentence.split())[:300], "now": now[:300]})
        for (s, e) in sorted(changes, reverse=True):
            text = text[:s] + changes[(s, e)] + text[e:]
        out.append((title, re.sub(r"[ \t]{2,}", " ", text).strip()))
    return out


# v60q: the statuses are the code's (§2's status lines, its «da verificare: il caso non dice …» and «(da verificare: …)»
# marks) -- the writer's prose never uses their words as ordinary words («non è da provare che…», «è provato che…», «non
# risulta che…»): each becomes a plain word of the same meaning (a closed list), quotations and the open-point formula
# untouched. Run on the writer's sections before any step adds a mark.
_STATUS_WORDS = ((re.compile(r"\bda provare\b", re.IGNORECASE), "da dimostrare"),
                 (re.compile(r"\bprovat([oaie])\b", re.IGNORECASE), r"dimostrat\1"),
                 (re.compile(r"\bnon risulta\b", re.IGNORECASE), "non emerge"),
                 (re.compile(r"\bnon risultano\b", re.IGNORECASE), "non emergono"),
                 (re.compile(r"\bda verificare\b(?!\s*:\s*il caso non (?:dice|indica))", re.IGNORECASE), "da accertare"))
_KEEP_SPANS_RE = re.compile(r"«[^»]*»|“[^”]*”|\(da verificare:[^)]*\)")


def status_words_out(sections: List[Tuple[str, str]], log: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """The writer's sections with the status words it uses as ordinary words replaced (_STATUS_WORDS); each change
    logged with its sentence."""
    from .plan_then_write import _sentence_spans

    def plain(piece: str) -> str:
        for pattern, word in _STATUS_WORDS:
            piece = pattern.sub(lambda m: _same_case(m.group(0), m.expand(word)), piece)
        return piece
    out = []
    for title, text in sections:
        if title == ANSWER_TITLES[3] or not text:
            out.append((title, text))
            continue
        pieces, at = [], 0
        for m in _KEEP_SPANS_RE.finditer(text):
            pieces += [plain(text[at:m.start()]), m.group(0)]
            at = m.end()
        new = "".join(pieces + [plain(text[at:])])
        if new != text:
            before = {text[a:b].strip() for a, b in _sentence_spans(text)}
            for a, b in _sentence_spans(new):
                if new[a:b].strip() not in before:
                    log.append({"section": title, "now": " ".join(new[a:b].split())[:300]})
        out.append((title, new))
    return out


def _same_case(old: str, new: str) -> str:
    return new[:1].upper() + new[1:] if old[:1].isupper() else new


# v57 (G: v56-p1 §3 "- [3] art. 589 c.p.", §2 "art. 43 c.p."): a line the writer gives only to an article's label,
# the list it was given echoed back
_SCAFFOLD_RE = re.compile(r"(?m)^[ \t]*(?:[-•][ \t]*)?(\[\d{1,2}\][ \t]*)?artt?\.[ \t]*\d+(?:-[a-z]+)?[ \t]+[^\n]{0,60}$")
SCAFFOLD_WORDS = 8
_CODE_WORDS = {"del", "della", "dello", "dei", "delle", "sul", "sulla", "di", "e", "n.", "n"}


def scaffolding_out(text: str) -> str:
    """The writer's text without its label-only lines ("[3] art. 589 c.p.": the mark kept, the label gone); a line of
    more than SCAFFOLD_WORDS words, or holding a sentence, is kept."""
    def strip(m):
        line = m.group(0)
        words = re.sub(r"\[\d{1,2}\]|[-•]", " ", line).split()
        label = re.sub(r"^.*?artt?\.\s*\d+(?:-[a-z]+)?\s+", "", line).split()
        codeish = all("." in w or w[:1].isupper() or w[:1].isdigit() or w.lower() in _CODE_WORDS for w in label)
        if len(words) > SCAFFOLD_WORDS or not codeish or re.search(r"[;:!?]\s+\S", line):
            return line
        return m.group(1) or ""
    return re.sub(r"\n{3,}", "\n\n", _SCAFFOLD_RE.sub(strip, text or "")).strip("\n")


_BOTH_WAYS_RE = re.compile(r"in via subordinata,?\s+ma\s+(?:in via\s+)?prioritari\w*", re.IGNORECASE)


_COMMA_OF_RE = re.compile(r"\b(?:comma|co\.)\s*(\d{1,2})\s*,?\s*(?:dell['’]\s*|dello\s+|del\s+)?art(?:icolo|\.)\s*"
                          r"(\d+(?:-[a-z]+)?)", re.IGNORECASE)
_ART_COMMA_RE = re.compile(r"\bart(?:icolo|\.)\s*(\d+(?:-[a-z]+)?)\s*(?:[a-z.]{2,8}\s*)?,?\s*(?:comma|co\.)\s*(\d{1,2})\b",
                           re.IGNORECASE)


def phantom_commi_out(sections: List[Tuple[str, str]], rules: List[Dict[str, Any]], log: List[Dict[str, Any]]
                      ) -> List[Tuple[str, str]]:
    """v58 (v57-h5: «come richiesto dal comma 9 dell'art. 628 c.p.»): a sentence citing a comma an article read here does
    not have (its commi numbered) is dropped whole; the rendered section is never read."""
    from . import plan_then_write as ptw
    commi: Dict[str, set] = {}
    for rule in rules:
        numbers = [k for k, _ in rule.get("commi") or []]
        if numbers and None not in numbers:
            commi.setdefault(str(rule["ref"][0]), set()).update(numbers)
    if not commi:
        return sections
    out = []
    for title, text in sections:
        if title == ANSWER_TITLES[3]:
            out.append((title, text))
            continue
        spans = []
        for s, e in ptw._sentence_spans(text):
            sentence = text[s:e]
            pairs = [(m.group(2), m.group(1)) for m in _COMMA_OF_RE.finditer(sentence)] + \
                    [(m.group(1), m.group(2)) for m in _ART_COMMA_RE.finditer(sentence)]
            if any(n in commi and int(k) not in commi[n] for n, k in pairs):
                spans.append((s, e))
                log.append({"section": title, "sentence": " ".join(sentence.split())[:300]})
        out.append((title, ptw._drop_spans(text, spans) if spans else text))
    return out


_HEADLESS_RE = re.compile(r"^\s*(?:[,;:)\]–—-]|[a-zà-ù])")


def _headless(text: str) -> bool:
    """A sentence cut at its head: a comma, a closing mark, a lower-case word -- an «a)» item or «art. N» aside."""
    return bool(_HEADLESS_RE.match(text) and not re.match(r"\s*(?:[a-zà-ù]\)|artt?\.\s*\d|ex\s+art)", text))


def headless_out(sections: List[Tuple[str, str]], log: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """v58 (v57-l3 §4: «-, che attribuisce al condominio…»): a sentence that starts with a comma, a closing mark or a
    lower-case word -- cut at a removed citation -- is dropped whole, never left headless; a list item's or a line's
    own start is a start. The rendered section is never read."""
    from . import plan_then_write as ptw
    out = []
    for title, text in sections:
        if title == ANSWER_TITLES[3]:
            out.append((title, text))
            continue
        spans = []
        for s, e in ptw._sentence_spans(text):
            sentence = text[s:e]
            line_start = s == 0 or text[s - 1] == "\n" or re.match(r"\s*[-•*]\s", text[text.rfind("\n", 0, s) + 1:s] + " ")
            body = re.sub(r"^\s*[-•*]\s+", "", sentence) if line_start else sentence
            if _headless(body):
                spans.append((s, e))
                log.append({"section": title, "sentence": " ".join(sentence.split())[:300]})
        out.append((title, ptw._drop_spans(text, spans) if spans else text))
    return out


def cosmetic_fixes(sections: List[Tuple[str, str]], titles: Sequence[str], log: List[Dict[str, Any]]
                   ) -> List[Tuple[str, str]]:
    """v57 (K): "in via subordinata, ma prioritariamente" (v56-h3 §1) is "in via principale". v60f (P9): no section name
    is repaired here any more -- the writer is told the answer's real sections (sections_note, read_only_block)."""
    out = []
    for title, text in sections:
        out.append((title, _BOTH_WAYS_RE.sub("in via principale", text or "")))
    return out


def unsupported_out(sections: List[Tuple[str, str]], rules: List[Dict[str, Any]], analysis: Dict[str, Any],
                    documents: List[Dict[str, Any]], log: List[Dict[str, Any]],
                    keep: Sequence[tuple] = ()) -> List[Tuple[str, str]]:
    """UNSUPPORTED_OUT: an article with requisites none of which the facts support has its sentences moved out of the
    answer and the norms sections into OTHER_PROFILES_TITLE (inserted before the conclusions). The chosen topics'
    anchors (keep) stay: the topic asked for is answered, its requisites «da provare». v57 (G): a sentence only stating an
    article's rule (rule_quote.rule_sentence: "L'art. 589 c.p. punisce …") is dropped, never moved."""
    from . import plan_then_write as ptw, rule_quote
    out_refs = []
    kept = {_key(r) for r in keep}
    for rule in rules:
        if _key(rule["ref"]) in kept:
            continue
        reqs = [r for k in rule.get("selected") or [] for r in (rule.get("requisites") or {}).get(k) or []]
        if reqs and not any(requisite_status(r, analysis) == PROVATO for r in reqs):
            out_refs.append(tuple(rule["ref"]))
    if not out_refs:
        return sections
    moved: List[str] = []
    done = []
    for title, text in sections:
        if title not in (ANSWER_TITLES[0], ANSWER_TITLES[1]):
            done.append((title, text))
            continue
        cited = ptw._citing(text, out_refs, documents)
        spans = sorted({span for c in cited for span in ptw._sentence_spans(text) if span[0] <= c[0] < span[1]})
        for s, e in spans:
            if rule_quote.rule_sentence(text[s:e], (), rule_quote.EXTENDED_VERB_RE) or \
                    not scaffolding_out(text[s:e]).strip() or _headless(text[s:e]):
                log.append({"section": title, "dropped": " ".join(text[s:e].split())[:300]})
                continue
            moved.append(" ".join(text[s:e].split()))
            log.append({"section": title, "articles": [_label(r) for c in cited if c[0] in range(s, e) for r in c[2]],
                        "sentence": " ".join(text[s:e].split())[:300]})
        done.append((title, ptw._drop_spans(text, spans) if spans else text))
    if not moved:
        return sections
    labels = ", ".join(f"art. {n} {ptw._code_key(c)}" for n, c in out_refs)
    body = (C.fill(C.OTHER_PROFILES_BODY, articles=labels) + "\n\n"
            + "\n".join(C.fill(C.OTHER_PROFILES_ITEM, sentence=m) for m in dict.fromkeys(moved)))
    at = next((i for i, (t, _) in enumerate(done) if t == ANSWER_TITLES[4]), len(done))
    return done[:at] + [(OTHER_PROFILES_TITLE, body)] + done[at:]


_SAID_PROVED_RE = re.compile(r"\b(?<!da\s)(?<!non\s)(?:(è|sono|risulta|risultano|appare|appaiono)\s+(?:già\s+)?)?"
                             r"(provat[oaie]|dimostrat[oaie]|soddisfatt[oaie]|integrat[oaie])\b", re.IGNORECASE)


def _covers(sentence: str, requisite: str) -> bool:
    """The sentence speaks of the requisite: half its content stems, two at least."""
    words = _stems(requisite)
    return len(words) >= 2 and len(words & _stems(sentence)) * 2 >= len(words)


def strict_status(sections: List[Tuple[str, str]], rules: List[Dict[str, Any]], analysis: Dict[str, Any],
                  log: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """A requisite's status is the facts' (requisite_status), never the writer's: a sentence on a requisite «da provare»
    calling it proved (provato, dimostrato, soddisfatto, integrato) says «da provare» instead."""
    open_reqs = [r for rule in rules for k in rule.get("selected") or [] for r in (rule.get("requisites") or {}).get(k) or []
                 if requisite_status(r, analysis) == DA_PROVARE]
    if not open_reqs:
        return sections
    from . import plan_then_write as ptw
    out = []
    for title, text in sections:
        changes: Dict[Tuple[int, int], str] = {}
        for s, e in ptw._sentence_spans(text):
            sentence = text[s:e]
            if not _SAID_PROVED_RE.search(sentence) or not any(_covers(sentence, r) for r in open_reqs):
                continue
            now = _SAID_PROVED_RE.sub(lambda m: ("sono da provare" if (m.group(1) or "").lower() in ("sono", "risultano",
                                                                                                   "appaiono")
                                                 else "è da provare" if m.group(1) else "da provare"), sentence)
            if now != sentence:
                changes[(s, e)] = now
                log.append({"section": title, "sentence": " ".join(sentence.split())[:300], "now": " ".join(now.split())[:300]})
        for (s, e) in sorted(changes, reverse=True):
            text = text[:s] + changes[(s, e)] + text[e:]
        out.append((title, text))
    return out


_ALREADY_CONDITIONAL_RE = re.compile(r"\b(?:se|qualora|ove|purché|purche|nel caso in cui|a condizione che|sempre che|"
                                     r"subordinat\w*|condizionat\w*|a condizione|in caso di|laddove|ipotesi)\b|"
                                     r"da provare|da verificare|da accertare|occorre provare|va provat|andrà provat|prova|"
                                     r"potrebbe|potrebbero|\bva\w*\s+(?:verificat|accertat)\w*",
                                     re.IGNORECASE)


def marked(sentence: str, points: Sequence[str]) -> Optional[str]:
    """v58: the sentence with the copy's marker (UNKNOWN_MARK, «(da verificare: …)») before its final stop -- once: a
    sentence already marked, or with no final stop of its own (a span cut inside a sentence), None."""
    if "(da verificare:" in sentence or not points:
        return None
    body = sentence.rstrip()
    stop = re.search(r"[.!?]+$", body)
    if not stop:
        return None
    mark = C.fill(C.UNKNOWN_MARK, point=C.LIST_JOIN.join(dict.fromkeys(points)))
    if not mark:
        return None
    if stop.group(0) == "." and _ABBREVIATION_END_RE.search(body):      # «… art. 1062 c.c.»: the code keeps its dot
        return body + mark + "." + sentence[len(body):]
    return body[:stop.start()] + mark + stop.group(0) + sentence[len(body):]


_ABBREVIATION_END_RE = re.compile(r"(?:\b[a-z]\.){2,}$|\b(?:ecc|art|artt|co|lett|n|cfr)\.$", re.IGNORECASE)
_LOWER_FIRST = {"il", "lo", "la", "l", "i", "gli", "le", "un", "uno", "una", "nessuna", "nessun", "ai", "al", "alla", "allo",
                "in", "per", "con", "dal", "dalla", "questo", "questa", "tale", "ne", "si", "non", "è", "anche", "inoltre",
                "pertanto", "quindi", "dunque"}


def _conditional(sentence: str, conditions: Sequence[str]) -> str:
    """The sentence opened by the copy's conditions ("Se ..., "), several joined by CONDITIONS_JOIN."""
    lead = re.match(r"^\s*", sentence).group(0)
    body = sentence[len(lead):]
    first = re.match(r"[^\W\d_]+", body)
    if first and first.group(0).lower() in _LOWER_FIRST:
        body = body[:1].lower() + body[1:]
    parts = [c.strip().rstrip(",").strip() for c in conditions if c.strip()]
    opening = parts[0] + "".join(C.CONDITIONS_JOIN + p[:1].lower() + p[1:] for p in parts[1:])
    return f"{lead}{C.capital(opening)}, {body}"


def conditional_on_unknown(sections: List[Tuple[str, str]], rules: List[Dict[str, Any]], analysis: Dict[str, Any],
                           documents: List[Dict[str, Any]], names: Sequence[str],
                           log: List[Dict[str, Any]], single: bool = False) -> List[Tuple[str, str]]:
    """A requisite «da provare» marks every sentence relying on it: a sentence citing its article (not a rule sentence,
    not already conditional or speaking of the proof) gets the copy's marker with the requisite's question label
    (marked: «(da verificare: …)», once per sentence, before its final stop) for a fact the copy conditions
    (CONDITIONS, by family:article or family). v58 (v57-h3: «Se le opere risultano esistenti alla separazione,» three
    times, once mid-paragraph: «, Questi fatti»): never a prefix."""
    from . import plan_then_write as ptw, rule_quote
    by_article: Dict[Tuple[str, str], List[str]] = {}
    for rule in rules:
        number = str(rule["ref"][0])
        own, general = [], []           # the copy's condition written for this article wins over the families' own
        for k in rule.get("selected") or []:
            for r in (rule.get("requisites") or {}).get(k) or []:
                if requisite_status(r, analysis) != DA_PROVARE:
                    continue
                names = _names_in(r)
                for name in names:
                    family = _FAMILY.get(name)
                    # the separation's condition speaks of the interclusion: only for a requisite that does (art. 1054)
                    if family == "atto_separazione" and "landlocked" not in names:
                        continue
                    point = C.QUESTION_SHORT.get(family) or family      # v58: the marker's words, never a prefix
                    if single:                                          # v60k: «il caso non dice …»
                        point = open_point_label({"template": family, "family": family}) or point
                    if family and C.CONDITIONS.get(f"{family}:{number}"):
                        own.append(point)
                    elif family and C.CONDITIONS.get(family):
                        general.append(point)
        if own or general:
            by_article[_key(rule["ref"])] = list(dict.fromkeys(own or general))
    if not by_article:
        return sections
    out = []
    for title, text in sections:
        changes: Dict[Tuple[int, int], str] = {}
        cited = ptw._citations(text, documents)
        for s, e in ptw._sentence_spans(text):
            sentence = text[s:e]
            if sentence.lstrip().startswith(("**", "#", "[")) or _ALREADY_CONDITIONAL_RE.search(sentence) \
                    or rule_quote.rule_sentence(sentence, names, rule_quote.EXTENDED_VERB_RE):
                continue
            conds = [c for start, _, refs in cited if s <= start < e for r in refs for c in by_article.get(_key(r), [])]
            now = marked(sentence, list(dict.fromkeys(conds))) if conds else None
            if now:
                changes[(s, e)] = now
                log.append({"section": title, "sentence": " ".join(sentence.split())[:300], "now": " ".join(now.split())[:300]})
        for (s, e) in sorted(changes, reverse=True):
            text = text[:s] + changes[(s, e)] + text[e:]
        out.append((title, text))
    return out


# v56 (v55-h3: «fin dal 1900», «non sarebbe dovuta alcuna indennità»): a sentence asserting the answer of a round-1
# question left unanswered, by family -- the possession's start, the separation's act (and art. 1054's consequence), the
# works at the separation, the access, the dates
_YEAR = r"(?:1[5-9]\d\d|20\d\d)"
_DAY = r"\b\d{1,2}(?:°)? (?:gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre) " + _YEAR
ASSERTS = {
    "possesso_inizio": (r"possed\w*|possesso|\busa\w*|utilizz\w*|transit\w*|passaggio|esercit\w*|percorr\w*",
                        rf"\b(?:fin\s+)?dal(?:l['’])?\s*(?:inizio del\s+|primi del\s+)?{_YEAR}\b|\bda (?:oltre |più di |almeno "
                        r"|circa )?(?:\d+|venti|trenta|quaranta|cinquanta|cento) anni\b|ultraventennal\w*|"
                        r"ventennal\w*|vent['’]anni|\boltre un secolo|da (?:sempre|tempo immemorabile)"),
    "atto_separazione": (r"", r"senza (?:alcuna |pagare (?:alcuna )?)?indennit\w*|non (?:è|sarebbe|sarà|era) dovut\w* "
                         r"(?:alcuna )?indennit\w*|(?:separazione|divisione|alienazione|vendita) (?:dei fondi |dei terreni )?"
                         r"(?:è |sono )?(?:avvenut\w*|intervenut\w*|operat\w*)|per effetto (?:della|dell['’]) (?:divisione|"
                         rf"vendita|alienazione)|(?:separat|divis|vendut|alienat)\w* (?:i fondi |i terreni )?nel {_YEAR}"),
    "opera_separazione": (r"", r"(?:esistev\w*|era\w* già|preesist\w*|già (?:presente|esistente|tracciat\w*))[^.;]{0,80}"
                          r"(?:separazion|divisio|alienazion)|(?:al momento della|alla|all['’]atto della) (?:separazione|"
                          r"divisione)[^.;]{0,80}(?:esistev|già|presente|visibil)"),
    "accesso_alternativo": (r"", r"(?:non ha|privo di|priva di|senza) (?:alcun |altro |alcun altro )?(?:accesso|uscita)|"
                            r"(?:è|risulta|sia) (?:totalmente |del tutto )?intercluso"),
    "data_fatto": (r"fatto|aggression\w*|evento|sinistro|reato|accadut\w*|avvenut\w*", rf"{_DAY}|\bnel {_YEAR}\b"),
    "data_messa_in_mora": (r"messa in mora|diffid\w*|contestazion\w*|raccomandat\w*", rf"{_DAY}"),
    "atto_notifica": (r"notific\w*|pubblicat\w*|comunicat\w*|provvediment\w*|ricorso",
                      rf"{_DAY}|(?:è|risulta|sia) (?:ancora )?(?:tempestiv\w*|in termine|nei termini|tardiv\w*)"),
    "data_misura": (r"ordinanz\w*|misura|custodia|arrest\w*|riesame", rf"{_DAY}|(?:è|risulta) (?:ancora )?(?:tempestiv\w*|"
                    r"in termine|tardiv\w*)"),
}


def unanswered(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The round-1 questions the user did not answer (the open ones), each once by family."""
    answered = set(state.get("answered") or {})
    out, seen = [], set()
    for q in state.get("questions") or []:
        if q.get("id") in answered or q.get("family") in seen:
            continue
        seen.add(q.get("family"))
        out.append(q)
    return out


def open_points_list(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """v60k: every open question but the client's -- each its own point (unanswered keeps one per family, and the first
    read's questions are all of one). v60l: never a legal characterisation (qualification); v60q: nor a party's state of
    mind (state_of_mind)."""
    answered = set(state.get("answered") or {})
    client, names = client_of(state), party_names_of(state)
    return [q for q in state.get("questions") or [] if q.get("id") not in answered and q.get("id") != "parte_assistita"
            and not qualification(q) and not state_of_mind(q, client, names)]


def open_point_label(q: Dict[str, Any], names: Sequence[str] = ()) -> str:
    """The open point inside a writer's sentence marker (UNKNOWN_MARK «(da verificare: …)»): «il caso non dice …»."""
    return (C.fill(C.OPEN_POINT, fact=open_point_fact(q, names)) or "").split(": ", 1)[-1]


def open_point_fact(q: Dict[str, Any], names: Sequence[str] = ()) -> str:
    """v60k: the fact an open question asks, as «il caso non dice …» reads it -- its template's (C.OPEN_POINT_FACTS),
    else the first read's own words, else its short label. v60l: lowercase after «il caso non dice» (lowered_start)."""
    fact = C.OPEN_POINT_FACTS.get(str(q.get("template") or q.get("family") or q.get("id") or ""))
    text = " ".join(str(fact or q.get("point") or q.get("short") or q.get("text") or "").split()).rstrip(" .?")
    return lowered_start(text, names)


def lowered_start(text: str, names: Sequence[str] = ()) -> str:
    """v60l: the text with a lowercase first letter -- not when its first word is a party's name (names) or an acronym
    (two capitals or more)."""
    m = re.match(r"[^\W\d_][\w'’.-]*", text or "")
    if not m:
        return text or ""
    word = m.group(0).split("'")[0].split("’")[0]
    own = {w for n in names or () for w in re.findall(r"[^\W\d_]+", str(n))}
    if word in own or (len(word) >= 2 and word.isupper()):
        return text
    return text[:1].lower() + text[1:]


def unknown_block(state: Dict[str, Any]) -> str:
    """v56: the open questions for the writer, each «non noto» (UNKNOWN_TITLE, the copy's UNKNOWN_LINE). v60k (single
    shot): each the fact the case does not state, «da verificare: il caso non dice …», never a question (the copy's
    OPEN_POINTS_WRITER_*). v60l (single shot): the legal characterisations the first read left open, the parere's own
    to settle (the copy's QUALIFICATIONS_WRITER_*). v60q (single shot): the parties' states of mind it left as missing,
    to argue from the facts (STATES_OF_MIND_WRITER_*)."""
    if single_mode(state):
        names = party_names_of(state)
        lines = [f"- {C.fill(C.OPEN_POINT, fact=open_point_fact(q, names))}" for q in open_points_list(state)
                 if open_point_fact(q, names)]
        block = f"{C.OPEN_POINTS_WRITER_TITLE}:\n" + "\n".join(lines) + f"\n{C.OPEN_POINTS_WRITER_RULE}" if lines else ""
        quals = [f"- {lowered_start(' '.join(str(x).split()).rstrip(' .?'), names)}"
                 for x in state.get("qualifications") or [] if str(x).strip()]
        if quals:
            block = "\n\n".join(b for b in (block, f"{C.QUALIFICATIONS_WRITER_TITLE}:\n" + "\n".join(quals)
                                             + f"\n{C.QUALIFICATIONS_WRITER_RULE}") if b)
        minds = [f"- {lowered_start(' '.join(str(x).split()).rstrip(' .?'), names)}"     # v60q
                 for x in state.get("states_of_mind") or [] if str(x).strip()]
        if minds:
            block = "\n\n".join(b for b in (block, f"{C.STATES_OF_MIND_WRITER_TITLE}:\n" + "\n".join(minds)
                                             + f"\n{C.STATES_OF_MIND_WRITER_RULE}") if b)
        return block
    lines = [C.fill(C.UNKNOWN_LINE, point=q.get("short") or q.get("family"), question=q.get("text")) for q in unanswered(state)]
    lines = [x for x in lines if x]
    return f"{C.UNKNOWN_TITLE}:\n" + "\n".join(lines) + f"\n{C.UNKNOWN_RULE}" if lines else ""


def unanswered_out(sections: List[Tuple[str, str]], state: Dict[str, Any], log: List[Dict[str, Any]],
                   names: Sequence[str] = ()) -> List[Tuple[str, str]]:
    """v56: a sentence asserting the answer of an open question (ASSERTS: its subject's words, when the family has them,
    and the assertion) -- not already conditional or speaking of the proof, not a heading -- gets the copy's marker
    (UNKNOWN_MARK, «da verificare: …») before its last stop: never an answer the user did not give. The rendered
    procedural section is never read; a rule sentence (the norm's own requisite, rule_quote.rule_sentence) neither."""
    from . import plan_then_write as ptw, rule_quote
    open_q = [q for q in unanswered(state) if q.get("family") in ASSERTS]
    if not open_q:
        return sections
    out = []
    for title, text in sections:
        if title == ANSWER_TITLES[3]:
            out.append((title, text))
            continue
        changes: Dict[Tuple[int, int], str] = {}
        for s, e in ptw._sentence_spans(text):
            sentence = text[s:e]
            if sentence.lstrip().startswith(("**", "#")) or _ALREADY_CONDITIONAL_RE.search(sentence) \
                    or rule_quote.rule_sentence(sentence, names, rule_quote.EXTENDED_VERB_RE):
                continue
            hit = [q for q in open_q for subject, claim in [ASSERTS[q["family"]]]
                   if re.search(claim, sentence, re.IGNORECASE)
                   and (not subject or re.search(subject, sentence, re.IGNORECASE))]
            if not hit:
                continue
            now = marked(sentence, [open_point_label(q, names) if single_mode(state) else q.get("short") or q["family"]
                                    for q in hit])
            if not now:
                continue
            changes[(s, e)] = now
            log.append({"section": title, "families": [q["family"] for q in hit],
                        "sentence": " ".join(sentence.split())[:300]})
        for (s, e) in sorted(changes, reverse=True):
            text = text[:s] + changes[(s, e)] + text[e:]
        out.append((title, text))
    return out


def citation_check(sections: List[Tuple[str, str]], allowed: Sequence[tuple], documents: List[Dict[str, Any]],
                   log: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """Every citation within the topic set and the computed block: a sentence citing another article goes
    (plan_then_write.remove_citations)."""
    from . import plan_then_write as ptw
    out = []
    for title, text in sections:
        bad = list(dict.fromkeys(r for _, _, refs in ptw._unsupported(text, list(allowed), documents) for r in refs))
        if bad:
            text, entries = ptw.remove_citations(text, bad, documents, title=title)
            log.extend({"section": title, **{k: v for k, v in e.items() if k in ("how", "kind", "removed", "sentence")}}
                       for e in entries)
        out.append((title, text))
    return out


def _allowed(state, analysis, refs, tool_rows, rendered, documents) -> List[tuple]:
    """The topic's articles, the tools' and the terms' articles, the fired rules' conclusions' and the rendered block's."""
    from . import defensive_generation as dg, plan_then_write as ptw
    out = list(refs) + [tuple(r["_article"]) for r in tool_rows if r.get("_article")]
    signals = _signals(analysis)
    texts = [str(t.get("ref") or "") for t in signals.get("terms") or []]
    texts += [str(r.get("conclusion") or "") for r in signals.get("fired") or []]
    texts += [rendered or "", str((signals.get("mediation") or {}).get("ref") or "")]
    for text in texts:
        out += [r for _, _, rs in ptw._citations(text, documents) for r in rs]
        out += dg._refs_in(text)
    return list(dict.fromkeys(tuple(r) for r in out))


def _split_draft(draft: str, titles: Sequence[str]) -> Optional[List[Tuple[str, str]]]:
    parts = re.split(r"(?m)^\*\*\d+\. ([^*\n]+)\*\*\s*$", draft)
    sections = [(parts[i].strip(), parts[i + 1].strip()) for i in range(1, len(parts) - 1, 2)]
    return sections if [t for t, _ in sections] == list(titles) else None


def _join(sections: Sequence[Tuple[str, str]]) -> str:
    return "\n\n".join(f"**{i}. {title}**\n\n{text}" for i, (title, text) in enumerate(sections, 1))


# --- v60: before and after writing ---------------------------------------------------------------------------------

def grade_chosen(state: Dict[str, Any], topics: Sequence[Dict[str, Any]]) -> List[str]:
    """G: the rows round 1 left ungraded in the chosen topics' Sezione or Capo (state["reserve"]) graded now; each graded
    2 or more joins its topic (an article of it, matched) and the session's rows. The labels added."""
    from . import defensive_generation as dg
    units = {t.get("unit") for t in topics if t.get("unit")}
    reserve = _restored_rows(state.get("reserve"))
    mine = [r for r in reserve if r.get("_unit") in units]
    if not mine:
        return []
    analysis = _restored_analysis(state["analysis"])
    limits = {"graded": True}
    with dg.grading_limits(limits):
        graded = dg._filter_applicable_articles(state.get("text") or "", state.get("request") or "", mine, analysis)
    added = []
    for row in graded:
        grade = row.get("_grade")
        topic = next((t for t in topics if t.get("unit") == row.get("_unit")), None)
        if not isinstance(grade, int) or grade < 2 or topic is None or not row.get("_article"):
            continue
        label = _label(_key(row["_article"]))
        if label in topic.get("articles") or []:
            continue
        planned = int(topic.get("planned") or len(topic.get("articles") or []))
        topic["articles"] = list(topic.get("articles") or [])[:planned] + [label] + list(topic.get("articles") or [])[planned:]
        topic["planned"] = min(planned + 1, TOPIC_ARTICLES)
        topic["matched"] = list(dict.fromkeys(list(topic.get("matched") or []) + [label]))
        topic.setdefault("grades", {})[label] = grade
        state["rows"] = list(state.get("rows") or []) + [_compact_row(row)]
        added.append(label)
    done = {_key(r["_article"]) for r in mine if r.get("_article")}
    state["reserve"] = [r for r in state.get("reserve") or [] if not r.get("_article") or _key(r["_article"]) not in done]
    _log("answer_grading", {"candidates": len(mine), "added": added, "graded": limits.get("graded_n")})
    return added


_CHARGED_RE = re.compile(r"a carico|indagat|imputat|convenut|ricorrent|accusat|denunciat", re.IGNORECASE)
# v60s (7): the extraction writes a convenuto as «controparte» (fact_signals._ROLE_SYNONYMS) -- with no client tagged,
# the party the action is directed at
_COUNTERPARTY_RE = re.compile(r"\bcontroparte\b", re.IGNORECASE)


def _assumed_party(state: Dict[str, Any], analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The party an answer assumes as the client when none is settled: the one the case puts a carico, convenuto or
    ricorrente (never the deceased)."""
    facts = _facts(analysis)
    dead = set(deceased(facts, case_text(state)))         # v60d: the deceased is never the client
    parties = [p for p in facts.get("parties") or [] if isinstance(p, dict) and p.get("name") and p["name"] not in dead]
    return next((p for p in parties if _CHARGED_RE.search(str(p.get("role") or ""))), None) or \
        next((p for p in parties if _COUNTERPARTY_RE.search(str(p.get("role") or ""))), None)      # v60s (7)


def prepare_side(state: Dict[str, Any]) -> None:
    """v60e: before the immediate answer picks its topics -- the client settled, else the one the answer will assume
    (settle_client) -- its side drives the topics' order (reside)."""
    analysis = _restored_analysis(state["analysis"])
    client = str(analysis.get("parte_assistita") or (state.get("answered") or {}).get("parte_assistita") or "")
    if not client and any(q.get("id") == "parte_assistita" for q in state.get("questions") or []):
        pick = _assumed_party(state, analysis)
        client = str((pick or {}).get("name") or "")
    if client:
        _isolated("reside", lambda: reside(state, client))


def settle_client(state: Dict[str, Any], analysis: Dict[str, Any]) -> Tuple[str, str]:
    """D: (the client, the line saying it was assumed). Stated or answered: (it, ""). Asked and unanswered: the party the
    case puts a carico, convenuto or ricorrente (never the first name), set in this answer's analysis only, with
    CLIENT_ASSUMED; none: ("", CLIENT_UNKNOWN). Not asked (the facts name the client): ("", "")."""
    answered = (state.get("answered") or {}).get("parte_assistita")
    if answered:
        client = str(analysis.get("parte_assistita") or answered)
        return client, (known_client_line(state, analysis, client) if single_mode(state) else "")      # v60s (7)
    if not any(q.get("id") == "parte_assistita" for q in state.get("questions") or []):
        known = _client_name(analysis) if single_mode(state) else ""      # v60s (7): named by the facts
        return "", (known_client_line(state, analysis, known) if known else "")
    facts = _facts(analysis)
    pick = _assumed_party(state, analysis)
    # v60d: after a reply that answered a question, never «Non mi hai indicato…» (the assumption is only logged);
    # v60k: a single-shot parere says it every time it is assumed (its first line)
    quiet = bool(state.get("answered_in_reply")) and not single_mode(state)
    if pick is None:
        _log("client_settled", {"how": "unknown", "line": not quiet})
        return "", "" if quiet else C.CLIENT_UNKNOWN
    label = frame_party_label(analysis, str(analysis.get(CASE_KEY) or ""), str(pick.get("role") or ""))   # v60r (5)
    role = label or str(pick.get("role") or "").strip()
    _log("party_label", {"party": pick["name"], "label": label, "extracted": pick.get("role") or ""})
    _apply_answer(facts, {"kind": "party", "names": [pick["name"]]}, pick["name"])
    analysis["parte_assistita"] = pick["name"]
    _log("client_settled", {"how": "assumed", "client": pick["name"], "line": not quiet})
    from . import party_names               # v60h (9)
    shown = party_names.label(pick["name"], str(analysis.get(CASE_KEY) or ""))
    if single_mode(state):                  # v60k: «Il parere è reso nell'interesse di X; se assisti un'altra parte…»
        line = (C.fill(C.SINGLE_SHOT_CLIENT_ROLE, client=shown, role=label) if label and label != _low(shown) else None) \
            or C.fill(C.SINGLE_SHOT_CLIENT, client=shown)                   # v60r (5): with the frame's label
        return pick["name"], "" if quiet else party_names.contract(line or "")
    return pick["name"], "" if quiet else party_names.contract(C.fill(C.CLIENT_ASSUMED, name=shown, role=role) or "")


def known_client_line(state: Dict[str, Any], analysis: Dict[str, Any], client: str) -> str:
    """v60s (7): the single-shot first line for a client the user stated or the facts name -- with the frame's label
    (frame_party_label) when the client is the party the action is directed at (its role: _CHARGED_RE), else with the
    counterparty's (a party a carico, else the extraction's «controparte»); neither: the client alone. Logged."""
    from . import party_names
    case = str(analysis.get(CASE_KEY) or "")
    facts = _facts(analysis)
    dead = set(deceased(facts, case_text(state)))
    parties = [p for p in facts.get("parties") or [] if isinstance(p, dict) and p.get("name") and p["name"] not in dead]
    mine = next((p for p in parties if _low(p["name"]) == _low(client)), {})
    shown = party_names.label(client, case)
    line, how = None, "solo cliente"
    if mine and _CHARGED_RE.search(str(mine.get("role") or "")):
        label = frame_party_label(analysis, case, str(mine.get("role") or ""))
        line, how = C.fill(C.SINGLE_SHOT_CLIENT_KNOWN_ROLE, client=shown, role=label), "cliente destinatario"
    else:
        others = [p for p in parties if _low(p["name"]) != _low(client)]
        other = next((p for p in others if _CHARGED_RE.search(str(p.get("role") or ""))), None) or \
            next((p for p in others if _COUNTERPARTY_RE.search(str(p.get("role") or ""))), None)
        if other:
            label = frame_party_label(analysis, case, str(other.get("role") or ""))
            line = C.fill(C.SINGLE_SHOT_CLIENT_COUNTERPARTY, client=shown, party=party_names.label(other["name"], case),
                          role=label)
            how = f"controparte {other['name']}" if line else how
    line = line or C.fill(C.SINGLE_SHOT_CLIENT_KNOWN, client=shown) or ""
    _log("client_line", {"client": client, "how": how, "line": line})
    return party_names.contract(line)


def hypotheses_block(topics: Sequence[Dict[str, Any]]) -> str:
    """A: the first read's hypotheses the chosen topics anchor, for the writer (never the unanchored ones)."""
    lines = []
    for t in topics:
        for h in t.get("hypotheses") or []:
            parts = [f"istituti: {', '.join(h['institutions'])}" if h.get("institutions") else "",
                     f"da provare: {'; '.join(h['to_prove'])}" if h.get("to_prove") else "",
                     f"prove: {'; '.join(h['evidence'])}" if h.get("evidence") else ""]
            extra = "; ".join(p for p in parts if p)
            whose = f" per {h['party']}" if h.get("party") else ""          # v60e
            lines.append(f"- linea {h.get('line')}{whose}: {h.get('claim')}" + (f" ({extra})" if extra else ""))
    return (HYPOTHESES_TITLE + ":\n" + "\n".join(lines) + "\n" + HYPOTHESES_RULE) if lines else ""


HYPOTHESES_TITLE = "LINEE DEL CASO (primo esame, prima della ricerca delle norme)"
HYPOTHESES_RULE = ("Usale per impostare la strategia principale e subordinata, la linea della controparte e le prove; "
                   "affermale solo dove i fatti e le norme fornite le sostengono, altrimenti scrivi che vanno verificate.")


def _read_back(draft: str, state: Dict[str, Any], analysis: Dict[str, Any], rules: Sequence[Dict[str, Any]],
               rows: Sequence[Dict[str, Any]], rendered: str, numbers: Sequence[int], log: Dict[str, Any],
               names: Sequence[str] = ()) -> str:
    """E (readback): the sources are the case (its text, the upload, the answered questions: case_text), the articles'
    texts and the computed block; the statuses, each rule's requisites' (v60g: the record's while it is active). v60g
    (2): the case's tagged sentences and the parties' names, for the acts of a party (readback.party_acts)."""
    from . import readback
    from .plan_then_write import _code_key
    statuses: Dict[str, List[str]] = {}
    for rule in rules or []:
        code = re.sub(r"\s+", "", _code_key(str(rule["ref"][1])).lower())
        label = f"{rule['ref'][0]} {code}"
        statuses[label] = [requisite_status(x, analysis) for reqs in (rule.get("requisites") or {}).values() for x in reqs]
    sources = [case_text(state), rendered] + [str((r.get("s") or {}).get("plain_text") or "") for r in rows]
    articles, rubriche = {}, {}     # v60c: what each article provides, for the grounding check
    for r in rows:
        if r.get("_article"):
            code = re.sub(r"\s+", "", _code_key(str(r["_article"][1])).lower())
            s = r.get("s") or {}
            label = f"{r['_article'][0]} {code}"
            articles[label] = f"{s.get('titolo_articolo') or ''} {s.get('plain_text') or ''}"
            rubriche[label] = str(s.get("titolo_articolo") or "")
    return readback.read_back(draft, sources, statuses, (ANSWER_TITLES[3],), log, state.get("round1_text") or "",
                              ", ".join(str(n) for n in numbers), articles=articles, case=case_text(state),
                              rubriche=rubriche, documents=ref_documents([tuple(r["_article"]) for r in rows
                                                                          if r.get("_article")], rows),
                              sentences=state.get("sentences") or case_sentences(state),
                              parties=party_names(state, analysis, names))


def party_names(state: Dict[str, Any], analysis: Dict[str, Any], names: Sequence[str] = ()) -> List[str]:
    """v60g (2): the parties' names -- the writer's (rule_quote.party_names), the client, the extracted parties'."""
    out = [str(n) for n in names or [] if str(n or "").strip()]
    client = str((state.get("client") or {}).get("name") or analysis.get("parte_assistita") or "")
    if client:
        out.append(client)
    for p in (_facts(analysis).get("parties") or []):
        if isinstance(p, dict) and str(p.get("name") or "").strip():
            out.append(str(p["name"]))
    return list(dict.fromkeys(x.strip() for x in out if x.strip()))


def write_answer(state: Dict[str, Any], numbers: Sequence[int]) -> Tuple[str, List[Dict[str, Any]]]:
    """Step 5: the answer on the chosen topics, from the cache (no retrieval). (text, citations). v60: under a stage
    meter, one journal line [VERBOSE:interactive_answer_stages] with each stage's seconds and calls."""
    from . import stage_meter
    meter = stage_meter.Meter()
    with meter.active():
        try:
            return _write_answer(state, numbers)
        finally:
            meter.stop()
            stages = meter.as_log()
            _log("answer_stages", stages)
            state.setdefault("timings", {})["answer_stages"] = {k: v for k, v in stages.items()
                                                                 if k.endswith("_s")}      # v60k


def _write_answer(state: Dict[str, Any], numbers: Sequence[int]) -> Tuple[str, List[Dict[str, Any]]]:
    from .stage_meter import mark
    from . import (defensive_generation as dg, plan_then_write as ptw, render_computed, repetition_guard, rule_quote,
                   text_tidy)
    t0 = time.time()
    lang = state.get("lang") or "it"
    analysis = _restored_analysis(state["analysis"])
    topics = [state["topics"][n - 1] for n in numbers if 0 < n <= len(state.get("topics") or [])]
    if not topics:
        return C.NO_TOPIC_TO_ANSWER, []
    mark("grade_chosen")
    t_grade = time.time()
    _isolated("grade_chosen", lambda: grade_chosen(state, topics))         # v60 (G): the chosen topics' rest graded
    state.setdefault("timings", {})["answer_grading_s"] = round(time.time() - t_grade, 1)
    mark("settle_client")
    client, assumption = _isolated("settle_client", lambda: settle_client(state, analysis), ("", "")) or ("", "")   # D
    if client:                  # v60e: the client's lines for the writer, the chosen topics as they are
        _isolated("reside", lambda: reside(state, client, reorder=False))
        persist_client(state, client, assumption)                           # v60f (B8)
    all_rows = _restored_rows(state.get("rows"))
    from .article_text import row_abrogated, row_misaligned     # v60c: an abrogated article is never written; v60e:
    dead = {_key(r["_article"]) for r in all_rows if r.get("_article") and (row_abrogated(r) or row_misaligned(r))}
    records = state.get("records") or {}
    mark("records_link")        # v60f: each chosen topic's requisites linked to the case's sentences, in parallel
    _isolated("records_link", lambda: link_records(state, topics, dead))
    if records:                 # v60i (2): the counterparty's articles whose elements fail are the client's argument
        _isolated("role_verdicts", lambda: role_verdicts(state))
    if records:                 # v60s (1): an offence issue with no offence article left goes, with all its articles
        topics = _isolated("issue_survival", lambda: surviving_issues(state, topics, all_rows, dead), topics) or []
        if not topics:
            return C.NO_TOPIC_TO_ANSWER, []
    if records:                 # v60p: each covered topic's anchor earns its §2 block
        _isolated("anchor_blocks", lambda: anchor_blocks(state, topics))
    cap_dropped: List[Tuple[str, str]] = []
    single = single_mode(state)
    cap = SINGLE_SHOT_ARTICLES if single else TOPIC_ARTICLES          # v60k: the single-shot answer's cap
    refs = capped_refs(topics, cap=cap, records=records if records else None, dropped=cap_dropped,  # v60f (A4): by weight
                       available={_key(r["_article"]) for r in all_rows if r.get("_article")})
    refs = [r for r in refs if _key(r) not in dead]
    beyond: List[Tuple[tuple, str]] = []
    if single:                  # v60k: what the cap leaves out, each with its topic, for the «non trattate» line
        for ref in cap_dropped:
            if _key(ref) in dead or _key(ref) in {_key(r) for r, _ in beyond}:     # v60q: once, with its first topic
                continue
            topic = next((t for t in topics if _key(ref) in {_key(r) for r in _topic_refs(t, all_=True)}), {})
            beyond.append((tuple(ref), str(topic.get("title") or "")))
        _log("single_shot_cap", {"cap": cap, "planned": [_label(r) for r in refs],
                                 "beyond": [{"article": _label(r), "topic": t} for r, t in beyond]})
    all_rows = [r for r in all_rows if not (r.get("_article") and _key(r["_article"]) in dead)]
    rows = [r for r in all_rows if r.get("_article") and _key(r["_article"]) in {_key(x) for x in refs}]
    for row in rows:            # v56: no footnote marker or editorial note reaches the writer or §2
        sec = row.get("s") or {}
        row["s"] = {**sec, "plain_text": rule_quote.clean_text(sec.get("plain_text")) if sec.get("plain_text") else
                    sec.get("plain_text"), "titolo_articolo": rule_quote.clean_rubrica(sec.get("titolo_articolo"))
                    if sec.get("titolo_articolo") else sec.get("titolo_articolo")}
    tool_rows = _restored_rows(state.get("tool_rows"))
    mark("documents")
    documents = _isolated("documents", _documents, []) or []
    case = case_text(state)
    articles = dg._source_block(rows, dg._ARTICLE_CHARS)
    mark("plan")
    issues = plan_inputs(state, analysis, topics, refs)        # v60f (c, M8): the chosen topics, their records, the angle
    plan = _isolated("plan", lambda: ptw.make_plan(case, "", analysis, articles, "", "", rows + tool_rows, {}), None)
    planned = plan is not None
    if plan is None:
        plan = _fallback_plan(state, analysis, topics, _rule_refs(rows, refs))
    if issues:                  # v60f (c): the plan argues the chosen topics, it never selects the issues
        from . import issue_records
        fix_log: List[Dict[str, Any]] = []
        plan = issue_records.fix_plan(plan, issues, fix_log)
        _log("plan_fixed", {"issues": [i["titolo"] for i in issues], "log": fix_log})
    for key in (ptw.PLAN_ISSUES_KEY, ptw.PLAN_ANGLE_KEY):
        analysis.pop(key, None)
    plan, analysis_p = ptw.for_the_plan(plan, analysis)
    if client:                  # v60 (D): the client settled (stated, answered or assumed) is the plan's
        plan = {**plan, "parte_assistita": client}
    if "obiezioni_norme" not in plan:
        plan = _isolated("move_gated", lambda: render_computed.move_gated(plan, analysis_p)[0], plan)
    who = f"{plan.get('parte_assistita') or ''} {plan.get('posizione') or ''}".strip() or party(analysis)
    from . import procedural_facts
    offences = _offences(topics, rows)
    facts_cp = procedural_facts.by_number(rows)
    offence_facts = {n: facts_cp[n] for n, _ in offences if n in facts_cp}
    if records:                 # v60h (1): each d'ufficio condition with its record status, the line's verdict
        offence_facts = {n: procedural_facts.with_statuses(f, (records.get(f"{n} c.p.") or {}).get("procedibilita") or [])
                         for n, f in offence_facts.items()}
        _log("procedibilita", {f"{n} c.p.": {"esito": f.get("ufficio_esito") or "", "condizioni": [
            {"comma": c.get("comma"), "text": c.get("text", "")[:160], "status": c.get("status"), "why": c.get("why")}
            for c in f.get("ufficio_condizioni") or []]} for n, f in offence_facts.items() if f.get("querela")})
    analysis_p = {**analysis_p, "own_analysis": own_analysis(state, topics),   # v60p: §4's terms by the parere's own
                  "offences": offences,          # v58 (J): §4's delitti
                  "offence_facts": offence_facts,              # v60 (C): from the text; v60h (1): with the statuses
                  "topic_refs": _matched_refs(topics),                       # v59: the mediation of the topics' matter
                  "listed_topics": [{"title": t.get("title"), "articles": t.get("articles"), "source": t.get("source"),
                                     **({"grades": t["grades"]} if t.get("grades") else {})}
                                    for t in state.get("topics") or []]}     # v59: §4's terms, the listed topics' only;
                                                                             # v60g (5): by their articles graded 2+
    if render_computed.store_only(analysis_p):      # v60h (2): the writer's conclusions and checks read the closed list's
        listed = _isolated("mediation_list", lambda: render_computed.listed_mediation(analysis_p), None)
        if listed is not None:
            analysis_p = {**analysis_p, "fact_signals": {**_signals(analysis_p), "mediation": {
                k: v for k, v in listed.items() if k != "matters"}}}
            _log("mediation_list", listed)
        labour = _isolated("labour_list", lambda: render_computed.labour_of(analysis_p), None)    # v60i (9)
        if labour is not None and labour.get("matter"):
            _log("labour_list", labour)
    mark("render")
    rendered = _isolated("render_computed", lambda: render_computed.render(analysis_p, who), state.get("rendered")) or ""
    if single:                  # v60k: §4's open points in the single-shot form, for the section and the writer alike
        rendered = open_points_wording(rendered)
    mark("jobs")
    analysis_p = {**analysis_p, CASE_KEY: analysis.get(CASE_KEY) or case}
    context = ptw.statement_context(analysis_p)
    named = dg._refs_in(case)
    from . import answer_ledger, citation_ledger, issue_records
    records_on = record_mode(state)
    # v60g (d): while the answer is written every status is the record's -- the jobs' rules block, the §2 status lines,
    # the checks, the read-back; (e) the ledger defers every removal to the one check run last (record_check)
    names_ = party_names_of(state)
    points = {q["id"]: open_point_fact(q, names_) for q in open_points_list(state)} if single and records_on else {}
    quals = list(state.get("qualifications") or []) if single and records_on else []      # v60l: «da verificare»
    # v60r (6): an element no fact touches is «da provare» -- v60p's «da verificare» for the anchors' silent ones is gone
    with issue_records.statuses_from(state.get("records") if records_on else None), \
            issue_records.points_from(state.get("records") if points or quals else None, points, quals):   # v60k
        jobs = _jobs(state, plan, analysis_p, rows + tool_rows, refs, topics, rendered, documents)
        glossed = glossed_refs(state, topics, analysis_p, case, section_refs(plan, rows, refs), rows)
        ledger = answer_ledger.Ledger(glossed, ref_documents(list(refs) + list(glossed), rows), defer=records_on)
        with answer_ledger.active(ledger):      # v60f: every removal after writing recorded, with the step's name
            citation_ledger.protect(list(dict.fromkeys(list(refs) + list(glossed))))     # v60f (X7): tidy's guard set
            try:
                draft = _reviewed(state, plan, analysis, analysis_p, jobs, lang, named, documents, rows, tool_rows, case,
                                  context, refs, topics, rendered, numbers, t0, planned, glossed, beyond, cap)
            finally:
                citation_ledger.protect(())
    if draft is None:
        return C.ANSWER_FAILED, []
    final = answer_ledger.sections_of(draft)
    unaccounted = ledger.unaccounted(final)
    buckets = bucket_articles(refs, glossed, draft, ledger, cap_dropped, rows,
                              state.get("records") if records_on else None,
                              _isolated("sanctions", lambda: sanction_map(section_refs(plan, rows, refs), rows), {}),
                              cap=cap)
    record_answer(state, buckets, ledger, unaccounted, numbers)
    if single:                                  # v60r (1): the record's rows go to the article citations only
        moved: Dict[str, Any] = {}
        draft, state["ledger_rows"] = _isolated("ledger_out", lambda: ledger_out(draft, state, moved),
                                                (draft, [])) or (draft, [])
        _log("ledger_rows", {**moved, "rows": len(state["ledger_rows"])})
    from . import party_names                   # v60h (9): a party named by role never capitalised as a name
    roles = party_names.role_names(_facts(analysis).get("parties") or [], case)
    if roles:
        lowered = party_names.lower_roles(draft, roles)
        if lowered != draft:
            _log("roles_lowered", {"roles": roles})
        draft = lowered
    if single:                                  # v60k: «Punti da precisare», v60q: at most OPEN_POINTS_MAX (0: none)
        found = _isolated("open_points", lambda: open_points(state, list(refs) + [r for r, _ in beyond]), None)
        if found and found.get("block"):
            draft = f"{draft}\n\n{found['block']}"
    if records_on:                              # v60s (4): §1 and §4 assert only what the record allows
        weak: List[Dict[str, Any]] = []
        draft = _isolated("asserted_only", lambda: asserted_only(draft, state, weak), draft) or draft
        if weak:
            _log("asserted_only", {"marked": weak})
    merged: List[str] = []                      # v60r (4): consecutive «(da verificare…)» markers, one
    draft = merged_markers(draft, merged)
    if merged:
        _log("markers_merged", {"runs": merged})
    if assumption:                              # v60 (D): the client assumed, said first
        draft = f"{assumption}\n\n{draft}"
    others = [C.fill(C.CLOSING_TOPIC, n=i, title=t["title"]) for i, t in enumerate(state["topics"], 1) if i not in numbers]
    others = [o for o in others if o]          # v56: no other topic, no closing (never «Oppure dimmi…» alone)
    closing = C.fill_parts(C.CLOSING, topics=C.LIST_JOIN.join(others)) if others else \
        (C.SINGLE_SHOT_CLOSING if single else "")
    if single:                                  # v60n: the topics not covered, one line each under «Altri profili»
        rest = _isolated("other_profiles", lambda: single_profiles(state, numbers, analysis), []) or []    # v60r (2)
        if rest:
            draft += "\n\n" + "\n".join([C.OTHER_PROFILES_TITLE, ""] + rest)
        closing = C.SINGLE_SHOT_CLOSING
    if closing:
        draft += "\n\n" + closing
    for n in numbers:
        state.setdefault("answers", {})[str(n)] = {"version": state.get("version"), "chars": len(draft)}
    mark("citations")
    cited = rows + tool_rows
    citations = _isolated("citations", lambda: [c for c in dg._rows_to_sources(dg._draft_sources(draft, cited, [], documents))
                                                if c.get("sections")], []) or []
    state.setdefault("timings", {})["answer_s"] = round(time.time() - t0, 1)
    _log("answer_timings", {k: v for k, v in state["timings"].items() if k.startswith("answer") or k == "records_link_s"})
    return draft, citations


def _reviewed(state, plan, analysis, analysis_p, jobs, lang, named, documents, rows, tool_rows, case, context, refs,
              topics, rendered, numbers, t0, planned, glossed=None, beyond=(), cap=TOPIC_ARTICLES) -> Optional[str]:
    """The sections written and every pass after them, to the final draft (before the client line and the closing);
    None when the writer fails. Every pass runs through plan_then_write._step or the answer ledger (v60f). v60g, with
    records: the writer under writer_rules (d); no pass deletes -- the ledger defers their removals -- and the one
    check against the record (record_check) runs last (e); unsupported_out is not run (every planned article is
    written, «da provare» when its record says so)."""
    from .stage_meter import mark
    from . import (answer_ledger, defensive_generation as dg, plan_then_write as ptw, record_check, render_computed,
                   repetition_guard, rule_quote, text_tidy, writer_rules)
    records_on = record_mode(state)
    rules_text = writer_rules.system_rules(plan.get("parte_assistita") or "", references=single_mode(state)) \
        if records_on else ""
    mark("write")
    sections = _isolated("write", lambda: ptw.write_sections(plan, jobs, lang, named, documents, rows + tool_rows, case,
                                                             context=context, sections_note=sections_note(),
                                                             writer_rules=rules_text), None)
    if not sections:
        _log("answer_failed", {"topics": [t["title"] for t in topics], "planned": planned})
        return None
    mark("review:scaffolding_cosmetic")
    sections = ptw._step("scaffolding_out", sections, lambda secs: [(t, scaffolding_out(x)) for t, x in secs])  # v57 (G)
    logs_k: List[Dict[str, Any]] = []
    sections = ptw._step("cosmetic_fixes", sections, lambda secs: cosmetic_fixes(
        secs, list(ANSWER_TITLES) + [OTHER_PROFILES_TITLE], logs_k))                                            # v57 (K)
    if records_on:              # v60q: the statuses are the code's -- never the writer's ordinary words
        status_log: List[Dict[str, Any]] = []
        sections = ptw._step("status_words", sections, lambda secs: status_words_out(secs, status_log))
        _log("status_words", {"rewritten": status_log})
    logs: Dict[str, List[Dict[str, Any]]] = {"term_consistency": [], "unsupported_out": [], "citation_check": [],
                                             "strict_status": [], "conditional": [], "objections": []}
    rule_refs = _rule_refs(rows, refs)
    mark("article_rules")
    rules = _isolated("article_rules", lambda: rule_quote.article_rules(plan, rows, analysis_p, rule_refs), []) or []
    sections = ptw._step("term_consistency", sections,
                         lambda secs: term_consistency(secs, context, analysis_p, logs["term_consistency"]))
    from . import procedural_facts
    offences = analysis_p.get("offences") or []
    logs["procedural_consistency"] = []         # v60 (C): the writer's procedural sentences against the article's text
    sections = ptw._step("procedural_consistency", sections, lambda secs: procedural_facts.consistency(
        secs, analysis_p["offence_facts"], {n: r for n, r in offences}, logs["procedural_consistency"],
        skip=(ANSWER_TITLES[3],)))

    def quoted(secs):           # v60q (single shot): references, never the comma's text -- a failing rule sentence goes
        done, log = rule_quote.check_sections(secs, plan, rows, analysis_p, documents, verbs=rule_quote.EXTENDED_VERB_RE,
                                              refs=rule_refs, quote=not single_mode(state))
        logs["rule_quote"] = [log]
        return done
    sections = ptw._step("rule_quote", sections, quoted)
    sections = ptw._step("strict_status", sections, lambda secs: strict_status(secs, rules, analysis_p,
                                                                              logs["strict_status"]))
    logs["phantom_commi"] = []
    sections = ptw._step("phantom_commi", sections, lambda secs: phantom_commi_out(secs, rules, logs["phantom_commi"]))
    mark("party_names")
    names = _isolated("party_names", lambda: rule_quote.party_names(analysis_p, plan), []) or []
    sections = ptw._step("conditional_on_unknown", sections, lambda secs: conditional_on_unknown(
        secs, rules, analysis_p, documents, names, logs["conditional"], single=single_mode(state)))
    logs["unanswered"] = []
    sections = ptw._step("unanswered_out", sections, lambda secs: unanswered_out(secs, state, logs["unanswered"], names))
    if plan.get("obiezioni_norme"):
        sections = ptw._step("gated_as_support", sections, lambda secs: [
            (title, render_computed.gated_as_support(title, text, plan, documents)[0] or text) for title, text in secs])
    sections = ptw._step("repetition_guard", sections, lambda secs: repetition_guard.guard_sections(secs, "interactive")[0])
    mark("allowed")
    allowed = _allowed(state, analysis_p, rule_refs or refs, tool_rows, rendered, documents)
    allowed = list(allowed) + [r for r, _ in beyond or ()]      # v60k: named in the «non trattate» line, never argued
    sections = ptw._step("citation_check", sections, lambda secs: citation_check(secs, allowed, documents,
                                                                                logs["citation_check"]))
    logs["headless"] = []
    sections = ptw._step("headless", sections, lambda secs: headless_out(secs, logs["headless"]))
    if dg.feature_flags.enabled(dg.RUBRICA_TEXT_CHECK_FLAG):
        sections = ptw._step("rubrica_stamp", sections, lambda secs: _stamped(secs, analysis_p, rows + tool_rows,
                                                                              documents, case))
    else:
        logs["rubrica_stamp"] = [{"skipped": "RUBRICA_TEXT_CHECK off: no stamp without its text check"}]
    before = list(sections)
    # v59 (v58-p1: 52 c.p.'s sentences moved to «Altri profili»): a topic's matched articles -- a table's own -- stay too
    anchors = list(dict.fromkeys(_parse_label(a) for t in topics for a in [t.get("anchor") or ""]
                                 + list(t.get("matched") or []) if a))
    mark("review:unsupported_out")
    if records_on:              # v60g (e): the record keeps every planned article in §2, its statuses as they are
        logs["unsupported_out"] = [{"skipped": "record: every planned article is written"}]
    else:
        sections = _isolated("unsupported_out", lambda: unsupported_out(sections, rules, analysis_p, documents,
                                                                        logs["unsupported_out"], anchors), sections) or before
        answer_ledger.note("unsupported_out", before, sections)    # v60f: it may add «Altri profili»: noted, not _step
    at = next((i for i, (t, _) in enumerate(sections) if t == ANSWER_TITLES[4]), len(sections))
    if rendered:            # nothing computed: no procedural section (never a placeholder)
        sections = sections[:at] + [(ANSWER_TITLES[3], with_copy_mediation(rendered, analysis_p))] + sections[at:]
    mark("assemble")
    titles = objections(state, topics, refs)
    logs["norms"] = []
    if glossed is None:
        glossed = glossed_refs(state, topics, analysis_p, case, section_refs(plan, rows, refs), rows)
    sanctions = _isolated("sanctions", lambda: sanction_map(section_refs(plan, rows, refs), rows), {}) or {}
    if sanctions:
        _log("sanctions", {_label(k): _label(v) if v else None for k, v in sanctions.items()})
    sections = ptw._step("assemble", sections, lambda secs: [
        (t, assemble_objections(x, titles, logs["objections"]) if t == ANSWER_TITLES[2]
         else assemble_norms(x, rules, analysis_p, rows, logs["norms"], section_refs(plan, rows, refs),
                             _pool_rubriche(state), glossed, anchors, keep=records_on, sanctions=sanctions,
                             beyond=[r for r, _ in beyond or ()])
         if t == ANSWER_TITLES[1] else x) for t, x in secs])
    before = list(sections)
    sections = [(t, x) for t, x in sections if (x or "").strip()]      # an objections section with no reply: gone
    answer_ledger.note("empty_sections", before, sections)
    draft = _join(sections)
    mark("tidy_final")
    draft = answer_ledger.draft_step("tidy", draft, lambda d: text_tidy.tidy(d)[0])
    logs["final"] = []
    verdicts = record_verdicts(state) if records_on else None
    lines = record_lines(state, sanctions, {_label(_key(r)): t for r, t in beyond or ()}, cap) \
        if records_on else None                                                     # v60i (1); v60k: the cap's
    if records_on:
        from . import issue_records
        asked_now = open_questions(state)
        _log("norm_blocks", {"blocks": [_label(_key(r)) for r in glossed],
                             "by_question": {label: issue_records.asked(state["records"][label], asked_now)
                                             for label in (_label(_key(r)) for r in glossed)
                                             if label in (state.get("records") or {})
                                             and issue_records.asked(state["records"][label], asked_now)},
                             "lines": {label: list(v) for label, v in (lines or {}).items()
                                       if label in {_label(_key(r)) for r in section_refs(plan, rows, refs)}}})
    draft = answer_ledger.draft_step("answer_final", draft, lambda d: answer_final(d, logs["final"], verdicts, lines))
    mark("readback")
    logs["readback"] = {}                       # v60 (E): contradictions, facts in no source, the lint
    draft = answer_ledger.draft_step("readback", draft, lambda d: _read_back(
        d, state, analysis_p, rules, rows + tool_rows, rendered, numbers, logs["readback"], names))
    if records_on:              # v60g (e): the one check, last, against the record -- the only one that removes
        mark("record_check")
        ctx = check_context(state, rows + tool_rows, refs, glossed, allowed, documents, case, rules)
        checked, logs["record_check"] = _isolated("record_check", lambda: record_check.check(
            answer_ledger.sections_of(draft), ctx, answer_ledger.current()), (None, {"error": "failed"})) \
            or (None, {"error": "failed"})
        if checked is not None:
            draft = answer_ledger.rejoin(draft, checked)
            draft = folded_norms(draft, verdicts, logs["final"], lines)
        _log("record_check", {k: v for k, v in logs["record_check"].items() if k != "rejected"}
             | {"rejected": len(logs["record_check"].get("rejected") or [])})
    _log("answer", {"topics": [t["title"] for t in topics], "planned": planned, "articles": [_label(r) for r in rule_refs],
                    "objections": len(titles), "rendered_chars": len(rendered),
                    **{k: v for k, v in logs.items()}}, (time.time() - t0) * 1000)
    return draft


# --- v59: the final pass, on the answer as the user reads it (after tidy, which can still empty a block) -------------

_NORM_HEADING_RE = re.compile(r"^\*\*Art\.\s+(\S+)\s+(.+?)(?:\s+—\s+.*?)?\*\*\s*$")
# v60d: a §2 gloss that declares its article inapplicable to the case («non si applica al caso», «non trova applicazione
# nella fattispecie», «la norma non è applicabile») -- outside quotations, not conditional, not about one comma, and no
# sentence of the gloss applying it
_NOT_APPLY_RE = re.compile(
    r"\bnon\s+(?:si\s+applica|(?:è|risulta|appare|sembra|sarebbe)\s+(?:direttamente\s+)?(?:applicabile|pertinente|"
    r"invocabile|rilevante)|trova\s+applicazione|viene\s+in\s+rilievo|può\s+(?:essere\s+)?(?:applicat\w+|invocat\w+))"
    r"|\b(?:è|risulta)\s+inapplicabile|\besula\s+dal(?:la)?\s+(?:caso|fattispecie|vicenda)"
    r"|^\W*(?:norma\s+)?non\s+applicabile\b", re.IGNORECASE)
_CASE_REF_RE = re.compile(r"\b(?:al|nel|alla|nella|ai|nei|per il|per la)\s+(?:presente\s+)?(?:caso|fattispecie|vicenda|"
                          r"fatti|specie)\b|\bqui\b|\bin concreto\b|\bin esame\b"
                          # v60i (6): «non si applica in questo caso», «nel caso di specie»
                          r"|\bin\s+(?:questo|quest['’]|tale|codesto)\s*(?:caso|fattispecie|vicenda|ipotesi)\b"
                          r"|\bnel\s+caso\s+(?:di\s+specie|concreto|in\s+esame)\b", re.IGNORECASE)
_NORM_SUBJECT_RE = re.compile(r"^\W*(?:(?:la|questa|tale)\s+(?:norma|disposizione|previsione)|(?:l['’]|quest['’]|"
                              r"tale\s+)\s*articolo|l['’]art\.\s*\d+|essa|non\s+si\s+applica|norma\s+non\s+"
                              r"applicabile|non\s+applicabile)\b", re.IGNORECASE)
# v60e: phrasings that declare it whatever their subject («non rileva (direttamente)», «non costituisce un problema»,
# «mantiene la sua efficacia», «non trova applicazione», «non è pertinente», «non si pone»)
# v60i (6): the PO's phrasings from the live answers -- «non è rilevante», «non è scaduto», «non costituisce una norma
# sostanziale», «è ancora in corso» (a term that bars nothing yet)
_NOT_APPLY_STRONG_RE = re.compile(
    r"\bnon\s+rileva(?:\s+direttamente)?\b|\bnon\s+costituisce\s+(?:un\s+|alcun\s+)?problema|\bmantiene\s+(?:la\s+sua\s+|"
    r"intatta\s+la\s+(?:sua\s+)?)?efficacia|\bnon\s+trova\s+applicazione|\bnon\s+(?:è|risulta|appare)\s+pertinente|"
    r"\bnon\s+si\s+pone\b"
    r"|\bnon\s+(?:è|risulta|appare|sembra)\s+(?:direttamente\s+)?rilevante|\bnon\s+(?:è|risulta)\s+(?:ancora\s+)?scadut[oa]\b"
    r"|\bnon\s+costituisce\s+(?:una\s+)?norma\s+sostanziale|\b(?:è|risulta)\s+ancora\s+in\s+corso\b", re.IGNORECASE)
_APPLIES_RE = re.compile(r"(?<!non )\b(?:si applica|è applicabile|trova applicazione|si applicano|sono applicabili)\b",
                         re.IGNORECASE)
_PARTIAL_RE = re.compile(r"\b(?:comma|co\.|ma|tuttavia|invece|bensì|salvo|se|qualora|ove|purché|nella misura)\b",
                         re.IGNORECASE)


_ABBREV_SAFE_SPLIT_RE = re.compile(r"(?<!\bn\.)(?<!\bnn\.)(?<!\bco\.)(?<!\bart\.)(?<!\bartt\.)(?<!\blett\.)(?<!\bss\.)"
                                   r"(?<!\bcfr\.)(?<=[.;!?])\s+", re.IGNORECASE)


def declares_inapplicable(gloss: str) -> bool:
    """v60d: the gloss says its article does not apply to the case (one sentence, _NOT_APPLY_RE with the case or the
    norm as its subject), and none of its sentences applies it."""
    plain = re.sub(r"«[^»]*»|\"[^\"]*\"", " ", gloss or "")
    # v60i (6): never split after an abbreviation («(comma 2, n. 2) non è rilevante» is one sentence about one comma)
    sentences = [x.strip() for x in _ABBREV_SAFE_SPLIT_RE.split(" ".join(plain.split())) if x.strip()]
    said = any((_NOT_APPLY_RE.search(x) and (_CASE_REF_RE.search(x) or _NORM_SUBJECT_RE.search(x))
                or _NOT_APPLY_STRONG_RE.search(x)) and not _PARTIAL_RE.search(x) for x in sentences)
    applied = any(_APPLIES_RE.search(x) and not _NOT_APPLY_RE.search(x) and not _NOT_APPLY_STRONG_RE.search(x)
                  for x in sentences)
    return said and not applied
_OTHER_NORMS_RE = re.compile(r"^" + re.escape(C.OTHER_NORMS_LINE.split("{", 1)[0]) + r"\s*(.*?)\.?\s*$")
_OTHER_ITEM_RE = re.compile(r"art\.\s*(\S+)\s+(.+?)\s*$")
_NOT_APPLICABLE_RE = re.compile(r"^" + re.escape(C.NOT_APPLICABLE_LINE.split("{", 1)[0]) + r"\s*(.*?)\.?\s*$")


_ITEM_RE = re.compile(r"^art\.\s*(\S+)\s+(.+?)(?:\s+\((.*)\))?\s*$")


def _line_items(text: str) -> List[Tuple[str, str, str]]:
    """v60i (1): the items of a closing line as (number, code, reason) -- split at the join outside «…» and (…), so a
    reason holding a comma stays one item. v60k: the last item's code given back the dot the line's own full stop took
    («c.c» -> «c.c.»)."""
    join = C.OTHER_NORMS_JOIN or ", "
    pieces, cur, depth, i = [], "", 0, 0
    while i < len(text or ""):
        ch = text[i]
        if ch in "(«":
            depth += 1
        elif ch in ")»":
            depth = max(0, depth - 1)
        if depth == 0 and text.startswith(join, i):
            pieces.append(cur)
            cur, i = "", i + len(join)
            continue
        cur += ch
        i += 1
    pieces.append(cur)
    out = []
    for piece in (x.strip() for x in pieces if x.strip()):
        m = _ITEM_RE.match(piece)
        if not m:
            out.append(("", piece, ""))
            continue
        code = m.group(2).strip()
        if re.fullmatch(r"(?:[a-z]{1,4}\.\s?)+[a-z]{1,4}", code):          # an abbreviation cut at its last dot
            code += "."
        out.append((m.group(1), code, m.group(3) or ""))
    return out


def norms_final(text: str, log: Optional[List[Dict[str, Any]]] = None,
                verdicts: Optional[Dict[str, str]] = None,
                lines: Optional[Dict[str, Tuple[str, str]]] = None) -> str:
    """§2 as read: an article heading followed by nothing (v58-h5 625, 52, 628-bis, 53; v58-p1 52, 53; v58-l3 1117,
    1118 -- tidy and the checks after assembly can empty a block) is no block; it is named once in the closing line
    (OTHER_NORMS_LINE) with those already there. verdicts (v60g P1: {"N code": the record's verdict}): «non
    applicabile» is the record's, never the gloss' -- no gloss is read for it (declares_inapplicable) and none goes; a bare
    block of a «non applicabile» article is named in NOT_APPLICABLE_LINE. lines (v60i 1: {"N code": (line, reason)},
    record_lines -- the articles with no block due): such an article with no paragraph left is one item of its line,
    its reason with it (a paragraph on it is the check's to remove first); every item of the record's lines carries its
    reason."""
    from . import issue_records
    lines = lines or {}
    rows = (text or "").split("\n")
    out: List[str] = []
    untreated: Dict[str, Tuple[str, str, str]] = {}
    inapplicable: Dict[str, Tuple[str, str, str]] = {}

    def label_of(number: str, code: str) -> str:
        return f"{number} {_code_label(code)}" if number else code

    def add(target, number, code, reason=""):
        key = label_of(number, code)
        reason = reason or (target.get(key) or ("", "", ""))[2] or (lines.get(key) or ("", ""))[1]
        target[key] = (number, code, reason)
    i = 0
    while i < len(rows):
        line = rows[i]
        m = _OTHER_NORMS_RE.match(line.strip())
        if m:
            for number, code, reason in _line_items(m.group(1)):
                add(untreated, number, code, reason)
            i += 1
            continue
        m = _NOT_APPLICABLE_RE.match(line.strip())
        if m:
            for number, code, reason in _line_items(m.group(1)):
                add(inapplicable, number, code, reason)
            i += 1
            continue
        h = _NORM_HEADING_RE.match(line.strip())
        if not h:
            out.append(line)
            i += 1
            continue
        j = i + 1
        while j < len(rows) and not _NORM_HEADING_RE.match(rows[j].strip()) and not _OTHER_NORMS_RE.match(rows[j].strip()) \
                and not _NOT_APPLICABLE_RE.match(rows[j].strip()):
            j += 1
        body_ij = "\n".join(rows[i + 1:j]).strip()
        number, code = h.group(1), h.group(2).strip()
        label = f"{number} {_code_label(code)}"
        if verdicts is not None:            # v60g: the record decides; a block with its gloss stays as written
            prose = any(x.strip() and not x.lstrip().startswith("- ") for x in rows[i + 1:j])
            due = lines.get(label)
            if due and not prose:
                # v60i (1): no block due (the verdict not «applicabile», or «da provare» with no sentence of the case
                # for it): one item of its line, with its reason
                add(inapplicable if due[0] == issue_records.LINE_NOT_APPLICABLE else untreated, number, code, due[1])
                if log is not None:
                    log.append({"one_line": label, "line": due[0], "reason": due[1]})
            elif verdicts.get(label) == issue_records.NOT_APPLICABLE and not prose:
                # v60h (5): a «non applicabile» article is one item of the line, never a block (its status lines go
                # with it; a paragraph on it is the check's to remove: record_check norma_non_applicabile)
                add(inapplicable, number, code)
                if log is not None:
                    log.append({"not_applicable": label, "by": "record"})
            elif body_ij:
                out += rows[i:j]
            elif verdicts.get(label) == issue_records.NOT_APPLICABLE:
                add(inapplicable, number, code)
                if log is not None:
                    log.append({"not_applicable": label, "by": "record"})
            else:
                add(untreated, number, code, C.LINE_REASON_NOT_WRITTEN if lines else "")
                if log is not None:
                    log.append({"bare": label})
            i = j
            continue
        if body_ij and declares_inapplicable(body_ij):      # v60d: one item of NOT_APPLICABLE_LINE, not a block
            add(inapplicable, number, code)
            if log is not None:
                log.append({"not_applicable": f"{number} {code}", "gloss": " ".join(body_ij.split())[:300]})
        elif body_ij:
            out += rows[i:j]
        else:
            add(untreated, number, code)
            if log is not None:
                log.append({"bare": f"{number} {code}"})
        i = j

    def item(number, code, reason, plain):
        if not number:
            return code
        if reason:
            return C.fill(C.NORM_ITEM_REASON, number=number, code=code, reason=reason) or f"art. {number} {code}"
        return C.fill(plain, number=number, code=code) or f"art. {number}"
    body = re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip("\n")
    for key in [k for k in untreated if k in inapplicable]:
        untreated.pop(key)
    closings = []
    if inapplicable:
        closings.append(C.fill(C.NOT_APPLICABLE_LINE, articles=C.OTHER_NORMS_JOIN.join(
            item(n, c, r, C.NOT_APPLICABLE_ITEM) for n, c, r in inapplicable.values()).rstrip(".")))
    if untreated:
        closings.append(C.fill(C.OTHER_NORMS_LINE, articles=C.OTHER_NORMS_JOIN.join(
            item(n, c, r, C.OTHER_NORMS_ITEM) for n, c, r in untreated.values()).rstrip(".")))
    closing = "\n\n".join(c for c in closings if c)
    if not closing:
        return body
    return f"{body}\n\n{closing}" if body else closing


_LEADING_CUT_RE = re.compile(r"^[ \t]+(?=\S)")
_BULLET_TAIL_RE = re.compile(r"^-\s*([,;:)\]].*)$")
_LABEL_BULLET_RE = re.compile(r"^\s*[-•*]\s+(?:\[\d{1,2}\]\s*)?(?:\*\*)?artt?\.\s*\d+(?:-[a-z]+)?(?:\s+[\w.]+){0,5}(?:\*\*)?\.?"
                              r"(?:\s+—\s+«[^»]*»?.*)?$", re.IGNORECASE)
_FINITE_VERB_RE = re.compile(
    r"\b(?:è|e'|sono|ha|hanno|può|possono|deve|devono|va|vanno|sta|fa|dà|era|erano|fu|sarà|saranno|sarebbe|sarebbero|"
    r"viola|violano|include|includono|presenta|presentano|integrano|integrino|indica|indicano|contesta|sostiene|"
    r"rileva|evidenzia|mostra|determina|implica|produce|ricorre|accerta|attesta|vieta|obbliga|autorizza|precisa|"
    r"chiarisce|ritiene|osserva|afferma|nega|limita|esclude|aggrava|attenua|condiziona|"
    r"risulta|risultano|sussiste|sussistono|richiede|richiedono|prevede|prevedono|dispone|consente|esclude|comporta|"
    r"applica|applicano|rientra|rientrano|costituisce|costituiscono|configura|spetta|impone|attribuisce|stabilisce|"
    r"deriva|dipende|manca|mancano|occorre|serve|basta|emerge|emergono|appare|appaiono|resta|restano|rimane|integra|"
    r"punisce|tutela|regola|disciplina|estende|riguarda|permette|presuppone|esige|giustifica|legittima|opera|"
    r"trova|trovano|risponde|rispondono|incide|grava|consiste|vale|valgono|segue|conferma|confermano|dimostra|"
    r"dimostrano|prova|provano|qualifica|chiede|chiedono|ottiene|nega|esclud\w+|\w+(?:ano|ono|ava|eva|iva|avano|evano|"
    r"rebbe|rebbero|erà|irà|ato|ata|ati|ate|ito|ita|iti|ite|uto|uta|uti|ute|ndo))\b", re.IGNORECASE)


def _drop_first_sentence(text: str) -> str:
    """text without its first sentence (a head cut at a removed citation: «(relativo all'uso dell'arma) costituisce …»)."""
    m = re.search(r"[.!?](?:[»”\"])?(?=\s+[A-ZÀ-Ü«(\[\-]|\s*$)", text)
    return text[m.end():].lstrip() if m else ""


def _fragment_start(text: str) -> bool:
    return _headless(text) or text.startswith("(")


def _orphans_out(text: str, log: List[Dict[str, Any]]) -> str:
    """v59 (v58-h5 §1 « (relativo all'uso dell'arma)…», v58-h7 §1 « La responsabilità di Caio…», §3 «(art. 321 c.p.\n-),
    non è sufficiente»): a line cut where a sentence head went -- its indent gone, a headless first sentence dropped; a
    bullet's tail on its own line («-), …») joined back to its bullet."""
    out: List[str] = []
    for line in (text or "").split("\n"):
        tail = _BULLET_TAIL_RE.match(line)
        if tail and out and out[-1].strip():
            out[-1] = out[-1].rstrip() + tail.group(1)
            log.append({"joined": tail.group(1)[:80]})
            continue
        if (not out or not out[-1].strip()) and re.match(r"^[—–]\s+(?=[A-ZÀ-Ü«])", line):
            line = re.sub(r"^[—–]\s+", "", line)                  # a paragraph opening on the dash of a cut head
            log.append({"dash": line[:80]})
        cut = _LEADING_CUT_RE.match(line)
        if cut and not re.match(r"^[ \t]+(?:[-*•]|\d+[.)])\s", line):
            line = line[cut.end():]
            line = re.sub(r"^[—–]\s+(?=[A-ZÀ-Ü«])", "", line)     # «— I contraenti possono …»: the head's dash only
            if _fragment_start(line):
                log.append({"headless": line[:120]})
                line = _drop_first_sentence(line)
                if not line.strip():
                    continue
        elif line and (not out or not out[-1].strip()) and _headless(line) and not re.match(r"^[-•*]\s+\S", line):
            log.append({"headless": line[:120]})
            line = _drop_first_sentence(line)
            if not line.strip():
                continue
        out.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip("\n")


_OPENER_RE = (r"(?:Questi|Questo|Questa|Queste|Tale|Tali|Il|Lo|La|Le|Gli|I|In|Per|Inoltre|Pertanto|Quindi|Ciò|Non|"
              r"Ne|Nel|Nella|Dal|Dalla|Ai|Al|Alla)\b")


def _markers_once(text: str, log: List[Dict[str, Any]]) -> str:
    """v59 (v57-h3 §5: «Se le opere risultano esistenti alla separazione,» twice, once before «Questi fatti»): in a
    paragraph, a «(da verificare: …)» or a copy's condition («Se …, », CONDITIONS) said once; a marker inside a sentence
    goes to its end."""
    prefixes = [c.strip() for c in C.CONDITIONS.values() if c.strip()]
    out = []
    for paragraph in re.split(r"(\n[ \t]*\n)", text or ""):
        seen: set = set()

        def once(m):
            key = m.group(0).strip().lower()
            if key in seen:
                log.append({"repeated": m.group(0).strip()[:80]})
                return ""
            seen.add(key)
            return m.group(0)
        paragraph = re.sub(r"\s*\(da verificare:[^)]*\)", once, paragraph)
        for prefix in prefixes:
            hits = list(re.finditer(re.escape(prefix) + r"\s*", paragraph))
            for k, m in reversed(list(enumerate(hits))):
                follows = paragraph[m.end():m.end() + 1]
                glued = re.match(_OPENER_RE, paragraph[m.end():])         # «…, Questi fatti»: a sentence's own start
                if k > 0 or glued:
                    rest = paragraph[m.end():]
                    if k > 0 and not follows.isupper():
                        rest = rest[:1].upper() + rest[1:]
                    paragraph = paragraph[:m.start()] + rest
                    log.append({"repeated": prefix[:80]})

        moved = []
        for m in re.finditer(r"\s*\(da verificare:[^)]*\)", paragraph):
            after = paragraph[m.end():]
            if after.strip() and not re.match(r"^[»”\"]?\s*[.!?;]", after):
                moved.append(m)
        for m in reversed(moved):
            after = paragraph[m.end():]
            stop = re.search(r"[.!?](?=\s|$)", after)
            if not stop:
                continue
            mark = m.group(0)
            paragraph = paragraph[:m.start()] + after[:stop.start()] + mark + after[stop.start():]
            log.append({"moved": mark.strip()[:80]})
        out.append(paragraph)
    return "".join(out)


def _scaffold_bullets_out(text: str, glosses: bool, log: List[Dict[str, Any]]) -> str:
    """v59 (v57-p1 §3, v55 «- [3] art. 1492 c.c.»): a bullet only an article's label; in «Altri profili» and the
    objections, a one-line gloss with no verb."""
    out = []
    for line in (text or "").split("\n"):
        m = re.match(r"^\s*[-•*]\s+(.*)$", line)
        plain = re.sub(r"«[^»]*»|\([^)]*\)", " ", m.group(1)) if m else ""
        if m and (_LABEL_BULLET_RE.match(line) or glosses and not _FINITE_VERB_RE.search(plain)
                  and not re.search(r"\bnon\s+(?:si\s+|lo\s+|la\s+|ne\s+|vi\s+|ci\s+)?[a-zà-ù]{2,}|\w+isc(?:e|ono)\b", plain)):
            log.append({"bullet": line.strip()[:120]})
            continue
        out.append(line)
    return "\n".join(out)


def answer_final(draft: str, log: Optional[List[Dict[str, Any]]] = None,
                 verdicts: Optional[Dict[str, str]] = None, lines: Optional[Dict[str, Tuple[str, str]]] = None) -> str:
    """v59: the last pass on the answer before its closing offer -- §2's bare headings into the closing line
    (norms_final); in every written section the cut lines (_orphans_out), a marker or a condition said once
    (_markers_once), a label-only bullet or, in «Altri profili» and
    the objections, a verbless one (_scaffold_bullets_out). The procedural section is the code's and stays as rendered.
    A section left with no sentence goes."""
    log = log if log is not None else []
    titles = list(ANSWER_TITLES) + [OTHER_PROFILES_TITLE]
    marks = [m for m in re.finditer(r"(?m)^\*\*(\d+)\.\s+([^*\n]+)\*\*\s*$", draft or "")
             if m.group(2).strip() in titles]
    if not marks:
        return draft
    head = draft[:marks[0].start()]
    sections = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(draft)
        sections.append((m.group(2).strip(), draft[m.end():end].strip("\n")))
    present = [t for t, _ in sections]
    out = []
    for title, text in sections:
        if title != ANSWER_TITLES[3]:
            text = _orphans_out(text, log)
            text = _markers_once(text, log)        # v60f (P9): _references_fixed gone, the writer names real sections
            text = _scaffold_bullets_out(text, title in (OTHER_PROFILES_TITLE, ANSWER_TITLES[2]), log)
        if title == ANSWER_TITLES[1]:
            text = norms_final(text, log, verdicts, lines)
        if title == OTHER_PROFILES_TITLE and not re.search(r"(?m)^\s*[-•*]\s+\S", text):
            log.append({"section_out": title})
            continue                # the intro alone, every item gone: no section
        if re.search(r"[A-Za-zÀ-ü]{3}", text):
            out.append((title, text.strip("\n")))
    return head + _join(out)


def with_copy_mediation(rendered: str, analysis: Dict[str, Any]) -> str:
    """The rendered procedural section with its mediation paragraph in the copy's words (MEDIATION), when mediation is a
    condition of the action."""
    from . import render_computed
    if not render_computed.mediation_of(analysis).get("required") or not C.MEDIATION \
            or "**" + render_computed.NEGOTIATION_TITLE + "**" in (rendered or ""):     # v60e: negotiation instead
        return rendered
    return re.sub(r"(\*\*Mediazione\*\*\n\n)[^\n]+", lambda m: m.group(1) + C.MEDIATION, rendered, count=1)


def _documents() -> List[Dict[str, Any]]:
    from .doc_lookup import _fetch_doc_names
    from .main import NEO4J_DATABASE, driver
    return _fetch_doc_names(driver, NEO4J_DATABASE)


def _stamped(sections, analysis, rows, documents, case) -> List[Tuple[str, str]]:
    """The rubrica stamp (dg._check_qualifications) with its text check (RUBRICA_TEXT_CHECK: the 1061 fix), over the
    written sections at once."""
    import neo4j
    from . import defensive_generation as dg
    from .main import NEO4J_DATABASE, driver
    titles = [t for t, _ in sections]
    with driver.session(database=NEO4J_DATABASE, default_access_mode=neo4j.READ_ACCESS) as session:
        draft = dg._check_qualifications(_join(sections), analysis, rows, session, documents, case, one_stem=True)
    return _split_draft(draft, titles) or sections


# --- the turn -------------------------------------------------------------------------------------------------------

# v60 (journal 19:12:19): what the user's own words already answer -- the client («Sei l'avvocato di Tizio») and the
# question they ask (the angle)
_NAME = r"([A-ZÀ-Ü][\w'’-]+(?:\s+[A-ZÀ-Ü][\w'’-]+)?)"
# v60d: a courtesy title or a «la società» before the name is not the name
_TITLE = r"(?i:(?:il\s+|la\s+|lo\s+|l['’]\s*)?(?:sig\.(?:ra)?\s*|signor[ae]?\s+|sig\.na\s*|dott\.(?:ssa)?\s*|" \
         r"dottor(?:essa)?\s+|ing\.\s*|avv\.\s*|prof\.(?:ssa)?\s*|societ[àa]\s+|ditta\s+|impresa\s+|(?:il|la)\s+))?"
_CLIENT_STATED_RE = re.compile(
    r"\b(?i:sei|sono|siamo|in qualit[àa] di|come|quale|nelle vesti di|assunte le vesti di)\s+(?i:l['’]\s*|il\s+|la\s+)?"
    r"(?i:avvocat[oa]|difensore|legale|consulente)\s+(?i:di|del|della|dello|dei|delle)\s+" + _TITLE + _NAME
    + r"|\b(?i:assisti|difendi|rappresenti|tuteli)\s+" + _TITLE + _NAME
    # v60d: the first person -- «Assisto X», «difendo X», «rappresento X», «seguo X», «il mio cliente è X»
    + r"|\b(?i:assisto|assistiamo|difendo|difendiamo|rappresento|rappresentiamo|tutelo|seguo|patrocino|agisco per(?: conto di)?)"
      r"\s+" + _TITLE + _NAME
    + r"|\b(?i:(?:il|la)\s+mi[oa]\s+(?:cliente|assistit[oa])|(?:il|la)\s+nostr[oa]\s+(?:cliente|assistit[oa]))"
      r"\s*(?i:è|e'|si chiama|:)?\s*" + _TITLE + _NAME)
# v60d: the client by the role the case gives it («difendo l'imputato», «assisto la persona offesa», «il mio cliente è
# il convenuto»): the party whose role holds the word
_ROLE_WORDS = (r"imputat[oa]|indagat[oa]|persona offesa|vittima|querelante|denunciante|attore|attrice|convenut[oa]|"
               r"ricorrente|resistente|appellante|appellat[oa]|locatore|locatrice|conduttore|conduttrice|venditore|"
               r"venditrice|acquirente|committente|appaltatore|datore di lavoro|lavorator[ea]|creditore|debitore|"
               r"erede|eredi|danneggiat[oa]|danneggiante|proprietari[oa]|condomin[oa]|amministratore|"
               # v60i (8): the roles a case names its parties by
               r"inquilin[oa]|affittuari[oa]|comproprietari[oa]|societ[àa]|aziend[ae]|ditt[ae]|impres[ae]|"
               r"vicin[oa]|consumator[ea]|fornitore|professionista|dipendente|compratore|compratrice|comodatari[oa]|"
               r"comodante|usufruttuari[oa]|mutuatari[oa]|legittimari[oa]|paziente|struttura|banca|"
               r"assicurazione|compagnia|condominio|comune|ente")
# v60i (8): the lawyer of a role -- «sono l'avvocato dell'inquilino», «come legale del proprietario» -- or, at the start
# of the message (a reply to «Chi assisti?»), «l'avvocato dell'inquilino» alone; inside a narrative («il legale del
# proprietario ha scritto») it is the other side's lawyer, never the client
_LAWYER_OF = (r"(?:il\s+|la\s+|l['’]\s*)?(?:legale|avvocat[oa]|difensore)\s+(?:del(?:la|lo|l['’])?|dei|degli|delle|di)")
_CLIENT_ROLE_RE = re.compile(
    r"(?:\b(?:assisto|assistiamo|difendo|difendiamo|rappresento|rappresentiamo|tutelo|seguo|patrocino|"
    r"(?:il|la)\s+mi[oa]\s+(?:cliente|assistit[oa])\s+(?:è|e'))"
    r"|\b(?:sono|sei|siamo|come|quale|in\s+qualit[àa]\s+di|nelle\s+vesti\s+di|assunte\s+le\s+vesti\s+di)\s+" + _LAWYER_OF
    + r")\s*(?:il|lo|la|l['’]|i|gli|le)?\s*(" + _ROLE_WORDS + r")\b",
    re.IGNORECASE)
_CLIENT_ROLE_BARE_RE = re.compile(r"^\W*" + _LAWYER_OF + r"\s*(?:il|lo|la|l['’]|i|gli|le)?\s*(" + _ROLE_WORDS
                                  + r")\b(?=\s*(?:[.,;:!?]|$))", re.IGNORECASE)
# v60i (8): a role word and the words that name the same party in the facts (its name or its role)
_ROLE_SYNONYMS = (
    (r"inquilin|conduttor|conduttric|affittuari|locatari", r"inquilin|conduttor|conduttric|affittuari|locatari"),
    (r"locator|locatric", r"locator|locatric"),
    (r"proprietari|comproprietari", r"proprietari"),
    (r"societ|aziend|ditt|impres", r"societ|aziend|ditt[ae]\b|impres|\bs\.?\s?r\.?\s?l\b|\bs\.?\s?p\.?\s?a\b|"
                                   r"\bs\.?\s?n\.?\s?c\b|\bs\.?\s?a\.?\s?s\b|\bsrl\b|\bspa\b"),
    (r"datore", r"datore|datrice|aziend|societ|ditt[ae]\b|impres"),
    (r"lavorator|dipendent", r"lavorator|dipendent|operai"),
    (r"acquirent|comprator|compratric", r"acquirent|comprator|compratric|consumator"),
    (r"venditor|venditric", r"venditor|venditric|alienant"),
)


def role_party(role: str, parties: Sequence[Dict[str, Any]], exclude: Sequence[str] = ()) -> str:
    """v60i (8): the one party a role word names -- its name or its role holds the word or a synonym of it
    (_ROLE_SYNONYMS: «l'inquilino» is the conduttore, «la società» the S.r.l.); "" when none or several."""
    low = _low(role)
    words = next((names for pattern, names in _ROLE_SYNONYMS if re.match(pattern, low)), re.escape(low[:6]))
    hits = [p["name"] for p in parties or [] if isinstance(p, dict) and p.get("name") and p["name"] not in exclude
            and re.search(words, f"{_low(p['name'])} | {_low(p.get('role'))}")]
    return hits[0] if len(set(hits)) == 1 else ""
_QUESTION_RE = re.compile(r"[^.?!\n]*\?")


def stated(text: str) -> Dict[str, str]:
    """{"client": the name the user says they assist, "client_role": the role word when they name it by its role,
    "angle": the last question the text asks, "date": a relative date resolved (fact_signals.relative_date)} (each ""
    if none)."""
    from . import fact_signals
    m = _CLIENT_STATED_RE.search(text or "")
    client = next((g for g in m.groups() if g), "") if m else ""
    role = ""
    if not client:
        r = _CLIENT_ROLE_RE.search(text or "") or _CLIENT_ROLE_BARE_RE.search((text or "").strip())
        role = r.group(1).lower() if r else ""
    questions = [" ".join(q.split()) for q in _QUESTION_RE.findall(text or "") if len(q.split()) >= 3]
    angle = re.sub(r"^(?:e|ma|quindi|dunque|allora)\s+", "", questions[-1], flags=re.IGNORECASE) if questions else ""
    return {"client": client, "client_role": role, "angle": angle[:1].upper() + angle[1:] if angle else "",
            "date": fact_signals.relative_date(text or "")}


_DECEASED_RE = re.compile(r"defunt|decedut|de cuius|\bmort[oa]\b|scompars[oa]\b|venut[oa] a mancare|ereditand",
                          re.IGNORECASE)


def deceased(facts: Dict[str, Any], case: str = "") -> List[str]:
    """v60d: the parties who died -- by their role («de cuius», «defunto», «deceduto») or by the case's words about them
    («la morte di X», «X è deceduto», «il defunto X»). Never a party to advise, never a client option."""
    out = []
    low = _low(case)
    for p in facts.get("parties") or []:
        if not isinstance(p, dict) or not str(p.get("name") or "").strip():
            continue
        name = str(p["name"]).strip()
        if _DECEASED_RE.search(str(p.get("role") or "")):
            out.append(name)
            continue
        n = re.escape(_low(name))
        if low and re.search(rf"(?:morte|decesso|scomparsa|defunt[oa]|de cuius|eredità|successione)\s+(?:del(?:la|lo)?\s+|di\s+)?"
                             rf"(?:sig\.(?:ra)?\s+|signor[ae]?\s+)?{n}\b|\b{n}\b[^.;]{{0,40}}?\b(?:è|era|sono)\s+"
                             rf"(?:mort[oa]|decedut[oa]|venut[oa] a mancare|scompars[oa])", low):
            out.append(name)
    return list(dict.fromkeys(out))


_RELATION_RE = re.compile(r"\b(?:figli[oae]?|coniug[ei]|moglie|marito|convivent[ei]|compagn[oa] di vita|unione civile|"
                          r"genitor[ei]|padre|madre|fratell[oi]|sorell[ae]|nipot[ei]|nonn[oi]|vedov[oa]|suocer[oa]|"
                          r"cognat[oa]|ered[ei]|legittimari[oa]|figliastr[oa])\b", re.IGNORECASE)


def relationships(case: str, top: int = 8) -> List[str]:
    """v60d: the case's sentences that state a relationship between persons (figli, coniuge, convivente ...), as the
    case writes them: facts the writer is given, never deduced."""
    out = []
    for sentence in re.split(r"(?<=[.;!?])\s+|\n+", case or ""):
        sentence = " ".join(sentence.split())
        if 8 <= len(sentence) <= 400 and _RELATION_RE.search(sentence) and sentence not in out:
            out.append(sentence)
    return out[:top]


RELATIONSHIPS_TITLE = "RAPPORTI TRA LE PERSONE (come il caso li dice)"
RELATIONSHIPS_RULE = ("Sono fatti del caso: tienine conto per le parti, la legittimazione e le pretese; non aggiungerne "
                      "altri. Chi è deceduto non è una parte e non è la parte assistita.")


def apply_stated(state: Dict[str, Any], said: Dict[str, str], renumber: Sequence[List[int]] = ()) -> None:
    """The client the user named is the client (no «Chi assisti» question); the question they asked is the angle: the
    topics it matches listed first and chosen ("procedi" takes them), every topic still listed."""
    name = said.get("client") or ""
    if not name and said.get("client_role"):        # v60d: by its role; v60i (8): its name or role, or a synonym
        facts0 = _facts(_restored_analysis(state["analysis"]))
        dead = set(deceased(facts0, case_text(state)))
        name = role_party(said["client_role"], facts0.get("parties") or [], list(dead))
        _log("client_by_role", {"role": said["client_role"], "party": name})
    if name:
        analysis = _restored_analysis(state["analysis"])
        facts = _facts(analysis)
        if facts:
            before = _derived(copy.deepcopy(facts), _areas(analysis))
            parties = facts.setdefault("parties", [])
            match = next((p for p in parties if p.get("name") and (_low(name) in _low(p["name"])
                                                                   or _low(p["name"]) in _low(name))), None)
            if match is None:
                match = {"name": name, "role": "", "age": ""}
                parties.append(match)
            _apply_answer(facts, {"kind": "party", "names": [match["name"]]}, match["name"])
            name = match["name"]
        analysis["parte_assistita"] = name
        state.setdefault("answered", {})["parte_assistita"] = name
        state["analysis"] = _jsonable(analysis)
        if facts:
            recompute(state, before)
        state["questions"] = [q for q in state.get("questions") or [] if q.get("id") != "parte_assistita"]
        _isolated("reside", lambda: reside(state, name, renumber))       # v60e: the client's side drives the topics
    angle = said.get("angle") or ""
    hits: List[int] = []
    if angle:
        state["angle"] = angle
        hits = _isolated("angle", lambda: match_angle(state, angle), []) or []
        topics = state.get("topics") or []
        if hits and _case_first_topics(topics):     # v60c: the angle orders last (topic_groups.order); procedi: immediate
            from . import topic_groups
            first = topics[hits[0] - 1]
            topic_groups.order(topics, angle=hits)
            state["angle_first"] = topics[0] is first
        elif hits:
            state["topics"] = [topics[i - 1] for i in hits] + [t for i, t in enumerate(topics, 1) if i not in hits]
            state["chosen"] = list(range(1, len(hits) + 1))
            state["angle_first"] = True
    _log("stated", {"client": name, "angle": angle, "angle_topics": hits})


def start(message: str, lang: str = "it", request: str = "") -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """A narrative case: (reply, interaction, state) of round 1. v60: the client and the question the user's own words
    state (the request when the case is an upload, else the message) are consumed (apply_stated)."""
    t0 = time.time()
    state = analyse_case(message, request, lang)
    said = stated(request or message)
    _isolated("stated", lambda: apply_stated(state, said))
    reply, field = round1(state)
    state["timings"]["round1_s"] = round(time.time() - t0, 1)
    field["seconds"] = state["timings"]["round1_s"]
    _log("round1", {"topics": len(state["topics"]), "questions": len(state["questions"]),
                    **{k: v for k, v in state["timings"].items() if v is not None}}, (time.time() - t0) * 1000)   # v60 (G)
    return reply, field, state


# --- v60k: single shot (CASE_FIRST_SINGLE_SHOT) ------------------------------------------------------------------------
# One message in, one complete parere out, the five sections: round 1 runs inside (first read, retrieval, grouping,
# records, demotion) and is never shown. The client from the message (stated, apply_stated), else assumed from the case
# (settle_client: SINGLE_SHOT_CLIENT, the parere's first line); its lines order the topics (reside). Every topic with
# graded rows is covered, the cap SINGLE_SHOT_ARTICLES. What the case does not state is «da verificare: il caso non dice
# …» where it counts; v60q: no «Punti da precisare» unless OPEN_POINTS_MAX says (open_points). A later message
# updates the parere from the cached analysis, the changed parts marked «(aggiornato)» (_single_update).

OTHER_PROFILES_MAX = 5          # v60r (2): the single-shot «Altri profili» lines, the best by score


def single_profiles(state: Dict[str, Any], numbers: Sequence[int], analysis: Dict[str, Any]) -> List[str]:
    """v60r (2): the single-shot «Altri profili» -- one line per topic not covered, its articles those inside the
    Sezione / Capo it is named by (state["units"]: an article known to lie outside it leaves the line; its anchor, the
    first left), past the code-area gate (an article of a code outside the frame's area only when admitted: area_gate's
    log); a line left with none goes. Ordered by score (the best grade, then the graded rows) and cut at
    OTHER_PROFILES_MAX. Logged with what each check took out."""
    units = state.get("units") or {}
    gate = state.get("area_gate") or {}
    families = gate.get("families") if "families" in gate else frame_families(analysis)
    admitted = {x.get("article") for x in gate.get("admitted") or []}

    def in_area(label: str) -> bool:
        family = CODE_FAMILY.get(_key(_parse_label(label))[1])
        return not family or not families or family in families or label in admitted
    log: Dict[str, Any] = {"outside_unit": {}, "outside_area": {}, "dropped": [], "reanchored": {}}
    items = []
    for i, t in enumerate(state.get("topics") or [], 1):
        if i in numbers:
            continue
        arts = [_label(_key(r)) for r in _topic_refs(t)]
        unit = str(t.get("unit") or "") if t.get("source") in ("sezione", "capo") else ""
        inside = [a for a in arts if not unit or a not in units or unit in units[a]]
        if len(inside) < len(arts):
            log["outside_unit"][t.get("title") or str(i)] = [a for a in arts if a not in inside]
        kept = [a for a in inside if in_area(a)]
        if len(kept) < len(inside):
            log["outside_area"][t.get("title") or str(i)] = [a for a in inside if a not in kept]
        if not kept:
            log["dropped"].append(t.get("title") or str(i))
            continue
        if arts and kept[0] != arts[0]:
            log["reanchored"][t.get("title") or str(i)] = kept[0]
        items.append(((int(t.get("best_grade") or 0), int(t.get("graded") or 0)), i, t, kept))
    items.sort(key=lambda x: (-x[0][0], -x[0][1], x[1]))
    log["listed"] = [{"topic": t.get("title"), "score": list(score), "articles": kept}
                     for score, _, t, kept in items[:OTHER_PROFILES_MAX]]
    log["cut"] = [t.get("title") for _, _, t, _ in items[OTHER_PROFILES_MAX:]]
    _log("single_shot_profiles", log)
    lines = [C.fill(C.OTHER_PROFILE_LINE_TITLE, title=t["title"], articles=_articles_line({"articles": kept}))
             for _, _, t, kept in items[:OTHER_PROFILES_MAX]]
    return [x for x in lines if x]


# v60r (4): consecutive «(da verificare…)» markers -- the writer's, conditional_on_unknown's, the record check's -- one
_MARKER_RUN_RE = re.compile(r"(?:[ \t]*\(da verificare(?::[^()]*)?\)){2,}", re.IGNORECASE)
_MARKER_RE = re.compile(r"\(da verificare(?::\s*([^()]*))?\)", re.IGNORECASE)


def merged_markers(text: str, log: Optional[List[str]] = None) -> str:
    """The text with each run of adjacent «(da verificare…)» markers as one, its contents deduplicated («; »)."""
    def one(m: "re.Match") -> str:
        items = list(dict.fromkeys(" ".join(x.split()) for x in _MARKER_RE.findall(m.group(0)) if x.strip()))
        lead = m.group(0)[:len(m.group(0)) - len(m.group(0).lstrip())]
        if log is not None:
            log.append(" ".join(m.group(0).split())[:200])
        return lead + (f"(da verificare: {'; '.join(items)})" if items else "(da verificare)")
    return _MARKER_RUN_RE.sub(one, text or "")


def covered_topics(state: Dict[str, Any]) -> List[int]:
    """The topics the single-shot answer covers. v60n: at most SINGLE_SHOT_TOPICS, those holding an article the user's
    text names, the main-line hypothesis or a row graded 3 (in that priority), in their order; none: the immediate
    answer's (default_choice). The rest, one line each under «Altri profili» (_write_answer). Logged. v60q: the
    criteria order every topic -- a named article, the main line, a row graded 3, then the topics meeting none, each
    group in the topics' order -- and the answer covers the first SINGLE_SHOT_TOPICS of that order, in that order: at
    least SINGLE_SHOT_TOPICS_MIN where as many exist (logged with the order)."""
    from . import defensive_generation as dg
    named = {_key(r) for r in dg._refs_in(case_text(state)) if len(r) == 2}
    why: Dict[int, List[str]] = {}
    for i, t in enumerate(state.get("topics") or [], 1):
        reasons = []
        if named & {_key(_parse_label(a)) for a in t.get("articles") or []}:
            reasons.append("articolo nominato")
        if any(h.get("line") == "principale" for h in t.get("hypotheses") or []):
            reasons.append("linea principale")
        if any(isinstance(g, int) and g >= 3 for g in (t.get("grades") or {}).values()):
            reasons.append("grado 3")
        if reasons:
            why[i] = reasons
    rank = ("articolo nominato", "linea principale", "grado 3")
    count = len(state.get("topics") or [])
    order = sorted(range(1, count + 1), key=lambda i: (min(rank.index(r) for r in why[i]) if i in why else len(rank), i))
    chosen = order[:SINGLE_SHOT_TOPICS] or default_choice(state)[:SINGLE_SHOT_TOPICS]
    floor = min(SINGLE_SHOT_TOPICS_MIN, count)
    _log("single_shot_topics", {"covered": chosen, "order": order,
                                "why": {str(i): why.get(i) or ["nessun criterio: per ordine"] for i in chosen},
                                "rest": [i for i in range(1, count + 1) if i not in chosen],
                                "beyond_cap": sorted(set(why) - set(chosen)),
                                "floor": {"min": floor, "ok": len([i for i in chosen if 0 < i <= count]) >= floor}})
    return chosen


def single_shot(message: str, lang: str = "it", request: str = "") -> Dict[str, Any]:
    """v60k: the complete parere for a narrative case -- {"answer", "interaction", "state", "citations", "kind"}, as
    follow_up's. Round 1 internally (analyse_case), the client the message states (apply_stated) or the one the answer
    assumes, whose side orders the topics (prepare_side); every graded topic answered (write_answer); timed
    (single_shot_timings)."""
    t0 = time.time()
    state = analyse_case(message, request, lang)
    state["mode"] = "single_shot"
    said = stated(request or message)
    _isolated("stated", lambda: apply_stated(state, said))
    if not (state.get("answered") or {}).get("parte_assistita"):
        _isolated("prepare_side", lambda: prepare_side(state))       # the assumed client's lines order the topics
    numbers = covered_topics(state)
    state["chosen"] = numbers
    t1 = time.time()
    state["state"] = FRAME_CONFIRMED
    text, citations = write_answer(state, numbers)
    state["state"] = ANSWERED
    state["last_answer"], state["last_snapshot"] = text, _snapshot(state)
    touch(state, "single shot")
    timings = single_shot_timings(state, t0, t1)
    _log("single_shot_funnel", single_shot_funnel(state, numbers))
    field = interaction(state, 3, ("deepen", "add_fact"))
    field["seconds"] = timings["total_s"]
    articles = _isolated("article_citations", lambda: article_citations(state, text), []) or []     # v60q
    return {"answer": text, "interaction": field, "state": state, "citations": citations, "kind": "answer",
            "article_citations": articles}


def article_citations(state: Dict[str, Any], text: str) -> List[Dict[str, Any]]:
    """v60q (single shot): the articles the parere cites, for the frontend to show on demand -- the body cites them as
    references and never quotes them. v60r (1): with the record's rows the body no longer prints (ledger_out). [{"id":
    "1062 c.c.", "number", "code", "rubrica", "commi": [{"n", "text"}] (none when its text was not read), "status" (its
    verdict, or its line: «non applicabile», «non trattata»), "reason", "in_body", "rows": [{"article", "comma",
    "requisite", "status", "supporting_sentence", "contradicting_sentence"}]}], in the order the parere first cites them,
    then the ones only the rows hold; an article with neither text nor rows is not listed. The record check reads the
    same texts server-side (check_context). Logged."""
    from . import article_text, citations, rule_quote
    rows = _restored_rows(state.get("rows")) + _restored_rows(state.get("tool_rows"))
    by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in rows:
        if row.get("_article"):
            by_key.setdefault(_key(row["_article"]), row)
    documents = ref_documents(list(by_key), rows)
    ledger: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in state.get("ledger_rows") or []:
        ledger.setdefault(_key(_parse_label(r["article"])), []).append(r)
    order = list(dict.fromkeys([_key(r) for r in citations.refs_in(text or "", documents) if len(r) == 2] + list(ledger)))
    out = []
    for key in order:
        row = by_key.get(key)
        commi = rule_quote.row_commi(row) if row else []
        mine = ledger.get(key) or []
        if not commi and not mine:
            continue
        head = next((r for r in mine if r.get("requisite") is None and r.get("comma") is None), {})
        rubrica = rule_quote.clean_rubrica(str(article_text.heading(row.get("s")) or "")).rstrip(" .") if row else ""
        out.append({"id": _label(key), "number": key[0], "code": key[1], "rubrica": rubrica,
                    "commi": [{"n": k, "text": t} for k, t in commi], "status": head.get("status") or "",
                    "reason": head.get("reason") or "", "in_body": bool(head.get("in_body", True)),
                    "rows": [{k: r.get(k) for k in ("article", "comma", "requisite", "status", "supporting_sentence",
                                                    "contradicting_sentence")} for r in mine if r is not head]})
    _log("article_citations", {"articles": [a["id"] for a in out], "rows": sum(len(a["rows"]) for a in out),
                               "chars": sum(len(c["text"]) for a in out for c in a["commi"])})
    return out


def single_shot_timings(state: Dict[str, Any], t0: float, t1: float) -> Dict[str, Any]:
    """[VERBOSE:interactive_single_shot_timings]: round 1's steps and stages, the answer's, the total against
    SINGLE_SHOT_TARGET_S."""
    timings = state.setdefault("timings", {})
    now = time.time()
    out = {"round1_s": round(t1 - t0, 1), "answer_s": round(now - t1, 1), "total_s": round(now - t0, 1),
           "target_s": SINGLE_SHOT_TARGET_S,
           "round1": {k: timings.get(k) for k in ("analysis_s", "retrieval_s", "first_read_s", "first_read_wait_s",
                                                  "grading_s", "topics_s", "records_s") if timings.get(k) is not None},
           "round1_stages": timings.get("round1_stages") or {},
           "answer": {k: v for k, v in timings.items() if k.startswith(("answer_", "records_link")) and
                      not isinstance(v, dict)},
           "answer_stages": timings.get("answer_stages") or {}}
    out["over_target"] = out["total_s"] > SINGLE_SHOT_TARGET_S
    timings["single_shot_s"] = out["total_s"]
    state["single_shot_timings"] = out
    _log("single_shot_timings", out)
    return out


def single_shot_funnel(state: Dict[str, Any], numbers: Sequence[int]) -> Dict[str, Any]:
    """The answer's funnel: topics listed and covered, the articles by their bucket, the open points asked."""
    from collections import Counter
    buckets = Counter((rec.get("answer") or {}).get("bucket") for rec in (state.get("records") or {}).values()
                      if rec.get("answer"))
    points = state.get("open_points") or {}
    return {"topics": len(state.get("topics") or []), "covered": list(numbers),
            "articles": {k: v for k, v in buckets.items() if k}, "open_points": len(points.get("points") or []),
            "asked": len(points.get("asked") or []), "dropped": len(points.get("dropped") or [])}


_TERM_CHANGE_RE = re.compile("^" + re.sub(r"\\\{\w+\\\}", ".+", re.escape(C.CHANGE_TERM)) + "$")
_RULE_CHANGE_RE = re.compile("^(?:" + "|".join(re.sub(r"\\\{\w+\\\}", ".+", re.escape(t))
                                               for t in (C.CHANGE_RULE_ON, C.CHANGE_RULE_OFF)) + ")$")


def open_points_max() -> int:
    """v60q: the «Punti da precisare» the single-shot answer asks -- OPEN_POINTS_MAX (the environment's, a number of 0 or
    more), default 0: no block."""
    try:
        return max(0, int(os.getenv(OPEN_POINTS_ENV) or OPEN_POINTS_MAX))
    except ValueError:
        return OPEN_POINTS_MAX


def open_points(state: Dict[str, Any], refs: Sequence[tuple]) -> Dict[str, Any]:
    """v60k: the facts the case does not state (each open question's: open_point_fact) and the ones asked in «Punti da
    precisare» -- at most open_points_max() (v60q: default 0, no block), each kept only when (a) its answer changes a term (its trial changes
    one: unlocks), a procedibilità verdict (it asks a d'ufficio condition of a planned offence) or an article's verdict
    (it asks a requisite of a planned article, or flips one or a rule), (b) it is specific to a named fact -- a generic
    rule question (generic_question) only when a computed term depends on it and the case states no date --, (c) no
    case sentence already answers it (asked_in_case). Specific before generic, then the most it changes. Logged
    (open_points | asked | dropped with the reason); in the state; {"points", "asked", "dropped", "block"}. v60l: a
    point is a fact a client can state -- a legal characterisation (qualification: the first read's, dropped when the
    questions were built, state["qualifications"], or any open question's) is never one, dropped "qualificazione"."""
    from . import fact_signals, issue_records
    labels = {issue_records._label(r) for r in refs}
    records = state.get("records") or {}
    by_q: Dict[str, set] = {}
    for label in labels:
        rec = records.get(label) or {}
        for r in rec.get("requisites") or []:
            if r.get("question") and r.get("status") == DA_PROVARE:
                by_q.setdefault(str(r["question"]), set()).add("esito")
        for c in rec.get("procedibilita") or []:
            if c.get("question") and c.get("status") == DA_PROVARE:
                by_q.setdefault(str(c["question"]), set()).add("procedibilità")
    sentences = fact_signals.case_sentences(case_text(state))
    dated = any(_VALUE_RE.search(x) or fact_signals.relative_date(x) for x in sentences)
    names = party_names_of(state)
    points, kept, dropped = [], [], []
    dropped += [{"point": " ".join(str(x).split())[:160], "reason": "qualificazione"}
                for x in state.get("qualifications") or [] if str(x).strip()]
    answered = set(state.get("answered") or {})
    dropped += [{"question": q["id"], "reason": "qualificazione"} for q in state.get("questions") or []
                if q.get("id") not in answered and q.get("id") != "parte_assistita" and qualification(q)]
    client = client_of(state)           # v60q: a party's state of mind -- argued, never a point
    dropped += [{"point": " ".join(str(x).split())[:160], "reason": "stato soggettivo"}
                for x in state.get("states_of_mind") or [] if str(x).strip()]
    dropped += [{"question": q["id"], "reason": "stato soggettivo"} for q in state.get("questions") or []
                if q.get("id") not in answered and q.get("id") != "parte_assistita" and not qualification(q)
                and state_of_mind(q, client, names)]
    for i, q in enumerate(open_points_list(state)):
        fact = open_point_fact(q, names)
        if fact:
            points.append({"question": q["id"], "point": C.fill(C.OPEN_POINT, fact=fact)})
        kinds = set(by_q.get(str(q.get("id"))) or ())
        unlocks = [str(u) for u in q.get("unlocks") or []]
        if any(_TERM_CHANGE_RE.match(u) for u in unlocks):
            kinds.add("termine")
        if any(u.startswith("requisito:") or _RULE_CHANGE_RE.match(u) for u in unlocks):
            kinds.add("esito")
        if not kinds:
            dropped.append({"question": q["id"], "reason": "la risposta non cambia termini, procedibilità né esiti"})
            continue
        said = asked_in_case(q, sentences, names)
        if said:
            dropped.append({"question": q["id"], "reason": f"nel caso: «{said[:120]}»"})
            continue
        generic = generic_question(q)
        if generic and "termine" not in kinds:
            dropped.append({"question": q["id"], "reason": "generica: nessun termine calcolato ne dipende"})
            continue
        if generic and dated:
            dropped.append({"question": q["id"], "reason": "generica: il caso indica una data"})
            continue
        kept.append((generic, -len(kinds), i, q, sorted(kinds)))
    kept.sort(key=lambda x: x[:3])
    most = open_points_max()            # v60q: default 0 -- no block; the points stay «da verificare» in the parere
    asked = kept[:most]
    beyond = f"oltre i {most} punti" if most else "nessun blocco (OPEN_POINTS_MAX 0): «da verificare» nel parere"
    dropped += [{"question": q["id"], "reason": beyond} for *_, q, _ in kept[most:]]
    items = [C.fill(C.OPEN_POINT_ITEM, n=n, question=q.get("text")) for n, (*_, q, _) in enumerate(asked, 1)]
    items = [x for x in items if x]
    block = f"**{C.OPEN_POINTS_TITLE}**\n\n" + "\n".join(items) if items else ""
    out = {"points": points, "asked": [{"question": q["id"], "changes": kinds} for *_, q, kinds in asked],
           "dropped": dropped, "block": block}
    state["open_points"] = {k: v for k, v in out.items() if k != "block"}
    _log("open_points", state["open_points"])
    return out


# v60k: §4's own wording of a fact the case does not state -- the computed terms' «DA VERIFICARE[, …]: il caso non
# indica …» and the procedure's «Data del fatto: non nota.» -- in the single-shot form (C.OPEN_POINT)
_TERM_UNKNOWN_RE = re.compile(r"\bda verificare(?:,[^:\n]*)?:\s*il caso non indica\b", re.IGNORECASE)
_DATE_UNKNOWN_RE = re.compile(r"\bData del fatto: non nota\.")


def open_points_wording(rendered: str) -> str:
    """The rendered procedural section with its open points as «da verificare: il caso non dice …»."""
    lead = C.OPEN_POINT.split("{fact}", 1)[0].rstrip()                 # «da verificare: il caso non dice»
    out = _TERM_UNKNOWN_RE.sub(lead, rendered or "")
    date = C.fill(C.OPEN_POINT, fact=C.OPEN_POINT_FACTS.get("data_fatto") or "la data del fatto")
    return _DATE_UNKNOWN_RE.sub((C.capital(date) + ".") if date else "Data del fatto: non nota.", out)


def _snapshot(state: Dict[str, Any]) -> Dict[str, Any]:
    """What an update compares: each record's verdict and each requisite's status and open question (an answered point
    changes its status line), each term's status and article."""
    analysis = _restored_analysis(state["analysis"]) if state.get("analysis") else {}
    return {"records": {label: [rec.get("verdict") or "", [[r.get("status"), r.get("question") or ""]
                                                           for r in rec.get("requisites") or []]]
                        for label, rec in (state.get("records") or {}).items()},
            "terms": {str(t.get("id")): [t.get("status") or "", str(t.get("ref") or "")]
                      for t in _signals(analysis).get("terms") or []}}


def changed_labels(before: Dict[str, Any], after: Dict[str, Any]) -> set:
    """The articles an update changed: a record's verdict or a requisite's status, a term's status (its article)."""
    out = {label for label, v in (after.get("records") or {}).items() if (before.get("records") or {}).get(label) != v}
    for tid, (status, ref) in (after.get("terms") or {}).items():
        if ((before.get("terms") or {}).get(tid) or [None])[0] != status:
            m = re.search(r"(\d+(?:-[a-z]+)?)\s+(.+)$", re.sub(r"^\s*artt?\.\s*", "", ref))
            if m:
                out.add(_label(_key((m.group(1), m.group(2).strip()))))
    return out


_SECTION_HEAD_RE = re.compile(r"^\*\*\d+\.\s+(.+?)\*\*\s*$")


def _bare(text: str) -> str:
    return " ".join(str(text or "").replace(C.UPDATED_MARK.strip(), " ").split()).strip()


def mark_updated(text: str, old: str, labels: set, words: str, log: Optional[Dict[str, Any]] = None) -> str:
    """v60k: the parts of an updated parere that changed, each marked «(aggiornato)» (C.UPDATED_MARK): a part is one the
    old parere does not hold (word for word) and that the update touches -- in «Profili processuali» (the code's) any
    new line; elsewhere a §2 block of a changed article (labels), a paragraph or line citing one, or holding two of the
    new facts' words. Section titles, the client line and «Punti da precisare» never."""
    if not old:
        return text
    from . import citations
    old_lines = {_bare(x) for x in old.split("\n") if x.strip()}
    old_paras = {_bare(x) for x in re.split(r"\n\s*\n", old) if x.strip()}
    new_words = _stems(words)
    section, article, n = "", "", 0

    def touched(chunk: str) -> bool:
        cited = {citations.label(r) for r in citations.refs_in(chunk)}
        return article in labels or bool(cited & labels) or len(_stems(chunk) & new_words) >= 2

    out = []
    for chunk in re.split(r"(\n\s*\n)", text):
        if not chunk.strip() or re.fullmatch(r"\n\s*\n", chunk):
            out.append(chunk)
            continue
        first = chunk.strip().split("\n")[0].strip()
        head = _SECTION_HEAD_RE.match(first)
        if head:
            section, article = head.group(1).strip(), ""
        elif first == f"**{C.OPEN_POINTS_TITLE}**":
            section, article = C.OPEN_POINTS_TITLE, ""
        if head or section == C.OPEN_POINTS_TITLE or not section:
            out.append(chunk)
            continue
        lines = chunk.split("\n")
        if any(x.lstrip().startswith(("- ", "**")) for x in lines):
            for j, line in enumerate(lines):
                h = _NORM_HEADING_RE.match(line.strip())
                if h:
                    article = f"{h.group(1)} {_code_label(h.group(2).strip())}"
                    if article in labels and _bare(chunk) not in old_paras:
                        lines[j], n = line.rstrip() + C.UPDATED_MARK, n + 1
                elif line.strip() and _bare(line) not in old_lines and (section == ANSWER_TITLES[3] or touched(line)):
                    lines[j], n = line.rstrip() + C.UPDATED_MARK, n + 1
            out.append("\n".join(lines))
        elif _bare(chunk) not in old_paras and (section == ANSWER_TITLES[3] or touched(chunk)):
            out.append(chunk.rstrip() + C.UPDATED_MARK + chunk[len(chunk.rstrip()):])
            n += 1
        else:
            out.append(chunk)
    if log is not None:
        log.update(marked=n, labels=sorted(labels))
    return "".join(out)


def _reply_words(parsed: Dict[str, Any], angle: str = "") -> str:
    """The words of what a reply adds: its answers, statements, corrections and angle."""
    def flat(value):
        if isinstance(value, dict):
            return " ".join(flat(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return " ".join(flat(v) for v in value)
        return str(value or "")
    return " ".join(flat(parsed.get(k)) for k in ("answers", "statements", "corrections")) + " " + (angle or "")


def _single_update(message: str, state: Dict[str, Any], parsed: Dict[str, Any], t0: float) -> Dict[str, Any]:
    """v60k: a message after a single-shot parere -- answers, a new fact, «approfondisci X», «e se…» (the reply
    interpreter's reading; a message it reads nothing in is an angle) -- the parere written again from the cached
    analysis, every topic it covered plus the ones the message names, the changed parts marked (mark_updated)."""
    from . import issue_records
    intents = parsed.get("intents") or []

    def out(answer, kind, citations=None):
        field = interaction(state, 3, ("deepen", "add_fact"))
        field["seconds"] = round(time.time() - t0, 1)
        state.setdefault("timings", {})[f"{kind}_s"] = field["seconds"]
        articles = (_isolated("article_citations", lambda: article_citations(state, answer), []) or []) \
            if kind == "regenerate" else []                                                         # v60q
        return {"answer": answer, "interaction": field, "state": state, "citations": citations or [], "kind": kind,
                **({"article_citations": articles} if articles else {})}
    gives = bool(parsed.get("answers") or parsed.get("corrections") or parsed.get("statements") or parsed.get("client"))
    if "domanda_flusso" in intents and len(intents) == 1 and not gives:
        touch(state, "flow question")
        return out(C.SINGLE_SHOT_HELP, "flow_help")
    if "documento" in intents and not gives:
        return {**out("", "document_request"), "document": True}
    before = state.get("last_snapshot") or _snapshot(state)
    if gives:
        apply_reply(state, parsed)
    angle = str(parsed.get("angle") or "")
    if not (gives or parsed.get("topics") or angle):
        angle = " ".join(message.split())[:300]           # «e se…», «approfondisci il possesso»: the angle
    hits = match_angle(state, angle) if angle else []
    if angle:
        state["angle"] = angle
    deepened = list(dict.fromkeys(list(parsed.get("topics") or []) + hits))
    numbers = list(dict.fromkeys(list(state.get("chosen") or covered_topics(state)) + deepened))
    numbers = [n for n in numbers if 0 < n <= len(state.get("topics") or [])]
    state["chosen"] = numbers
    state["state"] = FRAME_CONFIRMED
    text, citations = write_answer(state, numbers)
    state["state"] = ANSWERED
    after = _snapshot(state)
    labels = changed_labels(before, after) | {issue_records._label(r) for n in deepened
                                              for r in _topic_refs(state["topics"][n - 1])}
    log: Dict[str, Any] = {}
    text = mark_updated(text, state.get("last_answer") or "", labels, _reply_words(parsed, angle), log)
    state["last_answer"], state["last_snapshot"] = text, after
    state["pending_regenerate"] = []
    touch(state, "single shot update")
    _log("single_shot_update", {"intents": intents, "gave": gives, "angle": angle, "deepened": deepened, **log})
    return out(text, "regenerate", citations)


def follow_up(message: str, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """A reply within the interaction: {"answer", "interaction", "state", "citations", "kind"}, or None when this
    message is not one (a new case, or a question the interaction does not hold) -- the old path then."""
    t0 = time.time()
    phase = state.get("state")
    if is_new_case(message):
        return None
    parsed = parse_reply(message, state)
    parsed = reply_stated(message, state, parsed)         # v60d: the client and a relative date, from the words
    intents = parsed.get("intents") or []
    _log("reply", {"phase": phase, "how": parsed["how"], "intents": intents, "topics": parsed["topics"],
                   "confirm": parsed["confirm"], "answers": list(parsed["answers"]), "angle": parsed["angle"],
                   "corrections": len(parsed["corrections"]), "statements": len(parsed["statements"])})
    if single_mode(state) and phase == ANSWERED:           # v60k: the parere again, its changed parts marked
        return _single_update(message, state, parsed, t0)
    rnd = 1 if phase == TOPICS_OFFERED else 3 if phase == ANSWERED else 2

    def out(answer, field, kind, citations=None):
        field["seconds"] = round(time.time() - t0, 1)
        state.setdefault("timings", {})[f"{kind}_s"] = field["seconds"]
        return {"answer": answer, "interaction": field, "state": state, "citations": citations or [], "kind": kind}

    def answer(numbers, kind):
        state["state"] = FRAME_CONFIRMED
        text, citations = write_answer(state, numbers)
        state["state"] = ANSWERED
        state["pending_regenerate"] = []
        touch(state, kind)
        return out(text, interaction(state, 3, ("deepen", "add_fact")), kind, citations)

    def clarify():          # v60 (H): one clarifying question, never a fresh round 1
        question = {1: C.CLARIFY_TOPICS, 2: C.CLARIFY_CONFIRM, 3: C.CLARIFY_ANSWERED}[rnd]
        touch(state, "unclear")
        return out(C.fill(C.UNCLEAR, next=question) or question,
                   interaction(state, rnd, ("choose_topic", "proceed") if rnd < 3 else ("deepen", "add_fact")), "unclear")
    if "nuovo_caso" in intents and len(intents) == 1:
        return clarify()                # a new case is a narrative (is_new_case, above); a short message is none
    if "domanda_flusso" in intents and len(intents) == 1:
        touch(state, "flow question")
        return out(C.FLOW_HELP, interaction(state, rnd, ("choose_topic", "proceed") if rnd < 3 else ("deepen", "add_fact")),
                   "flow_help")
    if "documento" in intents and not (parsed["answers"] or parsed["corrections"] or parsed["statements"]
                                       or parsed.get("client")):
        if phase == ANSWERED:           # the API wraps the last topic answer (api._interactive_document)
            return {**out("", interaction(state, 3, ("deepen", "add_fact")), "document_request"), "document": True}
        if phase == TOPICS_OFFERED and not state.get("chosen"):
            if not parsed["topics"]:
                prepare_side(state)         # v60e: the client's side orders the topics the answer takes
            state["chosen"] = parsed["topics"] or default_choice(state)
        return {**answer(parsed["topics"] or state.get("chosen") or default_choice(state), "answer"), "document": True}
    changes = []
    if parsed["answers"] or parsed["corrections"] or parsed["statements"] or parsed.get("client"):
        changes = apply_reply(state, parsed)
    if parsed["angle"]:
        hits = match_angle(state, parsed["angle"])
        state["angle"] = parsed["angle"]
        _log("angle", {"angle": parsed["angle"], "topics": hits})
        parsed["topics"] = list(dict.fromkeys(list(parsed["topics"]) + hits))
    if parsed["topics"]:
        state["chosen"] = parsed["topics"]
    if phase == ANSWERED:
        if parsed["topics"] and not changes:
            return answer(parsed["topics"], "deepen")
        answered = bool(parsed["answers"] or parsed["corrections"] or parsed["statements"] or parsed.get("client"))
        if (changes or answered) and (parsed["confirm"] or "documento" in intents):
            return answer(list(state.get("chosen") or [1]), "regenerate")      # v60e: it asks for the answer too
        if changes or answered:
            state["pending_regenerate"] = list(state.get("chosen") or [1])
            touch(state, "new fact")
            text = restatement(state, phase_answered=True)       # v60e: one behaviour, the restatement
            return out(text, interaction(state, 3, ("regenerate", "deepen")), "new_fact")
        if parsed["confirm"] and state.get("pending_regenerate"):
            return answer(state["pending_regenerate"], "regenerate")
        return clarify()                # v60 (H): never the old path (a fresh round 1) inside the interaction
    if parsed["confirm"]:
        if phase == TOPICS_OFFERED and not state.get("chosen"):
            prepare_side(state)             # v60e
            state["chosen"] = default_choice(state)
        return answer(state.get("chosen") or default_choice(state), "answer")
    if not (changes or parsed["answers"] or parsed["corrections"] or parsed["statements"] or parsed["topics"]
            or parsed["angle"] or parsed.get("client")):
        return clarify()
    touch(state, "round 2")
    text, field = round2(state)
    if parsed["answers"] or parsed["corrections"] or parsed["statements"] or parsed.get("client"):
        text = restatement(state)           # v60e: a reply that answers questions: the restatement, one behaviour
    return out(text, field, "round2")
