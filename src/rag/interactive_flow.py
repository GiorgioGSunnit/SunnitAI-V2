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

Every step is isolated (_isolated): a failure is logged with its traceback (interactive_step_failed) and the turn goes
on without it. The API response carries `interaction` (interaction()): {schema, version, state, round, topics, facts,
questions, chosen, changes, actions}.
"""

import copy
import json
import logging
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


def capped_refs(topics: Sequence[Dict[str, Any]], cap: int = None) -> List[Tuple[str, str]]:
    """v57 (C: v55 h5 «2, 3, 5» -> 19 extras): the chosen topics' articles, at most TOPIC_ARTICLES across them -- every
    topic's matched article(s) first, in the topics' order, then the topics' other articles taken in turn."""
    cap = TOPIC_ARTICLES if cap is None else cap
    out: List[Tuple[str, str]] = []
    seen: set = set()

    def put(ref):
        if len(out) < cap and _key(ref) not in seen:
            seen.add(_key(ref))
            out.append(ref)
    for t in topics:
        for a in [t.get("anchor") or ""] + list(t.get("matched") or []):
            if a and _parse_label(a) in _topic_refs(t, all_=True):
                put(_parse_label(a))
    rest = [[r for r in _topic_refs(t) if _key(r) not in seen] for t in topics]
    for i in range(max((len(x) for x in rest), default=0)):
        for x in rest:
            if i < len(x):
                put(x[i])
    return out


def _topic_refs(topic: Dict[str, Any], all_: bool = False) -> List[Tuple[str, str]]:
    """A topic's planned articles (its first TOPIC_ARTICLES; all_: every one listed)."""
    arts = topic.get("articles") or []
    cut = min(TOPIC_ARTICLES, int(topic.get("planned") or len(arts)))       # v57: never the listed-only tail
    return [_parse_label(a) for a in (arts if all_ else arts[:cut])]


def _retrieve(text: str, request: str, lang: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Retrieval as _case_analysis_pipeline runs it, up to its pool texts, then the topics and their articles' texts:
    {"article_rows", "regime_rows", "topics", "topic_rows"}."""
    import neo4j
    from . import (article_text, comma_material, defensive_generation as dg, pool_gates, query_router, source_tiers,
                   topic_step)
    from .doc_lookup import _fetch_doc_names
    from .main import NEO4J_DATABASE, driver

    case = dg._case_text(text, request)
    route = query_router.route_for_pipeline(case, analysis)
    analysis["route"] = route["route"]
    hybrid = source_tiers.hybrid_enabled(route["route"])
    v2 = feature_flags.enabled(dg.CASE_ANALYSIS_FLAG)
    fusion = v2 and feature_flags.enabled(dg._CASE_POOL_FUSION_FLAG)
    store_ = article_text.ArticleStore()
    out: Dict[str, Any] = {"article_rows": [], "regime_rows": [], "topics": [], "others": [], "topic_rows": []}
    documents = _fetch_doc_names(driver, NEO4J_DATABASE)
    with driver.session(database=NEO4J_DATABASE, default_access_mode=neo4j.READ_ACCESS) as session:
        select = dg._select_articles_fusion if fusion else (dg._select_articles_v2 if v2 else dg._select_articles)
        picked = select(session, driver, NEO4J_DATABASE, text, request, analysis, store=store_)
        picked = dg._add_method_candidates(session, driver, NEO4J_DATABASE, analysis, picked, store=store_)
        picked = dg._add_regime_candidates(session, driver, NEO4J_DATABASE, analysis, picked, store=store_)
        regime_rows = dg._regime_rows(session, driver, NEO4J_DATABASE, analysis, store=store_)
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
        out.update(article_rows=article_rows, regime_rows=regime_rows)
        links = topic_step.GraphLinks(session, documents)
        others: List[Dict[str, Any]] = []
        topics = _isolated("topic_step", lambda: topic_step.build(analysis, article_rows + regime_rows, links, case,
                                                                   others=others), [])
        out["topics"], out["others"] = topics, others
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
        _isolated("fill_texts", lambda: article_text.fill_texts(session, rows + regime_rows, store_))
        if comma_material.enabled():
            _isolated("commi", lambda: comma_material.attach(session, rows, "strategy", store=store_))
        out["topic_rows"] = rows
    return out


def analyse_case(text: str, request: str = "", lang: str = "it") -> Dict[str, Any]:
    """Step 1: the analysis object, as a new state (TOPICS_OFFERED is set by round1)."""
    from . import defensive_generation as dg, render_computed
    t0 = time.time()
    analysis = dg._analyse_case_for_pipeline(text, lang, request)
    analysis[CASE_KEY] = dg._case_text(text, request)
    _isolated("unstated_rules", lambda: mark_unstated(analysis))
    t1 = time.time()
    found = _isolated("retrieval", lambda: _retrieve(text, request, lang, analysis),
                      {"article_rows": [], "regime_rows": [], "topics": [], "others": [], "topic_rows": []})
    t2 = time.time()
    rendered = _isolated("render_computed", lambda: render_computed.render(analysis, party(analysis)), "") or ""
    state = {
        "schema": SCHEMA, "version": 0, "state": "", "created": _now().isoformat(), "lang": lang,
        "text": text, "request": request, "additions": [],
        "analysis": _jsonable({**{k: v for k, v in analysis.items() if k != "topics"},
                               CASE_KEY: dg._case_text(text, request)}),
        "topics": _jsonable(found["topics"]), "others": _jsonable(found.get("others") or []),
        "rows": [_compact_row(r) for r in found["topic_rows"]],
        "tool_rows": [_compact_row(r) for r in found["regime_rows"]],
        "pool": [{"_article": list(r["_article"]), "_multi_signal": bool(r.get("_multi_signal")),
                  "rubrica": str((r.get("s") or {}).get("titolo_articolo") or "")}
                 for r in found["article_rows"] if r.get("_article")],
        "rendered": rendered, "answered": {}, "chosen": [], "angle": "", "changes": [], "answers": {},
        "timings": {"analysis_s": round(t1 - t0, 1), "retrieval_s": round(t2 - t1, 1)}}
    state["questions"] = _isolated("questions", lambda: questions(_restored_analysis(state["analysis"]), (),
                                                                  state["topics"], state["rows"]), [])
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
        text = C.fill(C.QUESTIONS.get(key), **slots)
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
            add("data_messa_in_mora", "data_messa_in_mora", ["events", i, ["date"]], {"counterparty": other},
                new_item={"what": f"messa in mora o contestazione a {other}", "kind": "diffida"})
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
    if named and name:
        return f"il terreno di {name}"
    return f"il fondo di {name}" if name else "il fondo del cliente"


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
    for topic in topics if topics is not None else [{"anchor": ""}]:
        names = _anchor_facts(topic, rows) if topics is not None else list(_FAMILY)
        for name in names:
            family = _FAMILY.get(name)
            if family and fact_state(name, facts, signals, case) == DA_PROVARE:
                wanted.setdefault(family, []).append(topic)
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
            ranked.append((-len(best), order, {**q, "unlocks": best,
                                              "topics": list(dict.fromkeys(t.get("anchor") or "" for t in asking))}))
    ranked.sort(key=lambda x: x[:2])
    first = client_question(facts, answered)
    return (first + [q for _, _, q in ranked])[:MAX_QUESTIONS]


def client_question(facts: Dict[str, Any], answered: Sequence[str] = ()) -> List[Dict[str, Any]]:
    """v57 (H): the parties the extraction left as possible clients (fact_signals.one_client: client_ambiguous) -- asked
    first, "Chi assisti …?"; v58: also whenever no party is the client (the parties named, no public body); [] when the
    client is known or the question answered."""
    names = [n for n in facts.get("client_ambiguous") or [] if n]
    parties = [p for p in facts.get("parties") or [] if isinstance(p, dict) and str(p.get("name") or "").strip()]
    if not any("cliente" in _low(p.get("role")).split(", ") for p in parties):
        # v58 (v57-l3, p2, h7: no party tagged cliente, no question): every time, the parties the case names
        names = names or [p["name"] for p in parties if not re.search(r"pubblica amministrazione|autorit", _low(p.get("role")))]
    if not names or "parte_assistita" in answered:
        return []
    joined = names[0] if len(names) == 1 else C.LIST_COMMA.join(names[:-1]) + C.CLIENT_JOIN + names[-1]
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


def known_facts(analysis: Dict[str, Any]) -> List[Dict[str, str]]:
    """The facts as they stand: parties with roles, the dated events, the computed terms (in the copy's words)."""
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
        text = C.fill(C.PARTY, name=p["name"], role=role) if role else C.fill(C.PARTY_NO_ROLE, name=p["name"])
        if client and _low(p["name"]) in client:
            text = C.fill(C.PARTY_CLIENT, party=text) or text
        out.append({"kind": "party", "name": p["name"], "role": role, "text": text or p["name"]})
    dated = [(e.get("date_iso") or "", e.get("date"), e.get("what")) for e in facts.get("events") or []
             if e.get("date") and e.get("what")]
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
    lines = [C.fill(C.ROUND1_OPENING, client=_client_name(analysis)), "", C.TOPICS_TITLE, ""]
    if state.get("topics"):
        for i, t in enumerate(state["topics"], 1):
            line = (C.fill(C.TOPIC_LINE, n=i, title=t["title"], articles=_articles_line(t), decides=_decides(t))
                    or C.fill(C.TOPIC_LINE_SHORT, n=i, title=t["title"], articles=_articles_line(t)) or t["title"])
            point = C.fill(C.TOPIC_POINT, point=C.LIST_JOIN.join(_to_verify(t, open_q)))
            lines.append(line + (point or ""))
    else:
        lines.append(C.NO_TOPICS)
    if state.get("others"):
        lines += ["", C.OTHER_PROFILES_TITLE, ""]
        lines += other_profile_lines(state["others"])
    facts = known_facts(analysis)
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
    return "\n".join(l for l in lines if l is not None), interaction(state, 1, ("choose_topic", "answer_questions", "proceed"))


# --- 3. the reply ---------------------------------------------------------------------------------------------------

_CONFIRM_RE = re.compile(r"^\W*(?:procedi|procediamo|ok(?:ay)?|vai|genera|sì|si|confermo|conferma|va bene|d['’]accordo|"
                         r"esatto|perfetto|certo)\W*(?:pure|grazie|così)?\W*$", re.IGNORECASE)
_ALL_RE = re.compile(r"^\W*(?:tutti|tutte|tutti gli argomenti|tutto)\W*$", re.IGNORECASE)
_NUMBERS_RE = re.compile(r"^\W*(?:(?:approfondisci|argoment[oi]|tema|punto|n\.?|numer[oi])\s*)?"
                         r"(\d{1,2})((?:\s*(?:,|;|e|ed|\+|/|\s)\s*\d{1,2})*)\W*$", re.IGNORECASE)
_ADD_RE = re.compile(r"^\W*(?:aggiungi|aggiungo|inoltre|preciso|precisazione|correggi|correzione)\b", re.IGNORECASE)

_REPLY_SYSTEM = (
    "Sei l'assistente di uno studio legale. L'utente risponde a un messaggio che gli proponeva degli ARGOMENTI numerati "
    "e delle DOMANDE con un id. Riporta la sua risposta in un JSON con ESATTAMENTE queste chiavi:\n"
    "- topics: i numeri degli argomenti che sceglie (lista, anche vuota);\n"
    "- confirm: true se chiede di procedere o conferma, altrimenti false;\n"
    "- answers: {id della domanda: risposta in breve} solo per le domande a cui risponde davvero; per sì/no scrivi «sì» "
    "o «no»; per una data, la data come la scrive; per l'atto di separazione, il tipo (vendita, divisione, donazione, "
    "successione) e la data se la dice;\n"
    "- angle: il profilo che chiede di approfondire se non è uno degli argomenti, in poche parole; altrimenti \"\";\n"
    "- corrections: le correzioni ai fatti, ciascuna {kind: \"party\", name, role} o {kind: \"date\", event, date};\n"
    "- statements: i fatti nuovi che afferma, ciascuno in una frase, come li dice.\n"
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
    empty = {"topics": [], "confirm": False, "answers": {}, "angle": "", "corrections": [], "statements": []}
    text = (message or "").strip()
    if _CONFIRM_RE.match(text):
        return {**empty, "how": "confirm", "confirm": True}
    if _ALL_RE.match(text):
        return {**empty, "how": "all", "topics": list(range(1, top + 1))}
    numbers = _numbers(text, top)
    if numbers:
        return {**empty, "how": "numbers", "topics": numbers}
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
    for qid, value in (raw.get("answers") or {}).items() if isinstance(raw.get("answers"), dict) else []:
        q = by_id.get(str(qid))
        value = " ".join(str(value or "").split())
        if not q or not value:
            continue
        if q["kind"] == "date":
            parsed = fact_signals.parse_date(value)
            if parsed:
                answers[q["id"]] = value
        elif q["kind"] == "transfer":
            kind = fact_signals._enum(value, fact_signals.TRANSFER_KINDS, fact_signals._TRANSFER_SYNONYMS)
            if kind and kind != "altro" or re.search(r"vend|divis|don|succe|eredit", value, re.IGNORECASE):
                answers[q["id"]] = value
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
    return {"topics": topics, "confirm": bool(raw.get("confirm")), "answers": answers,
            "angle": " ".join(str(raw.get("angle") or "").split())[:200], "corrections": corrections,
            "statements": statements}


def _stems(text: str) -> set:
    return {w[:6] for w in re.findall(r"[a-zà-ü]{4,}", _low(text))}


def _apply_answer(facts: Dict[str, Any], question: Dict[str, Any], value: str) -> None:
    from . import fact_signals
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


def apply_reply(state: Dict[str, Any], parsed: Dict[str, Any]) -> List[str]:
    """The reply's answers, corrections and statements put into the facts (the case text gains them as the client's
    precisazioni); then recompute(). The changes, in words (also in state["changes"])."""
    analysis = _restored_analysis(state["analysis"])
    facts = _facts(analysis)
    before = _derived(copy.deepcopy(facts), _areas(analysis)) if facts else None
    said: List[str] = []
    by_id = {q["id"]: q for q in state.get("questions") or []}
    for qid, value in (parsed.get("answers") or {}).items():
        q = by_id.get(qid)
        if not q or not facts:
            continue
        _apply_answer(facts, q, value)
        state.setdefault("answered", {})[qid] = value
        said.append(C.fill(C.ADDITION_ANSWER, question=q["text"], answer=value) or value)
    notes = [n for n in (_apply_correction(facts, c) for c in parsed.get("corrections") or [] if facts) if n]
    said += notes
    said += list(parsed.get("statements") or [])
    state.setdefault("additions", []).extend(s for s in said if s not in state["additions"])
    if not said:
        return []
    state["analysis"] = _jsonable(analysis)
    changes = notes + recompute(state, before)
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
    new = fact_signals.signals_from(copy.deepcopy(facts), old.get("problem") or "", areas, request=state.get("request") or "")
    if fact_signals.regimes_enabled():
        new["posture"] = fact_signals.claim_posture(case_text(state), new.get("facts"))
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
    _log("recomputed", {"changes": changes, "questions": [q["id"] for q in state["questions"]]})
    return changes


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
    based = [events_sentence(facts).rstrip(".")] + list(state.get("additions") or []) + list(state.get("changes") or [])
    verify = list(dict.fromkeys(re.sub(r"\s*\(domanda \d+\)$", "", v) for t in topics for v in _to_verify(t, open_q)))
    text = C.fill_parts(C.ROUND2, topics=C.LIST_JOIN.join(t["title"] for t in topics),
                        facts=C.LIST_JOIN.join(b for b in based if b), points=C.LIST_JOIN.join(verify),
                        parties=_parties_line(analysis))
    state["state"] = FRAME_RESTATED
    return text or C.CONFIRM_LINE, interaction(state, 2, ("confirm", "correct"))


# --- 5. the topic answer --------------------------------------------------------------------------------------------

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
    """The requisite's status from the facts the requisite's words name (fact_state, each): NON_RISULTA when one is known
    not to hold, PROVATO when every one holds, else DA_PROVARE -- from the extracted facts, the computed terms and the
    case text only, never from the model."""
    facts, signals = _facts(analysis), _signals(analysis)
    named = list(dict.fromkeys(name for pattern, name in _REQUISITE_FACTS if re.search(pattern, _low(requisite))))
    if not named or not facts:
        return DA_PROVARE
    states = [fact_state(n, facts, signals, str((analysis or {}).get(CASE_KEY) or "")) for n in named]
    return NON_RISULTA if NON_RISULTA in states else PROVATO if all(x == PROVATO for x in states) else DA_PROVARE


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
    from . import defensive_generation as dg
    keys = set()
    for t in topics:
        keys |= {_key(_parse_label(a)) for a in [t.get("anchor") or ""] + list(t.get("matched") or []) if a}
    signals = _signals(analysis)
    for rule in signals.get("fired") or []:
        keys |= {_key(tuple(r)) for r in rule.get("requires") or [] if len(r) == 2}
        keys |= {_key(r) for r in dg._refs_in(str(rule.get("conclusion") or ""))}
    # v58 (v57-l3: 1122-ter's «non si applica» paragraph): never a rule institution's resolved articles -- retrieval
    # resolved "uso delle parti comuni" to the videosurveillance article -- only what the rule itself names
    keys |= {_key(r) for r in dg._refs_in(case or "")}
    return [r for r in planned if _key(r) in keys]


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
                lines += [C.fill(C.REQUISITE_LINE, comma=label, requisite=r, status=requisite_status(r, analysis))
                          for r in reqs]
            elif _names_in(comma) or k in (rule.get("referring") or []):
                # v57: a comma line only where the comma names a fact (never «testo: da provare»); v59 (v58-h7 321): or
                # where it extends other articles' penalties -- its line, never a penalty as a requisite
                lines.append(C.fill(C.COMMA_LINE, comma=label, status=requisite_status(comma, analysis)))
        if len(lines) == 1 and chosen and _key(rule["ref"]) in (matched or set()):
            k, comma = chosen[0]
            lines.append(C.fill(C.COMMA_LINE, comma=C.fill(C.COMMA_LABEL, k=k) if k else C.COMMA_LABEL_ONLY,
                                status=requisite_status(comma, analysis)))
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


def _paragraph_article(paragraph: str, mark: Optional[int], refs: Sequence[tuple],
                       commi: Dict[int, List[str]]) -> Optional[int]:
    """The block (1-based) a paragraph belongs to: the block whose own comma it quotes («…», v58: first); else its mark's
    article when it cites it; else the first block article it cites; v57 (v56-p1: "Ai sensi del comma 2, …" under art. 42): else the
    block whose comma holds RULE_QUOTE's share (rule_quote.SHARE) of one of its sentences' content words, the most; else
    its mark; None when nothing says."""
    from . import defensive_generation as dg, rule_quote
    index = {_key(r): i for i, r in enumerate(refs, 1)}
    # v58 (v57-h7: 320's comma, quoted, under 318 because it cites 318 and 319): the article whose own comma holds the
    # quotation first; the cited numbers only then
    for quote in _QUOTE_RE.findall(paragraph):
        q = _plain(quote)
        hits = [i for i, texts in commi.items() if q and any(q in _plain(t) for t in texts)]
        if hits:
            return mark if mark in hits else hits[0]
    cited = list(dict.fromkeys(index[_key(r)] for r in dg._refs_in(paragraph) if _key(r) in index))
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
                   glossed: Optional[Sequence[tuple]] = None, matched: Optional[Sequence[tuple]] = None) -> str:
    """§2: one block per article (norm_blocks: heading and status lines, the code's, never the writer's), then the
    writer's paragraphs on it. v56: each paragraph goes to the article it cites, or whose comma it quotes, else to its
    "[N]" mark's (_paragraph_article) -- never to the previous heading because it follows it; a paragraph nothing places
    goes with the one before it, and before any placed one after the blocks (logged). No marks and as many paragraphs
    as articles, none citing: one per article, in order. v57 (B): glossed, the articles that earned an application
    (earned_refs), in the writer's numbering ("[N]" is the N-th of them); another article's block is its heading and
    comma line only, a paragraph placed on it is dropped (logged). v60: a matched article (a chosen topic's anchor, a
    table's own) the writer skips keeps its block with its comma line (norm_blocks), never «non trattate»."""
    from . import rule_quote
    refs = list(refs) if refs is not None else [r["ref"] for r in rules]
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
    where = [_paragraph_article(p, m, refs, commi) for m, p in paragraphs]
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
    placed = commi_moved(placed, refs, by_key, wrong)
    placed = {i: v for i, v in placed.items() if v}
    unearned = {i: placed.pop(i) for i in list(placed) if _key(refs[i - 1]) not in earned}
    # v58 (v57-h5, l3, h7, p2: bare headings, 3 of the 5 misuses left): an article whose text was not read gets no
    # block -- its paragraphs dropped --, nor one whose block would be its heading alone; both are named in one closing
    # line (OTHER_NORMS_LINE)
    textless = {i for i, ref in enumerate(refs, 1) if not (by_key.get(_key(ref)) or {}).get("commi")}
    no_text = {i: placed.pop(i) for i in list(placed) if i in textless}
    out, untreated = [], []
    for i, b in enumerate(blocks, 1):
        if i in textless or (len(b.split("\n")) == 1 and not placed.get(i)):
            untreated.append(refs[i - 1])
            continue
        out.append(b + "".join(f"\n\n{p}" for p in placed.get(i, [])))
    log.append({"norms": len(blocks), "applied": sorted(placed), "moved": moved, "unplaced": len(orphans), "commi": wrong,
                "dropped_unearned": {_label(refs[i - 1]): len(v) for i, v in unearned.items()},
                "dropped_no_text": {_label(refs[i - 1]): len(v) for i, v in no_text.items()},
                "untreated": [_label(r) for r in untreated]})
    closing = C.fill(C.OTHER_NORMS_LINE, articles=C.OTHER_NORMS_JOIN.join(
        C.fill(C.OTHER_NORMS_ITEM, number=r[0], code=_code_label(r[1])) or _label(r) for r in untreated).rstrip(".")
    ) if untreated else ""
    return "\n\n".join(out + orphans + ([closing] if closing else []))


def commi_moved(placed: Dict[int, List[str]], refs: Sequence[tuple], by_key: Dict[Tuple[str, str], Dict[str, Any]],
                log: List[Dict[str, Any]]) -> Dict[int, List[str]]:
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


def _jobs(state, plan, analysis, rows, refs, topics, rendered, documents) -> List[Tuple[str, str, str, int]]:
    from . import defensive_generation as dg, plan_then_write as ptw, protected_sources, render_computed, rule_quote
    who = plan.get("parte_assistita") or "la parte assistita"
    case = f"CASO:\n{case_text(state)}"
    titles = "; ".join(t["title"] for t in topics)
    shape = (" Questa risposta ha cinque sezioni: risposta e strategia; norme applicate ai fatti; obiezioni e repliche; "
             "profili processuali (già redatti); conclusioni. Tratta solo l'argomento indicato: " + titles + ".")

    def block(title: str, body: str) -> str:
        return f"{title}:\n{body}" if str(body or "").strip() else ""

    def material(*parts: str) -> str:
        return protected_sources.with_rule("\n\n".join(p for p in parts if p))
    fixed = block(ptw._CONCLUSIONS_TITLE, f"{ptw._conclusions(analysis)}\n{ptw._CONCLUSIONS_RULE}"
                  if ptw._conclusions(analysis) else "")
    fixed = "\n\n".join(b for b in (fixed, to_verify_block(analysis), unknown_block(state)) if b)
    to_sue = block(ptw._DEFENDANTS_TITLE, f"{ptw._defendants(analysis)}\n{ptw._DEFENDANTS_RULE}"
                   if ptw._defendants(analysis) else "")
    topic_lines = "\n".join(f"- {t['title']} ({_articles_line(t)})" + "".join(
        f"\n  da verificare: {v}" for v in _to_verify(t, state.get("questions") or [])) for t in topics)
    glossed = earned_refs(topics, analysis, case_text(state), section_refs(plan, rows, refs))
    rule_refs = [r for r in _rule_refs(rows, refs) if _key(r) in {_key(x) for x in glossed}]
    rules = rule_quote.material_block(plan, rows, analysis, rule_refs, status=lambda r: requisite_status(r, analysis),
                                      title=RULES_TITLE, rule_text=RULES_TEXT, force=True)
    numbered = "\n".join(f"{i}. art. {n} {ptw._code_key(c)}" for i, (n, c) in enumerate(glossed, 1))
    rules = f"{rules}\n\nNORME NELL'ORDINE DEI PARAGRAFI:\n{numbered}" if rules and numbered else rules
    texts = ptw._with_headings(dg._source_block([r for r in rows if r.get("_article") in rule_refs], dg._ARTICLE_CHARS),
                               ptw._headings(rows))
    read_only = render_computed.read_only_block(rendered or "")
    objected = objections(state, topics, refs)
    gated = render_computed.objections_block(plan)
    no_rulings = " Non citare pronunce."
    jobs = [
        (ANSWER_TITLES[0],
         f"SEZIONE: Risposta e strategia. La risposta diretta a {who} sull'argomento, in due-quattro paragrafi: che cosa "
         "può ottenere, con quale azione e su quale norma, e la strategia consigliata (prima la domanda principale, poi "
         "le subordinate). Non illustrare le norme nel dettaglio (sezione successiva)." + shape + no_rulings,
         material(case, fixed, block("ARGOMENTO", topic_lines), ptw._plan_text(plan, answers=True, rulings=False),
                  read_only), 900),
        (ANSWER_TITLES[1],
         f"SEZIONE: Norme applicate ai fatti. Per ciascuna norma di {RULES_TITLE}, nell'ordine, un paragrafo aperto dal "
         "numero della norma nell'elenco tra parentesi quadre su una riga nuova («[1] ...», «[2] ...»), senza titolo: la "
         "regola, riportando il comma tra «…» con il suo numero o restando dentro le sue parole, e come si applica ai "
         "fatti del caso, con il fatto che prova ciascun requisito o la prova da acquisire. Lo stato di ogni requisito è "
         "già scritto sopra il paragrafo: non ripeterlo e non contraddirlo. Solo le norme dell'elenco: le altre norme "
         "dell'argomento hanno già il loro titolo e non vanno commentate." + shape + no_rulings,
         material(case, fixed, to_sue, block("FATTI DEL CASO", ptw._facts_list(analysis)), rules,
                  block("TESTI DELLE NORME", texts)), 1700),
        (ANSWER_TITLES[2],
         f"SEZIONE: Obiezioni e repliche. Per ciascuna delle OBIEZIONI DELLA CONTROPARTE DA TRATTARE, nell'ordine e senza "
         f"aggiungerne: solo la replica di {who}, in due-tre frasi, sui fatti del caso e sulle norme fornite, aperta dal "
         "numero dell'obiezione tra parentesi quadre su una riga nuova («[1] ...», «[2] ...»); non riscrivere "
         "l'obiezione. Se la replica dipende da un fatto da verificare, dillo e indica la prova." + (f" Le norme di {render_computed.OBJECTIONS_TITLE} solo come argomento "
                                                  "della controparte." if gated else "") + shape + no_rulings,
         material(case, fixed, block("OBIEZIONI DELLA CONTROPARTE DA TRATTARE",
                                     "\n".join(f"{i}. {o}" for i, o in enumerate(objected, 1))), gated,
                  block("TESTI DELLE NORME", texts)), 1100),
        (ANSWER_TITLES[4],
         f"SEZIONE: Conclusioni. In due paragrafi: l'esito probabile per {who} su questo argomento; i rischi, cioè i "
         "requisiti da provare e le obiezioni più forti; la condotta consigliata e le prove da acquisire." + shape
         + no_rulings,
         material(fixed, ptw._plan_text(plan, subordinate=True, weak=True, rulings=False), rules, read_only),
         CONCLUSIONS_TOKENS),
    ]
    return [j for j in jobs if j[0] != ANSWER_TITLES[2] or objected]       # v58: no objection of the topic, no §3


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
_SECTION_NAMED_RE = re.compile(r"(?P<lead>,?\s*(?:come\s+(?:indicato|illustrato|visto|precisato)\s+)?(?:nella|alla|la)\s+"
                               r"sezione\s+)«(?P<name>[^»]{3,80})»", re.IGNORECASE)


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
    """v57 (K): "in via subordinata, ma prioritariamente" (v56-h3 §1) is "in via principale"; a section named that the
    answer does not have (v56-h10 §5 «Profili processuali e termini») is the title sharing most of its words, the
    reference dropped when none does."""
    from . import topic_step

    def section(m):
        name = m.group("name")
        if name in titles:
            return m.group(0)
        words = set(topic_step.tokens(name))
        best = max(titles, key=lambda t: len(words & set(topic_step.tokens(t))), default="")
        log.append({"section": name, "now": best if words & set(topic_step.tokens(best)) else ""})
        return f"{m.group('lead')}«{best}»" if words & set(topic_step.tokens(best)) else ""
    out = []
    for title, text in sections:
        text = _BOTH_WAYS_RE.sub("in via principale", text or "")
        out.append((title, _SECTION_NAMED_RE.sub(section, text)))
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
                           log: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
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


def unknown_block(state: Dict[str, Any]) -> str:
    """v56: the open questions for the writer, each «non noto» (UNKNOWN_TITLE, the copy's UNKNOWN_LINE)."""
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
            now = marked(sentence, [q.get("short") or q["family"] for q in hit])
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


def write_answer(state: Dict[str, Any], numbers: Sequence[int]) -> Tuple[str, List[Dict[str, Any]]]:
    """Step 5: the answer on the chosen topics, from the cache (no retrieval). (text, citations)."""
    from . import (defensive_generation as dg, plan_then_write as ptw, render_computed, repetition_guard, rule_quote,
                   text_tidy)
    t0 = time.time()
    lang = state.get("lang") or "it"
    analysis = _restored_analysis(state["analysis"])
    topics = [state["topics"][n - 1] for n in numbers if 0 < n <= len(state.get("topics") or [])]
    if not topics:
        return C.NO_TOPIC_TO_ANSWER, []
    refs = capped_refs(topics)
    all_rows = _restored_rows(state.get("rows"))
    rows = [r for r in all_rows if r.get("_article") and _key(r["_article"]) in {_key(x) for x in refs}]
    for row in rows:            # v56: no footnote marker or editorial note reaches the writer or §2
        sec = row.get("s") or {}
        row["s"] = {**sec, "plain_text": rule_quote.clean_text(sec.get("plain_text")) if sec.get("plain_text") else
                    sec.get("plain_text"), "titolo_articolo": rule_quote.clean_rubrica(sec.get("titolo_articolo"))
                    if sec.get("titolo_articolo") else sec.get("titolo_articolo")}
    tool_rows = _restored_rows(state.get("tool_rows"))
    documents = _isolated("documents", _documents, []) or []
    case = case_text(state)
    articles = dg._source_block(rows, dg._ARTICLE_CHARS)
    plan = _isolated("plan", lambda: ptw.make_plan(case, "", analysis, articles, "", "", rows + tool_rows, {}), None)
    planned = plan is not None
    if plan is None:
        plan = _fallback_plan(state, analysis, topics, _rule_refs(rows, refs))
    plan, analysis_p = ptw.for_the_plan(plan, analysis)
    if "obiezioni_norme" not in plan:
        plan = _isolated("move_gated", lambda: render_computed.move_gated(plan, analysis_p)[0], plan)
    who = f"{plan.get('parte_assistita') or ''} {plan.get('posizione') or ''}".strip() or party(analysis)
    analysis_p = {**analysis_p, "offences": _offences(topics, rows),          # v58 (J): §4's delitti
                  "topic_refs": _matched_refs(topics),                       # v59: the mediation of the topics' matter
                  "listed_topics": [{"title": t.get("title"), "articles": t.get("articles"), "source": t.get("source")}
                                    for t in state.get("topics") or []]}     # v59: §4's terms, the listed topics' only
    rendered = _isolated("render_computed", lambda: render_computed.render(analysis_p, who), state.get("rendered")) or ""
    analysis_p = {**analysis_p, CASE_KEY: analysis.get(CASE_KEY) or case}
    context = ptw.statement_context(analysis_p)
    named = dg._refs_in(case)
    jobs = _jobs(state, plan, analysis_p, rows + tool_rows, refs, topics, rendered, documents)
    sections = _isolated("write", lambda: ptw.write_sections(plan, jobs, lang, named, documents, rows + tool_rows, case,
                                                             context=context), None)
    if not sections:
        _log("answer_failed", {"topics": [t["title"] for t in topics], "planned": planned})
        return C.ANSWER_FAILED, []
    sections = [(t, scaffolding_out(x)) for t, x in sections]          # v57 (G)
    logs_k: List[Dict[str, Any]] = []
    sections = cosmetic_fixes(sections, list(ANSWER_TITLES) + [OTHER_PROFILES_TITLE], logs_k)      # v57 (K)
    logs: Dict[str, List[Dict[str, Any]]] = {"term_consistency": [], "unsupported_out": [], "citation_check": [],
                                             "strict_status": [], "conditional": [], "objections": []}
    rule_refs = _rule_refs(rows, refs)
    rules = _isolated("article_rules", lambda: rule_quote.article_rules(plan, rows, analysis_p, rule_refs), []) or []
    sections = ptw._step("term_consistency", sections,
                         lambda secs: term_consistency(secs, context, analysis_p, logs["term_consistency"]))

    def quoted(secs):
        done, log = rule_quote.check_sections(secs, plan, rows, analysis_p, documents, verbs=rule_quote.EXTENDED_VERB_RE,
                                              refs=rule_refs)
        logs["rule_quote"] = [log]
        return done
    sections = ptw._step("rule_quote", sections, quoted)
    sections = ptw._step("strict_status", sections, lambda secs: strict_status(secs, rules, analysis_p,
                                                                              logs["strict_status"]))
    logs["phantom_commi"] = []
    sections = ptw._step("phantom_commi", sections, lambda secs: phantom_commi_out(secs, rules, logs["phantom_commi"]))
    names = _isolated("party_names", lambda: rule_quote.party_names(analysis_p, plan), []) or []
    sections = ptw._step("conditional_on_unknown", sections, lambda secs: conditional_on_unknown(
        secs, rules, analysis_p, documents, names, logs["conditional"]))
    logs["unanswered"] = []
    sections = ptw._step("unanswered_out", sections, lambda secs: unanswered_out(secs, state, logs["unanswered"], names))
    if plan.get("obiezioni_norme"):
        sections = ptw._step("gated_as_support", sections, lambda secs: [
            (title, render_computed.gated_as_support(title, text, plan, documents)[0] or text) for title, text in secs])
    sections = ptw._step("repetition_guard", sections, lambda secs: repetition_guard.guard_sections(secs, "interactive")[0])
    allowed = _allowed(state, analysis_p, rule_refs or refs, tool_rows, rendered, documents)
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
    sections = _isolated("unsupported_out", lambda: unsupported_out(sections, rules, analysis_p, documents,
                                                                    logs["unsupported_out"], anchors), sections) or before
    at = next((i for i, (t, _) in enumerate(sections) if t == ANSWER_TITLES[4]), len(sections))
    if rendered:            # nothing computed: no procedural section (never a placeholder)
        sections = sections[:at] + [(ANSWER_TITLES[3], with_copy_mediation(rendered, analysis_p))] + sections[at:]
    titles = objections(state, topics, refs)
    logs["norms"] = []
    sections = [(t, assemble_objections(x, titles, logs["objections"]) if t == ANSWER_TITLES[2]
                 else assemble_norms(x, rules, analysis_p, rows, logs["norms"], section_refs(plan, rows, refs),
                                     _pool_rubriche(state), earned_refs(topics, analysis_p, case, section_refs(plan, rows, refs)),
                                     anchors)
                 if t == ANSWER_TITLES[1] else x)
                for t, x in sections]
    sections = [(t, x) for t, x in sections if (x or "").strip()]      # an objections section with no reply: gone
    draft = _join(sections)
    draft = _isolated("tidy", lambda: text_tidy.tidy(draft)[0], draft) or draft
    logs["final"] = []
    draft = _isolated("answer_final", lambda: answer_final(draft, logs["final"]), draft) or draft      # v59
    others = [C.fill(C.CLOSING_TOPIC, n=i, title=t["title"]) for i, t in enumerate(state["topics"], 1) if i not in numbers]
    others = [o for o in others if o]          # v56: no other topic, no closing (never «Oppure dimmi…» alone)
    closing = C.fill_parts(C.CLOSING, topics=C.LIST_JOIN.join(others)) if others else ""
    if closing:
        draft += "\n\n" + closing
    for n in numbers:
        state.setdefault("answers", {})[str(n)] = {"version": state.get("version"), "chars": len(draft)}
    cited = rows + tool_rows
    citations = _isolated("citations", lambda: [c for c in dg._rows_to_sources(dg._draft_sources(draft, cited, [], documents))
                                                if c.get("sections")], []) or []
    _log("answer", {"topics": [t["title"] for t in topics], "planned": planned, "articles": [_label(r) for r in rule_refs],
                    "objections": len(titles), "rendered_chars": len(rendered),
                    **{k: v for k, v in logs.items()}}, (time.time() - t0) * 1000)
    return draft, citations


# --- v59: the final pass, on the answer as the user reads it (after tidy, which can still empty a block) -------------

_NORM_HEADING_RE = re.compile(r"^\*\*Art\.\s+(\S+)\s+(.+?)(?:\s+—\s+.*?)?\*\*\s*$")
_OTHER_NORMS_RE = re.compile(r"^" + re.escape(C.OTHER_NORMS_LINE.split("{", 1)[0]) + r"\s*(.*?)\.?\s*$")
_OTHER_ITEM_RE = re.compile(r"art\.\s*(\S+)\s+(.+?)\s*$")


def norms_final(text: str, log: Optional[List[Dict[str, Any]]] = None) -> str:
    """§2 as read: an article heading followed by nothing (v58-h5 625, 52, 628-bis, 53; v58-p1 52, 53; v58-l3 1117,
    1118 -- tidy and the checks after assembly can empty a block) is no block; it is named once in the closing line
    (OTHER_NORMS_LINE) with those already there."""
    lines = (text or "").split("\n")
    out: List[str] = []
    untreated: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = _OTHER_NORMS_RE.match(line.strip())
        if m:
            untreated += [x.strip() for x in m.group(1).split(C.OTHER_NORMS_JOIN.strip() or ",") if x.strip()]
            i += 1
            continue
        h = _NORM_HEADING_RE.match(line.strip())
        if not h:
            out.append(line)
            i += 1
            continue
        j = i + 1
        while j < len(lines) and not _NORM_HEADING_RE.match(lines[j].strip()) and not _OTHER_NORMS_RE.match(lines[j].strip()):
            j += 1
        if "".join(lines[i + 1:j]).strip():
            out += lines[i:j]
        else:
            untreated.append(C.fill(C.OTHER_NORMS_ITEM, number=h.group(1), code=h.group(2).strip()) or f"art. {h.group(1)}")
            if log is not None:
                log.append({"bare": f"{h.group(1)} {h.group(2).strip()}"})
        i = j
    body = re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip("\n")
    untreated = list(dict.fromkeys(untreated))
    if not untreated:
        return body
    closing = C.fill(C.OTHER_NORMS_LINE, articles=C.OTHER_NORMS_JOIN.join(untreated).rstrip("."))
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


_DANGLING_RE = re.compile(r"(?P<lead>\s*\(?\s*(?:cfr\.|vedi|v\.)\s+(?:la\s+)?sezione\s*)(?P<q>[*«\"“]?)"
                          r"(?P<name>[^*»\"”)\n]{3,80}?)(?P<q2>[*»\"”]?)(?=\s*[),.;]|$)|"
                          r"(?P<lead2>\bsezione\s+)(?P<q3>[«*\"“])(?P<name2>[^»*\"”\n]{3,80})[»*\"”]", re.IGNORECASE)


def _references_fixed(text: str, titles: Sequence[str], log: List[Dict[str, Any]]) -> str:
    """v59 (v58-h10 §1 «(cfr. sezione *Profili processuali e termini*)»): a section named that the answer does not have
    is the present title sharing most of its words («Profili processuali»), the reference dropped when none does."""
    from . import topic_step
    present = {t.lower() for t in titles}

    def fix(m):
        name = (m.group("name") or m.group("name2") or "").strip(" .,;:")
        if not name or name.lower() in present or re.match(r"^(?:successiv|precedent|seguent|\d)", name.lower()):
            return m.group(0)
        words = set(topic_step.tokens(name))
        best = max(titles, key=lambda t: len(words & set(topic_step.tokens(t))), default="")
        log.append({"section": name, "now": best if words & set(topic_step.tokens(best)) else ""})
        if not words & set(topic_step.tokens(best)):
            return ""
        lead = m.group("lead") if m.group("lead") is not None else m.group("lead2")
        return f"{lead}«{best}»"
    text = _DANGLING_RE.sub(fix, text or "")
    return re.sub(r"\(\s*\)", "", text).replace(" .", ".").replace(" ,", ",")


def answer_final(draft: str, log: Optional[List[Dict[str, Any]]] = None) -> str:
    """v59: the last pass on the answer before its closing offer -- §2's bare headings into the closing line
    (norms_final); in every written section the cut lines (_orphans_out), a marker or a condition said once
    (_markers_once), a section named that is not there (_references_fixed), a label-only bullet or, in «Altri profili» and
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
            text = _markers_once(text, log)
            text = _references_fixed(text, present, log)
            text = _scaffold_bullets_out(text, title in (OTHER_PROFILES_TITLE, ANSWER_TITLES[2]), log)
        if title == ANSWER_TITLES[1]:
            text = norms_final(text, log)
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
    if not render_computed.mediation_of(analysis).get("required") or not C.MEDIATION:
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
        draft = dg._check_qualifications(_join(sections), analysis, rows, session, documents, case)
    return _split_draft(draft, titles) or sections


# --- the turn -------------------------------------------------------------------------------------------------------

def start(message: str, lang: str = "it", request: str = "") -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """A narrative case: (reply, interaction, state) of round 1."""
    t0 = time.time()
    state = analyse_case(message, request, lang)
    reply, field = round1(state)
    state["timings"]["round1_s"] = round(time.time() - t0, 1)
    field["seconds"] = state["timings"]["round1_s"]
    _log("round1", {"topics": len(state["topics"]), "questions": len(state["questions"])}, (time.time() - t0) * 1000)
    return reply, field, state


def follow_up(message: str, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """A reply within the interaction: {"answer", "interaction", "state", "citations", "kind"}, or None when this
    message is not one (a new case, or a question the interaction does not hold) -- the old path then."""
    t0 = time.time()
    phase = state.get("state")
    if is_new_case(message):
        return None
    parsed = parse_reply(message, state)
    _log("reply", {"phase": phase, "how": parsed["how"], "topics": parsed["topics"], "confirm": parsed["confirm"],
                   "answers": list(parsed["answers"]), "angle": parsed["angle"],
                   "corrections": len(parsed["corrections"]), "statements": len(parsed["statements"])})

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
    changes = []
    if parsed["answers"] or parsed["corrections"] or parsed["statements"]:
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
        if changes or parsed["statements"]:
            state["pending_regenerate"] = list(state.get("chosen") or [1])
            touch(state, "new fact")
            titles = "; ".join(state["topics"][n - 1]["title"] for n in state["pending_regenerate"]
                               if 0 < n <= len(state["topics"]))
            text = C.fill(C.NEW_FACT, changes="\n".join(C.fill(C.CHANGE_LINE, text=c) for c in changes or [C.NEW_FACT_NONE]),
                          titles=titles or "—")
            return out(text, interaction(state, 3, ("regenerate", "deepen")), "new_fact")
        if parsed["confirm"] and state.get("pending_regenerate"):
            return answer(state["pending_regenerate"], "regenerate")
        return None
    if parsed["confirm"]:
        if phase == TOPICS_OFFERED and not state.get("chosen"):
            state["chosen"] = [1]
        return answer(state.get("chosen") or [1], "answer")
    if parsed["how"] == "none" and not changes:
        text = C.fill(C.UNCLEAR, next=C.INSTRUCTION if phase == TOPICS_OFFERED else C.CONFIRM_LINE)
        return out(text, interaction(state, 1 if phase == TOPICS_OFFERED else 2, ("choose_topic", "proceed")), "unclear")
    touch(state, "round 2")
    text, field = round2(state)
    return out(text, field, "round2")
