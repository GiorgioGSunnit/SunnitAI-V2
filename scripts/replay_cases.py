#!/usr/bin/env python3
"""v79.16.2 (measurement only): N synthetic cases through the real pipeline entry, one JSON of what each stage decided,
and a one-page diff against a baseline run.

Each case of --cases (data/replay_cases/*.txt, the file's stem its id; six synthetic cases written with the script: two
civil claimants with numbered quesiti, one civil defence -- opposizione a decreto ingiuntivo --, two penal -- an indagato,
a persona offesa --, one lavoro; invented names and facts, none of a lawyer's, a tester's, the PM's or Antonelo's)
goes through interactive_flow.request_case_first (intake, the frame, the flow, the writer -- as the API runs it) against
the model --base-url / --model name (LLM_BASE_URL, LLM_MODEL: the GPU host), the flags as --env sets them (python-dotenv;
--set KEY=VALUE overrides one). Per case the script records:

  frame        the verified fields of the case frame (case_frame.held: cliente, ruolo, rito, stadio, addebito_o_domanda,
               istituto, parti, quesiti -- each as a short text) and the fields the rules filled
  fit          the applicability counters (fit_si, fit_si_verified, fit_dipende, fit_no, fit_unverified_ids, fit_removed,
               fit_kept_named, fit_kept_from_quesiti, norms_removed_by_fit, the governs and the recall counters)
  recall       v79.17.1 (item 9): the binding's set (principale) and the pretesa's recall (status, articles, recalled)
  analysis     the step's status, its leading / subordinate line (articolo, E-id, stato), the line tier the step's log
               line carries, the esiti per element of the lines' articles, the rejected lines, the fallback reason, the
               analysis_* and line_from_* counters
  retrieval    the «retrieval_quesiti» gate line (queries, source, reason, the labels added)
  qa           the Q&A counters (qa_questions_detected, qa_answer_reasked, qa_answer_insufficient, the qa_sentences_*)
  seconds      the wall time of the case
  strategy     v79.18.0 (B): the strategy step's line -- the ranked articles with their forza, the reordered flag, the
               risk word, the objections and contro-difesa answered -- and its counters (STRATEGY_COUNTERS)
  stages       v79.17.1 (item 8): the seconds per stage as the flow measured them -- frame (the frame step, inside the
               intake: frame_seconds), record (the fact record: records_s), extraction (round 1's analysis: analysis_s),
               retrieval (round 1's rest: retrieval_s), fit (applicability_s), analysis (the analysis step's), write (the
               answer's time less the fit's and the step's), total (the flow's single_shot_s). A measurement, no
               optimisation: the diff shows each stage «before → now» beside the wall seconds.

The JSON (--out) holds the version, the model, the flags and the cases; --baseline (a JSON this script wrote) adds the
diff: per case, every value that moved, «before → now» (the markdown --report, also printed). Nothing here changes
src/: the script reads the state the flow leaves and the log lines it writes (verbose_logger.vlog, wrapped, never
replaced: the server log keeps them).

  python scripts/replay_cases.py --env .env.v45 --base-url http://127.0.0.1:8000/v1 --model Qwen3 \\
      --out /var/log/maint/replay_79162.json --baseline /var/log/maint/replay_79161.json --report replay.md
  python scripts/replay_cases.py --env .env.v45 --only civ_attore_caparra_quesiti --out one.json

Exit 0 when every case ran; 1 when any case raised (the error is in its row) or the baseline cannot be read. The
measurement is not a score: no mean, no pass mark -- a table per case and the diff.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CASES_DIR = ROOT / "data" / "replay_cases"
FLAGS = ("CASE_FRAME", "APPLICABILITY_FIT", "ANALYSIS_STEP", "FACT_RECORD", "QUESTION_RETRIEVAL", "ELEMENT_SCHEMAS")
FIT_COUNTERS = ("fit_si", "fit_si_verified", "fit_dipende", "fit_dipende_verified", "fit_no", "fit_no_verified",
                "fit_unverified_ids", "fit_removed", "fit_kept_named", "fit_kept_from_quesiti", "norms_removed_by_fit",
                "fit_governs_pretesa", "fit_governs_quesiti", "fit_context_only",          # v79.17.0 (item 3)
                "fit_pretesa_recall", "fit_pretesa_recall_empty")                          # v79.17.1 (item 9)
FRAME_COUNTERS = ("frame_pretesa_verified", "frame_pretesa_derived", "frame_asked_side",      # v79.17.0 (item 1)
                  "binding_pretesa_applied", "binding_pretesa_excluded",                     # v79.17.0 (item 2)
                  "binding_set_size", "binding_empty_honesty",                                # v79.17.2 (item 1)
                  "frame_pretesa_article_dropped")                                             # v79.17.3 (item 3)
ANALYSIS_COUNTERS = ("analysis_calls", "analysis_reasked", "analysis_failed", "analysis_timeouts",
                     "analysis_line_rejected", "analysis_line_procedural_rejected", "analysis_line_undetermined",
                     "analysis_line_fallback_rubrica", "analysis_frame_charge_missing", "analysis_provato_downgraded",
                     "analysis_provato_no_pertinence", "analysis_provenance_from_record", "analysis_provenance_fallback",
                     "analysis_fallback_skipped_reason", "line_from_fit_si", "line_from_fit_dipende",
                     "line_from_unverified", "line_table_weaker",
                     "analysis_line_context_rejected", "analysis_line_quesito_only",       # v79.17.0 (item 2)
                     "analysis_line_outside_rito_rejected",                                  # v79.17.1 (item 9)
                     "analysis_line_outside_set_rejected", "analysis_binding_empty_honesty")   # v79.17.2 (items 1-2)
RETRIEVAL_COUNTERS = ("retrieval_quesiti_queries", "retrieval_quesiti_added")
QA_COUNTERS = ("qa_questions_detected", "qa_answer_reasked", "qa_answer_insufficient", "qa_answer_fragment",
               "qa_sentences_replaced", "qa_sentences_dropped", "qa_sentences_dropped_dependent",
               "prose_unfitted_article_dropped", "prose_excluded_article_dropped", "prose_unknown_id_dropped",
               "sections_dedup_dropped", "rulings_off_line_dropped", "sections_rito_blocks_dropped")
FIELD_KEYS = ("tipo", "contenuto", "contro", "valore", "intake", "articolo", "descrizione", "testo", "nome",
              "ruolo_nel_fatto", "lato")       # v79.17.0: the pretesa's fields first
RETRIEVAL_COUNTERS = RETRIEVAL_COUNTERS + ("retrieval_quesiti_gated", "family_rito_dropped")      # v79.17.1 (item 6)
ANALYSIS_COUNTERS = ANALYSIS_COUNTERS + ("candidates_no_elements_excluded",)                      # v79.17.1 (item 7)
STAGES = ("frame", "record", "extraction", "retrieval", "fit", "analysis", "strategy", "write", "total")   # v79.17.1 (item 8); v79.18.0
# v79.18.0 (B): the strategy step's counters, and its line per case (ranked articles + forza + the reordered flag)
STRATEGY_COUNTERS = ("strategy_calls", "strategy_reasked", "strategy_lines", "strategy_rank_reordered",
                     "strategy_line_rejected", "strategy_objections_unanswered", "strategy_rulings_cited",
                     "strategy_seconds", "strategy_failed", "strategy_bad_id", "strategy_passi_added",
                     "strategy_contro_difesa_added", "strategy_entries_uncited",
                     "strategy_riqualificazione_missing", "strategy_objection_mismatch", "strategy_passi_not_concrete")   # v79.18.1
RETRIEVAL_COUNTERS = RETRIEVAL_COUNTERS + ("retrieval_quesiti_standing",)                            # v79.18.1 (C3)
QA_COUNTERS = QA_COUNTERS + ("prose_rule_without_id_dropped", "rulings_no_passage_dropped",            # v79.18.1 (C7-C9)
                             "rulings_general_part_dropped", "timestamps_dropped")
FRAME_COUNTERS = FRAME_COUNTERS + ("binding_set_unverified",)                                       # v79.18.0 (A1)
ANALYSIS_COUNTERS = ANALYSIS_COUNTERS + ("analysis_line_fallback_set", "analysis_degree_downgraded")   # v79.18.0 (A2, 11)
# v79.18.2: item 2 the side re-ask, item 6 the alternativa tier (fit, binding, strategy), item 5 the matter gate's areas
STRATEGY_COUNTERS = STRATEGY_COUNTERS + ("strategy_side_reasked", "strategy_alternativa_demoted")
STRATEGY_COUNTERS = STRATEGY_COUNTERS + ("strategy_risk_capped",)                                   # v79.18.3 (item 1)
FIT_COUNTERS = FIT_COUNTERS + ("fit_no_kept_alternative",)
FRAME_COUNTERS = FRAME_COUNTERS + ("binding_set_alternative",)
RETRIEVAL_COUNTERS = RETRIEVAL_COUNTERS + ("topic_filter_areas", "topic_filter_demoted")
RETRIEVAL_STAGES_SHOWN = 6          # v79.18.2 (item 4): the retrieval's slowest sub-stages named in the row


# --- the measurement --------------------------------------------------------------------------------------------------

def _short(value: Any, limit: int = 90) -> str:
    """A frame field as a short text: a dict by its telling keys, a list as «n: a; b», a string as it is."""
    if isinstance(value, dict):
        parts = [str(value[k]) for k in FIELD_KEYS if value.get(k) not in (None, "", [], {})]
        text = " / ".join(parts) if parts else json.dumps(value, ensure_ascii=False)
    elif isinstance(value, list):
        text = f"{len(value)}: " + "; ".join(_short(x, 40) for x in value[:4]) + (" …" if len(value) > 4 else "")
    else:
        text = str(value or "")
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


def frame_fields(state: Dict[str, Any]) -> Dict[str, Any]:
    """The verified frame's fields as short texts, and the fields the rules filled (case_frame.held; {} with the
    frame off or absent)."""
    from src.rag import case_frame
    got = case_frame.held(state)
    if not got:
        return {"verified": [], "fallback": [], "fields": {}}
    frame = got.get("frame") or {}
    verified = list(got.get("verified") or [])
    derived = list(got.get("derived") or [])
    from src.rag import case_frame
    return {"verified": verified, "fallback": list(got.get("fallback") or []), "derived": derived,
            "fields": {f: _short(frame.get(f)) for f in verified + [x for x in derived if x not in verified]},
            "pretesa": case_frame.pretesa_text(got), "pretesa_line": case_frame.pretesa_line(got)}   # v79.17.0


def _event(events: Dict[str, List[Any]], name: str) -> Dict[str, Any]:
    """The first logged event of that name -- events as {name: [data, …]} or {name: [(data, ms), …]}."""
    for x in events.get(name) or []:
        data = x[0] if isinstance(x, (tuple, list)) else x
        if isinstance(data, dict):
            return data
    return {}


def analysis_fields(state: Dict[str, Any], events: Dict[str, List[Any]]) -> Dict[str, Any]:
    step = state.get("analysis_step") or {}
    analysis = step.get("analysis") or {}
    line = _event(events, "analysis_step")
    lines = []
    for kind in ("linea_principale", "linea_subordinata"):
        x = analysis.get(kind)
        if x:
            lines.append({"kind": kind, "articolo": x.get("articolo"), "eid": x.get("eid"), "stato": x.get("stato"),
                          "esito_in_gioco": x.get("esito_in_gioco")})
    return {"status": step.get("status") or line.get("status") or "",
            "line_tier": line.get("line_tier") or "",
            "honesty": str(analysis.get("honesty") or line.get("honesty") or ""),      # v79.17.2 (item 1)
            "fallback_skipped_reason": (step.get("counters") or {}).get("analysis_fallback_skipped_reason")
            or line.get("fallback_skipped_reason") or "",
            "lines": lines,
            "esiti": {art: [[e.get("eid"), e.get("esito")] for e in els]
                      for art, els in (analysis.get("elementi") or {}).items()},
            "rejected": [{"articolo": r.get("articolo"), "why": str(r.get("why") or "")[:120]}
                         for r in step.get("rejected") or []],
            "no_pertinence": [{"eid": x.get("eid"), "esito_modello": x.get("esito_modello")}
                              for x in line.get("no_pertinence") or []]}


def recall_fields(state: Dict[str, Any]) -> Dict[str, Any]:
    """v79.17.1 (item 9): the pretesa's recall as the fit left it (state["applicability"]["pretesa_recall"]) and the
    binding's sets -- {status, articles, dropped, principale, recalled}; {} when the fit did not run."""
    fit = state.get("applicability") or {}
    if not fit:
        return {}
    rec = fit.get("pretesa_recall") or {}
    binding = state.get("binding_pretesa") or {}
    return {"status": rec.get("status") or "", "articles": list(rec.get("articles") or []),
            "dropped": list(rec.get("dropped") or []), "principale": list(binding.get("principale") or []),
            "recalled": list(binding.get("recalled") or []),
            "reasons": {k: list(v) for k, v in (binding.get("reasons") or {}).items()},     # v79.17.3 (item 1)
            "pretesa_article": binding.get("pretesa_article") or "",
            "alternative": list(binding.get("alternative") or []),                          # v79.18.2 (item 6)
            "tiers": dict(binding.get("tiers") or {})}


def strategy_fields(state: Dict[str, Any]) -> Dict[str, Any]:
    """v79.18.0 (B): the strategy step's line (state["strategy"]: strategy_step.replay_summary) -- {status, ranked
    [{articolo, eid, tipo, rapporto, forza}], reordered, model_order, final_order, risk, objections, contro_difesa,
    prompt_hash, error}; {} when the step did not run."""
    got = state.get("strategy") or {}
    if not isinstance(got, dict) or not got:
        return {}
    return {k: got.get(k) for k in ("status", "ranked", "reordered", "model_order", "final_order", "risk",
                                    "objections", "contro_difesa", "riqualificazione", "prompt_hash", "error",
                                    "side", "side_source", "alternative",                   # v79.18.2 (items 2, 6)
                                    "risk_why", "risk_capped") if k in got}                 # v79.18.3 (item 1)


def retrieval_fields(state: Dict[str, Any], events: Dict[str, List[Any]]) -> Dict[str, Any]:
    gate = _event(events, "retrieval_quesiti") or state.get("quesiti_retrieval") or {}
    out = {k: gate.get(k) for k in ("queries", "source", "reason", "added") if k in gate}
    if gate.get("gated"):           # v79.17.1 (item 6): the questions' rows the family gate refused
        out["gated"] = [f"{x.get('article')} ({x.get('why') or x.get('family')})" for x in gate["gated"]]
    return out


def stage_seconds(state: Dict[str, Any]) -> Dict[str, float]:
    """v79.17.1 (item 8): {stage: seconds} from what the flow measured (STAGES; a stage the flow did not time is left
    out): frame -- the frame step's frame_seconds (state["case_frame"]["counters"]); record -- timings records_s;
    extraction / retrieval -- round 1's analysis_s / retrieval_s; fit -- applicability_s (else fit_seconds); analysis --
    the step's seconds (else analysis_seconds); write -- single_shot_timings answer_s less the fit's and the step's;
    total -- the flow's total_s."""
    state = state or {}
    t = state.get("timings") or {}
    c = state.get("block_counters") or {}
    cf = (state.get("case_frame") or {}).get("counters") or {}
    sst = state.get("single_shot_timings") or {}
    fit = t.get("applicability_s", c.get("fit_seconds"))
    analysis = (state.get("analysis_step") or {}).get("seconds", c.get("analysis_seconds"))
    strategy = t.get("strategy_s", c.get("strategy_seconds"))         # v79.18.0 (B): the strategy step's seconds
    answer = sst.get("answer_s")
    write = None
    if answer is not None:
        write = max(0.0, float(answer) - float(fit or 0) - float(analysis or 0) - float(strategy or 0))
    raw = {"frame": cf.get("frame_seconds", c.get("frame_seconds")), "record": t.get("records_s"),
           "extraction": t.get("analysis_s"), "retrieval": t.get("retrieval_s"), "fit": fit, "analysis": analysis,
           "strategy": strategy, "write": write, "total": sst.get("total_s", t.get("single_shot_s"))}
    out: Dict[str, float] = {}
    for k in STAGES:
        v = raw.get(k)
        if v is None:
            continue
        try:
            out[k] = round(float(v), 1)
        except (TypeError, ValueError):
            continue
    return out


def retrieval_stages(state: Dict[str, Any], n: int = RETRIEVAL_STAGES_SHOWN) -> List[Dict[str, Any]]:
    """v79.18.2 (item 4): the retrieval's sub-stages as round 1's meter measured them (timings round1_stages: <stage>_s;
    round1_counts: <stage>_llm / _embed / _graph), the slowest n first: [{stage, s, llm, embed, graph}]. The fusion's
    own sub-steps are marked select:current / searches / vote / fetch / crossref / spotted; [] when the flow did not
    time round 1."""
    t = (state or {}).get("timings") or {}
    secs = t.get("round1_stages") or {}
    counts = t.get("round1_counts") or {}
    out: List[Dict[str, Any]] = []
    for key, value in secs.items():
        if not str(key).endswith("_s") or key == "total_s":
            continue
        stage = str(key)[:-2]
        try:
            seconds = round(float(value), 1)
        except (TypeError, ValueError):
            continue
        out.append({"stage": stage, "s": seconds, **{k: int(counts.get(f"{stage}_{k}") or 0) for k in ("llm", "embed", "graph")}})
    out.sort(key=lambda x: -x["s"])
    return out[:n]


def retrieval_stages_text(stages: Sequence[Dict[str, Any]]) -> str:
    parts = []
    for x in stages or []:
        calls = ", ".join(f"{k} {x[k]}" for k in ("llm", "embed", "graph") if x.get(k))
        parts.append(f"{x['stage']} {x['s']}" + (f" ({calls})" if calls else ""))
    return "; ".join(parts) or "—"


def _counters(state: Dict[str, Any], keys: Sequence[str]) -> Dict[str, Any]:
    c = state.get("block_counters") or {}
    return {k: c[k] for k in keys if k in c}


def measure(case_id: str, state: Dict[str, Any], answer: str, seconds: float,
            events: Optional[Dict[str, List[Any]]] = None, error: str = "") -> Dict[str, Any]:
    """One case's row from the state the flow left, its answer, the wall seconds and the logged events."""
    events = events or {}
    return {"id": case_id, "error": error, "seconds": round(float(seconds), 1), "answer_chars": len(answer or ""),
            "asked": bool((state or {}).get("asked_side")),                     # v79.17.0 (item 1): the flow asked
            "frame": frame_fields(state), "frame_counters": _counters(state, FRAME_COUNTERS),
            "fit": _counters(state, FIT_COUNTERS),
            "recall": recall_fields(state),                                     # v79.17.1 (item 9)
            "analysis": analysis_fields(state, events),
            "analysis_counters": _counters(state, ANALYSIS_COUNTERS),
            "strategy": strategy_fields(state), "strategy_counters": _counters(state, STRATEGY_COUNTERS),   # v79.18.0
            "rites_block": list(state.get("rites_block") or []),                                            # v79.18.1 (C6)
            "retrieval": retrieval_fields(state, events), "retrieval_counters": _counters(state, RETRIEVAL_COUNTERS),
            "qa": _counters(state, QA_COUNTERS),
            "stages": stage_seconds(state),                                     # v79.17.1 (item 8)
            "retrieval_stages": retrieval_stages(state)}                        # v79.18.2 (item 4)


class _Capture:
    """verbose_logger.vlog wrapped for one case: the events kept by name, the original still called."""

    def __init__(self) -> None:
        self.events: Dict[str, List[Any]] = {}
        self._orig = None

    def __enter__(self) -> "_Capture":
        from src.rag import verbose_logger
        self._orig = verbose_logger.vlog

        def wrapped(event, data=None, duration_ms=None):
            self.events.setdefault(str(event), []).append((data, duration_ms))
            return self._orig(event, data, duration_ms)
        verbose_logger.vlog = wrapped
        return self

    def __exit__(self, *exc) -> None:
        from src.rag import verbose_logger
        verbose_logger.vlog = self._orig


def run_case(case_id: str, text: str) -> Dict[str, Any]:
    """The case through interactive_flow.request_case_first (the API's entry), measured; an exception is the row's
    error, never the run's end."""
    from src.rag import interactive_flow
    t0 = time.time()
    with _Capture() as cap:
        try:
            result = interactive_flow.request_case_first(text, "it", case_id=case_id)
        except Exception as exc:    # noqa: BLE001 -- the row says so
            traceback.print_exc()
            return measure(case_id, {}, "", time.time() - t0, cap.events, error=f"{type(exc).__name__}: {exc}"[:300])
    return measure(case_id, result.get("state") or {}, result.get("answer") or "", time.time() - t0, cap.events)


def load_cases(folder: Path, only: Optional[Sequence[str]] = None) -> List[Tuple[str, str]]:
    out = []
    for path in sorted(folder.glob("*.txt")):
        if only and path.stem not in only:
            continue
        text = path.read_text(encoding="utf-8").strip()
        if text:
            out.append((path.stem, text))
    return out


# --- the diff ---------------------------------------------------------------------------------------------------------

def _flat(prefix: str, value: Any, out: Dict[str, Any]) -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            _flat(f"{prefix}.{k}" if prefix else str(k), v, out)
    else:
        out[prefix] = value


def _row_values(row: Dict[str, Any]) -> Dict[str, Any]:
    """The comparable values of a row, flat: the frame's verified list, each field, every counter, the lines, the
    tier, the esiti, the gate line; seconds apart."""
    flat: Dict[str, Any] = {}
    a = row.get("analysis") or {}
    _flat("frame.fields", (row.get("frame") or {}).get("fields") or {}, flat)
    flat["frame.verified"] = ",".join((row.get("frame") or {}).get("verified") or [])
    flat["frame.pretesa"] = (row.get("frame") or {}).get("pretesa") or ""
    flat["asked"] = bool(row.get("asked"))
    for key in ("fit", "analysis_counters", "retrieval_counters", "qa", "retrieval", "frame_counters", "recall",
                "strategy_counters"):
        _flat(key, row.get(key) or {}, flat)
    st = row.get("strategy") or {}          # v79.18.0 (B): the strategy line, one comparable value
    if st:
        flat["strategy.status"] = st.get("status")
        flat["strategy.ranked"] = "; ".join(f"{x.get('articolo')} {x.get('eid')} {x.get('tipo')} [{x.get('forza')}]"
                                            for x in st.get("ranked") or [])
        flat["strategy.reordered"] = bool(st.get("reordered"))
        flat["strategy.risk"] = st.get("risk") or ""
        riq = st.get("riqualificazione") or {}
        flat["strategy.riqualificazione"] = (f"art. {riq.get('articolo')} (linea {riq.get('n')})" if riq.get("linea")
                                             else f"nessuna ({', '.join(riq.get('alternative') or [])})") if riq else ""
    flat["rites_block"] = "; ".join(f"{x.get('name')}: {x.get('status')}" for x in row.get("rites_block") or [])
    flat["analysis.status"] = a.get("status")
    flat["analysis.line_tier"] = a.get("line_tier")
    flat["analysis.fallback_skipped_reason"] = a.get("fallback_skipped_reason")
    flat["analysis.honesty"] = a.get("honesty") or ""
    for x in a.get("lines") or []:
        flat[f"analysis.{x.get('kind')}"] = f"{x.get('articolo')} {x.get('eid') or ''} {x.get('stato')}".strip()
    for art, els in (a.get("esiti") or {}).items():
        flat[f"analysis.esiti.{art}"] = ", ".join(f"{e[0]}={e[1]}" for e in els)
    flat["analysis.rejected"] = "; ".join(f"{r.get('articolo')}: {r.get('why')}" for r in a.get("rejected") or [])
    flat["error"] = row.get("error") or ""
    return flat


def _show(value: Any) -> str:
    if isinstance(value, list):
        return "[" + ", ".join(str(x) for x in value) + "]"
    return "—" if value in (None, "") else str(value)


def diff_rows(now: Sequence[Dict[str, Any]], before: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per case (by id): the values that moved, «before → now»; a case only on one side is said so."""
    old = {r.get("id"): r for r in before}
    out = []
    for row in now:
        prev = old.get(row.get("id"))
        if prev is None:
            out.append({"id": row.get("id"), "new": True, "changes": []})
            continue
        a, b = _row_values(prev), _row_values(row)
        changes = [{"key": k, "before": a.get(k), "now": b.get(k)} for k in sorted(set(a) | set(b))
                   if a.get(k) != b.get(k)]
        changes.append({"key": "seconds", "before": prev.get("seconds"), "now": row.get("seconds")})
        for k in STAGES:            # v79.17.1 (item 8): each stage's seconds beside the wall time, always shown
            a_s, b_s = (prev.get("stages") or {}).get(k), (row.get("stages") or {}).get(k)
            if a_s is not None or b_s is not None:
                changes.append({"key": f"seconds.{k}", "before": a_s, "now": b_s})
        out.append({"id": row.get("id"), "new": False, "changes": changes})
    for r in before:
        if r.get("id") not in {x.get("id") for x in now}:
            out.append({"id": r.get("id"), "gone": True, "changes": []})
    return out


# --- the report -------------------------------------------------------------------------------------------------------

def case_table(row: Dict[str, Any]) -> str:
    a = row.get("analysis") or {}
    lines = []
    lines.append(f"### {row.get('id')}" + (f" — ERROR {row['error']}" if row.get("error") else ""))
    lines.append("")
    lines.append("| Stage | Value |")
    lines.append("|---|---|")
    fr = row.get("frame") or {}
    lines.append(f"| pretesa | {_show(fr.get('pretesa'))}" + (" (derivata)" if "addebito_o_domanda" in (fr.get("derived") or []) else "") + " |")
    if row.get("asked"):
        lines.append("| asked | the flow asked for the side and stopped (frame_asked_side) |")
    lines.append(f"| frame verified | {_show(fr.get('verified'))} |")
    for f, v in (fr.get("fields") or {}).items():
        lines.append(f"| frame.{f} | {v} |")
    lines.append(f"| frame fallback | {_show(fr.get('fallback'))} |")
    lines.append("| frame counters | " + ", ".join(f"{k} {v}" for k, v in (row.get("frame_counters") or {}).items()) + " |")
    lines.append("| fit | " + ", ".join(f"{k} {v}" for k, v in (row.get("fit") or {}).items()) + " |")
    rc = row.get("recall") or {}         # v79.17.1 (item 9): the binding's set, the recall when the first pass named none
    if rc:
        lines.append(f"| binding / recall | principale {_show(rc.get('principale'))}; recall {_show(rc.get('status'))}"
                     f", articles {_show(rc.get('articles'))}, recalled {_show(rc.get('recalled'))}"
                     + (f", dropped {_show(rc.get('dropped'))}" if rc.get("dropped") else "")
                     + (", reasons " + "; ".join(f"{k}: {'+'.join(v)}" for k, v in rc["reasons"].items()) if rc.get("reasons") else "")
                     + (f", pretesa article {rc['pretesa_article']}" if rc.get("pretesa_article") else "") + " |")
    lines.append(f"| analysis status / tier / fallback | {_show(a.get('status'))} / {_show(a.get('line_tier'))} / "
                 f"{_show(a.get('fallback_skipped_reason'))} |")
    if a.get("honesty"):
        lines.append(f"| honesty line | {a['honesty']} |")
    for x in a.get("lines") or []:
        lines.append(f"| analysis.{x.get('kind')} | art. {x.get('articolo')} {x.get('eid') or ''} ({x.get('stato')}"
                     f"{', ' + str(x.get('esito_in_gioco')) if x.get('esito_in_gioco') else ''}) |")
    for art, els in (a.get("esiti") or {}).items():
        lines.append(f"| esiti {art} | " + ", ".join(f"{e[0]} {e[1]}" for e in els) + " |")
    if a.get("rejected"):
        lines.append("| rejected | " + "; ".join(f"{r.get('articolo')}: {r.get('why')}" for r in a["rejected"]) + " |")
    if a.get("no_pertinence"):
        lines.append("| no_pertinence (logged) | " + ", ".join(f"{x.get('eid')} {x.get('esito_modello')}"
                                                              for x in a["no_pertinence"]) + " |")
    lines.append("| analysis counters | " + ", ".join(f"{k} {v}" for k, v in (row.get("analysis_counters") or {}).items())
                 + " |")
    st = row.get("strategy") or {}          # v79.18.0 (B): the strategy line -- the ranked articles with their forza
    if st:
        ranked = "; ".join(f"{i}. art. {x.get('articolo')} {x.get('eid')} {x.get('tipo')}/{x.get('rapporto')} "
                           f"[{x.get('forza')}]" + (f" cita {', '.join(x['pronunce'])}" if x.get("pronunce") else "")
                           for i, x in enumerate(st.get("ranked") or [], 1))
        lines.append(f"| strategy | {_show(st.get('status'))}: {ranked or '—'}; reordered {int(bool(st.get('reordered')))}"
                     + (f" (model order {', '.join(st['model_order'])})" if st.get("reordered") else "")
                     + f"; risk {_show(st.get('risk'))}"
                     + (f"; objections answered {sum(1 for o in st['objections'] if o.get('answered'))}/{len(st['objections'])}"
                        if st.get("objections") else "")
                     + (f"; contro-difesa answered {sum(1 for o in st['contro_difesa'] if o.get('answered'))}/{len(st['contro_difesa'])}"
                        if st.get("contro_difesa") else "")
                     + (f"; error {st['error']}" if st.get("error") else "") + " |")
        riq = st.get("riqualificazione") or {}
        if riq:
            lines.append("| riqualificazione | " + (f"art. {riq.get('articolo')} (linea {riq.get('n')})" if riq.get("linea")
                                                    else f"nessuna tra {', '.join(riq.get('alternative') or []) or '—'}") + " |")
        lines.append("| strategy counters | " + (", ".join(f"{k} {v}" for k, v in (row.get("strategy_counters") or {}).items())
                                                  or "—") + " |")
    if row.get("rites_block"):          # v79.18.1 (C6): the three riti with their status
        lines.append("| rites block | " + "; ".join(f"{x.get('name')}: {x.get('status')}" for x in row["rites_block"]) + " |")
    r = row.get("retrieval") or {}
    lines.append(f"| retrieval_quesiti | queries {_show(r.get('queries'))}, source {_show(r.get('source'))}, reason "
                 f"{_show(r.get('reason'))}, added {_show(r.get('added'))}"
                 + (f", gated {_show(r.get('gated'))}" if r.get("gated") else "") + " |")
    lines.append("| retrieval counters | " + ", ".join(f"{k} {v}" for k, v in (row.get("retrieval_counters") or {}).items())
                 + " |")
    lines.append("| qa | " + ", ".join(f"{k} {v}" for k, v in (row.get("qa") or {}).items()) + " |")
    lines.append(f"| seconds | {row.get('seconds')} |")
    stages = row.get("stages") or {}      # v79.17.1 (item 8): where the time goes, stage by stage
    lines.append("| stages (s) | " + (", ".join(f"{k} {stages[k]}" for k in STAGES if k in stages) or "—") + " |")
    lines.append("| retrieval stages (s) | " + retrieval_stages_text(row.get("retrieval_stages") or []) + " |")   # v79.18.2
    return "\n".join(lines)


def report(run: Dict[str, Any], diff: Optional[List[Dict[str, Any]]], baseline_name: str = "") -> str:
    out = [f"# replay_cases — version {run.get('version')} — model {run.get('model')} @ {run.get('base_url') or '(env)'}",
           "", "flags: " + ", ".join(f"{k}={v}" for k, v in (run.get("flags") or {}).items()), ""]
    for row in run.get("cases") or []:
        out.append(case_table(row))
        out.append("")
    if diff is not None:
        out.append(f"## Diff against {baseline_name or 'the baseline'} (before → now)")
        out.append("")
        for d in diff:
            if d.get("new"):
                out.append(f"- **{d['id']}**: new (not in the baseline)")
            elif d.get("gone"):
                out.append(f"- **{d['id']}**: in the baseline only")
            else:
                moved = [c for c in d["changes"] if c["key"] != "seconds"]
                secs = next(c for c in d["changes"] if c["key"] == "seconds")
                out.append(f"- **{d['id']}**: {len(moved)} value(s) moved; seconds {_show(secs['before'])} → "
                           f"{_show(secs['now'])}")
                for c in moved:
                    out.append(f"  - {c['key']}: {_show(c['before'])} → {_show(c['now'])}")
        out.append("")
    return "\n".join(out)


def read_baseline(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("cases"), list):
        raise ValueError(f"baseline {path}: not a replay_cases JSON")
    return data


# --- main -------------------------------------------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--env", default=".env", help="the flags and keys (python-dotenv); absent: the process environment")
    ap.add_argument("--base-url", help="LLM_BASE_URL for this run (the GPU host's OpenAI-compatible endpoint)")
    ap.add_argument("--model", help="LLM_MODEL for this run")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="one env override, repeatable")
    ap.add_argument("--cases", type=Path, default=CASES_DIR, help=f"the folder of <id>.txt cases ({CASES_DIR})")
    ap.add_argument("--only", nargs="*", help="case ids (the file stems)")
    ap.add_argument("--out", help="the run's JSON")
    ap.add_argument("--baseline", help="a JSON this script wrote: the diff is against it")
    ap.add_argument("--report", help="the markdown (also printed)")
    args = ap.parse_args(argv)
    if os.path.isfile(args.env):
        from dotenv import load_dotenv
        load_dotenv(args.env)
    if args.base_url:
        os.environ["LLM_BASE_URL"] = args.base_url
    if args.model:
        os.environ["LLM_MODEL"] = args.model
    for item in args.set:
        key, _, value = item.partition("=")
        os.environ[key.strip()] = value
    try:
        baseline = read_baseline(args.baseline)
    except Exception as exc:    # noqa: BLE001
        print(f"replay_cases: {exc}", file=sys.stderr)
        return 1
    from src import version
    cases = load_cases(args.cases, args.only)
    if not cases:
        print(f"replay_cases: no case in {args.cases}", file=sys.stderr)
        return 1
    run = {"version": version.current(), "model": os.getenv("LLM_MODEL") or "", "base_url": os.getenv("LLM_BASE_URL") or "",
           "flags": {k: os.getenv(k) or "" for k in FLAGS}, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "cases": []}
    for case_id, text in cases:
        print(f"replay_cases: {case_id} …", file=sys.stderr)
        row = run_case(case_id, text)
        run["cases"].append(row)
        print(f"replay_cases: {case_id} done in {row['seconds']} s" + (f" ERROR {row['error']}" if row["error"] else ""),
              file=sys.stderr)
    diff = diff_rows(run["cases"], baseline["cases"]) if baseline else None
    run["diff"] = diff
    text = report(run, diff, args.baseline or "")
    if args.out:
        Path(args.out).write_text(json.dumps(run, ensure_ascii=False, indent=1), encoding="utf-8")
    if args.report:
        Path(args.report).write_text(text, encoding="utf-8")
    print(text)
    return 1 if any(r.get("error") for r in run["cases"]) else 0


if __name__ == "__main__":
    sys.exit(main())
