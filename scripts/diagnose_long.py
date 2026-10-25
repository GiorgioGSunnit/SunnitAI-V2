"""Read-only diagnosis of many long cases: which stage loses the expected articles, where unexpected ones come from.

Across many long narrative cases at once: which stage of the strategy pipeline loses the expected
articles (understanding, finding, ranking/selection), and where the articles an answer cites
without being expected come from.

Each case is one run of run_case_analysis_pipeline, the call dryrun_strategy.py makes (an eval
case's question as a chat message with no document; a --file as an upload with its --request),
with the code of the CURRENT FOLDER and the flags of --env (the live .env). Each case runs in its
own child process, at most 3 at once: the capture replaces module-level functions
(_resolve_institution, the fusion searches, the vote, the applicability check, ...) with wrappers
that record what they return and change nothing, which is only safe with one case per process.

Per case it records:
  1 UNDERSTANDING  the frame (areas, posture, tasks, central issues), the analysis' institutions,
                   each one resolved to an article with its method (rubrica, LLM title, heading
                   index, fulltext, the model's number, none) and the rubrica it landed on, which
                   is checked against the institution's words (unrelated / generic / no rubrica).
                   For each expected article: its rubrica, whether the understanding names it
                   (an institution resolved to it, or an institution, central issue, question
                   title or keyword whose words cover half its rubrica), whether the frame's
                   areas allow its code.
  2 FINDING        every article of the pool with its channels: user text, rulings' agreement,
                   v1's choice, institution, case-text fulltext and vector, distilled bullets,
                   frame queries, vote, method list, cross-reference, issue spotting; its fusion
                   rank; the vote's ruling count.
  3 RANKING        per article: survived the cap (and why: multi-signal, heading, fill position),
                   the applicability verdict or its exemption, in the draft prompt, in the plan
                   (and the plan check's notes), MENTIONED in the answer (tests/eval_narrative.py).
  4 PRECISION      every article the answer cites that is not expected (all of them for a --file
                   case, which has no answer key): its channels, exemption, plan protection, and
                   whether its stored rubrica fits the line that cites it -- the qualification
                   check's own comparison (_QUALIFICATION_SYSTEM), one extra call per case.

Then one report: (A) loss funnel of the expected articles, (B) origins of the articles cited
without being expected, by channel and by exemption, (C) the 20 most frequent of those,
(D) per area -- plus the understanding, finding and ranking figures behind them, and a line per case.

Cases by default: every case with expected articles among the 12 tuning ones (sempronio ... l4),
h1-h11 and n1-n9. --only picks some of them; --file/--request add documents (one --request per
--file, in order; a --file without one gets the generic request).

Run from the app folder, with its venv and live .env:
    cd /opt/chatbot && venv/bin/python scripts/diagnose_long.py \\
        --file "tests/po_cases/Leonardo Agricoltore.docx" --request "<the PO's message>"
    ... --only h3,n2              # some cases only
    ... --out /root/diag          # default: ./diagnose_long_runs/<date-time>
    ... --report-only DIR         # rebuild DIR/report.md from DIR/cases/*.json, no run
    ... --compare 20261025-0940 20261026-1010   # after the run: tables A and B, those runs and this one
    ... --compare OLD NEW --compare-only        # runs already made, no new run

--dump-llm (on by default; --no-dump-llm to skip) saves every chat call of a case -- frame, analysis,
distill, v1's choice, issue spotting, applicability batches, principles, plan, plan stance, plan check,
each section, stance, qualification, rubrica-fit -- with the exact messages, the raw reply, max_tokens,
temperature, a prompt token estimate and the duration, to cases/<id>/llm/NN_<stage>.txt (NN = call
order), through a wrapper of ai_chat._call_chat set in the child only. cases/<id>/material.txt holds
the draft/plan prompt's articles and rulings as placed there, each article with its provenance.

Writes only files under --out: cases/<id>.json (the capture and the answer's text: test cases
and the PO's test document, never a user's upload), cases/<id>/material.txt, cases/<id>/llm/,
logs/<id>.log, report.md, report.json, and
vote_idf.json when the vote is on without VOTE_POOL_IDF_FILE (the counts every process would
otherwise make for itself, made once). Every Neo4j session is read-only; nothing else is written.
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

DEFAULT_REQUEST = "Delinea una strategia difensiva per il cliente sulla base del documento allegato."
TUNING_IDS = ["sempronio", "sempronio_no_hint", "p1", "p2", "p3", "c1", "c2", "c3", "l1", "l2", "l3", "l4"]
HELDOUT_IDS = [f"h{i}" for i in range(1, 12)] + [f"n{i}" for i in range(1, 10)]
MAX_PARALLEL = 3
CASE_TIMEOUT = 1800

FLAGS = ["CASE_ANALYSIS_ENABLED", "CASE_POOL_FUSION_ENABLED", "FUSION_CLEAN_ENABLED", "CASE_FRAME_ENABLED",
         "STRUCTURED_ANALYSIS_ENABLED", "HEADING_INDEX_ENABLED", "ISSUE_SPOTTING_ENABLED",
         "APPLICABILITY_CHECK_ENABLED", "METHOD_CANDIDATES_ENABLED", "VOTE_POOL_ENABLED", "VOTE_POOL_MODE",
         "VOTE_POOL_K", "RULING_CARDS_ENABLED", "PLAN_THEN_WRITE_ENABLED", "PLAN_KEEP_APPLICABLE_ENABLED",
         "QUALIFICATION_CHECK_ENABLED", "DRAFT_CITATIONS_ONLY_ENABLED", "CLIENT_STANCE_CHECK_ENABLED",
         "STRATEGY_DRAFT_RULES_ENABLED", "METHOD_CANDIDATES_ENABLED", "CHECKLIST_GATED_ENABLED", "FRAME_SANITY_ENABLED",
         "RESOLUTION_STRICT_ENABLED", "CHECK_PRIORITY_ENABLED", "PLAN_KEEP_CHECKED_ENABLED", "CHECK_GRADED_ENABLED"]

# Furthest stage first-to-last; an expected article's stage is the first one it fails.
STAGES = ["LOST_AT_UNDERSTANDING", "NOT_IN_POOL", "BELOW_CAP", "REJECTED_BY_CHECK", "NOT_IN_PROMPT",
          "NOT_IN_PLAN", "NOT_WRITTEN", "MENTIONED"]
STAGE_MEANING = {
    "LOST_AT_UNDERSTANDING": "no channel found it, and nothing in the frame/analysis names it",
    "NOT_IN_POOL": "named by the understanding, but no channel found it",
    "BELOW_CAP": "found (fusion ranking, or the vote's top 20), cut before the pool",
    "REJECTED_BY_CHECK": "in the pool, dropped by the applicability check",
    "NOT_IN_PROMPT": "past the check, left out of the draft prompt by its trimming",
    "NOT_IN_PLAN": "in the prompt, not in the plan (PLAN_THEN_WRITE on)",
    "NOT_WRITTEN": "in the plan (or the prompt, without a plan), not MENTIONED in the answer",
    "MENTIONED": "MENTIONED with its code (tests/eval_narrative.py's primary score)",
}
CHANNELS = ["user_text", "institution", "fulltext", "vector", "distill", "frame", "vote", "method_list",
            "crossref", "rulings_agree", "v1_choice", "issue_spotting"]
VOTE_FOUND_TOP = 20     # vote_pool.TAG_TOP: a voted article below this is not "found"

_EVAL_TO_ABBR = {"cp": "c.p.", "cc": "c.c.", "cpc": "c.p.c.", "cpp": "c.p.p.", "cpa": "c.p.a."}
_GENERIC_RUBRICA = re.compile(
    r"^\s*(?:disposizion\w*\s+(?:general|comun|final|transitor|preliminar)|definizion|oggetto\b|ambito\b"
    r"|campo di applicazione|finalit|principi\w*\s+general|norme general|abrogazion|entrata in vigore|rinvi)",
    re.IGNORECASE)
_PREIMPORT = ("src.rag.plan_then_write", "src.rag.vote_pool", "src.rag.ruling_cards", "src.rag.method_candidates",
              "src.rag.issue_spotting", "src.rag.doc_lookup", "src.rag.nodes.retrieval", "src.rag.nodes.cypher")


def _lab(ref) -> str:
    return f"{ref[0]} {ref[1]}"


def _labs(refs) -> list:
    return [_lab(r) for r in refs or [] if r]


def _plain(value):
    """A JSON-safe snapshot: tuple keys as labels, tuples as lists."""
    if isinstance(value, dict):
        return {(k if isinstance(k, str) else _lab(k) if isinstance(k, tuple) and len(k) == 2 else str(k)): _plain(v)
                for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _item_label(item: dict) -> str:
    """An expected item's label: its own, else "art. 582/583 cp" for an any_of group."""
    return item.get("label") or f"art. {item.get('article') or '/'.join(map(str, item.get('any_of') or []))} {item['code']}"


def _eval_module():
    """tests/eval_narrative.py of the folder we run from: mentions() is the MENTIONED score."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("eval_narrative", os.path.join(os.getcwd(), "tests",
                                                                                 "eval_narrative.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_only_neo4j():
    import neo4j
    if getattr(neo4j.Driver.session, "_read_only", False):
        return
    real = neo4j.Driver.session

    def session(self, **config):
        config["default_access_mode"] = neo4j.READ_ACCESS
        return real(self, **config)
    session._read_only = True
    neo4j.Driver.session = session     # every session in this process is read-only


# =================================================================================================
# Child: one case, instrumented
# =================================================================================================

def _method(reason: str) -> str:
    """_resolve_institution's logged reason -> its method."""
    r = (reason or "").lower()
    for prefix, method in (("titolo_articolo", "heading"), ("title", "title"), ("heading index", "heading_index"),
                           ("fulltext", "fulltext"), ("model's numero", "model_number")):
        if r.startswith(prefix):
            return method
    return "none"


def _rubrica(row) -> tuple:
    s = (row or {}).get("s") or {}
    heading = (s.get("titolo_articolo") or "").strip()
    if heading:
        return heading, "titolo_articolo"
    heading = (s.get("title") or "").strip()
    return heading, "title" if heading else ""


def _checked_slots(dg) -> int:
    """How many articles the applicability check sends to the model in this process's settings."""
    if dg.feature_flags.enabled(getattr(dg, "CHECK_GRADED_FLAG", "CHECK_GRADED_ENABLED")):
        return getattr(dg, "_GRADED_MAX_CHECKED", dg._APPLICABILITY_MAX_CHECKED)
    return dg._APPLICABILITY_MAX_CHECKED


def _instrument(dg) -> dict:
    """Wrappers that record what the pipeline's steps return and change nothing."""
    import importlib
    for name in _PREIMPORT:
        importlib.import_module(name)
    from src.rag import vote_pool
    from src.rag.issue_spotting import _stems
    from src.rag.nodes.cypher import CROSSREF_SOURCE

    cap = {"events": [], "phase": [], "institutions": [], "searches": [], "hits_calls": [], "errors": [],
           "v1": None, "v1_sources": None, "v1_choice_failed": None, "v2": None, "fusion_raw": None,
           "bullets": [], "voted": [], "pool_rows": None, "method_candidates": [], "check": None,
           "prompt_labels": None}
    events = cap["events"]

    def vlog(event, data=None, duration_ms=None, **_):
        events.append([event, _plain(data), duration_ms])
    for name, module in list(sys.modules.items()):
        if name.startswith("src.rag") and callable(getattr(module, "vlog", None)):
            module.vlog = vlog

    def wrap(module, name, after, phase=None):
        real = getattr(module, name)

        def wrapper(*args, **kwargs):
            ctx = {"n0": len(events), "phase": cap["phase"][-1] if cap["phase"] else ""}
            if phase:
                cap["phase"].append(phase)
            try:
                result = real(*args, **kwargs)
            finally:
                if phase:
                    cap["phase"].pop()
            try:
                after(result, args, kwargs, ctx)
            except Exception as exc:       # a capture bug never changes the run
                cap["errors"].append(f"{name}: {type(exc).__name__}: {exc}")
            return result
        setattr(module, name, wrapper)

    def on_institution(rows, args, kwargs, ctx):
        istituto = (args[2] if len(args) > 2 else kwargs.get("istituto")) or {}
        parole = [p for p in ((args[3] if len(args) > 3 else kwargs.get("parole_chiave")) or []) if isinstance(p, str)]
        reason = next(((d or {}).get("reason") for e, d, _ in events[ctx["n0"]:] if e == "case_analysis_institution"),
                      "")
        nome = str(istituto.get("nome") or "")
        name_stems = _stems(nome)
        kw_stems = set().union(*(_stems(p) for p in parole)) if parole else set()
        articles, seen = [], set()
        for row in rows or []:
            if not row.get("_article") or _lab(row["_article"]) in seen:
                continue
            seen.add(_lab(row["_article"]))
            heading, field = _rubrica(row)
            stems = _stems(heading)
            articles.append({"label": _lab(row["_article"]), "rubrica": heading, "field": field,
                             "fits_name": bool(name_stems & stems), "fits_keyword": bool(kw_stems & stems),
                             "generic": bool(_GENERIC_RUBRICA.search(heading))})
        cap["institutions"].append({"phase": ctx["phase"], "nome": nome, "codice": str(istituto.get("codice") or ""),
                                    "numero_indicativo": str(istituto.get("numero_indicativo") or ""),
                                    "parole_chiave": parole, "reason": reason or "not found",
                                    "method": _method(reason), "articles": articles})

    def on_candidates(result, args, kwargs, ctx):
        facts, analysis, ruling_rows, code_refs = args[:4]
        counts = args[4] if len(args) > 4 else kwargs.get("ruling_counts")
        agreed = (sorted((r for r, n in counts.items() if n >= 2), key=lambda r: -counts[r]) if counts
                  else dg._cited_by_rulings(ruling_rows))
        damages = (dg._CIVIL_DAMAGES_NORMS if analysis.get("area") != "penale" and dg._DAMAGES_RE.search(facts)
                   else [])
        cap["v1_sources"] = {"code_search": _labs(code_refs), "rulings_cited": _labs(agreed[:6]),
                             "damages_backbone": _labs(damages), "model_norms": _labs(dg._norm_refs(analysis)),
                             "ruling_counts": {_lab(r): n for r, n in (counts or {}).items() if n >= 2}}

    def on_choose(result, args, kwargs, ctx):
        cap["v1_choice_failed"] = result is None

    def on_v1(result, args, kwargs, ctx):
        cap["v1"] = {k: _labs(result.get(k)) for k in ("keep", "named", "candidates", "chosen", "selected")}
        cap["v1"]["spotted"] = _labs(r.get("_article") for r in result.get("spotted") or [])

    def on_v2(result, args, kwargs, ctx):
        cap["v2"] = {"selected": _labs(result.get("selected")),
                     "institutions_resolved": _labs(result.get("institutions_resolved")),
                     "heading_matched": _labs(result.get("heading_matched"))}

    def on_search(channel):
        def after(result, args, kwargs, ctx):
            if ctx["phase"] != "fusion":
                return
            text = args[1] if len(args) > 1 else kwargs.get("text", "")
            if channel == "both":
                cap["searches"].append({"text": text, "channel": "vector", "refs": _labs(result[0])})
                cap["searches"].append({"text": text, "channel": "fulltext", "refs": _labs(result[1])})
            else:
                cap["searches"].append({"text": text, "channel": channel, "refs": _labs(result)})
        return after

    def on_hits(result, args, kwargs, ctx):
        if ctx["phase"] == "fusion":
            cap["hits_calls"].append((args, result))

    def on_bullets(result, args, kwargs, ctx):
        cap["bullets"] = [b.get("testo", "") for b in result or []]

    def on_vote(result, args, kwargs, ctx):
        cap["voted"] = list(result or [])

    def on_fusion(result, args, kwargs, ctx):
        main = cap["hits_calls"][-1] if cap["hits_calls"] else None
        hits = dict(main[1][0]) if main else {}
        signals = [list(lst) for lst in main[0]] if main else []
        keep, fused = list(result.get("keep") or []), list(result.get("candidates") or [])
        not_keep = [r for r in fused if r not in keep]
        how = vote_pool.mode() if vote_pool.enabled() else None
        if cap["voted"] and how and how[0] == "rrf":
            not_keep = [r for r in vote_pool.weighted_rrf(not_keep, cap["voted"], how[1], how[2]) if r not in keep]
        heading = {r for r in fused if _lab(r) in set((cap["v2"] or {}).get("heading_matched") or [])}
        exempt = {r: ("multi-signal" if hits.get(r, 0) >= 2 else "heading") for r in not_keep
                  if hits.get(r, 0) >= 2 or r in heading}
        fill = [r for r in not_keep if r not in exempt]
        cap["fusion_raw"] = {
            "fused": _labs(fused), "keep": _labs(keep), "named": _labs(result.get("named")),
            "chosen": _labs(result.get("chosen")), "selected": _labs(result.get("selected")),
            "spotted": _labs(r.get("_article") for r in result.get("spotted") or []),
            "signals": dict(zip(["current", "baseline", "distill", "frame"], (_labs(s) for s in signals))),
            "hits": {_lab(r): n for r, n in hits.items()}, "exempt_from_cap": {_lab(r): why for r, why in exempt.items()},
            "fill_order": _labs(fill), "cap": dg._FUSION_POOL_CAP, "vote_mode": ":".join(map(str, how)) if how else ""}
        cap["hits_calls"] = []

    def on_method(result, args, kwargs, ctx):
        rows = {}
        for row in result.get("article_rows") or []:
            ref = row.get("_article")
            if not ref:
                continue
            item = rows.setdefault(_lab(ref), {"label": _lab(ref), "sources": [], "rubrica": "", "field": "",
                                               "document": ((row.get("d") or {}).get("name") or "")})
            source = row.get("_source") or ""
            if source and source not in item["sources"]:
                item["sources"].append(source)
            for tag in ("_multi_signal", "_heading_match", "_vote"):
                if row.get(tag):
                    item[tag[1:]] = True
            if row.get("_vote_rank"):
                item["vote_rank"] = row["_vote_rank"]
            if not item["rubrica"]:
                item["rubrica"], item["field"] = _rubrica(row)
        cap["pool_rows"] = list(rows.values())
        cap["method_candidates"] = _labs(result.get("method_candidates") or [])
        cap["crossref_source"] = CROSSREF_SOURCE

    def on_check(result, args, kwargs, ctx):
        facts, request, rows = args[:3]
        to_check, texts = dg._applicability_candidates(facts, request, rows)
        named = set(dg._refs_in(dg._case_text(facts, request)))
        reasons = defaultdict(set)
        for row in rows:
            ref = row.get("_article")
            if not ref or ref in to_check:
                continue
            if ref in named:
                reasons[_lab(ref)].add("user_text")
            if row.get("_heading_match"):
                reasons[_lab(ref)].add("heading")
            if row.get("_source") == CROSSREF_SOURCE:
                reasons[_lab(ref)].add("crossref")
        exempt = {label: next(r for r in ("user_text", "heading", "crossref") if r in rs) for label, rs in reasons.items()}
        output = {}
        for row in result:
            if row.get("_article"):
                output.setdefault(_lab(row["_article"]), row.get("_applicability") or "kept")
        cap["check"] = {"input": list(dict.fromkeys(_lab(r["_article"]) for r in rows if r.get("_article"))),
                        "exempt": exempt, "no_text": [_lab(a) for a in to_check if _lab(a) not in texts],
                        "unchecked": list(texts)[_checked_slots(dg):], "output": output}

    def on_prompt(result, args, kwargs, ctx):
        cap["prompt_labels"] = re.findall(r"^\[art\. (.+?)\]$", result[0] or "", re.MULTILINE)
        cap["prompt_blocks"] = (result[0] or "", result[1] or "")

    wrap(dg, "_resolve_institution", on_institution)
    wrap(dg, "_candidate_articles", on_candidates)
    wrap(dg, "_choose_articles", on_choose)
    wrap(dg, "_select_articles", on_v1, phase="v1")
    wrap(dg, "_select_articles_v2", on_v2, phase="v2")
    wrap(dg, "_fusion_vector_search", on_search("vector"))
    wrap(dg, "_fusion_fulltext_search", on_search("fulltext"))
    wrap(dg, "_fusion_clean_searches", on_search("both"))
    wrap(dg, "_fusion_hits_and_best_rank", on_hits)
    wrap(dg, "_fusion_distill_bullets", on_bullets)
    wrap(vote_pool, "ruling_vote", on_vote)
    wrap(dg, "_select_articles_fusion", on_fusion, phase="fusion")
    wrap(dg, "_add_method_candidates", on_method)
    wrap(dg, "_filter_applicable_articles", on_check)
    wrap(dg, "_fit_parere_prompt", on_prompt)
    return cap


# The function a chat call is made from -> its stage name, innermost first.
_LLM_STAGES = {
    "_case_frame": "frame", "_analyse_case_v2": "case_analysis_v2", "_analyse_case": "case_analysis_v1",
    "_analyse_case_structured": "case_analysis_structured", "_structured_facts": "structured_facts",
    "_structured_grid": "structured_grid", "_fusion_distill_bullets": "distill", "_choose_articles": "v1_choice",
    "spot_issues": "issue_spotting", "spotted_rows": "issue_spotting", "_applicability_batch": "applicability",
    "_graded_batch": "applicability",
    "_extract_principles": "principles", "_issue_check": "plan_check", "_opposing_arguments": "stance",
    "_check_client_stance": "stance_rewrite", "make_plan": "plan", "_check_qualifications": "qualification",
    "_rubrica_fits": "rubrica_fit", "run_case_analysis_pipeline": "draft",
}
_SKIP_FRAMES = {"_chat", "_chat_json", "call_chat"}


def _llm_stage(frame) -> str:
    """The stage of the chat call made from `frame`: the first function up the stack with a known
    stage (a plan section is "section_<its title>"), else the nearest caller's own name."""
    nearest, names = "", []
    while frame is not None and len(names) < 60:
        name, module = frame.f_code.co_name, frame.f_globals.get("__name__", "")
        names.append(name)
        if not nearest and name not in _SKIP_FRAMES:
            nearest = name
        if name == "write" and module.endswith("plan_then_write"):
            job = frame.f_locals.get("job") or ("",)
            return "section_" + re.sub(r"[^a-z0-9]+", "_", str(job[0]).lower()).strip("_")[:40]
        if name == "_graded_batch" and frame.f_locals.get("retry"):
            return "applicability_retry"
        if name in _LLM_STAGES:
            stage = _LLM_STAGES[name]
            if name == "_opposing_arguments":
                outer = frame.f_back
                while outer is not None:
                    if outer.f_code.co_name == "make_plan":
                        return "plan_stance"
                    outer = outer.f_back
            return stage
        frame = frame.f_back
    return f"other_{nearest or 'unknown'}"


def _dump_llm(dg, dump_dir: str, t_start: float) -> None:
    """Every chat call of this process through a wrapper of ai_chat._call_chat (and of each module's
    own imported name for it), which saves the messages, the raw reply, the call's settings and its
    duration to dump_dir/NN_<stage>.txt, NN in the order the calls are made. The call itself is the
    real one, unchanged."""
    import threading
    from src.rag import ai_chat

    real = ai_chat._call_chat
    lock, counter = threading.Lock(), [0]
    os.makedirs(dump_dir, exist_ok=True)

    def call_chat(messages, *args, **kwargs):
        stage = _llm_stage(sys._getframe(1))
        if messages and str(getattr(messages[0], "content", "")).endswith(dg._JSON_RETRY_SUFFIX):
            stage += "_json_retry"
        with lock:
            counter[0] += 1
            number = counter[0]
        params = dict(zip(("max_tokens", "stop", "temperature"), args))
        params.update(kwargs)
        started = time.time()
        reply, error = None, None
        try:
            reply = real(messages, *args, **kwargs)
            return reply
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            parts, sizes = [], []
            for m in messages or []:
                kind = {"system": "SYSTEM", "human": "USER", "ai": "ASSISTANT"}.get(getattr(m, "type", ""), "MESSAGE")
                content = m.content if isinstance(m.content, str) else json.dumps(m.content, ensure_ascii=False)
                sizes.append(f"{kind.lower()} {dg._estimate_tokens(content)}")
                parts.append(f"===== {kind} =====\n{content}")
            total = sum(dg._estimate_tokens(m.content if isinstance(m.content, str) else str(m.content))
                        for m in messages or [])
            head = [f"stage: {stage}", f"call: {number:02d} (thread {threading.current_thread().name})",
                    f"max_tokens: {params.get('max_tokens')}",
                    f"temperature: {params['temperature'] if params.get('temperature') is not None else 'not passed (the chat model default)'}",
                    f"stop: {params.get('stop')}",
                    f"prompt estimate: {total} tokens ({', '.join(sizes)}; defensive_generation._estimate_tokens)",
                    f"started: +{started - t_start:.1f}s", f"duration: {time.time() - started:.1f}s",
                    f"status: {'ERROR ' + error if error else 'ok'}",
                    "note: max_tokens is what the caller asked; _call_chat lowers it itself on a context overflow"]
            body = "\n".join(head) + "\n\n" + "\n\n".join(parts) + "\n\n===== REPLY =====\n" + (
                reply if isinstance(reply, str) else "(no reply)") + "\n"
            try:
                with open(os.path.join(dump_dir, f"{number:02d}_{stage}.txt"), "w", encoding="utf-8") as fh:
                    fh.write(body)
            except OSError as exc:
                print(f"llm dump {number:02d}_{stage}: {exc}", file=sys.stderr)

    for name, module in list(sys.modules.items()):
        if name.startswith("src.rag") and getattr(module, "_call_chat", None) is real:
            module._call_chat = call_chat


def _provenance_line(t: dict, expected: bool) -> str:
    """One article's provenance, for material.txt."""
    origin = "+".join(t["channels"]) or "-"
    extra = t["details"] + [f"institution {i}" for i in t["institutions"]]
    plan = ("in plan" if t["in_plan"] else "not in plan") if t["in_plan"] is not None else "no plan"
    return (f">> provenance: {origin}" + (f" ({'; '.join(extra)})" if extra else "")
            + f" | fused rank {t['rank'] or '-'}, signals {'+'.join(t['signals']) or '-'}"
            + (f", vote rank {t['vote_rank']}" if t["vote_rank"] else "")
            + f" | cap: {t['cap']} | check: {t['check']} | {plan}"
            + (f" ({', '.join(t['plan_notes'])})" if t["plan_notes"] else "")
            + f" | answer: {'mentioned' if t['mentioned'] else 'not mentioned'}"
            + (" | EXPECTED" if expected else ""))


def _write_material(path: str, record: dict, blocks, ev) -> None:
    """material.txt: the articles and rulings exactly as the draft/plan prompt holds them
    (_fit_parere_prompt, after trimming and refill), a provenance line (">> ") under each article's
    label; then the articles left out by the trimming and those the check dropped."""
    case = analyse_case(record, ev)
    ekey = _ekey_fn(ev)
    traces = {ekey(t["label"]): t for t in case["pool_traces"]}
    expected = {(str(a).lower(), item["code"]) for item in record.get("expected") or []
                for a in item.get("any_of") or [item["article"]]}

    def line_for(label):
        k = ekey(label)
        t = traces.get(k)
        return _provenance_line(t, k in expected) if t else ">> provenance: not in the pool"

    tokens = (record.get("prompt") or {}).get("tokens") or {}
    out = [f"# {record['id']}: what the writer saw", "",
           "The ARTICOLI and SENTENZE blocks of the draft prompt, as _fit_parere_prompt built them (after trimming "
           "and refill); the one-call draft and the plan read these. Lines starting with '>> ' are added here, "
           "not part of the prompt. Each section's own material is in llm/NN_section_*.txt.",
           f"Prompt: {tokens.get('total')} est. tokens, budget {tokens.get('budget')}, step {tokens.get('step')}, "
           f"dropped to fit {tokens.get('dropped_for_size', '-')}, refilled {tokens.get('refilled', '-')}.", ""]
    if not blocks:
        out.append("(no draft prompt was built: the pipeline stopped before it)")
    else:
        articles, rulings = blocks
        out.append("========== ARTICOLI ==========")
        for text_line in articles.split("\n"):
            out.append(text_line)
            m = re.match(r"^\[art\. (.+?)\]$", text_line)
            if m:
                out.append(line_for(m.group(1)))
        out += ["", "========== SENTENZE ==========", rulings or "(none)"]
    left_out = tokens.get("articles_left_out") or []
    out += ["", f"========== LEFT OUT OF THE PROMPT BY ITS TRIMMING ({len(left_out)}) =========="]
    out += [f"{label}\n{line_for(label)}" for label in left_out] or ["(none)"]
    rejected = [t for t in case["pool_traces"] if t["check"] == "rejected"]
    out += ["", f"========== DROPPED BY THE APPLICABILITY CHECK ({len(rejected)}) =========="]
    out += [f"{t['label']}: requisito «{(t['dropped'] or {}).get('requisito', '')}», fatto "
            f"«{(t['dropped'] or {}).get('fatto', '')}», reason «{(t['dropped'] or {}).get('reason', '')}»\n"
            f"{_provenance_line(t, ekey(t['label']) in expected)}"
            for t in rejected] or ["(none)"]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")


def _wait_for_vote_counts(vote_pool, driver, database, log) -> None:
    """The vote is skipped until the IDF counts exist; a diagnosis waits for them instead."""
    if not vote_pool.enabled():
        return
    t0 = time.time()
    while vote_pool.idf_table(driver, database) is None:
        if not vote_pool._computed.get("started") or time.time() - t0 > 900:
            log("vote IDF counts not ready: the vote will be skipped in this case")
            return
        time.sleep(2)
    log(f"vote IDF counts ready ({time.time() - t0:.0f}s)")


def _rubrica_fits(dg, draft: str, refs: list, rows_by_ref: dict) -> dict:
    """{label: rubrica, citing line, compatibile, perche}: the qualification check's comparison
    (_QUALIFICATION_SYSTEM: the stored rubrica against the draft's line citing it), for refs."""
    items = {}
    for ref in refs:
        heading, field = _rubrica(rows_by_ref.get(ref))
        items[_lab(ref)] = {"rubrica": heading, "field": field, "context": dg._citation_context(draft, ref),
                            "compatibile": None, "perche": ""}
    ask = [(label, v) for label, v in items.items() if v["rubrica"] and v["context"]]
    for i in range(0, len(ask), 25):
        batch = ask[i:i + 25]
        human = "\n\n".join(f"[{label}]\nRubrica: {v['rubrica']}\nFrase: {v['context']}" for label, v in batch)
        try:
            verdicts = dg._chat_json([dg.SystemMessage(content=dg._QUALIFICATION_SYSTEM),
                                      dg.HumanMessage(content=human)],
                                     max_tokens=max(600, 90 * len(batch)), temperature=0)
        except Exception as exc:
            for label, v in batch:
                v["perche"] = f"check failed: {str(exc)[:120]}"
            continue
        keyed = {dg._applicability_key(label): label for label, _ in batch}
        for v in verdicts if isinstance(verdicts, list) else []:
            label = keyed.get(dg._applicability_key(str((v or {}).get("articolo") or ""))) if isinstance(v, dict) else None
            if label:
                items[label]["compatibile"] = v.get("compatibile")
                items[label]["perche"] = str(v.get("perche") or "")[:240]
    return items


def _expected_info(dg, ev, session, documents, expected, analysis, case_text, institutions) -> list:
    """Per expected item and alternative: its rubrica, whether the understanding names it, whether
    the frame's areas allow its code."""
    from src.rag import vote_pool
    from src.rag.issue_spotting import _stems, fetch_article

    allowed = vote_pool.allowed_codes(vote_pool.case_areas(analysis), vote_pool.named_codes(case_text))
    frame = analysis.get("frame") or {}
    names = [(f"central issue «{c.get('nome')}»", c.get("nome") or "") for c in frame.get("central_issue_candidates") or []]
    for q in analysis.get("questioni") or []:
        names.append((f"question «{q.get('titolo')}»", str(q.get("titolo") or "")))
        names += [(f"institution «{i.get('nome')}»", str(i.get("nome") or "")) for i in q.get("istituti") or []
                  if isinstance(i, dict)]
        names += [(f"keyword «{k}»", k) for k in q.get("parole_chiave") or [] if isinstance(k, str)]
    resolved = defaultdict(list)
    for inst in institutions:
        if inst["phase"] == "v2":
            for a in inst["articles"]:
                for key in ev.mentions(f"art. {a['label']}")[:1]:
                    resolved[key].append(f"{inst['nome']} ({inst['method']})")
    out = []
    for item in expected:
        alts = []
        for article in item.get("any_of") or [item["article"]]:
            number, key = dg._article_num(str(article)), item["code"]
            abbr = _EVAL_TO_ABBR.get(key)
            rows = []
            try:
                if abbr:
                    rows = dg._fetch_any(session, documents, [(number, abbr)])
                else:
                    for doc in documents:
                        if ev.code_of_document(doc.get("name") or "") == key:
                            rows = fetch_article(session, doc["id"], number)
                            if rows:
                                break
            except Exception as exc:
                rows = []
                print(f"expected {number} {key}: fetch failed: {exc}", file=sys.stderr)
            heading, field = _rubrica(rows[0] if rows else None)
            stems = _stems(heading)
            need = max(1, math.ceil(len(stems) / 2))
            named_by = [what for what, text in names if stems and len(_stems(text) & stems) >= need]
            ekey = (str(article).lower(), key)
            alts.append({"article": str(article).lower(), "code": key, "rubrica": heading, "field": field,
                         "in_corpus": bool(rows),
                         "resolved_by_institution": resolved.get(ekey, []),
                         "named_by": named_by[:6],
                         "area_allows": None if not abbr or abbr not in vote_pool.MAIN_CODES or allowed is None
                         else abbr in allowed})
        out.append({"label": _item_label(item), "alts": alts})
    return out


def run_child(spec_path: str, env_path: str) -> None:
    spec = json.load(open(spec_path, encoding="utf-8"))

    def log(message):
        print(f"[{spec['id']}] {message}", file=sys.stderr, flush=True)

    sys.path.insert(0, os.getcwd())                 # the code of the folder we run from
    from dotenv import load_dotenv
    load_dotenv(env_path)
    for name, value in (spec.get("settings") or {}).items():
        os.environ[name] = value
    import logging
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")
    logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)
    _read_only_neo4j()

    import src.rag.defensive_generation as dg
    from src.rag import feature_flags
    ev = _eval_module()
    cap = _instrument(dg)
    case_dir = os.path.join(os.path.dirname(spec["out"]), spec["id"])
    os.makedirs(case_dir, exist_ok=True)
    if spec.get("dump_llm", True):
        _dump_llm(dg, os.path.join(case_dir, "llm"), time.time())
    from src.rag import vote_pool
    from src.rag.doc_lookup import _fetch_doc_names
    from src.rag.main import driver, NEO4J_DATABASE
    log(f"code: {os.path.abspath(dg.__file__)}")
    _wait_for_vote_counts(vote_pool, driver, NEO4J_DATABASE, log)

    if spec.get("file"):
        from src.utils.document import extract_text_from_file
        text, message = extract_text_from_file(spec["file"]), spec.get("request") or DEFAULT_REQUEST
    else:
        text = message = spec["question"]
    request = "" if text.strip() == message.strip() else message   # run_defensive_pipeline's own rule

    t0 = time.time()
    result, status = None, "ok"
    try:
        result = dg.run_case_analysis_pipeline(text, spec.get("lang") or "it", request=request)
    except Exception as exc:
        status = f"ERROR {type(exc).__name__}: {str(exc)[:200]}"
        traceback.print_exc()
    elapsed = time.time() - t0
    log(f"{status}, {elapsed:.0f}s")

    result = result if isinstance(result, dict) else {}
    draft = result.get("draft") or ""
    analysis = result.get("analysis") or {}
    events = cap["events"]

    def last(name, with_key=None):
        found = [d for e, d, _ in events if e == name and (with_key is None or (isinstance(d, dict) and with_key in d))]
        return found[-1] if found else None

    pool_rows = cap["pool_rows"] or []
    rows_by_ref = {}
    record = {"id": spec["id"], "kind": "file" if spec.get("file") else "eval", "file": spec.get("file"),
              "request": request, "text_chars": len(text), "status": status, "seconds": round(elapsed, 1),
              "code": os.path.abspath(dg.__file__),
              "flags": {f: os.getenv(f) for f in FLAGS}, "expected": spec.get("expected"),
              "frame": analysis.get("frame"), "area": analysis.get("area"),
              "questioni": _plain([{k: q.get(k) for k in ("titolo", "categoria", "istituti", "parole_chiave", "voce")}
                                   for q in analysis.get("questioni") or [] if isinstance(q, dict)]),
              "institutions": cap["institutions"], "v1": cap["v1"], "v1_sources": cap["v1_sources"],
              "v1_choice_failed": cap["v1_choice_failed"], "v2": cap["v2"], "fusion": cap["fusion_raw"],
              "bullets": cap["bullets"], "searches": [], "vote": {
                  "ran": bool(last("case_pool_vote")), "info": last("case_pool_vote"),
                  "cut": last("case_pool_vote_cut"), "voted": _labs(cap["voted"][:100]),
                  "voted_total": len(cap["voted"])},
              "pool": pool_rows, "method_candidates": cap["method_candidates"],
              "crossref_source": cap.get("crossref_source"),
              "check": cap["check"], "check_result": last("case_analysis_applicability", "verdicts"),
              "check_calls": [d for e, d, _ in events if e == "case_analysis_applicability" and "batches" in (d or {})],
              "prompt": {"labels": cap["prompt_labels"], "tokens": last("case_analysis_prompt_tokens")},
              "plan": last("case_analysis_plan"), "plan_applicable": last("case_analysis_plan_applicable"),
              "qualification": last("case_analysis_qualification"),
              "draft": draft, "capture_errors": cap["errors"], "events": events}

    case_text = dg._case_text(text, request)
    kinds = {case_text: "baseline"}
    bullets = [b for b in cap["bullets"] if b]
    kinds.update({b: "distill" for b in bullets})
    if bullets:
        kinds.setdefault(" ".join(bullets), "distill")
    kinds.update({q: "frame" for q in dg._frame_queries(analysis.get("frame") or {}) if q not in kinds})
    record["searches"] = [{"kind": kinds.get(s["text"], "other"), "channel": s["channel"], "refs": s["refs"]}
                          for s in cap["searches"]]

    try:
        documents = _fetch_doc_names(driver, NEO4J_DATABASE)
        known = {tuple(p["label"].split(" ", 1)) for p in pool_rows}
        mentioned = dg._cited_articles(draft, known, documents) if draft else []
        flagged = []
        for m in dg._CITED_ARTICLE_RE.finditer(draft):
            code = dg._code_of(m.group(2))
            if code and draft.startswith(" [DA VERIFICARE", m.end()):
                flagged.append(_lab((dg._article_num(m.group(1)), code)))
        record["answer"] = {"chars": len(draft), "mentioned": _labs(mentioned),
                            "mentioned_eval": [list(x) for x in ev.mentions(draft)],
                            "marked_unverified": list(dict.fromkeys(flagged)),
                            "citations": len(result.get("citations") or [])}
        expected_keys = {(str(a).lower(), item["code"]) for item in spec.get("expected") or []
                         for a in item.get("any_of") or [item["article"]]}
        to_fit = [ref for ref in mentioned
                  if not expected_keys or not (set(ev.mentions(f"art. {_lab(ref)}")[:1]) & expected_keys)]
        with driver.session(database=NEO4J_DATABASE) as session:
            pool_refs = [tuple(p["label"].split(" ", 1)) for p in pool_rows]
            fetch = [ref for ref in dict.fromkeys(to_fit)]
            for row in dg._fetch_any(session, documents, fetch):
                if row.get("_article"):
                    rows_by_ref.setdefault(row["_article"], row)
            record["fits"] = _rubrica_fits(dg, draft, to_fit, rows_by_ref) if draft else {}
            record["in_pool_refs"] = len(pool_refs)
            if spec.get("expected"):
                record["expected_info"] = _expected_info(dg, ev, session, documents, spec["expected"], analysis,
                                                         case_text, cap["institutions"])
    except Exception as exc:
        record["capture_errors"].append(f"after the run: {type(exc).__name__}: {exc}")
        traceback.print_exc()
    record["feature_flags_seen"] = {"fusion": cap["fusion_raw"] is not None,
                                    "check": feature_flags.enabled("APPLICABILITY_CHECK_ENABLED"),
                                    "plan": feature_flags.enabled("PLAN_THEN_WRITE_ENABLED")}
    with open(spec["out"], "w", encoding="utf-8") as fh:
        json.dump(record, fh, ensure_ascii=False, indent=1, default=str)
    log(f"saved {spec['out']}")
    try:
        _write_material(os.path.join(case_dir, "material.txt"), json.loads(json.dumps(record, default=str)),
                        cap.get("prompt_blocks"), ev)
    except Exception as exc:
        log(f"material.txt failed: {type(exc).__name__}: {exc}")
        traceback.print_exc()


# =================================================================================================
# Pure: one case's record -> its traces (no app imports)
# =================================================================================================

def _ekey_fn(ev):
    cache = {}

    def ekey(label: str) -> tuple:
        """A pipeline label ("584 c.p.", "95 Codice degli Appalti 2026") -> the eval's (article, code
        key); a law the eval does not know keeps its own name, prefixed "?"."""
        if label not in cache:
            number, _, code = label.partition(" ")
            found = ev.mentions(f"art. {label}")
            if found and found[0][1]:
                cache[label] = (found[0][0], found[0][1])
            else:
                cache[label] = ((found[0][0] if found else number.lower()), "?" + code.strip().lower())
        return cache[label]
    return ekey


def _plan_label(entry: str) -> str:
    """"584 c.p. (contraddice)" -> "584 c.p."."""
    return re.sub(r"\s+\((?:contraddice|fuori tema)\)$", "", entry or "").strip()


def analyse_case(rec: dict, ev) -> dict:
    """Everything the report needs from one case, keyed by the eval's (article, code)."""
    ekey = _ekey_fn(ev)
    display = {}

    def K(label):
        k = ekey(label)
        display.setdefault(k, label)
        return k

    def keys(labels):
        return {K(label) for label in labels or []}

    fusion = rec.get("fusion") or {}
    v1, v2 = rec.get("v1") or {}, rec.get("v2") or {}
    v1_sources = rec.get("v1_sources") or {}
    flags = rec.get("feature_flags_seen") or {}

    channels, details = defaultdict(set), defaultdict(list)
    named = keys(fusion.get("named") or v1.get("named"))
    keep = keys(fusion.get("keep") or v1.get("keep"))
    for k in named:
        channels[k].add("user_text")
    for k in keep - named:
        channels[k].add("rulings_agree")
    for label in v1.get("chosen") or []:
        k = K(label)
        channels[k].add("v1_choice")
        details[k] += [f"v1:{src}" for src in ("code_search", "rulings_cited", "damages_backbone", "model_norms")
                       if label in (v1_sources.get(src) or [])]
    institution_of = defaultdict(list)
    for inst in rec.get("institutions") or []:
        if inst.get("phase") != "v2":
            continue
        for a in inst.get("articles") or []:
            k = K(a["label"])
            channels[k].add("institution")
            institution_of[k].append(f"{inst['nome']} [{inst['method']}]")
    search_channel = {"distill": "distill", "frame": "frame", "other": "other_search"}
    for s in rec.get("searches") or []:
        name = s["channel"] if s["kind"] == "baseline" else search_channel.get(s["kind"], "other_search")
        for label in s["refs"]:
            channels[K(label)].add(name)
    cut = (rec.get("vote") or {}).get("cut") or {}
    for label in cut.get("tagged") or []:
        channels[K(label)].add("vote")
    vote_added = keys(cut.get("added"))
    voted = [K(label) for label in (rec.get("vote") or {}).get("voted") or []]
    vote_rank = {}
    for i, k in enumerate(voted, 1):
        vote_rank.setdefault(k, i)
    for k in keys(rec.get("method_candidates")):
        channels[k].add("method_list")
    crossref_source = rec.get("crossref_source") or "crossref"
    pool = {}
    for row in rec.get("pool") or []:
        k = K(row["label"])
        pool.setdefault(k, row)
        if crossref_source in (row.get("sources") or []):
            channels[k].add("crossref")
    for k in keys(fusion.get("spotted") or v1.get("spotted")):
        channels[k].add("issue_spotting")

    fused = [K(label) for label in fusion.get("fused") or v2.get("selected") or v1.get("selected") or []]
    fused_rank = {}
    for i, k in enumerate(fused, 1):
        fused_rank.setdefault(k, i)
    selected = keys(fusion.get("selected") or v2.get("selected") or v1.get("selected"))
    exempt_cap = {K(label): why for label, why in (fusion.get("exempt_from_cap") or {}).items()}
    fill_pos = {}
    for i, label in enumerate(fusion.get("fill_order") or [], 1):
        fill_pos.setdefault(K(label), i)
    hits = {K(label): n for label, n in (fusion.get("hits") or {}).items()}
    signals_of = defaultdict(list)
    for name, labels in (fusion.get("signals") or {}).items():
        for k in keys(labels):
            signals_of[k].append(name)

    check = rec.get("check")
    check_result = rec.get("check_result") or {}
    verdicts = {K(label): v for label, v in (check_result.get("verdicts") or {}).items()}
    dropped = {K(d.get("article", "")): d for d in check_result.get("dropped") or []}
    voided = {K(d.get("article", "")) for d in check_result.get("voided_rejections") or []}
    prompt_labels = (rec.get("prompt") or {}).get("labels")
    in_prompt = keys(prompt_labels) if prompt_labels is not None else None
    plan = rec.get("plan")
    plan_on = flags.get("plan") and plan is not None and plan.get("issues", 0) > 0
    plan_norme = keys((plan or {}).get("norme"))
    plan_notes = defaultdict(list)
    checked = (plan or {}).get("checked") or {}
    for note in ("off_topic", "kept_by_heading", "kept_by_vote", "kept_as_applicable"):
        for entry in checked.get(note) or []:
            plan_notes[K(_plan_label(entry))].append(note)
    for label in (rec.get("plan_applicable") or {}).get("added_to_last_issue") or []:
        plan_notes[K(label)].append("added_to_last_issue")
    for label in ((plan or {}).get("removed") or {}).get("norme") or []:
        if isinstance(label, str):
            plan_notes[K(_plan_label(label))].append("removed_unknown")
    answer = rec.get("answer") or {}
    mentioned_eval = {tuple(x) for x in answer.get("mentioned_eval") or []}
    mentioned = [K(label) for label in answer.get("mentioned") or []]
    unverified = keys(answer.get("marked_unverified"))
    qualification_flagged = keys((rec.get("qualification") or {}).get("flagged"))

    def check_status(k):
        if check is None:
            return "check off" if not flags.get("check") else "no check run"
        if k not in keys(check.get("input")):
            return "not in pool"
        exempt = {K(label): why for label, why in (check.get("exempt") or {}).items()}
        if k in exempt:
            return f"exempt: {exempt[k]}"
        if k in dropped:
            return "rejected"
        if not check_result:
            return "check failed (kept)"
        # The pipeline's own list when it logs one (CHECK_PRIORITY_ENABLED reorders the check).
        if k in keys(check_result.get("unchecked_labels", check.get("unchecked"))):
            return f"unchecked (past {check_result.get('max_checked', 36)})"
        if k in keys(check.get("no_text")):
            return "no text (kept)"
        if k in voided:
            return "applicabile (rejection without a quote)"
        verdict = str(verdicts.get(k) or "missing").lower()
        return "incompatibile, kept (limit rule)" if verdict == "incompatibile" else verdict

    def cap_status(k):
        if k in keep:
            return "keep (uncapped)"
        if k in exempt_cap:
            return f"exempt ({exempt_cap[k]})"
        if k in fill_pos:
            pos = fill_pos[k]
            return f"fill #{pos} ({'kept' if k in selected else 'cut'}, cap {fusion.get('cap')})"
        if k in selected and k not in pool:
            return "selected, no row fetched"
        if k in vote_added:
            return "added by the vote"
        if k in pool:
            return "added after the cap"
        return "-"

    def trace(k):
        return {"label": display.get(k) or f"{k[0]} {k[1]}", "channels": [c for c in CHANNELS + ["other_search"]
                                                                          if c in channels.get(k, ())],
                "details": details.get(k, []), "institutions": institution_of.get(k, []),
                "signals": signals_of.get(k, []), "hits": hits.get(k, 0),
                "rank": fused_rank.get(k), "vote_rank": vote_rank.get(k), "cap": cap_status(k),
                "in_pool": k in pool, "check": check_status(k), "dropped": dropped.get(k),
                "in_prompt": None if in_prompt is None else k in in_prompt,
                "in_plan": (k in plan_norme) if plan_on else None, "plan_notes": plan_notes.get(k, []),
                "mentioned": k in mentioned_eval or k in set(mentioned),
                "marked_unverified": k in unverified, "flagged_by_qualification": k in qualification_flagged,
                "rubrica": (pool.get(k) or {}).get("rubrica", "")}

    def stage(k, info):
        t = trace(k)
        if k in mentioned_eval:
            return "MENTIONED", t
        found = k in fused_rank or k in pool or (vote_rank.get(k) or 10 ** 9) <= VOTE_FOUND_TOP
        if not found:
            understood = bool(info.get("resolved_by_institution") or info.get("named_by"))
            return ("NOT_IN_POOL" if understood else "LOST_AT_UNDERSTANDING"), t
        if k in selected and k not in pool:      # chosen, but no row for it in the corpus
            return "NOT_IN_POOL", t
        if k not in pool:
            return "BELOW_CAP", t
        if t["check"] == "rejected":
            return "REJECTED_BY_CHECK", t
        if t["in_prompt"] is False:
            return "NOT_IN_PROMPT", t
        if t["in_plan"] is False:
            return "NOT_IN_PLAN", t
        return "NOT_WRITTEN", t

    frame = rec.get("frame") or {}
    areas = frame.get("areas") or ([rec["area"]] if rec.get("area") else [])
    out = {"id": rec["id"], "kind": rec.get("kind"), "status": rec.get("status"), "seconds": rec.get("seconds"),
           "areas": areas, "area": areas[0] if areas else "?", "posture": frame.get("posture_type") or "",
           "tasks": frame.get("tasks") or [], "central": [c.get("nome") for c in frame.get("central_issue_candidates") or []],
           "keyed": bool(rec.get("expected")), "ok": bool(rec.get("draft")), "slots": [], "unexpected": [],
           "institutions": [i for i in rec.get("institutions") or [] if i.get("phase") == "v2"],
           "vote_rulings": ((rec.get("vote") or {}).get("info") or {}).get("rulings"),
           "pool_size": len(pool), "fused_size": len(fused), "prompt_size": len(in_prompt or ()),
           "plan_size": len(plan_norme) if plan_on else None, "mentioned_size": len(set(mentioned)),
           "capture_errors": rec.get("capture_errors") or [], "check_summary": {}}
    out["pool_traces"] = [trace(k) for k in pool]      # report.json: every pool article, with its verdict
    for t in out["pool_traces"]:
        out["check_summary"][t["check"]] = out["check_summary"].get(t["check"], 0) + 1

    expected_keys = set()
    for item, info in zip(rec.get("expected") or [], rec.get("expected_info") or [{}] * len(rec.get("expected") or [])):
        alt_infos = {(a["article"], a["code"]): a for a in (info or {}).get("alts") or []}
        best = None
        for article in item.get("any_of") or [item["article"]]:
            k = (str(article).lower(), item["code"])
            expected_keys.add(k)
            display.setdefault(k, f"{k[0]} {k[1]}")
            name, t = stage(k, alt_infos.get(k) or {})
            if best is None or STAGES.index(name) > STAGES.index(best[0]):
                best = (name, t, alt_infos.get(k) or {})
        name, t, alt = best
        out["slots"].append({**t, "label": _item_label(item), "pool_label": t["label"],
                             "stage": name, "from_memory": name == "MENTIONED" and not t["in_prompt"],
                             "understanding": alt, "in_expected_rubrica": alt.get("rubrica", "")})
    fits = rec.get("fits") or {}
    for k in dict.fromkeys(mentioned):
        if k in expected_keys:
            continue
        t = trace(k)
        fit = next((v for label, v in fits.items() if ekey(label) == k), None) or {}
        origin = t["channels"] or (["not_retrieved"] if not t["in_pool"] else ["other"])
        out["unexpected"].append({**t, "origin": origin, "fit": fit,
                                  "rubrica": t["rubrica"] or fit.get("rubrica", "")})
    return out


# =================================================================================================
# Pure: the report
# =================================================================================================

def _pct(n, d):
    return f"{100 * n / d:.0f}%" if d else "-"


def aggregate(cases: list) -> dict:
    keyed = [c for c in cases if c["keyed"] and c["ok"]]
    slots = [s for c in keyed for s in c["slots"]]
    stage_counts = Counter(s["stage"] for s in slots)
    wrong = [(c, u) for c in keyed for u in c["unexpected"]]
    any_channel = Counter(ch for _, u in wrong for ch in u["origin"])
    sole = Counter(u["origin"][0] for _, u in wrong if len(u["origin"]) == 1)
    by_check = Counter(u["check"] for _, u in wrong)
    by_plan = Counter(note for _, u in wrong for note in (u["plan_notes"] or ["-"]))
    fit = Counter("incompatible" if u["fit"].get("compatibile") is False else
                  "compatible" if u["fit"].get("compatibile") is True else "not checked" for _, u in wrong)
    per_article = defaultdict(lambda: {"cases": [], "origin": Counter(), "check": Counter(), "plan": Counter(),
                                       "fit": Counter(), "rubrica": "", "rank": [], "label": ""})
    for c, u in wrong:
        a = per_article[u["label"]]
        a["label"] = u["label"]
        a["cases"].append(c["id"])
        a["origin"].update(u["origin"])
        a["check"][u["check"]] += 1
        a["plan"].update(u["plan_notes"] or ["-"])
        a["fit"]["incompatible" if u["fit"].get("compatibile") is False else
                 "compatible" if u["fit"].get("compatibile") is True else "unchecked"] += 1
        a["rubrica"] = a["rubrica"] or u["rubrica"]
        if u["rank"]:
            a["rank"].append(u["rank"])
    unkeyed_labels = {u["label"] for c in cases if not c["keyed"] for u in c["unexpected"]}
    top = sorted(per_article.values(), key=lambda a: (-len(a["cases"]), a["label"]))[:20]

    institutions = [i for c in keyed for i in c["institutions"]]
    method_counts = Counter(i["method"] for i in institutions)
    resolved_articles = [(c["id"], i, a) for c in keyed for i in c["institutions"] for a in i["articles"]]
    unrelated = [(cid, i, a) for cid, i, a in resolved_articles
                 if a["rubrica"] and not a["fits_name"] and not a["fits_keyword"]]
    generic = [(cid, i, a) for cid, i, a in resolved_articles if a["generic"]]
    no_rubrica = [(cid, i, a) for cid, i, a in resolved_articles if not a["rubrica"]]

    areas = defaultdict(lambda: {"cases": 0, "slots": Counter(), "n": 0, "wrong": 0, "wrong_origin": Counter()})
    for c in keyed:
        g = areas[c["area"]]
        g["cases"] += 1
        g["n"] += len(c["slots"])
        g["slots"].update(s["stage"] for s in c["slots"])
        g["wrong"] += len(c["unexpected"])
        g["wrong_origin"].update(ch for u in c["unexpected"] for ch in u["origin"])
    return {"keyed": keyed, "slots": slots, "stage_counts": stage_counts, "wrong": wrong, "any_channel": any_channel,
            "sole": sole, "by_check": by_check, "by_plan": by_plan, "fit": fit, "top": top,
            "unkeyed_labels": unkeyed_labels, "institutions": institutions, "method_counts": method_counts,
            "resolved_articles": resolved_articles, "unrelated": unrelated, "generic": generic,
            "no_rubrica": no_rubrica, "areas": areas}


def _counter_line(counter: Counter, total: int, order=None) -> str:
    keys = order or [k for k, _ in counter.most_common()]
    return ", ".join(f"{k} {counter[k]} ({_pct(counter[k], total)})" for k in keys if counter.get(k)) or "none"


def render(cases: list, header: list) -> str:
    agg = aggregate(cases)
    keyed, slots, wrong = agg["keyed"], agg["slots"], agg["wrong"]
    n_slots = len(slots)
    L = ["# Long-case diagnosis", ""] + header + [""]
    failed = [c for c in cases if not c["ok"]]
    L.append(f"{len(cases)} cases run, {len(keyed)} with an answer key and an answer, {n_slots} expected-article "
             f"slots (an any_of group counts once, by its best alternative)."
             + (f" Failed: {', '.join(c['id'] + ' (' + str(c['status']) + ')' for c in failed)}." if failed else ""))
    errors = [f"{c['id']}: {e}" for c in cases for e in c["capture_errors"]]
    if errors:
        L.append(f"Capture errors (the runs themselves were unaffected): {'; '.join(errors[:10])}")

    # --- A
    L += ["", "## A. Loss funnel of the expected articles", "",
          "Each slot is counted at the first stage it fails.", "",
          "| stage | slots | % of slots | meaning |", "|---|---:|---:|---|"]
    for name in STAGES:
        L.append(f"| {name} | {agg['stage_counts'][name]} | {_pct(agg['stage_counts'][name], n_slots)} | "
                 f"{STAGE_MEANING[name]} |")
    lost = n_slots - agg["stage_counts"]["MENTIONED"]
    memory = sum(s["from_memory"] for s in slots)
    L.append("")
    L.append(f"Of the {lost} not mentioned: " + _counter_line(
        Counter({k: v for k, v in agg["stage_counts"].items() if k != "MENTIONED"}), lost,
        [s for s in STAGES if s != "MENTIONED"]) + ".")
    L.append(f"MENTIONED without being in the draft prompt (from the model's memory): {memory}.")

    # --- understanding
    insts = agg["institutions"]
    L += ["", "## 1. Understanding", ""]
    with_main = [s for s in slots if s["understanding"].get("area_allows") is not None]
    outside = [s for s in with_main if s["understanding"].get("area_allows") is False]
    L.append(f"- Frame areas vs expected codes: {len(outside)} of {len(with_main)} slots with a main code have a code "
             f"the frame's areas do not allow (vote_pool.allowed_codes; laws outside the five codes are not counted)"
             + (": " + ", ".join(f"{c['id']} {s['label']} (areas {'/'.join(c['areas']) or '-'})"
                                 for c in keyed for s in c["slots"] if s["understanding"].get("area_allows") is False)
                if outside else "") + ".")
    via = [s for s in slots if s["understanding"].get("resolved_by_institution")]
    named = [s for s in slots if s["understanding"].get("resolved_by_institution") or s["understanding"].get("named_by")]
    L.append(f"- Expected articles reached via an institution: {len(via)} of {n_slots} ({_pct(len(via), n_slots)}); "
             f"named by the understanding at all (institution resolved to it, or an institution, central issue, question "
             f"title or keyword covering half its rubrica): {len(named)} ({_pct(len(named), n_slots)}).")
    not_in_corpus = [f"{c['id']} {s['label']}" for c in keyed for s in c["slots"] if not s["understanding"].get("in_corpus", True)]
    if not_in_corpus:
        L.append(f"- Expected articles not found in the corpus by number (stage meaningless for them): "
                 f"{', '.join(not_in_corpus)}.")
    L.append(f"- Institutions in the analysis: {len(insts)} in {len(keyed)} cases; resolution method: "
             + _counter_line(agg["method_counts"], len(insts),
                             ["heading", "title", "heading_index", "fulltext", "model_number", "none"]) + ".")
    ra = agg["resolved_articles"]
    L.append(f"- Of the {len(ra)} articles they resolved to: rubrica sharing no word with the institution or its "
             f"keywords {len(agg['unrelated'])} ({_pct(len(agg['unrelated']), len(ra))}), generic rubrica "
             f"{len(agg['generic'])}, no rubrica {len(agg['no_rubrica'])}.")
    if agg["unrelated"] or agg["generic"]:
        L += ["", "| case | institution | method | article | rubrica |", "|---|---|---|---|---|"]
        for cid, i, a in (agg["unrelated"] + [g for g in agg["generic"] if g not in agg["unrelated"]])[:40]:
            L.append(f"| {cid} | {i['nome']} ({i['codice'] or '-'}) | {i['method']} | {a['label']} | "
                     f"{a['rubrica'] or '-'} ({a['field'] or '-'}) |")
    lost_u = [(c, s) for c in keyed for s in c["slots"] if s["stage"] == "LOST_AT_UNDERSTANDING"]
    if lost_u:
        L += ["", "Lost at understanding (expected article, its rubrica, the case's central issues):", ""]
        L += [f"- {c['id']} {s['label']} «{s['understanding'].get('rubrica') or '-'}» -- central: "
              f"{'; '.join(c['central']) or '-'}" for c, s in lost_u]

    # --- finding
    L += ["", "## 2. Finding", ""]
    found = [s for s in slots if s["stage"] not in ("LOST_AT_UNDERSTANDING", "NOT_IN_POOL")]
    L.append(f"- Expected found by some channel: {len(found)} of {n_slots} ({_pct(len(found), n_slots)}); in the pool: "
             f"{sum(s['in_pool'] for s in slots)}.")
    ch = Counter(c for s in found for c in s["channels"])
    sole = Counter(s["channels"][0] for s in found if len(s["channels"]) == 1)
    L.append(f"- Expected found, by channel (any; one article can have several): {_counter_line(ch, len(found), CHANNELS + ['other_search'])}.")
    L.append(f"- Found by one channel only: {_counter_line(sole, len(found), CHANNELS + ['other_search'])}.")
    rulings = [c["vote_rulings"] for c in keyed if c["vote_rulings"] is not None]
    if rulings:
        L.append(f"- Vote: rulings per case {min(rulings)}-{max(rulings)} (median {sorted(rulings)[len(rulings) // 2]}).")
    L.append(f"- Pool sizes (pre-check): median {sorted(c['pool_size'] for c in keyed)[len(keyed) // 2] if keyed else '-'}; "
             f"fused ranking median {sorted(c['fused_size'] for c in keyed)[len(keyed) // 2] if keyed else '-'}.")

    # --- ranking
    L += ["", "## 3. Ranking and selection", ""]
    below = [(c, s) for c in keyed for s in c["slots"] if s["stage"] == "BELOW_CAP"]
    if below:
        L.append("Expected found but cut before the pool: " + "; ".join(
            f"{c['id']} {s['label']} (fused rank {s['rank'] or '-'}, {s['cap']}, vote rank {s['vote_rank'] or '-'}, "
            f"{'+'.join(s['channels']) or '-'})" for c, s in below) + ".")
    rejected = [(c, s) for c in keyed for s in c["slots"] if s["stage"] == "REJECTED_BY_CHECK"]
    if rejected:
        L.append("")
        L.append("Expected rejected by the applicability check:")
        L += [f"- {c['id']} {s['label']}: graded 0, «{(s['dropped'] or {}).get('reason', '')}»"
              if (s["dropped"] or {}).get("grade") == 0 else
              f"- {c['id']} {s['label']}: requisito «{(s['dropped'] or {}).get('requisito', '')}», fatto "
              f"«{(s['dropped'] or {}).get('fatto', '')}»" for c, s in rejected]
    exp_checks = Counter(s["check"] for s in slots if s["in_pool"])
    L.append("")
    L.append(f"- Check status of the expected articles in the pool: {_counter_line(exp_checks, sum(exp_checks.values()))}.")
    all_checks = Counter()
    for c in keyed:
        all_checks.update(c["check_summary"])
    L.append(f"- Check status of every pool article (expected included, {sum(all_checks.values())}): "
             f"{_counter_line(all_checks, sum(all_checks.values()))}.")
    not_planned = [(c, s) for c in keyed for s in c["slots"] if s["stage"] in ("NOT_IN_PROMPT", "NOT_IN_PLAN", "NOT_WRITTEN")]
    if not_planned:
        L.append("- Expected past the check but not written: " + "; ".join(
            f"{c['id']} {s['label']} ({s['stage']}{', plan: ' + '/'.join(s['plan_notes']) if s['plan_notes'] else ''})"
            for c, s in not_planned) + ".")

    # --- B
    nw = len(wrong)
    L += ["", "## B. Articles cited without being expected: where they come from", "",
          f"{nw} citations in {len(keyed)} answers (an article per answer once; read by the pipeline's own "
          "_cited_articles). Not all are wrong: the answer key lists what must be there, not all that may.", ""]
    L.append(f"- By channel (any): {_counter_line(agg['any_channel'], nw, CHANNELS + ['other_search', 'not_retrieved', 'other'])}.")
    L.append(f"- By one channel only: {_counter_line(agg['sole'], nw, CHANNELS + ['other_search', 'not_retrieved', 'other'])}.")
    L.append(f"- By applicability status (exemption or verdict): {_counter_line(agg['by_check'], nw)}.")
    L.append(f"- By plan protection: {_counter_line(agg['by_plan'], nw)}.")
    L.append(f"- Stored rubrica vs the line citing it (qualification check's comparison): {_counter_line(agg['fit'], nw)}.")

    # --- C
    L += ["", "## C. The 20 most frequent articles cited without being expected", "",
          "| article | cases | origin (channel: times) | check | plan | rubrica fits | fused rank | rubrica |",
          "|---|---:|---|---|---|---|---|---|"]
    for a in agg["top"]:
        L.append(f"| {a['label']}{' *' if a['label'] in agg['unkeyed_labels'] else ''} | {len(a['cases'])} "
                 f"({', '.join(a['cases'][:6])}{'...' if len(a['cases']) > 6 else ''}) | "
                 f"{', '.join(f'{k}: {v}' for k, v in a['origin'].most_common())} | "
                 f"{', '.join(f'{k}: {v}' for k, v in a['check'].most_common())} | "
                 f"{', '.join(f'{k}: {v}' for k, v in a['plan'].most_common())} | "
                 f"{', '.join(f'{k}: {v}' for k, v in a['fit'].most_common())} | "
                 f"{'/'.join(map(str, a['rank'][:5])) or '-'} | {a['rubrica'] or '-'} |")
    L.append("")
    L.append("* also cited in a case without an answer key (--file).")

    # --- D
    L += ["", "## D. Per area (the frame's first area)", "",
          "| area | cases | slots | mentioned | " + " | ".join(s for s in STAGES if s != "MENTIONED")
          + " | not expected per answer | their top channels |",
          "|---|---:|---:|---:|" + "---:|" * (len(STAGES) - 1) + "---:|---|"]
    for area, g in sorted(agg["areas"].items(), key=lambda x: -x[1]["n"]):
        L.append(f"| {area} | {g['cases']} | {g['n']} | {g['slots']['MENTIONED']} ({_pct(g['slots']['MENTIONED'], g['n'])}) | "
                 + " | ".join(str(g["slots"][s]) for s in STAGES if s != "MENTIONED")
                 + f" | {g['wrong'] / g['cases']:.1f} | "
                 + ", ".join(f"{k} {v}" for k, v in g["wrong_origin"].most_common(3)) + " |")

    # --- per case
    L += ["", "## Per case", "",
          "| case | areas / posture | expected: stage | pool / prompt / plan / cited | not expected (origin) | time |",
          "|---|---|---|---|---|---:|"]
    for c in cases:
        exp = "; ".join(f"{s['label'].replace('art. ', '')}: {s['stage']}" for s in c["slots"]) or "(no key)"
        unexp = ", ".join(f"{u['label']} ({'+'.join(u['origin'])})" for u in c["unexpected"]) or "-"
        L.append(f"| {c['id']} | {'/'.join(c['areas']) or '-'} / {c['posture'] or '-'} | {exp} | {c['pool_size']} / "
                 f"{c['prompt_size']} / {c['plan_size'] if c['plan_size'] is not None else '-'} / {c['mentioned_size']} | "
                 f"{unexp} | {c['seconds'] or 0:.0f}s |")

    # --- unkeyed
    for c in cases:
        if c["keyed"]:
            continue
        L += ["", f"## {c['id']} (no answer key)", "",
              f"Areas {'/'.join(c['areas']) or '-'}, posture {c['posture'] or '-'}; tasks: {'; '.join(c['tasks']) or '-'}; "
              f"central issues: {'; '.join(c['central']) or '-'}.", "",
              "Institutions: " + "; ".join(
                  f"{i['nome']} -> {', '.join(a['label'] + ' «' + (a['rubrica'] or '-') + '»' for a in i['articles']) or 'none'}"
                  f" [{i['method']}]" for i in c["institutions"]), "",
              "| cited article | origin | fused rank | cap | check | plan | rubrica fits the line | rubrica |",
              "|---|---|---|---|---|---|---|---|"]
        for u in c["unexpected"]:
            fit = u["fit"]
            verdict = ("no" if fit.get("compatibile") is False else "yes" if fit.get("compatibile") is True else "-")
            L.append(f"| {u['label']} | {'+'.join(u['origin'])} | {u['rank'] or '-'} | {u['cap']} | {u['check']} | "
                     f"{'/'.join(u['plan_notes']) or ('yes' if u['in_plan'] else '-')} | {verdict}"
                     f"{': ' + fit['perche'] if fit.get('compatibile') is False and fit.get('perche') else ''} | "
                     f"{u['rubrica'] or '-'} |")
    return "\n".join(L) + "\n"


# =================================================================================================
# Pure: --audit-grades, why graded articles came back without a grade
# =================================================================================================

def _dump_parts(text: str) -> tuple:
    """({header field: value}, {SYSTEM/USER/REPLY...: content}) of an llm/NN_<stage>.txt file."""
    head, _, rest = text.partition("\n\n===== ")
    fields = dict(line.split(": ", 1) for line in head.splitlines() if ": " in line)
    parts = {}
    for chunk in re.split(r"(?:^|\n\n)===== ", "===== " + rest)[1:]:
        name, _, body = chunk.partition(" =====\n")
        parts[name.strip()] = body
    return fields, parts


def _loose(label: str) -> str:
    """A label with only its letters and digits ("Art. 640-ter c.p." -> "640terc p" squeezed)."""
    return re.sub(r"[^a-z0-9]", "", re.sub(r"^\s*art(?:icolo)?\.?\s*", "", str(label).lower()))


def _reply_objects(reply: str) -> tuple:
    """(objects read, kind) of a check reply: kind "list", "object, not a list", "cut", "unparseable",
    "empty"; a cut or wrapped reply's objects are read one by one, as the pipeline's _json_objects does."""
    text = re.sub(r"```(?:json)?\s*", "", reply or "").strip().rstrip("`").strip()
    if not text:
        return [], "empty"
    try:
        value = json.loads(text)
        if isinstance(value, list):
            return [v for v in value if isinstance(v, dict)], "list"
    except ValueError:
        value = None
    decoder, found, i = json.JSONDecoder(), [], text.find("{")
    while i >= 0:
        try:
            item, end = decoder.raw_decode(text, i)
        except ValueError:
            i = text.find("{", i + 1)
            continue
        if isinstance(item, dict):
            lists = [v for v in item.values() if isinstance(v, list)]
            found += [x for v in lists for x in v if isinstance(x, dict)] if lists and "a" not in item else [item]
        i = text.find("{", end)
    if isinstance(value, dict):
        return found, "object, not a list"
    if not text.endswith("]") and found:
        return found, "cut"
    return found, "cut" if text.startswith("[") else "unparseable"


def audit_grades(run_dir: str, ev) -> str:
    """For every graded applicability call of a run (its llm/ dumps): how the reply came back, and for
    every article sent, whether it got a grade and if not why -- the call cut short (the reply ends
    before the array does), the call failed, the id echoed differently from the one sent, or the model
    skipped it. Then the same for the expected articles the report has as "missing" at the check."""
    calls, outcome_of, mismatches, extra = Counter(), {}, [], Counter()
    sizes = []
    cases_dir = os.path.join(run_dir, "cases")
    rank = {"graded": 0, "entry without a valid grade": 1, "id mismatch": 2, "skipped by the model": 3,
            "lost: reply cut": 4, "lost: call failed": 5}
    for cid in sorted(os.listdir(cases_dir)):
        llm = os.path.join(cases_dir, cid, "llm")
        if not os.path.isdir(llm):
            continue
        for name in sorted(os.listdir(llm)):
            if "applicability" not in name:
                continue
            with open(os.path.join(llm, name), encoding="utf-8") as fh:
                fields, parts = _dump_parts(fh.read())
            if "voto di pertinenza" not in parts.get("SYSTEM", ""):
                continue
            sent = re.findall(r"^\[([^\]]+)\](?: (.+))?\nRubrica:", parts.get("USER", ""), re.MULTILINE)
            by_id = {i.strip(): (label or i).strip() for i, label in sent}
            by_loose = {_loose(label): label for label in by_id.values()}
            by_key = {" ".join(re.sub(r"^\s*art(?:icolo)?\.?\s*", "", label.lower()).split()): label
                      for label in by_id.values()}
            reply = parts.get("REPLY", "")
            if fields.get("status", "ok") != "ok":
                objects, kind = [], "error"
            else:
                objects, kind = _reply_objects(reply)
            calls[kind] += 1
            if kind == "cut":
                sizes.append((cid, name, len(sent), fields.get("max_tokens"), len(reply)))
            got = {}
            for item in objects:
                ref = str(item.get("a") or item.get("article") or "").strip()
                grade_ok = re.match(r"\s*[0-3]\b", str(item.get("g", item.get("grade", ""))))
                exact = by_id.get(ref) or by_key.get(" ".join(re.sub(r"^\s*art(?:icolo)?\.?\s*", "", ref.lower()).split()))
                if exact:
                    got.setdefault(exact, "graded" if grade_ok else "entry without a valid grade")
                elif by_loose.get(_loose(ref.strip("[] "))):
                    label = by_loose[_loose(ref.strip("[] "))]
                    got.setdefault(label, "id mismatch")
                    mismatches.append((cid, label, ref))
                else:
                    extra[(cid, ref)] += 1
            for label in by_id.values():
                fate = got.get(label) or ("lost: reply cut" if kind == "cut" else
                                          "lost: call failed" if kind in ("error", "unparseable", "empty",
                                                                         "object, not a list")
                                          else "skipped by the model")
                key = (cid, _loose(label))
                if key not in outcome_of or rank[fate] < rank[outcome_of[key][1]]:
                    outcome_of[key] = (label, fate)
    fates = Counter(fate for _, fate in outcome_of.values())
    L = [f"# Graded check audit: {os.path.basename(os.path.normpath(run_dir))}", "",
         f"{sum(calls.values())} graded calls; {len(outcome_of)} (case, article) pairs sent.", "",
         "## Calls by how the reply came back", "", "| reply | calls |", "|---|---:|"]
    L += [f"| {k} | {v} |" for k, v in calls.most_common()]
    L += ["", "A cut reply is not a JSON list, so before this fix the pipeline read none of its grades.", "",
          "## Every article sent, by its best outcome in the case", "", "| outcome | articles |", "|---|---:|"]
    L += [f"| {k} | {v} |" for k, v in sorted(fates.items(), key=lambda kv: rank[kv[0]])]
    _, cases = _load_cases(run_dir, ev)
    lost = [(c["id"], slot["label"], slot.get("pool_label") or slot["label"]) for c in cases for slot in c["slots"]
            if slot["stage"] == "NOT_IN_PROMPT" and slot.get("check") == "missing"]
    if lost:
        why = Counter()
        rows = []
        for cid, label, pool_label in lost:
            hit = next((f for (case, loose), (_, f) in outcome_of.items()
                        if case == cid and loose == _loose(pool_label)), None)
            fate = hit or "not found in the dumps"
            why[fate] += 1
            rows.append(f"| {cid} | {label} | {fate} |")
        L += ["", f"## The {len(lost)} expected articles NOT_IN_PROMPT with check \"missing\"", "",
              "| cause | articles |", "|---|---:|"] + [f"| {k} | {v} |" for k, v in why.most_common()]
        L += ["", "| case | expected | cause |", "|---|---|---|"] + rows
    if sizes:
        L += ["", "## Cut replies", "", "| case | call | articles sent | max_tokens | reply chars |", "|---|---|---:|---:|---:|"]
        L += [f"| {cid} | {name} | {n} | {mt} | {chars} |" for cid, name, n, mt, chars in sizes]
    if mismatches:
        L += ["", "## Ids echoed differently (first 20)", "", "| case | sent | returned |", "|---|---|---|"]
        L += [f"| {cid} | {label} | {ref} |" for cid, label, ref in mismatches[:20]]
    if extra:
        L += ["", f"## Ids in replies matching nothing sent: {sum(extra.values())} (first 20)", ""]
        L += [f"- {cid}: «{ref}» x{n}" for (cid, ref), n in extra.most_common(20)]
    return "\n".join(L) + "\n"


def _load_cases(out_dir: str, ev) -> tuple:
    """(records, analysed cases) of a run folder, in the cases' usual order."""
    records = []
    cases_dir = os.path.join(out_dir, "cases")
    for name in sorted(os.listdir(cases_dir)):
        if name.endswith(".json") and not name.endswith(".input.json"):
            with open(os.path.join(cases_dir, name), encoding="utf-8") as fh:
                records.append(json.load(fh))
    order = {cid: i for i, cid in enumerate(TUNING_IDS + HELDOUT_IDS)}
    records.sort(key=lambda r: (r.get("kind") == "file", order.get(r["id"], 999), r["id"]))
    return records, [analyse_case(r, ev) for r in records]


def write_report(out_dir: str, ev) -> str:
    records, cases = _load_cases(out_dir, ev)
    first = records[0] if records else {}
    flags = first.get("flags") or {}
    header = [f"Run: {out_dir}", f"Code: {first.get('code', '-')}",
              "Per case: cases/<id>.json (capture), cases/<id>/material.txt (what the writer saw), "
              "cases/<id>/llm/NN_<stage>.txt (every chat call, with --dump-llm).",
              "Flags: " + ", ".join(f"{k}={v}" for k, v in flags.items() if v is not None)]
    report = render(cases, header)
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as fh:
        fh.write(report)
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as fh:
        json.dump(cases, fh, ensure_ascii=False, indent=1, default=str)
    return os.path.join(out_dir, "report.md")


def _run_dir(name: str) -> str:
    """A run folder by path, or by its name under ./diagnose_long_runs ("20261025-0940")."""
    for path in (name, os.path.join("diagnose_long_runs", name)):
        if os.path.isdir(os.path.join(path, "cases")):
            return os.path.abspath(path)
    sys.exit(f"--compare: no run folder {name!r} (looked for {name}/cases and diagnose_long_runs/{name}/cases)")


def _side_by_side(title: str, counters: list, totals: list, names: list, order=None) -> list:
    """One table: a row per key, a column per run (n and % of that run's total), then the last run's
    change against each earlier one."""
    keys = order or sorted(set().union(*counters), key=lambda k: (-sum(c.get(k, 0) for c in counters), str(k)))
    head = f"| {title} | " + " | ".join(names) + " | " + " | ".join(f"change vs {n}" for n in names[:-1]) + " |"
    rows = [head, "|---|" + "---:|" * (2 * len(names) - 1)]
    for k in keys:
        values = [c.get(k, 0) for c in counters]
        if any(values):
            rows.append(f"| {k} | " + " | ".join(f"{v} ({_pct(v, t)})" for v, t in zip(values, totals)) + " | "
                        + " | ".join(f"{values[-1] - v:+d}" for v in values[:-1]) + " |")
    return rows


def compare(run_dirs: list, ev) -> str:
    """Tables A (loss funnel) and B (where the articles cited without being expected come from) of
    several runs side by side, the last one the newest, over the cases every run has with an answer key
    and an answer."""
    loaded = [_load_cases(d, ev) for d in run_dirs]
    names = [os.path.basename(os.path.normpath(d)) for d in run_dirs]
    by = [{c["id"]: c for c in cases if c["keyed"] and c["ok"]} for _, cases in loaded]
    common = [cid for cid in by[-1] if all(cid in b for b in by)]
    left_out = sorted(set().union(*by) - set(common))
    aggs = [aggregate([b[c] for c in common]) for b in by]
    flags = [((records or [{}])[0].get("flags") or {}) for records, _ in loaded]
    changed = [k for k in sorted(set().union(*flags)) if len({f.get(k) for f in flags}) > 1]
    L = [f"# diagnose_long: {' vs '.join(names)} (the last is the newest)", "",
         f"{len(common)} cases in every run, with an answer key and an answer"
         + (f"; left out (missing or failed in a run): {', '.join(left_out)}" if left_out else "") + ".",
         "Flags that differ: " + ("; ".join(f"{k} " + " / ".join(str(f.get(k)) for f in flags) for k in changed)
                                  or "none") + " (in run order).", "",
         "## Headline", "", "| | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
    rows = [("MENTIONED", [a["stage_counts"]["MENTIONED"] for a in aggs], lambda a: f" / {len(a['slots'])}"),
            ("not expected citations", [len(a["wrong"]) for a in aggs], lambda a: ""),
            ("not expected per answer", [round(len(a["wrong"]) / max(1, len(common)), 1) for a in aggs], lambda a: ""),
            ("rubrica-incompatible", [a["fit"]["incompatible"] for a in aggs], lambda a: "")]
    for title, values, suffix in rows:
        L.append(f"| {title} | " + " | ".join(f"{v}{suffix(a)}" for v, a in zip(values, aggs)) + " |")
    L += ["", "## A. Loss funnel of the expected articles", ""]
    L += _side_by_side("stage", [a["stage_counts"] for a in aggs], [len(a["slots"]) for a in aggs], names, STAGES)
    L += ["", "MENTIONED without being in the prompt: "
          + ", ".join(f"{n} {sum(s['from_memory'] for s in a['slots'])}" for n, a in zip(names, aggs)) + ".",
          "", "## B. Articles cited without being expected", ""]
    channel_order = CHANNELS + ["other_search", "not_retrieved", "other"]
    for title, key, order in (("by channel (any)", "any_channel", channel_order),
                              ("by one channel only", "sole", channel_order),
                              ("by applicability status", "by_check", None),
                              ("by plan protection", "by_plan", None),
                              ("rubrica vs citing line", "fit", None)):
        L += _side_by_side(title, [a[key] for a in aggs], [len(a["wrong"]) for a in aggs], names, order) + [""]
    L += ["## Per case", "", "| case | " + " | ".join(f"MENTIONED {n}" for n in names) + " | of | "
          + " | ".join(f"not expected {n}" for n in names) + " |", "|---|" + "---:|" * (2 * len(names) + 1)]
    for cid in common:
        mentioned = [sum(s["stage"] == "MENTIONED" for s in b[cid]["slots"]) for b in by]
        L.append(f"| {cid} | " + " | ".join(map(str, mentioned)) + f" | {len(by[-1][cid]['slots'])} | "
                 + " | ".join(str(len(b[cid]["unexpected"])) for b in by) + " |")
    return "\n".join(L) + "\n"


# =================================================================================================
# Parent
# =================================================================================================

def _vote_idf_file(out_dir: str) -> None:
    """VOTE_POOL_ENABLED without VOTE_POOL_IDF_FILE: every process counts the CITES edges for itself
    and skips the vote until done. The same counts (vote_pool._count's query) once, to a file."""
    from src.rag import feature_flags
    if not feature_flags.enabled("VOTE_POOL_ENABLED") or os.path.isfile(os.getenv("VOTE_POOL_IDF_FILE", "").strip()):
        return
    _read_only_neo4j()
    from src.rag.main import driver, NEO4J_DATABASE
    t0 = time.time()
    with driver.session(database=NEO4J_DATABASE) as session:
        n = session.run("MATCH (d:Document {document_type: 'interpretation'}) RETURN count(d) AS n").single()["n"]
        rows = session.run("MATCH (d:Document)-[:CONTAINS]->(:Section)-[r:CITES]->() "
                           "WITH r.code AS code, r.article AS article, count(DISTINCT d) AS df "
                           "RETURN code, article, df").data()
    path = os.path.join(out_dir, "vote_idf.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"n": n, "df": [[r["code"], r["article"], r["df"]] for r in rows]}, fh)
    os.environ["VOTE_POOL_IDF_FILE"] = path
    print(f"vote IDF: {len(rows)} articles over {n} rulings counted once in {time.time() - t0:.0f}s -> {path}",
          file=sys.stderr)


def _print_compare(run_dirs: list, ev) -> None:
    text = compare(run_dirs, ev)
    path = os.path.join(run_dirs[-1], "compare_" + "_".join(os.path.basename(os.path.normpath(d)) for d in run_dirs[:-1])
                        + ".md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    print("\n" + text + f"\nSaved to {path}")


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "document"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--only", help="comma-separated case ids (default: every case with expected articles "
                                       "among the 12 tuning ones, h1-h11, n1-n9)")
    parser.add_argument("--no-cases", action="store_true", help="only the --file documents")
    parser.add_argument("--file", action="append", default=[], help="a local .pdf/.docx/.txt, read as an upload is")
    parser.add_argument("--request", action="append", default=[], help="the user's message for each --file, in order")
    parser.add_argument("--lang", default="it")
    parser.add_argument("--env", default=os.path.join(os.getcwd(), ".env"))
    parser.add_argument("--out", help="output folder (default ./diagnose_long_runs/<date-time>)")
    parser.add_argument("--parallel", type=int, default=MAX_PARALLEL, help=f"cases at once (at most {MAX_PARALLEL})")
    parser.add_argument("--timeout", type=int, default=CASE_TIMEOUT, help="seconds per case")
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                        help="a switch for this run only, over the .env")
    parser.add_argument("--report-only", metavar="DIR", help="rebuild DIR/report.md from DIR/cases, no run")
    parser.add_argument("--compare", nargs="+", metavar="RUN",
                        help="earlier runs (a folder, or its name under ./diagnose_long_runs) whose tables A and B "
                             "are printed side by side with this run's, or with --report-only's DIR")
    parser.add_argument("--compare-only", action="store_true",
                        help="no run: compare the --compare runs among themselves, the last as the newest")
    parser.add_argument("--dump-llm", action=argparse.BooleanOptionalAction, default=True,
                        help="save every chat call (messages, raw reply, settings, duration) to "
                             "cases/<id>/llm/NN_<stage>.txt (default on; --no-dump-llm to skip)")
    parser.add_argument("--audit-grades", metavar="RUN",
                        help="no run: why articles sent to the graded check got no grade, from RUN's llm/ dumps "
                             "(written to RUN/grade_audit.md)")
    parser.add_argument("--child", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.child:
        run_child(args.child, args.env)
        return
    sys.path.insert(0, os.getcwd())
    ev = _eval_module()
    if args.audit_grades:
        run_dir = _run_dir(args.audit_grades)
        text = audit_grades(run_dir, ev)
        with open(os.path.join(run_dir, "grade_audit.md"), "w", encoding="utf-8") as fh:
            fh.write(text)
        print(text + f"\nWritten to {os.path.join(run_dir, 'grade_audit.md')}")
        return
    if args.compare_only:
        if not args.compare or len(args.compare) < 2:
            parser.error("--compare-only needs two runs or more in --compare")
        _print_compare([_run_dir(r) for r in args.compare], ev)
        return
    if args.report_only:
        print(f"Report: {write_report(args.report_only, ev)}")
        if args.compare:
            _print_compare([_run_dir(r) for r in args.compare] + [os.path.abspath(args.report_only)], ev)
        return

    from dotenv import load_dotenv
    load_dotenv(args.env)
    settings = {}
    for item in args.set:
        name, _, value = item.partition("=")
        settings[name.strip()] = value.strip()
        os.environ[name.strip()] = value.strip()

    all_cases = json.load(open(os.path.join(os.getcwd(), "tests", "eval_narrative_cases.json"), encoding="utf-8"))["cases"]
    by_id = {c["id"]: c for c in all_cases}
    if args.no_cases:
        chosen = []
    elif args.only:
        missing = [i for i in args.only.split(",") if i.strip() not in by_id]
        if missing:
            parser.error(f"unknown case ids: {', '.join(missing)}")
        chosen = [by_id[i.strip()] for i in args.only.split(",")]
    else:
        chosen = [by_id[i] for i in TUNING_IDS + HELDOUT_IDS if i in by_id and by_id[i].get("expected")]
    if len(args.request) > len(args.file):
        parser.error("more --request than --file")
    for path in args.file:
        if not os.path.isfile(path):
            parser.error(f"--file {path}: not found")

    out_dir = os.path.abspath(args.out or os.path.join("diagnose_long_runs", datetime.now().strftime("%Y%m%d-%H%M")))
    for sub in ("cases", "logs"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    try:
        _vote_idf_file(out_dir)
    except Exception as exc:
        print(f"vote IDF: counting once failed ({type(exc).__name__}: {exc}); each case counts for itself",
              file=sys.stderr)

    specs = []
    for case in chosen:
        specs.append({"id": case["id"], "question": case["question"], "expected": case.get("expected") or []})
    for i, path in enumerate(args.file):
        request = args.request[i] if i < len(args.request) else DEFAULT_REQUEST
        if i >= len(args.request):
            print(f"--file {path}: no --request, using the generic one", file=sys.stderr)
        specs.append({"id": _slug(os.path.splitext(os.path.basename(path))[0]), "file": os.path.abspath(path),
                      "request": request, "expected": None})
    for spec in specs:
        spec.update(lang=args.lang, settings=settings, dump_llm=args.dump_llm,
                    out=os.path.join(out_dir, "cases", f"{spec['id']}.json"))
        with open(os.path.join(out_dir, "cases", f"{spec['id']}.input.json"), "w", encoding="utf-8") as fh:
            json.dump(spec, fh, ensure_ascii=False)

    parallel = max(1, min(args.parallel, MAX_PARALLEL))
    print(f"{len(specs)} cases, {parallel} at a time, into {out_dir}", flush=True)
    t0 = time.time()

    def run(spec):
        started = time.time()
        log_path = os.path.join(out_dir, "logs", f"{spec['id']}.log")
        with open(log_path, "w", encoding="utf-8") as log:
            try:
                code = subprocess.run([sys.executable, os.path.abspath(__file__), "--child",
                                       os.path.join(out_dir, "cases", f"{spec['id']}.input.json"), "--env", args.env],
                                      cwd=os.getcwd(), stdout=log, stderr=subprocess.STDOUT,
                                      timeout=args.timeout).returncode
            except subprocess.TimeoutExpired:
                code = "timeout"
        done = os.path.isfile(spec["out"])
        print(f"  {spec['id']:<20s} {'ok' if done else 'FAILED (' + str(code) + ', see ' + log_path + ')'} "
              f"{time.time() - started:.0f}s  [{time.time() - t0:.0f}s elapsed]", flush=True)
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        list(pool.map(run, specs))

    print(f"\nAll cases in {time.time() - t0:.0f}s.")
    print(f"Report: {write_report(out_dir, ev)}")
    if args.compare:
        _print_compare([_run_dir(r) for r in args.compare] + [out_dir], ev)
    print("Nothing was written to the database.")


if __name__ == "__main__":
    main()
