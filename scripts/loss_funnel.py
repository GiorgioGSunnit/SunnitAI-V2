"""Read-only: for every expected article of every narrative case, the
furthest stage it reaches in the live case-analysis pipeline:

  NOT_IN_POOL          none of CURRENT/BASELINE/DISTILL found it, and it
                        isn't named by the user or agreed on by rulings
                        either (_select_articles_fusion's "keep").
  IN_POOL_CUT_BY_CAP    found, but ranked past _FUSION_POOL_CAP (reported
                        with its rank in the fused, keep-excluded order).
  DROPPED_BY_CHECK      reached "selected", but the applicability check
                        (APPLICABILITY_CHECK_ENABLED) ruled it incompatible
                        (reported with its reason).
  SELECTED_NOT_CITED    reached the final article_rows the draft prompt
                        saw, but never became a citation (most often: the
                        fetched row had no usable text -- _extract_
                        citations' own length gate).
  CITED                 appears in the pipeline's own "citations", the
                        same thing tests/eval_narrative.py's "cite X/47"
                        figure counts.

run_case_analysis_pipeline is called ONCE per case -- the same call
eval_narrative.py's number reflects -- with _select_articles_fusion
wrapped to capture its ranked pool and vlog wrapped to capture the
applicability check's own "dropped" list, so every stage comes from that
one execution, not a second, possibly different one (the LLM calls inside
it are not deterministic). Cases run sequentially, not in parallel, for
exactly this reason: the capture is done by temporarily replacing module-
level functions, which is not safe to do from more than one case at a time.

By default this uses whatever CASE_ANALYSIS_ENABLED / CASE_POOL_FUSION_ENABLED
/ APPLICABILITY_CHECK_ENABLED are actually set to in this environment (the
header prints them) -- pass --fusion to force the fusion path regardless
(same override check_applicability.py offers). If the live/forced path
never reaches _select_articles_fusion for a case, that case is reported as
an error, not silently scored as all-NOT_IN_POOL.

Run from the app folder so it uses the app's code, .env and venv:
    cd /opt/chatbot && venv/bin/python scripts/loss_funnel.py
    ... --only sempronio,l1          # just these cases
    ... --fusion                     # force the fusion path regardless of the flags
    ... --out /root/loss_funnel      # where the table lands (default ./loss_funnel_runs)

No DB writes; the only side effect is the LLM calls the live pipeline
itself makes (one draft generation per case, same as a real request).
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (ROOT, os.path.join(ROOT, "tests"), os.getcwd()):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(os.getcwd(), ".env"))

import neo4j  # noqa: E402

_open_session = neo4j.Driver.session


def _read_only_session(self, **config):
    config["default_access_mode"] = neo4j.READ_ACCESS
    return _open_session(self, **config)


neo4j.Driver.session = _read_only_session   # every session in this process is read-only
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

import eval_narrative as ev  # noqa: E402  (tests/eval_narrative.py: art_of, code_of_document, cited, _label)

import src.rag.defensive_generation as dg  # noqa: E402
from src.rag import feature_flags  # noqa: E402

DEFAULT_CASES = Path(__file__).resolve().parent.parent / "tests" / "eval_narrative_cases.json"
# The 12 non-"extra" narrative cases -- same set scripts/exp_distill.py uses,
# the ones Flavio's own 30.3/47 figure is counted over.
_NARRATIVE_CASE_IDS = ["sempronio", "sempronio_no_hint", "p1", "p2", "p3", "c1", "c2", "c3",
                       "l1", "l2", "l3", "l4"]

_STAGE_ORDER = ["NOT_IN_POOL", "IN_POOL_CUT_BY_CAP", "DROPPED_BY_CHECK", "SELECTED_NOT_CITED", "CITED"]
_STAGE_RANK = {s: i for i, s in enumerate(_STAGE_ORDER)}

# Same mapping scripts/exp_distill.py uses to bridge dg's own (number,
# abbr-or-document-name) tuples and the eval's (code, article) ones -- kept
# here too rather than imported, since exp_distill.py is a script, not a
# library this one should depend on importing.
_ABBR_TO_EVAL_CODE = {"c.p.": "cp", "c.c.": "cc", "c.p.c.": "cpc", "c.p.p.": "cpp"}


def _code_key(abbr_or_name):
    return _ABBR_TO_EVAL_CODE.get(abbr_or_name) or ev.code_of_document(abbr_or_name)


def _to_eval_ref(dg_ref):
    """(number, abbr-or-document-name) -- as _select_articles_fusion's own
    "keep"/"candidates" tag rows -- -> (eval code, article), or None when
    the document isn't one of the eval's known codes/laws."""
    number, abbr_or_name = dg_ref
    code = _code_key(abbr_or_name)
    return (code, number) if code else None


def _ordered_eval_refs(dg_refs):
    """dg_refs, mapped to eval refs, first occurrence kept, unmappable ones
    (a document the eval's CODES dict doesn't recognise) dropped."""
    seen, out = set(), []
    for ref in dg_refs:
        mapped = _to_eval_ref(ref)
        if mapped and mapped not in seen:
            seen.add(mapped)
            out.append(mapped)
    return out


def _parse_dropped_label(label):
    """"<number> <abbr-or-document-name>" -> (number, abbr_or_name) -- the
    inverse of _select_articles_fusion's own f"{n} {c}" label (see
    _applicability_candidates/_filter_applicable_articles)."""
    parts = (label or "").split(" ", 1)
    return (parts[0], parts[1]) if len(parts) == 2 else (label, "")


def _classify_one(ref, keep_eval, fused_rank, dropped_eval, cited_eval):
    """The furthest stage one (eval code, article) ref reaches -> (stage, detail)."""
    if ref in keep_eval:
        detail = "keep: named by the user or agreed on by rulings (uncapped)"
    elif ref in fused_rank:
        rank = fused_rank[ref]
        if rank > dg._FUSION_POOL_CAP:
            return "IN_POOL_CUT_BY_CAP", f"rank {rank} of {len(fused_rank)}"
        detail = f"rank {rank}"
    else:
        return "NOT_IN_POOL", ""
    if ref in dropped_eval:
        return "DROPPED_BY_CHECK", dropped_eval[ref]
    if ref in cited_eval:
        return "CITED", detail
    return "SELECTED_NOT_CITED", detail


def _classify(expected_item, keep_eval, fused_rank, dropped_eval, cited_eval):
    """One expected-article entry (a plain {code, article}, or an "any_of"
    group like c2's Codice del Consumo range) -> (stage, detail) for the
    BEST (furthest) alternative, same way tests/eval_narrative.py's own
    scoring treats an any_of group as satisfied by any one member."""
    alts = ([(expected_item["code"], a) for a in expected_item["any_of"]] if "any_of" in expected_item
            else [(expected_item["code"], expected_item["article"])])
    best_stage, best_detail = "NOT_IN_POOL", ""
    for ref in alts:
        stage, detail = _classify_one(ref, keep_eval, fused_rank, dropped_eval, cited_eval)
        if _STAGE_RANK[stage] > _STAGE_RANK[best_stage]:
            best_stage, best_detail = stage, detail
    return best_stage, best_detail


def _run_case(case, forced_path):
    """Runs the live pipeline once for this case, capturing the fusion
    pool and the applicability check's drops from that one execution.
    Returns {"keep_eval", "fused_rank", "dropped_eval", "cited_eval",
    "seconds", "pool_fusion"}."""
    facts = case["question"]
    captured = {}

    real_fusion = dg._select_articles_fusion

    def capturing_fusion(*a, **k):
        result = real_fusion(*a, **k)
        captured["picked"] = result
        return result

    dropped_labels = []
    real_vlog = dg.vlog

    def capturing_vlog(event, data=None, duration_ms=None):
        if event == "case_analysis_applicability" and isinstance(data, dict) and data.get("dropped"):
            dropped_labels.extend(data["dropped"])
        real_vlog(event, data, duration_ms)

    real_enabled = feature_flags.enabled
    overrides = {}
    if forced_path == "v1":
        overrides[dg.CASE_ANALYSIS_FLAG] = False
    elif forced_path == "v2":
        overrides[dg.CASE_ANALYSIS_FLAG] = True
        overrides[dg._CASE_POOL_FUSION_FLAG] = False
    elif forced_path == "fusion":
        overrides[dg.CASE_ANALYSIS_FLAG] = True
        overrides[dg._CASE_POOL_FUSION_FLAG] = True

    def forced_enabled(name, default=False):
        if name in overrides:
            return overrides[name]
        return real_enabled(name, default)

    dg._select_articles_fusion = capturing_fusion
    dg.vlog = capturing_vlog
    feature_flags.enabled = forced_enabled
    t0 = time.time()
    try:
        result = dg.run_case_analysis_pipeline(facts, "it", "")
    finally:
        dg._select_articles_fusion = real_fusion
        dg.vlog = real_vlog
        feature_flags.enabled = real_enabled
    seconds = time.time() - t0

    picked = captured.get("picked")
    if picked is None:
        raise RuntimeError(
            f"{case['id']}: _select_articles_fusion never ran (CASE_ANALYSIS_ENABLED and/or "
            f"CASE_POOL_FUSION_ENABLED are off in this environment) -- pass --fusion to force the path")

    keep_eval = set(_ordered_eval_refs(picked["keep"]))
    fused_minus_keep = [r for r in picked["candidates"] if r not in picked["keep"]]
    fused_rank = {}
    for i, ref in enumerate(_ordered_eval_refs(fused_minus_keep), start=1):
        fused_rank.setdefault(ref, i)

    dropped_eval = {}
    for d in dropped_labels:
        number, docname = _parse_dropped_label(d.get("article", ""))
        code = _code_key(docname)
        if code:
            dropped_eval[(code, number)] = d.get("reason") or d.get("requisito") or "incompatibile"

    cited_eval = {(code, art) for art, code, _name in ev.cited(result.get("citations")) if code}

    return {"keep_eval": keep_eval, "fused_rank": fused_rank, "dropped_eval": dropped_eval,
            "cited_eval": cited_eval, "seconds": seconds}


def _case_lines(case, run):
    lines = [f"\n## {case['id']}  ({run['seconds']:.1f}s)"]
    for item in case.get("expected", []):
        stage, detail = _classify(item, run["keep_eval"], run["fused_rank"], run["dropped_eval"], run["cited_eval"])
        label = ev._label(item)
        lines.append(f"  {stage:<20s} {label:<28s} {detail}")
    return lines


def _totals_lines(cases, runs):
    totals = {s: 0 for s in _STAGE_ORDER}
    n = 0
    for case in cases:
        run = runs.get(case["id"])
        if not run:
            continue
        for item in case.get("expected", []):
            stage, _ = _classify(item, run["keep_eval"], run["fused_rank"], run["dropped_eval"], run["cited_eval"])
            totals[stage] += 1
            n += 1
    lines = ["\n## TOTALS", f"  {n} expected-article slots across {len(runs)} cases"]
    for stage in _STAGE_ORDER:
        lines.append(f"  {stage:<20s} {totals[stage]}")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cases", default=str(DEFAULT_CASES))
    parser.add_argument("--only", help="comma-separated case ids")
    parser.add_argument("--out", default="loss_funnel_runs")
    path = parser.add_mutually_exclusive_group()
    path.add_argument("--v1", action="store_true", help="force the v1 path, regardless of the flags")
    path.add_argument("--v2", action="store_true", help="force the v2 path, regardless of the flags")
    path.add_argument("--fusion", action="store_true", help="force the fusion path, regardless of the flags")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    forced_path = "fusion" if args.fusion else "v2" if args.v2 else "v1" if args.v1 else None
    case_analysis_on = feature_flags.enabled(dg.CASE_ANALYSIS_FLAG)
    fusion_on = case_analysis_on and feature_flags.enabled(dg._CASE_POOL_FUSION_FLAG)
    applicability_on = feature_flags.enabled("APPLICABILITY_CHECK_ENABLED")
    path_name = forced_path or ("fusion" if fusion_on else "v2" if case_analysis_on else "v1")
    print(f"Path: {path_name}{' (forced)' if forced_path else ' (from the live flags)'}")
    print(f"APPLICABILITY_CHECK_ENABLED: {applicability_on} (DROPPED_BY_CHECK can only happen when this is true)")
    if path_name != "fusion":
        print("WARNING: the live/forced path is not fusion -- pass --fusion to inspect that path specifically.")

    all_cases = {c["id"]: c for c in json.loads(Path(args.cases).read_text(encoding="utf-8"))["cases"]}
    cases = [all_cases[cid] for cid in _NARRATIVE_CASE_IDS if cid in all_cases]
    if args.only:
        wanted = {c.strip() for c in args.only.split(",")}
        cases = [c for c in cases if c["id"] in wanted]
    if not cases:
        parser.error("no cases selected")

    print(f"Running {len(cases)} case(s) sequentially -- one live pipeline call (with a draft) each, "
         "this will take a while.\n")

    runs, lines = {}, []
    for case in cases:
        print(f"... {case['id']}", flush=True)
        try:
            runs[case["id"]] = _run_case(case, forced_path)
        except Exception as exc:
            logging.warning("%s: %s", case["id"], exc)
            lines.append(f"\n## {case['id']}\n  ERROR: {exc}")
            continue
        lines += _case_lines(case, runs[case["id"]])

    lines += _totals_lines(cases, runs)
    output = "\n".join(lines) + "\n"
    print(output)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "funnel.txt").write_text(output, encoding="utf-8")
    print(f"Wrote {out_dir / 'funnel.txt'}. Nothing was written to the database.")


if __name__ == "__main__":
    main()
