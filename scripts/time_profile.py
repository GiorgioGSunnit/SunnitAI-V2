"""Where a long answer's time goes, from a diagnose_long run's saved data: read-only, nothing runs, no app import.

    python3 scripts/time_profile.py diagnose_long_runs/rv2-20261031-0834-candidate [more run folders] [--out file.md]

Per case: every LLM call's stage, start and duration (the header of cases/<id>/llm/NN_<stage>.txt, so a run made with
--dump-llm, the default), the duration events of cases/<id>.json, and the case's total (case_analysis_pipeline, else
the record's seconds). rubrica_fit is diagnose_long's own call after the answer, outside the total: left out.

A stage's seconds in a case are wall time, the union of its calls: parallel calls do not add up. A group's seconds are
that union for its stages, or its logged windows when longer (they also hold the database work between the calls):
the groups run one after the other, so what is left of the total is "unattributed", time no call or window accounts
for. A case over its total (rounding: durations are saved to 0.1 s) counts 0.

Chat calls with no stage of their own: diagnose_long names "draft" any one made on the main thread (it stops at
run_case_analysis_pipeline) and other_<function> one made on a worker thread, so they are read here by their place:
before the check (the router's tie-break), between the check and the plan (the ruling fit), after the sections' start
(the post-writing checks). A "draft" in a case without a plan is the one-call answer. The logged events are shown in
their own table: they nest (the vote inside retrieval), so they are not added up.

Cases ran 3 at a time (diagnose_long's MAX_PARALLEL) on one LLM endpoint: these are the shares of a loaded run, not
production latency; a case alone is faster, and the LLM stages gain most.
"""

import argparse
import json
import os
import re
import statistics
from collections import defaultdict

GROUPS = ("understanding", "retrieval", "graded check", "plan", "sections", "post-writing")
STAGE_GROUP = {
    "frame": "understanding", "fact_signals": "understanding", "fact_issues": "understanding",
    "case_analysis_v1": "understanding", "case_analysis_v2": "understanding",
    "case_analysis_structured": "understanding", "structured_facts": "understanding", "structured_grid": "understanding",
    "distill": "retrieval", "v1_choice": "retrieval", "issue_spotting": "retrieval",
    "applicability": "graded check", "applicability_retry": "graded check", "ruling_fit": "graded check",
    "principles": "plan", "plan": "plan", "plan_stance": "plan", "plan_check": "plan",
    "citation_rewrite": "post-writing", "stance_rewrite": "post-writing", "qualification": "post-writing",
    "grounded_judge": "post-writing", "grounded_rewrite": "post-writing", "anchor_rewrite": "post-writing",
    "writer_review": "post-writing", "unsupported_claims": "post-writing", "complete_sections": "post-writing",
    "second_opinion": "post-writing",
}
# diagnose_long's other_<nearest function>, for the post-writing steps' worker threads.
OTHER_NAMES = {"other_review_section": "writer_review", "other_rewrite_unsupported": "unsupported_claims",
               "other_again": "complete_sections", "other_ask": "second_opinion"}
OUTSIDE = {"rubrica_fit"}
# Events timing a whole step: they also hold what is not a chat call. Sequential within their group, so they add up.
WINDOWS = {
    "retrieval": ("case_analysis_retrieval",),
    "graded check": ("case_analysis_applicability", "case_analysis_ruling_fit_llm"),
    "sections": ("case_analysis_sections_llm",),
    "post-writing": ("case_analysis_citation_discipline", "case_analysis_writer_rules",
                     "case_analysis_unsupported_claims", "anchored_writing_llm", "grounded_review_llm",
                     "case_analysis_qualification_llm", "case_analysis_second_opinion", "case_analysis_invariants"),
}
PARALLEL_NOTE = ("Cases ran 3 at a time (diagnose_long's MAX_PARALLEL) on one LLM endpoint: shares of a loaded run, not "
                 "production latency.")
_HEAD_RE = re.compile(r"^(stage|started|duration|status): (.*)$", re.MULTILINE)


def read_call(path: str) -> dict:
    """{"stage", "start", "seconds", "ok"} from an llm/NN_<stage>.txt header; the rest of the file is not read."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        head = fh.read(2000).partition("\n\n===== ")[0]
    fields = dict(_HEAD_RE.findall(head))
    number = lambda key: float(re.search(r"-?\d+(?:\.\d+)?", fields.get(key) or "0").group())   # noqa: E731
    return {"stage": fields.get("stage") or re.sub(r"^\d+_|\.txt$", "", os.path.basename(path)),
            "start": number("started"), "seconds": number("duration"),
            "ok": (fields.get("status") or "ok").startswith("ok")}


def named(calls: list) -> list:
    """The calls with their true stage and group ("stage", "group"), without those outside the answer."""
    def first(*stages):
        return min((c["start"] for c in calls if c["base"] in stages), default=None)
    for call in calls:
        call["base"] = re.sub(r"_json_retry$", "", call["stage"])
    check = first("applicability", "applicability_retry")
    plan = first("principles", "plan", "plan_stance", "plan_check")
    written = min((c["start"] for c in calls if c["base"].startswith("section_")), default=None)
    out = []
    for call in calls:
        stage = OTHER_NAMES.get(call["base"], call["base"])
        if stage in OUTSIDE:
            continue
        if stage.startswith("section_"):
            group = "sections"
        elif stage in STAGE_GROUP:
            group = STAGE_GROUP[stage]
        elif stage == "draft" and plan is None:
            group = "sections"
        elif stage == "draft" and written is None and call["start"] >= (plan or 0):
            group = "sections"           # the plan failed: the one-call answer
        elif written is not None and call["start"] >= written:
            group = "post-writing"
            stage = "post_writing_other" if stage == "draft" else stage
        elif check is not None and call["start"] >= check:
            group = "graded check" if plan is None or call["start"] < plan else "plan"
            stage = "ruling_fit" if stage == "draft" and group == "graded check" else stage
        else:
            group = "retrieval"
            stage = "router_tie" if stage == "draft" else stage
        out.append({**call, "stage": stage, "group": group})
    return out


def wall(calls: list) -> float:
    """Seconds covered by at least one of the calls."""
    total, end = 0.0, None
    for start, stop in sorted((c["start"], c["start"] + c["seconds"]) for c in calls):
        if end is None or start > end:
            total, end = total + stop - start, stop
        elif stop > end:
            total, end = total + stop - end, stop
    return total


def profile_case(rec: dict, calls: list) -> dict:
    events = [e for e in rec.get("events") or [] if isinstance(e, (list, tuple)) and len(e) >= 3 and e[2] is not None]
    by_event = defaultdict(list)
    for e in events:
        by_event[e[0]].append(e[2] / 1000)
    pipeline = by_event.pop("case_analysis_pipeline", None)
    total = pipeline[-1] if pipeline else float(rec.get("seconds") or 0)
    calls = named(calls)
    stages = defaultdict(list)
    for call in calls:
        stages[(call["group"], call["stage"])].append(call)
    groups = {}
    for group in GROUPS:
        windows = sum(sum(by_event.get(name, [])) for name in WINDOWS.get(group, ()))
        groups[group] = max(wall([c for c in calls if c["group"] == group]), windows)
    return {"id": rec.get("id"), "status": rec.get("status"), "total": total, "groups": groups,
            "unattributed": max(0.0, total - sum(groups.values())), "calls": len(calls),
            "failed_calls": sum(1 for c in calls if not c["ok"]),
            "stages": {k: {"calls": len(v), "seconds": wall(v)} for k, v in stages.items()},
            "events": {name: {"calls": len(v), "seconds": sum(v)} for name, v in by_event.items()}}


def load_run(run_dir: str) -> tuple:
    """([profile per case], [case ids without an llm/ folder])."""
    cases, no_dump = [], []
    folder = os.path.join(run_dir, "cases")
    for name in sorted(os.listdir(folder)):
        if not name.endswith(".json") or name.endswith(".input.json"):
            continue
        with open(os.path.join(folder, name), encoding="utf-8") as fh:
            rec = json.load(fh)
        llm = os.path.join(folder, name[:-5], "llm")
        if not os.path.isdir(llm):
            no_dump.append(rec.get("id"))
        calls = [read_call(os.path.join(llm, f)) for f in sorted(os.listdir(llm)) if f.endswith(".txt")] \
            if os.path.isdir(llm) else []
        cases.append(profile_case(rec, calls))
    return cases, no_dump


def _stats(values: list) -> str:
    return f"{statistics.mean(values):.1f} | {statistics.median(values):.1f} | {max(values):.1f}"


def render(run_dir: str, cases: list, no_dump: list) -> str:
    n = len(cases)
    grand = sum(c["total"] for c in cases) or 1
    totals = [c["total"] for c in cases]
    failed = [str(c["id"]) for c in cases if not str(c["status"] or "ok").startswith("ok")]
    L = [f"# Time profile: {os.path.basename(os.path.normpath(run_dir))}", "", PARALLEL_NOTE, "",
         f"{n} cases, per answer: mean {statistics.mean(totals):.0f}s, median {statistics.median(totals):.0f}s, max "
         f"{max(totals):.0f}s; {sum(c['calls'] for c in cases) / n:.1f} LLM calls."
         + (f" Failed: {', '.join(failed)}." if failed else "")
         + (f" Without LLM dumps (groups understated): {', '.join(map(str, no_dump))}." if no_dump else ""),
         "", "## By group", "", "| group | mean s | median s | max s | share of total |", "|---|---:|---:|---:|---:|"]
    for group in GROUPS:
        values = [c["groups"][group] for c in cases]
        L.append(f"| {group} | {_stats(values)} | {100 * sum(values) / grand:.1f}% |")
    values = [c["unattributed"] for c in cases]
    L.append(f"| unattributed | {_stats(values)} | {100 * sum(values) / grand:.1f}% |")
    L += ["", "## By LLM stage", "", "Seconds are wall time over the cases that have the stage; calls per case over all "
          "of them. Shares within a group can overlap the window it is measured by.", "",
          "| group | stage | calls per case | cases | mean s | median s | max s | share of total |",
          "|---|---|---:|---:|---:|---:|---:|---:|"]
    keys = {k for c in cases for k in c["stages"]}
    for key in sorted(keys, key=lambda k: (GROUPS.index(k[0]), k[1])):
        rows = [c["stages"][key] for c in cases if key in c["stages"]]
        seconds = [r["seconds"] for r in rows]
        L.append(f"| {key[0]} | {key[1]} | {sum(r['calls'] for r in rows) / n:.1f} | {len(rows)} | {_stats(seconds)} | "
                 f"{100 * sum(seconds) / grand:.1f}% |")
    L += ["", "## Logged duration events", "", "As logged, summed per case. They nest, so they do not add up.", "",
          "| event | calls per case | cases | mean s | median s | max s | share of total |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for name in sorted({k for c in cases for k in c["events"]},
                       key=lambda k: -sum(c["events"].get(k, {}).get("seconds", 0) for c in cases)):
        rows = [c["events"][name] for c in cases if name in c["events"]]
        seconds = [r["seconds"] for r in rows]
        L.append(f"| {name} | {sum(r['calls'] for r in rows) / n:.1f} | {len(rows)} | {_stats(seconds)} | "
                 f"{100 * sum(seconds) / grand:.1f}% |")
    L += ["", "## Per case", "", "| case | total s | LLM calls | " + " | ".join(GROUPS) + " | unattributed |",
          "|---|---:|---:|" + "---:|" * (len(GROUPS) + 1)]
    for c in cases:
        L.append(f"| {c['id']} | {c['total']:.0f} | {c['calls']} | "
                 + " | ".join(f"{c['groups'][g]:.0f}" for g in GROUPS) + f" | {c['unattributed']:.0f} |")
    return "\n".join(L) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("runs", nargs="+", help="diagnose_long run folders")
    parser.add_argument("--out", help="write the report here too")
    args = parser.parse_args()
    parts = []
    for run in args.runs:
        cases, no_dump = load_run(run)
        parts.append(render(run, cases, no_dump) if cases else f"# {run}: no cases\n")
    text = "\n".join(parts)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)


if __name__ == "__main__":
    main()
