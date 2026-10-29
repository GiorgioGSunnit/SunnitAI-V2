"""Read-only (29 Oct): where the expected articles found by the search are lost before the plan.

The relations measurement (28 Oct, 17 civil cases, run A = production's settings) found 58 of 64 expected articles in
the pool and 42 in the plan. For each expected article in the pool but not in the plan, this names the first step that
dropped it, from the events dryrun_strategy.py --save keeps:
  removed unchecked     past the applicability check's slots (evidence_cutoff "checked_slots"), removed unchecked
  graded 0, removed     the applicability check graded it 0 and incompatible (or the graded check removed it)
  graded 1              checked, grade 1: the plan takes only grade 2 as applicable (case_analysis_plan_applicable)
  prompt size           left out of the analysis prompt (case_analysis_prompt_tokens)
  graded 2, not taken   applicable, but the plan's issues did not take it
and, for each, whether it was MENTIONED in the answer anyway. Prints article numbers and counts only.

    python scripts/plan_loss.py --dir <folder with the saved runs> [--config A]
"""

import argparse
import importlib.util
import json
import os
import re
import sys
from collections import Counter
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("relations_compare", os.path.join(HERE, "relations_compare.py"))
rc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rc)

STEPS = ["removed unchecked", "graded 0, removed", "graded 1", "prompt size", "graded 2, not taken", "other"]


def _all(events: List[dict], name: str) -> List[dict]:
    return [e.get("data") for e in events if e.get("event") == name and isinstance(e.get("data"), dict)]


def stages(events: List[dict], keys) -> Dict[str, object]:
    """The plan stage's lists, as sets of (article, code key)."""
    applicability = [d for d in _all(events, "case_analysis_applicability") if "grades" in d or "removed_unchecked" in d]
    last = applicability[-1] if applicability else {}
    grades: Dict[tuple, int] = {}
    for label, grade in (last.get("grades") or {}).items():
        for k in keys([label]):
            grades[k] = grade
    removed_graded = set()
    for d in _all(events, "evidence_removal"):
        if d.get("stage") == "graded check":
            removed_graded |= keys([e.get("article") for e in d.get("entries") or []])
    removed_graded |= keys([e.get("article") for e in last.get("dropped") or []])
    prompt = set()
    for d in _all(events, "case_analysis_prompt_tokens"):
        prompt |= keys(d.get("articles_left_out"))
    plan_applicable = (rc.last_event(events, "case_analysis_plan_applicable") or {})
    return {"unchecked": keys(last.get("removed_unchecked")), "graded_removed": removed_graded, "grades": grades,
            "prompt": prompt, "applicable": keys(plan_applicable.get("applicable")),
            "max_issues": plan_applicable.get("max_issues")}


def step_of(wanted: set, s: Dict[str, object]) -> str:
    if wanted & s["unchecked"]:
        return "removed unchecked"
    if wanted & s["graded_removed"]:
        return "graded 0, removed"
    grades = [s["grades"][k] for k in wanted if k in s["grades"]]
    if grades and max(grades) == 1:
        return "graded 1"
    if wanted & s["prompt"]:
        return "prompt size"
    if (grades and max(grades) >= 2) or wanted & s["applicable"]:
        return "graded 2, not taken"
    return "other"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dir", required=True)
    p.add_argument("--config", default="A")
    p.add_argument("--cases", default="tests/eval_narrative_cases.json")
    p.add_argument("--code-folder", default=os.getcwd(), help="the folder holding tests/eval_narrative.py")
    args = p.parse_args()
    ev = rc.eval_module(args.code_folder)
    data = json.load(open(args.cases, encoding="utf-8"))
    items = data if isinstance(data, list) else (data.get("cases") or list(data.values()))
    expected_of = {c["id"]: c.get("expected") or [] for c in items if isinstance(c, dict)}
    expected_of.update({k: [{"code": "cc", "article": a} for a in v] for k, v in rc.EXTRA.items()})

    def keys(labels):
        return {ref for label in labels or [] if label for ref in ev.mentions(f"art. {label}")}

    totals, in_answer, n_expected, n_pool, n_plan = Counter(), Counter(), 0, 0, 0
    print(f"{'case':12} lost before the plan (step)")
    for f in sorted(os.listdir(args.dir)):
        m = re.match(rf"(.+)_{args.config}\.json$", f)
        if not m:
            continue
        name, record = m.group(1), json.load(open(os.path.join(args.dir, f), encoding="utf-8"))
        events = record.get("events") or []
        pool = keys((rc.last_event(events, "case_analysis_retrieval") or {}).get("article_refs"))
        plan = keys((rc.last_event(events, "case_analysis_plan") or {}).get("norme"))
        named = set(ev.mentions(record.get("answer") or record.get("draft") or ""))
        s = stages(events, keys)
        lost = []
        for label, wanted in rc.wanted_sets(expected_of.get(name, [])):
            n_expected += 1
            n_pool += bool(wanted & pool)
            n_plan += bool(wanted & plan)
            if wanted & pool and not wanted & plan:
                step = step_of(wanted, s)
                totals[step] += 1
                in_answer[step] += bool(wanted & named)
                lost.append(f"{label} ({step}{', in the answer anyway' if wanted & named else ''})")
        print(f"{name:12} " + ("; ".join(lost) or "-"))
    print(f"\nexpected {n_expected}: in the pool {n_pool}, in the plan {n_plan}; lost between them {n_pool - n_plan}")
    for step in STEPS:
        if totals[step]:
            print(f"  {totals[step]:>3}  {step}  (in the answer anyway: {in_answer[step]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
