"""Read-only: the expected norms the long cases miss, across diagnose_long runs.

Every report.json under DIR (default ./diagnose_reports) is a run, named by its folder. Only the 32 cases of
diagnose_long's TUNING_IDS + HELDOUT_IDS are read; a run holding fewer than all 32 is listed as partial and left
out of the stable-miss logic. Prints, in markdown:

  0  the runs, with how many of the 32 cases each holds
  1  every expected norm whose stage is not MENTIONED in some complete run, with its stage per run; stable (missing
     in every complete run) and flaky separated
  2  the stable misses by loss mechanism, with the slot fields of that mechanism, one row per run
  3  per mechanism, the count of stable misses and the distinct values of the fields that explain them
  4  the unexpected norms mentioned in every complete run for the same case with grade >= 2

A stable miss's mechanism is the one of the most runs (ties: the earliest in STAGE order). Nothing is written.

  python scripts/miss_analysis.py [DIR]
"""

import argparse
import glob
import importlib.util
import json
import os
import re
from collections import Counter, defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))


def _case_ids():
    spec = importlib.util.spec_from_file_location("diagnose_long", os.path.join(_HERE, "diagnose_long.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.TUNING_IDS + module.HELDOUT_IDS), list(module.STAGES)


CASE_IDS, STAGES = _case_ids()
MECHANISMS = [
    ("never in pool", {"LOST_AT_UNDERSTANDING", "NOT_IN_POOL"},
     ["channels", "hits", "rank", "in_corpus", "resolved_by_institution", "named_by"]),
    ("in pool, not in prompt", {"BELOW_CAP", "REJECTED_BY_CHECK", "NOT_IN_PROMPT"}, ["cap", "check", "dropped", "rank"]),
    ("in prompt, not in plan", {"NOT_IN_PLAN"}, ["plan_notes", "check"]),
    ("in plan, not mentioned", {"NOT_WRITTEN"}, ["plan_notes", "marked_unverified", "flagged_by_qualification"]),
]
# Table 3: the fields that tell the misses of a mechanism apart (stage itself for the first two).
EXPLAINING = {"never in pool": ["stage", "in_corpus"], "in pool, not in prompt": ["stage", "cap", "check", "dropped"],
              "in prompt, not in plan": ["check", "plan_notes"],
              "in plan, not mentioned": ["plan_notes", "marked_unverified", "flagged_by_qualification", "check"]}


def mechanism_of(stage):
    return next((name for name, stages, _ in MECHANISMS if stage in stages), "?")


def field(slot, name):
    if name in ("in_corpus", "resolved_by_institution", "named_by"):
        return (slot.get("understanding") or {}).get(name)
    return slot.get(name)


def cell(value, limit=90):
    if value is None:
        return "–"
    if isinstance(value, list):
        value = "; ".join(str(v) for v in dict.fromkeys(str(v) for v in value)) if value else "[]"
    text = " ".join(str(value).split()).replace("|", "\\|")
    return text if len(text) <= limit else text[:limit - 1] + "…"


def category(name, value):
    """A field's value for table 3: a fill position or a vote rank folded into its kind."""
    text = cell(value, 200)
    if name == "cap":
        text = re.sub(r"#\d+", "#N", text)
    if name == "plan_notes" and isinstance(value, list):
        text = "; ".join(sorted(set(map(str, value)))) or "[]"
    return text


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(cell(c, 200) if not isinstance(c, str) else c.replace("|", "\\|") for c in row) + " |"
            for row in rows]
    return "\n".join(out)


def load_runs(root):
    runs = {}
    for path in sorted(glob.glob(os.path.join(root, "**", "report.json"), recursive=True)):
        name = os.path.relpath(os.path.dirname(path), root)
        with open(path, encoding="utf-8") as fh:
            cases = json.load(fh)
        runs[name] = {c["id"]: c for c in cases if c.get("id") in CASE_IDS}
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dir", nargs="?", default="diagnose_reports")
    args = parser.parse_args()
    runs = load_runs(args.dir)
    if not runs:
        raise SystemExit(f"no report.json under {args.dir}")
    complete = [r for r, cases in runs.items() if len(cases) == len(CASE_IDS)]
    partial = [r for r in runs if r not in complete]

    print("## 0. Runs\n")
    print(table(["run", f"cases of the {len(CASE_IDS)}", "in stable-miss logic"],
                [[r, str(len(runs[r])), "yes" if r in complete else "no"] for r in runs]))
    if partial:
        print("\nPartial runs (left out): " + ", ".join(f"{r} (missing {', '.join(i for i in CASE_IDS if i not in runs[r])})"
                                                      for r in partial))
    if not complete:
        raise SystemExit("\nno run holds all the cases")

    # (case, norm label) -> run -> slot
    slots = defaultdict(dict)
    for r in complete:
        for cid in CASE_IDS:
            for slot in runs[r][cid].get("slots") or []:
                slots[(cid, slot["label"])][r] = slot
    order = {cid: i for i, cid in enumerate(CASE_IDS)}
    missed = sorted((k for k, per in slots.items() if any(s["stage"] != "MENTIONED" for s in per.values())),
                    key=lambda k: (order[k[0]], k[1]))
    stable = [k for k in missed if all(r in slots[k] and slots[k][r]["stage"] != "MENTIONED" for r in complete)]
    flaky = [k for k in missed if k not in stable]

    print(f"\n## 1. Expected norms not MENTIONED ({len(stable)} stable, {len(flaky)} flaky, {len(complete)} runs)\n")
    for title, keys in (("Stable (missing in every run)", stable), ("Flaky", flaky)):
        print(f"### {title}\n")
        print(table(["case", "norm"] + complete,
                    [[cid, label] + [(slots[(cid, label)].get(r) or {}).get("stage", "–") for r in complete]
                     for cid, label in keys]))
        print()

    assigned = {}
    for k in stable:
        counts = Counter(mechanism_of(slots[k][r]["stage"]) for r in complete)
        names = [m for m, _, _ in MECHANISMS]
        assigned[k] = max(counts, key=lambda m: (counts[m], -names.index(m) if m in names else -99))

    print("## 2. Stable misses by loss mechanism\n")
    for name, stages, fields in MECHANISMS:
        keys = [k for k in stable if assigned[k] == name]
        print(f"### {name} ({len(keys)})\n")
        if not keys:
            print("(none)\n")
            continue
        print(table(["case", "norm", "run", "stage"] + fields,
                    [[cid, label, r, slots[(cid, label)][r]["stage"]] + [field(slots[(cid, label)][r], f) for f in fields]
                     for cid, label in keys for r in complete]))
        print()

    print("## 3. Per mechanism: stable misses and the values that explain them\n")
    rows = []
    for name, stages, _ in MECHANISMS:
        keys = [k for k in stable if assigned[k] == name]
        if not keys:
            rows.append([name, "0", "–", "–"])
            continue
        for f in EXPLAINING[name]:
            # per value: in how many stable misses it occurs (any run whose stage is of this mechanism)
            seen = Counter()
            for k in keys:
                values = {category(f, field(slots[k][r], f)) for r in complete if slots[k][r]["stage"] in stages}
                seen.update(values)
            rows.append([name, str(len(keys)), f, "; ".join(f"{v} ×{n}" for v, n in seen.most_common())])
    print(table(["mechanism", "stable misses", "field", "distinct values (× stable misses showing it)"], rows))

    print(f"\n## 4. Unexpected norms mentioned in every run for the same case with grade ≥ 2\n")
    found = []
    for cid in CASE_IDS:
        per_run = []
        for r in complete:
            per_run.append({u["label"]: u for u in runs[r][cid].get("unexpected") or []})
        for label in sorted(set.intersection(*(set(p) for p in per_run))):
            grades = [re.match(r"grade (\d)", str(p[label].get("check") or "")) for p in per_run]
            if all(g and int(g.group(1)) >= 2 for g in grades):
                first = per_run[-1][label]
                found.append([cid, label, ", ".join(g.group(1) for g in grades), cell(first.get("rubrica")),
                              cell(first.get("origin")), cell(first.get("in_plan"))])
    print(table(["case", "norm", "grade per run (" + ", ".join(complete) + ")", "rubrica", "origin (last run)",
                 "in plan (last run)"], found) if found else "(none)")

    print("\n## 5. Flaky norms: the non-MENTIONED cells by stage\n")
    by_stage = defaultdict(list)
    for cid, label in flaky:
        for r in complete:
            slot = slots[(cid, label)].get(r)
            if slot and slot["stage"] != "MENTIONED":
                by_stage[slot["stage"]].append((cid, label, r, slot))
    stages = [s for s in STAGES if s in by_stage] + sorted(s for s in by_stage if s not in STAGES)
    print(table(["stage", "cells", "check (× cells)"],
                [[s, str(len(by_stage[s])), "; ".join(f"{v} ×{n}" for v, n in Counter(
                    cell(c[3].get("check")) for c in by_stage[s]).most_common())] for s in stages]))
    for s in stages:
        extra = ["plan_notes", "flagged_by_qualification"] if s == "NOT_WRITTEN" else []
        print(f"\n### {s} ({len(by_stage[s])})\n")
        print(table(["case", "norm", "run", "check"] + extra,
                    [[cid, label, r, field(slot, "check")] + [field(slot, f) for f in extra]
                     for cid, label, r, slot in by_stage[s]]))


if __name__ == "__main__":
    main()
