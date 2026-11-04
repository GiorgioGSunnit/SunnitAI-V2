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
  5  the flaky norms' non-MENTIONED cells by stage
  6  per run (partial ones too), the unexpected norms the qualification check found rubrica-incompatible
     (fit.compatibile false), with channels, grade, plan_source and whether the norm is a regime candidate or a
     procedural tool; then the diff between MISUSE_RUNS

A stable miss's mechanism is the one of the most runs (ties: the earliest in STAGE order). Nothing is written.

  python scripts/miss_analysis.py [DIR] [--misuse-runs A B]
"""

import argparse
import ast
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


# Table 6: the runs whose misused norms are diffed; the tables fact_signals' regime candidates and tools come from.
MISUSE_RUNS = ("v45-candidate", "v37-cards-off")
FACT_SIGNALS = os.path.join(_HERE, os.pardir, "src", "rag", "fact_signals.py")
REGIME_SOURCE = "regime_candidate"       # defensive_generation.REGIME_CANDIDATE_SOURCE: a channel once diagnose_long has it


def _ref_tables(path=FACT_SIGNALS):
    """{"regime": {label: [ids]}, "tool": {label: [ids]}} from fact_signals.REGIME_CANDIDATES and PROCEDURAL_TOOLS,
    read from the source (no import: it needs the app's dependencies); {} for both if the file is missing."""
    out = {"regime": defaultdict(list), "tool": defaultdict(list)}
    try:
        tree = ast.parse(open(path, encoding="utf-8").read())
    except OSError:
        return out
    names = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            names.update({t.id: node.value.value for t in node.targets if isinstance(t, ast.Name)})
    kinds = {"REGIME_CANDIDATES": "regime", "PROCEDURAL_TOOLS": "tool"}
    for node in tree.body:
        target = getattr(node, "target", None) or (node.targets[0] if isinstance(node, ast.Assign) else None)
        if not isinstance(target, ast.Name) or target.id not in kinds or not isinstance(node.value, ast.List):
            continue
        for item in node.value.elts:
            if not isinstance(item, ast.Dict):
                continue
            entry = {k.value: v for k, v in zip(item.keys, item.values) if isinstance(k, ast.Constant)}
            ident = entry["id"].value if isinstance(entry.get("id"), ast.Constant) else "?"
            for ref in getattr(entry.get("refs"), "elts", []):
                parts = [p.value if isinstance(p, ast.Constant) else names.get(getattr(p, "id", ""), "?")
                         for p in ref.elts]
                out[kinds[target.id]][_key(f"{parts[0]} {parts[1]}")].append(ident)
    return out


def _key(label):
    return " ".join(str(label).lower().replace(".", " ").split())


def misused(case):
    """The case's unexpected norms the qualification check found incompatible with their rubrica."""
    return [u for u in case.get("unexpected") or [] if (u.get("fit") or {}).get("compatibile") is False]


def provenance(u, tables):
    """Whether u is a regime candidate or a procedural tool: by the regime_candidate channel when the run has it,
    else by its article being in the table (marked "table only")."""
    out = []
    regimes = tables["regime"].get(_key(u["label"]))
    if REGIME_SOURCE in (u.get("channels") or []):
        out.append("regime candidate" + (f" ({', '.join(regimes)})" if regimes else ""))
    elif regimes:
        out.append(f"regime table only ({', '.join(regimes)})")
    tools = tables["tool"].get(_key(u["label"]))
    if tools:
        out.append(f"procedural tool ({', '.join(tools)})")
    return "; ".join(out) or "no"


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
    parser.add_argument("--misuse-runs", nargs=2, default=list(MISUSE_RUNS), metavar=("A", "B"),
                        help="table 6's diff: misused in A not in B, and vice versa")
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

    misuse_table(runs, args.misuse_runs)


def misuse_table(runs, pair):
    tables = _ref_tables()
    order = {cid: i for i, cid in enumerate(CASE_IDS)}
    print("\n## 6. Unexpected norms flagged rubrica-incompatible (fit.compatibile false)\n")
    print("plan_source: the row's plan_source if the report has one, else \"rule\" when its plan_notes say so, else –. "
          "regime / tool: \"regime table only\" means the article is in fact_signals.REGIME_CANDIDATES but the run "
          f"shows no {REGIME_SOURCE} channel.\n")
    for r in runs:
        rows = [[cid, u["label"], u.get("channels") or [], u.get("check"),
                 u.get("plan_source") or ("rule" if "rule" in (u.get("plan_notes") or []) else "–"),
                 provenance(u, tables), (u.get("fit") or {}).get("perche")]
                for cid in sorted(runs[r], key=order.get) for u in misused(runs[r][cid])]
        print(f"### {r} ({len(rows)} in {len({row[0] for row in rows})} cases)\n")
        print(table(["case", "norm", "channels", "grade", "plan_source", "regime / tool", "why (check)"], rows)
              if rows else "(none)")
        print()
    a, b = pair
    if a not in runs or b not in runs:
        print(f"(diff skipped: {', '.join(x for x in pair if x not in runs)} not under DIR)")
        return

    def status(run, cid, label):
        case = runs[run].get(cid)
        if case is None:
            return "case not in run"
        if any(s["label"] == label for s in case.get("slots") or []):
            return "expected norm"
        u = next((u for u in case.get("unexpected") or [] if u["label"] == label), None)
        if u is None:
            return "not unexpected (absent)"
        verdict = (u.get("fit") or {}).get("compatibile")
        return f"unexpected, {'compatible' if verdict else 'unchecked'}, {u.get('check')}"

    for one, other in ((a, b), (b, a)):
        theirs = {(cid, u["label"]) for cid, case in runs[other].items() for u in misused(case)}
        rows = [[cid, u["label"], u.get("channels") or [], u.get("check"), provenance(u, tables), status(other, cid, u["label"])]
                for cid in sorted(runs[one], key=order.get) for u in misused(runs[one][cid])
                if (cid, u["label"]) not in theirs]
        print(f"### Misused in {one}, not in {other} ({len(rows)})\n")
        print(table(["case", "norm", f"channels ({one})", f"grade ({one})", "regime / tool", f"in {other}"], rows)
              if rows else "(none)")
        print()


if __name__ == "__main__":
    main()
