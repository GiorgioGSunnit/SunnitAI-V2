"""Offline: EDGE_EXPAND's neighbour policies measured on the runs' dumps -- for every missed expected norm (not
MENTIONED), whether the policy reaches it, and how many rows it adds per case (v48).

    python scripts/edge_policy.py [DIR] [--runs v46-candidate v47/v47-candidate]

DIR (default ./diagnose_reports) holds article_links_out/ (units.csv, refers_to.csv: scripts/article_links.py's dry run)
and each run's report.json. Per case, from its pool_traces: the rows the check kept (check not "rejected", not grade 0,
not dropped) with their grade and signals, the articles in the prompt; then edge_expand.plan_expansion under each
policy (the module's parameters set for it):
  P1  per anchor the SEZIONE_N=2 nearest same-Sezione articles + the REFERS_N=2 REFERS_TO targets fewest cited, cap 40;
  P2  P1, cap 60;
  P3  P2 plus the 2 articles citing the anchor, same specificity rule;
  P4  P1 with anchors only graded >= 2 or carrying a signal.
A miss is reached when one of its articles is chosen (to be graded, or tagged when it is in the rows). Rows per case:
the ones sent to the check (new, or never graded). Prints a table policy x (misses reached, mean and max rows per
case), the misses no policy reaches (and how many are in the prompt already: anchors, not reachable), the totals over
the runs with the grading calls per case, and the three targets (h1 1218 c.c., h3 1158 c.c., h5 593 c.p.p.) under each."""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

for _k, _v in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
               ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(_k, _v)

import link_reach  # noqa: E402
from src.rag import edge_expand as ee  # noqa: E402

POLICIES = {"P1": {"MAX_ADDED": 40},
            "P2": {"MAX_ADDED": 60},
            "P3": {"MAX_ADDED": 60, "REFERS_SOURCES": True},
            "P4": {"MAX_ADDED": 40, "ANCHOR_MIN_GRADE": 2}}
BASE = {"SEZIONE_N": 2, "REFERS_N": 2, "REFERS_SOURCES": False, "ANCHOR_MIN_GRADE": None, "MAX_ADDED": 40}
TARGETS = (("h1", ("1218", "c.c.")), ("h3", ("1158", "c.c.")), ("h5", ("593", "c.p.p.")))


def case_rows(case: dict):
    """(rows the check kept, rows before the check, the prompt's article tags) from the case's pool_traces."""
    checked, before, prompt = [], [], []
    for t in case.get("pool_traces") or []:
        ref = link_reach.parse_label(t.get("label"))
        if not ref or not t.get("in_pool"):
            continue
        check = str(t.get("check") or "")
        g = re.match(r"grade (\d)", check)
        row = {"_article": ref, "_grade": int(g.group(1)) if g else None, "_signals": list(t.get("signals") or [])}
        if check.startswith("exempt: heading"):
            row["_heading_match"] = True
        elif check.startswith("exempt: crossref"):
            from src.rag.nodes.cypher import CROSSREF_SOURCE
            row["_source"] = CROSSREF_SOURCE
        elif check.startswith("unchecked"):
            row["_applicability"] = "non valutato"
        before.append(row)
        if check != "rejected" and not check.startswith("grade 0") and not t.get("dropped"):
            checked.append(row)
        if t.get("in_prompt"):
            prompt.append(f"[art. {ref[0]} {ref[1]}]")
    return checked, before, "\n".join(prompt)


def run_policy(run: dict, lookup, params: dict):
    """{case id: plan_expansion result} under params."""
    for name, value in {**BASE, **params}.items():
        setattr(ee, name, value)
    out = {}
    for cid, case in run.items():
        checked, before, prompt = case_rows(case)
        out[cid] = ee.plan_expansion(checked, before, lookup, prompt)
    return out


def missed(run: dict):
    return [(cid, s) for cid, case in run.items() for s in case.get("slots") or [] if s.get("stage") != "MENTIONED"]


def in_prompt(case: dict, slot: dict) -> bool:
    keys = {ee.key(r) for r in link_reach.slot_articles(slot)}
    return any(t.get("in_prompt") and (link_reach.parse_label(t.get("label")) or ("", ""))
               and ee.key(link_reach.parse_label(t.get("label"))) in keys for t in case.get("pool_traces") or [])


def chosen(plan: dict, refs) -> dict:
    """The chosen entry for one of refs, or None."""
    keys = {ee.key(r) for r in refs}
    return next((e for e in plan["ranked"] if e["key"] in keys), None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("dir", nargs="?", default="diagnose_reports")
    parser.add_argument("--runs", nargs="+", default=["v46-candidate", "v47/v47-candidate"])
    parser.add_argument("--policies", nargs="+", default=list(POLICIES))
    args = parser.parse_args()
    links = os.path.join(args.dir, "article_links_out")
    lookup = ee.CsvLookup(os.path.join(links, "units.csv"), os.path.join(links, "refers_to.csv"))
    saved = {name: getattr(ee, name) for name in BASE}
    totals: dict = {}
    try:
        for name in args.runs:
            run = link_reach.load_run(args.dir, name)
            if run is None:
                print(f"## {name}\n\n(no report.json)\n")
                continue
            misses = missed(run)
            anchored = sum(1 for cid, slot in misses if in_prompt(run[cid], slot))
            print(f"## {name}: {len(misses)} misses in {len(run)} cases; {anchored} of them already in the prompt "
                  f"(anchors themselves: no policy can add them)\n")
            print("| policy | misses reached | of which to grade / tag only | mean rows/case | max rows/case | "
                  "reach per 100 rows |\n|---|---:|---:|---:|---:|---:|")
            reached_by = {}
            for pol in args.policies:
                plans = run_policy(run, lookup, POLICIES[pol])
                rows = [len(p["grade"]) for p in plans.values()]
                hits = {}
                for cid, slot in misses:
                    e = chosen(plans[cid], link_reach.slot_articles(slot))
                    if e:
                        hits[(cid, slot.get("label"))] = e
                reached_by[pol] = hits
                total = totals.setdefault(pol, [0, 0, 0])
                total[0] += len(hits)
                total[1] += sum(rows)
                total[2] += len(rows)
                grade = sum(1 for e in hits.values() if e["action"] == "grade")
                print(f"| {pol} | {len(hits)} / {len(misses)} | {grade} / {len(hits) - grade} | "
                      f"{sum(rows) / len(rows):.1f} | {max(rows)} | {100 * len(hits) / max(1, sum(rows)):.2f} |")
            print()
            print("| miss | stage | " + " | ".join(args.policies) + " |\n|---|---|" + "---|" * len(args.policies))
            for cid, slot in misses:
                k = (cid, slot.get("label"))
                cells = [f"#{reached_by[p][k]['rank']} {'+'.join(c.split(':')[1] for c in reached_by[p][k]['channels'])}"
                         if k in reached_by[p] else "-" for p in args.policies]
                print(f"| {cid} {slot.get('label')} | {slot.get('stage')} | " + " | ".join(cells) + " |")
            never = [f"{cid} {slot.get('label')}" for cid, slot in misses
                     if not any((cid, slot.get("label")) in reached_by[p] for p in args.policies)]
            print(f"\nReached by no policy ({len(never)}): " + "; ".join(never) + "\n")
            if all(cid in run for cid, _ in TARGETS):
                print("| target | " + " | ".join(args.policies) + " |\n|---|" + "---|" * len(args.policies))
                for cid, ref in TARGETS:
                    cells = []
                    for p in args.policies:
                        slot = next((s for c, s in misses if c == cid and ee.key(ref) in
                                     {ee.key(r) for r in link_reach.slot_articles(s)}), None)
                        e = reached_by[p].get((cid, slot.get("label"))) if slot else None
                        cells.append(f"#{e['rank']} ({e['action']}, {', '.join(e['anchors'])})" if e else "-")
                    print(f"| {cid} {ref[0]} {ref[1]} | " + " | ".join(cells) + " |")
                print()
        if totals:
            print("All runs: | policy | misses reached | rows | mean rows/case | grading calls/case | reach per 100 rows |"
                  "\n|---|---:|---:|---:|---:|---:|")
            for pol, (hits, rows, cases) in totals.items():
                print(f"| {pol} | {hits} | {rows} | {rows / cases:.1f} | {ee.grading_calls(round(rows / cases))} | "
                      f"{100 * hits / max(1, rows):.2f} |")
    finally:
        for name, value in saved.items():
            setattr(ee, name, value)


if __name__ == "__main__":
    main()
