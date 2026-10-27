"""Relations agent v0, the measurement (28 Oct): the same cases answered with ARTICLE_RELATIONS_ENABLED off (A) and on
(B), each saved by scripts/dryrun_strategy.py --save as <dir>/<case>_A.json / <case>_B.json. For each case and in
total: the expected articles in the pool, in the plan, MENTIONED in the answer (tests/eval_narrative.py's primary
score, as dryrun_strategy's expected trace reads it); what the relations added, how many of those were expected, and
how many reached the answer; the time. Prints article numbers and counts only.

    cd /tmp/relations_v0/newcode && ../../../opt/chatbot/venv/bin/python scripts/relations_compare.py --dir ../out
"""

import argparse
import importlib.util
import json
import os
import re
import sys
from typing import Dict, List, Optional, Set, Tuple

# The lawyer's two cases (run with --text): the c.c. articles a correct answer rests on (our reading of the review).
EXTRA = {"barca": ["1490", "1491", "1492", "1494", "1495", "1497", "1512"],
         "piscina": ["2043", "2050", "2051", "2055", "2056", "2059", "2947"]}


def eval_module(folder: str):
    spec = importlib.util.spec_from_file_location("eval_narrative", os.path.join(folder, "tests", "eval_narrative.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def last_event(events: List[dict], name: str) -> Optional[dict]:
    found = [e.get("data") for e in events if e.get("event") == name and e.get("data") is not None]
    return found[-1] if found else None


def wanted_sets(expected: List[dict]) -> List[Tuple[str, Set[Tuple[str, str]]]]:
    """[(label, {(article, code key)})] of the expected items (any_of: one of them is enough)."""
    out = []
    for item in expected:
        arts = item.get("any_of") or [item.get("article")]
        wanted = {(str(a).lower(), item["code"]) for a in arts if a}
        out.append((f"{'/'.join(sorted(a for a, _ in wanted))} {item['code']}", wanted))
    return out


def trace(record: dict, expected: List[dict], mentions) -> dict:
    """For one saved run: expected in pool / plan / answer, the related additions, the time."""
    events = record.get("events") or []
    draft = record.get("answer") or record.get("draft") or ""

    def keys(labels):
        return {ref for label in labels or [] for ref in mentions(f"art. {label}")}
    pool = keys((last_event(events, "case_analysis_retrieval") or {}).get("article_refs"))
    plan = keys((last_event(events, "case_analysis_plan") or {}).get("norme"))
    named = set(mentions(draft))
    rows = []
    for label, wanted in wanted_sets(expected):
        rows.append({"label": label, "pool": bool(wanted & pool), "plan": bool(wanted & plan),
                     "answer": bool(wanted & named)})
    added = (last_event(events, "article_relations_added") or {}).get("added") or {}
    added_keys = {label: keys([label]) for label in added}
    expected_keys = set().union(*(w for _, w in wanted_sets(expected))) if expected else set()
    return {"rows": rows, "seconds": record.get("seconds"), "status": record.get("status"),
            "added": sorted(added), "added_expected": sorted(l for l, k in added_keys.items() if k & expected_keys),
            "added_in_answer": sorted(l for l, k in added_keys.items() if k & named)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dir", required=True, help="the folder with <case>_A.json and <case>_B.json")
    p.add_argument("--cases", default="tests/eval_narrative_cases.json")
    p.add_argument("--code-folder", default=os.getcwd(), help="the folder holding tests/eval_narrative.py")
    args = p.parse_args()
    ev = eval_module(args.code_folder)
    data = json.load(open(args.cases, encoding="utf-8"))
    items = data if isinstance(data, list) else (data.get("cases") or list(data.values()))
    expected_of = {c["id"]: c.get("expected") or [] for c in items if isinstance(c, dict)}
    expected_of.update({k: [{"code": "cc", "article": a} for a in v] for k, v in EXTRA.items()})
    names = sorted({m.group(1) for f in os.listdir(args.dir) for m in [re.match(r"(.+)_[AB]\.json$", f)] if m})
    totals = {cfg: {"pool": 0, "plan": 0, "answer": 0, "n": 0, "seconds": 0.0} for cfg in "AB"}
    added_total, added_expected, added_answer = 0, 0, 0
    print("case          expected   pool A>B   plan A>B   ANSWER A>B   related added (expected / in answer)")
    for name in names:
        runs = {}
        for cfg in "AB":
            path = os.path.join(args.dir, f"{name}_{cfg}.json")
            if os.path.isfile(path):
                runs[cfg] = trace(json.load(open(path, encoding="utf-8")), expected_of.get(name, []), ev.mentions)
        if set(runs) != {"A", "B"}:
            print(f"{name:12}  missing a run ({', '.join(sorted(runs)) or 'none'})")
            continue
        n = len(runs["A"]["rows"])
        cells = []
        for col in ("pool", "plan", "answer"):
            a, b = (sum(r[col] for r in runs[cfg]["rows"]) for cfg in "AB")
            cells.append(f"{a}>{b}".ljust(10 if col != "answer" else 12))
            for cfg, v in (("A", a), ("B", b)):
                totals[cfg][col] += v
        for cfg in "AB":
            totals[cfg]["n"] += n
            totals[cfg]["seconds"] += float(runs[cfg]["seconds"] or 0)
        b = runs["B"]
        added_total += len(b["added"])
        added_expected += len(b["added_expected"])
        added_answer += len(b["added_in_answer"])
        print(f"{name:12}  {n:<9}  " + " ".join(cells)
              + f" {len(b['added'])} ({', '.join(b['added_expected']) or '-'} / {', '.join(b['added_in_answer']) or '-'})"
              + ("" if all(r['status'] == 'ok' for r in runs.values()) else f"  status A {runs['A']['status']}, B {b['status']}"))
        changed = [(rb["label"], rb["answer"]) for ra, rb in zip(runs["A"]["rows"], b["rows"]) if ra["answer"] != rb["answer"]]
        if changed:
            print(f"{'':14}answer changed for: " + ", ".join(f"{l} ({'gained' if g else 'lost'})" for l, g in changed))
    print()
    for cfg, label in (("A", "A (today)        "), ("B", "B (with relations)")):
        t = totals[cfg]
        n = max(t["n"], 1)
        print(f"{label}  expected {t['n']}: pool {t['pool']} ({100 * t['pool'] / n:.0f}%), plan {t['plan']} "
              f"({100 * t['plan'] / n:.0f}%), MENTIONED {t['answer']} ({100 * t['answer'] / n:.0f}%); "
              f"time {t['seconds'] / 60:.0f} min")
    print(f"relations added {added_total} articles: {added_expected} of them expected, {added_answer} reached the answer")
    return 0


if __name__ == "__main__":
    sys.exit(main())
