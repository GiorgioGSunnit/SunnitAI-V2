"""Relations agent, test 2 (28 Oct): can the relations help the CHOICE among the articles already found? Offline, no
model call: reads the A runs of the measurement (scripts/relations_compare.py's folder: the pool and the plan of today's
pipeline) and the relations file (scripts/maint_relations.py).

Measurement v0 showed the pipeline already has 91% of the expected articles in its pool and loses a quarter of them at
the plan (pool 58 -> plan 42 of 64). Here, for each case and the Codice Civile articles of its pool:
  coherence(a) = the sum of the relation scores between a and every other article of the pool
  1. separation: how often an expected article has a higher coherence than an unexpected one (0.5 = no signal)
  2. rescue: of the expected articles the plan dropped, how many are among the pool's most coherent (top as many as
     the plan kept, at least 5), and how many unexpected articles that ranking would bring in with them
Prints article numbers and scores only.

    cd /tmp/relations_v0/newcode && /opt/chatbot/venv/bin/python scripts/relations_choice.py --dir ../out \\
        --relations /tmp/relations_v0/relations_cc.json
"""

import argparse
import json
import math
import os
import re
import sys
from typing import Dict, List, Optional, Set, Tuple

EXTRA = {"barca": ["1490", "1491", "1492", "1494", "1495", "1497", "1512"],
         "piscina": ["2043", "2050", "2051", "2055", "2056", "2059", "2947"]}
_CC = re.compile(r"^(\d+(?:-[a-z]+)?)\s+(?:c\.c\.|Codice Civile)", re.IGNORECASE)


def cc_numbers(labels: Optional[List[str]]) -> List[str]:
    """'1490 c.c.' / '1490 Codice Civile 2026' -> '1490', the others dropped, the order kept."""
    out = []
    for label in labels or []:
        m = _CC.match(str(label).strip())
        if m and m.group(1).lower() not in out:
            out.append(m.group(1).lower())
    return out


def last_event(events: List[dict], name: str) -> Optional[dict]:
    found = [e.get("data") for e in events if e.get("event") == name and e.get("data") is not None]
    return found[-1] if found else None


def relation_scores(related: Dict[str, List[list]]) -> Dict[Tuple[str, str], float]:
    """{(a, b): score} symmetric: the higher of a -> b and b -> a."""
    out: Dict[Tuple[str, str], float] = {}
    for a, rows in related.items():
        for b, score in rows:
            for pair in ((a, b), (b, a)):
                out[pair] = max(out.get(pair, 0.0), float(score))
    return out


def dampen_hubs(scores: Dict[Tuple[str, str], float]) -> Dict[Tuple[str, str], float]:
    """Each pair's score divided by sqrt(partners of a x partners of b): an article related to everything (1227, 1223,
    2043, 1218 in the first run, 28 Oct) stops winning by being common."""
    partners: Dict[str, int] = {}
    for a, _ in scores:
        partners[a] = partners.get(a, 0) + 1
    return {(a, b): s / math.sqrt(partners[a] * partners[b]) for (a, b), s in scores.items()}


def coherence(pool: List[str], scores: Dict[Tuple[str, str], float]) -> Dict[str, float]:
    return {a: sum(scores.get((a, b), 0.0) for b in pool if b != a) for a in pool}


def separation(coh: Dict[str, float], expected: Set[str]) -> Optional[float]:
    """P(an expected article's coherence > an unexpected one's), ties counting half; None without both kinds."""
    pos = [coh[a] for a in coh if a in expected]
    neg = [coh[a] for a in coh if a not in expected]
    if not pos or not neg:
        return None
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def case_report(pool: List[str], plan: List[str], expected: Set[str], scores: Dict[Tuple[str, str], float]) -> dict:
    coh = coherence(pool, scores)
    ranked = sorted(pool, key=lambda a: (-coh[a], pool.index(a)))
    k = max(len([a for a in plan if a in pool]), 5)
    top = set(ranked[:k])
    dropped = [a for a in pool if a in expected and a not in plan]
    return {"pool_cc": len(pool), "expected_in_pool": sorted(expected & set(pool), key=lambda a: int(re.match(r"\d+", a).group(0))),
            "plan_cc": [a for a in plan if a in pool], "separation": separation(coh, expected),
            "dropped_by_plan": dropped, "rescued": [a for a in dropped if a in top],
            "unexpected_in_top": sorted(a for a in top if a not in expected and a not in plan),
            "ranked": [(a, round(coh[a], 4), a in expected) for a in ranked[:k + 3]], "k": k}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dir", required=True, help="the measurement's folder (<case>_A.json)")
    p.add_argument("--relations", required=True)
    p.add_argument("--cases", default="tests/eval_narrative_cases.json")
    p.add_argument("--run", default="A", help="which runs to read (A: today's pipeline)")
    args = p.parse_args()
    rel = json.load(open(args.relations, encoding="utf-8"))
    scores = relation_scores(rel.get("related") or {})
    data = json.load(open(args.cases, encoding="utf-8"))
    items = data if isinstance(data, list) else (data.get("cases") or list(data.values()))
    expected_of = {c["id"]: {str(e["article"]).lower() for e in c.get("expected") or []
                             if isinstance(e, dict) and e.get("code") == "cc" and e.get("article")}
                   for c in items if isinstance(c, dict)}
    expected_of.update({k: set(v) for k, v in EXTRA.items()})
    modes = {"plain": scores, "hubs dampened": dampen_hubs(scores)}
    totals = {}
    for mode, mode_scores in modes.items():
        print(f"\n=== {mode}")
        totals[mode] = run_cases(args, mode_scores, expected_of)
    print("\n=== both")
    for mode, (seps, dropped, rescued, extra_in) in totals.items():
        print(f"{mode:14} separation {sum(seps) / max(len(seps), 1):.2f} (cases above 0.6: {sum(s > 0.6 for s in seps)}, "
              f"below 0.4: {sum(s < 0.4 for s in seps)}); dropped by the plan {dropped}, rescued {rescued}, "
              f"other articles brought in {extra_in}")
    return 0


def run_cases(args, scores, expected_of):
    seps, dropped, rescued, extra_in = [], 0, 0, 0
    print("case       pool c.c.  expected in pool   plan dropped -> rescued by coherence   separation   other articles the top brings in")
    for f in sorted(os.listdir(args.dir)):
        m = re.match(rf"(.+)_{args.run}\.json$", f)
        if not m:
            continue
        case = m.group(1)
        record = json.load(open(os.path.join(args.dir, f), encoding="utf-8"))
        events = record.get("events") or []
        pool = cc_numbers((last_event(events, "case_analysis_retrieval") or {}).get("article_refs"))
        plan = cc_numbers((last_event(events, "case_analysis_plan") or {}).get("norme"))
        expected = expected_of.get(case, set())
        if not expected or not pool:
            continue
        r = case_report(pool, plan, expected, scores)
        if r["separation"] is not None:
            seps.append(r["separation"])
        dropped += len(r["dropped_by_plan"])
        rescued += len(r["rescued"])
        extra_in += len(r["unexpected_in_top"])
        sep = "-" if r["separation"] is None else f"{r['separation']:.2f}"
        print(f"{case:10} {r['pool_cc']:<10} {','.join(r['expected_in_pool']) or '-':18} "
              f"{','.join(r['dropped_by_plan']) or '-':>12} -> {','.join(r['rescued']) or '-':24} {sep:12} "
              f"{len(r['unexpected_in_top'])} ({','.join(r['unexpected_in_top'][:6])})")
    print()
    if seps:
        print(f"separation, mean over {len(seps)} cases: {sum(seps) / len(seps):.2f} (0.50 = no signal, 1.00 = every expected "
              f"article more connected than every other); cases above 0.6: {sum(s > 0.6 for s in seps)}, below 0.4: "
              f"{sum(s < 0.4 for s in seps)}")
    print(f"expected articles the plan dropped: {dropped}; among the most connected: {rescued}; "
          f"other articles that ranking would add: {extra_in}")
    return seps, dropped, rescued, extra_in


if __name__ == "__main__":
    sys.exit(main())
