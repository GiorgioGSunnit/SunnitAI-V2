"""Offline, seconds, no database: FUSION_CLEAN and the vote combined, from exp_distill's vote_results.json.

Step 2a of the vote pool (24 Oct). scripts/exp_distill.py --vote found CITES_VOTE_60 about as good
as FUSION_CLEAN alone but complementary. This tries the ways VOTE_POOL_MODE can join them, with the
live code's own functions (src/rag/vote_pool.combine):

  rrf:1:1, rrf:1:0.5, rrf:2:1   weighted reciprocal rank, FUSION_CLEAN : vote. 2:1 orders exactly as
                                1:0.5 does (only the ratio counts); both are shown to make that visible.
  append:4, append:6, append:8  FUSION_CLEAN's top 16 (the live cap, _FUSION_POOL_CAP), then the first
                                N voted articles not among them, then the rest of FUSION_CLEAN.

Recall and noise @16/@20/@24, per set (tuning, h1-h11, n1-n9), and per variant the expected articles
it gains or loses against FUSION_CLEAN @20, case by case. The best variant is chosen on the TUNING
cases only -- the held-out sets never choose anything, they confirm -- by recall@20, then noise@20,
then recall@16; it is printed as the VOTE_POOL_MODE value to set.

The expected lists are the cases file's current ones (tests/eval_narrative_cases.json), so a
correction is picked up without rerunning exp_distill.

    python scripts/vote_pool_combine.py /root/exp_vote/vote_results.json
    ... --vote cards_vote_60          # the cards' vote instead of the citations'
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.rag.vote_pool import combine, parse_mode  # noqa: E402

DEFAULT_CASES = ROOT / "tests" / "eval_narrative_cases.json"
HEAD = 16   # the live pool's cap (defensive_generation._FUSION_POOL_CAP)
KS = (16, 20, 24)
VARIANTS = ["rrf:1:1", "rrf:1:0.5", "rrf:2:1", "append:4", "append:6", "append:8"]
GROUPS = (("tuning", "TUNING"), ("heldout", "HELD-OUT h1-h11"), ("heldout2", "HELD-OUT 2 n1-n9"))


def group_of(case_id: str, case: dict) -> str:
    if case:
        return case.get("set") or ("heldout" if case.get("heldout") else "tuning")
    return "heldout" if re.fullmatch(r"h\d+", case_id) else "heldout2" if re.fullmatch(r"n\d+", case_id) else "tuning"


def items(expected: list) -> list:
    """[(label, {(code, article)})], one per expected item ("any_of": any one of them)."""
    return [(f"{'/'.join(e.get('any_of') or [e['article']])} {e['code']}",
             {(e["code"], a) for a in (e.get("any_of") or [e["article"]])}) for e in expected]


def found(ranked: list, expected: list, k: int) -> list:
    """The labels of the expected items within ranked[:k] (exp_distill's _recall, by label)."""
    top = set(ranked[:k])
    return [label for label, refs in items(expected) if refs & top]


def noise(ranked: list, expected: list, k: int) -> int:
    """Articles in ranked[:k] that are not expected (exp_distill's _noise)."""
    wanted = set().union(*(refs for _, refs in items(expected))) if expected else set()
    return sum(1 for ref in ranked[:k] if ref not in wanted)


def table(rows: dict, cases: list) -> dict:
    """{method: {k: (found, total, noise)}} over `cases`."""
    out = {}
    for method, per_case in rows.items():
        out[method] = {k: (sum(len(found(per_case[c["id"]], c["expected"], k)) for c in cases),
                           sum(len(c["expected"]) for c in cases),
                           sum(noise(per_case[c["id"]], c["expected"], k) for c in cases)) for k in KS}
    return out


def best(scores: dict) -> str:
    return max((m for m in scores if m in VARIANTS),
               key=lambda m: (scores[m][20][0], -scores[m][20][2], scores[m][16][0], -VARIANTS.index(m)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("results", help="exp_distill --vote's vote_results.json")
    parser.add_argument("--vote", default="cites_vote_60", help="the vote to combine (cites_vote_60, cards_vote_60 ...)")
    parser.add_argument("--cases", default=str(DEFAULT_CASES))
    args = parser.parse_args()

    data = json.loads(Path(args.results).read_text(encoding="utf-8"))
    current = {c["id"]: c for c in json.loads(Path(args.cases).read_text(encoding="utf-8"))["cases"]}
    cases = [{"id": cid, "expected": (current.get(cid) or {}).get("expected") or expected,
              "group": group_of(cid, current.get(cid))} for cid, expected in data["cases"].items()]
    results = data["results"]
    missing = [c["id"] for c in cases if args.vote not in results.get(c["id"], {})]
    if missing:
        sys.exit(f"{args.vote} not in the results for {', '.join(missing)}")

    def refs(cid, key):
        return [tuple(r) for r in results[cid][key]]

    rows = {"fusion_clean": {c["id"]: refs(c["id"], "fusion_clean") for c in cases},
            args.vote: {c["id"]: refs(c["id"], args.vote) for c in cases}}
    for variant in VARIANTS:
        how = parse_mode(variant)
        rows[variant] = {c["id"]: combine(refs(c["id"], "fusion_clean"), refs(c["id"], args.vote), how, HEAD)
                         for c in cases}

    lines = [f"FUSION_CLEAN + {args.vote}, from {args.results}; expected lists from {args.cases}", ""]
    chosen = {}
    for group, title in GROUPS:
        group_cases = [c for c in cases if c["group"] == group]
        if not group_cases:
            continue
        scores = table(rows, group_cases)
        chosen[group] = best(scores)
        lines += ["=" * 86, f"{title}: {len(group_cases)} cases, {scores['fusion_clean'][20][1]} expected articles",
                  "=" * 86, f"{'variant':<16s}" + "".join(f"{'@' + str(k) + ' found / noise':<22s}" for k in KS)]
        for method in rows:
            lines.append(f"{method:<16s}" + "".join(f"{f'{f}/{t}  n{n}':<22s}" for f, t, n in
                                                      (scores[method][k] for k in KS)))
        if [rows["rrf:2:1"][c["id"]] for c in group_cases] == [rows["rrf:1:0.5"][c["id"]] for c in group_cases]:
            lines.append("(rrf:2:1 and rrf:1:0.5 rank every case identically: only the ratio of the weights counts)")
        lines += ["", f"Gained (+) / lost (-) against FUSION_CLEAN @20:"]
        for variant in VARIANTS:
            cells = []
            for c in group_cases:
                base = set(found(rows["fusion_clean"][c["id"]], c["expected"], 20))
                now = set(found(rows[variant][c["id"]], c["expected"], 20))
                if now != base:
                    cells.append(f"{c['id']} " + " ".join([f"+{x}" for x in sorted(now - base)]
                                                          + [f"-{x}" for x in sorted(base - now)]))
            lines.append(f"  {variant:<12s}" + ("; ".join(cells) or "no change"))
        lines.append("")

    pick = chosen.get("tuning")
    if pick:
        lines.append(f"BEST on the tuning cases (recall@20, then noise@20, then recall@16): {pick}")
        lines.append(f"  -> set VOTE_POOL_MODE={pick}")
        others = {g: m for g, m in chosen.items() if g != "tuning" and m != pick}
        if others:
            lines.append("  (the held-out sets alone would pick " + ", ".join(f"{g}: {m}" for g, m in others.items())
                         + " -- they confirm, they do not choose)")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
