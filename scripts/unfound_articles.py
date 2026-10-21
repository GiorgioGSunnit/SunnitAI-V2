"""Expected articles that no method of an exp_distill.py run found in its top k,
per case -- what no search channel reaches, so no ranking change can fix.

Reads the run's results.json only (no DB, no LLM):
    venv/bin/python scripts/unfound_articles.py /root/exp_distill_r13/results.json
    ... --k 20            # the cutoff (default 20)
    ... --methods fusion_clean,hyde_fusion_clean   # only these methods count

For each unfound article it also prints the best rank any method gave it
beyond k ("rank 34 in distill"), or "never retrieved".
"""

import argparse
import json
import sys
from pathlib import Path


def _expected_refs(entry):
    """An expected entry -> its acceptable (code, article) refs ("any_of" counts any one)."""
    if "any_of" in entry:
        return [(entry["code"], a) for a in entry["any_of"]]
    return [(entry["code"], entry["article"])]


def _label(entry):
    articles = "/".join(entry["any_of"]) if "any_of" in entry else entry["article"]
    return f"{articles} {entry['code']}"


def _ranked_methods(result, methods=None):
    """{method: [(code, article), ...]} for every ranked list in a case's result."""
    ranked = {}
    for name, value in result.items():
        if methods and name not in methods:
            continue
        if isinstance(value, list) and all(isinstance(r, list) and len(r) == 2 for r in value):
            ranked[name] = [tuple(r) for r in value]
    return ranked


def unfound(data, k=20, methods=None):
    """{case: [(expected entry, best (rank, method) beyond k or None)]} for the
    expected articles no method has in its top k."""
    report = {}
    for case_id, expected in data["cases"].items():
        ranked = _ranked_methods(data["results"].get(case_id) or {}, methods)
        missing = []
        for entry in expected:
            refs = set(_expected_refs(entry))
            ranks = [(i, m) for m, lst in ranked.items() for i, ref in enumerate(lst, 1) if ref in refs]
            if any(i <= k for i, _ in ranks):
                continue
            missing.append((entry, min(ranks) if ranks else None))
        report[case_id] = missing
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("results", help="an exp_distill.py results.json")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--methods", help="comma-separated: only these methods count")
    args = parser.parse_args(argv)
    data = json.loads(Path(args.results).read_text(encoding="utf-8"))
    methods = {m.strip() for m in args.methods.split(",")} if args.methods else None
    report = unfound(data, args.k, methods)
    total = sum(len(e) for e in data["cases"].values())
    missed = sum(len(m) for m in report.values())
    print(f"Expected articles no method found in the top {args.k}: {missed}/{total}\n")
    for case_id, missing in report.items():
        if not missing:
            print(f"{case_id}: none")
            continue
        print(f"{case_id}:")
        for entry, best in missing:
            where = f"best rank {best[0]} in {best[1]}" if best else "never retrieved"
            print(f"  - {_label(entry)}  ({where})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
