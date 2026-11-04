"""Read-only: the deterministic part of a dry run saved by scripts/dryrun_strategy.py --save, to compare two runs.

From run.json's "material" (written since 3 Nov): the article rows as the draft prompt is built from them and the
procedural tools' rows (regime_rows), each as document id, article, Section, the text chosen and the commi split. Which
articles reach the prompt depends on the model's calls upstream (the analysis, the choice, the applicability check); the
text and the commi of an article are a function of the corpus alone. So:

    python scripts/run_material.py run.json              one JSON line per row, in a stable order (diff two outputs)
    python scripts/run_material.py before.json after.json   the rows in one run only, and for the rows in both,
                                                         those whose text, commi or Section differ

Exit status 1 when two runs differ on a row both hold, else 0.
"""

import argparse
import hashlib
import json
import sys


def _key(row: dict) -> tuple:
    return (str(row.get("doc") or ""), tuple(row.get("article") or ()), str(row.get("section") or ""))


def rows_of(path: str) -> dict:
    """{"article_rows": {key: row}, "regime_rows": {key: row}} of a saved run."""
    with open(path, encoding="utf-8") as fh:
        material = json.load(fh).get("material")
    if not material:
        sys.exit(f"{path}: no \"material\" (saved by dryrun_strategy.py --save before 3 Nov, or the run failed early)")
    out = {}
    for part in ("article_rows", "regime_rows"):
        rows = {}
        for row in material.get(part) or []:
            rows.setdefault(_key(row), row)     # a row twice (two channels) counts once, as the prompt reads it once
        out[part] = dict(sorted(rows.items()))
    return out


def _digest(text) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()[:12]


def dump(path: str) -> None:
    for part, rows in rows_of(path).items():
        for key, row in rows.items():
            print(json.dumps({"part": part, "doc": key[0], "article": list(key[1]), "section": key[2],
                              "text": row.get("text"), "commi": row.get("commi"), "commi_from": row.get("commi_from")},
                             ensure_ascii=False, sort_keys=True))


def compare(a: str, b: str) -> int:
    left, right = rows_of(a), rows_of(b)
    differ = 0
    for part in ("article_rows", "regime_rows"):
        ra, rb = left[part], right[part]
        both = [k for k in ra if k in rb]
        print(f"## {part}: {len(ra)} in {a}, {len(rb)} in {b}, {len(both)} in both")
        for label, keys in (("only in " + a, [k for k in ra if k not in rb]), ("only in " + b, [k for k in rb if k not in ra])):
            if keys:
                print(f"{label}: " + ", ".join(f"{' '.join(k[1])} [{k[2]}]" for k in keys))
        for k in both:
            fields = [f for f in ("text", "commi", "commi_from") if ra[k].get(f) != rb[k].get(f)]
            if fields:
                differ += 1
                detail = "; ".join(f"{f}: {_digest(ra[k].get(f))} -> {_digest(rb[k].get(f))}" for f in fields)
                print(f"DIFFERS {' '.join(k[1])} [{k[2]}] {k[0]}: {detail}")
        print(f"rows in both that differ: {sum(1 for k in both if any(ra[k].get(f) != rb[k].get(f) for f in ('text', 'commi', 'commi_from')))}\n")
    return 1 if differ else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="one run.json to print, or two to compare")
    args = parser.parse_args()
    if len(args.runs) == 1:
        dump(args.runs[0])
    elif len(args.runs) == 2:
        sys.exit(compare(*args.runs))
    else:
        parser.error("one or two run.json files")


if __name__ == "__main__":
    main()
