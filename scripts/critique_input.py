#!/usr/bin/env python3
"""v70 (4): a lawyer's critique (docs/lawyer/<name>.md) into the --critique input of scripts/rubric_score.py -- the
yardstick, never a test fixture. His text is only cut: each non-empty line of a numbered section becomes a point with
a fixed id («1.1», «1.2» …) and his words verbatim (the file's word joiners out), so the same point is scored in every
batch and --baseline can say which one came back; «Giudizio complessivo» is kept whole. Each section's category, from
its heading: tesi, categoria (an error of category), coerenza, processuale, mancanza (what the parere lacks), pregio
(a merit to keep: «mantenuto» / «perso»), attesa (the strategy he would advise: «adottato» / «assente»). The case's
text comes with it (tests/eval_narrative_cases.json, by --case-id), so the scorer needs nothing else.

v71: the expected thesis (docs/lawyer/critiques/<id>.expected.json, when present, or --expected) goes in as
«tesi_attesa»: the scorer checks the soluzioni table against it for the points it names (his §1 and §7.2).

  python scripts/critique_input.py docs/lawyer/critica_penale_v2.md --case-id sempronio_no_hint
      -> docs/lawyer/critiques/sempronio_no_hint.critique.json
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "lawyer" / "critiques"
_HEADING_RE = re.compile(r"^(\d+)\.\s*(.+)$")
# a heading's category -- the first rule its words match
CATEGORY_RULES = (
    (r"\bpregi\b", "pregio"),
    (r"strategia che|consiglier", "attesa"),
    (r"contraddizion|incongruen", "coerenza"),
    (r"processual", "processuale"),
    (r"\bmanca\b|cosa manca", "mancanza"),
    (r"\btesi\b", "tesi"),
    (r"errore|errori", "categoria"),
)


def clean(line: str) -> str:
    return " ".join(line.replace("⁠", "").split())


def category(heading: str) -> str:
    low = heading.lower()
    return next((cat for rx, cat in CATEGORY_RULES if re.search(rx, low)), "altro")


def parse(text: str) -> Dict[str, Any]:
    giudizio, sections, current = "", [], None
    for raw in text.splitlines():
        line = clean(raw)
        if not line:
            continue
        m = _HEADING_RE.match(line)
        if m:
            current = {"n": int(m.group(1)), "heading": m.group(2).strip(), "category": category(m.group(2)),
                       "points": []}
            sections.append(current)
        elif current is None:
            giudizio = f"{giudizio} {line}".strip()
        else:
            current["points"].append(line)
    points: List[Dict[str, Any]] = []
    for sec in sections:
        for k, line in enumerate(sec["points"], 1):
            points.append({"id": f"{sec['n']}.{k}", "sezione": sec["heading"], "categoria": sec["category"],
                           "punto": line})
    return {"giudizio": giudizio, "points": points}


def case_text(case_id: str) -> str:
    data = json.loads((ROOT / "tests" / "eval_narrative_cases.json").read_text(encoding="utf-8"))
    return next((str(c.get("question") or "") for c in data.get("cases") or [] if c.get("id") == case_id), "")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("critique", type=Path)
    ap.add_argument("--case-id", required=True, help="the case the critique is about (its id in the eval cases)")
    ap.add_argument("--case", type=Path, help="the case text, when the id is not an eval case")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--expected", type=Path, help="the expected thesis (default: <out-dir>/<case-id>.expected.json)")
    args = ap.parse_args(argv)
    parsed = parse(args.critique.read_text(encoding="utf-8"))
    case = args.case.read_text(encoding="utf-8") if args.case else case_text(args.case_id)
    if not case:
        print(f"no case text for {args.case_id}: pass --case", file=sys.stderr)
        return 2
    expected_path = args.expected or args.out_dir / f"{args.case_id}.expected.json"
    expected = json.loads(expected_path.read_text(encoding="utf-8")) if expected_path.is_file() else None
    if expected:
        unknown = [x for x in expected.get("punti") or {} if x not in {p["id"] for p in parsed["points"]}]
        if unknown:
            print(f"expected thesis: unknown point ids {unknown}", file=sys.stderr)
            return 2
    out = {"id": args.case_id, "source": str(args.critique.relative_to(ROOT)) if args.critique.is_absolute() and
           ROOT in args.critique.parents else str(args.critique), "case": case, **parsed,
           **({"tesi_attesa": {k: v for k, v in expected.items() if not k.startswith("_")}} if expected else {})}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"{args.case_id}.critique.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    counts: Dict[str, int] = {}
    for p in parsed["points"]:
        counts[p["categoria"]] = counts.get(p["categoria"], 0) + 1
    print(f"wrote {path}: {len(parsed['points'])} points {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
