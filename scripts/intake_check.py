#!/usr/bin/env python3
"""v61: the intake alone (src/rag/intake.py) -- no retrieval, no answer -- over

  (a) the 31 sim cases: scripts/diagnose_long.py's TUNING_IDS + HELDOUT_IDS, from tests/eval_narrative_cases.json.
      The sim runs each through the case-first single shot, and today's API routes each there (the `today` column: its
      narrative check or its defence shortcut; «llm» where today's classifier decided). Expected: the default cell,
      deliverable parere and route case_first. Rito, role and stage are shown, not scored: when the rules say nothing,
      the client logic and the area frame decide them downstream, as before.
  (b) the grid, tests/data/intake_grid.yaml: synthetic one-paragraph requests, every value under a row's `expected`
      scored; a row passes when all match.

Prints, per set, a table prompt -> detected -> expected -> ✓/✗ and a totals row; then which fields the model filled.

  python scripts/intake_check.py                  # the rules, then the model for the fields they leave empty
  python scripts/intake_check.py --rules-only     # the rules and the defaults alone
  python scripts/intake_check.py --only grid      # one set (sim | grid)
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GRID = ROOT / "tests" / "data" / "intake_grid.yaml"
CASES = ROOT / "tests" / "eval_narrative_cases.json"
SIM_EXPECTED = {"deliverable": "parere", "route": "case_first"}
PROMPT_CHARS = 70


def sim_ids():
    """The sim's 31 case ids (diagnose_long's own lists)."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from diagnose_long import HELDOUT_IDS, TUNING_IDS
    return list(TUNING_IDS) + list(HELDOUT_IDS)


def today_route(message: str) -> str:
    """Today's deterministic routing of a request without a document: the narrative check (interactive_flow.routes_here)
    or the defence shortcut (calculation._DEFENCE_REQUEST_RE, not a short topic) -> case_first; else the LLM decided."""
    from src.rag import interactive_flow, intake
    from src.rag.calculation import _DEFENCE_REQUEST_RE
    if interactive_flow.routes_here(message):
        return "case_first (narrativa)"
    if _DEFENCE_REQUEST_RE.search(message) and not intake.topic_question(message):
        return "case_first (difesa)"
    return "llm"


def _short(text: str) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= PROMPT_CHARS else text[:PROMPT_CHARS - 1] + "…"


def _value(result: dict, key: str) -> str:
    if key == "blocks":
        return ",".join(b["kind"] for b in result.get("requested_blocks") or [])
    return str(result.get(key) or "")


def _detected(result: dict, keys) -> str:
    conf = result.get("confidence") or {}
    shown = []
    for key in keys:
        value = _value(result, key)
        how = conf.get(key)
        shown.append(f"{key}={value}" + (f" ({how})" if how and how != "rule" else ""))
    return "; ".join(shown)


def run_sim(intake, model: bool) -> tuple:
    cases = {c["id"]: c for c in json.load(open(CASES, encoding="utf-8"))["cases"]}
    rows, passed = [], 0
    keys = ("deliverable", "route", "rito", "role", "stage", "blocks")
    for cid in sim_ids():
        text = cases[cid]["question"]
        result = intake.detect(text, model=model)
        ok = all(_value(result, k) == v for k, v in SIM_EXPECTED.items())
        passed += ok
        rows.append((cid, _short(text), _detected(result, keys), today_route(text),
                     "parere / case_first", "✓" if ok else "✗", result))
    print("## (a) the 31 sim cases -- the default cell (deliverable parere, route case_first)\n")
    print("| id | prompt | detected | today | expected | |")
    print("|---|---|---|---|---|---|")
    for cid, prompt, detected, today, expected, mark, _ in rows:
        print(f"| {cid} | {prompt} | {detected} | {today} | {expected} | {mark} |")
    print(f"| **total** | | | | | **{passed}/{len(rows)}** |\n")
    return passed, len(rows), [r[-1] for r in rows]


def run_grid(intake, model: bool) -> tuple:
    import yaml
    grid = yaml.safe_load(open(GRID, encoding="utf-8"))
    rows, passed = [], 0
    for item in grid:
        expected = item["expected"]
        upload = item.get("upload") or ""
        result = intake.detect(item["prompt"], upload, model=model, refers_to_upload=bool(upload))
        wrong = [k for k, v in expected.items() if _value(result, k) != str(v)]
        passed += not wrong
        detected = _detected(result, list(expected))
        if wrong:
            detected += "  ✗ " + ", ".join(wrong)
        rows.append((item["id"], _short(item["prompt"]), detected,
                     "; ".join(f"{k}={v}" for k, v in expected.items()), "✓" if not wrong else "✗", result))
    print(f"## (b) the grid -- {GRID.relative_to(ROOT)}\n")
    print("| id | prompt | detected | expected | |")
    print("|---|---|---|---|---|")
    for rid, prompt, detected, expected, mark, _ in rows:
        print(f"| {rid} | {prompt} | {detected} | {expected} | {mark} |")
    print(f"| **total** | | | | **{passed}/{len(rows)}** |\n")
    return passed, len(rows), [r[-1] for r in rows]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--rules-only", action="store_true", help="no model call: the rules and the defaults alone")
    parser.add_argument("--only", choices=("sim", "grid"), help="one set")
    parser.add_argument("--env", default=".env", help="the environment file (the model's settings)")
    args = parser.parse_args(argv)
    if os.path.isfile(args.env):
        from dotenv import load_dotenv
        load_dotenv(args.env)
    from src.rag import intake
    model = not args.rules_only
    print(f"# intake check -- {'rules only' if args.rules_only else 'rules + model'}\n")
    totals, results = [], []
    if args.only in (None, "sim"):
        p, n, r = run_sim(intake, model)
        totals.append(f"sim {p}/{n}")
        results += r
    if args.only in (None, "grid"):
        p, n, r = run_grid(intake, model)
        totals.append(f"grid {p}/{n}")
        results += r
    filled = {}
    for r in results:
        for field, how in (r.get("confidence") or {}).items():
            if how == intake.MODEL:
                filled[field] = filled.get(field, 0) + 1
    print(f"**{' · '.join(totals)}** -- fields the model filled: "
          f"{', '.join(f'{k} {v}' for k, v in sorted(filled.items())) or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
