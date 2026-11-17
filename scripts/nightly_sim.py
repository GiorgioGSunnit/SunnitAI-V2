#!/usr/bin/env python3
"""v70: the nightly -- a five-cell sim with structural checks, not one score. The lesson of the v69 batch: the rubric
moved +0.8 while two sections went to zero; a mean hides a section that vanished. Each night, per posture cell
(tests/data/nightly_cells.json: civile/attore/pre_filing, civile/convenuto/primo_grado, penale/imputato/primo_grado,
penale/persona_offesa/pre_filing, lavoro/attore/pre_filing):

  1. intake: the cell read from the case as the API reads it (intake.request_intake, logged; v73: a cell without the
     intake line fails) -- a wrong rito, role or stage fails the cell
  2. the parere (interactive_flow.request_case_first: the flow with that intake, as the API runs it), timed: the thesis selection's and the writer's
     seconds (LLM_THINKING=per_call)
  3. its structure (lawyer_method.structure_checks): his ten sections present and non-empty, §4's tables, §6's table
     with the stragiudiziale option, three objections, §9's esito / grado / obiettivo, no «da verificare» outside §10;
     by the cell: the claimant's pre-filing mediazione / ATP / 700, the imputato's riti alternativi, the ratione
     temporis line
  4. once: the stage lists against v59 for the default cell (scripts/stage_diff.py)

The report (--out, markdown) has one row per cell and one column per check -- a check that passed yesterday and fails
today is in the «Regressions» list (--baseline: yesterday's --json-out). Exit 1 on any failure.

v77: the acceptance is this table for structure and regressions, and the lawyer's scores on the held-out set for
substance (scripts/heldout_export.py, scripts/heldout_score.py) -- the 31-case sim (diagnose_long --interactive-sim) is no
measure any more. Each run records the VERSION it ran (the report's first line, the JSON's "version"); a live cell's
answer must end with that version's footer (check version_footer); the report ends with the lawyer's latest mean per
version (the held-out folder's scores.jsonl: --heldout-dir, else HELDOUT_DIR, else /opt/chatbot/heldout).

v78 (RULINGS_RECORD on): every penal cell, and every civil cell whose charge is a c.c. fattispecie of the record, has at
least one ruling printed with its quoted passage in §5 (check ruling_passage; a cell with none fails); the report adds
the rulings record per cell (per article found / linked / kept, estremi parsed / unparsed, passages quoted / none,
orientations computed / to verify) and the penal cells' §5 verbatim.

v79 (A2, A3): per cell, the thesis selection's counts (candidates offered / parsed / valued / barred / unverifiable /
minoritaria, the main line chosen, whether it is the fallback) in their own table, compared with the baseline's;
--deterministic runs the thesis call once at temperature 0 (THESIS_DETERMINISTIC) for a version-to-version comparison
-- the default stays best-of-3.

  python scripts/nightly_sim.py --env .env --out nightly.md --json-out nightly.json --baseline last.json
  python scripts/nightly_sim.py --env .env --deterministic --out det.md --json-out det.json --baseline det.prev.json
  python scripts/nightly_sim.py --answers DIR      # offline: DIR/<cell id>.md, the checks only (no model, no DB)

Needs the server's environment (Neo4j, the model, LAWYER_METHOD=true, CASE_FIRST_SINGLE_SHOT=true, ...) except with
--answers. Proposed schedule (prod, UTC; outside the cards' GPU window, 21:00-22:30):
  15 2 * * * cd /opt/chatbot && cp -f /var/log/maint/nightly.json /var/log/maint/nightly.prev.json; venv/bin/python scripts/nightly_sim.py --env .env --out /var/log/maint/nightly.md \
      --json-out /var/log/maint/nightly.json --baseline /var/log/maint/nightly.prev.json >> /var/log/maint/nightly.log 2>&1
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CELLS_FILE = ROOT / "tests" / "data" / "nightly_cells.json"
CELL_KEYS = ("rito", "role", "stage")


def cells(path: Path = CELLS_FILE) -> List[Dict[str, Any]]:
    return [c for c in json.loads(path.read_text(encoding="utf-8")).get("cells") or [] if c.get("id")]


def cell_check(detected: Dict[str, Any], expected: Dict[str, Any]) -> Dict[str, Any]:
    wrong = {k: detected.get(k) for k in CELL_KEYS if expected.get(k) and detected.get(k) != expected[k]}
    return {"check": "intake_cell", "ok": not wrong, "why": "" if not wrong else f"read {wrong}"}


def intake_lines():
    """v73: the [VERBOSE:intake] lines logged while the block runs (intake.vlog wrapped, then restored)."""
    from contextlib import contextmanager
    from src.rag import intake

    @contextmanager
    def capture():
        seen: List[Dict[str, Any]] = []
        original = intake.vlog

        def vlog(event, data=None, duration_ms=None, **kwargs):
            if event == "intake":
                seen.append(data or {})
            return original(event, data, duration_ms, **kwargs)
        intake.vlog = vlog
        try:
            yield seen
        finally:
            intake.vlog = original
    return capture()


def intake_line_check(seen: List[Dict[str, Any]], state: Dict[str, Any], route: str) -> Dict[str, Any]:
    """v73: the cell ran as the API runs it -- the intake line logged once, before the flow, the flow given it (not
    reading its own), the API's route the case-first one. A cell without it fails."""
    from src.rag import intake
    why = ("no [VERBOSE:intake] line" if not seen else
           f"the flow read the intake itself ({state.get('intake_source')!r})" if state.get("intake_source") != "request"
           else f"the API would route it {route!r}, not case-first" if route != intake.CASE_FIRST else "")
    return {"check": "intake_line", "ok": not why, "why": why}


def run_cell(spec: Dict[str, Any]) -> Dict[str, Any]:
    """The cell live: intake, the parere, the checks; the error, when a step fails, as a failed check. v73: the intake
    and the flow as the API runs them (interactive_flow.request_case_first: intake.request_intake, then the flow)."""
    from src.rag import interactive_flow, lawyer_method
    out: Dict[str, Any] = {"id": spec["id"], "cell": spec["cell"], "checks": []}
    t0 = time.time()
    try:
        with intake_lines() as seen:
            result = interactive_flow.request_case_first(spec["text"], "it", case_id=spec["id"])
        detected = result["intake"]
        out["intake"] = {k: detected.get(k) for k in CELL_KEYS}
        print(f"{spec['id']}: intake {detected}", file=sys.stderr)
        out["checks"].append(intake_line_check(seen, result.get("state") or {}, result.get("api_route") or ""))
        out["checks"].append(cell_check(detected, spec["cell"]))
        out["answer"] = result.get("answer") or ""
        out["thesis_counts"] = (result.get("state") or {}).get("thesis_counts") or {}      # v79 (A2)
        timings = (result.get("state") or {}).get("timings") or {}
        out["timings"] = {"total_s": round(time.time() - t0, 1),
                          **{k: timings.get(k) for k in ("thesis_selection_s", "writer_s", "answer_s")}}
        out["checks"] += lawyer_method.structure_checks(out["answer"], spec["cell"])
        out["checks"].append(footer_check(out["answer"]))
        from src.rag import rulings
        if rulings.enabled():
            record = (result.get("state") or {}).get("rulings_record")
            out["rulings"] = (record or {}).get("log") or {}
            out["rulings"].update({k: (record or {}).get(k) for k in ("charge", "charge_cc_fattispecie")})
            check = ruling_check(out["answer"], spec["cell"], record)
            if check:
                out["checks"].append(check)
    except Exception as exc:          # noqa: BLE001 -- a failed cell is a failed check, the others still run
        out["checks"].append({"check": "run", "ok": False, "why": f"{type(exc).__name__}: {exc}"[:300]})
    return out


def footer_check(answer: str) -> Dict[str, Any]:
    """v77: the answer ends with «SunnitAI <version>», this checkout's VERSION."""
    from src import version
    ok = version.has_footer(answer, version.current())
    last = (answer or "").rstrip().rsplit("\n", 1)[-1].strip()[:80]
    return {"check": "version_footer", "ok": ok, "why": "" if ok else f"last line «{last}», not «{version.footer()}»"}


def section5(answer: str) -> str:
    from src.rag import lawyer_method
    return dict(lawyer_method.answer_sections(answer)).get(lawyer_method.title("orientamenti")) or ""


def ruling_check(answer: str, cell: Dict[str, Any], record: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """v78: a penal cell, or a civil cell charged with a c.c. fattispecie, needs a ruling quoted in §5; None when the
    cell is neither (no check)."""
    from src.rag import rulings
    penal = cell.get("family") == "penale"
    if not penal and not (record or {}).get("charge_cc_fattispecie"):
        return None
    n = rulings.quoted(section5(answer))
    why = "" if n else ("no rulings record" if record is None else
                        f"§5 quotes no passage (kept {len(record.get('kept') or [])}, charge {record.get('charge')})")
    return {"check": "ruling_passage", "ok": bool(n), "why": why}


def rulings_lines(results: List[Dict[str, Any]]) -> List[str]:
    """v78: per cell, the rulings record's counts; then the penal cells' §5 verbatim."""
    rows = [c for c in results if c.get("rulings") is not None]
    if not rows:
        return []
    lines = ["", "## Rulings in the record (per cell, per article)", "",
             "| cell | article | found | linked | graded | kept |", "|---|---|---|---|---|---|"]
    for c in rows:
        for label, a in (c["rulings"].get("articles") or {}).items():
            lines.append(f"| {c['id']} | {label} | {a.get('found', 0)} | {a.get('linked', 0)} | {a.get('graded', 0)} | "
                         f"{a.get('kept', 0)} |")
    lines += ["", "| cell | estremi parsed | unparsed | section unmapped | passages quoted | none | orientations "
                  "computed | to verify |", "|---|---|---|---|---|---|---|---|"]
    for c in rows:
        r = c["rulings"]
        e, p, o = r.get("estremi") or {}, r.get("passages") or {}, r.get("orientation") or {}
        lines.append(f"| {c['id']} | {e.get('parsed', 0)} | {e.get('unparsed', 0)} | {e.get('section_unmapped', 0)} | "
                     f"{p.get('quoted', 0)} | {p.get('none', 0)} | {o.get('computed', 0)} | {o.get('da_verificare', 0)} |")
    for c in results:
        if (c.get("cell") or {}).get("family") == "penale" and c.get("answer"):
            lines += ["", f"### §5 of {c['id']}", "", section5(c["answer"]) or "(no §5)"]
    return lines


COUNT_COLUMNS = ("offered", "parsed", "valued", "barred", "unverifiable", "minoritaria", "chosen", "main_is_fallback")


def counts_lines(results: List[Dict[str, Any]], baseline: Optional[Dict[str, Any]] = None) -> List[str]:
    """v79 (A2): per cell, the thesis selection's counts -- «before → now» where the baseline's differ."""
    rows = [c for c in results if c.get("thesis_counts")]
    if not rows:
        return []
    before = {c["id"]: c.get("thesis_counts") or {} for c in (baseline or {}).get("cells") or []}

    def cell(cid: str, key: str, now: Any) -> str:
        was = before.get(cid, {}).get(key)
        now_s = ("yes" if now else "no") if isinstance(now, bool) else str(now if now is not None else "—")
        if cid in before and key in before[cid] and was != now:
            was_s = ("yes" if was else "no") if isinstance(was, bool) else str(was)
            return f"{was_s} → {now_s}"
        return now_s
    lines = ["", "## Thesis selection (per cell)", "", "| cell | " + " | ".join(COUNT_COLUMNS) + " |",
             "|---" * (len(COUNT_COLUMNS) + 1) + "|"]
    for c in rows:
        t = c["thesis_counts"]
        lines.append(f"| {c['id']} | " + " | ".join(cell(c["id"], k, t.get(k)) for k in COUNT_COLUMNS) + " |")
    return lines


def lawyer_lines(folder: Optional[Path]) -> List[str]:
    """v77: the lawyer's latest mean per version on the held-out set (src/chatbot/heldout.py)."""
    from src.chatbot import heldout
    folder = folder or heldout.folder()
    table = heldout.means_table(heldout.scores(folder)) if folder and Path(folder).is_dir() else []
    return ["", "## The lawyer's scores per version (held-out set, 0-5)", ""] + (
        table or [f"- none recorded ({folder or 'no held-out folder'})"])


def offline_cell(spec: Dict[str, Any], folder: Path) -> Dict[str, Any]:
    from src.rag import lawyer_method
    path = folder / f"{spec['id']}.md"
    out: Dict[str, Any] = {"id": spec["id"], "cell": spec["cell"], "checks": []}
    if not path.is_file():
        out["checks"].append({"check": "run", "ok": False, "why": f"no {path.name}"})
        return out
    out["answer"] = path.read_text(encoding="utf-8")
    out["checks"] = lawyer_method.structure_checks(out["answer"], spec["cell"])
    from src.rag import rulings
    check = ruling_check(out["answer"], spec["cell"], None) if rulings.enabled() else None
    if check:           # v78: offline, the penal cells only (the record is not in a file)
        out["checks"].append(check)
    return out


def stage_check() -> Dict[str, Any]:
    """The stage lists against v59 (stage_diff.py, a git archive of the v59 commit): skipped -- not failed -- outside a
    git checkout (the server's copy): run it where the repository is."""
    if not (ROOT / ".git").exists():
        return {"ok": None, "report": "skipped: not a git checkout (run scripts/stage_diff.py in the repository)"}
    done = subprocess.run([sys.executable, str(ROOT / "scripts" / "stage_diff.py")], capture_output=True, text=True,
                          cwd=str(ROOT), timeout=900)
    return {"ok": done.returncode == 0, "report": done.stdout.strip() or done.stderr.strip()[-2000:]}


def regressions(results: List[Dict[str, Any]], baseline: Optional[Dict[str, Any]]) -> List[str]:
    before = {c["id"]: {k["check"]: k["ok"] for k in c.get("checks") or []} for c in (baseline or {}).get("cells") or []}
    out = []
    for c in results:
        for k in c["checks"]:
            if not k["ok"] and before.get(c["id"], {}).get(k["check"]):
                out.append(f"{c['id']}: {k['check']} ({k['why']})")
    return out


def report(results: List[Dict[str, Any]], stages: Optional[Dict[str, Any]], regressed: List[str], run_version: str = "",
           lawyer: Optional[List[str]] = None, baseline: Optional[Dict[str, Any]] = None, mode: str = "") -> str:
    names = list(dict.fromkeys(k["check"] for c in results for k in c["checks"]))
    lines = ["# Nightly: five cells, structural checks", "", f"Version: {run_version or '—'}"
             + (f" (baseline {baseline.get('version') or '—'})" if baseline else "") + (f"; {mode}" if mode else ""), "",
             "| cell | " + " | ".join(names) + " | thesis s | writer s |", "|---" * (len(names) + 3) + "|"]
    for c in results:
        by = {k["check"]: k["ok"] for k in c["checks"]}
        t = c.get("timings") or {}
        lines.append(f"| {c['id']} | " + " | ".join("✓" if by.get(n) else ("✗" if n in by else "·") for n in names)
                     + f" | {t.get('thesis_selection_s') or '—'} | {t.get('writer_s') or '—'} |")
    lines += ["", "## Failures", ""]
    fails = [f"- {c['id']}: {k['check']} — {k['why']}" for c in results for k in c["checks"] if not k["ok"]]
    lines += fails or ["- none"]
    lines += ["", "## Regressions (passed in the baseline, fail now)", ""] + ([f"- {x}" for x in regressed] or ["- none"])
    if stages is not None:
        lines += ["", "## Stage lists against v59 (default cell)", "", stages["report"]]
    lines += counts_lines(results, baseline)
    lines += rulings_lines(results)
    lines += lawyer or []
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--env", default=".env")
    ap.add_argument("--answers", type=Path, help="offline: a folder of <cell id>.md answers, the checks only")
    ap.add_argument("--only", nargs="*", help="cell ids")
    ap.add_argument("--out")
    ap.add_argument("--json-out")
    ap.add_argument("--baseline")
    ap.add_argument("--no-stages", action="store_true", help="skip the stage lists' diff against v59")
    ap.add_argument("--heldout-dir", type=Path, help="the held-out folder whose scores the report ends with")
    ap.add_argument("--reasoning-effort", help="v78.1.1: LLM_REASONING_EFFORT for this run (the calls with thinking on)")
    ap.add_argument("--deterministic", action="store_true",
                    help="v79: the thesis call once at temperature 0 (THESIS_DETERMINISTIC), for version-to-version runs")
    args = ap.parse_args(argv)
    if not args.answers and os.path.isfile(args.env):
        from dotenv import load_dotenv
        load_dotenv(args.env)
    if args.reasoning_effort is not None:       # v78.1.1: over the env file's, for this run
        os.environ["LLM_REASONING_EFFORT"] = args.reasoning_effort
    if args.deterministic:                      # v79 (A3): over the env file's, for this run
        os.environ["THESIS_DETERMINISTIC"] = "1"
    specs = [c for c in cells() if not args.only or c["id"] in args.only]
    results = [offline_cell(s, args.answers) if args.answers else run_cell(s) for s in specs]
    stages = None if args.no_stages else stage_check()
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8")) if args.baseline and \
        os.path.isfile(args.baseline) else None
    regressed = regressions(results, baseline)
    from src import version
    run_version = version.current()
    print(f"nightly: version {run_version}", file=sys.stderr)
    mode = "deterministic (one thesis sample, temperature 0)" if os.getenv("THESIS_DETERMINISTIC") else "best-of-3"
    text = report(results, stages, regressed, run_version, lawyer_lines(args.heldout_dir), baseline, mode)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps({"version": run_version, "mode": mode, "cells": results,
                                                   "stages_ok": None if stages is None else stages["ok"]},
                                                  ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(text)
    failed = any(not k["ok"] for c in results for k in c["checks"]) or (stages is not None and stages["ok"] is False)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
