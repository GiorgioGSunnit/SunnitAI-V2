#!/usr/bin/env python3
"""v77: the lawyer's filled pack (scripts/heldout_export.py) read back -- per case his seven section scores (0-5), his
notes and his overall judgement, recorded with the answer's version in <folder>/scores.jsonl (one line per case; a
case scored again: the later line counts). Then the table per version: cases scored, the mean of the case means, each
section's mean. This is the substance measure; the nightly (scripts/nightly_sim.py) appends the same table.

  python scripts/heldout_score.py pack_2026-11-17_filled.md          # record it, print the table
  python scripts/heldout_score.py pack.md --dry-run                  # parse and print, record nothing
  python scripts/heldout_score.py --summary                          # the table of what is recorded

A block with no voto at all is reported and not recorded (not yet scored); a voto out of 0-5 or not a number stops the
run before anything is written.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.chatbot import heldout  # noqa: E402
from heldout_export import SCORE_HEADING  # noqa: E402

_MARK_RE = re.compile(r"<!--\s*heldout:(\S+)\s+version:(\S+)\s*-->")
_SECTION_RE = re.compile(r"^\s*(\d)\.\s+\S")
_VOTO_RE = re.compile(r"^\s*-?\s*\**voto\**\s*:\s*(.*?)\s*$", re.I)
_NOTE_RE = re.compile(r"^\s*-?\s*\**note\**\s*:\s*(.*?)\s*$", re.I)
_GIUDIZIO_RE = re.compile(r"^\s*\**giudizio complessivo\**\s*:\s*(.*?)\s*$", re.I)


def parse(text: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """([{id, version, sections: {key: score or None}, notes: {key: text}, giudizio}], [errors])."""
    keys = [k for k, _ in heldout.SECTIONS]
    marks = list(_MARK_RE.finditer(text))
    cases, errors = [], []
    for i, m in enumerate(marks):
        chunk = text[m.end(): marks[i + 1].start() if i + 1 < len(marks) else len(text)]
        at = chunk.rfind(SCORE_HEADING)
        if at < 0:
            errors.append(f"{m.group(1)}: no «{SCORE_HEADING}» block")
            continue
        sections: Dict[str, Any] = {k: None for k in keys}
        notes: Dict[str, str] = {}
        giudizio, current, last = "", None, None
        for line in chunk[at + len(SCORE_HEADING):].split("\n"):
            sec, voto, note, giu = (_SECTION_RE.match(line), _VOTO_RE.match(line), _NOTE_RE.match(line),
                                    _GIUDIZIO_RE.match(line))
            if sec and 1 <= int(sec.group(1)) <= len(keys) and not voto and not note:
                current, last = keys[int(sec.group(1)) - 1], None
            elif voto and current:
                raw = voto.group(1).strip().strip("*").replace(",", ".")
                if raw:
                    try:
                        value = float(raw)
                    except ValueError:
                        errors.append(f"{m.group(1)} {current}: voto «{raw}» is not a number")
                        continue
                    lo, hi = heldout.SCALE
                    if not lo <= value <= hi:
                        errors.append(f"{m.group(1)} {current}: voto {raw} outside {lo}-{hi}")
                        continue
                    sections[current] = int(value) if value == int(value) else value
                last = None
            elif note and current:
                notes[current] = note.group(1).strip()
                last = ("note", current)
            elif giu:
                giudizio, last = giu.group(1).strip(), ("giudizio", None)
            elif line.strip() and last:     # a note or the judgement going on over more lines
                if last[0] == "note":
                    notes[last[1]] = (notes.get(last[1], "") + " " + line.strip()).strip()
                else:
                    giudizio = (giudizio + " " + line.strip()).strip()
        cases.append({"id": m.group(1), "version": m.group(2), "sections": sections,
                      "notes": {k: v for k, v in notes.items() if v}, "giudizio": giudizio})
    return cases, errors


def rows(cases: List[Dict[str, Any]], pack: str, versions: Dict[str, str]) -> List[Dict[str, Any]]:
    """The scores.jsonl lines: the cases with at least one voto; the version the held-out record's, else the pack's."""
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out = []
    for c in cases:
        mean = heldout.case_mean(c["sections"])
        if mean is None:
            continue
        out.append({"id": c["id"], "version": versions.get(c["id"]) or c["version"], "sections": c["sections"],
                    "mean": mean, "notes": c["notes"], "giudizio": c["giudizio"], "pack": pack, "scored_at": now})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("pack", nargs="?", help="the filled pack (Markdown)")
    ap.add_argument("--dir", type=Path, help="the held-out folder (default: HELDOUT_DIR, else /opt/chatbot/heldout)")
    ap.add_argument("--dry-run", action="store_true", help="parse and print, record nothing")
    ap.add_argument("--summary", action="store_true", help="no pack: the table of the recorded scores")
    args = ap.parse_args(argv)
    folder = args.dir or heldout.folder()
    if folder is None:
        print("no held-out folder: set HELDOUT_DIR or --dir", file=sys.stderr)
        return 1
    folder = Path(folder)
    if not args.pack and not args.summary:
        ap.error("a pack, or --summary")
    if args.pack:
        cases, errors = parse(Path(args.pack).read_text(encoding="utf-8"))
        if errors:
            print("not recorded:\n" + "\n".join(f"- {e}" for e in errors), file=sys.stderr)
            return 1
        versions = {r["id"]: r.get("version") for r in heldout.load(folder)} if folder.is_dir() else {}
        new = rows(cases, str(Path(args.pack).name), versions)
        blank = [c["id"] for c in cases if heldout.case_mean(c["sections"]) is None]
        for r in new:
            print(f"{r['id']} ({r['version']}): mean {r['mean']} — " +
                  ", ".join(f"{k} {v}" for k, v in r["sections"].items() if v is not None))
        if blank:
            print(f"not scored yet (no voto): {', '.join(blank)}")
        if not args.dry_run and new:
            folder.mkdir(parents=True, exist_ok=True)
            with open(folder / heldout.SCORES, "a", encoding="utf-8") as fh:
                for r in new:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"{len(new)} score(s) recorded in {folder / heldout.SCORES}")
    recorded = heldout.scores(folder) if folder.is_dir() else []
    if args.pack and args.dry_run:
        recorded += new         # the table as it would be once recorded
    table = heldout.means_table(recorded)
    print("\n".join(["", "## The lawyer's scores per version (held-out)", ""] + (table or ["- none recorded"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
