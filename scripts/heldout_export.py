#!/usr/bin/env python3
"""v77: the held-out pack for the lawyer -- N unscored case-first answers from the held-out set (src/chatbot/heldout.py:
/opt/chatbot/heldout/<date>_<id>.json, every case-first answer served live or on the test copy) as one Markdown file:
per case the request (and the uploaded case, when the case was a file), the parere as served, and a blank score block
with his seven sections (heldout.SECTIONS), each 0-5, and his notes. He fills the blocks and sends the file back;
scripts/heldout_score.py reads it.

Unscored: no score in <folder>/scores.jsonl, and not in a pack already exported and not yet scored (exports.jsonl;
--include-pending lists those too). The newest first. Only the first parere of a case (kind «answer»): --with-updates
adds the pareri written again after a follow-up.

  python scripts/heldout_export.py -n 10 --out /root/pack_2026-11-17.md
  python scripts/heldout_export.py -n 5 --version 77.0.0+abc1234 --source live --out pack.md
  python scripts/heldout_export.py --dir /tmp/heldout -n 3 --out pack.md --dry-run     # no exports.jsonl line
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.chatbot import heldout  # noqa: E402

MARK = "<!-- heldout:{id} version:{version} -->"
SCORE_HEADING = "### Valutazione dell'avvocato"
CASE_TEXT_CHARS = 20000


def exported(folder: Path) -> set:
    path = folder / heldout.EXPORTS
    if not path.is_file():
        return set()
    return {i for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
            for i in json.loads(line).get("ids") or []}


def pick(records: List[Dict[str, Any]], scored: set, pending: set, n: int, version: str = "", source: str = "",
         with_updates: bool = False) -> List[Dict[str, Any]]:
    out = []
    for rec in reversed(records):                   # newest first
        if rec["id"] in scored or rec["id"] in pending:
            continue
        if version and rec.get("version") != version:
            continue
        if source and rec.get("source") != source:
            continue
        if not with_updates and rec.get("kind") != "answer":
            continue
        out.append(rec)
        if len(out) >= n:
            break
    return out


def _quote(text: str) -> str:
    return "\n".join("> " + line if line.strip() else ">" for line in (text or "").strip().split("\n"))


def score_block() -> List[str]:
    lo, hi = heldout.SCALE
    lines = [SCORE_HEADING, "",
             f"Un voto da {lo} a {hi} per sezione ({lo} = gravemente insufficiente, {hi} = nulla da correggere; "
             f"per «Pregi»: {hi} = molti e solidi). Lasciare il voto vuoto se la sezione non si applica al caso.", ""]
    for i, (_, label) in enumerate(heldout.SECTIONS, 1):
        lines += [f"{i}. {label}", "   - voto: ", "   - note: ", ""]
    lines += ["Giudizio complessivo: ", ""]
    return lines


def render(picked: List[Dict[str, Any]], today: str) -> str:
    lines = [f"# Pareri da valutare — {today}", "",
             f"{len(picked)} pareri prodotti da SunnitAI su casi reali, nessuno già valutato. Per ciascuno: la "
             "richiesta, il parere come è stato consegnato, e sotto «Valutazione dell'avvocato» le sette sezioni da "
             "valutare. Si prega di non modificare le righe che iniziano con «<!--» né i titoli delle sezioni.", ""]
    for n, rec in enumerate(picked, 1):
        lines += ["---", "", MARK.format(id=rec["id"], version=rec.get("version") or heldout.version.UNKNOWN),
                  f"## Caso {n} ({rec['id']})", "", "### Richiesta", "", _quote(rec.get("prompt") or ""), ""]
        if rec.get("case_text"):
            case = rec["case_text"]
            if len(case) > CASE_TEXT_CHARS:
                case = case[:CASE_TEXT_CHARS] + "\n[…]"
            lines += ["### Caso (documento caricato)", "", _quote(case), ""]
        lines += ["### Parere", "", (rec.get("answer") or "").strip(), ""] + score_block()
    return "\n".join(lines).rstrip() + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("-n", type=int, default=10, help="how many cases (default 10)")
    ap.add_argument("--dir", type=Path, help="the held-out folder (default: HELDOUT_DIR, else /opt/chatbot/heldout)")
    ap.add_argument("--out", required=True, help="the pack (Markdown)")
    ap.add_argument("--version", default="", help="only the answers of this version")
    ap.add_argument("--source", choices=("live", "test"), default="", help="only the live or the test copy's answers")
    ap.add_argument("--with-updates", action="store_true", help="the pareri written again after a follow-up too")
    ap.add_argument("--include-pending", action="store_true", help="the cases of packs exported and not scored too")
    ap.add_argument("--dry-run", action="store_true", help="write the pack, do not record it in exports.jsonl")
    args = ap.parse_args(argv)
    folder = args.dir or heldout.folder()
    if folder is None or not Path(folder).is_dir():
        print(f"no held-out folder ({folder or 'set HELDOUT_DIR or --dir'})", file=sys.stderr)
        return 1
    folder = Path(folder)
    scored = {s["id"] for s in heldout.scores(folder)}
    pending = set() if args.include_pending else exported(folder) - scored
    picked = pick(heldout.load(folder), scored, pending, args.n, args.version, args.source, args.with_updates)
    if not picked:
        print("no unscored case to export", file=sys.stderr)
        return 1
    today = time.strftime("%Y-%m-%d")
    Path(args.out).write_text(render(picked, today), encoding="utf-8")
    if not args.dry_run:
        with open(folder / heldout.EXPORTS, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                 "pack": str(Path(args.out).resolve()), "ids": [r["id"] for r in picked]}) + "\n")
    print(f"{len(picked)} case(s) -> {args.out}: " + ", ".join(f"{r['id']} ({r.get('version')})" for r in picked))
    return 0


if __name__ == "__main__":
    sys.exit(main())
