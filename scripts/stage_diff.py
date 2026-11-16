#!/usr/bin/env python3
"""v70 (10): every stage list against the v59 behaviour for the default cell -- the procedural section (§4 then, §8 in
the lawyer's method) rendered for the same facts by the v59 tree (git archive of --ref: no posture, no case-first
gates) and by today's tree for the default cell (civile / attore / pre_filing, the case-first path: store_only, the
case text for the triggers). The report: per case, the block titles and the articles v59 rendered that today's does
not -- the losses -- and what today's adds. No model, no database: fact_signals.signals_from on the facts, then
render_computed.blocks.

  python scripts/stage_diff.py                                  # the fixture and tests/data/stage_diff_facts.json
  python scripts/stage_diff.py --ref 9422fd7 --facts my_facts.json --out diff.md

Each facts file: {"cases": [{"id", "case": the case text, "facts": fact_signals' facts (or "facts_file": a path from the
repo's root), "areas": [..], "topic_refs": the chosen topics' articles [[n, code]]}]}, or one facts object (its id the
file name). Every feature flag the tree reads is on in both runs (as in prod), unless
--no-flags. Exit 1 when a case lost a block or an article (the nightly reads it).
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
V59 = "9422fd7"
DEFAULT_CELL = {"rito": "civile", "family": "civile", "role": "attore", "stage": "pre_filing"}
DEFAULT_FACTS = [ROOT / "tests" / "data" / "stage_diff_facts.json"]
_REF_RE = re.compile(r"\bart(?:t)?\.\s*(\d+(?:-[a-z]+)?)\s*(?:,\s*co\.\s*\d+\s*,?\s*)?(c\.p\.c\.|c\.c\.|c\.p\.p\.|c\.p\.|d\.lgs\.[^,;)]*)?",
                     re.I)
_FLAG_RE = re.compile(r"""(?:enabled|_enabled)\(\s*["']([A-Z][A-Z0-9_]+)["']|^[A-Z_]*FLAG[A-Z_]*\s*=\s*["']([A-Z][A-Z0-9_]+)["']""",
                      re.M)

CHILD = r"""
import json, os, sys
sys.path.insert(0, os.getcwd())
for k, v in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
             ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(k, v)
from src.rag import fact_signals, render_computed
spec = json.loads(sys.stdin.read())
out = {}
for case in spec["cases"]:
    signals = fact_signals.signals_from(case["facts"], areas=case.get("areas"), rules=True)
    analysis = {"fact_signals": signals, "frame": {"areas": case.get("areas") or []},
                "topic_refs": case.get("topic_refs") or []}
    if spec["mode"] == "today":
        from src.rag.interactive_flow import CASE_KEY
        analysis.update({"posture": spec["cell"], "store_offences_only": True, CASE_KEY: case.get("case") or ""})
    try:
        out[case["id"]] = [[t, b] for t, b in render_computed.blocks(analysis, "")]
    except Exception as exc:
        out[case["id"]] = {"error": f"{type(exc).__name__}: {exc}"}
print(json.dumps(out, ensure_ascii=False))
"""


def flags(src: Path) -> List[str]:
    names = set()
    for path in (src / "src" / "rag").rglob("*.py"):
        for m in _FLAG_RE.finditer(path.read_text(encoding="utf-8", errors="replace")):
            names.add(m.group(1) or m.group(2))
    return sorted(names)


def load_cases(paths: List[Path]) -> List[Dict[str, Any]]:
    cases = []
    for p in paths:
        if not p.is_file():
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("cases"), list):
            for c in data["cases"]:
                if isinstance(c, dict) and c.get("facts_file"):
                    c = {**c, "facts": json.loads((ROOT / c["facts_file"]).read_text(encoding="utf-8"))}
                if isinstance(c, dict) and c.get("facts"):
                    cases.append(c)
        elif isinstance(data, dict):
            cases.append({"id": p.stem, "case": "", "facts": data, "areas": []})
    return cases


def extract(ref: str, into: Path) -> Path:
    """The ref's src/ and data/ (git archive), into a fresh directory."""
    archive = into / "tree.tar"
    subprocess.run(["git", "-C", str(ROOT), "archive", "-o", str(archive), ref, "src", "data"], check=True)
    with tarfile.open(archive) as tar:
        tar.extractall(into / "tree", filter="data")
    return into / "tree"


def render(tree: Path, cases: List[Dict[str, Any]], mode: str, on: List[str]) -> Dict[str, Any]:
    env = {**os.environ, **{f: "true" for f in on}, "PYTHONPATH": str(tree)}
    spec = json.dumps({"cases": cases, "mode": mode, "cell": DEFAULT_CELL})
    done = subprocess.run([sys.executable, "-I", "-c", CHILD], input=spec, capture_output=True, text=True, cwd=str(tree),
                          env=env, timeout=600)
    if done.returncode != 0:
        raise SystemExit(f"{mode} render failed:\n{done.stderr[-2000:]}")
    return json.loads(done.stdout.strip().splitlines()[-1])


def refs(blocks: List[List[str]]) -> set:
    text = "\n".join(f"{t}\n{b}" for t, b in blocks)
    return {f"{n} {(c or '').strip()}".strip() for n, c in _REF_RE.findall(text)}


def diff(old: Any, new: Any) -> Dict[str, Any]:
    """A v59 block is lost when today's has no block of its title and not all its articles (moved to other blocks is
    no loss)."""
    if isinstance(old, dict) or isinstance(new, dict):
        return {"error": (old if isinstance(old, dict) else new).get("error")}
    titles_old, titles_new = [t for t, _ in old], [t for t, _ in new]
    now = refs(new)
    return {"lost_blocks": [t for t, b in old if t not in titles_new and not (refs([[t, b]]) and refs([[t, b]]) <= now)],
            "added_blocks": [t for t in titles_new if t not in titles_old],
            "lost_articles": sorted(refs(old) - refs(new)), "added_articles": sorted(refs(new) - refs(old)),
            "v59": titles_old, "today": titles_new}


def report(results: Dict[str, Dict[str, Any]], ref: str) -> str:
    out = [f"# Stage lists: today's default cell ({'/'.join(DEFAULT_CELL[k] for k in ('rito', 'role', 'stage'))}) "
           f"against v59 ({ref})", "", "| case | lost blocks | lost articles | added blocks | added articles |",
           "|---|---|---|---|---|"]
    for cid, d in results.items():
        if d.get("error"):
            out.append(f"| {cid} | error: {d['error']} | | | |")
            continue
        out.append(f"| {cid} | {', '.join(d['lost_blocks']) or '—'} | {', '.join(d['lost_articles']) or '—'} | "
                   f"{', '.join(d['added_blocks']) or '—'} | {', '.join(d['added_articles']) or '—'} |")
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ref", default=V59)
    ap.add_argument("--facts", action="append", type=Path)
    ap.add_argument("--out")
    ap.add_argument("--no-flags", action="store_true")
    args = ap.parse_args(argv)
    cases = load_cases(args.facts or DEFAULT_FACTS)
    if not cases:
        print("no cases", file=sys.stderr)
        return 2
    with tempfile.TemporaryDirectory(prefix="stage_diff_") as tmp:
        old_tree = extract(args.ref, Path(tmp))
        on = [] if args.no_flags else flags(ROOT)
        old = render(old_tree, cases, "v59", [] if args.no_flags else flags(old_tree))
        new = render(ROOT, cases, "today", on)
    results = {c["id"]: diff(old.get(c["id"]), new.get(c["id"])) for c in cases}
    text = report(results, args.ref)
    if args.out:
        Path(args.out).write_text(text + "\n```json\n" + json.dumps(results, ensure_ascii=False, indent=1) + "\n```\n",
                                  encoding="utf-8")
    print(text)
    lost = any(d.get("lost_blocks") or d.get("lost_articles") or d.get("error") for d in results.values())
    return 1 if lost else 0


if __name__ == "__main__":
    sys.exit(main())
