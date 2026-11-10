"""Hands a Cassazione batch to the ingest at the ingest's own pace, and checks a batch before the first file moves.

The 2025 batch (5,307 files, Oct-Nov 2026) was fed by feed_queue.sh: once an hour, 240 files, only while the queue
held fewer than 100. With the model working the ingest drains ~60 files an hour, so that feeder skipped most hours
and the queue ran dry or sat at 190 for hours. This one tops the queue up to a target every run (a cron every ten
minutes), so the queue follows the ingest's real rate, at one worker or three; with several queue folders (one per
ingest service instance) it fills the emptiest first. A pause file stops it without touching the crontab, for the
cards window or a daytime stop. Nothing is deleted: a file already handed over (its name is in the manifest) is set
aside in <src>/_already_fed instead of being fed again, because the ingest REPLACES a document with the same name.

    maint_feed.py check --src /home/sunnit/pdf2026 --batch pdf2026 [--graph /opt/v2tools/health.env] [--set-aside]
        before the batch: counts, duplicate names inside the batch, files without the Italgiure code / the .name
        sidecar, files already fed or already in the graph; --set-aside moves those to <src>/_already_fed and
        <src>/_already_in_graph. Prints file codes only, never a sidecar's content (it can name the parties).
    maint_feed.py feed --src /home/sunnit/pdf2026 --batch pdf2026 --queue /opt/sunnitai-be/document_upload \
        [--queue /opt/sunnitai-be/document_upload_2] --target 100 --max 300
        cron, every ten minutes, as root:  */10 * * * * python3 /opt/v2tools/maint_feed.py feed ... >> /var/log/maint/feed_queue.log
    pause:  touch /var/log/maint/feed.pause     resume:  rm /var/log/maint/feed.pause

Manifest lines stay 'YYYY-MM-DD batch file' (maint_health.py reads them); the state file is read by the health
report's feeder line.
"""
import argparse
import datetime as dt
import glob
import hashlib
import json
import os
import re
import shutil
import sys
from collections import Counter
from typing import Dict, Iterable, List, Optional, Set

ITALGIURE = re.compile(r"(snpen|snciv)[@_]s\w\w[@_]a((?:19|20)\d{2})[@_]n\d+[@_]t[a-z]", re.IGNORECASE)
MANIFEST_LINE = re.compile(r"^(\d{4}-\d{2}-\d{2})\s+(\S+)\s+(\S.*)$")
ALREADY_FED, ALREADY_IN_GRAPH = "_already_fed", "_already_in_graph"


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def pdfs_in(folder: str) -> List[str]:
    """PDF file names directly in `folder` (not its side folders), sorted: the feed order."""
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join(folder, "*.pdf")))


def fed_names(manifest: str) -> Set[str]:
    """File names already handed to the queue, from every batch in the manifest."""
    names: Set[str] = set()
    if os.path.isfile(manifest):
        with open(manifest, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                m = MANIFEST_LINE.match(line.strip())
                if m:
                    names.add(m.group(3).strip())
    return names


def move_with_sidecar(src: str, name: str, dest_dir: str) -> None:
    os.makedirs(dest_dir, exist_ok=True)
    shutil.move(os.path.join(src, name), os.path.join(dest_dir, name))
    sidecar = os.path.splitext(name)[0] + ".name"
    if os.path.isfile(os.path.join(src, sidecar)):
        shutil.move(os.path.join(src, sidecar), os.path.join(dest_dir, sidecar))


def duplicate_stems(names: Iterable[str]) -> List[str]:
    """File names that repeat ignoring case and extension (the ingest keys documents by name)."""
    counts = Counter(os.path.splitext(n)[0].lower() for n in names)
    return sorted(s for s, c in counts.items() if c > 1)


def plan_feed(needs: Dict[str, int], candidates: Iterable[str], cap: int) -> List[tuple]:
    """Which file goes to which queue: each file to the queue with the largest remaining need, until every need is
    met or `cap` files are planned. Returns (file, queue) pairs in feed order."""
    needs = {q: max(n, 0) for q, n in needs.items()}
    plan: List[tuple] = []
    for name in candidates:
        if len(plan) >= cap or not any(needs.values()):
            break
        queue = max(needs, key=lambda q: (needs[q], -list(needs).index(q)))
        needs[queue] -= 1
        plan.append((name, queue))
    return plan


def feed(args) -> int:
    state = {"time": now_utc(), "batch": args.batch, "src": args.src, "paused": os.path.exists(args.pause_file),
             "fed": 0, "already_fed": 0, "queues": {}, "left": None, "failed": None}
    try:
        state["failed"] = sum(1 for f in os.listdir(args.failed_dir) if f.lower().endswith(".pdf"))
    except OSError:
        pass
    if not os.path.isdir(args.src):
        state["error"] = f"staging folder not found: {args.src}"
    elif state["paused"]:
        state["left"] = len(pdfs_in(args.src))
    else:
        needs = {}
        for queue in args.queue:
            os.makedirs(queue, exist_ok=True)
            needs[queue] = args.target - len(pdfs_in(queue))
            state["queues"][queue] = len(pdfs_in(queue))
        already = fed_names(args.manifest)
        candidates = []
        for name in pdfs_in(args.src):
            if name in already:
                move_with_sidecar(args.src, name, os.path.join(args.src, ALREADY_FED))
                state["already_fed"] += 1
            else:
                candidates.append(name)
        plan = plan_feed(needs, candidates, args.max)
        day = dt.date.today().isoformat()
        os.makedirs(os.path.dirname(args.manifest) or ".", exist_ok=True)
        with open(args.manifest, "a", encoding="utf-8") as manifest:
            for name, queue in plan:
                try:
                    move_with_sidecar(args.src, name, queue)
                except OSError as exc:
                    state["error"] = f"cannot move {name}: {type(exc).__name__}"
                    break
                manifest.write(f"{day} {args.batch} {name}\n")
                state["fed"] += 1
                state["queues"][queue] += 1
        state["left"] = len(pdfs_in(args.src))
    queues = ", ".join(f"{os.path.basename(q) or q} {n}" for q, n in state["queues"].items()) or "-"
    line = (f"{state['time']} batch={args.batch} " + ("PAUSED, " if state["paused"] else "")
            + f"fed {state['fed']} (queues now: {queues}), {state['left'] if state['left'] is not None else '?'} left"
            + (f", {state['already_fed']} already fed set aside" if state["already_fed"] else "")
            + (f", failed so far {state['failed']}" if state["failed"] is not None else "")
            + (f" -- {state['error']}" if state.get("error") else ""))
    print(line)
    if args.state:
        os.makedirs(os.path.dirname(args.state) or ".", exist_ok=True)
        with open(args.state, "w", encoding="utf-8") as fh:
            json.dump(state, fh, ensure_ascii=False, indent=1)
    return 1 if state.get("error") else 0


def graph_names(env_path: str, names: List[str]) -> Set[str]:
    """Which of these file names the graph already holds (Document.file_name or a name that contains the code)."""
    env = {}
    with open(env_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    from neo4j import GraphDatabase, READ_ACCESS
    found: Set[str] = set()
    driver = GraphDatabase.driver(env["NEO4J_URI"], auth=(env["NEO4J_USER"], env["NEO4J_PASSWORD"]))
    with driver.session(database=env.get("NEO4J_DATABASE", "neo4j"), default_access_mode=READ_ACCESS) as s:
        for i in range(0, len(names), 500):
            part = names[i:i + 500]
            stems = {os.path.splitext(n)[0].lower(): n for n in part}
            for r in s.run("MATCH (d:Document) WHERE d.file_name IN $names RETURN d.file_name AS f", names=part):
                found.add(r["f"])
            for r in s.run("UNWIND $stems AS st MATCH (d:Document) WHERE toLower(d.name) CONTAINS st "
                           "RETURN DISTINCT st", stems=list(stems)):
                found.add(stems[r["st"]])
    driver.close()
    return found


def check(args) -> int:
    if not os.path.isdir(args.src):
        print(f"staging folder not found: {args.src}")
        return 1
    names = pdfs_in(args.src)
    dup_stems = duplicate_stems(names)
    no_code = [n for n in names if not ITALGIURE.search(n)]
    years = Counter((ITALGIURE.search(n).group(2) if ITALGIURE.search(n) else "?") for n in names)
    branches = Counter((ITALGIURE.search(n).group(1).lower() if ITALGIURE.search(n) else "?") for n in names)
    no_sidecar, sidecar_hashes = [], Counter()
    for n in names:
        sidecar = os.path.join(args.src, os.path.splitext(n)[0] + ".name")
        if not os.path.isfile(sidecar):
            no_sidecar.append(n)
        else:
            with open(sidecar, "rb") as fh:
                sidecar_hashes[hashlib.sha1(fh.read().strip().lower()).hexdigest()] += 1
    same_display_name = sum(c - 1 for c in sidecar_hashes.values() if c > 1)
    fed = fed_names(args.manifest)          # read once: per name it was 35k reads of the manifest
    already = sorted(n for n in names if n in fed)
    in_graph: Set[str] = set()
    if args.graph:
        in_graph = graph_names(args.graph, names)

    def show(label, items):
        print(f"  {label}: {len(items):,}" + (f"  e.g. {', '.join(items[:5])}" if items else ""))

    print(f"Batch {args.batch} in {args.src}: {len(names):,} PDFs; by year "
          + ", ".join(f"{y} {c:,}" for y, c in sorted(years.items())) + "; by branch "
          + ", ".join(f"{b} {c:,}" for b, c in sorted(branches.items())))
    show("duplicate file names inside the batch (case-insensitive)", dup_stems)
    show("without the Italgiure code in the name (snpen/snciv, year, number: no branch, no year label)", no_code)
    show("without a .name sidecar (the display name then comes from the file name)", no_sidecar)
    print(f"  sidecars whose display name repeats another's: {same_display_name:,} "
          "(the ingest replaces a document with the same name: the earlier ruling would be deleted)")
    show("already handed to the queue (name in the manifest)", already)
    if args.graph:
        show("already in the graph (Document.file_name, or a name holding the code)", sorted(in_graph))
    if args.set_aside:
        for n in already:
            move_with_sidecar(args.src, n, os.path.join(args.src, ALREADY_FED))
        for n in sorted(in_graph - set(already)):
            move_with_sidecar(args.src, n, os.path.join(args.src, ALREADY_IN_GRAPH))
        print(f"  set aside: {len(already):,} to {ALREADY_FED}, {len(in_graph - set(already)):,} to {ALREADY_IN_GRAPH} "
              f"(moved, not deleted); {len(pdfs_in(args.src)):,} left to feed")
    problems = len(dup_stems) + len(no_code) + same_display_name + (0 if args.set_aside else len(already) + len(in_graph))
    print("OK to feed" if not problems else f"{problems:,} things to look at before feeding")
    return 0 if not problems else 2


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("feed", "check"):
        s = sub.add_parser(name)
        s.add_argument("--src", required=True, help="staging folder holding the batch's PDFs (+ .name sidecars)")
        s.add_argument("--batch", required=True, help="batch name written to the manifest, e.g. pdf2026")
        s.add_argument("--manifest", default="/var/log/maint/upload_batches.txt")
    f = sub.choices["feed"]
    f.add_argument("--queue", action="append", default=None,
                   help="ingest queue folder; repeat for a second ingest instance (default: the production queue)")
    f.add_argument("--target", type=int, default=100, help="files each queue is topped up to")
    f.add_argument("--max", type=int, default=300, help="most files moved in one run")
    f.add_argument("--pause-file", default="/var/log/maint/feed.pause")
    f.add_argument("--state", default="/var/log/maint/feed_state.json", help="JSON read by the health report")
    f.add_argument("--failed-dir", default="/opt/sunnitai-be/document_failed")
    c = sub.choices["check"]
    c.add_argument("--graph", metavar="ENV", help="also ask the graph (a health.env with NEO4J_* settings)")
    c.add_argument("--set-aside", action="store_true", help="move the already-fed / already-in-graph files out")
    args = p.parse_args(argv)
    if args.cmd == "feed":
        args.queue = args.queue or ["/opt/sunnitai-be/document_upload"]
        return feed(args)
    return check(args)


if __name__ == "__main__":
    sys.exit(main())
