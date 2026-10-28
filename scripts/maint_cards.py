"""Maintenance agent, job 4: the ruling cards of new rulings, every night (Flavio's brief, 27 Oct). Runs on prod, where
the card files and the model calls are -- e.g. at 21:00 UTC from root's crontab, with nothing else heavy running.

Each night:
  1. a lock (--lock) and a check that no card run is going: otherwise skipped, with a summary line saying why
  2. step 1's links refreshed into a new file, then swapped in (exp_ruling_links.py: no model call, ~5 minutes)
  3. for each branch, penale then civile: the rulings without a card carded into a WORK copy of the live card file
     (exp_ruling_cards.py, resumable; a work copy left by a crashed night is resumed), no new ruling after --stop-at
  4. each work copy swapped in for the live file in one rename when it holds at least as many lines, the previous file
     kept as <file>.prev: the app (src/rag/ruling_cards.store) reloads the cards once, never mid-run
  5. one summary line in --log-dir/cards_YYYYMMDD.txt: cards added per branch, still to card, failed, minutes

Dry run by default: refreshes nothing and cards nothing, says how many rulings each branch has to card (from the
current links file) and the commands it would run. --write does it. The cards are the live app's data: the files are
only ever replaced whole, never edited in place.

    /opt/chatbot/venv/bin/python /root/ruling_cards/maint_cards.py                          # dry run
    /opt/chatbot/venv/bin/python /root/ruling_cards/maint_cards.py --write --stop-at 00:45
"""

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time
from typing import Callable, Dict, List, Optional, Set, Tuple

try:
    import fcntl
except ImportError:          # Windows (the tests): no lock
    fcntl = None

BRANCH_FILES = {"penale": "cards_pen_v3.jsonl", "civile": "cards_civ_v1.jsonl"}
HERE = os.path.dirname(os.path.abspath(__file__))


def read_jsonl(path: str) -> List[dict]:
    out = []
    if not os.path.isfile(path):
        return out
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict):
                out.append(row)
    return out


def card_ids(path: str) -> Tuple[Set[str], int]:
    """(ids carded, ids whose only card failed)."""
    rows = read_jsonl(path)
    good = {r["id"] for r in rows if r.get("id") and not r.get("errore")}
    failed = {r["id"] for r in rows if r.get("id") and r.get("errore")} - good
    return good, len(failed)


def to_card(links: List[dict], branch: str, done: Set[str]) -> int:
    """How many rulings of the branch have no card: exp_ruling_cards.pick's rule (text read, civil closures skipped)."""
    return sum(1 for r in links if r.get("branch") == branch and r.get("id") not in done and r.get("chars")
               and (branch != "civile" or r.get("outcome") != "estinto"))


def lines(path: str) -> int:
    if not os.path.isfile(path):
        return 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        return sum(1 for _ in fh)


def swap(work: str, live: str) -> str:
    """Put the work copy live in one rename, the previous live file kept as <live>.prev; refused when the work copy
    has fewer lines (a card file only grows)."""
    before, after = lines(live), lines(work)
    if after < before:
        return f"refused: {os.path.basename(work)} has {after} lines, the live file {before}"
    if os.path.isfile(live):
        shutil.copy2(live, f"{live}.prev")
    os.replace(work, live)
    return f"live {before} -> {after} lines"


def busy(runner: Callable = subprocess.run) -> bool:
    """A card run going (the manual night script, or another job)."""
    try:
        out = runner(["ps", "-eo", "args"], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return False
    return any("exp_ruling_cards.py" in line and "maint_cards.py" not in line for line in out.splitlines())


class Lock:
    def __init__(self, path: str):
        self.path, self.fh = path, None

    def __enter__(self) -> bool:
        if fcntl is None:
            return True
        self.fh = open(self.path, "w")
        try:
            fcntl.flock(self.fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False

    def __exit__(self, *exc):
        if self.fh:
            self.fh.close()


def plan(args) -> List[Tuple[str, List[str]]]:
    """[(what, command)] of a night."""
    d, py = args.dir, args.python
    links_new = os.path.join(d, "ruling_links_new.jsonl")
    steps = [("links", [py, os.path.join(args.scripts, "exp_ruling_links.py"), "--out", links_new])]
    for branch in args.branches:
        work = os.path.join(d, f"work_{branch}.jsonl")
        steps.append((branch, [py, os.path.join(args.scripts, "exp_ruling_cards.py"), "--links",
                               os.path.join(d, "ruling_links.jsonl"), "--branch", branch, "--out", work,
                               "--workers", str(args.workers)] + (["--stop-at", args.stop_at] if args.stop_at else [])))
    return steps


def night(args, runner: Callable = subprocess.run, log: Callable = print) -> str:
    """One night's run (or its dry run); returns the summary line."""
    t0 = time.time()
    d = args.dir
    links_path = os.path.join(d, "ruling_links.jsonl")
    live = {b: os.path.join(d, BRANCH_FILES[b]) for b in args.branches}
    before = {b: card_ids(live[b]) for b in args.branches}
    if not args.write:
        links = read_jsonl(links_path)
        todo = {b: to_card(links, b, before[b][0]) for b in args.branches}
        for what, cmd in plan(args):
            log(f"would run ({what}): {' '.join(cmd)}")
        return ("DRY RUN: to card " + ", ".join(f"{b} {n:,}" for b, n in todo.items())
                + f" (links file: {len(links):,} rulings; new rulings since it are not counted)")
    if busy(runner):
        return "SKIPPED: a card run is already going"
    steps = plan(args)
    what, cmd = steps[0]
    links_new = cmd[-1]
    log(f"links: {' '.join(cmd)}")
    if runner(cmd, cwd=args.cwd).returncode != 0 or not os.path.isfile(links_new):
        return "FAILED: step 1's links refresh (see the log); nothing carded"
    os.replace(links_new, links_path)
    added, notes = {}, []
    for (branch, cmd) in steps[1:]:
        work = cmd[cmd.index("--out") + 1]
        if not (os.path.isfile(work) and lines(work) >= lines(live[branch])):   # else: a crashed night's, resumed
            if os.path.isfile(live[branch]):
                shutil.copy2(live[branch], work)
            else:
                open(work, "w").close()
        log(f"{branch}: {' '.join(cmd)}")
        code = runner(cmd, cwd=args.cwd).returncode
        if code != 0:
            notes.append(f"{branch} card run exited {code}")
        notes.append(f"{branch} {swap(work, live[branch])}")
        added[branch] = len(card_ids(live[branch])[0] - before[branch][0])
    links = read_jsonl(links_path)
    after = {b: card_ids(live[b]) for b in args.branches}
    todo = {b: to_card(links, b, after[b][0]) for b in args.branches}
    return (f"cards +{sum(added.values()):,} (" + ", ".join(f"{b} +{n:,}" for b, n in added.items()) + ")"
            + " · still to card " + ", ".join(f"{b} {n:,}" for b, n in todo.items())
            + " · failed " + ", ".join(f"{b} {after[b][1]}" for b in args.branches)
            + f" · links {len(links):,} rulings · {(time.time() - t0) / 60:.0f} min"
            + (" · " + "; ".join(notes) if notes else ""))


def main() -> int:
    p = argparse.ArgumentParser(description="Maintenance agent, job 4: the nightly ruling cards (dry run by default)")
    p.add_argument("--write", action="store_true", help="refresh the links, card, swap (without it: a dry run)")
    p.add_argument("--dir", default="/root/ruling_cards", help="the card files and step 1's links")
    p.add_argument("--scripts", default=HERE, help="the folder holding exp_ruling_links.py and exp_ruling_cards.py")
    p.add_argument("--cwd", default="/opt/chatbot", help="the app folder the card scripts run from")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--branches", default="penale,civile")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--stop-at", default="00:45", help="HH:MM server time: no new ruling after it ('' for none)")
    p.add_argument("--lock", default="/tmp/maint_cards.lock")
    p.add_argument("--log-dir", default="/var/log/maint")
    args = p.parse_args()
    args.branches = [b for b in args.branches.split(",") if b in BRANCH_FILES]
    with Lock(args.lock) as got:
        line = night(args) if got else "SKIPPED: another maint_cards run holds the lock"
    print(line)
    if args.write:
        os.makedirs(args.log_dir, exist_ok=True)
        with open(os.path.join(args.log_dir, f"cards_{dt.date.today().strftime('%Y%m%d')}.txt"), "a",
                  encoding="utf-8") as fh:
            fh.write(f"{dt.datetime.now().isoformat(timespec='minutes')} {line}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
