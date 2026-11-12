"""Lever 5 on be2 (prepared 11 Nov 2026): the waits of the chunk analysis become a setting.

What the journal of 10-11 Nov showed (763 rulings, two workers): a chunk call either answers within 30 s or never.
4.5% of the chunks hang on the first request; a fresh request for the same chunk answers in about 6 s half the
time. The code waits 60 s, then 120 s, and inside that second wait the client gives up after its own 90 s and
asks a third time. So a hung chunk already gets three requests, after 60, 150 and (too late) 180 s, and the
waiting is about 40% of all worker time.

ANALISI_CHUNK_TIMEOUTS in a worker's environment file lists the seconds each structured attempt waits:

    ANALISI_CHUNK_TIMEOUTS=40,40,40     # three requests, as today, each with more than a normal call needs

Without the setting nothing changes (default "60,120"). A value that cannot be read, or under 30 s (a normal call
takes up to 30 s; 15 s cut one call per file on 5 Nov), falls back to the default. The plain-text fallback after
the last attempt keeps its 30 s. The journal still prints "attempt N/2": with three waits the third reads "3/2".

One file, lex_package/analisi_parallel.py, checked against the checksum of be2's copy after lever 3. Lever 4
(patch_ingest_lever4.py, not applied) expects that same checksum: give it the new one before applying it.

    sudo python3 /home/sunnit/patch_ingest_lever5.py --check    # checksum only, changes nothing
    sudo python3 /home/sunnit/patch_ingest_lever5.py            # apply (prints what changed)
    sudo python3 /home/sunnit/patch_ingest_lever5.py --undo     # restore the newest backup
    python3 patch_ingest_lever5.py --selftest                   # how the setting is read; touches no file

Nothing else in the file is touched; no service is restarted by this script.
"""
import glob
import hashlib
import os
import shutil
import sys
import textwrap
import datetime as dt

PATH = "/opt/sunnitai-be/src/be/src/lex_package/analisi_parallel.py"
MD5 = "db2e23f06aaed0dec2ba943aa53c881f"          # after lever 3 (6 Nov 2026)
SEMAPHORE = ("        # chunks of one document sent at once; ANALISI_CHUNK_CONCURRENCY in .env (default 50)\n"
             "        semaphore = _asyncio.Semaphore(int(os.environ.get(\"ANALISI_CHUNK_CONCURRENCY\", \"50\")))\n")
SETTING = (
    "        # Seconds each structured attempt waits before asking again: ANALISI_CHUNK_TIMEOUTS in the worker's\n"
    "        # environment file, default \"60,120\" (since 6 Nov 2026). A call that hangs never answers and a fresh\n"
    "        # one mostly answers at once (journal of 11 Nov), so \"40,40,40\" asks again sooner. A normal call takes\n"
    "        # up to 30 s: a shorter wait, or a value that cannot be read, falls back to the default.\n"
    "        try:\n"
    "            chunk_timeouts = [int(s) for s in os.environ.get(\"ANALISI_CHUNK_TIMEOUTS\", \"60,120\").split(\",\")]\n"
    "        except ValueError:\n"
    "            chunk_timeouts = []\n"
    "        if not chunk_timeouts or min(chunk_timeouts) < 30:\n"
    "            chunk_timeouts = [60, 120]\n"
)
EDITS = [
    (SEMAPHORE, SEMAPHORE + SETTING),
    ("                for attempt, timeout_secs in enumerate([60, 120]):\n",
     "                for attempt, timeout_secs in enumerate(chunk_timeouts):\n"),
]
SUFFIX = ".bak_lever5_" + dt.date.today().strftime("%d%b").lower()


def md5(path):
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def check(path=PATH):
    have = md5(path)
    same = have == MD5
    print(f"{'OK  ' if same else 'DIFF'} {path} md5 {have}" + ("" if same else f" (expected {MD5})"))
    print("ready to apply" if same else "STOP: the file is not the expected copy; nothing would be changed")
    return 0 if same else 2


def apply(path=PATH):
    have = md5(path)
    if have != MD5:
        sys.exit(f"STOP: {path} is not the expected copy (md5 {have}); nothing changed")
    with open(path, encoding="utf-8") as fh:
        s = fh.read()
    for old, _ in EDITS:
        if s.count(old) != 1:
            sys.exit(f"STOP: expected text not found exactly once in {path}: {old[:60]!r}; nothing changed")
    backup = path + SUFFIX
    shutil.copy2(path, backup)
    for old, new in EDITS:
        s = s.replace(old, new)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(s)
    print(f"patched {path} ({len(EDITS)} edits); backup {backup}")
    print("done: ANALISI_CHUNK_TIMEOUTS=40,40,40 in .env.worker2 + a restart of both workers on empty queues")


def undo(path=PATH):
    backups = sorted(glob.glob(path + ".bak_lever5_*"), key=os.path.getmtime)
    if not backups:
        print(f"no backup {path}.bak_lever5_*: nothing restored")
        return
    shutil.copy2(backups[-1], path)
    print(f"restored {path} from {backups[-1]}")


def selftest():
    """The inserted lines, run on their own: what each value of the setting becomes."""
    cases = {None: [60, 120], "40,40,40": [40, 40, 40], " 45 , 45 ,45 ": [45, 45, 45], "90": [90],
             "": [60, 120], "40;40": [60, 120], "forty": [60, 120], "15,60": [60, 120], "40,,40": [60, 120]}
    kept = os.environ.get("ANALISI_CHUNK_TIMEOUTS")
    for value, expected in cases.items():
        os.environ.pop("ANALISI_CHUNK_TIMEOUTS", None)
        if value is not None:
            os.environ["ANALISI_CHUNK_TIMEOUTS"] = value
        scope = {"os": os}
        exec(textwrap.dedent(SETTING), scope)
        assert scope["chunk_timeouts"] == expected, (value, scope["chunk_timeouts"])
    os.environ.pop("ANALISI_CHUNK_TIMEOUTS", None)
    if kept is not None:
        os.environ["ANALISI_CHUNK_TIMEOUTS"] = kept
    print(f"selftest OK: {len(cases)} values of ANALISI_CHUNK_TIMEOUTS read as expected")


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--selftest" in args:
        selftest()
    elif "--check" in args:
        sys.exit(check())
    else:
        undo() if "--undo" in args else apply()
