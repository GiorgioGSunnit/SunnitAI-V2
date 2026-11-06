"""Lever 3 on be2 (6 Nov 2026): three small edits in Flavio's ingest, each with a dated backup, each asserted
against the exact text of the 6 Nov copy, so the script refuses to touch a file that changed since.

  1. lex_package/analisi_parallel.py: the first structured attempt per chunk waited 15 s, then gave up and asked
     again while the server was still answering (once per file at one worker: 181 timeouts in 183 files on 5 Nov).
     Now 60 s, the retry 120 s. The plain-text fallback keeps its 30 s.
  2. lex_package/analisi_parallel.py: the chunk burst per document (Semaphore(50)) becomes a setting,
     ANALISI_CHUNK_CONCURRENCY in /opt/sunnitai-be/.env, default 50 = today's behaviour. Lower it (8) when the
     ingest runs beside the users during the day.
  3. batch_ingest.py: the 5 s sleep after every file (2 days over 35,000 files) becomes 0.5 s.

    sudo python3 /home/sunnit/patch_ingest_lever3.py          # apply (prints what changed)
    sudo python3 /home/sunnit/patch_ingest_lever3.py --undo   # restore the backups
    sudo systemctl restart sunnitai-ingest                     # afterwards, when the queue is empty

Nothing else in the files is touched; the service is not restarted by this script.
"""
import hashlib
import shutil
import sys
import datetime as dt

ROOT = "/opt/sunnitai-be"
FILES = {
    f"{ROOT}/src/be/src/lex_package/analisi_parallel.py": {
        "md5": "3fa3c0dbe34ad5b12ad68880d437ea9b",
        "edits": [
            ("import logging\nimport time\n", "import logging\nimport os\nimport time\n"),
            ("        semaphore = _asyncio.Semaphore(50)  # max 5 concurrent RunPod calls\n",
             "        # chunks of one document sent at once; ANALISI_CHUNK_CONCURRENCY in .env (default 50)\n"
             "        semaphore = _asyncio.Semaphore(int(os.environ.get(\"ANALISI_CHUNK_CONCURRENCY\", \"50\")))\n"),
            ("                # Attempt 1: structured output with exponential backoff (15s, then 25s)\n"
             "                for attempt, timeout_secs in enumerate([15, 60]):\n",
             "                # Attempt 1: structured output, 60 s then 120 s (15 s gave up once per file while the\n"
             "                # server was still answering, and asked again: see the 5 Nov 2026 journal)\n"
             "                for attempt, timeout_secs in enumerate([60, 120]):\n"),
        ],
    },
    f"{ROOT}/batch_ingest.py": {
        "md5": "7d8897d38436a8761922e84dd03f253f",
        "edits": [
            ("        time.sleep(5)\n\n    return successes\n", "        time.sleep(0.5)\n\n    return successes\n"),
        ],
    },
}
SUFFIX = ".bak_lever3_" + dt.date.today().strftime("%d%b").lower()


def md5(path):
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def apply():
    for path, spec in FILES.items():
        have = md5(path)
        if have != spec["md5"]:
            sys.exit(f"STOP: {path} is not the 6 Nov copy (md5 {have}); nothing changed")
    for path, spec in FILES.items():
        with open(path, encoding="utf-8") as fh:
            s = fh.read()
        for old, new in spec["edits"]:
            if s.count(old) != 1:
                sys.exit(f"STOP: expected text not found exactly once in {path}: {old[:60]!r}; nothing changed")
        backup = path + SUFFIX
        shutil.copy2(path, backup)
        for old, new in spec["edits"]:
            s = s.replace(old, new)
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(s)
        print(f"patched {path} ({len(spec['edits'])} edits); backup {backup}")
    print("done: restart the ingest service when its queue is empty (sudo systemctl restart sunnitai-ingest)")


def undo():
    for path in FILES:
        backup = path + SUFFIX
        try:
            shutil.copy2(backup, path)
            print(f"restored {path} from {backup}")
        except FileNotFoundError:
            print(f"no backup for {path} with today's suffix ({backup}): look for *.bak_lever3_*")


if __name__ == "__main__":
    undo() if "--undo" in sys.argv[1:] else apply()
