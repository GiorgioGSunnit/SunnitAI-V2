"""Blind A/B of two configurations on one request: each runs through scripts/answer_docx.py (dryrun_strategy,
read-only) in its own process, at the same time; the two .docx are labelled "Risposta 1" and "Risposta 2" in
random order and carry no trace of their configuration; the key goes to a separate file.

    cd /opt/chatbot && venv/bin/python scripts/ab_answers.py --text "<the message>" \\
        --a "FACT_SIGNALS_ENABLED=false" --b "FACT_SIGNALS_ENABLED=true,GRADER_EVIDENCE_ENABLED=true" \\
        --out /tmp/ab_leonardo
    (or --file <document> --request "<the message>", or --case h3)

Writes <out>/Risposta 1.docx and <out>/Risposta 2.docx; the key (which is A, which is B, with each
configuration) and each run's full record go to <out>/chiave/, apart. Hand the reviewer the two .docx only.
--seed makes the order reproducible.
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def settings(text: str) -> list:
    """"A=1,B=2" -> ["A=1", "B=2"] (an empty string: the .env as it is)."""
    return [part.strip() for part in (text or "").split(",") if part.strip()]


def assign(seed=None) -> dict:
    """{"Risposta 1": "A" or "B", "Risposta 2": the other}, at random."""
    order = ["A", "B"]
    (random.Random(seed) if seed is not None else random.SystemRandom()).shuffle(order)
    return {"Risposta 1": order[0], "Risposta 2": order[1]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--file")
    source.add_argument("--case")
    parser.add_argument("--request", default="")
    parser.add_argument("--a", default="", help="configuration A: comma-separated NAME=VALUE over the .env")
    parser.add_argument("--b", default="", help="configuration B")
    parser.add_argument("--out", required=True, help="the folder to write into")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--env", default=os.path.join(os.getcwd(), ".env"))
    args = parser.parse_args()
    keys_dir = os.path.join(args.out, "chiave")
    os.makedirs(keys_dir, exist_ok=True)
    configs = {"A": settings(args.a), "B": settings(args.b)}
    key = assign(args.seed)
    source = (["--text", args.text] if args.text else ["--case", args.case] if args.case
              else ["--file", args.file] + (["--request", args.request] if args.request else []))
    jobs = {}
    for label, which in key.items():
        cmd = [sys.executable, os.path.join(HERE, "answer_docx.py"), *source, "--blind", "--label", label,
               "--out", os.path.join(args.out, f"{label}.docx"), "--env", args.env,
               "--save-json", os.path.join(keys_dir, f"{label}.json")]
        for item in configs[which]:
            cmd += ["--set", item]
        jobs[label] = subprocess.Popen(cmd, cwd=os.getcwd(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    t0 = time.time()
    results = {}
    for label, job in jobs.items():
        out, err = job.communicate()
        results[label] = {"exit": job.returncode, "output": out.strip()[-500:], "error": err.strip()[-1500:]}
    record = {"key": {label: {"configuration": which, "set": configs[which]} for label, which in key.items()},
              "source": source, "minutes": round((time.time() - t0) / 60, 1),
              "runs": {label: {"exit": r["exit"], "output": r["output"]} for label, r in results.items()}}
    with open(os.path.join(keys_dir, "chiave_ab.json"), "w", encoding="utf-8") as fh:
        json.dump(record, fh, ensure_ascii=False, indent=1)
    for label, r in results.items():
        print(f"{label}: {'ok' if r['exit'] == 0 else 'FAILED'} {r['output'] or r['error'][-300:]}")
    print(f"Key and records (do not share with the reviewer): {keys_dir}")


if __name__ == "__main__":
    main()
