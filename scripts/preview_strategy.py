"""Read-only: the strategy draft (parere) a chat user would get for a case,
with and without STRATEGY_DRAFT_RULES_ENABLED (round 3, item 4), so a lawyer
can compare them before the flag goes on for everyone.

Run from the app folder so it uses the app's code and .env (be2 or production):
    cd /opt/chatbot && venv/bin/python scripts/preview_strategy.py --facts-file /tmp/qt/sempronio.txt
    ... --text "Ho firmato un contratto perché mi hanno minacciato: posso annullarlo?"
    ... --rules on        # only one version (on / off); default: both
    ... --out /tmp/strategy_preview

It makes the same call as the chat (run_defensive_pipeline(message, message)),
so every other flag comes from .env as on the live service. The flag is set
only inside this process. Neo4j sessions are read-only, nothing is stored, and
the drafts are written to --out. The live answer also gets the
"BOZZA STRATEGIA DIFENSIVA" header, which is not shown here.
"""

import argparse
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (ROOT, os.getcwd()):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(os.getcwd(), ".env"))

import logging  # noqa: E402

import neo4j  # noqa: E402

_open_session = neo4j.Driver.session


def _read_only_session(self, **config):
    config["default_access_mode"] = neo4j.READ_ACCESS
    return _open_session(self, **config)


neo4j.Driver.session = _read_only_session   # every session in this process is read-only
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

FLAG = "STRATEGY_DRAFT_RULES_ENABLED"
_ARTICLE = re.compile(r"\bart(?:icol[oi]|t)?\.?\s*(\d+(?:[\s-]*(?:bis|ter|quater|quinquies|sexies))?)", re.IGNORECASE)


def _summary(draft: str) -> str:
    articles = sorted({" ".join(m.group(1).split()).lower() for m in _ARTICLE.finditer(draft)},
                      key=lambda a: (int(re.match(r"\d+", a).group()), a))
    unsure = len(re.findall(r"da verificare", draft, re.IGNORECASE))
    return (f"{len(draft.split())} words, {len(articles)} article numbers ({', '.join(articles) or 'none'}), "
            f"'da verificare' x{unsure}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--facts-file", help="text file with the user's message")
    source.add_argument("--text", help="the user's message itself")
    parser.add_argument("--rules", choices=("both", "on", "off"), default="both")
    parser.add_argument("--out", default="/tmp/strategy_preview", help="folder for the drafts")
    parser.add_argument("--lang", default="it")
    args = parser.parse_args()

    if args.facts_file:
        with open(args.facts_file, encoding="utf-8") as f:
            message = f.read().strip()
        name = os.path.splitext(os.path.basename(args.facts_file))[0]
    else:
        message = args.text.strip()
        name = "text"
    os.makedirs(args.out, exist_ok=True)

    from src.rag.defensive_generation import run_defensive_pipeline

    print(f"Other flags from .env: APPLICABILITY_CHECK_ENABLED={os.getenv('APPLICABILITY_CHECK_ENABLED', '(unset)')}, "
          f"ISSUE_SPOTTING_ENABLED={os.getenv('ISSUE_SPOTTING_ENABLED', '(unset)')}")
    for rules in (("off", "on") if args.rules == "both" else (args.rules,)):
        os.environ[FLAG] = "true" if rules == "on" else "false"
        t0 = time.time()
        draft = run_defensive_pipeline(message, message, args.lang).get("draft") or ""
        path = os.path.join(args.out, f"{name}-rules-{rules}.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(draft + "\n")
        print(f"\nrules {rules}: {time.time() - t0:.0f} s, {_summary(draft)}\n  saved in {path}")
    print("\nDone. Nothing was written to the database.")


if __name__ == "__main__":
    main()
