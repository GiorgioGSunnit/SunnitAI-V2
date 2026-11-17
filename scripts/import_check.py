#!/usr/bin/env python3
"""v77: the copy imports -- the API and the case-first flow, with the copy's env file loaded (no model, no database
call) -- and prints its VERSION. scripts/deploy.sh runs it on /root/v38 after the rsync, scripts/promote.sh on
/opt/chatbot before the restart. Exit 1 on any import error.

  venv/bin/python scripts/import_check.py --env .env.v45
"""

import argparse
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULES = ("src.chatbot.api", "src.rag.interactive_flow", "src.rag.lawyer_method", "src.rag.intake",
           "src.chatbot.heldout")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--env", default=".env", help="the copy's env file (loaded when present)")
    args = ap.parse_args(argv)
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    if os.path.isfile(args.env):
        from dotenv import load_dotenv
        load_dotenv(args.env)
    import importlib
    for name in MODULES:
        try:
            importlib.import_module(name)
        except Exception:           # noqa: BLE001 -- any failure is the check's answer
            print(f"import check FAILED on {name}:", file=sys.stderr)
            traceback.print_exc()
            return 1
    from src import state_paths, version
    print(f"import ok: {ROOT} version {version.current()} state {state_paths.state_dir()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
