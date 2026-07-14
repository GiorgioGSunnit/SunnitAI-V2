"""Interactive demo of the simulated LLM<->platform conversation.

Run from the repo root:

    .venv/bin/python calculation_platform/simulation/demo.py "quanto pago di tasse su un reddito di 42000 euro nel 2026"

or with no argument for an interactive session (Ctrl-D to exit):

    .venv/bin/python calculation_platform/simulation/demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.main import engine  # noqa: E402
from simulation.conversation import SimulatedConversation  # noqa: E402


def main() -> None:
    conversation = SimulatedConversation(engine)

    if len(sys.argv) > 1:
        reply = conversation.send(" ".join(sys.argv[1:]))
        print(f"[{reply.kind}]")
        print(reply.text)
        return

    print("Simulated conversation — type a sentence (Ctrl-D to exit).")
    while True:
        try:
            message = input("\nutente> ").strip()
        except EOFError:
            print()
            return
        if not message:
            continue
        reply = conversation.send(message)
        print(f"\nassistente [{reply.kind}]:\n{reply.text}")


if __name__ == "__main__":
    main()
