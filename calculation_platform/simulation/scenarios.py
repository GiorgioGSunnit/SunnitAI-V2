"""Three hardcoded end-to-end scenarios — the canonical demo of the flow.

Each scenario fixes a known sentence and shows every stage of the pipeline
explicitly, including the two payloads that are normally invisible:

  [1] the simulated LLM's structured call to the platform
      (what the real LLM would POST to /calculate), and
  [2] the platform's structured payload back to the LLM
      (result / steps / citations / warnings, or the structured error
      that drives a clarifying question).

Scenario 1: clear match, all inputs present -> straight to the result.
Scenario 2: clear match, one required variable missing -> the platform's
            structured error becomes a question; the follow-up answer
            completes the calculation.
Scenario 3: ambiguous sentence -> the user is asked to choose between
            (at most) 3 named formulas.

Run from the repo root:

    .venv/bin/python calculation_platform/simulation/scenarios.py
"""

import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.main import engine  # noqa: E402
from simulation.conversation import SimulatedConversation  # noqa: E402

SCENARIO_1_SENTENCE = "quanto pago di tasse su un reddito di 42000 euro nel 2026"
SCENARIO_2_SENTENCES = ["quanto pago di irpef nel 2026?", "il reddito è di 42000 euro"]
SCENARIO_3_SENTENCE = "quanto pago di interessi"


def _dump(payload) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def _play(title: str, sentences) -> None:
    print("=" * 72)
    print(title)
    print("=" * 72)
    conversation = SimulatedConversation(engine)
    for sentence in sentences:
        print(f'\nutente> "{sentence}"')
        reply = conversation.send(sentence)
        if reply.plan is not None:
            print("\n[0] Piano del router simulato:")
            print(_dump(reply.plan.model_dump()))
        if reply.tool_call is not None:
            print("\n[1] Chiamata simulata LLM -> piattaforma (POST /calculate):")
            print(_dump(asdict(reply.tool_call)))
        if reply.calculation is not None:
            print("\n[2] Payload della piattaforma -> LLM:")
            print(_dump(reply.calculation.model_dump()))
        print(f"\n[{reply.kind}] risposta mostrata all'utente:")
        print(reply.text)
    print()


def main() -> None:
    _play(
        "SCENARIO 1 — riconoscimento diretto: formula certa, input completi",
        [SCENARIO_1_SENTENCE],
    )
    _play(
        "SCENARIO 2 — variabile mancante: la piattaforma chiede, l'utente risponde",
        SCENARIO_2_SENTENCES,
    )
    _play(
        "SCENARIO 3 — ambiguita': scelta tra 3 formule",
        [SCENARIO_3_SENTENCE],
    )


if __name__ == "__main__":
    main()
