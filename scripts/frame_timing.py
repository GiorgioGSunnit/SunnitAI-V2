#!/usr/bin/env python3
"""v79.14.0: the case frame's timing (brief item 6: one call, at most 15 s).

Runs the real step (src/rag/case_frame.read: the one call with thinking off, temperature 0, a JSON reply, the
deadline, then the verification) on a few synthetic request texts (invented parties and facts, no real case) and prints
each call's seconds, its status, the fields verified and the ones left to the fallback, and the frame.

  python scripts/frame_timing.py                       the configured model (the env's LLM_* settings, as the service
                                                       reads them: run it on the server; no setting is overridden)
  python scripts/frame_timing.py --stub-latency 0      the code alone: a stand-in model answering at once
  python scripts/frame_timing.py --stub-latency 3      a stand-in model taking 3 s per call
  python scripts/frame_timing.py --repeat 3 --json out.json
  python scripts/frame_timing.py --frame               also print each frame field by field

Never run by the tests (it calls the model unless --stub-latency is given).
"""

import argparse
import json
import os
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# synthetic request texts: invented names and facts, the brief's five shapes
TEXTS = {
    "debitore": ("Assisto la società Orsa S.r.l., che ha stipulato con Delfino S.p.A. un contratto di fornitura di "
                 "componenti. Delfino pretende ora il pagamento di una penale di 18.000 euro per un ritardo nella "
                 "consegna e ha comunicato la risoluzione del contratto per inadempimento. Orsa sostiene che il "
                 "ritardo è dipeso da un ordine modificato da Delfino all'ultimo momento. Quali strategie difensive "
                 "può adottare Orsa? La penale è riducibile?"),
    "amministratore": ("Il mio cliente, Marco Lince, amministratore della società Gabbiano S.r.l., ha ricevuto un "
                       "avviso di garanzia: la Procura indaga per infedeltà patrimoniale per una cessione di beni "
                       "sociali a una società collegata a prezzo inferiore al valore. Quali sono i rischi penali per "
                       "il signor Lince e come impostare la difesa nella fase delle indagini?"),
    "due_parti": ("Anna Pino ha prestato 9.000 euro a Bruno Faggio con una scrittura privata; Faggio non ha restituito "
                  "la somma alla scadenza e sostiene che si trattava di una donazione. Assisto Anna Pino. Come può "
                  "recuperare la somma?"),
    "a_carico": ("Assisto Carla Ulivo, che ha subito un'aggressione davanti a un locale da parte di Dario Cedro, "
                 "riportando lesioni guaribili in trenta giorni. Quali reati sono configurabili a carico di Dario "
                 "Cedro? Carla Ulivo può costituirsi parte civile?"),
    "nessun_cliente": ("Tra la società Volpe S.r.l. e la società Lupo S.r.l. è sorta una controversia sul pagamento "
                       "di alcune fatture per lavori di manutenzione. Si chiede una valutazione della vicenda."),
}


def stub_call(latency: float):
    """A stand-in model: after `latency` seconds, a frame quoting the text itself (its first sentence as every perché,
    its questions verbatim) -- the verification's cost on a full reply, never a judgment of the case."""
    import re

    def call(messages, kwargs):
        time.sleep(latency)
        text = messages[-1].content.split("\n", 1)[1]
        sentences = [x.strip() for x in re.split(r"(?<=[.?!])\s+", text) if x.strip()]
        first = sentences[0].rstrip(".")
        why = f"«{first}»"
        return json.dumps({"parti": [], "cliente": None,
                           "ruolo": {"valore": "sconosciuto", "perche": why},
                           "rito": {"valore": "sconosciuto", "perche": why},
                           "stadio": {"valore": "sconosciuto", "perche": why},
                           "addebito_o_domanda": None, "istituto": None,
                           "quesiti": [x for x in sentences if x.endswith("?")]}, ensure_ascii=False)
    return call


def main(argv=None) -> int:
    ap_ = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--stub-latency", type=float, default=None,
                     help="seconds per call of a stand-in model (no model is called)")
    ap_.add_argument("--repeat", type=int, default=1, help="runs per text (default 1)")
    ap_.add_argument("--only", nargs="*", help="text ids")
    ap_.add_argument("--frame", action="store_true", help="print each frame field by field")
    ap_.add_argument("--json", default="", help="write the results here")
    args = ap_.parse_args(argv)
    from src.rag import case_frame
    call = stub_call(args.stub_latency) if args.stub_latency is not None else None
    cfg = case_frame.config()
    print(f"case frame timing: {len(args.only or TEXTS)} texts x {args.repeat}, deadline {cfg['deadline_s']} s, "
          f"max_tokens {cfg['max_tokens']}, prompt {case_frame.prompt_hash()}, model "
          f"{'stub %.1f s/call' % args.stub_latency if call else os.getenv('LLM_MODEL', '?')}")
    out, seconds = [], []
    for key, text in TEXTS.items():
        if args.only and key not in args.only:
            continue
        for run in range(max(1, args.repeat)):
            t0 = time.time()
            res = case_frame.read(text, call=call)
            wall = time.time() - t0
            seconds.append(res["seconds"])
            print(f"  {key} #{run + 1}: {res['seconds']:.2f} s (wall {wall:.2f} s), {res['status']}"
                  + (f" ({res['error']})" if res["error"] else "")
                  + f"; verified {', '.join(res['verified']) or '—'}; fallback {', '.join(res['fallback']) or '—'}")
            for row in res["log"][:8]:
                print(f"      unverified {row['field']}: {row['reason']}")
            if args.frame:
                print("\n".join(f"      {line}" for line in case_frame.printable(res)))
            out.append({"text": key, "run": run + 1, "wall_s": wall, **{k: res[k] for k in (
                "seconds", "status", "error", "verified", "fallback", "counters", "log", "frame")}})
    if seconds:
        print(f"per call: median {statistics.median(seconds):.2f} s, max {max(seconds):.2f} s "
              f"(budget {cfg['deadline_s']} s); over budget: {sum(1 for s in seconds if s > cfg['deadline_s'])}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
