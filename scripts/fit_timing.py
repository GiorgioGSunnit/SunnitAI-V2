#!/usr/bin/env python3
"""v79.13.0: the applicability judgment's timing (brief item 6: at most 20 s added per answer at 30 candidates).

Builds 30 synthetic candidate norms (invented articles 9001-9030 c.c., two or three commi each, realistic length) and a
synthetic case (six facts, certi and allegazioni), then runs the real batched step (src/rag/applicability.judge: batches
of at most 8, the bounded concurrency, the deadline, the verification) and prints the total seconds, each batch's
seconds and status, and the counters.

  python scripts/fit_timing.py                       the configured model (the env's LLM_* settings, as the service
                                                     reads them: run it on the server; no setting is overridden)
  python scripts/fit_timing.py --stub-latency 0      the code alone: a stand-in model answering at once
  python scripts/fit_timing.py --stub-latency 15     a stand-in model taking 15 s per call: shows the concurrency bound
  python scripts/fit_timing.py --n 30 --json out.json
  python scripts/fit_timing.py --flow --stub-latency 5
                                                     the step as the flow runs it (interactive_flow.applicability_fit:
                                                     the candidates' inputs, the reviewed notes, the judgment, the
                                                     effects and the counters) on a synthetic answer state; --repeat
                                                     adds a second, warm run

Never run by the tests (it calls the model unless --stub-latency is given).
"""

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

_SUBJECTS = ("il conduttore", "il depositario", "il mandatario", "il committente", "l'appaltatore", "il venditore")
_OBJECTS = ("la cosa locata", "il bene ricevuto in custodia", "la somma ricevuta", "l'opera commissionata",
            "il veicolo consegnato", "la merce venduta")


def synthetic_norms(n: int = 30):
    """n invented articles (9001 c.c. on): numbered commi of plausible length, a rubrica each."""
    out = []
    for i in range(n):
        s, o = _SUBJECTS[i % len(_SUBJECTS)], _OBJECTS[(i // 2) % len(_OBJECTS)]
        commi = [f"1. {s.capitalize()} che non restituisce {o} entro il termine convenuto è tenuto al risarcimento del "
                 f"danno, salvo che provi che l'inadempimento è derivato da causa a lui non imputabile.",
                 f"2. Se {o} è stata danneggiata durante il rapporto, {s} risponde del deterioramento, a meno che provi "
                 f"che esso è avvenuto per vetustà o per caso fortuito; la prova grava su di lui.",
                 f"3. Le disposizioni del presente articolo si applicano anche quando {o} è stata affidata a un terzo "
                 f"senza il consenso dell'altra parte, ferma la responsabilità solidale del terzo."][: 2 + i % 2]
        out.append({"article": f"{9001 + i} c.c.", "rubrica": f"Obblighi di restituzione ({i + 1})",
                    "text": "\n".join(commi)})
    return out


SYNTHETIC_FACTS = [
    {"id": "S1", "text": "Il cliente ha consegnato a Delta un veicolo in custodia il 3 marzo.", "certainty": "accertato"},
    {"id": "S2", "text": "Delta non ha restituito il veicolo entro il termine convenuto del 10 marzo.",
     "certainty": "accertato"},
    {"id": "S3", "text": "Delta sostiene che il veicolo è stato danneggiato per caso fortuito.", "certainty": "allegato"},
    {"id": "S4", "text": "Il veicolo è stato affidato da Delta a un terzo senza il consenso del cliente.",
     "certainty": "accertato"},
    {"id": "S5", "text": "Il cliente afferma di aver subito un danno di 2.000 euro.", "certainty": "allegato"},
    {"id": "S6", "text": "Il cliente chiede il risarcimento del danno.", "certainty": "accertato"},
]


def stub_call(latency: float):
    """A stand-in model: after `latency` seconds, a verifiable «si» for every norm of the batch."""
    from src.rag import applicability as ap

    def call(messages, kwargs):
        time.sleep(latency)
        human = messages[-1].content
        labels = [line.split(": ", 1)[1] for line in human.splitlines() if line.startswith("ARTICOLO: ")]
        items = [{"articolo": label, "fit": "si", "ambito": "rimedio", "condizione": "",       # v79.15.0: by ids
                  "perche": "I fatti (S2): «non restituisce».", "cita_norma": [f"A{ap.article_token(label)}.1"],
                  "cita_fatti": ["S2"]} for label in labels]
        return json.dumps({"giudizi": items}, ensure_ascii=False)
    return call


def flow_state(norms):
    """v79.13.0 (build B): a synthetic answer state -- the norms as records (one element each) and rows, the facts as
    the case's sentences -- for interactive_flow.applicability_fit."""
    records = {n["article"]: {"article": n["article"], "rubrica": n["rubrica"], "side": "weapon", "requisites": [
        {"id": f"R{i}", "comma": 1, "kind": "requisito", "status": "da provare", "text": "non restituisce"}]}
        for i, n in enumerate(norms, 1)}
    rows = [{"_article": tuple(n["article"].split(" ", 1)), "s": {"plain_text": n["text"],
                                                                  "titolo_articolo": n["rubrica"]}} for n in norms]
    state = {"records": records, "rows": [], "sentences": [dict(f, source="caso") for f in SYNTHETIC_FACTS],
             "text": " ".join(f["text"] for f in SYNTHETIC_FACTS), "request": "", "intake": {}, "analysis": {},
             "topics": [{"title": "Restituzione", "articles": list(records), "anchor": norms[0]["article"]}]}
    return state, rows


def main(argv=None) -> int:
    ap_ = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--n", type=int, default=30, help="candidate norms (default 30)")
    ap_.add_argument("--role", default="attore")
    ap_.add_argument("--stub-latency", type=float, default=None,
                     help="seconds per call of a stand-in model (no model is called)")
    ap_.add_argument("--json", default="", help="write the result here")
    ap_.add_argument("--repeat", action="store_true", help="with --flow: also a second, warm run")
    ap_.add_argument("--flow", action="store_true",
                     help="the step as the flow runs it (interactive_flow.applicability_fit) on a synthetic state")
    args = ap_.parse_args(argv)
    from src.rag import applicability
    call = stub_call(args.stub_latency) if args.stub_latency is not None else None
    cfg = applicability.config()
    print(f"applicability timing: {args.n} candidates, batch_size {cfg['batch_size']}, concurrency "
          f"{cfg['concurrency']}, deadline {cfg['deadline_s']} s, prompt {applicability.prompt_hash()}, model "
          f"{'stub %.1f s/call' % args.stub_latency if call else os.getenv('LLM_MODEL', '?')}")
    t0 = time.time()
    if args.flow:               # the flow's own step: inputs, notes, judge, effects, counters
        from src.rag import interactive_flow
        state, rows = flow_state(synthetic_norms(args.n))
        if call is not None:
            applicability._default_call = call
        interactive_flow.applicability_fit(state, {"role": args.role}, rows, set(), [])
        wall = time.time() - t0
        if args.repeat:         # warm: the modules and the notes read once already
            state, rows = flow_state(synthetic_norms(args.n))
            t1 = time.time()
            interactive_flow.applicability_fit(state, {"role": args.role}, rows, set(), [])
            print(f"flow step (cold): {wall:.2f} s; (warm, second run): {time.time() - t1:.2f} s")
            wall = time.time() - t1
        res = {"timing": (state.get("applicability") or {}).get("timing") or {"total_s": 0, "calls": []},
               "counters": {k: (state.get("block_counters") or {}).get(k) for k in applicability.FLOW_COUNTERS},
               "log": [{"article": j["article"], "reason": j["reason"]}
                       for j in (state.get("applicability") or {}).get("judgments") or [] if not j.get("verified")]}
        print(f"flow step: {wall:.2f} s wall (the judgment {res['timing'].get('total_s')} s)")
    else:
        res = applicability.judge(synthetic_norms(args.n), SYNTHETIC_FACTS, args.role, call=call)
        wall = time.time() - t0
    for c in res["timing"]["calls"]:
        print(f"  batch {c['batch']}: {c['norms']} norms, {c['seconds']:.2f} s, {c['status']}"
              + (f" ({c['error']})" if c["error"] else ""))
    slowest = max((c["seconds"] for c in res["timing"]["calls"]), default=0.0)
    print(f"total: {res['timing']['total_s']:.2f} s (wall {wall:.2f} s; slowest call {slowest:.2f} s; code overhead "
          f"{max(0.0, wall - slowest):.3f} s)")
    print("counters: " + ", ".join(f"{k}={v}" for k, v in res["counters"].items()))
    for row in res["log"][:10]:
        print(f"  unverified {row['article']}: {row['reason']}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"wall_s": wall, **res}, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
