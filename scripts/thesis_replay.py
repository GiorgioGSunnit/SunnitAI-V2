#!/usr/bin/env python3
"""v71: the thesis call's A/B -- logs/thesis_<case>.json (lawyer_method.thesis_log: the exact system and user messages,
max_tokens, the thinking setting, the reply, the record's elements and the assessment) sent again, unchanged, to another
model, and both replies checked the same way: each thesis's declared ids against the logged elements (a thesis
contesting an «integrato» element barred), the main line that results, the declared ids the record does not hold.

  python scripts/thesis_replay.py --log logs/thesis_sempronio_no_hint.json --base-url http://host:port/v1 --model NAME
  python scripts/thesis_replay.py --log ... --base-url ... --model ... --thinking off --out ab.json
  python scripts/thesis_replay.py --log ... --base-url ... --model ... --reasoning-effort low
  python scripts/thesis_replay.py --log ... --base-url ... --model ... --deterministic

v79 (A3): the reply kept is lawyer_method.choose's (the score, then the smallest digest of the parsed reply -- never the
order of arrival), both logged; --deterministic sends one sample at temperature 0, for a version-to-version comparison.

v78.1.1: with thinking on, reasoning_effort goes beside enable_thinking -- --reasoning-effort, else LLM_REASONING_EFFORT,
else none; the replay's output records it beside the log's own.

--base-url and --model are required (no default endpoint); the key from --api-key-env (default LLM_API_KEY).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"),
                   ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)
os.environ.setdefault("LLM_API_KEY", os.environ.get("LLM_API_KEY", "k"))


def records_from(elements: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """The logged elements back into records (one comma each: the statuses as logged)."""
    records: Dict[str, Dict[str, Any]] = {}
    for eid, e in elements.items():
        rec = records.setdefault(e.get("article") or "?", {"requisites": [], "side": e.get("side") or ""})
        rec["requisites"].append({"id": eid, "comma": 1, "kind": "requisito", "text": e.get("text") or "",
                                  "status": e.get("status") or "da provare"})
    return records


def check(plan_raw: Any, log: Dict[str, Any]) -> Dict[str, Any]:
    """The reply as the answer step reads it: v72 -- the logged candidates with the reply's valuations (a thesis it
    adds dropped), its score (lawyer_method.score_sample); a v71 log (no candidates): its own theses, ids kept when the
    record holds them. Then the coherence check."""
    from src.rag import lawyer_method as M
    known = log.get("elements") or {}
    candidates = log.get("candidates") or None
    unknown: List[str] = []
    clean_log: Dict[str, Any] = {}
    plan = M.clean_method(plan_raw, candidates, clean_log)
    if candidates is None:
        for s in plan["soluzioni"]:
            s["contesta"] = M.declared_ids(s.get("contesta"), known, unknown)
            s["poggia_su"] = M.declared_ids(s.get("poggia_su"), known, unknown)
    score = M.score_sample(plan_raw, candidates, known) if candidates else (None, [])
    sols, main = M.coherent_solutions(plan, records_from(known), log.get("assessment") or {}, (), None,
                                      log.get("posture") or {})
    return {"plan": plan, "main": sols[main]["tesi"],
            "main_is_fallback": bool(sols[main].get("fallback")) and not sols[main].get("candidate"),
            "barred": [{"tesi": s["tesi"], "contesta": s["barred"]} for s in sols if s.get("barred")],
            "unknown_ids": unknown, "theses": len(plan["soluzioni"]), "dropped": clean_log.get("dropped") or [],
            "score": score[0] if score[0] != float("-inf") else None, "score_why": score[1],
            "declared": sum(1 for s in plan["soluzioni"] if s.get("contesta") or s.get("poggia_su"))}


def template_kwargs(thinking: str, effort: str = "") -> Dict[str, Any]:
    """The extra body: none for «default»; enable_thinking, and reasoning_effort when thinking is on and one is given."""
    if thinking == "default":
        return {}
    kwargs: Dict[str, Any] = {"enable_thinking": thinking == "on"}
    if thinking == "on" and effort:
        kwargs["reasoning_effort"] = effort
    return {"chat_template_kwargs": kwargs}


def call(log: Dict[str, Any], base_url: str, model: str, api_key: str, thinking: str, max_tokens: int,
         n: int = 1, temperature: float = 0.0, effort: str = "") -> Dict[str, Any]:
    """The logged messages to the other model; v72: n replies (the log's best-of-n at its temperature), the best by
    the code's score kept."""
    from openai import OpenAI
    from src.rag.llm_reply import parse_json_reply
    from src.rag import lawyer_method as M
    client = OpenAI(base_url=base_url, api_key=api_key)
    extra = template_kwargs(thinking, effort)
    t0 = time.time()
    reply = client.chat.completions.create(model=model, messages=log["messages"], max_tokens=max_tokens,
                                           temperature=temperature, n=n, **({"extra_body": extra} if extra else {}))
    texts = [c.message.content or "" for c in reply.choices]
    samples = []
    for text in texts:
        try:
            parsed = parse_json_reply(text)
        except ValueError:
            parsed = {}
        value, why = M.score_sample(parsed, log.get("candidates") or [], log.get("elements") or {}) \
            if log.get("candidates") else (0.0, [])
        samples.append({"raw_reply": text, "parsed": parsed, "score": None if value == float("-inf") else value,
                        "why": why})
    best, choice = M.choose(samples)
    return {"raw_reply": samples[best]["raw_reply"], "parsed_best": samples[best]["parsed"], "samples": samples,
            "best": best, "choice": choice, "finish_reason": reply.choices[best].finish_reason,
            "seconds": round(time.time() - t0, 1)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--log", required=True, type=Path)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key-env", default="LLM_API_KEY")
    ap.add_argument("--thinking", choices=("on", "off", "default"), default="on")
    ap.add_argument("--reasoning-effort", help="with thinking on: chat_template_kwargs reasoning_effort (overrides "
                                               "LLM_REASONING_EFFORT for this run)")
    ap.add_argument("--max-tokens", type=int, help="default: the logged max_tokens (+ the thinking allowance when on)")
    ap.add_argument("--samples", type=int, help="replies per call (default: the log's n, else 1)")
    ap.add_argument("--deterministic", action="store_true", help="v79: one reply at temperature 0")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    from src.rag import ai_chat
    log = json.loads(args.log.read_text(encoding="utf-8"))
    tokens = args.max_tokens or int(log.get("max_tokens") or 2600) + (
        ai_chat.THINKING_EXTRA_TOKENS if args.thinking == "on" else 0)
    n = 1 if args.deterministic else args.samples or int(log.get("n") or 1)
    if args.reasoning_effort is not None:
        os.environ[ai_chat.REASONING_EFFORT_ENV] = args.reasoning_effort
    effort = ai_chat.reasoning_effort() if args.thinking == "on" else ""
    got = call(log, args.base_url, args.model, os.getenv(args.api_key_env) or "k", args.thinking, tokens, n,
               0.0 if args.deterministic else float(log.get("temperature") or 0.0) if n > 1 else 0.0, effort)
    parsed = got.pop("parsed_best") or {}
    original = check(log.get("parsed") or {}, log)
    replay = check(parsed, log)
    out = {"case_id": log.get("case_id"), "original": {"model": log.get("model"), "seconds": log.get("seconds"),
                                                       "reasoning_effort": log.get("reasoning_effort") or "",
                                                       **original},
           "replay": {"model": args.model, "base_url": args.base_url, "thinking": args.thinking,
                      "reasoning_effort": effort, "max_tokens": tokens, "deterministic": args.deterministic,
                      **got, "parsed": parsed, **replay},
           "same_main": original["main"] == replay["main"]}
    path = args.out or args.log.with_name(f"{args.log.stem}.replay.{args.model.replace('/', '_')}.json")
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for side in ("original", "replay"):
        r = out[side]
        print(f"{side:8} {r['model']}: main «{r['main'][:90]}»{' (fallback)' if r['main_is_fallback'] else ''}; "
              f"barred {len(r['barred'])}/{r['theses']}; declared ids {r['declared']}/{r['theses']}; "
              f"unknown ids {r['unknown_ids']}; {r.get('seconds')}s")
    print(f"same main line: {out['same_main']} -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
