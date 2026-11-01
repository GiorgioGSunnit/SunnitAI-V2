"""Read-only, no model, no database: which pareri of a diagnose_long run were written from a plan, and why the others were not.

    venv/bin/python scripts/plan_fallback_report.py diagnose_long_runs/model-qwen3 --llm-reply /tmp/llm_reply.py \\
        > plan_fallback_qwen3.txt

From each case's cases/<id>.json and llm/NN_<stage>.txt. A case went to one call when it has the event
case_analysis_draft_llm: a dump named "draft" proves nothing, it is the stage of any main-thread call without one of its
own. Per case not written from a plan: the cause, the exact step and error (plan_then_write_fallback, in runs after 1
Nov; before it, what the section dumps show), each plan call and section call's finish_reason, the first 300 characters
of the raw plan reply and the parse error. A step that failed safely is listed too (post_writing_step_failed): it never
causes a fallback. With --llm-reply (src/rag/llm_reply.py) also whether that parser reads what today's cannot.
"""

import argparse
import glob
import importlib.util
import json
import os
import re
import sys
from collections import Counter


def load_parser(path: str):
    spec = importlib.util.spec_from_file_location("llm_reply_new", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.parse_json_reply


def parse_today(raw: str):
    """ai_chat.parse_json_reply as deployed before llm_reply.py: (value, error)."""
    text = re.sub(r"```(?:json)?\s*", "", raw or "").strip().rstrip("`").strip()
    try:
        return json.loads(text), ""
    except ValueError as exc:
        error = f"{type(exc).__name__}: {exc}"
    decoder = json.JSONDecoder()
    for i, char in enumerate(text):
        if char in "[{":
            try:
                return decoder.raw_decode(text, i)[0], ""
            except ValueError:
                continue
    return None, f"no JSON value ({error})"


def dump_parts(text: str) -> tuple:
    """({header: value}, reply as it arrived)."""
    head, _, rest = text.partition("\n\n===== ")
    fields = dict(line.split(": ", 1) for line in head.splitlines() if ": " in line)
    raw = rest.split("===== RAW REPLY =====\n", 1)
    reply = raw[1] if len(raw) > 1 else rest.split("===== REPLY =====\n", 1)[-1]
    return fields, reply.rstrip("\n")


def shape(reply: str) -> str:
    text = reply.strip()
    if not text:
        return "empty reply"
    if text == "(no reply)":
        return "no reply"
    tags = []
    if re.search(r"<think>|</think>", text, re.I):
        tags.append("think block")
    if "```" in text:
        tags.append("fence")
    if not re.match(r"(?:<think>.*?</think>\s*)?(?:```(?:json)?\s*)?[\[{]", text, re.DOTALL | re.I):
        tags.append("prose first")
    return ", ".join(tags) or "plain JSON"


def stage_of(path: str) -> str:
    return re.sub(r"^\d+_", "", os.path.basename(path)[:-4])


def is_cut(reply: str) -> bool:
    """A JSON reply that never closes: the parsers then return a complete fragment from inside it."""
    text = re.sub(r"```(?:json)?\s*", "", re.sub(r"<think>.*?</think>", "", reply or "", flags=re.DOTALL)).strip()
    text = text.rstrip("`").strip()
    if text[:1] not in ("{", "["):
        return False
    try:
        json.loads(text)
        return False
    except ValueError as exc:
        return "Extra data" not in str(exc)


def named(events: list, name: str) -> list:
    return [e[1] for e in events if isinstance(e, (list, tuple)) and len(e) > 1 and e[0] == name and isinstance(e[1], dict)]


def case_report(json_path: str, new_parse=None) -> dict:
    case_id = os.path.basename(json_path)[:-5]
    with open(json_path, encoding="utf-8") as fh:
        case = json.load(fh)
    events = case.get("events") or [] if isinstance(case, dict) else []
    plan_events = named(events, "case_analysis_plan")
    dumps = sorted(glob.glob(os.path.join(json_path[:-5], "llm", "*.txt")))
    stages = Counter(stage_of(p) for p in dumps)
    calls, sections = [], []
    for path in dumps:
        stage = stage_of(path)
        if stage not in ("plan", "plan_json_retry") and not stage.startswith("section_"):
            continue
        with open(path, encoding="utf-8") as fh:
            fields, reply = dump_parts(fh.read())
        if stage.startswith("section_"):
            sections.append({"stage": stage, "status": fields.get("status", ""), "finish": fields.get("finish_reason", "?"),
                             "max_tokens": fields.get("max_tokens"), "chars": len(reply.strip())})
            continue
        value, error = parse_today(reply)
        fixed = ""
        if error and new_parse:
            try:
                new_parse(reply)
                fixed = "yes"
            except ValueError as exc:
                fixed = f"no: {exc}"
        calls.append({"stage": stage, "status": fields.get("status", ""), "finish": fields.get("finish_reason", "?"),
                      "max_tokens": fields.get("max_tokens"), "chars": len(reply), "shape": shape(reply), "error": error,
                      "cut": fields.get("finish_reason") == "length" or is_cut(reply), "fixed": fixed,
                      "head": " ".join(reply.split())[:300], "tail": " ".join(reply.split())[-80:]})
    issues = max([int(e.get("issues") or 0) for e in plan_events], default=None)
    removed = next((e.get("removed") for e in reversed(plan_events) if e.get("removed")), None)
    one_call = bool(named(events, "case_analysis_draft_llm") or any(e[0] == "case_analysis_draft_llm" for e in events
                                                                    if isinstance(e, (list, tuple)) and e))
    fell_back = named(events, "plan_then_write_fallback")
    safe_failures = named(events, "post_writing_step_failed")
    bad_sections = [s for s in sections if s["status"].startswith("ERROR") or not s["chars"]]
    last = calls[-1] if calls else None
    if not one_call:
        cause = ""
    elif not calls:
        cause = "no plan call"
    elif last["status"].startswith("ERROR"):
        cause = "plan call failed"
    elif last["cut"]:
        cause = "plan reply cut at max_tokens"
    elif last["error"]:
        cause = "plan reply has no JSON"
    elif issues == 0:
        cause = "plan complete, no usable issue"
    elif fell_back:
        cause = f"after the plan: {fell_back[-1].get('step')}"
    elif bad_sections:
        cause = "after the plan: section call"
    elif sections:
        cause = "after the plan: sections written, then an empty section (run without plan_then_write_fallback)"
    else:
        cause = "after the plan: no section call, unknown step (run without plan_then_write_fallback)"
    return {"id": case_id, "cause": cause, "one_call": one_call, "issues": issues, "removed": removed, "calls": calls,
            "sections": sections, "fell_back": fell_back, "safe_failures": safe_failures, "stages": stages}


def report(folder: str, new_parse=None) -> list:
    paths = sorted(glob.glob(os.path.join(folder, "cases", "*.json")))
    cases = [case_report(p, new_parse) for p in paths]
    listed = [c for c in cases if c["cause"]]
    causes = Counter(c["cause"] for c in listed)
    L = [f"{folder}: {len(cases)} cases: {len(cases) - len(listed)} written from a plan, {len(listed)} in one call; dumps: "
         f"{sum(c['stages'].get('plan', 0) for c in cases)} plan, "
         f"{sum(c['stages'].get('plan_json_retry', 0) for c in cases)} plan_json_retry, "
         f"{sum(c['stages'].get('draft', 0) for c in cases)} labelled draft (not all one-call drafts)", ""]
    L += [f"{n:>4}  {cause}" for cause, n in causes.most_common()]
    shapes = Counter(call["shape"] for c in cases for call in c["calls"])
    L += ["", "Plan replies by shape, every case: " + ", ".join(f"{k} {n}" for k, n in shapes.most_common())]
    finishes = Counter(s["finish"] for c in cases for s in c["sections"])
    L.append("Section calls by finish_reason, every case: " + ", ".join(f"{k} {n}" for k, n in finishes.most_common()))
    failed = Counter(f"{e.get('step')}: {str(e.get('error'))[:100]}" for c in cases for e in c["safe_failures"])
    L += ["", "Steps that failed safely, the sections kept (no fallback), every case:"]
    L += [f"{n:>4}  {what}" for what, n in failed.most_common()] or ["   none"]
    for c in listed:
        L += ["", f"--- {c['id']}: {c['cause']} (sections {len(c['sections'])}, issues {c['issues']})"]
        for entry in c["fell_back"]:
            L.append(f"  fallback at {entry.get('step')}: {str(entry.get('error'))[:300]}")
        if c["removed"]:
            L.append(f"  removed by clean_plan: {str(c['removed'])[:300]}")
        for entry in c["safe_failures"]:
            L.append(f"  failed safely: {entry.get('step')}: {str(entry.get('error'))[:200]}")
        if not c["calls"]:
            L.append("  stages: " + ", ".join(f"{k} {n}" for k, n in sorted(c["stages"].items()) if not k.startswith("section_"))[:400])
        for call in c["calls"]:
            L.append(f"  {call['stage']}: {call['status'][:120]}; finish {call['finish']}; max_tokens {call['max_tokens']}; "
                     f"{call['chars']} chars; {call['shape']}{'; cut' if call['cut'] else ''}")
            L.append(f"    error: {call['error'] or 'none'}" + (f"; new parser reads it: {call['fixed']}" if call["fixed"] else ""))
            L.append(f"    head: {call['head']}")
            if call["error"] or call["cut"]:
                L.append(f"    tail: {call['tail']}")
        for section in c["sections"]:
            if section["status"].startswith("ERROR") or not section["chars"] or section["finish"] == "length":
                L.append(f"  {section['stage']}: {section['status'][:160]}; finish {section['finish']}; "
                         f"max_tokens {section['max_tokens']}; {section['chars']} chars")
    return L


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run", help="the run folder, e.g. diagnose_long_runs/model-qwen3")
    parser.add_argument("--llm-reply", help="the new src/rag/llm_reply.py, to see whether it reads what today's cannot")
    args = parser.parse_args()
    if not glob.glob(os.path.join(args.run, "cases", "*.json")):
        sys.exit(f"no cases/*.json under {args.run}")
    print("\n".join(report(args.run, load_parser(args.llm_reply) if args.llm_reply else None)))


if __name__ == "__main__":
    main()
