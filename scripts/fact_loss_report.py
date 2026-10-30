"""Read-only: why the expected articles lost at "article resolved" with their trigger captured only as a fact were
lost, from FACT_ISSUES_ENABLED's own logs (case_analysis_fact_issues, case_analysis_fact_institution) and its LLM dump.

    venv/bin/python scripts/fact_loss_report.py fetch_audit_runs/<run>/audit.json diagnose_long_runs/<run> [--all]

The audit.json is fetch_audit's, with the understanding mechanism, over that diagnose_long run. For each lost
article (funnel lost_at "resolved", capture kind "fact"; --all: every lost article whose trigger was captured), the
institutions FACT_ISSUES named from its trigger fact (content words shared with the institution's "fatto", or the
article among what an institution resolved to), and the outcome:

  rejected by the rubrica filter   an institution of that fact resolved to it, keep_resolved refused it (no rubrica word
                                   shared) -- chapter-range fallback
  resolved to other articles       an institution of that fact resolved, to other articles only
  resolved to nothing              an institution of that fact, no article at all
  not named: cut by the cap        the model named it, beyond MAX_INSTITUTIONS (from the dump's reply)
  not named: dropped as a duplicate / with a number   the model named it, clean() dropped it
  not named: cap reached           no institution from that fact, and the model named MAX_INSTITUTIONS
  not named: under the cap         no institution from that fact, fewer than MAX_INSTITUTIONS named
  no FACT_ISSUES log               the flag off or the call failed for that case
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.getcwd())
from src.rag import fact_issues  # noqa: E402

OUTCOMES = ("rejected by the rubrica filter", "resolved to other articles", "resolved to nothing",
            "not named: cut by the cap", "not named: dropped as a duplicate / with a number",
            "not named: cap reached", "not named: under the cap", "no FACT_ISSUES log")
FIX = {"rejected by the rubrica filter": "let the institution resolve within its chapter range",
       "resolved to other articles": "the resolver (rubrica / heading match), not the cap",
       "resolved to nothing": "the resolver: the name does not reach a rubrica",
       "not named: cut by the cap": "raise the cap for long cases",
       "not named: dropped as a duplicate / with a number": "clean(): duplicates of central issues",
       "not named: cap reached": "raise the cap / name institutions for every decisive fact",
       "not named: under the cap": "name institutions for every decisive fact",
       "no FACT_ISSUES log": "FACT_ISSUES did not run"}
_STOP = {"della", "delle", "dello", "degli", "nella", "nelle", "sulla", "dalla", "alla", "alle", "come", "anche",
         "dopo", "prima", "quando", "questo", "questa", "sono", "essere", "stato", "stata", "parte", "tutto"}


def stems(text):
    return {w[:5] for w in re.findall(r"[a-zà-ù]{4,}", str(text or "").lower()) if w not in _STOP}


def same_fact(a, b) -> bool:
    x, y = stems(a), stems(b)
    shared = len(x & y)
    return shared >= 2 or (shared >= 1 and shared >= min(len(x), len(y)) / 2)


def article_keys(label):
    """{(number, code letters)} of a label: "art. 1490 c.c.", "1490 cc", "art. 582/583 cp"."""
    text = str(label or "").lower().replace("art.", " ").replace("artt.", " ")
    m = re.match(r"\s*([\d/\s\-a-z]*?\d[a-z\-]*)\s+(.*)$", text)
    if not m:
        return set()
    code = re.sub(r"[^a-z]", "", m.group(2))[:12]
    return {(n, code) for n in re.findall(r"\d+(?:-[a-z]+)?", m.group(1))}


def has_article(labels, keys) -> bool:
    for label in labels or []:
        for n, code in article_keys(label):
            if any(n == k and (code.startswith(c) or c.startswith(code)) for k, c in keys):
                return True
    return False


def raw_institutions(case_dir):
    """The institutions of the fact_issues call's reply in the case's LLM dump, [] without one."""
    for path in sorted(glob.glob(os.path.join(case_dir, "llm", "*_fact_issues.txt"))):
        text = open(path, encoding="utf-8").read()
        reply = text.split("===== REPLY =====", 1)[-1]
        m = re.search(r"\{.*\}", reply, re.DOTALL)
        try:
            data = json.loads(m.group(0)) if m else {}
        except ValueError:
            continue
        items = data.get("istituti") if isinstance(data, dict) else None
        if isinstance(items, list):
            return [i for i in items if isinstance(i, dict)]
    return []


def classify(lost, rec, case_dir):
    events = [e for e in rec.get("events") or [] if isinstance(e, (list, tuple)) and len(e) >= 2]
    logs = [d for e, d, *_ in events if e == "case_analysis_fact_issues" and isinstance(d, dict)]
    if not logs:
        return "no FACT_ISSUES log", [], {}
    named = [i for i in logs[-1].get("institutions") or [] if isinstance(i, dict)]
    resolutions = {}
    for e, d, *_ in events:
        if e == "case_analysis_fact_institution" and isinstance(d, dict) and d.get("fonte") == fact_issues.FONTE:
            r = resolutions.setdefault(d.get("institution"), {"kept": [], "rejected": []})
            r["kept"] += d.get("kept") or []
            r["rejected"] += d.get("rejected") or []
    keys = article_keys(lost["article"])
    fact = lost.get("fact") or ""
    mine = [i for i in named if same_fact(fact, i.get("fatto")) or has_article(
        (resolutions.get(i.get("nome")) or {}).get("kept", []) + (resolutions.get(i.get("nome")) or {}).get("rejected", []),
        keys)]
    detail = {"named": [i.get("nome") for i in named]}
    if mine:
        res = [resolutions.get(i.get("nome")) or {"kept": [], "rejected": []} for i in mine]
        detail["resolved"] = {i.get("nome"): r for i, r in zip(mine, res)}
        if any(has_article(r["rejected"], keys) for r in res):
            return "rejected by the rubrica filter", mine, detail
        if any(r["kept"] for r in res):
            return "resolved to other articles", mine, detail
        return "resolved to nothing", mine, detail
    raw = raw_institutions(case_dir)
    detail["raw"] = len(raw)
    from_fact = [k for k, i in enumerate(raw) if same_fact(fact, i.get("fatto"))]
    if from_fact:
        kept_names = {" ".join(str(i.get("nome") or "").lower().split()) for i in named}
        if all(k >= fact_issues.MAX_INSTITUTIONS for k in from_fact):
            return "not named: cut by the cap", [], detail
        if all(" ".join(str(raw[k].get("nome") or "").lower().split()) not in kept_names for k in from_fact):
            return "not named: dropped as a duplicate / with a number", [], detail
    if len(named) >= fact_issues.MAX_INSTITUTIONS or len(raw) >= fact_issues.MAX_INSTITUTIONS:
        return "not named: cap reached", [], detail
    return "not named: under the cap", [], detail


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audit", help="fetch_audit's audit.json (understanding mechanism)")
    parser.add_argument("run", help="the diagnose_long run folder it audited")
    parser.add_argument("--all", action="store_true", help="every lost article whose trigger was captured")
    args = parser.parse_args()
    audit = json.load(open(args.audit, encoding="utf-8"))
    run_name = os.path.basename(os.path.normpath(args.run))
    rows = []
    for case in audit.get("cases") or []:
        if case.get("run") not in (run_name, None):
            continue
        for lost in (case.get("understanding") or {}).get("lost") or []:
            if not lost.get("captured"):
                continue
            if not args.all and ((lost.get("funnel") or {}).get("lost_at") != "resolved"
                                 or (lost.get("capture_kind") or "fact") != "fact"):
                continue
            path = os.path.join(args.run, "cases", f"{case['id']}.json")
            if not os.path.exists(path):
                rows.append((case["id"], lost, "no FACT_ISSUES log", [], {"missing": path}))
                continue
            rec = json.load(open(path, encoding="utf-8"))
            outcome, mine, detail = classify(lost, rec, os.path.join(args.run, "cases", case["id"]))
            rows.append((case["id"], lost, outcome, mine, detail))
    if not rows:
        sys.exit(f"no lost article of run {run_name} in {args.audit} (wrong pair, or no understanding mechanism)")
    tally = Counter(outcome for _, _, outcome, _, _ in rows)
    named = sum(1 for _, _, outcome, _, _ in rows if not outcome.startswith(("not named", "no FACT")))
    print(f"# Lost at 'article resolved', trigger captured only as a fact: {len(rows)} (run {run_name})\n")
    print(f"An institution named from that fact: yes {named}, no {len(rows) - named}.\n")
    print("| outcome | articles | share | the general fix it points to |\n|---|---:|---:|---|")
    for outcome in OUTCOMES:
        if tally[outcome]:
            print(f"| {outcome} | {tally[outcome]} | {100 * tally[outcome] / len(rows):.0f}% | {FIX[outcome]} |")
    print("\n| case | article | trigger fact | named from it? | outcome | institutions (of that fact, or all named) |"
          "\n|---|---|---|---|---|---|")
    for cid, lost, outcome, mine, detail in rows:
        insts = "; ".join(f"{i.get('nome')} -> kept {detail['resolved'][i.get('nome')]['kept'] or '-'}, rejected "
                          f"{detail['resolved'][i.get('nome')]['rejected'] or '-'}" for i in mine) if mine else \
            f"{len(detail.get('named') or [])} named (raw reply {detail.get('raw', '-')}): " + \
            ", ".join(detail.get("named") or [])
        fact = " ".join(str(lost.get("fact") or "").split())[:140].replace("|", "/")
        print(f"| {cid} | {lost['article']} | {fact} | {'yes' if mine else 'no'} | {outcome} | "
              f"{insts.replace('|', '/')[:400]} |")


if __name__ == "__main__":
    main()
