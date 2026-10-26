"""Read-only, seconds: what FACT_SIGNALS / FACT_REGIMES read in a case and why each term and rule did or did not
come out. One extraction call (fact_signals.extract_facts), then:

  1. the model's reply as it came, and the FULL typed facts after validation (roles from the list, qualified
     roles only when the text states them, event kinds, dates kept only as written);
  2. the parties' qualifications (FACT_REGIMES), when --regimes or the flag is on;
  3. every legal term: whether it applies, the facts it runs from and to and how they were matched (by event kind
     or by words), its status;
  4. every rule of fact_signal_rules.yaml: each condition matched or not, with the fact it used.

No database, no pipeline, nothing written (unless --save).

Run from the app folder, with its venv and .env (for the model's key):
    cd /opt/chatbot && venv/bin/python scripts/fact_probe.py --file "<case>.docx" --request "<the message>"
    cd /opt/chatbot && venv/bin/python scripts/fact_probe.py --text "<the case as a chat message>"
    ... --areas diritti_reali,civile   # the frame's areas, as the pipeline would pass them (default: none, no filter)
    ... --today 2019-05-01             # the case's present when it states none (else terms run to the latest fact)
    ... --regimes                      # the FACT_REGIMES schema and qualifications even with the flag off
    ... --save probe.json              # the reply, the facts, the terms and the rules as JSON
"""

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (ROOT, os.getcwd()):
    if _path not in sys.path:
        sys.path.insert(0, _path)


def _mark(ok: bool) -> str:
    return "MATCHED  " if ok else "unmatched"


def report(fs, case_text: str, areas: list, raw_replies: list, facts, problem: str) -> tuple:
    """(lines, data): the probe's printout and its JSON, from an extraction already made."""
    lines = [f"Extraction: {'ok' if facts else 'FAILED'}" + (f" ({problem})" if problem else ""),
             f"Areas given: {', '.join(areas) or 'none (no family filter)'}", "",
             "## Model reply (as it came)"] + [r if isinstance(r, str) else json.dumps(r) for r in raw_replies]
    data = {"areas": areas, "raw_replies": raw_replies, "problem": problem, "facts": facts}
    if not facts:
        return lines, data
    qualifications = []
    if fs.regimes_enabled():
        facts, qualifications = fs.qualify_parties(facts)
    lines += ["", "## Typed facts (full)", json.dumps(facts, ensure_ascii=False, indent=1)]
    if fs.regimes_enabled():
        lines += ["", "## Qualifications (FACT_REGIMES)"] + (
            [f"- {q['party']}: {q['quality']} ({q['why']})" for q in qualifications] or ["(none)"])
    terms = fs.explain_terms(facts, areas)
    lines += ["", "## Terms"]
    for t in terms:
        entry = t["entry"]
        lines.append(f"- {t['id']}: " + (f"{entry['status']} -- {entry['text']}" if entry else "not computed"))
        lines += [f"    {w}" for w in t["why"]]
    computed = [t["entry"] for t in terms if t["entry"]]
    lines += ["", "## Rules"]
    rules = fs.explain_rules(facts, computed, areas=areas)
    for r in rules:
        state = "FIRED" if r["fired"] else ("skipped: " + r["skipped"] if r["skipped"] else "not fired")
        lines.append(f"- {r['id']}: {state}")
        for c in r["conditions"]:
            lines.append(f"    {_mark(c['matched'])} {c['text']}" + (f" -- «{c['fact']}»" if c["fact"] else ""))
    fired = [r["id"] for r in rules if r["fired"]]
    lines += ["", f"Terms computed: {len(computed)} ({', '.join(t['id'] for t in computed) or 'none'}). "
                  f"Rules fired: {len(fired)} ({', '.join(fired) or 'none'})."]
    data.update(facts=facts, qualifications=qualifications, terms=terms, rules=rules)
    return lines, data


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="One fact extraction, every term and rule explained.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--file", help="a local .pdf/.docx/.txt, read as an upload is")
    source.add_argument("--text", help="the case as a chat message")
    parser.add_argument("--request", default="", help="the user's message sent with --file")
    parser.add_argument("--areas", default="", help="comma-separated frame areas (default: none, no filter)")
    parser.add_argument("--today", help="YYYY-MM-DD, the case's present when it states none")
    parser.add_argument("--regimes", action="store_true", help="the FACT_REGIMES schema and qualifications")
    parser.add_argument("--env", default=os.path.join(os.getcwd(), ".env"))
    parser.add_argument("--save", help="write the reply, facts, terms and rules as JSON here")
    args = parser.parse_args(argv)

    from dotenv import load_dotenv
    load_dotenv(args.env)
    if args.regimes:
        os.environ["FACT_REGIMES_ENABLED"] = "true"
    from src.rag import fact_signals as fs
    if args.today:
        os.environ[fs.TODAY_ENV] = args.today
    if args.file:
        from src.utils.document import extract_text_from_file
        text = extract_text_from_file(args.file)
    else:
        text = args.text
    from src.rag.defensive_generation import _case_text
    case_text = _case_text(text, args.request) if args.file else text
    areas = [a.strip() for a in args.areas.split(",") if a.strip()]

    raw_replies: list = []
    t0 = time.time()
    facts, problem = fs.extract_facts(case_text, raw_replies)
    lines, data = report(fs, case_text, areas, raw_replies, facts, problem)
    print("\n".join(lines))
    print(f"\n{time.time() - t0:.1f}s, one extraction call. Nothing was written"
          + (f" but {args.save}." if args.save else "."))
    if args.save:
        with open(args.save, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=1, default=str)


if __name__ == "__main__":
    main()
