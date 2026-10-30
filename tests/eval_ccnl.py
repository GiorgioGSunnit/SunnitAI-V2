"""The CCNL path's eval (tests/eval_ccnl_cases.json), in process: each question through the chat's graph entry
(src/rag/main.run, which hands a ccnl turn to src/rag/ccnl_path.py) with the code of the current folder, the live
.env, CCNL_PATH_ENABLED=true unless --flag says otherwise, every Neo4j session read-only. Scored per case:

  SECTOR     the chosen sector matches the case's regex (on its folded name)
  VERSION    "testo consolidato": the base is the sector's Testo Consolidato and every signed renewal (CCNL,
             accordo di rinnovo) up to the date is among the amendments; "latest renewal": the sector's latest
             signed CCNL or accordo di rinnovo up to the date is in the chosen set
  CLAUSE     a clause found (reference, title, text) matches the case's regex
  FRAMEWORK  the expected articles the answer cites with their code or law ("art. 2109 c.c.", "art. 7 l.
             300/1970", "art. 7 dello Statuto dei lavoratori")
  AMBIGUITY  the answer states an assumption ("**Ipotesi**") exactly when the case expects one
  PROTECTED  verbatim overlaps with the clauses and other protected sources (src/rag/protected_sources.py): found /
             rewritten / dropped by the post-check, remaining in the final answer (target 0), and the protected
             excerpts shown in the sources (target 0) -- counted with PROTECTED_PARAPHRASE_ENABLED off too

Run from the app folder (it writes only under --out):
    cd /opt/chatbot && venv/bin/python tests/eval_narrative.py --ccnl --label first
    cd /opt/chatbot && venv/bin/python tests/eval_narrative.py --ccnl --debug 3
    cd /opt/chatbot && venv/bin/python tests/eval_narrative.py --ccnl --only m2,f2 --debug
    cd /opt/chatbot && venv/bin/python tests/eval_narrative.py --ccnl --label round4 --debug 14 --answers c1,c2,m4,e1,e2
    cd /opt/chatbot && venv/bin/python tests/eval_narrative.py --ccnl --label protected-on \\
        --flag PROTECTED_PARAPHRASE_ENABLED=true

--debug [N] (default 3; the first N cases, or the --only ones) prints for each case: the route, the sector
candidates with their scores and reasons, the date and how it was read, every document of the sector (kind,
validity) with the base, amendments and dropped ipotesi, the clauses found (order key, sections, reference, the
first characters), the framework articles wanted / found / missing, the rulings, the answer checks and the
answer. Also written to debug.txt. Every case also records the answer's sentences claiming the CCNL is silent or
concluding from a missing clause (ccnl_path.absence_claims: "non prevede / non specifica", "in assenza di
clausole ...", usi / equità applied with no clause found), counted in the summary (target 0). --answers IDS
prints those cases' answers, with their ids, at the end (and writes answers.txt).
"""

import json
import os
import re
import sys
import time
from datetime import date, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_CCNL_CASES = HERE / "eval_ccnl_cases.json"
SIGNED_RENEWALS = ("ccnl", "accordo di rinnovo")


def _fold(text):
    import unicodedata
    text = unicodedata.normalize("NFKD", (text or "").lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


# --- what the answer cites --------------------------------------------------------------------------------

_RUN_RE = re.compile(r"\b(?:artt?\.|art\b|articol[oi])\s*(\d+(?:-[a-z]+)?(?:\s*(?:,|e|ed)\s*\d+(?:-[a-z]+)?)*)",
                     re.IGNORECASE)
_LAW_RE = re.compile(r"(?:\b(?:l\.|legge)|\bd\.?\s?lgs\.?|\bdecreto legislativo)\s*(?:n\.?\s*)?(\d{1,4})\s*"
                     r"(?:/|del(?:l')?\s*(?:\d{1,2}\s+\w+\s+)?)\s*((?:19|20)\d{2})", re.IGNORECASE)
# "legge 20 maggio 1970, n. 300", "d.lgs. 8 aprile 2003, n. 66": the date before the number.
_LAW_DATED_RE = re.compile(r"(?:\b(?:l\.|legge)|\bd\.?\s?lgs\.?|\bdecreto legislativo)\s*(?:\d{1,2}\s+\w+\s+)?"
                           r"((?:19|20)\d{2})\s*,?\s*n\.?\s*(\d{1,4})\b", re.IGNORECASE)
_NAMED_LAWS = [(re.compile(r"statuto dei lavoratori", re.I), "l. 300/1970")]


def _label_after(text):
    """The code or law written at the start of `text` (just after an article): "c.c.", "Cost.", "l. 300/1970",
    "d.lgs. 66/2003"; None when none is within its first characters."""
    head = text[:90]
    m = re.match(r"^[\s,]*(?:(?:comma|co\.|lett\.)\s*[\w)]+[\s,]*)*(?:del(?:la|lo)?\s+|dello\s+)?", head, re.I)
    rest = head[m.end():] if m else head
    if re.match(r"c\.\s?c\.|cod(?:ice|\.)\s*civ", rest, re.I):
        return "c.c."
    if re.match(r"cost\.|costituzione", rest, re.I):
        return "Cost."
    kind = "l." if re.match(r"l\.|legge", rest, re.I) else "d.lgs."
    dated = _LAW_DATED_RE.match(rest)
    if dated:
        return f"{kind} {dated.group(2)}/{dated.group(1)}"
    law = _LAW_RE.match(rest)
    if law:
        return f"{kind} {law.group(1)}/{law.group(2)}"
    for pattern, label in _NAMED_LAWS:
        if pattern.match(rest):
            return label
    return None


def framework_mentions(answer):
    """{(article, label)} for every article the answer cites with its code or law."""
    found = set()
    for run in _RUN_RE.finditer(answer or ""):
        label = _label_after(answer[run.end():])
        if not label:
            continue
        for number in re.findall(r"\d+(?:-[a-z]+)?", run.group(1)):
            found.add((number, label))
    return found


# --- scoring -----------------------------------------------------------------------------------------------

def _version_right(expected, version):
    """True / False against the case's "version", None without the path's log."""
    if not version or not version.get("base"):
        return None
    when = version["date"]
    docs = version.get("sector_documents") or []
    chosen = [version["base"]] + list(version.get("amendments") or [])
    chosen_ids = {d["id"] for d in chosen}
    renewals = [d for d in docs if d["kind"] in SIGNED_RENEWALS and d.get("valid_from") and d["valid_from"] <= when]
    if expected == "testo consolidato":
        if version["base"]["kind"] != "testo consolidato":
            return False
        base_from = version["base"].get("valid_from") or ""
        later = [d for d in renewals if d["valid_from"] > base_from]
        return all(d["id"] in chosen_ids for d in later)
    if expected == "latest renewal":
        if not renewals:
            return version["base"]["kind"] in SIGNED_RENEWALS
        latest = max(r["valid_from"] for r in renewals)
        return any(d["id"] in chosen_ids for d in renewals if d["valid_from"] == latest)
    return None


def score_case(case, result):
    info = result.get("ccnl") or {}
    answer = result.get("answer") or ""
    contract = info.get("contract") or {}
    sector = contract.get("sector")
    sector_right = bool(sector and re.search(case["sector"], _fold(sector))) if info else None
    version_right = _version_right(case.get("version"), info.get("version"))
    found = (info.get("clauses") or {}).get("found") or []
    clause_right = any(re.search(case["clause"], f"{c.get('ref', '')} {c.get('text', '')}", re.I) for c in found) \
        if info else None
    cited = framework_mentions(answer)
    expected = [(f["article"], f["law"]) for f in case.get("framework") or []]
    framework = [{"article": f"{a} {law}", "cited": (a, law) in cited} for a, law in expected]
    flagged = "**Ipotesi**" in answer
    from src.rag.ccnl_path import NO_CLAUSE, absence_claims
    fixed = NO_CLAUSE.split("{")[0]
    # The body only: the header (contract, version, notes) is the path's own fixed text.
    body = answer.split("\n\n", 1)[1] if "\n\n" in answer else answer
    silence = absence_claims(body, bool((info.get("clauses") or {}).get("no_clause")))
    return {"sector": sector, "sector_right": sector_right, "version_right": version_right,
            "clause_found": clause_right, "framework": framework,
            "framework_cited": sum(1 for f in framework if f["cited"]), "framework_expected": len(framework),
            "ambiguity_flagged": flagged, "ambiguity_right": flagged == bool(case.get("ambiguous")),
            "cited_in_answer": sorted(f"{a} {law}" for a, law in cited), "silence_claims": silence,
            "no_clause_sentence": fixed in answer}


def run_case(case, events):
    from src.rag import protected_sources
    from src.rag.main import run
    del events[:]
    t0 = time.time()
    try:
        result = run(case["question"], raw_query=case["question"], skip_calculation=True)
        status = "ok"
    except Exception as exc:
        result, status = {}, f"ERROR {type(exc).__name__}: {str(exc)[:200]}"
    record = {"id": case["id"], "topic": case.get("topic"), "question": case["question"], "status": status,
              "seconds": round(time.time() - t0, 1), "route": result.get("route") or "graph",
              "path_ran": bool(result.get("ccnl")), "answer": result.get("answer") or "",
              "citations": result.get("citations") or [], "ccnl": result.get("ccnl") or {},
              "events": [{"event": e, "data": d} for e, d in events
                         if str(e).startswith("ccnl") or str(e).startswith("protected")]}
    clause_texts = [c.get("text") or "" for c in ((result.get("ccnl") or {}).get("clauses") or {}).get("found") or []]
    record["protected"] = protected_sources.run_record(events, record["answer"], result.get("raw_result") or [],
                                                       record["citations"], texts=clause_texts)
    record["score"] = score_case(case, result)
    return record


def _capture():
    events = []

    def vlog(event, data=None, duration_ms=None, **_):
        events.append((event, data))
    for name, module in list(sys.modules.items()):
        if name.startswith("src.rag") and callable(getattr(module, "vlog", None)):
            module.vlog = vlog
    return events


def _mark(value):
    return "-" if value is None else ("yes" if value else "NO")


def debug_report(case, record):
    s, info = record["score"], record["ccnl"]
    out = ["", "=" * 100, f"{case['id']} ({case.get('topic')}): {case['question']}", "=" * 100,
           f"STATUS {record['status']}; path ran: {record['path_ran']}; {record['seconds']}s"]
    for e in record["events"]:
        if e["event"] == "ccnl_route":
            out.append(f"ROUTE run={e['data'].get('run')}: {e['data'].get('why')}")
    contract = info.get("contract") or {}
    out.append(f"CONTRACT {contract.get('sector')} (how: {contract.get('how')}; ambiguous {contract.get('ambiguous')}; "
               f"alternatives {contract.get('alternatives')}) -> expected /{case['sector']}/: {_mark(s['sector_right'])}")
    for c in contract.get("candidates") or []:
        out.append(f"    {c['score']:>3}  {c['sector'][:60]:60} docs={c['docs']:<3} {'; '.join(c['why'])[:90]}")
    version = info.get("version") or {}
    if version:
        out.append(f"DATE {version.get('date')} ({version.get('date_from')}); {version.get('reason')}; "
                   f"expired {version.get('expired')} -> expected {case.get('version')}: {_mark(s['version_right'])}")
        chosen = {version["base"]["id"]: "BASE"}
        chosen.update({d["id"]: "AMEND" for d in version.get("amendments") or []})
        for d in sorted(version.get("sector_documents") or [], key=lambda d: d.get("valid_from") or ""):
            out.append(f"    {chosen.get(d['id'], ''):5} {str(d.get('valid_from')):10} .. {str(d.get('valid_until')):10} "
                       f"{d['kind']:18} {d['name'][:70]}")
        for d in version.get("dropped") or []:
            out.append(f"    dropped: {d['name']} ({d['why']})")
    clauses = info.get("clauses") or {}
    if clauses:
        out.append(f"CLAUSES topics {clauses.get('topics')}; {clauses.get('sections')} sections searched; order by "
                   f"{clauses.get('order_by')} -> expected /{case['clause']}/: {_mark(s['clause_found'])}")
        for c in clauses.get("found") or []:
            out.append(f"    [{c['doc'][:50]}] {c['ref']} (sections {c['sections']}; pay rows {c['rows']})")
            out.append(f"        {' '.join(c['text'].split())[:300]}")
    fw = info.get("framework") or {}
    if fw:
        out.append(f"FRAMEWORK wanted {fw.get('wanted')}; found {fw.get('found')}; missing {fw.get('missing')}")
        out.append(f"RULINGS {fw.get('rulings')}")
    if info.get("pay"):
        out.append(f"PAY {json.dumps(info['pay'], ensure_ascii=False)}")
    out.append(f"ANSWER CHECKS {json.dumps(info.get('answer') or {}, ensure_ascii=False)}")
    out.append(f"PROTECTED {json.dumps(record.get('protected') or {}, ensure_ascii=False)}")
    out.append(f"CITED IN THE ANSWER {s['cited_in_answer']}; expected "
               f"{[f['article'] + (' ok' if f['cited'] else ' MISSING') for f in s['framework']]}")
    out.append(f"AMBIGUITY expected {case.get('ambiguous')}, flagged {s['ambiguity_flagged']}: "
               f"{_mark(s['ambiguity_right'])}")
    out.append(f"NO-CLAUSE SENTENCE {s.get('no_clause_sentence')}; CCNL-SILENCE CLAIMS {s.get('silence_claims')}")
    out.append(f"ANSWER ({len(record['answer'])} chars):")
    out += ["    " + line for line in (record["answer"] or "<empty>").splitlines()]
    return out


def summary_lines(records, label):
    lines = [f"CCNL eval {label}".strip(), "",
             f"{'case':5} {'topic':13} {'path':5} {'sector':7} {'version':8} {'clause':7} {'framework':10} "
             f"{'ambig':6} {'secs':>5}"]
    for r in records:
        s = r["score"]
        lines.append(f"{r['id']:5} {str(r['topic'])[:13]:13} {'yes' if r['path_ran'] else 'no':5} "
                     f"{_mark(s['sector_right']):7} {_mark(s['version_right']):8} {_mark(s['clause_found']):7} "
                     f"{s['framework_cited']:>3}/{s['framework_expected']:<6} {_mark(s['ambiguity_right']):6} "
                     f"{r['seconds']:>5}" + ("" if r["status"] == "ok" else f"  {r['status']}"))
    t = totals(records)
    lines += ["", f"Path ran {t['path_ran']}/{t['cases']}; sector right {t['sector_right']}; version right "
                  f"{t['version_right']}; clause found {t['clause_found']}; framework cited {t['framework_cited']}/"
                  f"{t['framework_expected']}; ambiguity right {t['ambiguity_right']}/{t['cases']} (flagged when "
                  f"expected {t['ambiguity_flagged_expected']}/{t['ambiguity_expected']}); answers claiming the "
                  f"CCNL's silence {t['silence']}",
              _protected_line(records)]
    return lines


def _protected_line(records):
    from src.rag import protected_sources
    return protected_sources.report_line(protected_sources.sum_records(r.get("protected") for r in records))


def totals(records):
    def count(key):
        return sum(1 for r in records if r["score"][key])
    return {"cases": len(records), "path_ran": sum(1 for r in records if r["path_ran"]),
            "sector_right": count("sector_right"), "version_right": count("version_right"),
            "clause_found": count("clause_found"), "framework_cited": sum(r["score"]["framework_cited"] for r in records),
            "framework_expected": sum(r["score"]["framework_expected"] for r in records),
            "ambiguity_right": count("ambiguity_right"),
            "silence": sum(1 for r in records if r["score"].get("silence_claims")),
            "ambiguity_expected": sum(1 for r in records if r.get("expected_ambiguous")),
            "ambiguity_flagged_expected": sum(1 for r in records if r.get("expected_ambiguous")
                                              and r["score"]["ambiguity_flagged"]),
            **{f"protected_{k}": v for k, v in _protected_sum(records).items()}}


def _protected_sum(records):
    from src.rag import protected_sources
    return protected_sources.sum_records(r.get("protected") for r in records)


def main(args):
    """Called by tests/eval_narrative.py --ccnl."""
    from eval_direct import _setup
    settings = {"CCNL_PATH_ENABLED": "true", **dict(item.split("=", 1) for item in args.set_flags or [])}
    _setup(settings)
    import importlib
    importlib.import_module("src.rag.main")
    events = _capture()
    debug = getattr(args, "debug", None)
    cases = json.loads(Path(args.ccnl_cases).read_text(encoding="utf-8"))["cases"]
    if args.only:
        cases = [c for c in cases if c["id"] in {x.strip() for x in args.only.split(",")}]
    elif debug:
        cases = cases[:debug]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.out) / (f"ccnl-{stamp}" + (f"-{args.label}" if args.label else ""))
    run_dir.mkdir(parents=True, exist_ok=True)
    records, debug_lines = [], []
    for repeat in range(1, max(1, args.repeat) + 1):
        for case in cases:
            record = run_case(case, events)
            record["repeat"], record["expected_ambiguous"] = repeat, bool(case.get("ambiguous"))
            records.append(record)
            if debug:
                block = debug_report(case, record)
                debug_lines += block
                print("\n".join(block), flush=True)
            name = f"{case['id']}" + (f"-r{repeat}" if args.repeat > 1 else "")
            (run_dir / f"{name}.json").write_text(json.dumps(record, ensure_ascii=False, indent=1, default=str),
                                                  encoding="utf-8")
            s = record["score"]
            print(f"{case['id']} r{repeat}: path {record['path_ran']}, sector {_mark(s['sector_right'])}, version "
                  f"{_mark(s['version_right'])}, clause {_mark(s['clause_found'])}, framework "
                  f"{s['framework_cited']}/{s['framework_expected']}, ambiguity {_mark(s['ambiguity_right'])}, "
                  f"protected overlaps {record['protected']['found']}/{record['protected']['rewritten']}/"
                  f"{record['protected']['dropped']}/{record['protected']['remaining']}, excerpts shown "
                  f"{record['protected']['excerpts_shown']}, {record['seconds']}s", flush=True)
    lines = summary_lines(records, args.label)
    summary = {"label": args.label, "settings": settings, "today": date.today().isoformat(), "totals": totals(records),
               "cases": [{"id": r["id"], "repeat": r["repeat"], "protected": r.get("protected"),
                          **{k: v for k, v in r["score"].items() if k != "cited_in_answer"}} for r in records]}
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    (run_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if debug:
        (run_dir / "debug.txt").write_text("\n".join(debug_lines) + "\n", encoding="utf-8")
    wanted = [x.strip() for x in (getattr(args, "answers", None) or "").split(",") if x.strip()]
    if wanted:
        answers = []
        for r in records:
            if r["id"] in wanted:
                answers += ["", "=" * 100, f"{r['id']}: {r['question']}", "=" * 100, r["answer"] or "<empty>"]
        (run_dir / "answers.txt").write_text("\n".join(answers) + "\n", encoding="utf-8")
        print("\n".join(answers))
    print("\n".join(lines))
    print(f"\nWritten to {run_dir}")
