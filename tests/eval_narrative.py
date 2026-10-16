"""Acceptance set for narrative and strategy questions, run against a live V2.

Posts each case in eval_narrative_cases.json to /api/chat on --base-url, as a
logged-in user, each in a new chat session, and scores the answer per expected
article, code-aware (the article number AND the code or law it belongs to):

  CITED       a citation from that code has a section with that article
  NAMED-ONLY  the answer names it (with that code) but cites nothing for it
  NAMED-BARE  the answer names the number without saying which code
  MISSING     neither

It also lists every article the answer names but doesn't cite, forbidden
articles that were cited (FAIL) and forbidden names that were used (FLAG).

It only reads the app's answers; the one thing it leaves behind is a chat
session per case for the account it logs in as, so use a dedicated account.

  # credentials from the environment (EVAL_EMAIL, EVAL_PASSWORD, optional EVAL_TOTP), else prompted
  python tests/eval_narrative.py --base-url http://localhost:8000
  python tests/eval_narrative.py --base-url http://localhost:8000 --token <JWT> --only sempronio,c2
  # each case 3 times, to separate a real change from model/run variance
  python tests/eval_narrative.py --base-url http://localhost:8000 --repeat 3 --label flag-off
  # what changed between two runs
  python tests/eval_narrative.py --compare eval_runs/<run A>/summary.json eval_runs/<run B>/summary.json

Each run writes eval_runs/<timestamp>[-label]/: one <case>.json per case
(question, answer, citations, scoring) — <case>-rN.json instead when
--repeat > 1 — and summary.json / summary.txt. --repeat > 1 also writes
aggregate.json / aggregate.txt: per case, how many of the repeats CITED
each expected article (e.g. "2/3"), and the average CITED count per case
and overall — the numbers to compare flag-off vs flag-on on, since a single
run's score moves by several articles between identical runs (model/serving
variance, not a real change).
"""

import argparse
import getpass
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_CASES = HERE / "eval_narrative_cases.json"

_SUFFIX = r"(?:bis|ter|quater|quinquies|sexies|septies|octies|novies|decies)"

# code key -> (citation document_name pattern, pattern for the code written in the answer)
CODES = {
    "cpp": (r"codice di procedura penale",
            r"\bc\.?\s?p\.?\s?p\b\.?|codice di procedura penale|cod\.\s*proc\.\s*pen"),
    "cpc": (r"codice di procedura civile",
            r"\bc\.?\s?p\.?\s?c\b\.?|codice di procedura civile|cod\.\s*proc\.\s*civ"),
    "cp": (r"codice penale(?!\s+commentat)",
           r"\bc\.?\s?p\b(?!\.?\s?[cp]\b)\.?|codice penale|cod\.\s*pen"),
    "cc": (r"codice civile",
           r"\bc\.?\s?c\b(?!\.?\s?[cp]\b)\.?|codice civile|cod\.\s*civ"),
    "consumo": (r"consumo|206\s*/\s*2005",
                r"cod(?:ice)?\.?\s*(?:del\s+)?cons(?:umo)?\b|d\.?\s?lgs\.?\s*(?:n\.\s*)?206\s*/\s*2005"),
    "l392": (r"(?=.*\b392\b)(?=.*\b1978\b)|equo canone",
             r"\bl(?:egge)?\.?\s*(?:n\.\s*)?392\s*(?:/|del\s+)\s*1978|equo canone"),
}
_CODE_ORDER = ["cpp", "cpc", "cp", "cc", "consumo", "l392"]   # longer abbreviations first

# A section name's article: "1434", "1434.0.0", "1434.1", "1434_2", "624-bis", "624bis",
# "Art. 52", "Articolo 5" -- and never "14340" for 1434.
_SECTION_ART = re.compile(rf"^\s*(?:art(?:icolo)?\.?\s*)?(\d+)(?:[\s-]*({_SUFFIX}))?(?![0-9A-Za-z])", re.I)
# Article mentions in the answer, including runs ("artt. 1490 e 1495").
_NUMBER = rf"\d+(?:[\s-]*{_SUFFIX})?(?![0-9A-Za-z])"
_MENTION_RUN = re.compile(rf"\b(?:artt?\.?|articol[oi])\s*{_NUMBER}(?:\s*(?:,|e|ed|&)\s*{_NUMBER})*", re.I)
_MENTION_NUM = re.compile(rf"(\d+)(?:[\s-]*({_SUFFIX}))?(?![0-9A-Za-z])", re.I)


def art_of(section_name):
    """The article number a section name stands for ("624-bis"), or None."""
    m = _SECTION_ART.match(section_name or "")
    if not m:
        return None
    return f"{m.group(1)}-{m.group(2).lower()}" if m.group(2) else m.group(1)


def code_of_document(document_name):
    for key in _CODE_ORDER:
        if re.search(CODES[key][0], document_name or "", re.I):
            return key
    return None


def _code_in_text(text):
    """The code named at the start of `text` (just after an article number)."""
    for key in _CODE_ORDER:
        m = re.search(CODES[key][1], text, re.I)
        if m and m.start() <= 12:          # "c.p.", ", c.p.", "del codice penale"
            return key
    return None


def mentions(answer):
    """(article, code key or None) for every article the answer names."""
    found = []
    for run in _MENTION_RUN.finditer(answer or ""):
        prefix = re.match(r"\b(?:artt?\.?|articol[oi])\s*", run.group(0), re.I).end()
        code = _code_in_text(answer[run.end():run.end() + 60])
        for m in _MENTION_NUM.finditer(run.group(0)[prefix:]):
            number = f"{m.group(1)}-{m.group(2).lower()}" if m.group(2) else m.group(1)
            if (number, code) not in found:
                found.append((number, code))
    return found


def cited(citations):
    """(article, code key or None, document name) for every cited section."""
    out = []
    for c in citations or []:
        name = c.get("document_name") or ""
        code = code_of_document(name)
        for s in c.get("sections") or []:
            article = art_of(s.get("name"))
            if article:
                out.append((article, code, name))
    return out


def _articles(item):
    return [str(a).lower() for a in (item.get("any_of") or [item["article"]])]


def _label(item):
    return item.get("label") or f"art. {item['article']} {item['code']}"


def score(case, answer, citations):
    cites = cited(citations)
    named = mentions(answer)
    expected = []
    for item in case.get("expected", []):
        arts, code = _articles(item), item["code"]
        if any(a in arts and c == code for a, c, _ in cites):
            status = "CITED"
        elif any(a in arts and c == code for a, c in named):
            status = "NAMED-ONLY"
        elif any(a in arts and c is None for a, c in named):
            status = "NAMED-BARE"
        else:
            status = "MISSING"
        expected.append({"expected": _label(item), "status": status})

    cited_keys = {(a, c) for a, c, _ in cites}
    cited_numbers = {a for a, _, _ in cites}
    named_not_cited = [
        f"art. {a} {c}" if c else f"art. {a} (no code)"
        for a, c in named
        if ((a, c) not in cited_keys if c else a not in cited_numbers)
    ]
    failures = [
        f"FAIL cited art. {f['article']} {f['code']}" + (f" ({f['why']})" if f.get("why") else "")
        for f in case.get("forbidden_cited", [])
        if (str(f["article"]).lower(), f["code"]) in cited_keys
    ]
    failures += [
        f"FLAG named /{f['pattern']}/" + (f" ({f['why']})" if f.get("why") else "")
        for f in case.get("forbidden_named", [])
        if re.search(f["pattern"], answer or "", re.I)
    ]
    docs = [
        {"expected": d.get("label") or d["doc"],
         "status": "CITED" if any(re.search(d["doc"], (c.get("document_name") or ""), re.I)
                                  for c in citations or []) else "MISSING"}
        for d in case.get("expected_docs", [])
    ]
    return {"expected": expected + docs, "named_not_cited": named_not_cited, "failures": failures,
            "cited": sorted({f"art. {a} {c or '?'}" for a, c, _ in cites})}


# --- HTTP --------------------------------------------------------------------

def _post(url, payload, token=None, timeout=60):
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _error_detail(exc):
    try:
        return json.loads(exc.read().decode("utf-8")).get("detail") or exc.reason
    except (ValueError, AttributeError):
        return exc.reason


def login(base_url):
    email = os.getenv("EVAL_EMAIL") or input("Email: ")
    password = os.getenv("EVAL_PASSWORD") or getpass.getpass("Password: ")
    payload = {"email": email, "password": password}
    if os.getenv("EVAL_TOTP"):
        payload["totp_code"] = os.environ["EVAL_TOTP"]
    try:
        return _post(f"{base_url}/api/auth/login", payload)["access_token"]
    except urllib.error.HTTPError as exc:
        if exc.code == 401 and exc.headers.get("X-2FA-Required") and "totp_code" not in payload:
            payload["totp_code"] = input("2FA code: ").strip()     # the account has two-factor login on
            try:
                return _post(f"{base_url}/api/auth/login", payload)["access_token"]
            except urllib.error.HTTPError as again:
                exc = again
        sys.exit(f"Login failed ({exc.code}): {_error_detail(exc)}")


def ask(base_url, token, question, timeout):
    t0 = time.time()
    body = _post(f"{base_url}/api/chat", {"message": question}, token=token, timeout=timeout)
    return body, round(time.time() - t0, 1)


# --- Reports -------------------------------------------------------------------

def _summary_lines(results):
    lines = []
    for r in results:
        if r.get("error"):
            lines.append(f"## {r['id']}: ERROR {r['error']}")
            continue
        s = r["score"]
        counts = {k: sum(1 for e in s["expected"] if e["status"] == k)
                  for k in ("CITED", "NAMED-ONLY", "NAMED-BARE", "MISSING")}
        lines.append(f"## {r['id']}{' (extra)' if r.get('extra') else ''}  {r['seconds']} s  "
                     + "  ".join(f"{k} {v}" for k, v in counts.items() if v))
        for e in s["expected"]:
            lines.append(f"   {e['status']:10s} {e['expected']}")
        for f in s["failures"]:
            lines.append(f"   {f}")
        if s["named_not_cited"]:
            lines.append(f"   named, not cited: {', '.join(s['named_not_cited'])}")
    return lines


def _aggregate(results, repeat):
    """Per case: {expected_label: (cited_count, repeat)}, average CITED count
    per case, and the overall average — results is the flat list of per-
    repeat records (each carrying "id" and "score"), grouped back by id."""
    by_case = {}
    for r in results:
        by_case.setdefault(r["id"], []).append(r)

    per_case = {}
    totals = []
    for case_id, runs in by_case.items():
        ok_runs = [r for r in runs if not r.get("error")]
        labels = list(dict.fromkeys(e["expected"] for r in ok_runs for e in r["score"]["expected"]))
        counts = {label: sum(1 for r in ok_runs for e in r["score"]["expected"]
                             if e["expected"] == label and e["status"] == "CITED")
                  for label in labels}
        per_case[case_id] = {"counts": counts, "n": len(ok_runs), "errors": len(runs) - len(ok_runs)}
        if ok_runs:
            totals.append(sum(counts.values()) / len(ok_runs))
    overall_avg = sum(totals) / len(totals) if totals else 0.0
    return per_case, overall_avg


def _aggregate_lines(per_case, overall_avg, repeat):
    lines = [f"Averages over {repeat} repeats per case (CITED rate per expected article):"]
    for case_id, data in per_case.items():
        n = data["n"]
        avg = sum(data["counts"].values()) / n if n else 0.0
        err = f"  ({data['errors']} error(s))" if data["errors"] else ""
        lines.append(f"## {case_id}  avg CITED {avg:.2f}/{len(data['counts'])}{err}")
        for label, c in data["counts"].items():
            lines.append(f"   {c}/{n}  {label}")
    lines.append(f"\nOVERALL average CITED per case: {overall_avg:.2f} (across {len(per_case)} cases, "
                 f"{repeat} repeats each)")
    return lines


def compare(old_path, new_path):
    old = {r["id"]: r for r in json.loads(Path(old_path).read_text(encoding="utf-8"))["results"]}
    new = {r["id"]: r for r in json.loads(Path(new_path).read_text(encoding="utf-8"))["results"]}
    for case_id in [c for c in new if c in old]:
        if old[case_id].get("error") or new[case_id].get("error"):
            print(f"## {case_id}: error in one run")
            continue
        before = {e["expected"]: e["status"] for e in old[case_id]["score"]["expected"]}
        after = {e["expected"]: e["status"] for e in new[case_id]["score"]["expected"]}
        changes = [f"   {k}: {before.get(k, '-')} -> {v}" for k, v in after.items() if before.get(k) != v]
        fails = (set(new[case_id]["score"]["failures"]) - set(old[case_id]["score"]["failures"]))
        print(f"## {case_id}: " + ("no change" if not changes and not fails else ""))
        for line in changes:
            print(line)
        for f in sorted(fails):
            print(f"   new: {f}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Acceptance set for narrative and strategy questions.")
    parser.add_argument("--base-url", help="e.g. http://localhost:8000 (on the server)")
    parser.add_argument("--token", help="a JWT instead of logging in")
    parser.add_argument("--cases", default=str(DEFAULT_CASES))
    parser.add_argument("--only", help="comma-separated case ids")
    parser.add_argument("--skip-extra", action="store_true", help="only Flavio's cases")
    parser.add_argument("--out", default="eval_runs", help="folder for the run outputs")
    parser.add_argument("--label", default="", help="added to the run folder name, e.g. baseline")
    parser.add_argument("--timeout", type=int, default=400, help="seconds per question")
    parser.add_argument("--repeat", type=int, default=1,
                        help="run every case N times; report per-article CITED rate and averages")
    parser.add_argument("--compare", nargs=2, metavar=("OLD_SUMMARY", "NEW_SUMMARY"))
    args = parser.parse_args(argv)

    if args.compare:
        compare(*args.compare)
        return 0
    if not args.base_url:
        parser.error("--base-url is required (or use --compare)")
    base_url = args.base_url.rstrip("/")
    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))["cases"]
    if args.only:
        wanted = {c.strip() for c in args.only.split(",")}
        cases = [c for c in cases if c["id"] in wanted]
    if args.skip_extra:
        cases = [c for c in cases if not c.get("extra")]

    token = args.token or login(base_url)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.out) / (f"{stamp}-{args.label}" if args.label else stamp)
    run_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for case in cases:
        for i in range(args.repeat):
            label = f"{case['id']} (r{i + 1}/{args.repeat})" if args.repeat > 1 else case["id"]
            print(f"... {label}", flush=True)
            record = {"id": case["id"], "extra": bool(case.get("extra")), "question": case["question"]}
            if args.repeat > 1:
                record["repeat"] = i
            try:
                body, seconds = ask(base_url, token, case["question"], args.timeout)
                record.update(seconds=seconds, answer=body.get("answer", ""), citations=body.get("citations", []),
                              status_messages=body.get("status_messages", []),
                              score=score(case, body.get("answer", ""), body.get("citations", [])))
            except (urllib.error.URLError, TimeoutError, ValueError, KeyError) as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
            file_name = f"{case['id']}-r{i + 1}.json" if args.repeat > 1 else f"{case['id']}.json"
            (run_dir / file_name).write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            results.append(record)

    lines = _summary_lines(results)
    (run_dir / "summary.json").write_text(
        json.dumps({"run": run_dir.name, "base_url": base_url, "repeat": args.repeat, "results": results},
                   ensure_ascii=False, indent=2),
        encoding="utf-8")
    (run_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    if args.repeat > 1:
        per_case, overall_avg = _aggregate(results, args.repeat)
        agg_lines = _aggregate_lines(per_case, overall_avg, args.repeat)
        (run_dir / "aggregate.json").write_text(
            json.dumps({"run": run_dir.name, "repeat": args.repeat, "overall_avg": overall_avg,
                       "per_case": per_case}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        (run_dir / "aggregate.txt").write_text("\n".join(agg_lines) + "\n", encoding="utf-8")
        print("\n" + "\n".join(agg_lines))

    print(f"\nSaved in {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
