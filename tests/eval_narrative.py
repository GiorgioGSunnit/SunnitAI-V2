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

Held-out cases (19 Oct; "heldout": true in eval_narrative_cases.json --
h1-h11, real exam-style cases from a practising lawyer, NOT Flavio's own
list) are a separate, later check on whatever the tuning set already
converged on: never used to write a prompt, pick a fixture, or decide
whether a change helped, and never fed into scripts/exp_distill.py's case
set either. Excluded by default (a normal run only costs what it always
did); --include-heldout adds them to the same run, reported in their own
section with their own average, never mixed into the tuning set's; --heldout-
only runs just them. Their "expected" lists are themselves provisional
("expected_status": "draft — pending lawyer validation") -- scripts/
check_heldout_articles.py checks whether the documents they name even
exist in the corpus, separately from whether the numbers are right.

24 Oct: the primary score is MENTIONED -- the expected article named in the
answer text with its code (CITED or NAMED-ONLY above), what the lawyer reads.
CITED is reported second. A second held-out set from the lawyer, n1-n9
("set": "heldout2"), and the two cases of that batch already used to tune plan
mode, s1-s2 ("set": "seen2"): each set gets its own section and averages,
never mixed with another's. Cases with a set run only when asked:

  python tests/eval_narrative.py --base-url http://localhost:8000 --set heldout2 --repeat 3 --label ptw-on
  python tests/eval_narrative.py --set seen2 ...            # s1-s2
  # the saved answers scored again with the current expected lists (no requests),
  # into <run dir>/rescore-<timestamp>/ -- after the lawyer corrects them
  python tests/eval_narrative.py --rescore eval_runs/<run dir>
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
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

# Every Latin suffix spelling, normalised to the canonical one the corpus uses
# ("21 novies" -> "21-nonies"): the app's own table (src/rag/article_suffix.py, no
# other imports). The list here before 24 Oct had no "nonies", so heldout2 n3's
# 21-nonies l.241 could never match.
from src.rag.article_suffix import SUFFIX as _SUFFIX, canonical_suffix  # noqa: E402

# code key -> (citation document_name pattern, pattern for the code written in the answer)
CODES = {
    "cpp": (r"codice di procedura penale",
            r"\bc\.?\s?p\.?\s?p\b\.?|codice di procedura penale|cod\.\s*proc\.\s*pen"),
    "cpc": (r"codice di procedura civile",
            r"\bc\.?\s?p\.?\s?c\b\.?|codice di procedura civile|cod\.\s*proc\.\s*civ"),
    # cpa (held-out h10/h11, 19 Oct): checked before "cp" in _CODE_ORDER, same
    # reason cpp/cpc are -- otherwise "c.p.a." would match "cp"'s own pattern.
    "cpa": (r"codice del processo amministrativo",
            r"\bc\.?\s?p\.?\s?a\b\.?|codice del processo amministrativo|cod\.\s*proc\.\s*amm"),
    "cp": (r"codice penale(?!\s+commentat)",
           r"\bc\.?\s?p\b(?!\.?\s?[cp]\b)\.?|codice penale|cod\.\s*pen"),
    "cc": (r"codice civile",
           r"\bc\.?\s?c\b(?!\.?\s?[cp]\b)\.?|codice civile|cod\.\s*civ"),
    "consumo": (r"consumo|206\s*/\s*2005",
                r"cod(?:ice)?\.?\s*(?:del\s+)?cons(?:umo)?\b|d\.?\s?lgs\.?\s*(?:n\.\s*)?206\s*/\s*2005"),
    "l392": (r"(?=.*\b392\b)(?=.*\b1978\b)|equo canone",
             r"\bl(?:egge)?\.?\s*(?:n\.\s*)?392\s*(?:/|del\s+)\s*1978|equo canone"),
    # l241 (held-out h10, 19 Oct): "Legge 7 agosto 1990, n. 241" on procedimento
    # amministrativo, art. 21-septies. Not verified against the live corpus
    # (no DB access here) -- also matches "Legge sul Procedimento
    # Amministrativo", in case that's the document's real name instead; see
    # scripts/check_heldout_articles.py.
    "l241": (r"(?=.*\b241\b)(?=.*\b1990\b)|legge sul procedimento amministrativo",
             r"\bl(?:egge)?\.?\s*(?:n\.\s*)?241\s*(?:/|del\s+)\s*(?:19)?90\b|legge sul procedimento amministrativo"),
    # dlgs36 (heldout2 n4, 24 Oct): D.Lgs. 31 marzo 2023, n. 36, Codice dei contratti
    # pubblici -- in the corpus possibly as "Codice degli Appalti 2026" (arts 95 and 98:
    # scripts/check_heldout_articles.py says which document has them).
    "dlgs36": (r"(?:d\.?\s?lgs\.?|decreto legislativo)(?=.{0,40}\b36\b)(?=.{0,40}\b2023\b)|contratti pubblici"
               r"|codice degli appalti",
               r"d\.?\s?lgs\.?\s*(?:n\.\s*)?36\s*(?:/|del\s+)\s*2023|codice dei contratti(?:\s+pubblici)?"
               r"|codice degli appalti"),
    # tuel (heldout2 n5, 24 Oct): Testo unico degli enti locali, D.Lgs. 267/2000, art. 252.
    "tuel": (r"\btuel\b|testo unico.{0,40}enti locali"
             r"|(?:d\.?\s?lgs\.?|decreto legislativo)(?=.{0,40}\b267\b)(?=.{0,40}\b2000\b)",
             r"\bt\.?\s?u\.?\s?e\.?\s?l\b\.?|testo unico.{0,40}enti locali"
             r"|d\.?\s?lgs\.?\s*(?:n\.\s*)?267\s*(?:/|del\s+)\s*2000"),
}
_CODE_ORDER = ["cpp", "cpc", "cpa", "cp", "cc", "consumo", "l392", "l241", "dlgs36", "tuel"]   # longer abbreviations first

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
    return f"{m.group(1)}-{canonical_suffix(m.group(2))}" if m.group(2) else m.group(1)


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
            number = f"{m.group(1)}-{canonical_suffix(m.group(2))}" if m.group(2) else m.group(1)
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
        is_cited = any(a in arts and c == code for a, c, _ in cites)
        is_mentioned = any(a in arts and c == code for a, c in named)
        if is_cited:
            status = "CITED"
        elif is_mentioned:
            status = "NAMED-ONLY"
        elif any(a in arts and c is None for a, c in named):
            status = "NAMED-BARE"
        else:
            status = "MISSING"
        expected.append({"expected": _label(item), "status": status,
                         "mentioned": is_mentioned, "cited": is_cited})

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

# Primary metric (24 Oct): the expected article MENTIONED in the answer text with its
# code -- what the lawyer reads. CITED (a citation from that code has the section) is
# secondary. Items without the flag (expected_docs, or runs saved before 24 Oct and
# not rescored) count for neither.
_GROUP_TITLES = {
    "tuning": None,
    "heldout": "HELD-OUT SET h1-h11 (provisional expected articles -- never used to tune behaviour)",
    "heldout2": "HELD-OUT SET 2, n1-n9 (provisional expected articles -- never used to tune behaviour)",
    "seen2": "SEEN, s1-s2 (used to tune plan mode -- not held out, never in heldout2's average)",
}


def group_of(record):
    """The report section a result belongs to: its case's "set", else "heldout" or "tuning"."""
    return record.get("set") or ("heldout" if record.get("heldout") else "tuning")


def _grouped(results):
    groups = {}
    for r in results:
        groups.setdefault(group_of(r), []).append(r)
    return sorted(groups.items(), key=lambda g: list(_GROUP_TITLES).index(g[0]) if g[0] in _GROUP_TITLES else 99)


def _banner(group):
    title = _GROUP_TITLES.get(group, f"SET {group}")
    return ["\n" + "=" * 70, title, "=" * 70] if title else []


def _summary_group_lines(results):
    lines = []
    mentioned_counts, cited_counts = [], []
    for r in results:
        if r.get("error"):
            lines.append(f"## {r['id']}: ERROR {r['error']}")
            continue
        s = r["score"]
        counts = {k: sum(1 for e in s["expected"] if e["status"] == k)
                  for k in ("CITED", "NAMED-ONLY", "NAMED-BARE", "MISSING")}
        articles = [e for e in s["expected"] if "mentioned" in e]
        mentioned = sum(1 for e in articles if e["mentioned"])
        mentioned_counts.append(mentioned)
        cited_counts.append(counts["CITED"])
        lines.append(f"## {r['id']}{' (extra)' if r.get('extra') else ''}  {r['seconds']} s  "
                     f"MENTIONED {mentioned}/{len(articles)}  "
                     + "  ".join(f"{k} {v}" for k, v in counts.items() if v)
                     + ("" if s["expected"] else "  (NO EXPECTED ARTICLES)"))
        for e in s["expected"]:
            flag = "" if "mentioned" not in e else ("  mentioned" if e["mentioned"] else "  not mentioned")
            lines.append(f"   {e['status']:10s} {e['expected']}{flag}")
        for f in s["failures"]:
            lines.append(f"   {f}")
        if s["named_not_cited"]:
            lines.append(f"   named, not cited: {', '.join(s['named_not_cited'])}")
    if cited_counts:
        n = len(cited_counts)
        lines.append(f"\naverage MENTIONED per case: {sum(mentioned_counts) / n:.2f}; "
                     f"average CITED per case: {sum(cited_counts) / n:.2f} (across {n} case(s))")
    return lines


def _summary_lines(results):
    """Each set in its own section with its own averages -- held-out and seen cases
    never mixed into the tuning set's, see the module docstring."""
    lines = []
    for group, rows in _grouped(results):
        lines += _banner(group) + _summary_group_lines(rows)
    return lines


def _aggregate(results, repeat):
    """Per case: {expected_label: count} of repeats that CITED it and of repeats that
    MENTIONED it, the average of each per case, and the overall averages -- results is
    the flat list of per-repeat records (each carrying "id" and "score"), grouped back by id."""
    by_case = {}
    for r in results:
        by_case.setdefault(r["id"], []).append(r)

    per_case = {}
    totals, totals_m = [], []
    for case_id, runs in by_case.items():
        ok_runs = [r for r in runs if not r.get("error")]
        labels = list(dict.fromkeys(e["expected"] for r in ok_runs for e in r["score"]["expected"]))
        counts = {label: sum(1 for r in ok_runs for e in r["score"]["expected"]
                             if e["expected"] == label and e["status"] == "CITED")
                  for label in labels}
        mentioned = {label: sum(1 for r in ok_runs for e in r["score"]["expected"]
                                if e["expected"] == label and e.get("mentioned"))
                     for label in labels}
        avg_seconds = sum(r.get("seconds", 0.0) for r in ok_runs) / len(ok_runs) if ok_runs else 0.0
        per_case[case_id] = {"counts": counts, "mentioned": mentioned, "n": len(ok_runs),
                             "errors": len(runs) - len(ok_runs), "avg_seconds": avg_seconds}
        if ok_runs:
            totals.append(sum(counts.values()) / len(ok_runs))
            totals_m.append(sum(mentioned.values()) / len(ok_runs))
    overall_avg = sum(totals) / len(totals) if totals else 0.0
    overall_mentioned = sum(totals_m) / len(totals_m) if totals_m else 0.0
    return per_case, overall_avg, overall_mentioned


def _aggregate_lines(per_case, overall_avg, overall_mentioned, repeat):
    lines = [f"Averages over {repeat} repeats per case (MENTIONED / CITED rate per expected article):"]
    for case_id, data in per_case.items():
        n = data["n"]
        avg = sum(data["counts"].values()) / n if n else 0.0
        avg_m = sum(data.get("mentioned", {}).values()) / n if n else 0.0
        err = f"  ({data['errors']} error(s))" if data["errors"] else ""
        lines.append(f"## {case_id}  avg MENTIONED {avg_m:.2f}/{len(data['counts'])}  avg CITED {avg:.2f}  "
                     f"avg {data.get('avg_seconds', 0.0):.1f} s{err}")
        for label, c in data["counts"].items():
            lines.append(f"   mentioned {data.get('mentioned', {}).get(label, 0)}/{n}  cited {c}/{n}  {label}")
    total_n = sum(d["n"] for d in per_case.values())
    overall_seconds = (sum(d.get("avg_seconds", 0.0) * d["n"] for d in per_case.values()) /
                       total_n) if total_n else 0.0
    lines.append(f"\nOVERALL average MENTIONED per case: {overall_mentioned:.2f}; average CITED per case: "
                 f"{overall_avg:.2f} (across {len(per_case)} cases, {repeat} repeats each) — "
                 f"overall average {overall_seconds:.1f} s per run")
    return lines


def write_reports(results, run_dir, repeat, base_url):
    """summary.json/.txt, and with repeat > 1 aggregate.json/.txt, one section per set."""
    lines = _summary_lines(results)
    (run_dir / "summary.json").write_text(
        json.dumps({"run": run_dir.name, "base_url": base_url, "repeat": repeat, "results": results},
                   ensure_ascii=False, indent=2),
        encoding="utf-8")
    (run_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    if repeat > 1:
        agg_lines, agg_json = [], {"run": run_dir.name, "repeat": repeat}
        for group, rows in _grouped(results):
            per_case, overall_avg, overall_mentioned = _aggregate(rows, repeat)
            agg_lines += _banner(group) + _aggregate_lines(per_case, overall_avg, overall_mentioned, repeat)
            prefix = "" if group == "tuning" else f"{group}_"
            agg_json.update({f"{prefix}overall_avg": overall_avg, f"{prefix}overall_mentioned": overall_mentioned,
                             f"{prefix}per_case": per_case})
        (run_dir / "aggregate.json").write_text(
            json.dumps(agg_json, ensure_ascii=False, indent=2), encoding="utf-8")
        (run_dir / "aggregate.txt").write_text("\n".join(agg_lines) + "\n", encoding="utf-8")
        print("\n" + "\n".join(agg_lines))


def _case_fields(case):
    return {"extra": bool(case.get("extra")), "heldout": bool(case.get("heldout")),
            "set": case.get("set"), "seen": bool(case.get("seen"))}


def rescore(run_dir, cases):
    """Scores every saved answer in run_dir again with the current expected lists, into
    run_dir/rescore-<timestamp>/ -- the saved answers and the first scores stay as they are."""
    run_dir = Path(run_dir)
    by_id = {c["id"]: c for c in cases}
    files = sorted(f for f in run_dir.glob("*.json") if f.name not in ("summary.json", "aggregate.json"))
    if not files:
        sys.exit(f"no saved answers (<case>.json) in {run_dir}")
    try:
        meta = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        meta = {}
    results, unknown = [], []
    for f in files:
        record = json.loads(f.read_text(encoding="utf-8"))
        case = by_id.get(record.get("id"))
        if case is None:
            unknown.append(record.get("id") or f.name)
            continue
        record.update(_case_fields(case))
        if not record.get("error"):
            record["score"] = score(case, record.get("answer", ""), record.get("citations", []))
        results.append(record)
    results.sort(key=lambda r: (list(by_id).index(r["id"]), r.get("repeat", 0)))
    if unknown:
        print(f"not in the cases file any more, skipped: {', '.join(sorted(set(unknown)))}")
    repeat = meta.get("repeat") or max((r.get("repeat", 0) + 1 for r in results), default=1)
    out = run_dir / f"rescore-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out.mkdir()
    write_reports(results, out, repeat, meta.get("base_url", ""))
    print(f"\nRescored {len(results)} answer(s) from {run_dir} into {out}")
    return 0


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




def select(cases, sets=None, only=None, include_heldout=False, heldout_only=False, skip_extra=False):
    """The cases a run asks for. --only and --set pick from every case; otherwise the
    default excludes held-out cases and cases of a named set (heldout2, seen2), so a
    plain run costs what it always did."""
    if only:
        wanted = {c.strip() for c in only.split(",")}
        cases = [c for c in cases if c["id"] in wanted]
    elif sets:
        wanted = {s.strip() for s in sets.split(",")}
        cases = [c for c in cases if c.get("set") in wanted]
    elif heldout_only:
        cases = [c for c in cases if c.get("heldout") and not c.get("set")]
    elif include_heldout:
        cases = [c for c in cases if not c.get("set")]
    else:
        cases = [c for c in cases if not c.get("heldout") and not c.get("set")]
    if skip_extra:
        cases = [c for c in cases if not c.get("extra")]
    return cases


def main(argv=None):
    parser = argparse.ArgumentParser(description="Acceptance set for narrative and strategy questions.")
    parser.add_argument("--base-url", help="e.g. http://localhost:8000 (on the server)")
    parser.add_argument("--token", help="a JWT instead of logging in")
    parser.add_argument("--cases", default=str(DEFAULT_CASES))
    parser.add_argument("--only", help="comma-separated case ids (from any set)")
    parser.add_argument("--set", dest="sets", help="run only the cases of these sets, e.g. heldout2 or heldout2,seen2")
    parser.add_argument("--skip-extra", action="store_true", help="only Flavio's cases")
    heldout = parser.add_mutually_exclusive_group()
    heldout.add_argument("--include-heldout", action="store_true",
                         help="also run the held-out set (h1-h11), reported in its own section")
    heldout.add_argument("--heldout-only", action="store_true", help="run only the held-out set (h1-h11)")
    parser.add_argument("--out", default="eval_runs", help="folder for the run outputs")
    parser.add_argument("--label", default="", help="added to the run folder name, e.g. baseline")
    parser.add_argument("--timeout", type=int, default=400, help="seconds per question")
    parser.add_argument("--repeat", type=int, default=1,
                        help="run every case N times; report per-article MENTIONED/CITED rates and averages")
    parser.add_argument("--compare", nargs=2, metavar=("OLD_SUMMARY", "NEW_SUMMARY"))
    parser.add_argument("--rescore", metavar="RUN_DIR",
                        help="score a saved run's answers again with the current expected lists (no requests)")
    args = parser.parse_args(argv)

    if args.compare:
        compare(*args.compare)
        return 0
    all_cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))["cases"]
    if args.rescore:
        return rescore(args.rescore, all_cases)
    if not args.base_url:
        parser.error("--base-url is required (or use --compare / --rescore)")
    base_url = args.base_url.rstrip("/")
    cases = select(all_cases, args.sets, args.only, args.include_heldout, args.heldout_only, args.skip_extra)
    if not cases:
        parser.error("no case matches the selection")
    empty = [c["id"] for c in cases if not c.get("expected") and not c.get("expected_docs")]
    if empty:
        print(f"WARNING: no expected articles yet for {', '.join(empty)} -- answers are saved, "
              f"score them later with --rescore")

    token = args.token or login(base_url)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.out) / (f"{stamp}-{args.label}" if args.label else stamp)
    run_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for case in cases:
        for i in range(args.repeat):
            label = f"{case['id']} (r{i + 1}/{args.repeat})" if args.repeat > 1 else case["id"]
            print(f"... {label}", flush=True)
            record = {"id": case["id"], **_case_fields(case), "question": case["question"]}
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

    write_reports(results, run_dir, args.repeat, base_url)
    print(f"\nSaved in {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
