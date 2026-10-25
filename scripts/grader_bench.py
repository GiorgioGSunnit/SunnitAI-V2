"""Re-run ONLY the graded applicability step on the saved pools of a diagnose_long run, for several
versions of the grader's prompt, and compare how they grade the expected articles and the others.

Read-only and in-process: nothing of the pipeline runs but the graded check itself (one call per 12
articles, as the pipeline makes it). Per case, the articles the run sent to the grader and the exact
rubrica and text it saw come from the run's llm/ dumps (cases/<id>/llm/NN_applicability*.txt, so the
run needs --dump-llm, the default); the case's frame, issues and pool tags come from cases/<id>.json;
the similar rulings' votes are counted again (vote_pool.ruling_vote, read-only), since runs before
GRADER_EVIDENCE_ENABLED did not save them. Exempt articles (named, cross-referenced, found by rubrica)
and those past the checked slots were never graded, so they are not here.

Versions (defensive_generation._graded_system):
  saved  the grades the run itself logged
  v0     the live prompt (CHECK_GRADED_ENABLED), run again: how much the grades move between runs
  v1     + the evidence per article (searches, similar rulings applying it, area) and its rule
  v2     + the grade per numbered issue of the case = GRADER_EVIDENCE_ENABLED
  v3     + the worked examples = GRADER_EVIDENCE_ENABLED with GRADER_EXAMPLES_ENABLED

    cd /opt/chatbot && venv/bin/python scripts/grader_bench.py <run folder> [--variants v0,v3] [--only c1,h3]

Report: <run>/grader_bench_<date-time>.md (and .json): per version, the grades of the expected and
the other articles, the share graded 2 or more, the expected ones graded 1 or less with the reason.
"""

import argparse
import importlib.util
import json
import os
import re
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

VARIANTS = {
    "v0": {"evidence": False, "issues": False, "examples": False},
    "v1": {"evidence": True, "issues": False, "examples": False},
    "v2": {"evidence": True, "issues": True, "examples": False},
    "v3": {"evidence": True, "issues": True, "examples": True},
}
GRADES = (0, 1, 2, 3, None)


def _diagnose_long():
    spec = importlib.util.spec_from_file_location(
        "diagnose_long", os.path.join(os.path.dirname(os.path.abspath(__file__)), "diagnose_long.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- pure: the run's inputs ----------------------------------------------------------------------

_NEW_ENTRY = re.compile(r"^\[(A\d+)\] ([^\n]+)\n(Rubrica: .*?)(?=\n\n\[A\d+\] |\n\n===== |\Z)",
                        re.MULTILINE | re.DOTALL)
_OLD_ENTRY = re.compile(r"^\[([^\]\n]+)\]\n(Rubrica: .*?)(?=\n\n\[[^\]\n]+\]\n|\n\n===== |\Z)",
                        re.MULTILINE | re.DOTALL)


def graded_inputs(llm_dir: str, dump_parts) -> dict:
    """{label: (rubrica, text)} of every article the run's graded calls sent, in the order first sent."""
    out = {}
    for name in sorted(os.listdir(llm_dir)) if os.path.isdir(llm_dir) else []:
        if "applicability" not in name:
            continue
        with open(os.path.join(llm_dir, name), encoding="utf-8") as fh:
            _, parts = dump_parts(fh.read())
        if "voto di pertinenza" not in parts.get("SYSTEM", ""):
            continue
        user = parts.get("USER", "")
        user = user[user.find("Articoli candidati"):] if "Articoli candidati" in user else user
        entries = [(label, body) for _, label, body in _NEW_ENTRY.findall(user)] or _OLD_ENTRY.findall(user)
        for label, body in entries:
            rubrica = re.match(r"Rubrica: (.*)", body).group(1).strip()
            text = body.split("Testo: ", 1)[1].strip() if "Testo: " in body else ""
            out.setdefault(label.strip(), ("" if rubrica == "-" else rubrica, text))
    return out


def pool_tags(rec: dict, dg) -> dict:
    """{label: row tags} the evidence reads, from the run's record: the fusion's signals, the pool's
    multi-signal / vote / method-list tags, the institutions resolved for a central issue."""
    tags = {}
    for row in rec.get("pool") or []:
        tags[row["label"]] = {"_multi_signal": bool(row.get("multi_signal")), "_vote": bool(row.get("vote")),
                              "_source": (dg.METHOD_CANDIDATES_SOURCE
                                          if row.get("sources") == [dg.METHOD_CANDIDATES_SOURCE] else ""),
                              "_signals": []}
    for name, labels in ((rec.get("fusion") or {}).get("signals") or {}).items():
        for label in labels or []:
            tags.setdefault(label, {"_signals": []})["_signals"].append(name)
    analysis = {"frame": rec.get("frame") or {}, "questioni": rec.get("questioni") or []}
    central = dg._central_questions(analysis)
    names = {str(i.get("nome") or "").lower() for idx, q in enumerate(analysis["questioni"]) if idx in central
             for i in q.get("istituti") or [] if isinstance(i, dict)}
    for inst in rec.get("institutions") or []:
        if inst.get("phase") == "v2" and str(inst.get("nome") or "").lower() in names:
            for a in inst.get("articles") or []:
                tags.setdefault(a["label"], {"_signals": []})["_central"] = True
    return tags


def summarise(results: list) -> dict:
    """Per version: grade counts and share >= 2, expected and not."""
    out = {}
    for item in results:
        for version, graded in item["grades"].items():
            s = out.setdefault(version, {"exp": Counter(), "other": Counter()})
            s["exp" if item["expected"] else "other"][graded["grade"]] += 1
    for s in out.values():
        for side in ("exp", "other"):
            n = sum(s[side].values())
            s[f"{side}_n"] = n
            s[f"{side}_hi"] = sum(s[side][g] for g in (2, 3))
            s[f"{side}_pct"] = 100 * s[f"{side}_hi"] / n if n else 0.0
    return out


def render(run_dir: str, results: list, versions: list, seconds: float, skipped: list) -> str:
    summary = summarise(results)
    cases = sorted({r["case"] for r in results})
    L = [f"# Grader bench: {os.path.basename(os.path.normpath(run_dir))}", "",
         f"{len(cases)} cases, {len(results)} graded articles ({sum(r['expected'] for r in results)} expected); "
         f"{seconds / 60:.1f} min." + (f" Skipped: {', '.join(skipped)}." if skipped else ""), "",
         "v2 is GRADER_EVIDENCE_ENABLED, v3 adds GRADER_EXAMPLES_ENABLED; saved is what the run logged. Exempt and unchecked articles are not graded, "
         "so not here.", "",
         "| version | expected | expected >= 2 | other | other >= 2 | expected 0/1/2/3/none | other 0/1/2/3/none |",
         "|---|---:|---:|---:|---:|---|---|"]
    for v in ["saved"] + versions:
        s = summary.get(v)
        if not s:
            continue
        L.append(f"| {v} | {s['exp_n']} | {s['exp_hi']} ({s['exp_pct']:.0f}%) | {s['other_n']} | {s['other_hi']} "
                 f"({s['other_pct']:.0f}%) | " + "/".join(str(s["exp"][g]) for g in GRADES) + " | "
                 + "/".join(str(s["other"][g]) for g in GRADES) + " |")
    base = "v0" if "v0" in versions else "saved"
    for v in versions:
        if v == base:
            continue
        up = Counter(r["expected"] for r in results if v in r["grades"] and base in r["grades"]
                     and (r["grades"][base]["grade"] or 0) <= 1 < (r["grades"][v]["grade"] or 0))
        down = Counter(r["expected"] for r in results if v in r["grades"] and base in r["grades"]
                       and (r["grades"][v]["grade"] or 0) <= 1 < (r["grades"][base]["grade"] or 0))
        L.append(f"\n{v} vs {base}: expected up to >= 2: {up[True]}, down: {down[True]}; others up: {up[False]}, "
                 f"down: {down[False]}.")
    for v in ["saved"] + versions:
        low = [r for r in results if r["expected"] and v in r["grades"] and (r["grades"][v]["grade"] or 0) <= 1]
        if not low:
            continue
        L += ["", f"## {v}: expected articles graded 1 or less ({len(low)})", "",
              "| case | article | grade | issue | reason | evidence |", "|---|---|---:|---|---|---|"]
        for r in low:
            g = r["grades"][v]
            grade = "none" if g["grade"] is None else g["grade"]
            L.append(f"| {r['case']} | {r['label']} | {grade} | {str(g.get('issue') or '')[:60]} | "
                     f"{str(g.get('reason') or '')[:120]} | {'; '.join(r['evidence'])[:160]} |")
    return "\n".join(L) + "\n"


# --- the run ---------------------------------------------------------------------------------------

def bench_case(rec: dict, inputs: dict, versions: list, dg, ev_key, cases_by_id: dict, vote: bool) -> list:
    """The graded results of one case, one entry per article sent, with every version's grade."""
    from src.rag import vote_pool

    if rec.get("kind") == "file":
        from src.utils.document import extract_text_from_file
        facts, request = extract_text_from_file(rec["file"]), rec.get("request") or ""
    else:
        facts, request = cases_by_id[rec["id"]]["question"], ""
    analysis = {"frame": rec.get("frame") or {}, "questioni": rec.get("questioni") or [], "area": rec.get("area")}
    stats = {}
    if vote:
        from src.rag.main import NEO4J_DATABASE, driver
        with driver.session(database=NEO4J_DATABASE) as session:
            vote_pool.ruling_vote(session, driver, NEO4J_DATABASE, dg._case_text(facts, request),
                                  dg._CASE_POOL_VECTOR_INDEX, analysis, stats)
    tags = pool_tags(rec, dg)
    to_check, texts = {}, {}
    for label, (rubrica, text) in inputs.items():
        number, _, code = label.partition(" ")
        ref = (number, code)
        row = {"_article": ref, "s": {"titolo_articolo": rubrica, "plain_text": text}, **tags.get(label, {})}
        if ref in stats:
            row["_vote_stats"] = stats[ref]
        to_check[ref], texts[label] = [row], text
    context = dg._evidence_context(facts, request, analysis)
    expected = {(str(a).lower(), item["code"]) for item in rec.get("expected") or []
                for a in item.get("any_of") or [item["article"]]}
    saved = ((rec.get("check_result") or {}).get("grades") or {})
    results = [{"case": rec["id"], "label": label, "expected": ev_key(label) in expected,
                "evidence": dg._evidence_lines(to_check[tuple(label.partition(" ")[::2])], context),
                "grades": {"saved": {"grade": saved.get(label), "issue": ((rec.get("check_result") or {}).get("issues")
                                                                          or {}).get(label, ""), "reason": ""}}}
               for label in texts]
    for version in versions:
        flags = VARIANTS[version]
        issues = dg._graded_issues(analysis) if flags["issues"] else None
        verdicts, _ = dg._graded_verdicts(
            dg._graded_case(facts, request, analysis, issues),
            dg._graded_texts(texts, to_check, context if flags["evidence"] else None),
            dg._graded_system(**flags), issues)
        for item in results:
            v = verdicts.get(dg._applicability_key(item["label"])) or {}
            item["grades"][version] = {"grade": dg._grade(v), "issue": v.get("issue") or "",
                                       "reason": v.get("reason") or ""}
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run", help="a diagnose_long run folder, or its name under ./diagnose_long_runs")
    parser.add_argument("--variants", default="v0,v1,v2,v3", help="comma-separated (default v0,v1,v2,v3)")
    parser.add_argument("--only", help="comma-separated case ids")
    parser.add_argument("--workers", type=int, default=3, help="cases at once (each makes up to 3 calls at once)")
    parser.add_argument("--no-vote", action="store_true", help="do not count the similar rulings' votes again")
    parser.add_argument("--env", default=os.path.join(os.getcwd(), ".env"))
    args = parser.parse_args()
    versions = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in versions if v not in VARIANTS]
    if unknown:
        parser.error(f"unknown versions: {', '.join(unknown)} (known: {', '.join(VARIANTS)})")

    sys.path.insert(0, os.getcwd())
    from dotenv import load_dotenv
    load_dotenv(args.env)
    dl = _diagnose_long()
    run_dir = dl._run_dir(args.run)
    dl._read_only_neo4j()
    import logging
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")
    logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)
    idf = os.path.join(run_dir, "vote_idf.json")
    if os.path.isfile(idf) and not os.getenv("VOTE_POOL_IDF_FILE"):
        os.environ["VOTE_POOL_IDF_FILE"] = idf
    import src.rag.defensive_generation as dg
    ev = dl._eval_module()
    ekey = dl._ekey_fn(ev)
    cases_by_id = {c["id"]: c for c in json.load(open(os.path.join(os.getcwd(), "tests", "eval_narrative_cases.json"),
                                                      encoding="utf-8"))["cases"]}
    only = {i.strip() for i in (args.only or "").split(",") if i.strip()}

    jobs, skipped = [], []
    cases_dir = os.path.join(run_dir, "cases")
    for name in sorted(os.listdir(cases_dir)):
        if not name.endswith(".json") or name.endswith(".input.json"):
            continue
        with open(os.path.join(cases_dir, name), encoding="utf-8") as fh:
            rec = json.load(fh)
        if (only and rec["id"] not in only) or not rec.get("expected"):
            continue
        inputs = graded_inputs(os.path.join(cases_dir, rec["id"], "llm"), dl._dump_parts)
        if not inputs:
            skipped.append(f"{rec['id']} (no graded call in its llm/ dumps)")
            continue
        jobs.append((rec, inputs))
    print(f"{len(jobs)} cases, {sum(len(i) for _, i in jobs)} articles, versions {', '.join(versions)}", flush=True)

    def run(job):
        rec, inputs = job
        t = time.time()
        try:
            out = bench_case(rec, inputs, versions, dg, ekey, cases_by_id, not args.no_vote)
            print(f"[{rec['id']}] {len(out)} articles, {time.time() - t:.0f}s", flush=True)
            return out
        except Exception as exc:
            traceback.print_exc()
            skipped.append(f"{rec['id']} ({type(exc).__name__}: {str(exc)[:80]})")
            return []

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        results = [item for out in pool.map(run, jobs) for item in out]
    text = render(run_dir, results, versions, time.time() - t0, skipped)
    stamp = time.strftime("%Y%m%d-%H%M")
    path = os.path.join(run_dir, f"grader_bench_{stamp}")
    with open(path + ".md", "w", encoding="utf-8") as fh:
        fh.write(text)
    with open(path + ".json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=1)
    print(text + f"\nWritten to {path}.md (and .json)")


if __name__ == "__main__":
    main()
