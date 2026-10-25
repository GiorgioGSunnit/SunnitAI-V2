"""The fact-signal rules on the cases of a diagnose_long run, before a full run: which expected articles
the run lost at LOST_AT_UNDERSTANDING a fired rule would reach, and which rules fire without reaching
any expected article (possible false triggers).

Read-only and in-process, as scripts/grader_bench.py: per case of the run with an answer key, one call
(fact_signals.extract_facts) on the case's text, the terms in code, the rules, then each fired
institution resolved against the database as the pipeline resolves it (_resolve_institution, with
the live .env's RESOLUTION_STRICT). Nothing else of the pipeline runs; minutes for 30 cases.

    cd /opt/chatbot && venv/bin/python scripts/fact_signals_bench.py <run folder> [--only h3,n2]

Report: <run>/fact_signals_bench_<date-time>.md (and .json).
"""

import argparse
import importlib.util
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor


def _diagnose_long():
    spec = importlib.util.spec_from_file_location(
        "diagnose_long", os.path.join(os.path.dirname(os.path.abspath(__file__)), "diagnose_long.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bench_case(case: dict, text: str, fs, resolve) -> dict:
    """One case: its facts, terms, fired rules, and per fired institution the articles it resolves to."""
    signals = fs.analyse(text)
    resolved = {}
    for rule in signals["fired"]:
        for name in rule["institutions"]:
            if name not in resolved:
                resolved[name] = resolve(name, rule["codice"])
    lost = [s for s in case["slots"] if s["stage"] == "LOST_AT_UNDERSTANDING"]
    return {"case": case["id"], "signals": signals, "resolved": resolved, "lost": lost}


def render(run_dir: str, results: list, ekey, keys_of, seconds: float, skipped: list) -> str:
    reach, false, lost_total = [], [], 0
    lines = []
    for r in results:
        expected = keys_of[r["case"]]
        lost_keys = {}
        for s in r["lost"]:
            lost_keys[ekey(s["pool_label"])] = s["label"]
        lost_total += len(r["lost"])
        fired = r["signals"]["fired"]
        lines.append(f"### {r['case']}")
        lines.append("Termini: " + ("; ".join(t["text"] for t in r["signals"]["terms"]) or "nessuno"))
        if r["signals"].get("problem"):
            lines.append(f"Estrazione: {r['signals']['problem']}")
        for rule in fired:
            labels = [label for name in rule["institutions"] for label in r["resolved"].get(name, [])]
            hits = [label for label in labels if ekey(label) in expected]
            recovered = [lost_keys[ekey(label)] for label in labels if ekey(label) in lost_keys]
            reach += [(r["case"], rule["id"], label) for label in dict.fromkeys(recovered)]
            if not hits:
                false.append((r["case"], rule["id"], labels))
            lines.append(f"- {rule['id']} → {', '.join(rule['institutions'])}: "
                         + (", ".join(f"**{l}**" if ekey(l) in expected else l for l in labels) or "nessun articolo")
                         + (f" — recupera {', '.join(recovered)}" if recovered else ""))
        if not fired:
            lines.append("- nessuna regola")
        lines.append("")
    head = [f"# Fact signals bench: {os.path.basename(os.path.normpath(run_dir))}", "",
            f"{len(results)} cases, {seconds / 60:.1f} min." + (f" Skipped: {', '.join(skipped)}." if skipped else ""),
            "", f"LOST_AT_UNDERSTANDING in the run: {lost_total}; reached by a fired rule: {len(reach)}"
            + (" (" + ", ".join(f"{c} {l} ← {rule}" for c, rule, l in reach) + ")" if reach else "") + ".",
            f"Fired rules reaching no expected article (possible false triggers): {len(false)}"
            + (" (" + "; ".join(f"{c} {rule}" for c, rule, _ in false) + ")" if false else "") + ".",
            "", "Articles in bold are expected in that case.", ""]
    return "\n".join(head + lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run", help="a diagnose_long run folder, or its name under ./diagnose_long_runs")
    parser.add_argument("--only", help="comma-separated case ids")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--env", default=os.path.join(os.getcwd(), ".env"))
    args = parser.parse_args()

    sys.path.insert(0, os.getcwd())
    from dotenv import load_dotenv
    load_dotenv(args.env)
    dl = _diagnose_long()
    run_dir = dl._run_dir(args.run)
    dl._read_only_neo4j()
    import logging
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")
    logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)
    import src.rag.defensive_generation as dg
    from src.rag import fact_signals as fs
    from src.rag.doc_lookup import _fetch_doc_names
    from src.rag.main import NEO4J_DATABASE, driver
    ev = dl._eval_module()
    ekey = dl._ekey_fn(ev)
    documents = _fetch_doc_names(driver, NEO4J_DATABASE)
    by_id = {c["id"]: c for c in json.load(open(os.path.join(os.getcwd(), "tests", "eval_narrative_cases.json"),
                                                encoding="utf-8"))["cases"]}
    only = {i.strip() for i in (args.only or "").split(",") if i.strip()}
    records, cases = dl._load_cases(run_dir, ev)
    jobs, skipped, keys_of = [], [], {}
    for rec, case in zip(records, cases):
        if not case["keyed"] or (only and rec["id"] not in only):
            continue
        if rec.get("kind") == "file":
            from src.utils.document import extract_text_from_file
            text = dg._case_text(extract_text_from_file(rec["file"]), rec.get("request") or "")
        elif rec["id"] in by_id:
            text = by_id[rec["id"]]["question"]
        else:
            skipped.append(f"{rec['id']} (no text)")
            continue
        keys_of[rec["id"]] = {(str(a).lower(), item["code"]) for item in rec.get("expected") or []
                              for a in item.get("any_of") or [item["article"]]}
        jobs.append((case, text))

    def resolve(name: str, codice: str) -> list:
        with driver.session(database=NEO4J_DATABASE) as session:
            rows = dg._resolve_institution(session, documents, {"nome": name, "codice": codice,
                                                                 "numero_indicativo": ""}, [])
        return list(dict.fromkeys(f"{r['_article'][0]} {r['_article'][1]}" for r in rows if r.get("_article")))

    def run(job):
        case, text = job
        t = time.time()
        try:
            out = bench_case(case, text, fs, resolve)
            print(f"[{case['id']}] {len(out['signals']['fired'])} rules, {time.time() - t:.0f}s", flush=True)
            return out
        except Exception as exc:
            traceback.print_exc()
            skipped.append(f"{case['id']} ({type(exc).__name__}: {str(exc)[:80]})")
            return None

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        results = [r for r in pool.map(run, jobs) if r]
    text = render(run_dir, results, ekey, keys_of, time.time() - t0, skipped)
    path = os.path.join(run_dir, f"fact_signals_bench_{time.strftime('%Y%m%d-%H%M')}")
    with open(path + ".md", "w", encoding="utf-8") as fh:
        fh.write(text)
    with open(path + ".json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=1, default=str)
    print(text + f"\nWritten to {path}.md (and .json)")


if __name__ == "__main__":
    main()
