"""The direct questions' eval (tests/eval_direct_cases.json), in process: each question through the chat's graph
(src/rag/main.run) with the code of the current folder, the live .env and any --flag settings, every Neo4j session
read-only. Scored per case:

  MENTIONED          the expected articles the answer names with their code (tests/eval_narrative.py's score)
  CITED              the expected articles among the answer's citations (the same score)
  ROUTE              the router's route (ROUTER_ENABLED), else "direct (router off)"
  NON-PRIMARY HITS   rows the answer was built from whose document is neither primary nor special (support
                     rulings excluded) -- the authority tier's leak; with PRIMARY_FIRST_ENABLED also the
                     filter's own check after each node (source_tier_hits, must be 0)
  RULINGS            the rulings the answer cites: how many CITE an article the answer names, how many only a
                     Cassation-boilerplate one (target 0), how many none

Run from the app folder (it writes only under --out):
    cd /opt/chatbot && venv/bin/python tests/eval_narrative.py --direct --label off
    cd /opt/chatbot && venv/bin/python tests/eval_narrative.py --direct --label on \\
        --flag ROUTER_ENABLED=true --flag PRIMARY_FIRST_ENABLED=true
    cd /opt/chatbot && venv/bin/python tests/eval_narrative.py --direct --compare-direct \\
        eval_runs/direct-<A>/summary.json eval_runs/direct-<B>/summary.json

--debug [N] (default 3; the first N cases, or the --only ones) prints for each case: the route and its reasons,
the graph's result keys and the answer, the rows every article-finding node returned (document, section, the
document_type and the exact key it was read from), what the primary filter kept or dropped and why (with
PRIMARY_FIRST_ENABLED and a direct route), and how MENTIONED was computed. Also written to debug.txt.
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_DIRECT_CASES = HERE / "eval_direct_cases.json"
EVAL_CODE_TO_CITES = {"cp": "c.p.", "cc": "c.c.", "cpc": "c.p.c.", "cpp": "c.p.p.", "cpa": "c.p.a."}
ALLOWED_TYPES = ("primary", "special", "unknown")


def _setup(settings):
    sys.path.insert(0, os.getcwd())
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.getcwd(), ".env"))
    for name, value in settings.items():
        os.environ[name] = value
    import logging
    import neo4j
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")
    logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)
    real = neo4j.Driver.session

    def read_only(self, **config):
        config["default_access_mode"] = neo4j.READ_ACCESS
        return real(self, **config)
    neo4j.Driver.session = read_only


def _capture():
    """Every src.rag module's vlog records into one list (the run changes nothing else)."""
    events = []

    def vlog(event, data=None, duration_ms=None, **_):
        events.append((event, data))
    for name, module in list(sys.modules.items()):
        if name.startswith("src.rag") and callable(getattr(module, "vlog", None)):
            module.vlog = vlog
    return events


def answer_refs(answer, mentions):
    """The (article, CITES code) pairs the answer names, for the codes the CITES edges know."""
    return [(a, EVAL_CODE_TO_CITES[c]) for a, c in mentions(answer) if c in EVAL_CODE_TO_CITES]


def run_case(case, ev, events):
    from src.rag import query_router, source_tiers
    from src.rag.main import NEO4J_DATABASE, driver, run

    del events[:]
    t0 = time.time()
    try:
        # As the chat calls it (src/chatbot/session.py): the user's own words as raw_query too -- the route
        # (source_tiers.graph_route) is read from them, not from the query the graph rewrites.
        result = run(case["question"], raw_query=case["question"], skip_calculation=True)
        status = "ok"
    except Exception as exc:
        result, status = {}, f"ERROR {type(exc).__name__}: {str(exc)[:200]}"
    seconds = round(time.time() - t0, 1)
    answer = result.get("answer") or ""
    citations = result.get("citations") or []
    scored = ev.score(case, answer, citations)
    mentioned = sum(1 for e in scored["expected"] if e.get("mentioned"))
    cited = sum(1 for e in scored["expected"] if e.get("cited"))
    route = next((d.get("route") for e, d in events if e == "route_chosen" and isinstance(d, dict)), None)
    if route is None:
        route = query_router.classify(case["question"], log=False)["route"] if query_router.enabled() \
            else "direct (router off)"
    rows = [r for r in result.get("raw_result") or [] if r.get("_source") != source_tiers.SUPPORT_SOURCE]
    rulings = [c.get("document_id") for c in citations if (c.get("document_type") or "") == "interpretation"
               and c.get("document_id")]
    with driver.session(database=NEO4J_DATABASE) as session:
        types = source_tiers.doc_types(session, rows)
        cites = source_tiers.ruling_citations(session, rulings)
    fit = source_tiers.ruling_answer_fit({r: cites.get(r, set()) for r in rulings},
                                         answer_refs(answer, ev.mentions))
    by_type = {}
    for kind in types:
        by_type[kind] = by_type.get(kind, 0) + 1
    after = sum((d or {}).get("non_primary_after", 0) for e, d in events if e == "source_tier_hits")
    return {"id": case["id"], "question": case["question"], "status": status, "seconds": seconds, "route": route,
            "mentioned": mentioned, "cited": cited, "expected": len(scored["expected"]), "scoring": scored,
            "result_keys": sorted(result), "rows": len(rows),
            "rows_by_type": by_type, "non_primary_hits": sum(n for k, n in by_type.items() if k not in ALLOWED_TYPES),
            "non_primary_after_filter": after,
            "support": [d for e, d in events if e == "source_tier_support"],
            "rulings": {k: v for k, v in fit.items() if k != "per_ruling"}, "rulings_shared": fit["per_ruling"],
            "answer": answer, "citations": citations}


def debug_report(record, events, node_rows, ev, max_rows=25):
    """--debug: the lines that show where a case's articles were found, kept or lost, and how it was scored."""
    out = ["", "=" * 100, f"{record['id']}: {record['question']}", "=" * 100]
    route = next((d for e, d in events if e == "route_chosen" and isinstance(d, dict)), None)
    out.append(f"ROUTE {record['route']}" + (f" -- {'; '.join(route.get('reasons') or [])}" if route else
                                              " (no route_chosen event: router off or cached)"))
    out.append(f"STATUS {record['status']}; result keys: {', '.join(record['result_keys'])}")
    out.append(f"ANSWER (the result's 'answer', {len(record['answer'])} chars):")
    out += ["    " + line for line in (record["answer"] or "<empty>").splitlines()]
    out.append(f"NODES ({len(node_rows)} wrapped node outputs; 'type from' is the key the document_type was read from):")
    for item in node_rows:
        if item.get("filter"):
            decisions = item["decisions"]
            kept = sum(1 for d in decisions if d["kept"])
            out.append(f"  [{item['stage']}] filter ON: {len(decisions)} rows, kept {kept}, dropped {len(decisions) - kept}")
            for d in decisions[:max_rows]:
                out.append(f"      {'KEEP' if d['kept'] else 'DROP'} {str(d['doc'])[:45]:45} s={str(d['section'])[:14]:14} "
                           f"type={d['type']:14} from={d['type_from'][:60]}  ({d['why'][:70]})")
        else:
            rows = item["rows"]
            out.append(f"  [{item['stage']}] filter off (route {item.get('route')}): {len(rows)} rows")
            for r in rows[:max_rows]:
                out.append(f"      {str(r['doc'])[:45]:45} s={str(r['section'])[:14]:14} type={str(r['type']):14} "
                           f"from={r['type_from']}")
        hidden = len(item.get("decisions") or item.get("rows") or []) - max_rows
        if hidden > 0:
            out.append(f"      ... {hidden} more")
    for e, d in events:
        if e in ("source_tier_hits", "source_tier_rerun", "source_tier_support", "article_code_check"):
            out.append(f"  {e}: {json.dumps(d, ensure_ascii=False, default=str)[:400]}")
    out.append(f"FINAL raw_result: {record['rows']} rows (support rulings excluded); by type {record['rows_by_type']}")
    named = ev.mentions(record["answer"])
    named = ", ".join(f"{a} {c or '(no code)'}" for a, c in named) or "none"
    out.append(f"MENTIONED = expected articles the answer names WITH their code (eval_narrative.mentions); "
               f"named: {named}")
    for item in record["scoring"]["expected"]:
        out.append(f"    {item['expected']:28} {item['status']:11} mentioned={item['mentioned']} cited={item['cited']}")
    out.append(f"  cited sections: {', '.join(record['scoring']['cited']) or 'none'}")
    out.append(f"  => MENTIONED {record['mentioned']}/{record['expected']}, CITED {record['cited']}/{record['expected']}")
    return out


def summary_lines(records, label):
    lines = [f"Direct eval {label}".strip(), ""]
    lines.append(f"{'case':5} {'route':20} {'MENT':>6} {'CITED':>6} {'rows':>5} {'non-prim':>8} {'rulings':>7} "
                 f"{'cite ans':>8} {'boilerpl':>8} {'unrel':>5}")
    for r in records:
        rl = r["rulings"]
        lines.append(f"{r['id']:5} {str(r['route'])[:20]:20} {r['mentioned']:>3}/{r['expected']:<2} "
                     f"{r.get('cited', 0):>3}/{r['expected']:<2} {r.get('rows', 0):>5} {r['non_primary_hits']:>8} {rl['rulings']:>7} {rl['citing_answer_article']:>8} "
                     f"{rl['boilerplate_only']:>8} {rl['unrelated']:>5}" + ("" if r["status"] == "ok" else f"  {r['status']}"))
    n = len(records) or 1
    total_rulings = sum(r["rulings"]["rulings"] for r in records)
    lines += ["", f"MENTIONED: {sum(r['mentioned'] for r in records)}/{sum(r['expected'] for r in records)} "
                  f"({sum(r['mentioned'] for r in records) / n:.2f}/case); "
              f"CITED: {sum(r.get('cited', 0) for r in records)}/{sum(r['expected'] for r in records)}",
              f"Non-primary hits: {sum(r['non_primary_hits'] for r in records)} "
              f"(after the filter: {sum(r['non_primary_after_filter'] for r in records)}, must be 0)",
              f"Rulings cited: {total_rulings}; citing an answer article: "
              f"{sum(r['rulings']['citing_answer_article'] for r in records)}; boilerplate-only: "
              f"{sum(r['rulings']['boilerplate_only'] for r in records)} (target 0); unrelated: "
              f"{sum(r['rulings']['unrelated'] for r in records)}"]
    return lines


def totals(records):
    return {"cases": len(records), "mentioned": sum(r["mentioned"] for r in records),
            "cited": sum(r.get("cited", 0) for r in records), "rows": sum(r.get("rows", 0) for r in records),
            "expected": sum(r["expected"] for r in records),
            "non_primary_hits": sum(r["non_primary_hits"] for r in records),
            "non_primary_after_filter": sum(r["non_primary_after_filter"] for r in records),
            "rulings": sum(r["rulings"]["rulings"] for r in records),
            "citing_answer_article": sum(r["rulings"]["citing_answer_article"] for r in records),
            "boilerplate_only": sum(r["rulings"]["boilerplate_only"] for r in records),
            "unrelated": sum(r["rulings"]["unrelated"] for r in records)}


def compare_direct(old_path, new_path):
    old, new = (json.loads(Path(p).read_text(encoding="utf-8")) for p in (old_path, new_path))
    print(f"{'':28} {'A':>10} {'B':>10}\n{'':28} {old.get('label') or 'A':>10} {new.get('label') or 'B':>10}")
    for key in ("mentioned", "cited", "expected", "rows", "non_primary_hits", "non_primary_after_filter", "rulings",
                "citing_answer_article", "boilerplate_only", "unrelated"):
        print(f"{key:28} {old['totals'].get(key, ''):>10} {new['totals'].get(key, ''):>10}")
    old_cases = {c["id"]: c for c in old["cases"]}
    print("\nper case: MENTIONED A -> B, route A -> B")
    for c in new["cases"]:
        o = old_cases.get(c["id"]) or {}
        print(f"  {c['id']:5} {o.get('mentioned', '-')}->{c['mentioned']}/{c['expected']}  "
              f"{o.get('route', '-')} -> {c['route']}")


def main(args):
    """Called by tests/eval_narrative.py --direct."""
    if args.compare_direct:
        compare_direct(*args.compare_direct)
        return
    settings = dict(item.split("=", 1) for item in args.set_flags or [])
    _setup(settings)
    import importlib
    ev = importlib.import_module("eval_narrative") if "eval_narrative" in sys.modules else None
    if ev is None:
        sys.path.insert(0, str(HERE))
        ev = importlib.import_module("eval_narrative")
    importlib.import_module("src.rag.main")
    events = _capture()
    debug = getattr(args, "debug", None)
    if debug:
        from src.rag import source_tiers
        source_tiers.DEBUG_ROWS = []
    cases = json.loads(Path(args.direct_cases).read_text(encoding="utf-8"))["cases"]
    if args.only:
        cases = [c for c in cases if c["id"] in {x.strip() for x in args.only.split(",")}]
    elif debug:
        cases = cases[:debug]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.out) / (f"direct-{stamp}" + (f"-{args.label}" if args.label else ""))
    run_dir.mkdir(parents=True, exist_ok=True)
    records, debug_lines = [], []
    for repeat in range(1, max(1, args.repeat) + 1):
        for case in cases:
            if debug:
                del source_tiers.DEBUG_ROWS[:]
            record = run_case(case, ev, events)
            if debug:
                block = debug_report(record, events, source_tiers.DEBUG_ROWS, ev)
                debug_lines += block
                print("\n".join(block), flush=True)
            record["repeat"] = repeat
            records.append(record)
            name = f"{case['id']}" + (f"-r{repeat}" if args.repeat > 1 else "")
            (run_dir / f"{name}.json").write_text(json.dumps(record, ensure_ascii=False, indent=1, default=str),
                                                  encoding="utf-8")
            print(f"{case['id']} r{repeat}: {record['route']}, MENTIONED {record['mentioned']}/{record['expected']}, "
                  f"non-primary {record['non_primary_hits']}, rulings {record['rulings']['rulings']} "
                  f"({record['rulings']['boilerplate_only']} boilerplate-only), {record['seconds']}s", flush=True)
    lines = summary_lines(records, args.label)
    summary = {"label": args.label, "settings": settings, "totals": totals(records),
               "cases": [{k: r[k] for k in ("id", "repeat", "route", "mentioned", "cited", "expected", "rows",
                                             "non_primary_hits",
                                             "non_primary_after_filter", "rulings")} for r in records]}
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    (run_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if debug:
        (run_dir / "debug.txt").write_text("\n".join(debug_lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWritten to {run_dir}")
