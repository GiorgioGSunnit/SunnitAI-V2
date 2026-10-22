"""Read-only: would the Situation nodes find the articles our tuning cases expect?

The database holds about 27,700 Situation nodes: short everyday stories, about five
per article of the six main codes (c.c., c.p., c.p.c., c.p.p., Codice del Consumo,
c.p.a.), each with the article it illustrates and an embedding in the vector index
situation_embeddings_v2. Nothing in V2 uses them yet. Users describe stories and the
codes are written in legal language, which is where the pipeline loses most expected
articles (NOT_IN_POOL in loss_funnel.py), so matching the user's story against these
stories could bring the right articles into the pool.

For each tuning case (tests/eval_narrative_cases.json; held-out cases are refused):
embed the case text the way the app embeds a query, take the k nearest situations,
and rank the articles they illustrate two ways: by best score, and by how many of the
article's situations are among the k (ties by best score). Reports the rank of every
expected article, recall@10/@20/@30 per case and overall, and, with --reference <an
eval run folder>, the "new finds": expected articles the live pipeline cited in fewer
than half its runs that the situations rank in the top 20.

Run from the app folder (prod), so it uses the app's code, .env and embedding model:
    cd /opt/chatbot && venv/bin/python scripts/exp_situations.py
    ... --reference /root/eval/eval_runs/20261020-093048-reference-20oct
    ... --only sempronio,c2      ... --k 200

Nothing is written: Neo4j sessions are read-only, one embedding call per case, no
LLM calls. The situations were written by a model and some are legally wrong, so a
good recall here means "a useful candidate source", never "a reliable answer".
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (ROOT, os.path.join(ROOT, "tests"), "/root/eval", os.getcwd()):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(os.getcwd(), ".env"))

import neo4j  # noqa: E402

_open_session = neo4j.Driver.session


def _read_only_session(self, **config):
    config["default_access_mode"] = neo4j.READ_ACCESS
    return _open_session(self, **config)


neo4j.Driver.session = _read_only_session   # every session in this process is read-only
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

import eval_narrative as ev  # noqa: E402  (art_of, code_of_document, _label: the eval's own scoring)

TUNING_CASES = ["sempronio", "sempronio_no_hint", "p1", "p2", "p3", "c1", "c2", "c3", "l1", "l2", "l3", "l4"]
INDEX = os.getenv("SITUATION_VECTOR_INDEX", "situation_embeddings_v2")
KS = (10, 20, 30)


def _default_cases_path() -> Path:
    for candidate in (Path(ROOT) / "tests" / "eval_narrative_cases.json", Path("/root/eval/eval_narrative_cases.json")):
        if candidate.exists():
            return candidate
    return Path(ROOT) / "tests" / "eval_narrative_cases.json"


# --- Pure helpers (unit-tested in tests/test_exp_situations.py) ---------------

def rank_articles(rows):
    """Situation rows (document, article, score; any order) -> two rankings of
    (eval code, article): by best score, and by number of the article's
    situations among the rows (ties by best score). Rows whose document or
    article the eval can't read are skipped."""
    best, hits = {}, {}
    for row in rows:
        code = ev.code_of_document(row.get("document") or "")
        article = ev.art_of(str(row.get("article") or ""))
        if not code or not article:
            continue
        ref = (code, article)
        score = float(row.get("score") or 0.0)
        best[ref] = max(score, best.get(ref, float("-inf")))
        hits[ref] = hits.get(ref, 0) + 1
    by_best = sorted(best, key=lambda r: (-best[r], -hits[r], r))
    by_hits = sorted(best, key=lambda r: (-hits[r], -best[r], r))
    return by_best, by_hits


def expected_rank(ranked, item):
    """1-based rank of an expected item in `ranked`, or None. An "any_of" item
    (c2's Codice del Consumo range) counts at its best-ranked article."""
    articles = [str(a).lower() for a in (item.get("any_of") or [item["article"]])]
    ranks = [ranked.index((item["code"], a)) + 1 for a in articles if (item["code"], a) in ranked]
    return min(ranks) if ranks else None


def recall(ranked, expected, k):
    """(found, total) among the expected items within the top k."""
    found = sum(1 for item in expected if (r := expected_rank(ranked, item)) is not None and r <= k)
    return found, len(expected)


def reference_rates(aggregate):
    """An eval run's aggregate.json -> {case id: {expected label: cited rate}}."""
    rates = {}
    for case_id, data in (aggregate.get("per_case") or {}).items():
        n = data.get("n") or 0
        rates[case_id] = {label: (count / n if n else 0.0) for label, count in (data.get("counts") or {}).items()}
    return rates


def new_finds(expected, ranked, rates, k=20):
    """Expected items the live pipeline cited in fewer than half its runs
    (per `rates`, {label: rate}) that the situations rank within the top k."""
    out = []
    for item in expected:
        label = ev._label(item)
        rank = expected_rank(ranked, item)
        if label in rates and rates[label] < 0.5 and rank is not None and rank <= k:
            out.append((label, rank, rates[label]))
    return out


# --- Live run -------------------------------------------------------------

def _embedding_check(session, embedding_model):
    """The situations' embedding model and size against the app's query embedding."""
    probe = list(embedding_model.embed_query("controllo"))
    rows = session.run("MATCH (x:Situation) WHERE x.embedding IS NOT NULL "
                       "RETURN x.model AS model, size(x.embedding) AS dims, count(*) AS n").data()
    print(f"App query embedding: {len(probe)} dimensions")
    for row in rows:
        warning = "" if row["dims"] == len(probe) else "   <-- DIFFERENT SIZE: results will be meaningless"
        print(f"Situations: model={row['model']!r} dims={row['dims']} count={row['n']}{warning}")


def _situation_rows(session, embedding_model, text, k):
    embedding = list(embedding_model.embed_query(text))
    return session.run(
        "CALL db.index.vector.queryNodes($index, $k, $embedding) YIELD node, score "
        "RETURN node.document AS document, node.article AS article, score ORDER BY score DESC",
        index=INDEX, k=k, embedding=embedding,
    ).data()


def _fmt_ref(ref):
    return f"{ref[1]} {ref[0]}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cases", default=str(_default_cases_path()))
    parser.add_argument("--only", help="comma-separated tuning case ids (default: the 12)")
    parser.add_argument("--k", type=int, default=150, help="nearest situations per case (default 150)")
    parser.add_argument("--reference", help="an eval run folder with aggregate.json, to mark new finds")
    args = parser.parse_args()

    from src.rag.ai_chat import embedding_model
    from src.rag.main import NEO4J_DATABASE, driver

    all_cases = {c["id"]: c for c in json.loads(Path(args.cases).read_text(encoding="utf-8"))["cases"]}
    ids = [i.strip() for i in args.only.split(",")] if args.only else TUNING_CASES
    refused = [i for i in ids if all_cases.get(i, {}).get("heldout")]
    if refused:
        print(f"Skipped held-out cases (never used for tuning): {', '.join(refused)}")
    cases = [all_cases[i] for i in ids if i in all_cases and i not in refused]
    rates = {}
    if args.reference:
        rates = reference_rates(json.loads((Path(args.reference) / "aggregate.json").read_text(encoding="utf-8")))

    totals = {(order, k): [0, 0] for order in ("best", "hits") for k in KS}
    all_new = []
    with driver.session(database=NEO4J_DATABASE) as session:
        _embedding_check(session, embedding_model)
        print(f"Index {INDEX}, {args.k} nearest situations per case\n")
        for case in cases:
            rows = _situation_rows(session, embedding_model, case["question"], args.k)
            by_best, by_hits = rank_articles(rows)
            expected = case["expected"]
            print(f"## {case['id']}  ({len(by_best)} distinct articles among {len(rows)} situations)")
            for item in expected:
                rb, rh = expected_rank(by_best, item), expected_rank(by_hits, item)
                ref = rates.get(case["id"], {}).get(ev._label(item))
                live = f"   live cited {ref:.0%}" if ref is not None else ""
                print(f"   {ev._label(item):45s} rank by score {rb or '-':>4}   by count {rh or '-':>4}{live}")
            print(f"   top 10 by score: {', '.join(_fmt_ref(r) for r in by_best[:10])}")
            for order, ranked in (("best", by_best), ("hits", by_hits)):
                for k in KS:
                    found, total = recall(ranked, expected, k)
                    totals[(order, k)][0] += found
                    totals[(order, k)][1] += total
            finds = new_finds(expected, by_best, rates.get(case["id"], {}))
            all_new += [(case["id"], *f) for f in finds]
            print()

    print("Recall over all cases (expected articles found):")
    for order, name in (("best", "ranked by score"), ("hits", "ranked by count")):
        cells = "   ".join(f"@{k} {totals[(order, k)][0]}/{totals[(order, k)][1]}" for k in KS)
        print(f"   {name:16s} {cells}")
    if args.reference:
        print(f"\nNew finds (live pipeline cited < 50% of runs, situations rank them in the top 20): {len(all_new)}")
        for case_id, label, rank, rate in all_new:
            print(f"   {case_id:18s} {label:40s} rank {rank:>3}   live cited {rate:.0%}")
    print("\nDone. Nothing was written.")


if __name__ == "__main__":
    main()
