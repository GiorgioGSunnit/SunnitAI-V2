"""Does the ingest's model analysis make rulings easier to find? Self-retrieval on the 2025 batch, read-only.

Nine in ten rulings of the 2025 batch went into the graph without the model analysis (2-5 Nov 2026, the model name
in the ingest's .env was stale): sections split by size, no number or date. The rest went in with it: sections by
the ruling's structure, number and date filled. The app searches the sections' plain text through the vector index
and never reads anything else the analysis produces, so the only thing it can change in the answers is where the
section boundaries fall. This script measures that: for N rulings of each group, it asks the index the way the app
does (Qwen query instruction + text, k=30, index section_embeddings_v2) with three queries built from the ruling's
own card, and records the rank of the ruling's first section in the result.

  - facts+issues: the card's parole_chiave_fatti + parole_chiave_questioni (a lawyer's case description)
  - argument:     the card's argomento_decisivo (the legal reasoning)
  - quote:        the card's citazione (a verbatim sentence: both groups should hit; a sanity check of the text)

Groups: the rulings of one batch (--batch NAME: the file names the feeder's manifest lists under that batch;
default: the 2025 batch by file-name pattern) with document_number set (analysis) vs NULL (no analysis: the cheap
path, INGEST_SKIP_ANALYSIS, leaves it NULL). Both groups are sampled with the same court-section mix.
Output: per group and query, hit@10, hit@30, share of hits above the app's 0.55 floor, median rank; sections per
ruling and chars per section; JSON to --out. Counts only, no ruling text is printed.

    scp scripts/self_retrieval_check.py sunnit@185.199.24.31:~/ ; scp C:\\tplstage\\cards_6nov\\cards_pen_v3.jsonl sunnit@185.199.24.31:~/
    cd /opt/chatbot && sudo venv/bin/python -I /home/sunnit/self_retrieval_check.py --cards /home/sunnit/cards_pen_v3.jsonl
Runs in a few minutes: 2 x 200 rulings x 3 queries = 1,200 embedding calls (batched) and 1,200 index queries.
    python3 -I self_retrieval_check.py --report /tmp/self_retrieval_check.json     # the table of a saved run
    # the first Cassazione batch, worker 1 (analysis) against worker 2 (cheap path), penal and civil cards:
    ... --batch cass_batch1 --cards /home/sunnit/cards_pen_v3.jsonl,/home/sunnit/cards_civ_v1.jsonl
"""
import argparse
import json
import logging
import random
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional

QUERY_INSTRUCTION = ("Instruct: Given a legal question in Italian, retrieve the passages of laws, rulings and "
                     "commentary that answer it\nQuery: ")                       # src/rag/embedding_config.py
YEAR_2025 = re.compile(r"[@_]a2025[@_]n", re.IGNORECASE)
SEZIONE = re.compile(r"[@_]s(\d{2})[@_]", re.IGNORECASE)   # court section in the Italgiure file name (s70 = settima)
MANIFEST_LINE = re.compile(r"^(\d{4}-\d{2}-\d{2})\s+(\S+)\s+(\S.*)$")   # maint_feed.py: "YYYY-MM-DD batch file"
QUERIES = ("facts+issues", "argument", "quote")


def load_env(path: str) -> Dict[str, str]:
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def card_queries(card: dict) -> Dict[str, str]:
    """The three query texts of a card; a query is dropped when its field is empty."""
    def join(keys):
        parts = []
        for k in keys:
            v = card.get(k) or []
            parts += [str(x).strip() for x in (v if isinstance(v, list) else [v]) if str(x).strip()]
        return ", ".join(parts)
    out = {"facts+issues": join(("parole_chiave_fatti", "parole_chiave_questioni")),
           "argument": join(("argomento_decisivo",)), "quote": (card.get("citazione") or "").strip()}
    return {k: v for k, v in out.items() if v}


def latest_cards(paths: str) -> Dict[str, dict]:
    """One card per ruling id from one or more comma-separated card files: the last good one wins (a retried card
    follows a failed one)."""
    cards: Dict[str, dict] = {}
    for path in [p.strip() for p in paths.split(",") if p.strip()]:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                rid = row.get("id")
                if rid and not row.get("errore"):
                    cards[rid] = row
    return cards


def batch_files(manifest: str, batch: str) -> List[str]:
    """The file names the feeder handed to the queue under this batch name."""
    names = []
    with open(manifest, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = MANIFEST_LINE.match(line.strip())
            if m and m.group(2) == batch:
                names.append(m.group(3).strip())
    return names


def rank_of(rows: List[dict], doc_id: str) -> Optional[int]:
    """1-based rank of the first result belonging to the ruling, None when absent."""
    for i, r in enumerate(rows, 1):
        if r["doc"] == doc_id:
            return i
    return None


def summarise(ranks: List[Optional[int]], scores: List[Optional[float]], k: int, floor: float) -> dict:
    n = len(ranks)
    hits = [r for r in ranks if r is not None]
    return {"rulings": n, "hit@10": sum(1 for r in hits if r <= 10) / n if n else None,
            f"hit@{k}": len(hits) / n if n else None,
            "hit_above_floor": sum(1 for r, s in zip(ranks, scores) if r is not None and s is not None and s >= floor) / n if n else None,
            "median_rank_of_hits": statistics.median(hits) if hits else None}


def print_report(result: dict) -> None:
    """The shape lines and the table, from a fresh result or a saved --out file."""
    k = result["k"]
    for group, g in result["groups"].items():
        mix = ", ".join(f"s{sez} {n}" for sez, n in sorted(g.get("sections_mix", {}).items()))
        print(f"{group}: {g['in_graph']:,} in the graph, {g['with_card']:,} with a complete card, {g['sampled']} sampled"
              f" ({mix}); sections per ruling {g['sections_per_ruling']:.1f}, {g['chars_per_section']:,.0f} chars per section")
    print("\n== Self-retrieval: rank of the ruling's own section among the top %d ==" % k)
    print(f"{'group':<12} {'query':<13} {'rulings':>7} {'hit@10':>7} {'hit@' + str(k):>7} {'>=floor':>8} {'med.rank':>8}")
    pct = lambda x: f"{100 * x:.0f}%" if x is not None else "-"  # noqa: E731
    for group, g in result["groups"].items():
        for q, m in g["queries"].items():
            med = m["median_rank_of_hits"]
            print(f"{group:<12} {q:<13} {m['rulings']:>7} {pct(m['hit@10']):>7} {pct(m[f'hit@{k}']):>7} "
                  f"{pct(m['hit_above_floor']):>8} {med if med is not None else '-':>8}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cards", help="card files copied from prod (/root/ruling_cards), comma-separated")
    p.add_argument("--batch", help="rulings of this batch (the feeder's manifest) instead of the 2025 file-name pattern")
    p.add_argument("--manifest", default="/var/log/maint/upload_batches.txt", help="maint_feed.py's manifest")
    p.add_argument("--report", metavar="JSON", help="print the table of a saved --out file; no database, no cards")
    p.add_argument("--env", default="/opt/v2tools/health.env", help="NEO4J_URI / USER / PASSWORD / DATABASE")
    p.add_argument("--embed-env", default="/opt/sunnitai-be/.env",
                   help="LLM_EMBEDDING_BASE_URL / LLM_EMBEDDING_MODEL / LLM_EMBEDDING_API_KEY (the ingest's)")
    p.add_argument("--index", default="section_embeddings_v2")
    p.add_argument("--per-group", type=int, default=200)
    p.add_argument("--k", type=int, default=30, help="results per query, the app's k for sections")
    p.add_argument("--floor", type=float, default=0.55, help="the app's minimum score for sections")
    p.add_argument("--seed", type=int, default=6)
    p.add_argument("--no-match-sections", dest="match_sections", action="store_false",
                   help="sample each group freely instead of with the same court-section mix")
    p.add_argument("--out", default="/tmp/self_retrieval_check.json")
    args = p.parse_args()
    if args.report:
        with open(args.report, encoding="utf-8") as fh:
            print_report(json.load(fh))
        return 0
    if not args.cards:
        p.error("--cards is required (or --report JSON)")
    # the server answers every vector query with a deprecation notice (queryNodes -> SEARCH) that the driver logs
    # as a warning: one line per query, thousands per run
    logging.getLogger("neo4j").setLevel(logging.ERROR)

    env = load_env(args.env)
    emb = load_env(args.embed_env)
    from neo4j import GraphDatabase, READ_ACCESS
    from openai import OpenAI
    client = OpenAI(base_url=emb["LLM_EMBEDDING_BASE_URL"], api_key=emb.get("LLM_EMBEDDING_API_KEY") or "EMPTY")
    model = emb["LLM_EMBEDDING_MODEL"]

    cards = latest_cards(args.cards)
    print(f"cards: {len(cards):,} rulings with a card")
    driver = GraphDatabase.driver(env["NEO4J_URI"], auth=(env["NEO4J_USER"], env["NEO4J_PASSWORD"]))
    result = {"index": args.index, "k": args.k, "floor": args.floor, "per_group": args.per_group,
              "batch": args.batch or "2025", "groups": {}}
    with driver.session(database=env.get("NEO4J_DATABASE", "neo4j"), default_access_mode=READ_ACCESS) as s:
        names = [r["name"] for r in s.run("SHOW INDEXES YIELD name, type WHERE type = 'VECTOR' RETURN name")]
        if args.index not in names:
            sys.exit(f"vector index {args.index} not found here; vector indexes: {names}")
        if args.batch:
            names = batch_files(args.manifest, args.batch)
            print(f"batch {args.batch}: {len(names):,} files in the manifest")
            docs = s.run("MATCH (d:Document {document_type: 'interpretation'}) WHERE d.file_name IN $names "
                         "RETURN d.id AS id, d.file_name AS file, d.document_number IS NOT NULL AS analysed",
                         names=names).data()
        else:
            docs = s.run("MATCH (d:Document {document_type: 'interpretation'}) "
                         "WHERE d.file_name =~ '(?i).*[@_]a2025[@_]n.*' "
                         "RETURN d.id AS id, d.file_name AS file, d.document_number IS NOT NULL AS analysed").data()
        sez_of = {d["id"]: (SEZIONE.search(d["file"] or "").group(1) if SEZIONE.search(d["file"] or "") else "?")
                  for d in docs}
        groups = {"analysis": [d["id"] for d in docs if d["analysed"]],
                  "no_analysis": [d["id"] for d in docs if not d["analysed"]]}
        print(f"{args.batch or '2025'} rulings in the graph: " + ", ".join(f"{g} {len(ids):,}" for g, ids in groups.items()))
        rng = random.Random(args.seed)
        complete = {g: [i for i in ids if i in cards and len(card_queries(cards[i])) == 3] for g, ids in groups.items()}
        for g, ids in complete.items():
            mix = Counter(sez_of[i] for i in ids)
            print(f"  {g}: {len(ids):,} with a complete card; by court section "
                  + ", ".join(f"s{k} {n:,}" for k, n in sorted(mix.items())))
        samples = {}
        if args.match_sections:
            # the same court-section mix in both groups: the smaller group's sample sets the count per section,
            # capped by what every group holds in that section, so both samples end up the same size
            small = min(complete, key=lambda g: len(complete[g]))
            want = Counter(sez_of[i] for i in rng.sample(complete[small], min(args.per_group, len(complete[small]))))
            by_sez = {g: defaultdict(list) for g in complete}
            for g, ids in complete.items():
                for i in ids:
                    by_sez[g][sez_of[i]].append(i)
            want = Counter({sez: min([n] + [len(by_sez[g][sez]) for g in complete]) for sez, n in want.items()})
            for g in complete:
                samples[g] = [i for sez, n in want.items() for i in rng.sample(by_sez[g][sez], n)]
            print("  samples matched on court section: " + ", ".join(f"s{k} {n}" for k, n in sorted(want.items())))
        else:
            samples = {g: rng.sample(ids, min(args.per_group, len(ids))) for g, ids in complete.items()}
        for group, sample in samples.items():
            with_card = complete[group]
            print(f"\n{group}: sampling {len(sample)}")
            shape = s.run("MATCH (d:Document)-[:CONTAINS]->(x:Section) WHERE d.id IN $ids "
                          "WITH d, count(x) AS n, avg(size(coalesce(x.plain_text, ''))) AS chars "
                          "RETURN avg(n) AS sections, avg(chars) AS chars, min(n) AS min_n, max(n) AS max_n",
                          ids=sample).single()
            print(f"  sections per ruling {shape['sections']:.1f} (min {shape['min_n']}, max {shape['max_n']}), "
                  f"{shape['chars']:,.0f} chars per section")
            texts = [(rid, q, text) for rid in sample for q, text in card_queries(cards[rid]).items()]
            vectors = []
            t0 = time.time()
            for start in range(0, len(texts), 32):
                batch = [QUERY_INSTRUCTION + t for _, _, t in texts[start:start + 32]]
                resp = client.embeddings.create(model=model, input=batch)
                vectors += [d.embedding for d in resp.data]
            print(f"  {len(texts)} queries embedded in {time.time() - t0:.0f}s")
            ranks, scores = defaultdict(list), defaultdict(list)
            t0 = time.time()
            for (rid, q, _), vec in zip(texts, vectors):
                rows = s.run("CALL db.index.vector.queryNodes($index, $k, $v) YIELD node, score "
                             "MATCH (d:Document)-[:CONTAINS]->(node) RETURN d.id AS doc, score ORDER BY score DESC",
                             index=args.index, k=args.k, v=vec).data()
                r = rank_of(rows, rid)
                ranks[q].append(r)
                scores[q].append(rows[r - 1]["score"] if r else None)
            print(f"  {len(texts)} index queries in {time.time() - t0:.0f}s")
            result["groups"][group] = {"in_graph": len(groups[group]), "with_card": len(with_card), "sampled": len(sample),
                                       "sections_mix": dict(Counter(sez_of[i] for i in sample)),
                                       "sections_per_ruling": shape["sections"], "chars_per_section": shape["chars"],
                                       "queries": {q: summarise(ranks[q], scores[q], args.k, args.floor) for q in QUERIES}}
    driver.close()

    print()
    print_report(result)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=1)
    print(f"\njson: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
