"""Multilingual cross-encoders against the Qwen graded check, offline, on one diagnose_long run's saved data.

bge-reranker-v2-m3 by default; jina-reranker-v2 only with --models bge,jina (its licence is non-commercial and its
modelling code comes from the Hub). No model of ours runs, no answer is written, Neo4j is only read, and nothing in src/
is called but pure helpers (the area's codes, article_text.py). Per case of the run:

  candidates  what the check received (check.input, else the pool), each once by its eval key
  text        the rubrica and text the graded call was sent (llm/NN_applicability*.txt). A pool row holds no text, so
              every other candidate (exempt, past the check's slots), and a text under SHORT_TEXT characters, is read
              from Neo4j, read-only, from the document its pool row names, by article_text.fetch_article, the
              pipeline's own reading; once, kept in <run>/rerank_bench_texts.json. material.txt and the pool's rubrica
              are only the fallback, for an article not found or with --no-fetch; the report counts each source. All
              are cut to --doc-chars (default 400: what the grader reads). The grader's own evidence lines
              ("Trovato da", "Materia", ...) are not given to the rerankers.
  grades      check_result.verdicts: the logged result after the retry, grade 0 included
  fusion      fusion.fused, then the other candidates in their order
  expected    the record's expected items; an any_of group is one slot, hit by its best alternative

Queries: whole = the case text + request; issues = each issue the grader is given (frame questions, central issues,
analysis questions, at most 10), an article's score the max over them (a case without issues uses whole).

Rankings: fusion; qwen (exempt, then grade 3, 2, 1, without a grade, 0; fusion rank within each); each model on each
query; <model>+pin, the same with the exempt articles first in fusion order, as the pipeline would still keep them if
a reranker took the check's place: "+pin" is the row to set against qwen.

Metrics: Recall@10/@20 of the slots and the ceiling (slots with an alternative among the candidates); MRR (1 / rank of
the case's best expected article, 0 when none is there); lost / gained against Qwen's selection (exempt + graded
--keep-grade or more), taking from each ranking as many articles as Qwen selected; unrelated share of the top 10 (graded
0-1 by Qwen, or of a main code outside the case's areas; an expected article never counts), with apart the other-area
share, the only part that does not rest on Qwen, and the share never graded.

Scores are saved per case under <run>/rerank_bench_scores/, so a run stopped halfway resumes; --report-only rebuilds the
report from them. Report: <run>/rerank_bench_<date-time>.md (and .json).

    cd /opt/chatbot && nice -n 19 /opt/rerank_bench/venv/bin/python scripts/rerank_bench.py \\
        diagnose_long_runs/rv2-20261031-0834-candidate [--models bge,jina] [--device cpu] [--batch 16] [--threads 4]

Needs the Neo4j settings of --env (default ./.env), unless --no-fetch or --report-only.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import re
import statistics
import sys
import time
from datetime import datetime

MODELS = {
    "bge": {"name": "BAAI/bge-reranker-v2-m3", "max_length": 512, "remote_code": False, "revision": None},
    # Its modelling code comes from the Hub and runs here: pinned to the commit tested, which needs transformers<5.
    "jina": {"name": "jinaai/jina-reranker-v2-base-multilingual", "max_length": 1024, "remote_code": True,
             "revision": "9cfeff2df7d40d1b78e75e5e9cebec92a99813c9"},
}
MODES = ("whole", "issues")
MAX_ISSUES = 10                  # defensive_generation._GRADED_MAX_ISSUES
UNRELATED_GRADES = (0, 1)
SHORT_TEXT = 60                  # article_text.SHORT_FIRST: a shorter text is a fragment, not the article
GRADED_CHARS = 400               # defensive_generation._GRADED_TEXT_CHARS: a graded call's text stops here
CODE_PREFIXES = {"c.c.": "Codice Civile", "c.p.c.": "Codice di procedura civile", "c.p.": "Codice Penale",
                 "c.p.p.": "Codice di procedura penale", "c.p.a.": "Codice del processo amministrativo",
                 "c.d.s.": "Codice della strada"}     # defensive_generation._CODE_PREFIXES, for a row without its document
TIER = {3: 1, 2: 2, 1: 3, None: 4, 0: 5}      # qwen's order; exempt articles are tier 0
_SCRIPTS = os.path.dirname(os.path.abspath(__file__))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_SCRIPTS, f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- pure: a case from the saved run --------------------------------------------------------------------------------

def material_texts(path: str) -> dict:
    """{label: text} of the articles in a case's material.txt, without the lines added for the reader (">> ")."""
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    block = text.partition("========== ARTICOLI ==========")[2].partition("========== SENTENZE ==========")[0]
    pieces = re.split(r"(?m)^\[art\. (.+?)\]$", block)
    out = {}
    for label, body in zip(pieces[1::2], pieces[2::2]):
        lines = [ln for ln in body.split("\n") if ln.strip() and not ln.startswith(">> ")]
        out.setdefault(label.strip(), " ".join(" ".join(lines).split()))
    return out


def case_issues(rec: dict) -> list:
    """What the grader is given as the case's issues (defensive_generation._graded_issues, which needs the app)."""
    frame = rec.get("frame") or {}
    items = [str(t).strip() for t in frame.get("tasks") or [] if str(t or "").strip()]
    items += [str(c.get("nome")).strip() for c in frame.get("central_issue_candidates") or []
              if isinstance(c, dict) and c.get("nome")]
    items += [re.sub(r"^Questione centrale:\s*", "", str(q.get("titolo"))).strip()
              for q in rec.get("questioni") or [] if isinstance(q, dict) and q.get("titolo")]
    seen, out = set(), []
    for item in items:
        if item.lower() not in seen:
            seen.add(item.lower())
            out.append(item)
    return out[:MAX_ISSUES]


def allowed_for(rec: dict, text: str):
    """(the main codes of the case's areas or None when unknown, the main codes)."""
    sys.path.insert(0, os.getcwd())
    from src.rag import vote_pool
    frame_areas = [str(a).strip().lower() for a in (rec.get("frame") or {}).get("areas") or [] if str(a).strip()]
    area = str(rec.get("area") or "").strip().lower()
    try:
        areas = vote_pool.case_areas({"frame": rec.get("frame") or {}, "area": rec.get("area")})
    except ImportError:          # the bench's venv has not the app's packages
        areas = frame_areas or ([area] if area else [])
    return vote_pool.allowed_codes(areas, vote_pool.named_codes(text)), vote_pool.MAIN_CODES


def document_text(label: str, rubrica: str, text: str, doc_chars: int) -> str:
    return " ".join(p for p in (f"Art. {label}.", rubrica, text[:doc_chars]) if p)


def build_case(rec: dict, query: str, inputs: dict, material: dict, ekey, allowed, main_codes, doc_chars: int) -> dict:
    """The case as the benchmark reads it, or {"skip": why}. inputs: grader_bench.graded_inputs."""
    check = rec.get("check") or {}
    pool = {row["label"]: row for row in rec.get("pool") or []}
    labels = check.get("input") or list(pool)
    verdicts = (rec.get("check_result") or {}).get("verdicts") or {}
    if not labels:
        return {"skip": "no pool"}
    if not rec.get("expected"):
        return {"skip": "no answer key"}
    if not any(str(v).startswith("grade ") for v in verdicts.values()):
        return {"skip": "no grades (CHECK_GRADED off?)"}
    fused = {}
    for i, label in enumerate((rec.get("fusion") or {}).get("fused") or []):
        fused.setdefault(ekey(label), i)
    exempt = {ekey(label) for label in check.get("exempt") or {}}
    grades = {}
    for label, verdict in verdicts.items():
        m = re.match(r"grade (\d)", str(verdict))
        if m:
            grades[ekey(label)] = int(m.group(1))
    issues = {ekey(label): issue for label, issue in ((rec.get("check_result") or {}).get("issues") or {}).items()}
    candidates, seen = [], set()
    for i, label in enumerate(labels):
        key = ekey(label)
        if key in seen:
            continue
        seen.add(key)
        rubrica = (pool.get(label) or {}).get("rubrica") or ""
        if label in inputs:
            rubrica, text, source = inputs[label][0] or rubrica, inputs[label][1], "graded call"
        elif material.get(label):
            text, source = material[label], "material.txt"
        else:
            text, source = "", "rubrica only" if rubrica else "label only"
        code = label.partition(" ")[2].strip()
        candidates.append({"key": key, "label": label, "rubrica": rubrica, "source": source, "text": text,
                           "document": (pool.get(label) or {}).get("document") or "",
                           "doc": document_text(label, rubrica, text, doc_chars),
                           "fusion": fused.get(key, len(fused) + i), "exempt": key in exempt,
                           "grade": grades.get(key), "issue": issues.get(key) or "",
                           "other_area": bool(allowed is not None and code in main_codes and code not in allowed)})
    slots = [{"label": item.get("label") or f"art. {'/'.join(map(str, item.get('any_of') or [item['article']]))} "
                                            f"{item['code']}",
              "keys": [(str(a).lower(), item["code"]) for a in item.get("any_of") or [item["article"]]]}
             for item in rec["expected"]]
    seconds = sum((e[2] or 0) / 1000 for e in rec.get("events") or []
                  if isinstance(e, (list, tuple)) and len(e) >= 3 and e[0] == "case_analysis_applicability")
    return {"id": rec["id"], "query": query, "issues": case_issues(rec), "candidates": candidates, "slots": slots,
            "qwen_seconds": seconds}


# --- the articles' real text ----------------------------------------------------------------------------------------

def is_thin(candidate: dict, doc_chars: int) -> bool:
    """The candidate's saved text is not the article's, or stops before the --doc-chars asked."""
    if candidate["source"] != "graded call" or len(candidate["text"]) < SHORT_TEXT:
        return True
    return doc_chars > GRADED_CHARS and len(candidate["text"]) >= GRADED_CHARS


def document_ids(session, document: str, code: str) -> list:
    """The ids of the document the pool row names (the pipeline's "N <name>" label names it too); a code by its
    prefix, as defensive_generation._fetch_articles reads it, for a row without one."""
    name = document or ("" if code in CODE_PREFIXES else code)
    if name:
        ids = [r["id"] for r in session.run("MATCH (d:Document {name: $name}) RETURN d.id AS id", name=name).data()]
        if ids:
            return ids
    if code not in CODE_PREFIXES:
        return []
    return [r["id"] for r in session.run("MATCH (d:Document) WHERE d.document_type = 'primary' "
                                         "AND d.name STARTS WITH $prefix RETURN d.id AS id",
                                         prefix=CODE_PREFIXES[code]).data()]


def read_article(session, document: str, label: str) -> tuple:
    """(rubrica, text) of the article, the fullest among the document's copies, by article_text.fetch_article; ("", "")
    when not found."""
    from src.rag import article_text
    number, _, code = label.partition(" ")
    best = ("", "")
    for doc_id in document_ids(session, document, code.strip()):
        rows = article_text.fetch_article(session, doc_id, number)
        text = " ".join(" ".join(((r.get("s") or {}).get("plain_text") or (r.get("s") or {}).get("abstract") or "").split())
                        for r in rows).strip()
        if len(text) > len(best[1]):
            section = rows[0].get("s") or {}
            best = ((section.get("titolo_articolo") or section.get("title") or "").strip(), text)
    return best


def apply_texts(cases: list, session, cache: dict, doc_chars: int) -> dict:
    """The thin candidates with their article's text, in place: from `cache` ({"document|label": [rubrica, text]}),
    else from `session` (None: the cache alone). A thin candidate not found keeps its saved fallback. Counts returned."""
    counts = {"thin": 0, "fetched": 0, "not_found": 0, "reads": 0}
    for case in cases:
        for c in case["candidates"]:
            if not is_thin(c, doc_chars):
                continue
            counts["thin"] += 1
            key = f"{c['document']}|{c['label']}"
            if key not in cache and session is not None:
                cache[key] = list(read_article(session, c["document"], c["label"]))
                counts["reads"] += 1
            rubrica, text = cache.get(key) or ("", "")
            if not text:
                counts["not_found"] += 1
                continue
            c["rubrica"] = c["rubrica"] or rubrica
            c["text"], c["source"] = text, "neo4j"
            c["doc"] = document_text(c["label"], c["rubrica"], text, doc_chars)
            counts["fetched"] += 1
    return counts


def fetch_texts(cases: list, path: str, args) -> dict:
    """apply_texts with the saved texts at `path`, and a read-only session unless --no-fetch or --report-only."""
    cache = {}
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            cache = json.load(fh)
    if args.no_fetch or args.report_only:
        return apply_texts(cases, None, cache, args.doc_chars)
    import neo4j
    from dotenv import load_dotenv
    load_dotenv(args.env)
    uri, database = os.getenv("NEO4J_URI"), os.getenv("NEO4J_DATABASE", "neo4j")
    if not uri:
        sys.exit(f"NEO4J_URI is not set ({args.env}): give --env, or --no-fetch to score the saved fallbacks")
    with neo4j.GraphDatabase.driver(uri, auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))) as driver:
        with driver.session(database=database, default_access_mode=neo4j.READ_ACCESS) as session:
            counts = apply_texts(cases, session, cache, args.doc_chars)
    with open(path + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(cache, fh, ensure_ascii=False)
    os.replace(path + ".tmp", path)
    return counts


# --- pure: rankings and metrics -------------------------------------------------------------------------------------

def fusion_order(candidates: list) -> list:
    return sorted(candidates, key=lambda c: c["fusion"])


def qwen_order(candidates: list) -> list:
    return sorted(candidates, key=lambda c: (0 if c["exempt"] else TIER[c["grade"]], c["fusion"]))


def score_order(candidates: list, scores: dict, pin: bool = False) -> list:
    """By score, fusion rank on a tie; pin: the exempt articles first, in fusion order."""
    order = sorted(candidates, key=lambda c: (-scores.get(c["label"], float("-inf")), c["fusion"]))
    if pin:
        return fusion_order([c for c in order if c["exempt"]]) + [c for c in order if not c["exempt"]]
    return order


def selection(candidates: list, keep_grade: int) -> set:
    """Qwen's selection: the exempt articles and those graded keep_grade or more."""
    return {c["key"] for c in candidates if c["exempt"] or (c["grade"] is not None and c["grade"] >= keep_grade)}


def metrics(case: dict, order: list, keep_grade: int) -> dict:
    position = {c["key"]: i for i, c in enumerate(order, 1)}
    selected = selection(case["candidates"], keep_grade)
    n = len(selected)
    out = {"slots": len(case["slots"]), "ceiling": 0, "hit10": 0, "hit20": 0, "lost": [], "gained": [], "selected": n}
    best = None
    expected = set()
    for slot in case["slots"]:
        expected.update(slot["keys"])
        ranks = [position[k] for k in slot["keys"] if k in position]
        rank = min(ranks) if ranks else None
        if rank is None:
            continue
        out["ceiling"] += 1
        out["hit10"] += rank <= 10
        out["hit20"] += rank <= 20
        best = rank if best is None else min(best, rank)
        kept = any(k in selected for k in slot["keys"])
        if kept and rank > n:
            out["lost"].append(slot["label"])
        elif not kept and rank <= n:
            out["gained"].append(slot["label"])
    out["rr"] = 1 / best if best else 0.0
    top = order[:10]
    others = [c for c in top if c["key"] not in expected]
    out["top"] = len(top)
    out["unrelated"] = sum(1 for c in others if c["other_area"] or c["grade"] in UNRELATED_GRADES)
    out["other_area"] = sum(1 for c in others if c["other_area"])
    out["never_graded"] = sum(1 for c in others if c["grade"] is None and not c["exempt"])
    return out


def rankings(case: dict, scores: dict) -> dict:
    """{ranking name: order}; scores: {model: {mode: {label: score}}}."""
    candidates = case["candidates"]
    out = {"fusion": fusion_order(candidates), "qwen": qwen_order(candidates)}
    for model, by_mode in scores.items():
        for mode, table in by_mode.items():
            out[f"{model} {mode}"] = score_order(candidates, table)
            out[f"{model}+pin {mode}"] = score_order(candidates, table, pin=True)
    return out


def summarise(rows: list) -> dict:
    n = max(1, sum(r["slots"] for r in rows))
    top = max(1, sum(r["top"] for r in rows))
    return {"cases": len(rows), "slots": sum(r["slots"] for r in rows), "ceiling": sum(r["ceiling"] for r in rows),
            "hit10": sum(r["hit10"] for r in rows), "hit20": sum(r["hit20"] for r in rows),
            "recall10": 100 * sum(r["hit10"] for r in rows) / n, "recall20": 100 * sum(r["hit20"] for r in rows) / n,
            "ceiling_pct": 100 * sum(r["ceiling"] for r in rows) / n,
            "mrr": sum(r["rr"] for r in rows) / max(1, len(rows)),
            "lost": sum(len(r["lost"]) for r in rows), "gained": sum(len(r["gained"]) for r in rows),
            "unrelated": 100 * sum(r["unrelated"] for r in rows) / top,
            "other_area": 100 * sum(r["other_area"] for r in rows) / top,
            "never_graded": 100 * sum(r["never_graded"] for r in rows) / top}


def examples(cases: list, ranked: dict, keep_grade: int, limit: int) -> list:
    """Up to `limit` expected articles Qwen selected and a reranker's ranking loses, one per (case, article): the +pin
    rows first (a loss there is what replacing the check would cost), worst rank first."""
    found = {}
    for pin_first in (True, False):
        for case in cases:
            selected = selection(case["candidates"], keep_grade)
            by_key = {c["key"]: c for c in case["candidates"]}
            for name, order in ranked[case["id"]].items():
                if name.split()[0] in ("fusion", "qwen") or ("+pin" in name) != pin_first:
                    continue
                position = {c["key"]: i for i, c in enumerate(order, 1)}
                for slot in case["slots"]:
                    keys = [k for k in slot["keys"] if k in position]
                    if not any(k in selected for k in keys):
                        continue
                    key = min(keys, key=lambda k: position[k])
                    if position[key] <= len(selected):
                        continue
                    item = found.setdefault((case["id"], slot["label"]), {
                        "case": case["id"], "article": slot["label"], "pin": pin_first, "rankers": [],
                        "selected": len(selected), "pool": len(order), "candidate": by_key[key], "worst": 0})
                    if item["pin"] == pin_first:
                        item["rankers"].append((name, position[key]))
                        item["worst"] = max(item["worst"], position[key] / max(1, len(order)))
    items = sorted(found.values(), key=lambda i: (not i["pin"], -i["worst"], i["case"], i["article"]))
    return items[:limit]


# --- the models -----------------------------------------------------------------------------------------------------

class Scorer:
    """A cross-encoder through sentence-transformers: predict([(query, document)]) -> [score]."""

    def __init__(self, short: str, device: str, batch: int, threads: int):
        import torch
        from sentence_transformers import CrossEncoder
        if threads:
            torch.set_num_threads(threads)
        spec = MODELS[short]
        self.batch, self.max_length = batch, spec["max_length"]
        self.model = CrossEncoder(spec["name"], device=device, max_length=spec["max_length"],
                                  trust_remote_code=spec["remote_code"], revision=spec["revision"])

    def predict(self, pairs: list) -> list:
        return [float(s) for s in self.model.predict(pairs, batch_size=self.batch, show_progress_bar=False)]

    def tokens(self, text: str) -> int:
        return len(self.model.tokenizer(text, truncation=False)["input_ids"])


def score_case(scorer, case: dict) -> dict:
    """{"whole": {label: score}, "issues": {label: score}, seconds per mode, pairs, the queries' tokens}."""
    docs = [(c["label"], c["doc"]) for c in case["candidates"]]
    t0 = time.time()
    whole = dict(zip((d[0] for d in docs), scorer.predict([(case["query"], d[1]) for d in docs])))
    whole_seconds = time.time() - t0
    issues = case["issues"]
    t0 = time.time()
    if issues:
        flat = scorer.predict([(issue, d[1]) for issue in issues for d in docs])
        by_issue = [flat[i:i + len(docs)] for i in range(0, len(flat), len(docs))]
        best = {d[0]: max(row[i] for row in by_issue) for i, d in enumerate(docs)}
    else:
        best = dict(whole)
    return {"whole": whole, "issues": best, "seconds": {"whole": whole_seconds, "issues": time.time() - t0},
            "pairs": len(docs) * (1 + len(issues)), "query_tokens": scorer.tokens(case["query"]),
            "max_length": getattr(scorer, "max_length", None)}


def inputs_hash(case: dict) -> str:
    """Of what a model reads for the case: saved scores of other queries or texts are not reused."""
    parts = [case["query"], *case["issues"], *(c["doc"] for c in case["candidates"])]
    return hashlib.sha1("\n".join(parts).encode("utf-8")).hexdigest()


def load_scores(path: str, case: dict):
    """The saved scores of a case, or None when missing or made from other inputs."""
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as fh:
        saved = json.load(fh)
    return saved if saved.get("inputs") == inputs_hash(case) else None


def score_all(cases: list, models: list, scores_dir: str, args) -> dict:
    """{model: {case id: saved scores}}: read when already there, else scored and written at once."""
    out = {}
    for short in models:
        folder = os.path.join(scores_dir, short)
        os.makedirs(folder, exist_ok=True)
        out[short] = {}
        scorer = None
        for n, case in enumerate(cases, 1):
            path = os.path.join(folder, re.sub(r"[^\w@.-]+", "_", case["id"]) + ".json")
            saved = load_scores(path, case)
            if saved is None:
                if args.report_only:
                    continue
                if scorer is None:
                    print(f"{short}: loading {MODELS[short]['name']} on {args.device}", flush=True)
                    scorer = Scorer(short, args.device, args.batch, args.threads)
                saved = {**score_case(scorer, case), "inputs": inputs_hash(case)}
                with open(path + ".tmp", "w", encoding="utf-8") as fh:
                    json.dump(saved, fh, ensure_ascii=False)
                os.replace(path + ".tmp", path)
                print(f"  {short} {n}/{len(cases)} {case['id']}: {saved['pairs']} pairs, "
                      f"{sum(saved['seconds'].values()):.0f}s", flush=True)
            out[short][case["id"]] = saved
    return out


# --- the report -----------------------------------------------------------------------------------------------------

def _ranking_rows(names: list, by_ranking: dict) -> list:
    L = ["| ranking | Recall@10 | Recall@20 | ceiling | MRR | lost | gained | unrelated top 10 | other area | "
         "never graded |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in names:
        s = summarise(by_ranking[name])
        L.append(f"| {name} | {s['recall10']:.1f}% ({s['hit10']}/{s['slots']}) | {s['recall20']:.1f}% ({s['hit20']}/"
                 f"{s['slots']}) | {s['ceiling_pct']:.1f}% ({s['ceiling']}) | {s['mrr']:.3f} | {s['lost']} | "
                 f"{s['gained']} | {s['unrelated']:.1f}% | {s['other_area']:.1f}% | {s['never_graded']:.1f}% |")
    return L


def render(run_dir: str, cases: list, skipped: list, table: dict, ranked: dict, saved: dict, keep_grade: int,
           doc_chars: int, limit: int, fetched: dict) -> str:
    names = list(table)
    base = [n for n in names if n in ("fusion", "qwen")]
    sources, expected_sources = {}, {}
    for case in cases:
        keys = {k for slot in case["slots"] for k in slot["keys"]}
        for c in case["candidates"]:
            sources[c["source"]] = sources.get(c["source"], 0) + 1
            if c["key"] in keys:
                expected_sources[c["source"]] = expected_sources.get(c["source"], 0) + 1
    L = [f"# Reranker bench: {os.path.basename(os.path.normpath(run_dir))}", "",
         f"{len(cases)} cases, {sum(len(c['slots']) for c in cases)} expected slots, "
         f"{statistics.mean(len(c['candidates']) for c in cases):.0f} candidates per case, Qwen's selection "
         f"{statistics.mean(len(selection(c['candidates'], keep_grade)) for c in cases):.0f} per case (exempt + graded "
         f"{keep_grade} or more)." + (" Skipped: " + "; ".join(skipped) + "." if skipped else ""), "",
         "Lost / gained: expected slots Qwen selected and the ranking leaves out, or the reverse, taking as many "
         "articles as Qwen selected in that case. +pin keeps the exempt articles first: compare it with qwen. Unrelated "
         "rests on Qwen's own grades, so it favours qwen by construction; other area does not. An expected article "
         "never counts as unrelated.", "",
         "Text given to the rerankers (first " + str(doc_chars) + " characters): "
         + ", ".join(f"{k} {v}" for k, v in sorted(sources.items(), key=lambda x: -x[1]))
         + "; of the expected articles among the candidates: "
         + ", ".join(f"{k} {v}" for k, v in sorted(expected_sources.items(), key=lambda x: -x[1])) + ".", "",
         f"Read from Neo4j (article_text.fetch_article): {fetched['fetched']} of the {fetched['thin']} candidates "
         f"without their full text; not found, left on their saved fallback: {fetched['not_found']}."]
    for mode in MODES:
        chosen = base + [n for n in names if n.endswith(f" {mode}")]
        L += ["", f"## Query: {mode}", ""] + _ranking_rows(chosen, table)
    timing = ["## Time per case", "", "| ranking | pairs per case | mean | median | max | seconds per 100 pairs |",
              "|---|---:|---:|---:|---:|---:|"]
    qwen = [c["qwen_seconds"] for c in cases if c["qwen_seconds"]]
    if qwen:
        timing.append(f"| Qwen graded check (from the run, wall time) | - | {statistics.mean(qwen):.1f}s | "
                      f"{statistics.median(qwen):.1f}s | {max(qwen):.1f}s | - |")
    tokens = []
    for short, by_case in saved.items():
        rows = list(by_case.values())
        if not rows:
            continue
        for mode in MODES:
            times = [r["seconds"][mode] for r in rows]
            docs = [len(c["candidates"]) * (1 if mode == "whole" else max(1, len(c["issues"]))) for c in cases
                    if c["id"] in by_case]
            timing.append(f"| {short} {mode} | {statistics.mean(docs):.0f} | {statistics.mean(times):.1f}s | "
                          f"{statistics.median(times):.1f}s | {max(times):.1f}s | "
                          f"{100 * sum(times) / max(1, sum(docs)):.1f}s |")
        over = sum(1 for r in rows if r.get("max_length") and r.get("query_tokens", 0) > r["max_length"] * 0.8)
        tokens.append(f"{short}: {over} of {len(rows)} cases with a case text over 80% of the model's window "
                      f"({rows[0].get('max_length')} tokens), its end cut in the whole query")
    L += [""] + timing
    if tokens:
        L += ["", "; ".join(tokens) + ". In a case where the request closes the text, that is the part lost."]
    L += ["", "## Per case"]
    for name in names:
        L += ["", f"### {name}", "", "| case | hit@10 | hit@20 | in pool | of | RR | lost | gained |",
              "|---|---:|---:|---:|---:|---:|---|---|"]
        for case, row in zip(cases, table[name]):
            L.append(f"| {case['id']} | {row['hit10']} | {row['hit20']} | {row['ceiling']} | {row['slots']} | "
                     f"{row['rr']:.2f} | {', '.join(row['lost']) or '-'} | {', '.join(row['gained']) or '-'} |")
    picked = examples(cases, ranked, keep_grade, limit)
    L += ["", f"## Examples: expected articles Qwen kept and a reranker loses ({len(picked)})", ""]
    for i, item in enumerate(picked, 1):
        c = item["candidate"]
        kept = "exempt" if c["exempt"] else f"grade {c['grade']}"
        L += [f"**{i}. {item['case']}, {item['article']}** ({c['label']}: {c['rubrica'] or 'no rubrica'}). Qwen: {kept}"
              + (f", for «{c['issue']}»" if c["issue"] else "") + f"; selected {item['selected']} of {item['pool']}.",
              "Ranks: " + ", ".join(f"{name} {rank}" for name, rank in item["rankers"]) + f". Text: {c['source']}.",
              f"> {c['doc'][:300]}", ""]
    if not picked:
        L.append("None.")
    return "\n".join(L) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run", help="a diagnose_long run folder, made with --dump-llm (the default)")
    parser.add_argument("--models", default="bge", help="comma-separated: bge, jina (default bge; jina is opt-in: "
                                                            "non-commercial licence, remote code)")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, mps")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4, help="CPU threads (0: torch's default, all of them)")
    parser.add_argument("--doc-chars", type=int, default=400, help="article text per candidate (the grader reads 400)")
    parser.add_argument("--keep-grade", type=int, default=2, help="Qwen's selection: this grade or more")
    parser.add_argument("--only", help="comma-separated case ids")
    parser.add_argument("--examples", type=int, default=10)
    parser.add_argument("--report-only", action="store_true", help="no model, no database: from the saved scores")
    parser.add_argument("--no-fetch", action="store_true", help="no Neo4j: the texts saved in the run only")
    parser.add_argument("--env", default=os.path.join(os.getcwd(), ".env"), help="the Neo4j settings")
    args = parser.parse_args()

    sys.path.insert(0, os.getcwd())
    dl, gb = _load("diagnose_long"), _load("grader_bench")
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in models if m not in MODELS]
    if unknown:
        parser.error(f"unknown models: {', '.join(unknown)}")
    questions = {c["id"]: c["question"] for c in json.load(
        open(os.path.join("tests", "eval_narrative_cases.json"), encoding="utf-8"))["cases"]}
    ekey = dl._ekey_fn(dl._eval_module())
    only = {x.strip() for x in args.only.split(",")} if args.only else None
    cases, skipped = [], []
    cases_dir = os.path.join(args.run, "cases")
    for name in sorted(os.listdir(cases_dir)):
        if not name.endswith(".json") or name.endswith(".input.json"):
            continue
        with open(os.path.join(cases_dir, name), encoding="utf-8") as fh:
            rec = json.load(fh)
        base = dl._rep(rec["id"])[0]
        if only and rec["id"] not in only and base not in only:
            continue
        if rec.get("kind") == "file" or base not in questions:
            skipped.append(f"{rec['id']} (a document: no answer key)")
            continue
        query = "\n\n".join(p for p in (questions[base], rec.get("request") or "") if p.strip())
        folder = os.path.join(cases_dir, rec["id"])
        allowed, main_codes = allowed_for(rec, query)
        case = build_case(rec, query, gb.graded_inputs(os.path.join(folder, "llm"), dl._dump_parts),
                          material_texts(os.path.join(folder, "material.txt")), ekey, allowed, main_codes,
                          args.doc_chars)
        if "skip" in case:
            skipped.append(f"{rec['id']} ({case['skip']})")
        else:
            cases.append(case)
    if not cases:
        sys.exit("No usable case: " + "; ".join(skipped))
    print(f"{len(cases)} cases, {sum(len(c['candidates']) for c in cases)} candidates", flush=True)
    fetched = fetch_texts(cases, os.path.join(args.run, "rerank_bench_texts.json"), args)
    print(f"Article texts: {fetched['fetched']} of {fetched['thin']} thin candidates filled from Neo4j "
          f"({fetched['reads']} read now), {fetched['not_found']} not found", flush=True)

    saved = score_all(cases, models, os.path.join(args.run, "rerank_bench_scores"), args)
    if args.report_only:
        missing = [(m, c["id"]) for m in models for c in cases if c["id"] not in saved[m]]
        if missing:
            print(f"--report-only: {len(missing)} (model, case) without scores, those cases left out", file=sys.stderr)
            cases = [c for c in cases if all(c["id"] in saved[m] for m in models)]
    table, ranked = {}, {}
    for case in cases:
        scores = {m: {mode: saved[m][case["id"]][mode] for mode in MODES} for m in models}
        ranked[case["id"]] = rankings(case, scores)
        for name, order in ranked[case["id"]].items():
            table.setdefault(name, []).append(metrics(case, order, args.keep_grade))
    text = render(args.run, cases, skipped, table, ranked, saved, args.keep_grade, args.doc_chars, args.examples,
                  fetched)
    stem = os.path.join(args.run, f"rerank_bench_{datetime.now().strftime('%Y%m%d-%H%M')}")
    with open(stem + ".md", "w", encoding="utf-8") as fh:
        fh.write(text)
    with open(stem + ".json", "w", encoding="utf-8") as fh:
        json.dump({"summary": {n: summarise(rows) for n, rows in table.items()},
                   "cases": {n: dict(zip((c["id"] for c in cases), rows)) for n, rows in table.items()}},
                  fh, ensure_ascii=False, indent=1, default=str)
    print(f"Report: {stem}.md")


if __name__ == "__main__":
    main()
