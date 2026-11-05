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
  v_vote v0's prompt under GRADER_VOTE: each article alone, three votes, the second highest kept
         (defensive_generation._voted_verdicts; GRADER_VOTE_TEMPERATURE from --env); each vote in the .json
  v_bin  the same material, a yes/no question instead of a grade (BINARY_SYSTEM): "si applica al caso?", and the
         article's clause that applies, quoted word for word from its text; a "sì" whose quote is not in the text
         counts as "no" (quote_in_text). Yes is recorded as grade 2, no as 0, so it is compared on the line of 2 only

    cd /opt/chatbot && venv/bin/python scripts/grader_bench.py <run folder> [--variants v0,v3] [--only c1,h3]

Report: <run>/grader_bench_<date-time>.md (and .json): per version, the grades of the expected and
the other articles, the share graded 2 or more, the expected ones graded 1 or less with the reason;
per version, its flips against the grades the run saved (any change, and across the line of 2); for
v_vote, how many articles' votes split.
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
    "v_vote": {"evidence": False, "issues": False, "examples": False, "vote": True},
    "v_bin": {"evidence": False, "issues": False, "examples": False, "binary": True},
}
BINARY_SYSTEM = (
    "Sei un avvocato italiano. Ti vengono dati i fatti di un caso e alcuni articoli di legge, ciascuno con la sua rubrica "
    "e il suo testo reale. Per ciascun articolo rispondi a una sola domanda: si applica al caso? Rispondi «sì» solo se "
    "una disposizione del testo dell'articolo si applica ai fatti del caso, e riporta quella disposizione copiandola "
    "parola per parola dal Testo, senza cambiare, aggiungere o togliere nulla. Se non si applica, rispondi «no» e lascia "
    "vuota la clausola. Rispondi solo con un array JSON, un oggetto per articolo, nell'ordine dato: "
    "{\"a\": \"A1\", \"applica\": \"sì\" o \"no\", \"clausola\": \"la disposizione copiata dal Testo\"}.")
BINARY_BATCH = 12
BINARY_TOKENS = (200, 160)        # max_tokens: base, per article (its quote)
PROMPT_KEYS = ("evidence", "issues", "examples")
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


def _plain_words(text: str) -> str:
    """Text compared for the quote: typographic quotes and apostrophes as plain ones, spaces collapsed, lower case."""
    text = (text or "").replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"').replace("«", "").replace("»", "")
    return " ".join(text.split()).lower().strip(" .;,:\"'")


def quote_in_text(quote: str, text: str) -> bool:
    """A non-empty quote found in the article's text, as a substring (_plain_words on both)."""
    q = _plain_words(quote)
    return bool(q) and q in _plain_words(text)


def binary_verdicts(dg, case: str, texts: dict, articles: dict) -> dict:
    """{label: {"grade": 2 or 0, "applies", "said", "quote", "quote_in_text"}} for v_bin: BINARY_BATCH articles per call,
    three calls at once, as the graded check sends them (`texts`, dg._graded_texts); `articles` the text each quote must be
    in. An article the reply leaves out has no grade."""
    from concurrent.futures import ThreadPoolExecutor
    from langchain_core.messages import HumanMessage, SystemMessage

    labels = list(texts)
    groups = [labels[i:i + BINARY_BATCH] for i in range(0, len(labels), BINARY_BATCH)]

    def ask(group):
        ids = {f"A{i}": label for i, label in enumerate(group, 1)}
        human = (f"Fatti del caso:\n{case}\n\nArticoli, {len(ids)}: un oggetto per ciascuno degli identificativi "
                 f"{', '.join(ids)}.\n\n" + "\n\n".join(f"[{key}] {label}\n{texts[label]}" for key, label in ids.items()))
        try:
            raw = dg._chat([SystemMessage(content=BINARY_SYSTEM), HumanMessage(content=human)],
                           max_tokens=BINARY_TOKENS[0] + BINARY_TOKENS[1] * len(ids), temperature=0)
        except Exception as exc:
            print(f"v_bin call failed: {exc}", flush=True)
            return {}
        out = {}
        for item in dg._json_objects(raw):
            key = re.match(r"\[?\s*(A\d+)\b", str(item.get("a") or item.get("id") or ""), re.IGNORECASE)
            label = ids.get(key.group(1).upper()) if key else None
            said = str(item.get("applica") or item.get("risposta") or "").strip().lower()
            if not label or label in out or said not in ("sì", "si", "no"):
                continue
            quote = " ".join(str(item.get("clausola") or "").split())
            found = quote_in_text(quote, articles.get(label, ""))
            applies = said != "no" and found
            out[label] = {"grade": 2 if applies else 0, "applies": applies, "said": "sì" if said != "no" else "no",
                          "quote": quote[:400], "quote_in_text": found, "issue": "", "reason": quote[:200]}
        return out
    with ThreadPoolExecutor(max_workers=max(1, min(3, len(groups)))) as pool:
        return {k: v for part in pool.map(ask, groups) for k, v in part.items()}


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


def flip_counts(results: list, version: str) -> dict:
    """How a version's grades differ from the run's own: over the articles both graded, those graded otherwise, those
    on the other side of 2 (applicable or not), how many of these expected; for v_vote, the articles whose votes split."""
    both = [r for r in results if r["grades"].get("saved", {}).get("grade") is not None
            and r["grades"].get(version, {}).get("grade") is not None]
    if not both:
        return {}
    binary = any("applies" in r["grades"][version] for r in both)
    crossed = [r for r in both if (r["grades"]["saved"]["grade"] >= 2) != (r["grades"][version]["grade"] >= 2)]
    # a yes/no answer has no grade to differ by: it changes only across the line of 2
    changed = crossed if binary else [r for r in both if r["grades"]["saved"]["grade"] != r["grades"][version]["grade"]]
    voted = [r for r in results if "votes" in r["grades"].get(version, {})]
    return {"n": len(both), "changed": len(changed), "changed_pct": 100 * len(changed) / len(both),
            "quote_missing": sum(1 for r in both if r["grades"][version].get("said") == "sì"
                                 and not r["grades"][version].get("quote_in_text")) if binary else None,
            "to_yes": sum(1 for r in crossed if r["grades"][version]["grade"] >= 2),
            "to_no": sum(1 for r in crossed if r["grades"][version]["grade"] < 2),
            "crossed": len(crossed), "crossed_pct": 100 * len(crossed) / len(both),
            "crossed_expected": sum(1 for r in crossed if r["expected"]),
            "split": sum(1 for r in voted if len({g for g in r["grades"][version]["votes"] if g is not None}) > 1)
            if voted else None}


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
    L += ["", "## Flips against the grades the run saved", "",
          "v_bin answers yes/no (yes = 2, no = 0): it is compared on the line of 2 only, so its \"changed\" is its "
          "\"across\". \"sì, quote not in the text\": yes answers counted as no.", "",
          "| version | articles | changed | across the line of 2 (to >= 2 / to < 2) | of them expected | votes split | "
          "sì, quote not in the text |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for v in versions:
        flips = flip_counts(results, v)
        if flips:
            L.append(f"| {v} | {flips['n']} | {flips['changed']} ({flips['changed_pct']:.0f}%) | {flips['crossed']} "
                     f"({flips['crossed_pct']:.0f}%; {flips['to_yes']} / {flips['to_no']}) | {flips['crossed_expected']} | "
                     + (str(flips["split"]) if flips["split"] is not None else "-") + " | "
                     + (str(flips["quote_missing"]) if flips["quote_missing"] is not None else "-") + " |")
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
        if flags.get("binary"):
            got = binary_verdicts(dg, dg._graded_case(facts, request, analysis), dg._graded_texts(texts, to_check, None),
                                  {label: inputs[label][1] for label in texts})
            for item in results:
                item["grades"][version] = got.get(item["label"]) or {"grade": None, "issue": "", "reason": ""}
            continue
        issues = dg._graded_issues(analysis) if flags["issues"] else None
        grade = dg._voted_verdicts if flags.get("vote") else dg._graded_verdicts
        verdicts, _ = grade(
            dg._graded_case(facts, request, analysis, issues),
            dg._graded_texts(texts, to_check, context if flags["evidence"] else None),
            dg._graded_system(**{k: flags[k] for k in PROMPT_KEYS}), issues)
        for item in results:
            v = verdicts.get(dg._applicability_key(item["label"])) or {}
            item["grades"][version] = {"grade": dg._grade(v), "issue": v.get("issue") or "",
                                       "reason": v.get("reason") or "", **({"votes": v["votes"]} if "votes" in v else {})}
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
