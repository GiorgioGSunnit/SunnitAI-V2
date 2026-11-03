"""Read-only against Neo4j: the former article reads (whole nodes, one query per article and spelling) against the new
ones (named properties through article_text.ArticleStore), article by article, on the real corpus.

The sample: every (code, article) the eval cases expect or forbid (tests/eval_narrative_cases.json, any_of groups
expanded), the unexpected articles of diagnose_long reports when --reports is given, and the articles of the tables
(method_candidates' lists, practical_checklist's items, fact_signals' procedural tools, regimes, damages and terms,
the civil damages backbone). Per article, the former Cypher is inline here, as it stood before 3 Nov, never imported:
  - _fetch_articles' query per spelling (the six codes: name prefix, ORDER BY s.name LIMIT 300), and its rule:
    rows grouped by document, article_rows, the first spelling with rows, at most 6;
  - _sections' query per document and spelling (fetch_article, fill_texts, the commi), whole nodes.
The new path is the code of the folder this runs from: defensive_generation._fetch_articles and article_text.
fetch_many with one ArticleStore for the whole sample (as one request), then the store's rows.

Compared per article: (1) the rows the fetch returns (document, Section, text, every projected field, next); (2) per
document and spelling, the Section rows (the fragments by name, each with its projected fields); (3) the text fill_texts'
rule chooses from them; (4) the commi comma_material splits. Every mismatch printed, then a summary and the traffic of
both: queries, rows, bytes of the returned rows as JSON, floats returned (vectors). Exit 1 on any mismatch.

    cd /root/v38 && /opt/chatbot/venv/bin/python scripts/store_equivalence.py --env /opt/chatbot/.env \\
        [--reports diagnose_reports] [--only c.p.] [--verbose]

Nothing is written: every session is READ_ACCESS, the driver is this script's own and closed at the end.
"""

import argparse
import glob
import importlib.util
import json
import os
import re
import sys
from collections import Counter, defaultdict

OLD_FETCH = ("MATCH (d:Document)-[:CONTAINS]->(s:Section) "
             "WHERE d.document_type = 'primary' AND d.name STARTS WITH $prefix "
             "AND (s.name = $n OR s.name STARTS WITH $n + '.' OR s.name STARTS WITH $n + '_') "
             "OPTIONAL MATCH (s)-[:NEXT]->(after:Section) "
             "RETURN d, s, after.name AS next ORDER BY s.name LIMIT 300")
OLD_SECTIONS = ("UNWIND $items AS it "
                "MATCH (d:Document {id: it.doc})-[:CONTAINS]->(s:Section) "
                "WHERE (s.name = it.num OR s.name STARTS WITH it.num + '.' OR s.name STARTS WITH it.num + '_') "
                "OPTIONAL MATCH (s)-[:NEXT]->(after:Section) "
                "RETURN it.doc AS doc, it.num AS num, d, s, after.name AS next")
DOCUMENTS = ("MATCH (d:Document)-[:CONTAINS]->(:Section) WHERE d.document_type IN ['primary', 'special', 'ccnl'] "
             "AND coalesce(d.visibility, 'public') = 'public' "
             "RETURN DISTINCT d.id AS id, d.name AS name, d.document_type AS document_type, "
             "coalesce(d.aliases, []) AS aliases")
EVAL_TO_CODE = {"cp": "c.p.", "cc": "c.c.", "cpp": "c.p.p.", "cpc": "c.p.c.", "cpa": "c.p.a."}


# --- traffic -------------------------------------------------------------------------------------------------

def _floats(value) -> int:
    if isinstance(value, dict):
        return sum(_floats(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        if value and all(isinstance(v, float) for v in value):
            return len(value)
        return sum(_floats(v) for v in value)
    return 0


class Counting:
    """A session whose every run is tallied under the current phase: queries, rows, JSON bytes, floats."""

    def __init__(self, session):
        self.session, self.phase = session, "other"
        self.tally = defaultdict(lambda: [0, 0, 0, 0])

    def run(self, query, parameters=None, **params):
        rows = self.session.run(query, parameters, **params).data()
        t = self.tally[self.phase]
        t[0] += 1
        t[1] += len(rows)
        t[2] += sum(len(json.dumps(r, default=str, ensure_ascii=False).encode("utf-8")) for r in rows)
        t[3] += sum(_floats(r) for r in rows)
        return type("Result", (), {"data": lambda self_: rows})()


# --- the sample ----------------------------------------------------------------------------------------------

def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample(dg, reports: str) -> list:
    """[(number, code, source)]: code a key of _CODE_PREFIXES, an eval key ("consumo"), or a law's name."""
    from src.rag import fact_signals, method_candidates, practical_checklist

    out = []
    cases = json.load(open(os.path.join("tests", "eval_narrative_cases.json"), encoding="utf-8"))["cases"]
    for case in cases:
        for field in ("expected", "forbidden_cited"):
            for item in case.get(field) or []:
                for number in item.get("any_of") or [item["article"]]:
                    out.append((str(number), EVAL_TO_CODE.get(item["code"], item["code"]), f"eval {field}"))
    for path in sorted(glob.glob(os.path.join(reports, "**", "report.json"), recursive=True)) if reports else []:
        for case in json.load(open(path, encoding="utf-8")):
            for u in case.get("unexpected") or []:
                number, _, code = str(u.get("label") or "").partition(" ")
                if number and code:
                    out.append((number, code, "report unexpected"))
    for _, (_, refs) in list(method_candidates.AREA_ARTICLES.items()) + list(method_candidates.POSTURE_ARTICLES.items()):
        out += [(n, c, "method_candidates") for n, c in refs]
    for table in (practical_checklist.POSTURE_CHECKS, practical_checklist.AREA_CHECKS,
                  practical_checklist.VICTIM_POSTURE_CHECKS):
        for items in table.values():
            out += [(n, c, "practical_checklist") for item in items for n, c in practical_checklist.item_refs(item)]
    out += [(n, c, "procedural_tools") for t in fact_signals.PROCEDURAL_TOOLS for n, c in t["refs"]]
    out += [(n, c, "criminal_by_side") for t in fact_signals.CRIMINAL_BY_SIDE.values() for n, c in t["refs"]]
    out += [(n, c, "damages_death") for n, c in fact_signals.DAMAGES_DEATH["refs"]]
    for regime in fact_signals.REGIMES:
        numbers = [str(k) for lo, hi in regime["ranges"] for k in range(lo, hi + 1)] + (
            [regime["articolo"]] if regime.get("articolo") else [])
        out += [(n, regime["codice"], "regimes") for n in numbers]
    for term in fact_signals.TERMS:
        for m in fact_signals._REF_RE.finditer(term.ref or ""):
            out += [(dg._article_num(n), m.group(2), "terms") for n in re.split(r"\s*(?:,|\be\b)\s*", m.group(1)) if n]
    out.append(("5", "d.lgs. 28/2010", "mediation"))
    out += [(n, c, "civil_damages") for n, c in dg._CIVIL_DAMAGES_NORMS]
    seen, unique = {}, []
    for number, code, source in out:
        key = (number, code)
        if key not in seen:
            seen[key] = source
            unique.append((number, code, source))
    return unique


def resolve(code: str, documents: list, eval_codes: dict):
    """The document of a code outside the six: an eval key by its document pattern, else a law name (resolve_code)."""
    from src.rag.issue_spotting import resolve_code
    if code in eval_codes:
        hits = [d for d in documents if d.get("document_type") == "primary" and re.search(eval_codes[code][0], d["name"], re.I)]
        return min(hits, key=lambda d: len(d["name"])) if hits else None
    return resolve_code(code, documents)


# --- old and new ---------------------------------------------------------------------------------------------

def old_fetch_code(cs, AT, number_variants, prefix, number):
    """(_fetch_articles' rows as before, {spelling: the query's rows}) for one article of the six codes."""
    raws, found = {}, []
    for spelling in number_variants(number):
        raw = cs.run(OLD_FETCH, prefix=prefix, n=spelling).data()
        raws[spelling] = raw
        by_doc = {}
        for row in raw:
            by_doc.setdefault((row.get("d") or {}).get("id"), []).append(row)
        found = [r for doc_rows in by_doc.values() for r in AT.article_rows(spelling, doc_rows)]
        if found:
            break
    return found[:6], raws


def old_sections(cs, doc, spelling):
    """The former _sections' rows for one (document, spelling): whole nodes."""
    return [{"d": r["d"], "s": r["s"], "next": r.get("next")}
            for r in cs.run(OLD_SECTIONS, items=[{"doc": doc, "num": spelling}]).data()]


def old_fetch_article(cs, AT, number_variants, doc, number):
    for spelling in number_variants(number):
        rows = AT.article_rows(spelling, old_sections(cs, doc, spelling))
        if rows:
            return rows
    return []


def shown(AT, row) -> dict:
    """A row as a reader sees it: s and d restricted to the fields read, without the properties the node lacks."""
    return {"d": AT.present({k: (row.get("d") or {}).get(k) for k in AT.DOCUMENT_FIELDS}),
            "s": AT.present({k: (row.get("s") or {}).get(k) for k in AT.SECTION_FIELDS}), "next": row.get("next")}


def _canon(rows) -> list:
    return sorted(json.dumps(r, sort_keys=True, ensure_ascii=False, default=str) for r in rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", default=os.path.join(os.getcwd(), ".env"))
    parser.add_argument("--reports", default="", help="a folder of diagnose_long report.json files (unexpected articles)")
    parser.add_argument("--only", default="", help="only the articles of this code (\"c.p.\", \"consumo\", ...)")
    parser.add_argument("--verbose", action="store_true", help="print every article checked, not only mismatches")
    args = parser.parse_args()

    sys.path.insert(0, os.getcwd())                 # the code of the folder we run from: the new path
    from dotenv import load_dotenv
    load_dotenv(args.env)
    import neo4j
    import src.rag.defensive_generation as dg
    from src.rag import article_text as AT, comma_material
    from src.rag.article_suffix import number_variants
    eval_codes = _load(os.path.join("tests", "eval_narrative.py"), "eval_narrative").CODES
    print(f"code: {os.path.abspath(dg.__file__)}", file=sys.stderr)

    driver = neo4j.GraphDatabase.driver(os.environ["NEO4J_URI"], auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]))
    mismatches, kinds, sources, not_found = [], Counter(), Counter(), []
    checked = 0
    try:
        with driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j"), default_access_mode=neo4j.READ_ACCESS) as raw:
            cs = Counting(raw)
            cs.phase = "documents"
            documents = cs.run(DOCUMENTS).data()
            items = [i for i in sample(dg, args.reports) if not args.only or i[1] == args.only]
            store = AT.ArticleStore()

            def miss(kind, label, detail):
                kinds[kind] += 1
                mismatches.append(f"MISMATCH {kind} {label}: {detail}")

            for number, code, source in items:
                label = f"art. {number} {code}"
                sources[source] += 1
                prefix = dg._CODE_PREFIXES.get(code)
                if prefix:
                    cs.phase = "old fetch (per article and spelling)"
                    old_rows, raws = old_fetch_code(cs, AT, number_variants, prefix, number)
                    cs.phase = "new (store)"
                    new_rows = dg._fetch_articles(cs, [(number, code)], store)
                    docs = store.prefix_docs(cs, [prefix])[prefix]
                    old_docs = list(dict.fromkeys((r.get("d") or {}).get("id") for rows in raws.values() for r in rows))
                    if [d for d in old_docs if d not in docs]:
                        miss("documents", label, f"old query's documents {old_docs} not all among the prefix's {docs}")
                    for spelling, rows in raws.items():       # the query's own cut, emulated on the store's rows
                        grouped = store.sections(cs, [(doc, spelling) for doc in docs])
                        cut = sorted((r for doc in docs for r in grouped.get((doc, spelling), [])),
                                     key=lambda r: str((r.get("s") or {}).get("name") or ""))[:300]
                        if _canon(shown(AT, r) for r in rows) != _canon(shown(AT, r) for r in cut):
                            miss("fetch query rows", f"{label} [{spelling}]",
                                 f"old {len(rows)} rows, new {len(cut)}: names only old "
                                 f"{sorted({r['s'].get('name') for r in rows} - {r['s'].get('name') for r in cut})[:8]}, "
                                 f"only new {sorted({r['s'].get('name') for r in cut} - {r['s'].get('name') for r in rows})[:8]}"
                                 + (" (same names: a projected field differs)" if Counter(r["s"].get("name") for r in rows)
                                    == Counter(r["s"].get("name") for r in cut) else ""))
                        if [(r["d"].get("id"), r["s"].get("name")) for r in rows] != \
                                [(r["d"].get("id"), r["s"].get("name")) for r in cut]:
                            if _canon(shown(AT, r) for r in rows) == _canon(shown(AT, r) for r in cut):
                                miss("fetch query order", f"{label} [{spelling}]", "same rows, another order (documents' ties)")
                else:
                    doc = resolve(code, documents, eval_codes)
                    if not doc:
                        not_found.append(f"{label} (no document)")
                        continue
                    docs = [doc["id"]]
                    cs.phase = "old fetch (per article and spelling)"
                    old_rows = old_fetch_article(cs, AT, number_variants, doc["id"], number)
                    cs.phase = "new (store)"
                    new_rows = AT.fetch_many(cs, [(doc["id"], number)], store)[(doc["id"], number)]
                checked += 1
                # (1) the rows the fetch returns, in order
                old_seen, new_seen = [shown(AT, r) for r in old_rows], [shown(AT, r) for r in new_rows]
                if old_seen != new_seen:
                    miss("fetch rows", label, f"old {[(r['d'].get('id'), r['s'].get('name')) for r in old_seen]} "
                                              f"new {[(r['d'].get('id'), r['s'].get('name')) for r in new_seen]}"
                         + (" (same rows, another order)" if _canon(old_seen) == _canon(new_seen) else
                            " (same documents and Sections: a field differs)"
                            if [(r['d'].get('id'), r['s'].get('name')) for r in old_seen]
                            == [(r['d'].get('id'), r['s'].get('name')) for r in new_seen] else ""))
                if any(set(r.get("s") or {}) - set(AT.SECTION_FIELDS) or set(r.get("d") or {}) - set(AT.DOCUMENT_FIELDS)
                       for r in new_rows):
                    miss("projection", label, "a new row holds a property outside the fields read")
                if not old_rows:
                    not_found.append(f"{label} (no rows, old and new)" if not new_rows else f"{label} (no old rows)")
                # (2)-(4) per document and spelling: the Section rows, fill_texts' text, the commi
                for doc in docs:
                    old_commi = new_commi = None
                    for spelling in number_variants(number):
                        cs.phase = "old _sections (per document and spelling)"
                        old = old_sections(cs, doc, spelling)
                        cs.phase = "new (store)"
                        new = store.sections(cs, [(doc, spelling)]).get((doc, spelling), [])
                        where = f"{label} [{spelling}] in {doc}"
                        if _canon(shown(AT, r) for r in old) != _canon(shown(AT, r) for r in new):
                            names_old = Counter(r["s"].get("name") for r in old)
                            names_new = Counter(r["s"].get("name") for r in new)
                            miss("sections", where, f"fragments only old {sorted((names_old - names_new).elements())[:8]}, "
                                                    f"only new {sorted((names_new - names_old).elements())[:8]}"
                                 + (" (same names: a projected field differs)" if names_old == names_new else ""))
                        if any(set(r["s"]) - set(AT.SECTION_FIELDS) or set(r["d"]) - set(AT.DOCUMENT_FIELDS) for r in new):
                            miss("projection", where, "a store row holds a property outside the fields read")
                        if old or new:
                            texts = [AT.choose_text(spelling, *AT._fields(AT._best_by_name(rows))) for rows in (old, new)]
                            if texts[0] != texts[1]:
                                miss("text", where, f"old {len(texts[0] or '')} chars, new {len(texts[1] or '')} chars")
                            commi = [comma_material.commi_from_sections(spelling, *AT._fields(AT._best_by_name(rows)))
                                     for rows in (old, new)]
                            if commi[0] != commi[1]:
                                miss("commi", where, f"old {len(commi[0][0])} parts ({commi[0][1]}), new {len(commi[1][0])} "
                                                     f"({commi[1][1]})" + (" (as many parts: their text differs)"
                                                                           if len(commi[0][0]) == len(commi[1][0]) else ""))
                            if old_commi is None and commi[0][0]:
                                old_commi = commi[0]
                    # what comma_material reads for the article (_read: the first spelling with commi), through the store
                    cs.phase = "new (store)"
                    new_commi = comma_material._read(cs, [(doc, number)], store).get((doc, number))
                    if (old_commi or None) != (new_commi or None):
                        miss("commi read", f"{label} in {doc}", f"old {old_commi and len(old_commi[0])} parts, "
                                                                f"new {new_commi and len(new_commi[0])} parts")
                if args.verbose:
                    print(f"checked {label} ({source}): {len(old_rows)} rows", file=sys.stderr)
    finally:
        driver.close()

    print("\n".join(mismatches) if mismatches else "No mismatch.")
    print(f"\n## Summary\n\narticles in the sample: {sum(sources.values())} ({', '.join(f'{s} {n}' for s, n in sources.most_common())})")
    print(f"articles compared: {checked}; mismatches: {len(mismatches)}" + (
        " (" + ", ".join(f"{k} {n}" for k, n in kinds.most_common()) + ")" if kinds else ""))
    print(f"not found: {len(not_found)}" + (": " + "; ".join(not_found[:40]) + (" ..." if len(not_found) > 40 else "")
                                             if not_found else ""))
    print(f"store: {store.stats()}")
    print("\n## Traffic (this sample)\n\n| reads | queries | rows | MB as JSON | floats |\n|---|---|---|---|---|")
    for phase, (q, r, b, f) in cs.tally.items():
        print(f"| {phase} | {q} | {r} | {b / 1e6:.2f} | {f} |")
    old = [sum(v) for v in zip(*(t for p, t in cs.tally.items() if p.startswith("old")))] or [0, 0, 0, 0]
    new = cs.tally.get("new (store)", [0, 0, 0, 0])
    print(f"\nold, both reads: {old[0]} queries, {old[2] / 1e6:.2f} MB, {old[3]} floats; "
          f"new: {new[0]} queries, {new[2] / 1e6:.2f} MB, {new[3]} floats. The old pipeline repeated the _sections read "
          "in fill_texts and in the commi; here it is counted once.")
    sys.exit(1 if mismatches else 0)


if __name__ == "__main__":
    main()
