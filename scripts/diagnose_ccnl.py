"""Read-only diagnosis of the CCNL eval's misses (tests/eval_ccnl_cases.json), with the code of the current folder:

  1 key        the sectors of commercio / terziario / turismo / pubblici esercizi the database has (documents
               per sector: kind, validity, name), and for the key cases (--key-cases, default c1,c2,t1,t2) the
               candidates ccnl_path.identify_contract scores, its choice and ambiguity against the case's key,
               and every sector the key's regex accepts
  2 clauses    for the clause misses (--clause-cases, default c1,c2,m2,s2,m4,e1,f2): the contract and version used,
               the topic's keywords and heading phrases, the top sections as ranked now (heading score, lexical
               score, cosine, name, 120 characters), the clauses selected now, and whether the clause exists in
               the chosen documents: the Sections whose text holds each phrase of --phrases (per topic), with
               their rank, and the lines opening with "Art. N" that hold the phrase with what article_headings
               reads there (a heading the parser misses shows "NOT PARSED")
  3 framework  for the framework misses (--framework-cases, default m1,m4,e1): each wanted article, the
               primary documents its law resolves to (by number and year, then by title) and whether the
               article was fetched

Run from the app folder:
    cd /opt/chatbot && venv/bin/python scripts/diagnose_ccnl.py
    cd /opt/chatbot && venv/bin/python scripts/diagnose_ccnl.py --clause-cases m2 --top 20 --no-embed

Every session is read-only; nothing is written to the database. Writes report.txt under --out.
"""

import argparse
import json
import os
import re
import sys
from datetime import date, datetime
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (ROOT, os.getcwd()):
    if _path not in sys.path:
        sys.path.insert(0, _path)

KEY_SECTORS_RE = r"terziar|commerc|turism|pubblici esercizi|alberg|distribuzione|ristoraz|confcommercio|confesercenti|fipe"
# The phrases a clause on the topic carries, searched in the chosen documents' texts.
PHRASES = {"prova": ["periodo di prova", "patto di prova", "assunzione in prova"],
           "disciplinare": ["provvedimenti disciplinari", "sanzioni disciplinari", "codice disciplinare",
                            "norme disciplinari"],
           "preavviso": ["preavviso", "termini di preavviso", "dimissioni"],
           "malattia": ["malattia", "comporto", "conservazione del posto"], "ferie": ["ferie"],
           "retribuzione": ["minimi tabellari", "retribuzione tabellare"]}


def _session():
    from scripts.audit_ccnl import _read_only_driver
    driver, database = _read_only_driver()
    return driver, driver.session(database=database)


def key_report(session, cases, ids, cp):
    sectors = cp.sector_index(session)
    L = ["## 1 Key: commercio / terziario / turismo / pubblici esercizi", ""]
    matching = sorted(s for s in sectors if re.search(KEY_SECTORS_RE, cp._fold(s)))
    L.append(f"{len(matching)} sectors match /{KEY_SECTORS_RE}/:")
    for sector in matching:
        docs = [cp._doc_view(d) for d in cp.sector_documents(session, sector)]
        L.append(f"- {sector}  ({len(docs)} docs; folded '{cp._fold(sector)}')")
        for d in sorted(docs, key=lambda d: d["from"] or date.min):
            L.append(f"    {str(d['from']):10} .. {str(d['until']):10} {d['kind']:18} {cp.short_name(d)[:80]}")
    for case in [c for c in cases if c["id"] in ids]:
        got = cp.identify_contract(case["question"], sectors, llm=False)
        accepted = [s for s in sectors if re.search(case["sector"], cp._fold(s))]
        right = bool(got["sector"] and re.search(case["sector"], cp._fold(got["sector"])))
        L += ["", f"### {case['id']}: {case['question']}",
              f"key: sector /{case['sector']}/, ambiguous {case.get('ambiguous')}",
              f"chosen: {got['sector']} ({got['how']}); ambiguous {got['ambiguous']}; alternatives "
              f"{got['alternatives']} -> sector {'right' if right else 'WRONG'}, ambiguity "
              f"{'right' if got['ambiguous'] == bool(case.get('ambiguous')) else 'WRONG'}",
              f"sectors the key's regex accepts ({len(accepted)}): {accepted}", "candidates:"]
        L += [f"    {c['score']:>3} {c['sector'][:60]:60} docs={c['docs']:<3} {'; '.join(c['why'])}"
              for c in got["candidates"]]
    return L


def _prepare(session, case, cp, embed, today):
    sectors = cp.sector_index(session)
    contract = cp.identify_contract(case["question"], sectors, llm=False)
    if not contract["sector"]:
        return None
    when, how = cp.relevant_date(case["question"], today)
    version = cp.choose_version(cp.sector_documents(session, contract["sector"]), when)
    chosen = [version["base"]] + version["amendments"]
    topics = cp.topics_of(case["question"])
    keywords = cp.topic_keywords(case["question"], topics)
    embedding = None
    if embed:
        from src.rag.lookups import _cached_embed_query
        embedding = list(_cached_embed_query(f"{case['question']}\n{' '.join(keywords)}"))
    docs = cp.sector_documents(session, contract["sector"])
    rows = cp.document_sections(session, [d["id"] for d in chosen], embedding)
    for r in rows:
        r["text"] = cp.clean(r.get("text"))
    # Every full text of the sector and the chosen documents: sections, with text, characters.
    stats = []
    for d in {x["id"]: x for x in chosen + [cp._doc_view(x) for x in docs if cp.doc_kind(x) in cp.BASE_KINDS]}.values():
        doc_rows = rows if d["id"] in {c["id"] for c in chosen} else [
            {**r, "text": cp.clean(r.get("text"))} for r in cp.document_sections(session, [d["id"]], None)]
        mine = [r for r in doc_rows if r["doc"] == d["id"]]
        stats.append(f"    {cp.short_name(d)[:80]:80} {len(mine):>4} sections, "
                     f"{sum(1 for r in mine if (r.get('text') or '').strip()):>4} with text, "
                     f"{cp.text_chars(mine, d['id']):>8} chars{'  (chosen)' if d in chosen else ''}")
        if d["kind"] in cp.BASE_KINDS:                 # Sections hanging deeper than CONTAINS (not searched)
            deep = session.run("MATCH (d:Document {id: $id})-[*1..4]->(s:Section) RETURN count(DISTINCT s) AS n, "
                               "sum(size(coalesce(s.plain_text, ''))) AS chars", id=d["id"]).single()
            stats.append(f"        within 4 hops of any relation: {deep['n']} sections, {deep['chars']} chars")
    chosen, rows, text_from, base_chars = cp.read_base_text(session, docs, version, chosen, rows, when, embedding)
    return {"contract": contract, "when": when, "how": how, "version": version, "chosen": chosen,
            "stats": stats, "text_from": text_from, "base_chars": base_chars,
            "topics": topics, "keywords": keywords, "headings": cp.topic_headings(topics), "rows": rows}


def clause_report(session, cases, ids, cp, embed, top, today):
    L = ["", "## 2 Clause misses"]
    for case in [c for c in cases if c["id"] in ids]:
        p = _prepare(session, case, cp, embed, today)
        L += ["", f"### {case['id']} ({case.get('topic')}): {case['question']}"]
        if p is None:
            L.append("no contract identified")
            continue
        v = p["version"]
        L += [f"contract {p['contract']['sector']}; date {p['when']} ({p['how']}); base {cp.short_name(v['base'])}; "
              f"amendments {[cp.short_name(d) for d in v['amendments']]}",
              f"topics {p['topics']}; keywords {p['keywords']}; headings {p['headings']}",
              f"query for the cosine: {case['question']} + the keywords; {len(p['rows'])} sections in the chosen "
              f"documents"]
        L.append("documents (sections, with text, characters):")
        L += p["stats"]
        L.append(f"base text {p['base_chars']} chars; clauses read from "
                 f"{cp.short_name(p['text_from']) if p['text_from'] else 'the base'}")
        by_doc, order_used = cp.group_ordered(p["rows"])
        L.append(f"order used: {order_used}")
        ranked = cp.rank_sections(p["rows"], p["keywords"], p["headings"], case["question"])
        names = {d["id"]: cp.short_name(d).split(" — ")[-1][:30] for d in p["chosen"]}
        rank_of = {id(r): i for i, r in enumerate(ranked)}
        L.append(f"top {top} sections as ranked now (head, lex, cos, document, name, text):")
        for r in ranked[:top]:
            sim = f"{r['sim']:.3f}" if r.get("sim") is not None else "  -  "
            L.append(f"    h{r['_head']} lex{r['_lex']:>5.1f} cos{sim} [{names.get(r['doc'], r['doc'])[:30]}] "
                     f"{str(r.get('name'))[:40]!r}: {' '.join((r.get('text') or '').split())[:120]}")
        clauses, _ = cp.select_clauses(p["rows"], p["chosen"], p["keywords"], headings=p["headings"],
                                       question=case["question"])
        L.append("clauses selected now:")
        L += [f"    [{names.get(c['doc']['id'], '')}] {cp.clause_reference(c)} (sections {c['span']}): "
              f"{' '.join(c['text'].split())[:160]}" for c in clauses] or ["    none"]
        L.append("does the clause exist in the chosen documents? (sections whose text holds the phrase)")
        for phrase in PHRASES.get(case.get("topic"), []):
            hits = [r for r in p["rows"] if re.search(rf"\b{re.escape(cp._fold(phrase))}\b", cp._fold(r.get("text") or ""))]
            L.append(f"    '{phrase}': {len(hits)} sections")
            for r in hits[:4]:
                pos = rank_of.get(id(r))
                text = r.get("text") or ""
                at = cp._fold(text).find(cp._fold(phrase))
                L.append(f"        rank {pos if pos is not None else 'not ranked'}; [{names.get(r['doc'], '')}] "
                         f"{str(r.get('name'))[:40]!r} h{cp.heading_score(r, p['headings'], case['question'])}: "
                         f"{' '.join(text[max(0, at - 30):at + 90].split())}")
            heading_lines = [(r, m) for r in p["rows"] for m in re.finditer(
                rf"(?im)^[ \t>*#|]*art(?:icolo|\.)?\s*\d+[^\n]{{0,80}}", r.get("text") or "")
                if cp._fold(phrase) in cp._fold(m.group(0))]
            for r, m in heading_lines[:6]:
                parsed = [h for h in cp.article_headings(r.get("text") or "") if abs(h[0] - m.start()) <= 8]
                pos = rank_of.get(id(r))
                L.append(f"        heading line, rank {pos if pos is not None else 'not ranked'}: "
                         f"{' '.join(m.group(0).split())[:90]!r} -> "
                         f"{(parsed[0][1] + ' - ' + parsed[0][2]) if parsed else 'NOT PARSED'}")
    return L


def framework_report(session, cases, ids, cp, today):
    from src.rag.article_codes import law_label
    L = ["", "## 3 Framework misses"]
    for case in [c for c in cases if c["id"] in ids]:
        topics = cp.topics_of(case["question"])
        refs = cp.framework_refs(topics)
        rows, missing = cp.fetch_framework(session, refs)
        expected = [f"{f['article']} {f['law']}" for f in case.get("framework") or []]
        wanted = [f"{n} {label}" for n, label in refs]
        fetched = [" ".join(r["_article"]) for r in rows]
        L += ["", f"### {case['id']}: {case['question']}", f"topics {topics}; expected {expected}",
              f"wanted {wanted}; fetched {fetched}; missing {missing}",
              f"expected but not in the topic's list: {[e for e in expected if tuple(e.split(' ', 1)) not in refs] or 'none'}"]
        for label in sorted({l for _, l in refs if l not in ("c.c.", "Cost.")}):
            parts = cp._law_parts(label)
            by_number = session.run("MATCH (d:Document) WHERE d.document_type = 'primary' AND d.name CONTAINS $n "
                                    "AND d.name CONTAINS $y RETURN d.name AS name LIMIT 20",
                                    n=parts[0], y=parts[1]).data() if parts else []
            L.append(f"    {label}: by number and year {[(r['name'][:70], law_label(r['name'])) for r in by_number]}")
            if cp.LAW_ALIASES.get(label):
                by_title = session.run("MATCH (d:Document) WHERE d.document_type = 'primary' AND "
                                       "any(a IN $a WHERE toLower(d.name) CONTAINS a) RETURN d.name AS name LIMIT 10",
                                       a=cp.LAW_ALIASES[label]).data()
                L.append(f"        by title {cp.LAW_ALIASES[label]}: {[r['name'][:70] for r in by_title]}")
    L.append("\n(Whether the answer then cited a fetched article is in the eval's --debug: 'CITED IN THE ANSWER'.)")
    return L


def main(argv=None):
    parser = argparse.ArgumentParser(description="Read-only diagnosis of the CCNL eval's misses.")
    parser.add_argument("--cases", default=os.path.join(ROOT, "tests", "eval_ccnl_cases.json"))
    parser.add_argument("--key-cases", default="c1,c2,t1,t2")
    parser.add_argument("--clause-cases", default="c1,c2,m2,s2,m4,e1,f2")
    parser.add_argument("--framework-cases", default="m1,m4,e1")
    parser.add_argument("--top", type=int, default=10, help="sections shown per clause case (default 10)")
    parser.add_argument("--no-embed", action="store_true", help="lexical and heading scores only, no embedding call")
    parser.add_argument("--out", default="ccnl_diagnosis")
    args = parser.parse_args(argv)
    from src.rag import ccnl_path as cp
    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))["cases"]
    split = lambda s: {x.strip() for x in (s or "").split(",") if x.strip()}  # noqa: E731
    today = date.today()
    driver, session = _session()
    try:
        lines = [f"# CCNL diagnosis ({datetime.now():%Y-%m-%d %H:%M})"]
        lines += key_report(session, cases, split(args.key_cases), cp)
        lines += clause_report(session, cases, split(args.clause_cases), cp, not args.no_embed, args.top, today)
        lines += framework_report(session, cases, split(args.framework_cases), cp, today)
    finally:
        session.close()
        driver.close()
    folder = os.path.join(args.out, datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(folder, exist_ok=True)
    Path(folder, "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWritten to {folder}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
