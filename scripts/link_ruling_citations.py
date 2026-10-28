"""Link each ruling Section to the code articles it cites: (Section)-[:CITES]->(Section).

Step 2 of the ruling-cards plan. exp_ruling_links.py showed that nothing links a ruling
to the articles it applies. Section.references holds only the page ("Pg. 3", flatten.py),
so the citations come from each ruling Section's plain_text, read by
exp_ruling_links.cited_articles(): the app's citation patterns ("art. 56 c.p.",
"artt. 110 e 117 cod. pen.", "art. 41, secondo comma, c.p.", "art. 593 c.p.p."),
numbers normalised by article_suffix ("609 bis" -> "609-bis"). A number with no code
after it (often a special law's) is counted, not linked.

Each "N code" is resolved the way the app reads articles (_fetch_articles): a primary
Document whose name starts with the code's prefix ("Codice Penale"), and in it, for
each spelling of N (number_variants), the Section "N.0.0", else "N", else the first
"N.x". The link goes to the article, not to the comma a ruling names. Every code's
Section names are read once at the start, so resolving costs no query.

Dry run by default: how many rulings and citations resolve, the unresolved ones by
reason. --write MERGEs the edges (idempotent), --tx-edges per transaction, --pause
seconds between batches of --batch rulings. The edges carry created_by, so --undo
deletes exactly them. Since 29 Oct the code may also follow a list, an ordinal or a page
footer ("art. 606, comma 1, lett. b) ed e), cod. proc. pen."); the edges only that rule
finds carry rule "past_lists": the summary counts them, --undo --only-past-lists --write
deletes just them.

Run on be2 from the app folder, with exp_ruling_links.py next to this file:
    cd /opt/chatbot && venv/bin/python /tmp/link_ruling_citations.py            # dry run
    ... --write
    ... --undo --write
Only counts are printed, never ruling text. --out gets one JSON line per unresolved
citation (ruling id, section id, citation, reason).
"""

import argparse
import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exp_ruling_links import _PQM_RE, cited_articles, code_of_run  # noqa: E402  (sets up sys.path and .env)

from src.rag.article_suffix import number_variants  # noqa: E402
from src.rag.defensive_generation import _ARTICLE_RUN_RE, _CODE_PREFIXES, _RUN_NUMBER_RE  # noqa: E402

CREATED_BY = "link_ruling_citations"
# Edges only the 29 Oct rule finds (the code past a list, an ordinal or a page footer: exp_ruling_links.code_of_run)
# carry rule: PAST_LISTS, so the dry run counts them and --undo --only-past-lists removes just them.
PAST_LISTS = "past_lists"
CODE_PREFIXES = {**_CODE_PREFIXES, "Cost.": "Costituzione"}


def bare_numbers(text: str) -> int:
    """How many article numbers the text cites with no code after them."""
    text = _PQM_RE.sub(" ; ", text or "")
    return sum(len(_RUN_NUMBER_RE.findall(m.group(1))) for m in _ARTICLE_RUN_RE.finditer(text)
               if not code_of_run([], text[m.end():m.end() + 120])[1])


class Codes:
    """Every code's Documents on be2 and their Section names, read once."""

    def __init__(self, session):
        self.docs = {}       # code -> [(doc id, doc name, {section name: section id})], most sections first
        for code, prefix in CODE_PREFIXES.items():
            found = []
            for r in session.run("MATCH (d:Document) WHERE d.document_type = 'primary' AND d.name STARTS WITH $p "
                                 "RETURN d.id AS id, d.name AS name", p=prefix):
                names = {}
                # Of two Sections with one name, the longer text: the app's tie-break
                # (_dedupe_sections) prefers the fuller one too.
                for s in session.run("MATCH (d:Document {id: $id})-[:CONTAINS]->(s:Section) "
                                     "RETURN s.id AS id, s.name AS name, size(coalesce(s.plain_text, '')) AS len "
                                     "ORDER BY len", id=r["id"]):
                    if s["name"]:
                        names[s["name"]] = s["id"]
                found.append((r["id"], r["name"], names))
            self.docs[code] = sorted(found, key=lambda d: -len(d[2]))

    def resolve(self, number: str, code: str):
        """(section id, how) or (None, reason)."""
        docs = self.docs.get(code)
        if not docs:
            return None, "code_not_on_be2"
        for _doc_id, _name, names in docs:
            for spelling in number_variants(number):
                if f"{spelling}.0.0" in names:
                    return names[f"{spelling}.0.0"], "article"
                if spelling in names:
                    return names[spelling], "article"
                parts = sorted(n for n in names if n.startswith(spelling + "."))
                if parts:
                    return names[parts[0]], "first_fragment"
        return None, "article_not_in_code"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--write", action="store_true", help="create the edges (default: dry run)")
    p.add_argument("--undo", action="store_true", help=f"delete the CITES edges created_by '{CREATED_BY}' (with --write)")
    p.add_argument("--only-past-lists", action="store_true",
                   help=f"with --undo: only the edges with rule '{PAST_LISTS}' (the 29 Oct rule's)")
    p.add_argument("--batch", type=int, default=50, help="rulings per read")
    p.add_argument("--tx-edges", type=int, default=500)
    p.add_argument("--pause", type=float, default=1.0, help="seconds between batches")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--top", type=int, default=25)
    p.add_argument("--out", default="/tmp/ruling_citations_unresolved.jsonl")
    args = p.parse_args()

    import neo4j
    from src.rag.main import NEO4J_DATABASE, driver

    mode = neo4j.WRITE_ACCESS if args.write else neo4j.READ_ACCESS
    with driver.session(database=NEO4J_DATABASE, default_access_mode=mode) as session:
        before = session.run("MATCH ()-[r:CITES]->() RETURN r.created_by AS by, count(*) AS c").data()
        print("CITES edges now: " + (", ".join(f"{r['by'] or '(no created_by)'} {r['c']:,}" for r in before) or "none"))

        if args.undo:
            if not args.write:
                sys.exit("--undo deletes edges: add --write")
            total = 0
            match = ("MATCH ()-[r:CITES {created_by: $by, rule: $rule}]->()" if args.only_past_lists
                     else "MATCH ()-[r:CITES {created_by: $by}]->()")
            while True:
                n = session.run(match + " WITH r LIMIT $n DELETE r RETURN count(*) AS c",
                                by=CREATED_BY, rule=PAST_LISTS, n=args.tx_edges).single()["c"]
                total += n
                if not n:
                    break
                time.sleep(0.2)
            print(f"Deleted {total:,} CITES edges created by {CREATED_BY}"
                  + (f" with rule {PAST_LISTS}." if args.only_past_lists else "."))
            return

        codes = Codes(session)
        print("Code documents on be2 (primary, name starts with the prefix):")
        for code, docs in codes.docs.items():
            listing = "; ".join(f"{name} [{doc_id}] {len(names):,} sections" for doc_id, name, names in docs)
            print(f"  {code:7} {listing or 'NONE'}")

        query = "MATCH (d:Document {document_type: 'interpretation'}) RETURN d.id AS id ORDER BY id"
        ids = [r["id"] for r in session.run(query + (" LIMIT $n" if args.limit else ""), n=args.limit)]
        print(f"\nReading {len(ids):,} rulings, {args.batch} per query"
              + ("" if args.write else " (dry run: nothing is written)"), file=sys.stderr)

        sections = no_text = bare = written = 0
        rulings_citing, rulings_linked = set(), set()
        how, reasons, per_code, per_code_ok = Counter(), Counter(), Counter(), Counter()
        top_ok, top_missing, past_lists = Counter(), Counter(), Counter()
        t0 = time.time()
        with open(args.out, "w", encoding="utf-8") as out:
            for start in range(0, len(ids), args.batch):
                if start:
                    time.sleep(args.pause)
                rows = session.run("MATCH (d:Document)-[:CONTAINS]->(s:Section) WHERE d.id IN $ids "
                                   "RETURN d.id AS doc, s.id AS id, s.plain_text AS text",
                                   ids=ids[start:start + args.batch]).data()
                edges = []
                for row in rows:
                    sections += 1
                    text = row["text"] or ""
                    if not text.strip():
                        no_text += 1
                        continue
                    bare += bare_numbers(text)
                    before_lists = set(cited_articles(text, past_lists=False))
                    for citation in cited_articles(text):
                        number, code = citation.split(" ", 1)
                        rulings_citing.add(row["doc"])
                        per_code[code] += 1
                        target, result = codes.resolve(number, code)
                        if target is None:
                            reasons[result] += 1
                            top_missing[f"{citation}  ({result})"] += 1
                            out.write(json.dumps({"ruling": row["doc"], "section": row["id"], "citation": citation,
                                                  "reason": result}, ensure_ascii=False) + "\n")
                            continue
                        how[result] += 1
                        per_code_ok[code] += 1
                        top_ok[citation] += 1
                        rulings_linked.add(row["doc"])
                        # rule None removes the property (SET r += ...): an edge the earlier rule also finds is untagged
                        props = {"article": number, "code": code, "how": result, "created_by": CREATED_BY, "rule": None}
                        if citation not in before_lists:
                            props["rule"] = PAST_LISTS
                            past_lists[code] += 1
                        edges.append({"a": row["id"], "b": target, "props": props})
                if args.write:
                    for i in range(0, len(edges), args.tx_edges):
                        written += (getattr(session, "execute_write", None) or session.write_transaction)(lambda tx, part: tx.run(
                            "UNWIND $rows AS row MATCH (a:Section {id: row.a}) MATCH (b:Section {id: row.b}) "
                            "MERGE (a)-[r:CITES]->(b) SET r += row.props RETURN count(r) AS c",
                            rows=part).single()["c"], edges[i:i + args.tx_edges])
                done = min(start + args.batch, len(ids))
                print(f"  {done:,}/{len(ids):,} rulings, {time.time() - t0:.0f}s", file=sys.stderr)

    cited, resolved = sum(per_code.values()), sum(how.values())
    pct = lambda n, d: f"{100 * n / d:.0f}%" if d else "-"  # noqa: E731
    print(f"\n== Summary ==" + ("" if args.write else " (dry run: nothing written)"))
    print(f"Rulings {len(ids):,}; sections {sections:,} ({no_text:,} without plain_text)")
    print(f"Rulings citing a code article: {len(rulings_citing):,} ({pct(len(rulings_citing), len(ids))}); "
          f"with at least one resolved: {len(rulings_linked):,} ({pct(len(rulings_linked), len(ids))})")
    print(f"Citations (section, article): {cited:,}; resolved {resolved:,} ({pct(resolved, cited)})"
          + "".join(f", {k} {v:,}" for k, v in how.most_common()))
    print(f"Of the resolved, found only past a list, an ordinal or a page footer (rule {PAST_LISTS}, 29 Oct): "
          f"{sum(past_lists.values()):,}" + "".join(f", {c} {n:,}" for c, n in past_lists.most_common()))
    print("Unresolved by reason: " + (", ".join(f"{k} {v:,}" for k, v in reasons.most_common()) or "none"))
    print(f"Article numbers with no code after them (special laws, or the code named earlier): {bare:,} -- not linked")
    print("By code: " + ", ".join(f"{c} {per_code_ok[c]:,}/{n:,}" for c, n in per_code.most_common()))
    print(f"Most cited, resolved (top {args.top}): " + ", ".join(f"{c} {n:,}" for c, n in top_ok.most_common(args.top)))
    print(f"Most cited, unresolved (top {args.top}):")
    for c, n in top_missing.most_common(args.top):
        print(f"  {n:>7,}  {c}")
    if args.write:
        print(f"CITES edges written (MERGE): {written:,}")
    print(f"Each unresolved citation: {args.out}")


if __name__ == "__main__":
    main()
