"""Trace one Cassazione ruling through the pipeline, READ-ONLY: what the graph holds of it, the article links the
linker made (each with the rule that made it), step 1's link record, its ruling card (which model and prompt
version wrote it), and the commands that find its batch and the answers it was offered to. For "a ruling in an
answer looks wrong: where did it come from?" (the PM's tracing request, 4 Nov).

Runs on prod, where the card and link files are, against the app's own graph (.env):
    cd /opt/chatbot && venv/bin/python scripts/trace_ruling.py --number 14172/2026
    cd /opt/chatbot && venv/bin/python scripts/trace_ruling.py --id LEGAL_DOC::23b985db...
    cd /opt/chatbot && venv/bin/python scripts/trace_ruling.py --file snpen@s10@a2025@n00001@tS

Prints fields, counts and labels only: never a section's text, never a card's quote. Several rulings matching a
number (the same number in two years, or civil and criminal) are listed, and the trace needs the --id.
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

DEFAULT_CARDS = "/root/ruling_cards/cards_pen_v3.jsonl,/root/ruling_cards/cards_civ_v1.jsonl"
DEFAULT_LINKS = "/root/ruling_cards/ruling_links.jsonl"
QUOTE_FIELDS = ("citazione", "citazione_scartata", "citazione_ricostruita", "testo", "text")


# --- pure helpers (tested) -----------------------------------------------------------------------

def parse_number(value: str) -> Tuple[str, Optional[str]]:
    """'14172/2026' -> ('14172', '2026'); 'n. 14172 del 2026' -> the same; '14172' -> ('14172', None)."""
    m = re.search(r"(\d{1,6})\s*(?:/|del|-)\s*((?:19|20)\d{2})", value)
    if m:
        return str(int(m.group(1))), m.group(2)
    m = re.search(r"\d{1,6}", value)
    return (str(int(m.group(0))) if m else ""), None


def number_matches(stored, number: str) -> bool:
    """Whether a document_number ('14172', '014172', '14172/2026', 'n. 14172') is this number, not one containing it."""
    return bool(re.search(rf"(?<!\d)0*{re.escape(number)}(?!\d)", str(stored or "")))


def year_of(doc: dict) -> Optional[str]:
    m = re.search(r"[@_ ]a((?:19|20)\d{2})[@_ ]n", str(doc.get("file_name") or "") + " " + str(doc.get("name") or ""))
    if m:
        return m.group(1)
    m = re.search(r"((?:19|20)\d{2})", str(doc.get("date_enacted") or ""))
    return m.group(1) if m else None


PARTY_MARKERS = re.compile(r"\s+(?:sul ricorso |)(?:propost[oa] d[aiel]|ricorrente|nei confronti|contro|a carico d[i'])\b.*$",
                           re.IGNORECASE)


def safe_name(name: Optional[str]) -> str:
    """A document name without the parties: the ingest names a ruling 'Sentenza ... sul ricorso proposto da <party>',
    and no name is ever printed by this tool."""
    return PARTY_MARKERS.sub(" [...]", str(name or "").strip())[:80]


def branch_of(doc: dict) -> str:
    blob = (str(doc.get("file_name") or "") + " " + str(doc.get("name") or "")).lower()
    return "penale" if "snpen" in blob or "[penale]" in blob else "civile" if "snciv" in blob or "[civile]" in blob else "?"


def stem_of(file_value: str) -> str:
    """The file name without folder, .pdf and .clean, separators unified to '@' for matching."""
    stem = os.path.basename(file_value.strip())
    stem = re.sub(r"\.pdf$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"\.clean$", "", stem, flags=re.IGNORECASE)
    return stem.replace("_", "@").replace(" ", "@")


def find_records(path: str, doc_id: str) -> List[dict]:
    """The JSON lines whose "id" is the document's, from a card or link file (read line by line, no text kept)."""
    out = []
    if not os.path.isfile(path):
        return out
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if doc_id not in line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if isinstance(rec, dict) and rec.get("id") == doc_id:
                out.append(rec)
    return out


def card_summary(card: dict) -> Dict[str, object]:
    """The card's fields worth showing, quotes replaced by their length and whether one was verified."""
    out: Dict[str, object] = {}
    for key in ("prompt_version", "modello", "branch", "esito", "a_favore_di", "parte_favorita", "ricorrente",
                "controricorrente", "completo", "letta_in_parte", "errore", "articoli_scartati"):
        if key in card:
            out[key] = card[key]
    for key in ("articoli_applicati", "argomento_decisivo", "parole_chiave_fatti", "parole_chiave_questioni"):
        if card.get(key):
            out[key] = card[key]
    quote = card.get("citazione") or ""
    out["citazione"] = f"{len(quote)} chars, verified {bool(card.get('citazione_verificata', bool(quote)))}"
    assert not any(k in out and k != "citazione" for k in QUOTE_FIELDS)
    return out


def cites_lines(rows: Iterable[dict]) -> List[str]:
    """One line per cited article: 'art. 612-bis c.p. (Codice Penale 2026): 3 links, how explicit 3, rule past_lists 1'."""
    by_article: Dict[Tuple[str, str, str], List[dict]] = {}
    for r in rows:
        by_article.setdefault((str(r.get("article")), str(r.get("code")), str(r.get("law") or "?")), []).append(r)
    lines = []
    for (article, code, law), edges in sorted(by_article.items(), key=lambda x: -sum(e.get("n", 1) for e in x[1])):
        n = sum(e.get("n", 1) for e in edges)
        hows = Counter()
        rules = Counter()
        for e in edges:
            hows[str(e.get("how"))] += e.get("n", 1)
            rules[str(e.get("rule"))] += e.get("n", 1)
        lines.append(f"art. {article} {code} ({law}): {n} link{'s' if n != 1 else ''}, how "
                     + ", ".join(f"{h} {k}" for h, k in hows.most_common()) + "; rule "
                     + ", ".join(f"{r} {k}" for r, k in rules.most_common()))
    return lines


# --- graph ---------------------------------------------------------------------------------------

DOC_FIELDS = ("RETURN d.id AS id, d.name AS name, d.file_name AS file_name, d.document_number AS document_number, "
              "d.date_enacted AS date_enacted, d.document_type AS document_type, d.status AS status, "
              "left(coalesce(d.hash, ''), 12) AS hash, keys(d) AS keys, "
              "size([(d)-[:CONTAINS]->(s:Section) | s]) AS sections")


def find_documents(session, args) -> List[dict]:
    if args.id:
        return session.run(f"MATCH (d:Document {{id: $id}}) {DOC_FIELDS}", id=args.id).data()
    if args.file:
        stem = stem_of(args.file)
        rows = session.run("MATCH (d:Document {document_type: 'interpretation'}) "
                           "WHERE d.file_name CONTAINS $a OR d.file_name CONTAINS $b OR d.name CONTAINS $c "
                           f"{DOC_FIELDS} LIMIT 10", a=stem, b=stem.replace("@", "_"), c=stem.replace("@", " ")).data()
        return rows
    number, year = parse_number(args.number)
    rows = session.run("MATCH (d:Document {document_type: 'interpretation'}) "
                       "WHERE toString(d.document_number) CONTAINS $n "
                       f"{DOC_FIELDS} LIMIT 200", n=number).data()
    rows = [r for r in rows if number_matches(r.get("document_number"), number)]
    if year:
        rows = [r for r in rows if year_of(r) in (year, None)]
    if args.branch:
        rows = [r for r in rows if branch_of(r) == args.branch]
    return rows[:10]


def cites_of(session, doc_id: str) -> List[dict]:
    return session.run(
        "MATCH (d:Document {id: $id})-[:CONTAINS]->(:Section)-[c:CITES]->(t:Section) "
        "OPTIONAL MATCH (law:Document)-[:CONTAINS]->(t) "
        "RETURN c.code AS code, c.article AS article, c.how AS how, c.rule AS rule, c.created_by AS created_by, "
        "law.name AS law, count(*) AS n ORDER BY n DESC", id=doc_id).data()


def main() -> int:
    p = argparse.ArgumentParser(description="Trace one Cassazione ruling through the pipeline (read-only)")
    which = p.add_mutually_exclusive_group(required=True)
    which.add_argument("--number", help="the ruling's number and year, e.g. 14172/2026")
    which.add_argument("--id", help="the document id (LEGAL_DOC::...)")
    which.add_argument("--file", help="the Italgiure file name, e.g. snpen@s10@a2025@n00001@tS")
    p.add_argument("--branch", choices=("penale", "civile"), help="with --number: only this branch")
    p.add_argument("--env", default=os.path.join(os.getcwd(), ".env"))
    p.add_argument("--cards", default=DEFAULT_CARDS, help="comma-separated card files")
    p.add_argument("--links", default=DEFAULT_LINKS, help="step 1's link file")
    args = p.parse_args()

    from dotenv import load_dotenv
    load_dotenv(args.env)
    from neo4j import GraphDatabase, READ_ACCESS
    driver = GraphDatabase.driver(os.environ["NEO4J_URI"], auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]))
    try:
        with driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j"), default_access_mode=READ_ACCESS) as session:
            docs = find_documents(session, args)
            if not docs:
                print("no ruling found")
                return 1
            if len(docs) > 1:
                print(f"{len(docs)} rulings match; rerun with --id (or --branch penale/civile):")
                for d in docs:
                    print(f"  {d['id']}  {branch_of(d)}  n. {d.get('document_number')}  {d.get('date_enacted')}  "
                          f"{safe_name(d.get('name'))}")
                return 1
            d = docs[0]
            cites = cites_of(session, d["id"])
    finally:
        driver.close()

    print("== document")
    for key in ("id", "file_name", "document_number", "date_enacted", "document_type", "status", "hash", "sections"):
        print(f"  {key}: {d.get(key)}")
    print(f"  name: {safe_name(d.get('name'))}   branch: {branch_of(d)}")
    print(f"  properties: {', '.join(sorted(d.get('keys') or []))}")
    print(f"\n== article links (CITES): {sum(r['n'] for r in cites)} edges, {len(cites_lines(cites))} articles")
    for line in cites_lines(cites)[:40]:
        print("  " + line)
    if cites:
        print("  made by: " + ", ".join(f"{k} {v}" for k, v in Counter(str(r.get('created_by')) for r in cites).items()))

    links = find_records(args.links, d["id"])
    print(f"\n== step 1 link record ({args.links}): {'found' if links else 'none'}")
    for rec in links[:1]:
        for key in ("branch", "court", "number", "date", "outcome", "appellant", "pqm_anchored", "chars"):
            if key in rec:
                print(f"  {key}: {rec[key]}")
        print(f"  articles: {len(rec.get('articles') or [])}: {', '.join(map(str, (rec.get('articles') or [])[:20]))}")

    for path in [x for x in args.cards.split(",") if x]:
        cards = find_records(path, d["id"])
        print(f"\n== card ({path}): {'found' if cards else 'none'}")
        for card in cards[:1]:
            for key, value in card_summary(card).items():
                print(f"  {key}: {value}")

    stem = stem_of(d.get("file_name") or "")
    print("\n== where next")
    print(f"  batch (on be2):   grep -F '{stem}' /var/log/maint/upload_batches.txt")
    print(f"  answers (prod):   journalctl -u chatbot | grep -c '{d['id']}'      # offered/cited events, verbose logging on")
    return 0


if __name__ == "__main__":
    sys.exit(main())
