#!/usr/bin/env python3
"""v60h (12): one report for ingestion -- docs/ingestion_issues.md (and .json) -- of what the store holds wrong, each
node with its id, re-runnable (read only on the graph):

  1. misaligned nodes: an article whose rubrica belongs to another article's text (article_text.shifted, confirmed by
     an adjacent article -- the check the flow applies at load, scripts/misaligned_articles.py);
  2. rubriche carrying footnote markers: a rubrica the answer step's cleaning changes (rule_quote.clean_rubrica: «[1025]»,
     «(1)», a number glued to the last word), raw and cleaned;
  3. abrogated articles still active: an article node whose own text is its abrogation (article_text.is_abrogated:
     «(abrogato)», «Articolo abrogato da …»), served as a live article -- no property on the node says otherwise (the
     node's status-like properties, when any, are listed);
  4. non-normative document names: a Document of the types the answers cite from (primary, special, ccnl:
     doc_lookup._fetch_doc_names) whose name names no normative act (NORMATIVE_RE) -- a ruling (RULING_RE) or anything
     else -- with its id and type.

  python scripts/ingestion_issues.py                         # every primary and special document; docs/ingestion_issues.md
  python scripts/ingestion_issues.py --codes c.c.,c.p.       # the article checks on some codes only
  python scripts/ingestion_issues.py --all-types             # check 4 over every document type
"""

import argparse
import datetime as dt
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DOCS_QUERY = ("MATCH (d:Document) WHERE d.document_type IN ['primary', 'special'] "
              "RETURN d.id AS id, d.name AS name ORDER BY d.name")
SECTIONS_QUERY = ("MATCH (d:Document {id: $doc})-[:CONTAINS]->(s:Section) WHERE s.titolo_articolo IS NOT NULL "
                  "RETURN coalesce(s.id, elementId(s)) AS id, s.name AS name, s.titolo_articolo AS rubrica, "
                  "s.plain_text AS text, [k IN keys(s) WHERE k =~ '(?i).*(abrog|vigen|status|stato|force|attiv).*'] "
                  "AS status_keys")
NAMES_QUERY = ("MATCH (d:Document) WHERE $all OR d.document_type IN ['primary', 'special', 'ccnl'] "
               "RETURN d.id AS id, d.name AS name, d.document_type AS type ORDER BY d.name")
from src.rag.citations import NORMATIVE_NAME_RE as NORMATIVE_RE  # noqa: E402  (shared with article_links --unitless)
RULING_RE = re.compile(r"\b(?:sentenz\w*|ordinanz\w*|cass(?:azione|\.)|corte|tribunale|tar\b|consiglio di stato|"
                       r"pronunci\w*|massim\w*|decision\w*|n\.\s?\d+/\d{4})", re.IGNORECASE)


def _canonical(rows, article_number):
    articles = {}
    for row in rows:
        number = article_number(row["name"] or "")
        if number and row["name"] in (number, f"{number}.0.0") and number not in articles:
            articles[number] = row
    return articles


def _md_table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        out.append("| " + " | ".join(str(x).replace("|", "/").replace("\n", " ")[:160] for x in r) + " |")
    return out


def main(argv=None) -> None:
    import neo4j
    from src.rag import article_text, citations
    from src.rag.main import NEO4J_DATABASE, driver
    from src.rag.rule_quote import clean_rubrica

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--codes", default="", help="sigle or document names for checks 1-3 (default: all)")
    parser.add_argument("--all-types", action="store_true", help="check 4 over every document type")
    parser.add_argument("--out", default=str(ROOT / "docs" / "ingestion_issues.md"))
    args = parser.parse_args(argv)
    wanted = {citations.code_key(c) for c in args.codes.split(",") if c.strip()}
    misaligned, markers, abrogated = [], [], []
    checked = Counter()
    with driver.session(database=NEO4J_DATABASE, default_access_mode=neo4j.READ_ACCESS) as session:
        for doc in session.run(DOCS_QUERY).data():
            code = citations.code_key(doc["name"])
            if wanted and code not in wanted:
                continue
            articles = _canonical(session.run(SECTIONS_QUERY, doc=doc["id"]).data(), article_text.article_number)
            checked[code] += len(articles)
            for number, row in articles.items():
                rubrica, text = row["rubrica"] or "", row["text"] or ""
                neighbours = [(articles[n]["rubrica"], articles[n]["text"]) for n in article_text.neighbour_numbers(number)
                              if n in articles]
                hit = article_text.shifted(rubrica, text, neighbours)
                if hit:
                    bad = article_text.misaligned(rubrica, text) or {}
                    misaligned.append({"code": code, "article": number, "id": str(row["id"]), "rubrica": rubrica,
                                       "first_comma": bad.get("first_comma", ""), "neighbour": hit})
                cleaned = clean_rubrica(rubrica)
                if " ".join(cleaned.split()).strip(" .") != " ".join(rubrica.split()).strip(" ."):
                    markers.append({"code": code, "article": number, "id": str(row["id"]), "raw": rubrica,
                                    "cleaned": cleaned})
                if article_text.is_abrogated(text) or (not text.strip() and article_text.is_abrogated(rubrica)):
                    abrogated.append({"code": code, "article": number, "id": str(row["id"]), "rubrica": rubrica,
                                      "text": " ".join(text.split())[:160], "status_keys": row.get("status_keys") or []})
        names = session.run(NAMES_QUERY, all=bool(args.all_types)).data()
    non_normative = []
    for d in names:
        name = " ".join(str(d.get("name") or "").split())
        if NORMATIVE_RE.search(name):
            continue
        non_normative.append({"id": str(d["id"]), "name": name, "type": d.get("type") or "",
                              "kind": "pronuncia" if RULING_RE.search(name) else "altro"})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {"built": dt.datetime.now().isoformat(timespec="seconds"), "articles_checked": dict(checked),
              "misaligned": misaligned, "rubriche_with_markers": markers, "abrogated_active": abrogated,
              "non_normative_documents": non_normative, "documents_checked": len(names)}
    out.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    kinds = Counter(x["kind"] for x in non_normative)
    types = Counter(x["type"] for x in non_normative)
    lines = ["# Ingestion issues", "",
             f"Generated by `scripts/ingestion_issues.py` on {report['built']} (read only; re-run to refresh). "
             f"Articles checked: {sum(checked.values())} in {len(checked)} documents; document names checked: "
             f"{len(names)}.", "",
             "| issue | nodes |", "|---|---|",
             f"| 1. misaligned nodes (rubrica of another article's text) | {len(misaligned)} |",
             f"| 2. rubriche carrying footnote markers | {len(markers)} |",
             f"| 3. abrogated articles still active | {len(abrogated)} |",
             f"| 4. non-normative document names | {len(non_normative)} ({dict(kinds)}; by type {dict(types)}) |", "",
             "## 1. Misaligned nodes", ""]
    lines += _md_table(["code", "article", "id", "rubrica", "first comma", "confirmed by"],
                       [(x["code"], x["article"], x["id"], x["rubrica"], x["first_comma"], x["neighbour"])
                        for x in misaligned])
    lines += ["", "## 2. Rubriche carrying footnote markers", ""]
    lines += _md_table(["code", "article", "id", "raw", "cleaned"],
                       [(x["code"], x["article"], x["id"], x["raw"], x["cleaned"]) for x in markers])
    lines += ["", "## 3. Abrogated articles still active", "",
              "Served as live articles: their text is the abrogation. Status-like node properties, when any, listed.", ""]
    lines += _md_table(["code", "article", "id", "rubrica", "text", "status properties"],
                       [(x["code"], x["article"], x["id"], x["rubrica"], x["text"], ", ".join(x["status_keys"]) or "—")
                        for x in abrogated])
    lines += ["", "## 4. Non-normative document names", "",
              "Documents of the cited types whose name names no normative act (code, law, decree, regulation, "
              "directive, collective agreement, …).", ""]
    lines += _md_table(["id", "name", "type", "kind"], [(x["id"], x["name"], x["type"], x["kind"]) for x in non_normative])
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"misaligned {len(misaligned)}, markers {len(markers)}, abrogated {len(abrogated)}, "
          f"non-normative {len(non_normative)} of {len(names)}; {out}")


if __name__ == "__main__":
    main()
