"""Read-only audit of the collective agreements (document_type 'ccnl') before the CCNL path is built
(CCNL_PATH_ENABLED, step 0). Reports:

  1 documents   how many, their labels, every property key and how often it is set (with sample values for the
                dates, sector and anything that looks like a link to another version)
  2 naming      how names read: agreement type (CCNL / testo consolidato / accordo di rinnovo / accordo / ipotesi /
                verbale / protocollo / other), whether a date is in the name, whether a sector is set
  3 versions    documents per sector (d.sector, else the name without type and date): how many sectors have 1,
                2-3, 4-10, >10 documents; the 20 largest, with their types and date span
  4 sections    per document: how many sections, how many article-shaped ("12", "12.0.0", "Art. 12"), how many
                with titolo_articolo, how many with a v2 embedding; the Section property keys
  5 pay tables  the sections on pay (minimi, tabelle retributive, paga base): text laid out as a table
                (pipes, tabs, rows of amounts) or prose; any non-Section node a CCNL contains
  6 links       relationships between a CCNL document and other Documents, by type and direction, and any
                property naming a predecessor
  7 examples    10 representative documents (by type and by the largest sectors), each with its properties and
                its first sections (name, heading, the first 300 characters)

Run from the app folder so it uses the app's .env:
    cd /opt/chatbot && venv/bin/python scripts/audit_ccnl.py                  # report to stdout and a file
    cd /opt/chatbot && venv/bin/python scripts/audit_ccnl.py --examples 10 --sections 6 --out ccnl_audit

Every session is read-only; nothing is written to the database.
"""

import argparse
import json
import logging
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (ROOT, os.getcwd()):
    if _path not in sys.path:
        sys.path.insert(0, _path)

TYPE = "ccnl"

# --- reading names (pure, tested in tests/test_audit_ccnl.py) ----------------------------------------------

# Most specific first: "Ipotesi di accordo di rinnovo" is an ipotesi, "Testo consolidato del CCNL" a consolidato,
# "Verbale di accordo" a verbale.
_AGREEMENT_TYPES = [
    ("ipotesi", re.compile(r"\bipotesi\b", re.I)),
    ("testo consolidato", re.compile(r"\btesto (?:unico |coordinato )?consolidat\w*|\bconsolidat\w*\b|"
                                     r"\btesto coordinato\b", re.I)),
    ("accordo di rinnovo", re.compile(r"\baccordo\b.{0,30}\brinnov\w*|\brinnovo\b", re.I)),
    ("verbale", re.compile(r"\bverbal\w*\b", re.I)),             # "Verbale di accordo": a verbale
    ("protocollo", re.compile(r"\bprotocoll\w*\b", re.I)),       # "Protocollo d'intesa": a protocollo
    ("accordo", re.compile(r"\baccord\w*\b|\bintesa\b", re.I)),
    ("ccnl", re.compile(r"\bc\.?c\.?n\.?l\.?\b|\bcontratto collettivo\b", re.I)),
]
_MONTHS = ("gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre")
_MONTH_NUMBER = {m: i + 1 for i, m in enumerate(_MONTHS.split("|"))}
_DATE_NUMERIC_RE = re.compile(r"\b(\d{1,2})[./-](\d{1,2})[./-]((?:19|20)\d{2})\b")
_DATE_WORDS_RE = re.compile(rf"\b(\d{{1,2}})°?\s+({_MONTHS})\s+((?:19|20)\d{{2}})\b", re.I)
_YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")
_ARTICLE_RE = re.compile(r"^\s*(?:art(?:icolo)?\.?\s*)?(\d+)(?:\s*-?\s*[a-z]+)?(?:\.\d+)*\s*$", re.I)
_LINK_KEY_RE = re.compile(r"prev|prec|supersed|replac|sostitu|rinnov|parent|origin|base|version|vers", re.I)
_DATE_KEY_RE = re.compile(r"date|data|valid|vigor|decorr|scaden|firm|sign|from|until|start|end|year|anno", re.I)
_PAY_RE = re.compile(r"minim[io] (?:tabellar|contrattual|retributiv)\w*|tabell\w* (?:retributiv|salarial)\w*|"
                     r"paga base|retribuzione tabellare|minimi conglobati|importi mensili", re.I)
_AMOUNT_RE = re.compile(r"(?:€\s*)?\b\d{1,3}(?:\.\d{3})*,\d{2}\b|€\s*\d+")
_EMBEDDING_KEYS_RE = re.compile(r"embedding", re.I)


def agreement_type(name: str) -> str:
    """The kind of agreement a document's name says it is; "other" when it says none."""
    for kind, pattern in _AGREEMENT_TYPES:
        if pattern.search(name or ""):
            return kind
    return "other"


def name_date(name: str) -> Tuple[Optional[str], str]:
    """(the date in the name as YYYY-MM-DD, or YYYY when only a year is written; how it was found:
    "numeric", "words", "year", "none"). The last date written wins (names put the signing date last)."""
    text = name or ""
    found = []
    for m in _DATE_NUMERIC_RE.finditer(text):
        day, month, year = int(m.group(1)), int(m.group(2)), m.group(3)
        if 1 <= day <= 31 and 1 <= month <= 12:
            found.append((m.start(), f"{year}-{month:02d}-{day:02d}", "numeric"))
    for m in _DATE_WORDS_RE.finditer(text):
        found.append((m.start(), f"{m.group(3)}-{_MONTH_NUMBER[m.group(2).lower()]:02d}-{int(m.group(1)):02d}",
                      "words"))
    if found:
        _, value, how = max(found)
        return value, how
    years = _YEAR_RE.findall(text)
    return (years[-1], "year") if years else (None, "none")


def sector_key(doc: Dict[str, Any]) -> str:
    """The document's sector: its "sector" property, else its name without agreement words, dates and
    punctuation (so the versions of one contract group together)."""
    sector = str(doc.get("sector") or "").strip()
    if sector:
        return sector.lower()
    name = str(doc.get("name") or "")
    name = _DATE_NUMERIC_RE.sub(" ", name)
    name = _DATE_WORDS_RE.sub(" ", name)
    name = _YEAR_RE.sub(" ", name)
    name = re.sub(r"\b(?:c\.?c\.?n\.?l\.?|contratto collettivo nazionale(?: di lavoro)?|ipotesi|accordo|accordi|di|"
                  r"del|della|dei|per|il|la|lo|i|le|rinnovo|testo|consolidato|coordinato|verbale|protocollo|"
                  r"stipulato|sottoscritto|firmato|sottoscrizione|integrativo|nazionale|lavoro|dipendenti|"
                  r"addetti|settore|n|art|e|ed|in|al|dal|da|dalle|alle)\b", " ", name, flags=re.I)
    return " ".join(re.sub(r"[^\w\s]", " ", name.lower()).split()) or "(no sector)"


def article_shaped(section_name: str) -> bool:
    """A section named as an article: "12", "12.0.0", "12-bis", "Art. 12", "Articolo 12"."""
    return bool(_ARTICLE_RE.match(section_name or ""))


def table_shape(text: str) -> str:
    """How a section's text holds figures: "markdown table" (pipe rows), "tab-separated", "rows of amounts"
    (three or more lines with two or more amounts each), "amounts in prose", or "no amounts"."""
    text = text or ""
    lines = [line for line in text.splitlines() if line.strip()]
    if sum(1 for line in lines if line.count("|") >= 2) >= 2:
        return "markdown table"
    if sum(1 for line in lines if "\t" in line) >= 3:
        return "tab-separated"
    if sum(1 for line in lines if len(_AMOUNT_RE.findall(line)) >= 2) >= 3:
        return "rows of amounts"
    return "amounts in prose" if _AMOUNT_RE.search(text) else "no amounts"


def pick_examples(docs: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    """n representative documents: one per agreement type first (the one with most sections), then the
    largest sectors' most recent documents, no document twice."""
    chosen: List[Dict[str, Any]] = []
    seen = set()

    def take(doc):
        if doc["id"] not in seen and len(chosen) < n:
            seen.add(doc["id"])
            chosen.append(doc)

    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for doc in docs:
        by_type[doc["type"]].append(doc)
    for kind in [k for k, _ in _AGREEMENT_TYPES] + ["other"]:
        if by_type.get(kind):
            take(max(by_type[kind], key=lambda d: (d.get("sections") or 0, d["id"])))
    by_sector: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for doc in docs:
        by_sector[doc["sector"]].append(doc)
    for _, group in sorted(by_sector.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        take(max(group, key=lambda d: (str(d.get("date") or ""), d["id"])))
        if len(chosen) >= n:
            break
    return chosen


def _short(value: Any, limit: int = 120) -> Any:
    if isinstance(value, (list, tuple)):
        if value and all(isinstance(v, (int, float)) for v in value[:5]):
            return f"<vector of {len(value)}>"
        return [_short(v, limit) for v in value[:5]] + (["..."] if len(value) > 5 else [])
    text = str(value)
    return text if len(text) <= limit else text[:limit] + "..."


def _bucket(n: int) -> str:
    return "1" if n == 1 else "2-3" if n <= 3 else "4-10" if n <= 10 else ">10"


# --- reading the graph ----------------------------------------------------------------------------------

def _read_only_driver():
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.getcwd(), ".env"))
    import neo4j
    logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)
    real = neo4j.Driver.session

    def read_only(self, **config):
        config["default_access_mode"] = neo4j.READ_ACCESS
        return real(self, **config)
    neo4j.Driver.session = read_only
    driver = neo4j.GraphDatabase.driver(os.getenv("NEO4J_URI"),
                                        auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")))
    return driver, os.getenv("NEO4J_DATABASE", "neo4j")


def audit(session, n_examples: int, n_sections: int, vector_property: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    # 1 documents
    out["labels"] = {"+".join(sorted(r["labels"])): r["n"] for r in session.run(
        "MATCH (d) WHERE d.document_type = $t RETURN labels(d) AS labels, count(*) AS n", t=TYPE).data()}
    keys = session.run(
        "MATCH (d:Document) WHERE d.document_type = $t UNWIND keys(d) AS k RETURN k, count(*) AS n ORDER BY n DESC",
        t=TYPE).data()
    out["document_keys"] = {r["k"]: r["n"] for r in keys}
    interesting = [k for k in out["document_keys"]
                   if (_DATE_KEY_RE.search(k) or _LINK_KEY_RE.search(k) or k in ("sector", "source", "type"))
                   and not _EMBEDDING_KEYS_RE.search(k)]
    out["document_key_samples"] = {}
    for k in interesting:
        rows = session.run(f"MATCH (d:Document) WHERE d.document_type = $t AND d.`{k}` IS NOT NULL "
                           f"RETURN d.`{k}` AS v LIMIT 8", t=TYPE).data()
        out["document_key_samples"][k] = [_short(r["v"], 80) for r in rows]

    docs = session.run(
        "MATCH (d:Document) WHERE d.document_type = $t "
        "OPTIONAL MATCH (d)-[:CONTAINS]->(s:Section) "
        "WITH d, count(s) AS sections, "
        "     sum(CASE WHEN s.titolo_articolo IS NOT NULL AND trim(toString(s.titolo_articolo)) <> '' "
        "              THEN 1 ELSE 0 END) AS headed, "
        f"     sum(CASE WHEN s.`{vector_property}` IS NOT NULL THEN 1 ELSE 0 END) AS embedded, "
        "     collect(s.name)[..400] AS names "
        "RETURN d.id AS id, d.name AS name, d.sector AS sector, d.valid_until AS valid_until, "
        "       sections, headed, embedded, names", t=TYPE).data()
    records = []
    for r in docs:
        date, how = name_date(r["name"] or "")
        names = [n for n in r["names"] or [] if n is not None]
        records.append({"id": r["id"], "name": r["name"], "type": agreement_type(r["name"] or ""),
                        "date": date, "date_from": how, "sector": sector_key(r), "sector_set": bool(r["sector"]),
                        "valid_until": str(r["valid_until"]) if r["valid_until"] is not None else None,
                        "sections": r["sections"], "headed": r["headed"], "embedded": r["embedded"],
                        "article_shaped": sum(1 for n in names if article_shaped(str(n))),
                        "names_sampled": len(names)})
    out["documents"] = len(records)

    # 2 naming
    out["agreement_types"] = dict(Counter(d["type"] for d in records).most_common())
    out["date_in_name"] = dict(Counter(d["date_from"] for d in records).most_common())
    out["sector_property_set"] = sum(d["sector_set"] for d in records)
    out["valid_until_set"] = sum(1 for d in records if d["valid_until"])
    out["name_samples_by_type"] = {kind: [d["name"] for d in records if d["type"] == kind][:6]
                                   for kind in out["agreement_types"]}

    # 3 versions per sector
    sectors: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for d in records:
        sectors[d["sector"]].append(d)
    out["sectors"] = len(sectors)
    out["documents_per_sector"] = dict(Counter(_bucket(len(g)) for g in sectors.values()))
    out["largest_sectors"] = [
        {"sector": s, "documents": len(g), "types": dict(Counter(d["type"] for d in g).most_common()),
         "dates": [min((str(d["date"]) for d in g if d["date"]), default=None),
                   max((str(d["date"]) for d in g if d["date"]), default=None)],
         "names": [d["name"] for d in sorted(g, key=lambda d: str(d["date"] or ""))][-4:]}
        for s, g in sorted(sectors.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:20]]

    # 4 sections
    with_sections = [d for d in records if d["sections"]]
    total = sum(d["sections"] for d in records) or 1
    sampled = sum(d["names_sampled"] for d in records) or 1
    out["sections"] = {
        "total": sum(d["sections"] for d in records),
        "documents_without_sections": len(records) - len(with_sections),
        "per_document": dict(Counter(_bucket(d["sections"]) if d["sections"] else "0" for d in records)),
        "article_shaped_share": round(sum(d["article_shaped"] for d in records) / sampled, 3),
        "documents_mostly_articles": sum(1 for d in with_sections if d["names_sampled"]
                                         and d["article_shaped"] / d["names_sampled"] >= 0.5),
        "titolo_articolo_share": round(sum(d["headed"] for d in records) / total, 3),
        f"{vector_property}_share": round(sum(d["embedded"] for d in records) / total, 3),
    }
    out["section_keys"] = {r["k"]: r["n"] for r in session.run(
        "MATCH (d:Document)-[:CONTAINS]->(s:Section) WHERE d.document_type = $t "
        "WITH s LIMIT 20000 UNWIND keys(s) AS k RETURN k, count(*) AS n ORDER BY n DESC", t=TYPE).data()}
    out["section_name_samples"] = Counter(
        re.sub(r"\d+", "N", str(n)) for d in docs for n in (d["names"] or [])[:30] if n is not None).most_common(15)

    # 5 pay tables
    pay = session.run(
        "MATCH (d:Document)-[:CONTAINS]->(s:Section) WHERE d.document_type = $t AND s.plain_text IS NOT NULL "
        "AND toLower(s.plain_text) =~ '(?s).*(minim[io] (tabellar|contrattual|retributiv)|tabell[ae] (retributiv|salarial)"
        "|paga base|retribuzione tabellare|minimi conglobati).*' "
        "RETURN d.name AS doc, s.name AS section, s.titolo_articolo AS heading, s.plain_text AS text LIMIT 300",
        t=TYPE).data()
    shapes = Counter(table_shape(r["text"]) for r in pay)
    out["pay_sections"] = {"sampled": len(pay), "shapes": dict(shapes.most_common()),
                           "examples": [{"doc": r["doc"], "section": r["section"], "heading": r["heading"],
                                         "shape": table_shape(r["text"]), "text": (r["text"] or "")[:500]}
                                        for r in _one_per_shape(pay)]}
    out["contained_labels"] = {"+".join(sorted(r["labels"])): r["n"] for r in session.run(
        "MATCH (d:Document)-[:CONTAINS]->(x) WHERE d.document_type = $t RETURN labels(x) AS labels, count(*) AS n",
        t=TYPE).data()}

    # 6 links
    out["relationships"] = [
        {"direction": r["dir"], "type": r["type"], "other": r["other"], "n": r["n"]} for r in session.run(
            "MATCH (d:Document)-[r]->(o:Document) WHERE d.document_type = $t "
            "RETURN 'out' AS dir, type(r) AS type, o.document_type AS other, count(*) AS n "
            "UNION ALL MATCH (d:Document)<-[r]-(o:Document) WHERE d.document_type = $t "
            "RETURN 'in' AS dir, type(r) AS type, o.document_type AS other, count(*) AS n", t=TYPE).data()]
    out["predecessor_keys"] = {k: n for k, n in out["document_keys"].items() if _LINK_KEY_RE.search(k)}
    out["other_relationship_types"] = [
        {"type": r["type"], "n": r["n"]} for r in session.run(
            "MATCH (d:Document)-[r]-() WHERE d.document_type = $t AND type(r) <> 'CONTAINS' "
            "RETURN type(r) AS type, count(*) AS n ORDER BY n DESC", t=TYPE).data()]

    # 7 examples
    examples = []
    for d in pick_examples(records, n_examples):
        props = session.run("MATCH (d:Document {id: $id}) RETURN properties(d) AS p", id=d["id"]).single()["p"]
        first = session.run(
            "MATCH (d:Document {id: $id})-[:CONTAINS]->(s:Section) "
            "RETURN s.name AS name, s.titolo_articolo AS heading, s.title AS title, s.plain_text AS text, "
            "       keys(s) AS keys "
            "ORDER BY coalesce(s.order, s.position, s.index, 0), s.name LIMIT $k", id=d["id"], k=n_sections).data()
        examples.append({"name": d["name"], "type": d["type"], "sector": d["sector"], "date": d["date"],
                         "sections": d["sections"], "article_shaped": d["article_shaped"], "headed": d["headed"],
                         "properties": {k: _short(v) for k, v in props.items() if not _EMBEDDING_KEYS_RE.search(k)},
                         "first_sections": [{"name": s["name"], "heading": s["heading"] or s["title"],
                                             "text": " ".join(str(s["text"] or "").split())[:300],
                                             "keys": sorted(k for k in s["keys"] if not _EMBEDDING_KEYS_RE.search(k))}
                                            for s in first]})
    out["examples"] = examples
    return out


def _one_per_shape(rows: Iterable[Dict[str, Any]], per_shape: int = 2) -> List[Dict[str, Any]]:
    taken: Dict[str, int] = defaultdict(int)
    out = []
    for r in rows:
        shape = table_shape(r["text"])
        if taken[shape] < per_shape:
            taken[shape] += 1
            out.append(r)
    return out


# --- the report ------------------------------------------------------------------------------------------

def report(a: Dict[str, Any], vector_property: str) -> List[str]:
    L = [f"# CCNL audit ({a['documents']} documents with document_type '{TYPE}')", ""]
    L += ["## 1 Documents", f"Labels: {a['labels']}", "", "| property | set on |", "|---|---:|"]
    L += [f"| {k} | {n} |" for k, n in a["document_keys"].items()]
    L += ["", "Sample values:"] + [f"- {k}: {v}" for k, v in a["document_key_samples"].items()]
    L += ["", "## 2 Naming", f"Agreement type (from the name): {a['agreement_types']}",
          f"Date in the name: {a['date_in_name']}",
          f"sector property set: {a['sector_property_set']}/{a['documents']}; "
          f"valid_until set: {a['valid_until_set']}/{a['documents']}", ""]
    for kind, names in a["name_samples_by_type"].items():
        L += [f"- {kind}:"] + [f"    - {n}" for n in names]
    L += ["", "## 3 Versions per sector", f"Sectors: {a['sectors']}; documents per sector: {a['documents_per_sector']}",
          "", "| sector | docs | types | dates (from names) | latest names |", "|---|---:|---|---|---|"]
    for s in a["largest_sectors"]:
        L.append(f"| {s['sector'][:50]} | {s['documents']} | {s['types']} | {s['dates'][0]} .. {s['dates'][1]} | "
                 f"{'; '.join(str(n)[:60] for n in s['names'])} |")
    sec = a["sections"]
    L += ["", "## 4 Sections",
          f"Total {sec['total']}; documents without sections {sec['documents_without_sections']}; "
          f"per document {sec['per_document']}",
          f"Article-shaped names: {sec['article_shaped_share']:.1%} of names; documents mostly articles: "
          f"{sec['documents_mostly_articles']}",
          f"titolo_articolo set: {sec['titolo_articolo_share']:.1%}; {vector_property} set: "
          f"{sec[f'{vector_property}_share']:.1%}",
          f"Section keys (first 20,000 sections): {a['section_keys']}",
          f"Section name shapes: {a['section_name_samples']}"]
    pay = a["pay_sections"]
    L += ["", "## 5 Pay tables", f"Pay sections sampled: {pay['sampled']}; shapes: {pay['shapes']}",
          f"Nodes a CCNL contains, by label: {a['contained_labels']}", ""]
    for e in pay["examples"]:
        L += [f"- [{e['shape']}] {e['doc']} / {e['section']} ({e['heading']}):", "```", e["text"], "```"]
    L += ["", "## 6 Links", "| dir | relationship | other document type | n |", "|---|---|---|---:|"]
    L += [f"| {r['direction']} | {r['type']} | {r['other']} | {r['n']} |" for r in a["relationships"]] or ["| - | none | - | 0 |"]
    L += [f"Non-CONTAINS relationships of any kind: {a['other_relationship_types'] or 'none'}",
          f"Properties that may name a predecessor: {a['predecessor_keys'] or 'none'}"]
    L += ["", "## 7 Representative documents"]
    for i, e in enumerate(a["examples"], 1):
        L += ["", f"### {i}. {e['name']}",
              f"type {e['type']}; sector {e['sector']}; date {e['date']}; {e['sections']} sections "
              f"({e['article_shaped']} article-shaped, {e['headed']} with titolo_articolo)",
              f"properties: {json.dumps(e['properties'], ensure_ascii=False, default=str)}"]
        for s in e["first_sections"]:
            L.append(f"- `{s['name']}` | {s['heading']} | {s['text']}  (keys: {', '.join(s['keys'])})")
    return L


def main(argv=None):
    parser = argparse.ArgumentParser(description="Read-only audit of the CCNL documents (document_type 'ccnl').")
    parser.add_argument("--examples", type=int, default=10, help="representative documents shown (default 10)")
    parser.add_argument("--sections", type=int, default=5, help="first sections shown per document (default 5)")
    parser.add_argument("--out", default="ccnl_audit", help="folder for report.md and audit.json")
    args = parser.parse_args(argv)
    driver, database = _read_only_driver()
    from src.rag import embedding_config
    with driver.session(database=database) as session:
        result = audit(session, args.examples, args.sections, embedding_config.VECTOR_PROPERTY)
    driver.close()
    lines = report(result, embedding_config.VECTOR_PROPERTY)
    folder = os.path.join(args.out, datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(folder, "audit.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1, default=str)
    print("\n".join(lines))
    print(f"\nWritten to {folder}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
