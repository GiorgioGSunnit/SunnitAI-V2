#!/usr/bin/env python3
"""v60h (4): build data/institutions_<code>.json -- for each code, the institutions a practitioner expects (name,
rubricated name, articles), generated once from the store's own structure, never hand-written per case. Read only on
the graph; writes the JSON files and prints, per code, the structure it found.

The structure is not the same in every code (the first build gave c.p.a. no unit), so each code is probed first and its
index built from the first source that has units:

  1. units: Unit nodes linked to the code's article Sections by any relationship, at any level (TITOLO, CAPO, SEZIONE,
     PARAGRAFO, LIBRO, PARTE; «Title», «chapter» … read as those), and every unit above them up the CHILD_OF chain
     (scripts/article_links.py: an article's deepest unit holds it), each unit an institution named by its title;
  2. properties: a structural property on the article Sections (capo, titolo, sezione, libro, parte, a breadcrumb or
     path), its values the units;
  3. headings: heading Sections («Titolo II – …», «Capo I – …») in the NEXT chain, each article under the last heading
     of each level before it;
  4. rubriche: none of these -- each article an institution of its own, named by its rubrica (level ARTICOLO).

v60n: with units (1-3), every article is also an entry of its own -- a leaf, level ARTICOLO, named by its rubrica, its
"parent" the deepest unit holding it -- so the index channel (fact_resolver.index_channel, which ranks an article first
on a tie) resolves a single-article institution («Truffa» -> 640 c.p.); a generic rubrica («Circostanze aggravanti») is
no leaf.

A code's documents are those whose name reads as the code (citations.code_key), else whose name starts with its
prefix, case aside (dg._CODE_PREFIXES), else whose name holds one of its ALIASES; every document type. The probe --
documents (id, name, type, articles), unit levels and relationships to the articles and to the documents, the article
Sections' properties, heading Sections found, the source used -- is printed and saved in the file ("structure").

  python scripts/build_institutions.py                          # c.c., c.p., c.p.c., c.p.p., c.p.a.
  python scripts/build_institutions.py --codes c.p.a.           # one code
  python scripts/build_institutions.py --probe --codes c.p.a.   # the structure only, nothing written
  python scripts/build_institutions.py --out /tmp/institutions  # elsewhere than data/
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

LEVEL_NAMES = {"LIBRO": "LIBRO", "BOOK": "LIBRO", "PARTE": "PARTE", "PART": "PARTE", "TITOLO": "TITOLO",
               "TITLE": "TITOLO", "CAPO": "CAPO", "CHAPTER": "CAPO", "SEZIONE": "SEZIONE", "SECTION": "SEZIONE",
               "PARAGRAFO": "PARAGRAFO", "PARAGRAPH": "PARAGRAFO", "§": "PARAGRAFO"}
LEVEL_ORDER = ("PARAGRAFO", "SEZIONE", "CAPO", "TITOLO", "PARTE", "LIBRO", "ARTICOLO")
ALIASES = {"c.p.a.": r"processo amministrativo|104/2010", "c.p.c.": r"procedura civile", "c.p.p.": r"procedura penale",
           "c.c.": r"^codice civile", "c.p.": r"^codice penale"}
STRUCT_KEYS = ("libro", "parte", "titolo", "capo", "sezione", "paragrafo", "book", "part", "chapter")
PATH_KEYS = ("breadcrumb", "path", "title_path", "hierarchy", "struttura", "percorso")
_HEADING_RE = re.compile(r"^\s*(libro|parte|titolo|capo|sezione|paragrafo|§)\s+([ivxlcdm\d]+(?:[-\s]*(?:bis|ter|quater|"
                         r"quinquies|sexies|septies|octies))?)\b", re.IGNORECASE)

DOCS_QUERY = "MATCH (d:Document) RETURN d.id AS id, d.name AS name, d.document_type AS type"
ARTICLES_QUERY = ("MATCH (d:Document)-[:CONTAINS]->(s:Section) WHERE d.id IN $docs AND s.titolo_articolo IS NOT NULL "
                  "RETURN d.id AS doc, count(s) AS n")
ARTICLE_UNITS_QUERY = ("MATCH (d:Document)-[:CONTAINS]->(s:Section) WHERE d.id IN $docs AND s.titolo_articolo IS NOT NULL "
                       "MATCH (u:Unit)-[r]-(s) RETURN type(r) AS rel, u.level AS level, count(*) AS n")
DOC_UNITS_QUERY = "MATCH (d:Document)-[r]-(u:Unit) WHERE d.id IN $docs RETURN type(r) AS rel, u.level AS level, count(*) AS n"
# the article's own unit and, up its CHILD_OF chain (scripts/article_links.py links an article to its deepest unit only),
# every unit above it: the Capo holds the articles of its Sezioni, the Titolo those of its Capi
UNITS_QUERY = ("MATCH (d:Document)-[:CONTAINS]->(s:Section) WHERE d.id IN $docs AND s.titolo_articolo IS NOT NULL "
               "MATCH (u0:Unit)-[]-(s) MATCH (u0)-[:CHILD_OF*0..6]->(u:Unit) "
               "RETURN DISTINCT elementId(u) AS uid, u.level AS level, u.name AS unit, s.name AS name, "
               "s.titolo_articolo AS rubrica")
ANCESTOR_UNITS_QUERY = ("MATCH (d:Document)-[:CONTAINS]->(s:Section) WHERE d.id IN $docs AND s.titolo_articolo IS NOT NULL "
                        "MATCH (u0:Unit)-[]-(s) MATCH (u0)-[:CHILD_OF*1..6]->(u:Unit) "
                        "RETURN u.level AS level, count(DISTINCT u) AS n")
KEYS_QUERY = ("MATCH (d:Document)-[:CONTAINS]->(s:Section) WHERE d.id IN $docs AND s.titolo_articolo IS NOT NULL "
              "UNWIND keys(s) AS k RETURN k, count(*) AS n")
VALUES_QUERY = ("MATCH (d:Document)-[:CONTAINS]->(s:Section) WHERE d.id IN $docs AND s.titolo_articolo IS NOT NULL "
                "RETURN s.name AS name, s.titolo_articolo AS rubrica, s[$key] AS value")
CHAIN_QUERY = ("MATCH (d:Document)-[:CONTAINS]->(s:Section) WHERE d.id IN $docs "
               "OPTIONAL MATCH (s)-[:NEXT]->(n:Section) "
               "RETURN elementId(s) AS sid, s.name AS name, s.title AS title, s.titolo_articolo AS rubrica, "
               "elementId(n) AS next")


def level_of(raw) -> str:
    """A unit level as one of LEVEL_ORDER where it reads as one («Capo», «chapter»), else its own name upper case."""
    text = str(raw or "").strip().upper()
    return LEVEL_NAMES.get(text, text or "?")


def _order(number: str) -> tuple:
    m = re.match(r"(\d+)(?:-([a-z]+))?", number)
    return (int(m.group(1)), m.group(2) or "") if m else (10 ** 9, number)


def _rank(level: str) -> int:
    return LEVEL_ORDER.index(level) if level in LEVEL_ORDER else len(LEVEL_ORDER)


def _rubricated(unit: str) -> str:
    return _HEADING_RE.sub("", " ".join(str(unit or "").split())).lstrip(" –—:.-") or " ".join(str(unit or "").split())


def entries(rows, article_number, unit_title, generic, clean_rubrica):
    """[{name, rubricated, level, articles, rubriche}] from rows {uid, level, unit, name, rubrica}, one per unit, its
    articles in order; a generic or empty unit name left out. v60n: then one leaf per article of a unit (leaves)."""
    units = {}
    for r in rows:
        number = article_number(r.get("name") or "")
        if not number or r.get("name") not in (number, f"{number}.0.0"):
            continue
        u = units.setdefault(r["uid"], {"level": level_of(r.get("level")), "unit": r.get("unit") or "", "rubriche": {}})
        u["rubriche"][number] = clean_rubrica(str(r.get("rubrica") or "")).strip(" .")
    out = []
    for u in units.values():
        name = unit_title(u["unit"])
        if not name or generic(name):
            continue
        articles = sorted(u["rubriche"], key=_order)
        out.append({"name": name, "rubricated": _rubricated(u["unit"]), "level": u["level"], "articles": articles,
                    "rubriche": {a: u["rubriche"][a] for a in articles}})
    out += leaves(units.values(), unit_title, generic)
    out.sort(key=lambda e: (_order(e["articles"][0]) if e["articles"] else (10 ** 9, ""), _rank(e["level"])))
    return out


def leaves(units, unit_title, generic):
    """v60n: one entry per article of the units (never of an ARTICOLO unit: the rubriche source is leaves already) --
    {name and rubricated: its rubrica, level ARTICOLO, articles [it], rubriche, parent: the deepest named unit holding
    it, parent_level}; an article with no rubrica, or a generic one, none."""
    best = {}
    for u in units:
        if u["level"] == "ARTICOLO":
            continue
        name = unit_title(u["unit"])
        for number, rubrica in u["rubriche"].items():
            rank = _rank(u["level"]) if name and not generic(name) else len(LEVEL_ORDER) + 1
            if number not in best or rank < best[number][0]:
                best[number] = (rank, rubrica, name if rank <= len(LEVEL_ORDER) else "", u["level"])
    out = []
    for number, (_, rubrica, parent, level) in best.items():
        if rubrica and not generic(rubrica):
            out.append({"name": rubrica, "rubricated": rubrica, "level": "ARTICOLO", "articles": [number],
                        "rubriche": {number: rubrica}, "parent": parent, "parent_level": level if parent else ""})
    return out


def property_rows(values, key: str):
    """Rows (entries' shape) from a structural property's values on the article Sections: a level key («capo»), its
    value the unit; a path key («breadcrumb»), each of its segments that reads as a heading («Capo II – …»)."""
    rows = []
    for v in values:
        value = v.get("value")
        if value in (None, "", []):
            continue
        if key.lower() in PATH_KEYS:
            parts = value if isinstance(value, list) else re.split(r"\s*(?:>|/|\||»|›)\s*", str(value))
            for part in parts:
                m = _HEADING_RE.match(str(part))
                if m:
                    rows.append({"uid": f"{key}:{part}", "level": m.group(1), "unit": str(part), "name": v["name"],
                                 "rubrica": v.get("rubrica")})
        else:
            rows.append({"uid": f"{key}:{value}", "level": key, "unit": str(value), "name": v["name"],
                         "rubrica": v.get("rubrica")})
    return rows


def chain_rows(sections):
    """Rows (entries' shape) from the NEXT chain: each article under the last heading Section of each level before it
    («Titolo II – …», «Capo I – …» as a Section's name or title). [] when no heading is found or the chain is broken."""
    by_id = {s["sid"]: s for s in sections}
    pointed = {s.get("next") for s in sections if s.get("next")}
    heads = [s for s in sections if s["sid"] not in pointed]
    rows, seen = [], set()
    for head in heads:
        current, open_units = head, {}
        while current is not None and current["sid"] not in seen:
            seen.add(current["sid"])
            label = next((x for x in (current.get("title"), current.get("name")) if x and _HEADING_RE.match(str(x))), None)
            if label and not current.get("rubrica"):
                level = level_of(_HEADING_RE.match(label).group(1))
                open_units[level] = (current["sid"], label)
                for lower in LEVEL_ORDER[:_rank(level)]:      # a new Capo closes the Sezioni under the old one
                    open_units.pop(lower, None)
            elif current.get("rubrica"):
                for level, (uid, label_) in open_units.items():
                    rows.append({"uid": uid, "level": level, "unit": label_, "name": current.get("name"),
                                 "rubrica": current.get("rubrica")})
            current = by_id.get(current.get("next"))
    return rows


def rubrica_rows(values):
    """Rows (entries' shape) with each article its own institution, named by its rubrica (level ARTICOLO)."""
    return [{"uid": f"art:{v['name']}", "level": "ARTICOLO", "unit": v.get("rubrica") or "", "name": v["name"],
             "rubrica": v.get("rubrica")} for v in values if v.get("rubrica")]


def code_documents(docs, code: str, code_key, prefix: str):
    """The code's documents: by code_key, else by prefix (case aside), else by ALIASES."""
    by_key = [d for d in docs if code_key(d.get("name") or "") == code]
    if by_key:
        return by_key, "code_key"
    low = (prefix or "").lower()
    by_prefix = [d for d in docs if low and str(d.get("name") or "").lower().startswith(low)]
    if by_prefix:
        return by_prefix, "prefix"
    alias = ALIASES.get(code)
    return ([d for d in docs if alias and re.search(alias, str(d.get("name") or ""), re.IGNORECASE)], "alias")


def probe(session, docs):
    """The structure of the documents' articles (see the module docstring)."""
    articles = {r["doc"]: r["n"] for r in session.run(ARTICLES_QUERY, docs=docs).data()}
    unit_levels = Counter()
    for r in session.run(ARTICLE_UNITS_QUERY, docs=docs).data():
        unit_levels[f"{r['rel']}:{r['level']}"] += r["n"]
    doc_levels = Counter()
    for r in session.run(DOC_UNITS_QUERY, docs=docs).data():
        doc_levels[f"{r['rel']}:{r['level']}"] += r["n"]
    keys = {r["k"]: r["n"] for r in session.run(KEYS_QUERY, docs=docs).data()}
    above = {str(r["level"]): r["n"] for r in session.run(ANCESTOR_UNITS_QUERY, docs=docs).data()} \
        if unit_levels else {}
    return {"articles": articles, "article_unit_levels": dict(unit_levels), "units_above": above,
            "document_unit_levels": dict(doc_levels), "section_keys": keys}


def build(session, docs, structure, helpers):
    """(entries, source) from the first source with units: units, properties, headings, rubriche."""
    article_number, unit_title, generic, clean_rubrica = helpers
    if structure["article_unit_levels"]:
        found = entries(session.run(UNITS_QUERY, docs=docs).data(), *helpers)
        if found:
            return found, "units"
    for key in [k for k in structure["section_keys"] if k.lower() in STRUCT_KEYS + PATH_KEYS]:
        values = session.run(VALUES_QUERY, docs=docs, key=key).data()
        found = entries(property_rows(values, key), *helpers)
        if found:
            return found, f"property:{key}"
    sections = session.run(CHAIN_QUERY, docs=docs).data()
    structure["heading_sections"] = sum(1 for s in sections if not s.get("rubrica") and any(
        x and _HEADING_RE.match(str(x)) for x in (s.get("title"), s.get("name"))))
    found = entries(chain_rows(sections), *helpers)
    if found:
        return found, "headings"
    values = [{"name": s.get("name"), "rubrica": s.get("rubrica")} for s in sections if s.get("rubrica")]
    return entries(rubrica_rows(values), article_number, lambda unit: clean_rubrica(unit).strip(" ."),
                   lambda name: False, clean_rubrica), "rubriche"


def main(argv=None) -> None:
    import neo4j
    from src.rag import article_text, citations, defensive_generation as dg, fact_resolver, topic_groups
    from src.rag.rule_quote import clean_rubrica
    from src.rag.main import NEO4J_DATABASE, driver

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--codes", default="c.c.,c.p.,c.p.c.,c.p.p.,c.p.a.")
    parser.add_argument("--out", default=str(ROOT / "data"))
    parser.add_argument("--probe", action="store_true", help="print the structure only, write nothing")
    args = parser.parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    helpers = (article_text.article_number, topic_groups.unit_title, topic_groups.generic, clean_rubrica)
    summary = []
    with driver.session(database=NEO4J_DATABASE, default_access_mode=neo4j.READ_ACCESS) as session:
        all_docs = session.run(DOCS_QUERY).data()
        for code in [c.strip() for c in args.codes.split(",") if c.strip()]:
            matched, how = code_documents(all_docs, code, citations.code_key, dg._CODE_PREFIXES.get(code, ""))
            ids = [d["id"] for d in matched]
            print(f"\n## {code}: {len(matched)} document(s) by {how}")
            for d in matched:
                print(f"   {d['id']}  «{d['name']}»  ({d.get('type')})")
            if not ids:
                summary.append((code, "no document", "—", "—", "—", 0, 0))
                continue
            structure = probe(session, ids)
            structure.update(documents=[{"id": d["id"], "name": d["name"], "type": d.get("type")} for d in matched],
                             matched_by=how)
            print(f"   articles per document: {structure['articles']}")
            print(f"   unit levels on the articles (relationship:level): {structure['article_unit_levels'] or 'none'}")
            print(f"   units above them (CHILD_OF), by level: {structure['units_above'] or 'none'}")
            print(f"   unit levels on the documents: {structure['document_unit_levels'] or 'none'}")
            print(f"   article Section properties: {sorted(structure['section_keys'])}")
            if args.probe:
                continue
            found, source = build(session, ids, structure, helpers)
            structure["source"] = source
            levels = dict(Counter(e["level"] for e in found))
            structure["index_levels"] = levels
            print(f"   built from {source}: {len(found)} institutions {levels}"
                  + (f"; heading sections {structure['heading_sections']}" if "heading_sections" in structure else ""))
            path = out_dir / f"institutions_{fact_resolver.code_slug(code)}.json"
            path.write_text(json.dumps({"code": code, "built": dt.date.today().isoformat(), "structure": structure,
                                        "institutions": found}, ensure_ascii=False, indent=1), encoding="utf-8")
            summary.append((code, how, ", ".join(f"{k} {v}" for k, v in sorted(structure["article_unit_levels"].items()))
                            or "none", source, ", ".join(f"{k} {v}" for k, v in levels.items()), len(found),
                            sum(structure["articles"].values())))
    print("\n| code | documents by | unit types on the articles | index built from | index levels | institutions | "
          "articles |")
    print("|---|---|---|---|---|---|---|")
    for row in summary:
        print("| " + " | ".join(str(x) for x in row) + " |")


if __name__ == "__main__":
    main()
