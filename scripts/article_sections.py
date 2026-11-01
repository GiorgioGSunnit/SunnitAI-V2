"""Read-only: how articles are really stored as Sections, to fix article_text on what the data shows.

Run from the app folder with its venv and .env:

    venv/bin/python scripts/article_sections.py --out article_sections.txt

1 ARTICLES  the reviewed examples (--articles "61 c.p.,177 c.c." for others). Per primary document of the code: what
            article_text.fetch_article returns today (length and last words), then every Section of the article, in
            name order: id, name, type, any order field, length, the first 80 characters. Marks: HIDDEN = a name today's
            lookups cannot see ("N_2"), UNUSED = seen but outside the rule ("N.2_3", "N.0.1"), DATE, NOTE = holds an
            editorial footnote.
2 PROBES    words of the text the reviewers found missing, searched in every Section of the code: if they turn up under
            the article, assembling can restore them; under another name, they are misfiled; nowhere, only a re-ingestion
            can. A phrase can also occur in another article: read the names.
3 CENSUS    per code, over all its Sections: the shapes of the names, which shapes hide text, how many articles read today
            end at a colon, how footnotes open and whether text follows one, the "abrogato" forms, the fields a Section has
            and its relationships (is there any stored order?).
"""

import argparse
import os
import re
import sys
from collections import Counter

CODE_PREFIXES = {"c.c.": "Codice Civile", "c.p.c.": "Codice di procedura civile", "c.p.": "Codice Penale",
                 "c.p.p.": "Codice di procedura penale"}      # defensive_generation._CODE_PREFIXES
ARTICLES = ("61 c.p.", "640 c.p.", "628 c.p.", "589 c.p.", "590 c.p.", "582 c.p.", "624-bis c.p.", "177 c.c.", "179 c.c.",
            "337-ter c.c.", "593 c.p.p.", "113 c.p.c.", "658 c.p.c.")
PROBES = {"61 c.p.": "motivi abietti o futili", "640 c.p.": "a danno dello Stato", "628 c.p.": "persona travisata",
          "589 c.p.": "da due a sette anni", "590 c.p.": "di cui al secondo comma",
          "177 c.c.": "Costituiscono oggetto della comunione", "179 c.c.": "il coniuge era proprietario",
          "337-ter c.c.": "attuali esigenze del figlio"}
ORDER_FIELDS = ("order", "position", "index", "ordine", "seq")
SECTION_SIZE = 4000     # the BE's MAX_SECTION_CHARS: a longer text is cut into "name", "name_1", "name_2"...

NAME_RE = re.compile(r"^(\d+(?:-[a-z]+)?)((?:[._].*)?)$", re.IGNORECASE)
# An editorial note: "(1) Articolo aggiunto dall'art. 9", "(3) Comma così modificato", "(2) La Corte costituzionale".
NOTE_RE = re.compile(r"(?:^|(?<=[\s.;:]))\((\d{1,3})\)\s+(?=[A-ZÀ-Ü][^\n]{0,120}?(?:modificat|aggiunt|inserit|abrogat|"
                     r"sostituit|soppress|rinumerat|dichiarat|introdott|Corte costituzionale|[Vv]edi\b))")
ABROGATED_RE = re.compile(r"[(\[]?\babrogat[oaie]\b[)\].]*", re.IGNORECASE)
LIST_RE = re.compile(r"^\s*(?:\d{1,2}\)|\d{1,2}[-.]?(?:bis|ter|quater)\)|[a-z]\)|[a-z]-?(?:bis|ter|quater)\))\s")


def article_of(name: str) -> str:
    match = NAME_RE.match((name or "").strip())
    return match.group(1).lower() if match else ""


def shape(name: str) -> str:
    """"640.2_3" -> "N.k_j"; the 0 of "N.0.0" and "N.k.0" kept, as the lookups depend on it."""
    match = NAME_RE.match((name or "").strip())
    if not match:
        return "other"
    tail = re.sub(r"(?<=\.)[1-9]\d*[a-z]?(?=\.|_|$)", "k", match.group(2))
    return "N" + re.sub(r"(?<=_)\d+$", "j", tail)


def status(name: str, number: str) -> str:
    """What today's rule does with the name: "" read, "HIDDEN" the lookups never fetch it, "UNUSED" fetched, not joined."""
    tail = name[len(number):]
    if tail in ("", ".0.0") or re.fullmatch(r"\.[1-9]\d*\.0", tail):
        return ""
    return "UNUSED" if tail.startswith(".") else "HIDDEN"


def sort_key(name: str) -> list:
    return [(0, int(p), "") if p.isdigit() else (1, 0, p) for p in re.split(r"[._]", name or "")]


def plain(text: str) -> str:
    return " ".join((text or "").split())


def note_at(text: str):
    """(start of the first editorial note, characters of text after the whole note block's paragraph) or None."""
    match = NOTE_RE.search(text or "")
    return match.start() if match else None


def section_line(row: dict, number: str) -> str:
    from src.rag import article_text
    name, text = row.get("name") or "", row.get("plain_text") or ""
    order = [f"{k}={row[k]}" for k in ORDER_FIELDS if row.get(k) is not None]
    marks = [status(name, number)] if status(name, number) else []
    if article_text.in_force_only(text):
        marks.append("DATE")
    if note_at(text) is not None:
        marks.append(f"NOTE@{note_at(text)}")
    if len(text) >= SECTION_SIZE - 100:
        marks.append("FULL-SIZE")
    return (f"  {name:<16} id={str(row.get('id'))[:18]:<18} type={str(row.get('type')):<10} len={len(text):<6}"
            f"{' '.join(order)} {' '.join(marks)}\n      | {plain(text)[:80]}")


def census(rows: list) -> list:
    """Lines on one document's Sections: {"name", "plain_text"}."""
    shapes, hidden, notes_open, note_tail, abrogated, list_names = Counter(), Counter(), Counter(), Counter(), Counter(), Counter()
    samples, by_article, kinds = {}, {}, {}
    for row in rows:
        name, text = row.get("name") or "", row.get("plain_text") or ""
        number, form = article_of(name), shape(name)
        shapes[form] += 1
        samples.setdefault(form, name)
        if number:
            by_article.setdefault(number, {})[name] = text
            kinds[name] = row.get("type") or ""
            if status(name, number) and text.strip():
                hidden[(status(name, number), form)] += 1
        start = note_at(text)
        if start is not None:
            notes_open[" ".join(text[start:].split()[1:3]).lower()] += 1
            middle = re.search(r"\n\s*(?:\d{1,2}\.\s|[a-z]\)\s|\d{1,2}\)\s)", text[start:])
            note_tail["a comma or list item after the note" if middle else "nothing but note after it"] += 1
            note_tail["the whole Section is a note" if start == 0 else "text before the note"] += 1
        for form_seen in ABROGATED_RE.findall(text[:200] if len(text) > 200 else text):
            abrogated[form_seen.lower()] += 1
        if LIST_RE.match(text):
            list_names[form] += 1
    from src.rag import article_text
    colon, sample_colon = 0, []
    for number, texts in by_article.items():
        if article_text.choose_text(number, texts, kinds).rstrip().endswith(":"):
            colon += 1
            sample_colon.append(number)
    out = [f"  sections {len(rows):,}, articles {len(by_article):,}",
           f"  articles read today that end at a colon: {colon} (e.g. {', '.join(sample_colon[:15])})",
           "  name shapes (count, example):"]
    out += [f"    {form:<12} {n:>6}  {samples[form]}" for form, n in shapes.most_common(20)]
    out += ["  text today's rule does not read:"]
    out += [f"    {why:<7} {form:<12} {n:>6}" for (why, form), n in hidden.most_common()] or ["    none"]
    out += ["  a Section opening with a list marker, by name shape: "
            + (", ".join(f"{k} {n}" for k, n in list_names.most_common(8)) or "none"),
            "  footnotes open with: " + (", ".join(f"{k} {n}" for k, n in notes_open.most_common(12)) or "none"),
            "  footnotes: " + (", ".join(f"{k} {n}" for k, n in note_tail.most_common()) or "none"),
            "  abrogato forms (short Sections and first 200 characters): "
            + (", ".join(f"{k} {n}" for k, n in abrogated.most_common(8)) or "none")]
    return out


def read_document(session, doc_id: str) -> list:
    fields = ", ".join(f".{k}" for k in ("id", "name", "type", "plain_text", "titolo_articolo") + ORDER_FIELDS)
    return [{**r["s"], "_fields": r["fields"]} for r in session.run(
        f"MATCH (d:Document {{id: $doc}})-[:CONTAINS]->(s:Section) RETURN s {{{fields}}} AS s, keys(s) AS fields",
        doc=doc_id).data()]


def report(session, wanted: list) -> list:
    from src.rag import article_text
    from src.rag.article_suffix import number_variants
    documents = {}
    for code, prefix in CODE_PREFIXES.items():
        documents[code] = [(r["id"], r["name"], read_document(session, r["id"])) for r in session.run(
            "MATCH (d:Document) WHERE d.document_type = 'primary' AND d.name STARTS WITH $prefix "
            "RETURN d.id AS id, d.name AS name ORDER BY d.name", prefix=prefix).data()
            if code != "c.p." or not r["name"].startswith("Codice di procedura penale")]
    L = ["=== 1 THE ARTICLES AND 2 THE PROBES ==="]
    for ref in wanted:
        number, _, code = ref.partition(" ")
        L += ["", f"--- art. {ref}"]
        for doc_id, name, rows in documents.get(code, []):
            spelling = next((v for v in number_variants(number)
                             if any(article_of(r.get("name")) == v for r in rows)), None)
            if not spelling:
                L.append(f"{name}: no Section")
                continue
            mine = sorted((r for r in rows if article_of(r.get("name")) == spelling), key=lambda r: sort_key(r.get("name")))
            today = article_text.fetch_article(session, doc_id, number)
            text = "\n".join(plain((r.get("s") or {}).get("plain_text")) for r in today)
            L += [f"{name}: {len(mine)} Sections; rubrica: {(mine[0].get('titolo_articolo') or '-')[:60]}",
                  f"  today: {len(text)} chars, ends: ...{text[-80:]}"]
            L += [section_line(r, spelling) for r in mine]
        if ref in PROBES:
            phrase = plain(PROBES[ref]).lower()
            hits = [f"{name}: {r.get('name')}" for _, name, rows in documents.get(code, []) for r in rows
                    if phrase in plain(r.get("plain_text")).lower()]
            L.append(f"PROBE \"{PROBES[ref]}\": " + ("; ".join(hits[:8]) + (f" (+{len(hits) - 8})" if len(hits) > 8 else "")
                                                     if hits else "NOT IN ANY SECTION OF THE CODE"))
    L += ["", "=== 3 THE CENSUS ==="]
    for code, docs in documents.items():
        for _, name, rows in docs:
            L += ["", f"--- {name} ({code})"] + census(rows)
    sections = [r for docs in documents.values() for _, _, rows in docs for r in rows]
    fields = Counter(k for r in sections for k in r["_fields"])
    relations = session.run("MATCH (d:Document)-[:CONTAINS]->(s:Section) WHERE d.document_type = 'primary' WITH s LIMIT 2000 "
                            "MATCH (s)-[r]-() RETURN type(r) AS type, count(*) AS n ORDER BY n DESC").data()
    L += ["", f"Section fields: {', '.join(f'{k} {n}' for k, n in fields.most_common())}",
          f"Relationships of 2000 Sections: {', '.join(str(r['type']) + ' ' + str(r['n']) for r in relations)}"]
    return L


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--articles", help='default: the reviewed examples; e.g. "61 c.p.,177 c.c."')
    parser.add_argument("--env", default=os.path.join(os.getcwd(), ".env"))
    parser.add_argument("--out", default="article_sections.txt")
    args = parser.parse_args()
    sys.path.insert(0, os.getcwd())
    from dotenv import load_dotenv
    load_dotenv(args.env)
    uri = os.getenv("NEO4J_URI")
    if not uri:
        sys.exit(f"NEO4J_URI is not set ({args.env})")
    wanted = [a.strip() for a in args.articles.split(",")] if args.articles else list(ARTICLES)
    import neo4j
    with neo4j.GraphDatabase.driver(uri, auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))) as driver:
        with driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j"),
                            default_access_mode=neo4j.READ_ACCESS) as session:
            lines = report(session, wanted)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"File: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
