"""Read-only: which articles of the main codes are stored as in force although they look repealed?

Found on 17 Nov 2026: the Codice Penale holds arts. 519-526 (repealed in 1996) with their old text, a heading that is
only a footnote mark «(1)» and no word of the repeal -- the note was lost at loading, so the app's own guard
(article_text.is_abrogated, which reads the stored text) cannot see them.

Per code document: its articles; how many say «abrogat» in their text; how many have a heading without any word
(empty or only marks such as «(1)»), split into those whose text says «abrogat» and those whose text does not. The
second group are the candidates, printed as article numbers. A code where most headings have no words (the
Constitution has no headings) gets counts only. A candidate is a suspect, not a verdict: an article in force can lack
a heading too.

Prints the names of the code documents, article numbers and counts: no text, and no document outside the main codes.

    cd /opt/chatbot && sudo venv/bin/python -I /home/sunnit/repealed_articles.py      # be2, reads health.env
    python ops/be2/repealed_articles.py --selftest                                     # no database
"""
import re
import statistics
import sys
from collections import defaultdict

CODE_PREFIXES = ("Codice Penale", "Codice di procedura penale", "Codice Civile", "Codice di procedura civile",
                 "Codice del processo amministrativo", "Codice della strada", "Costituzione")
ARTICLE_NAME = re.compile(r"^(\d+(?:-[a-z]+)?)(?:\.|$)")
LISTED = 600            # candidate numbers printed per document


def article_of(name):
    """'519.0.0' / '519.1' / '519' -> '519'; '609-bis.0.0' -> '609-bis'; a section that is no article -> None."""
    m = ARTICLE_NAME.match(str(name or "").strip())
    return m.group(1) if m else None


def has_words(heading) -> bool:
    return bool(re.search(r"[A-Za-zÀ-ÿ]", str(heading or "")))


def articles(rows) -> dict:
    """rows = dicts name, heading, chars, says (one per section) -> {article: {heading, chars, says}}: the heading of
    the article's first piece, the characters of all its pieces, says = any piece says «abrogat»."""
    out, first = {}, {}
    for r in rows:
        a = article_of(r.get("name"))
        if a is None:
            continue
        e = out.setdefault(a, {"heading": "", "chars": 0, "says": False})
        e["chars"] += int(r.get("chars") or 0)
        e["says"] = e["says"] or bool(r.get("says"))
        if a not in first or str(r["name"]) < first[a]:
            first[a], e["heading"] = str(r["name"]), r.get("heading") or ""
    return out


def number_key(a: str):
    m = re.match(r"(\d+)(?:-([a-z]+))?", a)
    return (int(m.group(1)), m.group(2) or "")


def ranges(numbers) -> str:
    """['519', '520', '521', '530', '609-bis'] -> '519-521, 530, 609-bis' (suffix articles stay single)."""
    out, run = [], []

    def flush():
        if run:
            out.append(str(run[0]) if len(run) == 1 else f"{run[0]}-{run[-1]}")
            run.clear()
    for a in sorted(set(numbers), key=number_key):
        n, suffix = number_key(a)
        if suffix:
            flush()
            out.append(a)
        elif run and n == run[-1] + 1:
            run.append(n)
        else:
            flush()
            run.append(n)
    flush()
    return ", ".join(out)


def report(doc_name: str, rows) -> list:
    arts = articles(rows)
    total = len(arts)
    marked = [a for a, e in arts.items() if e["says"]]
    bare = [a for a, e in arts.items() if not has_words(e["heading"])]
    bare_marked = [a for a in bare if arts[a]["says"]]
    candidates = [a for a in bare if not arts[a]["says"]]
    med = lambda xs: int(statistics.median(arts[a]["chars"] for a in xs)) if xs else 0
    out = [f"{' '.join(str(doc_name).split())[:60]}: {total:,} articles; text says 'abrogat': {len(marked):,}; "
           f"heading without words: {len(bare):,} (text says 'abrogat': {len(bare_marked):,}; does not: {len(candidates):,})",
           f"  median text length: marked {med(marked):,} chars; candidates {med(candidates):,} chars; all {med(list(arts)):,} chars"]
    if total and len(bare) > total / 2:
        out.append("  most headings have no words in this document: no candidate list")
    elif candidates:
        shown = sorted(candidates, key=number_key)[:LISTED]
        out.append(f"  candidates ({len(candidates):,}): {ranges(shown)}" + (" ..." if len(candidates) > LISTED else ""))
    return out


def selftest() -> None:
    assert article_of("519.0.0") == "519" and article_of("609-bis.1") == "609-bis" and article_of("52") == "52"
    assert article_of("Libro II") is None and article_of(None) is None
    assert has_words("Omicidio (1)") and not has_words("(1)") and not has_words("") and not has_words(None)
    assert ranges(["521", "519", "520", "530", "609-bis", "609-ter", "8"]) == "8, 519-521, 530, 609-bis, 609-ter"
    rows = [{"name": "519.1", "heading": "piece two", "chars": 100, "says": False},
            {"name": "519.0.0", "heading": "(1)", "chars": 400, "says": False},          # candidate: 500 chars in all
            {"name": "520.0.0", "heading": "(1)", "chars": 300, "says": False},          # candidate
            {"name": "530.0.0", "heading": " (2) ", "chars": 20, "says": True},          # bare heading, marked
            {"name": "575.0.0", "heading": "Omicidio (1)", "chars": 470, "says": False},
            {"name": "600.0.0", "heading": "Riduzione in schiavitù", "chars": 900, "says": True},
            {"name": "52.0.0", "heading": "Difesa legittima", "chars": 34, "says": False},
            {"name": "61.0.0", "heading": "Circostanze aggravanti comuni", "chars": 800, "says": False},
            {"name": "Titolo XII", "heading": "Dei delitti contro la persona", "chars": 50, "says": False}]
    arts = articles(rows)
    assert set(arts) == {"52", "61", "519", "520", "530", "575", "600"}
    assert arts["519"] == {"heading": "(1)", "chars": 500, "says": False}
    text = "\n".join(report("Codice  Penale 2026", rows))
    assert "Codice Penale 2026: 7 articles; text says 'abrogat': 2; heading without words: 3 (text says 'abrogat': 1; does not: 2)" in text, text
    assert "candidates (2): 519-520" in text and "marked 460 chars; candidates 400 chars" in text, text
    bare = [{"name": str(n), "heading": "", "chars": 10, "says": False} for n in range(1, 6)]
    assert "no candidate list" in "\n".join(report("Costituzione", bare))
    print("selftest OK")


def main() -> int:
    if "--selftest" in sys.argv:
        selftest()
        return 0
    env_path = sys.argv[sys.argv.index("--env") + 1] if "--env" in sys.argv else "/opt/v2tools/health.env"
    env = {}
    with open(env_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    from neo4j import GraphDatabase, READ_ACCESS
    driver = GraphDatabase.driver(env["NEO4J_URI"], auth=(env["NEO4J_USER"], env["NEO4J_PASSWORD"]))
    out = []
    with driver.session(database=env.get("NEO4J_DATABASE", "neo4j"), default_access_mode=READ_ACCESS) as s:
        docs = defaultdict(dict)
        for prefix in CODE_PREFIXES:
            for d in s.run("MATCH (d:Document {document_type:'primary'}) WHERE d.name STARTS WITH $p "
                           "RETURN d.id AS id, d.name AS name", {"p": prefix}).data():
                docs[d["id"]] = d
        for d in sorted(docs.values(), key=lambda d: str(d["name"])):
            rows = s.run("MATCH (d:Document {id: $id})-[:CONTAINS]->(s:Section) WHERE s.name IS NOT NULL "
                         "RETURN s.name AS name, coalesce(s.titolo_articolo, s.title) AS heading, "
                         "size(coalesce(s.plain_text, '')) AS chars, "
                         "toLower(coalesce(s.plain_text, '')) CONTAINS 'abrogat' AS says", {"id": d["id"]}).data()
            out += report(d["name"] or "", rows) + [""]
    driver.close()
    print(f"code documents read: {len(docs)}\n")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
