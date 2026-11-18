"""Read-only, counts only: two ingestion points raised by the 17 Nov 2026 strategy review.

  1. Special laws. The linker writes citation links (CITES) for six codes and the Constitution only. Printed: the
     links per code, and how many rulings name the drugs act (d.P.R. 309/1990) or the building act (d.P.R. 380/2001)
     in their text -- the size of what a link to special laws would add.
  2. Repealed articles of the penal code. The answers applied arts. 519, 521 and 523 (repealed in 1996) as in force.
     Printed, for arts. 519-526 and three in-force controls (52, 575, 609-bis): the stored heading, the length of
     the stored text and whether the text says it is repealed; then how many articles of the code say so at all.

Prints headings of penal-code articles (public law) and counts; never a ruling's name, number or text.

    cd /opt/chatbot && sudo venv/bin/python -I /home/sunnit/ingest_marks.py      # be2, reads health.env
    python ops/be2/ingest_marks.py --selftest                                    # no database
"""
import re
import sys

SPECIAL_LAWS = (      # (label, full-text query: the number written "309/1990" or "309 del 1990")
    ("d.P.R. 309/1990 (drugs act)", '"309 1990" OR "309 del 1990"'),
    ("d.P.R. 380/2001 (building act)", '"380 2001" OR "380 del 2001"'),
)
REPEALED = [str(n) for n in range(519, 527)]          # arts. 519-526 c.p., repealed by L. 66/1996
CONTROLS = ["52", "575", "609-bis"]
CODE_PREFIX = "Codice Penale"


def article_of(name) -> str:
    """'519.0.0' / '519.1' / '519' -> '519'; '609-bis.0.0' -> '609-bis'."""
    return str(name or "").split(".", 1)[0].strip()


def links_lines(rows) -> list:
    """rows = dicts code, n."""
    total = sum(r["n"] for r in rows)
    out = [f"citation links (CITES) by code: {total:,} in all"]
    out += [f"  {str(r['code']):<8} {r['n']:>9,}" for r in sorted(rows, key=lambda r: -r["n"])]
    return out


def special_lines(counts) -> list:
    """counts = [(label, parts, rulings)]."""
    out = ["rulings that name a special law in their text (the linker does not link these):"]
    out += [f"  {label}: {rulings:,} rulings, {parts:,} parts" for label, parts, rulings in counts]
    return out


def articles_lines(doc_name: str, sections: int, rows, marked: int, articles: int) -> list:
    """rows = dicts name, heading, chars, says (the text says 'abrogat'): the first piece of each article is shown."""
    first = {}
    for r in sorted(rows, key=lambda r: str(r["name"])):
        first.setdefault(article_of(r["name"]), r)
    out = [f"{' '.join(doc_name.split())[:60]}: {sections:,} sections; articles whose text says 'abrogat': {marked:,} of {articles:,}",
           f"  {'article':<9}{'stored':>7}{'text chars':>12}{'says repealed':>15}  heading"]
    for group, numbers in (("repealed in 1996", REPEALED), ("in force (control)", CONTROLS)):
        out.append(f"  -- {group}")
        for n in numbers:
            r = first.get(n)
            if not r:
                out.append(f"  {n:<9}{'no':>7}")
                continue
            heading = " ".join(str(r.get("heading") or "").split())[:70]
            out.append(f"  {n:<9}{'yes':>7}{r['chars']:>12,}{('yes' if r['says'] else 'no'):>15}  {heading}")
    return out


def selftest() -> None:
    assert article_of("519.0.0") == "519" and article_of("609-bis.1") == "609-bis" and article_of("52") == "52"
    text = "\n".join(links_lines([{"code": "c.p.", "n": 10}, {"code": "c.c.", "n": 25}]))
    assert "35 in all" in text and text.index("c.c.") < text.index("c.p."), text
    text = "\n".join(special_lines([("d.P.R. 309/1990 (drugs act)", 1200, 300)]))
    assert "d.P.R. 309/1990 (drugs act): 300 rulings, 1,200 parts" in text, text
    rows = [{"name": "519.1", "heading": "second piece", "chars": 5, "says": False},
            {"name": "519.0.0", "heading": "Rapina  carnale", "chars": 412, "says": False},
            {"name": "521.0.0", "heading": None, "chars": 11, "says": True},
            {"name": "52.0.0", "heading": "Difesa legittima", "chars": 900, "says": False}]
    text = "\n".join(articles_lines("Codice Penale 2026", 4000, rows, 37, 800))
    assert "articles whose text says 'abrogat': 37 of 800" in text
    assert re.search(r"519\s+yes\s+412\s+no\s+Rapina carnale$", text, re.M), text          # the first piece, not '519.1'
    assert re.search(r"521\s+yes\s+11\s+yes", text) and re.search(r"520\s+no$", text, re.M), text
    assert re.search(r"52\s+yes\s+900\s+no\s+Difesa legittima$", text, re.M) and re.search(r"575\s+no$", text, re.M), text
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
        out += links_lines(s.run("MATCH ()-[c:CITES]->() RETURN c.code AS code, count(*) AS n").data())
        counts = []
        for label, query in SPECIAL_LAWS:
            r = s.run("CALL db.index.fulltext.queryNodes('section_fulltext', $q) YIELD node "
                      "MATCH (d:Document {document_type:'interpretation'})-[:CONTAINS]->(node) "
                      "RETURN count(DISTINCT node) AS parts, count(DISTINCT d) AS rulings", {"q": query}).single()
            counts.append((label, r["parts"], r["rulings"]))
        out += [""] + special_lines(counts) + [""]
        wanted = REPEALED + CONTROLS
        for d in s.run("MATCH (d:Document {document_type:'primary'}) WHERE d.name STARTS WITH $p "
                       "RETURN d.id AS id, d.name AS name, COUNT { (d)-[:CONTAINS]->(:Section) } AS sections "
                       "ORDER BY sections DESC", p=CODE_PREFIX).data():
            rows = s.run("MATCH (d:Document {id: $id})-[:CONTAINS]->(s:Section) "
                         "WHERE split(s.name, '.')[0] IN $wanted "
                         "RETURN s.name AS name, coalesce(s.titolo_articolo, s.title) AS heading, "
                         "size(coalesce(s.plain_text, '')) AS chars, "
                         "toLower(coalesce(s.plain_text, '')) CONTAINS 'abrogat' AS says", id=d["id"], wanted=wanted).data()
            m = s.run("MATCH (d:Document {id: $id})-[:CONTAINS]->(s:Section) WHERE s.name IS NOT NULL "
                      "WITH split(s.name, '.')[0] AS article, "
                      "max(CASE WHEN toLower(coalesce(s.plain_text, '')) CONTAINS 'abrogat' THEN 1 ELSE 0 END) AS says "
                      "RETURN count(article) AS articles, sum(says) AS marked", id=d["id"]).single()
            out += articles_lines(d["name"] or "", d["sections"], rows, m["marked"], m["articles"]) + [""]
    driver.close()
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
