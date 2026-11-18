"""Read-only: do the Cassazione rulings in the graph carry what the answer needs to show a ruling (Flavio, 17 Nov
2026)? Three things per ruling: an abstract on its sections, CITES to the articles it cites, and the estremi
(Document.document_number, Document.date_enacted; the file name always gives archive, section, year and number).

Per archive and year: rulings, with a number, with a date, with both, number equal to the file name's, date in the
code's year (or the one before), read by the linker (cites_linked_at), with at least one CITES, and the CITES links
in all. Then the sections of
a random sample of rulings per row: without an abstract, abstract under 20 characters, the ingest's error string,
without text, median abstract length, and the same for the sections that cite an article.
Prints counts only: never a name, a number, a date or a text.

    cd /opt/chatbot && sudo venv/bin/python -I /home/sunnit/ruling_completeness.py      # be2, reads health.env
    python ops/be2/ruling_completeness.py --selftest                                     # no database
"""
import random
import re
import statistics
import sys
from collections import Counter, defaultdict

CODE = re.compile(r"(snpen|snciv)[@_ ]s(\w)\w[@_ ]a((?:19|20)\d{2})[@_ ]n(\d+)[@_ ]t([a-z])", re.IGNORECASE)
ERROR_TEXT = "[Errore analisi comma]"
STAMP = "copia non ufficiale"       # the PDF stamp of the source files, found inside quoted passages (17 Nov 2026)
SAMPLE = 200            # rulings per archive and year whose sections are read


def group_of(file_name, name):
    """(archive, year, number) from the Italgiure code in the file name or the display name; None without one."""
    m = CODE.search(file_name or "") or CODE.search(name or "")
    return (m.group(1).lower(), m.group(3), m.group(4).lstrip("0")) if m else None


def filled(value) -> bool:
    return value is not None and str(value).strip() != ""


def number_agrees(stored, number: str) -> bool:
    """The stored document_number holds the file name's number as a whole number ('14172', '014172', 'n. 14172/2025')."""
    return bool(number) and bool(re.search(rf"(?<!\d)0*{re.escape(number)}(?!\d)", str(stored or "")))


def year_fits(stored, year: str) -> bool:
    """The stored date's year is the code's year or the one before (decided late in a year, published the next)."""
    return any(int(y) in (int(year), int(year) - 1) for y in re.findall(r"(?:19|20)\d{2}", str(stored or "")))


def documents_table(docs, cited) -> list:
    """docs = dicts id, f, n, num, date, linked; cited = {ruling id: its CITES links} (only rulings with at least one)."""
    rows, uncoded = defaultdict(Counter), 0
    for d in docs:
        g = group_of(d.get("f"), d.get("n"))
        if not g:
            uncoded += 1
            continue
        c, has_num, has_date = rows[g[:2]], filled(d.get("num")), filled(d.get("date"))
        c["rulings"] += 1
        c["number"] += has_num
        c["date"] += has_date
        c["both"] += has_num and has_date
        c["agrees"] += has_num and number_agrees(d.get("num"), g[2])
        c["year"] += has_date and year_fits(d.get("date"), g[1])
        c["linked"] += bool(d.get("linked"))
        c["cited"] += d.get("id") in cited
        c["links"] += cited.get(d.get("id"), 0)
    keys = ("rulings", "number", "date", "both", "agrees", "year", "linked", "cited", "links")
    out = [f"rulings in the graph: {len(docs):,}; without an Italgiure code in the name (not in the table): {uncoded:,}",
           "  archive year " + "".join(f"{h:>9}" for h in ("rulings", "number", "date", "both", "n = file", "date yr",
                                                             "linker", "w/ CITES", "CITES"))]      # CITES = the links
    for g in sorted(rows):
        out.append(f"  {g[0]:<7} {g[1]} " + "".join(f"{rows[g][k]:>9,}" for k in keys))
    return out


def sample_ids(docs, size: int = SAMPLE) -> dict:
    """{(archive, year): up to `size` ruling ids, picked at random with a fixed seed}."""
    by_group = defaultdict(list)
    for d in docs:
        g = group_of(d.get("f"), d.get("n"))
        if g:
            by_group[g[:2]].append(d["id"])
    rng = random.Random(17)
    return {g: rng.sample(sorted(ids), min(size, len(ids))) for g, ids in by_group.items()}


def sections_table(rows) -> list:
    """rows = dicts g (archive, year), id, a (abstract chars), t (text chars), err, cites, stamp -- one per section."""
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r["g"])].append(r)
    out = ["sections of a random sample of rulings (the abstract is what the answer can show of a ruling):",
           "  archive year " + "".join(f"{h:>9}" for h in ("rulings", "sections", "no abstr", "under 20", "error tx",
                                                             "no text", "median", "citing", "cit w/o", "stamp", "rul w/st"))]
    # cit w/o = citing, no abstract; stamp = sections whose text holds the PDF stamp; rul w/st = rulings with one or more
    for g in sorted(groups):
        rs = groups[g]
        lengths = [r["a"] for r in rs if r["a"]]
        citing = [r for r in rs if r.get("cites")]
        cells = (len({r["id"] for r in rs}), len(rs), sum(not r["a"] for r in rs), sum(0 < r["a"] < 20 for r in rs),
                 sum(bool(r.get("err")) for r in rs), sum(not r["t"] for r in rs),
                 int(statistics.median(lengths)) if lengths else 0, len(citing), sum(not r["a"] for r in citing),
                 sum(bool(r.get("stamp")) for r in rs), len({r["id"] for r in rs if r.get("stamp")}))
        out.append(f"  {g[0]:<7} {g[1]} " + "".join(f"{c:>9,}" for c in cells))
    return out


def selftest() -> None:
    docs = [{"id": "a", "f": "snpen@s10@a2025@n00012@tS.clean.pdf", "n": "[PENALE] x", "num": "12", "date": "2025-03-01", "linked": True},
            {"id": "b", "f": "snpen@s10@a2025@n00013@tS.clean.pdf", "n": "[PENALE] y", "num": "9913/2024", "date": None, "linked": False},
            {"id": "c", "f": "snciv_s50_a2025_n00100_tO.clean.pdf", "n": "[CIVILE] z", "num": None, "date": "12/01/2025", "linked": True},
            {"id": "d", "f": None, "n": "[CIVILE] snciv s50 a2025 n00101 tO.clean", "num": "n. 101/2025", "date": "2024-12-20", "linked": True},
            {"id": "e", "f": "x.pdf", "n": "Sentenza Rossi contro Bianchi", "num": "7", "date": "2019-01-01", "linked": False}]
    text = "\n".join(documents_table(docs, {"a": 3, "d": 1}))
    assert "rulings in the graph: 5; without an Italgiure code in the name (not in the table): 1" in text
    #                         rulings number date both agrees year linked cited links
    assert re.search(r"snciv   2025\s+2\s+1\s+2\s+1\s+1\s+2\s+2\s+1\s+1$", text, re.M), text
    assert re.search(r"snpen   2025\s+2\s+2\s+1\s+1\s+1\s+1\s+1\s+1\s+3$", text, re.M), text
    assert not number_agrees("9913/2024", "13") and number_agrees("014172", "14172") and not number_agrees("114172", "14172")
    assert year_fits("2024-12-20", "2025") and not year_fits("2023-05-01", "2025") and not year_fits(None, "2025")
    picked = sample_ids(docs, 1)
    assert set(picked) == {("snpen", "2025"), ("snciv", "2025")} and all(len(v) == 1 for v in picked.values())
    rows = [{"g": ("snpen", "2025"), "id": "a", "a": 400, "t": 3000, "err": False, "cites": True, "stamp": True},
            {"g": ("snpen", "2025"), "id": "a", "a": 0, "t": 2000, "err": False, "cites": True, "stamp": True},
            {"g": ("snpen", "2025"), "id": "b", "a": 12, "t": 0, "err": False, "cites": False, "stamp": False},
            {"g": ("snpen", "2025"), "id": "b", "a": 600, "t": 900, "err": True, "cites": False, "stamp": True}]
    text2 = "\n".join(sections_table(rows))
    #                        rulings sections no_abstr under20 error no_text median citing citing_no_abstr stamp rulings_with_stamp
    assert re.search(r"snpen   2025\s+2\s+4\s+1\s+1\s+1\s+1\s+400\s+2\s+1\s+3\s+2$", text2, re.M), text2
    assert not any(w in text + text2 for w in ("Rossi", "Bianchi", "9913", "2024-12-20"))    # counts only
    print("selftest OK")


def main() -> int:
    if "--selftest" in sys.argv:
        selftest()
        return 0
    env_path = sys.argv[sys.argv.index("--env") + 1] if "--env" in sys.argv else "/opt/v2tools/health.env"
    size = int(sys.argv[sys.argv.index("--sample") + 1]) if "--sample" in sys.argv else SAMPLE
    env = {}
    with open(env_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    from neo4j import GraphDatabase, READ_ACCESS
    driver = GraphDatabase.driver(env["NEO4J_URI"], auth=(env["NEO4J_USER"], env["NEO4J_PASSWORD"]))
    with driver.session(database=env.get("NEO4J_DATABASE", "neo4j"), default_access_mode=READ_ACCESS) as s:
        docs = s.run("MATCH (d:Document {document_type:'interpretation'}) RETURN d.id AS id, d.file_name AS f, "
                     "d.name AS n, d.document_number AS num, d.date_enacted AS date, "
                     "d.cites_linked_at IS NOT NULL AS linked").data()
        cited = {r["id"]: r["n"] for r in s.run("MATCH (d:Document {document_type:'interpretation'})-[:CONTAINS]->"
                                                "(:Section)-[c:CITES]->() RETURN d.id AS id, count(c) AS n").data()}
        rows = []
        for g, ids in sample_ids(docs, size).items():
            for r in s.run("MATCH (d:Document) WHERE d.id IN $ids MATCH (d)-[:CONTAINS]->(s:Section) "
                           "RETURN d.id AS id, size(trim(coalesce(s.abstract, ''))) AS a, "
                           "size(trim(coalesce(s.plain_text, ''))) AS t, coalesce(s.abstract, '') CONTAINS $err AS err, "
                           "EXISTS { (s)-[:CITES]->() } AS cites, "
                           "coalesce(s.plain_text, '') CONTAINS $stamp AS stamp", ids=ids, err=ERROR_TEXT, stamp=STAMP).data():
                rows.append({"g": g, **r})
    driver.close()
    print("\n".join(documents_table(docs, cited) + [""] + sections_table(rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
