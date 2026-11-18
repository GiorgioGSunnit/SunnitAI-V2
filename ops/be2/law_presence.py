"""Read-only: are these laws in the graph as primary documents? Asked after the 17 Nov 2026 strategy review: no
article of L. 24/2017, d.lgs. 23/2015, L. 300/1970 or L. 604/1966 was among the candidates of two test cases.
Three laws known to be there are listed too, as a control of the search itself.

Prints, per law, the matching document names (only names that match a statute's own title or number) with their
section counts, or NOT FOUND. Other names are never printed: the law tier holds some private documents.

    cd /opt/chatbot && sudo venv/bin/python -I /home/sunnit/law_presence.py      # be2, reads health.env
    python ops/be2/law_presence.py --selftest                                    # no database
"""
import re
import sys

LAWS = (
    ("L. 24/2017 (responsabilità sanitaria, Gelli-Bianco)", r"24/2017|24 del 2017|gelli|responsabilit[aà] sanitaria|esercenti le professioni sanitarie|sicurezza delle cure"),
    ("d.lgs. 23/2015 (tutele crescenti)", r"23/2015|23 del 2015|tutele crescenti|jobs act"),
    ("L. 300/1970 (Statuto dei lavoratori)", r"300/1970|300 del 1970|statuto dei lavoratori"),
    ("L. 604/1966 (licenziamenti individuali)", r"604/1966|604 del 1966|licenziamenti individuali"),
    ("control: L. 241/1990", r"241/1990|procedimento amministrativo"),
    ("control: d.P.R. 309/1990", r"309/1990|stupefacenti"),
    ("control: d.P.R. 380/2001", r"380/2001|materia edilizia"),
)


def report(docs) -> list:
    """docs = (name, file_name, sections) of every primary document. The report's lines."""
    out = [f"primary documents in the graph: {len(docs):,}"]
    for label, pattern in LAWS:
        rx = re.compile(pattern, re.IGNORECASE)
        hits = [(n or f or "", s) for n, f, s in docs if rx.search(f"{n or ''} {f or ''}")]
        if not hits:
            out.append(f"  {label}: NOT FOUND")
        for name, sections in sorted(hits)[:6]:
            out.append(f"  {label}: {' '.join(name.split())[:80]} ({sections:,} sections)")
    labour = sum(bool(re.search(r"lavor", f"{n or ''} {f or ''}", re.IGNORECASE)) for n, f, _ in docs)
    out.append(f"primary documents with 'lavor' in the name: {labour:,} (names not printed)")
    return out


def selftest() -> None:
    docs = [("Legge sul Procedimento Amministrativo 2026", "l241.pdf", 120), ("Statuto dei lavoratori (L. 300/1970)", None, 41),
            ("Contratto di lavoro Rossi Mario", "contratto.pdf", 3), (None, "testo_unico_stupefacenti.pdf", 300)]
    text = "\n".join(report(docs))
    assert "primary documents in the graph: 4" in text
    assert "L. 300/1970 (Statuto dei lavoratori): Statuto dei lavoratori (L. 300/1970) (41 sections)" in text
    assert "L. 24/2017 (responsabilità sanitaria, Gelli-Bianco): NOT FOUND" in text
    assert "control: L. 241/1990: Legge sul Procedimento Amministrativo 2026 (120 sections)" in text
    assert "control: d.P.R. 309/1990: testo_unico_stupefacenti.pdf (300 sections)" in text
    assert "with 'lavor' in the name: 2" in text and "Rossi" not in text       # a private name is counted, never printed
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
    with driver.session(database=env.get("NEO4J_DATABASE", "neo4j"), default_access_mode=READ_ACCESS) as s:
        docs = [(r["n"], r["f"], r["s"]) for r in s.run(
            "MATCH (d:Document {document_type:'primary'}) "
            "RETURN d.name AS n, d.file_name AS f, COUNT { (d)-[:CONTAINS]->(:Section) } AS s")]
    driver.close()
    print("\n".join(report(docs)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
