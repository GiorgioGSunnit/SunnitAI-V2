"""Read-only: which Cassazione rulings the graph holds, by archive, year and act type, and how the civil ones are
named. For a new batch: does the graph already hold decrees (tD) and interim orders (tI), and what display name
do the civil rulings carry. Prints counts and code shapes only (digits -> 9), never a title: a name without the
Italgiure code is only counted.

    cd /opt/chatbot && sudo venv/bin/python -I /home/sunnit/ruling_names.py            # be2, reads health.env
    python ops/be2/ruling_names.py --selftest                                        # no database
"""
import re
import sys
from collections import Counter

CODE = re.compile(r"(snpen|snciv)[@_ ]s(\w)\w[@_ ]a((?:19|20)\d{2})[@_ ]n\d+[@_ ]t([a-z])", re.IGNORECASE)
LABEL = re.compile(r"^\s*\[(civile|penale)\]", re.IGNORECASE)
TAIL = re.compile(r"([@_ ]o[a-z])?(\.clean)?(\.pdf)?", re.IGNORECASE)


def summarise(rows) -> list:
    """rows = (file_name, name) of every document. The report's lines."""
    table, naming, shapes = Counter(), Counter(), Counter()
    for file_name, name in rows:
        file_name, name = file_name or "", name or ""
        m = CODE.search(file_name) or CODE.search(name)
        if not m:
            continue
        archive, year, act = m.group(1).lower(), m.group(3), "t" + m.group(4).upper()
        table[(archive, year, act)] += 1
        in_name = CODE.search(name)
        kind = ("label + code" if LABEL.match(name) and in_name else "code, no label" if in_name
                else "label, no code" if LABEL.match(name) else "a title (no label, no code)")
        naming[(archive, year, kind)] += 1
        if in_name and archive == "snciv":                       # only the code's own text: no title can be in it
            label = LABEL.match(name)
            tail = name[in_name.end():]                          # shown only when it is the file's own suffix
            tail = tail if TAIL.fullmatch(tail) else " +more"
            shapes[(label.group(0).strip().upper() + " " if label else "")
                   + re.sub(r"\d+", "9", in_name.group(0)) + tail] += 1
    acts = sorted({k[2] for k in table})
    out = [f"rulings with an Italgiure code: {sum(table.values()):,}",
           "  archive year " + "".join(f"{a:>8}" for a in acts) + "   total"]
    for archive, year in sorted({k[:2] for k in table}):
        out.append(f"  {archive:<7} {year} " + "".join(f"{table[(archive, year, a)]:>8,}" for a in acts)
                   + f"{sum(table[(archive, year, a)] for a in acts):>8,}")
    out.append("how they are named (Document.name):")
    for (archive, year, kind), n in sorted(naming.items()):
        out.append(f"  {archive:<7} {year}  {kind}: {n:,}")
    out.append("civil names that hold the code, most common shapes (digits -> 9):")
    out += [f"  {n:>7,}  {s}" for s, n in shapes.most_common(8)] or ["  none"]
    return out


def selftest() -> None:
    rows = [("snciv_s50_a2026_n01978_tO.clean.pdf", "[CIVILE] snciv_s50_a2026_n01978_tO.clean"),
            ("snciv_s50_a2026_n01979_tO.clean.pdf", "[CIVILE] snciv_s50_a2026_n01979_tO Rossi contro Bianchi"),
            ("snciv_sU0_a2026_n12726_tD.clean.pdf", "Ordinanza sul ricorso di Mario Rossi"),
            ("snpen@s10@a2025@n00001@tS.clean.pdf", "[PENALE] snpen s10 a2025 n00001 tS.clean"),
            ("contratto.pdf", "Contratto Rossi"), (None, None)]
    text = "\n".join(summarise(rows))
    assert "rulings with an Italgiure code: 4" in text
    assert "snciv   2026  a title (no label, no code): 1" in text and "snciv   2026  label + code: 2" in text
    assert "snpen   2025  label + code: 1" in text
    assert "1  [CIVILE] snciv_s9_a9_n9_tO.clean\n" in text + "\n" and "1  [CIVILE] snciv_s9_a9_n9_tO +more" in text
    assert not any(w in text for w in ("Rossi", "Bianchi", "Contratto", "ricorso"))       # no title is printed
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
        rows = [(r["f"], r["n"]) for r in s.run("MATCH (d:Document) RETURN d.file_name AS f, d.name AS n")]
    driver.close()
    print("\n".join(summarise(rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
