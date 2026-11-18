#!/usr/bin/env python3
"""v79.9 (B4): the special laws the institution modules expect (data/corpus_expected.json) that the graph's primary
documents do not hold -- READ-ONLY. The list goes to the ingestion side; it is the same data the parere's fixed line
reads (src/rag/corpus_laws.py).

v79.10 (A4, archive names): the archive's document names carry no law numbers; each law maps to its exact document
names («archive_names»). The script reads the primary documents' names live from the graph (read-only), prints how many
it read (EXPECTED_NAMES: the user's count of the archive), then per module and for the defaults (and the «codes») each
law -> its archive names -> present / NOT IN ARCHIVE (mapped, no such name read) / UNMAPPED (no archive name: not in
the archive), and the names read that no law, code or special-law table entry maps. Exit 1 when a module's law is not
present (unmapped or not in the archive).

    cd /opt/chatbot && venv/bin/python scripts/corpus_gaps.py --env .env [--json]
    python scripts/corpus_gaps.py --selftest                        # no database
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EXPECTED_NAMES = 87


def _row(key: str, result) -> str:
    from src.rag import corpus_laws
    names = corpus_laws.archive_names(key)
    shown = " · ".join(f"«{n}»" for n in names) or "—"
    return f"| {key} | {shown} | {result['status'].get(key) or corpus_laws.UNMAPPED} |"


def report(result) -> str:
    from src.rag import corpus_laws
    exp = corpus_laws.expected()
    n = result["names_read"]
    lines = [f"primary document names read: {n} (expected {EXPECTED_NAMES})" +
             ("" if n == EXPECTED_NAMES else " -- DIFFERENT COUNT")]
    sections = [(f"module {mod}", laws) for mod, laws in (exp.get("modules") or {}).items()]
    sections += [("defaults", list(exp.get("defaults") or [])), ("codes", list(exp.get("codes") or {}))]
    for title, keys in sections:
        lines += ["", f"## {title}", "", "| law | archive names | status |", "|---|---|---|"]
        lines += [_row(k, result) for k in keys]
    lines += ["", "## archive names no law maps", ""]
    lines += [f"- «{x}»" for x in result["names_unmapped"]] or ["- none"]
    every = sorted({x for v in result["modules"].values() for x in v} | set(result["defaults"]))
    lines += ["", "Missing, for ingestion: " + (", ".join(every) if every else "none")]
    lines.append("Module laws not present: " + (", ".join(result["unmatched_modules"]) or "none"))
    return "\n".join(lines)


def run(names) -> dict:
    from src.rag import corpus_laws, special_laws
    return corpus_laws.gaps(names, special_laws.archive_names())


def exit_code(result) -> int:
    """v79.10 (A4): 1 when a module's law is not present (unmapped or not in the archive), else 0."""
    return 1 if result.get("unmatched_modules") else 0


def selftest() -> int:
    from src.rag import corpus_laws
    names = ["Riforma del Lavoro Fornero 2026", "Codice del Consumo 2026", "Legge sul Procedimento Amministrativo 2026",
             "Legge  sul divorzio   2024", "Legge 15 luglio 1966, n. 604", "Legge 7 agosto 1990, n. 241",
             "Codice Civile 2026", "Codice del processo amministrativo 2025"]
    got = run(names)
    lic = got["modules"]["licenziamento"]
    assert lic == ["L. 604/1966", "L. 300/1970", "d.lgs. 23/2015", "d.lgs. 81/2015"], lic
    assert got["status"]["L. 92/2012"] == corpus_laws.PRESENT and got["status"]["L. 604/1966"] == corpus_laws.UNMAPPED
    assert got["status"]["d.P.R. 380/2001"] == corpus_laws.NOT_IN_ARCHIVE
    assert got["status"]["L. 898/1970"] == corpus_laws.PRESENT                 # whitespace-normalised
    assert got["status"]["c.p.a."] == corpus_laws.PRESENT
    assert got["modules"]["consumo"] == [] and got["modules"]["amministrativo"] == ["d.P.R. 380/2001"]
    assert "L. 24/2017" in got["modules"]["responsabilita_sanitaria"]
    assert got["names_unmapped"] == ["Codice Civile 2026", "Legge 15 luglio 1966, n. 604",
                                     "Legge 7 agosto 1990, n. 241"]
    assert exit_code(got) == 1 and got["names_read"] == len(names)
    assert exit_code(run(names + ["Legge sul divorzio 2024"])) == 1      # the unmapped module laws stay unmatched
    print(report(got))
    print("selftest ok")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", default=".env")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    if args.env and Path(args.env).exists():
        from dotenv import load_dotenv
        load_dotenv(args.env)
    import neo4j
    from src.rag import corpus_laws
    from src.rag.main import NEO4J_DATABASE, driver
    with driver.session(database=NEO4J_DATABASE, default_access_mode=neo4j.READ_ACCESS) as session:
        names = corpus_laws.fetch_names(session)
    got = run(names)
    print(json.dumps(got, ensure_ascii=False, indent=2) if args.json else report(got))
    return exit_code(got)


if __name__ == "__main__":
    sys.exit(main())
