#!/usr/bin/env python3
"""v79.9 (B4): the special laws the institution modules expect (data/corpus_expected.json) that the graph's primary
documents do not hold -- READ-ONLY. The list goes to the ingestion side; it is the same data the parere's fixed line
reads (src/rag/corpus_laws.py).

    cd /opt/chatbot && venv/bin/python scripts/corpus_gaps.py --env .env [--json]
    python scripts/corpus_gaps.py --selftest                        # no database
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def report(result) -> str:
    from src.rag import corpus_laws
    exp = corpus_laws.expected()
    lines = [f"primary document names read: {result['names_read']}", "", "| module | expected | missing |",
             "|---|---|---|"]
    for mod, laws in (exp.get("modules") or {}).items():
        miss = result["modules"].get(mod) or []
        lines.append(f"| {mod} | {', '.join(laws)} | {', '.join(miss) or '—'} |")
    lines.append(f"| (defaults) | {', '.join(exp.get('defaults') or [])} | {', '.join(result['defaults']) or '—'} |")
    every = sorted({x for v in result["modules"].values() for x in v} | set(result["defaults"]))
    lines += ["", "Missing, for ingestion: " + (", ".join(every) if every else "none")]
    return "\n".join(lines)


def selftest() -> int:
    from src.rag import corpus_laws
    names = ["Legge 15 luglio 1966, n. 604", "Codice del consumo", "Decreto legislativo 4 marzo 2015, n. 23"]
    got = corpus_laws.gaps(names)
    assert "L. 604/1966" not in got["modules"]["licenziamento"]
    assert "d.lgs. 23/2015" not in got["modules"]["licenziamento"]
    assert "L. 300/1970" in got["modules"]["licenziamento"] and "d.lgs. 81/2015" in got["modules"]["licenziamento"]
    assert got["modules"]["consumo"] == [] and "L. 24/2017" in got["modules"]["responsabilita_sanitaria"]
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
    got = corpus_laws.gaps(names)
    print(json.dumps(got, ensure_ascii=False, indent=2) if args.json else report(got))
    return 0


if __name__ == "__main__":
    sys.exit(main())
