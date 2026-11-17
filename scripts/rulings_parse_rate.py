#!/usr/bin/env python3
"""v78: the estremi parse rate on a sample of the graph's rulings, READ-ONLY -- to be reported before RULINGS_RECORD renders
anything. Per sampled ruling (Document.document_type 'interpretation', random): the Italgiure code of its file name
(src/rag/rulings.parse_estremi: archive, section code, year, number, act), the chamber the section code maps to
(data/ruling_sections.json, provisional until Antonelo confirms it), the date_enacted read as a date, whether
document_number agrees with the file name's number (never trusted over it), whether the linker stamped it
(cites_linked_at).

Prints counts and shapes only: an unparsed file name is shown as its shape (digits -> 9, every word of four letters or
more -> «w»), never a title.

    cd /opt/chatbot && venv/bin/python scripts/rulings_parse_rate.py --env .env               # 1,000 rulings
    ... --sample 2000 --json-out /root/parse_rate.json
    python scripts/rulings_parse_rate.py --selftest                                         # no database
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rag import rulings  # noqa: E402

SAMPLE_QUERY = ("MATCH (d:Document) WHERE d.document_type = 'interpretation' WITH d, rand() AS r ORDER BY r LIMIT $n "
                "RETURN d.file_name AS file_name, d.name AS name, d.document_number AS document_number, "
                "toString(d.date_enacted) AS date_enacted, d.cites_linked_at IS NOT NULL AS linked")


def shape(text: str) -> str:
    return re.sub(r"[A-Za-zÀ-ÿ]{4,}", "w", re.sub(r"\d", "9", str(text or "")))[:60]


def number_agrees(stored: Any, number: int) -> bool:
    return bool(re.search(rf"(?<!\d)0*{number}(?!\d)", str(stored or "")))


def summarise(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"sample": len(rows), "parsed": 0, "unparsed": 0, "parsed_from_name_only": 0,
                           "archive_year": Counter(), "sections": Counter(), "section_unmapped": Counter(),
                           "acts": Counter(), "dated": 0, "number_agrees": 0, "number_disagrees": 0,
                           "number_missing": 0, "linked": 0, "unparsed_shapes": Counter(),
                           "sections_confirmed": bool(rulings.sections_map().get("confirmed"))}
    for r in rows:
        out["linked"] += bool(r.get("linked"))
        p = rulings.parse_estremi(r.get("file_name") or "")
        if p is None:
            out["unparsed"] += 1
            if rulings.parse_estremi(r.get("name") or ""):
                out["parsed_from_name_only"] += 1
            out["unparsed_shapes"][shape(r.get("file_name") or "(no file_name)")] += 1
            continue
        out["parsed"] += 1
        out["archive_year"][f"{p['archive']} {p['year']}"] += 1
        out["sections"][p["section"]] += 1
        if not rulings.chamber(p):
            out["section_unmapped"][p["section"]] += 1
        out["acts"][p["act"]] += 1
        out["dated"] += rulings.ruling_date(r.get("date_enacted"), p["year"]) is not None
        if r.get("document_number") in (None, ""):
            out["number_missing"] += 1
        elif number_agrees(r["document_number"], p["number"]):
            out["number_agrees"] += 1
        else:
            out["number_disagrees"] += 1
    return out


def report(s: Dict[str, Any]) -> str:
    n = max(1, s["sample"])
    parsed = max(1, s["parsed"])
    pct = lambda a, b: f"{100 * a / b:.1f}%"
    lines = [f"# Estremi parse rate: {s['sample']:,} rulings sampled", "",
             f"- parsed from the file name: {s['parsed']:,} ({pct(s['parsed'], n)}); unparsed {s['unparsed']:,} "
             f"(of which the name holds a code: {s['parsed_from_name_only']:,})",
             f"- stamped by the linker (cites_linked_at): {s['linked']:,} ({pct(s['linked'], n)})",
             f"- date_enacted read as a date: {s['dated']:,} of the parsed ({pct(s['dated'], parsed)}); the others "
             f"print «data n.d.»",
             f"- document_number vs the file name's number: agrees {s['number_agrees']:,}, disagrees "
             f"{s['number_disagrees']:,}, missing {s['number_missing']:,} (the file name is used)",
             f"- section mapping (data/ruling_sections.json): {'confirmed' if s['sections_confirmed'] else 'PROVISIONAL, pending Antonelo'}",
             "", "| section code | rulings | chamber printed |", "|---|---|---|"]
    for code, k in sorted(s["sections"].items(), key=lambda x: -x[1]):
        room = rulings.chamber({"section": code})
        lines.append(f"| {code} | {k:,} | {room or '— (unmapped)'} |")
    lines += ["", "| archive year | rulings |", "|---|---|"]
    lines += [f"| {k} | {v:,} |" for k, v in sorted(s["archive_year"].items())]
    lines += ["", "acts: " + ", ".join(f"t{k} {v:,}" for k, v in sorted(s["acts"].items()))]
    if s["unparsed_shapes"]:
        lines += ["", "unparsed file names, most common shapes:"]
        lines += [f"  {v:>6,}  {k}" for k, v in s["unparsed_shapes"].most_common(10)]
    return "\n".join(lines)


def selftest() -> None:
    rows = [{"file_name": "snpen@s10@a2025@n00001@tS.clean.pdf", "document_number": "1", "date_enacted": "2025-01-09",
             "linked": True},
            {"file_name": "snciv_sU0_a2026_n12726_tD.clean.pdf", "document_number": "12727", "date_enacted": None,
             "linked": True},
            {"file_name": "snciv_s9X_a2026_n00042_tO.pdf", "document_number": None, "date_enacted": "1999-01-01",
             "linked": False},
            {"file_name": "Ordinanza Rossi contro Bianchi.pdf", "name": "[CIVILE] snciv_s30_a2025_n00007_tO",
             "linked": False}]
    s = summarise(rows)
    assert s["parsed"] == 3 and s["unparsed"] == 1 and s["parsed_from_name_only"] == 1
    assert s["number_agrees"] == 1 and s["number_disagrees"] == 1 and s["number_missing"] == 1 and s["dated"] == 1
    assert s["section_unmapped"] == Counter({"9X": 1}) and s["sections"]["U0"] == 1
    text = report(s)
    assert "Rossi" not in text and "Bianchi" not in text and "w w w w.pdf" in text
    assert "| U0 | 1 | Sez. Un. |" in text and "| 9X | 1 | Sez. 9-X |" not in text
    print("selftest OK")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--env", default=".env")
    ap.add_argument("--sample", type=int, default=1000)
    ap.add_argument("--json-out")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        selftest()
        return 0
    if os.path.isfile(args.env):
        from dotenv import load_dotenv
        load_dotenv(args.env)
    import neo4j
    from src.rag.main import NEO4J_DATABASE, driver
    with driver.session(database=NEO4J_DATABASE, default_access_mode=neo4j.READ_ACCESS) as session:
        rows = session.run(SAMPLE_QUERY, n=args.sample).data()
    s = summarise(rows)
    print(report(s))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps({k: dict(v) if isinstance(v, Counter) else v for k, v in s.items()},
                                                  ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
