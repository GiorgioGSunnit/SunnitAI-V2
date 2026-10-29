"""Hide chosen Documents from every search without deleting them (29 Oct): private test uploads that a superadmin
account sent straight into the shared library, stored as document_type 'primary' with no visibility (so public).

The app's searches filter either by document_type (the strategy path: 'primary', 'special', 'interpretation') or by
visibility (nodes/retrieval.py, source_tiers: coalesce(visibility, 'public') = 'public' OR owner_id = the user). A
quarantined Document gets both: document_type 'quarantine' and visibility 'private' with owner_id 'quarantine', which
no user has. Its old type is kept in quarantine_from_type; --undo puts everything back. Nothing is deleted.

  1. --list FILE   every Document typed primary / special / legislation, one per line (id, kind, matched word, type,
                   name) into FILE on the server, the suspected ones first: to read there, not to paste anywhere
  2. --ids FILE    the ids to quarantine, one per line (# comments allowed): a dry run, counts only
  3. --ids FILE --write            quarantine them
     --undo --write                put back every quarantined Document (or only those in --ids)

    cd /opt/chatbot && venv/bin/python /root/quarantine_docs.py --list /root/law_tier_docs.tsv
"""

import argparse
import datetime as dt
import os
import re
import sys
from typing import List

PRIVATE = re.compile(
    r"timesheet|verbale di sommarie|^procura|contratto di consulenza|ristrutturazione|nota per|"
    r"accordo di (?:collaborazione|riservatezza)|plcgrp|plcimpairment|polgrp|polvalutazione|policypcg|job posting|"
    r"seminar mat|istanza correzione|avviso di pagamento|^documento_|^doc2026|termini commerciali|"
    r"modulo cancellazione|nota eseguito lavori|problemi i transportit|gestione aspettative|rischio reputazionale|"
    r"impairment test|segnalazioni di gruppo|pianificazione e controllo|^a\d{5}-\d{5}\.pdf|^boe-a|^noa\.es|^ue_es|"
    r"^e_\d+|^orden trm|nomina del difensore|locazione",
    re.IGNORECASE)
COURT = re.compile(r"^ordinanza|^sentenza \d+/\d{4}|ecli:it", re.IGNORECASE)
QUARANTINE = "quarantine"


def kind_of(name: str):
    """(kind, the word that decided it) for a Document name."""
    m = PRIVATE.search(name or "")
    if m:
        return "private-looking", m.group(0).lower()
    m = COURT.search(name or "")
    return ("court ruling", m.group(0).lower()) if m else ("other", "")


def read_ids(path: str) -> List[str]:
    """The ids in a file, one per line; blank lines and # comments skipped; a tab-separated line (a copy of a --list
    line) gives its first field."""
    ids = []
    for line in open(path, encoding="utf-8"):
        line = line.split("#", 1)[0].strip()
        if line:
            ids.append(line.split("\t", 1)[0].strip())
    return list(dict.fromkeys(ids))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--list", metavar="FILE")
    p.add_argument("--ids", metavar="FILE")
    p.add_argument("--undo", action="store_true")
    p.add_argument("--write", action="store_true")
    p.add_argument("--reason", default="private test upload (29 Oct 2026)")
    args = p.parse_args()
    if not (args.list or args.ids or args.undo):
        p.error("one of --list, --ids, --undo")

    sys.path.insert(0, os.getcwd())
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.getcwd(), ".env"))
    import neo4j
    from src.rag.main import NEO4J_DATABASE, driver

    mode = neo4j.WRITE_ACCESS if args.write else neo4j.READ_ACCESS
    with driver.session(database=NEO4J_DATABASE, default_access_mode=mode) as session:
        if args.list:
            rows = session.run("MATCH (d:Document) WHERE d.document_type IN ['primary', 'special', 'legislation'] "
                               "RETURN d.id AS id, d.name AS name, d.document_type AS type").data()
            order = {"private-looking": 0, "court ruling": 1, "other": 2}
            lines = sorted(((*kind_of(r["name"]), r) for r in rows), key=lambda x: (order[x[0]], x[2]["name"] or ""))
            with open(args.list, "w", encoding="utf-8") as fh:
                fh.write("# id\tkind\tmatched\ttype\tname -- delete the lines to keep, the rest is quarantined\n")
                for kind, word, r in lines:
                    fh.write(f"{r['id']}\t{kind}\t{word}\t{r['type']}\t{r['name']}\n")
            counts = {k: sum(1 for x in lines if x[0] == k) for k in order}
            print(f"{len(lines)} documents written to {args.list}: " + ", ".join(f"{k} {n}" for k, n in counts.items()))
            return 0

        ids = read_ids(args.ids) if args.ids else None
        if args.undo:
            where = "d.document_type = $q" + (" AND d.id IN $ids" if ids is not None else "")
            n = session.run(f"MATCH (d:Document) WHERE {where} RETURN count(d) AS n", q=QUARANTINE, ids=ids).single()["n"]
            print(f"quarantined documents to put back: {n}" + ("" if args.write else " (dry run: add --write)"))
            if args.write and n:
                done = session.run(
                    f"MATCH (d:Document) WHERE {where} "
                    "SET d.document_type = coalesce(d.quarantine_from_type, 'primary') "
                    "REMOVE d.visibility, d.owner_id, d.quarantine_from_type, d.quarantined_at, d.quarantine_reason "
                    "RETURN count(d) AS n", q=QUARANTINE, ids=ids).single()["n"]
                print(f"put back: {done}")
            return 0

        found = session.run("MATCH (d:Document) WHERE d.id IN $ids RETURN d.document_type AS type, count(d) AS n",
                            ids=ids).data()
        n_found = sum(r["n"] for r in found)
        print(f"ids in the file: {len(ids)}; found: {n_found} (" + ", ".join(f"{r['type']} {r['n']}" for r in found)
              + f"); not found: {len(ids) - n_found}" + ("" if args.write else " -- dry run: add --write"))
        if args.write:
            done = session.run(
                "MATCH (d:Document) WHERE d.id IN $ids AND d.document_type <> $q "
                "SET d.quarantine_from_type = d.document_type, d.document_type = $q, d.visibility = 'private', "
                "d.owner_id = $q, d.quarantined_at = $at, d.quarantine_reason = $reason RETURN count(d) AS n",
                ids=ids, q=QUARANTINE, at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                reason=args.reason).single()["n"]
            print(f"quarantined: {done}  (undo: --undo --write)")
    driver.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
