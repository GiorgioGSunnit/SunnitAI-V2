"""Read-only: the documents that may belong on the protected list (PROTECTED_PARAPHRASE_ENABLED,
src/rag/protected_sources.py), to fill PROTECTED_DOCUMENTS_FILE by exact name.

Lists every Document that is not 'special' or 'ccnl' (those are protected by type already) and either
  - has a name of a book, manual or collection (manuale, trattato, commentario, compendio, lineamenti, istituzioni
    di, lezioni di, corso di diritto, "Principî/Principi di diritto", massime / massimario, dispensa, seminario,
    slide, formulario, prontuario), or
  - carries a reproduction notice in its first --scan sections (protected_sources.has_notice)
with its document_type, its number of sections, which rule matched and, for a notice, the notice as written.
'primary' documents are listed too: a mislabelled book is the case the name list is for (a law's notice is only
stripped, never protects it).

Run from the app folder so it uses the app's .env:
    cd /opt/chatbot && venv/bin/python scripts/protected_candidates.py
    cd /opt/chatbot && venv/bin/python scripts/protected_candidates.py --scan 40 --names-only > /tmp/candidates.txt

--names-only prints the names alone, one per line: review them, then keep the ones that are protected works in
the file PROTECTED_DOCUMENTS_FILE names. Every session is read-only; nothing is written.
"""

import argparse
import os
import re
import sys

BOOK_NAME_RE = re.compile(
    r"manuale|trattato|commentari|compendio|lineamenti|istituzioni di|lezioni di|corso di diritto|"
    r"princip[iîì]\s+di\s+diritto|massim|dispens|seminari|slide|formulario|prontuario|codice\s+\w+\s+commentato",
    re.IGNORECASE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scan", type=int, default=20, help="sections read per document for a notice")
    parser.add_argument("--names-only", action="store_true", help="the names alone, one per line")
    args = parser.parse_args()

    sys.path.insert(0, os.getcwd())
    sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
    from audit_ccnl import _read_only_driver
    from src.rag import protected_sources
    driver, database = _read_only_driver()
    found = []
    with driver.session(database=database) as session:
        docs = session.run(
            "MATCH (d:Document) WHERE coalesce(d.document_type, '') <> 'special' "
            "AND coalesce(d.document_type, '') <> 'ccnl' "
            "OPTIONAL MATCH (d)-[:CONTAINS]->(s:Section) "
            "RETURN d.id AS id, d.name AS name, d.document_type AS type, count(s) AS sections").data()
        for doc in docs:
            why = []
            if BOOK_NAME_RE.search(doc["name"] or ""):
                why.append("name")
            texts = session.run(
                "MATCH (d:Document {id: $id})-[:CONTAINS]->(s:Section) "
                "RETURN coalesce(s.plain_text, s.abstract, '') AS text LIMIT $n", id=doc["id"], n=args.scan).data()
            notice = next((t["text"] for t in texts if protected_sources.has_notice(t["text"])), None)
            if notice is not None:
                m = next(p.search(notice) for p in protected_sources._NOTICE_RES if p.search(notice))
                why.append(f"notice: {' '.join(m.group(0).split())[:80]}")
            if why:
                found.append((doc, why))
    driver.close()
    found.sort(key=lambda x: (str(x[0]["type"]), str(x[0]["name"])))
    if args.names_only:
        print("\n".join(doc["name"] for doc, _ in found if doc["name"]))
        return
    names = protected_sources.protected_names()
    print(f"{len(found)} candidates of {len(docs)} documents (not special/ccnl); 'listed' = already in the list\n")
    for doc, why in found:
        listed = "listed " if protected_sources.norm_name(doc["name"]) in names else ""
        print(f"{listed}{str(doc['type']):15} {doc['sections']:>6} sections  {doc['name']}  [{'; '.join(why)}]")


if __name__ == "__main__":
    main()
