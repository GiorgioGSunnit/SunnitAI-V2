"""Repairs two losses of the code loader in the main codes, from a list checked article by article
(code_articles_verified_17nov.json, next to this script):

  REPEALED  articles the code no longer has, stored with their old text and no word of the repeal (c.p. 519-526 ...).
            The text of every piece of the article becomes the repeal note («Articolo abrogato dall'art. 1 della
            L. 15 febbraio 1996, n. 66.»), which the app's guard reads (article_text.is_abrogated, live and newer).
            The old text stays on the node in plain_text_before_codefix.
  HEADINGS  articles in force whose heading was lost (stored as a footnote mark, «(1)»). titolo_articolo becomes the
            official heading; the old value stays in titolo_articolo_before_codefix; the heading vector is cleared,
            to be rebuilt by the app's scripts/build_heading_index.py --write.

Articles struck down by the Constitutional Court are listed and NOT touched (the guard does not read that wording).

Without --write nothing is written: the plan is printed, with the number to pass to --expect. --write applies exactly
that plan in one transaction, checks it and commits, or changes nothing. --undo puts the old values back on the nodes
this tool marked (property codefix) and on no other. Only documents named like the four codes are read.

    cd /opt/chatbot && sudo venv/bin/python -I /home/sunnit/fix_code_articles.py                      # the plan
    cd /opt/chatbot && sudo venv/bin/python -I /home/sunnit/fix_code_articles.py --pieces             # the plan, piece by piece
    cd /opt/chatbot && sudo venv/bin/python -I /home/sunnit/fix_code_articles.py --write --expect N   # apply it
    cd /opt/chatbot && sudo venv/bin/python -I /home/sunnit/fix_code_articles.py --undo               # take it back
    python ops/be2/fix_code_articles.py --selftest                                                      # no database
"""
import json
import os
import re
import sys

STAMP = "2026-11-18"
SHORT = {"Codice Penale": "c.p.", "Codice di procedura penale": "c.p.p.", "Codice Civile": "c.c.",
         "Codice di procedura civile": "c.p.c."}
REASON = re.compile(r"^(?:repealed|replaced): (?P<act>.+?), art\. (?P<art>[\w-]+)(?:, comma (?P<comma>\d+))?$")
GUARD = re.compile(r"\b(?:l['’]\s*)?(?:articolo|art\.)\s+(?:è\s+stato\s+|e'\s+stato\s+)?abrogat[oa]\b", re.IGNORECASE)  # the live guard's

Q_DOCS = "MATCH (d:Document {document_type:'primary'}) WHERE d.name STARTS WITH $p RETURN d.id AS id, d.name AS name"
Q_PIECES = ("MATCH (d:Document {id: $id})-[:CONTAINS]->(s:Section) WHERE s.name IS NOT NULL AND "
            "any(n IN $nums WHERE s.name = n OR s.name STARTS WITH n + '.' OR s.name STARTS WITH n + '_') "
            "RETURN elementId(s) AS eid, s.name AS name, s.titolo_articolo AS heading, s.title AS title, "
            "coalesce(s.plain_text, '') AS text, s.codefix AS codefix")
Q_SET_TEXT = ("UNWIND $rows AS row MATCH (s) WHERE elementId(s) = row.eid "
              "SET s.plain_text_before_codefix = coalesce(s.plain_text_before_codefix, s.plain_text), "
              "s.plain_text = row.note, s.codefix = $stamp RETURN count(s) AS n")
NONE = "<none>"         # the backup of a heading the node did not have
Q_SET_HEAD = ("UNWIND $rows AS row MATCH (s) WHERE elementId(s) = row.eid "
              "SET s.titolo_articolo_before_codefix = coalesce(s.titolo_articolo_before_codefix, coalesce(s.titolo_articolo, $none)), "
              "s.titolo_articolo = row.heading, s.heading_embedding = null, s.codefix = $stamp RETURN count(s) AS n")
Q_UNDO_TEXT = ("UNWIND $eids AS eid MATCH (s) WHERE elementId(s) = eid AND s.codefix IS NOT NULL "
               "AND s.plain_text_before_codefix IS NOT NULL "
               "SET s.plain_text = s.plain_text_before_codefix REMOVE s.plain_text_before_codefix RETURN count(s) AS n")
Q_UNDO_HEAD = ("UNWIND $eids AS eid MATCH (s) WHERE elementId(s) = eid AND s.codefix IS NOT NULL "
               "AND s.titolo_articolo_before_codefix IS NOT NULL "
               "SET s.titolo_articolo = CASE WHEN s.titolo_articolo_before_codefix = $none THEN null "
               "ELSE s.titolo_articolo_before_codefix END, s.heading_embedding = null "
               "REMOVE s.titolo_articolo_before_codefix RETURN count(s) AS n")
Q_UNMARK = "UNWIND $eids AS eid MATCH (s) WHERE elementId(s) = eid AND s.codefix IS NOT NULL REMOVE s.codefix RETURN count(s) AS n"


def note_of(reason: str) -> str:
    """'repealed: L. 15 febbraio 1996, n. 66, art. 1' -> «Articolo abrogato dall'art. 1 della L. 15 febbraio 1996, n. 66.»"""
    m = REASON.match(reason)
    assert m, f"repeal reason not understood: {reason}"
    comma = f", comma {m['comma']}," if m["comma"] else ""
    return f"Articolo abrogato dall'art. {m['art']}{comma} {'della' if m['act'].startswith('L.') else 'del'} {m['act']}."


def has_words(value) -> bool:
    return bool(re.search(r"[A-Za-zÀ-ÿ]", str(value or "")))


def same(a, b) -> bool:
    norm = lambda s: re.sub(r"\W+", " ", re.sub(r"\(\d+\)", " ", str(s or "").lower())).strip()
    return norm(a) == norm(b)


def code_document(prefix: str, docs) -> tuple:
    """(the code's own document or None, the names of the others the prefix matched). The code itself is named by the
    prefix alone or with a year / edition («Codice Penale 2026»); «Codice Penale Militare ...» is another document."""
    own = re.compile(rf"^{re.escape(prefix)}(?:\s+(?:Edizione\s+)?\d{{4}})?\s*$", re.IGNORECASE)
    mine = [d for d in docs if own.match(str(d["name"] or ""))]
    others = [str(d["name"]) for d in docs if d not in mine]
    return (mine[0] if len(mine) == 1 else None), others, len(mine)


CODE_WORDS = set("codice penale civile di procedura militare pace guerra edizione e leggi complementari della del navigazione".split())


def shown(name) -> str:
    """A document's name only when every word of it is a word of a code's name: other documents typed as laws may be
    private, and their names are not printed."""
    words = re.findall(r"[^\W\d]+", str(name or "").lower())
    return f"«{' '.join(str(name).split())[:60]}»" if words and all(w in CODE_WORDS for w in words) else "<name not shown>"


def pieces_of(rows, number: str) -> list:
    return sorted((r for r in rows if r["name"] == number or r["name"].startswith((number + ".", number + "_"))),
                  key=lambda r: r["name"])


REFERENCE = {"Codice Penale": "575", "Codice di procedura penale": "384", "Codice Civile": "2043",
             "Codice di procedura civile": "163"}          # a sound article per code, shown with --pieces to compare


def piece_line(p) -> str:
    text = " ".join(p["text"].split())
    head = text[:38] if re.match(r"\W*(?:Cass|Trib|Corte|App)", text) else text[:60]      # a case-law note: its reference only
    return (f"        {p['name']:<22} {len(p['text']):>5} chars  heading «{str(p['heading'] or '')[:28]}»  "
            f"title «{str(p['title'] or '')[:28]}»  «{head}»")


def plan(verified: dict, state: dict, detail: bool = False) -> dict:
    """state = {prefix: {"docs": [...], "rows": [...]}} -> the lines to print, the nodes to change, what blocks a write.
    detail: every piece of the repealed articles and of one sound article per code, one line each."""
    out = {"lines": [], "text": [], "head": [], "blocked": [], "eids": []}
    say = out["lines"].append
    for prefix, short in SHORT.items():
        doc, others, matched = code_document(prefix, state.get(prefix, {}).get("docs") or [])
        rows = state.get(prefix, {}).get("rows") or []
        say(f"== {prefix}: " + (f"document {shown(doc['name'])}" if doc else f"{matched} documents named like the code")
            + (f"; {len(others)} other document(s) with this start of name, not touched: {', '.join(shown(o) for o in others[:8])}"
               if others else ""))
        if not doc:
            out["blocked"].append(f"{prefix}: {matched} documents named like the code (one expected)")
            continue
        if detail:
            say(f"   for comparison, a sound article, {short} {REFERENCE[prefix]}:")
            out["lines"] += [piece_line(p) for p in pieces_of(rows, REFERENCE[prefix])]
        for number, reason in sorted((verified["not_in_force"].get(prefix) or {}).items()):
            ps = pieces_of(rows, number)
            out["eids"] += [p["eid"] for p in ps]
            if reason.startswith("struck down"):
                say(f"   {short} {number}: struck down by the Constitutional Court, {len(ps)} piece(s): NOT touched")
                continue
            if detail:
                out["lines"] += [piece_line(p) for p in ps]
            note = note_of(reason)
            todo = [p for p in ps if p["text"].strip() and p["text"].strip() != note]
            mention = [p["name"] for p in todo if "abrogat" in p["text"].lower()]
            shape = ", ".join(f"{p['name']} {len(p['text'])} chars" for p in ps[:6]) + (" ..." if len(ps) > 6 else "")
            if not ps:
                say(f"   {short} {number}: REPEALED, not found in the document")
            elif mention:
                say(f"   {short} {number}: REPEALED, but the text of {mention} already speaks of a repeal: left alone, to look at")
            elif not todo:
                say(f"   {short} {number}: REPEALED, already says so ({shape})")
            else:
                first = " ".join(todo[0]["text"].split())[:50]
                say(f"   {short} {number}: REPEALED -> text of {len(todo)} of {len(ps)} piece(s) becomes «{note}» ({shape}; "
                    f"now «{first}…», heading «{str(ps[0]['heading'] or '')[:30]}»)")
                out["text"] += [{"eid": p["eid"], "note": note} for p in todo]
        for number, official in sorted((verified["in_force_heading"].get(prefix) or {}).items()):
            ps = pieces_of(rows, number)
            out["eids"] += [p["eid"] for p in ps]
            worded = [p for p in ps if has_words(p["heading"])]
            other = [p for p in worded if not same(p["heading"], official)]
            todo = [p for p in ps if not has_words(p["heading"])]      # every piece of a sound article carries the heading
            titles = sorted({" ".join(str(p["title"]).split())[:70] for p in ps if has_words(p["title"])})
            title_note = (f"; title property «{titles[0]}»" + ("" if same(titles[0], official) else " (DIFFERS from the list)")) if titles else ""
            if not ps:
                say(f"   {short} {number}: HEADING, not found in the document")
            elif other:
                say(f"   {short} {number}: HEADING, already has one, «{str(other[0]['heading'])[:60]}»: left alone, to look at")
            elif not todo:
                say(f"   {short} {number}: HEADING, already «{official[:60]}»")
            else:
                say(f"   {short} {number}: HEADING «{str(todo[0]['heading'] or '')[:20]}» -> «{official}» on {len(todo)} of {len(ps)} piece(s){title_note}")
                out["head"] += [{"eid": p["eid"], "heading": official} for p in todo]
        for number, why in sorted((verified.get("leave_as_is", {}).get(prefix) or {}).items()):
            say(f"   {short} {number}: left as it is ({why[:70]})")
    both = {x["eid"] for x in out["text"]} & {x["eid"] for x in out["head"]}
    assert not both, "a node is in both lists"
    return out


def read_state(runner, verified: dict) -> dict:
    state = {}
    for prefix in SHORT:
        docs = runner.run(Q_DOCS, {"p": prefix}).data()
        doc = code_document(prefix, docs)[0]
        nums = sorted(set(verified["not_in_force"].get(prefix) or {}) | set(verified["in_force_heading"].get(prefix) or {})
                      | {REFERENCE[prefix]})
        rows = runner.run(Q_PIECES, {"id": doc["id"], "nums": nums}).data() if doc and nums else []
        state[prefix] = {"docs": docs, "rows": rows}
    return state


def selftest() -> None:
    assert note_of("repealed: L. 15 febbraio 1996, n. 66, art. 1") == "Articolo abrogato dall'art. 1 della L. 15 febbraio 1996, n. 66."
    assert note_of("replaced: D.Lgs. 6 settembre 2005, n. 206, art. 142") == "Articolo abrogato dall'art. 142 del D.Lgs. 6 settembre 2005, n. 206."
    assert note_of("repealed: D.Lgs. 27 gennaio 2010, n. 39, art. 37, comma 9") == \
        "Articolo abrogato dall'art. 37, comma 9, del D.Lgs. 27 gennaio 2010, n. 39."
    assert all(GUARD.search(note_of(r)) for r in ("repealed: L. 22 maggio 1978, n. 194, art. 22",
                                                   "repealed: D.Lgs. 27 gennaio 2010, n. 39, art. 37, comma 9"))
    assert not GUARD.search("Chiunque, con violenza o minaccia, costringe taluno")
    docs = [{"id": "cp", "name": "Codice Penale 2026"}, {"id": "m", "name": "Codice Penale Militare di Pace"}]
    assert code_document("Codice Penale", docs)[0]["id"] == "cp" and code_document("Codice Penale", docs)[1] == ["Codice Penale Militare di Pace"]
    assert code_document("Codice Penale", docs + [{"id": "x", "name": "Codice Penale 2025"}])[0] is None
    assert same("Delitto tentato (1) (2)", "delitto tentato") and not same("Rapina", "Furto")
    assert shown("Codice Penale 2026") == "«Codice Penale 2026»" and shown("Codice Civile contratto Rossi") == "<name not shown>"
    row = lambda eid, name, heading, text, title=None: {"eid": eid, "name": name, "heading": heading, "title": title, "text": text, "codefix": None}
    rows = [row("a", "519.0.0", "(1)", "Chiunque, con violenza"), row("b", "519.1", None, "Chiunque"), row("c", "519.2", None, ""),
            row("d", "5190.0.0", "Altro", "testo"), row("e", "131-bis.0.0", "(1)", "Nei reati", "Esclusione della punibilità"),
            row("f", "131-bis.1", "", "Nei reati"), row("g", "56.0.0", "Delitto tentato (1)", "Chi compie"),
            row("h", "402.0.0", "(1)", "Chiunque pubblicamente vilipende")]
    verified = {"not_in_force": {"Codice Penale": {"519": "repealed: L. 15 febbraio 1996, n. 66, art. 1",
                                                    "402": "struck down: Corte cost., sent. n. 508/2000"}},
                "in_force_heading": {"Codice Penale": {"131-bis": "Esclusione della punibilità per particolare tenuità del fatto",
                                                        "56": "Delitto tentato", "999": "Non esiste"}}}
    state = {"Codice Penale": {"docs": docs, "rows": rows}, "Codice di procedura penale": {"docs": [], "rows": []},
             "Codice Civile": {"docs": [{"id": "x", "name": "Codice Civile 2026"}], "rows": []},
             "Codice di procedura civile": {"docs": [{"id": "x", "name": "Codice di procedura civile"}], "rows": []}}
    p = plan(verified, state)
    assert [x["eid"] for x in p["text"]] == ["a", "b"], p["text"]                  # not the empty piece, not 5190, not 402
    assert [x["eid"] for x in p["head"]] == ["e", "f"], p["head"]                  # every piece without a heading; not art. 56
    shown_pieces = "\n".join(plan(verified, state, detail=True)["lines"])
    assert "519.1 " in shown_pieces and "«Chiunque»" in shown_pieces and "a sound article, c.p. 575" in shown_pieces, shown_pieces
    text = "\n".join(p["lines"])
    assert "c.p. 402: struck down" in text and "c.p. 56: HEADING, already" in text and "c.p. 999: HEADING, not found" in text, text
    assert "DIFFERS from the list" in text and p["blocked"] == ["Codice di procedura penale: 0 documents named like the code (one expected)"]
    for x in p["text"]:                                                            # after the write the plan is empty
        next(r for r in rows if r["eid"] == x["eid"])["text"] = x["note"]
    for x in p["head"]:
        next(r for r in rows if r["eid"] == x["eid"])["heading"] = x["heading"]
    again = plan(verified, state)
    assert not again["text"] and not again["head"], again["lines"]
    rows[1]["text"] = "Il comma è stato abrogato"                                   # a text that already speaks of a repeal
    assert not plan(verified, state)["text"] and "left alone, to look at" in "\n".join(plan(verified, state)["lines"])
    print("selftest OK")


def main() -> int:
    if "--selftest" in sys.argv:
        selftest()
        return 0
    arg = lambda flag, default: sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default
    here = os.path.dirname(os.path.abspath(__file__))
    with open(arg("--list", os.path.join(here, "code_articles_verified_17nov.json")), encoding="utf-8") as fh:
        verified = json.load(fh)
    env = {}
    with open(arg("--env", "/opt/v2tools/health.env"), encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    write, undo = "--write" in sys.argv, "--undo" in sys.argv
    assert not (write and undo), "--write and --undo together"
    import logging
    logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)     # «property codefix does not exist» before the first write
    from neo4j import GraphDatabase, READ_ACCESS
    driver = GraphDatabase.driver(env["NEO4J_URI"], auth=(env["NEO4J_USER"], env["NEO4J_PASSWORD"]))
    database = env.get("NEO4J_DATABASE", "neo4j")
    try:
        if not (write or undo):
            with driver.session(database=database, default_access_mode=READ_ACCESS) as s:
                p = plan(verified, read_state(s, verified), detail="--pieces" in sys.argv)
            print("\n".join(p["lines"]))
            n = len(p["text"]) + len(p["head"])
            print(f"\nNOTHING WRITTEN. The plan changes {n} nodes: text of {len(p['text'])}, heading of {len(p['head'])}.")
            for b in p["blocked"]:
                print(f"BLOCKED, a write would refuse: {b}")
            if n and not p["blocked"]:
                print(f"to apply exactly this plan: add  --write --expect {n}")
            return 0
        with driver.session(database=database) as s:
            tx = s.begin_transaction()
            try:
                p = plan(verified, read_state(tx, verified))
                if undo:
                    eids = sorted(set(p["eids"]))
                    t = tx.run(Q_UNDO_TEXT, {"eids": eids}).data()[0]["n"]
                    h = tx.run(Q_UNDO_HEAD, {"eids": eids, "none": NONE}).data()[0]["n"]
                    m = tx.run(Q_UNMARK, {"eids": eids}).data()[0]["n"]
                    tx.commit()
                    print(f"UNDONE: text put back on {t} nodes, heading on {h} nodes, mark removed from {m} nodes.")
                    print("The heading vectors of those nodes are cleared: rebuild them with scripts/build_heading_index.py --write.")
                    return 0
                n = len(p["text"]) + len(p["head"])
                expect = int(arg("--expect", "-1"))
                if p["blocked"] or n != expect or not n:
                    tx.rollback()
                    print(f"NOTHING WRITTEN: the plan now changes {n} nodes, --expect says {expect}; blocked: {p['blocked'] or 'no'}."
                          " Run without --write and read the plan again.")
                    return 1
                t = tx.run(Q_SET_TEXT, {"rows": p["text"], "stamp": f"{STAMP} repealed"}).data()[0]["n"] if p["text"] else 0
                h = tx.run(Q_SET_HEAD, {"rows": p["head"], "stamp": f"{STAMP} heading", "none": NONE}).data()[0]["n"] if p["head"] else 0
                left = plan(verified, read_state(tx, verified))
                if (t, h) != (len(p["text"]), len(p["head"])) or left["text"] or left["head"]:
                    tx.rollback()
                    print(f"NOTHING WRITTEN: the check after the write failed (text {t} of {len(p['text'])}, heading {h} of "
                          f"{len(p['head'])}, still to change {len(left['text']) + len(left['head'])}). Rolled back.")
                    return 1
                tx.commit()
            except BaseException:
                tx.rollback()
                raise
        print(f"WRITTEN: text of {t} nodes, heading of {h} nodes. Old values kept on the nodes (…_before_codefix).")
        print("Next: rebuild the heading vectors of the changed nodes with the app's scripts/build_heading_index.py --write.")
        return 0
    finally:
        driver.close()


if __name__ == "__main__":
    sys.exit(main())
