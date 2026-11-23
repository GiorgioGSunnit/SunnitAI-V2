"""Read-only: are the Cassazione rulings our answers cite really in the database, and are the passages quoted from them
really in their text?

Input: a JSON list made on the PC from the answers of a test run, one item per ruling cited:
    {"branch": "pen" | "civ", "section": "2", "number": "34225", "year": "2025",
     "where": ["case run", ...], "passages": ["the passage as the answer prints it", ...]}
For each ruling: is there a document with that archive, year and number; does the section the answer prints agree with
the file's; is each passage in the ruling's text -- word for word once punctuation and spacing are removed, else the
share of its six-word runs found there (the answer drops page stamps and masks names, so a true passage can differ).

Prints counts and the rulings' public references (court, section, number, year). Never a passage, never a ruling's text.

    cd /opt/chatbot && sudo venv/bin/python -I /home/sunnit/verify_cited_rulings.py /home/sunnit/cited_rulings.json
    python ops/be2/verify_cited_rulings.py --selftest                                    # no database
"""
import json
import re
import sys
from collections import Counter, defaultdict

CODE = re.compile(r"(snpen|snciv)[@_ ]s(\w\w)[@_ ]a((?:19|20)\d{2})[@_ ]n(\d+)[@_ ]t([a-z])", re.IGNORECASE)
RUN, NEAR = 6, 0.8           # words in a run; share of a passage's runs that makes it "nearly word for word"


def words(text: str) -> list:
    return re.findall(r"[a-zà-ù0-9]+", (text or "").lower())


def where_found(passage: str, text_words: list) -> tuple:
    """("whole" | "nearly" | "no" | "empty", share of the passage's RUN-word runs that are in the text)."""
    w = words(passage)
    if not w:
        return "empty", 0.0
    if " " + " ".join(w) + " " in " " + " ".join(text_words) + " ":
        return "whole", 1.0
    grams = {tuple(text_words[i:i + RUN]) for i in range(len(text_words) - RUN + 1)}
    runs = [tuple(w[i:i + RUN]) for i in range(max(1, len(w) - RUN + 1))]
    share = sum(r in grams for r in runs) / len(runs)
    return ("nearly" if share >= NEAR else "no"), share


def key_of(file_name, name):
    """(branch, year, number) and the section code, from the Italgiure code of the file name or the display name."""
    m = CODE.search(file_name or "") or CODE.search(name or "")
    if not m:
        return None, None
    return ("pen" if m.group(1).lower() == "snpen" else "civ", m.group(3), str(int(m.group(4)))), m.group(2).upper()


def label(item: dict) -> str:
    return f"Cass. {item['branch']}., Sez. {item['section']}, n. {item['number']}/{item['year']}"


def kind_of(file_name, name) -> str:
    """The kind of decision, from the file's code: the answer code ranks judgments first, and orders and the penal
    filter section last (rulings.DEMOTED_ACTS, DEMOTED_SECTIONS)."""
    m = CODE.search(file_name or "") or CODE.search(name or "")
    if not m:
        return "no code"
    branch, sec, act = ("pen" if m.group(1).lower() == "snpen" else "civ"), m.group(2).lower(), m.group(5).upper()
    if sec.startswith("u"):
        return branch + " Joint Sections"
    if branch == "pen" and sec == "70":
        return "pen Section VII"
    if act in ("D", "I"):
        return branch + " decrees, interlocutory"
    return branch + (" judgments" if act == "S" else " orders") + ", ordinary sections"


def check(items: list, docs: list, texts: dict) -> list:
    """The report's lines. docs: [{id, f, n}]; texts: {doc id: [section texts]}."""
    held = defaultdict(list)                                  # (branch, year, number) -> [(doc id, section code, kind)]
    all_kinds = Counter()                                     # (year, kind) over the whole database
    for d in docs:
        key, section = key_of(d.get("f"), d.get("n"))
        if key:
            kind = kind_of(d.get("f"), d.get("n"))
            held[key].append((d["id"], section, kind))
            all_kinds[(key[1], kind)] += 1
    out, missing, other_section, counts, not_found, nearly, used_kinds = [], [], [], Counter(), [], [], Counter()
    for it in items:
        mine = held.get((it["branch"], it["year"], str(int(it["number"]))))
        if not mine:
            missing.append(it)
            continue
        used_kinds[(it["year"], mine[0][2])] += 1
        if not any(sec[:1] == str(it["section"]).strip().upper()[:1] for _, sec, _ in mine):
            other_section.append((it, sorted({sec for _, sec, _ in mine})))
        text_words = words("\n".join(t for doc, _, _ in mine for t in texts.get(doc, [])))
        for i, p in enumerate(it.get("passages") or [], 1):
            how, share = where_found(p, text_words)
            counts[how] += 1
            if how == "no":
                not_found.append((it, i, len(p), share))
            elif how == "nearly":
                nearly.append((it, i, len(p), share))
    n = len(items)
    out.append(f"rulings cited: {n} (" + ", ".join(f"{b} {c}" for b, c in sorted(Counter(i['branch'] for i in items).items()))
               + f"; by year {dict(sorted(Counter(i['year'] for i in items).items()))})")
    out.append(f"in the database with that archive, year and number: {n - len(missing)}; NOT in the database: {len(missing)}")
    for it in missing:
        out.append(f"   not held: {label(it)}   cited in: {', '.join(it.get('where') or [])}")
    out.append(f"section printed agrees with the file's: {n - len(missing) - len(other_section)}; differs: {len(other_section)}")
    for it, secs in other_section:
        out.append(f"   {label(it)}: the file's section code is {', '.join(secs)}   cited in: {', '.join(it.get('where') or [])}")
    total = sum(counts.values())
    out.append(f"passages quoted from those rulings: {total}; in the ruling's text word for word: {counts['whole']}; "
               f"nearly ({NEAR:.0%} or more of the {RUN}-word runs): {counts['nearly']}; NOT found: {counts['no']}"
               + (f"; empty: {counts['empty']}" if counts["empty"] else ""))
    for it, i, size, share in nearly:
        out.append(f"   nearly: {label(it)}, passage {i} ({size} chars): {share:.0%} of its runs")
    for it, i, size, share in not_found:
        out.append(f"   NOT found: {label(it)}, passage {i} ({size} chars): {share:.0%} of its runs   cited in: {', '.join(it.get('where') or [])}")
    out.append("kind of decision, per year: rulings the answers used / rulings of that kind in the database")
    for year in sorted({y for y, _ in used_kinds}):
        total_used = sum(n for (y, _), n in used_kinds.items() if y == year)
        total_held = sum(n for (y, _), n in all_kinds.items() if y == year)
        for kind in sorted({k for y, k in all_kinds if y == year}):
            u, h = used_kinds[(year, kind)], all_kinds[(year, kind)]
            out.append(f"   {year}  {kind:<34}{u:>5} used ({u / total_used:>4.0%})   {h:>7,} held ({h / total_held:>4.0%})")
    return out


def selftest() -> None:
    text = "La Corte osserva che il reato di rapina impropria si consuma quando la violenza segue immediatamente la sottrazione. Rigetta."
    tw = words(text)
    assert where_found("il reato di rapina impropria si consuma quando la violenza segue", tw) == ("whole", 1.0)
    assert where_found("IL REATO di rapina impropria, si consuma: quando la violenza segue", tw)[0] == "whole"      # punctuation, case
    assert where_found("si consuma quando la violenza [...] segue immediatamente la sottrazione", tw)[0] == "whole"   # a mark, no word dropped
    how, share = where_found("che il reato di rapina impropria si consuma quando [...] segue immediatamente la sottrazione", tw)
    assert how == "no" and 0 < share < NEAR, (how, share)      # short passage, two words dropped: half of its runs broken
    long_text = " ".join(f"parola{i}" for i in range(80))
    edited = " ".join(f"parola{i}" for i in range(10, 70) if i != 40)
    assert where_found(edited, words(long_text))[0] == "nearly"
    assert where_found("un principio del tutto diverso che la sentenza non contiene affatto", tw)[0] == "no"
    assert where_found("rapina", words("rapinatore")) == ("no", 0.0)                 # whole words only
    assert key_of("snpen@s20@a2025@n34225@tS.clean.pdf", None) == (("pen", "2025", "34225"), "20")
    assert key_of(None, "snciv_sL0_a2026_n00012_tO") == (("civ", "2026", "12"), "L0")
    docs = [{"id": "a", "f": "snpen@s20@a2025@n34225@tS.clean.pdf", "n": None}, {"id": "b", "f": "snciv@sL0@a2026@n00012@tO.pdf", "n": None}]
    items = [{"branch": "pen", "section": "2", "number": "34225", "year": "2025", "where": ["x A"], "passages": ["il reato di rapina impropria si consuma quando la violenza segue", "un principio del tutto diverso che la sentenza non contiene affatto"]},
             {"branch": "civ", "section": "3", "number": "12", "year": "2026", "where": ["y B"], "passages": []},
             {"branch": "pen", "section": "5", "number": "1", "year": "2026", "where": ["z A"], "passages": ["qualcosa"]}]
    rep = "\n".join(check(items, docs, {"a": [text[:60], text[60:]], "b": ["testo"]}))
    assert "NOT in the database: 1" in rep and "not held: Cass. pen., Sez. 5, n. 1/2026" in rep, rep
    assert "differs: 1" in rep and "the file's section code is L0" in rep, rep
    assert "word for word: 1" in rep and "NOT found: 1" in rep, rep
    assert "rapina" not in rep and "principio" not in rep, "the report must not print a passage"
    assert kind_of("snpen@s70@a2025@n00030@tO.clean.pdf", None) == "pen Section VII"
    assert kind_of("snciv@sU0@a2026@n12726@tS", None) == "civ Joint Sections" and kind_of(None, "snciv_s30_a2026_n1_tD") == "civ decrees, interlocutory"
    assert kind_of("snciv@sL0@a2026@n00012@tO.pdf", None) == "civ orders, ordinary sections" and kind_of("x.pdf", None) == "no code"
    assert "2025  pen judgments, ordinary sections      1 used (100%)         1 held (100%)" in rep, rep
    assert "2026  civ orders, ordinary sections         1 used (100%)         1 held (100%)" in rep, rep
    print("selftest OK")


def main() -> int:
    if "--selftest" in sys.argv:
        selftest()
        return 0
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    env_path = sys.argv[sys.argv.index("--env") + 1] if "--env" in sys.argv else "/opt/v2tools/health.env"
    if "--env" in sys.argv:
        args.remove(env_path)
    if len(args) != 1:
        print(__doc__)
        return 1
    with open(args[0], encoding="utf-8") as fh:
        items = json.load(fh)
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
        docs = s.run("MATCH (d:Document {document_type:'interpretation'}) RETURN d.id AS id, d.file_name AS f, d.name AS n").data()
        wanted = {(it["branch"], it["year"], str(int(it["number"]))) for it in items}
        ids = [d["id"] for d in docs if key_of(d.get("f"), d.get("n"))[0] in wanted]
        texts = defaultdict(list)
        for start in range(0, len(ids), 100):
            for row in s.run("MATCH (d:Document)-[:CONTAINS]->(s:Section) WHERE d.id IN $ids "
                             "RETURN d.id AS id, s.plain_text AS t", ids=ids[start:start + 100]).data():
                texts[row["id"]].append(row["t"] or "")
    driver.close()
    print(f"rulings in the database: {len(docs):,}; documents read for this check: {len(ids)}")
    print("\n".join(check(items, docs, texts)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
