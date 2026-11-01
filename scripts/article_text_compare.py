"""Read-only: what the new article_text changes, before it is deployed. The running code is "before"; the new files are
loaded from a folder of their own, so nothing in the app changes. From the app folder, with its venv and .env:

    mkdir -p /tmp/at_new            # holding the new src/rag/article_text.py and src/rag/comma_material.py
    venv/bin/python scripts/article_text_compare.py --new /tmp/at_new --out article_text_compare.md

For the four codes: before and after for the reviewed articles; per code, how many articles change and how their lengths
are distributed; how many exceed the per-article prompt budget (_ARTICLE_CHARS) and what comma_material.trim keeps of the
longest; and what to look at: a text that shrank by more than its notes can explain, a note left inside, a colon at the end.
"""

import argparse
import importlib.util
import os
import re
import statistics
import sys
from collections import Counter

CODE_PREFIXES = {"c.c.": "Codice Civile", "c.p.c.": "Codice di procedura civile", "c.p.": "Codice Penale",
                 "c.p.p.": "Codice di procedura penale"}      # defensive_generation._CODE_PREFIXES
ARTICLES = ("61 c.p.", "640 c.p.", "628 c.p.", "589 c.p.", "590 c.p.", "582 c.p.", "624-bis c.p.", "177 c.c.", "179 c.c.",
            "337-ter c.c.", "593 c.p.p.", "113 c.p.c.", "658 c.p.c.")
BUDGET = 1500       # defensive_generation._ARTICLE_CHARS
NAME_RE = re.compile(r"^(\d+(?:-[a-z]+)?)(?:[._]|$)", re.IGNORECASE)
NOTE_LEFT_RE = re.compile(r"\(\d{1,3}\)\s+[A-Z][^.]{0,80}?\bdall['’]\s*art\.")


def load(folder: str, name: str):
    """The file as a module of src.rag, so its relative imports work, under a name of its own."""
    spec = importlib.util.spec_from_file_location(f"src.rag._new_{name}", os.path.join(folder, f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def by_article(rows: list) -> dict:
    out = {}
    for row in rows:
        match = NAME_RE.match(row["s"].get("name") or "")
        if match:
            out.setdefault(match.group(1).lower(), []).append(row)
    return out


def read(module, number: str, rows: list, seen_by_old: bool) -> str:
    """The article's text by that module; the old lookups never fetched "N_j"."""
    if seen_by_old:
        rows = [r for r in rows if r["s"]["name"] == number or r["s"]["name"].startswith(number + ".")]
    found = module.article_rows(number, rows)
    return "\n".join(((r.get("s") or {}).get("plain_text") or "").strip() for r in found).strip()


def flat(text: str) -> str:
    return " ".join((text or "").split())


def ends(text: str) -> str:
    return "..." + flat(text)[-90:]


def distribution(lengths: list) -> str:
    if not lengths:
        return "-"
    ordered = sorted(lengths)
    return (f"median {int(statistics.median(ordered)):,}, p90 {ordered[int(len(ordered) * .9)]:,}, max {ordered[-1]:,}, "
            f"<=1,500: {sum(n <= BUDGET for n in ordered):,}, 1,501-4,000: {sum(BUDGET < n <= 4000 for n in ordered):,}, "
            f">4,000: {sum(n > 4000 for n in ordered):,}")


def compare(session, old, new, trim, split, wanted=ARTICLES) -> list:
    L = ["# article_text: before and after", ""]
    tables, reviewed, oversize, checks = [], [], [], []
    for code, prefix in CODE_PREFIXES.items():
        docs = [d for d in session.run("MATCH (d:Document) WHERE d.document_type = 'primary' AND d.name STARTS WITH $prefix "
                                       "RETURN d.id AS id, d.name AS name ORDER BY d.name", prefix=prefix).data()
                if code != "c.p." or not d["name"].startswith("Codice di procedura penale")]
        for doc in docs:
            rows = [{"d": {"id": doc["id"]}, "s": r["s"], "next": r["next"]} for r in session.run(
                "MATCH (d:Document {id: $doc})-[:CONTAINS]->(s:Section) OPTIONAL MATCH (s)-[:NEXT]->(after:Section) "
                "RETURN s {.name, .type, .plain_text, .titolo_articolo} AS s, after.name AS next", doc=doc["id"]).data()]
            before, after, count = [], [], Counter()
            for number, mine in by_article(rows).items():
                was, now = read(old, number, mine, True), read(new, number, mine, False)
                before.append(len(was))
                after.append(len(now))
                count["articles"] += 1
                count["changed"] += flat(was) != flat(now)
                count["longer"] += len(now) > len(was)
                count["shorter"] += len(now) < len(was)
                count["colon_before"] += was.rstrip().endswith(":")
                count["colon_after"] += now.rstrip().endswith(":")
                count["split_" + split(now)[1]] += 1
                ref = f"{number} {code}"
                if len(now) > BUDGET:
                    oversize.append((len(now), ref, now))
                if not now and was:
                    checks.append(f"art. {ref}: the text is gone ({len(was)} characters before)")
                elif len(now) < len(was) and flat(now)[:60] != flat(was)[:60]:
                    checks.append(f"art. {ref}: shorter and opens differently ({len(was)} -> {len(now)}): {flat(now)[:70]}")
                if NOTE_LEFT_RE.search(now):
                    checks.append(f"art. {ref}: a note may be left: {NOTE_LEFT_RE.search(now).group(0)[:80]}")
                if now.rstrip().endswith(":"):
                    checks.append(f"art. {ref}: still ends at a colon: {ends(now)}")
                if ref in wanted:
                    reviewed.append((wanted.index(ref), ref, was, now))
            tables.append(f"| {doc['name']} | {count['articles']:,} | {count['changed']:,} | {count['longer']:,} | "
                          f"{count['shorter']:,} | {count['colon_before']} → {count['colon_after']} |")
            L_lengths = (f"- **{doc['name']}** before: {distribution(before)}\n"
                         f"- **{doc['name']}** after: {distribution(after)}\n"
                         f"- **{doc['name']}** split into: numbered {count['split_numbered']:,}, paragraphs "
                         f"{count['split_lines']:,} (both get \"(co. K)\"), flat text, sentences, no label "
                         f"{count['split_sentences']:,}")
            tables.append(("lengths", L_lengths))
    rows_table = [t for t in tables if isinstance(t, str)]
    L += ["## Articles that change, per code", "", "| document | articles | changed | longer | shorter | ending at a colon |",
          "|---|---:|---:|---:|---:|---|"] + rows_table + ["", "## Length, in characters", ""]
    L += [t[1] for t in tables if isinstance(t, tuple)]
    oversize.sort(key=lambda x: -x[0])
    L += ["", f"## Over the per-article budget ({BUDGET:,} characters)", "",
          f"{len(oversize):,} articles. In a prompt each is trimmed by comma_material.trim: the opening, then the commi and "
          "list items sharing most words with the issue, gaps marked \"[…]\", never the first 1,500 characters. Longest 10, "
          "as an example of what stays with no issue given:", ""]
    for size, ref, text in oversize[:10]:
        L.append(f"- art. {ref}: {size:,} → {len(trim(text, BUDGET))}: {flat(trim(text, BUDGET))[:160]}")
    L += ["", "## Before and after, the reviewed articles"]
    for _, ref, was, now in sorted(reviewed):
        L += ["", f"### art. {ref}", "", f"Before: {len(was):,} characters, {ends(was)}", "",
              f"After: {len(now):,} characters, {ends(now)}", "", f"> {flat(now)}"]
    L += ["", f"## To look at ({len(checks):,})", ""] + [f"- {c}" for c in checks[:300]]
    if len(checks) > 300:
        L.append(f"- and {len(checks) - 300:,} more")
    return L


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--new", required=True, help="folder holding the new article_text.py and comma_material.py")
    parser.add_argument("--env", default=os.path.join(os.getcwd(), ".env"))
    parser.add_argument("--out", default="article_text_compare.md")
    args = parser.parse_args()
    sys.path.insert(0, os.getcwd())
    from dotenv import load_dotenv
    load_dotenv(args.env)
    uri = os.getenv("NEO4J_URI")
    if not uri:
        sys.exit(f"NEO4J_URI is not set ({args.env})")
    from src.rag import article_text as old
    new, comma = load(args.new, "article_text"), load(args.new, "comma_material")
    import neo4j
    with neo4j.GraphDatabase.driver(uri, auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))) as driver:
        with driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j"),
                            default_access_mode=neo4j.READ_ACCESS) as session:
            lines = compare(session, old, new, comma.trim, comma.article_parts)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"File: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
