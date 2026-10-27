"""Relations agent v0 (28 Oct): builds the Codice Civile article relations into a FILE the app can read behind
ARTICLE_RELATIONS_ENABLED (src/rag/article_relations.py). Never writes to the graph: it only reads it.

From the data we already have (no model call):
  - the civil ruling cards' applied articles      -> articles applied together
  - Flavio's CITES edges (code c.c.)              -> articles cited together
  - the articles' own texts                       -> an article naming another
  fused by reciprocal rank into each article's top related articles (scripts/exp_relations.py measured them: 43% of
  a case's other expected articles in an expected article's top 10, against 29% for the neighbouring numbers);
  plus each article's profiles: the keywords of the rulings applying it, and its own text and rubrica.

Dry run by default: prints what it would write (counts, and the related articles of a few articles). --write writes
--out (atomically: a temporary file, then a rename, so the app never reads half a file).

    cd /opt/chatbot && venv/bin/python /tmp/relations_v0/newcode/scripts/maint_relations.py \\
        --cards /root/ruling_cards/cards_civ_v1.jsonl --out /root/relations/relations_cc.json --write
"""

import argparse
import datetime as dt
import importlib.util
import json
import os
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, List, Set

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
_spec = importlib.util.spec_from_file_location("exp_relations", os.path.join(_HERE, "exp_relations.py"))
rel = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rel)

from src.rag import article_relations as ar  # noqa: E402

TOP_RELATED = 15
VERSION = 1


def build(cards: List[dict], cited: Dict[str, Set[str]], texts: Dict[str, str], headings: Dict[str, str],
          generic_share: float = rel.GENERIC_SHARE, top: int = TOP_RELATED) -> dict:
    """The file's content, from the civil cards, {ruling: c.c. articles cited} and the code's texts and rubriche."""
    applied = [rel.card_articles(c) for c in cards]
    with_cc = [a for a in applied if a]
    freq = Counter(a for g in with_cc for a in g)
    generic = {a for a, k in freq.items() if k > generic_share * max(len(with_cc), 1)} | rel.BOILERPLATE
    co_applied = rel.CoRelation(with_cc, exclude=generic)
    co_cited = rel.CoRelation(cited.values(), exclude=generic)
    refers = rel.RefersTo({a: rel.text_refs(t, a) for a, t in texts.items()}, set(texts)) if texts else None
    articles = set(co_applied.single) | set(co_cited.single) | (set(refers.out) if refers else set())
    related, via = {}, {}
    for a in sorted(articles, key=lambda x: (rel.number_of(x), x)):
        lists = {"applied": co_applied.neighbours(a, "cosine"), "cited": co_cited.neighbours(a, "cosine"),
                 "text": refers.neighbours(a) if refers else []}
        fused = rel.fuse([v for v in lists.values() if v])[:top]
        if not fused:
            continue
        related[a] = [[b, round(s, 5)] for b, s in fused]
        via[a] = {b: ",".join(name for name, v in lists.items() if b in {x for x, _ in v}) for b, _ in fused}
    ruling_docs: Dict[str, Counter] = defaultdict(Counter)
    for card, arts in zip(cards, applied):
        words = " ".join(str(k) for k in (card.get("argomento_decisivo") or []) + (card.get("parole_chiave_questioni") or [])
                         + (card.get("parole_chiave_fatti") or []))
        for a in arts - generic:
            ruling_docs[a].update(ar.stems(words))
    profiles = {"rulings": ar.profile_vectors(ruling_docs)}
    if texts:
        profiles["text"] = ar.profile_vectors({a: Counter(ar.stems(f"{headings.get(a, '')} {headings.get(a, '')} {t}"))
                                               for a, t in texts.items()})
    return {"version": VERSION, "code": ar.CODE, "built_at": dt.datetime.now().isoformat(timespec="seconds"),
            "sources": {"cards": len(cards), "cards_with_cc": len(with_cc), "rulings_citing_cc": len(cited),
                        "articles_with_text": len(texts)},
            "generic": sorted(generic, key=rel.number_of), "related": related, "via": via, "profiles": profiles}


def write_atomically(data: dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, path)


def main() -> int:
    p = argparse.ArgumentParser(description="Relations agent v0: the c.c. article relations file (graph read-only)")
    p.add_argument("--cards", default="/root/ruling_cards/cards_civ_v1.jsonl")
    p.add_argument("--env", default=os.path.join(os.getcwd(), ".env"))
    p.add_argument("--out", default="/root/relations/relations_cc.json")
    p.add_argument("--write", action="store_true", help="write --out (without it: a dry run, nothing written)")
    p.add_argument("--show", default="1490,2051,1218,2043,1453,1117")
    args = p.parse_args()
    t0 = time.time()
    from dotenv import load_dotenv
    from neo4j import GraphDatabase, READ_ACCESS
    load_dotenv(args.env)
    cards = rel.read_cards(args.cards)
    driver = GraphDatabase.driver(os.getenv("NEO4J_URI"), auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")))
    try:
        with driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j"), default_access_mode=READ_ACCESS) as session:
            texts, headings = rel.code_articles(session)
            cited = rel.cites_groups(session)
    finally:
        driver.close()
    data = build(cards, cited, texts, headings)
    size = len(json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    print(f"sources: {data['sources']}")
    print(f"articles with related articles: {len(data['related']):,}; pairs: "
          f"{sum(len(v) for v in data['related'].values()):,}; profiles: "
          + ", ".join(f"{k} {len(v['vectors']):,}" for k, v in data["profiles"].items())
          + f"; file {size / 1e6:.1f} MB ({time.time() - t0:.0f}s)")
    for a in [x for x in (rel.key(s) for s in args.show.split(",")) if x]:
        top = data["related"].get(a, [])[:8]
        print(f"  {a} c.c. -> " + (", ".join(f"{b} ({data['via'][a][b]})" for b, _ in top) or "-"))
    if args.write:
        write_atomically(data, args.out)
        print(f"written: {args.out}")
    else:
        print("dry run: nothing written (add --write)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
