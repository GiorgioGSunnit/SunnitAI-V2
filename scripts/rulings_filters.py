#!/usr/bin/env python3
"""v79.7 (item 4): what the passage picker's filters drop in the corpus, READ-ONLY -- per article (default: the penal
nightly cells' charges, 572, 582, 624, 648 c.p.), the rulings that CITE it (src/rag/rulings.fetch, linked only):
how many abstracts miss the rubrica's head noun (rulings.on_rubrica), how many are Sez. 7 or ordinanze; then, over their
citing sections' sentences that hold the head noun (a proxy for the element's nouns: no record here), how many each
sentence filter drops (rulings.sentence_filter, without the element test) -- and, among the sentences dropped for «no
rule verb», the verb forms they use (a word after «è / sono / deve / va / può / si», and the finite forms ending as
Italian third persons do), counted, so the rule-verb list (rulings._RULE_RE) is extended with what the corpus writes,
not with what we guess.

Prints counts and verb forms only, never a sentence.

    cd /opt/chatbot && venv/bin/python scripts/rulings_filters.py --env .env
    ... --articles "572 c.p." "582 c.p." --json-out /root/rulings_filters.json
    python scripts/rulings_filters.py --selftest                                            # no database
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rag import rulings  # noqa: E402

DEFAULT_ARTICLES = ("572 c.p.", "582 c.p.", "624 c.p.", "648 c.p.")
DOCS_QUERY = "MATCH (d:Document) WHERE d.document_type IN ['primary', 'special'] RETURN d.id AS id, d.name AS name"
SECTION_QUERY = ("MATCH (d:Document {id: $doc})-[:CONTAINS]->(s:Section) WHERE s.name IN [$num, $num + '.0.0'] "
                 "RETURN s.name AS num, s.titolo_articolo AS rubrica LIMIT 1")
_AUX_RE = re.compile(r"\b(?:è|sono|sia|siano|deve|devono|va|vanno|può|possono|si|non)\s+([a-zàèéìòù]{4,})", re.I)
_FINITE_RE = re.compile(r"\b([a-zàèéìòù]{4,}(?:a|ano|e|ono|isce|iscono))\b", re.I)
_NOT_VERBS = set("""della delle dello nella nelle nello sulla sulle alla alle dalla dalle quella quelle questa queste
essere stessa stesse prima dopo come anche ovvero oppure sempre mentre perché poiché quale quali tale tali ogni altra
altre parte persona pena cosa causa norma legge corte sentenza condotta fattispecie natura misura forma prova""".split())


def verb_forms(sentence: str) -> List[str]:
    """The verb-like forms of a sentence: «è configurabile» -> «è configurabile», a finite form («integra», «esige»)."""
    out = [f"{m.group(0).split()[0].lower()} {m.group(1).lower()}" for m in _AUX_RE.finditer(sentence or "")]
    out += [w.lower() for w in _FINITE_RE.findall(sentence or "") if w.lower() not in _NOT_VERBS]
    return out


def measure(article: str, rubrica: str, rulings_found: Iterable[Dict[str, Any]], texts: Dict[str, str]
            ) -> Dict[str, Any]:
    head = rulings.head_stem(rubrica)
    c: Counter = Counter()
    verbs: Counter = Counter()
    for g in rulings_found:
        if not g.get("linked"):
            continue
        c["linked"] += 1
        p = rulings.parse_estremi(g.get("file_name") or "")
        if p and rulings.demoted(p):
            c["rank_demoted"] += 1
        if not rulings.on_rubrica(g.get("abstract") or "", rubrica):
            c["rubrica"] += 1
            continue
        for sid in g.get("sections") or []:
            for s in rulings.sentences(texts.get(sid) or ""):
                if head and not any(w.startswith(head) for w in re.findall(r"[a-zàèéìòù]{3,}", s.lower())):
                    continue
                c["sentences_on_head"] += 1
                failed = rulings.sentence_filter(s, [])
                failed = "" if failed == "element_nouns" else failed      # no element here: that test is not run
                c[failed or "passes"] += 1
                if failed == "rule_verb":
                    verbs.update(verb_forms(s))
    return {"article": article, "rubrica": rubrica, "head": head, "counts": dict(c),
            "verbs_in_rejected": verbs.most_common(40)}


def report(rows: List[Dict[str, Any]]) -> str:
    keys = ("linked", "rubrica", "rank_demoted", "sentences_on_head", "length", "mid_reference", "rule_verb",
            "procedural", "passes")
    lines = ["# Passage picker filters on the corpus", "", "| article | head | " + " | ".join(keys) + " |",
             "|---" * (len(keys) + 2) + "|"]
    for r in rows:
        lines.append(f"| {r['article']} | {r['head']}- | " + " | ".join(str(r["counts"].get(k, 0)) for k in keys) + " |")
    for r in rows:
        lines += ["", f"## {r['article']}: verb forms in the sentences dropped for «no rule verb»", ""]
        lines += [f"- {v}: {n}" for v, n in r["verbs_in_rejected"]] or ["- none"]
    return "\n".join(lines) + "\n"


def selftest() -> None:
    found = [{"linked": True, "file_name": "snpen@s50@a2025@n00001@tS.clean.pdf", "abstract": "Maltrattamenti in famiglia",
              "sections": ["S1"]},
             {"linked": True, "file_name": "snpen@s70@a2025@n00002@tO.clean.pdf", "abstract": "Truffa",
              "sections": ["S2"]},
             {"linked": False, "file_name": "x", "abstract": "", "sections": []}]
    texts = {"S1": "I maltrattamenti esigono una serie di atti vessatori ripetuti nel tempo. I maltrattamenti sussistono "
                   "quando la condotta è abituale. Il ricorrente lamenta che i maltrattamenti non sussistano."}
    got = measure("572 c.p.", "Maltrattamenti contro familiari e conviventi", found, texts)
    c = got["counts"]
    assert c["linked"] == 2 and c["rubrica"] == 1 and c["rank_demoted"] == 1 and c["sentences_on_head"] == 3, c
    assert c["passes"] == 1 and c["rule_verb"] == 1 and c["procedural"] == 1, c
    assert ("esigono", 1) in got["verbs_in_rejected"], got["verbs_in_rejected"]
    assert "| 572 c.p. | maltratt- | 2 | 1 | 1 | 3 |" in report([got])
    print("selftest OK")


def _article(session, label: str, docs: Dict[str, str]) -> Tuple[str, str, str]:
    """(document id, section name, rubrica) of the article -- its code's document as repeal_audit.py finds it."""
    num, _, code = label.partition(" ")
    doc = docs.get(code)
    row = session.run(SECTION_QUERY, doc=doc, num=num).data() if doc else []
    return (doc, row[0]["num"], row[0]["rubrica"] or "") if row else ("", "", "")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--env", default=".env")
    ap.add_argument("--articles", nargs="*", default=list(DEFAULT_ARTICLES))
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
    rows = []
    from src.rag import citations
    with driver.session(database=NEO4J_DATABASE, default_access_mode=neo4j.READ_ACCESS) as session:
        docs: Dict[str, str] = {}
        for d in session.run(DOCS_QUERY).data():
            docs.setdefault(citations.code_key(d["name"]), d["id"])
        for label in args.articles:
            doc, num, rubrica = _article(session, label, docs)
            if not doc:
                rows.append({"article": label, "rubrica": "", "head": "", "counts": {"unread": 1},
                             "verbs_in_rejected": []})
                continue
            found = list(rulings.fetch(session, doc, num).values())
            texts = rulings.texts(session, [s for g in found if g.get("linked") for s in g.get("sections") or []])
            rows.append(measure(label, rubrica, found, texts))
    text = report(rows)
    print(text)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rows, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
