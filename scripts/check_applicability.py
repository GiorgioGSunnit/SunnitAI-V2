"""Read-only: which articles the strategy (parere) pipeline selects for a case,
and the applicability verdict for each one, with the reason.

Run from the app folder so it uses the app's code and .env (be2 or production):
    cd /opt/chatbot && venv/bin/python scripts/check_applicability.py --facts-file /tmp/sempronio.txt
    ... --request "the user's message, when the case is an uploaded document"
    ... --articles "116 c.p.,589 c.p.,590 c.p."   # check these articles only, skip the selection
    ... --old-prompt                                # also ask 6db1125's original prompt, for comparison

Neo4j sessions are read-only and nothing is written. It calls the chat model
for the case analysis, the article choice and the verdicts, never for an
answer. It ignores APPLICABILITY_CHECK_ENABLED: it always shows the check.
"""

import argparse
import json
import logging
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (ROOT, os.getcwd()):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(os.getcwd(), ".env"))

import neo4j  # noqa: E402

_open_session = neo4j.Driver.session


def _read_only_session(self, **config):
    config["default_access_mode"] = neo4j.READ_ACCESS
    return _open_session(self, **config)


neo4j.Driver.session = _read_only_session   # every session in this process is read-only
# "property does not exist" warnings for the visibility filter (no private documents yet): noise here.
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

# 6db1125's prompt, kept here only to show what it answered on production.
_OLD_PROMPT = (
    "Sei un giurista. Per ogni articolo elencato, leggi il suo testo reale e giudica se si applica "
    "davvero ai fatti del caso (non se è genericamente pertinente all'area del diritto). "
    "Rispondi SOLO con un array JSON, un oggetto per articolo, nessun altro testo:\n"
    '[{"article": "<come indicato>", "applicable": true|false, "reason": "<una frase>"}]'
)


def _label(article):
    return f"{article[0]} {article[1]}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--facts-file", required=True, help="text file with the case")
    parser.add_argument("--request", default="", help="the user's message, if the case is a document")
    parser.add_argument("--articles", help='comma-separated, e.g. "116 c.p.,589 c.p."')
    parser.add_argument("--old-prompt", action="store_true", help="also run 6db1125's original prompt")
    args = parser.parse_args()

    import src.rag.defensive_generation as dg
    from langchain_core.messages import HumanMessage, SystemMessage
    from src.rag.main import NEO4J_DATABASE, driver

    with open(args.facts_file, encoding="utf-8") as f:
        facts = f.read().strip()
    named = dg._refs_in(dg._case_text(facts, args.request))
    print(f"Articles named in the user's text (exempt): {', '.join(map(_label, named)) or 'none'}")

    with driver.session(database=NEO4J_DATABASE) as session:
        if args.articles:
            refs = [r for piece in args.articles.split(",") for r in dg._refs_in("art. " + piece.strip())]
            rows = dg._fetch_articles(session, refs)
            print(f"Checking only: {', '.join(map(_label, refs))}")
        else:
            analysis = dg._analyse_case(facts, "it", args.request)
            print("\nCase analysis questions:")
            for q in analysis.get("questioni", []):
                print(f"  - {q.get('titolo')}  (model's norms: {', '.join(q.get('norme') or []) or '-'})")
            picked = dg._select_articles(session, driver, NEO4J_DATABASE, facts, args.request, analysis)
            keep = picked["keep"]
            print("\nSelection:")
            print(f"  keep, named by the user : {', '.join(map(_label, picked['named'])) or 'none'}")
            print(f"  keep, rulings agree     : {', '.join(_label(k) for k in keep if k not in picked['named']) or 'none'}")
            print(f"  candidates              : {', '.join(map(_label, picked['candidates']))}")
            print(f"  model's choice          : {', '.join(map(_label, picked['chosen'] or []))}")
            print(f"  selected                : {', '.join(map(_label, picked['selected']))}")
            rows = picked["article_rows"]
            crossref = sorted({_label(r['_article']) for r in rows
                               if r.get("_source") == "broaden_article_crossref" and r.get("_article")})
            print(f"  added by cross-reference: {', '.join(crossref) or 'none'}")
            spotted = sorted({_label(r["_article"]) for r in picked.get("spotted") or []})
            print(f"  added by issue spotting : {', '.join(spotted) or 'none (ISSUE_SPOTTING_ENABLED off or nothing found)'}")

    to_check, texts = dg._applicability_candidates(facts, args.request, rows)
    print(f"\nSent to the check: {', '.join(texts) or 'none'}")
    verdicts = dg._applicability_verdicts(facts, args.request, texts)
    if verdicts is None:
        print("The check failed (see the warning above): every article would be kept.")
        verdicts = {}
    print("\nVerdicts (an article is dropped only if 'incompatibile' names a requirement and a fact):")
    for label in texts:
        v = verdicts.get(dg._applicability_key(label)) or {}
        decision = "DROP" if dg._drops_article(v, texts[label]) else "keep"
        if decision == "keep" and str(v.get("verdict", "")).lower() == "incompatibile" and dg._is_limit_rule(texts[label]):
            decision = "keep (limit rule)"
        print(f"  {decision}  {label:12s} {v.get('verdict', 'no verdict'):13s} {v.get('reason', '')}")
        if v.get("requisito") or v.get("fatto"):
            print(f"        requirement: {v.get('requisito', '')} | fact: {v.get('fatto', '')}")

    if args.old_prompt and texts:
        human = (f"Fatti del caso:\n{dg._case_text(facts, args.request)[:2000]}\n\n"
                 "Articoli candidati (testo reale dalla banca dati):\n"
                 + "\n\n".join(f"[{label}]\n{text}" for label, text in texts.items()))
        raw = dg._call_chat([SystemMessage(content=_OLD_PROMPT), HumanMessage(content=human)], max_tokens=700)
        print("\n6db1125's original prompt answered:")
        try:
            for v in json.loads(re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()):
                print(f"  {'keep' if v.get('applicable', True) else 'DROP'}  {v.get('article')}: {v.get('reason')}")
        except Exception:
            print(f"  (unparseable) {raw[:400]}")
    print("\nDone. Nothing was written.")


if __name__ == "__main__":
    main()
