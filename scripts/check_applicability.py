"""Read-only: which articles the strategy (parere) pipeline selects for a case,
and the applicability verdict for each one, with the reason.

Run from the app folder so it uses the app's code and .env (be2 or production):
    cd /opt/chatbot && venv/bin/python scripts/check_applicability.py --facts-file /tmp/sempronio.txt
    ... --request "the user's message, when the case is an uploaded document"
    ... --articles "116 c.p.,589 c.p.,590 c.p."   # check these articles only, skip the selection
    ... --old-prompt                                # also ask 6db1125's original prompt, for comparison
    ... --v1 / --v2                                 # force a path regardless of CASE_ANALYSIS_ENABLED

Follows CASE_ANALYSIS_ENABLED from the environment by default, same as the
app (it used to always show the v1 path, ignoring the flag — fixed 17 Oct,
after a prod run with the flag set still showed "keep / candidates / model's
choice", the v1 vocabulary, giving no way to diagnose what v2 actually did).
Pass --v1 or --v2 to check one path regardless of what's set. With v2, also
prints each questione's "istituti" and, for each one, the article it
resolved to (or none) and which of titolo_articolo / title / full-text / the
model's own number found it — see defensive_generation._resolve_institution.

Neo4j sessions are read-only and nothing is written. It calls the chat model
for the case analysis, the article choice (v1) and the verdicts, never for
an answer. It ignores APPLICABILITY_CHECK_ENABLED: it always shows the check.
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


def _istituto_label(i):
    num = f", num. indicativo {i['numero_indicativo']}" if i.get("numero_indicativo") else ""
    return f"{i['nome']!r} ({i['codice']}{num})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--facts-file", required=True, help="text file with the case")
    parser.add_argument("--request", default="", help="the user's message, if the case is a document")
    parser.add_argument("--articles", help='comma-separated, e.g. "116 c.p.,589 c.p."')
    parser.add_argument("--old-prompt", action="store_true", help="also run 6db1125's original prompt")
    path = parser.add_mutually_exclusive_group()
    path.add_argument("--v1", action="store_true", help="force the v1 path, regardless of CASE_ANALYSIS_ENABLED")
    path.add_argument("--v2", action="store_true", help="force the v2 path, regardless of CASE_ANALYSIS_ENABLED")
    args = parser.parse_args()

    import src.rag.defensive_generation as dg
    from src.rag import feature_flags
    from langchain_core.messages import HumanMessage, SystemMessage
    from src.rag.main import NEO4J_DATABASE, driver

    use_v2 = args.v2 or (feature_flags.enabled(dg.CASE_ANALYSIS_FLAG) and not args.v1)
    forced = " (forced)" if (args.v1 or args.v2) else " (from CASE_ANALYSIS_ENABLED)"
    print(f"Path: {'v2' if use_v2 else 'v1'}{forced}")

    with open(args.facts_file, encoding="utf-8") as f:
        facts = f.read().strip()
    named = dg._refs_in(dg._case_text(facts, args.request))
    print(f"Articles named in the user's text (exempt): {', '.join(map(_label, named)) or 'none'}")

    # Captures _resolve_institution's vlog("case_analysis_institution", ...)
    # calls so they can be printed below, in addition to the usual stdout
    # line vlog itself prints when VERBOSE_LOGGING=true (not assumed here).
    institution_log = []
    original_vlog = dg.vlog
    if use_v2:
        def _capture_vlog(event, data=None, duration_ms=None):
            if event == "case_analysis_institution":
                institution_log.append(data)
            original_vlog(event, data, duration_ms)
        dg.vlog = _capture_vlog

    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            if args.articles:
                refs = [r for piece in args.articles.split(",") for r in dg._refs_in("art. " + piece.strip())]
                rows = dg._fetch_articles(session, refs)
                print(f"Checking only: {', '.join(map(_label, refs))}")
            elif use_v2:
                analysis = dg._analyse_case_v2(facts, "it", args.request)
                print("\nCase analysis questions (v2):")
                for q in analysis.get("questioni", []):
                    istituti = ", ".join(_istituto_label(i) for i in q.get("istituti") or []) or "-"
                    print(f"  - {q.get('titolo')} [{q.get('categoria') or '-'}]")
                    print(f"      istituti: {istituti}")
                picked = dg._select_articles_v2(session, driver, NEO4J_DATABASE, facts, args.request, analysis)
                keep = picked["keep"]
                print("\nSelection (v2 = v1's own selection, union institution-resolved):")
                print(f"  keep, named by the user   : {', '.join(map(_label, picked['named'])) or 'none'}")
                print(f"  keep, rulings agree       : "
                     f"{', '.join(_label(k) for k in keep if k not in picked['named']) or 'none'}")
                print(f"  candidates (v1)           : {', '.join(map(_label, picked['candidates']))}")
                print(f"  model's choice (v1)       : {', '.join(map(_label, picked['chosen'] or []))}")
                print(f"  institutions resolved (v2): "
                     f"{', '.join(map(_label, picked.get('institutions_resolved') or [])) or 'none'}")
                print(f"  selected                  : {', '.join(map(_label, picked['selected']))}")
                rows = picked["article_rows"]
                crossref = sorted({_label(r['_article']) for r in rows
                                   if r.get("_source") == "broaden_article_crossref" and r.get("_article")})
                print(f"  added by cross-reference  : {', '.join(crossref) or 'none'}")
                spotted = sorted({_label(r["_article"]) for r in picked.get("spotted") or []})
                print(f"  added by issue spotting   : {', '.join(spotted) or 'none'}")

                print("\nInstitution -> article resolution:")
                for entry in institution_log:
                    resolved = entry.get("resolved") or "NOT FOUND"
                    print(f"  {entry['institution']!r:45s} -> {resolved:22s} ({entry['reason']})")
            else:
                analysis = dg._analyse_case(facts, "it", args.request)
                print("\nCase analysis questions (v1):")
                for q in analysis.get("questioni", []):
                    print(f"  - {q.get('titolo')}  (model's norms: {', '.join(q.get('norme') or []) or '-'})")
                picked = dg._select_articles(session, driver, NEO4J_DATABASE, facts, args.request, analysis)
                keep = picked["keep"]
                print("\nSelection:")
                print(f"  keep, named by the user : {', '.join(map(_label, picked['named'])) or 'none'}")
                print(f"  keep, rulings agree     : "
                     f"{', '.join(_label(k) for k in keep if k not in picked['named']) or 'none'}")
                print(f"  candidates              : {', '.join(map(_label, picked['candidates']))}")
                print(f"  model's choice          : {', '.join(map(_label, picked['chosen'] or []))}")
                print(f"  selected                : {', '.join(map(_label, picked['selected']))}")
                rows = picked["article_rows"]
                crossref = sorted({_label(r['_article']) for r in rows
                                   if r.get("_source") == "broaden_article_crossref" and r.get("_article")})
                print(f"  added by cross-reference: {', '.join(crossref) or 'none'}")
                spotted = sorted({_label(r["_article"]) for r in picked.get("spotted") or []})
                print(f"  added by issue spotting : {', '.join(spotted) or 'none (ISSUE_SPOTTING_ENABLED off or nothing found)'}")
    finally:
        dg.vlog = original_vlog

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
