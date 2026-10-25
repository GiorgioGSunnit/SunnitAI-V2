"""Dry run of the strategy pipeline on one uploaded document, outside the live service.

Calls run_defensive_pipeline exactly as the chat's document flow does (the document's text,
the user's request, the session language), in this process, with the code of the CURRENT
FOLDER. The same command run from /opt/chatbot (the live code) and from a copy holding new
files shows before and after, with no restart and no effect on users. Neo4j is read through
the pipeline's own read-only sessions; nothing is written (no DOCX, no session, no row).

Made for the overflow fixes of 23 Oct: the upload of 07:25 UTC made the applicability check
fail twice (87-89 articles in one call, "maximum context length").

Prints only status, sizes and timings, never the document's or the answer's text:
  - ok, or the exception's type and the first 160 characters of its message;
  - the applicability check: articles checked, batches, failed batches, unchecked, duration,
    error -- from its vlog events (the live code logs fewer fields);
  - the draft prompt: estimated tokens, budget, trim step, articles dropped to fit;
  - the answer's length and citation count, and the total time.

The document is an upload (--document-id), a local file read as an upload is (--file:
.pdf/.docx/.txt), or an eval case's question sent as the chat sends a message with no
document (--case c1: tests/eval_narrative_cases.json). --save writes the report AND the answer's text to a JSON file, to compare
two runs side by side: for test cases (the PM's of 23 Oct), not users' documents.

Run from the folder whose code you want to test, with the live .env:
    cd /opt/chatbot && venv/bin/python /tmp/dryrun_strategy.py --document-id <uuid>
    cd /tmp/chatbot_new && /opt/chatbot/venv/bin/python /tmp/dryrun_strategy.py \\
        --document-id <uuid> --env /opt/chatbot/.env
"""

import argparse
import json
import os
import sys
import time

DEFAULT_REQUEST = "Delinea una strategia difensiva per il cliente sulla base del documento allegato."


def summarise(events: list, status: str, result, elapsed: float, document_chars: int) -> list:
    """The report lines, from the pipeline's vlog events [(event, data, duration_ms)]. No text."""
    out = [f"document: {document_chars} characters", f"status: {status}", f"total time: {elapsed:.0f}s"]
    for event, data, duration in events:
        data = data or {}
        if event == "case_analysis_retrieval":
            out.append(f"retrieval: {data.get('articles')} article rows, {data.get('rulings')} rulings")
        elif event == "case_analysis_applicability" and isinstance(data.get("checked"), int):
            parts = [f"checked {data['checked']}"]
            for key in ("batches", "failed_batches"):
                if key in data:
                    parts.append(f"{key.replace('_', ' ')} {data[key]}")
            if duration is not None:
                parts.append(f"{duration / 1000:.1f}s")
            if data.get("error"):
                parts.append(f"error: {str(data['error'])[:120]}")
            out.append("applicability call: " + ", ".join(parts))
        elif event == "case_analysis_applicability":
            out.append(f"applicability result: {len(data.get('checked') or [])} judged, "
                       f"{len(data.get('dropped') or [])} dropped, {data.get('unchecked', 0)} unchecked")
        elif event == "case_analysis_ruling_cards":
            chosen = "; ".join(f"{c.get('riferimento')} ({c.get('score')}, {c.get('a_favore_di') or '-'}, "
                               f"{c.get('role') or '-'}, {'+'.join(c.get('anchors') or []) or '-'})"
                               for c in data.get("chosen") or []) or "none"
            out.append(f"ruling cards: {data.get('cards')} loaded, {data.get('central')} central articles, "
                       f"chosen: {chosen}, missing in the database {len(data.get('missing_in_db') or [])}")
        elif event == "case_analysis_ruling_cards_reading":
            out.append(f"ruling cards in the reading: {data.get('card_rows')} card rulings + {data.get('searched')} "
                       f"searched, {data.get('kept')} read, {data.get('cards_kept')} of them from cards")
        elif event == "case_analysis_plan":
            removed = data.get("removed") or {}
            out.append(f"plan: {data.get('issues')} issues, {len(data.get('norme') or [])} articles, "
                       f"{len(data.get('pronunce') or [])} of {data.get('references_available')} rulings, "
                       f"removed {len(removed.get('norme') or [])} articles and {len(removed.get('pronunce') or [])} "
                       f"rulings, replanned {'yes' if data.get('replanned') else 'no'}")
            checked = data.get("checked")
            if checked is not None:
                out.append(f"plan check: off-topic {', '.join(checked.get('off_topic') or []) or '-'}; "
                           f"contradicted {'; '.join(checked.get('contradicted') or []) or '-'}"
                           + (f"; kept as voted {', '.join(checked['kept_by_vote'])}"
                              if checked.get("kept_by_vote") else "")
                           + (f"; kept as applicable {', '.join(checked['kept_as_applicable'])}"
                              if checked.get("kept_as_applicable") else ""))
        elif event == "case_analysis_plan_applicable":
            out.append(f"plan breadth: {len(data.get('applicable') or [])} applicable (up to {data.get('max_issues')} "
                       f"issues); placed by the plan {', '.join(data.get('placed') or []) or '-'}; added to the last "
                       f"issue {', '.join(data.get('added_to_last_issue') or []) or '-'}")
        elif event in ("case_analysis_plan_llm", "case_analysis_plan_check_llm", "case_analysis_sections_llm",
                       "case_analysis_draft_llm") and duration is not None:
            out.append(f"{event[len('case_analysis_'):-len('_llm')]} call: {duration / 1000:.1f}s")
        elif event == "case_pool_vote":
            took = f", {duration / 1000:.1f}s" if duration is not None else ""
            if data.get("skipped") or data.get("error"):
                out.append(f"vote: {data.get('skipped') or data.get('error')}{took}")
            else:
                out.append(f"vote ({data.get('mode')}, K {data.get('k')}): {data.get('rulings')} rulings, "
                           f"{data.get('carded')} carded{took}; top: {', '.join(data.get('top') or []) or '-'}")
                for reason, refs in (data.get("dropped") or {}).items():
                    out.append(f"vote dropped ({reason}): {', '.join(refs)}")
        elif event == "case_pool_vote_cut":
            out.append(f"vote in the pool: added {', '.join(data.get('added') or []) or '-'}; "
                       f"tagged _vote {', '.join(data.get('tagged') or []) or '-'}")
        elif event == "case_pool_vote_idf":
            out.append(f"vote IDF counts: {'ready' if data.get('ready') else 'FAILED'}"
                       + (f", {duration / 1000:.0f}s" if duration is not None else ""))
        elif event == "case_analysis_qualification":
            line = (f"qualification check: {len(data.get('checked') or [])} checked, flagged "
                    f"{', '.join(data.get('flagged') or []) or '-'}")
            if "removed" in data:      # FLAGGED_REMOVE_ENABLED
                kept = data.get("kept") or {}
                line += (f"; removed {', '.join(data['removed']) or '-'}; kept "
                         + (", ".join(f"{label} ({why})" for label, why in kept.items()) or "-")
                         + f"; mark dropped (wording) {', '.join(data.get('note_dropped') or []) or '-'}")
            out.append(line)
        elif event == "case_analysis_citation_hygiene":
            out.append(f"not a law, removed: {', '.join(data.get('removed') or []) or '-'} "
                       f"({'; '.join(f'{c}: {t}' for c, t in (data.get('types') or {}).items())})")
        elif event == "case_analysis_citation_discipline":
            out.append(f"citations taken out of the sections: {data.get('count', 0)} "
                       f"({data.get('rephrased', 0)} rephrased, {data.get('dropped', 0)} dropped): "
                       + (", ".join(r for e in data.get("removed") or [] for r in e.get("removed") or []) or "-"))
        elif event == "case_analysis_prompt_tokens":
            out.append(f"draft prompt: {data.get('total')} est. tokens, budget {data.get('budget')}, "
                       f"step {data.get('step')}, dropped to fit {data.get('dropped_for_size', '-')}, "
                       f"refilled {data.get('refilled', '-')}, left out {len(data.get('articles_left_out') or [])}")
    if isinstance(result, dict):
        out.append(f"answer: {len(result.get('draft') or '')} characters, "
                   f"{len(result.get('citations') or [])} citations")
    out += vote_trace(events, (result or {}).get("draft") if isinstance(result, dict) else "")
    return out


def _eval_module():
    """tests/eval_narrative.py of the folder we run from: its mentions() is the MENTIONED score."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("eval_narrative",
                                                  os.path.join(os.getcwd(), "tests", "eval_narrative.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expected_trace(expected: list, events: list, draft: str) -> list:
    """--case: each expected article in the pool (the retrieval's article rows), in the plan, and
    MENTIONED in the answer with its code (tests/eval_narrative.py's primary score). A pool or plan
    label ("584 c.p.", "95 Codice degli Appalti 2026") is read as the eval reads an answer."""
    ev = _eval_module()

    def keys(labels):
        return {ref for label in labels or [] for ref in ev.mentions(f"art. {label}")}
    pool = next((d.get("article_refs") for e, d, _ in reversed(events)
                 if e == "case_analysis_retrieval" and d and "article_refs" in d), None)
    plan = next((d for e, d, _ in reversed(events) if e == "case_analysis_plan" and d), {})
    in_pool, in_plan, named = keys(pool), keys(plan.get("norme")), set(ev.mentions(draft or ""))
    lines = ["expected article: pool / plan / MENTIONED in the answer"]
    for item in expected:
        wanted = {(str(a).lower(), item["code"]) for a in (item.get("any_of") or [item["article"]])}
        cells = ["?" if pool is None else "yes" if wanted & in_pool else "no",
                 "yes" if wanted & in_plan else "no", "yes" if wanted & named else "no"]
        lines.append(f"  {'/'.join(sorted(a for a, _ in wanted)) + ' ' + item['code']:<22s} " + " / ".join(cells))
    return lines


def vote_trace(events: list, draft: str) -> list:
    """Each article the vote put in the pool (VOTE_POOL_ENABLED): past the applicability check,
    in the draft prompt, offered to the plan, in the plan, in the answer's text. [] without a vote."""
    def last(name):
        found = [d or {} for e, d, _ in events if e == name]
        return found[-1] if found else None
    cut = last("case_pool_vote_cut")
    if not cut or not cut.get("tagged"):
        return []
    kept = set((last("case_pool_vote_kept") or {}).get("kept") or [])
    left_out = set((last("case_analysis_prompt_tokens") or {}).get("articles_left_out") or [])
    plan = last("case_analysis_plan") or {}
    offered, in_plan = set(plan.get("voted_offered") or []), set(plan.get("norme") or [])
    from src.rag.defensive_generation import _refs_in
    in_answer = {f"{n} {c}" for n, c in _refs_in(draft or "")}
    lines = ["voted article: applicable / prompt / offered to plan / plan / answer"]
    for label in cut["tagged"]:
        marks = [label in kept, label in kept and label not in left_out, label in offered, label in in_plan,
                 label in in_answer]
        lines.append(f"  {label:<16s} " + " / ".join("yes" if m else "no" for m in marks)
                     + ("  (added by the vote)" if label in (cut.get("added") or []) else ""))
    if not plan:
        lines.append("  (no plan: PLAN_THEN_WRITE off or the plan failed)")
    return lines


def saved_run(code: str, request: str, status: str, elapsed: float, lines: list, events: list, result) -> dict:
    """What --save writes: the report, the vlog events and the answer (draft + citations)."""
    result = result if isinstance(result, dict) else {}
    return {"code": code, "request": request, "status": status, "seconds": round(elapsed, 1),
            "report": lines, "events": [{"event": e, "data": d, "duration_ms": ms} for e, d, ms in events],
            "draft": result.get("draft"), "answer": result.get("draft"), "citations": result.get("citations")}


def _document_text(document_id: str) -> str:
    from src.db.base import SessionLocal
    from src.db.models import UserDocument
    from src.utils.document import extract_text_from_file

    db = SessionLocal()
    try:
        row = db.query(UserDocument).filter(UserDocument.id == document_id).first()
    finally:
        db.close()
    if row is None or not os.path.exists(row.storage_path or ""):
        sys.exit(f"document {document_id}: not found or no longer on disk")
    return extract_text_from_file(row.storage_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--document-id", help="a user_documents id")
    source.add_argument("--file", help="a local .pdf/.docx/.txt, read as an upload is")
    source.add_argument("--case", help="an eval case id (tests/eval_narrative_cases.json), sent as a chat message")
    parser.add_argument("--request", default=DEFAULT_REQUEST, help="the user's message (default: a generic one)")
    parser.add_argument("--lang", default="it")
    parser.add_argument("--env", default=os.path.join(os.getcwd(), ".env"))
    parser.add_argument("--save", help="JSON file for the report and the answer's text (test cases only)")
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                        help="a switch for this run only, over the .env (e.g. PLAN_THEN_WRITE_ENABLED=true)")
    args = parser.parse_args()

    sys.path.insert(0, os.getcwd())                 # the code of the folder we run from
    from dotenv import load_dotenv
    load_dotenv(args.env)
    for item in args.set:
        name, _, value = item.partition("=")
        os.environ[name.strip()] = value.strip()
        print(f"set for this run: {name.strip()}={value.strip()}", file=sys.stderr)

    import src.rag.defensive_generation as dg
    print(f"code: {os.path.abspath(dg.__file__)}", file=sys.stderr)

    events = []
    dg.vlog = lambda event, data=None, *a, **k: events.append((event, data, k.get("duration_ms")))
    for module in ("ruling_cards", "vote_pool"):   # modules that log through their own vlog
        try:
            setattr(__import__(f"src.rag.{module}", fromlist=["vlog"]), "vlog", dg.vlog)
        except ImportError:
            pass

    request = args.request
    if args.case:
        cases = {c["id"]: c for c in json.load(open(os.path.join(os.getcwd(), "tests", "eval_narrative_cases.json"),
                                                     encoding="utf-8"))["cases"]}
        if args.case not in cases:
            sys.exit(f"no case {args.case!r} in tests/eval_narrative_cases.json")
        text = request = cases[args.case]["question"]   # the chat with no document: the message is both
    elif args.file:
        from src.utils.document import extract_text_from_file
        text = extract_text_from_file(args.file)
    else:
        text = _document_text(args.document_id)
    t0 = time.time()
    result = None
    try:
        result = dg.run_defensive_pipeline(text, request, args.lang)
        status = "ok"
    except Exception as exc:
        status = f"ERROR {type(exc).__name__}: {str(exc)[:160]}"
    elapsed = time.time() - t0
    lines = summarise(events, status, result, elapsed, len(text))
    if args.case:
        lines += expected_trace(cases[args.case].get("expected") or [], events,
                                (result or {}).get("draft") if isinstance(result, dict) else "")
    print("\n".join(lines))
    if args.save:
        record = saved_run(os.path.abspath(dg.__file__), request, status, elapsed, lines, events, result)
        record["settings"] = args.set
        with open(args.save, "w", encoding="utf-8") as fh:
            json.dump(record, fh, ensure_ascii=False, indent=1, default=str)
        print(f"\nSaved to {args.save}, the answer included. Nothing else was written.")
    else:
        print("\nNothing was written: no document, no session, no database row.")


if __name__ == "__main__":
    main()
