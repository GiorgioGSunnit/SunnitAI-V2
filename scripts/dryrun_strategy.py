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

The document is an upload (--document-id) or a local file read as an upload is (--file:
.pdf/.docx/.txt). --save writes the report AND the answer's text to a JSON file, to compare
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
        elif event == "case_analysis_plan":
            removed = data.get("removed") or {}
            out.append(f"plan: {data.get('issues')} issues, {len(data.get('norme') or [])} articles, "
                       f"{len(data.get('pronunce') or [])} of {data.get('references_available')} rulings, "
                       f"removed {len(removed.get('norme') or [])} articles and {len(removed.get('pronunce') or [])} "
                       f"rulings, replanned {'yes' if data.get('replanned') else 'no'}")
        elif event in ("case_analysis_plan_llm", "case_analysis_sections_llm", "case_analysis_draft_llm") \
                and duration is not None:
            out.append(f"{event[len('case_analysis_'):-len('_llm')]} call: {duration / 1000:.1f}s")
        elif event == "case_analysis_prompt_tokens":
            out.append(f"draft prompt: {data.get('total')} est. tokens, budget {data.get('budget')}, "
                       f"step {data.get('step')}, dropped to fit {data.get('dropped_for_size', '-')}, "
                       f"refilled {data.get('refilled', '-')}, left out {len(data.get('articles_left_out') or [])}")
    if isinstance(result, dict):
        out.append(f"answer: {len(result.get('draft') or '')} characters, "
                   f"{len(result.get('citations') or [])} citations")
    return out


def saved_run(code: str, request: str, status: str, elapsed: float, lines: list, events: list, result) -> dict:
    """What --save writes: the report, the vlog events and the answer (draft + citations)."""
    result = result if isinstance(result, dict) else {}
    return {"code": code, "request": request, "status": status, "seconds": round(elapsed, 1),
            "report": lines, "events": [{"event": e, "data": d, "duration_ms": ms} for e, d, ms in events],
            "draft": result.get("draft"), "citations": result.get("citations")}


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

    if args.file:
        from src.utils.document import extract_text_from_file
        text = extract_text_from_file(args.file)
    else:
        text = _document_text(args.document_id)
    t0 = time.time()
    result = None
    try:
        result = dg.run_defensive_pipeline(text, args.request, args.lang)
        status = "ok"
    except Exception as exc:
        status = f"ERROR {type(exc).__name__}: {str(exc)[:160]}"
    elapsed = time.time() - t0
    lines = summarise(events, status, result, elapsed, len(text))
    print("\n".join(lines))
    if args.save:
        record = saved_run(os.path.abspath(dg.__file__), args.request, status, elapsed, lines, events, result)
        record["settings"] = args.set
        with open(args.save, "w", encoding="utf-8") as fh:
            json.dump(record, fh, ensure_ascii=False, indent=1, default=str)
        print(f"\nSaved to {args.save}, the answer included. Nothing else was written.")
    else:
        print("\nNothing was written: no document, no session, no database row.")


if __name__ == "__main__":
    main()
