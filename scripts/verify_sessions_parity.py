"""
Read-only parity check: does Postgres hold everything that sessions.json holds?

Run this BEFORE switching the chat backend to read conversations from Postgres.
After the switch the backend never reads sessions.json again, and its first
save rewrites that file with only the sessions touched since the restart — so
anything that exists only in the file is gone from that point on.

Nothing here writes to the sessions file or to the database:
  * the file is opened read-only;
  * the Postgres transaction is set to READ ONLY before the first query;
  * --report refuses to overwrite the sessions file.

Usage (from the repository root):
    python scripts/verify_sessions_parity.py
    python scripts/verify_sessions_parity.py --sessions-file /opt/chatbot/data/sessions.json
    python scripts/verify_sessions_parity.py --report parity_report.json

DATABASE_URL must be set (environment or --env-file). There is deliberately no
fallback: the script prints which database it is about to read and stops
without one.

Exit status: 0 = parity, 1 = sessions at risk, 2 = the check could not run.
"""

import argparse
import json
import os
import sys
import uuid
from typing import Any, Callable, Dict, List, Optional

# Categories that mean "this session would be lost or damaged by the switch".
AT_RISK = (
    "missing_in_postgres",
    "unmirrorable_id",
    "owner_mismatch",
    "postgres_behind",
    "content_differs",
)

BATCH_SIZE = 200


# ---------------------------------------------------------------------------
# Pure comparison logic (no I/O, so it can be tested without a database)
# ---------------------------------------------------------------------------

def _parse_uuid(value: Any) -> Optional[uuid.UUID]:
    try:
        return uuid.UUID(str(value))
    except (ValueError, TypeError, AttributeError):
        return None


def compare(
    json_sessions: List[Dict[str, Any]],
    fetch_rows: Callable[[List[uuid.UUID]], Dict[uuid.UUID, Dict[str, Any]]],
    postgres_total: Optional[int] = None,
) -> Dict[str, Any]:
    """Classify every session in the file against what Postgres holds.

    `fetch_rows` takes a list of UUIDs and returns {uuid: {"user_id", "tenant_id",
    "title", "messages"}} for the rows that exist.
    """
    report: Dict[str, Any] = {c: [] for c in AT_RISK}
    report.update({
        "anonymous_memory_only": [],   # expected: never mirrored, by design
        "postgres_ahead": [],          # informational: Postgres has more than the file
        "title_differs": [],           # informational
        "ok": [],
    })
    seen: Dict[str, int] = {}
    to_check: List[Dict[str, Any]] = []

    for s in json_sessions:
        sid = str(s.get("session_id", ""))
        seen[sid] = seen.get(sid, 0) + 1
        if not s.get("user_id") or not s.get("tenant_id"):
            report["anonymous_memory_only"].append({
                "session_id": sid, "messages": len(s.get("messages") or []),
            })
        elif _parse_uuid(sid) is None:
            report["unmirrorable_id"].append({
                "session_id": sid, "user_id": s.get("user_id"),
                "messages": len(s.get("messages") or []),
            })
        else:
            to_check.append(s)

    rows: Dict[uuid.UUID, Dict[str, Any]] = {}
    for i in range(0, len(to_check), BATCH_SIZE):
        ids = [_parse_uuid(s["session_id"]) for s in to_check[i:i + BATCH_SIZE]]
        rows.update(fetch_rows(ids))

    for s in to_check:
        sid = str(s["session_id"])
        row = rows.get(_parse_uuid(sid))
        n_json = len(s.get("messages") or [])
        if row is None:
            report["missing_in_postgres"].append({
                "session_id": sid, "user_id": s.get("user_id"), "messages": n_json,
            })
            continue
        pg_msgs = row.get("messages") if isinstance(row.get("messages"), list) else []
        n_pg = len(pg_msgs)
        if str(row.get("user_id")) != str(s.get("user_id")):
            report["owner_mismatch"].append({
                "session_id": sid, "file_user": s.get("user_id"), "postgres_user": str(row.get("user_id")),
            })
        elif n_pg < n_json:
            report["postgres_behind"].append({
                "session_id": sid, "file_messages": n_json, "postgres_messages": n_pg,
            })
        elif n_pg > n_json:
            report["postgres_ahead"].append({
                "session_id": sid, "file_messages": n_json, "postgres_messages": n_pg,
            })
        elif (s.get("messages") or []) != pg_msgs:
            report["content_differs"].append({"session_id": sid, "messages": n_json})
        else:
            if (s.get("title") or "") != (row.get("title") or ""):
                report["title_differs"].append({
                    "session_id": sid, "file_title": s.get("title"), "postgres_title": row.get("title"),
                })
            report["ok"].append(sid)

    dupes = {k: v for k, v in seen.items() if v > 1}
    matched = len(rows)     # distinct Postgres rows that correspond to a session in the file
    summary = {
        "sessions_in_file": len(json_sessions),
        "duplicate_ids_in_file": dupes,
        "postgres_conversations_total": postgres_total,
        "postgres_rows_not_in_file": (postgres_total - matched) if postgres_total is not None else None,
        **{k: len(v) for k, v in report.items()},
    }
    at_risk = sum(len(report[c]) for c in AT_RISK)
    return {"summary": summary, "at_risk_total": at_risk, "details": report}


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_json_sessions(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:        # read-only
        data = json.load(f)
    sessions = data.get("sessions") if isinstance(data, dict) else None
    if not isinstance(sessions, list):
        raise ValueError(f"{path}: expected an object with a 'sessions' list")
    return sessions


def _load_env(env_file: Optional[str]) -> None:
    if env_file and os.path.exists(env_file):
        try:
            from dotenv import load_dotenv
            load_dotenv(env_file)
        except ImportError:
            pass


def _open_readonly_db():
    """Return (db, description) for a session whose transaction cannot write."""
    from sqlalchemy import text
    from sqlalchemy.engine import make_url

    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. Refusing to guess which database to read "
            "(src/db/base.py would fall back to a localhost default)."
        )
    target = make_url(url)
    description = f"{target.host or 'local socket'}/{target.database}"

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.db.base import SessionLocal

    db = SessionLocal()
    db.execute(text("SET TRANSACTION READ ONLY"))        # must be the first statement
    return db, description


def _make_fetcher(db) -> Callable[[List[uuid.UUID]], Dict[uuid.UUID, Dict[str, Any]]]:
    from src.db.models import Conversation

    def fetch(ids: List[uuid.UUID]) -> Dict[uuid.UUID, Dict[str, Any]]:
        rows = db.query(
            Conversation.id, Conversation.user_id, Conversation.tenant_id,
            Conversation.title, Conversation.messages,
        ).filter(Conversation.id.in_(ids)).all()
        return {
            r.id: {"user_id": r.user_id, "tenant_id": r.tenant_id, "title": r.title, "messages": r.messages}
            for r in rows
        }
    return fetch


def _count_postgres(db) -> int:
    from sqlalchemy import func
    from src.db.models import Conversation
    return db.query(func.count(Conversation.id)).scalar() or 0


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_LABELS = {
    "missing_in_postgres": "AT RISK  in the file, NOT in Postgres",
    "unmirrorable_id": "AT RISK  id is not a UUID, can never be stored in Postgres",
    "owner_mismatch": "AT RISK  Postgres row belongs to a different user",
    "postgres_behind": "AT RISK  Postgres has FEWER messages than the file",
    "content_differs": "AT RISK  same message count, different content",
    "anonymous_memory_only": "expected  no user/tenant: in-memory by design, never mirrored",
    "postgres_ahead": "info      Postgres has MORE messages than the file",
    "title_differs": "info      title differs",
    "ok": "OK        identical",
}


def print_report(result: Dict[str, Any], max_list: int) -> None:
    s = result["summary"]
    print(f"\nSessions in file .............. {s['sessions_in_file']}")
    if s["duplicate_ids_in_file"]:
        print(f"Duplicate ids in file ......... {len(s['duplicate_ids_in_file'])}")
    if s["postgres_conversations_total"] is not None:
        print(f"Conversations in Postgres ..... {s['postgres_conversations_total']}"
              f"  ({s['postgres_rows_not_in_file']} not in the file)")
    print()
    for key in (*AT_RISK, "anonymous_memory_only", "postgres_ahead", "title_differs", "ok"):
        print(f"  {s[key]:6d}  {_LABELS[key]}")
    for key in AT_RISK + ("anonymous_memory_only",):
        items = result["details"][key]
        if not items:
            continue
        print(f"\n--- {_LABELS[key]} ({len(items)}) ---")
        for it in items[:max_list]:
            print("   ", it)
        if len(items) > max_list:
            print(f"    ... {len(items) - max_list} more (use --report for the full list)")
    print()
    if result["at_risk_total"]:
        print(f"RESULT: NOT SAFE TO SWITCH - {result['at_risk_total']} session(s) exist only (or more completely) in the file.")
    else:
        print("RESULT: PARITY - every identified session in the file is in Postgres with identical messages.")
        if s["anonymous_memory_only"]:
            print(f"        ({s['anonymous_memory_only']} anonymous session(s) will be dropped by design; confirm they are dev/test data.)")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sessions-file", default=os.environ.get("SESSIONS_FILE", "/opt/chatbot/data/sessions.json"))
    ap.add_argument("--env-file", default="/opt/chatbot/.env", help="dotenv file to load DATABASE_URL from, if it exists")
    ap.add_argument("--max-list", type=int, default=20, help="ids to print per category")
    ap.add_argument("--report", help="write the full result as JSON to this path")
    args = ap.parse_args(argv)

    if args.report and os.path.abspath(args.report) == os.path.abspath(args.sessions_file):
        print("Refusing to write the report over the sessions file.", file=sys.stderr)
        return 2

    _load_env(args.env_file)
    try:
        sessions = load_json_sessions(args.sessions_file)
    except (OSError, ValueError) as exc:
        print(f"Cannot read sessions file: {exc}", file=sys.stderr)
        return 2

    try:
        db, target = _open_readonly_db()
    except Exception as exc:
        print(f"Cannot open the database: {exc}", file=sys.stderr)
        return 2

    print(f"Sessions file : {args.sessions_file}  ({len(sessions)} sessions)")
    print(f"Database      : {target}  (read-only transaction)")
    try:
        result = compare(sessions, _make_fetcher(db), postgres_total=_count_postgres(db))
    except Exception as exc:
        print(f"Comparison failed: {exc}", file=sys.stderr)
        return 2
    finally:
        db.rollback()
        db.close()

    print_report(result, args.max_list)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        print(f"Full report written to {args.report}")
    return 1 if result["at_risk_total"] else 0


if __name__ == "__main__":
    sys.exit(main())
