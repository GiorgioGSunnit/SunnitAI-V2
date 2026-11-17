"""v77: the held-out set -- every case-first answer served (live, /opt/chatbot; the test copy, /root/v38 through
scripts/serve_test.sh) saved as one JSON file, /opt/chatbot/heldout/<date>_<id>.json: the prompt, the intake, the answer,
its article_citations, the version, the timings. scripts/heldout_export.py hands N unscored ones to the lawyer,
scripts/heldout_score.py reads his scores back (heldout/scores.jsonl). These cases are the substance measure: no rule
is tuned on them.

HELDOUT_DIR overrides the folder; without it the folder is /opt/chatbot/heldout when /opt/chatbot exists (the
server), else nothing is saved (a laptop, the tests). A failed save is logged and never touches the answer."""

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from .. import version

logger = logging.getLogger(__name__)

ENV = "HELDOUT_DIR"
EXPORTS = "exports.jsonl"
# The lawyer's seven sections, as his critique of the Sempronio parere is built (docs/lawyer/critica_penale_v2.md,
# docs/lawyer/critiques/sempronio_no_hint.critique.json): each scored 0-5 on the parere, 5 the best. The labels are his
# own headings, generalised; pending his confirmation (17 Nov 2026) -- a change of label keeps the key
SECTIONS = (("tesi", "Tesi centrale"),
            ("categoria", "Errori di categoria (qualificazione, elemento soggettivo, norme fuori posto)"),
            ("coerenza", "Contraddizioni e incongruenze interne"),
            ("processuale", "Profili processuali"),
            ("mancanze", "Cosa manca"),
            ("pregi", "Pregi"),
            ("strategia", "Strategia"))
SCALE = (0, 5)
SERVER_DIR = Path("/opt/chatbot/heldout")
SCORES = "scores.jsonl"
TIMING_KEYS = ("total_s", "round1_s", "answer_s", "thesis_selection_s", "writer_s")


def folder() -> Optional[Path]:
    configured = (os.getenv(ENV) or "").strip()
    if configured:
        return Path(configured)
    return SERVER_DIR if SERVER_DIR.parent.is_dir() else None


def source() -> str:
    """Which copy served it: «live» (/opt/chatbot), «test» (any other checkout: /root/v38)."""
    return "live" if str(version.ROOT).startswith("/opt/chatbot") else "test"


def timings(state: Dict[str, Any]) -> Dict[str, Any]:
    shot = state.get("single_shot_timings") or {}
    raw = state.get("timings") or {}
    out = {k: shot.get(k, raw.get(k)) for k in TIMING_KEYS}
    return {k: v for k, v in out.items() if v is not None}


def record(session_id: str, prompt: str, result: Dict[str, Any], mode: str, intake: Optional[Dict[str, Any]] = None,
           case_text: Optional[str] = None) -> Dict[str, Any]:
    state = result.get("state") or {}
    now = time.gmtime()
    return {"id": f"{time.strftime('%Y-%m-%d', now)}_{uuid.uuid4().hex[:8]}",
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", now),
            "version": version.current(), "source": source(), "session_id": session_id, "mode": mode,
            "kind": result.get("kind") or "", "prompt": prompt or "",
            **({"case_text": case_text} if case_text else {}),
            "intake": intake or {}, "answer": result.get("answer") or "",
            "article_citations": result.get("article_citations") or [], "timings": timings(state),
            **({"rulings": rulings_used(state)} if state.get("rulings_record") else {})}


def rulings_used(state: Dict[str, Any]) -> Dict[str, Any]:
    """v78: the rulings the answer used (RULINGS_RECORD) -- the kept ones with their passage, the orientations, the
    lawyer's pronunce di riferimento."""
    record = state.get("rulings_record") or {}
    return {k: record.get(k) for k in ("kept", "orientations", "landmarks")}


def save(session_id: str, prompt: str, result: Dict[str, Any], mode: str, intake: Optional[Dict[str, Any]] = None,
         case_text: Optional[str] = None) -> Optional[Path]:
    """The answer as <folder>/<date>_<id>.json, or None (no folder, or the write failed: logged)."""
    target = folder()
    if target is None:
        return None
    try:
        rec = record(session_id, prompt, result, mode, intake, case_text)
        target.mkdir(parents=True, exist_ok=True)
        path = target / f"{rec['id']}.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")
        os.replace(tmp, path)
        return path
    except Exception as exc:        # noqa: BLE001 -- the held-out copy never fails the answer
        logger.warning("heldout: the answer of session %s was not saved: %s", session_id, exc)
        return None


def load(target: Path) -> List[Dict[str, Any]]:
    """The saved records, oldest first (the id begins with the date)."""
    out = []
    for path in sorted(Path(target).glob("*.json")):
        try:
            out.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError) as exc:
            logger.warning("heldout: %s unreadable: %s", path.name, exc)
    return out


def scores(target: Path) -> List[Dict[str, Any]]:
    """The scores recorded so far (scripts/heldout_score.py), one per line of <folder>/scores.jsonl."""
    path = Path(target) / SCORES
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def latest_scores(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One score per case, the last recorded (a case scored again replaces its earlier score)."""
    by: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        by[row["id"]] = row
    return list(by.values())


def case_mean(sections: Dict[str, Any]) -> Optional[float]:
    values = [v for v in (sections or {}).values() if isinstance(v, (int, float))]
    return round(sum(values) / len(values), 2) if values else None


def version_means(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per version (in the order first scored): the cases scored, the mean of their means, each section's mean."""
    out: Dict[str, Dict[str, Any]] = {}
    for row in latest_scores(rows):
        if row.get("mean") is None:
            continue
        v = out.setdefault(row.get("version") or version.UNKNOWN, {"version": row.get("version") or version.UNKNOWN,
                                                                   "means": [], "sections": {}})
        v["means"].append(row["mean"])
        for key, value in (row.get("sections") or {}).items():
            if isinstance(value, (int, float)):
                v["sections"].setdefault(key, []).append(value)
    return [{"version": v["version"], "cases": len(v["means"]), "mean": round(sum(v["means"]) / len(v["means"]), 2),
             "sections": {k: round(sum(x) / len(x), 2) for k, x in v["sections"].items()}} for v in out.values()]


def means_table(rows: List[Dict[str, Any]]) -> List[str]:
    """The Markdown table of version_means, the sections by their keys; [] when nothing is scored."""
    means = version_means(rows)
    if not means:
        return []
    keys = [k for k, _ in SECTIONS]
    lines = ["| version | cases | mean (0-5) | " + " | ".join(keys) + " |", "|---" * (len(keys) + 3) + "|"]
    for m in means:
        lines.append(f"| {m['version']} | {m['cases']} | {m['mean']} | "
                     + " | ".join(str(m["sections"].get(k, "—")) for k in keys) + " |")
    return lines
