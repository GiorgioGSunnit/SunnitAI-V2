"""Cross-encoder reranking of retrieved rows."""

import logging
import os

from .verbose_logger import vlog

logger = logging.getLogger(__name__)


_RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "false").lower() == "true"
_RERANKER_URL = os.getenv("RERANKER_URL", "http://217.160.8.129:8002/v1/rerank")


def _format_for_reranker(row: dict) -> str:
    """Build structured reranker input that emphasises title and
    document context over raw text length. Works for any document
    regardless of name, version, or section length."""
    s = row.get("s") or {}
    d = row.get("d") or {}

    name = s.get("name", "")
    abstract = (s.get("abstract") or "").strip()
    plain_text = (s.get("plain_text") or "").strip()
    doc_name = d.get("name", "")

    parts = []

    # Document context — skip raw filenames
    if doc_name and not doc_name.lower().endswith(
            ('.pdf', '.docx', '.xlsx', '.txt')):
        parts.append(f"Fonte: {doc_name}")

    # Section identifier
    if name:
        parts.append(f"Articolo: {name}")

    # Content — combine abstract and plain_text for maximum signal
    # Abstract already has title prepended (e.g. "Omicidio - ...")
    # Plain text has the actual legal provision
    content = abstract or plain_text
    if plain_text and plain_text not in (abstract or ""):
        content = f"{content} {plain_text}"[:500]
    else:
        content = content[:500]

    if content:
        parts.append(content)

    return " | ".join(parts)


def rerank_results(query: str, rows: list, top_k: int = 12) -> list:
    """Re-sort rows by reranker score. Fail-open: returns rows unchanged on any error."""
    if not _RERANKER_ENABLED or not rows:
        return rows
    if all(r.get("_source") == "clarification" for r in rows):
        return rows  # skip reranker for clarification rerank — scores already set
    import time as _time
    import requests
    t0 = _time.time()
    try:
        reranker_query = (
            f"Instruct: Given a legal query in Italian, retrieve the most relevant legal document sections\n"
            f"Query: {query}"
        )
        payload = {
            "model": os.getenv("RERANKER_MODEL", "reranker"),
            "query": reranker_query,
            "documents": [
                _format_for_reranker(row) for row in rows
            ],
        }
        api_key = os.getenv("LLM_API_KEY", "")
        resp = requests.post(
            _RERANKER_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=10,
        )
        resp.raise_for_status()
        scored = resp.json().get("results", [])
        if not scored:
            return rows
        for result in scored:
            original_idx = result["index"]
            score = result["relevance_score"]
            if original_idx < len(rows):
                rows[original_idx]["_reranker_score"] = score
        reranker_top = sorted(rows, key=lambda r: r.get("_reranker_score", 0), reverse=True)[:top_k - 2]

        bm25_rows = [r for r in rows if r.get("_source") == "bm25"]
        reranker_ids = {
            (r.get("s") or {}).get("id") for r in reranker_top
            if (r.get("s") or {}).get("id")
        }
        BM25_INJECTION_MIN_RERANKER_SCORE = 0.3
        bm25_candidates = [
            r for r in bm25_rows
            if (r.get("s") or {}).get("id") not in reranker_ids
            and r.get("_reranker_score", 0) >= BM25_INJECTION_MIN_RERANKER_SCORE
        ]
        bm25_top = sorted(
            bm25_candidates, key=lambda r: r.get("_reranker_score", 0), reverse=True
        )[:2]

        merged_ids = reranker_ids | {
            (r.get("s") or {}).get("id") for r in bm25_top
            if (r.get("s") or {}).get("id")
        }
        slots_remaining = top_k - len(reranker_top) - len(bm25_top)
        overflow = [
            r for r in sorted(rows, key=lambda r: r.get("_reranker_score", 0), reverse=True)
            if (r.get("s") or {}).get("id") not in merged_ids
        ][:slots_remaining]

        reranked = reranker_top + bm25_top + overflow
        logger.info(
            "Reranker merge: reranker_top=%d bm25_injected=%d overflow=%d total=%d",
            len(reranker_top), len(bm25_top), len(overflow), len(reranked),
        )

        score_debug = []
        for r in reranked:
            s = r.get("s") or {}
            if hasattr(s, "get"):
                name = s.get("name") or s.get("title") or "?"
            else:
                name = str(s)[:20]
            score_debug.append((name, round(r.get("_reranker_score", 0), 3)))
        logger.info("Reranker scores (top %d): %r", len(score_debug), score_debug)
        vlog("reranker", {"input_count": len(rows), "output_count": len(reranked)}, (_time.time() - t0) * 1000)
        return reranked
    except Exception as exc:
        logger.warning("Reranker failed (fail-open): %s", exc)
        vlog("reranker", {"input_count": len(rows), "output_count": len(rows), "error": str(exc)[:120]}, (_time.time() - t0) * 1000)
        return rows
