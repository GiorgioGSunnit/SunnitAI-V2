"""Validate and normalize parsed JSON files before graph extraction."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

logger = logging.getLogger(__name__)

_REQUIRED_CHUNK_FIELDS = ("chunk_id", "chunk_type", "text_en")


def _normalize_chunk(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """Strip whitespace from string fields and ensure text_en is non-empty."""
    out = dict(chunk)
    if isinstance(out.get("text_en"), str):
        out["text_en"] = out["text_en"].strip()
    if isinstance(out.get("title"), str):
        out["title"] = out["title"].strip()
    return out


def _is_valid_chunk(chunk: Dict[str, Any]) -> bool:
    for field in _REQUIRED_CHUNK_FIELDS:
        if not chunk.get(field):
            return False
    return True


def process_file(
    in_path: Path,
    out_dir: Path,
    drop_invalid: bool = False,
    pretty: bool = False,
) -> Tuple[Path, Dict[str, Any]]:
    """Validate and normalize a parsed JSON file.

    Args:
        in_path:      Path to the parser-produced JSON file.
        out_dir:      Directory to write the normalized output file.
        drop_invalid: If True, omit chunks that fail validation; otherwise keep them.
        pretty:       If True, write indented JSON.

    Returns:
        (norm_path, stats) where stats contains chunk counts.
    """
    in_path = Path(in_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with in_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    chunks = payload.get("chunks", [])
    total = len(chunks)
    invalid = 0
    normalized: list = []

    for chunk in chunks:
        chunk = _normalize_chunk(chunk)
        if not _is_valid_chunk(chunk):
            invalid += 1
            logger.warning("Invalid chunk in %s: %s", in_path.name, chunk.get("chunk_id"))
            if drop_invalid:
                continue
        normalized.append(chunk)

    payload["chunks"] = normalized
    norm_path = out_dir / f"{in_path.stem}.normalized.json"
    indent = 2 if pretty else None
    with norm_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=indent)

    stats = {"total": total, "invalid": invalid, "written": len(normalized)}
    return norm_path, stats
