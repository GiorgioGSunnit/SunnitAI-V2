import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

logger = logging.getLogger(__name__)


def process_file(
    source_path,
    output_dir,
    drop_invalid: bool = False,
    pretty: bool = True,
) -> Tuple[str, Dict[str, Any]]:
    """Validate and normalize a parsed JSON file.

    Reads the JSON from ``source_path``, applies minimal normalization, and
    writes the result to ``output_dir``. Returns (output_path_str, stats_dict).

    ``drop_invalid`` is accepted for interface compatibility but is not yet
    enforced — invalid records are always kept and flagged in stats.
    """
    source_path = Path(source_path)
    output_dir = Path(output_dir)

    with source_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    stats: Dict[str, Any] = {"source": source_path.name, "issues": []}

    if not isinstance(data, dict):
        stats["issues"].append("root is not a dict")
    else:
        if "document_id" not in data:
            stats["issues"].append("missing document_id")
            logger.warning("validate_and_normalize: %s has no document_id", source_path.name)
        if "chunks" not in data:
            data["chunks"] = []
            stats["issues"].append("missing chunks — defaulted to []")
            logger.warning("validate_and_normalize: %s has no chunks, defaulted to []", source_path.name)

    out_path = output_dir / f"{source_path.stem}.normalized.json"
    indent = 2 if pretty else None
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)

    logger.debug("validate_and_normalize: wrote %s (%d issues)", out_path.name, len(stats["issues"]))
    return str(out_path), stats
