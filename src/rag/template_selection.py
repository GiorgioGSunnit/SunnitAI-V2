"""Choose catalog templates for a generation request by meaning (embeddings).

Every catalog template has a precomputed vector (label + tipo_atto +
description, built by scripts/template_selection/embed_catalog.py and shipped
next to the catalog). A request is embedded on the same server and model, then:

  best score below FLOOR           -> [] ("the catalog has nothing for this")
  best minus second >= MARGIN      -> the best template alone (auto-select)
  otherwise                        -> the templates within WINDOW of the best,
                                      2 to 5 of them (the "Quale variante" picker)

Calibrated on 123 labelled requests against the 5,145-template catalog (Sept
2026): right template in the top 5 for 109/109, no wrong auto-select, 14/14
requests for documents the catalog lacks refused. See
scripts/template_selection/calibrate_v2.py.

select_templates() returns None whenever it cannot decide - vectors missing or
out of step with the catalog, embedding server down, numpy/httpx unavailable -
and the caller then uses the keyword method it always had.
"""
import json
import logging
import os
import threading
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

_DATA_DIR = os.path.dirname(os.getenv(
    "SYSTEM_TEMPLATES_CATALOG_PATH", "/opt/chatbot/data/system_templates/catalog_enriched.json"))
VECTORS_PATH = os.getenv("TEMPLATE_VECTORS_PATH", os.path.join(_DATA_DIR, "template_vectors.npy"))
INDEX_PATH = os.getenv("TEMPLATE_VECTORS_INDEX", os.path.join(_DATA_DIR, "template_index.json"))
FLOOR = float(os.getenv("TEMPLATE_SELECT_FLOOR", "0.62"))
MARGIN = float(os.getenv("TEMPLATE_SELECT_MARGIN", "0.10"))
WINDOW = float(os.getenv("TEMPLATE_SELECT_WINDOW", "0.05"))
MAX_CANDIDATES = 5
# Templates without a vector: above this share of the catalog, selection by
# meaning is switched off rather than silently ignoring part of the catalog.
_MAX_MISSING_SHARE = 0.02

# The query side of the embedding model needs an instruction; without it the
# right template ranked first in 25% of test requests instead of 89%.
INSTRUCT = ("Instruct: Data la richiesta di un avvocato, trova il modello di atto giuridico "
            "italiano che corrisponde al documento richiesto\nQuery: ")

_lock = threading.Lock()
_cache: Dict[str, Any] = {"key": None, "matrix": None, "rows": None}


def _load(catalog: Sequence[Dict[str, Any]]):
    """(matrix, catalog index per matrix row) for this catalog, or None."""
    catalog_key = (len(catalog), catalog[0].get("filename") if catalog else None,
                   catalog[-1].get("filename") if catalog else None)
    with _lock:
        if _cache["key"] == catalog_key:
            return (_cache["matrix"], _cache["rows"]) if _cache["matrix"] is not None else None
        _cache.update(key=catalog_key, matrix=None, rows=None)
        try:
            import numpy as np
            matrix = np.load(VECTORS_PATH).astype(np.float32)
            with open(INDEX_PATH, encoding="utf-8") as f:
                index = [x["filename"] for x in json.load(f)]
        except Exception as exc:
            logger.warning("template selection by meaning off: cannot load vectors (%s)", exc)
            return None
        if len(index) != matrix.shape[0]:
            logger.warning("template selection by meaning off: %d vectors for %d index rows",
                           matrix.shape[0], len(index))
            return None
        row_of = {fn: r for r, fn in enumerate(index)}
        pos = [(i, row_of.get(e.get("filename"))) for i, e in enumerate(catalog)]
        missing = sum(1 for _, r in pos if r is None)
        if missing > _MAX_MISSING_SHARE * max(1, len(catalog)):
            logger.warning("template selection by meaning off: %d of %d templates have no vector "
                           "(re-run embed_catalog.py after changing the catalog)", missing, len(catalog))
            return None
        if missing:
            logger.warning("template selection: %d templates have no vector and cannot be chosen", missing)
        pos = [(i, r) for i, r in pos if r is not None]
        m = matrix[[r for _, r in pos]]
        m /= np.linalg.norm(m, axis=1, keepdims=True)
        _cache.update(matrix=m, rows=[i for i, _ in pos])
        logger.info("template selection by meaning on: %d templates, %d dims", m.shape[0], m.shape[1])
        return m, _cache["rows"]


def _embed(text: str, dims: int):
    """The request's vector, straight from the embedding server as plain text -
    the same way the template vectors were made. None on any failure."""
    try:
        import httpx
        import numpy as np
        base = (os.getenv("EMBEDDING_BASE_URL") or "").rstrip("/")
        if not base:
            return None
        key = os.getenv("EMBEDDING_API_KEY", os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY")))
        r = httpx.post(base + "/embeddings",
                       json={"model": os.getenv("EMBEDDING_MODEL"), "input": [INSTRUCT + text]},
                       headers={"Authorization": "Bearer " + key} if key else {}, timeout=15)
        r.raise_for_status()
        v = np.asarray(r.json()["data"][0]["embedding"], dtype=np.float32)[:dims]
        return v / np.linalg.norm(v)
    except Exception as exc:
        logger.warning("template selection: embedding the request failed (%s)", exc)
        return None


def decide(scores, floor: float = FLOOR, margin: float = MARGIN,
           window: float = WINDOW) -> List[Tuple[int, float]]:
    """(position, score) of the templates to offer: [] / one / a picker of 2-5."""
    import numpy as np
    order = np.argsort(-scores)
    s1 = float(scores[order[0]])
    if s1 < floor:
        return []
    if len(order) == 1 or s1 - float(scores[order[1]]) >= margin:
        return [(int(order[0]), s1)]
    picked = [int(i) for i in order[:MAX_CANDIDATES] if scores[i] >= s1 - window]
    if len(picked) < 2:
        picked = [int(i) for i in order[:2]]
    return [(i, float(scores[i])) for i in picked]


def select_templates(message: str, catalog: Sequence[Dict[str, Any]],
                     exclude_codici: Optional[Set[str]] = None) -> Optional[List[Tuple[int, float]]]:
    """Catalog indices (with scores) to offer for this request, [] when the
    catalog has nothing close enough, or None when selection by meaning is not
    available and the caller should fall back.

    exclude_codici: areas the request rules out (e.g. civil procedure when it
    names the c.p.p.). Dropped only if something above the floor remains.
    """
    loaded = _load(catalog)
    if loaded is None:
        return None
    matrix, rows = loaded
    q = _embed(message, matrix.shape[1])
    if q is None:
        return None
    scores = matrix @ q
    if exclude_codici:
        keep = [catalog[i].get("codice") not in exclude_codici for i in rows]
        masked = scores.copy()
        masked[[not k for k in keep]] = -1.0
        if masked.max() >= FLOOR:
            scores = masked
    return [(rows[p], s) for p, s in decide(scores)]
