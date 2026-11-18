"""v79.9 (B2, B4): the special laws each institution module needs, checked against the graph's primary documents.

data/corpus_expected.json lists, per module, the laws expected (the same list as the module's «laws» in
data/institution_modules.json) and how each is matched: a primary document whose name holds the law's number and year
(«n. 604» and «1966», or «604/1966») or one of its aliases. The names are read once (warm(), at the API's startup) and
cached for the process; a case whose module has a law missing gets one fixed line in §1 and §9
(lawyer_method.MISSING_LAWS_LINE). When the graph could not be read, nothing is said to be missing (missing() is None).

scripts/corpus_gaps.py --env .env prints the missing laws for ingestion.
"""

import json
import logging
import os
import re
import threading
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

EXPECTED_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "corpus_expected.json")
NAMES_QUERY = ("MATCH (d:Document) WHERE d.document_type = 'primary' "
               "RETURN d.name AS name, coalesce(d.aliases, []) AS aliases")

_EXPECTED: Dict[str, Any] = {}
_NAMES: Dict[str, Any] = {}
_LOCK = threading.Lock()


def expected() -> Dict[str, Any]:
    """data/corpus_expected.json (read once; {} when missing, logged)."""
    if "d" not in _EXPECTED:
        try:
            with open(EXPECTED_FILE, encoding="utf-8") as f:
                _EXPECTED["d"] = json.load(f)
        except (OSError, ValueError) as exc:
            logger.warning("corpus_laws: %s not read: %s", EXPECTED_FILE, exc)
            _EXPECTED["d"] = {}
    return _EXPECTED["d"]


def present(law: str, names: Sequence[str]) -> bool:
    """A law is in the graph: a document name with its number («n. 604», «604/1966») and its year, or an alias."""
    spec = (expected().get("laws") or {}).get(law) or {}
    num, year = str(spec.get("number") or ""), str(spec.get("year") or "")
    aliases = [str(a).lower() for a in spec.get("aliases") or []]
    by_number = re.compile(rf"\b(?:n\.?\s*º?\s*|numero\s+){re.escape(num)}\b|\b{re.escape(num)}\s*/\s*{re.escape(year)}\b",
                           re.IGNORECASE) if num else None
    for name in names or []:
        low = str(name or "").lower()
        if aliases and any(a in low for a in aliases):
            return True
        if by_number and year and year in low and by_number.search(low):
            return True
    return False


def fetch_names(session) -> List[str]:
    """The primary documents' names and aliases, from a read session."""
    out: List[str] = []
    for r in session.run(NAMES_QUERY):
        out.append(str(r["name"] or ""))
        out += [str(a) for a in r["aliases"] or []]
    return [x for x in out if x]


def set_names(names: Optional[Sequence[str]]) -> None:
    """The cache (None: the graph could not be read -- nothing is missing)."""
    with _LOCK:
        _NAMES["names"] = list(names) if names is not None else None


def names() -> Optional[List[str]]:
    """The cached names, read on first use when warm() did not run; None when the graph could not be read."""
    if "names" not in _NAMES:
        warm()
    return _NAMES.get("names")


def warm() -> Optional[List[str]]:
    """Read the names once (the API's startup); logged, never raises."""
    try:
        import neo4j
        from .main import NEO4J_DATABASE, driver
        with driver.session(database=NEO4J_DATABASE, default_access_mode=neo4j.READ_ACCESS) as session:
            got = fetch_names(session)
        set_names(got)
        logger.info("corpus_laws: %d primary document names cached", len(got))
    except Exception as exc:        # noqa: BLE001 -- unknown: no line is printed
        logger.warning("corpus_laws: the primary documents' names not read: %s", exc)
        set_names(None)
    return _NAMES.get("names")


def missing(laws: Sequence[str], known: Optional[Sequence[str]] = None) -> Optional[List[str]]:
    """The laws of the list absent from the graph; None when the names are unknown."""
    known = names() if known is None else known
    if known is None:
        return None
    return [x for x in laws or [] if not present(x, known)]


def gaps(known: Sequence[str]) -> Dict[str, Any]:
    """Per module and for the defaults, the expected laws missing from the names given (corpus_gaps.py)."""
    exp = expected()
    mods = {m: [x for x in laws if not present(x, known)] for m, laws in (exp.get("modules") or {}).items()}
    return {"modules": mods, "defaults": [x for x in exp.get("defaults") or [] if not present(x, known)],
            "names_read": len(known)}
