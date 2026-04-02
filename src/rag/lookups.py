"""Neo4j lookup strategies: B-tree, full-text, and vector search."""

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Pattern, Set, Tuple

from neo4j.exceptions import Neo4jError

from .ai_chat import embedding_model
from .lookup_indexes import (
    BTREE_LOOKUPS,
    FULLTEXT_INDEXES,
    VECTOR_K,
)
from .utils import canonical_name

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants & patterns
# ---------------------------------------------------------------------------

DEFAULT_FULLTEXT_LIMIT = 3
GENERIC_REFERENCE_PATTERN = re.compile(r"\bno\.?\b", re.IGNORECASE)
_AL_PREFIX_PATTERN = re.compile(r"^al[\s\-]+", re.IGNORECASE)

LEGAL_ACT_TYPE_ALIASES: Dict[str, str] = {
    "decree": "Decree",
    "amiri decree": "Decree",
    "emiri decree": "Decree",
    "decree law": "DecreeLaw",
    "decision": "Decision",
    "ministerial resolution": "MinisterialResolution",
    "committee resolution": "CommitteeResolution",
    "regulation": "Regulation",
}

VECTOR_INDEX_SETTINGS: Dict[str, Dict[str, Any]] = {
    "company_embeddings": {"k": 1, "min_score": 0.35},
    "legalparty_embeddings": {"k": 1, "min_score": 0.35},
    "institution_embeddings": {"k": 1, "min_score": 0.3},
    "legalact_embeddings": {"k": 2, "min_score": 0.3},
    "document_embeddings": {"min_score": 0.25},
    "section_embeddings": {"min_score": 0.25},
    "tender_embeddings": {"k": 2, "min_score": 0.28},
    "contract_embeddings": {"k": 2, "min_score": 0.28},
    "penalty_embeddings": {"k": 1, "min_score": 0.32},
    "award_embeddings": {"k": 2, "min_score": 0.28},
}

LABEL_VECTOR_HINTS: Dict[str, List[str]] = {
    "LegalAct": ["legalact_embeddings"],
    "Resolution": ["legalact_embeddings"],
    "Institution": ["institution_embeddings"],
    "Company": ["company_embeddings"],
    "Person": ["person_embeddings"],
    "Court": ["court_embeddings"],
    "CourtCase": ["courtcase_embeddings"],
    "LegalParty": ["legalparty_embeddings"],
    "Tender": ["tender_embeddings"],
    "Award": ["award_embeddings"],
    "Contract": ["contract_embeddings"],
    "ChangeOrder": ["changeorder_embeddings"],
    "Auction": ["auction_embeddings"],
    "Penalty": ["penalty_embeddings"],
    "Meeting": ["document_embeddings"],
    "Article": ["article_embeddings"],
    "Section": ["section_embeddings"],
    "Clause": ["clause_embeddings"],
    "Document": ["document_embeddings"],
    "Complaint": ["complaint_embeddings"],
    "Vote": ["vote_embeddings"],
    "Correction": ["correction_embeddings"],
    "Addendum": ["addendum_embeddings"],
    "Topic": ["topic_embeddings"],
}

DEFAULT_VECTOR_INDEXES = ["document_embeddings"]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParsedLegalAct:
    act_type: str
    act_number: int
    act_year: int


@dataclass
class EntityLookupHints:
    allowed_labels: Optional[Set[str]]
    vector_indexes: List[str]
    fulltext_indexes: List[str]
    fulltext_limit: int
    legal_act: Optional[ParsedLegalAct]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _strip_al_prefix(value: str) -> str:
    return _AL_PREFIX_PATTERN.sub("", value).strip()


def _build_btree_candidates(value: str, property_name: str) -> List[str]:
    raw = (value or "").strip()
    if not raw:
        return []

    candidates: List[str] = []
    seen: Set[str] = set()

    def _add(candidate: Optional[str]) -> None:
        if not candidate:
            return
        lowered = candidate.lower()
        if lowered and lowered not in seen:
            seen.add(lowered)
            candidates.append(lowered)

    _add(raw)

    if property_name in {"normalized_name"}:
        canonical = canonical_name(raw)
        _add(canonical)
        if canonical:
            _add(canonical.replace("-", " "))
            _add(_strip_al_prefix(canonical))
            sans_hyphen = canonical.replace("-", " ")
            _add(_strip_al_prefix(sans_hyphen))

    return candidates


def _unique_preserve_order(values: Iterable) -> List[str]:
    return list(dict.fromkeys(values))


def _normalize_numeric_token(token: Optional[str]) -> Optional[int]:
    if token is None:
        return None
    cleaned = re.sub(r"[^0-9]", "", token)
    if not cleaned:
        return None
    try:
        return int(cleaned)
    except ValueError:
        return None


def _build_legal_act_patterns(keyword: str) -> List[Pattern[str]]:
    escaped = re.escape(keyword)
    return [
        re.compile(
            rf"{escaped}\s*(?:no\.?|number)?\s*(?P<number>[0-9()]+)\s*/\s*(?P<year>\d{{4}})",
            re.IGNORECASE,
        ),
        re.compile(
            rf"{escaped}\s*(?:no\.?|number)?\s*(?P<number>[0-9()]+)\s+(?:of|for\s+the\s+year|for\s+year|year)\s*(?P<year>\d{{4}})",
            re.IGNORECASE,
        ),
        re.compile(
            rf"{escaped}\s*(?:no\.?|number)?\s*(?P<number>[0-9()]+)\s*\(?(?P<year>\d{{4}})\)?",
            re.IGNORECASE,
        ),
    ]


def _parse_legal_act_reference(text: str) -> Optional[ParsedLegalAct]:
    candidate = text.strip()
    if not candidate:
        return None
    lowered = candidate.lower()
    for keyword, normalized_type in LEGAL_ACT_TYPE_ALIASES.items():
        if keyword not in lowered:
            continue
        for pattern in _build_legal_act_patterns(keyword):
            match = pattern.search(candidate)
            if not match:
                continue
            number = _normalize_numeric_token(match.group("number"))
            year = _normalize_numeric_token(match.group("year"))
            if number is None or year is None:
                continue
            return ParsedLegalAct(
                act_type=normalized_type,
                act_number=number,
                act_year=year,
            )
    return None


# ---------------------------------------------------------------------------
# Entity hint computation
# ---------------------------------------------------------------------------

def compute_entity_hints(entity: str) -> EntityLookupHints:
    normalized = entity.strip()
    lowered = normalized.lower()

    allowed_labels: Set[str] = set()
    vector_indexes: List[str] = []
    fulltext_indexes: List[str] = []
    fulltext_limit = DEFAULT_FULLTEXT_LIMIT

    def add_label_hint(label: str) -> None:
        allowed_labels.add(label)
        vector_indexes.extend(LABEL_VECTOR_HINTS.get(label, []))

    legal_act = _parse_legal_act_reference(normalized)
    if legal_act:
        add_label_hint("LegalAct")
        fulltext_indexes.append("legal_text_keywords")
        fulltext_limit = min(fulltext_limit, 2)

    if any(
        keyword in lowered
        for keyword in ("decree", "decision", "regulation", "resolution", "law")
    ):
        add_label_hint("LegalAct")
        if "resolution" in lowered:
            add_label_hint("Resolution")
        fulltext_indexes.append("legal_text_keywords")

    if "bank" in lowered:
        add_label_hint("Institution")
        fulltext_indexes.extend(["entity_names", "legal_text_keywords"])

    if re.search(
        r"\b(ministry|authority|agency|commission|council|committee|department)\b",
        lowered,
    ):
        add_label_hint("Institution")
        fulltext_indexes.append("entity_names")

    if re.search(r"\b(company|co\.?|corp|corporation|ltd|inc|holding)\b", lowered):
        add_label_hint("Company")
        fulltext_indexes.append("entity_names")

    if "court" in lowered:
        add_label_hint("Court")
        fulltext_indexes.append("entity_names")

    if re.search(r"court case|case\s+no\.", lowered):
        add_label_hint("CourtCase")
        fulltext_indexes.append("legal_text_keywords")

    if "tender" in lowered:
        add_label_hint("Tender")
        fulltext_indexes.append("event_keywords")

    if "contract" in lowered:
        add_label_hint("Contract")
        fulltext_indexes.append("event_keywords")

    if "penalty" in lowered:
        add_label_hint("Penalty")
        fulltext_indexes.append("event_keywords")

    if "auction" in lowered:
        add_label_hint("Auction")
        fulltext_indexes.append("event_keywords")

    if "award" in lowered:
        add_label_hint("Award")
        fulltext_indexes.append("event_keywords")

    if "meeting" in lowered or "session" in lowered:
        add_label_hint("Meeting")
        fulltext_indexes.append("event_keywords")

    if re.search(r"\barticle\s+\d+", lowered):
        add_label_hint("Article")
        fulltext_indexes.append("legal_text_keywords")

    if re.search(r"\bsection\s+\d+", lowered):
        add_label_hint("Section")
        fulltext_indexes.append("legal_text_keywords")

    if re.search(r"\bclause\s+\d+", lowered):
        add_label_hint("Clause")
        fulltext_indexes.append("legal_text_keywords")

    if not vector_indexes:
        vector_indexes = list(DEFAULT_VECTOR_INDEXES)
    else:
        vector_indexes = _unique_preserve_order(vector_indexes)

    if not fulltext_indexes:
        fulltext_indexes = ["entity_names", "legal_text_keywords", "event_keywords"]

    fulltext_indexes = [
        index
        for index in _unique_preserve_order(fulltext_indexes)
        if index in FULLTEXT_INDEXES
    ]

    if GENERIC_REFERENCE_PATTERN.search(normalized) and not any(
        keyword in lowered
        for keyword in ("decree", "decision", "regulation", "resolution", "law")
    ):
        fulltext_indexes = [idx for idx in fulltext_indexes if idx != "entity_names"]

    if allowed_labels and "LegalAct" in allowed_labels:
        fulltext_indexes = [
            idx
            for idx in fulltext_indexes
            if idx != "entity_names" or "Institution" in allowed_labels
        ]

    return EntityLookupHints(
        allowed_labels=allowed_labels or None,
        vector_indexes=vector_indexes,
        fulltext_indexes=fulltext_indexes,
        fulltext_limit=fulltext_limit,
        legal_act=legal_act,
    )


# ---------------------------------------------------------------------------
# Neo4j lookup functions
# ---------------------------------------------------------------------------

def legal_act_lookup(
    session, parsed: Optional[ParsedLegalAct]
) -> Iterable[Dict[str, Any]]:
    if not parsed:
        return []

    logger.debug(
        "Deterministic LegalAct lookup",
        extra={
            "act_type": parsed.act_type,
            "act_number": parsed.act_number,
            "act_year": parsed.act_year,
        },
    )

    queries: List[Tuple[str, Dict[str, Any]]] = [
        (
            (
                "MATCH (n:LegalAct) "
                "WHERE n.act_type = $act_type "
                "AND n.act_number = $act_number "
                "AND n.act_year = $act_year "
                "RETURN elementId(n) AS element_id, labels(n) AS labels"
            ),
            {
                "act_type": parsed.act_type,
                "act_number": parsed.act_number,
                "act_year": parsed.act_year,
            },
        ),
        (
            (
                "MATCH (n:LegalAct) "
                "WHERE n.act_type = $act_type "
                "AND toString(n.act_number) = $act_number "
                "AND toString(n.act_year) = $act_year "
                "RETURN elementId(n) AS element_id, labels(n) AS labels"
            ),
            {
                "act_type": parsed.act_type,
                "act_number": str(parsed.act_number),
                "act_year": str(parsed.act_year),
            },
        ),
    ]

    seen_ids: Set[str] = set()
    for query, params in queries:
        try:
            records = session.run(query, **params)
        except Neo4jError as exc:
            logger.warning(
                "LegalAct deterministic lookup failed",
                extra={"params": params, "error": str(exc)},
            )
            continue
        for record in records:
            element_id = record["element_id"]
            if element_id in seen_ids:
                continue
            seen_ids.add(element_id)
            yield {
                "element_id": element_id,
                "labels": record["labels"],
                "source": "legalact:key",
            }


def btree_lookup(
    session, value: str, *, allowed_labels: Optional[Set[str]] = None
) -> Iterable[Dict[str, Any]]:
    for config in BTREE_LOOKUPS:
        if allowed_labels and config.label not in allowed_labels:
            continue
        candidates = _build_btree_candidates(value, config.property)
        if not candidates:
            continue
        query = (
            f"MATCH (n:{config.label}) "
            f"WHERE toLower(n.{config.property}) IN $candidates "
            "RETURN elementId(n) AS element_id, labels(n) AS labels"
        )
        try:
            records = session.run(query, candidates=candidates)
        except Neo4jError as exc:
            logger.warning(
                "B-tree lookup failed",
                extra={
                    "entity": value,
                    "label": config.label,
                    "property": config.property,
                    "error": str(exc),
                },
            )
            continue
        for record in records:
            yield {
                "element_id": record["element_id"],
                "labels": record["labels"],
                "source": "btree",
            }


def fulltext_lookup(
    session,
    value: str,
    *,
    indexes: Optional[List[str]] = None,
    limit: int = DEFAULT_FULLTEXT_LIMIT,
    allowed_labels: Optional[Set[str]] = None,
) -> Iterable[Dict[str, Any]]:
    limit_value = max(1, limit)
    indexes_to_use = indexes or FULLTEXT_INDEXES
    matches: List[Dict[str, Any]] = []
    for index_name in indexes_to_use:
        label_clause = (
            "WHERE any(label IN labels(node) WHERE label IN $labels) "
            if allowed_labels
            else ""
        )
        query = (
            "CALL db.index.fulltext.queryNodes($index, $text) "
            "YIELD node, score "
            f"{label_clause}"
            "RETURN elementId(node) AS element_id, labels(node) AS labels, score "
            "ORDER BY score DESC "
            f"LIMIT {limit_value}"
        )
        params: Dict[str, Any] = {"index": index_name, "text": value}
        if allowed_labels:
            params["labels"] = list(allowed_labels)
        try:
            records = session.run(query, **params)
        except Neo4jError as exc:
            logger.warning(
                "Full-text lookup failed",
                extra={
                    "entity": value,
                    "index": index_name,
                    "error": str(exc),
                },
            )
            continue
        matches.extend(
            {
                "element_id": record["element_id"],
                "labels": record["labels"],
                "source": f"fulltext:{index_name}",
                "score": record.get("score"),
            }
            for record in records
        )
    return matches


def vector_lookup(
    session,
    value: str,
    *,
    indexes: List[str],
    index_settings: Optional[Dict[str, Dict[str, Any]]] = None,
    k: int = VECTOR_K,
    source_prefix: str = "vector",
) -> Iterable[Dict[str, Any]]:
    try:
        embedding = embedding_model.embed_query(value)
    except Exception as exc:
        logger.warning(
            "Embedding generation failed",
            extra={"value": value, "error": str(exc)},
        )
        return []

    query = (
        "CALL db.index.vector.queryNodes($index, $k, $embedding) "
        "YIELD node, score "
        "RETURN elementId(node) AS element_id, labels(node) AS labels, score ORDER BY score DESC"
    )
    matches: List[Dict[str, Any]] = []
    for index in indexes:
        config = (index_settings or {}).get(index, {})
        k_value = max(1, int(config.get("k", k)))
        min_score = config.get("min_score")
        try:
            records = session.run(query, index=index, k=k_value, embedding=embedding)
        except Neo4jError as exc:
            logger.warning(
                "Vector lookup failed",
                extra={
                    "value": value,
                    "index": index,
                    "k": k_value,
                    "error": str(exc),
                },
            )
            continue
        for record in records:
            score = record.get("score")
            if min_score is not None and score is not None and score < min_score:
                continue
            matches.append(
                {
                    "element_id": record["element_id"],
                    "labels": record["labels"],
                    "source": f"{source_prefix}:{index}",
                    "score": score,
                }
            )
    return matches
