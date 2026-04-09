"""LangGraph node functions for the RAG agent pipeline."""

import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed
from typing import Any, Dict, List, Optional, Set

from langchain_core.messages import HumanMessage, SystemMessage
from neo4j.exceptions import Neo4jError

from ..preprocessing.schema.schema import entities as schema_entities
from ..preprocessing.schema.schema import relations as schema_relations
from .ai_chat import _call_chat, invoke_entity_extraction
from .cypher_logger import log_cypher_event, log_cypher_multiline
from .language import SessionLang, normalize_lang
from .prompts import (
    legal_consultant_system_prefix,
    synthesis_empty_system,
    synthesis_error_system,
    synthesis_human_footer,
    synthesis_system_message,
)
from .lookup_indexes import (
    CONTEXT_NODE_LIMIT,
    CONTEXT_VECTOR_INDEXES,
    ENTITY_VECTOR_INDEXES,
    FULLTEXT_INDEXES,
)
from .lookups import (
    LABEL_VECTOR_HINTS,
    VECTOR_INDEX_SETTINGS,
    ParsedLegalAct,
    btree_lookup,
    fulltext_lookup,
    legal_act_lookup,
    vector_lookup,
)
from .models import DocumentEntities
from .utils import (
    _build_schema_text,
    _clean_cypher,
    _enforce_relation_directions,
    canonical_name,
)

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# Optional bypass for context retrieval input:
# when enabled, we ignore generalized/user query and retrieve context from a fixed phrase.
RAG_USE_DEFAULT_CONTEXT = _env_bool("RAG_USE_DEFAULT_CONTEXT", False)
RAG_DEFAULT_CONTEXT_QUERY = os.getenv("RAG_DEFAULT_CONTEXT_QUERY", "leggi italiane").strip()
# If disabled, skip LLM query optimization and use user phrase directly as keyword seed.
CYPHER_OPTIMIZATION_ENABLED = _env_bool("CYPHER_OPTIMIZATION_ENABLED", True)
KEYWORD_EXTRACTION_ENABLED = _env_bool("KEYWORD_EXTRACTION_ENABLED", True)
DECOMPOSE_LLM_ENABLED = _env_bool("DECOMPOSE_LLM_ENABLED", True)
RAG_DEFAULT_KEYWORDS = [
    k.strip()
    for k in os.getenv("RAG_DEFAULT_KEYWORDS", "normativa,legge,decreto").split(",")
    if k.strip()
]
CYTHER_GENERATION_TIMEOUT_SECONDS = float(
    os.getenv("CYPHER_GENERATION_TIMEOUT_SECONDS", "120")
)
CYTHER_FALLBACK_KEYWORD = os.getenv("CYPHER_FALLBACK_KEYWORD", "codice civile").strip() or "codice civile"
_ALLOWED_VECTOR_INDEXES = set(CONTEXT_VECTOR_INDEXES + ENTITY_VECTOR_INDEXES)
_VECTOR_QUERY_INDEX_RE = re.compile(
    r"db\.index\.vector\.queryNodes\(\s*['\"]([^'\"]+)['\"]",
    re.IGNORECASE,
)


def _strip_reasoning_and_markup(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""
    low = raw.lower()
    marker = "</think>"
    if marker in low:
        idx = low.rfind(marker)
        raw = raw[idx + len(marker):].strip()
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
    return raw


def _sanitize_generalized_phrase(raw_text: str, fallback_query: str) -> str:
    cleaned = _strip_reasoning_and_markup(raw_text)
    if not cleaned:
        cleaned = (fallback_query or "").strip()
    first_line = cleaned.splitlines()[0].strip() if cleaned else ""
    if not first_line:
        first_line = (fallback_query or "").strip()
    # Keep compact: max 8 words as requested by the prompt.
    words = first_line.split()
    if len(words) > 8:
        first_line = " ".join(words[:8])
    return first_line


def _escape_cypher_string(value: str) -> str:
    return (value or "").replace("\\", "\\\\").replace("'", "\\'")


def _select_fallback_keyword(state: Dict[str, Any]) -> str:
    keywords = state.get("retrieval_keywords") or []
    for kw in keywords:
        clean = (kw or "").strip()
        if clean:
            return clean
    generalized = (state.get("generalized_query") or "").strip()
    if generalized:
        return generalized
    return CYTHER_FALLBACK_KEYWORD


def _base_fallback_cypher() -> str:
    # Requested emergency fallback query when normal Cypher generation is missing/late.
    return (
        "MATCH (n)\n"
        "WHERE (n:Legal_doc OR n:Legal_action)\n"
        "AND (\n"
        "    toLower(n.title) CONTAINS toLower($keyword) OR\n"
        "    toLower(n.description) CONTAINS toLower($keyword) OR\n"
        "    toLower(n.text) CONTAINS toLower($keyword)\n"
        ")\n"
        "RETURN n;"
    )


def _fallback_cypher_payload(state: Dict[str, Any], reason: str, attempt: str) -> Dict[str, Any]:
    keyword = _select_fallback_keyword(state)
    cypher = _base_fallback_cypher()
    log_cypher_event(
        "b_fallback_base",
        f"{attempt}: using base fallback Cypher",
        detail={"reason": reason, "keyword": keyword},
    )
    log_cypher_multiline(
        "b_draft",
        f"{attempt}: base fallback Cypher (timeout/empty generation guard)",
        cypher,
    )
    return {
        "cypher_query": cypher,
        "cypher_params": {"keyword": keyword},
        "cypher_generation_error": None,
        "cypher_attempt": attempt,
    }


def _call_chat_timeout(messages: List[Any], llm_context: Optional[Dict[str, Any]] = None) -> str:
    with ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(_call_chat, messages, llm_context)
        return fut.result(timeout=CYTHER_GENERATION_TIMEOUT_SECONDS)


def _session_lang(state: Dict[str, Any]) -> SessionLang:
    return normalize_lang(state.get("session_language"))


_ENTITY_LABELS_TEXT = ", ".join(DocumentEntities.allowed_labels())
SCHEMA_TEXT = _build_schema_text()
RELATION_HINTS = "\n".join(
    f"- {item['from']} -[:{item['type']}]-> {item['to']}" for item in schema_relations
)


# ---------------------------------------------------------------------------
# Node A: Query decomposition
# ---------------------------------------------------------------------------

def decompose_query(state: Dict[str, Any]) -> Dict[str, Any]:
    query = state["query"]
    lang = _session_lang(state)
    logger.info("Starting query decomposition", extra={"query": query})

    log_cypher_multiline(
        "a_query",
        "user question (verbatim — start of RAG pipeline)",
        query,
        delimiter_label="USER_QUESTION",
    )

    if not DECOMPOSE_LLM_ENABLED:
        generalized = (query or "").strip()
        retrieval_keywords = (RAG_DEFAULT_KEYWORDS[:3] or [generalized][:1])
        log_cypher_event(
            "a_generalized",
            "decompose LLM disabled: using raw user phrase as generalized query",
            detail=generalized,
        )
        log_cypher_event(
            "a_keywords",
            "decompose LLM disabled: using configured default keywords",
            detail=retrieval_keywords,
        )
        return {
            **state,
            "generalized_query": generalized,
            "retrieval_keywords": retrieval_keywords,
            # No LLM extraction => no entities/relationships from query text.
            "entities": [],
            "extracted_relationships": [],
        }

    entity_extraction_prompt = (
        "Schema:\n{schema}\n\n"
        "Based on the schema, extract a graph of nodes and relationships from the following question.\n"
        "Question: {query}\n\n"
        "Instructions:\n"
        "1. Identify all distinct entities (nodes). For each node, you MUST assign a temporary `id` (e.g., 'node1', 'node2').\n"
        "2. For each node, you MUST include a `label` and a `properties` field. The `properties` field can be an empty object (`{{}}`) if no specific properties are mentioned.\n"
        "3. Populate the `properties` object according to the schema:\n"
        "   - For 'Company', 'Institution', 'Person', 'Court', or 'LegalParty', extract its full name into the 'name' property.\n"
        "   - For a 'LegalAct', extract 'act_type', 'act_number', 'act_year'.\n"
        "   - For a 'Document', extract 'issue_number', 'document_title', 'document_date'.\n"
        "4. Identify relationships between nodes. The 'type' must be one of the types defined in the schema for the given source and target nodes.\n"
        "5. Format the output as a single JSON object.\n\n"
        "Example:\n"
        "Question: 'Who was appointed by the Ministry of Oil in Decree No. 46 of 2025?'\n"
        'Result: {{"graph": {{"nodes": ['
        '{{"id": "node1", "label": "Person", "properties": {{\'role\': \'Undersecretary\'}}}},'
        '{{"id": "node2", "label": "Institution", "properties": {{"name": "Ministry of Oil"}}}},'
        '{{"id": "node3", "label": "LegalAct", "properties": {{"act_type": "Decree", "act_number": "46", "act_year": "2025"}}}}'
        '], "relationships": ['
        '{{"source_id": "node2", "target_id": "node1", "type": "APPOINTS"}},'
        '{{"source_id": "node3", "target_id": "node1", "type": "APPOINTS"}}'
        "]}}}}"
    ).format(schema=SCHEMA_TEXT, query=query)
    entity_extraction_messages = [
        SystemMessage(
            content=(
                f"{legal_consultant_system_prefix(lang)} "
                "You are an expert graph extractor for legal documents. "
                "Identify nodes and relationships from the user's query based on the provided graph schema."
            )
        ),
        HumanMessage(content=entity_extraction_prompt),
    ]
    if CYPHER_OPTIMIZATION_ENABLED:
        # Step 1 (generalize) and Step 2 (entity extraction) are independent —
        # run them in parallel to cut decomposition latency roughly in half.
        generalization_messages = [
            SystemMessage(
                content=(
                    f"{legal_consultant_system_prefix(lang)} "
                    "You generalize user questions about legal documents into concise search-focused phrases."
                )
            ),
            HumanMessage(
                content=(
                    "Original question: {query}\n"
                    "Respond with a short generalized phrase (max 8 words) capturing the main topic."
                ).format(query=query)
            ),
        ]

        with ThreadPoolExecutor(max_workers=2) as pool:
            future_generalize = pool.submit(_call_chat, generalization_messages)
            future_entities = pool.submit(
                invoke_entity_extraction, entity_extraction_messages
            )

            generalized_raw = future_generalize.result()
            entities_payload = future_entities.result()
            generalized = _sanitize_generalized_phrase(generalized_raw, query)

        logger.info(f"Generalized query: '{generalized}'")
        log_cypher_event(
            "a_generalized",
            "generalized topic phrase (used for context / vector retrieval)",
            detail=generalized,
        )

        # Step 1b: Keywords (1–3) — depends on generalized, so runs after.
        # Can be disabled to bypass additional LLM call.
        if KEYWORD_EXTRACTION_ENABLED:
            kw_raw = _call_chat(
                [
                    SystemMessage(
                        content=(
                            f"{legal_consultant_system_prefix(lang)} "
                            "Extract one to three keywords or short noun phrases that capture the core legal subject matter. "
                            "Output only a comma-separated list, no numbering or extra text."
                        )
                    ),
                    HumanMessage(
                        content=f"Question:\n{query}\n\nGeneralized topic:\n{generalized}\n\nKeywords:"
                    ),
                ]
            )
            kw_clean = _strip_reasoning_and_markup(kw_raw or "")
            retrieval_keywords = [k.strip() for k in kw_clean.split(",") if k.strip()][:3]
            if not retrieval_keywords:
                retrieval_keywords = [generalized] if generalized else [query]
            log_cypher_event(
                "a_keywords",
                "extracted keywords",
                detail=retrieval_keywords,
            )
        else:
            retrieval_keywords = (RAG_DEFAULT_KEYWORDS[:3] or [generalized][:1])
            log_cypher_event(
                "a_keywords",
                "keyword extraction disabled: using configured default keywords",
                detail=retrieval_keywords,
            )
    else:
        # Bypass optimization: keep original phrase as-is for Cypher keywording.
        generalized = query
        retrieval_keywords = [query.strip()] if query and query.strip() else []
        entities_payload = invoke_entity_extraction(entity_extraction_messages)
        log_cypher_event(
            "a_generalized",
            "cypher optimization disabled: using raw user phrase as generalized query",
            detail=generalized,
        )
        log_cypher_event(
            "a_keywords",
            "cypher optimization disabled: using full user phrase as keyword seed",
            detail=retrieval_keywords,
        )

    # Step 3: Validate and normalize the extracted graph
    raw_graph = entities_payload.graph

    schema_nodes = {
        item["label"]: set(item["properties"]) | set(item["key"])
        for item in schema_entities
    }
    schema_rels = {
        (item["from"], item["type"]): item["to"] for item in schema_relations
    }

    valid_nodes = {}
    temp_id_to_label = {}
    nodes_to_discard = set()

    for node in raw_graph.nodes:
        node_dict = node.model_dump()
        temp_id = node_dict.get("id")
        label = node_dict.get("label")
        properties = node_dict.get("properties", {})

        if label not in schema_nodes:
            logger.warning(f"Invalid node label '{label}'. Discarding node {temp_id}.")
            nodes_to_discard.add(temp_id)
            continue

        valid_properties = {
            prop: value
            for prop, value in properties.items()
            if prop in schema_nodes[label] or prop == "name"
        }

        node_dict["properties"] = valid_properties
        valid_nodes[temp_id] = node_dict
        temp_id_to_label[temp_id] = label

    valid_relationships = []
    for rel in raw_graph.relationships:
        rel_dict = rel.model_dump()
        source_id = rel_dict.get("source_id")
        target_id = rel_dict.get("target_id")

        if source_id in nodes_to_discard or target_id in nodes_to_discard:
            continue

        source_label = temp_id_to_label.get(source_id)
        target_label = temp_id_to_label.get(target_id)
        rel_type = rel_dict.get("type")

        if not all([source_label, target_label, rel_type]):
            continue

        if (source_label, rel_type) not in schema_rels or schema_rels.get(
            (source_label, rel_type)
        ) != target_label:
            logger.warning(
                f"Invalid relationship '{source_label}-[:{rel_type}]->{target_label}'. Discarding."
            )
            continue

        valid_relationships.append(rel_dict)

    # Pass 2: Post-process and normalize
    labels_with_normalized_name = {
        "Company", "Institution", "Person", "Court", "LegalParty",
    }

    final_valid_nodes = {}
    for temp_id, node in valid_nodes.items():
        if temp_id in nodes_to_discard:
            continue

        label = node.get("label")
        properties = node.get("properties", {}).copy()

        if label in labels_with_normalized_name and "name" in properties:
            raw_name = properties.pop("name")
            if raw_name:
                properties["normalized_name"] = canonical_name(raw_name)

        key_properties = set(
            next(
                (item["key"] for item in schema_entities if item["label"] == label), []
            )
        )
        if key_properties and not key_properties.issubset(properties.keys()):
            if not properties or len(properties) == 0:
                logger.warning(f"Node {temp_id} ('{label}') has no properties. Discarding.")
                nodes_to_discard.add(temp_id)
                continue
            else:
                logger.info(
                    f"Node {temp_id} ('{label}') missing key properties but has: {list(properties.keys())}. Keeping as type hint."
                )

        node["properties"] = properties
        final_valid_nodes[temp_id] = node

    processed_entities = list(final_valid_nodes.values())
    final_relationships = valid_relationships

    logger.info(
        "Decomposed query: generalized='%s', entities=%d, relationships=%d",
        generalized,
        len(processed_entities),
        len(final_relationships),
    )

    return {
        **state,
        "generalized_query": generalized,
        "retrieval_keywords": retrieval_keywords,
        "entities": processed_entities,
        "extracted_relationships": final_relationships,
    }


# ---------------------------------------------------------------------------
# Node B: Entity linking
# ---------------------------------------------------------------------------

def entity_linking(state: Dict[str, Any], driver, database: str) -> Dict[str, Any]:
    extracted = state.get("entities", [])
    if not extracted:
        logger.warning("Entity linking skipped: no extracted entities present")
        return {"entry_nodes": []}

    entries: Dict[str, Dict[str, Any]] = {}
    node_id_map: Dict[str, str] = {}

    with driver.session(database=database) as session:

        def merge_entry(match: Dict[str, Any], entity: Dict[str, Any]) -> None:
            element_id = match["element_id"]
            temp_id = entity.get("id")
            if temp_id and temp_id not in node_id_map:
                node_id_map[temp_id] = element_id
            if element_id in entries:
                entries[element_id]["sources"].add(match.get("source", "unknown"))
            else:
                entries[element_id] = {
                    "element_id": element_id,
                    "labels": match.get("labels", []),
                    "sources": {match.get("source", "unknown")},
                    "entity_props": entity.get("properties", {}),
                }

        for entity in extracted:
            label = entity.get("label")
            properties = entity.get("properties", {})
            if not label or not properties:
                continue

            precise_match_found = False

            # LegalAct composite key
            if label == "LegalAct" and all(
                k in properties for k in ["act_type", "act_number", "act_year"]
            ):
                parsed = ParsedLegalAct(
                    act_type=properties["act_type"],
                    act_number=properties["act_number"],
                    act_year=properties["act_year"],
                )
                for match in legal_act_lookup(session, parsed):
                    merge_entry(match, entity)
                    precise_match_found = True
                if precise_match_found:
                    continue

            # Composite key lookups for Article, Clause, CourtCase, Section
            composite_lookups = {
                "Article": (["parent_act_key", "index"], "parent_act_key", "index"),
                "Clause": (["parent_article_key", "index"], "parent_article_key", "index"),
                "CourtCase": (["document_id", "chunk_id"], "document_id", "chunk_id"),
                "Section": (["document_id", "chunk_id"], "document_id", "chunk_id"),
            }
            if label in composite_lookups:
                keys, *_ = composite_lookups[label]
                if all(k in properties for k in keys):
                    query = f"MATCH (n:{label}) WHERE " + " AND ".join(
                        f"n.{k} = ${k}" for k in keys
                    ) + " RETURN elementId(n) AS element_id, labels(n) AS labels"
                    try:
                        records = session.run(query, **{k: properties[k] for k in keys})
                        for record in records:
                            merge_entry(
                                {
                                    "element_id": record["element_id"],
                                    "labels": record["labels"],
                                    "source": f"btree:composite_{label.lower()}",
                                },
                                entity,
                            )
                            precise_match_found = True
                        if precise_match_found:
                            continue
                    except Neo4jError as exc:
                        logger.warning(f"{label} composite lookup failed: {exc}")

            # Simple ID lookups
            id_key_map = {
                "Penalty": "penalty_id", "Contract": "contract_id",
                "Tender": "tender_id", "Award": "award_id",
                "Meeting": "meeting_id", "Auction": "auction_id",
                "Asset": "asset_id", "Document": "document_id",
                "Resolution": "resolution_id", "Complaint": "complaint_id",
                "Vote": "vote_id", "Correction": "correction_id",
                "Addendum": "addendum_id", "ChangeOrder": "change_order_id",
            }

            if label in id_key_map:
                id_key = id_key_map[label]
                if id_key in properties:
                    query = (
                        f"MATCH (n:{label}) WHERE n.{id_key} = ${id_key} "
                        "RETURN elementId(n) AS element_id, labels(n) AS labels"
                    )
                    try:
                        records = session.run(query, **{id_key: properties[id_key]})
                        for record in records:
                            merge_entry(
                                {
                                    "element_id": record["element_id"],
                                    "labels": record["labels"],
                                    "source": f"btree:id_{label.lower()}",
                                },
                                entity,
                            )
                            precise_match_found = True
                        if precise_match_found:
                            continue
                    except Neo4jError as exc:
                        logger.warning(f"{label} ID lookup failed: {exc}")

            # B-tree property lookup
            from .lookup_indexes import BTREE_LOOKUPS
            for prop_name, prop_value in properties.items():
                btree_config = next(
                    (c for c in BTREE_LOOKUPS if c.label == label and c.property == prop_name),
                    None,
                )
                if btree_config:
                    for match in btree_lookup(session, prop_value, allowed_labels={label}):
                        merge_entry(match, entity)
                        precise_match_found = True

            if precise_match_found:
                continue

            # Fallback: vector/fulltext on property values
            search_value = " ".join(str(v) for v in properties.values())

            vector_indexes = LABEL_VECTOR_HINTS.get(label, [])
            if vector_indexes:
                for match in vector_lookup(
                    session, search_value, indexes=vector_indexes,
                    index_settings=VECTOR_INDEX_SETTINGS, source_prefix="vector_targeted",
                ):
                    merge_entry(match, entity)

            fulltext_indexes = [idx for idx in FULLTEXT_INDEXES if label in idx]
            if fulltext_indexes:
                for match in fulltext_lookup(
                    session, search_value, indexes=fulltext_indexes, allowed_labels={label},
                ):
                    merge_entry(match, entity)

    entry_nodes = [
        {**entry, "sources": sorted(list(entry["sources"]))}
        for entry in entries.values()
    ]
    logger.info("Entity linking produced %d entry nodes", len(entry_nodes))

    # Fallback with generalized query if no entries found
    if not entry_nodes:
        generalized_query = state.get("generalized_query")
        if generalized_query:
            with driver.session(database=database) as session:
                fallback_matches = vector_lookup(
                    session, generalized_query,
                    indexes=CONTEXT_VECTOR_INDEXES,
                    index_settings=VECTOR_INDEX_SETTINGS,
                    source_prefix="context_fallback",
                )

            aggregated: Dict[str, Dict[str, Any]] = {}
            for match in fallback_matches:
                element_id = match["element_id"]
                existing = aggregated.get(element_id)
                if not existing:
                    aggregated[element_id] = {
                        "element_id": element_id,
                        "labels": match.get("labels", []),
                        "sources": {match.get("source", "unknown")},
                        "score": match.get("score"),
                    }
                else:
                    existing["sources"].add(match.get("source", "unknown"))
                    score = match.get("score")
                    if score is not None and (existing.get("score") is None or score > existing["score"]):
                        existing["score"] = score

            sorted_nodes = sorted(
                aggregated.values(),
                key=lambda item: item.get("score") or 0,
                reverse=True,
            )

            for node_data in sorted_nodes[:CONTEXT_NODE_LIMIT]:
                entry_nodes.append({
                    "element_id": node_data["element_id"],
                    "labels": node_data.get("labels", []),
                    "entities": sorted([generalized_query]),
                    "sources": sorted(list(node_data["sources"])),
                })

    return {"entry_nodes": entry_nodes, "node_id_map": node_id_map}


# ---------------------------------------------------------------------------
# Node C: Context retrieval
# ---------------------------------------------------------------------------

def context_retrieval(state: Dict[str, Any], driver, database: str) -> Dict[str, Any]:
    requested = state.get("generalized_query") or state.get("query")
    if RAG_USE_DEFAULT_CONTEXT:
        context_seed = RAG_DEFAULT_CONTEXT_QUERY or "leggi italiane"
        logger.info(
            "Context retrieval: default context enabled (seed='%s', original='%s')",
            context_seed,
            requested,
        )
    else:
        context_seed = requested

    if not context_seed:
        return {"context_nodes": []}

    with driver.session(database=database) as session:
        matches = vector_lookup(
            session, context_seed, indexes=CONTEXT_VECTOR_INDEXES,
            index_settings=VECTOR_INDEX_SETTINGS, source_prefix="context",
        )

    aggregated: Dict[str, Dict[str, Any]] = {}
    for match in matches:
        element_id = match["element_id"]
        labels = match.get("labels", []) or []
        score = match.get("score")
        source = match.get("source", "context")

        existing = aggregated.get(element_id)
        if not existing:
            aggregated[element_id] = {
                "element_id": element_id,
                "labels": list(labels),
                "sources": {source},
                "score": score,
            }
            continue
        existing["sources"].add(source)
        if labels and not existing["labels"]:
            existing["labels"] = list(labels)
        if score is not None and (existing.get("score") is None or score > existing["score"]):
            existing["score"] = score

    context_nodes = sorted(
        (
            {
                "element_id": data["element_id"],
                "labels": data.get("labels", []),
                "sources": sorted(data["sources"]),
                "score": data.get("score"),
            }
            for data in aggregated.values()
        ),
        key=lambda item: item.get("score") or 0,
        reverse=True,
    )[:CONTEXT_NODE_LIMIT]

    logger.info(
        "Context retrieval produced %d nodes (seed='%s')",
        len(context_nodes),
        context_seed,
    )
    return {"context_nodes": context_nodes}


# ---------------------------------------------------------------------------
# Node D1: Intersection Cypher generation
# ---------------------------------------------------------------------------

def _format_entry_lines(nodes: List[Dict[str, Any]]) -> str:
    if not nodes:
        return "(none)"
    return "\n".join(
        f'- elementId: "{item["element_id"]}", labels: {", ".join(item.get("labels", [])) or "Unknown"}, '
        f"entities: {', '.join(item.get('entities', [])) or 'Unknown'}"
        for item in nodes
    )


def _format_context_lines(nodes: List[Dict[str, Any]]) -> str:
    if not nodes:
        return "(none)"
    return "\n".join(
        f'- elementId: "{item["element_id"]}", labels: {", ".join(item.get("labels", [])) or "Unknown"}, '
        f"sources: {', '.join(item.get('sources', [])) or 'Unknown'}, score: {item.get('score') or 0:.4f}"
        for item in nodes
    )


def generate_cypher_intersection(state: Dict[str, Any]) -> Dict[str, Any]:
    lang = _session_lang(state)
    entry_nodes = state.get("entry_nodes") or []
    context_nodes = state.get("context_nodes") or []
    extracted_relationships = state.get("extracted_relationships", [])
    node_id_map = state.get("node_id_map", {})

    log_cypher_event(
        "b_prepare",
        "main nodes identified (entry_nodes + context_nodes) before intersection Cypher",
        detail={
            "entry_count": len(entry_nodes),
            "context_count": len(context_nodes),
            "entry_ids": [n.get("element_id") for n in entry_nodes[:12]],
            "context_ids": [n.get("element_id") for n in context_nodes[:12]],
            "keywords": state.get("retrieval_keywords") or [],
        },
    )

    if not entry_nodes:
        log_cypher_event(
            "b_skip",
            "intersection: no Cypher generated (no entry nodes from entity linking)",
            detail={
                "cypher_generation_error": "Entity linking returned no entry nodes.",
                "note": "If context_nodes exist, graph tries context_only Cypher; if both entry and context are empty, Neo4j is skipped.",
            },
        )
        return _fallback_cypher_payload(
            state,
            "Entity linking returned no entry nodes.",
            "intersection",
        )

    if not context_nodes:
        log_cypher_event(
            "b_skip",
            "intersection: no Cypher generated (no context nodes from semantic retrieval)",
            detail={
                "cypher_generation_error": "Context retrieval returned no candidate nodes.",
                "next_graph_route": "fallback",
            },
        )
        return _fallback_cypher_payload(
            state,
            "Context retrieval returned no candidate nodes.",
            "intersection",
        )

    entry_block = _format_entry_lines(entry_nodes)
    context_block = _format_context_lines(context_nodes)

    rel_context_parts = []
    for rel in extracted_relationships:
        source_eid = node_id_map.get(rel["source_id"])
        target_eid = node_id_map.get(rel["target_id"])
        if source_eid and target_eid:
            rel_context_parts.append(
                f"elementId(source)='{source_eid}' AND elementId(target)='{target_eid}' AND type='{rel['type']}'"
            )

    relationship_context = (
        "Relationships to consider:\n" + "\n".join(rel_context_parts)
        if rel_context_parts
        else "No specific relationships were extracted."
    )

    grouped_entries: Dict[str, List[str]] = {}
    for item in entry_nodes:
        for label in item.get("labels", []):
            grouped_entries.setdefault(label, []).append(item["element_id"])

    grouped_context: Dict[str, List[str]] = {}
    for item in context_nodes:
        for label in item.get("labels", []):
            grouped_context.setdefault(label, []).append(item["element_id"])

    grouped_entries_text = (
        "\n".join(
            "{}: [{}]".format(label, ", ".join('"' + eid + '"' for eid in ids))
            for label, ids in grouped_entries.items()
        )
        or "(no grouped entry IDs)"
    )
    grouped_context_text = (
        "\n".join(
            "{}: [{}]".format(label, ", ".join('"' + eid + '"' for eid in ids))
            for label, ids in grouped_context.items()
        )
        or "(no grouped context IDs)"
    )

    messages = [
        SystemMessage(
            content=(
                f"{legal_consultant_system_prefix(lang)} "
                "You are a Cypher expert focusing on legal data. "
                "Intersect subject nodes (entity linking) with context nodes (semantic search) to answer the user's question."
            )
        ),
        HumanMessage(
            content=(
                "Original question: {question}\n"
                "Schema:\n{schema}\n\n"
                "Relation directions:\n{relations}\n\n"
                "Subject entry nodes:\n{entries}\n\n"
                "Subject IDs by label:\n{entry_groups}\n\n"
                "Context candidate nodes:\n{contexts}\n\n"
                "Context IDs by label:\n{context_groups}\n\n"
                "{relationship_context}\n\n"
                "Construct ONE SINGLE Cypher query that finds paths between the Subject and Context nodes. "
                "You MUST use the relationships extracted from the query as the primary guide for pathfinding. "
                "To filter nodes by their ID, you MUST use the `elementId()` function in a WHERE clause. "
                "CRITICAL: Return ONLY ONE complete Cypher query with ONE RETURN statement at the end."
            ).format(
                question=state["query"],
                schema=SCHEMA_TEXT,
                relations=RELATION_HINTS,
                entries=entry_block,
                entry_groups=grouped_entries_text,
                contexts=context_block,
                context_groups=grouped_context_text,
                relationship_context=relationship_context,
            )
        ),
    ]
    try:
        prompt = _call_chat_timeout(
            messages,
            {
                "cypher_query": state.get("cypher_query"),
                "cypher_attempt": "intersection",
            },
        )
    except FuturesTimeoutError:
        return _fallback_cypher_payload(
            state,
            f"LLM generation timeout after {CYTHER_GENERATION_TIMEOUT_SECONDS}s",
            "intersection",
        )
    except Exception as exc:
        return _fallback_cypher_payload(
            state,
            f"LLM generation error: {exc}",
            "intersection",
        )

    cypher = _clean_cypher(prompt)
    cypher = _enforce_relation_directions(cypher)
    logger.info("Generated intersection Cypher: %s", cypher)
    log_cypher_multiline(
        "b_draft",
        "intersection: Cypher string as generated (next step: execute on Neo4j)",
        cypher,
    )

    if not cypher:
        return _fallback_cypher_payload(
            state,
            "LLM generated empty Cypher",
            "intersection",
        )
    return {
        "cypher_query": cypher,
        "cypher_params": {},
        "cypher_generation_error": None,
        "cypher_attempt": "intersection",
    }


def generate_cypher_context_only(state: Dict[str, Any]) -> Dict[str, Any]:
    """When entity linking finds no anchors, still query Neo4j from semantic-search context nodes."""
    lang = _session_lang(state)
    context_nodes = state.get("context_nodes") or []
    if not context_nodes:
        log_cypher_event(
            "b_skip",
            "context_only: no Cypher generated (no vector context nodes)",
            detail={"cypher_generation_error": "Context-only path: no vector context nodes."},
        )
        return _fallback_cypher_payload(
            state,
            "Context-only path: no vector context nodes.",
            "context_only",
        )

    context_block = _format_context_lines(context_nodes)
    grouped_context: Dict[str, List[str]] = {}
    for item in context_nodes:
        for label in item.get("labels", []):
            grouped_context.setdefault(label, []).append(item["element_id"])
    grouped_context_text = (
        "\n".join(
            "{}: [{}]".format(label, ", ".join('"' + eid + '"' for eid in ids))
            for label, ids in grouped_context.items()
        )
        or "(no grouped context IDs)"
    )

    log_cypher_event(
        "b_prepare",
        "context-only Cypher (no entry nodes)",
        detail={
            "context_count": len(context_nodes),
            "keywords": state.get("retrieval_keywords") or [],
        },
    )

    messages = [
        SystemMessage(
            content=(
                f"{legal_consultant_system_prefix(lang)} "
                "You are a Cypher expert. There was no precise entity match for the user's question, "
                "but semantic search produced anchor nodes below. Write ONE query that starts from those elementIds, "
                "expands along valid schema relationships (short paths), and returns substantive legal content: "
                "Article, Section, Clause, LegalAct, Penalty, Contract, Document, or related nodes useful to answer the question. "
                "Filter anchors with elementId() in WHERE. Prefer paths that yield text or normative metadata. "
                "CRITICAL: Return ONLY ONE complete Cypher query with ONE RETURN at the end."
            )
        ),
        HumanMessage(
            content=(
                "Original question: {question}\n"
                "Generalized topic: {generalized}\n"
                "Keywords: {keywords}\n\n"
                "Schema:\n{schema}\n\n"
                "Relation directions:\n{relations}\n\n"
                "Semantic anchor nodes (use these elementIds):\n{contexts}\n\n"
                "Context IDs by label:\n{context_groups}\n\n"
                "Construct ONE Cypher query to retrieve material from the graph that best answers the question."
            ).format(
                question=state["query"],
                generalized=state.get("generalized_query") or state["query"],
                keywords=", ".join(state.get("retrieval_keywords") or []),
                schema=SCHEMA_TEXT,
                relations=RELATION_HINTS,
                contexts=context_block,
                context_groups=grouped_context_text,
            )
        ),
    ]
    try:
        prompt = _call_chat_timeout(
            messages,
            {
                "cypher_query": state.get("cypher_query"),
                "cypher_attempt": "context_only",
            },
        )
    except FuturesTimeoutError:
        return _fallback_cypher_payload(
            state,
            f"LLM generation timeout after {CYTHER_GENERATION_TIMEOUT_SECONDS}s",
            "context_only",
        )
    except Exception as exc:
        return _fallback_cypher_payload(
            state,
            f"LLM generation error: {exc}",
            "context_only",
        )

    cypher = _clean_cypher(prompt)
    cypher = _enforce_relation_directions(cypher)
    logger.info("Generated context-only Cypher: %s", cypher)
    log_cypher_multiline(
        "b_draft",
        "context_only: Cypher string as generated (next step: execute on Neo4j)",
        cypher,
    )

    if not cypher:
        return _fallback_cypher_payload(
            state,
            "LLM generated empty Cypher",
            "context_only",
        )
    return {
        "cypher_query": cypher,
        "cypher_params": {},
        "cypher_generation_error": None,
        "cypher_attempt": "context_only",
    }


# ---------------------------------------------------------------------------
# Node D2: Fallback Cypher generation
# ---------------------------------------------------------------------------

def generate_cypher_fallback(state: Dict[str, Any]) -> Dict[str, Any]:
    lang = _session_lang(state)
    entry_nodes = state.get("entry_nodes") or []
    extracted_relationships = state.get("extracted_relationships", [])

    log_cypher_event(
        "b_prepare",
        "fallback Cypher generation",
        detail={"entry_count": len(entry_nodes), "keywords": state.get("retrieval_keywords") or []},
    )

    if not entry_nodes:
        log_cypher_event(
            "b_skip",
            "fallback: no Cypher generated (no entry nodes)",
            detail={"cypher_generation_error": "Fallback: no subject entry nodes available."},
        )
        return _fallback_cypher_payload(
            state,
            "Fallback: no subject entry nodes available.",
            "fallback",
        )

    entry_block = _format_entry_lines(entry_nodes)
    grouped_entries: Dict[str, List[str]] = {}
    for item in entry_nodes:
        for label in item.get("labels", []):
            grouped_entries.setdefault(label, []).append(item["element_id"])

    grouped_entries_text = (
        "\n".join(
            "{}: [{}]".format(label, ", ".join('"' + eid + '"' for eid in ids))
            for label, ids in grouped_entries.items()
        )
        or "(no grouped entry IDs)"
    )

    fallback_reason = state.get("cypher_generation_error") or "Intersection attempt returned no rows."
    context_summary = _format_context_lines(state.get("context_nodes") or [])

    rel_context_parts = []
    for rel in extracted_relationships:
        source_node = next(
            (n for n in state["entities"] if n.get("id") == rel["source_id"]), None
        )
        target_node = next(
            (n for n in state["entities"] if n.get("id") == rel["target_id"]), None
        )
        if source_node and target_node:
            rel_context_parts.append(
                f"({source_node['label']})-[:{rel['type']}]->({target_node['label']})"
            )

    relationship_context = (
        "The user's query implies these connections:\n" + "\n".join(rel_context_parts)
        if rel_context_parts
        else ""
    )

    messages = [
        SystemMessage(
            content=(
                f"{legal_consultant_system_prefix(lang)} "
                "Generate resilient Cypher for the SunnitAI graph. "
                "Reuse subject entry node IDs as anchors and traverse relationships using elementId() filters only."
            )
        ),
        HumanMessage(
            content=(
                "Original question: {question}\n"
                "Reason for fallback: {reason}\n\n"
                "Schema:\n{schema}\n\n"
                "Relation directions:\n{relations}\n\n"
                "Subject entry nodes:\n{entries}\n\n"
                "Subject IDs by label:\n{entry_groups}\n\n"
                "Context hints:\n{contexts}\n\n"
                "{relationship_context}\n\n"
                "Generate a SINGLE Cypher query that starts from the subject IDs using elementId() filters. "
                "CRITICAL: Return ONLY ONE complete Cypher query with ONE RETURN statement at the end."
            ).format(
                question=state["query"],
                reason=fallback_reason,
                schema=SCHEMA_TEXT,
                relations=RELATION_HINTS,
                entries=entry_block,
                entry_groups=grouped_entries_text,
                contexts=context_summary,
                relationship_context=relationship_context,
            )
        ),
    ]
    try:
        prompt = _call_chat_timeout(
            messages,
            {
                "cypher_query": state.get("cypher_query"),
                "cypher_attempt": "fallback",
            },
        )
    except FuturesTimeoutError:
        return _fallback_cypher_payload(
            state,
            f"LLM generation timeout after {CYTHER_GENERATION_TIMEOUT_SECONDS}s",
            "fallback",
        )
    except Exception as exc:
        return _fallback_cypher_payload(
            state,
            f"LLM generation error: {exc}",
            "fallback",
        )

    cypher = _clean_cypher(prompt)
    cypher = _enforce_relation_directions(cypher)
    logger.info("Generated fallback Cypher: %s", cypher)
    log_cypher_multiline(
        "b_draft",
        "fallback: Cypher string as generated (next step: execute on Neo4j)",
        cypher,
    )

    if not cypher:
        return _fallback_cypher_payload(
            state,
            "LLM generated empty Cypher",
            "fallback",
        )
    return {
        "cypher_query": cypher,
        "cypher_params": {},
        "cypher_generation_error": None,
        "cypher_attempt": "fallback",
    }


def generate_cypher_reformulation(state: Dict[str, Any]) -> Dict[str, Any]:
    """Regenerate Cypher after a poor quality verdict (max two rounds handled upstream)."""
    lang = _session_lang(state)
    entry_nodes = state.get("entry_nodes") or []
    extracted_relationships = state.get("extracted_relationships", [])
    feedback = state.get("quality_feedback") or "Prior result lacked concrete legal detail."
    previous = (state.get("cypher_query") or "").strip()

    if not entry_nodes:
        log_cypher_event(
            "b_skip",
            "reformulation: no Cypher generated (no entry nodes)",
            detail={"cypher_generation_error": "Reformulation: no subject entry nodes."},
        )
        return {
            "cypher_query": None,
            "cypher_generation_error": "Reformulation: no subject entry nodes.",
            "cypher_attempt": "reformulation",
        }

    entry_block = _format_entry_lines(entry_nodes)
    grouped_entries: Dict[str, List[str]] = {}
    for item in entry_nodes:
        for label in item.get("labels", []):
            grouped_entries.setdefault(label, []).append(item["element_id"])
    grouped_entries_text = (
        "\n".join(
            "{}: [{}]".format(label, ", ".join('"' + eid + '"' for eid in ids))
            for label, ids in grouped_entries.items()
        )
        or "(no grouped entry IDs)"
    )

    rel_context_parts = []
    for rel in extracted_relationships:
        source_node = next(
            (n for n in state["entities"] if n.get("id") == rel["source_id"]), None
        )
        target_node = next(
            (n for n in state["entities"] if n.get("id") == rel["target_id"]), None
        )
        if source_node and target_node:
            rel_context_parts.append(
                f"({source_node['label']})-[:{rel['type']}]->({target_node['label']})"
            )
    relationship_context = (
        "The user's query implies these connections:\n" + "\n".join(rel_context_parts)
        if rel_context_parts
        else ""
    )

    log_cypher_event(
        "b_reformulate",
        "reformulating Cypher with evaluation feedback",
        detail={"feedback": feedback[:2000], "previous_len": len(previous)},
    )

    prompt = _call_chat(
        [
            SystemMessage(
                content=(
                    f"{legal_consultant_system_prefix(lang)} "
                    "Generate improved Cypher for the SunnitAI graph. "
                    "Address the critique; broaden paths or add node/relationship patterns where useful. "
                    "Use elementId() filters on subject anchors; respect schema directions."
                )
            ),
            HumanMessage(
                content=(
                    "Original question: {question}\n"
                    "Critique of previous retrieval: {feedback}\n\n"
                    "Previous Cypher (may be suboptimal):\n{previous}\n\n"
                    "Schema:\n{schema}\n\n"
                    "Relation directions:\n{relations}\n\n"
                    "Subject entry nodes:\n{entries}\n\n"
                    "Subject IDs by label:\n{entry_groups}\n\n"
                    "{relationship_context}\n\n"
                    "Produce ONE improved Cypher query with ONE RETURN. "
                    "Return ONLY the query, nothing else."
                ).format(
                    question=state["query"],
                    feedback=feedback,
                    previous=previous or "(none)",
                    schema=SCHEMA_TEXT,
                    relations=RELATION_HINTS,
                    entries=entry_block,
                    entry_groups=grouped_entries_text,
                    relationship_context=relationship_context,
                )
            ),
        ],
        llm_context={
            "cypher_query": state.get("cypher_query"),
            "cypher_attempt": "reformulation",
        },
    )

    cypher = _clean_cypher(prompt)
    cypher = _enforce_relation_directions(cypher)
    logger.info("Generated reformulation Cypher: %s", cypher)
    log_cypher_multiline(
        "b_draft",
        "reformulation: Cypher string as generated (next step: execute on Neo4j)",
        cypher,
    )

    return {
        "cypher_query": cypher,
        "cypher_generation_error": None,
        "cypher_attempt": "reformulation",
    }


# ---------------------------------------------------------------------------
# Node E: Cypher execution
# ---------------------------------------------------------------------------

def _enrich_with_source_metadata(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    enriched_references = []
    for record in data:
        reference = {"data": record, "sources": []}
        for key, value in record.items():
            if isinstance(value, dict):
                if "properties" in value and "labels" in value:
                    labels = value.get("labels", [])
                    props = value.get("properties", {})
                    source_info = {
                        "type": labels[0] if labels else "Unknown",
                        "id": value.get("elementId"),
                    }
                    if "Document" in labels:
                        source_info["document_id"] = props.get("document_id")
                        source_info["document_title"] = props.get("document_title")
                        source_info["document_date"] = props.get("document_date")
                    elif "LegalAct" in labels:
                        source_info["act_type"] = props.get("act_type")
                        source_info["act_number"] = props.get("act_number")
                        source_info["act_year"] = props.get("act_year")
                    elif "Article" in labels:
                        source_info["parent_act_key"] = props.get("parent_act_key")
                        source_info["index"] = props.get("index")
                        source_info["heading"] = props.get("heading")
                    elif "Section" in labels:
                        source_info["document_id"] = props.get("document_id")
                        source_info["chunk_id"] = props.get("chunk_id")
                        source_info["title"] = props.get("title")
                    if props.get("text_en"):
                        source_info["text_preview"] = props.get("text_en")[:200] + "..."
                    reference["sources"].append(source_info)
        enriched_references.append(reference)
    return enriched_references


def execute_cypher(state: Dict[str, Any], driver, database: str) -> Dict[str, Any]:
    cypher = state.get("cypher_query")
    cypher_params = state.get("cypher_params") or {}
    attempt = state.get("cypher_attempt", "unknown")
    if not cypher:
        log_cypher_event(
            "c_skip",
            "Neo4j: query not executed (empty cypher_query)",
            detail={
                "cypher_attempt": attempt,
                "cypher_generation_error": state.get("cypher_generation_error"),
            },
        )
        return {
            "raw_result": [],
            "execution_error": state.get("cypher_generation_error"),
            "neo4j_executed": False,
        }

    # Guardrail: if LLM-generated Cypher references non-existing vector indexes,
    # force deterministic fallback query instead of surfacing Neo4j procedure errors.
    requested_vector_indexes = {
        match.group(1).strip()
        for match in _VECTOR_QUERY_INDEX_RE.finditer(cypher or "")
        if match.group(1).strip()
    }
    invalid_vector_indexes = sorted(
        idx for idx in requested_vector_indexes if idx not in _ALLOWED_VECTOR_INDEXES
    )
    if invalid_vector_indexes:
        keyword = _select_fallback_keyword(state)
        fallback_query = _base_fallback_cypher()
        log_cypher_event(
            "c_execute",
            "Cypher rejected due to unknown vector index; forcing fallback query",
            detail={
                "invalid_vector_indexes": invalid_vector_indexes,
                "allowed_vector_indexes": sorted(_ALLOWED_VECTOR_INDEXES),
                "fallback_keyword": keyword,
            },
        )
        cypher = fallback_query
        cypher_params = {"keyword": keyword}
        attempt = "fallback_guard_vector_index"

    # Exact string passed to Neo4j driver (verbatim, including whitespace)
    log_cypher_multiline(
        "c_execute",
        f"Query submitted to Neo4j database={database!r} attempt={attempt!r} (exact string below)",
        cypher,
    )

    exec_t0 = time.perf_counter()
    try:
        with driver.session(database=database) as session:
            run_t0 = time.perf_counter()
            records = session.run(cypher, cypher_params)
            data = [record.data() for record in records]
            run_elapsed_ms = int((time.perf_counter() - run_t0) * 1000)
    except Neo4jError as exc:
        logger.error("Cypher execution failed during %s attempt: %s", attempt, exc)
        log_cypher_event(
            "c_execute",
            f"Neo4j driver error after submit attempt={attempt!r} database={database!r}",
            detail={
                "error": str(exc),
                "elapsed_ms": int((time.perf_counter() - exec_t0) * 1000),
            },
        )
        return {
            "raw_result": [],
            "execution_error": str(exc),
            "neo4j_executed": True,
        }

    logger.info("Cypher execution (%s) returned %d rows", attempt, len(data))
    log_cypher_event(
        "c_execute",
        f"Neo4j execution finished attempt={attempt!r} rows={len(data)}",
        detail={
            "result_column_keys": list(data[0].keys()) if data else [],
            "run_elapsed_ms": run_elapsed_ms,
            "total_elapsed_ms": int((time.perf_counter() - exec_t0) * 1000),
        },
    )
    preview_rows = data[:10]
    log_cypher_multiline(
        "c_execute",
        f"Neo4j response payload preview rows={len(data)} (showing up to 10)",
        json.dumps(preview_rows, ensure_ascii=False, indent=2),
        delimiter_label="NEO4J_RESULT",
    )
    enriched_references = _enrich_with_source_metadata(data)

    return {
        "raw_result": data,
        "execution_error": None,
        "references": enriched_references,
        "neo4j_executed": True,
    }


# ---------------------------------------------------------------------------
# Node F1: Retrieval quality evaluation
# ---------------------------------------------------------------------------


def evaluate_retrieval_quality(state: Dict[str, Any]) -> Dict[str, Any]:
    """LLM critique of retrieved rows before synthesis; may trigger reformulation (max two)."""
    lang = _session_lang(state)
    data = state.get("raw_result") or []
    status = list(state.get("status_messages") or [])
    if state.get("cypher_attempt") != "reformulation":
        if lang == "it":
            status.append(
                "Seconda fase: valutazione critica dei risultati recuperati dal database…"
            )
        elif lang == "es":
            status.append(
                "Segunda fase: evaluación crítica de los resultados recuperados de la base de datos…"
            )
        else:
            status.append(
                "Second phase: critical evaluation of results retrieved from the database…"
            )

    summarized_data = _summarize_for_synthesis(data, max_records=8)
    serialized = json.dumps(summarized_data, ensure_ascii=False, indent=2)

    r_before = int(state.get("quality_reformulation_round") or 0)
    log_cypher_event(
        "d_evaluate_start",
        "critical retrieval evaluation (LLM) — starting",
        detail={
            "user_query": state["query"],
            "row_count": len(data),
            "cypher_attempt": state.get("cypher_attempt"),
            "quality_reformulation_round_before": r_before,
        },
    )

    verdict_raw = _call_chat(
        [
            SystemMessage(
                content=(
                    "You judge whether Neo4j rows are sufficient to answer the user's legal question with concrete substance "
                    "(articles, acts, parties, obligations, dates, or defined terms). "
                    "Reply with exactly two lines: "
                    "Line 1: OK or POOR (uppercase). "
                    "Line 2: one short sentence explaining why."
                )
            ),
            HumanMessage(
                content=(
                    "Question:\n{q}\n\n"
                    "Summarized rows:\n{rows}\n\n"
                    "Verdict:"
                ).format(q=state["query"], rows=serialized[:45000])
            ),
        ],
        llm_context={
            "cypher_query": state.get("cypher_query"),
            "cypher_attempt": state.get("cypher_attempt"),
        },
    )
    lines = [ln.strip() for ln in (verdict_raw or "").splitlines() if ln.strip()]
    head = lines[0].upper() if lines else "OK"
    poor = head.startswith("POOR")
    feedback = lines[1] if len(lines) > 1 else ""

    log_cypher_multiline(
        "d_evaluate_llm",
        "raw LLM verdict output (line 1: OK|POOR, line 2: reason)",
        verdict_raw or "",
        delimiter_label="LLM_VERDICT",
    )

    r = int(state.get("quality_reformulation_round") or 0)
    if not poor:
        decision = "OK -> route synthesize_answer"
        ok_flag = True
        r_after = r
        fb_out = None
    elif r < 2:
        decision = f"POOR -> route generate_cypher_reformulation (round {r} -> {r + 1})"
        ok_flag = False
        r_after = r + 1
        fb_out = feedback
    else:
        decision = "POOR -> route synthesize_answer (reformulation cap reached; max 2 retries done)"
        ok_flag = True
        r_after = r
        fb_out = feedback

    log_cypher_event(
        "d_evaluate_decision",
        decision,
        detail={
            "verdict_line": head[:200],
            "feedback": feedback,
            "retrieval_quality_ok": ok_flag,
            "quality_reformulation_round_after": r_after,
        },
    )

    if not poor:
        return {
            "retrieval_quality_ok": True,
            "quality_feedback": None,
            "status_messages": status,
            "retrieval_evaluated": True,
        }
    if r < 2:
        return {
            "retrieval_quality_ok": False,
            "quality_reformulation_round": r + 1,
            "quality_feedback": feedback,
            "status_messages": status,
            "retrieval_evaluated": True,
        }
    return {
        "retrieval_quality_ok": True,
        "quality_feedback": feedback,
        "status_messages": status,
        "retrieval_evaluated": True,
    }


# ---------------------------------------------------------------------------
# Node F: Answer synthesis
# ---------------------------------------------------------------------------

def _summarize_for_synthesis(
    data: List[Dict[str, Any]], max_records: int = 5
) -> List[Dict[str, Any]]:
    summarized = []
    total_chars = 0
    MAX_TOTAL_CHARS = 50000

    for record in data[:max_records]:
        summary_record = {}
        for key, value in record.items():
            if isinstance(value, dict) and "properties" in value:
                props = value["properties"]
                labels = value.get("labels", [])
                summary_props = {"labels": labels}

                if "LegalAct" in labels:
                    summary_props.update({
                        "act_type": props.get("act_type"),
                        "act_number": props.get("act_number"),
                        "act_year": props.get("act_year"),
                        "title": (props.get("title") or "")[:100],
                    })
                elif "Person" in labels:
                    summary_props.update({"name": props.get("name"), "role": props.get("role")})
                elif "Company" in labels or "Institution" in labels:
                    summary_props.update({
                        "name": props.get("name"),
                        "normalized_name": props.get("normalized_name"),
                    })
                elif "Article" in labels:
                    snippet = ""
                    for key in ("text_en", "text_it", "text_es", "text_ar"):
                        v = props.get(key)
                        if isinstance(v, str) and v.strip():
                            snippet = v[:150]
                            break
                    summary_props.update({
                        "index": props.get("index"),
                        "heading": (props.get("heading") or "")[:100],
                        "text_snippet": snippet,
                    })
                elif "Document" in labels:
                    summary_props.update({
                        "document_id": props.get("document_id"),
                        "document_title": (props.get("document_title") or "")[:100],
                        "document_date": props.get("document_date"),
                    })
                else:
                    summary_props.update({
                        "title": (props.get("title") or "")[:80],
                        "name": props.get("name"),
                        "text_en": (props.get("text_en") or "")[:80],
                    })

                summary_record[key] = {k: v for k, v in summary_props.items() if v is not None}
            elif value is None:
                continue
            else:
                if isinstance(value, str) and len(value) > 100:
                    summary_record[key] = value[:100] + "..."
                else:
                    summary_record[key] = value

        record_json = json.dumps(summary_record, ensure_ascii=False)
        total_chars += len(record_json)
        if total_chars > MAX_TOTAL_CHARS:
            break
        summarized.append(summary_record)

    return summarized


def synthesize_answer(state: Dict[str, Any]) -> Dict[str, Any]:
    lang = _session_lang(state)
    error = state.get("execution_error") or state.get("cypher_generation_error")
    data = state.get("raw_result", [])

    qfb = state.get("quality_feedback")
    log_cypher_event(
        "z_pipeline_terminal",
        "retrieval pipeline snapshot — entering final answer synthesis",
        detail={
            "user_query": state["query"],
            "session_language": state.get("session_language"),
            "generalized_query": state.get("generalized_query"),
            "retrieval_keywords": state.get("retrieval_keywords"),
            "entry_nodes_count": len(state.get("entry_nodes") or []),
            "context_nodes_count": len(state.get("context_nodes") or []),
            "cypher_attempt": state.get("cypher_attempt"),
            "cypher_generated": bool(state.get("cypher_query")),
            "neo4j_executed": state.get("neo4j_executed"),
            "neo4j_row_count": len(data),
            "cypher_generation_error": state.get("cypher_generation_error"),
            "execution_error": state.get("execution_error"),
            "critical_evaluation_ran": bool(state.get("retrieval_evaluated")),
            "retrieval_quality_ok": state.get("retrieval_quality_ok"),
            "quality_reformulation_round": state.get("quality_reformulation_round"),
            "quality_feedback_excerpt": (qfb[:400] + "...") if isinstance(qfb, str) and len(qfb) > 400 else qfb,
        },
    )

    if error:
        answer = _call_chat(
            [
                SystemMessage(content=synthesis_error_system(lang)),
                HumanMessage(
                    content=(
                        "Original question: {question}\n"
                        "Internal retrieval note (do not quote literally or discuss IT systems): {error}\n\n"
                        "Provide the legal consultation as instructed."
                    ).format(question=state["query"], error=error)
                    + synthesis_human_footer(lang)
                ),
            ],
            llm_context={
                "cypher_query": state.get("cypher_query"),
                "cypher_attempt": state.get("cypher_attempt"),
            },
        )
        return {
            "answer": answer,
            "references": state.get("raw_result", []) or [],
            "status_messages": state.get("status_messages") or [],
        }

    if not data:
        answer = _call_chat(
            [
                SystemMessage(content=synthesis_empty_system(lang)),
                HumanMessage(
                    content=(
                        "Original question: {question}\n"
                        "The knowledge graph query returned no rows.\n\n"
                        "Provide the legal consultation as instructed."
                    ).format(question=state["query"])
                    + synthesis_human_footer(lang)
                ),
            ],
            llm_context={
                "cypher_query": state.get("cypher_query"),
                "cypher_attempt": state.get("cypher_attempt"),
            },
        )
        return {
            "answer": answer,
            "references": [],
            "status_messages": state.get("status_messages") or [],
        }

    summarized_data = _summarize_for_synthesis(data)
    serialized = json.dumps(summarized_data, ensure_ascii=False, indent=2)
    answer = _call_chat(
        [
            SystemMessage(content=synthesis_system_message(lang)),
            HumanMessage(
                content=(
                    "Question: {question}\n"
                    "Data: {data}\n"
                    "Write a concise factual answer. Quote short passages in their original language from the data; "
                    "explain and synthesize in the session language."
                ).format(question=state["query"], data=serialized)
                + synthesis_human_footer(lang)
            ),
        ],
        llm_context={
            "cypher_query": state.get("cypher_query"),
            "cypher_attempt": state.get("cypher_attempt"),
        },
    )
    return {
        "answer": answer,
        "references": data,
        "status_messages": state.get("status_messages") or [],
    }


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def route_after_intersection(state: Dict[str, Any]) -> str:
    cypher = state.get("cypher_query")
    attempt = state.get("cypher_attempt")
    if cypher:
        return "run"
    if attempt != "intersection":
        return "abort"
    if not state.get("entry_nodes"):
        if state.get("context_nodes"):
            return "context_explore"
        return "abort"
    return "fallback"


def route_after_execution(state: Dict[str, Any]) -> str:
    if state.get("execution_error"):
        return "answer"
    if state.get("raw_result"):
        return "evaluate"
    if state.get("cypher_attempt") == "intersection":
        return "retry"
    return "answer"


def route_after_evaluation(state: Dict[str, Any]) -> str:
    if state.get("retrieval_quality_ok"):
        return "synthesize"
    return "reformulate"
