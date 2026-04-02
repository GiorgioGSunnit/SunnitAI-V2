"""LangGraph node functions for the RAG agent pipeline."""

import json
import logging
from typing import Any, Dict, List, Optional, Set

from langchain_core.messages import HumanMessage, SystemMessage
from neo4j.exceptions import Neo4jError

from ..preprocessing.schema.schema import entities as schema_entities
from ..preprocessing.schema.schema import relations as schema_relations
from .ai_chat import _call_chat, structured_entities_model
from .lookup_indexes import (
    CONTEXT_NODE_LIMIT,
    CONTEXT_VECTOR_INDEXES,
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
    logger.info("Starting query decomposition", extra={"query": query})

    # Step 1: Generalize the query
    generalization_prompt = (
        "Original question: {query}\n"
        "Respond with a short generalized phrase (max 8 words) capturing the main topic."
    ).format(query=query)
    generalized = _call_chat(
        [
            SystemMessage(
                content="You generalize user questions about legal documents into concise search-focused phrases."
            ),
            HumanMessage(content=generalization_prompt),
        ]
    )
    logger.info(f"Generalized query: '{generalized}'")

    # Step 2: Extract structured entities and relationships
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

    entities_payload = structured_entities_model.invoke(
        [
            SystemMessage(
                content=(
                    "You are an expert graph extractor for legal documents. "
                    "Your task is to identify nodes and relationships from the user's query based on the provided graph schema."
                )
            ),
            HumanMessage(content=entity_extraction_prompt),
        ]
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
    generalized = state.get("generalized_query") or state.get("query")
    if not generalized:
        return {"context_nodes": []}

    with driver.session(database=database) as session:
        matches = vector_lookup(
            session, generalized, indexes=CONTEXT_VECTOR_INDEXES,
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

    logger.info("Context retrieval produced %d nodes", len(context_nodes))
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
    entry_nodes = state.get("entry_nodes") or []
    context_nodes = state.get("context_nodes") or []
    extracted_relationships = state.get("extracted_relationships", [])
    node_id_map = state.get("node_id_map", {})

    if not entry_nodes:
        return {
            "cypher_query": None,
            "cypher_generation_error": "Entity linking returned no entry nodes.",
            "cypher_attempt": "intersection",
        }

    if not context_nodes:
        return {
            "cypher_query": None,
            "cypher_generation_error": "Context retrieval returned no candidate nodes.",
            "cypher_attempt": "intersection",
        }

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

    prompt = _call_chat(
        [
            SystemMessage(
                content=(
                    "You are a Cypher expert focusing on legal data. "
                    "Your goal is to intersect subject nodes (from entity linking) with context nodes (semantic search results) to answer the user's question."
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
    )

    cypher = _clean_cypher(prompt)
    cypher = _enforce_relation_directions(cypher)
    logger.info("Generated intersection Cypher: %s", cypher)

    return {
        "cypher_query": cypher,
        "cypher_generation_error": None,
        "cypher_attempt": "intersection",
    }


# ---------------------------------------------------------------------------
# Node D2: Fallback Cypher generation
# ---------------------------------------------------------------------------

def generate_cypher_fallback(state: Dict[str, Any]) -> Dict[str, Any]:
    entry_nodes = state.get("entry_nodes") or []
    extracted_relationships = state.get("extracted_relationships", [])

    if not entry_nodes:
        return {
            "cypher_query": None,
            "cypher_generation_error": "Fallback: no subject entry nodes available.",
            "cypher_attempt": "fallback",
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

    prompt = _call_chat(
        [
            SystemMessage(
                content=(
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
    )

    cypher = _clean_cypher(prompt)
    cypher = _enforce_relation_directions(cypher)
    logger.info("Generated fallback Cypher: %s", cypher)

    return {
        "cypher_query": cypher,
        "cypher_generation_error": None,
        "cypher_attempt": "fallback",
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
    attempt = state.get("cypher_attempt", "unknown")
    if not cypher:
        return {
            "raw_result": [],
            "execution_error": state.get("cypher_generation_error"),
        }

    try:
        with driver.session(database=database) as session:
            records = session.run(cypher)
            data = [record.data() for record in records]
    except Neo4jError as exc:
        logger.error("Cypher execution failed during %s attempt: %s", attempt, exc)
        return {"raw_result": [], "execution_error": str(exc)}

    logger.info("Cypher execution (%s) returned %d rows", attempt, len(data))
    enriched_references = _enrich_with_source_metadata(data)

    return {
        "raw_result": data,
        "execution_error": None,
        "references": enriched_references,
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
                    summary_props.update({
                        "index": props.get("index"),
                        "heading": (props.get("heading") or "")[:100],
                        "text_en": (props.get("text_en") or "")[:150],
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
    error = state.get("execution_error") or state.get("cypher_generation_error")
    data = state.get("raw_result", [])

    if error:
        answer = _call_chat(
            [
                SystemMessage(
                    content="Explain issues answering queries about legal documents clearly and concisely."
                ),
                HumanMessage(
                    content=(
                        "Original question: {question}\n"
                        "Issue: {error}\n"
                        "Respond with a brief explanation and suggest a follow-up question if applicable."
                    ).format(question=state["query"], error=error)
                ),
            ]
        )
        return {"answer": answer, "references": state.get("raw_result", []) or []}

    if not data:
        answer = _call_chat(
            [
                SystemMessage(
                    content="Summarize data from Neo4j queries about legal documents."
                ),
                HumanMessage(
                    content=(
                        "Original question: {question}\n"
                        "No rows were returned. Respond with a concise answer indicating the absence of results."
                    ).format(question=state["query"])
                ),
            ]
        )
        return {"answer": answer, "references": []}

    summarized_data = _summarize_for_synthesis(data)
    serialized = json.dumps(summarized_data, ensure_ascii=False, indent=2)
    answer = _call_chat(
        [
            SystemMessage(
                content="Compose clear answers about penalties, contracts, and legal acts using retrieved data."
            ),
            HumanMessage(
                content=(
                    "Question: {question}\n"
                    "Data: {data}\n"
                    "Write a concise factual answer."
                ).format(question=state["query"], data=serialized)
            ),
        ]
    )
    return {"answer": answer, "references": data}


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
        return "abort"
    return "fallback"


def route_after_execution(state: Dict[str, Any]) -> str:
    if state.get("execution_error"):
        return "answer"
    if state.get("raw_result"):
        return "answer"
    if state.get("cypher_attempt") == "intersection":
        return "retry"
    return "answer"
