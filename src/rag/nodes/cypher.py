"""Cypher generation, execution and retrieval-quality nodes for the RAG agent pipeline."""

import json
import logging
import re
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from neo4j.exceptions import Neo4jError

from ..ai_chat import _call_chat
from ..answer_processing import _fetch_allowed_doc_ids, _visibility_filter
from ..cypher_logger import log_cypher_event, log_cypher_multiline
from ..doc_lookup import _select_schema_for_query
from ..formatting import (
    _collect_labels,
    _enrich_with_source_metadata,
    _format_context_lines,
    _format_entry_lines,
    _session_lang,
    _summarize_for_synthesis,
)
from ..prompts import legal_consultant_system_prefix
from ..utils import _clean_cypher, _enforce_relation_directions
from ..verbose_logger import vlog

logger = logging.getLogger(__name__)


# Max nodes to pass into Cypher generation prompts (keeps tokens under control)
_MAX_ENTRY_NODES_FOR_PROMPT = 8
_MAX_CONTEXT_NODES_FOR_PROMPT = 6


# ---------------------------------------------------------------------------
# Node D1: Intersection Cypher generation
# ---------------------------------------------------------------------------

def generate_cypher_intersection(state: Dict[str, Any], driver=None, database: str = "neo4j") -> Dict[str, Any]:
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
        # Last-resort: try a direct keyword search before skipping
        keywords = state.get("retrieval_keywords") or []
        keyword = keywords[0] if keywords else None
        if keyword and driver:
            try:
                with driver.session(database=database) as kw_session:
                    kw_records = kw_session.run(
                        "MATCH (n) "
                        "WHERE (n:Document OR n:Section OR n:LegalAct) "
                        "AND ("
                        "  toLower(n.name) CONTAINS toLower($keyword) OR "
                        "  toLower(n.description) CONTAINS toLower($keyword) OR "
                        "  toLower(n.abstract) CONTAINS toLower($keyword)"
                        ") "
                        "AND NOT ('LegalAct' IN labels(n) AND (n.name IS NULL OR n.name = '' OR n.name IN ['string', 'Fonte non classificata'])) "
                        "AND NOT ('Section' IN labels(n) AND (n.abstract IS NULL OR n.abstract = '') AND (n.plain_text IS NULL OR n.plain_text = '')) "
                        "RETURN elementId(n) AS element_id, labels(n) AS labels "
                        "LIMIT 5",
                        keyword=keyword,
                    )
                    kw_rows = [r.data() for r in kw_records]
            except Exception as exc:
                logger.warning("b_keyword_fallback failed: %s", exc)
                kw_rows = []
            kw_entry_nodes = [
                {"element_id": row["element_id"], "labels": row["labels"], "sources": ["b_keyword_fallback"]}
                for row in kw_rows
                if row.get("element_id")
            ]
            if kw_entry_nodes:
                log_cypher_event(
                    "b_keyword_fallback",
                    "keyword fallback search found nodes — generating Cypher",
                    detail={"keyword": keyword, "count": len(kw_entry_nodes)},
                )
                ids_literal = "[" + ", ".join("'" + n["element_id"] + "'" for n in kw_entry_nodes) + "]"
                cypher = (
                    "MATCH (d:Document)-[:CONTAINS]->(s:Section)\n"
                    "WHERE elementId(d) IN " + ids_literal + "\n"
                    "RETURN d, s\n"
                    "LIMIT 8"
                )
                log_cypher_multiline("b_keyword_fallback", "keyword fallback Cypher", cypher)
                return {
                    "entry_nodes": kw_entry_nodes,
                    "cypher_query": cypher,
                    "cypher_generation_error": None,
                    "cypher_attempt": "intersection",
                }
        log_cypher_event(
            "b_skip",
            "intersection: no Cypher generated (no entry nodes from entity linking)",
            detail={
                "cypher_generation_error": "Entity linking returned no entry nodes.",
                "note": "If context_nodes exist, graph tries context_only Cypher; if both entry and context are empty, Neo4j is skipped.",
            },
        )
        return {
            "cypher_query": None,
            "cypher_generation_error": "Entity linking returned no entry nodes.",
            "cypher_attempt": "intersection",
        }

    if not context_nodes:
        log_cypher_event(
            "b_skip",
            "intersection: no Cypher generated (no context nodes from semantic retrieval)",
            detail={
                "cypher_generation_error": "Context retrieval returned no candidate nodes.",
                "next_graph_route": "fallback",
            },
        )
        return {
            "cypher_query": None,
            "cypher_generation_error": "Context retrieval returned no candidate nodes.",
            "cypher_attempt": "intersection",
        }

    turn_count = state.get("turn_count", 1)
    if turn_count == 1:
        entry_ids = [n["element_id"] for n in entry_nodes if n.get("element_id")]
        context_ids = [n["element_id"] for n in context_nodes if n.get("element_id")]
        # Only use section-level IDs so we don't pull arbitrary sections from a matched Document node
        section_ids = [
            eid for eid in dict.fromkeys(entry_ids + context_ids)
            if any(
                "Section" in (n.get("labels") or [])
                for n in (entry_nodes + context_nodes)
                if n.get("element_id") == eid
            )
        ] or list(dict.fromkeys(entry_ids + context_ids))
        ids_literal = "[" + ", ".join("'" + eid + "'" for eid in section_ids) + "]"
        cypher = (
            "MATCH (d:Document)-[:CONTAINS]->(s:Section)\n"
            "WHERE elementId(s) IN " + ids_literal + "\n"
            "RETURN d, s\n"
            "LIMIT 8"
        )
        log_cypher_multiline(
            "b_tier1",
            "intersection: Tier 1 fixed Cypher (first message, no LLM call)",
            cypher,
        )
        return {
            "cypher_query": cypher,
            "cypher_generation_error": None,
            "cypher_attempt": "intersection",
        }

    # Cap nodes to keep prompt within token limits
    capped_entries = entry_nodes[:_MAX_ENTRY_NODES_FOR_PROMPT]
    capped_context = context_nodes[:_MAX_CONTEXT_NODES_FOR_PROMPT]

    entry_block = _format_entry_lines(capped_entries)
    context_block = _format_context_lines(capped_context)

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
    for item in capped_entries:
        for label in item.get("labels", []):
            grouped_entries.setdefault(label, []).append(item["element_id"])

    grouped_context: Dict[str, List[str]] = {}
    for item in capped_context:
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

    # Three-step schema selection: Steps 1 & 2 narrow labels and rel types
    anchor_labels = _collect_labels(capped_entries) | _collect_labels(capped_context)
    selected_labels, selected_rel_types = _select_schema_for_query(
        question=state["query"],
        keywords=state.get("retrieval_keywords") or [],
        anchor_labels=anchor_labels,
    )
    labels_line = ", ".join(selected_labels)
    rel_types_line = ", ".join(selected_rel_types) if selected_rel_types else "(none selected)"

    # Step 3: Cypher generation with LLM-filtered labels and rel types
    prompt = _call_chat(
        [
            SystemMessage(
                content=(
                    f"{legal_consultant_system_prefix(lang)} "
                    "You are a Cypher expert. Generate ONE Cypher query. "
                    "Rules: max 2 hops, no variable-length paths (no *), "
                    "filter nodes with elementId() only, LIMIT 8. "
                    "The ONLY valid Cypher patterns are: "
                    "MATCH (d:Document)-[:CONTAINS]->(s:Section) "
                    "MATCH (s:Section)-[:PART_OF]->(d:Document) "
                    "MATCH (d:Document)-[:PUBLISHED]->(la:LegalAct) "
                    "Do NOT use any other relationship type. Do NOT use HAS_CHUNK, "
                    "REFERENCES, AMENDS, REPEALS or any other relationship. "
                    "A valid intersection query example: "
                    "MATCH (d:Document)-[:CONTAINS]->(s:Section) "
                    "WHERE elementId(d) IN ['id1', 'id2'] "
                    "AND elementId(s) IN ['id3', 'id4'] "
                    "RETURN d, s "
                    "LIMIT 8 "
                    "CRITICAL: elementId is a function in Neo4j 5, NOT a property. "
                    "Never write: MATCH (n {elementId: '...'}) "
                    "Always write: MATCH (n) WHERE elementId(n) = '...' "
                    "For multiple nodes use: WHERE elementId(n) IN ['...', '...'] "
                    "Return ONLY the Cypher query, nothing else."
                )
            ),
            HumanMessage(
                content=(
                    "Original question: {question}\n\n"
                    "Node labels in scope: {labels}\n"
                    "Allowed relationship types: {rel_types}\n\n"
                    "Subject entry nodes:\n{entries}\n\n"
                    "Subject IDs by label:\n{entry_groups}\n\n"
                    "Context candidate nodes:\n{contexts}\n\n"
                    "Context IDs by label:\n{context_groups}\n\n"
                    "{relationship_context}\n\n"
                    "Construct ONE Cypher query that finds paths between Subject and Context nodes. "
                    "Use elementId() to filter nodes. Max 2 hops, no variable-length paths. "
                    "LIMIT 8. ONE RETURN statement at the end."
                ).format(
                    question=state["query"],
                    labels=labels_line,
                    rel_types=rel_types_line,
                    entries=entry_block,
                    entry_groups=grouped_entries_text,
                    contexts=context_block,
                    context_groups=grouped_context_text,
                    relationship_context=relationship_context,
                )
            ),
        ],
        max_tokens=500,
    )

    cypher = _clean_cypher(prompt)
    cypher = _enforce_relation_directions(cypher)
    logger.info("Generated intersection Cypher: %s", cypher)
    log_cypher_multiline(
        "b_draft",
        "intersection: Cypher string as generated (next step: execute on Neo4j)",
        cypher,
    )

    return {
        "cypher_query": cypher,
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
        return {
            "cypher_query": None,
            "cypher_generation_error": "Context-only path: no vector context nodes.",
            "cypher_attempt": "context_only",
        }

    capped_context = context_nodes[:_MAX_CONTEXT_NODES_FOR_PROMPT]
    context_block = _format_context_lines(capped_context)
    grouped_context: Dict[str, List[str]] = {}
    for item in capped_context:
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
            "context_count": len(capped_context),
            "keywords": state.get("retrieval_keywords") or [],
        },
    )

    # Three-step schema selection: Steps 1 & 2 narrow labels and rel types
    anchor_labels = _collect_labels(capped_context)
    selected_labels, selected_rel_types = _select_schema_for_query(
        question=state["query"],
        keywords=state.get("retrieval_keywords") or [],
        anchor_labels=anchor_labels,
    )
    labels_line = ", ".join(selected_labels)
    rel_types_line = ", ".join(selected_rel_types) if selected_rel_types else "(none selected)"

    # Step 3: Cypher generation with LLM-filtered labels and rel types
    prompt = _call_chat(
        [
            SystemMessage(
                content=(
                    f"{legal_consultant_system_prefix(lang)} "
                    "You are a Cypher expert. Generate ONE Cypher query. "
                    "Rules: max 2 hops, no variable-length paths (no *), "
                    "filter nodes with elementId() only, LIMIT 8. "
                    "The ONLY valid Cypher patterns are: "
                    "MATCH (d:Document)-[:CONTAINS]->(s:Section) "
                    "MATCH (s:Section)-[:PART_OF]->(d:Document) "
                    "MATCH (d:Document)-[:PUBLISHED]->(la:LegalAct) "
                    "Do NOT use any other relationship type. Do NOT use HAS_CHUNK, "
                    "REFERENCES, AMENDS, REPEALS or any other relationship. "
                    "A valid intersection query example: "
                    "MATCH (d:Document)-[:CONTAINS]->(s:Section) "
                    "WHERE elementId(d) IN ['id1', 'id2'] "
                    "AND elementId(s) IN ['id3', 'id4'] "
                    "RETURN d, s "
                    "LIMIT 8 "
                    "CRITICAL: elementId is a function in Neo4j 5, NOT a property. "
                    "Never write: MATCH (n {elementId: '...'}) "
                    "Always write: MATCH (n) WHERE elementId(n) = '...' "
                    "For multiple nodes use: WHERE elementId(n) IN ['...', '...'] "
                    "Return ONLY the Cypher query, nothing else."
                )
            ),
            HumanMessage(
                content=(
                    "Original question: {question}\n"
                    "Generalized topic: {generalized}\n"
                    "Keywords: {keywords}\n\n"
                    "Node labels in scope: {labels}\n"
                    "Allowed relationship types: {rel_types}\n\n"
                    "Semantic anchor nodes (use these elementIds):\n{contexts}\n\n"
                    "Context IDs by label:\n{context_groups}\n\n"
                    "Construct ONE Cypher query to retrieve material from the graph that best answers the question. "
                    "Use elementId() to filter nodes. Max 2 hops, no variable-length paths. "
                    "LIMIT 8. ONE RETURN statement at the end."
                ).format(
                    question=state["query"],
                    generalized=state.get("generalized_query") or state["query"],
                    keywords=", ".join(state.get("retrieval_keywords") or []),
                    labels=labels_line,
                    rel_types=rel_types_line,
                    contexts=context_block,
                    context_groups=grouped_context_text,
                )
            ),
        ],
        max_tokens=500,
    )

    cypher = _clean_cypher(prompt)
    cypher = _enforce_relation_directions(cypher)
    logger.info("Generated context-only Cypher: %s", cypher)
    log_cypher_multiline(
        "b_draft",
        "context_only: Cypher string as generated (next step: execute on Neo4j)",
        cypher,
    )

    return {
        "cypher_query": cypher,
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
        return {
            "cypher_query": None,
            "cypher_generation_error": "Fallback: no subject entry nodes available.",
            "cypher_attempt": "fallback",
        }

    context_nodes = state.get("context_nodes") or []

    # Cap nodes to avoid exceeding token limits
    capped_entries = entry_nodes[:_MAX_ENTRY_NODES_FOR_PROMPT]
    capped_context = context_nodes[:_MAX_CONTEXT_NODES_FOR_PROMPT]

    entry_block = _format_entry_lines(capped_entries)
    grouped_entries: Dict[str, List[str]] = {}
    for item in capped_entries:
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
    context_summary = _format_context_lines(capped_context)

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

    # Three-step schema selection: Steps 1 & 2 narrow labels and rel types
    anchor_labels = _collect_labels(capped_entries) | _collect_labels(capped_context)
    selected_labels, selected_rel_types = _select_schema_for_query(
        question=state["query"],
        keywords=state.get("retrieval_keywords") or [],
        anchor_labels=anchor_labels,
    )
    labels_line = ", ".join(selected_labels)
    rel_types_line = ", ".join(selected_rel_types) if selected_rel_types else "(none selected)"

    # Step 3: Cypher generation with LLM-filtered labels and rel types
    prompt = _call_chat(
        [
            SystemMessage(
                content=(
                    f"{legal_consultant_system_prefix(lang)} "
                    "You are a Cypher expert. Generate ONE Cypher query. "
                    "Rules: max 2 hops, no variable-length paths (no *), "
                    "filter nodes with elementId() only, LIMIT 8. "
                    "The ONLY valid Cypher patterns are: "
                    "MATCH (d:Document)-[:CONTAINS]->(s:Section) "
                    "MATCH (s:Section)-[:PART_OF]->(d:Document) "
                    "MATCH (d:Document)-[:PUBLISHED]->(la:LegalAct) "
                    "Do NOT use any other relationship type. Do NOT use HAS_CHUNK, "
                    "REFERENCES, AMENDS, REPEALS or any other relationship. "
                    "A valid intersection query example: "
                    "MATCH (d:Document)-[:CONTAINS]->(s:Section) "
                    "WHERE elementId(d) IN ['id1', 'id2'] "
                    "AND elementId(s) IN ['id3', 'id4'] "
                    "RETURN d, s "
                    "LIMIT 8 "
                    "CRITICAL: elementId is a function in Neo4j 5, NOT a property. "
                    "Never write: MATCH (n {elementId: '...'}) "
                    "Always write: MATCH (n) WHERE elementId(n) = '...' "
                    "For multiple nodes use: WHERE elementId(n) IN ['...', '...'] "
                    "Return ONLY the Cypher query, nothing else."
                )
            ),
            HumanMessage(
                content=(
                    "Original question: {question}\n"
                    "Reason for fallback: {reason}\n\n"
                    "Node labels in scope: {labels}\n"
                    "Allowed relationship types: {rel_types}\n\n"
                    "Subject entry nodes:\n{entries}\n\n"
                    "Subject IDs by label:\n{entry_groups}\n\n"
                    "Context hints:\n{contexts}\n\n"
                    "{relationship_context}\n\n"
                    "Generate ONE Cypher query starting from subject IDs using elementId() filters. "
                    "Max 2 hops, no variable-length paths. LIMIT 8. ONE RETURN statement at the end."
                ).format(
                    question=state["query"],
                    reason=fallback_reason,
                    labels=labels_line,
                    rel_types=rel_types_line,
                    entries=entry_block,
                    entry_groups=grouped_entries_text,
                    contexts=context_summary,
                    relationship_context=relationship_context,
                )
            ),
        ],
        max_tokens=500,
    )

    cypher = _clean_cypher(prompt)
    cypher = _enforce_relation_directions(cypher)
    logger.info("Generated fallback Cypher: %s", cypher)
    log_cypher_multiline(
        "b_draft",
        "fallback: Cypher string as generated (next step: execute on Neo4j)",
        cypher,
    )

    return {
        "cypher_query": cypher,
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

    # Cap nodes to avoid exceeding token limits
    capped_entries = entry_nodes[:_MAX_ENTRY_NODES_FOR_PROMPT]

    entry_block = _format_entry_lines(capped_entries)
    grouped_entries: Dict[str, List[str]] = {}
    for item in capped_entries:
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

    # Three-step schema selection: Steps 1 & 2 narrow labels and rel types
    anchor_labels = _collect_labels(capped_entries)
    selected_labels, selected_rel_types = _select_schema_for_query(
        question=state["query"],
        keywords=state.get("retrieval_keywords") or [],
        anchor_labels=anchor_labels,
    )
    labels_line = ", ".join(selected_labels)
    rel_types_line = ", ".join(selected_rel_types) if selected_rel_types else "(none selected)"

    # Step 3: Cypher generation with LLM-filtered labels and rel types
    prompt = _call_chat(
        [
            SystemMessage(
                content=(
                    f"{legal_consultant_system_prefix(lang)} "
                    "You are a Cypher expert. Generate ONE improved Cypher query. "
                    "Address the critique; broaden paths or add patterns where useful. "
                    "Rules: max 2 hops, no variable-length paths (no *), "
                    "filter nodes with elementId() only, LIMIT 8. "
                    "The ONLY valid Cypher patterns are: "
                    "MATCH (d:Document)-[:CONTAINS]->(s:Section) "
                    "MATCH (s:Section)-[:PART_OF]->(d:Document) "
                    "MATCH (d:Document)-[:PUBLISHED]->(la:LegalAct) "
                    "Do NOT use any other relationship type. Do NOT use HAS_CHUNK, "
                    "REFERENCES, AMENDS, REPEALS or any other relationship. "
                    "A valid intersection query example: "
                    "MATCH (d:Document)-[:CONTAINS]->(s:Section) "
                    "WHERE elementId(d) IN ['id1', 'id2'] "
                    "AND elementId(s) IN ['id3', 'id4'] "
                    "RETURN d, s "
                    "LIMIT 8 "
                    "CRITICAL: elementId is a function in Neo4j 5, NOT a property. "
                    "Never write: MATCH (n {elementId: '...'}) "
                    "Always write: MATCH (n) WHERE elementId(n) = '...' "
                    "For multiple nodes use: WHERE elementId(n) IN ['...', '...'] "
                    "Return ONLY the Cypher query, nothing else."
                )
            ),
            HumanMessage(
                content=(
                    "Original question: {question}\n"
                    "Critique of previous retrieval: {feedback}\n\n"
                    "Previous Cypher (may be suboptimal):\n{previous}\n\n"
                    "Node labels in scope: {labels}\n"
                    "Allowed relationship types: {rel_types}\n\n"
                    "Subject entry nodes:\n{entries}\n\n"
                    "Subject IDs by label:\n{entry_groups}\n\n"
                    "{relationship_context}\n\n"
                    "Produce ONE improved Cypher query with ONE RETURN. "
                    "Max 2 hops, no variable-length paths. LIMIT 8."
                ).format(
                    question=state["query"],
                    feedback=feedback,
                    previous=previous or "(none)",
                    labels=labels_line,
                    rel_types=rel_types_line,
                    entries=entry_block,
                    entry_groups=grouped_entries_text,
                    relationship_context=relationship_context,
                )
            ),
        ],
        max_tokens=500,
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

def execute_cypher(state: Dict[str, Any], driver, database: str) -> Dict[str, Any]:
    cypher = state.get("cypher_query")
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
            "raw_result": [r for r in state.get("raw_result", []) if r.get("_source") == "bm25"],
            "execution_error": state.get("cypher_generation_error"),
            "neo4j_executed": False,
        }

    # Exact string passed to Neo4j driver (verbatim, including whitespace)
    log_cypher_multiline(
        "c_execute",
        f"Query submitted to Neo4j database={database!r} attempt={attempt!r} (exact string below)",
        cypher,
    )

    try:
        import time as _time
        _t0 = _time.time()
        with driver.session(database=database) as session:
            records = session.run(cypher)
            data = [record.data() for record in records]
        vlog("neo4j_query", {"attempt": attempt, "cypher_length": len(cypher), "row_count": len(data)}, (_time.time() - _t0) * 1000)
    except Neo4jError as exc:
        logger.error("Cypher execution failed during %s attempt: %s", attempt, exc)
        log_cypher_event(
            "c_execute",
            f"Neo4j driver error after submit attempt={attempt!r} database={database!r}",
            detail={"error": str(exc)},
        )
        return {
            "raw_result": [r for r in state.get("raw_result", []) if r.get("_source") == "bm25"],
            "execution_error": str(exc),
            "neo4j_executed": True,
        }

    logger.info("Cypher execution (%s) returned %d rows", attempt, len(data))
    log_cypher_event(
        "c_execute",
        f"Neo4j execution finished attempt={attempt!r} rows={len(data)}",
        detail={"result_column_keys": list(data[0].keys()) if data else []},
    )

    # Tier 1 fallback: intersection returned 0 rows but vector search has context nodes.
    # Trust the vector search results and use them directly so synthesis has something to work
    # with and route_after_execution routes to "evaluate" instead of "retry".
    if (
        not data
        and state.get("turn_count") == 1
        and attempt == "intersection"
    ):
        context_nodes = state.get("context_nodes") or []
        if context_nodes:
            logger.info(
                "Tier 1 intersection returned 0 rows; falling back to %d context nodes from vector search",
                len(context_nodes),
            )
            log_cypher_event(
                "c_tier1_fallback",
                f"Tier 1 intersection empty — using {len(context_nodes)} vector-search context nodes as raw_result",
                detail={"context_node_count": len(context_nodes)},
            )
            return {
                "raw_result": context_nodes,
                "execution_error": None,
                "neo4j_executed": True,
            }

    # Apply document scoping to intersection results if a law hint is present
    doc_hint = state.get("law_hint_doc_id")
    if doc_hint and data:
        before = len(data)
        data = [
            row for row in data
            if any(
                isinstance(v, dict) and v.get("id", "").startswith(doc_hint)
                for v in row.values()
            )
        ]
        logger.info(
            "execute_cypher: scoped intersection results to %r — %d/%d rows kept",
            doc_hint, len(data), before,
        )

    user_id = state.get("user_id") or ""
    tenant_id = state.get("tenant_id") or ""
    if user_id or tenant_id:
        with driver.session(database=database) as _s:
            allowed_ids = _fetch_allowed_doc_ids(_s, user_id, tenant_id)
        data = [
            r for r in data
            if (r.get("d") or {}).get("id") in allowed_ids
            or (r.get("s") or {}).get("id", "").startswith("DOCUMENT_SECTION::")
        ]

    enriched_references = _enrich_with_source_metadata(data)

    existing_raw = state.get("raw_result", [])
    bm25_rows = [r for r in existing_raw if r.get("_source") == "bm25"]
    merged = list(data) + bm25_rows

    return {
        "raw_result": merged,
        "execution_error": None,
        "references": enriched_references,
        "neo4j_executed": True,
    }


# ---------------------------------------------------------------------------
# Node F1: Retrieval quality evaluation
# ---------------------------------------------------------------------------


def evaluate_retrieval_quality(state: Dict[str, Any], driver=None, database: str = "neo4j") -> Dict[str, Any]:
    """LLM critique of retrieved rows before synthesis; may trigger reformulation (max two)."""
    lang = _session_lang(state)
    data = state.get("raw_result") or []
    total_row_count = len(data)  # includes BM25 rows pre-populated by context_retrieval
    status = list(state.get("status_messages") or [])
    # Skip LLM evaluation when all rows come from a direct article lookup —
    # article_router matched by article number, result is already exact, no judgement needed.
    if data and all(r.get("_source") == "bm25" for r in data) and state.get("bm25_doc_ids") and state.get("bm25_from_article_lookup"):
        return {
            **state,
            "retrieval_quality_ok": True,
            "status_messages": status,
        }
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

    summarized_data = _summarize_for_synthesis(data, max_records=25)
    for rec in summarized_data:
        if rec.get("_source") == "bm25":
            rec["_source"] = "[BM25] direct fulltext match"
    serialized = json.dumps(summarized_data, ensure_ascii=False)

    r_before = int(state.get("quality_reformulation_round") or 0)
    log_cypher_event(
        "d_evaluate_start",
        "critical retrieval evaluation (LLM) — starting",
        detail={
            "user_query": state["query"],
            "row_count": total_row_count,
            "cypher_attempt": state.get("cypher_attempt"),
            "quality_reformulation_round_before": r_before,
        },
    )

    keywords = state.get("retrieval_keywords") or []
    q_short = ", ".join(keywords) if keywords else state["query"][:100]
    import time as _time
    _t0 = _time.time()
    verdict_raw = _call_chat(
        [
            SystemMessage(
                content=(
                    "You judge whether a set of retrieved legal sections contains at least one section that directly addresses the user's question. "
                    "Reply with exactly two lines: "
                    "Line 1: OK or POOR (uppercase). "
                    "Line 2: one short sentence explaining why. "
                    "Mark OK if ANY single section in the retrieved set directly addresses the question — "
                    "one relevant section among many irrelevant ones is enough to mark OK. "
                    "Mark POOR only if EVERY section is completely unrelated to the question, or the set is entirely empty. "
                    "When in doubt, mark OK. "
                    "If ANY row is tagged '[BM25] direct fulltext match', treat it as a strong relevance signal — "
                    "return OK if its plain_text or abstract addresses the question, regardless of other rows."
                )
            ),
            HumanMessage(
                content=(
                    "Question:\n{q}\n\n"
                    "Summarized rows:\n{rows}\n\n"
                    "Verdict:"
                ).format(q=q_short, rows=serialized[:4000])
            ),
        ],
        max_tokens=80,
    )
    lines = [ln.strip() for ln in (verdict_raw or "").splitlines() if ln.strip()]
    head = lines[0].upper() if lines else "POOR"
    poor = head.startswith("POOR")
    feedback = lines[1] if len(lines) > 1 else ""
    bm25_rows_in_result = sum(
        1 for r in data
        if r.get("_source") == "bm25" or "bm25" in (r.get("sources") or ())
    )
    vlog("evaluator_llm", {"query": state["query"][:80], "row_count": total_row_count, "bm25_rows_in_result": bm25_rows_in_result, "verdict": head, "reason": feedback[:120], "verdict_overridden_by_bm25": bm25_rows_in_result > 0 and poor}, (_time.time() - _t0) * 1000)

    log_cypher_multiline(
        "d_evaluate_llm",
        "raw LLM verdict output (line 1: OK|POOR, line 2: reason)",
        verdict_raw or "",
        delimiter_label="LLM_VERDICT",
    )

    if bm25_rows_in_result > 0 and poor and state.get("bm25_from_article_lookup"):
        bm25_only = [r for r in data if r.get("_source") == "bm25"]
        log_cypher_event(
            "d_evaluate_bm25_override",
            f"BM25 found {bm25_rows_in_result} relevant sections from article lookup — overriding POOR verdict",
            detail={"bm25_rows_in_result": bm25_rows_in_result, "llm_verdict": head, "feedback": feedback},
        )
        bm25_doc_ids = list({r.get("d", {}).get("id", "") for r in bm25_only if r.get("d", {}).get("id")})
        return {
            "retrieval_quality_ok": True,
            "raw_result": bm25_only,
            "quality_feedback": feedback,
            "status_messages": status,
            "retrieval_evaluated": True,
            "bm25_doc_ids": bm25_doc_ids,
            "bm25_from_article_lookup": True,
        }

    # Second override: scoped BM25 returned results for a general query
    # The fulltext index already confirmed these sections exist in the right document
    if bm25_rows_in_result > 0 and poor and not state.get("bm25_from_article_lookup"):
        bm25_ids = [
            r.get("d", {}).get("id", "")
            for r in data
            if r.get("_source") == "bm25"
        ]
        if bm25_ids:
            logger.info("BM25 general override fired — bm25_rows=%d, routing to synthesis", bm25_rows_in_result)
            return {
                "retrieval_quality_ok": True,
                "bm25_doc_ids": bm25_ids,
                "retrieval_evaluated": True,
                "status_messages": state.get("status_messages", []),
                "raw_result": [r for r in data if "bm25" in str(r.get("_source", ""))],
            }

    # Override POOR when the queried article is directly present in the retrieved data
    if poor and total_row_count >= 1:
        _art_ref = re.search(r'\b(?:articolo|art\.?)\s*(\d+)', state["query"], re.IGNORECASE)
        if _art_ref:
            art_num = _art_ref.group(1)
            article_rows = [
                r for r in data
                if isinstance(r.get("s"), dict) and r["s"].get("name") == art_num
            ]
            feedback_mentions_article = art_num in feedback
            if article_rows or feedback_mentions_article:
                _override_rows = article_rows or data
                _override_doc_ids = list({
                    r.get("d", {}).get("id", "")
                    for r in _override_rows
                    if isinstance(r.get("d"), dict) and r.get("d", {}).get("id")
                })
                log_cypher_event(
                    "d_evaluate_article_override",
                    f"Article {art_num} found in data (rows={len(_override_rows)}) — overriding POOR verdict",
                    detail={"art_num": art_num, "article_rows": len(article_rows), "feedback_mentions": feedback_mentions_article},
                )
                return {
                    "retrieval_quality_ok": True,
                    "raw_result": _override_rows,
                    "quality_feedback": None,
                    "status_messages": status,
                    "retrieval_evaluated": True,
                    "bm25_doc_ids": _override_doc_ids,
                }

    bm25_doc_ids_from_data = [
        r.get("d", {}).get("id", "")
        for r in data
        if r.get("_source") == "bm25" and r.get("d", {}).get("id", "")
    ]

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
            "bm25_doc_ids": state.get("bm25_doc_ids") or bm25_doc_ids_from_data,
        }
    if r < 2:
        # On the first evaluation of an intersection attempt, if vector search produced
        # context nodes, trust them directly rather than entering reformulation.
        if r == 0 and state.get("cypher_attempt") == "intersection":
            log_cypher_event(
                "d_evaluate_fallback_debug",
                "checking vector-search fallback eligibility",
                detail={
                    "r": r,
                    "cypher_attempt": state.get("cypher_attempt"),
                    "context_nodes_count": len(state.get("context_nodes") or []),
                },
            )
            context_nodes = state.get("context_nodes") or []
            if context_nodes and driver:
                element_ids = [n["element_id"] for n in context_nodes if n.get("element_id")]
                fetched: List[Dict[str, Any]] = []
                if element_ids:
                    _user_id = state.get("user_id") or ""
                    _tenant_id = state.get("tenant_id") or ""
                    try:
                        with driver.session(database=database) as neo4j_session:
                            records = neo4j_session.run(
                                "MATCH (d:Document)-[:CONTAINS]->(s:Section)\n"
                                "WHERE elementId(s) IN $element_ids\n"
                                f"AND {_visibility_filter()}\n"
                                "RETURN d, s",
                                element_ids=element_ids,
                                user_id=_user_id,
                                tenant_id=_tenant_id,
                            )
                            fetched = [record.data() for record in records]
                    except Exception as exc:
                        logger.warning("Vector-search fallback Neo4j fetch failed: %s", exc)
                if fetched:
                    # Merge fetched rows with BM25 rows already in raw_result
                    bm25_existing = list(state.get("raw_result") or [])
                    fetched_section_ids = {
                        row["s"].get("id") for row in fetched
                        if isinstance(row.get("s"), dict) and row["s"].get("id")
                    }
                    merged_result = list(fetched)
                    for row in bm25_existing:
                        row_sid = row["s"].get("id") if isinstance(row.get("s"), dict) else None
                        if row_sid and row_sid not in fetched_section_ids:
                            merged_result.append(row)
                            fetched_section_ids.add(row_sid)
                    log_cypher_event(
                        "d_evaluate_vector_fallback",
                        f"POOR on intersection round 0 — bypassing reformulation, fetched {len(fetched)} records from {len(context_nodes)} vector-search context nodes",
                        detail={"context_node_count": len(context_nodes), "fetched_row_count": len(fetched), "bm25_row_count": len(merged_result) - len(fetched), "feedback": feedback},
                    )
                    return {
                        "retrieval_quality_ok": True,
                        "raw_result": merged_result,
                        "quality_feedback": feedback,
                        "status_messages": status,
                        "retrieval_evaluated": True,
                        "retrieval_fallback": False,
                    }
        return {
            "retrieval_quality_ok": state.get("retrieval_quality_ok", False),
            "quality_reformulation_round": r + 1,
            "quality_feedback": feedback,
            "status_messages": status,
            "retrieval_evaluated": True,
        }
    # Cap reached: retrieved data was POOR quality. Clear raw_result so synthesize_answer
    # uses the empty/no-data path rather than synthesizing from irrelevant results.
    # retrieval_quality_ok stays True so route_after_evaluation still routes to "synthesize".
    return {
        "retrieval_quality_ok": True,
        "raw_result": [],
        "quality_feedback": feedback,
        "status_messages": status,
        "retrieval_evaluated": True,
        "retrieval_fallback": True,
    }
