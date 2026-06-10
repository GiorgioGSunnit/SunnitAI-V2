import json
import os
import re
import unicodedata
from typing import Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ..preprocessing.schema.schema import entities as schema_entities
from ..preprocessing.schema.schema import relations as schema_relations

_CODE_BLOCK_PATTERN = re.compile(r"```(?:cypher)?\n(.*?)\n```", re.DOTALL)
_AL_PREFIX_PATTERN = re.compile(r"^\b(al-?|el-?)\b", re.IGNORECASE)

# Pre-built label list — used as fallback in _select_schema_for_query Step 1
_ALL_SCHEMA_LABELS: List[str] = [item["label"] for item in schema_entities]


def canonical_name(value: str) -> Optional[str]:
    if value is None:
        return None
    
    normalized = unicodedata.normalize("NFKC", value)
    normalized = _AL_PREFIX_PATTERN.sub("", normalized).strip()
    normalized = normalized.lower()
    normalized = re.sub(r"[^\w\s\-]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    
    try:
        normalized = normalized.encode("ascii", "ignore").decode("ascii")
    except Exception:
        pass
        
    return normalized or None


def _build_schema_text() -> str:
    """Full schema text (used for entity extraction where all types matter)."""
    entities_block = "\n".join(
        f'- {item["label"]} ({", ".join(item["key"])})' for item in schema_entities
    )
    relations_block = "\n".join(
        f'- {item["from"]} -[:{item["type"]}]-> {item["to"]}' for item in schema_relations
    )
    return (
        "Node Labels and Primary Keys:\n"
        f"{entities_block}\n\n"
        "Relationship Types:\n"
        f"{relations_block}"
    )


def _build_filtered_schema_text(relevant_labels: set) -> str:
    """Schema text filtered to only node types and relations relevant to a query.

    Includes any node type that appears in relevant_labels, plus any node type
    that is connected to a relevant label via a relationship (1-hop neighborhood).
    """
    if not relevant_labels:
        return _build_schema_text()

    # Expand to 1-hop neighbors so the LLM can traverse from anchors
    expanded = set(relevant_labels)
    for rel in schema_relations:
        if rel["from"] in relevant_labels or rel["to"] in relevant_labels:
            expanded.add(rel["from"])
            expanded.add(rel["to"])

    entities_block = "\n".join(
        f'- {item["label"]} ({", ".join(item["key"])})'
        for item in schema_entities
        if item["label"] in expanded
    )
    relations_block = "\n".join(
        f'- {item["from"]} -[:{item["type"]}]-> {item["to"]}'
        for item in schema_relations
        if item["from"] in expanded and item["to"] in expanded
    )
    return (
        "Node Labels and Primary Keys:\n"
        f"{entities_block}\n\n"
        "Relationship Types:\n"
        f"{relations_block}"
    )


def _build_filtered_relation_hints(relevant_labels: set) -> str:
    """Relation hints filtered to only relationships touching relevant labels."""
    if not relevant_labels:
        return "\n".join(
            f"- {item['from']} -[:{item['type']}]-> {item['to']}" for item in schema_relations
        )

    expanded = set(relevant_labels)
    for rel in schema_relations:
        if rel["from"] in relevant_labels or rel["to"] in relevant_labels:
            expanded.add(rel["from"])
            expanded.add(rel["to"])

    return "\n".join(
        f"- {item['from']} -[:{item['type']}]-> {item['to']}"
        for item in schema_relations
        if item["from"] in expanded and item["to"] in expanded
    )


def _strict_filter_relations(selected_labels: set) -> List[Dict]:
    """Return schema relations where BOTH endpoints are in selected_labels.

    No 1-hop expansion — only edges that directly connect the chosen labels.
    Used as the Python pre-filter before the Step 2 LLM call in the
    three-step Cypher generation pipeline.
    Returns all relations if selected_labels is empty.
    """
    if not selected_labels:
        return list(schema_relations)
    return [
        r for r in schema_relations
        if r["from"] in selected_labels and r["to"] in selected_labels
    ]


def _parse_json_list(response: str) -> List[str]:
    """Robustly parse an LLM response that should be a JSON array of strings.

    Strips markdown fences, finds the first [...] block, removes trailing
    commas before ], and attempts JSON parsing. Returns [] on any failure.
    """
    if not response:
        return []
    # Strip markdown fences (```json ... ``` or ``` ... ```)
    text = re.sub(r"```(?:json)?\s*", "", response).strip().rstrip("`").strip()
    # Extract the first [...] block — the LLM may add prose before/after
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if match:
        text = match.group(0)
    # Remove trailing commas before ] — a common LLM formatting mistake
    text = re.sub(r",\s*\]", "]", text)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(x) for x in parsed if x]
    except (json.JSONDecodeError, ValueError):
        pass
    return []


def _enforce_relation_directions(cypher: str) -> str:
    # Sanitizer: rewrite known relation patterns to enforce directionality
    def _replace(pattern: str, template: str, text: str) -> str:
        def repl(match: re.Match[str]) -> str:
            return template.format(**match.groupdict())

        return re.sub(pattern, repl, text)

    cypher = _replace(
        r"\(\s*(?P<inst_var>[A-Za-z_][\w]*)\s*:\s*Institution\s*\)\s*-\s*\[:IMPOSED_BY\]->\s*\(\s*(?P<pen_var>[A-Za-z_][\w]*)\s*:\s*Penalty\s*\)",
        "({pen_var}:Penalty)-[:IMPOSED_BY]->({inst_var}:Institution)",
        cypher,
    )
    cypher = _replace(
        r"\(\s*(?P<comp_var>[A-Za-z_][\w]*)\s*:\s*Company\s*\)\s*-\s*\[:IMPOSED_ON\]->\s*\(\s*(?P<pen_var>[A-Za-z_][\w]*)\s*:\s*Penalty\s*\)",
        "({pen_var}:Penalty)-[:IMPOSED_ON]->({comp_var}:Company)",
        cypher,
    )
    return cypher


def _clean_cypher(raw_cypher: str) -> str:
    # Clean Cypher from markdown code blocks and remove comments/blank lines
    if not raw_cypher:
        return ""
    match = _CODE_BLOCK_PATTERN.search(raw_cypher)
    if match:
        cypher = match.group(1)
    else:
        cypher = raw_cypher

    lines = [
        line.strip()
        for line in cypher.split("\n")
        if line.strip() and not line.strip().startswith("//")
    ]
    return "\n".join(lines)
