import os
import re
import unicodedata
from typing import Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ..preprocessing.schema.schema import entities as schema_entities
from ..preprocessing.schema.schema import relations as schema_relations

_CODE_BLOCK_PATTERN = re.compile(r"```(?:cypher)?\n(.*?)\n```", re.DOTALL)
_AL_PREFIX_PATTERN = re.compile(r"^\b(al-?|el-?)\b", re.IGNORECASE)


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
    # Build schema text for LLM prompts
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
