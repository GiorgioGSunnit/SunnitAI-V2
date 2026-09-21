"""Pure formatting and conversion helpers shared by the RAG graph nodes."""

import json
from typing import Any, Dict, List, Set

from .language import SessionLang, normalize_lang


def _session_lang(state: Dict[str, Any]) -> SessionLang:
    return normalize_lang(state.get("session_language"))


def _collect_labels(nodes: List[Dict[str, Any]]) -> Set[str]:
    """Extract all unique labels from a list of node dicts."""
    labels: Set[str] = set()
    for n in nodes:
        for lbl in n.get("labels", []):
            labels.add(lbl)
    return labels


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


def _summarize_for_synthesis(
    data: List[Dict[str, Any]], max_records: int = 5, is_comparison: bool = False
) -> List[Dict[str, Any]]:
    summarized = []
    total_chars = 0
    MAX_TOTAL_CHARS = 4000 if is_comparison else 6000

    for record in data[:max_records]:
        if is_comparison and record.get("_source") == "comparison":
            if total_chars > MAX_TOTAL_CHARS:
                break
            comparison_count = sum(1 for r in summarized if r.get("_source") == "comparison")
            if comparison_count >= 5:
                continue
            rec = {k: v for k, v in record.items() if k not in ("embedding", "vettore")}
            for node_key in ("s", "s2"):
                if isinstance(rec.get(node_key), dict):
                    rec[node_key] = {
                        k: v for k, v in rec[node_key].items()
                        if k not in ("embedding", "vettore", "embedding_dim")
                    }
                    if rec[node_key].get("plain_text"):
                        rec[node_key]["plain_text"] = rec[node_key]["plain_text"][:500]
                    if rec[node_key].get("abstract"):
                        rec[node_key]["abstract"] = rec[node_key]["abstract"][:200]
            rec_json = json.dumps(rec, ensure_ascii=False)
            total_chars += len(rec_json)
            summarized.append(rec)
            continue
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
                    abstract = (props.get("abstract") or props.get("description") or "")[:200]
                    if abstract:
                        summary_props["abstract"] = abstract
                    plain_text = (props.get("plain_text") or props.get("text") or "")[:150]
                    if plain_text:
                        summary_props["plain_text"] = plain_text

                summary_record[key] = {k: v for k, v in summary_props.items() if v is not None}
            elif isinstance(value, dict):
                # Flat property dict (no "properties" wrapper) — infer labels from key name
                node_id = value.get("id") or ""
                labels = (
                    ["Document"] if (key == "d" or node_id.startswith("LEGAL_DOC::"))
                    else ["Section"] if key == "s"
                    else []
                )
                is_doc = "Document" in labels
                flat_props: Dict[str, Any] = {"labels": labels} if labels else {}
                if is_doc:
                    if value.get("name"):
                        flat_props["name"] = value["name"]
                    description = (value.get("description") or "")[:200]
                    if description:
                        flat_props["description"] = description
                    if node_id:
                        flat_props["id"] = node_id
                elif key == "s":
                    # Section node — article number, full text, abstract, parent doc name
                    if value.get("name"):
                        flat_props["name"] = value["name"]
                    abstract = (value.get("abstract") or "")[:200]
                    if abstract:
                        flat_props["abstract"] = abstract
                    plain_text = (value.get("plain_text") or value.get("text") or "")[:500]
                    if plain_text:
                        flat_props["plain_text"] = plain_text
                    d_node = record.get("d") or {}
                    doc_name = (
                        d_node.get("name") or d_node.get("nomedocumento")
                        or d_node.get("document_title") or ""
                    )
                    if doc_name:
                        flat_props["document_name"] = doc_name
                else:
                    # Generic flat node
                    title = (value.get("title") or "")[:80]
                    if title:
                        flat_props["title"] = title
                    if value.get("name"):
                        flat_props["name"] = value["name"]
                    text_en = (value.get("text_en") or "")[:80]
                    if text_en:
                        flat_props["text_en"] = text_en
                    abstract = (value.get("abstract") or value.get("description") or "")[:200]
                    if abstract:
                        flat_props["abstract"] = abstract
                    plain_text = (value.get("plain_text") or value.get("text") or "")[:150]
                    if plain_text:
                        flat_props["plain_text"] = plain_text
                if flat_props:
                    summary_record[key] = flat_props
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

    if len(summarized) > 10:
        summarized = summarized[:10]
        summarized.append({"note": "results truncated to 10 items"})
    return summarized


def _node_to_dict(node) -> dict:
    """Convert a Neo4j node or plain dict to a plain dict."""
    if node is None:
        return {}
    if isinstance(node, dict):
        return node
    try:
        return dict(node)
    except Exception:
        return {}

