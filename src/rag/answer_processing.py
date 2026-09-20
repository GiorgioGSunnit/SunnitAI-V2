"""Answer post-processing, citation extraction and Neo4j visibility helpers."""

import re
import urllib.parse
from typing import Any, Dict, List, Optional

from neo4j.graph import Node as Neo4jNode


def _visibility_filter(alias: str = "d") -> str:
    """Returns a Cypher WHERE clause fragment for document visibility.
    Documents without a visibility property are treated as public.
    """
    return (
        f"(coalesce({alias}.visibility, 'public') = 'public' "
        f"OR {alias}.owner_id = $user_id "
        f"OR {alias}.tenant_id = $tenant_id) "
        f"AND coalesce({alias}.archived, false) = false"
    )


def _fetch_allowed_doc_ids(session, user_id: str, tenant_id: str) -> set:
    """Fetch all document IDs visible to this user."""
    result = session.run("""
        MATCH (d:Document)
        WHERE coalesce(d.visibility, 'public') = 'public'
           OR d.owner_id = $user_id
           OR d.tenant_id = $tenant_id
        RETURN d.id AS id
    """, user_id=user_id or "", tenant_id=tenant_id or "")
    return {r["id"] for r in result}


_VAGUE_CLOSING_PATTERNS = re.compile(
    r"potrebbe esaminare|potrebbe essere utile|un approfondimento|potrebbe approfondire"
    r"|potremmo esaminare|possiamo esaminare"
    r"|could examine|could be useful|may be useful"
    r"|podría examinar|podría ser útil"
    r"|potrebbe essere interessante|sarebbe interessante|vale la pena esplorare"
    r"|it would be interesting|it is worth exploring|valdría la pena|sería interesante",
    re.IGNORECASE,
)


def _strip_vague_closing(text: str) -> str:
    """Remove a trailing vague/open-ended sentence from an LLM answer."""
    if not text:
        return text
    parts = re.split(r"(?<=[.!?])\s+", text.rstrip())
    if len(parts) > 1 and _VAGUE_CLOSING_PATTERNS.search(parts[-1]):
        return " ".join(parts[:-1]).rstrip()
    return text


def _strip_hallucinated_fonti(answer: str) -> str:
    """Remove Fonti:/Fonte: sections written by the LLM — replaced programmatically."""
    if not answer:
        return answer
    # Strip inline "Fonti:" sections appended by the LLM
    for marker in ["Fonti:", "Fonti :", "Sources:", "Fuentes:"]:
        idx = answer.find(marker)
        if idx != -1:
            answer = answer[:idx].strip()
    # Also strip line-starting Fonti patterns
    lines = answer.splitlines()
    clean = []
    for line in lines:
        if re.match(r'^\s*Fonti\s*:', line, re.IGNORECASE):
            break
        clean.append(line)
    return "\n".join(clean).strip()


def _extract_citations(
    raw_result: List[Dict[str, Any]],
    answer: str = "",
    doc_refs: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Extract deduplicated structured citations from Neo4j result rows.

    When answer and/or doc_refs are provided, filters to only citations whose
    document_name appears in the answer text OR whose name/id matches a detected
    document reference. Without those args, returns all citations unfiltered.
    """
    docs: Dict[str, Dict] = {}

    for record in raw_result:
        for key, value in record.items():
            if not isinstance(value, (dict, Neo4jNode)):
                continue
            node_id = value.get("id") or ""

            is_doc = key == "d" or node_id.startswith("LEGAL_DOC::")
            is_section = key == "s" or node_id.startswith("DOCUMENT_SECTION::")

            if is_doc:
                doc_id = node_id or value.get("document_id") or ""
                doc_name = value.get("name") or value.get("document_title") or doc_id
                doc_name = re.sub(r'^\s*﻿?\[[A-Z]+\]\s*', '', doc_name or '').strip()
                if not doc_id:
                    continue
                if doc_id not in docs:
                    docs[doc_id] = {"title": None, "sections": {}, "document_type": None}
                docs[doc_id]["title"] = doc_name
                docs[doc_id]["document_type"] = value.get("document_type")

            elif is_section:
                section_name = value.get("name") or value.get("title") or ""
                parts = node_id.split("::")
                doc_id = ("LEGAL_DOC::" + parts[1]) if len(parts) >= 3 else (value.get("document_id") or "")
                if not doc_id:
                    continue
                if doc_id not in docs:
                    docs[doc_id] = {"title": None, "sections": {}, "document_type": None}
                # Skip embedding property — large vector, not needed here
                value_safe = {k: v for k, v in value.items() if k not in ("embedding", "vettore")}
                section_plain_text = (value_safe.get("plain_text") or value_safe.get("text") or "").strip()
                section_title_raw = (value_safe.get("title") or "").strip() or None
                section_abstract = (value_safe.get("abstract") or "").strip()
                section_title = section_title_raw
                if not section_title:
                    try:
                        section_title = section_abstract.strip("- ").split(" - ")[0].strip()[:100] if section_abstract else None
                    except Exception:
                        section_title = None
                section_score = record.get("_reranker_score")
                # Key by node_id, not section_name: name is only guaranteed unique
                # for numbered code articles. Dottrina/special chunks derive name from
                # a truncated prose snippet, so distinct chunks can share the same name
                # and would otherwise be silently collapsed by setdefault().
                section_key = node_id or section_name
                if section_name and section_name != "0" and section_plain_text and len(section_plain_text) > 20 and not section_plain_text.strip().startswith("Torna indietro"):
                    docs[doc_id]["sections"].setdefault(section_key, {
                        "name": section_name,
                        "plain_text": section_plain_text,
                        "title": section_title,
                        "score": section_score,
                    })

    def _section_sort_key(item):
        _key, sec = item
        name = sec.get("name") or ""
        score = sec.get("score")
        if score is None:
            return (1, 0, name)
        return (0, -score, name)

    results = [
        {
            "document_name": re.sub(r'^\s*\[[A-Z]+\]\s*', '', info["title"] or doc_id),
            "document_id": doc_id,
            "document_type": info.get("document_type"),
            "sections": [
                {
                    "name": sec["name"],
                    "title": sec["title"],
                    "plain_text": sec["plain_text"],
                    "score": sec.get("score"),
                    "url": (
                        f"/api/documents/{urllib.parse.quote(doc_id, safe='')}/"
                        f"sections/{urllib.parse.quote(sec['name'], safe='')}"
                    ),
                }
                for _key, sec in sorted(info["sections"].items(), key=_section_sort_key)
                if len(sec["name"]) <= 200 and len(sec["plain_text"]) > 10
            ],
        }
        for doc_id, info in docs.items()
    ]
    if not answer and not doc_refs:
        return results
    if doc_refs:
        refs = doc_refs
        return [
            c for c in results
            if any(
                ref in (c.get("document_name") or "") or ref in (c.get("document_id") or "")
                for ref in refs
            )
        ]
    return results


# _citation_is_relevant removed — citation quality handled by reranker threshold (0.65)
# Kept here for reference in case embedding-based citation filtering is needed in future
# def _citation_is_relevant(answer: str, section_text: str, threshold: float = 0.80) -> bool:
#     if not section_text or not answer:
#         return False
#     try:
#         if len(section_text) <= 800:
#             effective_threshold = threshold - 0.08   # 0.72 for short sections
#         elif len(section_text) <= 3000:
#             effective_threshold = threshold           # 0.80 for medium sections
#         else:
#             effective_threshold = threshold - 0.12   # 0.68 for very long sections
#         answer_emb = _embed_query_with_prefix(answer[:500])
#         section_emb = _embed_query_with_prefix(section_text[:500])
#         dot = sum(a * b for a, b in zip(answer_emb, section_emb))
#         norm_a = sum(a * a for a in answer_emb) ** 0.5
#         norm_b = sum(b * b for b in section_emb) ** 0.5
#         similarity = dot / (norm_a * norm_b + 1e-9)
#         return similarity >= effective_threshold
#     except Exception:
#         return True  # keep citation on error


_GAP_PHRASES = [
    "non è presente nei documenti",
    "non ho documentazione specifica",
    "non ho informazioni specifiche",
    "not present in my knowledge base",
    "no specific documentation",
    "no tengo documentacion especifica",
    "non trovo informazioni",
    "non sono presenti nei documenti",
    "non ci siano documenti specifici",
    "nonostante non ci siano",
    "non sono presenti documenti",
    "non risultano documenti",
    "non trovo documenti",
    "among the documents provided",
    "the documents provided do not",
    "tra i documenti forniti non",
    "nei documenti forniti non",
    "documents do not contain",
    "non è trattata nei documenti",
    "non sono trattati nei documenti",
    "non viene trattata nei documenti",
    "non risulta nei documenti",
    "trattano temi diversi",
    "i documenti presenti trattano",
    "not addressed in the documents",
    "not covered in the documents",
    "no se trata en los documentos",
    "non contengono informazioni specifiche",
    "non contiene informazioni specifiche",
    "non contengono informazioni su",
    "i documenti forniti non contengono",
    "le fonti disponibili non contengono",
    "non contengono il testo specifico",
    "non contengono dettagli specifici",
    "nessuno dei documenti forniti contiene",
    "nessuno dei documenti contiene",
    "non contiene tale articolo",
    "non contengono tale articolo",
    "nessun documento fornito contiene",
    "i documenti disponibili non contengono",
    "non è possibile fornire dettagli specifici",
    "non è possibile fornire informazioni specifiche",
    "non è specificamente menzionata nei documenti",
    "non è specificamente trattata nei documenti",
    "non è menzionata nei documenti forniti",
    "non sono specificamente menzionati nei documenti",
    "non è esplicitamente menzionata nei documenti",
    "non viene menzionata nei documenti",
    "posso aiutarla con domande correlate",
    "posso aiutarti con domande correlate",
]


# Structural pattern catching gap-acknowledgment phrasings the model invents
# that aren't on the literal _GAP_PHRASES list (e.g. "non ho trovato
# informazioni sufficienti", a real production case the literal list missed).
# Matches the SHAPE of an "I found nothing" sentence rather than exact
# wording: a negation + a finding/information verb + an information noun,
# optionally followed by a qualifier like "sufficienti", "specifiche", or
# a reference to "documenti"/"database"/"fonti".
_GAP_STRUCTURAL_PATTERN = re.compile(
    r'\bnon\s+(?:ho\s+trovato|trovo|ho|sono\s+riuscito\s+a\s+trovare|'
    r'sono\s+stat[oi]\s+in\s+grado\s+di\s+trovare)\s+'
    r'(?:informazioni|dati|documentazione|dettagli)\b'
    r'.{0,40}'
    r'(?:sufficient\w*|specific\w*|necessari\w*|nei?\s+documenti|nel\s+database|'
    r'nella\s+base|nelle?\s+fonti)?',
    re.IGNORECASE,
)


def _is_primary_gap_response(answer: str) -> bool:
    """True only when the answer is primarily a gap acknowledgment.

    Checks both a structural regex pattern (catches phrasings the model
    invents that don't match the literal phrase list — e.g. "non ho trovato
    informazioni sufficienti") and the literal _GAP_PHRASES list, then
    verifies the earliest match falls within the first min(150, max(100,
    15%)) of the answer. Trailing disclaimers after substantive content
    never fire; only opening gap sentences do.
    """
    answer_lower = answer.lower()
    earliest_idx = len(answer)

    _struct_match = _GAP_STRUCTURAL_PATTERN.search(answer_lower)
    if _struct_match:
        earliest_idx = min(earliest_idx, _struct_match.start())

    for phrase in _GAP_PHRASES:
        idx = answer_lower.find(phrase)
        if idx != -1:
            earliest_idx = min(earliest_idx, idx)

    if earliest_idx == len(answer):
        return False
    threshold = min(80, max(50, int(len(answer) * 0.08)))
    return earliest_idx < threshold
