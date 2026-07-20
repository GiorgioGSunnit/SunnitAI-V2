"""LangGraph node functions for the RAG agent pipeline."""

import itertools
import json
import logging
import os
import re
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Set

from langchain_core.messages import HumanMessage, SystemMessage
from neo4j.exceptions import Neo4jError
from neo4j.graph import Node as Neo4jNode

from ..preprocessing.schema.schema import entities as schema_entities
from ..preprocessing.schema.schema import relations as schema_relations
from .ai_chat import _call_chat, structured_entities_model, embedding_model, _embed_query_with_prefix
from .verbose_logger import vlog, Timer
from .cypher_logger import log_cypher_event, log_cypher_multiline
from .language import SessionLang, language_display_name, normalize_lang
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
    FULLTEXT_INDEXES,
)
from .lookups import (
    LABEL_VECTOR_HINTS,
    VECTOR_INDEX_SETTINGS,
    ParsedLegalAct,
    bm25_lookup,
    btree_lookup,
    fulltext_lookup,
    legal_act_lookup,
    vector_lookup,
)
from .models import DocumentEntities
from .utils import (
    _ALL_SCHEMA_LABELS,
    _build_schema_text,
    _clean_cypher,
    _enforce_relation_directions,
    _parse_json_list,
    _strict_filter_relations,
    canonical_name,
)

logger = logging.getLogger(__name__)

_INTENT_CLASSIFIER_TIMEOUT = 8  # seconds


def _visibility_filter(alias: str = "d") -> str:
    """Returns a Cypher WHERE clause fragment for document visibility.
    Documents without a visibility property are treated as public.
    """
    return (
        f"(coalesce({alias}.visibility, 'public') = 'public' "
        f"OR {alias}.owner_id = $user_id "
        f"OR {alias}.tenant_id = $tenant_id)"
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


def _node_to_dict(v: Any) -> Any:
    return dict(v) if hasattr(v, "items") and not isinstance(v, dict) else v


# Max nodes to pass into Cypher generation prompts (keeps tokens under control)
_MAX_ENTRY_NODES_FOR_PROMPT = 8
_MAX_CONTEXT_NODES_FOR_PROMPT = 6


def _session_lang(state: Dict[str, Any]) -> SessionLang:
    return normalize_lang(state.get("session_language"))


# ---------------------------------------------------------------------------
# Document reference detection
# ---------------------------------------------------------------------------

_DOC_REF_NAMED_PATTERN = re.compile(
    r'(?:'
    r'NOA\.\d{2}\.\d{4}\.\d+'   # NOA.XX.YYYY.XXXXXXX
    r'|BOE-A-\d{4}-\d+'          # BOE-A-YYYY-XXXXX
    r'|T2LE_[A-Z0-9]+'           # T2LE_XXXXX
    r')'
)
_DOC_REF_FILE_PATTERN = re.compile(r'\S+\.(?:pdf|docx|md)\b', re.IGNORECASE)
# Uppercase token with at least one digit, separated by dots/dashes/underscores, min 3 chars total
_DOC_REF_CODE_PATTERN = re.compile(r'\b[A-Z][A-Z0-9]*[-_.](?:[A-Z0-9]+[-_.])*[A-Z0-9]*\d[A-Z0-9]*\b')


def _extract_document_references(query: str) -> List[str]:
    """Detect document reference patterns in the query using regex.

    Matches named formats (NOA, BOE, T2LE), file extensions, and generic
    uppercase codes with digits separated by dots/dashes/underscores.
    """
    seen: set = set()
    refs: List[str] = []
    for pat in (_DOC_REF_NAMED_PATTERN, _DOC_REF_FILE_PATTERN, _DOC_REF_CODE_PATTERN):
        for m in pat.findall(query):
            if m not in seen:
                seen.add(m)
                refs.append(m)
    return refs


def _collect_labels(nodes: List[Dict[str, Any]]) -> Set[str]:
    """Extract all unique labels from a list of node dicts."""
    labels: Set[str] = set()
    for n in nodes:
        for lbl in n.get("labels", []):
            labels.add(lbl)
    return labels


# ---------------------------------------------------------------------------
# Article number detection
# ---------------------------------------------------------------------------

_ARTICLE_PATTERNS = [
    re.compile(r'\bart\.?\s*(\d+)\s*(?:bis|ter|quater|quinquies)?\b', re.IGNORECASE),
    re.compile(r'\barticolo\s+(\d+)\s*(?:bis|ter|quater|quinquies)?\b', re.IGNORECASE),
    re.compile(r'\bartículo\s+(\d+)\b', re.IGNORECASE),
    re.compile(r'\b§\s*(\d+)\b'),
]
_LAW_PATTERNS = [
    re.compile(r'codice\s+civile', re.IGNORECASE),
    re.compile(r'codice\s+penale', re.IGNORECASE),
    re.compile(r'codice\s+di\s+procedura\s+(?:civile|penale)', re.IGNORECASE),
    re.compile(r'codice\s+dei\s+contratti(?:\s+pubblici)?', re.IGNORECASE),
    re.compile(r'codice\s+degli\s+appalti', re.IGNORECASE),
    re.compile(r'contratti\s+pubblici', re.IGNORECASE),
    re.compile(r'codice\s+appalti', re.IGNORECASE),
    re.compile(r'codice\s+del\s+processo\s+\w+', re.IGNORECASE),
    re.compile(r'd\.?\s*lgs\.?\s*[\d/]+', re.IGNORECASE),
    re.compile(r'legge\s+[\d/]+', re.IGNORECASE),
    re.compile(r'costituzione', re.IGNORECASE),
]

# Cache of document names fetched from Neo4j — keyed by (user_id, tenant_id)
# Each entry is (docs_list, timestamp). TTL: 5 minutes.
_DOC_NAMES_CACHE: dict = {}
_DOC_NAMES_CACHE_LOCK = __import__('threading').Lock()
_DOC_NAMES_CACHE_TTL = 300  # seconds

_STOPWORDS = {
    'il', 'lo', 'la', 'i', 'gli', 'le', 'un', 'uno', 'una', 'di', 'del', 'della',
    'dei', 'degli', 'delle', 'a', 'ad', 'al', 'alla', 'ai', 'agli', 'alle', 'da',
    'dal', 'dalla', 'dai', 'dagli', 'dalle', 'in', 'nel', 'nella', 'nei', 'negli',
    'nelle', 'su', 'sul', 'sulla', 'sui', 'sugli', 'sulle', 'con', 'per', 'tra',
    'fra', 'e', 'o', 'ma', 'che', 'chi', 'cui', 'non', 'si', 'mi', 'ti', 'ci',
    'vi', 'lo', 'li', 'ne', 'the', 'of', 'and', 'or', 'in', 'to', 'a', 'is',
    'cosa', 'dice', 'come', 'quando', 'dove', 'perché', 'quale', 'quali', 'quanto',
    'articolo', 'art', 'comma', 'decreto', 'legge', 'n', 'del', 'pdf',
}


def _fetch_doc_names(driver, database: str, user_id: str = "", tenant_id: str = "") -> list[dict]:
    """Fetch all document id+name pairs from Neo4j, with TTL-based caching."""
    import time
    global _DOC_NAMES_CACHE
    cache_key = (user_id, tenant_id)
    with _DOC_NAMES_CACHE_LOCK:
        cached = _DOC_NAMES_CACHE.get(cache_key)
        if cached and (time.time() - cached[1]) < _DOC_NAMES_CACHE_TTL:
            return cached[0]
        try:
            with driver.session(database=database) as session:
                result = session.run(
                    "MATCH (d:Document)-[:CONTAINS]->(:Section) "
                    "WHERE coalesce(d.visibility, 'public') = 'public' "
                    "OR d.owner_id = $user_id OR d.tenant_id = $tenant_id "
                    "RETURN DISTINCT d.id AS id, d.name AS name, "
                    "coalesce(d.aliases, []) AS aliases",
                    user_id=user_id,
                    tenant_id=tenant_id,
                )
                docs = [
                    {
                        "id": r["id"],
                        "name": r["name"],
                        "aliases": r["aliases"] or [],
                    }
                    for r in result if r["name"] and r["id"]
                ]
            _DOC_NAMES_CACHE[cache_key] = (docs, time.time())
            logger.info("_fetch_doc_names: cached %d document names", len(docs))
            return docs
        except Exception as exc:
            logger.warning("_fetch_doc_names failed: %s", exc)
            return []


# Legacy function — replaced by _classify_query_intent in decompose_query
# Kept for reference only, no longer called by context_retrieval
def _dynamic_law_hint(query: str, driver, database: str) -> str:
    """
    Match query tokens against all document names in Neo4j.
    Returns the document id of the best-matching document,
    or empty string if no meaningful match is found.
    Uses regex patterns first for speed, then falls back to token matching.
    """
    # Tokenize query — lowercase, strip punctuation, remove stopwords
    tokens = {
        re.sub(r'[^\w]', '', t).lower()
        for t in query.split()
    }
    tokens = {t for t in tokens if t and t not in _STOPWORDS and len(t) > 2}
    if not tokens:
        return ""

    docs = _fetch_doc_names(driver, database)
    best_id = ""
    best_score = 0
    best_name_token_count = 999

    for doc in docs:
        name = doc.get("name", "")
        doc_id = doc.get("id", "")
        # Build searchable text from name + aliases
        alias_text = " ".join(doc.get("aliases", []))
        searchable = f"{name} {alias_text}"
        name_tokens = {
            re.sub(r'[^\w]', '', t).lower()
            for t in searchable.split()
        }
        name_tokens = {t for t in name_tokens if t and t not in _STOPWORDS and len(t) > 2}
        score = len(tokens & name_tokens)
        if score > best_score or (score == best_score and len(name_tokens) < best_name_token_count):
            best_score = score
            best_id = doc_id
            best_name = name
            best_name_token_count = len(name_tokens)

    # Only scope BM25 to a single document for specific article lookup queries.
    # General legal questions (no article reference) should search the full corpus.
    # Require score >= 3 AND an article reference in the query to avoid false locks.
    _has_article_ref = bool(re.search(
        r'\bart\.?\s*\d+|articolo\s+\d+|comma\s+\d+|art\s+\d+',
        query, re.IGNORECASE
    ))
    # With article reference: score >= 2 is enough to scope BM25 to that document
    if _has_article_ref and best_score >= 2:
        logger.info(
            "_dynamic_law_hint: article lookup — matched %r id=%r (score=%d) for query %r",
            best_name, best_id, best_score, query[:60],
        )
        return best_id
    # Without article reference: require score >= 4 to avoid locking BM25
    # to a document just because its name appears in a general query
    if not _has_article_ref and best_score >= 4:
        logger.info(
            "_dynamic_law_hint: strong match — matched %r id=%r (score=%d) for query %r",
            best_name, best_id, best_score, query[:60],
        )
        return best_id

    return ""


def _classify_query_intent(
    query: str,
    driver,
    database: str,
    user_id: str = "",
    tenant_id: str = "",
) -> dict:
    """
    LLM-based query intent classification with grounded document resolution.
    Replaces token-intersection _dynamic_law_hint for document scoping.

    Returns:
        intent: concept_in_doc | concept_across_docs | doc_comparison | regular
        doc_a_id: resolved document id or ""
        doc_b_id: resolved document id or ""
        entity_a: first concept or ""
        entity_b: second concept or ""
    """
    _default = {"intent": "regular", "doc_a_id": "", "doc_b_id": "",
                "entity_a": "", "entity_b": ""}
    try:
        docs = _fetch_doc_names(driver, database,
                                user_id=user_id, tenant_id=tenant_id)
        if not docs:
            return _default

        doc_list = "\n".join(
            f"- {d['name']}" for d in docs
            if d.get("name") and not re.search(
                r'\.(pdf|docx|xlsx|txt)$', d['name'], re.IGNORECASE
            )
        )

        system_prompt = (
            "You are a query classifier for an Italian legal document system.\n"
            "Given a user query and a list of available documents, classify the "
            "intent and identify any document references.\n\n"
            f"Available documents:\n{doc_list}\n\n"
            "Respond ONLY with valid JSON, no markdown, no explanation:\n"
            '{"intent": "...", "doc_a": "...", "doc_b": "...", '
            '"entity_a": "...", "entity_b": "..."}\n\n'
            "Valid intent values (ONLY these four, no others):\n"
            "  concept_in_doc | concept_across_docs | doc_comparison | regular\n\n"
            "Intent rules:\n"
            "- concept_in_doc: user asks about one OR two concepts within "
            "a SINGLE named document. The document must be EXPLICITLY named "
            "in the query. Use this ONLY when the query is clearly scoped to "
            "that one document.\n"
            "  Example: 'differenza tra dolo e colpa nel codice penale'\n"
            "- concept_across_docs: user EXPLICITLY asks to COMPARE the same "
            "concept across TWO DIFFERENT documents — requires BOTH doc_a AND "
            "doc_b. The user must be asking for a comparison or difference, "
            "NOT just mentioning two related documents in a substantive legal "
            "question.\n"
            "  CORRECT example: 'differenza tra responsabilità nel codice "
            "civile e nel codice penale' — explicitly asks for a difference.\n"
            "  WRONG example: 'sanzioni previste dal GDPR e dal codice della "
            "privacy' — mentions two documents but asks a substantive question "
            "about sanctions, NOT a comparison. Use concept_in_doc or regular.\n"
            "- doc_comparison: user wants to compare two documents broadly "
            "— requires BOTH doc_a AND doc_b. Must explicitly request a "
            "comparison, not merely mention two documents.\n"
            "- regular: use this for ALL other cases, including:\n"
            "  * general legal questions with no specific document named\n"
            "  * cross-domain queries (e.g. criminal + civil/regulatory topics together)\n"
            "  * queries that mention a legal topic that appears in a document name "
            "but are asking a general question (e.g. 'conseguenze penali per chi "
            "viola la privacy' — do NOT lock to GDPR, use regular)\n"
            "  * any query where you are uncertain\n\n"
            "CRITICAL RULES:\n"
            "1. concept_across_docs and doc_comparison require TWO different documents "
            "explicitly named in the query AND an explicit request for comparison or "
            "difference — merely mentioning two documents in a substantive legal "
            "question (e.g. asking about sanctions, obligations, or rights that "
            "span two related laws) is NOT a comparison — use concept_in_doc "
            "with the most relevant document, or regular.\n"
            "2. When both a national Italian law (codice, testo unico, decreto "
            "legislativo) and an EU regulation (regolamento UE, GDPR, direttiva) "
            "are mentioned, set doc_a to the national law — it is almost always "
            "the primary source for Italian practitioners. Set doc_b to the EU "
            "regulation only if the question specifically asks about EU-level rules.\n"
            "2. concept_in_doc requires the document name to appear EXPLICITLY in the "
            "query — do NOT infer it from topic keywords alone.\n"
            "3. Cross-domain queries (combining criminal law with civil/regulatory topics, "
            "or asking about consequences across multiple legal areas) must use 'regular' "
            "so all relevant documents are searched.\n"
            "4. When in doubt, always use 'regular'. A wrong 'concept_in_doc' gives "
            "incomplete answers; 'regular' always gives complete answers.\n"
            "5. Leave doc_a and doc_b as empty strings for 'regular' intent.\n"
            "6. For concept_in_doc with two documents, set doc_a to the document "
            "that most directly answers the question (usually the national law), "
            "and doc_b to the secondary reference."
        )

        from openai import OpenAI
        client = OpenAI(
            base_url=os.getenv("LLM_BASE_URL", ""),
            api_key=os.getenv("LLM_API_KEY", ""),
        )
        response = client.chat.completions.create(
            model=os.getenv("LLM_MODEL", ""),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            temperature=0,
            max_tokens=150,
            timeout=_INTENT_CLASSIFIER_TIMEOUT,
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:].strip()
        result = json.loads(raw)

        name_to_id = {d["name"]: d["id"] for d in docs}
        doc_a_id = name_to_id.get(result.get("doc_a", ""), "")
        doc_b_id = name_to_id.get(result.get("doc_b", ""), "")

        intent = result.get("intent", "regular")
        # Normalize common LLM shorthand aliases before validation
        if intent == "comparison":
            intent = "doc_comparison"
        if intent not in ("concept_in_doc", "concept_across_docs",
                          "doc_comparison", "regular"):
            intent = "regular"

        logger.info(
            "_classify_query_intent: intent=%r doc_a=%r doc_b=%r "
            "entity_a=%r entity_b=%r",
            intent, result.get("doc_a"), result.get("doc_b"),
            result.get("entity_a"), result.get("entity_b"),
        )

        return {
            "intent": intent,
            "doc_a_id": doc_a_id,
            "doc_b_id": doc_b_id,
            "entity_a": result.get("entity_a", ""),
            "entity_b": result.get("entity_b", ""),
        }

    except Exception as e:
        logger.warning(
            "_classify_query_intent failed: %s — falling back to regular", e
        )
        return _default


def _extract_article_references(query: str) -> List[tuple]:
    """Return list of (article_number_str, full_match_str) for each unique article found."""
    results = []
    seen: Set[str] = set()
    for pat in _ARTICLE_PATTERNS:
        for m in pat.finditer(query):
            number = m.group(1)
            if number not in seen:
                seen.add(number)
                results.append((number, m.group(0).strip()))
    return results


def _extract_law_hint(query: str) -> str:
    """Return the first law/code reference found in query, lowercased, or empty string."""
    for pat in _LAW_PATTERNS:
        m = pat.search(query)
        if m:
            return m.group(0).lower()
    return ""


def _select_schema_for_query(
    question: str,
    keywords: List[str],
    anchor_labels: Set[str],
) -> tuple:
    """Steps 1 and 2 of the three-step Cypher generation pipeline.

    Step 1 — Label selection: LLM picks relevant node labels from the full
             25-label schema list.
    Step 2 — Relationship selection: Python strictly pre-filters relations to
             only those whose both endpoints are in the Step 1 result, then
             LLM picks which of those are actually needed for this query.

    Returns (selected_labels: list[str], selected_rel_types: list[str]).

    Fallback policy (critical for weak/small models):
      - Any unexpected exception            → full schema labels + all candidate types
      - Step 1 empty or malformed JSON      → all 25 schema labels
      - Step 2 empty or malformed JSON      → all types from the pre-filtered candidates
      - No candidate rels after pre-filter  → (labels, [])
    """
    try:
        all_labels_str = ", ".join(_ALL_SCHEMA_LABELS)
        anchor_str = ", ".join(sorted(anchor_labels)) if anchor_labels else "(none)"
        keywords_str = ", ".join(keywords) if keywords else "(none)"

        # --- Step 1: Label selection ---
        step1_raw = _call_chat(
            [
                SystemMessage(
                    content=(
                        "You are a schema filter for a legal knowledge graph. "
                        "Given a question and keywords, select only the node labels "
                        "relevant to answering it. Return a JSON array of label strings. "
                        "No explanation."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Schema labels:\n{all_labels_str}\n\n"
                        f"Already-matched labels (always include these): {anchor_str}\n"
                        f"Question: {question}\n"
                        f"Keywords: {keywords_str}\n\n"
                        'Return a JSON array only. Example: ["LegalAct", "Person"]'
                    )
                ),
            ]
        )
        selected_labels = _parse_json_list(step1_raw or "")
        if not selected_labels:
            logger.warning(
                "_select_schema_for_query: Step 1 empty/invalid — falling back to all schema labels"
            )
            selected_labels = list(_ALL_SCHEMA_LABELS)

        # --- Step 2: Relationship selection (Python pre-filter first) ---
        candidate_rels = _strict_filter_relations(set(selected_labels))
        if not candidate_rels:
            return selected_labels, []

        candidate_lines = "\n".join(
            f"- {r['from']} -[:{r['type']}]-> {r['to']}" for r in candidate_rels
        )
        step2_raw = _call_chat(
            [
                SystemMessage(
                    content=(
                        "You are a schema filter for a legal knowledge graph. "
                        "Given selected node labels and candidate relationships, pick only "
                        "the relationship types needed to answer the question. "
                        "Return a JSON array of type strings. No explanation."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Selected labels: {selected_labels}\n\n"
                        f"Candidate relationships:\n{candidate_lines}\n\n"
                        f"Question: {question}\n\n"
                        'Return a JSON array of type strings only. Example: ["ISSUED_BY", "APPOINTS"]'
                    )
                ),
            ]
        )
        selected_rel_types = _parse_json_list(step2_raw or "")
        if not selected_rel_types:
            logger.warning(
                "_select_schema_for_query: Step 2 empty/invalid — falling back to all candidate types"
            )
            selected_rel_types = list({r["type"] for r in candidate_rels})

        return selected_labels, selected_rel_types

    except Exception:
        logger.exception(
            "_select_schema_for_query: unexpected error — returning full schema fallback"
        )
        all_rels = _strict_filter_relations(set(_ALL_SCHEMA_LABELS))
        return list(_ALL_SCHEMA_LABELS), list({r["type"] for r in all_rels})


_ENTITY_LABELS_TEXT = ", ".join(DocumentEntities.allowed_labels())
SCHEMA_TEXT = _build_schema_text()  # full schema — used only for entity extraction
RELATION_HINTS = "\n".join(
    f"- {item['from']} -[:{item['type']}]-> {item['to']}" for item in schema_relations
)


# ---------------------------------------------------------------------------
# Node A: Query decomposition
# ---------------------------------------------------------------------------

_OFF_TOPIC_REDIRECTS = {
    "it": (
        "💡 Sono specializzato in consulenza legale e non sono in grado di rispondere a questa domanda. "
        "Posso aiutarti con questioni di diritto civile, penale, amministrativo o con l'analisi di documenti legali. "
        "C'è qualcosa di legale su cui posso assisterti?"
    ),
    "es": (
        "💡 Estoy especializado en consultoría legal y no puedo responder a esta pregunta. "
        "Puedo ayudarte con cuestiones de derecho civil, penal, administrativo o con el análisis de documentos legales. "
        "¿Hay algo legal en lo que pueda ayudarte?"
    ),
    "en": (
        "💡 I specialise in legal consultation and am unable to answer this question. "
        "I can help you with civil, criminal, administrative law or legal document analysis. "
        "Is there something legal I can assist you with?"
    ),
}


_COMPARISON_PATTERNS = re.compile(
    r'\b(confronta|confronto|paragona|paragon[ao]|'
    r'compare|comparison|differences?\s+between|'
    r'compara|comparaci[oó]n|diferencias?\s+entre)\b',
    re.IGNORECASE,
)


def _is_legal_query(query: str, lang: str) -> bool:
    """Return True if the query is legal/professional; False if off-topic. Fails safe (True)."""
    try:
        response = _call_chat(
            [
                SystemMessage(
                    content="You are a legal assistant classifier. Reply with only LEGAL or OFFTOPIC."
                ),
                HumanMessage(
                    content=(
                        "Is this a legal or professional question?\n"
                        "Question: {query}\n"
                        "Reply with only: LEGAL or OFFTOPIC"
                    ).format(query=query)
                ),
            ],
            max_tokens=5,
        )
        return "LEGAL" in (response or "").upper()
    except Exception:
        return True


def decompose_query(state: Dict[str, Any], driver=None, database: str = "neo4j") -> Dict[str, Any]:
    state["turn_count"] = state.get("turn_count", 0) + 1
    query = state["query"]
    lang = _session_lang(state)

    if not _is_legal_query(query, lang):
        redirect = _OFF_TOPIC_REDIRECTS.get(lang, _OFF_TOPIC_REDIRECTS["en"])
        logger.info("Off-topic query detected, skipping pipeline: %s", query[:80])
        return {
            **state,
            "answer": redirect,
            "citations": [],
            "references": [],
            "status_messages": [],
            "off_topic": True,
        }

    is_comparison = bool(_COMPARISON_PATTERNS.search(query))
    logger.info(f"decompose_query: is_comparison={is_comparison} for query={query[:50]!r}")
    comparison_name_messages = None
    if is_comparison:
        comparison_name_messages = [
            SystemMessage(
                content=(
                    "Extract exactly two legal document names being compared. "
                    "Return only the two names separated by '|||'. No explanation."
                )
            ),
            HumanMessage(content=f"Query: {query}"),
        ]

    logger.info("Starting query decomposition", extra={"query": query})

    log_cypher_multiline(
        "a_query",
        "user question (verbatim — start of RAG pipeline)",
        query,
        delimiter_label="USER_QUESTION",
    )

    # Step 1 (generalize) and Step 2 (entity extraction) are independent —
    # run them in parallel to cut decomposition latency roughly in half.
    generalization_messages = [
        SystemMessage(
            content=f"You are a legal search assistant. Respond in {language_display_name(lang)}."
        ),
        HumanMessage(
            content=(
                "Original question: {query}\n"
                "The question may be phrased colloquially or informally. Translate it into a concise "
                "formal legal search phrase (max 8 words) capturing the main legal topic, regardless "
                "of how casually it was expressed."
            ).format(query=query)
        ),
    ]

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
                "You are an expert graph extractor for legal documents. "
                "Identify nodes and relationships from the user's query based on the provided graph schema. "
                f"Respond in {language_display_name(lang)} where any text fields are needed."
            )
        ),
        HumanMessage(content=entity_extraction_prompt),
    ]

    query_variants_messages = [
        SystemMessage(
            content=f"You are a legal search assistant. Respond in {language_display_name(lang)}."
        ),
        HumanMessage(
            content=(
                "Generate 3 alternative ways to phrase this legal question for broader search coverage. "
                "Include both formal legal terminology and more colloquial phrasings a non-lawyer might use. "
                "Return as comma-separated phrases, no numbering, no explanation.\n\n"
                f"Question: {query}"
            )
        ),
    ]

    with ThreadPoolExecutor(max_workers=4) as pool:
        future_generalize = pool.submit(_call_chat, generalization_messages, 60)
        future_entities = pool.submit(structured_entities_model.invoke, entity_extraction_messages)
        future_variants = pool.submit(_call_chat, query_variants_messages, 100)
        future_comparison = (
            pool.submit(_call_chat, comparison_name_messages, 80)
            if comparison_name_messages else None
        )

        generalized = future_generalize.result()
        entities_payload = future_entities.result()
        variants_raw = future_variants.result()
        try:
            comparison_names_raw = future_comparison.result() if future_comparison else ""
        except Exception:
            comparison_names_raw = ""

    comparison_doc_ids: List[str] = []
    if is_comparison and comparison_names_raw:
        comparison_doc_ids = [
            p.strip() for p in comparison_names_raw.split("|||") if p.strip()
        ][:2]

    query_variants = [v.strip() for v in (variants_raw or "").split(",") if v.strip()][:5]

    logger.info(f"Generalized query: '{generalized}'")
    log_cypher_event(
        "a_generalized",
        "generalized topic phrase (used for context / vector retrieval)",
        detail=generalized,
    )
    log_cypher_event(
        "a_variants",
        "query variant phrasings (used for multi-shot vector retrieval)",
        detail=query_variants,
    )

    # Pre-processing: detect document reference patterns before LLM keyword extraction
    doc_refs = _extract_document_references(query)

    # Step 1b: Keywords (up to 5) — depends on generalized, so runs after
    ref_instruction = (
        " These document references were found in the query and must be preserved "
        f"exactly as-is in the keywords: {', '.join(doc_refs)}. Do not translate or interpret them."
        if doc_refs else ""
    )
    kw_raw = _call_chat(
        [
            SystemMessage(
                content=(
                    f"{legal_consultant_system_prefix(lang)} "
                    "Extract 5 specific legal terms or phrases from this question that would most likely appear verbatim in relevant legal documents. "
                    "Focus on specific concepts, not generic categories. "
                    f"No explanation, comma-separated.{ref_instruction}"
                )
            ),
            HumanMessage(
                content=f"Question:\n{query}\n\nGeneralized topic:\n{generalized}\n\nKeywords:"
            ),
        ],
        max_tokens=60,
    )
    llm_keywords = [k.split('\n')[0].strip() for k in (kw_raw or "").split(",") if k.split('\n')[0].strip()][:5]
    _sentence_starters = re.compile(
        r"^(noto|prevede|stabilisce|dispone|indica|riporta)\b", re.IGNORECASE
    )
    llm_keywords = [
        k for k in llm_keywords
        if len(k) <= 50
        and '"' not in k and "'" not in k
        and not _sentence_starters.match(k)
    ]
    # Prepend detected doc refs so they're always present regardless of LLM output
    if doc_refs:
        existing = set(llm_keywords)
        retrieval_keywords = [r for r in doc_refs if r not in existing] + llm_keywords
    else:
        retrieval_keywords = llm_keywords

    # Keyword-derived article references: the keyword-extraction LLM often
    # correctly names a specific article (e.g. "art. 575 c.p.") even when
    # the user's own query text has no number in it (e.g. "omicidio doloso").
    # _extract_article_references only scans the raw query, missing these —
    # so we also scan the extracted keywords themselves and surface any
    # article numbers found, for article_router to use as a fallback.
    keyword_article_refs: List[tuple] = []
    _seen_kw_articles: Set[str] = set()
    for kw in retrieval_keywords:
        for pat in _ARTICLE_PATTERNS:
            for m in pat.finditer(kw):
                number = m.group(1)
                if number not in _seen_kw_articles:
                    _seen_kw_articles.add(number)
                    keyword_article_refs.append((number, m.group(0).strip()))

    log_cypher_event(
        "a_keywords",
        "extracted keywords",
        detail=retrieval_keywords,
    )
    logger.info("DEBUG retrieval_keywords=%r keyword_article_refs=%r", retrieval_keywords, keyword_article_refs)

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

    user_id = state.get("user_id") or ""
    tenant_id = state.get("tenant_id") or ""
    intent_result = _classify_query_intent(
        query, driver, database,
        user_id=user_id, tenant_id=tenant_id,
    )

    # --- Deterministic pre-classifier rules ---
    # Applied BEFORE the LLM classifier result, for patterns where the
    # LLM is known to be non-deterministic. These rules always win.
    _privacy_gdpr_pattern = bool(re.search(
        r'\b(gdpr|regolamento\s+(?:generale\s+)?(?:ue|europeo|sulla\s+protezione))\b',
        query, re.IGNORECASE
    ) and re.search(
        r'\b(privacy|dati\s+personali|codice\s+della\s+privacy)\b',
        query, re.IGNORECASE
    ))
    if _privacy_gdpr_pattern and not intent_result["doc_a_id"]:
        # Force doc_a = Codice della Privacy 2026 (national law takes priority)
        # Force doc_b = Regolamento generale sulla protezione dei dati 2019
        _privacy_id = next(
            (d["id"] for d in _fetch_doc_names(driver, database)
             if d["name"] == "Codice della Privacy 2026"), ""
        )
        _gdpr_id = next(
            (d["id"] for d in _fetch_doc_names(driver, database)
             if d["name"] == "Regolamento generale sulla protezione dei dati 2019"), ""
        )
        if _privacy_id:
            intent_result["doc_a_id"] = _privacy_id
            intent_result["law_hint_doc_id"] = _privacy_id
        if _gdpr_id:
            intent_result["doc_b_id"] = _gdpr_id
        if _privacy_id and _gdpr_id:
            intent_result["intent"] = "concept_across_docs"
            logger.info(
                "[decompose_query] deterministic rule: privacy+GDPR query "
                "forced to concept_across_docs doc_a=Privacy doc_b=GDPR"
            )
    # --- End deterministic pre-classifier rules ---

    classifier_intent = intent_result["intent"]
    if classifier_intent == "doc_comparison":
        # Explicit broad document comparison — route to comparison pipeline
        is_comparison = True
        clf_doc_ids = [
            d for d in [intent_result["doc_a_id"], intent_result["doc_b_id"]]
            if d
        ]
        if len(clf_doc_ids) >= 2:
            comparison_doc_ids = clf_doc_ids
    elif classifier_intent == "concept_across_docs":
        # Conceptual query spanning two documents — treat like concept_in_doc:
        # search both documents via normal RAG rather than comparison pipeline.
        is_comparison = False

    return {
        **state,
        "generalized_query": generalized,
        "retrieval_keywords": retrieval_keywords,
        "keyword_article_refs": keyword_article_refs,
        "document_references": doc_refs,
        "entities": processed_entities,
        "extracted_relationships": final_relationships,
        "query_variants": query_variants,
        "is_comparison": is_comparison,
        "comparison_doc_ids": comparison_doc_ids,
        "query_intent": intent_result["intent"],
        "law_hint_doc_id": intent_result["doc_a_id"],
        "law_hint_doc_id_b": intent_result["doc_b_id"],
        "intent_entity_a": intent_result["entity_a"],
        "intent_entity_b": intent_result["entity_b"],
    }


# ---------------------------------------------------------------------------
# Node A1: Article number router (runs before vector search)
# ---------------------------------------------------------------------------

def article_router(state: Dict[str, Any], driver, database: str) -> Dict[str, Any]:
    query = state.get("query", "")
    user_id = state.get("user_id") or ""
    tenant_id = state.get("tenant_id") or ""
    article_refs = _extract_article_references(query)

    # Fallback: the raw query may name a concept ("omicidio doloso") with no
    # number, while the keyword-extraction LLM (run earlier in decompose_query)
    # correctly identified the specific article it maps to (e.g. "art. 575
    # c.p."). Use that as a second source of article references before giving
    # up on the exact-match path entirely.
    used_keyword_fallback = False
    if not article_refs:
        kw_refs = state.get("keyword_article_refs") or []
        if kw_refs:
            article_refs = kw_refs
            used_keyword_fallback = True

    # Prefer the document already identified by _classify_query_intent
    # (set in decompose_query). Only fall back to _dynamic_law_hint when
    # the classifier didn't identify a document — avoids wrong-codice matches
    # e.g. "articolo 90 del codice penale" matching "Codice di procedura penale"
    law_hint = state.get("law_hint_doc_id") or _dynamic_law_hint(query, driver, database)

    # Dedicated, single-purpose article lookup — deterministic fallback when
    # neither the raw query nor the shared keyword-extraction step surfaced
    # a number. This is a focused LLM call with ONE job (name the article),
    # not competing with 4 other extraction slots, so it is far less likely
    # to skip the number than the general keyword-extraction prompt.
    #
    # SCOPE GUARD: only attempt this for queries whose phrasing signals a
    # specific crime/penalty/single-provision question. Broad conceptual
    # questions ("cosa prevede", "requisiti di", "differenze tra") often
    # span multiple articles and must NOT be force-narrowed to one — doing
    # so produced a wrong, overconfident answer in testing (e.g. contract
    # validity question incorrectly narrowed to a single unrelated article).
    _single_provision_signal = bool(re.search(
        r'\b(pena\s+per|pene\s+per|reato\s+di|conseguenze\s+del|conseguenze\s+penali|'
        r'sanzioni\s+per|punito\s+con|punizione\s+per|delitto\s+di|'
        r'responsabilit[àa]\s+civile|responsabilit[àa]\s+penale|'
        r'differenz[ae]\s+tra|cosa\s+(?:prevede|dice|stabilisce)\s+il\s+codice|'
        r'istituto\s+giuridico|disciplina\s+di|elementi\s+del\s+reato)\b',
        query, re.IGNORECASE
    ))
    if not article_refs and _single_provision_signal:
        try:
            # Single multi-article JSON call — ask for up to 3 articles directly.
            # Corpus validation ensures only articles that actually exist in the
            # database are used, preventing hallucinated article numbers from
            # corrupting retrieval.
            _multi_system = (
                "Sei un esperto di diritto penale e civile italiano. "
                "Data una domanda su reati o istituti giuridici, identifica "
                "gli articoli principali del Codice Penale o Civile (massimo 3). "
                "Rispondi SOLO con JSON array, nessun testo aggiuntivo:\n"
                '[{"num": "575", "codice": "penale"}]\n\n'
                "Esempi:\n"
                "- 'omicidio doloso' -> [{\"num\": \"575\", \"codice\": \"penale\"}]\n"
                "- 'truffa' -> [{\"num\": \"640\", \"codice\": \"penale\"}]\n"
                "- 'furto' -> [{\"num\": \"624\", \"codice\": \"penale\"}, {\"num\": \"625\", \"codice\": \"penale\"}]\n"
                "- 'rapina' -> [{\"num\": \"628\", \"codice\": \"penale\"}]\n"
                "- 'lesioni personali' -> [{\"num\": \"582\", \"codice\": \"penale\"}]\n"
                "- 'diffamazione' -> [{\"num\": \"595\", \"codice\": \"penale\"}]\n"
                "- 'responsabilità civile e penale' -> [{\"num\": \"185\", \"codice\": \"penale\"}, {\"num\": \"2043\", \"codice\": \"civile\"}]\n"
                "- 'atti osceni' -> [{\"num\": \"527\", \"codice\": \"penale\"}]\n"
                "- 'bancarotta fraudolenta' -> [{\"num\": \"216\", \"codice\": \"altro\"}]\n"
                "- 'violazione privacy' -> [{\"num\": \"167\", \"codice\": \"privacy\"}]\n"
                "- 'trattamento illecito dati personali' -> [{\"num\": \"167\", \"codice\": \"privacy\"}]\n"
                "- 'responsabilità medico errore professionale' -> [{\"num\": \"2043\", \"codice\": \"civile\"}, {\"num\": \"1218\", \"codice\": \"civile\"}]\n"
                "Se la domanda non riguarda articoli specifici: []\n"
                "Rispondi SOLO con il JSON array."
            )
            _multi_raw = _call_chat(
                [SystemMessage(content=_multi_system), HumanMessage(content=query)],
                max_tokens=80,
            ).strip()
            _multi_clean = _multi_raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            try:
                _multi_results = json.loads(_multi_clean) if _multi_clean.startswith("[") else []
            except Exception:
                _multi_results = []

            _codice_to_doc = {
                "penale": "Codice Penale 2026",
                "civile": "Codice Civile 2026",
                "privacy": "Codice della Privacy 2026",
                "lavoro": "Codice Civile 2026",
            }
            _validated_refs = []
            if _multi_results and isinstance(_multi_results, list):
                for entry in _multi_results[:3]:
                    _num = str(entry.get("num", "")).strip()
                    _codice = entry.get("codice", "").strip().lower()
                    if not re.match(r'^\d+(?:-(?:bis|ter|quater|quinquies))?$', _num):
                        continue
                    _doc_name = _codice_to_doc.get(_codice, "")
                    # Corpus validation: only use article if it actually
                    # exists in the database — prevents hallucinated numbers
                    # from corrupting retrieval entirely.
                    # Prefer scoping to the document already identified by
                    # _classify_query_intent (law_hint_doc_id) when no
                    # explicit codice mapping exists — avoids matching the
                    # same article number across unrelated documents.
                    _scope_doc = _doc_name
                    if not _scope_doc:
                        # Try law_hint_doc_id first (set by _classify_query_intent)
                        _hint_id = state.get("law_hint_doc_id") or ""
                        if _hint_id:
                            with driver.session(database=database) as _ns:
                                _name_row = _ns.run(
                                    "MATCH (d:Document {id: $id}) RETURN d.name AS name",
                                    id=_hint_id,
                                ).single()
                                if _name_row:
                                    _scope_doc = _name_row["name"]
                    # No further fallback — if _scope_doc is still empty here,
                    # corpus validation runs unscoped across all documents.
                    # The reranker handles relevance filtering on the results.
                    with driver.session(database=database) as _vs:
                        _exists = _vs.run(
                            "MATCH (d:Document)-[:CONTAINS]->(s:Section) "
                            "WHERE (s.name = $num OR s.name STARTS WITH $num + '.') "
                            "AND ($doc = '' OR d.name = $doc) "
                            "RETURN count(s) AS cnt",
                            num=_num, doc=_scope_doc,
                        ).single()["cnt"]
                    if _exists > 0 and not _doc_name:
                        _doc_name = _scope_doc
                    if _exists > 0:
                        _validated_refs.append((_num, f"art. {_num}", _doc_name))
                        logger.info(
                            "[article_router] validated article %r in %r (%d nodes)",
                            _num, _doc_name or "any", _exists,
                        )
                    else:
                        logger.warning(
                            "[article_router] rejected hallucinated article %r in %r",
                            _num, _doc_name,
                        )

            if _validated_refs:
                article_refs = [(_num, ref) for _num, ref, _ in _validated_refs]
                used_keyword_fallback = True
                state["_multi_article_doc_names"] = {
                    _num: _doc_name for _num, _, _doc_name in _validated_refs
                }
                logger.info(
                    "[article_router] lookup found %r for query %r",
                    _validated_refs, query[:60],
                )
                if not law_hint and _validated_refs[0][2]:
                    _first_doc = _validated_refs[0][2]
                    with driver.session(database=database) as _sess:
                        _doc_row = _sess.run(
                            "MATCH (d:Document {name: $name}) RETURN d.id AS id",
                            name=_first_doc,
                        ).single()
                        if _doc_row:
                            law_hint = _doc_row["id"]

        except Exception as exc:
            logger.warning("[article_router] article lookup failed: %s", exc)

    if not article_refs:
        vlog("article_router", {"article_refs_found": [], "law_hint": law_hint, "results_found": 0})
        return {
            "article_router_fired": False,
            "article_refs_found": [],
            "law_hint_doc_id": state.get("law_hint_doc_id") or law_hint,
        }

    all_refs = [ref for _, ref in article_refs]

    _multi_doc_names = state.get("_multi_article_doc_names") or {}
    all_data: List[Dict[str, Any]] = []
    for article_number, article_ref in article_refs:
        # Use per-article document scoping when multi-article lookup provided
        # specific code names (e.g. 624->penale, 2043->civile)
        _article_law_hint = law_hint
        if _multi_doc_names.get(article_number):
            _doc_name = _multi_doc_names[article_number]
            with driver.session(database=database) as _s:
                _row = _s.run(
                    "MATCH (d:Document {name: $name}) RETURN d.id AS id",
                    name=_doc_name,
                ).single()
                if _row:
                    _article_law_hint = _row["id"]
        try:
            with driver.session(database=database) as session:
                # Primary search: exact match or standard sub-node prefix
                records = session.run(
                    "MATCH (d:Document)-[:CONTAINS]->(s:Section)\n"
                    "WHERE (s.name = $article_number OR s.name STARTS WITH $article_number + '.')\n"
                    "AND ($law_hint = '' OR d.id = $law_hint)\n"
                    f"AND {_visibility_filter()}\n"
                    "RETURN d, s LIMIT 15",
                    article_ref=article_ref,
                    article_number=article_number,
                    law_hint=_article_law_hint,
                    user_id=user_id,
                    tenant_id=tenant_id,
                )
                primary_data = [record.data() for record in records]

                # Fallback: prefixed names (e.g. "raccolte_usi.9.0.0") —
                # only used when primary search returns nothing, to avoid
                # false matches on sub-nodes of other articles (e.g. "87.9")
                if not primary_data:
                    # Match prefixed section names like "raccolte_usi.9.0.0".
                    # Starts with lowercase letter (rules out "87.9.0").
                    # After the article number: either end of string or a dot
                    # followed by anything — prevents matching "foo.91.0" when
                    # searching for article 9. Verified in Neo4j regex engine.
                    _prefixed_pattern = f"^[a-z][a-z0-9_]*[.]{article_number}([.].*)?$"
                    records_fallback = session.run(
                        "MATCH (d:Document)-[:CONTAINS]->(s:Section)\n"
                        "WHERE s.name =~ $pattern\n"
                        "AND ($law_hint = '' OR d.id = $law_hint)\n"
                        f"AND {_visibility_filter()}\n"
                        "RETURN d, s LIMIT 15",
                        pattern=_prefixed_pattern,
                        article_ref=article_ref,
                        article_number=article_number,
                        law_hint=_article_law_hint,
                        user_id=user_id,
                        tenant_id=tenant_id,
                    )
                    records = records_fallback
                else:
                    # Wrap primary_data back into an iterable the downstream
                    # code can call .data() on — store result directly
                    pass
                if primary_data:
                    data = primary_data
                else:
                    data = [record.data() for record in records]
                data = [{k: _node_to_dict(v) for k, v in row.items()} for row in data]
                for row in data:
                    row["_source"] = "bm25"

                # Merge fragment nodes: same article name + same document → one row
                # This fixes ingestion splits where one article became N Section nodes
                merged: dict[tuple, dict] = {}
                for row in data:
                    d = row.get("d") or {}
                    s = row.get("s") or {}
                    key = (d.get("id", ""), s.get("name", ""))
                    if key not in merged:
                        merged[key] = {
                            "d": d,
                            "s": {**s},
                            "_source": "bm25",
                        }
                    else:
                        # Append fragment text with a newline separator
                        existing_text = merged[key]["s"].get("plain_text") or ""
                        new_text = s.get("plain_text") or ""
                        if new_text and new_text not in existing_text:
                            merged[key]["s"]["plain_text"] = existing_text + "\n" + new_text
                data = list(merged.values())
                logger.info(
                    "[article_router] merged fragments: %d raw rows → %d sections",
                    sum(1 for row in data), len(data),
                )

                # Prefer the single clean base node when one exists and is
                # the only node found (mirrors the base-detection heuristic
                # in post_process.py for title grounding).
                if len(data) > 1:
                    _exact = [
                        row for row in data
                        if (row.get("s") or {}).get("name") == article_number
                    ]
                    _zero_zero = [
                        row for row in data
                        if (row.get("s") or {}).get("name") == f"{article_number}.0.0"
                    ]
                    if _exact:
                        data = _exact
                    elif len(data) == 1 and _zero_zero:
                        data = _zero_zero

                # When multiple genuinely distinct fragments remain (e.g. art.
                # 640's base crime plus 10 real aggravating-circumstance
                # variants), they are not safe to merge or arbitrarily drop —
                # but dumping all of them is noisy when the user asked a
                # general question that only the base provision answers.
                # Rerank them against the actual query using the same
                # reranker trusted elsewhere in this pipeline, and keep only
                # the top-scoring fragments — this is relevance filtering,
                # not content loss, since the reranker score reflects how
                # well each fragment actually answers THIS question.
                if len(data) > 3:
                    _reranked = rerank_results(query, data, top_k=4)
                    if _reranked:
                        data = _reranked
                        logger.info(
                            "[article_router] reranked %d fragments for art. %s down to %d relevant",
                            len(data), article_number, len(_reranked),
                        )
        except Neo4jError as exc:
            logger.warning("[article_router] Cypher failed for ref=%r: %s", article_ref, exc)
            continue

        vlog(
            "article_router",
            {"article_ref": article_ref, "law_hint": law_hint, "results_found": len(data)},
        )

        if data:
            logger.info(
                "[article_router] matched via %s: article_ref=%r, %d rows",
                "keyword-derived reference" if used_keyword_fallback else "query text",
                article_ref, len(data),
            )
            # Rerank per-article before accumulating — prevents all fragments
            # from multiple articles surviving together when only one article
            # is actually relevant to the question (e.g. furto returning both
            # 624 and 625 fragments when only 624 answers the general query).
            if len(data) > 2:
                _per_article_reranked = rerank_results(query, data, top_k=2)
                if _per_article_reranked:
                    data = _per_article_reranked
            all_data.extend(data)

    if all_data:
        enriched = _enrich_with_source_metadata(all_data)
        logger.info(
            "[article_router] returning %d total rows across %d article(s)",
            len(all_data), len(article_refs),
        )
        return {
            "article_router_fired": True,
            "article_refs_found": all_refs,
            "raw_result": all_data,
            "references": enriched,
            "execution_error": None,
            "neo4j_executed": True,
            "cypher_attempt": "article_router",
            "bm25_from_article_lookup": True,
            "law_hint_doc_id": state.get("law_hint_doc_id") or law_hint,
        }

    vlog(
        "article_router",
        {"article_refs": all_refs, "law_hint": law_hint, "results_found": 0},
    )
    return {
        "article_router_fired": False,
        "article_refs_found": all_refs,
        "law_hint_doc_id": state.get("law_hint_doc_id") or law_hint,
    }


# ---------------------------------------------------------------------------
# Node B: Entity linking
# ---------------------------------------------------------------------------

def entity_linking(state: Dict[str, Any], driver, database: str) -> Dict[str, Any]:
    extracted = state.get("entities", [])
    doc_refs = state.get("document_references") or []

    if not extracted and not doc_refs:
        logger.warning("Entity linking skipped: no extracted entities present")
        return {"entry_nodes": []}

    entries: Dict[str, Dict[str, Any]] = {}
    node_id_map: Dict[str, str] = {}

    with driver.session(database=database) as session:
        # Direct name lookup for detected document references — always resolved first
        for ref in doc_refs:
            try:
                records = session.run(
                    "MATCH (d:Document) WHERE d.name CONTAINS $ref "
                    "RETURN elementId(d) AS element_id, labels(d) AS labels",
                    ref=ref,
                )
                for record in records:
                    element_id = record["element_id"]
                    if element_id not in entries:
                        entries[element_id] = {
                            "element_id": element_id,
                            "labels": record["labels"],
                            "sources": {"name_lookup:doc_ref"},
                        }
                    else:
                        entries[element_id]["sources"].add("name_lookup:doc_ref")
            except Neo4jError as exc:
                logger.warning("Document name lookup for ref '%s' failed: %s", ref, exc)

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

    user_id = state.get("user_id") or ""
    tenant_id = state.get("tenant_id") or ""
    query_variants = state.get("query_variants") or []
    original_query = state.get("query", "")
    search_texts = [generalized] + query_variants + ([original_query] if original_query != generalized else [])

    # Resolve document scope for this query — shared across vector, BM25, and intersection
    vector_doc_hint = (state.get("law_hint_doc_id") or
                       _dynamic_law_hint(original_query, driver, database))

    with driver.session(database=database) as session:
        all_matches: List[Dict[str, Any]] = []
        for i, text in enumerate(search_texts):
            prefix = "context" if i == 0 else f"context_variant_{i}"
            all_matches.extend(
                vector_lookup(
                    session, text, indexes=CONTEXT_VECTOR_INDEXES,
                    index_settings=VECTOR_INDEX_SETTINGS, source_prefix=prefix,
                )
            )

        # If a document hint was found, filter vector matches to that document only
        if vector_doc_hint and all_matches:
            element_ids = [m["element_id"] for m in all_matches]
            scoped = session.run(
                "MATCH (d:Document)-[:CONTAINS]->(s:Section) "
                "WHERE elementId(s) IN $ids AND d.id = $doc_id "
                f"AND {_visibility_filter()} "
                "RETURN elementId(s) AS eid",
                ids=element_ids,
                doc_id=vector_doc_hint,
                user_id=user_id,
                tenant_id=tenant_id,
            ).data()
            scoped_ids = {r["eid"] for r in scoped}
            all_matches = [m for m in all_matches if m["element_id"] in scoped_ids]
            logger.info(
                "Vector search scoped to document %r — %d/%d matches kept",
                vector_doc_hint, len(all_matches), len(element_ids),
            )

    matches = all_matches

    law_hint_doc_id_b = state.get("law_hint_doc_id_b") or ""
    if (state.get("query_intent") in ("concept_across_docs", "concept_in_doc")
            and law_hint_doc_id_b
            and law_hint_doc_id_b != vector_doc_hint):
        with driver.session(database=database) as session_b:
            all_matches_b: List[Dict[str, Any]] = []
            for i, text in enumerate(search_texts):
                prefix = f"context_b_{i}" if i > 0 else "context_b"
                all_matches_b.extend(
                    vector_lookup(
                        session_b, text, indexes=CONTEXT_VECTOR_INDEXES,
                        index_settings=VECTOR_INDEX_SETTINGS,
                        source_prefix=prefix,
                    )
                )
            if all_matches_b:
                element_ids_b = [m["element_id"] for m in all_matches_b]
                scoped_b = session_b.run(
                    "MATCH (d:Document)-[:CONTAINS]->(s:Section) "
                    "WHERE elementId(s) IN $ids AND d.id = $doc_id "
                    "RETURN elementId(s) AS eid",
                    ids=element_ids_b,
                    doc_id=law_hint_doc_id_b,
                ).data()
                scoped_ids_b = {r["eid"] for r in scoped_b}
                matched_b = [m for m in all_matches_b
                             if m["element_id"] in scoped_ids_b]
                all_matches.extend(matched_b)
                # Cap combined matches to avoid synthesis overflow
                if len(all_matches) > 30:
                    all_matches = all_matches[:30]

    # Fetch BM25 section content directly
    # When a law hint is active, use the generalized/keyword query so the document
    # name itself doesn't dominate the fulltext match. Fall back to original_query.
    bm25_doc_hint = (state.get("law_hint_doc_id") or
                     _dynamic_law_hint(original_query, driver, database))
    _BM25_SCORE_THRESHOLD = 5.0 if bm25_doc_hint else 8.5
    if bm25_doc_hint:
        # Use retrieval_keywords instead of full query when scoped to one document.
        # Full query contains stop words ("quali", "sono", "le", "secondo") that
        # match everywhere in the corpus, drowning out the specific legal term.
        # Keywords (e.g. "truffa", "640", "frode") give much more precise results.
        _kw = state.get("retrieval_keywords") or []
        if _kw:
            # For single-document scoped BM25, use ONLY the most domain-specific term.
            # Common legal words like "pene", "condanna", "reato" match everywhere.
            # We want the specific crime/concept name (e.g. "truffa", "omicidio").
            _it_stops = {
                "di", "del", "della", "dei", "degli", "delle", "il", "lo", "la",
                "i", "gli", "le", "un", "uno", "una", "e", "o", "a", "da", "in",
                "con", "su", "per", "tra", "fra", "al", "dal", "nel", "sul",
                "che", "non", "si", "è", "ha", "sono", "era", "ai", "alle",
                "come", "se", "ma", "anche", "secondo", "previste", "previsto",
                # Common legal words that appear everywhere — too broad for scoped BM25
                "pene", "pena", "reato", "delitto", "articolo", "comma", "codice",
                "penale", "civile", "legge", "decreto", "norma", "disposizione",
                "condanna", "procedura", "processo",
            }
            _words = []
            for phrase in _kw:
                for word in phrase.lower().split():
                    w = re.sub(r'[^\w]', '', word)
                    if w and len(w) > 3 and w not in _it_stops and w not in _words:
                        _words.append(w)
            # Use only the 2 most specific terms — fewer terms = more precise BM25
            bm25_query = " ".join(_words[:2]) if _words else original_query
        else:
            bm25_query = original_query
    else:
        _kw = state.get("retrieval_keywords") or []
        bm25_query = " ".join(_kw) if _kw else (
            state.get("generalized_query") or original_query)
    entity_a = state.get("intent_entity_a") or ""
    entity_b = state.get("intent_entity_b") or ""
    if (state.get("query_intent") == "concept_in_doc"
            and entity_a and entity_b
            and entity_a not in bm25_query
            and entity_b not in bm25_query):
        bm25_query = f"{entity_a} {entity_b} {bm25_query}".strip()
    bm25_k = 150 if bm25_doc_hint else 15
    bm25_hits = bm25_lookup(bm25_query, driver, database, k=bm25_k)
    filtered_hits = [(eid, score) for eid, score in bm25_hits if score >= _BM25_SCORE_THRESHOLD]

    # For concept_across_docs with a second document, run a second BM25
    # pass scoped to doc_b and merge — doc_b only gets vector search
    # coverage otherwise, which frequently loses to higher-scoring content
    # from unrelated corpus documents during the reranker merge.
    law_hint_doc_id_b = state.get("law_hint_doc_id_b") or ""
    if (state.get("query_intent") == "concept_across_docs"
            and law_hint_doc_id_b
            and law_hint_doc_id_b != bm25_doc_hint):
        bm25_hits_b = bm25_lookup(bm25_query, driver, database, k=150)
        filtered_hits_b = [
            (eid, score) for eid, score in bm25_hits_b
            if score >= _BM25_SCORE_THRESHOLD
        ]
        logger.info(
            "[context_retrieval] doc_b BM25: %d hits for %r",
            len(filtered_hits_b), bm25_query,
        )
    else:
        filtered_hits_b = []
        law_hint_doc_id_b = ""

    vlog("bm25_search", {"results_found": len(bm25_hits), "results_above_threshold": len(filtered_hits)})
    raw_result: List[Dict[str, Any]] = []
    if filtered_hits:
        with driver.session(database=database) as session:
            if bm25_doc_hint:
                bm25_rows = session.run(
                    "MATCH (d:Document)-[:CONTAINS]->(s:Section) "
                    "WHERE elementId(s) IN $ids AND d.id = $doc_id "
                    f"AND {_visibility_filter()} "
                    "RETURN d, s",
                    ids=[eid for eid, _ in filtered_hits],
                    doc_id=bm25_doc_hint,
                    user_id=user_id,
                    tenant_id=tenant_id,
                ).data()
                logger.info(
                    "BM25 scoped to document %r — %d rows",
                    bm25_doc_hint, len(bm25_rows),
                )
                # Supplement with k=300 when scoped results are sparse
                if bm25_doc_hint and len(bm25_rows) < 10:
                    extra_hits = bm25_lookup(
                        bm25_query, driver, database, k=300
                    )
                    extra_filtered = [
                        (eid, score) for eid, score in extra_hits
                        if score >= _BM25_SCORE_THRESHOLD
                    ]
                    if extra_filtered:
                        with driver.session(database=database) as _sess:
                            extra_rows = _sess.run(
                                "MATCH (d:Document)-[:CONTAINS]->(s:Section) "
                                "WHERE elementId(s) IN $ids AND d.id = $doc_id "
                                "AND (coalesce(d.visibility, 'public') = 'public' "
                                "OR d.owner_id = $user_id "
                                "OR d.tenant_id = $tenant_id) "
                                "RETURN d, s",
                                ids=[eid for eid, _ in extra_filtered],
                                doc_id=bm25_doc_hint,
                                user_id=user_id,
                                tenant_id=tenant_id,
                            ).data()
                        existing_ids = {
                            (r.get("s") or {}).get("id") for r in bm25_rows
                        }
                        for row in extra_rows:
                            if (row.get("s") or {}).get("id") not in existing_ids:
                                bm25_rows.append(row)
            else:
                bm25_rows = session.run(
                    "MATCH (d:Document)-[:CONTAINS]->(s:Section) "
                    "WHERE elementId(s) IN $ids "
                    f"AND {_visibility_filter()} "
                    "RETURN d, s",
                    ids=[eid for eid, _ in filtered_hits],
                    user_id=user_id,
                    tenant_id=tenant_id,
                ).data()
        for row in bm25_rows:
            row["_source"] = "bm25"
            raw_result.append(row)

    # Second BM25 pass scoped to doc_b for concept_across_docs
    if filtered_hits_b and law_hint_doc_id_b:
        with driver.session(database=database) as _sess_b:
            bm25_rows_b = _sess_b.run(
                "MATCH (d:Document)-[:CONTAINS]->(s:Section) "
                "WHERE elementId(s) IN $ids AND d.id = $doc_id "
                f"AND {_visibility_filter()} "
                "RETURN d, s",
                ids=[eid for eid, _ in filtered_hits_b],
                doc_id=law_hint_doc_id_b,
                user_id=user_id,
                tenant_id=tenant_id,
            ).data()
        for row in bm25_rows_b:
            row["_source"] = "bm25_doc_b"
            raw_result.append(row)
        logger.info(
            "[context_retrieval] doc_b BM25 added %d rows",
            len(bm25_rows_b),
        )

    # Article-number targeted lookup: "articolo 100" / "art. 100" → match Section.name directly
    _art_rows: List[Dict[str, Any]] = []
    _art_match = re.search(r'\b(?:articolo|art\.?)\s*(\d+)', original_query, re.IGNORECASE)
    if _art_match:
        art_num = _art_match.group(1)
        _doc_hint_pat = re.search(
            r'\b(codice\s+civile|codice\s+penale|codice\s+del\s+\w+|'
            r'codice\s+dei\s+contratti|contratti\s+pubblici|codice\s+appalti|'
            r'd\.lgs\.?\s*\d+|decreto\s+legislativo\s+\d+|regolamento|'
            r'legge\s+n\.?\s*\d+)\b',
            original_query, re.IGNORECASE,
        )
        doc_hint = _doc_hint_pat.group(1).strip() if _doc_hint_pat else None
        if doc_hint:
            _art_cypher = (
                "MATCH (d:Document)-[:CONTAINS]->(s:Section) "
                "WHERE s.name = $art_num AND toLower(d.name) CONTAINS toLower($doc_hint) "
                f"AND {_visibility_filter()} "
                "RETURN d, s, elementId(s) AS s_eid"
            )
            _art_params: Dict[str, Any] = {"art_num": art_num, "doc_hint": doc_hint, "user_id": user_id, "tenant_id": tenant_id}
        else:
            _art_cypher = (
                "MATCH (d:Document)-[:CONTAINS]->(s:Section) "
                "WHERE s.name = $art_num "
                f"AND {_visibility_filter()} "
                "RETURN d, s, elementId(s) AS s_eid"
            )
            _art_params = {"art_num": art_num, "user_id": user_id, "tenant_id": tenant_id}
        with driver.session(database=database) as _art_session:
            _art_rows = _art_session.run(_art_cypher, **_art_params).data()
        for row in _art_rows:
            row["_source"] = "bm25"
        raw_result.extend(_art_rows)
        vlog("article_number_lookup", {"article_number": art_num, "doc_hint": doc_hint, "results_found": len(_art_rows)})

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

    for row in _art_rows:
        eid = row.get("s_eid", "")
        if eid and eid not in aggregated:
            aggregated[eid] = {
                "element_id": eid,
                "labels": ["Section"],
                "sources": {"bm25"},
                "score": 999.0,
            }

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
    return {
        "context_nodes": context_nodes,
        "raw_result": raw_result,
        "law_hint_doc_id": vector_doc_hint or "",
    }


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
                        "retrieval_fallback": True,
                    }
        return {
            "retrieval_quality_ok": False,
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


# ---------------------------------------------------------------------------
# Node F: Answer synthesis
# ---------------------------------------------------------------------------

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
                if not doc_id:
                    continue
                if doc_id not in docs:
                    docs[doc_id] = {"title": None, "sections": {}}
                docs[doc_id]["title"] = doc_name

            elif is_section:
                section_name = value.get("name") or value.get("title") or ""
                parts = node_id.split("::")
                doc_id = ("LEGAL_DOC::" + parts[1]) if len(parts) >= 3 else (value.get("document_id") or "")
                if not doc_id:
                    continue
                if doc_id not in docs:
                    docs[doc_id] = {"title": None, "sections": {}}
                section_plain_text = (value.get("plain_text") or value.get("text") or "").strip()
                section_title = (value.get("title") or "").strip() or None
                if not section_title:
                    section_abstract = (value.get("abstract") or "").strip()
                    title_match = re.match(r"^-\s*(.+?)\s*-", section_abstract)
                    section_title = title_match.group(1) if title_match else None
                section_score = record.get("_reranker_score")
                if section_name and section_name != "0" and section_plain_text:
                    docs[doc_id]["sections"].setdefault(section_name, {
                        "plain_text": section_plain_text,
                        "title": section_title,
                        "score": section_score,
                    })

    def _section_sort_key(item):
        name, sec = item
        score = sec.get("score")
        if score is None:
            return (1, 0, name)
        return (0, -score, name)

    results = [
        {
            "document_name": info["title"] or doc_id,
            "document_id": doc_id,
            "sections": [
                {
                    "name": name,
                    "title": sec["title"],
                    "plain_text": sec["plain_text"],
                    "score": sec.get("score"),
                    "url": (
                        f"/api/documents/{urllib.parse.quote(doc_id, safe='')}/"
                        f"sections/{urllib.parse.quote(name, safe='')}"
                    ),
                }
                for name, sec in sorted(info["sections"].items(), key=_section_sort_key)
                if len(name) <= 200 and len(sec["plain_text"]) > 10
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


def _citation_is_relevant(answer: str, section_text: str, threshold: float = 0.80) -> bool:
    if not section_text or not answer:
        return False
    try:
        if len(section_text) <= 800:
            effective_threshold = threshold - 0.08   # 0.72 for short sections
        elif len(section_text) <= 3000:
            effective_threshold = threshold           # 0.80 for medium sections
        else:
            effective_threshold = threshold - 0.12   # 0.68 for very long sections
        answer_emb = _embed_query_with_prefix(answer[:500])
        section_emb = _embed_query_with_prefix(section_text[:500])
        dot = sum(a * b for a, b in zip(answer_emb, section_emb))
        norm_a = sum(a * a for a in answer_emb) ** 0.5
        norm_b = sum(b * b for b in section_emb) ** 0.5
        similarity = dot / (norm_a * norm_b + 1e-9)
        return similarity >= effective_threshold
    except Exception:
        return True  # keep citation on error


_RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "false").lower() == "true"
_RERANKER_URL = os.getenv("RERANKER_URL", "http://217.160.8.129:8002/v1/rerank")


def _format_for_reranker(row: dict) -> str:
    """Build structured reranker input that emphasises title and
    document context over raw text length. Works for any document
    regardless of name, version, or section length."""
    s = row.get("s") or {}
    d = row.get("d") or {}

    name = s.get("name", "")
    abstract = (s.get("abstract") or "").strip()
    plain_text = (s.get("plain_text") or "").strip()
    doc_name = d.get("name", "")

    parts = []

    # Document context — skip raw filenames
    if doc_name and not doc_name.lower().endswith(
            ('.pdf', '.docx', '.xlsx', '.txt')):
        parts.append(f"Fonte: {doc_name}")

    # Section identifier
    if name:
        parts.append(f"Articolo: {name}")

    # Content — combine abstract and plain_text for maximum signal
    # Abstract already has title prepended (e.g. "Omicidio - ...")
    # Plain text has the actual legal provision
    content = abstract or plain_text
    if plain_text and plain_text not in (abstract or ""):
        content = f"{content} {plain_text}"[:500]
    else:
        content = content[:500]

    if content:
        parts.append(content)

    return " | ".join(parts)


def rerank_results(query: str, rows: list, top_k: int = 12) -> list:
    """Re-sort rows by reranker score. Fail-open: returns rows unchanged on any error."""
    if not _RERANKER_ENABLED or not rows:
        return rows
    if all(r.get("_source") == "clarification" for r in rows):
        return rows  # skip reranker for clarification rerank — scores already set
    import time as _time
    import requests
    t0 = _time.time()
    try:
        reranker_query = (
            f"Instruct: Given a legal query in Italian, retrieve the most relevant legal document sections\n"
            f"Query: {query}"
        )
        payload = {
            "model": os.getenv("RERANKER_MODEL", "reranker"),
            "query": reranker_query,
            "documents": [
                _format_for_reranker(row) for row in rows
            ],
        }
        api_key = os.getenv("LLM_API_KEY", "")
        resp = requests.post(
            _RERANKER_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=10,
        )
        resp.raise_for_status()
        scored = resp.json().get("results", [])
        if not scored:
            return rows
        for result in scored:
            original_idx = result["index"]
            score = result["relevance_score"]
            if original_idx < len(rows):
                rows[original_idx]["_reranker_score"] = score
        reranker_top = sorted(rows, key=lambda r: r.get("_reranker_score", 0), reverse=True)[:top_k - 2]

        bm25_rows = [r for r in rows if r.get("_source") == "bm25"]
        reranker_ids = {
            (r.get("s") or {}).get("id") for r in reranker_top
            if (r.get("s") or {}).get("id")
        }
        BM25_INJECTION_MIN_RERANKER_SCORE = 0.3
        bm25_candidates = [
            r for r in bm25_rows
            if (r.get("s") or {}).get("id") not in reranker_ids
            and r.get("_reranker_score", 0) >= BM25_INJECTION_MIN_RERANKER_SCORE
        ]
        bm25_top = sorted(
            bm25_candidates, key=lambda r: r.get("_reranker_score", 0), reverse=True
        )[:2]

        merged_ids = reranker_ids | {
            (r.get("s") or {}).get("id") for r in bm25_top
            if (r.get("s") or {}).get("id")
        }
        slots_remaining = top_k - len(reranker_top) - len(bm25_top)
        overflow = [
            r for r in sorted(rows, key=lambda r: r.get("_reranker_score", 0), reverse=True)
            if (r.get("s") or {}).get("id") not in merged_ids
        ][:slots_remaining]

        reranked = reranker_top + bm25_top + overflow
        logger.info(
            "Reranker merge: reranker_top=%d bm25_injected=%d overflow=%d total=%d",
            len(reranker_top), len(bm25_top), len(overflow), len(reranked),
        )

        score_debug = []
        for r in reranked:
            s = r.get("s") or {}
            if hasattr(s, "get"):
                name = s.get("name") or s.get("title") or "?"
            else:
                name = str(s)[:20]
            score_debug.append((name, round(r.get("_reranker_score", 0), 3)))
        logger.info("Reranker scores (top %d): %r", len(score_debug), score_debug)
        vlog("reranker", {"input_count": len(rows), "output_count": len(reranked)}, (_time.time() - t0) * 1000)
        return reranked
    except Exception as exc:
        logger.warning("Reranker failed (fail-open): %s", exc)
        vlog("reranker", {"input_count": len(rows), "output_count": len(rows), "error": str(exc)[:120]}, (_time.time() - t0) * 1000)
        return rows


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
    threshold = min(150, max(100, int(len(answer) * 0.15)))
    return earliest_idx < threshold


def synthesize_answer(state: Dict[str, Any]) -> Dict[str, Any]:
    lang = _session_lang(state)
    error = state.get("execution_error") or state.get("cypher_generation_error")
    data = rerank_results(state.get("query", ""), state.get("raw_result") or [])
    logger.info("synthesize_answer: data=%d, raw_result=%d, bm25_doc_ids=%s", len(data), len(state.get('raw_result') or []), state.get('bm25_doc_ids'))

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

    tone = int(state.get("tone") or 2)
    standing = int(state.get("standing") or 2)
    response_length = int(state.get("response_length") or 2)

    if error:
        answer = _call_chat(
            [
                SystemMessage(content=synthesis_error_system(lang, tone=tone, standing=standing, length=response_length)),
                HumanMessage(
                    content=(
                        "Original question: {question}\n"
                        "Internal retrieval note (do not quote literally or discuss IT systems): {error}\n\n"
                        "Provide the legal consultation as instructed."
                    ).format(question=state["query"], error=error)
                    + synthesis_human_footer(lang)
                ),
            ]
        )
        answer = _strip_vague_closing(answer)
        return {
            "answer": answer,
            "references": state.get("raw_result", []) or [],
            "status_messages": state.get("status_messages") or [],
        }

    if not data:
        answer = _call_chat(
            [
                SystemMessage(content=synthesis_empty_system(lang, tone=tone, standing=standing, length=response_length)),
                HumanMessage(
                    content=(
                        "Original question: {question}\n"
                        "The knowledge graph query returned no rows.\n\n"
                        "Provide the legal consultation as instructed."
                    ).format(question=state["query"])
                    + synthesis_human_footer(lang)
                ),
            ]
        )
        answer = _strip_vague_closing(answer)
        return {
            "answer": answer,
            "references": [],
            "status_messages": state.get("status_messages") or [],
        }

    _is_cmp = state.get("is_comparison", False)
    summarized_data = _summarize_for_synthesis(data, is_comparison=_is_cmp)
    serialized = json.dumps(summarized_data, ensure_ascii=False)
    char_cap = 4000 if _is_cmp else 15000
    if len(serialized) > char_cap:
        serialized = json.dumps(
            _summarize_for_synthesis(data, max_records=3,
                                     is_comparison=_is_cmp),
            ensure_ascii=False,
            indent=None if _is_cmp else 2,
        )
    all_citations = _extract_citations(data)
    citation_strings = [
        f"Fonte: {c['document_name']}" if not c["sections"]
        else f"Fonte: {c['document_name']}, sezione: {c['sections'][0]['name']}" if len(c["sections"]) == 1
        else f"Fonte: {c['document_name']}, sezioni: {', '.join(s['name'] for s in c['sections'])}"
        for c in all_citations
    ]

    human_parts = [
        f"Question: {state['query']}\n",
        f"Data: {serialized}\n",
    ]
    if citation_strings:
        human_parts.append("Fonti disponibili:\n" + "\n".join(citation_strings) + "\n")
    if state.get("bm25_doc_ids") and state.get("bm25_from_article_lookup"):
        human_parts.append(
            "Answer ONLY using the data provided above. Do not use any knowledge outside of the retrieved data. "
            "The retrieved data contains sections of the requested article — synthesize them into a coherent answer. "
            "Do NOT say the article is not present — it IS present in the data above. "
            "Quote relevant passages directly and explain their meaning."
        )
    elif state.get("bm25_doc_ids") and not state.get("bm25_from_article_lookup"):
        human_parts.append(
            "Answer ONLY using the data provided above. "
            "The retrieved sections are directly relevant to the question — "
            "use them as your primary source. "
            "Do NOT say the information is not present — it IS present in "
            "the data above. Cite the specific sections that address the "
            "question most directly."
        )
    else:
        human_parts.append(
            "Answer ONLY using the data provided above. Do not use any knowledge outside of the retrieved data. "
            "If the retrieved data does not contain enough information to answer the question, you MUST say one of these phrases: "
            "'non è presente nei documenti' or 'non trovo informazioni nei documenti forniti'. "
            "Never invent, infer, or extrapolate beyond what is explicitly stated in the data. "
            "Quote short passages in their original language from the data; explain and synthesize in the session language."
        )
    if citation_strings:
        human_parts.append(
            "\nIf your answer draws from the retrieved data, end with "
            "a 'Fonti:' line citing the relevant documents and sections from the list above. "
            "If no retrieved data was used, omit the Fonti line entirely."
        )

    if state.get("retrieval_fallback"):
        _lang = state.get("session_language", "it")
        if _lang == "es":
            fallback_answer = "El tema de su consulta no está presente en los documentos disponibles en mi base de conocimiento. Le recomiendo consultar las fuentes oficiales pertinentes para obtener información precisa. Si lo desea, puedo ayudarle con temas relacionados disponibles en mi base documental."
        elif _lang == "en":
            fallback_answer = "The topic of your query is not present in the documents available in my knowledge base. I recommend consulting the relevant official sources for accurate information. If you wish, I can help you with related topics available in my knowledge base."
        else:
            fallback_answer = "L'argomento della sua domanda non è presente nei documenti disponibili nella mia base di conoscenza. Le consiglio di consultare le fonti ufficiali pertinenti per ottenere informazioni precise. Se desidera, posso aiutarla con domande correlate presenti nella mia base documentale."
        return {"answer": fallback_answer, "citations": [], "references": []}

    log_cypher_event(
        "e_synthesize_start",
        "synthesis LLM call starting",
        detail={"citations_count": len(all_citations)},
    )
    import time as _time
    _t1 = _time.time()
    if state.get("is_comparison"):
        system_prompt = (
            "Sei un assistente legale. Confronta i due documenti forniti nei dati. "
            "Usa bullet points con •. Un bullet per tema. "
            "Per ogni bullet: Documento 1 dice X, Documento 2 dice Y. "
            "Solo testo semplice, niente markdown **."
        )
    else:
        system_prompt = synthesis_system_message(
            lang,
            retrieval_fallback=state.get("retrieval_fallback", False),
            is_comparison=False,
            tone=tone,
            standing=standing,
            length=response_length,
        )
    answer = _call_chat(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content="".join(human_parts) + synthesis_human_footer(lang)),
        ],
        max_tokens=600,
    )
    vlog("synthesis_llm", {"citations_count": len(all_citations), "answer_length": len(answer)}, (_time.time() - _t1) * 1000)
    log_cypher_event(
        "e_synthesize_end",
        "synthesis LLM call complete",
        detail={"answer_length": len(answer)},
    )
    answer = _strip_vague_closing(answer)
    bm25_doc_ids = state.get("bm25_doc_ids") or []
    existing_doc_refs = state.get("document_references") or []
    merged_doc_refs = list(set(existing_doc_refs + bm25_doc_ids))
    citations = _extract_citations(
        data, answer=answer, doc_refs=merged_doc_refs if merged_doc_refs else None
    )
    # Limit to top 5 most-cited documents to reduce noise
    if len(citations) > 5:
        citations = sorted(citations, key=lambda c: len(c['sections']), reverse=True)[:5]
    # Filter citations to only sections that survived reranker
    if data and any(r.get('_reranker_score') is not None for r in data) and not state.get("is_comparison"):
        reranker_texts = {
            r.get('s', {}).get('plain_text', '')[:100]
            for r in data
            if r.get('_reranker_score') is not None and r.get('_reranker_score', 0) >= 0.25
        }
        citations = [
            {**c, 'sections': [
                s for s in c['sections']
                if s.get('plain_text', '')[:100] in reranker_texts
                or c.get('document_id', '') in bm25_doc_ids
            ]}
            for c in citations
        ]
        citations = [c for c in citations if c['sections']]

    # Keyword relevance filter: drop sections that don't contain at least one
    # meaningful query keyword. Prevents sections that match a single legal term
    # (e.g. "nullità") from appearing when the query is about a different context
    # (e.g. "nullità matrimonio" vs "nullità contratti").
    keywords = [
        k.lower() for k in (state.get("retrieval_keywords") or [])
        if len(k) > 3
    ]
    if keywords and not state.get("is_comparison"):
        # Build a set of section plain_text prefixes that came directly from BM25 —
        # only these specific sections bypass the keyword filter, not the whole document
        bm25_section_texts = {
            r.get('s', {}).get('plain_text', '')[:100]
            for r in data
            if r.get('_source') == 'bm25'
        }

        # Also build individual tokens from keyword phrases for partial matching
        keyword_tokens = {
            token
            for kw in keywords
            for token in kw.split()
            if len(token) > 4
        }

        def _section_matches_keywords(section: dict) -> bool:
            text = (section.get('plain_text') or '').lower()
            # Match full phrase first
            if any(kw in text for kw in keywords):
                return True
            # Fall back to individual token matching — require at least 2 tokens to match
            # to avoid false positives from common legal terms
            token_hits = sum(1 for t in keyword_tokens if t in text)
            return token_hits >= 2

        def _section_from_bm25(section: dict) -> bool:
            return section.get('plain_text', '')[:100] in bm25_section_texts

        citations = [
            {**c, 'sections': [
                s for s in c['sections']
                if _section_matches_keywords(s) or _section_from_bm25(s)
            ]}
            for c in citations
        ]
        citations = [c for c in citations if c['sections']]

    # Answer reference filter: only keep sections whose article number is
    # explicitly mentioned in the answer, or whose plain_text has substantial
    # overlap with the answer content. This ensures cited sections were actually used.
    answer_lower = answer.lower()
    cited_section_pattern = re.compile(r'\b(?:articolo|art\.?|sezione|sez\.?)\s*(\d+(?:[.\-]\d+)*(?:[\s\-]*(?:bis|ter|quater))?)', re.IGNORECASE)
    answer_article_refs = {m.group(1).strip().lower().replace(' ', '') for m in cited_section_pattern.finditer(answer)}
    # Also add base article numbers (e.g. "124" from "124.0.0")
    answer_article_refs |= {ref.split('.')[0] for ref in answer_article_refs}

    # Reranker-score-based citation filter.
    # Trust the reranker's semantic relevance score rather than keyword matching.
    # Sections with score >= 0.5 are always included.
    # Sections with score 0.3-0.5 are included only if article number also in answer.
    # Sections with score < 0.3 are excluded (noise).
    # BM25 article-lookup sections bypass the filter entirely (already highly targeted).
    if not state.get("is_comparison"):
        def _section_passes_quality(section: dict, doc_id: str) -> bool:
            # BM25 article lookup — always include
            if doc_id in bm25_doc_ids and state.get("bm25_from_article_lookup"):
                return True
            score = section.get("score")
            if score is None:
                # No reranker score — fall back to article reference check
                name = (section.get('name') or '').lower().replace(' ', '')
                base_name = name.split('.')[0]
                return name in answer_article_refs or base_name in answer_article_refs
            if score >= 0.5:
                return True
            if score >= 0.3:
                # Medium confidence — only include if article number in answer
                name = (section.get('name') or '').lower().replace(' ', '')
                base_name = name.split('.')[0]
                return name in answer_article_refs or base_name in answer_article_refs
            return False

        filtered = [
            {**c, 'sections': [
                s for s in c['sections']
                if _section_passes_quality(s, c.get('document_id', ''))
            ]}
            for c in citations
        ]
        # Only apply if filter keeps at least 1 section
        if any(fc['sections'] for fc in filtered):
            citations = [c for c in filtered if c['sections']]

    answer = _strip_hallucinated_fonti(answer)

    # Citation quality filter disabled — corpus too small for meaningful filtering
    # Gap phrase detection handles hallucination prevention instead
    filtered_citations = citations
    citations = filtered_citations

    # Hard stop enforcement: if the answer acknowledges a gap, clear all citations.
    # Comparison answers legitimately say "document X doesn't cover this" — skip gap detection.
    if state.get("is_comparison"):
        is_gap = False
    else:
        is_gap = _is_primary_gap_response(answer)
    if is_gap and citations:
        if not (state.get("bm25_doc_ids") and state.get("bm25_from_article_lookup")) and not state.get("is_clarification_rerank"):
            citations = []
            logger.debug("Hard stop detected: citations cleared")

    # Truncate answer at gap phrase if hard stop detected
    if is_gap:
        answer_lower = answer.lower()
        for phrase in _GAP_PHRASES:
            idx = answer_lower.find(phrase)
            if idx != -1:
                gap_sentence_end = answer.find('.', idx + len(phrase))
                if gap_sentence_end == -1:
                    gap_sentence_end = len(answer) - 1
                period_count = 0
                cut_pos = len(answer)
                for i, ch in enumerate(answer[gap_sentence_end + 1:], start=gap_sentence_end + 1):
                    if ch == '.':
                        period_count += 1
                        if period_count == 2:
                            cut_pos = i + 1
                            break
                answer = answer[:cut_pos].strip()
                if not (state.get("bm25_doc_ids") and state.get("bm25_from_article_lookup")) and not state.get("is_clarification_rerank"):
                    citations = []
                logger.debug("Hard stop: answer truncated after 3-sentence polite response")
                break

    # Remove redundant "In definitiva" closing after gap acknowledgment
    if is_gap:
        for closing in ["In definitiva,", "In definitiva ", "In summary,", "In summary ", "En definitiva,"]:
            idx = answer.find(closing)
            if idx != -1:
                last_period = answer.rfind('.', 0, idx)
                if last_period != -1:
                    answer = answer[:last_period + 1].strip()
                break

    # Final citation clear for primary gap responses
    if is_gap:
        if not (state.get("bm25_doc_ids") and state.get("bm25_from_article_lookup")) and not state.get("is_clarification_rerank"):
            citations = []

    if citations:
        fonti_line = "Fonti: " + ", ".join(
            f"{c['document_name']} sezioni: {', '.join(s['name'] for s in c['sections'])}"
            for c in citations
        )
        answer = answer.rstrip() + "\n\n" + fonti_line
    return {
        "answer": answer,
        "references": data,
        "citations": citations,
        "status_messages": state.get("status_messages") or [],
    }


# ---------------------------------------------------------------------------
# Node: Clarification flow
# ---------------------------------------------------------------------------


def generate_clarifying_question(state: Dict[str, Any]) -> Dict[str, Any]:
    """When the answer draws on 2+ distinct source documents, ask the user
    one contextual clarifying question to narrow down which context applies,
    and stash the candidate sections so the next turn can re-rank instead of
    re-retrieving from scratch."""
    if state.get("pending_sections"):
        return {}
    citations = state.get("citations") or []
    unique_doc_names = {c.get("document_name") for c in citations if c.get("document_name")}
    if len(unique_doc_names) < 2:
        return {}

    system = (
        "Sei un assistente legale italiano. Hai appena risposto a una domanda legale citando più fonti diverse. "
        "Genera UNA SOLA domanda di chiarimento in italiano, breve e specifica, che aiuti a capire quale contesto "
        "si applica alla situazione dell'utente. La domanda deve essere contestuale alla risposta data, non generica."
    )
    human = (
        f"Domanda originale: {state['query']}\n\n"
        f"Risposta data: {state['answer']}\n\n"
        f"Fonti citate: {[c['document_name'] for c in citations]}"
    )

    try:
        clarifying_question = _call_chat(
            [SystemMessage(content=system), HumanMessage(content=human)],
            max_tokens=150,
        ).strip()
    except Exception as e:
        logger.warning("generate_clarifying_question: LLM call failed: %s", e)
        return {}

    pending_sections: List[Dict[str, Any]] = [
        {
            "document_name": c.get("document_name"),
            "document_id": c.get("document_id"),
            "name": s.get("name"),
            "title": s.get("title"),
            "plain_text": s.get("plain_text"),
            "score": s.get("score"),
        }
        for c in citations
        for s in (c.get("sections") or [])
    ]

    return {
        "answer": state.get("answer", "") + "\n" + clarifying_question,
        "awaiting_clarification": True,
        "pending_sections": pending_sections,
    }


def rerank_from_clarification(state: Dict[str, Any]) -> Dict[str, Any]:
    """Re-score the sections retrieved last turn against the user's
    clarification message, instead of re-running retrieval from scratch."""
    pending_sections = state.get("pending_sections") or []
    query = state.get("query", "")

    if not pending_sections:
        return {
            "raw_result": [],
            "context_nodes": [],
            "awaiting_clarification": False,
        }

    system = (
        "Sei un assistente legale italiano. L'utente ha chiarito il contesto della sua domanda. "
        "Devi classificare le seguenti sezioni di testi legali in base alla loro rilevanza per il contesto chiarito dall'utente. "
        "Restituisci SOLO un array JSON con TUTTI gli indici (0-based) in ordine di rilevanza, dal più al meno pertinente. Includi tutti gli indici nell'array. "
        "Esempio: [2, 0, 4]"
    )
    sections_list = "\n".join(
        f"{i}. {s.get('document_name', '')} — {s.get('title') or s.get('name') or ''}: "
        f"{(s.get('plain_text') or '')[:200]}"
        for i, s in enumerate(pending_sections)
    )
    human = f"Chiarimento utente: {query}\n\nSezioni disponibili:\n{sections_list}"

    try:
        raw = _call_chat(
            [SystemMessage(content=system), HumanMessage(content=human)],
            max_tokens=200,
        )
        text = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
        indices = json.loads(text)
        if not isinstance(indices, list):
            raise ValueError("LLM did not return a JSON array")
    except Exception as e:
        logger.warning("rerank_from_clarification: LLM call/parse failed: %s", e)
        indices = list(range(len(pending_sections)))

    selected = [
        pending_sections[i] for i in indices
        if isinstance(i, int) and 0 <= i < len(pending_sections)
    ]

    new_raw_result = [
        {
            "d": {
                "id": s.get("document_id", f"LEGAL_DOC::{s.get('document_name', '')}"),
                "name": s.get("document_name"),
            },
            "s": {
                "id": f"DOCUMENT_SECTION::{s.get('document_id', '').replace('LEGAL_DOC::', '')}{'::'}{s.get('name', '')}",
                "name": s.get("name"),
                "title": s.get("title"),
                "plain_text": s.get("plain_text"),
                "score": s.get("score"),
            },
            "_reranker_score": 0.9,  # treat clarification-selected sections as high-confidence
            "_source": "clarification",
        }
        for s in selected
    ]

    clarification_doc_ids = list({row["d"]["id"] for row in new_raw_result if row.get("d", {}).get("id")})

    logger.info("rerank_from_clarification: returning %d rows, bm25_doc_ids=%s", len(new_raw_result), clarification_doc_ids)

    return {
        "raw_result": new_raw_result,
        "context_nodes": new_raw_result,
        "bm25_doc_ids": clarification_doc_ids,
        "awaiting_clarification": False,
        "is_clarification_rerank": True,
        "retrieval_quality_ok": True,
    }


def _resolve_by_name(name: str, session, user_id: str = "", tenant_id: str = "") -> Optional[str]:
    """Fallback doc-ID resolver: tries each candidate token and accepts only a unique match."""
    _STOPWORDS = {"del", "dei", "delle", "della", "dello", "gli", "per", "con", "tra", "fra", "sul", "sulla", "verbale"}
    raw_tokens = [t for t in name.split() if len(t) > 3 and t.lower() not in _STOPWORDS]
    # Prioritise: all-caps tokens first (acronyms/proper names), then by descending length
    candidate_tokens = (
        [t for t in raw_tokens if t.upper() == t]
        + sorted([t for t in raw_tokens if t.upper() != t], key=len, reverse=True)
    )
    for token in candidate_tokens:
        results = list(session.run(
            "MATCH (d:Document) WHERE d.name CONTAINS $token "
            f"AND {_visibility_filter()} "
            "RETURN d.id AS id LIMIT 2",
            token=token.upper(),
            user_id=user_id,
            tenant_id=tenant_id,
        ))
        if len(results) == 1:
            return results[0]["id"]
    return None


# ---------------------------------------------------------------------------
# Node: Cross-document comparison retrieval
# ---------------------------------------------------------------------------

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


def comparison_retrieval(state: Dict[str, Any], driver, database: str) -> Dict[str, Any]:
    """Fetch section pairs from two documents for side-by-side comparison synthesis."""
    names = state.get("comparison_doc_ids") or []
    query = state.get("query", "")
    keywords = state.get("retrieval_keywords") or []
    user_id = state.get("user_id") or ""
    tenant_id = state.get("tenant_id") or ""

    # Resolve names → document IDs (accept either a human name or a bare doc ID)
    doc_ids: List[str] = []
    with driver.session(database=database) as session:
        for name in names[:2]:
            result = session.run(
                "MATCH (d:Document)-[:CONTAINS]->(:Section) "
                "WHERE (toLower(d.name) CONTAINS toLower($name) OR d.id = $name) "
                f"AND {_visibility_filter()} "
                "RETURN d.id AS id LIMIT 1",
                name=name,
                user_id=user_id,
                tenant_id=tenant_id,
            ).single()
            if result and result["id"]:
                doc_ids.append(result["id"])

    # If name lookup failed, try splitting the query at conjunctions/versus
    if len(doc_ids) < 2:
        parts = re.split(
            r'\b(?:e|and|y|vs\.?|versus|rispetto\s+a|con)\b',
            query, maxsplit=1, flags=re.IGNORECASE,
        )
        for part in parts[:2]:
            hint = _dynamic_law_hint(part.strip(), driver, database)
            if hint and hint not in doc_ids:
                doc_ids.append(hint)

    # Fallback: token-based CONTAINS search for any still-unresolved names
    if len(doc_ids) < 2:
        with driver.session(database=database) as session:
            for name in names[:2]:
                fid = _resolve_by_name(name, session, user_id=user_id, tenant_id=tenant_id)
                if fid and fid not in doc_ids:
                    doc_ids.append(fid)

    doc_id_1 = doc_ids[0] if len(doc_ids) >= 1 else None
    doc_id_2 = doc_ids[1] if len(doc_ids) >= 2 else None

    if not doc_id_2 or doc_id_1 == doc_id_2:
        logger.warning(
            "comparison_retrieval: could not resolve two distinct documents (got %r) — re-routing as regular query",
            doc_ids,
        )
        return {
            "is_comparison": False,
            "raw_result": [],
            "retrieval_quality_ok": False,
            "neo4j_executed": False,
        }

    with driver.session(database=database) as session:
        count1 = (session.run(
            "MATCH (d:Document {id: $id})-[:CONTAINS]->(s:Section) "
            f"WHERE {_visibility_filter()} "
            "RETURN count(s) AS cnt",
            id=doc_id_1,
            user_id=user_id,
            tenant_id=tenant_id,
        ).single() or {}).get("cnt", 0)
        count2 = (session.run(
            "MATCH (d:Document {id: $id})-[:CONTAINS]->(s:Section) "
            f"WHERE {_visibility_filter()} "
            "RETURN count(s) AS cnt",
            id=doc_id_2,
            user_id=user_id,
            tenant_id=tenant_id,
        ).single() or {}).get("cnt", 0)

    is_short = count1 < 50 and count2 < 50
    fetch_limit = 999 if is_short else 30
    rank_limit = 999 if is_short else 3
    pair_limit = min(count1 + count2, 100) if is_short else 20

    # Fetch sections with plain_text from each document
    with driver.session(database=database) as session:
        rows1 = session.run(
            "MATCH (d:Document {id: $doc_id})-[:CONTAINS]->(s:Section) "
            "WHERE s.plain_text IS NOT NULL AND s.plain_text <> '' "
            f"AND {_visibility_filter()} "
            "RETURN d, s ORDER BY s.name LIMIT $lim",
            doc_id=doc_id_1, lim=fetch_limit,
            user_id=user_id, tenant_id=tenant_id,
        ).data()
        rows2 = session.run(
            "MATCH (d:Document {id: $doc_id})-[:CONTAINS]->(s:Section) "
            "WHERE s.plain_text IS NOT NULL AND s.plain_text <> '' "
            f"AND {_visibility_filter()} "
            "RETURN d, s ORDER BY s.name LIMIT $lim",
            doc_id=doc_id_2, lim=fetch_limit,
            user_id=user_id, tenant_id=tenant_id,
        ).data()

    if not rows1 or not rows2:
        return {
            "comparison_doc_ids": doc_ids,
            "raw_result": [],
            "execution_error": "No sections with plain_text found in one or both documents",
            "neo4j_executed": True,
        }

    # Short documents: concatenate full text of both docs into a single pair so
    # the LLM receives complete content without section-pairing losses.
    if is_short:
        doc1_text = "\n\n".join(
            r["s"]["plain_text"] for r in rows1 if r.get("s") and r["s"].get("plain_text")
        )
        doc2_text = "\n\n".join(
            r["s"]["plain_text"] for r in rows2 if r.get("s") and r["s"].get("plain_text")
        )
        logger.info(
            "comparison_retrieval (short): doc1=%r (%d sections), doc2=%r (%d sections)",
            doc_id_1, len(rows1), doc_id_2, len(rows2),
        )
        return {
            "raw_result": [{
                "d": rows1[0]["d"],
                "s": {"name": "verbale_completo", "plain_text": doc1_text, "id": doc_id_1},
                "d2": rows2[0]["d"],
                "s2": {"name": "verbale_completo", "plain_text": doc2_text, "id": doc_id_2},
                "_source": "comparison",
            }],
            "is_comparison": True,
            "comparison_doc_ids": [doc_id_1, doc_id_2],
            "retrieval_quality_ok": True,
            "neo4j_executed": True,
            "execution_error": None,
        }

    # Rank sections by keyword relevance against the query
    kw_lower = (
        [k.lower() for k in keywords]
        if keywords
        else [w for w in query.lower().split() if len(w) > 3]
    )

    def _relevance(row: Dict) -> float:
        text = ((row.get("s") or {}).get("plain_text") or "").lower()
        return sum(1.0 for kw in kw_lower if kw in text)

    ranked1 = sorted(rows1, key=_relevance, reverse=True)[:rank_limit]
    ranked2 = sorted(rows2, key=_relevance, reverse=True)[:rank_limit]

    # Prefer pairing sections that share the same article name across documents
    by_name1 = {
        (r.get("s") or {}).get("name", ""): r
        for r in ranked1 if (r.get("s") or {}).get("name")
    }
    by_name2 = {
        (r.get("s") or {}).get("name", ""): r
        for r in ranked2 if (r.get("s") or {}).get("name")
    }

    pairs: List[Dict[str, Any]] = []
    paired1: set = set()
    paired2: set = set()

    for name, r1 in by_name1.items():
        if name in by_name2:
            r2 = by_name2[name]
            pairs.append({
                "d": _node_to_dict(r1.get("d")),
                "s": _node_to_dict(r1.get("s")),
                "d2": _node_to_dict(r2.get("d")),
                "s2": _node_to_dict(r2.get("s")),
                "_source": "comparison",
            })
            paired1.add(id(r1))
            paired2.add(id(r2))

    # Fill remaining slots — use zip_longest so sections from the longer document
    # are not silently dropped when one side has fewer sections than the other.
    rem1 = [r for r in ranked1 if id(r) not in paired1]
    rem2 = [r for r in ranked2 if id(r) not in paired2]
    for r1, r2 in itertools.zip_longest(rem1, rem2):
        pairs.append({
            "d": _node_to_dict(r1.get("d") if r1 else None),
            "s": _node_to_dict(r1.get("s") if r1 else None),
            "d2": _node_to_dict(r2.get("d") if r2 else None),
            "s2": _node_to_dict(r2.get("s") if r2 else None),
            "_source": "comparison",
        })

    logger.info(
        "comparison_retrieval: doc1=%r (%d sections), doc2=%r (%d sections), pairs=%d",
        doc_id_1, len(rows1), doc_id_2, len(rows2), len(pairs),
    )

    return {
        "comparison_doc_ids": doc_ids,
        "raw_result": pairs[:3],
        "neo4j_executed": True,
        "execution_error": None,
    }


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def route_after_decompose(state: Dict[str, Any]) -> str:
    if state.get("off_topic"):
        return "off_topic"
    if state.get("is_comparison"):
        return "comparison"
    return "legal"


def route_after_article_router(state: Dict[str, Any]) -> str:
    return "fired" if state.get("article_router_fired") else "pass"


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
        error = state["execution_error"]
        attempt = state.get("cypher_attempt", "")
        if attempt not in ("fallback", "reformulation") and any(
            kw in error for kw in ("SyntaxError", "Invalid")
        ):
            return "retry"
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
