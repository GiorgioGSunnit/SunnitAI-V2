"""Document-name lookup, query classification and reference extraction helpers."""

import json
import logging
import os
import re
from typing import List, Set

from langchain_core.messages import HumanMessage, SystemMessage

from .ai_chat import _call_chat
from .utils import _ALL_SCHEMA_LABELS, _parse_json_list, _strict_filter_relations

logger = logging.getLogger(__name__)


_INTENT_CLASSIFIER_TIMEOUT = 8  # seconds


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


# ---------------------------------------------------------------------------
# Article number detection
# ---------------------------------------------------------------------------

_ARTICLE_PATTERNS = [
    re.compile(r'\bartt?\.?\s*(\d+)\s*(?:bis|ter|quater|quinquies)?\b', re.IGNORECASE),
    re.compile(r'\barticol[oi]\s+(\d+)\s*(?:bis|ter|quater|quinquies)?\b', re.IGNORECASE),
    re.compile(r'\bartículo\s+(\d+)\b', re.IGNORECASE),
    re.compile(r'\b§\s*(\d+)\b'),
    # Bare number shorthand lawyers use without "art." — e.g. "612 bis cp"
    re.compile(r'\b(\d{3})\s*(?:bis|ter|quater|quinquies)?\s*(?:c\.?p\.?|cod\.?\s*pen\.?)\b', re.IGNORECASE),
    # Second (and later) numbers in a comma-separated "artt." list — e.g. "artt. 582, 583"
    re.compile(r'\bartt\.?\s*\d+\s*,\s*(\d+)\b', re.IGNORECASE),
    # Bare number preceded by "ex" — Italian legal shorthand: "ex 576", "ex 612 bis"
    re.compile(r'\bex\s+(\d{3})\s*(?:bis|ter|quater|quinquies)?\b', re.IGNORECASE),
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
                    "WHERE d.document_type IN ['primary', 'special', 'ccnl'] "
                    "AND (coalesce(d.visibility, 'public') = 'public' "
                    "OR d.owner_id = $user_id OR d.tenant_id = $tenant_id) "
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

    # Pre-check: standard Italian legal code abbreviations
    if _has_article_ref:
        _CODE_ABBR = {
            r'\bc\.?p\.?p\.?\b': 'Codice di procedura penale',  # cpp first
            r'\bc\.?p\.?a\.?\b': 'Codice del processo amministrativo',  # cpa first
            r'\bc\.?p\.?\b': 'Codice Penale',  # cp last
            r'\bc\.?c\.?\b': 'Codice Civile',
        }
        for pattern, name_fragment in _CODE_ABBR.items():
            if re.search(pattern, query, re.IGNORECASE):
                matches = [
                    doc for doc in _fetch_doc_names(driver, database)
                    if name_fragment.lower() in doc.get('name', '').lower()
                ]
                if matches:
                    # Prefer documents whose name starts with the fragment (exact code, not commentary)
                    primary_match = next(
                        (d for d in matches if d.get('name', '').lower().startswith(name_fragment.lower())),
                        matches[0]
                    )
                    logger.info(
                        '_dynamic_law_hint: abbr match — %r for query %r',
                        primary_match['name'], query[:60]
                    )
                    return primary_match['id']

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
            if d.get("name")
            and not re.search(r'\.(pdf|docx|xlsx|txt)$', d['name'], re.IGNORECASE)
            and not d.get("name", "").startswith("[CCNL]")  # exclude CCNL docs from intent classification
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
    _suffix_re = re.compile(r'\b(bis|ter|quater|quinquies)\b', re.IGNORECASE)
    for pat in _ARTICLE_PATTERNS:
        for m in pat.finditer(query):
            number = m.group(1)
            full_match = m.group(0).strip()
            # Check if bis/ter/quater/quinquies follows the number in the full match
            suffix_match = _suffix_re.search(full_match[len(number):])
            if suffix_match:
                number = f"{number}-{suffix_match.group(1).lower()}"
            if number not in seen:
                seen.add(number)
                results.append((number, full_match))
    return results


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


# Whitelist: meta-queries about the conversation itself
_CONVERSATION_META_PATTERNS = re.compile(
    r'\b(riassumi|riassumere|abbiamo discusso|hai detto|ho chiesto|'
    r'cosa abbiamo|di cosa (si tratta|abbiamo)|recap|riepilog\w*|'
    r'precedente|prima hai|hai menzionato|torna (su|a)|'
    r'summarize|summary|what did we|we discussed)\b',
    re.IGNORECASE
)


def _is_legal_query(query: str, lang: str) -> bool:
    """Return True if the query is legal/professional; False if off-topic. Fails safe (True)."""
    # Always allow meta-queries about the conversation
    if _CONVERSATION_META_PATTERNS.search(query):
        return True
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
