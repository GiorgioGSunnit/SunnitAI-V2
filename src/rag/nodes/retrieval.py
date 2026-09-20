"""Query decomposition and retrieval nodes for the RAG agent pipeline."""

import itertools
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Set

from langchain_core.messages import HumanMessage, SystemMessage
from neo4j.exceptions import Neo4jError

from ...preprocessing.schema.schema import (
    entities as schema_entities,
    relations as schema_relations,
)
from ..ai_chat import _call_chat, structured_entities_model
from ..answer_processing import _visibility_filter
from ..cypher_logger import log_cypher_event, log_cypher_multiline
from ..doc_lookup import (
    _ARTICLE_PATTERNS,
    _classify_query_intent,
    _dynamic_law_hint,
    _extract_article_references,
    _extract_document_references,
    _fetch_doc_names,
    _is_legal_query,
)
from ..formatting import _enrich_with_source_metadata, _node_to_dict, _session_lang
from ..language import language_display_name
from ..lookup_indexes import CONTEXT_NODE_LIMIT, CONTEXT_VECTOR_INDEXES, FULLTEXT_INDEXES
from ..lookups import (
    LABEL_VECTOR_HINTS,
    ParsedLegalAct,
    VECTOR_INDEX_SETTINGS,
    bm25_lookup,
    btree_lookup,
    fulltext_lookup,
    legal_act_lookup,
    vector_lookup,
)
from ..prompts import legal_consultant_system_prefix
from ..reranker import rerank_results
from ..utils import _build_schema_text, canonical_name
from ..verbose_logger import vlog

logger = logging.getLogger(__name__)


SCHEMA_TEXT = _build_schema_text()  # full schema — used only for entity extraction


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
            from ..lookup_indexes import BTREE_LOOKUPS
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

    # Special-document sections are excluded from the main retrieval — they're
    # only surfaced via the dedicated special_dottrina search below.
    raw_result = [
        row for row in raw_result
        if row.get("d", {}).get("document_type") != "special"
    ]

    # Parallel special-source (dottrina) search: surfaces sections from
    # document_type='special' documents alongside the primary/secondary
    # retrieval, so synthesize_answer can build a doctrinal addendum.
    # Comparison mode has its own retrieval shape — skip there.
    special_dottrina_rows: List[Dict[str, Any]] = []
    if not state.get("is_comparison"):
        special_query = search_texts[0] if search_texts else generalized
        special_keywords = state.get("retrieval_keywords") or []
        with driver.session(database=database) as _special_session:
            existing_ids = {(r.get("s") or {}).get("id") for r in raw_result}

            # 1. BM25 keyword search on section_fulltext — scoped to special books
            bm25_rows: List[Dict[str, Any]] = []
            if special_keywords:
                bm25_query = " ".join(special_keywords[:6])
                try:
                    bm25_results = _special_session.run(
                        "CALL db.index.fulltext.queryNodes('section_fulltext', $q) "
                        "YIELD node, score "
                        "MATCH (d:Document)-[:CONTAINS]->(node) "
                        "WHERE d.document_type = 'special' "
                        "RETURN d, node AS s, score "
                        "ORDER BY score DESC LIMIT 4",
                        q=bm25_query,
                    ).data()
                    for row in bm25_results:
                        row["_source"] = "special_dottrina_bm25"
                        row["_reranker_score"] = None
                        bm25_rows.append(row)
                except Exception as e:
                    logger.warning("dottrina BM25 search failed: %s", e)

            # 2. Vector search as fallback/supplement
            vector_rows: List[Dict[str, Any]] = []
            commentato_matches = vector_lookup(
                _special_session, special_query,
                indexes=["commentato_section_embeddings"], k=3,
                source_prefix="special_dottrina",
            )
            fiscalita_matches = vector_lookup(
                _special_session, special_query,
                indexes=["fiscalita_section_embeddings"], k=3,
                source_prefix="special_dottrina",
            )
            special_matches = commentato_matches + fiscalita_matches
            special_eids = [m["element_id"] for m in special_matches]
            if special_eids:
                vector_results = _special_session.run(
                    "MATCH (d:Document)-[:CONTAINS]->(s:Section) "
                    "WHERE elementId(s) IN $eids "
                    "RETURN d, s",
                    eids=special_eids,
                ).data()
                for row in vector_results:
                    row["_source"] = "special_dottrina"
                    row["_reranker_score"] = None
                    vector_rows.append(row)

            # 3. Merge — BM25 first, then vector, dedup by section id (and against raw_result)
            seen_ids: set = set()
            for row in bm25_rows + vector_rows:
                s_id = (row.get("s") or {}).get("id", "")
                if not s_id or s_id in existing_ids or s_id in seen_ids:
                    continue
                seen_ids.add(s_id)
                special_dottrina_rows.append(row)
                if len(special_dottrina_rows) >= 5:
                    break
        vlog("special_dottrina_search", {"results_found": len(special_dottrina_rows),
                                          "bm25_count": len(bm25_rows),
                                          "vector_count": len(vector_rows)})

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
        "special_dottrina_rows": special_dottrina_rows,
        "law_hint_doc_id": vector_doc_hint or "",
    }


def dottrina_search(state: Dict[str, Any], driver, database: str) -> Dict[str, Any]:
    """Dedicated dottrina search node — runs on all paths to synthesize_answer.
    Searches commentato and fiscalita indexes and populates special_dottrina_rows
    without touching raw_result or any other state keys.
    """
    if state.get("is_comparison"):
        return {}
    if state.get("special_dottrina_rows"):
        return {}  # already populated by context_retrieval

    query = state.get("query", "") or state.get("generalized_query", "")
    keywords = state.get("retrieval_keywords") or []
    special_dottrina_rows: List[Dict[str, Any]] = []

    with driver.session(database=database) as session:
        # 1. BM25 keyword search on section_fulltext — scoped to special books
        bm25_rows: List[Dict[str, Any]] = []
        if keywords:
            bm25_query = " ".join(keywords[:6])
            try:
                bm25_results = session.run(
                    "CALL db.index.fulltext.queryNodes('section_fulltext', $q) "
                    "YIELD node, score "
                    "MATCH (d:Document)-[:CONTAINS]->(node) "
                    "WHERE d.document_type = 'special' "
                    "RETURN d, node AS s, score "
                    "ORDER BY score DESC LIMIT 4",
                    q=bm25_query,
                ).data()
                for row in bm25_results:
                    row["_source"] = "special_dottrina_bm25"
                    row["_reranker_score"] = None
                    bm25_rows.append(row)
            except Exception as e:
                logger.warning("dottrina BM25 search failed: %s", e)

        # 2. Vector search as fallback/supplement
        vector_rows: List[Dict[str, Any]] = []
        commentato_matches = vector_lookup(
            session, query,
            indexes=["commentato_section_embeddings"], k=3,
            source_prefix="special_dottrina",
        )
        fiscalita_matches = vector_lookup(
            session, query,
            indexes=["fiscalita_section_embeddings"], k=3,
            source_prefix="special_dottrina",
        )
        special_matches = commentato_matches + fiscalita_matches
        special_eids = [m["element_id"] for m in special_matches]
        if special_eids:
            vector_results = session.run(
                "MATCH (d:Document)-[:CONTAINS]->(s:Section) "
                "WHERE elementId(s) IN $eids "
                "RETURN d, s",
                eids=special_eids,
            ).data()
            for row in vector_results:
                row["_source"] = "special_dottrina"
                row["_reranker_score"] = None
                vector_rows.append(row)

        # 3. Merge — BM25 first, then vector, dedup by section id
        seen_ids: set = set()
        for row in bm25_rows + vector_rows:
            s_id = (row.get("s") or {}).get("id", "")
            if s_id and s_id not in seen_ids:
                seen_ids.add(s_id)
                special_dottrina_rows.append(row)
            if len(special_dottrina_rows) >= 5:
                break

    vlog("dottrina_search", {"results_found": len(special_dottrina_rows),
                              "bm25_count": len(bm25_rows),
                              "vector_count": len(vector_rows)})
    return {"special_dottrina_rows": special_dottrina_rows}


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
