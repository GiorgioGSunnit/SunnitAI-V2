"""Answer synthesis and clarification nodes for the RAG agent pipeline."""

import json
import logging
import re
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ..ai_chat import _call_chat
from ..answer_processing import (
    _GAP_PHRASES,
    _extract_citations,
    _is_primary_gap_response,
    _strip_hallucinated_fonti,
    _strip_vague_closing,
)
from ..cypher_logger import log_cypher_event
from ..formatting import _session_lang, _summarize_for_synthesis
from ..prompts import (
    synthesis_empty_system,
    synthesis_error_system,
    synthesis_human_footer,
    synthesis_system_message,
)
from ..reranker import rerank_results
from ..verbose_logger import vlog

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Node F: Answer synthesis
# ---------------------------------------------------------------------------

def synthesize_answer(state: Dict[str, Any]) -> Dict[str, Any]:
    lang = _session_lang(state)
    error = state.get("execution_error") or state.get("cypher_generation_error")
    data = rerank_results(state.get("query", ""), state.get("raw_result") or [])
    logger.info("synthesize_answer: data=%d, raw_result=%d, bm25_doc_ids=%s", len(data), len(state.get('raw_result') or []), state.get('bm25_doc_ids'))
    _dottrina_only = False

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

    def _with_history(system_content: str, human_content: str):
        """Build a message list with full conversation history injected."""
        msgs = [SystemMessage(content=system_content)]
        for msg in (state.get("chat_history") or []):
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                msgs.append(HumanMessage(content=content))
            elif role == "assistant":
                msgs.append(AIMessage(content=content))
            elif role == "system":
                msgs.append(SystemMessage(content=content))
        msgs.append(HumanMessage(content=human_content))
        return msgs

    if error:
        answer = _call_chat(
            _with_history(
                synthesis_error_system(lang, tone=tone, standing=standing, length=response_length),
                (
                    "Original question: {question}\n"
                    "Internal retrieval note (do not quote literally or discuss IT systems): {error}\n\n"
                    "Provide the legal consultation as instructed."
                ).format(question=state["query"], error=error)
                + synthesis_human_footer(lang),
            )
        )
        answer = _strip_vague_closing(answer)
        return {
            "answer": answer,
            "references": state.get("raw_result", []) or [],
            "status_messages": state.get("status_messages") or [],
        }

    if not data:
        # Main corpus has nothing but dottrina might — fall through to the dottrina
        # section below by treating the special sections as the primary source.
        _fallback_special_rows = state.get("special_dottrina_rows") or []
        if _fallback_special_rows:
            data = _fallback_special_rows
            _dottrina_only = True
        else:
            answer = _call_chat(
                _with_history(
                    synthesis_empty_system(lang, tone=tone, standing=standing, length=response_length),
                    (
                        "Original question: {question}\n"
                        "The knowledge graph query returned no rows.\n\n"
                        "Provide the legal consultation as instructed."
                    ).format(question=state["query"])
                    + synthesis_human_footer(lang),
                )
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
    char_cap = 4000 if _is_cmp else 3000
    if len(serialized) > char_cap:
        serialized = json.dumps(
            _summarize_for_synthesis(data, max_records=2,
                                     is_comparison=_is_cmp),
            ensure_ascii=False,
            indent=None if _is_cmp else 2,
        )
    all_citations = _extract_citations(data)
    primary_citations = [c for c in all_citations if c.get("document_type") in (None, "primary", "ccnl")]
    secondary_citations = [c for c in all_citations if c.get("document_type") == "interpretation"]
    # Special/dottrina sources are retrieved and carried separately from the main
    # pipeline (context_retrieval's dedicated state key) — never part of `data`.
    special_dottrina_rows = state.get("special_dottrina_rows") or []
    open("/tmp/dottrina_trace.log", "a").write(f"special_dottrina_rows len={len(special_dottrina_rows)}\n")
    special_citations = _extract_citations(special_dottrina_rows)
    open("/tmp/dottrina_trace.log", "a").write(
        f"special_citations len={len(special_citations)} "
        f"first_sections={len(special_citations[0]['sections']) if special_citations else 'n/a'}\n"
    )
    # Tiered synthesis only applies to the standard (non-comparison) answer flow —
    # comparison mode builds Data from paired document rows and has its own prompt shape.
    is_tiered = bool(secondary_citations) and not state.get("is_comparison")

    def _citation_source_blocks(cites: List[Dict[str, Any]], label: str) -> List[str]:
        blocks = []
        for c in cites:
            text = "\n\n".join(
                s["plain_text"] for s in c.get("sections", []) if s.get("plain_text")
            )
            if text:
                blocks.append(f"[{label} - {c['document_name']}]\n{text}")
        return blocks

    if is_tiered:
        main_citations = primary_citations + secondary_citations
        data_blob = "\n\n".join(
            _citation_source_blocks(primary_citations, "FONTE PRIMARIA")
            + _citation_source_blocks(secondary_citations, "FONTE SECONDARIA")
        )
        if len(data_blob) > char_cap:
            data_blob = data_blob[:char_cap]
    else:
        main_citations = all_citations
        data_blob = serialized

    citation_strings = [
        f"Fonte: {c['document_name']}" if not c["sections"]
        else f"Fonte: {c['document_name']}, sezione: {c['sections'][0]['name']}" if len(c["sections"]) == 1
        else f"Fonte: {c['document_name']}, sezioni: {', '.join(s['name'] for s in c['sections'])}"
        for c in main_citations
    ]

    human_parts = [
        f"Question: {state['query']}\n",
        f"Data: {data_blob}\n",
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

    if state.get("retrieval_fallback") and not data:
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
    if is_tiered:
        system_prompt = system_prompt + (
            "\nSTRUTTURA DELLA RISPOSTA (obbligatoria quando sono presenti più tipi di fonti):\n"
            "1. FONTI PRIMARIE: inizia citando cosa stabilisce la legge, riportando il testo normativo con precisione.\n"
            "2. FONTI SECONDARIE: aggiungi come la giurisprudenza ha interpretato la norma, con attribuzione esplicita "
            "(es. 'La Corte di Cassazione ha stabilito che...', 'Secondo la sentenza n. X...').\n"
            "Se una fonte non è disponibile, ometti quella sezione senza menzionarne l'assenza. "
            "Non inventare contenuti non presenti nelle fonti."
        )
    if not _dottrina_only:
        answer = _call_chat(
            _with_history(system_prompt, "".join(human_parts) + synthesis_human_footer(lang)),
            max_tokens=600,
        )
    else:
        answer = ""
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
                or c.get('document_type') == 'ccnl'
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
    # Accepts "articolo 5", "art. 5", "sezione 5" and - the form the model
    # actually writes - "sezioni: 371-ter.0.0, 371-ter_3": plural, colon, list.
    cited_section_pattern = re.compile(
        r'\b(?:articol[oi]|art\.?|sezion[ei]|sez\.?)\s*:?\s*'
        r'(\d+(?:[.\-_]\w+)*(?:[\s\-]*(?:bis|ter|quater))?)',
        re.IGNORECASE,
    )
    answer_article_refs = {m.group(1).strip().lower().replace(' ', '') for m in cited_section_pattern.finditer(answer)}
    # Also add base article numbers (e.g. "124" from "124.0.0")
    answer_article_refs |= {ref.split('.')[0] for ref in answer_article_refs}
    # Whitespace-stripped answer, so a section name can be matched verbatim
    # regardless of how the model punctuated the citation around it.
    answer_normalised = answer.lower().replace(" ", "")

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
            # Dottrina sections (special books) — always include, never filter by article ref
            if doc_id in {c.get("document_id") for c in special_citations}:
                return True
            score = section.get("score")
            if score is None:
                # No reranker score — fall back to article reference check
                name = (section.get('name') or '').lower().replace(' ', '')
                base_name = name.split('.')[0]
                # Full name matched verbatim too: the extraction pattern cannot
                # anticipate every citation style. Full name only, never base_name,
                # or a bare "40" would match any stray number in the prose.
                return (name in answer_article_refs
                        or base_name in answer_article_refs
                        or (len(name) > 3 and name in answer_normalised))
            if score >= 0.65:
                return True
            if score >= 0.3:
                # Medium confidence — only include if article number in answer
                name = (section.get('name') or '').lower().replace(' ', '')
                base_name = name.split('.')[0]
                # Full name matched verbatim too: the extraction pattern cannot
                # anticipate every citation style. Full name only, never base_name,
                # or a bare "40" would match any stray number in the prose.
                return (name in answer_article_refs
                        or base_name in answer_article_refs
                        or (len(name) > 3 and name in answer_normalised))
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

    answer_before_gap = answer

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

    # Strip trailing question if LLM appended one despite instructions
    import re as _re
    answer = _re.sub(r'\n[A-Z][^\n]{0,500}\?[^\n]*$', '', answer.rstrip()).rstrip()

    # Built at the end of this function instead: the dottrina block below adds
    # its own sources to `citations`. Building it here listed only the main
    # citations while the frontend received the merged list, so dottrina sources
    # showed in the side panel but never in the answer.

    if _dottrina_only and not special_citations:
        special_citations = all_citations

    if (special_citations or _dottrina_only) and not state.get("is_comparison"):
        special_context = "\n\n".join(_citation_source_blocks(special_citations, "DOTTRINA"))
        open("/tmp/dottrina_trace.log", "a").write(f"special_context len={len(special_context)}\n")
        if special_context:
            dottrina_system = (
                "Sei un assistente legale italiano specializzato in dottrina giuridica. Ti vengono forniti estratti "
                "da opere dottrinali e commentari giuridici, insieme alla risposta principale già redatta. "
                "DIVIETO ASSOLUTO DI CITAZIONE VERBATIM: non riportare MAI frasi intere o brani dal testo originale, "
                "nemmeno tra virgolette. Riformula SEMPRE con parole completamente diverse, mantenendo solo il "
                "significato. Se senti la tentazione di copiare una frase, riscrivila da zero con struttura "
                "sintattica diversa. Non aggiungere contenuti non presenti "
                "negli estratti. Non omettere concetti chiave presenti negli estratti. Regole: parafrasa fedelmente "
                "preservando l'essenza e il significato originale degli autori; attribuisci sempre la fonte con "
                "naturalezza nel testo (es. 'Secondo il commentario...', 'La dottrina rileva che...'); includi "
                "riferimenti bibliografici e note a piè di pagina se presenti negli estratti; non usare virgolette "
                "o apici. Non porre domande all'utente e non chiedere chiarimenti. "
                "Usa solo il contenuto degli estratti forniti. Se un estratto non è direttamente pertinente, "
                "citalo brevemente in relazione alla domanda senza inventare contenuti aggiuntivi. "
                "VIETATO ripetere o parafrasare la risposta principale già fornita sopra. "
                "Aggiungi solo ciò che gli estratti dottrinali contengono in aggiunta. "
                "Non porre mai domande all'utente. Non chiedere mai chiarimenti o informazioni aggiuntive. "
                "Termina sempre con un punto fermo."
            )
            dottrina_human = (
                f"Domanda originale: {state['query']}\n\n"
                f"Risposta principale già fornita:\n{'' if _is_primary_gap_response(answer_before_gap.split('---')[0].strip()) else answer_before_gap}\n\n"
                f"Estratti dottrinali disponibili:\n{special_context}\n\n"
                f"IMPORTANTE: Devi sempre fornire una nota dottrinale basata sugli estratti sopra. "
                f"Non restituire mai una stringa vuota."
            )
            dottrina_answer = _call_chat(
                [SystemMessage(content=dottrina_system), HumanMessage(content=dottrina_human)],
                max_tokens=600,
                stop=["Nel contesto", "Puoi precisare", "Vuoi specificare", "Hai ulteriori"],
            )
            open("/tmp/dottrina_trace.log", "a").write(f"dottrina_answer len={len(dottrina_answer)} preview={repr(dottrina_answer[:200])}\n")
            dottrina_answer = re.sub(r'^[\s"\'“”]+', '', dottrina_answer).strip()
            dottrina_answer = re.sub(r'©[^\n]{0,100}', '', dottrina_answer)
            dottrina_answer = re.sub(r'\n[^\n]{0,600}\?[^\n]*$', '', dottrina_answer.rstrip()).rstrip()
            if dottrina_answer and dottrina_answer.strip():
                answer = (
                    answer.rstrip()
                    + "\n\n---\n**Nota dottrinale:**\n"
                    + dottrina_answer.strip()
                )
                existing_doc_ids = {c.get("document_id") for c in citations}
                citations = citations + [
                    c for c in special_citations if c.get("document_id") not in existing_doc_ids
                ]

                # If main answer is a gap response but dottrina has content, promote dottrina to main answer
                if dottrina_answer and _is_primary_gap_response(answer_before_gap.split('---')[0].strip()):
                    answer = dottrina_answer.strip()

    answer = re.sub(r'\n[A-Z][^\n]{0,500}\?[^\n]*$', '', answer.rstrip()).rstrip()

    # Dottrina sources are merged into `citations` above, so build the Fonti
    # line from the complete list. Kept above the dottrina note so the layout is
    # unchanged; if that note became the whole answer there is no marker and it
    # simply goes at the end.
    if citations:
        fonti_line = "Fonti: " + ", ".join(
            f"{c['document_name']} sezioni: {', '.join(s['name'] for s in c['sections'])}"
            for c in citations
        )
        _marker = "\n\n---\n**Nota dottrinale:**"
        if _marker in answer:
            _head, _sep, _tail = answer.partition(_marker)
            answer = _head.rstrip() + "\n\n" + fonti_line + _sep + _tail
        else:
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

    def _normalize_type(t):
        return "corpus" if t in (None, "primary", "interpretation") else t
    unique_doc_types = {_normalize_type(c.get("document_type")) for c in citations}
    if len(unique_doc_types) <= 1:
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
            "document_type": c.get("document_type"),
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
                "document_type": s.get("document_type"),
            },
            "s": {
                "id": f"DOCUMENT_SECTION::{s.get('document_id', '').replace('LEGAL_DOC::', '')}{'::'}{s.get('name', '')}",
                "name": s.get("name"),
                "title": s.get("title"),
                "plain_text": s.get("plain_text"),
                "score": s.get("score"),
            },
            "_reranker_score": s.get("score") or 0.9,  # carry the original score forward; fall back to high-confidence only if it was never set
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
