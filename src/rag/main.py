"""RAG agent pipeline — thin orchestrator over modular components.

Modules:
  - lookups.py      : Neo4j lookup strategies (B-tree, full-text, vector)
  - graph_nodes.py  : LangGraph node functions (decompose, link, generate, execute, synthesize)
  - utils.py        : Cypher cleaning, canonical naming, schema text
  - ai_chat.py      : LLM / embedding model initialization
  - models.py       : Pydantic data models
  - lookup_indexes.py : Index configuration constants
"""

import atexit
import asyncio
import json
import logging
import os
from functools import partial
from typing import Any, Dict, List, Optional, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from neo4j import Driver, GraphDatabase

from .calculation import (
    calculation_gate,
    calculation_node,
    route_after_calculation,
    route_after_gate,
)
from .graph_nodes import (
    article_router,
    comparison_retrieval,
    context_retrieval,
    decompose_query,
    entity_linking,
    evaluate_retrieval_quality,
    execute_cypher,
    generate_clarifying_question,
    generate_cypher_context_only,
    generate_cypher_fallback,
    generate_cypher_intersection,
    generate_cypher_reformulation,
    rerank_from_clarification,
    route_after_article_router,
    route_after_decompose,
    route_after_evaluation,
    route_after_execution,
    route_after_intersection,
    synthesize_answer,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# State type
# ---------------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    awaiting_clarification: Optional[bool]
    pending_sections: List[Dict[str, Any]]
    pending_calculation: Optional[Dict[str, Any]]
    calculation_match: Optional[Dict[str, Any]]
    calc_route: Optional[str]
    calculation_result: Optional[Dict[str, Any]]
    is_clarification_rerank: bool
    turn_count: int
    query: str
    raw_query: Optional[str]
    session_language: str
    generalized_query: str
    retrieval_keywords: List[str]
    document_references: List[str]
    entities: List[str]
    entry_nodes: List[Dict[str, Any]]
    cypher_query: Optional[str]
    cypher_generation_error: Optional[str]
    raw_result: List[Dict[str, Any]]
    execution_error: Optional[str]
    answer: str
    references: List[Any]
    cypher_attempt: str
    extracted_relationships: List[Dict[str, Any]]
    node_id_map: Dict[str, str]
    context_nodes: List[Dict[str, Any]]
    retrieval_quality_ok: Optional[bool]
    quality_reformulation_round: int
    quality_feedback: Optional[str]
    status_messages: List[str]
    neo4j_executed: Optional[bool]
    retrieval_evaluated: Optional[bool]
    retrieval_fallback: Optional[bool]
    citations: List[Dict[str, Any]]
    off_topic: Optional[bool]
    query_variants: List[str]
    article_router_fired: Optional[bool]
    article_refs_found: List[str]
    keyword_article_refs: List[tuple]
    bm25_doc_ids: List[str]
    bm25_from_article_lookup: Optional[bool]
    law_hint_doc_id: Optional[str]
    query_intent: Optional[str]
    law_hint_doc_id_b: Optional[str]
    intent_entity_a: Optional[str]
    intent_entity_b: Optional[str]
    comparison_doc_ids: List[str]
    is_comparison: bool
    user_id: Optional[str]
    tenant_id: Optional[str]
    tone: int
    standing: int
    response_length: int


# ---------------------------------------------------------------------------
# Neo4j driver (module-level singleton)
# ---------------------------------------------------------------------------

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

driver: Driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
atexit.register(driver.close)


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(compile_graph: bool = True):
    """Wire the RAG state machine and optionally compile it."""
    graph = StateGraph(AgentState)

    # Nodes that need the Neo4j driver are wrapped with partial
    graph.add_node("decompose_query", partial(decompose_query, driver=driver, database=NEO4J_DATABASE))
    graph.add_node(
        "article_router",
        partial(article_router, driver=driver, database=NEO4J_DATABASE),
    )
    graph.add_node(
        "entity_linking",
        partial(entity_linking, driver=driver, database=NEO4J_DATABASE),
    )
    graph.add_node(
        "context_retrieval",
        partial(context_retrieval, driver=driver, database=NEO4J_DATABASE),
    )
    graph.add_node(
        "generate_cypher_intersection",
        partial(generate_cypher_intersection, driver=driver, database=NEO4J_DATABASE),
    )
    graph.add_node("generate_cypher_context_only", generate_cypher_context_only)
    graph.add_node("generate_cypher_fallback", generate_cypher_fallback)
    graph.add_node(
        "execute_cypher",
        partial(execute_cypher, driver=driver, database=NEO4J_DATABASE),
    )
    graph.add_node(
        "evaluate_retrieval_quality",
        partial(evaluate_retrieval_quality, driver=driver, database=NEO4J_DATABASE),
    )
    graph.add_node("generate_cypher_reformulation", generate_cypher_reformulation)
    graph.add_node(
        "comparison_retrieval",
        partial(comparison_retrieval, driver=driver, database=NEO4J_DATABASE),
    )
    graph.add_node("synthesize_answer", synthesize_answer)
    graph.add_node("generate_clarifying_question", generate_clarifying_question)
    graph.add_node("rerank_from_clarification", rerank_from_clarification)
    graph.add_node("calculation_gate", calculation_gate)
    graph.add_node("calculation_node", calculation_node)

    # Edges
    def route_entry(state):
        try:
            if state.get("pending_calculation"):
                return "calculation_node"
            if state.get("awaiting_clarification"):
                return "rerank_from_clarification"
        except Exception:
            logger.exception("Graph entry router failed; using the fail-safe gate")
        return "calculation_gate"

    graph.set_conditional_entry_point(
        route_entry,
        {
            "calculation_node": "calculation_node",
            "rerank_from_clarification": "rerank_from_clarification",
            "calculation_gate": "calculation_gate",
        },
    )
    graph.add_conditional_edges(
        "calculation_gate",
        route_after_gate,
        {"calculate": "calculation_node", "normal": "decompose_query"},
    )
    graph.add_conditional_edges(
        "calculation_node",
        route_after_calculation,
        {"fallback": "decompose_query", "end": END},
    )
    graph.add_conditional_edges(
        "decompose_query",
        route_after_decompose,
        {"legal": "article_router", "off_topic": END, "comparison": "comparison_retrieval"},
    )
    graph.add_conditional_edges(
        "article_router",
        route_after_article_router,
        {"fired": "evaluate_retrieval_quality", "pass": "entity_linking"},
    )
    graph.add_edge("entity_linking", "context_retrieval")
    graph.add_edge("context_retrieval", "generate_cypher_intersection")
    graph.add_conditional_edges(
        "generate_cypher_intersection",
        route_after_intersection,
        {
            "run": "execute_cypher",
            "fallback": "generate_cypher_fallback",
            "abort": "synthesize_answer",
            "context_explore": "generate_cypher_context_only",
        },
    )
    graph.add_edge("generate_cypher_context_only", "execute_cypher")
    graph.add_edge("generate_cypher_fallback", "execute_cypher")
    graph.add_conditional_edges(
        "execute_cypher",
        route_after_execution,
        {
            "answer": "synthesize_answer",
            "retry": "generate_cypher_fallback",
            "evaluate": "evaluate_retrieval_quality",
        },
    )
    graph.add_conditional_edges(
        "evaluate_retrieval_quality",
        route_after_evaluation,
        {
            "synthesize": "synthesize_answer",
            "reformulate": "generate_cypher_reformulation",
        },
    )
    graph.add_edge("generate_cypher_reformulation", "execute_cypher")
    def route_after_comparison(state):
        return "synthesize"

    graph.add_conditional_edges(
        "comparison_retrieval",
        route_after_comparison,
        {"synthesize": "synthesize_answer"},
    )
    graph.add_edge("synthesize_answer", "generate_clarifying_question")

    def route_after_clarifying_question(state):
        return "awaiting" if state.get("awaiting_clarification") else "done"

    graph.add_conditional_edges(
        "generate_clarifying_question",
        route_after_clarifying_question,
        {"awaiting": END, "done": END},
    )
    graph.add_edge("rerank_from_clarification", "synthesize_answer")

    return graph.compile() if compile_graph else graph


# ---------------------------------------------------------------------------
# Compiled graph singleton (built once, reused across all queries)
# ---------------------------------------------------------------------------

_compiled_graph = None


def _get_compiled_graph():
    """Return the compiled LangGraph, building it once on first call."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph(compile_graph=True)
    return _compiled_graph


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(query: str, session_language: str = "it",
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        tone: int = 2,
        standing: int = 2,
        response_length: int = 2,
        awaiting_clarification: bool = False,
        pending_sections: Optional[List[Dict[str, Any]]] = None,
        pending_calculation: Optional[Dict[str, Any]] = None,
        raw_query: Optional[str] = None) -> Dict[str, Any]:
    """Run a single query through the agent graph."""
    compiled = _get_compiled_graph()
    initial_state: AgentState = {
        "query": query,
        "raw_query": raw_query,
        "session_language": session_language or "it",
        "quality_reformulation_round": 0,
        "status_messages": [],
        "turn_count": 0,
        "bm25_doc_ids": [],
        "raw_result": [],
        "is_comparison": False,
        "comparison_doc_ids": [],
        "off_topic": False,
        "article_router_fired": False,
        "neo4j_executed": False,
        "retrieval_quality_ok": False,
        "cypher_query": None,
        "execution_error": None,
        "cypher_generation_error": None,
        "retrieval_evaluated": False,
        "cypher_attempt": None,
        "law_hint_doc_id": None,
        "user_id": user_id,
        "tenant_id": tenant_id,
        "tone": tone,
        "standing": standing,
        "response_length": response_length,
        "awaiting_clarification": awaiting_clarification,
        "pending_sections": pending_sections or [],
        "pending_calculation": pending_calculation,
    }
    return compiled.invoke(initial_state)


async def run_async(query: str, session_language: str = "it") -> Dict[str, Any]:
    """Async wrapper — runs the synchronous graph in an executor."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, run, query, session_language)


async def run_batch(
    queries: List[str], max_concurrent: int = 5
) -> List[Dict[str, Any]]:
    """Process multiple queries concurrently."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def bounded_run(query: str, index: int) -> Dict[str, Any]:
        async with semaphore:
            logger.info(f"[{index + 1}/{len(queries)}] Processing: {query[:50]}...")
            try:
                result = await run_async(query)
                return {"query": query, "success": True, **result}
            except Exception as e:
                logger.error(f"[{index + 1}/{len(queries)}] Failed: {e}")
                return {
                    "query": query,
                    "success": False,
                    "error": str(e),
                    "answer": f"Error processing query: {e}",
                }

    results = await asyncio.gather(
        *[bounded_run(query, i) for i, query in enumerate(queries)],
        return_exceptions=False,
    )
    return results


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Agentic RAG pipeline for the SunnitAI graph"
    )
    parser.add_argument("query", type=str, nargs="?", help="User question")
    parser.add_argument("--batch", type=str, help="JSON file with multiple queries")
    parser.add_argument("--max-concurrent", type=int, default=5)
    parser.add_argument("--output", type=str, help="Output file for batch results")
    args = parser.parse_args()

    if args.batch:
        with open(args.batch, "r", encoding="utf-8") as f:
            batch_data = json.load(f)
        if isinstance(batch_data, dict):
            queries = batch_data.get("questions", batch_data.get("queries", []))
        else:
            queries = batch_data

        results = asyncio.run(run_batch(queries, max_concurrent=args.max_concurrent))
        output = {
            "total_queries": len(queries),
            "successful": sum(1 for r in results if r.get("success")),
            "failed": sum(1 for r in results if not r.get("success")),
            "results": results,
        }
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)
        else:
            print(json.dumps(output, ensure_ascii=False, indent=2))
        return

    if not args.query:
        parser.error("Either provide a query or use --batch with a JSON file")

    result = run(args.query)
    output = {
        "query": args.query,
        "answer": result.get("answer", "No answer generated."),
        "references": result.get("references", []),
        "cypher_query": result.get("cypher_query"),
        "entry_nodes": result.get("entry_nodes", []),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
