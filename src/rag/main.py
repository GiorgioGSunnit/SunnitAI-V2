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

from .graph_nodes import (
    context_retrieval,
    decompose_query,
    entity_linking,
    evaluate_retrieval_quality,
    execute_cypher,
    generate_cypher_context_only,
    generate_cypher_fallback,
    generate_cypher_intersection,
    generate_cypher_reformulation,
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
    query: str
    session_language: str
    generalized_query: str
    retrieval_keywords: List[str]
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
    graph.add_node("decompose_query", decompose_query)
    graph.add_node(
        "entity_linking",
        partial(entity_linking, driver=driver, database=NEO4J_DATABASE),
    )
    graph.add_node(
        "context_retrieval",
        partial(context_retrieval, driver=driver, database=NEO4J_DATABASE),
    )
    graph.add_node("generate_cypher_intersection", generate_cypher_intersection)
    graph.add_node("generate_cypher_context_only", generate_cypher_context_only)
    graph.add_node("generate_cypher_fallback", generate_cypher_fallback)
    graph.add_node(
        "execute_cypher",
        partial(execute_cypher, driver=driver, database=NEO4J_DATABASE),
    )
    graph.add_node(
        "evaluate_retrieval_quality",
        evaluate_retrieval_quality,
    )
    graph.add_node("generate_cypher_reformulation", generate_cypher_reformulation)
    graph.add_node("synthesize_answer", synthesize_answer)

    # Edges
    graph.set_entry_point("decompose_query")
    graph.add_edge("decompose_query", "entity_linking")
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
    graph.add_edge("synthesize_answer", END)

    return graph.compile() if compile_graph else graph


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(query: str, session_language: str = "it") -> Dict[str, Any]:
    """Run a single query through the agent graph."""
    compiled = build_graph()
    initial_state: AgentState = {
        "query": query,
        "session_language": session_language or "it",
        "quality_reformulation_round": 0,
        "status_messages": [],
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
