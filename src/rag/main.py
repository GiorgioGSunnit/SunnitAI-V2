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
import time
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
from .cypher_logger import log_cypher_event, log_cypher_multiline
from .lookup_indexes import CONTEXT_VECTOR_INDEXES, ENTITY_VECTOR_INDEXES

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
    cypher_params: Dict[str, Any]
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
NEO4J_CONNECTION_TIMEOUT_SECONDS = float(
    os.getenv("NEO4J_CONNECTION_TIMEOUT_SECONDS", "20")
)
NEO4J_MAX_CONNECTION_LIFETIME_SECONDS = int(
    os.getenv("NEO4J_MAX_CONNECTION_LIFETIME_SECONDS", "1800")
)

driver: Driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD),
    connection_timeout=NEO4J_CONNECTION_TIMEOUT_SECONDS,
    max_connection_lifetime=NEO4J_MAX_CONNECTION_LIFETIME_SECONDS,
)
atexit.register(driver.close)

_VECTOR_INDEX_LABELS: Dict[str, str] = {
    "document_embeddings": "Document",
    "legalact_embeddings": "LegalAct",
    "article_embeddings": "Article",
    "clause_embeddings": "Clause",
    "section_embeddings": "Section",
    "institution_embeddings": "Institution",
    "person_embeddings": "Person",
    "company_embeddings": "Company",
    "court_embeddings": "Court",
    "courtcase_embeddings": "CourtCase",
    "legalparty_embeddings": "LegalParty",
    "tender_embeddings": "Tender",
    "award_embeddings": "Award",
    "contract_embeddings": "Contract",
    "changeorder_embeddings": "ChangeOrder",
    "auction_embeddings": "Auction",
    "penalty_embeddings": "Penalty",
}


def _ensure_runtime_vector_indexes() -> None:
    """Ensure vector indexes required by RAG lookups exist in Neo4j."""
    dims = int(os.getenv("EMBEDDING_DIMENSIONS", "1024"))
    required = sorted(set(ENTITY_VECTOR_INDEXES + CONTEXT_VECTOR_INDEXES))
    with driver.session(database=NEO4J_DATABASE) as session:
        for index_name in required:
            label = _VECTOR_INDEX_LABELS.get(index_name)
            if not label:
                logger.warning(
                    "No label mapping configured for vector index %s", index_name
                )
                continue
            query = (
                f"CREATE VECTOR INDEX {index_name} IF NOT EXISTS "
                f"FOR (n:{label}) ON (n.embedding) "
                f"OPTIONS {{indexConfig: {{`vector.dimensions`: {dims}, "
                f"`vector.similarity_function`: 'cosine'}}}}"
            )
            session.run(query).consume()
    logger.info("Runtime vector indexes ensured (count=%d)", len(required))


try:
    _ensure_runtime_vector_indexes()
except Exception as exc:
    # Non-fatal: API can still start, diagnostics will show missing indexes.
    logger.warning("Runtime vector index bootstrap failed: %s", exc)


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

def run(
    query: str,
    session_language: str = "it",
    trace_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a single query through the agent graph."""
    trace = trace_id or "n/a"
    t0 = time.perf_counter()

    logger.info(
        "[rag.trace:%s] start query_len=%d lang=%s",
        trace,
        len(query or ""),
        session_language or "it",
    )
    compiled = _get_compiled_graph()
    initial_state: AgentState = {
        "query": query,
        "session_language": session_language or "it",
        "quality_reformulation_round": 0,
        "status_messages": [],
    }
    try:
        result = compiled.invoke(initial_state)
        logger.info(
            "[rag.trace:%s] end ok elapsed_ms=%d rows=%d",
            trace,
            int((time.perf_counter() - t0) * 1000),
            len(result.get("raw_result") or []),
        )
        return result
    except Exception:
        logger.exception(
            "[rag.trace:%s] end error elapsed_ms=%d",
            trace,
            int((time.perf_counter() - t0) * 1000),
        )
        raise


def run_diagnostics(query: str, session_language: str = "it") -> Dict[str, Any]:
    """Run early RAG stages only and return per-step timings."""
    state: AgentState = {
        "query": query,
        "session_language": session_language or "it",
        "quality_reformulation_round": 0,
        "status_messages": [],
    }
    step_timings_ms: Dict[str, int] = {}
    trace: List[Dict[str, Any]] = []

    diag_error: Optional[str] = None
    failed_step: Optional[str] = None

    def _run_step(name: str, fn):
        t0 = time.perf_counter()
        out = fn(state)
        elapsed = int((time.perf_counter() - t0) * 1000)
        state.update(out or {})
        step_timings_ms[name] = elapsed
        trace.append(
            {
                "step": name,
                "elapsed_ms": elapsed,
                "keys_added": sorted(list((out or {}).keys())),
            }
        )

    try:
        _run_step("decompose_query", decompose_query)
        _run_step(
            "entity_linking",
            lambda s: entity_linking(s, driver=driver, database=NEO4J_DATABASE),
        )
        _run_step(
            "context_retrieval",
            lambda s: context_retrieval(s, driver=driver, database=NEO4J_DATABASE),
        )
        _run_step("generate_cypher_intersection", generate_cypher_intersection)
        route = route_after_intersection(state)
        if route == "fallback":
            _run_step("generate_cypher_fallback", generate_cypher_fallback)
        elif route == "context_explore":
            _run_step("generate_cypher_context_only", generate_cypher_context_only)
    except Exception as exc:
        failed_step = trace[-1]["step"] if trace else "decompose_query"
        diag_error = str(exc)
        route = "error"

    return {
        "query": query,
        "session_language": session_language or "it",
        "step_timings_ms": step_timings_ms,
        "trace": trace,
        "route_after_intersection": route,
        "diag_error": diag_error,
        "failed_step": failed_step,
        "entry_nodes_count": len(state.get("entry_nodes") or []),
        "context_nodes_count": len(state.get("context_nodes") or []),
        "cypher_attempt": state.get("cypher_attempt"),
        "cypher_query_preview": (state.get("cypher_query") or "")[:500],
        "cypher_generation_error": state.get("cypher_generation_error"),
        "retrieval_keywords": state.get("retrieval_keywords") or [],
        "generalized_query": state.get("generalized_query"),
    }


def run_diagnostics_full(
    query: str,
    session_language: str = "it",
    max_transitions: int = 20,
) -> Dict[str, Any]:
    """Run the full RAG flow step-by-step and return timings + routes.

    This mirrors graph routing logic while exposing per-step timings to
    identify bottlenecks/timeouts in production.
    """
    state: AgentState = {
        "query": query,
        "session_language": session_language or "it",
        "quality_reformulation_round": 0,
        "status_messages": [],
    }
    step_timings_ms: Dict[str, int] = {}
    trace: List[Dict[str, Any]] = []
    routes: List[Dict[str, Any]] = []
    failed_step: Optional[str] = None
    diag_error: Optional[str] = None
    transitions = 0
    finished = False
    log_cypher_multiline(
        "diag_start",
        "Diagnostics full run: user request",
        query or "",
        delimiter_label="USER_QUESTION",
    )

    def _run_step(name: str, fn) -> None:
        nonlocal failed_step
        t0 = time.perf_counter()
        out = fn(state)
        elapsed = int((time.perf_counter() - t0) * 1000)
        state.update(out or {})
        failed_step = name
        step_timings_ms[name] = step_timings_ms.get(name, 0) + elapsed
        trace.append(
            {
                "step": name,
                "elapsed_ms": elapsed,
                "raw_result_rows": len(state.get("raw_result") or []),
                "status_count": len(state.get("status_messages") or []),
            }
        )

    try:
        _run_step("decompose_query", decompose_query)
        log_cypher_event(
            "diag_data",
            "retrieval keywords extracted",
            detail={
                "retrieval_keywords": state.get("retrieval_keywords") or [],
                "generalized_query": state.get("generalized_query"),
            },
        )
        _run_step(
            "entity_linking",
            lambda s: entity_linking(s, driver=driver, database=NEO4J_DATABASE),
        )
        _run_step(
            "context_retrieval",
            lambda s: context_retrieval(s, driver=driver, database=NEO4J_DATABASE),
        )
        _run_step("generate_cypher_intersection", generate_cypher_intersection)

        route = route_after_intersection(state)
        routes.append({"after": "generate_cypher_intersection", "route": route})
        log_cypher_event(
            "diag_route",
            "post-intersection routing decision",
            detail={"route": route},
        )

        if route == "fallback":
            _run_step("generate_cypher_fallback", generate_cypher_fallback)
        elif route == "context_explore":
            _run_step("generate_cypher_context_only", generate_cypher_context_only)
        elif route == "abort":
            _run_step("synthesize_answer", synthesize_answer)
            finished = True

        while not finished and transitions < max_transitions:
            transitions += 1
            _run_step(
                "execute_cypher",
                lambda s: execute_cypher(s, driver=driver, database=NEO4J_DATABASE),
            )
            route_exec = route_after_execution(state)
            routes.append({"after": "execute_cypher", "route": route_exec})
            log_cypher_event(
                "diag_route",
                "post-execute routing decision",
                detail={"route": route_exec, "transition": transitions},
            )

            if route_exec == "answer":
                _run_step("synthesize_answer", synthesize_answer)
                finished = True
                break

            if route_exec == "retry":
                _run_step("generate_cypher_fallback", generate_cypher_fallback)
                continue

            if route_exec == "evaluate":
                _run_step("evaluate_retrieval_quality", evaluate_retrieval_quality)
                route_eval = route_after_evaluation(state)
                routes.append(
                    {"after": "evaluate_retrieval_quality", "route": route_eval}
                )
                log_cypher_event(
                    "diag_route",
                    "post-evaluation routing decision",
                    detail={"route": route_eval, "transition": transitions},
                )
                if route_eval == "synthesize":
                    _run_step("synthesize_answer", synthesize_answer)
                    finished = True
                    break
                if route_eval == "reformulate":
                    _run_step(
                        "generate_cypher_reformulation",
                        generate_cypher_reformulation,
                    )
                    continue

            diag_error = f"Unknown execution route: {route_exec}"
            break

        if not finished and not diag_error and transitions >= max_transitions:
            diag_error = (
                f"Max transitions reached ({max_transitions}) before pipeline completion"
            )
    except Exception as exc:
        diag_error = str(exc)

    log_cypher_event(
        "diag_end",
        "Diagnostics full run completed",
        detail={
            "finished": finished,
            "failed_step": failed_step,
            "diag_error": diag_error,
            "transitions": transitions,
        },
    )

    return {
        "query": query,
        "session_language": session_language or "it",
        "step_timings_ms": step_timings_ms,
        "trace": trace,
        "routes": routes,
        "diag_error": diag_error,
        "failed_step": failed_step,
        "transitions": transitions,
        "finished": finished,
        "entry_nodes_count": len(state.get("entry_nodes") or []),
        "context_nodes_count": len(state.get("context_nodes") or []),
        "raw_result_count": len(state.get("raw_result") or []),
        "has_answer": bool((state.get("answer") or "").strip()),
        "status_messages": state.get("status_messages") or [],
        "cypher_attempt": state.get("cypher_attempt"),
        "cypher_query_preview": (state.get("cypher_query") or "")[:800],
        "cypher_generation_error": state.get("cypher_generation_error"),
        "execution_error": state.get("execution_error"),
        "retrieval_keywords": state.get("retrieval_keywords") or [],
        "generalized_query": state.get("generalized_query"),
    }


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
