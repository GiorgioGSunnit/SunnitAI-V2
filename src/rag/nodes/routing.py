"""LangGraph conditional-edge routing functions for the RAG agent pipeline."""

from typing import Any, Dict


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
