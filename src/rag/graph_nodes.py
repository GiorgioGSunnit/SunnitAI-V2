"""LangGraph node functions for the RAG agent pipeline.

Thin facade: the implementations live in the ``nodes`` subpackage
(retrieval, cypher, synthesis, routing) and are re-exported here.
"""

from .nodes.cypher import (
    evaluate_retrieval_quality,
    execute_cypher,
    generate_cypher_context_only,
    generate_cypher_fallback,
    generate_cypher_intersection,
    generate_cypher_reformulation,
)
from .nodes.retrieval import (
    _resolve_by_name,
    article_router,
    comparison_retrieval,
    context_retrieval,
    decompose_query,
    dottrina_search,
    entity_linking,
)
from .nodes.routing import (
    route_after_article_router,
    route_after_decompose,
    route_after_evaluation,
    route_after_execution,
    route_after_intersection,
)
from .nodes.synthesis import (
    generate_clarifying_question,
    rerank_from_clarification,
    synthesize_answer,
)

__all__ = [
    "_resolve_by_name",
    "article_router",
    "comparison_retrieval",
    "context_retrieval",
    "decompose_query",
    "dottrina_search",
    "entity_linking",
    "evaluate_retrieval_quality",
    "execute_cypher",
    "generate_clarifying_question",
    "generate_cypher_context_only",
    "generate_cypher_fallback",
    "generate_cypher_intersection",
    "generate_cypher_reformulation",
    "rerank_from_clarification",
    "route_after_article_router",
    "route_after_decompose",
    "route_after_evaluation",
    "route_after_execution",
    "route_after_intersection",
    "synthesize_answer",
]
