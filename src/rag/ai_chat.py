"""LLM and embedding model initialization.

Supports:
  - Chat: OpenAI, or any OpenAI-compatible endpoint (vLLM, Ollama, RunPod)
  - Embeddings: OpenAI, or local HuggingFace models (no API key needed)

Configuration via environment variables (see .env.example).
"""

import logging
import os
import random
import time
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage

from .cypher_logger import log_cypher_event
from .models import DocumentEntities, ExtractedGraph

load_dotenv()
logger = logging.getLogger(__name__)


def _optional_positive_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        v = int(raw)
        return v if v > 0 else default
    except ValueError:
        logger.warning("Invalid %s=%r — using default %s", name, raw, default)
        return default


def _optional_positive_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        v = float(raw)
        return v if v > 0 else default
    except ValueError:
        logger.warning("Invalid %s=%r — using default %s", name, raw, default)
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# Chat model
# ---------------------------------------------------------------------------

from langchain_openai import ChatOpenAI

_chat_kwargs = {
    "model": os.getenv("LLM_MODEL", os.getenv("OPENAI_MODEL", "qwen2.5-32b")),
    "temperature": 0,
    "api_key": os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY")),
    "timeout": _optional_positive_float("LLM_REQUEST_TIMEOUT_SECONDS", 60.0),
    "max_retries": _optional_positive_int("LLM_MAX_RETRIES", 1),
}

_llm_base_url = os.getenv(
    "LLM_BASE_URL", "https://m3vke16xgzhstu-8000.proxy.runpod.net/v1"
)
if _llm_base_url:
    _chat_kwargs["base_url"] = _llm_base_url

chat_model = ChatOpenAI(**_chat_kwargs)
LLM_FALLBACK_ATTEMPTS = _optional_positive_int("LLM_FALLBACK_ATTEMPTS", 3)
LLM_FALLBACK_BASE_DELAY_SECONDS = _optional_positive_float(
    "LLM_FALLBACK_BASE_DELAY_SECONDS", 0.8
)
LLM_FRESH_CLIENT_PER_CALL = _env_bool("LLM_FRESH_CLIENT_PER_CALL", False)
LLM_ENTITY_FRESH_CLIENT_PER_CALL = _env_bool("LLM_ENTITY_FRESH_CLIENT_PER_CALL", False)

# Optional cap on all chat completions (set on small context models, e.g. 16k total window).
_llm_max_out = os.getenv("LLM_MAX_OUTPUT_TOKENS", "").strip()
_llm_max_out_value = (
    _optional_positive_int("LLM_MAX_OUTPUT_TOKENS", 8192) if _llm_max_out else None
)
if _llm_max_out:
    chat_model = chat_model.bind(max_tokens=_llm_max_out_value)

# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------

_embedding_provider = os.getenv("EMBEDDING_PROVIDER", "auto").lower()
_embed_base_url = os.getenv("EMBEDDING_BASE_URL")
_embed_api_key = os.getenv(
    "EMBEDDING_API_KEY",
    os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY")),
)

def _init_embedding_model():
    """Initialize the embedding model based on config.

    Priority:
      1. EMBEDDING_PROVIDER=local  → always use HuggingFace
      2. EMBEDDING_PROVIDER=openai → always use OpenAI
      3. EMBEDDING_PROVIDER=auto (default):
         - If EMBEDDING_BASE_URL is set → use OpenAI client with custom base
         - If OPENAI_API_KEY is set → use OpenAI
         - Otherwise → use local HuggingFace model
    """
    if _embedding_provider == "local":
        return _init_local_embeddings()

    if _embedding_provider == "openai":
        return _init_openai_embeddings()

    # Auto-detect
    if _embed_base_url:
        return _init_openai_embeddings()

    if os.getenv("OPENAI_API_KEY"):
        return _init_openai_embeddings()

    return _init_local_embeddings()


def _init_openai_embeddings():
    from langchain_openai import OpenAIEmbeddings
    kwargs = {
        "model": os.getenv("EMBEDDING_MODEL", os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")),
        "api_key": _embed_api_key,
    }
    if _embed_base_url:
        kwargs["base_url"] = _embed_base_url
    logger.info("Using OpenAI embeddings: model=%s", kwargs["model"])
    return OpenAIEmbeddings(**kwargs)


def _init_local_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
    logger.info("Using local HuggingFace embeddings: model=%s", model_name)
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


embedding_model = _init_embedding_model()

# ---------------------------------------------------------------------------
# Structured output model (entity extraction)
# ---------------------------------------------------------------------------

# Entity JSON must stay small. `chat_model.bind(max_tokens=…)` is not always honored by some
# OpenAI-compatible stacks with json_mode; a dedicated ChatOpenAI with `max_tokens` set at
# construction ensures the provider receives an explicit output budget.
_entity_max = _optional_positive_int("LLM_ENTITY_EXTRACTION_MAX_TOKENS", 16384)
_entity_llm = ChatOpenAI(**{**_chat_kwargs, "max_tokens": _entity_max})
structured_entities_model = _entity_llm.with_structured_output(
    DocumentEntities, method="json_mode"
)


def _build_chat_client_for_call():
    client = ChatOpenAI(**_chat_kwargs)
    if _llm_max_out_value:
        client = client.bind(max_tokens=_llm_max_out_value)
    return client


def _build_structured_entities_model_for_call():
    entity_client = ChatOpenAI(**{**_chat_kwargs, "max_tokens": _entity_max})
    return entity_client.with_structured_output(DocumentEntities, method="json_mode")


def invoke_entity_extraction(
    messages: List[Union[SystemMessage, HumanMessage]],
) -> DocumentEntities:
    """Run structured entity extraction; degrade to an empty graph on token-limit / parse failures."""
    try:
        model = (
            _build_structured_entities_model_for_call()
            if LLM_ENTITY_FRESH_CLIENT_PER_CALL
            else structured_entities_model
        )
        return model.invoke(messages)
    except Exception as exc:
        name = type(exc).__name__
        msg_l = str(exc).lower()
        if (
            "LengthFinishReason" in name
            or "length_finish_reason" in msg_l
            or "length limit" in msg_l
        ):
            logger.warning(
                "Entity extraction hit output/token limit; using empty graph. %s: %s",
                name,
                str(exc)[:240],
            )
            return DocumentEntities(
                graph=ExtractedGraph(nodes=[], relationships=[])
            )
        raise


def _extract_status_code(exc: Exception) -> Optional[int]:
    """Best-effort extraction of HTTP status from OpenAI/httpx style exceptions."""
    for attr in ("status_code", "http_status", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    if isinstance(value, int):
        return value
    return None


def _call_chat(
    messages: List[Union[SystemMessage, HumanMessage]],
    llm_context: Optional[Dict[str, Any]] = None,
) -> str:
    """Call chat model with resilient fallback for transient upstream failures."""

    last_exc: Exception | None = None
    for attempt in range(1, LLM_FALLBACK_ATTEMPTS + 1):
        try:
            t0 = time.perf_counter()
            model = _build_chat_client_for_call() if LLM_FRESH_CLIENT_PER_CALL else chat_model
            response = model.invoke(messages)
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            content = (response.content or "").strip()
            # Some upstream gateways may return HTML error pages in content.
            if content.lstrip().lower().startswith("<!doctype html"):
                raise RuntimeError("LLM upstream gateway returned HTML error page")
            if "<html" in content[:200].lower() and "cloudflare" in content[:500].lower():
                raise RuntimeError("LLM upstream gateway returned HTML Cloudflare page")
            prompt_preview = ""
            if messages:
                try:
                    prompt_preview = str(messages[-1].content)[:220]
                except Exception:
                    prompt_preview = "(unavailable)"
            log_cypher_event(
                "llm_call",
                "LLM call completed",
                detail={
                    "attempt": attempt,
                    "elapsed_ms": elapsed_ms,
                    "model": _chat_kwargs.get("model"),
                    "base_url": _llm_base_url,
                    "prompt_preview": prompt_preview,
                    "response_len": len(content),
                    "status_code": 200,
                    "cypher_query": (llm_context or {}).get("cypher_query"),
                    "cypher_attempt": (llm_context or {}).get("cypher_attempt"),
                },
            )
            return content
        except Exception as exc:
            last_exc = exc
            msg = str(exc)
            status_code = _extract_status_code(exc)
            log_cypher_event(
                "llm_call",
                "LLM call failed",
                detail={
                    "attempt": attempt,
                    "error": msg[:300],
                    "model": _chat_kwargs.get("model"),
                    "base_url": _llm_base_url,
                    "status_code": status_code,
                    "cypher_query": (llm_context or {}).get("cypher_query"),
                    "cypher_attempt": (llm_context or {}).get("cypher_attempt"),
                },
            )
            transient = any(
                token in msg.lower()
                for token in (
                    "timeout",
                    "readtimeout",
                    "apitimeouterror",
                    "bad gateway",
                    "502",
                    "503",
                    "504",
                    "cloudflare",
                )
            )
            logger.warning(
                "LLM call failed (attempt %d/%d, transient=%s): %s",
                attempt,
                LLM_FALLBACK_ATTEMPTS,
                transient,
                msg[:300],
            )
            if (not transient) or attempt >= LLM_FALLBACK_ATTEMPTS:
                break
            # Exponential backoff with a small jitter.
            delay = LLM_FALLBACK_BASE_DELAY_SECONDS * (2 ** (attempt - 1))
            delay = delay + random.uniform(0, 0.25)
            time.sleep(delay)

    raise RuntimeError(
        "LLM service temporarily unavailable. Please retry in a few moments."
    ) from last_exc
