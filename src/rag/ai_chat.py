"""LLM and embedding model initialization.

Supports:
  - Chat: OpenAI, or any OpenAI-compatible endpoint (vLLM, Ollama, RunPod)
  - Embeddings: OpenAI, or local HuggingFace models (no API key needed)

Configuration via environment variables (see .env.example).
"""

import logging
import os
from typing import List, Union

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage

from .models import DocumentEntities

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chat model
# ---------------------------------------------------------------------------

from langchain_openai import ChatOpenAI

_chat_kwargs = {
    "model": os.getenv("LLM_MODEL", os.getenv("OPENAI_MODEL", "nemotron-2-30B-A3B")),
    "temperature": 0,
    "api_key": os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY")),
}

_llm_base_url = os.getenv(
    "LLM_BASE_URL", "https://m3vke16xgzhstu-8000.proxy.runpod.net/v1"
)
if _llm_base_url:
    _chat_kwargs["base_url"] = _llm_base_url

chat_model = ChatOpenAI(**_chat_kwargs)

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

structured_entities_model = chat_model.with_structured_output(
    DocumentEntities, method="json_mode"
)


def _call_chat(messages: List[Union[SystemMessage, HumanMessage]]) -> str:
    """Call the chat model and trim whitespace from the response."""
    response = chat_model.invoke(messages)
    return response.content.strip()
