"""LLM and embedding model initialization.

Supports:
  - Chat: OpenAI, or any OpenAI-compatible endpoint (vLLM, Ollama, RunPod)
  - Embeddings: OpenAI, or local HuggingFace models (no API key needed)

Configuration via environment variables (see .env.example).
"""

import logging
import re
import os
from typing import List, Optional, Union

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .models import DocumentEntities

load_dotenv()

from . import embedding_config  # noqa: E402
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chat model
# ---------------------------------------------------------------------------

from langchain_openai import ChatOpenAI

_chat_kwargs = {
    "model": os.getenv("LLM_MODEL", os.getenv("OPENAI_MODEL", "nemotron-2-30B-A3B")),
    "temperature": 0,
    "api_key": os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY")),
    "request_timeout": 240,
    "max_tokens": 1500,
}

_llm_base_url = os.getenv("LLM_BASE_URL")
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
        **embedding_config.openai_embedding_kwargs(),
    }
    if _embed_base_url:
        kwargs["base_url"] = _embed_base_url
    logger.info("Using OpenAI embeddings: model=%s, query mode=%s", kwargs["model"], embedding_config.QUERY_MODE)
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


def _embed_query_with_prefix(text: str) -> list:
    """Embed a search query: in text mode, with Qwen's query instruction."""
    if embedding_config.text_mode():
        return embedding_model.embed_query(embedding_config.QUERY_INSTRUCTION + text)
    return embedding_model.embed_query(text)

# ---------------------------------------------------------------------------
# Structured output model (entity extraction)
# ---------------------------------------------------------------------------

structured_entities_model = chat_model.bind(max_tokens=2500).with_structured_output(
    DocumentEntities, method="json_mode"
)


_CONTEXT_LIMIT_RE = re.compile(
    r"maximum context length is (\d+) tokens.*?prompt contains at least (\d+) input tokens",
    re.IGNORECASE | re.DOTALL,
)
_MIN_ANSWER_TOKENS = 400
# The server reports the prompt as "at least N" tokens, a lower bound. Prod,
# 19 Oct (held-out h9): the first error counted 15,018 prompt tokens, the retry
# was sized on "at least 15,001" with a 16-token margin, asked for one token
# more than fit, and failed again.
_RETRY_MARGIN_TOKENS = 300


def _call_chat(
    messages: List[Union[SystemMessage, HumanMessage]],
    max_tokens: Optional[int] = None,
    stop: Optional[List[str]] = None,
    temperature: Optional[float] = None,
) -> str:
    """Call the chat model and trim whitespace from the response.

    A long prompt plus a fixed answer budget can exceed the serving context: a
    25k-character template asked for 4000 output tokens against a 10000-token
    limit failed by a single token, and the user got a 500. Rather than fail,
    ask for a shorter answer that fits.

    temperature: chat_model is already constructed with temperature=0 (see
    _chat_kwargs above), so every call is already deterministic by default
    without passing this. Pass it explicitly only where a caller wants that
    guaranteed regardless of chat_model's own construction ever changing —
    not to lower it further, since 0 is already the floor.
    """
    bind_kwargs = {}
    if max_tokens is not None:
        bind_kwargs["max_tokens"] = max_tokens
    if stop is not None:
        bind_kwargs["stop"] = stop
    if temperature is not None:
        bind_kwargs["temperature"] = temperature
    model = chat_model.bind(**bind_kwargs) if bind_kwargs else chat_model
    try:
        response = model.invoke(messages)
    except Exception as exc:
        m = _CONTEXT_LIMIT_RE.search(str(exc))
        if not m:
            raise
        limit, prompt_tokens = int(m.group(1)), int(m.group(2))
        room = limit - prompt_tokens - _RETRY_MARGIN_TOKENS
        if room < _MIN_ANSWER_TOKENS:
            logger.error(
                "prompt of %d tokens leaves only %d of the %d-token context for the "
                "answer: too little to retry", prompt_tokens, room, limit,
            )
            raise
        logger.warning(
            "prompt of %d tokens against a %d-token context: retrying with "
            "max_tokens=%d instead of %s", prompt_tokens, limit, room, max_tokens,
        )
        bind_kwargs["max_tokens"] = room
        response = chat_model.bind(**bind_kwargs).invoke(messages)
    return response.content.strip()


def parse_json_reply(raw: Optional[str]):
    """The first JSON array or object in a model reply, ignoring code fences
    and any text before or after it. A strict json.loads failed on production
    (16 Oct, "Extra data"): the model wrote the list asked for and then more
    text, and the strategy pipeline lost its article choice. Raises ValueError
    when the reply holds no JSON."""
    import json
    text = re.sub(r"```(?:json)?\s*", "", raw or "").strip().rstrip("`").strip()
    try:
        return json.loads(text)
    except ValueError:
        pass
    decoder = json.JSONDecoder()
    for i, char in enumerate(text):
        if char in "[{":
            try:
                return decoder.raw_decode(text, i)[0]
            except ValueError:
                continue
    raise ValueError("no JSON value in the model's reply")


def _call_chat_with_tools(
    messages: List[Union[SystemMessage, HumanMessage]],
    tools: List[dict],
    tool_choice=None,
    max_tokens: Optional[int] = None,
) -> AIMessage:
    """Call the chat model with tools and return the full AI message."""
    try:
        model = chat_model.bind_tools(
            tools,
            tool_choice=tool_choice,
            parallel_tool_calls=False,
        )
    except TypeError:
        # Compatibility with older langchain-openai versions that do not expose
        # the parallel_tool_calls keyword on bind_tools.
        model = chat_model.bind_tools(tools, tool_choice=tool_choice)
    if max_tokens is not None:
        model = model.bind(max_tokens=max_tokens)
    return model.invoke(messages)
