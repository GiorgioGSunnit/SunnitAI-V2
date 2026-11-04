"""LLM and embedding model initialization.

Supports:
  - Chat: OpenAI, or any OpenAI-compatible endpoint (vLLM, Ollama, RunPod)
  - Embeddings: OpenAI, or local HuggingFace models (no API key needed)

Configuration via environment variables (see .env.example).
"""

import logging
import re
import sys
import threading
import os
from typing import List, Optional, Union

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .llm_reply import parse_json_reply, strip_think  # noqa: F401  (parse_json_reply: imported from here by the pipeline)
from .models import DocumentEntities
from .verbose_logger import vlog

load_dotenv()

from . import embedding_config  # noqa: E402
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chat model
# ---------------------------------------------------------------------------

from langchain_openai import ChatOpenAI

# Qwen3 (the production model since 2 Nov) reasons before it answers, and the reasoning counts against max_tokens:
# vLLM strips it from the reply, so a section can come back cut with no reasoning in sight. Prod, 2-4 Nov: 3 writer
# sections, 1 rewrite, 1 chat answer and 4 session titles cut in 8 chat requests. LLM_THINKING=false asks the chat
# template not to reason (enable_thinking=False, as scripts/exp_ruling_cards.py does for the cards: 20/20 complete,
# other models' templates ignore it). Unset keeps the model's default: the field is vLLM's, OpenAI's API rejects it.
_NO_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}
_OFF = {"false", "0", "no", "off"}


def _thinking_kwargs(setting: Optional[str]) -> dict:
    """ChatOpenAI kwargs for the LLM_THINKING setting: the no-reasoning body when it says no, else nothing."""
    if (setting or "").strip().lower() in _OFF:
        return {"extra_body": _NO_THINKING}
    return {}


_chat_kwargs = {
    "model": os.getenv("LLM_MODEL", os.getenv("OPENAI_MODEL", "nemotron-2-30B-A3B")),
    "temperature": 0,
    "api_key": os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY")),
    "request_timeout": 240,
    "max_tokens": 1500,
    **_thinking_kwargs(os.getenv("LLM_THINKING")),
}


def _announce_thinking(setting: Optional[str]) -> Optional[str]:
    """The startup notice for a thinking switch that is off, printed as well as logged: the service runs under
    uvicorn, which leaves the root logger unconfigured, so an INFO line never reaches the journal (4 Nov: the deploy
    check counted 0). The verbose logger prints for the same reason."""
    if not _thinking_kwargs(setting):
        return None
    notice = f"LLM_THINKING={setting.strip()}: the chat model is asked not to reason before answering"
    logger.info(notice)
    print(notice, flush=True)
    return notice


_announce_thinking(os.getenv("LLM_THINKING"))

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
# The same overflow in its other wordings (prod logs, 20-23 Oct): checked on characters before
# tokenizing ("your prompt contains 65396 characters (more than 53760 characters ...)"), and an
# answer length over the whole context ("max_completion_tokens=19580 ..."). Neither was retried.
_CONTEXT_CHARS_RE = re.compile(r"maximum context length is (\d+) tokens.*?prompt contains (\d+) characters",
                               re.IGNORECASE | re.DOTALL)
_CONTEXT_SIZE_RE = re.compile(r"max_model_len\D{0,3}(\d+)|maximum context length is (\d+)", re.IGNORECASE)
# Our own prompt estimate, used when the server gives only a lower bound: Italian legal text
# runs about 3 characters per token, 2.5 errs towards a shorter answer that fits.
_RETRY_CHARS_PER_TOKEN = 2.5


def _overflow_sizes(error_text: str, messages) -> Optional[tuple]:
    """(context limit, prompt tokens) from a context-overflow error, or None if it isn't one.

    The prompt size is the larger of what the server reports and our own estimate from the
    prompt's characters: "at least N input tokens" is only a lower bound -- the server stops
    counting at the room left after the requested answer. Prod, 23 Oct (applicability check):
    a 19,140-token request was told "at least 861", the retry asked for 18,839, was told "at
    least 1,162", and failed again; the prompt was some 20,000 tokens."""
    estimate = int(sum(len(getattr(m, "content", "") or "") for m in messages or []) / _RETRY_CHARS_PER_TOKEN)
    m = _CONTEXT_LIMIT_RE.search(error_text)
    if m:
        return int(m.group(1)), max(int(m.group(2)), estimate)
    m = _CONTEXT_CHARS_RE.search(error_text)
    if m:
        return int(m.group(1)), max(int(int(m.group(2)) / _RETRY_CHARS_PER_TOKEN), estimate)
    if "max_completion_tokens" in error_text:
        m = _CONTEXT_SIZE_RE.search(error_text)
        if m:
            return int(m.group(1) or m.group(2)), estimate
    return None


_PLUMBING = {"_caller", "_reply_text", "_call_chat", "_chat", "_chat_json", "_cut_json", "call_chat", "once"}
# The last reply on this thread as it arrived, for diagnose_long's dumps: _call_chat returns it cleaned.
last_reply = threading.local()


def _caller() -> str:
    """The function that asked for this reply, past the plumbing, for the logs."""
    frame = sys._getframe(1)
    while frame is not None and frame.f_code.co_name in _PLUMBING:
        frame = frame.f_back
    return frame.f_code.co_name if frame is not None else "?"


def _reply_text(response, max_tokens: Optional[int] = None) -> str:
    """The reply without its reasoning. A reply that held some is logged, and so is one that came back empty. max_tokens:
    what the call really had, which an overflow may have lowered below what its caller asked."""
    raw = response.content or ""
    finish = (getattr(response, "response_metadata", None) or {}).get("finish_reason")
    last_reply.raw, last_reply.finish_reason, last_reply.max_tokens = raw, finish, max_tokens
    call = _caller()
    vlog("llm_call", {"call": call, "finish_reason": finish, "chars": len(raw), "max_tokens": max_tokens})
    if finish == "length":
        logger.warning("model reply cut at its token limit: %s (%d characters, max_tokens %s)", call, len(raw), max_tokens)
    text = strip_think(raw)
    if text != raw.strip():
        logger.info("model reply: reasoning removed (%d characters of %d)", len(raw) - len(text), len(raw))
        vlog("llm_reply_think_stripped", {"removed": len(raw) - len(text), "kept": len(text)})
    if not text:
        logger.warning("model reply empty after removing its reasoning (finish_reason=%s, %d characters raw)",
                       finish, len(raw))
    return text


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
        sizes = _overflow_sizes(str(exc), messages)
        if not sizes:
            raise
        limit, prompt_tokens = sizes
        room = limit - prompt_tokens - _RETRY_MARGIN_TOKENS
        if max_tokens is not None:
            room = min(room, max_tokens)     # never ask for a longer answer than the caller did
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
    return _reply_text(response, bind_kwargs.get("max_tokens"))


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
    response = model.invoke(messages)
    if isinstance(response.content, str):
        response.content = strip_think(response.content)
    return response
