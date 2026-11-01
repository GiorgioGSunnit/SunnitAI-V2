"""What a model's reply holds, whatever it came wrapped in. One place for every path: _call_chat strips the reasoning from
every reply, free text included, and every JSON-expecting call reads through parse_json_reply.

Qwen3 (vLLM with --reasoning-parser qwen3) answers inside "<think>...</think>", empty or not, in code fences, or after
prose. Reasoning can hold brackets of its own, so it goes before any JSON is looked for. No dependency beyond the
standard library: scripts/plan_fallback_report.py loads this file alone.
"""

import json
import re
from typing import Any, Optional

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"</think>", re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r"<think>", re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json)?\s*")


def strip_think(text: Optional[str]) -> str:
    """The reply without its reasoning: every "<think>...</think>", all before a stray "</think>" (the server's parser
    took the opening tag), all from an unclosed "<think>" on (the reply was cut while reasoning: no answer in it)."""
    text = _THINK_RE.sub("", text or "")
    closes = list(_THINK_CLOSE_RE.finditer(text))
    if closes:
        text = text[closes[-1].end():]
    opened = _THINK_OPEN_RE.search(text)
    return (text[:opened.start()] if opened else text).strip()


def clean_json_text(raw: Optional[str]) -> str:
    return _FENCE_RE.sub("", strip_think(raw)).strip().rstrip("`").strip()


def parse_json_reply(raw: Optional[str]) -> Any:
    """The outermost JSON array or object in a model reply, ignoring reasoning, code fences and any text around it: of
    several, the longest ("[1] ... {the plan}"), the first on a tie. A strict json.loads failed on production (16 Oct,
    "Extra data"): the model wrote the list asked for and then more text, and the strategy pipeline lost its article
    choice. Raises ValueError when the reply holds no JSON."""
    text = clean_json_text(raw)
    try:
        return json.loads(text)
    except ValueError:
        pass
    decoder = json.JSONDecoder()
    best, size, i = None, 0, 0
    while i < len(text):
        if text[i] in "[{":
            try:
                value, end = decoder.raw_decode(text, i)
            except ValueError:
                i += 1
                continue
            if end - i > size:
                best, size = value, end - i
            i = end
        else:
            i += 1
    if size:
        return best
    raise ValueError("no JSON value in the model's reply")


def salvage_json(raw: Optional[str]) -> Any:
    """What a reply cut short holds, from its outermost value: read up to its last complete object or list, the open
    brackets closed, so an item cut in the middle is left out. parse_json_reply would return one complete item from
    inside. ValueError when nothing is complete."""
    text = clean_json_text(raw)
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if not starts:
        raise ValueError("no JSON value in the reply")
    start = min(starts)
    stack, in_string, escape, safe = [], False, False, None
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if stack:
                stack.pop()
            safe = (i + 1, list(stack))
            if not stack:
                break
    if not safe:
        raise ValueError("the reply is cut before any complete value")
    end, still_open = safe
    return json.loads(text[start:end] + "".join(reversed(still_open)))
