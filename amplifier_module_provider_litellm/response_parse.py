"""Parse a proxy chat-completion response into a kernel ChatResponse.

Pure (dict in -> ChatResponse out), so it unit-tests without a network.

Preserves the three proven extension fields:
- `thinking_blocks` (with signature) and/or `reasoning_content` -> ThinkingBlock
- `tool_calls` -> ToolCallBlock + ChatResponse.tool_calls
- Anthropic-style cache usage -> Usage.cache_read_tokens / cache_write_tokens
Block order is thinking -> text -> tool calls, so reasoning leads the assistant
turn (required when echoed back across a tool round-trip).
"""

from __future__ import annotations

import json
from decimal import Decimal
from decimal import InvalidOperation
from typing import Any

from amplifier_core.message_models import ChatResponse
from amplifier_core.message_models import TextBlock
from amplifier_core.message_models import ThinkingBlock
from amplifier_core.message_models import ToolCall
from amplifier_core.message_models import ToolCallBlock
from amplifier_core.message_models import Usage


def _parse_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        except json.JSONDecodeError:
            return {"_raw": raw}
    return {}


def _to_decimal(value: Any) -> Decimal | None:
    """Coerce a proxy-supplied cost (str/number) to Decimal; None if unparseable.

    Usage.cost_usd is a Decimal field (the kernel rejects float), and LiteLLM
    returns the cost as a header string like "0.00024".
    """
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None


def _parse_usage(
    raw: dict[str, Any] | None, response_cost: Any = None
) -> Usage | None:
    cost_usd = _to_decimal(response_cost)
    if not raw and cost_usd is None:
        return None
    raw = raw or {}
    details = raw.get("prompt_tokens_details") or {}
    cache_read = raw.get("cache_read_input_tokens")
    if cache_read is None:
        cache_read = details.get("cached_tokens")
    cache_write = raw.get("cache_creation_input_tokens")
    completion_details = raw.get("completion_tokens_details") or {}

    input_tokens = int(raw.get("prompt_tokens", 0) or 0)
    output_tokens = int(raw.get("completion_tokens", 0) or 0)
    total_tokens = int(raw.get("total_tokens") or (input_tokens + output_tokens))

    # Optional/extension metrics passed via dict-unpack so they round-trip on
    # any core version (Usage uses extra="allow"); avoids hard-coding field
    # names that older cores may not declare.
    extras: dict[str, Any] = {}
    if completion_details.get("reasoning_tokens") is not None:
        extras["reasoning_tokens"] = int(completion_details["reasoning_tokens"])
    if cache_read is not None:
        extras["cache_read_tokens"] = int(cache_read)
    if cache_write is not None:
        extras["cache_write_tokens"] = int(cache_write)
    if cost_usd is not None:
        extras["cost_usd"] = cost_usd

    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        **extras,
    )


def parse_response(data: dict[str, Any]) -> ChatResponse:
    choices = data.get("choices") or []
    if not choices:
        raise ValueError("litellm response contained no choices")
    choice = choices[0]
    message = choice.get("message") or {}

    blocks: list[Any] = []
    tool_calls: list[ToolCall] = []

    # --- Reasoning (thinking first) -----------------------------------------
    thinking_blocks = message.get("thinking_blocks")
    if isinstance(thinking_blocks, list) and thinking_blocks:
        for tb in thinking_blocks:
            if not isinstance(tb, dict):
                continue
            blocks.append(
                ThinkingBlock(
                    thinking=tb.get("thinking", "") or "",
                    signature=tb.get("signature"),
                )
            )
    else:
        reasoning = message.get("reasoning_content")
        if reasoning:
            blocks.append(ThinkingBlock(thinking=reasoning))

    # --- Text ---------------------------------------------------------------
    content = message.get("content")
    if isinstance(content, str) and content:
        blocks.append(TextBlock(text=content))
    elif isinstance(content, list):
        for part in content:
            if (
                isinstance(part, dict)
                and part.get("type") == "text"
                and part.get("text")
            ):
                blocks.append(TextBlock(text=part["text"]))

    # --- Tool calls ---------------------------------------------------------
    for tc in message.get("tool_calls") or []:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") or {}
        name = fn.get("name", "")
        args = _parse_arguments(fn.get("arguments"))
        call_id = tc.get("id", "")
        blocks.append(ToolCallBlock(id=call_id, name=name, input=args))
        tool_calls.append(ToolCall(id=call_id, name=name, arguments=args))

    return ChatResponse(
        content=blocks,
        tool_calls=tool_calls,
        usage=_parse_usage(data.get("usage"), data.get("_litellm_response_cost")),
        finish_reason=choice.get("finish_reason"),
        metadata={"model": data.get("model"), "id": data.get("id")},
    )
