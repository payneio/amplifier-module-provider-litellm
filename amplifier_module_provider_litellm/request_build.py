"""Build an OpenAI-with-extensions wire payload from a kernel ChatRequest.

Pure functions only (no network) so they unit-test without a proxy.

Responsibilities:
- Serialize kernel Messages/blocks to OpenAI chat format, echoing assistant
  `thinking_blocks` (with signature) verbatim for reasoning continuity across
  tool turns.
- Gate every optional param against the model's live allow-list.
- Resolve extended-thinking using the provider-anthropic precedence:
  kwargs override > request.reasoning_effort > config default budget; default OFF.
"""

from __future__ import annotations

import json
from typing import Any

from .caching import apply_cache_control
from .capabilities import CapabilityMap

# Effort -> thinking budget (tokens). Anthropic requires budget when thinking is
# enabled; medium/high/xhigh scale up from the configured default.
_EFFORT_BUDGET = {
    "low": 4096,
    "medium": 8192,
    "high": 16384,
    "xhigh": 32768,
}


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        return str(value)


def _serialize_messages(messages: list[Any]) -> list[dict[str, Any]]:
    """Convert kernel Messages to OpenAI chat-format dicts.

    A single kernel Message can fan out to multiple wire messages (e.g. tool
    results become separate role="tool" messages).
    """
    out: list[dict[str, Any]] = []
    for m in messages:
        role = getattr(m, "role", "user")
        content = getattr(m, "content", "")

        if isinstance(content, str):
            wire: dict[str, Any] = {"role": role, "content": content}
            name = getattr(m, "name", None)
            if name:
                wire["name"] = name
            tcid = getattr(m, "tool_call_id", None)
            if tcid and role == "tool":
                wire["tool_call_id"] = tcid
            out.append(wire)
            continue

        text_parts: list[dict[str, Any]] = []
        thinking_blocks: list[dict[str, Any]] = []
        tool_calls: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []

        for b in content:
            btype = getattr(b, "type", None)
            if btype == "text":
                text_parts.append({"type": "text", "text": b.text})
            elif btype == "thinking":
                tb: dict[str, Any] = {
                    "type": "thinking",
                    "thinking": getattr(b, "thinking", ""),
                }
                sig = getattr(b, "signature", None)
                if sig:
                    tb["signature"] = sig
                thinking_blocks.append(tb)
            elif btype == "redacted_thinking":
                thinking_blocks.append(
                    {"type": "redacted_thinking", "data": getattr(b, "data", "")}
                )
            elif btype == "tool_call":
                tool_calls.append(
                    {
                        "id": b.id,
                        "type": "function",
                        "function": {
                            "name": b.name,
                            "arguments": _stringify(getattr(b, "input", {})),
                        },
                    }
                )
            elif btype == "tool_result":
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": getattr(b, "tool_call_id", None),
                        "content": _stringify(getattr(b, "output", "")),
                    }
                )

        # Pure tool-result message -> emit the tool messages only.
        if tool_results and not text_parts and not tool_calls and not thinking_blocks:
            out.extend(tool_results)
            continue

        wire = {"role": role}
        wire["content"] = text_parts if text_parts else ""
        if thinking_blocks and role == "assistant":
            wire["thinking_blocks"] = thinking_blocks
        if tool_calls and role == "assistant":
            wire["tool_calls"] = tool_calls
        out.append(wire)
        if tool_results:
            out.extend(tool_results)

    return out


def _serialize_tools(tools: list[Any] | None) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    wire: list[dict[str, Any]] = []
    for t in tools:
        wire.append(
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": getattr(t, "description", "") or "",
                    "parameters": getattr(t, "parameters", {}) or {},
                },
            }
        )
    return wire


def _resolve_thinking(
    request: Any, config: dict[str, Any], kwargs: dict[str, Any]
) -> tuple[bool, int, str | None]:
    """Return (enabled, budget_tokens, effort) per locked precedence."""
    effort = kwargs.get("reasoning_effort") or getattr(
        request, "reasoning_effort", None
    )
    override = kwargs.get("extended_thinking")
    if "reasoning" in kwargs and override is None:
        override = kwargs.get("reasoning")

    if override is True:
        enabled = True
    elif override is False:
        enabled = False
    else:
        enabled = effort is not None

    default_budget = int(config.get("thinking_budget_tokens", 8192))
    budget = _EFFORT_BUDGET.get(effort or "", default_budget)
    return enabled, budget, effort


def build_payload(
    request: Any,
    caps: CapabilityMap,
    config: dict[str, Any],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Construct the wire payload, gating optional params on live capabilities."""
    model = (
        kwargs.get("model")
        or getattr(request, "model", None)
        or config.get("default_model")
        or config.get("model")
    )
    if not model:
        raise ValueError("No model specified for litellm completion request.")

    messages = _serialize_messages(list(getattr(request, "messages", []) or []))
    tools = _serialize_tools(getattr(request, "tools", None))

    payload: dict[str, Any] = {"model": model, "messages": messages}
    if tools:
        payload["tools"] = tools
        tool_choice = getattr(request, "tool_choice", None)
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

    # --- Standard generation params (gated) ---------------------------------
    def _gated(param: str, value: Any) -> None:
        if value is not None and caps.supports_param(model, param):
            payload[param] = value

    _gated("temperature", getattr(request, "temperature", None))
    _gated("top_p", getattr(request, "top_p", None))
    _gated("stop", getattr(request, "stop", None))

    max_out = getattr(request, "max_output_tokens", None)
    if max_out is not None:
        if caps.supports_param(model, "max_completion_tokens"):
            payload["max_completion_tokens"] = max_out
        elif caps.supports_param(model, "max_tokens"):
            payload["max_tokens"] = max_out

    # --- Extended thinking / reasoning (opt-in, default OFF) ----------------
    # Prefer the OpenAI-portable `reasoning_effort` knob and let the proxy
    # translate it to whatever each vendor's current thinking API wants
    # (Anthropic adaptive, OpenAI reasoning, etc.). Only fall back to the raw
    # Anthropic `thinking` block when a model exposes `thinking` but NOT
    # `reasoning_effort`. Translation is the proxy's job, not ours.
    enabled, budget, effort = _resolve_thinking(request, config, kwargs)
    if enabled:
        if caps.supports_param(model, "reasoning_effort"):
            payload["reasoning_effort"] = effort or "medium"
        elif caps.supports_param(model, "thinking"):
            payload["thinking"] = {"type": "enabled", "budget_tokens": budget}

    # --- Prompt caching breakpoints (only if model supports it) -------------
    if caps.supports_prompt_caching(model):
        apply_cache_control(payload["messages"], payload.get("tools"))

    return payload
