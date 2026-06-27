"""Prompt-cache breakpoint placement (Anthropic-style, via the proxy).

Decision (locked): "static prefix + rolling conversation breakpoint".
- One breakpoint at the end of the cacheable static prefix (system prompt and,
  if present, the tool definitions) -- this rarely changes turn to turn.
- One rolling breakpoint on the last content block of the LAST message of the
  PRIOR turn, so the growing conversation prefix gets cached and is read back
  on the next turn.

Anthropic allows up to 4 breakpoints; we use up to 2. cache_control must live
on a structured content *part*, so string content is promoted to a one-element
text-part list before the marker is attached.

Only applied when the model's live capabilities report prompt-cache support.
This module mutates the already-serialized OpenAI-format wire structures.
"""

from __future__ import annotations

from typing import Any

_EPHEMERAL = {"type": "ephemeral"}


def _mark_last_part(content: Any) -> Any:
    """Attach cache_control to the last text-ish part of a message content.

    Accepts either a string (promoted to a single text part) or a list of
    parts. Returns the new content value (always a list when marked).
    """
    if isinstance(content, str):
        if not content:
            return content
        return [{"type": "text", "text": content, "cache_control": _EPHEMERAL}]
    if isinstance(content, list) and content:
        # Find the last part we can safely annotate (prefer text parts).
        for part in reversed(content):
            if isinstance(part, dict):
                part["cache_control"] = _EPHEMERAL
                return content
    return content


def apply_cache_control(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
) -> None:
    """Place static-prefix and rolling breakpoints. Mutates in place.

    Caller is responsible for only invoking this when the target model
    supports prompt caching.
    """
    # --- Static prefix breakpoint -------------------------------------------
    # Prefer marking the tool definitions (they sit after the system prompt in
    # the cached prefix); otherwise mark the system message.
    if tools:
        last_tool = tools[-1]
        if isinstance(last_tool, dict):
            last_tool["cache_control"] = _EPHEMERAL
    else:
        for msg in messages:
            if msg.get("role") in ("system", "developer"):
                msg["content"] = _mark_last_part(msg.get("content"))
                break

    # --- Rolling conversation breakpoint ------------------------------------
    # The last message of the prior turn = the last message that is NOT the
    # in-flight final user message. Mark its last content part so the whole
    # conversation prefix up to here is cached for next turn.
    if len(messages) >= 2:
        # Walk backward past the final message to the previous one.
        target = messages[-2]
        if isinstance(target, dict) and target.get("content") is not None:
            target["content"] = _mark_last_part(target.get("content"))
