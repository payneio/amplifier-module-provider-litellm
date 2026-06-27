"""Live smoke test against the running LiteLLM proxy.

Exercises the real provider code path end-to-end:
  1. mount() -> live /model/info capability load (fail-loud)
  2. list_models()
  3. plain completion
  4. extended-thinking completion (thinking block + signature returned)
  5. prompt-cache write then read across two calls

Run: source litellm/.env first (for LITELLM_MASTER_KEY), or rely on env.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from amplifier_core.message_models import ChatRequest  # noqa: E402
from amplifier_core.message_models import Message  # noqa: E402

import amplifier_module_provider_litellm as prov  # noqa: E402

BASE_URL = os.environ.get("LITELLM_BASE_URL", "http://localhost:4000")
MODEL = os.environ.get("SMOKE_MODEL", "claude-opus-4-8")


def line(msg: str) -> None:
    print(f"\n=== {msg} ===", flush=True)


async def main() -> int:
    cfg = {"base_url": BASE_URL, "thinking_budget_tokens": 2048, "request_timeout": 120}
    line("mount() + live capability load")
    provider = await prov.mount(None, cfg)
    models = await provider.list_models()
    print(f"mounted; {len(models)} models known; target={MODEL}")
    caps = provider.caps.get(MODEL)
    assert caps is not None, f"{MODEL} not in capability map"
    print(
        f"caps: caching={caps.supports_prompt_caching} reasoning={caps.supports_reasoning} "
        f"thinking_param={'thinking' in caps.supported_params}"
    )

    # 1) plain completion --------------------------------------------------
    line("plain completion")
    req = ChatRequest(
        model=MODEL,
        messages=[Message(role="user", content="Reply with exactly: PONG")],
        max_output_tokens=32,
    )
    resp = await provider.complete(req)
    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    print("text:", repr(text.strip()))
    assert "PONG" in text.upper(), "plain completion did not return PONG"

    # 2) extended thinking -------------------------------------------------
    line("extended-thinking completion")
    req2 = ChatRequest(
        model=MODEL,
        messages=[Message(role="user", content="What is 17 * 23? Think step by step.")],
        max_output_tokens=2048,
    )
    resp2 = await provider.complete(req2, extended_thinking=True)
    thinking = [b for b in resp2.content if getattr(b, "type", "") == "thinking"]
    text2 = "".join(b.text for b in resp2.content if getattr(b, "type", "") == "text")
    print(f"thinking blocks: {len(thinking)}; first sig present: "
          f"{bool(thinking and thinking[0].signature)}")
    print("answer:", text2.strip()[:120])
    assert thinking, "no thinking blocks returned with extended_thinking=True"
    assert "391" in text2, "thinking answer wrong (expected 391)"

    # 3) cache write then read --------------------------------------------
    line("prompt caching (write then read)")
    big_system = "You are a helpful assistant. " + ("Context paragraph. " * 600)
    msgs = [
        Message(role="system", content=big_system),
        Message(role="user", content="Say READY"),
    ]
    r_write = await provider.complete(ChatRequest(model=MODEL, messages=msgs, max_output_tokens=16))
    r_read = await provider.complete(ChatRequest(model=MODEL, messages=msgs, max_output_tokens=16))

    def cache(u):
        return (
            getattr(u, "cache_write_tokens", None),
            getattr(u, "cache_read_tokens", None),
        )

    w = cache(r_write.usage)
    rd = cache(r_read.usage)
    print(f"call1 (write) cache_write/read = {w}")
    print(f"call2 (read)  cache_write/read = {rd}")
    assert rd[1], "second call did not report cache_read_tokens (caching not working)"

    line("ALL SMOKE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
