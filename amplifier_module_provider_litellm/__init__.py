"""Amplifier provider for a LiteLLM proxy (single front-door).

One provider that fronts every model the proxy exposes, exploiting LiteLLM's
OpenAI-compatible extensions: Anthropic prompt caching, extended thinking, and
reasoning round-tripping. Per-model feature detection comes live from the
proxy's /model/info; all per-vendor translation is the proxy's job.

mount() eagerly fetches the capability map and FAILS LOUD if it can't -- a
single front-door that can't reach its proxy is useless, and mounting in a
degraded "everything plain" state would silently strip caching/thinking from
every call.
"""

from __future__ import annotations

import asyncio
import os
from decimal import Decimal
from typing import Any

import httpx

from amplifier_core.message_models import ChatResponse
from amplifier_core.message_models import ToolCall
from amplifier_core.models import ModelInfo
from amplifier_core.models import ProviderInfo

from .capabilities import CapabilityMap
from .errors import LiteLLMConfigError
from .errors import LiteLLMRequestError
from .errors import is_transient_status
from .request_build import build_payload
from .response_parse import parse_response

__all__ = ["LiteLLMProvider", "mount"]

_DEFAULT_TIMEOUT = 600.0
_DEFAULT_MAX_RETRIES = 2


class LiteLLMProvider:
    """Provider implementation backed by a LiteLLM proxy."""

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        config: dict[str, Any] | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        # Signature is deliberately compatible with the amplifier-app-cli
        # lightweight provider loader, which instantiates the class directly as
        # (base_url, api_key, config={}) with no http client and without calling
        # mount()/init_capabilities(). client and caps are therefore lazy.
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._config = dict(config or {})
        self._client = client
        self._caps: CapabilityMap | None = None
        self._max_retries = int(self._config.get("max_retries", _DEFAULT_MAX_RETRIES))
        # Running session cost (USD), summed from each response's proxy-reported
        # cost. Surfaced to the CLI via the "session.cost" contributor (below).
        self._cost_total: Decimal | None = None
        # Public attributes the amplifier orchestrator reads off a provider.
        # _select_provider() picks the lowest `priority` number; without this
        # attribute it defaults to 100 and a higher-priority native provider
        # (e.g. anthropic) wins, so traffic never reaches the proxy.
        self.priority = int(self._config.get("priority", 100))
        self.default_model = self._config.get("default_model")
        self.raw = bool(self._config.get("raw", False))

    # ----------------------------------------------------------------- mount
    def _ensure_client(self) -> httpx.AsyncClient:
        """Return the http client, creating it lazily for non-mount() callers."""
        if self._client is None:
            timeout = float(self._config.get("request_timeout", _DEFAULT_TIMEOUT))
            self._client = httpx.AsyncClient(
                timeout=timeout,
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
        return self._client

    async def init_capabilities(self) -> None:
        """Fetch the live capability map. Fail loud on error (called at mount)."""
        self._caps = await CapabilityMap.fetch(self._ensure_client(), self._base_url)

    async def _ensure_caps(self) -> CapabilityMap:
        """Lazily load the capability map (CLI model-listing path skips mount)."""
        if self._caps is None:
            await self.init_capabilities()
        assert self._caps is not None
        return self._caps

    @property
    def caps(self) -> CapabilityMap:
        if self._caps is None:
            raise RuntimeError("LiteLLMProvider used before init_capabilities()")
        return self._caps

    # ---------------------------------------------------------------- protocol
    @property
    def config(self) -> dict[str, Any]:
        """Public view of the provider config (read by the orchestrator)."""
        return self._config

    @property
    def name(self) -> str:
        return "litellm"

    def get_info(self) -> ProviderInfo:
        return ProviderInfo(
            id="litellm",
            display_name="LiteLLM Proxy",
            credential_env_vars=["LITELLM_MASTER_KEY", "LITELLM_API_KEY"],
            capabilities=["tools", "thinking", "prompt_caching"],
            defaults={
                "timeout": self._config.get("request_timeout", _DEFAULT_TIMEOUT),
                "max_retries": self._max_retries,
            },
        )

    async def list_models(self) -> list[ModelInfo]:
        caps = await self._ensure_caps()
        models: list[ModelInfo] = []
        for mc in caps.all():
            caps_list = ["tools"]
            if mc.supports_reasoning:
                caps_list.append("thinking")
            if mc.supports_prompt_caching:
                caps_list.append("prompt_caching")
            models.append(
                ModelInfo(
                    id=mc.name,
                    display_name=mc.name,
                    context_window=int(mc.raw.get("max_input_tokens", 0) or 0),
                    max_output_tokens=int(mc.raw.get("max_output_tokens", 0) or 0),
                    capabilities=caps_list,
                )
            )
        return models

    async def complete(self, request: Any, **kwargs: Any) -> ChatResponse:
        caps = await self._ensure_caps()
        payload = build_payload(request, caps, self._config, kwargs)
        data = await self._post_completion(payload)
        response = parse_response(data)
        self._accumulate_cost(response)
        return response

    def parse_tool_calls(self, response: ChatResponse) -> list[ToolCall]:
        return list(response.tool_calls or [])

    def _accumulate_cost(self, response: ChatResponse) -> None:
        """Add this response's proxy-reported cost to the running session total."""
        cost = getattr(response.usage, "cost_usd", None) if response.usage else None
        if cost is not None:
            self._cost_total = (self._cost_total or Decimal("0")) + cost

    def session_cost(self) -> dict[str, str] | None:
        """Contributor for the kernel's 'session.cost' capability.

        Returns the accumulated cost as a string (the CLI's cost display sums
        these), or None when no cost data has been seen yet (shown as '?').
        """
        if self._cost_total is None:
            return None
        return {"cost_usd": str(self._cost_total)}

    async def close(self) -> None:
        """Close the underlying http client (best-effort; safe to call twice)."""
        client, self._client = self._client, None
        if client is not None:
            await client.aclose()

    # ------------------------------------------------------------------ http
    async def _post_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = self._base_url + "/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self._api_key}"}
        client = self._ensure_client()
        attempt = 0
        while True:
            try:
                resp = await client.post(url, json=payload, headers=headers)
            except httpx.HTTPError as exc:
                # Transport-level failure -> transient, retry with backoff.
                if attempt < self._max_retries:
                    await asyncio.sleep(0.5 * (2**attempt))
                    attempt += 1
                    continue
                raise LiteLLMRequestError(
                    f"litellm request to {url} failed: {exc}", transient=True
                ) from exc

            if resp.status_code < 400:
                data = resp.json()
                # LiteLLM reports per-request cost in a response header, not the
                # body. Thread it through so parse_response can surface it on
                # Usage.cost_usd and the session cost accumulates.
                cost = resp.headers.get("x-litellm-response-cost")
                if cost is not None:
                    data["_litellm_response_cost"] = cost
                return data

            # Fail loud on non-transient (e.g. 400): never silently strip-and-retry.
            if is_transient_status(resp.status_code) and attempt < self._max_retries:
                await asyncio.sleep(0.5 * (2**attempt))
                attempt += 1
                continue
            raise LiteLLMRequestError(
                f"litellm returned {resp.status_code}: {resp.text}",
                status=resp.status_code,
                transient=is_transient_status(resp.status_code),
            )


# amplifier-app-cli's provider_loader looks up the class by the convention
# name `Litellm` + `Provider`. Our canonical class keeps the LiteLLM brand
# casing, so expose an alias for the CLI's exact-match lookup.
LitellmProvider = LiteLLMProvider


async def mount(coordinator: Any, config: dict[str, Any] | None = None) -> Any:
    """Entry point: build the provider, load capabilities (fail loud), and
    self-register with the coordinator.

    The amplifier kernel calls mount(coordinator, config) and expects the module
    to register itself into the coordinator via coordinator.mount("providers",
    provider, name=...). The return value is treated as an optional cleanup
    callable -- NOT as the provider. Returning the provider instance (and never
    self-mounting) is why the kernel reported the provider 'missing' even though
    mount() succeeded.

    coordinator may be None for non-kernel callers (smoke test, the CLI's
    lightweight provider_loader); in that case we skip registration and just
    return the live provider so those paths keep working.
    """
    config = dict(config or {})

    base_url = config.get("base_url") or os.environ.get("LITELLM_BASE_URL")
    if not base_url:
        raise LiteLLMConfigError(
            "litellm provider requires 'base_url' (config) or LITELLM_BASE_URL (env)."
        )

    api_key = config.get("api_key")
    if not api_key:
        key_env = config.get("api_key_env", "LITELLM_MASTER_KEY")
        api_key = os.environ.get(key_env) or os.environ.get("LITELLM_API_KEY")
    if not api_key:
        raise LiteLLMConfigError(
            "litellm provider requires an API key via 'api_key', 'api_key_env', "
            "LITELLM_MASTER_KEY, or LITELLM_API_KEY."
        )

    timeout = float(config.get("request_timeout", _DEFAULT_TIMEOUT))
    client = httpx.AsyncClient(
        timeout=timeout,
        headers={"Authorization": f"Bearer {api_key}"},
    )

    provider = LiteLLMProvider(base_url, api_key, config, client)
    await provider.init_capabilities()  # fail loud if /model/info unreachable

    # Non-kernel callers (smoke test, CLI lightweight loader) pass coordinator=None.
    # They use the returned provider directly and don't expect a cleanup callable.
    if coordinator is None:
        return provider

    # Kernel path: self-register, then hand back a cleanup callable.
    await coordinator.mount("providers", provider, name=provider.name)

    # Surface per-session cost to the CLI's cost display (the proxy reports it
    # per request; we accumulate and contribute it here, like the other providers).
    if hasattr(coordinator, "register_contributor"):
        coordinator.register_contributor(
            "session.cost", "provider-litellm", provider.session_cost
        )

    async def cleanup() -> None:
        await provider.close()

    return cleanup
