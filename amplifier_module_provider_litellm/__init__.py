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
        api_key: str,
        config: dict[str, Any],
        client: httpx.AsyncClient,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._config = config
        self._client = client
        self._caps: CapabilityMap | None = None
        self._max_retries = int(config.get("max_retries", _DEFAULT_MAX_RETRIES))

    # ----------------------------------------------------------------- mount
    async def init_capabilities(self) -> None:
        """Fetch the live capability map. Fail loud on error (called at mount)."""
        self._caps = await CapabilityMap.fetch(self._client, self._base_url)

    @property
    def caps(self) -> CapabilityMap:
        if self._caps is None:
            raise RuntimeError("LiteLLMProvider used before init_capabilities()")
        return self._caps

    # ---------------------------------------------------------------- protocol
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
        models: list[ModelInfo] = []
        for mc in self.caps.all():
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
        payload = build_payload(request, self.caps, self._config, kwargs)
        data = await self._post_completion(payload)
        return parse_response(data)

    def parse_tool_calls(self, response: ChatResponse) -> list[ToolCall]:
        return list(response.tool_calls or [])

    # ------------------------------------------------------------------ http
    async def _post_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = self._base_url + "/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self._api_key}"}
        attempt = 0
        while True:
            try:
                resp = await self._client.post(url, json=payload, headers=headers)
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
                return resp.json()

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


async def mount(coordinator: Any, config: dict[str, Any] | None = None) -> Any:
    """Entry point: build the provider and eagerly load capabilities (fail loud)."""
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
    return provider
