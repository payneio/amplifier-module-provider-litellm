"""Live per-model capability map sourced from the proxy's /model/info.

Decision (locked): capabilities come LIVE from the proxy, not from a static
table or model-name heuristics. Every optional wire param is gated against the
per-model `supported_openai_params` allow-list. An unknown model falls back to
plain OpenAI passthrough (no extensions) -- the documented default, not a
silent degradation.
"""

from __future__ import annotations

from typing import Any

import httpx

from .errors import LiteLLMCapabilityError


class ModelCaps:
    """Capabilities for a single model as reported by the proxy."""

    __slots__ = (
        "name",
        "supported_params",
        "supports_prompt_caching",
        "supports_reasoning",
        "raw",
    )

    def __init__(
        self,
        name: str,
        supported_params: set[str],
        supports_prompt_caching: bool,
        supports_reasoning: bool,
        raw: dict[str, Any],
    ) -> None:
        self.name = name
        self.supported_params = supported_params
        self.supports_prompt_caching = supports_prompt_caching
        self.supports_reasoning = supports_reasoning
        self.raw = raw


class CapabilityMap:
    """Maps model name -> ModelCaps. Built once at mount, refreshable."""

    def __init__(self, models: dict[str, ModelCaps]) -> None:
        self._models = models

    # ------------------------------------------------------------------ build
    @classmethod
    async def fetch(cls, client: httpx.AsyncClient, base_url: str) -> "CapabilityMap":
        """GET {base_url}/model/info and build the map. Fail loud on error."""
        url = base_url.rstrip("/") + "/model/info"
        try:
            resp = await client.get(url)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:  # noqa: BLE001 - fail loud at mount, wrap for clarity
            raise LiteLLMCapabilityError(
                f"Could not fetch capability info from {url}: {exc}. "
                "The litellm provider cannot mount without it."
            ) from exc

        rows = payload.get("data", payload) if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise LiteLLMCapabilityError(
                f"Unexpected /model/info shape from {url}: expected a list of models."
            )

        models: dict[str, ModelCaps] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            name = row.get("model_name") or row.get("model") or ""
            if not name:
                continue
            info = row.get("model_info") or {}
            supported = info.get("supported_openai_params") or []
            caps = ModelCaps(
                name=name,
                supported_params={str(p) for p in supported},
                supports_prompt_caching=bool(
                    info.get("supports_prompt_caching", False)
                ),
                supports_reasoning=bool(
                    info.get("supports_reasoning", False)
                    or "reasoning_effort" in supported
                    or "thinking" in supported
                ),
                raw=info,
            )
            models[name] = caps
        if not models:
            raise LiteLLMCapabilityError(
                f"/model/info at {url} returned no usable models."
            )
        return cls(models)

    # ------------------------------------------------------------------ query
    def get(self, model: str | None) -> ModelCaps | None:
        if not model:
            return None
        return self._models.get(model)

    def is_known(self, model: str | None) -> bool:
        return self.get(model) is not None

    def supports_param(self, model: str | None, param: str) -> bool:
        caps = self.get(model)
        if caps is None:
            return False
        return param in caps.supported_params

    def supports_prompt_caching(self, model: str | None) -> bool:
        caps = self.get(model)
        return bool(caps and caps.supports_prompt_caching)

    def supports_reasoning(self, model: str | None) -> bool:
        caps = self.get(model)
        return bool(caps and caps.supports_reasoning)

    def model_names(self) -> list[str]:
        return list(self._models.keys())

    def all(self) -> list[ModelCaps]:
        return list(self._models.values())
