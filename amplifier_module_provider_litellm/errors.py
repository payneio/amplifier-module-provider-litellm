"""Fail-loud error types and transient-retry classification.

Design contract (locked):
- Misconfiguration / unreachable capability source -> fail loud (raise).
- An unexpected 4xx (e.g. a param the model's allow-list claimed to support but
  the proxy rejected) -> surface it. NEVER silently strip-and-retry.
- Transient errors (network, 408/409/429, 5xx) -> retry with backoff.
"""

from __future__ import annotations

# Status codes we treat as transient (safe to retry with backoff).
_TRANSIENT_STATUS = {408, 409, 429}


class LiteLLMError(Exception):
    """Base error for the LiteLLM provider."""


class LiteLLMConfigError(LiteLLMError):
    """Required configuration is missing or invalid. Fail loud at mount."""


class LiteLLMCapabilityError(LiteLLMError):
    """The proxy's /model/info could not be reached/parsed at mount. Fail loud."""


class LiteLLMRequestError(LiteLLMError):
    """A completion request failed.

    `status` is the HTTP status (None for transport-level failures).
    `transient` indicates whether a retry is warranted.
    """

    def __init__(
        self, message: str, *, status: int | None = None, transient: bool = False
    ) -> None:
        super().__init__(message)
        self.status = status
        self.transient = transient


def is_transient_status(status: int) -> bool:
    """Return True if an HTTP status should be retried."""
    return status in _TRANSIENT_STATUS or status >= 500
