# amplifier-module-provider-litellm

An Amplifier provider module that fronts a [LiteLLM](https://github.com/BerriAI/litellm)
proxy as a **single front-door** over every model the proxy exposes.

Instead of treating each model as plain OpenAI, this provider exploits LiteLLM's
OpenAI-compatible extensions and lets the proxy do all per-vendor translation:

- **Live per-model capabilities** — fetched from the proxy's `/model/info` at
  mount time. Every optional wire param is gated against that model's
  `supported_openai_params` allow-list. An unknown model falls back to plain
  OpenAI passthrough (no extensions) rather than silently degrading.
- **Reasoning / extended thinking** — prefers the portable `reasoning_effort`
  knob and lets the proxy translate it into each vendor's current thinking API
  (Anthropic adaptive, OpenAI reasoning, etc.). Falls back to the raw Anthropic
  `thinking` block only when a model exposes `thinking` but not
  `reasoning_effort`. Assistant `thinking_blocks` (with signatures) are echoed
  back across tool turns for reasoning continuity. Default **OFF**.
- **Prompt caching** — Anthropic-style breakpoints (static prefix + rolling
  conversation), applied only when the model reports prompt-cache support. Cache
  read/write tokens are surfaced in `Usage`.
- **Fail loud** — bad config or an unreachable `/model/info` raises at mount; a
  non-transient HTTP status (e.g. 400) surfaces instead of a silent
  strip-and-retry. Only transport errors and 408/409/429/5xx are retried with
  backoff.

## Install

```bash
amplifier bundle add git+https://github.com/payneio/amplifier-module-provider-litellm@main
```

Or reference it directly from a bundle / `settings.yaml`:

```yaml
providers:
  - id: litellm
    module: provider-litellm
    source: git+https://github.com/payneio/amplifier-module-provider-litellm@main
    config:
      base_url: http://localhost:4000
      api_key: ${LITELLM_MASTER_KEY}
      default_model: claude-opus-4-8
```

## Configuration

| Key | Required | Default | Purpose |
|-----|----------|---------|---------|
| `base_url` | yes (or `LITELLM_BASE_URL`) | — | Proxy base URL |
| `api_key` | yes (or `api_key_env` / `LITELLM_MASTER_KEY` / `LITELLM_API_KEY`) | — | Proxy key |
| `api_key_env` | no | `LITELLM_MASTER_KEY` | Env var name to read the key from |
| `default_model` | no | — | Model used when a request doesn't name one |
| `thinking_budget_tokens` | no | `8192` | Budget for the raw `thinking` fallback |
| `request_timeout` | no | `600` | HTTP timeout (seconds) |
| `max_retries` | no | `2` | Retries for transient failures |

## Smoke test

`smoke_test.py` exercises the real code path end-to-end against a running proxy
(mount + live capability load, plain completion, extended thinking, and prompt
cache write→read):

```bash
export LITELLM_BASE_URL=http://localhost:4000
export LITELLM_MASTER_KEY=...   # your proxy key
python smoke_test.py
```

## License

MIT
