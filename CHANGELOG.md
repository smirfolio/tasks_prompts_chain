# Changelog

All notable changes to `tasks_prompts_chain` are documented in this file.

## [0.2.2] - 2026-11-23

### Added

- **Provider-level JSON output**: `PromptTemplate` and prompt dicts accept
  `response_format`, forwarded to OpenAI-compatible and Cerebras
  `chat.completions.create` calls. Anthropic’s native SDK ignores it. A
  rejected `json_schema` is downgraded once to `json_object`; if
  `response_format` is still unsupported, it is omitted for subsequent calls
  on that client. These are pre-generation 400/422 rejections, so the retry
  cannot double-bill tokens.
- **Finish-reason visibility**:
  `ClientLLMSDK.get_last_finish_reason()`,
  `TasksPromptsChain.get_finish_reason(placeholder)`, and
  `ChainEvent.finish_reason` expose values such as `stop`, `length`, and
  `end_turn`, so callers can distinguish truncated output from malformed
  output.
- **Deterministic token-limit selection**: the chain chooses the request
  parameter before sending the request:
  - OpenAI SDK + official OpenAI endpoint → `max_completion_tokens`;
  - OpenAI SDK + custom base URL → `max_tokens`;
  - Anthropic native and Cerebras → `max_tokens`;
  - `model_options["max_tokens_parameter"]` explicitly overrides the OpenAI
    choice with either `"max_tokens"` or `"max_completion_tokens"`.
- **6500 compatibility fallback**: missing, `None`, `0`, and negative token
  limits fall back to `6500`. The chain always sends a positive limit; 6500
  is a compatibility fallback, not a model-specific maximum.
- **Tests**: added coverage for JSON-mode forwarding and fallback,
  `json_schema` downgrade, finish-reason exposure, deterministic token
  parameter selection, the 6500 fallback, and the explicit
  `max_tokens_parameter` override.
- Deterministic `max_completion_tokens` support: the OpenAI SDK uses
  `max_completion_tokens` for the official OpenAI endpoint and `max_tokens`
  for custom OpenAI-compatible base URLs. Anthropic and Cerebras always use
  `max_tokens`.
- `max_tokens_parameter` override for proxies and Azure-style endpoints.
- 6500 compatibility fallback for missing, `None`, `0`, or negative limits.

### Changed

- Removed the error-driven `max_tokens` ↔ `max_completion_tokens` negotiation.
  The chain no longer sends a request, inspects a provider rejection, and
  resends with the other parameter. The correct parameter is selected before
  the request, so a failed request is never retried just to change the
  token-limit key.
- Existing `max_tokens` configurations keep working. Custom
  OpenAI-compatible endpoints continue to use `max_tokens`; the official
  OpenAI endpoint uses `max_completion_tokens`.
- Removed the error-driven token-parameter retry. The parameter is selected
  before the request; existing `max_tokens` configurations remain compatible.

### Preserved

- Existing public constructors, streaming chunks, output formats,
  `<tasks-sys>Done</tasks-sys>` sentinel, observer events, and
  `stream_options` fallback behavior are unchanged.
- No new runtime dependencies. Python 3.8+ remains supported.
  If you want only the max-token part, use this shorter version:
 
## [0.2.1] - 2026-11-22

### Added

- **Optional `reasoning_effort` request option**: `model_options` may now
  include `reasoning_effort` (for example `"low"`, `"medium"`, `"high"`).
  `ClientLLMSDK` forwards it on OpenAI-compatible and Cerebras
  `chat.completions.create` calls. The option is optional and omitted when
  absent, so existing callers send exactly the same requests as before.
- `ClientLLMSDK` accepts an optional `request_options` argument so the chain
  can carry request-level options separately from the client constructor
  kwargs. Direct constructors may still pass `reasoning_effort` in
  `model_options`.
- `tests/test_reasoning_effort.py` covers OpenAI/Cerebras forwarding, the
  omitted case, and the chain-constructor path.

## [0.2.0] - 2026-11-21

### Added

- **Observer / usage callback**: `TasksPromptsChain(..., observer=...)` emits one
  `prompt_end` event per prompt and one `chain_end` event per `execute_chain`
  call. Observer exceptions are swallowed so telemetry can never break a chain.
- **Per-prompt metadata**: `PromptTemplate(..., metadata={...})` and the dict
  prompt key `"metadata"` attach arbitrary JSON-like metadata (for example
  `{"request_kind": "key_feature"}`) to observer events. The observer receives
  a copy, so mutating an event cannot corrupt chain state.
- **Normalized token usage**: new `TokenUsage` dataclass with
  `prompt_tokens`, `completion_tokens`, `total_tokens`, `cached_tokens`,
  `reasoning_tokens`, and `source` (`"provider"` or `"missing"`). Usage is
  captured from OpenAI-compatible, Anthropic, and Cerebras streams.
- **`include_usage` option**: requests provider token usage through
  `stream_options={"include_usage": True}`. It defaults to `True` when an
  observer is provided and to `False` otherwise, so existing requests are
  unchanged.
- **Provider rejection fallback**: if an OpenAI-compatible provider rejects
  `stream_options` with a 400/422 error naming the parameter, the request is
  retried once without it. The rejected request never generated tokens, so the
  retry cannot double-bill.
- **New public events module**: `ChainEvent`, `TokenUsage`,
  `EVENT_PROMPT_END`, and `EVENT_CHAIN_END` are exported from the package.
- **`ClientLLMSDK.get_last_usage()`**: returns the usage captured during the
  most recent generation call.
- **Test suite**: `tests/test_observer.py` covers observer events, usage
  capture, metadata isolation, stream options, fallback, error paths,
  stop-placeholder handling, and Anthropic usage merging.

### Changed

- The outer chain error now uses `raise ... from exc`, preserving the original
  provider exception in the `__cause__` chain. The error message is unchanged.
- Fixed a latent `UnboundLocalError` in provider stream loops when a chunk was
  `None` or contained only usage.

### Preserved

- All existing public classes, methods, arguments, output formats, yielded text
  chunks, and the `<tasks-sys>Done</tasks-sys>` sentinel are unchanged.
- No new runtime dependencies. Python 3.8+ remains supported.

## [0.1.2] - 2025-06-28

- Added `stop_placeholder` support to halt chain execution when the LLM returns
  a guardrail marker.
- Improved Anthropic compatibility; provider dispatch still happens by class
  name (`AsyncOpenAI`, `AsyncAnthropic`, `AsyncCerebras`).
- Added multi-provider support and template output formatting.

## [0.1.0] - 2025-02-09

- First public release with sequential prompt chains, streaming, system
  prompts, placeholder substitution, multi-model support, and output templates.
