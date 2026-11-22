# Changelog

All notable changes to `tasks_prompts_chain` are documented in this file.

## [Unreleased]

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
