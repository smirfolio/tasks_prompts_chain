from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional


# Event type constants. Plain strings keep the public surface small and make
# the values trivially serializable for telemetry pipelines.
EVENT_PROMPT_END = "prompt_end"
EVENT_CHAIN_END = "chain_end"


@dataclass(frozen=True)
class TokenUsage:
    """Normalized token usage for one prompt or an aggregated chain."""

    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    source: str = "missing"  # "provider" | "missing"


@dataclass(frozen=True)
class ChainEvent:
    """One observer event.

    type:
      - "prompt_end": emitted once per prompt, after success or failure.
      - "chain_end": emitted once per execute_chain call, in a finally.

    Events carry metadata and counters only: never prompt text, response text,
    messages, API keys, or provider URLs.
    """

    type: str
    prompt_index: int          # 1-based; for chain_end == prompt_count
    prompt_count: int
    output_placeholder: Optional[str]
    llm_id: Optional[str]
    model: Optional[str]
    metadata: Dict[str, Any]
    outcome: str               # "ok" | "error"
    started_at: datetime       # timezone-aware UTC
    ended_at: datetime         # timezone-aware UTC
    duration_ms: int
    first_token_ms: Optional[int]
    usage: TokenUsage
    response_chars: int
    final_result_placeholder: Optional[str] = None
    error: Optional[BaseException] = None
    error_type: Optional[str] = None
