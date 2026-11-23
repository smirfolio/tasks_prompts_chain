from typing import Optional

from .events import TokenUsage


def _get(obj, name):
    """Read an attribute or dict key from a provider object, safely."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


#: Compatibility fallback used when the caller does not provide a positive
#: ``max_tokens``. The chain always sends this value so providers never see
#: ``max_tokens=0`` or ``None``. Callers should set the model's actual maximum
#: output when it differs from this fallback.
DEFAULT_MAX_TOKENS = 6500


def _as_int(value):
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _positive_int(value) -> Optional[int]:
    """Return an int > 0, or None for unset/invalid/non-positive values."""
    parsed = _as_int(value)
    return parsed if parsed is not None and parsed > 0 else None


def _normalize_usage(raw) -> TokenUsage:
    """Normalize OpenAI/Anthropic/Cerebras usage payloads to TokenUsage."""
    prompt_tokens = _as_int(_get(raw, "prompt_tokens"))
    completion_tokens = _as_int(_get(raw, "completion_tokens"))
    total_tokens = _as_int(_get(raw, "total_tokens"))

    # Anthropic native names.
    if prompt_tokens is None:
        prompt_tokens = _as_int(_get(raw, "input_tokens"))
    if completion_tokens is None:
        completion_tokens = _as_int(_get(raw, "output_tokens"))

    prompt_details = _get(raw, "prompt_tokens_details")
    cached_tokens = _as_int(_get(prompt_details, "cached_tokens"))
    if cached_tokens is None:
        cached_tokens = _as_int(_get(raw, "cache_read_input_tokens"))

    completion_details = _get(raw, "completion_tokens_details")
    reasoning_tokens = _as_int(_get(completion_details, "reasoning_tokens"))

    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens

    return TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cached_tokens=cached_tokens,
        reasoning_tokens=reasoning_tokens,
        source="provider",
    )


def _merge_usage(current: TokenUsage, new: TokenUsage) -> TokenUsage:
    """Merge partial usage payloads, keeping any value already captured."""
    prompt_tokens = (
        new.prompt_tokens if new.prompt_tokens is not None else current.prompt_tokens
    )
    completion_tokens = (
        new.completion_tokens
        if new.completion_tokens is not None
        else current.completion_tokens
    )
    total_tokens = (
        new.total_tokens if new.total_tokens is not None else current.total_tokens
    )
    cached_tokens = (
        new.cached_tokens if new.cached_tokens is not None else current.cached_tokens
    )
    reasoning_tokens = (
        new.reasoning_tokens
        if new.reasoning_tokens is not None
        else current.reasoning_tokens
    )

    # Anthropic reports input usage at message_start and output usage at
    # message_delta. The payloads often carry partial totals, so prefer an
    # explicit new total and otherwise derive it from the merged components.
    if new.total_tokens is None:
        if prompt_tokens is not None and completion_tokens is not None:
            total_tokens = prompt_tokens + completion_tokens
        else:
            total_tokens = current.total_tokens

    return TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cached_tokens=cached_tokens,
        reasoning_tokens=reasoning_tokens,
        source="provider",
    )


def _is_stream_options_rejection(exc) -> bool:
    """Return True for a 400/422 rejection caused by stream_options."""
    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
    if status_code not in (400, 422):
        return False

    message = str(exc).lower()
    # A JSON-mode rejection is handled by the response_format fallback, not
    # here, even when the provider uses a generic "unknown parameter" phrase.
    if any(
        marker in message
        for marker in ("response_format", "json_object", "json_schema")
    ):
        return False
    markers = (
        "stream_options",
        "include_usage",
        "unknown parameter",
        "unrecognized parameter",
        "unsupported parameter",
        "extra inputs",
        "unexpected keyword",
    )
    return any(marker in message for marker in markers)


def _is_response_format_rejection(exc) -> bool:
    """Return True for a 400/422 rejection caused by ``response_format``.

    Keep the markers specific. A generic ``unknown parameter`` error may be
    about ``reasoning_effort`` or ``stream_options``; treating it as a JSON
    mode rejection would silently disable enforcement for unrelated reasons.
    """
    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
    if status_code not in (400, 422):
        return False

    message = str(exc).lower()
    markers = (
        "response_format",
        "response format",
        "json_object",
        "json_schema",
        "json mode",
        "structured output",
        "structured outputs",
    )
    return any(marker in message for marker in markers)


class ClientLLMSDK:
    """
    A class to handle LLM SDKs for various providers.
    This class is designed to work with different LLM SDKs by
    dynamically loading the appropriate class based on the provider.
    """
    def __init__(self, AsyncLLmAi, model_options, request_options=None):
        """
        Initialize with any LLM SDK class.

        :param llm_class: The LLM class to be used (e.g., openai.AsyncOpenAI, anthropic.AsyncAnthropic)
        :param kwargs: Additional parameters for initializing the LLM instance
        :param request_options: Optional request-level options (e.g.
            ``reasoning_effort``) kept separate from client constructor kwargs.
        """
        self.llm_class_name = AsyncLLmAi.__name__  # Store the class type
        client_kwargs = {"api_key": model_options["api_key"]}
        if "base_url" in model_options:
            client_kwargs["base_url"] = model_options["base_url"]
         # Instantiate the LLM
        self.client = AsyncLLmAi(**client_kwargs)
        # Keep the full request options so the token-limit parameter can be
        # selected deterministically from the SDK class and base URL.
        self.request_options = request_options or {}
        # Optional OpenAI-compatible thinking control. The caller includes
        # this only for models whose provider advertises thinking support.
        self.reasoning_effort = self.request_options.get(
            "reasoning_effort"
        ) or model_options.get("reasoning_effort")
        self.last_usage = TokenUsage(source="missing")
        self.last_raw_usage = None
        # None = not probed yet, True = last response_format call succeeded,
        # False = this provider rejected it; omit it on subsequent calls.
        self._response_format_supported: Optional[bool] = None
        self.last_finish_reason: Optional[str] = None

    def get_last_usage(self) -> TokenUsage:
        """Return the usage captured during the most recent generate call."""
        return self.last_usage

    def get_last_finish_reason(self) -> Optional[str]:
        """Return the provider finish reason for the most recent call."""
        return self.last_finish_reason

    def _select_token_parameter(self) -> str:
        """Choose the token-limit parameter before the request is sent.

        No guessing and no retry:

        - Anthropic native and Cerebras always use ``max_tokens``.
        - The OpenAI SDK uses ``max_completion_tokens`` for the official
          OpenAI endpoint (no custom ``base_url``).
        - An OpenAI SDK pointed at a custom base URL uses ``max_tokens``,
          preserving compatibility with OpenAI-compatible providers.
        - ``model_options["max_tokens_parameter"]`` explicitly overrides the
          OpenAI choice when a proxy or Azure-style endpoint needs the other
          parameter.
        """
        if self.llm_class_name != "AsyncOpenAI":
            return "max_tokens"

        explicit = self.request_options.get("max_tokens_parameter")
        if explicit in ("max_tokens", "max_completion_tokens"):
            return explicit

        base_url = self.request_options.get("base_url") or ""
        if not base_url or str(base_url).startswith("https://api.openai.com"):
            return "max_completion_tokens"
        return "max_tokens"

    def _extract_content(self, chunk) -> Optional[str]:
        """Extract content from different response formats"""
        if self.llm_class_name == "AsyncAnthropic":
            # Handle different Anthropic event types
            if hasattr(chunk, 'type'):
                if chunk.type == 'content_block_delta':
                    return chunk.delta.text
                elif chunk.type == 'message_stop':
                    return None
            return None
        else:
            # Handle OpenAI and other formats
            if hasattr(chunk, 'choices') and chunk.choices:
                if hasattr(chunk.choices[0], 'delta'):
                    return chunk.choices[0].delta.content
                elif hasattr(chunk.choices[0], 'message'):
                    return chunk.choices[0].message.content
        return None

    def _capture_usage(self, chunk) -> None:
        """Capture provider-reported usage from a stream chunk, if any."""
        raw_usage = _get(chunk, "usage")
        if raw_usage is None:
            message = _get(chunk, "message")
            raw_usage = _get(message, "usage")
        if raw_usage is not None:
            self.last_raw_usage = raw_usage
            self.last_usage = _merge_usage(
                self.last_usage, _normalize_usage(raw_usage)
            )

    def _capture_finish_reason(self, chunk) -> None:
        """Capture the provider finish/stop reason from a stream chunk."""
        if self.llm_class_name == "AsyncAnthropic":
            if _get(chunk, "type") == "message_delta":
                stop_reason = _get(_get(chunk, "delta"), "stop_reason")
                if stop_reason:
                    self.last_finish_reason = stop_reason
            return

        choices = _get(chunk, "choices")
        if not choices:
            return
        finish_reason = _get(choices[0], "finish_reason")
        if finish_reason:
            self.last_finish_reason = finish_reason

    async def _create_with_fallbacks(self, create_kwargs):
        """Create a chat completion, shedding unsupported optional params.

        ``response_format`` and ``stream_options`` are optional. A provider
        that rejects either gets one retry without the offending parameter. A
        rejected ``json_schema`` is downgraded to ``json_object`` once before
        being dropped entirely. These are pre-generation 400/422 rejections,
        so the retries cannot double-bill tokens.

        The token-limit parameter is selected before this call; this helper
        does not retry or negotiate ``max_tokens``/``max_completion_tokens``.
        """
        while True:
            try:
                response = await self.client.chat.completions.create(**create_kwargs)
                if "response_format" in create_kwargs:
                    self._response_format_supported = True
                return response
            except Exception as exc:
                response_format = create_kwargs.get("response_format")
                if (
                    response_format is not None
                    and _is_response_format_rejection(exc)
                ):
                    if response_format.get("type") == "json_schema":
                        create_kwargs["response_format"] = {"type": "json_object"}
                        continue
                    create_kwargs.pop("response_format", None)
                    self._response_format_supported = False
                    continue
                if (
                    "stream_options" in create_kwargs
                    and _is_stream_options_rejection(exc)
                ):
                    create_kwargs.pop("stream_options", None)
                    continue
                raise

    async def generat_response(self, **kwargs):
        """
        Generate a response from the LLM.

        :param prompt: The prompt to be sent to the LLM
        :param kwargs: Additional parameters for generating the response
        :return: The generated response
        """
        model = kwargs.get("model", "gpt-4")
        messages= kwargs.get("messages", [])
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = _positive_int(kwargs.get("max_tokens")) or DEFAULT_MAX_TOKENS
        max_completion_tokens = _positive_int(
            kwargs.get("max_completion_tokens")
        )
        stream = kwargs.get("stream", True)
        stream_options = kwargs.get("stream_options")
        response_format = kwargs.get("response_format")

        # Every call starts with a clean usage and finish-reason record.
        self.last_usage = TokenUsage(source="missing")
        self.last_raw_usage = None
        self.last_finish_reason = None

        if self.llm_class_name == "AsyncOpenAI":  # OpenAI SDK
            create_kwargs = dict(
                model=model,
                messages=messages,
                temperature=temperature,
                stream=stream,
            )
            if stream_options is not None:
                create_kwargs["stream_options"] = stream_options
            if self.reasoning_effort:
                create_kwargs["reasoning_effort"] = self.reasoning_effort
            if (
                response_format is not None
                and self._response_format_supported is not False
            ):
                create_kwargs["response_format"] = response_format

            token_parameter = self._select_token_parameter()
            token_value = max_completion_tokens or max_tokens
            if token_parameter == "max_completion_tokens":
                create_kwargs["max_completion_tokens"] = token_value
            else:
                create_kwargs["max_tokens"] = token_value

            response = await self._create_with_fallbacks(create_kwargs)

            async for chunk in response:
                delta = None
                self._capture_usage(chunk)
                self._capture_finish_reason(chunk)
                if chunk is not None:
                   if isinstance(chunk, str):
                            delta = chunk
                   else:
                       # Handle different response formats
                       delta = self._extract_content(chunk)
                if delta is not None:
                    yield delta
                

        elif self.llm_class_name == "AsyncAnthropic":  # Anthropic SDK
            # Extract system message if present
            system_message = ""
            filtered_messages = []

            for message in messages:
                if message.get("role") == "system" and message.get("content") != None:
                    system_message = message.get("content")
                else:
                    filtered_messages.append(message)

            # Update messages without system message
            messages = filtered_messages
            # Anthropic's native SDK does not accept stream_options. Usage is
            # still captured when the native event stream includes it.
            response = await self.client.messages.create(
                system=system_message,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream
            )

            async for chunk in response:
                delta = None
                self._capture_usage(chunk)
                self._capture_finish_reason(chunk)
                if chunk is not None:
                   if isinstance(chunk, str):
                            delta = chunk
                   else:
                       # Handle different response formats
                       delta = self._extract_content(chunk)
                if delta is not None:
                    yield delta

        elif self.llm_class_name == "AsyncCerebras":  # AsyncCerebras SDK
            create_kwargs = dict(
                model=model,
                messages=messages,
                temperature=temperature,
                stream=stream,
            )
            if stream_options is not None:
                create_kwargs["stream_options"] = stream_options
            if self.reasoning_effort:
                create_kwargs["reasoning_effort"] = self.reasoning_effort
            if (
                response_format is not None
                and self._response_format_supported is not False
            ):
                create_kwargs["response_format"] = response_format

            # Cerebras always uses max_tokens. max_completion_tokens is
            # intentionally ignored for this SDK.
            create_kwargs["max_tokens"] = max_tokens

            response = await self._create_with_fallbacks(create_kwargs)

            async for chunk in response:
                delta = None
                self._capture_usage(chunk)
                self._capture_finish_reason(chunk)
                if chunk is not None:
                    if isinstance(chunk, str):
                        delta = chunk
                    else:
                        choices = _get(chunk, "choices")
                        if choices:
                            delta = _get(_get(choices[0], "delta"), "content")
                if delta is not None:
                    yield delta
        else:
            raise NotImplementedError(f"Unsupported LLM: {self.llm_class_name}")
