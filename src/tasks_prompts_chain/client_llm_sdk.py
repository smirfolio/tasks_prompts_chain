from typing import Optional

from .events import TokenUsage


def _get(obj, name):
    """Read an attribute or dict key from a provider object, safely."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _as_int(value):
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


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


class ClientLLMSDK:
    """
    A class to handle LLM SDKs for various providers.
    This class is designed to work with different LLM SDKs by
    dynamically loading the appropriate class based on the provider.
    """
    def __init__(self, AsyncLLmAi, model_options):
        """
        Initialize with any LLM SDK class.

        :param llm_class: The LLM class to be used (e.g., openai.AsyncOpenAI, anthropic.AsyncAnthropic)
        :param kwargs: Additional parameters for initializing the LLM instance
        """
        self.llm_class_name = AsyncLLmAi.__name__  # Store the class type
        client_kwargs = {"api_key": model_options["api_key"]}
        if "base_url" in model_options:
            client_kwargs["base_url"] = model_options["base_url"]
         # Instantiate the LLM
        self.client = AsyncLLmAi(**client_kwargs)
        self.last_usage = TokenUsage(source="missing")
        self.last_raw_usage = None

    def get_last_usage(self) -> TokenUsage:
        """Return the usage captured during the most recent generate call."""
        return self.last_usage

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
        max_tokens = kwargs.get("max_tokens", 512)
        stream = kwargs.get("stream", True)
        stream_options = kwargs.get("stream_options")

        # Every call starts with a clean usage record.
        self.last_usage = TokenUsage(source="missing")
        self.last_raw_usage = None

        if self.llm_class_name == "AsyncOpenAI":  # OpenAI SDK
            create_kwargs = dict(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
            )
            if stream_options is not None:
                create_kwargs["stream_options"] = stream_options

            try:
                response = await self.client.chat.completions.create(**create_kwargs)
            except Exception as exc:
                if stream_options is not None and _is_stream_options_rejection(exc):
                    # The first request was rejected before generation, so the
                    # retry cannot double-bill. Providers that do not support
                    # stream_options still work, just without usage.
                    create_kwargs.pop("stream_options", None)
                    response = await self.client.chat.completions.create(**create_kwargs)
                else:
                    raise

            async for chunk in response:
                delta = None
                self._capture_usage(chunk)
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
                max_tokens=max_tokens,
                stream=stream,
            )
            if stream_options is not None:
                create_kwargs["stream_options"] = stream_options

            try:
                response = await self.client.chat.completions.create(**create_kwargs)
            except Exception as exc:
                if stream_options is not None and _is_stream_options_rejection(exc):
                    create_kwargs.pop("stream_options", None)
                    response = await self.client.chat.completions.create(**create_kwargs)
                else:
                    raise

            async for chunk in response:
                delta = None
                self._capture_usage(chunk)
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
