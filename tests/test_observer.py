import asyncio
import os
import sys
import unittest
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from types import SimpleNamespace

# Make the src layout importable when tests are run from the repository root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from tasks_prompts_chain import (
    ChainEvent,
    EVENT_CHAIN_END,
    EVENT_PROMPT_END,
    OutputFormat,
    PromptTemplate,
    TasksPromptsChain,
    TokenUsage,
)
from tasks_prompts_chain.client_llm_sdk import (
    _as_int,
    _get,
    _is_stream_options_rejection,
    _merge_usage,
    _normalize_usage,
)


# ---------------------------------------------------------------------------
# Fake provider SDKs. ClientLLMSDK dispatches on the class __name__, so the
# public fake classes are built with type() to get the exact provider names.
# ---------------------------------------------------------------------------

class _FakeClientBase:
    scripts = []
    instances = []

    @classmethod
    def configure(cls, scripts):
        cls.scripts = list(scripts)
        cls.instances = []

    def __init__(self, api_key=None, base_url=None, **kwargs):
        self.api_key = api_key
        self.base_url = base_url
        self.calls = []
        self._scripts = list(type(self).scripts)
        type(self).instances.append(self)

    async def _create(self, **kwargs):
        self.calls.append(dict(kwargs))
        if not self._scripts:
            raise RuntimeError("no scripted responses left")
        script = self._scripts.pop(0)
        if isinstance(script, BaseException):
            raise script

        async def gen():
            for chunk in script:
                yield chunk

        return gen()


class _FakeOpenAI(_FakeClientBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )


class _FakeAnthropic(_FakeClientBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.messages = SimpleNamespace(create=self._create)


class _FakeCerebras(_FakeClientBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )


FakeOpenAI = type("AsyncOpenAI", (_FakeOpenAI,), {})
FakeAnthropic = type("AsyncAnthropic", (_FakeAnthropic,), {})
FakeCerebras = type("AsyncCerebras", (_FakeCerebras,), {})


def text_chunk(text):
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=text))],
        usage=None,
    )


def usage_chunk(prompt=10, completion=5, total=15, cached=2, reasoning=3):
    return SimpleNamespace(
        choices=[],
        usage=SimpleNamespace(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
            prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning),
        ),
    )


def anthropic_message_start(input_tokens=100):
    return SimpleNamespace(
        type="message_start",
        message=SimpleNamespace(
            usage=SimpleNamespace(input_tokens=input_tokens)
        ),
    )


def anthropic_text_delta(text):
    return SimpleNamespace(
        type="content_block_delta",
        delta=SimpleNamespace(text=text),
    )


def anthropic_message_delta(output_tokens=20):
    return SimpleNamespace(
        type="message_delta",
        usage=SimpleNamespace(output_tokens=output_tokens),
    )


class FakeBadRequestError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.status_code = 400


class FakeUnprocessableError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.status_code = 422


class ObserverTests(unittest.IsolatedAsyncioTestCase):
    def make_chain(
        self,
        llm_class=FakeOpenAI,
        observer=None,
        include_usage=None,
        model="test-model",
        final_result_placeholder="final",
    ):
        llm_configs = [
            {
                "llm_id": "llm1",
                "llm_class": llm_class,
                "model_options": {
                    "model": model,
                    "api_key": "test-key",
                    "temperature": 0.0,
                    "max_tokens": 32,
                },
            }
        ]
        return TasksPromptsChain(
            llm_configs,
            final_result_placeholder=final_result_placeholder,
            observer=observer,
            include_usage=include_usage,
        )

    async def consume(self, chain, prompts, streamout=True):
        chunks = []
        async for chunk in chain.execute_chain(prompts, streamout=streamout):
            chunks.append(chunk)
        return chunks

    async def test_backward_compatibility_without_observer(self):
        FakeOpenAI.configure([[text_chunk("hello")]])
        chain = self.make_chain()

        chunks = await self.consume(
            chain, [{"prompt": "test prompt", "output_placeholder": "out"}]
        )

        self.assertEqual(chunks, ["hello", "<tasks-sys>Done</tasks-sys>"])
        self.assertEqual(chain.get_result("out"), "hello")
        calls = FakeOpenAI.instances[0].calls
        self.assertEqual(len(calls), 1)
        self.assertNotIn("stream_options", calls[0])

    async def test_observer_events(self):
        FakeOpenAI.configure([[text_chunk("one")], [text_chunk("two")]])
        events = []
        chain = self.make_chain(observer=events.append)

        prompts = [
            {
                "prompt": "first prompt",
                "output_placeholder": "first",
                "llm_id": "llm1",
                "metadata": {"request_kind": "key_feature"},
            },
            {
                "prompt": "second prompt",
                "output_placeholder": "second",
                "llm_id": "llm1",
            },
        ]

        chunks = await self.consume(chain, prompts)

        self.assertEqual(chunks, ["one", "two", "<tasks-sys>Done</tasks-sys>"])

        prompt_events = [e for e in events if e.type == EVENT_PROMPT_END]
        chain_events = [e for e in events if e.type == EVENT_CHAIN_END]
        self.assertEqual(len(prompt_events), 2)
        self.assertEqual(len(chain_events), 1)

        first, second = prompt_events
        self.assertEqual(first.prompt_index, 1)
        self.assertEqual(first.prompt_count, 2)
        self.assertEqual(first.output_placeholder, "first")
        self.assertEqual(first.llm_id, "llm1")
        self.assertEqual(first.model, "test-model")
        self.assertEqual(first.metadata, {"request_kind": "key_feature"})
        self.assertEqual(first.outcome, "ok")
        self.assertIsNotNone(first.started_at.tzinfo)
        self.assertIsNotNone(first.ended_at.tzinfo)
        self.assertGreaterEqual(first.duration_ms, 0)
        self.assertGreaterEqual(first.first_token_ms, 0)
        self.assertEqual(first.response_chars, 3)

        self.assertEqual(second.prompt_index, 2)
        self.assertEqual(second.prompt_count, 2)
        self.assertEqual(second.output_placeholder, "second")
        self.assertEqual(second.metadata, {})
        self.assertEqual(second.outcome, "ok")
        self.assertEqual(second.response_chars, 3)

        chain_event = chain_events[0]
        self.assertEqual(chain_event.prompt_index, 2)
        self.assertEqual(chain_event.prompt_count, 2)
        self.assertEqual(chain_event.metadata, {})
        self.assertEqual(chain_event.final_result_placeholder, "final")
        self.assertEqual(chain_event.outcome, "ok")
        self.assertEqual(chain_event.response_chars, 3)

    async def test_usage_and_reasoning(self):
        FakeOpenAI.configure(
            [[text_chunk("hi"), usage_chunk(10, 5, 15, 2, 3)]]
        )
        events = []
        chain = self.make_chain(observer=events.append)

        await self.consume(
            chain, [{"prompt": "p", "output_placeholder": "out"}]
        )

        prompt_event = [e for e in events if e.type == EVENT_PROMPT_END][0]
        expected = TokenUsage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            cached_tokens=2,
            reasoning_tokens=3,
            source="provider",
        )
        self.assertEqual(prompt_event.usage, expected)
        self.assertEqual(prompt_event.usage.source, "provider")

        chain_event = [e for e in events if e.type == EVENT_CHAIN_END][0]
        self.assertEqual(chain_event.usage, expected)

    async def test_missing_usage(self):
        FakeOpenAI.configure([[text_chunk("hi")]])
        events = []
        chain = self.make_chain(observer=events.append)

        await self.consume(
            chain, [{"prompt": "p", "output_placeholder": "out"}]
        )

        prompt_event = [e for e in events if e.type == EVENT_PROMPT_END][0]
        self.assertEqual(prompt_event.usage.source, "missing")
        self.assertIsNone(prompt_event.usage.prompt_tokens)
        self.assertIsNone(prompt_event.usage.completion_tokens)
        self.assertIsNone(prompt_event.usage.total_tokens)
        self.assertIsNone(prompt_event.usage.cached_tokens)
        self.assertIsNone(prompt_event.usage.reasoning_tokens)

        chain_event = [e for e in events if e.type == EVENT_CHAIN_END][0]
        self.assertEqual(chain_event.usage.source, "missing")

    async def test_stream_options_opt_in_matrix(self):
        prompt = [{"prompt": "p", "output_placeholder": "out"}]

        # Observer set, include_usage unset -> usage requested.
        FakeOpenAI.configure([[text_chunk("a")]])
        events = []
        chain = self.make_chain(observer=events.append)
        await self.consume(chain, prompt)
        call = FakeOpenAI.instances[0].calls[0]
        self.assertEqual(call["stream_options"], {"include_usage": True})
        self.assertEqual(len(events), 2)

        # Observer set, include_usage=False -> no usage request.
        FakeOpenAI.configure([[text_chunk("b")]])
        events = []
        chain = self.make_chain(observer=events.append, include_usage=False)
        await self.consume(chain, prompt)
        call = FakeOpenAI.instances[0].calls[0]
        self.assertNotIn("stream_options", call)
        self.assertEqual(len(events), 2)

        # No observer, include_usage=True -> usage requested, no events.
        FakeOpenAI.configure([[text_chunk("c")]])
        chain = self.make_chain(include_usage=True)
        await self.consume(chain, prompt)
        call = FakeOpenAI.instances[0].calls[0]
        self.assertEqual(call["stream_options"], {"include_usage": True})

        # No observer, include_usage unset -> default request unchanged.
        FakeOpenAI.configure([[text_chunk("d")]])
        chain = self.make_chain()
        await self.consume(chain, prompt)
        call = FakeOpenAI.instances[0].calls[0]
        self.assertNotIn("stream_options", call)

    async def test_stream_options_fallback(self):
        FakeOpenAI.configure(
            [
                FakeBadRequestError(
                    "Bad request: unknown parameter stream_options"
                ),
                [text_chunk("fallback")],
            ]
        )
        events = []
        chain = self.make_chain(observer=events.append)

        chunks = await self.consume(
            chain, [{"prompt": "p", "output_placeholder": "out"}]
        )

        self.assertEqual(chunks, ["fallback", "<tasks-sys>Done</tasks-sys>"])
        calls = FakeOpenAI.instances[0].calls
        self.assertEqual(len(calls), 2)
        self.assertIn("stream_options", calls[0])
        self.assertNotIn("stream_options", calls[1])

        prompt_events = [e for e in events if e.type == EVENT_PROMPT_END]
        chain_events = [e for e in events if e.type == EVENT_CHAIN_END]
        self.assertEqual(len(prompt_events), 1)
        self.assertEqual(prompt_events[0].usage.source, "missing")
        self.assertEqual(prompt_events[0].outcome, "ok")
        self.assertEqual(len(chain_events), 1)
        self.assertEqual(chain_events[0].outcome, "ok")

    async def test_stream_options_fallback_422(self):
        FakeOpenAI.configure(
            [
                FakeUnprocessableError(
                    "Unsupported parameter: include_usage is not supported"
                ),
                [text_chunk("fallback")],
            ]
        )
        chain = self.make_chain(observer=lambda event: None)

        chunks = await self.consume(
            chain, [{"prompt": "p", "output_placeholder": "out"}]
        )

        self.assertEqual(chunks, ["fallback", "<tasks-sys>Done</tasks-sys>"])
        calls = FakeOpenAI.instances[0].calls
        self.assertEqual(len(calls), 2)
        self.assertIn("stream_options", calls[0])
        self.assertNotIn("stream_options", calls[1])

    async def test_cerebras_stream_options_usage_and_fallback(self):
        # Normal path: usage requested, captured, and emitted. Raw string
        # chunks are also accepted by the Cerebras branch.
        FakeCerebras.configure([["raw", text_chunk("text"), usage_chunk()]])
        events = []
        chain = self.make_chain(llm_class=FakeCerebras, observer=events.append)

        chunks = await self.consume(
            chain, [{"prompt": "p", "output_placeholder": "out"}]
        )

        self.assertEqual(chunks, ["raw", "text", "<tasks-sys>Done</tasks-sys>"])
        self.assertEqual(chain.get_result("out"), "rawtext")
        call = FakeCerebras.instances[0].calls[0]
        self.assertEqual(call["stream_options"], {"include_usage": True})

        prompt_event = [e for e in events if e.type == EVENT_PROMPT_END][0]
        self.assertEqual(prompt_event.usage.source, "provider")
        self.assertEqual(prompt_event.usage.prompt_tokens, 10)
        self.assertEqual(prompt_event.usage.completion_tokens, 5)

        # Fallback path: a 422 rejection retries once without stream_options.
        FakeCerebras.configure(
            [
                FakeUnprocessableError("unknown parameter stream_options"),
                [text_chunk("fallback")],
            ]
        )
        events = []
        chain = self.make_chain(llm_class=FakeCerebras, observer=events.append)

        chunks = await self.consume(
            chain, [{"prompt": "p", "output_placeholder": "out"}]
        )

        self.assertEqual(chunks, ["fallback", "<tasks-sys>Done</tasks-sys>"])
        calls = FakeCerebras.instances[0].calls
        self.assertEqual(len(calls), 2)
        self.assertIn("stream_options", calls[0])
        self.assertNotIn("stream_options", calls[1])
        prompt_event = [e for e in events if e.type == EVENT_PROMPT_END][0]
        self.assertEqual(prompt_event.usage.source, "missing")

    async def test_cerebras_provider_error_is_wrapped(self):
        FakeCerebras.configure([RuntimeError("cerebras boom")])
        events = []
        chain = self.make_chain(llm_class=FakeCerebras, observer=events.append)

        with self.assertRaises(Exception) as ctx:
            async for _ in chain.execute_chain(
                [{"prompt": "p", "output_placeholder": "out"}]
            ):
                pass

        self.assertEqual(
            str(ctx.exception),
            "Error in prompt chain execution at prompt 0: cerebras boom",
        )
        prompt_event = [e for e in events if e.type == EVENT_PROMPT_END][0]
        self.assertEqual(prompt_event.outcome, "error")
        self.assertEqual(prompt_event.error_type, "RuntimeError")
        chain_event = [e for e in events if e.type == EVENT_CHAIN_END][0]
        self.assertEqual(chain_event.outcome, "error")

    async def test_chain_usage_aggregates_multiple_prompts(self):
        FakeOpenAI.configure(
            [
                [text_chunk("a"), usage_chunk(10, 5, 15, 2, 3)],
                [text_chunk("b"), usage_chunk(4, 6, 10, 1, 0)],
            ]
        )
        events = []
        chain = self.make_chain(observer=events.append)

        await self.consume(
            chain,
            [
                {"prompt": "p1", "output_placeholder": "out1"},
                {"prompt": "p2", "output_placeholder": "out2"},
            ],
        )

        prompt_events = [e for e in events if e.type == EVENT_PROMPT_END]
        self.assertEqual(
            [e.usage for e in prompt_events],
            [
                TokenUsage(10, 5, 15, 2, 3, "provider"),
                TokenUsage(4, 6, 10, 1, 0, "provider"),
            ],
        )
        chain_event = [e for e in events if e.type == EVENT_CHAIN_END][0]
        self.assertEqual(chain_event.usage, TokenUsage(14, 11, 25, 3, 3, "provider"))

    async def test_include_usage_without_observer_captures_usage(self):
        FakeOpenAI.configure([[text_chunk("hi"), usage_chunk()]])
        chain = self.make_chain(include_usage=True)

        await self.consume(chain, [{"prompt": "p", "output_placeholder": "out"}])

        self.assertIsNone(chain.observer)
        self.assertIs(chain.include_usage, True)
        usage = chain.clients["llm1"]["client"].get_last_usage()
        self.assertEqual(usage, TokenUsage(10, 5, 15, 2, 3, "provider"))
        call = FakeOpenAI.instances[0].calls[0]
        self.assertEqual(call["stream_options"], {"include_usage": True})

    async def test_invalid_llm_id_emits_error_events(self):
        FakeOpenAI.configure([[text_chunk("unused")]])
        events = []
        chain = self.make_chain(observer=events.append)

        with self.assertRaises(Exception) as ctx:
            async for _ in chain.execute_chain(
                [
                    {
                        "prompt": "p",
                        "output_placeholder": "out",
                        "llm_id": "does-not-exist",
                    }
                ]
            ):
                pass

        self.assertIn(
            "LLM with id 'does-not-exist' not found", str(ctx.exception)
        )
        prompt_events = [e for e in events if e.type == EVENT_PROMPT_END]
        chain_events = [e for e in events if e.type == EVENT_CHAIN_END]
        self.assertEqual(len(prompt_events), 1)
        self.assertEqual(prompt_events[0].outcome, "error")
        self.assertEqual(prompt_events[0].error_type, "ValueError")
        self.assertEqual(prompt_events[0].llm_id, "does-not-exist")
        self.assertEqual(prompt_events[0].usage.source, "missing")
        self.assertEqual(len(chain_events), 1)
        self.assertEqual(chain_events[0].outcome, "error")

    async def test_observer_exception_is_swallowed(self):
        FakeOpenAI.configure([[text_chunk("hello")]])
        seen = []

        def observer(event):
            seen.append(event)
            raise RuntimeError("observer boom")

        chain = self.make_chain(observer=observer)
        chunks = await self.consume(
            chain, [{"prompt": "p", "output_placeholder": "out"}]
        )

        self.assertEqual(chunks, ["hello", "<tasks-sys>Done</tasks-sys>"])
        self.assertEqual(len(seen), 2)
        self.assertEqual(
            [e.type for e in seen], [EVENT_PROMPT_END, EVENT_CHAIN_END]
        )

    async def test_prompt_error_event(self):
        FakeOpenAI.configure([RuntimeError("provider boom")])
        events = []
        chain = self.make_chain(observer=events.append)

        with self.assertRaises(Exception) as ctx:
            async for _ in chain.execute_chain(
                [{"prompt": "p", "output_placeholder": "out"}]
            ):
                pass

        exc = ctx.exception
        self.assertEqual(
            str(exc),
            "Error in prompt chain execution at prompt 0: provider boom",
        )
        self.assertIsInstance(exc.__cause__, RuntimeError)

        prompt_events = [e for e in events if e.type == EVENT_PROMPT_END]
        chain_events = [e for e in events if e.type == EVENT_CHAIN_END]
        self.assertEqual(len(prompt_events), 1)
        self.assertEqual(prompt_events[0].outcome, "error")
        self.assertEqual(prompt_events[0].error_type, "RuntimeError")
        self.assertIsInstance(prompt_events[0].error, RuntimeError)

        self.assertEqual(len(chain_events), 1)
        self.assertEqual(chain_events[0].outcome, "error")
        self.assertEqual(chain_events[0].error_type, "Exception")
        self.assertIs(chain_events[0].error, exc)

    async def test_stop_placeholder_event(self):
        FakeOpenAI.configure(
            [
                [
                    text_chunk("%%Not_project_description%%"),
                    usage_chunk(),
                ]
            ]
        )
        events = []
        chain = self.make_chain(observer=events.append)

        with self.assertRaises(Exception) as ctx:
            async for _ in chain.execute_chain(
                [
                    {
                        "prompt": "p",
                        "output_placeholder": "out",
                        "stop_placeholder": "%%Not_project_description%%",
                    }
                ]
            ):
                pass

        self.assertIn(
            "Invalid project description. Please provide a valid project description.",
            str(ctx.exception),
        )

        prompt_event = [e for e in events if e.type == EVENT_PROMPT_END][0]
        self.assertEqual(prompt_event.outcome, "error")
        self.assertEqual(prompt_event.usage.source, "provider")
        self.assertEqual(prompt_event.usage.prompt_tokens, 10)
        self.assertEqual(prompt_event.usage.completion_tokens, 5)
        self.assertEqual(prompt_event.usage.total_tokens, 15)
        self.assertEqual(prompt_event.response_chars, len("%%Not_project_description%%"))

        chain_event = [e for e in events if e.type == EVENT_CHAIN_END][0]
        self.assertEqual(chain_event.outcome, "error")

    async def test_metadata_isolation_and_backward_compatibility(self):
        FakeOpenAI.configure(
            [[text_chunk("a")], [text_chunk("b")], [text_chunk("c")]]
        )
        seen_metadata = []
        chain_event_metadata = []

        def observer(event):
            if event.type == EVENT_PROMPT_END:
                seen_metadata.append(dict(event.metadata))
                # Mutating the event must not corrupt chain state or later
                # events.
                event.metadata["injected"] = "mutation"
            elif event.type == EVENT_CHAIN_END:
                chain_event_metadata.append(dict(event.metadata))

        chain = self.make_chain(observer=observer)
        template = PromptTemplate(
            prompt="second",
            output_placeholder="out2",
            llm_id="llm1",
            metadata={"request_kind": "bom"},
        )
        prompts = [
            {
                "prompt": "first",
                "output_placeholder": "out1",
                "llm_id": "llm1",
                "metadata": {"request_kind": "core_requirements"},
            },
            template,
            {"prompt": "third", "output_placeholder": "out3", "llm_id": "llm1"},
        ]

        await self.consume(chain, prompts)

        self.assertEqual(
            seen_metadata,
            [
                {"request_kind": "core_requirements"},
                {"request_kind": "bom"},
                {},
            ],
        )
        self.assertEqual(template.metadata, {"request_kind": "bom"})
        self.assertEqual(chain_event_metadata, [{}])

        # A non-dict metadata value must not raise and must be treated as {}.
        template_with_bad_metadata = PromptTemplate(
            prompt="bad metadata", metadata=["not", "a", "dict"]
        )
        self.assertEqual(template_with_bad_metadata.metadata, {})

    async def test_anthropic_usage_merge(self):
        FakeAnthropic.configure(
            [
                [
                    anthropic_message_start(100),
                    anthropic_text_delta("hello"),
                    anthropic_message_delta(20),
                ]
            ]
        )
        events = []
        chain = self.make_chain(llm_class=FakeAnthropic, observer=events.append)

        chunks = await self.consume(
            chain, [{"prompt": "p", "output_placeholder": "out"}]
        )

        self.assertEqual(chunks, ["hello", "<tasks-sys>Done</tasks-sys>"])
        usage = chain.clients["llm1"]["client"].get_last_usage()
        self.assertEqual(usage.prompt_tokens, 100)
        self.assertEqual(usage.completion_tokens, 20)
        self.assertEqual(usage.total_tokens, 120)
        self.assertEqual(usage.source, "provider")

        prompt_event = [e for e in events if e.type == EVENT_PROMPT_END][0]
        self.assertEqual(prompt_event.usage, usage)

    async def test_streamout_false_regression(self):
        FakeOpenAI.configure([[text_chunk("hello")]])
        events = []
        chain = self.make_chain(observer=events.append)

        chunks = await self.consume(
            chain,
            [{"prompt": "p", "output_placeholder": "out"}],
            streamout=False,
        )

        self.assertEqual(chunks, ["<tasks-sys>Done</tasks-sys>"])
        self.assertEqual(chain.get_result("out"), "hello")
        self.assertEqual(
            [e.type for e in events], [EVENT_PROMPT_END, EVENT_CHAIN_END]
        )

    async def test_usage_resets_between_calls(self):
        FakeOpenAI.configure([[text_chunk("hi"), usage_chunk()], [text_chunk("bye")]])
        events = []
        chain = self.make_chain(observer=events.append)

        await self.consume(chain, [{"prompt": "p", "output_placeholder": "out"}])
        await self.consume(chain, [{"prompt": "p2", "output_placeholder": "out2"}])

        prompt_events = [e for e in events if e.type == EVENT_PROMPT_END]
        self.assertEqual(len(prompt_events), 2)
        self.assertEqual(prompt_events[0].usage.source, "provider")
        self.assertEqual(prompt_events[1].usage.source, "missing")
        self.assertIsNone(prompt_events[1].usage.prompt_tokens)


class UsageHelperTests(unittest.TestCase):
    def test_get_handles_none_dict_and_objects(self):
        self.assertIsNone(_get(None, "value"))
        self.assertEqual(_get({"prompt_tokens": 7}, "prompt_tokens"), 7)
        self.assertIsNone(_get({"prompt_tokens": 7}, "missing"))
        obj = SimpleNamespace(value=3)
        self.assertEqual(_get(obj, "value"), 3)
        self.assertIsNone(_get(obj, "missing"))

    def test_as_int_handles_bad_values(self):
        self.assertEqual(_as_int("12"), 12)
        self.assertEqual(_as_int(3.9), 3)
        self.assertIsNone(_as_int(None))
        self.assertIsNone(_as_int("not-a-number"))
        self.assertIsNone(_as_int(object()))

    def test_normalize_dict_payload_derives_total(self):
        usage = _normalize_usage(
            {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": None,
                "prompt_tokens_details": {"cached_tokens": 2},
                "completion_tokens_details": {"reasoning_tokens": 3},
            }
        )
        self.assertEqual(usage, TokenUsage(10, 5, 15, 2, 3, "provider"))

    def test_normalize_anthropic_payload(self):
        usage = _normalize_usage(
            {
                "input_tokens": 100,
                "output_tokens": 20,
                "cache_read_input_tokens": 5,
            }
        )
        self.assertEqual(usage, TokenUsage(100, 20, 120, 5, None, "provider"))

    def test_merge_usage_derives_final_total(self):
        start = _normalize_usage({"input_tokens": 100, "output_tokens": 1})
        merged = _merge_usage(start, _normalize_usage({"output_tokens": 20}))
        self.assertEqual(merged, TokenUsage(100, 20, 120, None, None, "provider"))

    def test_stream_options_rejection_predicate(self):
        class FakeError(Exception):
            def __init__(self, message, status_code=None):
                super().__init__(message)
                if status_code is not None:
                    self.status_code = status_code

        self.assertTrue(
            _is_stream_options_rejection(
                FakeError("unknown parameter stream_options", 400)
            )
        )
        self.assertTrue(
            _is_stream_options_rejection(
                FakeError("include_usage is unsupported", 422)
            )
        )
        self.assertFalse(
            _is_stream_options_rejection(
                FakeError("unknown parameter stream_options", 500)
            )
        )
        self.assertFalse(
            _is_stream_options_rejection(FakeError("bad request", 400))
        )
        self.assertFalse(
            _is_stream_options_rejection(FakeError("stream_options", None))
        )


class PublicSurfaceTests(unittest.TestCase):
    def test_new_symbols_are_exported(self):
        import tasks_prompts_chain

        for name in (
            "TasksPromptsChain",
            "PromptTemplate",
            "OutputFormat",
            "ChainEvent",
            "TokenUsage",
            "EVENT_PROMPT_END",
            "EVENT_CHAIN_END",
        ):
            self.assertIn(name, tasks_prompts_chain.__all__)
            self.assertTrue(hasattr(tasks_prompts_chain, name))

    def test_token_usage_defaults_and_event_fields(self):
        usage = TokenUsage()
        self.assertIsNone(usage.prompt_tokens)
        self.assertIsNone(usage.completion_tokens)
        self.assertIsNone(usage.total_tokens)
        self.assertIsNone(usage.cached_tokens)
        self.assertIsNone(usage.reasoning_tokens)
        self.assertEqual(usage.source, "missing")

        field_names = set(ChainEvent.__dataclass_fields__)
        self.assertTrue(
            {
                "type",
                "prompt_index",
                "prompt_count",
                "output_placeholder",
                "llm_id",
                "model",
                "metadata",
                "outcome",
                "started_at",
                "ended_at",
                "duration_ms",
                "first_token_ms",
                "usage",
                "response_chars",
                "final_result_placeholder",
                "error",
                "error_type",
            }.issubset(field_names)
        )

    def test_output_format_unchanged(self):
        self.assertEqual(OutputFormat.JSON.value, "JSON")
        self.assertEqual(OutputFormat.MARKDOWN.value, "MARKDOWN")
        self.assertEqual(OutputFormat.CSV.value, "CSV")
        self.assertEqual(OutputFormat.TEXT.value, "TEXT")

    def test_prompt_template_metadata_defaults_to_empty_dict(self):
        template = PromptTemplate(prompt="p")
        self.assertEqual(template.metadata, {})
        self.assertIsNot(template.metadata, PromptTemplate(prompt="q").metadata)

    def test_prompt_template_copies_metadata(self):
        source = {"request_kind": "bom"}
        template = PromptTemplate(prompt="p", metadata=source)
        source["request_kind"] = "changed"
        self.assertEqual(template.metadata, {"request_kind": "bom"})

    def test_events_and_usage_are_frozen(self):
        usage = TokenUsage()
        with self.assertRaises(FrozenInstanceError):
            usage.prompt_tokens = 1

        now = datetime.now(timezone.utc)
        event = ChainEvent(
            type=EVENT_PROMPT_END,
            prompt_index=1,
            prompt_count=1,
            output_placeholder="out",
            llm_id="llm1",
            model="test-model",
            metadata={},
            outcome="ok",
            started_at=now,
            ended_at=now,
            duration_ms=0,
            first_token_ms=None,
            usage=usage,
            response_chars=0,
        )
        with self.assertRaises(FrozenInstanceError):
            event.outcome = "error"


if __name__ == "__main__":
    unittest.main()
