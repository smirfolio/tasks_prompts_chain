"""Provider JSON mode and finish-reason tests.

These tests use fake clients whose class names match the package's dispatch
keys (``AsyncOpenAI`` / ``AsyncAnthropic`` / ``AsyncCerebras``) so they run
without the real provider SDKs installed.

Run from the repository root:

    pytest tests/test_json_output.py -v
"""

from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from tasks_prompts_chain import (
    EVENT_PROMPT_END,
    TasksPromptsChain,
)


# ---------------------------------------------------------------------------
# Fake provider objects
# ---------------------------------------------------------------------------


class FakeProviderError(Exception):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


class FakeOpenAIChoice:
    def __init__(self, content=None, finish_reason=None):
        self.delta = SimpleNamespace(content=content)
        self.finish_reason = finish_reason


class FakeOpenAChunk:
    def __init__(self, content=None, finish_reason=None):
        self.choices = [FakeOpenAIChoice(content, finish_reason)]


class FakeAnthropicDelta:
    def __init__(self, text=None, stop_reason=None):
        self.text = text
        self.stop_reason = stop_reason


class FakeAnthropicChunk:
    def __init__(self, chunk_type, delta=None):
        self.type = chunk_type
        self.delta = delta


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


class _FakeOpenAIBase(_FakeClientBase):
    def __init__(self, api_key=None, base_url=None, **kwargs):
        super().__init__(api_key, base_url, **kwargs)
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )


class _FakeCerebrasBase(_FakeOpenAIBase):
    pass


class _FakeAnthropicBase(_FakeClientBase):
    def __init__(self, api_key=None, base_url=None, **kwargs):
        super().__init__(api_key, base_url, **kwargs)
        self.messages = SimpleNamespace(create=self._create)


FakeOpenAI = type("AsyncOpenAI", (_FakeOpenAIBase,), {})
FakeCerebras = type("AsyncCerebras", (_FakeCerebrasBase,), {})
FakeAnthropic = type("AsyncAnthropic", (_FakeAnthropicBase,), {})


def make_chain(client_class, observer=None):
    return TasksPromptsChain(
        [
            {
                "llm_id": "test",
                "llm_class": client_class,
                "model_options": {
                    "model": "fake-model",
                    "api_key": "test-key",
                    "temperature": 0.1,
                    "max_tokens": 128,
                },
            }
        ],
        final_result_placeholder="final",
        observer=observer,
    )


async def run_chain(chain, prompts):
    chunks = []
    async for chunk in chain.execute_chain(prompts):
        chunks.append(chunk)
    return chunks


def one_prompt(**overrides):
    prompt = {
        "prompt": "Hello",
        "output_format": "JSON",
        "output_placeholder": "phase",
        "llm_id": "test",
    }
    prompt.update(overrides)
    return prompt


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class JsonModeForwardingTests(unittest.IsolatedAsyncioTestCase):
    async def test_response_format_is_forwarded_to_openai(self):
        FakeOpenAI.configure([[FakeOpenAChunk("ok", finish_reason="stop")]])
        chain = make_chain(FakeOpenAI)

        await run_chain(
            chain,
            [one_prompt(response_format={"type": "json_object"})],
        )

        client = FakeOpenAI.instances[0]
        self.assertEqual(
            client.calls[0]["response_format"], {"type": "json_object"}
        )

    async def test_response_format_is_forwarded_to_cerebras(self):
        FakeCerebras.configure([[FakeOpenAChunk("ok", finish_reason="stop")]])
        chain = make_chain(FakeCerebras)

        await run_chain(
            chain,
            [one_prompt(response_format={"type": "json_object"})],
        )

        client = FakeCerebras.instances[0]
        self.assertEqual(
            client.calls[0]["response_format"], {"type": "json_object"}
        )

    async def test_anthropic_ignores_response_format(self):
        FakeAnthropic.configure(
            [
                [
                    FakeAnthropicChunk(
                        "content_block_delta",
                        FakeAnthropicDelta(text="ok"),
                    ),
                    FakeAnthropicChunk(
                        "message_delta",
                        FakeAnthropicDelta(stop_reason="end_turn"),
                    ),
                ]
            ]
        )
        chain = make_chain(FakeAnthropic)

        await run_chain(
            chain,
            [one_prompt(response_format={"type": "json_object"})],
        )

        client = FakeAnthropic.instances[0]
        self.assertNotIn("response_format", client.calls[0])


class JsonModeFallbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_rejection_retries_without_response_format(self):
        FakeOpenAI.configure(
            [
                FakeProviderError("Unsupported parameter: response_format"),
                [FakeOpenAChunk("ok", finish_reason="stop")],
            ]
        )
        chain = make_chain(FakeOpenAI)

        chunks = await run_chain(
            chain,
            [one_prompt(response_format={"type": "json_object"})],
        )

        client = FakeOpenAI.instances[0]
        self.assertIn("response_format", client.calls[0])
        self.assertNotIn("response_format", client.calls[1])
        self.assertIn("ok", "".join(chunks))

    async def test_rejection_is_cached_for_later_prompts(self):
        FakeOpenAI.configure(
            [
                FakeProviderError("Unsupported parameter: response_format"),
                [FakeOpenAChunk("first", finish_reason="stop")],
                [FakeOpenAChunk("second", finish_reason="stop")],
            ]
        )
        chain = make_chain(FakeOpenAI)

        await run_chain(
            chain,
            [
                one_prompt(
                    output_placeholder="one",
                    response_format={"type": "json_object"},
                )
            ],
        )
        await run_chain(
            chain,
            [
                one_prompt(
                    output_placeholder="two",
                    response_format={"type": "json_object"},
                )
            ],
        )

        client = FakeOpenAI.instances[0]
        self.assertIn("response_format", client.calls[0])
        self.assertNotIn("response_format", client.calls[1])
        # Second execute_chain must not probe again.
        self.assertNotIn("response_format", client.calls[2])

    async def test_json_schema_downgrades_to_json_object(self):
        FakeOpenAI.configure(
            [
                FakeProviderError("Unsupported response_format: json_schema"),
                [FakeOpenAChunk("ok", finish_reason="stop")],
            ]
        )
        chain = make_chain(FakeOpenAI)

        await run_chain(
            chain,
            [
                one_prompt(
                    response_format={
                        "type": "json_schema",
                        "json_schema": {"name": "x", "schema": {}},
                    }
                )
            ],
        )

        client = FakeOpenAI.instances[0]
        self.assertEqual(client.calls[0]["response_format"]["type"], "json_schema")
        self.assertEqual(
            client.calls[1]["response_format"], {"type": "json_object"}
        )

    async def test_stream_options_fallback_still_works(self):
        FakeOpenAI.configure(
            [
                FakeProviderError("Unsupported parameter: stream_options"),
                [FakeOpenAChunk("ok", finish_reason="stop")],
            ]
        )
        # include_usage is enabled by the observer, so the first request
        # really carries stream_options and the fallback path is exercised.
        chain = make_chain(FakeOpenAI, observer=lambda event: None)

        await run_chain(chain, [one_prompt()])

        client = FakeOpenAI.instances[0]
        self.assertIn("stream_options", client.calls[0])
        self.assertNotIn("stream_options", client.calls[1])


class FinishReasonTests(unittest.IsolatedAsyncioTestCase):
    async def test_openai_finish_reason_is_exposed(self):
        events = []
        FakeOpenAI.configure([[FakeOpenAChunk("ok", finish_reason="stop")]])
        chain = make_chain(FakeOpenAI, observer=events.append)

        await run_chain(chain, [one_prompt()])

        self.assertEqual(chain.get_finish_reason("phase"), "stop")
        prompt_events = [e for e in events if e.type == EVENT_PROMPT_END]
        self.assertEqual(prompt_events[0].finish_reason, "stop")

    async def test_anthropic_finish_reason_is_exposed(self):
        FakeAnthropic.configure(
            [
                [
                    FakeAnthropicChunk(
                        "content_block_delta",
                        FakeAnthropicDelta(text="ok"),
                    ),
                    FakeAnthropicChunk(
                        "message_delta",
                        FakeAnthropicDelta(stop_reason="end_turn"),
                    ),
                ]
            ]
        )
        chain = make_chain(FakeAnthropic)

        await run_chain(chain, [one_prompt()])

        self.assertEqual(chain.get_finish_reason("phase"), "end_turn")


class InstructionTests(unittest.IsolatedAsyncioTestCase):
    async def test_legacy_json_instruction_is_unchanged(self):
        FakeOpenAI.configure([[FakeOpenAChunk("ok", finish_reason="stop")]])
        chain = make_chain(FakeOpenAI)

        await run_chain(chain, [one_prompt()])

        client = FakeOpenAI.instances[0]
        content = client.calls[0]["messages"][-1]["content"]
        self.assertTrue(
            content.endswith("Please provide your response in JSON format.")
        )
        self.assertNotIn(
            "Return only a single valid JSON value", content
        )

    async def test_provider_json_mode_appends_hardening(self):
        FakeOpenAI.configure([[FakeOpenAChunk("ok", finish_reason="stop")]])
        chain = make_chain(FakeOpenAI)

        await run_chain(
            chain,
            [one_prompt(response_format={"type": "json_object"})],
        )

        client = FakeOpenAI.instances[0]
        content = client.calls[0]["messages"][-1]["content"]
        self.assertIn("Return only a single valid JSON value", content)
        self.assertIn("Do not use markdown code fences", content)


if __name__ == "__main__":
    unittest.main()
