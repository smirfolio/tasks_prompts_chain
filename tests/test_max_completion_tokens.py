"""Deterministic max_tokens / max_completion_tokens selection tests.

There is no error-driven retry. The chain chooses the parameter before the
request:

- OpenAI SDK + official OpenAI endpoint -> max_completion_tokens
- OpenAI SDK + custom base URL -> max_tokens
- Anthropic native / Cerebras -> max_tokens
- model_options["max_tokens_parameter"] can override the OpenAI choice
"""

from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from tasks_prompts_chain import TasksPromptsChain


class FakeProviderError(Exception):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


class FakeChunk:
    def __init__(self, content="ok", finish_reason="stop"):
        self.choices = [
            SimpleNamespace(
                delta=SimpleNamespace(content=content),
                finish_reason=finish_reason,
            )
        ]
        self.usage = None


class _FakeClientBase:
    instances = []

    @classmethod
    def reset(cls):
        cls.instances = []

    def __init__(self, api_key=None, base_url=None, **kwargs):
        self.calls = []
        type(self).instances.append(self)

    async def _create(self, **kwargs):
        self.calls.append(dict(kwargs))
        if getattr(type(self), "raise_error", None):
            raise type(self).raise_error

        async def gen():
            yield FakeChunk()

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


def make_chain(client_class, **model_options):
    options = {
        "model": "fake-model",
        "api_key": "test-key",
        "temperature": 0.1,
    }
    options.update(model_options)
    return TasksPromptsChain(
        [
            {
                "llm_id": "test",
                "llm_class": client_class,
                "model_options": options,
            }
        ],
        final_result_placeholder="final",
    )


async def run_chain(chain):
    async for _chunk in chain.execute_chain(
        [
            {
                "prompt": "Hello",
                "output_format": "TEXT",
                "output_placeholder": "phase",
                "llm_id": "test",
            }
        ]
    ):
        pass


class TokenParameterSelectionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        FakeOpenAI.reset()
        FakeCerebras.reset()
        FakeAnthropic.reset()
        FakeOpenAI.raise_error = None
        FakeCerebras.raise_error = None
        FakeAnthropic.raise_error = None

    async def test_openai_official_uses_max_completion_tokens(self):
        chain = make_chain(FakeOpenAI)

        await run_chain(chain)

        call = FakeOpenAI.instances[0].calls[0]
        self.assertEqual(call["max_completion_tokens"], 6500)
        self.assertNotIn("max_tokens", call)

    async def test_openai_custom_base_url_uses_max_tokens(self):
        chain = make_chain(
            FakeOpenAI, base_url="https://example.com/v1"
        )

        await run_chain(chain)

        call = FakeOpenAI.instances[0].calls[0]
        self.assertEqual(call["max_tokens"], 6500)
        self.assertNotIn("max_completion_tokens", call)

    async def test_explicit_max_completion_tokens_value_official(self):
        chain = make_chain(FakeOpenAI, max_completion_tokens=4096)

        await run_chain(chain)

        call = FakeOpenAI.instances[0].calls[0]
        self.assertEqual(call["max_completion_tokens"], 4096)
        self.assertNotIn("max_tokens", call)

    async def test_explicit_max_completion_tokens_value_custom_endpoint(self):
        chain = make_chain(
            FakeOpenAI,
            base_url="https://example.com/v1",
            max_completion_tokens=4096,
        )

        await run_chain(chain)

        call = FakeOpenAI.instances[0].calls[0]
        self.assertEqual(call["max_tokens"], 4096)
        self.assertNotIn("max_completion_tokens", call)

    async def test_max_tokens_parameter_override_for_official_openai(self):
        chain = make_chain(
            FakeOpenAI, max_tokens_parameter="max_tokens"
        )

        await run_chain(chain)

        call = FakeOpenAI.instances[0].calls[0]
        self.assertEqual(call["max_tokens"], 6500)
        self.assertNotIn("max_completion_tokens", call)

    async def test_max_tokens_parameter_override_for_custom_endpoint(self):
        chain = make_chain(
            FakeOpenAI,
            base_url="https://example.com/v1",
            max_tokens_parameter="max_completion_tokens",
        )

        await run_chain(chain)

        call = FakeOpenAI.instances[0].calls[0]
        self.assertEqual(call["max_completion_tokens"], 6500)
        self.assertNotIn("max_tokens", call)

    async def test_both_values_prefers_max_completion_tokens(self):
        chain = make_chain(
            FakeOpenAI,
            max_tokens=1024,
            max_completion_tokens=4096,
        )

        await run_chain(chain)

        call = FakeOpenAI.instances[0].calls[0]
        self.assertEqual(call["max_completion_tokens"], 4096)
        self.assertNotIn("max_tokens", call)

    async def test_cerebras_always_uses_max_tokens(self):
        chain = make_chain(
            FakeCerebras,
            max_completion_tokens=4096,
            max_tokens_parameter="max_completion_tokens",
        )

        await run_chain(chain)

        call = FakeCerebras.instances[0].calls[0]
        self.assertEqual(call["max_tokens"], 6500)
        self.assertNotIn("max_completion_tokens", call)

    async def test_anthropic_always_uses_max_tokens(self):
        chain = make_chain(
            FakeAnthropic,
            max_completion_tokens=4096,
            max_tokens_parameter="max_completion_tokens",
        )

        await run_chain(chain)

        call = FakeAnthropic.instances[0].calls[0]
        self.assertEqual(call["max_tokens"], 6500)
        self.assertNotIn("max_completion_tokens", call)

    async def test_no_error_driven_retry(self):
        FakeOpenAI.raise_error = FakeProviderError(
            "Unsupported parameter: max_tokens. "
            "Use max_completion_tokens instead."
        )
        chain = make_chain(FakeOpenAI)

        with self.assertRaises(Exception):
            await run_chain(chain)

        # Exactly one request: the chain does not retry with another
        # token-limit parameter.
        self.assertEqual(len(FakeOpenAI.instances[0].calls), 1)


if __name__ == "__main__":
    unittest.main()
