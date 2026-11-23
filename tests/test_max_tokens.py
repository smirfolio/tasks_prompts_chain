"""Token-limit fallback tests.

The chain must always send a positive token limit. Missing, None, zero, and
negative inputs fall back to 6500. The parameter name depends on the SDK and
endpoint: this file uses a custom OpenAI-compatible base URL, so it exercises
the max_tokens path for all three providers.

Run from the repository root:

    pytest tests/test_max_tokens.py -v
"""

from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from tasks_prompts_chain import TasksPromptsChain
from tasks_prompts_chain.client_llm_sdk import DEFAULT_MAX_TOKENS


_UNSET = object()


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

        async def gen():
            yield _good_chunk()

        return gen()


class _FakeOpenAI(_FakeClientBase):
    def __init__(self, api_key=None, base_url=None, **kwargs):
        super().__init__(api_key, base_url, **kwargs)
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )


class _FakeCerebras(_FakeOpenAI):
    pass


class _FakeAnthropic(_FakeClientBase):
    def __init__(self, api_key=None, base_url=None, **kwargs):
        super().__init__(api_key, base_url, **kwargs)
        self.messages = SimpleNamespace(create=self._create)


FakeOpenAI = type("AsyncOpenAI", (_FakeOpenAI,), {})
FakeCerebras = type("AsyncCerebras", (_FakeCerebras,), {})
FakeAnthropic = type("AsyncAnthropic", (_FakeAnthropic,), {})


def _good_chunk():
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content="ok"),
                finish_reason="stop",
            )
        ],
        usage=None,
    )


def make_chain(client_class, max_tokens=_UNSET):
    model_options = {
        "model": "fake-model",
        "api_key": "test-key",
        # Custom base URL -> OpenAI SDK uses max_tokens, so this file can
        # assert the same parameter for OpenAI, Cerebras, and Anthropic.
        "base_url": "https://example.com/v1",
        "temperature": 0.1,
    }
    if max_tokens is not _UNSET:
        model_options["max_tokens"] = max_tokens
    return TasksPromptsChain(
        [
            {
                "llm_id": "test",
                "llm_class": client_class,
                "model_options": model_options,
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


class MaxTokensFallbackTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        FakeOpenAI.reset()
        FakeCerebras.reset()
        FakeAnthropic.reset()

    async def _assert_fallback(self, client_class):
        for value in (_UNSET, None, 0, -1):
            with self.subTest(value=value):
                client_class.reset()
                chain = make_chain(client_class, value)
                await run_chain(chain)
                self.assertEqual(
                    client_class.instances[0].calls[0]["max_tokens"],
                    DEFAULT_MAX_TOKENS,
                )

    async def test_openai_falls_back_to_6500(self):
        await self._assert_fallback(FakeOpenAI)

    async def test_cerebras_falls_back_to_6500(self):
        await self._assert_fallback(FakeCerebras)

    async def test_anthropic_falls_back_to_6500(self):
        await self._assert_fallback(FakeAnthropic)

    async def test_explicit_values_are_forwarded(self):
        for client_class in (FakeOpenAI, FakeCerebras, FakeAnthropic):
            with self.subTest(client=client_class.__name__):
                client_class.reset()
                chain = make_chain(client_class, 1024)
                await run_chain(chain)
                self.assertEqual(
                    client_class.instances[0].calls[0]["max_tokens"],
                    1024,
                )


if __name__ == "__main__":
    unittest.main()
