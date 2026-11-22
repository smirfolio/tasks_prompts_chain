"""Tests for optional reasoning_effort passthrough in ClientLLMSDK."""

import os
import sys
from types import SimpleNamespace

import pytest

# Make the src layout importable when tests are run from the repository root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from tasks_prompts_chain import TasksPromptsChain
from tasks_prompts_chain.client_llm_sdk import ClientLLMSDK


class FakeCompletions:
    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)

        async def stream():
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="ok")
                    )
                ]
            )

        return stream()


def make_client_class(class_name: str):
    class FakeClient:
        def __init__(self, **kwargs):
            self.init_kwargs = kwargs
            self.completions = FakeCompletions()
            self.chat = SimpleNamespace(completions=self.completions)

    FakeClient.__name__ = class_name
    return FakeClient


@pytest.mark.asyncio
async def test_openai_forwards_reasoning_effort():
    client_class = make_client_class("AsyncOpenAI")
    sdk = ClientLLMSDK(
        client_class,
        {"api_key": "key", "reasoning_effort": "high"},
    )

    chunks = [
        chunk
        async for chunk in sdk.generat_response(
            model="model",
            messages=[],
            stream=True,
        )
    ]

    assert chunks == ["ok"]
    assert sdk.client.completions.calls[0]["reasoning_effort"] == "high"


@pytest.mark.asyncio
async def test_openai_omits_reasoning_effort_when_absent():
    client_class = make_client_class("AsyncOpenAI")
    sdk = ClientLLMSDK(client_class, {"api_key": "key"})

    _ = [
        chunk
        async for chunk in sdk.generat_response(
            model="model",
            messages=[],
            stream=True,
        )
    ]

    assert "reasoning_effort" not in sdk.client.completions.calls[0]


@pytest.mark.asyncio
async def test_cerebras_forwards_reasoning_effort():
    client_class = make_client_class("AsyncCerebras")
    sdk = ClientLLMSDK(
        client_class,
        {"api_key": "key", "reasoning_effort": "max"},
    )

    _ = [
        chunk
        async for chunk in sdk.generat_response(
            model="model",
            messages=[],
            stream=True,
        )
    ]

    assert sdk.client.completions.calls[0]["reasoning_effort"] == "max"


def test_chain_constructor_forwards_request_option():
    client_class = make_client_class("AsyncOpenAI")
    chain = TasksPromptsChain(
        [
            {
                "llm_id": "user",
                "llm_class": client_class,
                "model_options": {
                    "model": "model",
                    "api_key": "key",
                    "reasoning_effort": "high",
                },
            }
        ],
        final_result_placeholder="final",
    )

    assert chain.clients["user"]["client"].reasoning_effort == "high"
