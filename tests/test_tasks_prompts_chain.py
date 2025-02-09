import pytest
from tasks_prompts_chain import TasksPromptsChain, PromptTemplate, OutputFormat
from unittest.mock import AsyncMock, MagicMock

class MockResponse:
    def __init__(self, content):
        self.choices = [MagicMock(delta=MagicMock(content=content))]

@pytest.fixture
def mock_chain():
    return TasksPromptsChain(
        model_options={
            "model": "gpt-3.5-turbo",
            "api_key": "test-key",
            "temperature": 0.7,
            "max_tokens": 4120,
            "stream": True
        },
        final_result_placeholder="final_result"
    )

@pytest.fixture
def mock_chain_with_system():
    return TasksPromptsChain(
        model_options={
            "model": "gpt-3.5-turbo",
            "api_key": "test-key",
            "temperature": 0.7,
            "max_tokens": 4120,
            "stream": True
        },
        system_prompt="You are a test assistant",
        final_result_placeholder="final_result",
        system_apply_to_all_prompts=True
    )

def test_prompt_template_initialization():
    template = PromptTemplate(
        prompt="Test prompt",
        output_format="JSON",
        output_placeholder="test_result"
    )
    
    assert template.prompt == "Test prompt"
    assert template.output_format == OutputFormat.JSON
    assert template.output_placeholder == "test_result"

def test_prompt_chain_initialization(mock_chain):
    assert mock_chain.model == "gpt-3.5-turbo"
    assert mock_chain.final_result_placeholder == "final_result"
    assert mock_chain._results == {}
    assert mock_chain._output_template is None

def test_prompt_chain_with_system_initialization(mock_chain_with_system):
    assert mock_chain_with_system.system_prompt == "You are a test assistant"
    assert mock_chain_with_system.system_apply_to_all_prompts is True

def test_template_output_before_execution(mock_chain):
    template = "<result>{{test_placeholder}}</result>"
    mock_chain.template_output(template)
    assert mock_chain._output_template == template

def test_template_output_after_execution(mock_chain):
    mock_chain._results = {"some_result": "value"}
    with pytest.raises(Exception) as exc_info:
        mock_chain.template_output("some template")
    assert str(exc_info.value) == "template_output must be called before execute_chain"

def test_get_result_nonexistent_placeholder(mock_chain):
    assert mock_chain.get_result("nonexistent") is None

@pytest.mark.asyncio
async def test_execute_chain_basic(mocker):
    chain = TasksPromptsChain(
        model_options={
            "model": "gpt-3.5-turbo",
            "api_key": "test-key",
            "temperature": 0.7,
            "max_tokens": 4120,
            "stream": True
        },
        final_result_placeholder="final"
    )
    
    # Create mock responses
    mock_stream = [
        MockResponse("This is "),
        MockResponse("a test "),
        MockResponse("response")
    ]
    
    # Create a proper async generator
    async def mock_stream_generator():
        for item in mock_stream:
            yield item
    
    # Create a mock for the create method
    mock_create = AsyncMock()
    mock_create.return_value = type('AsyncIterator', (), {
        '__aiter__': lambda self: mock_stream_generator(),
    })()
    
    # Patch the create method
    mocker.patch.object(
        chain.client.chat.completions,
        'create',
        mock_create
    )
    
    prompts = [
        {
            "prompt": "Test prompt",
            "output_format": "TEXT",
            "output_placeholder": "result1"
        }
    ]
    
    responses = []
    async for response in chain.execute_chain(prompts):
        responses.append(response)
    
    assert len(responses) == 3
    assert responses[-1] == "This is a test response"
    assert chain.get_result("result1") == "This is a test response"

def test_output_format_enum():
    assert OutputFormat.JSON.value == "JSON"
    assert OutputFormat.MARKDOWN.value == "MARKDOWN"
    assert OutputFormat.CSV.value == "CSV"
    assert OutputFormat.TEXT.value == "TEXT"
import pytest
from tasks_prompts_chain import TasksPromptsChain, PromptTemplate, OutputFormat
from typing import List, Dict
import json
import asyncio
from unittest.mock import AsyncMock, MagicMock

class MockResponse:
    def __init__(self, content):
        self.choices = [MagicMock(delta=MagicMock(content=content))]

@pytest.fixture
def mock_chain():
    return TasksPromptsChain(
        model_options={
            "model": "gpt-3.5-turbo",
            "api_key": "test-key",
            "temperature": 0.7,
            "max_tokens": 4120,
            "stream": True
        },
        final_result_placeholder="final_result"
    )

@pytest.fixture
def mock_chain_with_system():
    return TasksPromptsChain(
        model_options={
            "model": "gpt-3.5-turbo",
            "api_key": "test-key",
            "temperature": 0.7,
            "max_tokens": 4120,
            "stream": True
        },
        final_result_placeholder="final_result",
        system_prompt="You are a test assistant",
        system_apply_to_all_prompts=True
    )

def test_prompt_template_initialization():
    template = PromptTemplate(
        prompt="Test prompt",
        output_format="JSON",
        output_placeholder="test_result"
    )
    
    assert template.prompt == "Test prompt"
    assert template.output_format == OutputFormat.JSON
    assert template.output_placeholder == "test_result"

def test_prompt_chain_initialization(mock_chain):
    assert mock_chain.model == "gpt-3.5-turbo"
    assert mock_chain.final_result_placeholder == "final_result"
    assert mock_chain._results == {}
    assert mock_chain._output_template is None

def test_prompt_chain_with_system_initialization(mock_chain_with_system):
    assert mock_chain_with_system.system_prompt == "You are a test assistant"
    assert mock_chain_with_system.system_apply_to_all_prompts is True

def test_template_output_before_execution(mock_chain):
    template = "<result>{{test_placeholder}}</result>"
    mock_chain.template_output(template)
    assert mock_chain._output_template == template

def test_template_output_after_execution(mock_chain):
    mock_chain._results = {"some_result": "value"}
    with pytest.raises(Exception) as exc_info:
        mock_chain.template_output("some template")
    assert str(exc_info.value) == "template_output must be called before execute_chain"

def test_get_result_nonexistent_placeholder(mock_chain):
    assert mock_chain.get_result("nonexistent") is None

def test_output_format_enum():
    assert OutputFormat.JSON.value == "JSON"
    assert OutputFormat.MARKDOWN.value == "MARKDOWN"
    assert OutputFormat.CSV.value == "CSV"
    assert OutputFormat.TEXT.value == "TEXT"
