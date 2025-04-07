import pytest
import sys
import os

# Add the parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.tasks_prompts_chain.tasks_prompts_chain import TasksPromptsChain, PromptTemplate, OutputFormat
from unittest.mock import MagicMock, patch
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from cerebras.cloud.sdk import AsyncCerebras

class MockTextDelta:
    def __init__(self, text):
        self.text = text
        self.type = "text_delta"

class MockDelta:
    def __init__(self, text):
        self.text = text

class MockAnthropicResponse:
    def __init__(self, text):
        self.type = "content_block_delta"
        self.delta = MockDelta(text)
        self.index = 0

class MockOpenAIResponse:
    def __init__(self, content):
        self.choices = [MagicMock(delta=MagicMock(content=content))]

@pytest.fixture
def mock_anthropic_chain():
    llm_configs=[
            {
                "llm_id": "claude",
                "llm_class": AsyncAnthropic,
                "model_options": {
                    "model": "claude-3-haiku-20240307",
                    "api_key": "test-key",
                    "temperature": 0.7,
                    "max_tokens": 4120,
                }
            }
        ]
    return TasksPromptsChain(
        llm_configs,
        "",
        "final_result"
    )

@pytest.fixture
def mock_openai_chain():
    llm_configs=[
            {
                "llm_id": "gpt",
                "llm_class": AsyncOpenAI,
                "model_options": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "test-key",
                    "temperature": 0.7,
                    "max_tokens": 4120,
                }
            }
        ]
    return TasksPromptsChain(
        llm_configs,
        "",
        "final_result"
    )

@pytest.fixture
def mock_cerebras_chain():
    llm_configs=[
            {
                "llm_id": "llama",
                "llm_class": AsyncCerebras,
                "model_options": {
                    "model": "llama-3.3-70b",
                    "api_key": "test-key",
                    "temperature": 0.7,
                    "max_tokens": 4120,
                }
            }
        ]
    return TasksPromptsChain(
        llm_configs,
        "",
        "final_result"
    )

@pytest.fixture
def mock_chain_with_system():
    llm_configs=[
            {
                "llm_id": "gpt",
                "llm_class": AsyncOpenAI,
                "model_options": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "test-key",
                    "temperature": 0.7,
                    "max_tokens": 4120,
                }
            }
        ]
    return TasksPromptsChain(
        llm_configs,
        "You are a test assistant",
        "final_result",
        True
    )

def test_prompt_template_initialization():
    template = PromptTemplate(
        prompt="Test prompt",
        output_format="JSON",
        output_placeholder="test_result",
    )
    
    assert template.prompt == "Test prompt"
    assert template.output_format == OutputFormat.JSON
    assert template.output_placeholder == "test_result"

### Test OpenAi
def test_openai_prompt_chain_initialization(mock_openai_chain):
    assert isinstance(mock_openai_chain, TasksPromptsChain)
    assert mock_openai_chain.final_result_placeholder == "final_result"
    assert mock_openai_chain._results == {}
    assert mock_openai_chain._output_template is None

def test_openai_template_output_before_execution(mock_openai_chain):
    template = "<r>{{test_placeholder}}</r>"
    mock_openai_chain.template_output(template)
    assert mock_openai_chain._output_template == template

def test_openai_template_output_after_execution(mock_openai_chain):
    mock_openai_chain._results = {"some_result": "value"}
    with pytest.raises(Exception) as exc_info:
        mock_openai_chain.template_output("some template")
    assert str(exc_info.value) == "template_output must be called before execute_chain"

def test_openai_get_result_nonexistent_placeholder(mock_openai_chain):
    assert mock_openai_chain.get_result("nonexistent") is None

##### test anthropic
def test_anthropic_prompt_chain_initialization(mock_anthropic_chain):
    assert isinstance(mock_anthropic_chain, TasksPromptsChain)
    assert mock_anthropic_chain.final_result_placeholder == "final_result"
    assert mock_anthropic_chain._results == {}
    assert mock_anthropic_chain._output_template is None

def test_anthropic_template_output_before_execution(mock_anthropic_chain):
    template = "<r>{{test_placeholder}}</r>"
    mock_anthropic_chain.template_output(template)
    assert mock_anthropic_chain._output_template == template

def test_anthropic_template_output_after_execution(mock_anthropic_chain):
    mock_anthropic_chain._results = {"some_result": "value"}
    with pytest.raises(Exception) as exc_info:
        mock_anthropic_chain.template_output("some template")
    assert str(exc_info.value) == "template_output must be called before execute_chain"

def test_anthropic_get_result_nonexistent_placeholder(mock_anthropic_chain):
    assert mock_anthropic_chain.get_result("nonexistent") is None

def test_prompt_chain_with_system_initialization(mock_chain_with_system):
    assert mock_chain_with_system.system_prompt == "You are a test assistant"
    assert mock_chain_with_system.system_apply_to_all_prompts is True


##### test cerebras
def test_anthropic_prompt_chain_initialization(mock_anthropic_chain):
    assert isinstance(mock_anthropic_chain, TasksPromptsChain)
    assert mock_anthropic_chain.final_result_placeholder == "final_result"
    assert mock_anthropic_chain._results == {}
    assert mock_anthropic_chain._output_template is None

def test_anthropic_template_output_before_execution(mock_anthropic_chain):
    template = "<r>{{test_placeholder}}</r>"
    mock_anthropic_chain.template_output(template)
    assert mock_anthropic_chain._output_template == template

def test_anthropic_template_output_after_execution(mock_anthropic_chain):
    mock_anthropic_chain._results = {"some_result": "value"}
    with pytest.raises(Exception) as exc_info:
        mock_anthropic_chain.template_output("some template")
    assert str(exc_info.value) == "template_output must be called before execute_chain"

def test_anthropic_get_result_nonexistent_placeholder(mock_anthropic_chain):
    assert mock_anthropic_chain.get_result("nonexistent") is None

def test_prompt_chain_with_system_initialization(mock_chain_with_system):
    assert mock_chain_with_system.system_prompt == "You are a test assistant"
    assert mock_chain_with_system.system_apply_to_all_prompts is True

##### test cerebras
def test_cerebras_prompt_chain_initialization(mock_cerebras_chain):
    assert isinstance(mock_cerebras_chain, TasksPromptsChain)
    assert mock_cerebras_chain.final_result_placeholder == "final_result"
    assert mock_cerebras_chain._results == {}
    assert mock_cerebras_chain._output_template is None

def test_cerebras_template_output_before_execution(mock_cerebras_chain):
    template = "<r>{{test_placeholder}}</r>"
    mock_cerebras_chain.template_output(template)
    assert mock_cerebras_chain._output_template == template

def test_cerebras_template_output_after_execution(mock_cerebras_chain):
    mock_cerebras_chain._results = {"some_result": "value"}
    with pytest.raises(Exception) as exc_info:
        mock_cerebras_chain.template_output("some template")
    assert str(exc_info.value) == "template_output must be called before execute_chain"

def test_cerebras_get_result_nonexistent_placeholder(mock_cerebras_chain):
    assert mock_cerebras_chain.get_result("nonexistent") is None

def test_prompt_chain_with_system_initialization(mock_chain_with_system):
    assert mock_chain_with_system.system_prompt == "You are a test assistant"
    assert mock_chain_with_system.system_apply_to_all_prompts is True
    
def test_output_format_enum():
    assert OutputFormat.JSON.value == "JSON"
    assert OutputFormat.MARKDOWN.value == "MARKDOWN"
    assert OutputFormat.CSV.value == "CSV"
    assert OutputFormat.TEXT.value == "TEXT"

@pytest.mark.asyncio
async def test_execute_openai_chain():
    llm_configs=[
            {
                "llm_id": "gpt",
                "llm_class": AsyncOpenAI,
                "model_options": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "test-key",
                    "temperature": 0.7,
                    "max_tokens": 4120,
                }
            }
        ]
    chain = TasksPromptsChain(
        llm_configs,
        final_result_placeholder="final"
    )
    
    # Create mock responses
    mock_stream = [
        MockOpenAIResponse("This is "),
        MockOpenAIResponse("a test "),
        MockOpenAIResponse("response")
    ]
    
    # Create a proper async generator
    async def mock_stream_generator():
        for item in mock_stream:
            yield item
    
    # Set up the mock
    with patch('src.tasks_prompts_chain.client_llm_sdk.ClientLLMSDK.generat_response') as mock_generate:
        # Create an async generator mock
        async def mock_gen(**kwargs):
            yield "This is "
            yield "a test "
            yield "response"
        
        mock_generate.return_value = mock_gen()
        
        prompts = [
            {
                "prompt": "Test prompt",
                "output_format": "TEXT",
                "llm_id": "gpt",
                "output_placeholder": "result1"
            }
        ]
        
        responses = []
        async for response in chain.execute_chain(prompts):
            responses.append(response)
        
        # Check responses excluding the final system message
        actual_responses = [r for r in responses if r != "<tasks-sys>Done</tasks-sys>"]
        assert len(actual_responses) == 3
        assert actual_responses[0] == "This is "
        assert actual_responses[1] == "a test "
        assert actual_responses[2] == "response"
        assert chain.get_result("result1") == "This is a test response"

@pytest.mark.asyncio
async def test_execute_anthropic_chain():
    llm_configs=[
            {
              "llm_id": "claude",
              "llm_class": AsyncAnthropic,
              "model_options": {
                "model": "claude-3-haiku-20240307",
                "api_key": "test-key",
                "temperature": 0.7,
                "max_tokens": 4120,
                }
            }
        ]
    chain = TasksPromptsChain(
        llm_configs,
        final_result_placeholder="final"
    )
    
    with patch('src.tasks_prompts_chain.client_llm_sdk.ClientLLMSDK.generat_response') as mock_generate:
        # Create an async generator mock
        async def mock_gen(**kwargs):
            yield "This is "
            yield "a test "
            yield "response"
        
        mock_generate.return_value = mock_gen()
        
        prompts = [
            {
                "prompt": "Test prompt",
                "output_format": "TEXT",
                "output_placeholder": "result1",
                "llm_id": "claude"
            }
        ]
        
        responses = []
        async for response in chain.execute_chain(prompts):
            responses.append(response)
        
        # Check responses excluding the final system message
        actual_responses = [r for r in responses if r != "<tasks-sys>Done</tasks-sys>"]
        assert len(actual_responses) == 3
        assert actual_responses[0] == "This is "
        assert actual_responses[1] == "a test "
        assert actual_responses[2] == "response"
        assert chain.get_result("result1") == "This is a test response"