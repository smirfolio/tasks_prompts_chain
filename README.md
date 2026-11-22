# TasksPromptsChain
[![PyPI Downloads](https://static.pepy.tech/badge/tasks-prompts-chain)](https://pepy.tech/projects/tasks-prompts-chain)

A Mini Python library for creating and executing chains of prompts using multiple LLM providers with streaming support and output template formatting.

## Features

- Sequential prompt chain execution
- Streaming responses
- Template-based output formatting
- System prompt support
- Placeholder replacement between prompts
- Stop placeholder string, that can be defined, so if the LLM responds with this placeholder, the chain prompt will interrupt, as an LLM error handler
- Multiple output formats (JSON, Markdown, CSV, Text)
- Async/await support
- Support for multiple LLM providers (OpenAI, Anthropic, Cerebras, etc.)
- Multi-model support - use different models for different prompts in the chain
- Observer / usage telemetry with per-prompt metadata and normalized token usage

## Dependencies

Please install typing-extensions and the SDK for your preferred LLM providers:

For OpenAI:
```bash
pip install typing-extensions
pip install openai
```

For Anthropic:
```bash
pip install typing-extensions
pip install anthropic
```

For Cerebras:
```bash
pip install typing-extensions
pip install cerebras
```

To Install the library: 
```
pip install tasks_prompts_chain
```

## Installation from source code

### For Users required from source gitHub repo
```bash
pip install -r requirements/requirements.txt
```

### For Developers required from source gitHub repo
```bash
pip install -r requirements/requirements.txt
pip install -r requirements/requirements-dev.txt
```

## Quick Start

```python
from tasks_prompts_chain import TasksPromptsChain
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from cerebras import AsyncCerebras

async def main():
    # Initialize the chain with multiple LLM configurations
    llm_configs = [
        {
            "llm_id": "gpt",  # Unique identifier for this LLM
            "llm_class": AsyncOpenAI,  # LLM SDK class
            "model_options": {
                "model": "gpt-4o",
                "api_key": "your-openai-api-key",
                "temperature": 0.1,
                "max_tokens": 4120,
            }
        },
        {
            "llm_id": "claude",  # Unique identifier for this LLM
            "llm_class": AsyncAnthropic,  # LLM SDK class
            "model_options": {
                "model": "claude-3-sonnet-20240229",
                "api_key": "your-anthropic-api-key",
                "temperature": 0.1,
                "max_tokens": 8192,
            }
        },
        {
            "llm_id": "llama",  # Unique identifier for this LLM
            "llm_class": AsyncCerebras,  # LLM SDK class
            "model_options": {
                "model": "llama-3.3-70b",
                "api_key": "your-cerebras-api-key",
                "base_url": "https://api.cerebras.ai/v1",
                "temperature": 0.1,
                "max_tokens": 4120,
            }
        }
    ]

    chain = TasksPromptsChain(
        llm_configs,
        final_result_placeholder="design_result"
    )

    # Define your prompts - specify which LLM to use for each prompt
    prompts = [
        {
            "prompt": "Create a design concept for a luxury chocolate bar, if not inspired respond with this string %%cant_be_inspired%% so the chain prompt query will be stopped",
            "output_format": "TEXT",
            "output_placeholder": "design_concept",
            "llm_id": "gpt",  # Use the GPT model for this prompt
            "stop_placeholder": "%%cant_be_inspired%%" # the stop placeholder string that will interrupt the prompt chain query
        },
        {
            "prompt": "Based on this concept: {{design_concept}}, suggest a color palette",
            "output_format": "JSON",
            "output_placeholder": "color_palette",
            "llm_id": "claude"  # Use the Claude model for this prompt
        },
        {
            "prompt": "Based on the design and colors: {{design_concept}} and {{color_palette}}, suggest packaging materials",
            "output_format": "MARKDOWN",
            "output_placeholder": "packaging",
            "llm_id": "llama"  # Use the Cerebras model for this prompt
        }
    ]

    # Stream the responses
    async for chunk in chain.execute_chain(prompts):
        print(chunk, end="", flush=True)

    # Get specific results
    design = chain.get_result("design_concept")
    colors = chain.get_result("color_palette")
    packaging = chain.get_result("packaging")
```

## Advanced Usage

### Using System Prompts

```python
chain = TasksPromptsChain(
    llm_configs=[
        {
            "llm_id": "default_model",
            "llm_class": AsyncOpenAI,
            "model_options": {
                "model": "gpt-4o",
                "api_key": "your-openai-api-key",
                "temperature": 0.1,
                "max_tokens": 4120,
            }
        }
    ],
    final_result_placeholder="result",
    system_prompt="You are a professional design expert specialized in luxury products",
    system_apply_to_all_prompts=True
)
```

### Using Cerebras Models

```python
from cerebras import AsyncCerebras

llm_configs = [
    {
        "llm_id": "cerebras",
        "llm_class": AsyncCerebras,
        "model_options": {
            "model": "llama-3.3-70b",
            "api_key": "your-cerebras-api-key",
            "base_url": "https://api.cerebras.ai/v1",
            "temperature": 0.1,
            "max_tokens": 4120,
        }
    }
]

chain = TasksPromptsChain(
    llm_configs,
    final_result_placeholder="result",
)
```

### Custom API Endpoint

```python
llm_configs = [
    {
        "llm_id": "custom_endpoint",
        "llm_class": AsyncOpenAI,
        "model_options": {
            "model": "your-custom-model",
            "api_key": "your-api-key",
            "base_url": "https://your-custom-endpoint.com/v1",
            "temperature": 0.1,
            "max_tokens": 4120,
        }
    }
]

chain = TasksPromptsChain(
    llm_configs,
    final_result_placeholder="result",
)
```

### Using Templates

You must call this set method before the execution of the prompting query (chain.execute_chain(prompts))

```python
# Set output template before execution
chain.template_output("""
<result>
    <design>
    ### Design Concept:
    {{design_concept}}
    </design>
    
    <colors>
    ### Color Palette:
    {{color_palette}}
    </colors>
</result>
""")
```
then retrieves the final result within the template : 

```python
# print out the final result in the well formated template
print(chain.get_final_result_within_template())
```


## Observer / usage telemetry

Attach a synchronous observer to receive structured timing and token usage for
every prompt and for the whole chain. No wrapping or monkey-patching of the
provider SDK is required.

```python
from tasks_prompts_chain import TasksPromptsChain, EVENT_PROMPT_END, EVENT_CHAIN_END


def observer(event):
    if event.type == EVENT_PROMPT_END:
        print(
            event.prompt_index,
            event.output_placeholder,
            event.metadata.get("request_kind"),
            event.usage.prompt_tokens,
            event.usage.completion_tokens,
            event.duration_ms,
            event.first_token_ms,
        )
    elif event.type == EVENT_CHAIN_END:
        print("chain total", event.usage)


async def main():
    chain = TasksPromptsChain(
        llm_configs,
        observer=observer,
        # include_usage defaults to True when an observer is set
    )

    async for chunk in chain.execute_chain(
        [
            {
                "prompt": "List the key features.",
                "output_placeholder": "key_features",
                "metadata": {"request_kind": "key_feature"},
            },
        ]
    ):
        print(chunk, end="", flush=True)
```

### Event types

| Event | When | `prompt_index` |
|---|---|---|
| `prompt_end` | Once per prompt, after success or failure | 1-based prompt number |
| `chain_end` | Once per `execute_chain` call, in a `finally` | `prompt_count` |

`chain_end` usage is an aggregate of the successful prompt usages. Use the
`prompt_end` events when per-prompt granularity is needed. Every `execute_chain`
call produces a fresh set of events, including calls made by an outer retry
loop.

### `ChainEvent` fields

| Field | Description |
|---|---|
| `type` | `"prompt_end"` or `"chain_end"` |
| `prompt_index` | 1-based prompt number; equals `prompt_count` for `chain_end` |
| `prompt_count` | Number of prompts in the chain |
| `output_placeholder` | Prompt output placeholder (`final_result_placeholder` for `chain_end`) |
| `llm_id` | LLM configuration ID (`None` for `chain_end`) |
| `model` | Model name (`None` for `chain_end`) |
| `metadata` | Copy of the per-prompt metadata (`{}` for `chain_end`) |
| `outcome` | `"ok"` or `"error"` |
| `started_at` / `ended_at` | Timezone-aware UTC timestamps |
| `duration_ms` | Wall-clock duration in milliseconds |
| `first_token_ms` | Time to first non-empty text delta, or `None` |
| `usage` | Normalized `TokenUsage` |
| `response_chars` | Number of response characters (no response text is included) |
| `final_result_placeholder` | Configured final result placeholder |
| `error` / `error_type` | Original exception and its class name on failure |

Events never contain prompt text, response text, messages, API keys, or
provider URLs.

### `TokenUsage` fields

| Field | Description |
|---|---|
| `prompt_tokens` | Input tokens |
| `completion_tokens` | Output tokens |
| `total_tokens` | Total tokens (derived when the provider omits it) |
| `cached_tokens` | Cached input tokens, when reported |
| `reasoning_tokens` | Reasoning tokens, when reported |
| `source` | `"provider"` when a usage payload was received, otherwise `"missing"` |

### Usage request behavior

- With an observer, the chain requests
  `stream_options={"include_usage": True}` from OpenAI-compatible providers.
- Pass `include_usage=False` to keep the original request shape and receive
  events with `usage.source == "missing"` (unless the provider sends usage
  anyway).
- Pass `include_usage=True` without an observer to capture usage internally.
- Anthropic's native SDK does not accept `stream_options`; usage is captured
  when its event stream reports it.
- If an OpenAI-compatible endpoint rejects `stream_options` with a 400/422
  error naming the parameter, the request is retried once without it.

### Important: keep the observer fast

The observer is **synchronous** and runs inline in the event loop. It must be
fast and non-blocking. If you need to perform I/O (writing logs to a database,
sending an HTTP request, etc.), enqueue the event and process it elsewhere.
Exceptions raised by the observer are swallowed, so telemetry failures can
never break the prompt chain.

## API Reference

### TasksPromptsChain Class

#### Constructor Parameters

- `llm_configs` (List[Dict]): List of LLM configurations, each containing:
  - `llm_id` (str): Unique identifier for this LLM configuration
  - `llm_class`: The LLM class to use (e.g., `AsyncOpenAI`, `AsyncAnthropic`, `AsyncCerebras`)
  - `model_options` (Dict): Configuration for the LLM:
    - `model` (str): The model identifier
    - `api_key` (str): Your API key for the LLM provider
    - `temperature` (float): Temperature setting for response generation
    - `max_tokens` (int): Maximum tokens in generated responses
    - `base_url` (Optional[str]): Custom API endpoint URL
- `system_prompt` (Optional[str]): System prompt for context
- `final_result_placeholder` (str): Name for the final result placeholder
- `system_apply_to_all_prompts` (Optional[bool]): Apply system prompt to all prompts
- `observer` (Optional[Callable[[ChainEvent], None]]): Synchronous observer called with `prompt_end` and `chain_end` events. Must be fast and non-blocking; exceptions are swallowed.
- `include_usage` (Optional[bool]): Request provider token usage. Defaults to `True` when `observer` is set, otherwise `False`.

#### Methods

- `execute_chain(prompts: List[Dict], streamout: bool = True) -> AsyncGenerator[str, None]`
  - Executes the prompt chain and streams responses
  
- `template_output(template: str) -> None`
  - Sets the output template format
  
- `get_final_result_within_template(self) -> Optional[str]`
  - Retrieves the final query result with the defined template in template_output();

- `get_result(placeholder: str) -> Optional[str]`
  - Retrieves a specific result by placeholder

### Prompt Format

Each prompt in the chain can be defined as a dictionary:
```python
{
    "prompt": str,              # The actual prompt text
    "output_format": str,       # "JSON", "MARKDOWN", "CSV", or "TEXT"
    "output_placeholder": str,  # Identifier for accessing this result
    "llm_id": str,              # Optional: ID of the LLM to use for this prompt
    "stop_placeholder": str,    # Optional: The stop string placeholder that may interrupt the chaining prompt query, defined in a prompt, and may be returned by the LLM
    "metadata": dict            # Optional: per-prompt metadata copied into observer events
}
```

`PromptTemplate` accepts the same values as keyword arguments, including the
optional `metadata` argument:

```python
from tasks_prompts_chain import PromptTemplate

prompt = PromptTemplate(
    prompt="Extract the core requirements.",
    output_format="JSON",
    output_placeholder="core_requirements",
    llm_id="gpt",
    metadata={"request_kind": "core_requirements"},
)
```

Invalid (non-dict) metadata is treated as an empty dict instead of raising.

## Supported LLM Providers

TasksPromptsChain currently supports the following LLM providers:

1. **OpenAI** - via `AsyncOpenAI` from the `openai` package
2. **Anthropic** - via `AsyncAnthropic` from the `anthropic` package
3. **Cerebras** - via `AsyncCerebras` from the `cerebras` package

Each provider has different capabilities and models. The library adapts the API calls to work with each provider's specific requirements.

## Error Handling

The library includes comprehensive error handling:
- Template validation
- API error handling
- Placeholder validation
- LLM validation (checks if specified LLM ID exists)
- stop_placeholder to validate the LLM output and stop the chain prompt execution

Errors are raised with descriptive messages indicating the specific issue and prompt number where the error occurred.

## Best Practices

1. Always set templates before executing the chain
2. Use meaningful placeholder names
3. Implement a stop_placeholder in each prompt, to catch the LLM error defined bad responses to stop the subsequent requests
3. Handle streaming responses appropriately
4. Choose appropriate models for different types of tasks
5. Use system prompts for consistent context
6. Select the best provider for specific tasks:
   - OpenAI is great for general purpose applications
   - Anthropic (Claude) excels at longer contexts and complex reasoning
   - Cerebras is excellent for high-performance AI tasks

## How You Can Get Involved
✅ Try out tasks_prompts_chain: Give our software a try in your own setup and let us know how it goes - your experience helps us improve!

✅ Find a bug: Found something that doesn't work quite right? We'd appreciate your help in documenting it so we can fix it together.

✅ Fixing Bugs: Even small code contributions make a big difference! Pick an issue that interests you and share your solution with us.

✅ Share your thoughts: Have an idea that would make this project more useful? We're excited to hear your thoughts and explore new possibilities together!

Your contributions, big or small, truly matter to us. We're grateful for any help you can provide and look forward to welcoming you to our community!

### Developer Contribution Workflow
1. Fork the Repository: Create your own copy of the project by clicking the "Fork" button on our GitHub repository.
2. Clone Your Fork: 
``` bash
git clone git@github.com:<your-username>/tasks_prompts_chain.git
cd tasks_prompts_chain/
```
3. Set Up Development Environment
``` bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install development dependencies
pip install -r requirements/requirements-dev.txt
```
4. Stay Updated
```bash
# Add the upstream repository
git remote add upstream https://github.com/original-owner/tasks_prompts_chain.git

# Fetch latest changes from upstream
git fetch upstream
git merge upstream/main
```
#### Making Changes
1. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b bugfix/issue-you-are-fixing
```
2. Implement Your Changes
    - Write tests for your changes when applicable
    - Ensure existing tests pass with pytest
    - Follow our code style guidelines

3. Commit Your Changes
```bash
git add .
git commit -m "Your descriptive commit message"
```
4. Push to Your Fork
```bash
git push origin feature/your-feature-name
```
5. Create a Pull Request
6. Code Review Process
    - Maintainers will review your PR
    - Address any requested changes
    - Once approved, your contribution will be merged!

## Release Notes

### 0.2.0 - Observer and usage telemetry

- Added the `observer` callback and `prompt_end` / `chain_end` events
- Added per-prompt `metadata` (dict prompts and `PromptTemplate`)
- Added normalized `TokenUsage` with provider-reported prompt, completion,
  total, cached, and reasoning tokens
- Added `include_usage` and provider `stream_options` support, with a safe
  one-time fallback when a provider rejects the parameter
- Added `ClientLLMSDK.get_last_usage()`
- Added `ChainEvent`, `TokenUsage`, `EVENT_PROMPT_END`, and `EVENT_CHAIN_END`
  exports
- All existing behavior, output chunks, sentinel, and error messages are
  preserved; no new runtime dependencies

### 0.1.0 - Breaking Changes

- **Complete API redesign**: The constructor now requires a list of LLM configurations instead of a single LLM class
- **Multi-model support**: Use different models for different prompts in the chain
- **Constructor changes**: Replaced `AsyncLLmAi` and `model_options` with `llm_configs`
- **New provider support**: Added official support for Cerebras models
- **Removed dependencies**: No longer directly depends on OpenAI SDK
- **Prompt configuration**: Added `llm_id` field to prompt dictionaries to specify which LLM to use

Users upgrading from version 0.0.x will need to modify their code to use the new API structure.

## License

MIT License
