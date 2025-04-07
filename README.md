# TasksPromptsChain

A Mini Python library for creating and executing chains of prompts using OpenAI's API with streaming support and output template formatting.

## Features

- Sequential prompt chain execution
- Streaming responses
- Template-based output formatting
- System prompt support
- Placeholder replacement between prompts
- Multiple output formats (JSON, Markdown, CSV, Text)
- Async/await support
- Support for multiple LLM providers (OpenAI and Anthropic)

## Dependencies

Please install typing-extensions and the SDK for your preferred LLM provider:

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
from openai import AsyncOpenAI  # Import the LLM SDK you want to use

async def main():
    # Initialize the chain with OpenAI
    chain = TasksPromptsChain(
        AsyncLLmAi=AsyncOpenAI,  # Specify which LLM SDK to use
        model_options={
            "model": "gpt-3.5-turbo",
            "api_key": "your-api-key",
            "temperature": 0.1,
            "max_tokens": 4120,
            "stream": True
        },
        final_result_placeholder="design_result"
    )

    # Define your prompts
    prompts = [
        {
            "prompt": "Create a design concept for a luxury chocolate bar",
            "output_format": "TEXT",
            "output_placeholder": "design_concept"
        },
        {
            "prompt": "Based on this concept: {{design_concept}}, suggest a color palette",
            "output_format": "JSON",
            "output_placeholder": "color_palette"
        }
    ]

    # Stream the responses
    async for chunk in chain.execute_chain(prompts):
        print(chunk, end="", flush=True)

    # Get specific results
    design = chain.get_result("design_concept")
    colors = chain.get_result("color_palette")
```

## Advanced Usage

### Using Anthropic (Claude)

```python
from tasks_prompts_chain import TasksPromptsChain
from anthropic import AsyncAnthropic

chain = TasksPromptsChain(
    AsyncLLmAi=AsyncAnthropic,  # Use Anthropic's SDK
    model_options={
            "model": "claude-3-haiku-20240307",  # Use a Claude model
            "api_key": "your-anthropic-api-key",
            "temperature": 0.1,
            "max_tokens": 4120,
            "stream": True
        },
    final_result_placeholder="result"
)
```

### Using System Prompts

```python
from openai import AsyncOpenAI

chain = TasksPromptsChain(
    AsyncLLmAi=AsyncOpenAI,
    model_options={
            "model": "gpt-3.5-turbo",
            "api_key": "your-api-key",
            "temperature": 0.1,
            "max_tokens": 4120,
            "stream": True
        },
    final_result_placeholder="result",
    system_prompt="You are a professional design expert specialized in luxury products",
    system_apply_to_all_prompts=True
)
```

### Custom API Endpoint

```python
from openai import AsyncOpenAI

chain = TasksPromptsChain(
    AsyncLLmAi=AsyncOpenAI,
    model_options={
            "model": "your-custom-model",
            "api_key": "your-api-key",
            "base_url": "https://your-custom-endpoint.com/v1",
            "temperature": 0.1,
            "max_tokens": 4120,
            "stream": True
        },
    final_result_placeholder="result",
)
```

### Using Templates

You must call this set method befor the excution of the prompting query (chain.execute_chain(prompts))

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


## API Reference

### TasksPromptsChain Class

#### Constructor Parameters

- `AsyncLLmAi`: The LLM SDK class to use (e.g., `AsyncOpenAI` or `AsyncAnthropic`)
- `model_options` (Dict): Configuration for the LLM:
  - `model` (str): The model identifier (e.g., 'gpt-3.5-turbo' or 'claude-3-haiku-20240307')
  - `api_key` (str): Your API key for the LLM provider
  - `temperature` (float): Temperature setting for response generation
  - `max_tokens` (int): Maximum tokens in generated responses
  - `base_url` (Optional[str]): Custom API endpoint URL
- `system_prompt` (Optional[str]): System prompt for context
- `final_result_placeholder` (str): Name for the final result placeholder
- `system_apply_to_all_prompts` (Optional[bool]): Apply system prompt to all prompts

#### Methods

- `execute_chain(prompts: List[Dict], temperature: float = 0.7) -> AsyncGenerator[str, None]`
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
    "prompt": str,           # The actual prompt text
    "output_format": str,    # "JSON", "MARKDOWN", "CSV", or "TEXT"
    "output_placeholder": str # Identifier for accessing this result
}
```

## Error Handling

The library includes comprehensive error handling:
- Template validation
- API error handling
- Placeholder validation

Errors are raised with descriptive messages indicating the specific issue and prompt number where the error occurred.

## Best Practices

1. Always set templates before executing the chain
2. Use meaningful placeholder names
3. Handle streaming responses appropriately
4. Consider temperature settings based on your use case
5. Use system prompts for consistent context
6. Choose the appropriate LLM provider for your needs:
   - OpenAI is great for general purpose applications
   - Anthropic (Claude) excels at longer contexts and complex reasoning

## How You Can Get Involved
✅ Try out tasks_prompts_chain: Give our software a try in your own setup and let us know how it goes - your experience helps us improve!

✅ Find a bug: ound something that doesn't work quite right? We'd appreciate your help in documenting it so we can fix it together.

✅ Fixing Bugs: ven small code contributions make a big difference! Pick an issue that interests you and share your solution with us.

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

## License

MIT License
