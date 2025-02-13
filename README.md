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

## Dependancies

Please install typing-extensions and openai python packages
```bash
pip install typing-extensions
pip install openai
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

async def main():
    # Initialize the chain
    chain = TasksPromptsChain(
        model="gpt-3.5-turbo",
        api_key="your-api-key",
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

### Using Templates

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

### Using System Prompts

```python
chain = TasksPromptsChain(
    model="gpt-3.5-turbo",
    api_key="your-api-key",
    final_result_placeholder="result",
    system_prompt="You are a professional design expert specialized in luxury products",
    system_apply_to_all_prompts=True
)
```

### Custom API Endpoint

```python
chain = TasksPromptsChain(
    model="gpt-3.5-turbo",
    api_key="your-api-key",
    final_result_placeholder="result",
    base_url="https://your-custom-endpoint.com/v1"
)
```

## API Reference

### TasksPromptsChain Class

#### Constructor Parameters

- `model` (str): The model identifier (e.g., 'gpt-3.5-turbo')
- `api_key` (str): Your OpenAI API key
- `final_result_placeholder` (str): Name for the final result placeholder
- `system_prompt` (Optional[str]): System prompt for context
- `system_apply_to_all_prompts` (Optional[bool]): Apply system prompt to all prompts
- `base_url` (Optional[str]): Custom API endpoint URL

#### Methods

- `execute_chain(prompts: List[Dict], temperature: float = 0.7) -> AsyncGenerator[str, None]`
  - Executes the prompt chain and streams responses
  
- `template_output(template: str) -> None`
  - Sets the output template format
  
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

## License

MIT License
