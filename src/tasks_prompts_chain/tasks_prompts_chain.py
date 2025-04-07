#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: tasks_prompts_chain.py
Author: Samir Ben Sghaier
Date: 2025-02-08
Version: 1.0
Description: 
    A Python library for creating and executing chains of prompts using 
    OpenAI's SDK with streaming support and template formatting. 
Contact: ben.sghaier.samir@gmail.com
GitHub: https://github.com/smirfolio
Dependencies: openai, typing-extensions

License: APACHE 2.0 License
Copyright 2025 Samir Ben Sghaier - Smirfolio

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

"""
from typing import List, Optional, Dict, Union, AsyncGenerator, TypedDict
from enum import Enum
from .client_llm_sdk import ClientLLMSDK

class OutputFormat(Enum):
    JSON = "JSON"
    MARKDOWN = "MARKDOWN"
    CSV = "CSV"
    TEXT = "TEXT"

class ModelOptions(TypedDict, total=False):
    model: str
    api_key: str 
    base_url: Optional[str]
    temperature: Optional[float]
    max_tokens: Optional[int]

class PromptTemplate:
    def __init__(self, prompt: str, output_format: str = "TEXT", output_placeholder: Optional[str] = None):
        self.prompt = prompt
        self.output_format = OutputFormat(output_format.upper())
        self.output_placeholder = output_placeholder

class TasksPromptsChain:
    """A utility class for creating and executing prompt chains using OpenAI's API."""
    
    def __init__(self,
                 injectedLLM,
                 model_options: ModelOptions,
                 system_prompt: Optional[str] = None,
                 final_result_placeholder: Optional[str] = None,
                 system_apply_to_all_prompts: bool = False):
        """
        Initialize the TasksPromptsChain with OpenAI configuration.
        
        Args:
            model_options (ModelOptions): Dictionary containing model configuration:
                - model (str): The model identifier to use (e.g., 'gpt-3.5-turbo')
                - api_key (str): The OpenAI API key
                - base_url (str, optional): API endpoint URL
                - temperature (float, optional): Temperature parameter (default: 0.7)
                - max_tokens (int, optional): Maximum tokens (default: 4120)
            system_prompt (str, optional): System prompt to set context for the LLM
            final_result_placeholder (str, optional): The placeholder name for the final result
            system_apply_to_all_prompts (bool): Whether to apply system prompt to all prompts
        """
        self.model = model_options.get("model", "gpt-3.5-turbo")
        self.temperature = model_options.get("temperature", 0.7)
        self.max_tokens = model_options.get("max_tokens", 4120)
        client_kwargs = {"api_key": model_options["api_key"]}
        if "base_url" in model_options:
            client_kwargs["base_url"] = model_options["base_url"]
        self.client = ClientLLMSDK(injectedLLM, client_kwargs)
        self.system_prompt = system_prompt
        self.system_apply_to_all_prompts = system_apply_to_all_prompts
        self.final_result_placeholder = final_result_placeholder or "final_result"
        self._results = {}
        self._output_template = None
        self._final_output_template = None
        self._current_stream_buffer = ""
    
    def get_reflection(self):
        """
        Get the reflection of the class instance.
        
        Returns:
            str: The string representation of the class instance
        """
        return self.AsyncLLmAi
    def set_output_template(self, template: str) -> None:
        """
        Set the output template to be used for streaming responses.
        Must be called before execute_chain if template formatting is desired.
        
        Args:
            template (str): Template string containing placeholders in {{placeholder}} format
        """
        self._output_template = template

    def _format_current_stream(self) -> str:
        """
        Format the current stream buffer using the template.
        
        Returns:
            str: Formatted output using the template
        """
        if not self._output_template:
            return self._current_stream_buffer
            
        output = self._output_template
        # Replace all existing results
        for placeholder, value in self._results.items():
            output = output.replace(f"{{{{{placeholder}}}}}", value or "")
        # Replace current streaming placeholder
        output = output.replace(f"{{{{{self.final_result_placeholder}}}}}", self._current_stream_buffer)
        self._final_output_template=output
        return output

    async def execute_chain(self, prompts: List[Union[Dict, PromptTemplate]], streamout = True) -> AsyncGenerator[str, None]:
        """
        Execute a chain of prompts sequentially, with placeholder replacement.
        
        Args:
            prompts (List[Union[Dict, PromptTemplate]]): List of prompt templates or dicts with structure:
                                                        {
                                                            "prompt": str,
                                                            "output_format": str,
                                                            "output_placeholder": str
                                                        }
        Returns:
            List[str]: List of responses for each prompt
        """
        responses = []
        placeholder_values = {}

        try:
            for i, prompt_data in enumerate(prompts):
                # Convert dict to PromptTemplate if necessary
                if isinstance(prompt_data, dict):
                    prompt_template = PromptTemplate(
                        prompt=prompt_data["prompt"],
                        output_format=prompt_data.get("output_format", "TEXT"),
                        output_placeholder=prompt_data.get("output_placeholder")
                    )
                else:
                    prompt_template = prompt_data

                # Replace placeholders in the prompt
                current_prompt = prompt_template.prompt
                for placeholder, value in placeholder_values.items():
                    current_prompt = current_prompt.replace(f"{{{{{placeholder}}}}}", value)

                # Format system message based on output format
                format_instruction = ""
                if prompt_template.output_format != OutputFormat.TEXT:
                    format_instruction = f"\nPlease provide your response in {prompt_template.output_format.value} format."

                messages = []
                if self.system_prompt and (i == 0 or self.system_apply_to_all_prompts):
                    messages.append({"role": "system", "content": self.system_prompt})

                messages.append({"role": "user", "content": current_prompt + format_instruction})
                
                streamResponse = self.client.generat_response(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stream=True
                )
                
                response_content = ""
                self._current_stream_buffer = ""
                
                async for chunk in streamResponse:
                    if chunk is not None:
                        delta = chunk
                        response_content += delta
                        self._current_stream_buffer = response_content
                        self._format_current_stream()
                        responses.append(response_content)
                        # Store response with placeholder if specified
                        if prompt_template.output_placeholder:
                            placeholder_values[prompt_template.output_placeholder] = response_content
                            self._results[prompt_template.output_placeholder] = response_content
                        if streamout:
                            yield delta            

        except Exception as e:
            raise Exception(f"Error in prompt chain execution at prompt {i}: {str(e)}")

        # Store the last response with the final result placeholder
        if responses:
            self._results[self.final_result_placeholder] = responses[-1]
            yield "<tasks-sys>Done</tasks-sys>"

    def get_result(self, placeholder: str) -> Optional[str]:
        """
        Get the result of a specific prompt by its placeholder.
        
        Args:
            placeholder (str): The output_placeholder value used in the prompt
            
        Returns:
            Optional[str]: The response for that placeholder if it exists, None otherwise
        """
        return self._results.get(placeholder)

    def get_final_result_within_template(self) -> Optional[str]:
        """
        Get tje final result
        Returns:
            Optional[str]: The response for that placeholder if it exists, None otherwise
        """
        return self._final_output_template
    
    def template_output(self, template: str) -> None:
        """
        Set the output template for streaming responses.
        Must be called before execute_chain.
        
        Args:
            template (str): Template string containing placeholders in {{placeholder}} format
            
        Raises:
            Exception: If called after execute_chain has already been run
        """
        if len(self._results) > 0:
            raise Exception("template_output must be called before execute_chain")
        self.set_output_template(template)

