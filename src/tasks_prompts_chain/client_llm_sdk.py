from typing import  Optional

class ClientLLMSDK:
    """
    A class to handle LLM SDKs for various providers.
    This class is designed to work with different LLM SDKs by
    dynamically loading the appropriate class based on the provider.
    """
    def __init__(self, AsyncLLmAi, model_options):
        """
        Initialize with any LLM SDK class.

        :param llm_class: The LLM class to be used (e.g., openai.AsyncOpenAI, anthropic.AsyncAnthropic)
        :param kwargs: Additional parameters for initializing the LLM instance
        """
        self.llm_class_name = AsyncLLmAi.__name__  # Store the class type
        client_kwargs = {"api_key": model_options["api_key"]}
        if "base_url" in model_options:
            client_kwargs["base_url"] = model_options["base_url"]
         # Instantiate the LLM
        self.client = AsyncLLmAi(**client_kwargs)

    def _extract_content(self, chunk) -> Optional[str]:
        """Extract content from different response formats"""
        if self.llm_class_name == "AsyncAnthropic":
            # Handle different Anthropic event types
            if hasattr(chunk, 'type'):
                if chunk.type == 'content_block_delta':
                    return chunk.delta.text
                elif chunk.type == 'message_stop':
                    return None
            return None
        else:
            # Handle OpenAI and other formats
            if hasattr(chunk, 'choices') and chunk.choices:
                if hasattr(chunk.choices[0], 'delta'):
                    return chunk.choices[0].delta.content
                elif hasattr(chunk.choices[0], 'message'):
                    return chunk.choices[0].message.content
        return None

    async def generat_response(self, **kwargs):
        """
        Generate a response from the LLM.

        :param prompt: The prompt to be sent to the LLM
        :param kwargs: Additional parameters for generating the response
        :return: The generated response
        """
        model = kwargs.get("model", "gpt-4")
        messages= kwargs.get("messages", [])
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 512)
        stream = kwargs.get("stream", True)
        
        if self.llm_class_name == "AsyncOpenAI":  # OpenAI SDK
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream
            )
            
            async for chunk in response:
                if chunk is not None:
                   if isinstance(chunk, str):
                            delta = chunk
                   else:
                       # Handle different response formats
                       delta = self._extract_content(chunk)
                if delta is not None:
                    yield delta
                

        elif self.llm_class_name == "AsyncAnthropic":  # Anthropic SDK
            # Extract system message if present
            system_message = ""
            filtered_messages = []
            
            for message in messages:
                if message.get("role") == "system" and message.get("content") != None:
                    system_message = message.get("content")
                else:
                    filtered_messages.append(message)
            
            # Update messages without system message
            messages = filtered_messages
            response = await self.client.messages.create(
                system=system_message,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream
            )
            
            async for chunk in response:
                if chunk is not None:
                   if isinstance(chunk, str):
                            delta = chunk
                   else:
                       # Handle different response formats
                       delta = self._extract_content(chunk)
                if delta is not None:
                    yield delta
        
        elif self.llm_class_name == "AsyncCerebras":  # AsyncCerebras SDK
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream
            )
            
            async for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        else:
            raise NotImplementedError(f"Unsupported LLM: {self.llm_class_name}")
