"""
Anthropic provider implementation for Just-Prompt
"""
import os
import re
import time
from typing import List, Optional, Dict, Any, Tuple

import anthropic
from anthropic import Anthropic

from just_prompt.atoms.shared.data_types import PromptResponse


class AnthropicProvider:
    """Anthropic provider implementation"""
    
    def __init__(self):
        """Initialize the Anthropic provider with API key"""
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        self.client = Anthropic(api_key=self.api_key)
    
    async def list_models(self) -> List[str]:
        """List available models from Anthropic"""
        try:
            # Anthropic API doesn't have a specific endpoint for listing models
            # We'll return a static list of known models
            return [
                "claude-3-opus-20240229",
                "claude-3-sonnet-20240229",
                "claude-3-haiku-20240307",
                "claude-3-7-sonnet-20250219",
                "claude-2.1",
                "claude-2.0",
                "claude-instant-1.2"
            ]
        except Exception as e:
            # Handle any errors that occur during model listing
            return await self._handle_error(e)
        
    async def generate(self, prompt: str, model: str) -> PromptResponse:
        """Generate a response for the given prompt using the specified model"""
        try:
            # Parse model name to extract thinking tokens if specified
            base_model, thinking_tokens = self._parse_model_with_thinking_tokens(model)
            
            # Set up the message request
            message_params = {
                "model": base_model,
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": prompt}]
            }
            
            # Add thinking tokens if specified
            if thinking_tokens:
                message_params["thinking_tokens"] = thinking_tokens
                
            # Call Anthropic API
            response = self.client.messages.create(**message_params)
            
            # Calculate token usage if available
            tokens = None
            if hasattr(response, 'usage'):
                tokens = response.usage.input_tokens + response.usage.output_tokens
            
            return PromptResponse(
                model=model,
                content=response.content[0].text,
                tokens=tokens
            )
        except Exception as e:
            # Handle any errors that occur during generation
            return await self._handle_error(e, retry_count=0, prompt=prompt, model=model)
    
    def _parse_model_with_thinking_tokens(self, model: str) -> Tuple[str, Optional[int]]:
        """Parse model name that might include thinking tokens suffix
        
        Example: "claude-3-7-sonnet-20250219:4k" -> ("claude-3-7-sonnet-20250219", 4096)
        """
        # Check if model includes a thinking tokens suffix
        pattern = r"^(.*?):(\d+)([km])?$"
        match = re.match(pattern, model)
        
        if not match:
            # No thinking tokens specified
            return model, None
        
        base_model = match.group(1)
        token_value = int(match.group(2))
        unit = match.group(3)
        
        # Convert to actual token value
        if unit == 'k':
            token_value *= 1024
        elif unit == 'm':
            token_value *= 1024 * 1024
            
        return base_model, token_value
        
    async def _handle_error(self, error: Exception, retry_count: int = 0, **kwargs) -> Any:
        """Handle errors with appropriate retry logic"""
        # Maximum number of retries
        max_retries = 3
        
        # Handle rate limiting errors
        if isinstance(error, anthropic.RateLimitError) and retry_count < max_retries:
            # Exponential backoff: wait longer between each retry
            wait_time = 2 ** retry_count
            time.sleep(wait_time)
            
            # Extract prompt and model from kwargs if they exist
            prompt = kwargs.get("prompt")
            model = kwargs.get("model")
            
            if prompt and model:
                # Retry the generate method with incremented retry count
                retry_count += 1
                return await self.generate(prompt, model)
            else:
                # If we don't have enough information to retry, re-raise the error
                raise error
        
        # Handle authentication errors
        elif isinstance(error, anthropic.AuthenticationError):
            raise ValueError(f"Anthropic API key is invalid: {str(error)}")
        
        # Handle API errors
        elif isinstance(error, anthropic.APIError):
            if retry_count < max_retries:
                # Wait a bit and retry
                time.sleep(1)
                
                # Extract prompt and model from kwargs if they exist
                prompt = kwargs.get("prompt")
                model = kwargs.get("model")
                
                if prompt and model:
                    # Retry the generate method with incremented retry count
                    retry_count += 1
                    return await self.generate(prompt, model)
            
            # If we've exceeded max retries or don't have enough info, re-raise
            raise ValueError(f"Anthropic API error: {str(error)}")
        
        # Handle other errors
        else:
            raise ValueError(f"Error occurred when calling Anthropic API: {str(error)}") 