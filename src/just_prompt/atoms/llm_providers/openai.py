"""
OpenAI provider implementation for Just-Prompt
"""
import os
import time
from typing import List, Optional, Dict, Any

import openai
from openai import OpenAI
from openai.types.completion import Completion

from just_prompt.atoms.shared.data_types import PromptResponse


class OpenAIProvider:
    """OpenAI provider implementation"""
    
    def __init__(self):
        """Initialize the OpenAI provider with API key"""
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.client = OpenAI(api_key=self.api_key)
        
    async def list_models(self) -> List[str]:
        """List available models from OpenAI"""
        try:
            models = self.client.models.list()
            # Filter to include only relevant models for text generation
            relevant_models = [
                model.id for model in models.data
                if model.id.startswith(("gpt-", "text-"))
            ]
            return sorted(relevant_models)
        except Exception as e:
            # Handle any errors that occur during model listing
            return await self._handle_error(e)

    async def generate(self, prompt: str, model: str) -> PromptResponse:
        """Generate a response for the given prompt using the specified model"""
        try:
            if model.startswith("gpt-"):
                # For chat models
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1024
                )
                # Calculate token usage if available
                tokens = response.usage.total_tokens if hasattr(response, 'usage') else None
                return PromptResponse(
                    model=model,
                    content=response.choices[0].message.content,
                    tokens=tokens
                )
            else:
                # For completion models (older models)
                response = self.client.completions.create(
                    model=model,
                    prompt=prompt,
                    temperature=0.7,
                    max_tokens=1024
                )
                # Calculate token usage if available
                tokens = response.usage.total_tokens if hasattr(response, 'usage') else None
                return PromptResponse(
                    model=model,
                    content=response.choices[0].text,
                    tokens=tokens
                )
        except Exception as e:
            # Handle any errors that occur during generation
            return await self._handle_error(e, retry_count=0, prompt=prompt, model=model)
    
    async def _handle_error(self, error: Exception, retry_count: int = 0, **kwargs) -> Any:
        """Handle errors with appropriate retry logic"""
        # Maximum number of retries
        max_retries = 3
        
        # Handle rate limiting errors
        if isinstance(error, openai.RateLimitError) and retry_count < max_retries:
            # Exponential backoff: wait longer between each retry
            wait_time = 2 ** retry_count
            time.sleep(wait_time)
            
            # Extract prompt and model from kwargs if they exist
            prompt = kwargs.get("prompt")
            model = kwargs.get("model")
            
            if prompt and model:
                # Retry the generate method
                return await self.generate(prompt, model)
            else:
                # If we don't have enough information to retry, re-raise the error
                raise error
        
        # Handle authentication errors
        elif isinstance(error, openai.AuthenticationError):
            raise ValueError(f"OpenAI API key is invalid: {str(error)}")
        
        # Handle API errors
        elif isinstance(error, openai.APIError):
            if retry_count < max_retries:
                # Wait a bit and retry
                time.sleep(1)
                
                # Extract prompt and model from kwargs if they exist
                prompt = kwargs.get("prompt")
                model = kwargs.get("model")
                
                if prompt and model:
                    # Retry the generate method with incremented retry count
                    return await self.generate(prompt, model)
            
            # If we've exceeded max retries or don't have enough info, re-raise
            raise ValueError(f"OpenAI API error: {str(error)}")
        
        # Handle other errors
        else:
            raise ValueError(f"Error occurred when calling OpenAI API: {str(error)}") 