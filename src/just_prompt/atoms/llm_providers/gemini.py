"""
Google Gemini provider implementation for Just-Prompt
"""
import os
import time
from typing import List, Optional, Dict, Any

import google.generativeai as genai
from google.generativeai.types import GenerateContentResponse

from just_prompt.atoms.shared.data_types import PromptResponse


class GeminiProvider:
    """Google Gemini provider implementation"""
    
    def __init__(self):
        """Initialize the Gemini provider with API key"""
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        genai.configure(api_key=self.api_key)
    
    async def list_models(self) -> List[str]:
        """List available models from Google Gemini"""
        try:
            # Fetch models from Gemini API
            models = genai.list_models()
            # Filter to only include text models
            gemini_models = [
                model.name.split('/')[-1] 
                for model in models 
                if "generateContent" in model.supported_generation_methods
            ]
            return gemini_models
        except Exception as e:
            return await self._handle_error(e)
        
    async def generate(self, prompt: str, model: str) -> PromptResponse:
        """Generate a response for the given prompt using the specified model"""
        try:
            # Configure the generation model
            generation_config = {
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 2048,
            }
            
            # Create the model and generate content
            model_instance = genai.GenerativeModel(
                model_name=model,
                generation_config=generation_config
            )
            
            response = model_instance.generate_content(prompt)
            
            # Calculate token usage if available
            tokens = None
            if hasattr(response, 'usage_metadata'):
                tokens = response.usage_metadata.total_token_count
            
            # Extract text from response
            content = response.text if hasattr(response, 'text') else self._extract_text_from_response(response)
            
            return PromptResponse(
                model=model,
                content=content,
                tokens=tokens
            )
        except Exception as e:
            return await self._handle_error(e, retry_count=0, prompt=prompt, model=model)
    
    def _extract_text_from_response(self, response: GenerateContentResponse) -> str:
        """Extract text from various response formats"""
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and candidate.content:
                return candidate.content.parts[0].text
        
        # Fallback to string representation
        return str(response)
    
    async def _handle_error(self, error: Exception, retry_count: int = 0, **kwargs) -> Any:
        """Handle errors with appropriate retry logic"""
        # Maximum number of retries
        max_retries = 3
        
        # Handle rate limiting errors
        if "quota" in str(error).lower() and retry_count < max_retries:
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
        elif "authentication" in str(error).lower() or "api key" in str(error).lower():
            raise ValueError(f"Google Gemini API key is invalid: {str(error)}")
        
        # Handle API errors
        elif retry_count < max_retries and kwargs.get("prompt") and kwargs.get("model"):
            # Wait a bit and retry
            time.sleep(1)
            
            # Retry the generate method with incremented retry count
            retry_count += 1
            return await self.generate(kwargs["prompt"], kwargs["model"])
        
        # Handle other errors
        else:
            raise ValueError(f"Error occurred when calling Google Gemini API: {str(error)}") 