"""
Tests for the OpenAI provider
"""
import os
import pytest
import unittest.mock as mock
from unittest.mock import AsyncMock, MagicMock

import openai
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice, ChatCompletionUsage
from openai.pagination import SyncPage

from just_prompt.atoms.llm_providers.openai import OpenAIProvider
from just_prompt.atoms.shared.data_types import PromptResponse


class TestOpenAIProvider:
    """Tests for the OpenAI provider"""

    @mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    def test_init(self):
        """Test initialization"""
        provider = OpenAIProvider()
        assert provider.api_key == "test_key"

    @mock.patch.dict(os.environ, {})
    def test_init_missing_key(self):
        """Test initialization with missing API key"""
        with pytest.raises(ValueError, match="OPENAI_API_KEY environment variable not set"):
            OpenAIProvider()

    @mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    @mock.patch("openai.OpenAI")
    async def test_list_models(self, mock_openai):
        """Test listing models"""
        # Setup mock
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        # Create mock model data
        mock_models = [
            MagicMock(id="gpt-4"),
            MagicMock(id="gpt-3.5-turbo"),
            MagicMock(id="text-davinci-003"),
            MagicMock(id="davinci"), # Should be filtered out
            MagicMock(id="embedding-ada") # Should be filtered out
        ]
        
        # Setup return value for models.list()
        mock_client.models.list.return_value = SyncPage(data=mock_models)
        
        # Initialize provider and call list_models
        provider = OpenAIProvider()
        models = await provider.list_models()
        
        # Check that we called models.list()
        mock_client.models.list.assert_called_once()
        
        # Check that we got the expected list of models (filtered and sorted)
        assert models == ["gpt-3.5-turbo", "gpt-4", "text-davinci-003"]

    @mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    @mock.patch("openai.OpenAI")
    async def test_generate_chat(self, mock_openai):
        """Test generating a chat completion response"""
        # Setup mock
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        # Create mock response
        mock_usage = ChatCompletionUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        mock_message = ChatCompletionMessage(content="Test response", role="assistant")
        mock_choice = Choice(index=0, message=mock_message, finish_reason="stop")
        mock_response = ChatCompletion(
            id="test_id",
            choices=[mock_choice],
            created=1234567890,
            model="gpt-4",
            object="chat.completion",
            usage=mock_usage
        )
        
        # Setup return value for chat.completions.create()
        mock_client.chat.completions.create.return_value = mock_response
        
        # Initialize provider and call generate
        provider = OpenAIProvider()
        response = await provider.generate("Test prompt", "gpt-4")
        
        # Check that we called chat.completions.create() with the right arguments
        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-4",
            messages=[{"role": "user", "content": "Test prompt"}],
            temperature=0.7,
            max_tokens=1024
        )
        
        # Check that we got the expected response
        assert isinstance(response, PromptResponse)
        assert response.model == "gpt-4"
        assert response.content == "Test response"
        assert response.tokens == 30

    @mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    @mock.patch("openai.OpenAI")
    async def test_generate_completion(self, mock_openai):
        """Test generating a completion response"""
        # Setup mock
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        # Create mock response for completions
        mock_choice = MagicMock()
        mock_choice.text = "Test response"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage.total_tokens = 30
        
        # Setup return value for completions.create()
        mock_client.completions.create.return_value = mock_response
        
        # Initialize provider and call generate
        provider = OpenAIProvider()
        response = await provider.generate("Test prompt", "text-davinci-003")
        
        # Check that we called completions.create() with the right arguments
        mock_client.completions.create.assert_called_once_with(
            model="text-davinci-003",
            prompt="Test prompt",
            temperature=0.7,
            max_tokens=1024
        )
        
        # Check that we got the expected response
        assert isinstance(response, PromptResponse)
        assert response.model == "text-davinci-003"
        assert response.content == "Test response"
        assert response.tokens == 30

    @mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    @mock.patch("openai.OpenAI")
    @mock.patch("time.sleep")
    async def test_handle_rate_limit_error(self, mock_sleep, mock_openai):
        """Test handling rate limit errors"""
        # Setup mock
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        # First call raises a rate limit error, second call succeeds
        mock_client.chat.completions.create.side_effect = [
            openai.RateLimitError("Rate limit exceeded"),
            MagicMock(
                choices=[MagicMock(message=MagicMock(content="Retry response"))],
                usage=MagicMock(total_tokens=25)
            )
        ]
        
        # Initialize provider and call generate
        provider = OpenAIProvider()
        response = await provider.generate("Test prompt", "gpt-4")
        
        # Check that sleep was called
        mock_sleep.assert_called_once_with(1)  # First retry = 2^0 = 1 second
        
        # Check that we got the expected response after retry
        assert response.model == "gpt-4"
        assert response.content == "Retry response"
        assert response.tokens == 25

    @mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    @mock.patch("openai.OpenAI")
    async def test_handle_authentication_error(self, mock_openai):
        """Test handling authentication errors"""
        # Setup mock
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        # Raise an authentication error
        mock_client.chat.completions.create.side_effect = openai.AuthenticationError("Invalid API key")
        
        # Initialize provider and call generate
        provider = OpenAIProvider()
        
        # Check that we raise the expected error
        with pytest.raises(ValueError, match="OpenAI API key is invalid"):
            await provider.generate("Test prompt", "gpt-4") 