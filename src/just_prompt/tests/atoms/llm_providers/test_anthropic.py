"""
Tests for the Anthropic provider
"""
import os
import re
import pytest
import unittest.mock as mock
from unittest.mock import AsyncMock, MagicMock

import anthropic

from just_prompt.atoms.llm_providers.anthropic import AnthropicProvider
from just_prompt.atoms.shared.data_types import PromptResponse


class TestAnthropicProvider:
    """Tests for the Anthropic provider"""

    @mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"})
    def test_init(self):
        """Test initialization"""
        provider = AnthropicProvider()
        assert provider.api_key == "test_key"

    @mock.patch.dict(os.environ, {})
    def test_init_missing_key(self):
        """Test initialization with missing API key"""
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY environment variable not set"):
            AnthropicProvider()

    @mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"})
    async def test_list_models(self):
        """Test listing models"""
        provider = AnthropicProvider()
        models = await provider.list_models()
        
        # Check that we got the expected list of models
        assert isinstance(models, list)
        assert "claude-3-opus-20240229" in models
        assert "claude-3-sonnet-20240229" in models
        assert "claude-3-haiku-20240307" in models

    @mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"})
    @mock.patch("anthropic.Anthropic")
    async def test_generate(self, mock_anthropic):
        """Test generating a response"""
        # Setup mock
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        
        # Create mock response
        mock_content = MagicMock()
        mock_content.text = "Test response"
        mock_response = MagicMock()
        mock_response.content = [mock_content]
        
        # Add usage information if available in the response
        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 20
        
        # Setup return value for messages.create()
        mock_client.messages.create.return_value = mock_response
        
        # Initialize provider and call generate
        provider = AnthropicProvider()
        response = await provider.generate("Test prompt", "claude-3-sonnet-20240229")
        
        # Check that we called messages.create() with the right arguments
        mock_client.messages.create.assert_called_once_with(
            model="claude-3-sonnet-20240229",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Test prompt"}]
        )
        
        # Check that we got the expected response
        assert isinstance(response, PromptResponse)
        assert response.model == "claude-3-sonnet-20240229"
        assert response.content == "Test response"
        assert response.tokens == 30

    @mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"})
    @mock.patch("anthropic.Anthropic")
    async def test_generate_with_thinking_tokens(self, mock_anthropic):
        """Test generating a response with thinking tokens specified"""
        # Setup mock
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        
        # Create mock response
        mock_content = MagicMock()
        mock_content.text = "Test response"
        mock_response = MagicMock()
        mock_response.content = [mock_content]
        
        # Setup return value for messages.create()
        mock_client.messages.create.return_value = mock_response
        
        # Initialize provider and call generate with thinking tokens
        provider = AnthropicProvider()
        response = await provider.generate("Test prompt", "claude-3-sonnet-20240229:4k")
        
        # Check that we called messages.create() with the right arguments
        mock_client.messages.create.assert_called_once_with(
            model="claude-3-sonnet-20240229",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Test prompt"}],
            thinking_tokens=4096
        )
        
        # Check that we got the expected response
        assert isinstance(response, PromptResponse)
        assert response.model == "claude-3-sonnet-20240229:4k"
        assert response.content == "Test response"

    def test_parse_model_with_thinking_tokens(self):
        """Test parsing model names with thinking tokens suffix"""
        provider = AnthropicProvider()
        
        # Test regular model name
        model, tokens = provider._parse_model_with_thinking_tokens("claude-3-sonnet-20240229")
        assert model == "claude-3-sonnet-20240229"
        assert tokens is None
        
        # Test with kilobyte token suffix
        model, tokens = provider._parse_model_with_thinking_tokens("claude-3-sonnet-20240229:4k")
        assert model == "claude-3-sonnet-20240229"
        assert tokens == 4096
        
        # Test with raw number token suffix
        model, tokens = provider._parse_model_with_thinking_tokens("claude-3-sonnet-20240229:2000")
        assert model == "claude-3-sonnet-20240229"
        assert tokens == 2000
        
        # Test with megabyte token suffix
        model, tokens = provider._parse_model_with_thinking_tokens("claude-3-sonnet-20240229:1m")
        assert model == "claude-3-sonnet-20240229"
        assert tokens == 1048576

    @mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"})
    @mock.patch("anthropic.Anthropic")
    @mock.patch("time.sleep")
    async def test_handle_rate_limit_error(self, mock_sleep, mock_anthropic):
        """Test handling rate limit errors"""
        # Setup mock
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        
        # First call raises a rate limit error, second call succeeds
        mock_content = MagicMock()
        mock_content.text = "Retry response"
        mock_response = MagicMock()
        mock_response.content = [mock_content]
        
        mock_client.messages.create.side_effect = [
            anthropic.RateLimitError("Rate limit exceeded"),
            mock_response
        ]
        
        # Initialize provider and call generate
        provider = AnthropicProvider()
        response = await provider.generate("Test prompt", "claude-3-sonnet-20240229")
        
        # Check that sleep was called
        mock_sleep.assert_called_once_with(1)  # First retry = 2^0 = 1 second
        
        # Check that we got the expected response after retry
        assert response.model == "claude-3-sonnet-20240229"
        assert response.content == "Retry response"

    @mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"})
    @mock.patch("anthropic.Anthropic")
    async def test_handle_authentication_error(self, mock_anthropic):
        """Test handling authentication errors"""
        # Setup mock
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        
        # Raise an authentication error
        mock_client.messages.create.side_effect = anthropic.AuthenticationError("Invalid API key")
        
        # Initialize provider and call generate
        provider = AnthropicProvider()
        
        # Check that we raise the expected error
        with pytest.raises(ValueError, match="Anthropic API key is invalid"):
            await provider.generate("Test prompt", "claude-3-sonnet-20240229") 