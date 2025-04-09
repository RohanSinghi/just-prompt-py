"""
Tests for the Google Gemini provider
"""
import os
import pytest
import unittest.mock as mock
from unittest.mock import AsyncMock, MagicMock, patch

from just_prompt.atoms.llm_providers.gemini import GeminiProvider
from just_prompt.atoms.shared.data_types import PromptResponse


class TestGeminiProvider:
    """Tests for the Google Gemini provider"""

    @mock.patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"})
    @mock.patch("google.generativeai.configure")
    def test_init(self, mock_configure):
        """Test initialization"""
        provider = GeminiProvider()
        assert provider.api_key == "test_key"
        mock_configure.assert_called_once_with(api_key="test_key")

    @mock.patch.dict(os.environ, {})
    def test_init_missing_key(self):
        """Test initialization with missing API key"""
        with pytest.raises(ValueError, match="GEMINI_API_KEY environment variable not set"):
            GeminiProvider()

    @mock.patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"})
    @mock.patch("google.generativeai.configure")
    @mock.patch("google.generativeai.list_models")
    async def test_list_models(self, mock_list_models, mock_configure):
        """Test listing models"""
        # Setup mock models
        mock_model1 = MagicMock()
        mock_model1.name = "models/gemini-pro"
        mock_model1.supported_generation_methods = ["generateContent"]
        
        mock_model2 = MagicMock()
        mock_model2.name = "models/gemini-ultra"
        mock_model2.supported_generation_methods = ["generateContent"]
        
        mock_model3 = MagicMock()
        mock_model3.name = "models/embedding-model"
        mock_model3.supported_generation_methods = ["embedContent"]
        
        mock_list_models.return_value = [mock_model1, mock_model2, mock_model3]
        
        # Initialize provider and call list_models
        provider = GeminiProvider()
        models = await provider.list_models()
        
        # Check that we filtered correctly
        assert "gemini-pro" in models
        assert "gemini-ultra" in models
        assert len(models) == 2  # Only text models, not embedding models

    @mock.patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"})
    @mock.patch("google.generativeai.configure")
    @mock.patch("google.generativeai.GenerativeModel")
    async def test_generate(self, mock_gen_model, mock_configure):
        """Test generating a response"""
        # Setup model mock
        mock_model_instance = MagicMock()
        mock_gen_model.return_value = mock_model_instance
        
        # Setup response
        mock_response = MagicMock()
        mock_response.text = "Test response"
        
        # Add usage information
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.total_token_count = 42
        
        mock_model_instance.generate_content.return_value = mock_response
        
        # Initialize provider and call generate
        provider = GeminiProvider()
        response = await provider.generate("Test prompt", "gemini-pro")
        
        # Check that we created the model with the right parameters
        mock_gen_model.assert_called_once_with(
            model_name="gemini-pro",
            generation_config={
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 2048,
            }
        )
        
        # Check that we generated content with the right prompt
        mock_model_instance.generate_content.assert_called_once_with("Test prompt")
        
        # Check that we got the expected response
        assert isinstance(response, PromptResponse)
        assert response.model == "gemini-pro"
        assert response.content == "Test response"
        assert response.tokens == 42

    @mock.patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"})
    @mock.patch("google.generativeai.configure")
    @mock.patch("google.generativeai.GenerativeModel")
    async def test_generate_without_text_attribute(self, mock_gen_model, mock_configure):
        """Test generating a response when response has no text attribute"""
        # Setup model mock
        mock_model_instance = MagicMock()
        mock_gen_model.return_value = mock_model_instance
        
        # Setup response with candidates structure instead of text attribute
        mock_candidate = MagicMock()
        mock_content = MagicMock()
        mock_part = MagicMock()
        mock_part.text = "Test response via candidates"
        
        mock_content.parts = [mock_part]
        mock_candidate.content = mock_content
        
        mock_response = MagicMock()
        mock_response.text = None  # No text attribute
        mock_response.candidates = [mock_candidate]
        
        mock_model_instance.generate_content.return_value = mock_response
        
        # Initialize provider and call generate
        provider = GeminiProvider()
        response = await provider.generate("Test prompt", "gemini-pro")
        
        # Check that we got the expected response extracted from candidates
        assert response.content == "Test response via candidates"

    @mock.patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"})
    @mock.patch("google.generativeai.configure")
    @mock.patch("google.generativeai.GenerativeModel")
    @mock.patch("time.sleep")
    async def test_handle_quota_error(self, mock_sleep, mock_gen_model, mock_configure):
        """Test handling quota exceeded errors"""
        # Setup model mock
        mock_model_instance = MagicMock()
        mock_gen_model.return_value = mock_model_instance
        
        # First call raises a quota error, second call succeeds
        mock_response = MagicMock()
        mock_response.text = "Retry response"
        
        quota_error = Exception("Quota exceeded for this API key")
        
        mock_model_instance.generate_content.side_effect = [
            quota_error,
            mock_response
        ]
        
        # Initialize provider and call generate
        provider = GeminiProvider()
        response = await provider.generate("Test prompt", "gemini-pro")
        
        # Check that sleep was called for exponential backoff
        mock_sleep.assert_called_once_with(1)  # First retry = 2^0 = 1 second
        
        # Check that we generated content twice (one error, one success)
        assert mock_model_instance.generate_content.call_count == 2
        
        # Check that we got the expected response after retry
        assert response.model == "gemini-pro"
        assert response.content == "Retry response"

    @mock.patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"})
    @mock.patch("google.generativeai.configure")
    @mock.patch("google.generativeai.GenerativeModel")
    async def test_handle_authentication_error(self, mock_gen_model, mock_configure):
        """Test handling authentication errors"""
        # Setup model mock
        mock_model_instance = MagicMock()
        mock_gen_model.return_value = mock_model_instance
        
        # Raise an authentication error
        auth_error = Exception("Invalid API key")
        mock_model_instance.generate_content.side_effect = auth_error
        
        # Initialize provider and call generate
        provider = GeminiProvider()
        
        # Check that we raise the expected error
        with pytest.raises(ValueError, match="Google Gemini API key is invalid"):
            await provider.generate("Test prompt", "gemini-pro") 