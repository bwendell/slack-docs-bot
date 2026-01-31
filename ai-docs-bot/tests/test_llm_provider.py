"""Unit tests for the LLM provider factory.

Tests the get_llm() function with comprehensive coverage of:
- OpenAI provider instantiation
- Ollama provider instantiation with health checks
- Caching behavior
- Error handling for connection failures
- Error handling for missing models
- Error handling for invalid JSON responses
- Error handling for invalid providers
- System prompt passing
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.retrieval.llm_provider import get_llm, _llm_cache


@pytest.fixture
def clear_cache():
    """Clear the LLM cache before each test.
    
    CRITICAL: This fixture ensures test isolation by clearing the shared
    _llm_cache dictionary before each test runs.
    
    Yields:
        None
    """
    _llm_cache.clear()
    yield
    _llm_cache.clear()


class TestOpenAIProvider:
    """Tests for OpenAI LLM provider instantiation."""

    def test_returns_openai_instance(self, clear_cache):
        """Test that OpenAI provider returns OpenAI instance.
        
        Mocks:
        - get_settings() to return openai as provider
        - OpenAI constructor to track instantiation
        
        Expected:
        - Instance of OpenAI is returned
        - OpenAI constructor called with correct parameters
        """
        mock_settings = Mock()
        mock_settings.LLM_PROVIDER = "openai"
        mock_settings.OPENAI_API_KEY = "test-key"
        mock_settings.OPENAI_API_BASE = "https://api.openai.com/v1"
        mock_settings.LLM_MODEL = "gpt-4"
        
        with patch("src.retrieval.llm_provider.get_settings", return_value=mock_settings):
            with patch("src.retrieval.llm_provider.OpenAI") as mock_openai_class:
                mock_openai_instance = Mock()
                mock_openai_class.return_value = mock_openai_instance
                
                result = get_llm()
                
                assert result == mock_openai_instance
                mock_openai_class.assert_called_once_with(
                    api_key="test-key",
                    api_base="https://api.openai.com/v1",
                    model="gpt-4",
                    temperature=0.1,
                    max_tokens=1024,
                    system_prompt=None,
                )

    def test_system_prompt_passed_to_openai(self, clear_cache):
        """Test that system_prompt is passed to OpenAI constructor.
        
        Mocks:
        - get_settings() to return openai as provider
        - OpenAI constructor to track instantiation
        
        Expected:
        - system_prompt parameter is passed to OpenAI constructor
        """
        mock_settings = Mock()
        mock_settings.LLM_PROVIDER = "openai"
        mock_settings.OPENAI_API_KEY = "test-key"
        mock_settings.OPENAI_API_BASE = "https://api.openai.com/v1"
        mock_settings.LLM_MODEL = "gpt-4"
        
        custom_prompt = "You are a helpful assistant."
        
        with patch("src.retrieval.llm_provider.get_settings", return_value=mock_settings):
            with patch("src.retrieval.llm_provider.OpenAI") as mock_openai_class:
                mock_openai_instance = Mock()
                mock_openai_class.return_value = mock_openai_instance
                
                result = get_llm(system_prompt=custom_prompt)
                
                assert result == mock_openai_instance
                mock_openai_class.assert_called_once_with(
                    api_key="test-key",
                    api_base="https://api.openai.com/v1",
                    model="gpt-4",
                    temperature=0.1,
                    max_tokens=1024,
                    system_prompt=custom_prompt,
                )


class TestOllamaProvider:
    """Tests for Ollama LLM provider instantiation."""

    def test_returns_ollama_instance(self, clear_cache):
        """Test that Ollama provider returns Ollama instance.
        
        Mocks:
        - get_settings() to return ollama as provider
        - requests.get() to return successful health check response
        - Ollama constructor to track instantiation
        
        Expected:
        - Instance of Ollama is returned
        - Ollama constructor called with correct parameters
        - Health check request made to Ollama API
        """
        mock_settings = Mock()
        mock_settings.LLM_PROVIDER = "ollama"
        mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
        mock_settings.OLLAMA_MODEL = "llama3.2"
        mock_settings.OLLAMA_TIMEOUT = 120.0
        mock_settings.OLLAMA_CONTEXT_WINDOW = 8192
        
        mock_response = Mock()
        mock_response.json.return_value = {
            "models": [
                {"name": "llama3.2"},
                {"name": "mistral"},
            ]
        }
        
        with patch("src.retrieval.llm_provider.get_settings", return_value=mock_settings):
            with patch("src.retrieval.llm_provider.requests.get", return_value=mock_response) as mock_get:
                with patch("src.retrieval.llm_provider.Ollama") as mock_ollama_class:
                    mock_ollama_instance = Mock()
                    mock_ollama_class.return_value = mock_ollama_instance
                    
                    result = get_llm()
                    
                    assert result == mock_ollama_instance
                    mock_get.assert_called_once_with(
                        "http://localhost:11434/api/tags",
                        timeout=2,
                    )
                    mock_ollama_class.assert_called_once_with(
                        model="llama3.2",
                        base_url="http://localhost:11434",
                        request_timeout=120.0,
                        context_window=8192,
                        system_prompt=None,
                    )

    def test_system_prompt_passed_to_ollama(self, clear_cache):
        """Test that system_prompt is passed to Ollama constructor.
        
        Mocks:
        - get_settings() to return ollama as provider
        - requests.get() to return successful health check response
        - Ollama constructor to track instantiation
        
        Expected:
        - system_prompt parameter is passed to Ollama constructor
        """
        mock_settings = Mock()
        mock_settings.LLM_PROVIDER = "ollama"
        mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
        mock_settings.OLLAMA_MODEL = "llama3.2"
        mock_settings.OLLAMA_TIMEOUT = 120.0
        mock_settings.OLLAMA_CONTEXT_WINDOW = 8192
        
        mock_response = Mock()
        mock_response.json.return_value = {
            "models": [{"name": "llama3.2"}]
        }
        
        custom_prompt = "You are a Python expert."
        
        with patch("src.retrieval.llm_provider.get_settings", return_value=mock_settings):
            with patch("src.retrieval.llm_provider.requests.get", return_value=mock_response):
                with patch("src.retrieval.llm_provider.Ollama") as mock_ollama_class:
                    mock_ollama_instance = Mock()
                    mock_ollama_class.return_value = mock_ollama_instance
                    
                    result = get_llm(system_prompt=custom_prompt)
                    
                    assert result == mock_ollama_instance
                    mock_ollama_class.assert_called_once_with(
                        model="llama3.2",
                        base_url="http://localhost:11434",
                        request_timeout=120.0,
                        context_window=8192,
                        system_prompt=custom_prompt,
                    )


class TestCaching:
    """Tests for LLM instance caching behavior."""

    def test_cache_returns_same_instance_for_identical_key(self, clear_cache):
        """Test that cache returns same instance for identical (provider, system_prompt) key.
        
        Mocks:
        - get_settings() to return openai as provider
        - OpenAI constructor to track instantiation
        
        Expected:
        - First call instantiates OpenAI
        - Second call with same parameters returns cached instance
        - OpenAI constructor called only once
        """
        mock_settings = Mock()
        mock_settings.LLM_PROVIDER = "openai"
        mock_settings.OPENAI_API_KEY = "test-key"
        mock_settings.OPENAI_API_BASE = "https://api.openai.com/v1"
        mock_settings.LLM_MODEL = "gpt-4"
        
        with patch("src.retrieval.llm_provider.get_settings", return_value=mock_settings):
            with patch("src.retrieval.llm_provider.OpenAI") as mock_openai_class:
                mock_openai_instance = Mock()
                mock_openai_class.return_value = mock_openai_instance
                
                # First call
                result1 = get_llm()
                assert result1 == mock_openai_instance
                assert mock_openai_class.call_count == 1
                
                # Second call with same parameters
                result2 = get_llm()
                assert result2 == mock_openai_instance
                assert mock_openai_class.call_count == 1  # Still 1, not 2
                
                # Both results are identical
                assert result1 is result2

    def test_cache_returns_different_instance_for_different_system_prompt(self, clear_cache):
        """Test that cache returns different instance for different system_prompt.
        
        Mocks:
        - get_settings() to return openai as provider
        - OpenAI constructor to track instantiation
        
        Expected:
        - First call with system_prompt=None caches one instance
        - Second call with system_prompt="different" caches different instance
        - OpenAI constructor called twice
        - Both instances are different
        """
        mock_settings = Mock()
        mock_settings.LLM_PROVIDER = "openai"
        mock_settings.OPENAI_API_KEY = "test-key"
        mock_settings.OPENAI_API_BASE = "https://api.openai.com/v1"
        mock_settings.LLM_MODEL = "gpt-4"
        
        with patch("src.retrieval.llm_provider.get_settings", return_value=mock_settings):
            with patch("src.retrieval.llm_provider.OpenAI") as mock_openai_class:
                mock_instance1 = Mock()
                mock_instance2 = Mock()
                mock_openai_class.side_effect = [mock_instance1, mock_instance2]
                
                # First call with no system_prompt
                result1 = get_llm()
                assert result1 == mock_instance1
                
                # Second call with different system_prompt
                result2 = get_llm(system_prompt="Custom prompt")
                assert result2 == mock_instance2
                
                # OpenAI constructor called twice
                assert mock_openai_class.call_count == 2
                
                # Both results are different
                assert result1 is not result2


class TestOllamaConnectionErrors:
    """Tests for Ollama connection error handling."""

    def test_connection_error_when_ollama_unreachable(self, clear_cache):
        """Test ConnectionError when Ollama is unreachable.
        
        Mocks:
        - get_settings() to return ollama as provider
        - requests.get() to raise RequestException
        
        Expected:
        - ConnectionError is raised with helpful message
        - Error message includes base URL
        """
        mock_settings = Mock()
        mock_settings.LLM_PROVIDER = "ollama"
        mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
        mock_settings.OLLAMA_MODEL = "llama3.2"
        
        with patch("src.retrieval.llm_provider.get_settings", return_value=mock_settings):
            with patch("src.retrieval.llm_provider.requests.get") as mock_get:
                import requests
                mock_get.side_effect = requests.RequestException("Connection refused")
                
                with pytest.raises(ConnectionError) as exc_info:
                    get_llm()
                
                error_message = str(exc_info.value)
                assert "Cannot connect to Ollama" in error_message
                assert "http://localhost:11434" in error_message
                assert "Is Ollama running?" in error_message

    def test_connection_error_when_model_not_found(self, clear_cache):
        """Test ConnectionError when requested model is not found in Ollama.
        
        Mocks:
        - get_settings() to return ollama as provider
        - requests.get() to return response without requested model
        
        Expected:
        - ConnectionError is raised with helpful message
        - Error message includes missing model name
        - Error message includes available models
        - Error message includes ollama pull command
        """
        mock_settings = Mock()
        mock_settings.LLM_PROVIDER = "ollama"
        mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
        mock_settings.OLLAMA_MODEL = "llama3.2"
        
        mock_response = Mock()
        mock_response.json.return_value = {
            "models": [
                {"name": "mistral"},
                {"name": "neural-chat"},
            ]
        }
        
        with patch("src.retrieval.llm_provider.get_settings", return_value=mock_settings):
            with patch("src.retrieval.llm_provider.requests.get", return_value=mock_response):
                with pytest.raises(ConnectionError) as exc_info:
                    get_llm()
                
                error_message = str(exc_info.value)
                assert "Model 'llama3.2' not found" in error_message
                assert "mistral" in error_message
                assert "neural-chat" in error_message
                assert "ollama pull llama3.2" in error_message

    def test_connection_error_when_model_not_found_empty_list(self, clear_cache):
        """Test ConnectionError when Ollama returns empty model list.
        
        Mocks:
        - get_settings() to return ollama as provider
        - requests.get() to return response with empty models list
        
        Expected:
        - ConnectionError is raised with helpful message
        - Error message indicates no models available
        """
        mock_settings = Mock()
        mock_settings.LLM_PROVIDER = "ollama"
        mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
        mock_settings.OLLAMA_MODEL = "llama3.2"
        
        mock_response = Mock()
        mock_response.json.return_value = {"models": []}
        
        with patch("src.retrieval.llm_provider.get_settings", return_value=mock_settings):
            with patch("src.retrieval.llm_provider.requests.get", return_value=mock_response):
                with pytest.raises(ConnectionError) as exc_info:
                    get_llm()
                
                error_message = str(exc_info.value)
                assert "Model 'llama3.2' not found" in error_message
                assert "none" in error_message.lower()

    def test_connection_error_on_invalid_json_response(self, clear_cache):
        """Test ConnectionError when Ollama returns invalid JSON.
        
        Mocks:
        - get_settings() to return ollama as provider
        - requests.get() to return response with invalid JSON
        
        Expected:
        - ConnectionError is raised with helpful message
        - Error message indicates invalid response
        """
        mock_settings = Mock()
        mock_settings.LLM_PROVIDER = "ollama"
        mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
        mock_settings.OLLAMA_MODEL = "llama3.2"
        
        mock_response = Mock()
        mock_response.json.side_effect = ValueError("Invalid JSON")
        
        with patch("src.retrieval.llm_provider.get_settings", return_value=mock_settings):
            with patch("src.retrieval.llm_provider.requests.get", return_value=mock_response):
                with pytest.raises(ConnectionError) as exc_info:
                    get_llm()
                
                error_message = str(exc_info.value)
                assert "Invalid response from Ollama" in error_message
                assert "http://localhost:11434" in error_message


class TestInvalidProvider:
    """Tests for invalid provider error handling."""

    def test_value_error_for_invalid_provider(self, clear_cache):
        """Test ValueError when LLM_PROVIDER is invalid.
        
        Mocks:
        - get_settings() to return invalid provider
        
        Expected:
        - ValueError is raised with helpful message
        - Error message includes invalid provider name
        - Error message indicates valid options
        """
        mock_settings = Mock()
        mock_settings.LLM_PROVIDER = "invalid_provider"
        
        with patch("src.retrieval.llm_provider.get_settings", return_value=mock_settings):
            with pytest.raises(ValueError) as exc_info:
                get_llm()
            
            error_message = str(exc_info.value)
            assert "Invalid LLM_PROVIDER" in error_message
            assert "invalid_provider" in error_message
            assert "openai" in error_message
            assert "ollama" in error_message
