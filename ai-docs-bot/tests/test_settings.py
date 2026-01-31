"""Unit tests for LLM settings in src.config.settings module."""

import unittest
from unittest.mock import patch
from src.config.settings import get_settings


class TestLLMSettings(unittest.TestCase):
    """Test suite for LLM-related settings fields."""

    def test_llm_provider_default_is_openai(self):
        """Test default value for LLM_PROVIDER is 'openai'."""
        with patch.dict("os.environ", {}, clear=False):
            settings = get_settings()
            self.assertEqual(settings.LLM_PROVIDER, "openai")

    def test_ollama_model_default_is_llama3_2(self):
        """Test default value for OLLAMA_MODEL is 'llama3.2'."""
        with patch.dict("os.environ", {}, clear=False):
            settings = get_settings()
            self.assertEqual(settings.OLLAMA_MODEL, "llama3.2")

    def test_ollama_base_url_default(self):
        """Test default value for OLLAMA_BASE_URL is 'http://localhost:11434'."""
        with patch.dict("os.environ", {}, clear=False):
            settings = get_settings()
            self.assertEqual(settings.OLLAMA_BASE_URL, "http://localhost:11434")

    def test_ollama_timeout_default_is_120_0_float(self):
        """Test default value for OLLAMA_TIMEOUT is 120.0 (float type)."""
        with patch.dict("os.environ", {}, clear=False):
            settings = get_settings()
            self.assertEqual(settings.OLLAMA_TIMEOUT, 120.0)
            self.assertIsInstance(settings.OLLAMA_TIMEOUT, float)

    def test_ollama_context_window_default_is_8192_int(self):
        """Test default value for OLLAMA_CONTEXT_WINDOW is 8192 (int type)."""
        with patch.dict("os.environ", {}, clear=False):
            settings = get_settings()
            self.assertEqual(settings.OLLAMA_CONTEXT_WINDOW, 8192)
            self.assertIsInstance(settings.OLLAMA_CONTEXT_WINDOW, int)

    def test_ollama_timeout_correctly_cast_to_float(self):
        """Test OLLAMA_TIMEOUT is correctly cast to float type."""
        with patch.dict("os.environ", {"OLLAMA_TIMEOUT": "45.5"}, clear=False):
            settings = get_settings()
            self.assertEqual(settings.OLLAMA_TIMEOUT, 45.5)
            self.assertIsInstance(settings.OLLAMA_TIMEOUT, float)

    def test_ollama_context_window_correctly_cast_to_int(self):
        """Test OLLAMA_CONTEXT_WINDOW is correctly cast to int type."""
        with patch.dict("os.environ", {"OLLAMA_CONTEXT_WINDOW": "16384"}, clear=False):
            settings = get_settings()
            self.assertEqual(settings.OLLAMA_CONTEXT_WINDOW, 16384)
            self.assertIsInstance(settings.OLLAMA_CONTEXT_WINDOW, int)

    def test_llm_provider_env_override(self):
        """Test environment variable override for LLM_PROVIDER."""
        with patch.dict("os.environ", {"LLM_PROVIDER": "ollama"}, clear=False):
            settings = get_settings()
            self.assertEqual(settings.LLM_PROVIDER, "ollama")

    def test_ollama_model_env_override(self):
        """Test environment variable override for OLLAMA_MODEL."""
        with patch.dict("os.environ", {"OLLAMA_MODEL": "mistral"}, clear=False):
            settings = get_settings()
            self.assertEqual(settings.OLLAMA_MODEL, "mistral")

    def test_ollama_base_url_env_override(self):
        """Test environment variable override for OLLAMA_BASE_URL."""
        custom_url = "http://custom-host:12345"
        with patch.dict("os.environ", {"OLLAMA_BASE_URL": custom_url}, clear=False):
            settings = get_settings()
            self.assertEqual(settings.OLLAMA_BASE_URL, custom_url)

    def test_ollama_timeout_env_override(self):
        """Test environment variable override for OLLAMA_TIMEOUT."""
        with patch.dict("os.environ", {"OLLAMA_TIMEOUT": "300.0"}, clear=False):
            settings = get_settings()
            self.assertEqual(settings.OLLAMA_TIMEOUT, 300.0)
            self.assertIsInstance(settings.OLLAMA_TIMEOUT, float)

    def test_ollama_context_window_env_override(self):
        """Test environment variable override for OLLAMA_CONTEXT_WINDOW."""
        with patch.dict("os.environ", {"OLLAMA_CONTEXT_WINDOW": "32768"}, clear=False):
            settings = get_settings()
            self.assertEqual(settings.OLLAMA_CONTEXT_WINDOW, 32768)
            self.assertIsInstance(settings.OLLAMA_CONTEXT_WINDOW, int)

    def test_multiple_llm_env_overrides_together(self):
        """Test multiple LLM environment variable overrides work together."""
        env_vars = {
            "LLM_PROVIDER": "ollama",
            "OLLAMA_MODEL": "neural-chat",
            "OLLAMA_BASE_URL": "http://gpu-server:11434",
            "OLLAMA_TIMEOUT": "60.0",
            "OLLAMA_CONTEXT_WINDOW": "4096",
        }
        with patch.dict("os.environ", env_vars, clear=False):
            settings = get_settings()
            self.assertEqual(settings.LLM_PROVIDER, "ollama")
            self.assertEqual(settings.OLLAMA_MODEL, "neural-chat")
            self.assertEqual(settings.OLLAMA_BASE_URL, "http://gpu-server:11434")
            self.assertEqual(settings.OLLAMA_TIMEOUT, 60.0)
            self.assertEqual(settings.OLLAMA_CONTEXT_WINDOW, 4096)

    def test_settings_independence_between_calls(self):
        """Test that settings calls are independent and don't share state."""
        # First call with default values
        with patch.dict("os.environ", {}, clear=False):
            settings1 = get_settings()
            self.assertEqual(settings1.LLM_PROVIDER, "openai")

        # Second call with overridden values
        with patch.dict("os.environ", {"LLM_PROVIDER": "ollama"}, clear=False):
            settings2 = get_settings()
            self.assertEqual(settings2.LLM_PROVIDER, "ollama")

        # Third call back to defaults - should not be affected by second call
        with patch.dict("os.environ", {}, clear=False):
            settings3 = get_settings()
            self.assertEqual(settings3.LLM_PROVIDER, "openai")


if __name__ == "__main__":
    unittest.main()
