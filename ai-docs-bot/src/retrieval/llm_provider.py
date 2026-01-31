"""LLM Provider Factory.

Provides a factory function to instantiate LLM providers (OpenAI or Ollama)
based on configuration, with health checks and helpful error messages.
"""

from typing import Any

import requests
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI

from src.config.settings import get_settings

# Cache to avoid repeated health checks
_llm_cache: dict[str, Any] = {}


def get_llm(system_prompt: str | None = None) -> OpenAI | Ollama:
    """Get an LLM instance based on configuration.
    
    Returns an OpenAI or Ollama LLM instance based on the LLM_PROVIDER
    environment variable. Caches the instance per process to avoid
    repeated health checks for Ollama.
    
    Args:
        system_prompt: Optional system prompt to use for the LLM.
        
    Returns:
        An OpenAI or Ollama LLM instance.
        
    Raises:
        ValueError: If LLM_PROVIDER is not 'openai' or 'ollama'.
        ConnectionError: If Ollama provider is selected but unreachable.
    """
    settings = get_settings()
    provider = settings.LLM_PROVIDER
    
    # Check cache first (key includes system_prompt to handle different contexts)
    cache_key = f"{provider}:{system_prompt}"
    if cache_key in _llm_cache:
        return _llm_cache[cache_key]
    
    if provider == "openai":
        llm = OpenAI(
            api_key=settings.OPENAI_API_KEY,
            api_base=settings.OPENAI_API_BASE,
            model=settings.LLM_MODEL,
            temperature=0.1,
            max_tokens=1024,
            system_prompt=system_prompt,
        )
        _llm_cache[cache_key] = llm
        return llm
    
    elif provider == "ollama":
        # Health check for Ollama
        base_url = settings.OLLAMA_BASE_URL
        model = settings.OLLAMA_MODEL
        
        try:
            response = requests.get(
                f"{base_url}/api/tags",
                timeout=2,
            )
            response.raise_for_status()
        except (requests.RequestException, requests.Timeout) as e:
            raise ConnectionError(
                f"Cannot connect to Ollama at {base_url}. Is Ollama running?"
            ) from e
        
        # Check if model exists
        try:
            tags_data = response.json()
            models = tags_data.get("models", [])
            model_names = [m.get("name", "") for m in models]
            
            # Check for exact match or match with :latest suffix
            model_found = (
                model in model_names or 
                f"{model}:latest" in model_names
            )
            
            if not model_names or not model_found:
                raise ConnectionError(
                    f"Model '{model}' not found in Ollama. "
                    f"Available models: {', '.join(model_names) if model_names else 'none'}. "
                    f"Run: ollama pull {model}"
                )
        except ValueError as e:
            raise ConnectionError(
                f"Invalid response from Ollama at {base_url}: {e}"
            ) from e
        
        llm = Ollama(
            model=model,
            base_url=base_url,
            request_timeout=settings.OLLAMA_TIMEOUT,
            context_window=settings.OLLAMA_CONTEXT_WINDOW,
            system_prompt=system_prompt,
        )
        _llm_cache[cache_key] = llm
        return llm
    
    else:
        raise ValueError(
            f"Invalid LLM_PROVIDER: {provider}. Must be 'openai' or 'ollama'."
        )
