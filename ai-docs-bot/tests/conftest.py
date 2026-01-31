"""Pytest fixtures for E2E testing with local LLM."""

import os
import subprocess
import time
import pytest
import requests
import tempfile
from typing import List
from llama_index.core import Document


def _start_ollama_server() -> bool:
    """Start Ollama server if not already running.
    
    Returns:
        bool: True if server is running (started or already up), False otherwise.
    """
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # Check if already running
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=2)
        if response.status_code == 200:
            return True
    except (requests.RequestException, requests.Timeout):
        pass
    
    # Try to start Ollama server
    try:
        # Start ollama serve in the background
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        
        # Wait for server to be ready (up to 30 seconds)
        for _ in range(30):
            time.sleep(1)
            try:
                response = requests.get(f"{base_url}/api/tags", timeout=2)
                if response.status_code == 200:
                    return True
            except (requests.RequestException, requests.Timeout):
                continue
        
        return False
    except FileNotFoundError:
        # Ollama binary not found
        return False


def is_ollama_available() -> bool:
    """Check if Ollama is running and reachable.
    
    First checks if server is running, then tries to start it if not.
    
    Returns:
        bool: True if Ollama API is accessible, False otherwise.
    """
    try:
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        response = requests.get(f"{base_url}/api/tags", timeout=2)
        if response.status_code == 200:
            return True
    except (requests.RequestException, requests.Timeout):
        pass
    
    # Try to start server
    return _start_ollama_server()


def _get_ollama_model_name() -> str:
    """Get the correct Ollama model name, checking for :latest suffix.
    
    The Ollama API returns model names with :latest suffix, but users
    often configure just the base name. This function checks what's
    actually available.
    
    Returns:
        str: The model name as it appears in Ollama (with tag if needed).
    """
    base_model = os.getenv("OLLAMA_MODEL", "llama3.2")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=2)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            
            # Check for exact match first
            if base_model in model_names:
                return base_model
            
            # Check for model with :latest suffix
            if f"{base_model}:latest" in model_names:
                return f"{base_model}:latest"
    except (requests.RequestException, requests.Timeout, ValueError):
        pass
    
    return base_model


# Custom marker for tests requiring Ollama
# This marker both skips tests when Ollama is unavailable AND
# ensures the ensure_ollama_running fixture is used
requires_ollama = pytest.mark.skipif(
    not is_ollama_available(),
    reason="Ollama not available - skipping local LLM tests"
)


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "requires_ollama: mark test as requiring Ollama to be running"
    )


@pytest.fixture(scope="session")
def ensure_ollama_running(request):
    """Session-scoped fixture to ensure Ollama server is running.
    
    This fixture starts the Ollama server if it's not already running.
    It's automatically used by tests marked with @requires_ollama via
    the pytest_runtest_setup hook.
    
    Yields:
        bool: True if Ollama is available, False otherwise.
    """
    available = is_ollama_available()
    yield available


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    """Auto-use ensure_ollama_running fixture for @requires_ollama tests."""
    if item.get_closest_marker("requires_ollama"):
        # Request the fixture - this ensures Ollama is started before the test
        item.fixturenames.insert(0, "ensure_ollama_running")


@pytest.fixture
def ollama_model() -> str:
    """Return configured Ollama model name.
    
    Resolves the model name to what's actually available in Ollama,
    handling the :latest suffix automatically.
    
    Returns:
        str: Ollama model identifier as it appears in Ollama.
    """
    return _get_ollama_model_name()


@pytest.fixture
def ephemeral_chroma():
    """Create an ephemeral ChromaDB client for test isolation.
    
    Uses chromadb.EphemeralClient() which stores data in-memory only,
    ensuring tests don't pollute production data and have fast cleanup.
    
    Yields:
        chromadb.EphemeralClient: In-memory ChromaDB client.
    """
    import chromadb
    client = chromadb.EphemeralClient()
    yield client
    # Cleanup happens automatically when client goes out of scope


@pytest.fixture
def sample_documents() -> List[Document]:
    """Sample documents for testing the RAG pipeline.
    
    Provides realistic test data with metadata for document ingestion
    and retrieval testing.
    
    Returns:
        List[Document]: List of sample LlamaIndex Documents.
    """
    return [
        Document(
            text="Python is a programming language known for its simplicity and readability. "
                 "It was created by Guido van Rossum and first released in 1991.",
            metadata={"source_path": "docs/python.md", "source_type": "docs"}
        ),
        Document(
            text="FastAPI is a modern web framework for building APIs with Python. "
                 "It provides automatic OpenAPI documentation and high performance.",
            metadata={"source_path": "docs/fastapi.md", "source_type": "docs"}
        ),
        Document(
            text="ChromaDB is an open-source vector database for AI applications. "
                 "It supports similarity search using embeddings.",
            metadata={"source_path": "docs/chromadb.md", "source_type": "docs"}
        ),
    ]


@pytest.fixture
def temp_chroma_dir():
    """Create a temporary directory for ChromaDB persistence in tests.
    
    Provides isolated storage for ChromaDB when testing persistent backends.
    Automatically cleaned up after the test.
    
    Yields:
        str: Path to temporary directory.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir
