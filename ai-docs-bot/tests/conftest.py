"""Pytest fixtures for E2E testing with local LLM."""

import os
import pytest
import requests
import tempfile
from typing import List
from llama_index.core import Document


def is_ollama_available() -> bool:
    """Check if Ollama is running and reachable.
    
    Returns:
        bool: True if Ollama API is accessible, False otherwise.
    """
    try:
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        response = requests.get(f"{base_url}/api/tags", timeout=2)
        return response.status_code == 200
    except (requests.RequestException, requests.Timeout):
        return False


# Skip decorator for tests requiring Ollama
requires_ollama = pytest.mark.skipif(
    not is_ollama_available(),
    reason="Ollama not available - skipping local LLM tests"
)


@pytest.fixture
def ollama_model() -> str:
    """Return configured Ollama model name.
    
    Returns:
        str: Ollama model identifier from environment or default.
    """
    return os.getenv("OLLAMA_MODEL", "llama3.2")


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
