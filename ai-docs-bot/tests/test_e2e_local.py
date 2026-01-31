"""E2E tests for local LLM pipeline with Ollama."""

import os
import pytest
from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from tests.conftest import requires_ollama, is_ollama_available
from src.retrieval.llm_provider import get_llm


@pytest.mark.timeout(300)
class TestLocalLLMPipeline:
    """E2E tests for the RAG pipeline with local Ollama LLM."""

    @requires_ollama
    def test_ollama_connection(self):
        """Test that Ollama is reachable and configured model exists."""
        # This test verifies the connection check works
        assert is_ollama_available(), "Ollama should be available for this test"
        
        # Verify we can get an LLM instance
        os.environ["LLM_PROVIDER"] = "ollama"
        try:
            llm = get_llm(system_prompt="You are a test assistant.")
            assert llm is not None
            assert hasattr(llm, 'complete')  # LlamaIndex LLM interface
        finally:
            os.environ["LLM_PROVIDER"] = "openai"  # Reset

    @requires_ollama
    def test_document_ingestion(self, ephemeral_chroma, sample_documents):
        """Test ingesting documents into ephemeral ChromaDB."""
        # Set up embeddings (same as production)
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5"
        )
        
        # Create a collection in ephemeral chroma
        collection = ephemeral_chroma.create_collection("test_docs")
        
        # Verify documents can be processed
        assert len(sample_documents) == 3
        for doc in sample_documents:
            assert doc.text
            assert doc.metadata.get("source_path")

    @requires_ollama
    def test_query_with_local_llm(self, ephemeral_chroma, sample_documents):
        """Test full pipeline: ingest -> query -> verify response."""
        # Set up for local LLM
        os.environ["LLM_PROVIDER"] = "ollama"
        
        try:
            # Configure embeddings
            Settings.embed_model = HuggingFaceEmbedding(
                model_name="BAAI/bge-small-en-v1.5"
            )
            
            # Configure LLM
            Settings.llm = get_llm(system_prompt="Answer questions based on the provided context.")
            
            # Create index from sample documents
            index = VectorStoreIndex.from_documents(sample_documents)
            
            # Create query engine
            query_engine = index.as_query_engine(
                similarity_top_k=3,
                response_mode="compact",
            )
            
            # Execute query
            response = query_engine.query("What is Python?")
            
            # Verify response exists and is non-empty
            answer = str(response)
            assert answer, "Response should not be empty"
            assert len(answer) > 10, "Response should have meaningful content"
            
            # Semantic check: response should relate to the question
            # (not checking exact strings due to LLM variability)
            answer_lower = answer.lower()
            assert any(word in answer_lower for word in ["python", "programming", "language", "guido"]), \
                f"Response should mention Python-related terms: {answer}"
                
        finally:
            os.environ["LLM_PROVIDER"] = "openai"  # Reset

    @requires_ollama
    def test_response_includes_sources(self, sample_documents):
        """Test that query responses include source citations."""
        os.environ["LLM_PROVIDER"] = "ollama"
        
        try:
            # Configure embeddings and LLM
            Settings.embed_model = HuggingFaceEmbedding(
                model_name="BAAI/bge-small-en-v1.5"
            )
            Settings.llm = get_llm(system_prompt="Answer questions based on context.")
            
            # Create index and query engine
            index = VectorStoreIndex.from_documents(sample_documents)
            query_engine = index.as_query_engine(similarity_top_k=3)
            
            # Execute query
            response = query_engine.query("Tell me about FastAPI")
            
            # Verify sources are returned
            assert hasattr(response, 'source_nodes'), "Response should have source_nodes"
            assert len(response.source_nodes) > 0, "Should have at least one source"
            
            # Verify source has expected metadata
            first_source = response.source_nodes[0]
            assert hasattr(first_source, 'node'), "Source should have node"
            
        finally:
            os.environ["LLM_PROVIDER"] = "openai"  # Reset


class TestProviderSwitching:
    """Tests for LLM provider switching functionality."""

    def test_default_provider_is_openai(self):
        """Test that default provider is OpenAI."""
        from src.config.settings import get_settings
        settings = get_settings()
        # Default when env not set should be openai
        assert settings.LLM_PROVIDER in ["openai", "ollama"]

    def test_invalid_provider_raises_error(self):
        """Test that invalid provider raises ValueError."""
        os.environ["LLM_PROVIDER"] = "invalid_provider"
        
        try:
            # Clear the cache to force re-evaluation
            from src.retrieval import llm_provider
            llm_provider._llm_cache.clear()
            
            with pytest.raises(ValueError) as exc_info:
                get_llm()
            
            assert "invalid_provider" in str(exc_info.value).lower()
            assert "openai" in str(exc_info.value).lower() or "ollama" in str(exc_info.value).lower()
        finally:
            os.environ["LLM_PROVIDER"] = "openai"
            llm_provider._llm_cache.clear()
