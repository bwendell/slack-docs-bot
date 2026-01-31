# tests/test_rag.py
import pytest
from src.retrieval.query_engine import QueryResult, Source
from src.bot.responses import format_response
from src.ingestion.github_loader import should_include_file, INCLUDE_EXTENSIONS

class TestQueryResult:
    def test_query_result_creation(self):
        result = QueryResult(
            answer="Test answer",
            sources=[Source("snippet", "path", "docs", 0.9)]
        )
        assert result.answer == "Test answer"
        assert len(result.sources) == 1

class TestResponseFormatting:
    def test_format_response_with_sources(self):
        result = QueryResult(
            answer="The answer",
            sources=[
                Source("snippet", "https://docs.example.com", "docs", 0.9)
            ]
        )
        formatted = format_response(result)
        assert "The answer" in formatted
        assert "Sources:" in formatted

class TestFileFiltering:
    def test_includes_python_files(self):
        assert ".py" in INCLUDE_EXTENSIONS

    def test_includes_markdown_files(self):
        assert ".md" in INCLUDE_EXTENSIONS

    def test_excludes_images(self):
        assert ".png" not in INCLUDE_EXTENSIONS
        assert ".jpg" not in INCLUDE_EXTENSIONS
