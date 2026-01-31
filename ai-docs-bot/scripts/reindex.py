#!/usr/bin/env python3
"""Manual re-indexing script for the knowledge base."""
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import get_settings
from src.ingestion.docs_loader import load_docs_from_sitemap
from src.ingestion.github_loader import load_github_repo
from src.storage.chroma_store import (
    get_chroma_client,
    create_index_from_documents,
)

def main():
    parser = argparse.ArgumentParser(description="Re-index the knowledge base")
    parser.add_argument("--docs-only", action="store_true", help="Only index docs")
    parser.add_argument("--code-only", action="store_true", help="Only index code")
    args = parser.parse_args()

    settings = get_settings()
    documents = []

    # Clear existing collection
    print("Clearing existing index...")
    client = get_chroma_client()
    try:
        client.delete_collection("knowledge_base")
    except ValueError:
        pass  # Collection doesn't exist

    # Load docs
    if not args.code_only:
        print(f"Loading docs from {settings.DOCS_SITEMAP_URL}...")
        try:
            docs = load_docs_from_sitemap(settings.DOCS_SITEMAP_URL)
            print(f"  Loaded {len(docs)} documents from docs site")
            documents.extend(docs)
        except Exception as e:
            print(f"  Error loading docs: {e}")

    # Load code
    if not args.docs_only:
        print(f"Loading code from {settings.GITHUB_REPO_URL}...")
        try:
            code_docs = load_github_repo(settings.GITHUB_REPO_URL)
            print(f"  Loaded {len(code_docs)} documents from repo")
            documents.extend(code_docs)
        except Exception as e:
            print(f"  Error loading code: {e}")

    if not documents:
        print("No documents to index!")
        sys.exit(1)

    # Create index
    print(f"Creating index with {len(documents)} documents...")
    index = create_index_from_documents(documents)
    print("Done! Index created successfully.")

if __name__ == "__main__":
    main()
