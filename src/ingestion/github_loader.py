"""GitHub repository ingestion via cloning and file loading."""
import os
import tempfile
from typing import List, Optional
from git import Repo
from llama_index.core import SimpleDirectoryReader, Document

# File extensions to include
INCLUDE_EXTENSIONS = {
    '.py', '.ts', '.tsx', '.js', '.jsx',
    '.md', '.rst', '.txt',
    '.json', '.yaml', '.yml', '.toml',
    '.html', '.css', '.scss',
    '.sql', '.sh', '.bash',
    '.go', '.rs', '.java', '.kt',
}

# Directories to exclude
EXCLUDE_DIRS = {
    'node_modules', 'vendor', 'dist', 'build',
    '.git', '.github', '__pycache__', '.pytest_cache',
    'venv', '.venv', 'env', '.env',
    'coverage', '.coverage', 'htmlcov',
}

MAX_FILE_SIZE = 100 * 1024  # 100KB


def should_include_file(file_path: str) -> bool:
    """Check if a file should be included in the index."""
    # Check extension
    _, ext = os.path.splitext(file_path)
    if ext.lower() not in INCLUDE_EXTENSIONS:
        return False

    # Check file size (only if file exists)
    if os.path.exists(file_path):
        try:
            if os.path.getsize(file_path) > MAX_FILE_SIZE:
                return False
        except OSError:
            return False

    # Check if in excluded directory
    parts = file_path.split(os.sep)
    if any(part in EXCLUDE_DIRS for part in parts):
        return False

    return True


def clone_repo(repo_url: str, target_dir: Optional[str] = None) -> str:
    """Clone a git repository and return the path.

    Args:
        repo_url: URL of the git repository
        target_dir: Optional directory to clone into (uses temp if None)

    Returns:
        Path to the cloned repository
    """
    if target_dir is None:
        target_dir = tempfile.mkdtemp(prefix='repo_')

    if os.path.exists(os.path.join(target_dir, '.git')):
        # Already cloned, just pull
        repo = Repo(target_dir)
        repo.remotes.origin.pull()
    else:
        Repo.clone_from(repo_url, target_dir)

    return target_dir


def load_github_repo(repo_url: str) -> List[Document]:
    """Load all documents from a GitHub repository.

    Args:
        repo_url: URL of the GitHub repository

    Returns:
        List of LlamaIndex Document objects
    """
    repo_dir = clone_repo(repo_url)

    # Get all files that pass the filter
    all_files = []
    for root, dirs, files in os.walk(repo_dir):
        # Prune excluded directories
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

        for file in files:
            file_path = os.path.join(root, file)
            if should_include_file(file_path):
                all_files.append(file_path)

    if not all_files:
        return []

    # Load documents
    reader = SimpleDirectoryReader(input_files=all_files)
    documents = reader.load_data()

    # Add metadata
    repo_name = repo_url.rstrip('/').split('/')[-1]
    for doc in documents:
        rel_path = os.path.relpath(doc.metadata.get('file_path', ''), repo_dir)
        doc.metadata['source'] = f"{repo_name}/{rel_path}"
        doc.metadata['source_type'] = 'code'
        doc.metadata['repo_url'] = repo_url

    return documents
