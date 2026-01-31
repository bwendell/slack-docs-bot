"""ChromaDB vector store with persistence and HuggingFace embeddings."""
import chromadb
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from src.config.settings import get_settings

_settings = get_settings()

# Configure embeddings globally
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

# Configure chunking strategy
# - chunk_size: 1000 chars for good context without too much noise
# - chunk_overlap: 200 chars to maintain context across chunk boundaries
Settings.node_parser = SentenceSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)


def get_chroma_client():
    """Get persistent ChromaDB client."""
    return chromadb.PersistentClient(path=_settings.CHROMA_PERSIST_DIR)


def get_or_create_collection(name: str = "knowledge_base"):
    """Get or create a ChromaDB collection."""
    client = get_chroma_client()
    return client.get_or_create_collection(name)


def get_vector_store():
    """Get ChromaDB vector store."""
    collection = get_or_create_collection()
    return ChromaVectorStore(chroma_collection=collection)


def get_storage_context():
    """Get storage context with ChromaDB vector store."""
    return StorageContext.from_defaults(vector_store=get_vector_store())


def get_index():
    """Get existing index from ChromaDB."""
    vector_store = get_vector_store()
    return VectorStoreIndex.from_vector_store(vector_store)


def create_index_from_documents(documents):
    """Create new index from documents."""
    storage_context = get_storage_context()
    return VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context
    )
