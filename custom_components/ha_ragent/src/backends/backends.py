from custom_components.ha_ragent.src.backends.database.base_backend import ABaseDbBackend
from custom_components.ha_ragent.src.backends.database.chromadb_backend import ChromaDbBackend
from custom_components.ha_ragent.src.backends.database.faiss_backend import FaissDbBackend
from custom_components.ha_ragent.src.backends.database.mongodb_backend import MongoDbBackend
from custom_components.ha_ragent.src.backends.embedder.base_backend import ABaseEmbedder
from custom_components.ha_ragent.src.backends.embedder.ollama_backend import OllamaEmbedder
from custom_components.ha_ragent.src.backends.embedder.openai_backend import OpenAiEmbedder
from custom_components.ha_ragent.src.backends.llm.base_backend import ALlmBaseBackend
from custom_components.ha_ragent.src.backends.llm.ollama_backend import OllamaLlmBackend
from custom_components.ha_ragent.src.backends.llm.openai_backend import OpenAiLlmBackend
from custom_components.ha_ragent.src.const import (
    BACKEND_EMBEDDING_TYPE_OLLAMA,
    BACKEND_EMBEDDING_TYPE_OPENAI_COMPATIBLE,
    BACKEND_LLM_TYPE_OLLAMA,
    BACKEND_LLM_TYPE_OPENAI_COMPATIBLE,
    BACKEND_VECTOR_DB_TYPE_CHROMA,
    BACKEND_VECTOR_DB_TYPE_FAISS,
    BACKEND_VECTOR_DB_TYPE_MONGODB,
)


def vector_db_to_class(vector_db_type: str) -> ABaseDbBackend:
    """Map a vector database type string to its backend class."""
    backend_to_class = {
        BACKEND_VECTOR_DB_TYPE_MONGODB: MongoDbBackend,
        BACKEND_VECTOR_DB_TYPE_CHROMA: ChromaDbBackend,
        BACKEND_VECTOR_DB_TYPE_FAISS: FaissDbBackend,
    }
    if vector_db_type not in backend_to_class:
        raise ValueError(f"Invalid vector database type: {vector_db_type}")
    return backend_to_class[vector_db_type]


def embedding_backend_to_class(backend_type: str) -> ABaseEmbedder:
    """Map an embedding backend type string to its backend class."""
    backend_to_class = {
        BACKEND_EMBEDDING_TYPE_OLLAMA: OllamaEmbedder,
        BACKEND_EMBEDDING_TYPE_OPENAI_COMPATIBLE: OpenAiEmbedder,
    }
    if backend_type not in backend_to_class:
        raise ValueError(f"Invalid embedding backend type: {backend_type}")
    return backend_to_class[backend_type]


def llm_backend_to_class(backend_type: str) -> ALlmBaseBackend:
    """Map an LLM backend type string to its backend class."""
    backend_to_class = {
        BACKEND_LLM_TYPE_OLLAMA: OllamaLlmBackend,
        BACKEND_LLM_TYPE_OPENAI_COMPATIBLE: OpenAiLlmBackend,
    }
    if backend_type not in backend_to_class:
        raise ValueError(f"Invalid LLM backend type: {backend_type}")
    return backend_to_class[backend_type]
