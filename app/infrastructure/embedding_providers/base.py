from abc import ABC, abstractmethod
from typing import List


class BaseEmbeddingProvider(ABC):
    """
    Abstract adapter interface for vector embedding providers.
    Follows Clean Architecture Dependency Inversion:
    application domain services (like IngestionService) interact with BaseEmbeddingProvider,
    while infrastructure adapters implement SDK-specific details (like Gemini GenAI SDK).
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Provider identifier string (e.g., 'gemini', 'openai')."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Embedding model identifier (e.g. 'gemini-embedding-001')."""
        ...

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Dimensionality of the output embedding vector (e.g., 768)."""
        ...

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Generate an embedding vector for a single text input."""
        ...

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embedding vectors for a batch of text inputs in a single call."""
        ...
