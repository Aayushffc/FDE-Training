from typing import List, Optional

from google import genai
from google.genai import types

from app.core.config import settings
from app.infrastructure.embedding_providers.base import BaseEmbeddingProvider


class GeminiEmbeddingProvider(BaseEmbeddingProvider):
    """
    Google Gemini embedding provider adapter using the modern google-genai SDK.
    Communicates with Gemini to produce 768-dimensional float vector embeddings.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        dimensions: Optional[int] = None,
    ) -> None:
        resolved_key = api_key if api_key is not None else settings.gemini_api_key
        if not resolved_key or not resolved_key.strip():
            raise ValueError(
                "GEMINI_API_KEY is not configured. Please set GEMINI_API_KEY in your environment or .env file."
            )

        self._api_key = resolved_key.strip()
        self._model = model or settings.gemini_embedding_model or "gemini-embedding-001"
        self._dimensions = dimensions or settings.embedding_dimensions or 768
        self.client = genai.Client(api_key=self._api_key)

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed_text(self, text: str) -> List[float]:
        """
        Generate a 768-dimensional embedding vector for a single text input.
        """
        if not text or not text.strip():
            raise ValueError("Text content cannot be empty for embedding generation.")

        config = types.EmbedContentConfig(output_dimensionality=self._dimensions)
        response = self.client.models.embed_content(
            model=self._model,
            contents=text.strip(),
            config=config,
        )

        if not response.embeddings or not response.embeddings[0].values:
            raise RuntimeError(
                f"Gemini API returned an empty embedding response for model '{self._model}'."
            )

        return list(response.embeddings[0].values)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embedding vectors for a batch of text strings in a single Gemini call.
        """
        if not texts:
            return []

        cleaned_texts = [t.strip() for t in texts]
        config = types.EmbedContentConfig(output_dimensionality=self._dimensions)
        response = self.client.models.embed_content(
            model=self._model,
            contents=cleaned_texts,
            config=config,
        )

        if not response.embeddings:
            raise RuntimeError(
                f"Gemini API returned no embeddings for batch of size {len(texts)}."
            )

        return [list(emb.values) for emb in response.embeddings]
