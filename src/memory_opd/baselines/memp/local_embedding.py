"""LangChain adapter for the frozen local FastEmbed checkpoint."""

from __future__ import annotations

from fastembed import TextEmbedding
from langchain_core.embeddings import Embeddings


class FastEmbedLangChainEmbeddings(Embeddings):
    def __init__(self, model: str, cache_dir: str):
        self.model = model
        self._backend = TextEmbedding(model_name=model, cache_dir=cache_dir)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._backend.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._backend.embed([text]))).tolist()
