"""Embedding generator using sentence-transformers.

Provides batch embedding with normalisation and a simple
in-memory cache to avoid re-computing identical texts.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from rag_system.config import EmbeddingConfig
from rag_system.core.chunker import Chunk

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate embeddings for text chunks."""

    def __init__(self, config: EmbeddingConfig) -> None:
        self._cfg = config
        logger.info("Loading embedding model: %s", config.model_name)
        self._model = SentenceTransformer(
            config.model_name,
            device=config.device,
        )
        # Quick cache: text hash -> embedding (avoids recomputation within a session)
        self._cache: dict[str, np.ndarray] = {}

    @property
    def dimension(self) -> int:
        return self._cfg.dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of raw text strings.  Returns list of float vectors."""
        if not texts:
            return []

        # Check cache hits
        to_compute: list[tuple[int, str]] = []  # (original_index, text)
        results: dict[int, list[float]] = {}

        for i, t in enumerate(texts):
            key = t[:200]  # use prefix as cache key for speed
            if key in self._cache:
                results[i] = self._cache[key].tolist()
            else:
                to_compute.append((i, t))

        if to_compute:
            batch_texts = [t for _, t in to_compute]
            embeddings = self._model.encode(
                batch_texts,
                batch_size=self._cfg.batch_size,
                normalize_embeddings=self._cfg.normalize,
                show_progress_bar=len(batch_texts) > 20,
            )
            for (orig_idx, text), emb in zip(to_compute, embeddings):
                vec = emb.tolist() if isinstance(emb, np.ndarray) else list(emb)
                results[orig_idx] = vec
                self._cache[text[:200]] = np.array(vec)

        return [results[i] for i in range(len(texts))]

    def embed_chunks(self, chunks: list[Chunk]) -> list[list[float]]:
        """Embed a list of ``Chunk`` objects by their text."""
        texts = [c.text for c in chunks]
        return self.embed_texts(texts)

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        return self.embed_texts([query])[0]

    def clear_cache(self) -> None:
        self._cache.clear()
