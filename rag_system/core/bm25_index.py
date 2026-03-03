"""BM25 keyword retrieval index for the RAG system.

Provides a BM25Okapi-based keyword index as a complement to dense vector
retrieval.  During ingestion the index is built from all chunks and
persisted to disk via pickle.  At query time, the index scores each chunk
against the tokenised query and returns the top-k results.

Classes:
    BM25Tokenizer — Lightweight whitespace + stop-word tokenizer.
    BM25Index     — Build, persist, search the BM25 index.
"""

from __future__ import annotations

import logging
import pickle
import string
from pathlib import Path
from typing import Optional

from rank_bm25 import BM25Okapi

from rag_system.config import BM25Config
from rag_system.core.chunker import Chunk

logger = logging.getLogger(__name__)


# ======================================================================
# Tokenizer
# ======================================================================
class BM25Tokenizer:
    """Simple whitespace + punctuation tokenizer with stop-word removal.

    Deliberately avoids heavy NLP dependencies (no nltk, no spacy).
    Sufficient for the financial-products domain this system targets.
    """

    STOP_WORDS: frozenset[str] = frozenset({
        "a", "an", "the", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "shall", "can",
        "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "and",
        "but", "or", "nor", "not", "so", "yet", "both", "either",
        "neither", "each", "every", "all", "any", "few", "more",
        "most", "other", "some", "such", "no", "only", "own", "same",
        "than", "too", "very", "just", "about", "above", "also",
        "it", "its", "this", "that", "these", "those", "i", "me",
        "my", "we", "our", "you", "your", "he", "him", "his", "she",
        "her", "they", "them", "their", "what", "which", "who",
        "whom", "how", "when", "where", "why",
    })

    # Pre-compute translation table once
    _PUNCT_TABLE = str.maketrans("", "", string.punctuation)

    @classmethod
    def tokenize(cls, text: str) -> list[str]:
        """Lowercase, strip punctuation, remove stop words and single chars."""
        text = text.lower().translate(cls._PUNCT_TABLE)
        return [t for t in text.split() if t not in cls.STOP_WORDS and len(t) > 1]


# ======================================================================
# BM25 Index
# ======================================================================
class BM25Index:
    """BM25 keyword index for all ingested chunks.

    Builds a global ``BM25Okapi`` index from chunk texts and persists it
    to disk via pickle for fast reload across restarts.
    """

    def __init__(self, config: BM25Config) -> None:
        self._cfg = config
        self._index_dir = Path(config.index_dir)
        self._index_dir.mkdir(parents=True, exist_ok=True)

        self._bm25: Optional[BM25Okapi] = None
        self._chunk_ids: list[str] = []
        self._chunk_texts: list[str] = []
        self._chunk_metadatas: list[dict] = []

        # Attempt to load a previously persisted index
        self._load()

    # ------------------------------------------------------------------
    # Build (called during ingestion)
    # ------------------------------------------------------------------
    def build_from_chunks(self, chunks: list[Chunk]) -> None:
        """Build (or rebuild) the BM25 index from *all* chunks."""
        if not chunks:
            logger.warning("No chunks provided — BM25 index not built")
            return

        self._chunk_ids = [c.chunk_id for c in chunks]
        self._chunk_texts = [c.text for c in chunks]
        self._chunk_metadatas = [c.metadata for c in chunks]

        tokenized_corpus = [BM25Tokenizer.tokenize(t) for t in self._chunk_texts]

        self._bm25 = BM25Okapi(
            tokenized_corpus,
            k1=self._cfg.k1,
            b=self._cfg.b,
        )

        self._save()
        logger.info(
            "Built BM25 index: %d documents, k1=%.2f, b=%.2f",
            len(chunks), self._cfg.k1, self._cfg.b,
        )

    # ------------------------------------------------------------------
    # Search (called at query time)
    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        top_k: int = 10,
        url_filter: Optional[str] = None,
    ) -> list[dict]:
        """Score *query* against the BM25 index.

        Returns a list of dicts ``{id, score, document, metadata}``
        ordered by descending raw BM25 score.  Scores are **not**
        normalised — the caller must normalise before merging.
        """
        if self._bm25 is None or not self._chunk_ids:
            return []

        tokenized_query = BM25Tokenizer.tokenize(query)
        if not tokenized_query:
            return []

        scores = self._bm25.get_scores(tokenized_query)

        # Pair scores with chunk info; apply optional URL filter
        scored: list[tuple[float, int]] = []
        for idx, score in enumerate(scores):
            if score <= 0:
                continue
            if url_filter and self._chunk_metadatas[idx].get("source_url") != url_filter:
                continue
            scored.append((score, idx))

        scored.sort(key=lambda x: x[0], reverse=True)
        scored = scored[:top_k]

        return [
            {
                "id": self._chunk_ids[idx],
                "score": score,
                "document": self._chunk_texts[idx],
                "metadata": self._chunk_metadatas[idx],
            }
            for score, idx in scored
        ]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _save(self) -> None:
        path = self._index_path()
        data = {
            "chunk_ids": self._chunk_ids,
            "chunk_texts": self._chunk_texts,
            "chunk_metadatas": self._chunk_metadatas,
            "bm25": self._bm25,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Saved BM25 index to %s (%d chunks)", path, len(self._chunk_ids))

    def _load(self) -> None:
        path = self._index_path()
        if not path.exists():
            return
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self._chunk_ids = data["chunk_ids"]
            self._chunk_texts = data["chunk_texts"]
            self._chunk_metadatas = data["chunk_metadatas"]
            self._bm25 = data["bm25"]
            logger.info("Loaded BM25 index from disk: %d chunks", len(self._chunk_ids))
        except Exception as e:
            logger.warning("Failed to load BM25 index: %s", e)

    def delete(self) -> None:
        """Remove the persisted index from disk and clear in-memory state."""
        path = self._index_path()
        if path.exists():
            path.unlink()
            logger.info("Deleted BM25 index file")
        self._bm25 = None
        self._chunk_ids = []
        self._chunk_texts = []
        self._chunk_metadatas = []

    def _index_path(self) -> Path:
        return self._index_dir / "bm25_index.pkl"

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def is_ready(self) -> bool:
        """True when the index has been built and is queryable."""
        return self._bm25 is not None and len(self._chunk_ids) > 0

    @property
    def count(self) -> int:
        """Number of chunks in the index."""
        return len(self._chunk_ids)
