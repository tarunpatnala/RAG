"""ChromaDB vector store with full CRUD, search, and metadata filtering.

Wraps ChromaDB's persistent client to store chunks with embeddings,
supporting cosine / L2 / inner-product distance and rich metadata
filters for hybrid retrieval.
"""

from __future__ import annotations

import logging
from typing import Optional

import chromadb
from chromadb.config import Settings

from rag_system.config import VectorStoreConfig
from rag_system.core.chunker import Chunk

logger = logging.getLogger(__name__)


class VectorStore:
    """Persistent vector store backed by ChromaDB."""

    def __init__(self, config: VectorStoreConfig) -> None:
        self._cfg = config
        self._client = chromadb.PersistentClient(
            path=config.persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )

        distance_fn = {
            "cosine": "cosine",
            "l2": "l2",
            "ip": "ip",
        }.get(config.distance_metric, "cosine")

        self._collection = self._client.get_or_create_collection(
            name=config.collection_name,
            metadata={"hnsw:space": distance_fn},
        )
        logger.info(
            "VectorStore ready: collection=%s  count=%d  distance=%s",
            config.collection_name,
            self._collection.count(),
            distance_fn,
        )

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------
    def upsert_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> int:
        """Add or update chunks in the collection.  Returns count upserted."""
        if not chunks:
            return 0

        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [self._clean_metadata(c.metadata) for c in chunks]

        # ChromaDB has a batch limit of ~41666; process in batches
        batch_size = 5000
        total = 0
        for i in range(0, len(ids), batch_size):
            end = i + batch_size
            self._collection.upsert(
                ids=ids[i:end],
                embeddings=embeddings[i:end],
                documents=documents[i:end],
                metadatas=metadatas[i:end],
            )
            total += len(ids[i:end])

        logger.info("Upserted %d chunks into vector store", total)
        return total

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(
        self,
        query_embedding: list[float],
        top_k: Optional[int] = None,
        where: Optional[dict] = None,
        where_document: Optional[dict] = None,
    ) -> list[dict]:
        """Semantic search.  Returns list of dicts with keys:
        ``id``, ``document``, ``metadata``, ``distance``.
        """
        k = top_k or self._cfg.search_top_k
        kwargs: dict = {
            "query_embeddings": [query_embedding],
            "n_results": min(k, self._collection.count() or k),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        if where_document:
            kwargs["where_document"] = where_document

        results = self._collection.query(**kwargs)
        return self._format_results(results)

    def search_by_ids(self, chunk_ids: list[str]) -> list[dict]:
        """Retrieve specific chunks by their IDs."""
        if not chunk_ids:
            return []
        result = self._collection.get(
            ids=chunk_ids,
            include=["documents", "metadatas"],
        )
        items: list[dict] = []
        for i in range(len(result["ids"])):
            items.append({
                "id": result["ids"][i],
                "document": result["documents"][i] if result["documents"] else "",
                "metadata": result["metadatas"][i] if result["metadatas"] else {},
                "distance": 0.0,
            })
        return items

    def search_by_url(
        self,
        query_embedding: list[float],
        url: str,
        top_k: int = 5,
    ) -> list[dict]:
        """Search within chunks from a specific source URL."""
        return self.search(
            query_embedding,
            top_k=top_k,
            where={"source_url": url},
        )

    # ------------------------------------------------------------------
    # Management
    # ------------------------------------------------------------------
    def get_indexed_urls(self) -> set[str]:
        """Return the set of unique source URLs currently in the collection."""
        total = self._collection.count()
        if total == 0:
            return set()
        result = self._collection.get(
            limit=total,
            include=["metadatas"],
        )
        urls: set[str] = set()
        if result["metadatas"]:
            for meta in result["metadatas"]:
                if meta and "source_url" in meta:
                    urls.add(meta["source_url"])
        return urls

    def delete_by_url(self, url: str) -> None:
        """Remove all chunks from a specific URL."""
        self._collection.delete(where={"source_url": url})
        logger.info("Deleted chunks for URL: %s", url)

    def delete_all(self) -> None:
        """Clear the entire collection."""
        self._client.delete_collection(self._cfg.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._cfg.collection_name,
            metadata={"hnsw:space": self._cfg.distance_metric},
        )
        logger.info("Cleared vector store")

    @property
    def count(self) -> int:
        return self._collection.count()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _clean_metadata(meta: dict) -> dict:
        """ChromaDB requires metadata values to be str, int, float, or bool."""
        clean: dict = {}
        for k, v in meta.items():
            if v is None:
                continue
            if isinstance(v, (str, int, float, bool)):
                clean[k] = v
            elif isinstance(v, list):
                clean[k] = ", ".join(str(x) for x in v)
            else:
                clean[k] = str(v)
        return clean

    @staticmethod
    def _format_results(raw: dict) -> list[dict]:
        """Flatten ChromaDB's nested result format."""
        items: list[dict] = []
        if not raw["ids"] or not raw["ids"][0]:
            return items
        for i in range(len(raw["ids"][0])):
            items.append({
                "id": raw["ids"][0][i],
                "document": raw["documents"][0][i] if raw["documents"] else "",
                "metadata": raw["metadatas"][0][i] if raw["metadatas"] else {},
                "distance": raw["distances"][0][i] if raw["distances"] else 0.0,
            })
        return items
