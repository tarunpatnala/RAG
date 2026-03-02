"""Semantic retrieval cache — avoids re-running expensive retrieval + LLM
generation for queries that are identical or semantically similar to
previously answered queries.

Two-tier caching strategy:
  1. **Exact match** — hash the query string; O(1) lookup.
  2. **Semantic match** — embed the query and compare cosine similarity
     against cached query embeddings; returns a hit when similarity
     exceeds a configurable threshold (default 0.92).

Cache entries are automatically invalidated when the underlying content
changes (tracked via a store-level content fingerprint derived from the
set of content hashes in the vector store).

Persistence: SQLite database at ``data/cache/retrieval_cache.db``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------
@dataclass
class CacheEntry:
    """A single cached query → response mapping."""
    query: str
    query_hash: str
    query_embedding: list[float]
    answer: str
    sources_json: str         # serialised RetrievalResult list
    content_fingerprint: str  # invalidate when content changes
    top_k: int
    url_filter: str
    created_at: float
    hit_count: int = 0
    last_hit_at: float = 0.0


class RetrievalCache:
    """Persistent semantic query cache backed by SQLite."""

    def __init__(
        self,
        db_path: str,
        similarity_threshold: float = 0.92,
        ttl_seconds: float = 86_400,    # 24 hours default
        max_entries: int = 1_000,
    ) -> None:
        self._db_path = db_path
        self._threshold = similarity_threshold
        self._ttl = ttl_seconds
        self._max_entries = max_entries
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

        # In-memory index of embeddings for fast semantic matching
        self._embedding_index: list[tuple[str, np.ndarray]] = []  # (query_hash, embedding)
        self._load_embedding_index()

    # ------------------------------------------------------------------
    # Database setup
    # ------------------------------------------------------------------
    def _init_db(self) -> None:
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS retrieval_cache (
                query_hash          TEXT PRIMARY KEY,
                query               TEXT NOT NULL,
                query_embedding     TEXT NOT NULL,
                answer              TEXT NOT NULL,
                sources_json        TEXT NOT NULL,
                content_fingerprint TEXT NOT NULL,
                top_k               INTEGER NOT NULL,
                url_filter          TEXT NOT NULL DEFAULT '',
                created_at          REAL NOT NULL,
                hit_count           INTEGER NOT NULL DEFAULT 0,
                last_hit_at         REAL NOT NULL DEFAULT 0
            )
        """)
        self._conn.commit()

    def _load_embedding_index(self) -> None:
        """Load all query embeddings into memory for fast cosine search."""
        cur = self._conn.execute(
            "SELECT query_hash, query_embedding FROM retrieval_cache"
        )
        self._embedding_index = []
        for row in cur.fetchall():
            qhash = row[0]
            emb = np.array(json.loads(row[1]), dtype=np.float32)
            self._embedding_index.append((qhash, emb))
        logger.debug("Loaded %d cached query embeddings", len(self._embedding_index))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def lookup(
        self,
        query: str,
        query_embedding: list[float],
        content_fingerprint: str,
        top_k: int,
        url_filter: str = "",
    ) -> Optional[CacheEntry]:
        """Look up a cached response.

        Tries exact match first, then semantic match.
        Returns None on cache miss.
        """
        now = time.time()

        # ── Tier 1: Exact match ───────────────────────────────────────
        qhash = self._hash_query(query, top_k, url_filter)
        entry = self._get_entry(qhash)
        if entry and self._is_valid(entry, content_fingerprint, now):
            self._record_hit(qhash, now)
            logger.info("Retrieval cache HIT (exact): %s", query[:80])
            return entry

        # ── Tier 2: Semantic match ────────────────────────────────────
        if not self._embedding_index:
            return None

        q_emb = np.array(query_embedding, dtype=np.float32)
        best_hash: Optional[str] = None
        best_sim = -1.0

        for cached_hash, cached_emb in self._embedding_index:
            sim = self._cosine_similarity(q_emb, cached_emb)
            if sim > best_sim:
                best_sim = sim
                best_hash = cached_hash

        if best_sim >= self._threshold and best_hash:
            entry = self._get_entry(best_hash)
            if entry and self._is_valid(entry, content_fingerprint, now):
                # Also verify top_k and url_filter match
                if entry.top_k >= top_k and entry.url_filter == url_filter:
                    self._record_hit(best_hash, now)
                    logger.info(
                        "Retrieval cache HIT (semantic, sim=%.3f): '%s' matched cached '%s'",
                        best_sim, query[:60], entry.query[:60],
                    )
                    return entry

        return None

    def store(
        self,
        query: str,
        query_embedding: list[float],
        answer: str,
        sources_json: str,
        content_fingerprint: str,
        top_k: int,
        url_filter: str = "",
    ) -> None:
        """Store a query result in the cache."""
        qhash = self._hash_query(query, top_k, url_filter)
        now = time.time()

        self._conn.execute(
            """
            INSERT INTO retrieval_cache
                (query_hash, query, query_embedding, answer, sources_json,
                 content_fingerprint, top_k, url_filter, created_at, hit_count, last_hit_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0)
            ON CONFLICT(query_hash) DO UPDATE SET
                answer              = excluded.answer,
                sources_json        = excluded.sources_json,
                query_embedding     = excluded.query_embedding,
                content_fingerprint = excluded.content_fingerprint,
                top_k               = excluded.top_k,
                created_at          = excluded.created_at,
                hit_count           = 0,
                last_hit_at         = 0
            """,
            (
                qhash, query,
                json.dumps(query_embedding),
                answer, sources_json,
                content_fingerprint,
                top_k, url_filter, now,
            ),
        )
        self._conn.commit()

        # Update in-memory index
        emb = np.array(query_embedding, dtype=np.float32)
        # Remove old entry if exists, then add new
        self._embedding_index = [
            (h, e) for h, e in self._embedding_index if h != qhash
        ]
        self._embedding_index.append((qhash, emb))

        # Evict oldest entries if over capacity
        self._evict_if_needed()

        logger.debug("Cached query: %s", query[:80])

    def invalidate_all(self) -> None:
        """Clear the entire retrieval cache."""
        self._conn.execute("DELETE FROM retrieval_cache")
        self._conn.commit()
        self._embedding_index.clear()
        logger.info("Retrieval cache cleared")

    def stats(self) -> dict:
        """Return cache statistics."""
        cur = self._conn.execute("""
            SELECT
                COUNT(*) as total_entries,
                COALESCE(SUM(hit_count), 0) as total_hits,
                COALESCE(MAX(last_hit_at), 0) as last_hit_at
            FROM retrieval_cache
        """)
        row = cur.fetchone()
        return {
            "total_entries": row[0],
            "total_hits": row[1],
            "last_hit_at": row[2],
            "similarity_threshold": self._threshold,
            "ttl_seconds": self._ttl,
        }

    # ------------------------------------------------------------------
    # Content fingerprinting
    # ------------------------------------------------------------------
    @staticmethod
    def compute_content_fingerprint(content_hashes: list[str]) -> str:
        """Compute a fingerprint from the set of content hashes in the store.

        If any document's content changes, the fingerprint changes,
        invalidating all cache entries that relied on the old content.
        """
        combined = "|".join(sorted(content_hashes))
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _hash_query(self, query: str, top_k: int, url_filter: str) -> str:
        """Deterministic hash for exact-match lookup."""
        raw = f"{query.strip().lower()}||{top_k}||{url_filter}"
        return hashlib.sha256(raw.encode()).hexdigest()[:20]

    def _get_entry(self, query_hash: str) -> Optional[CacheEntry]:
        cur = self._conn.execute(
            "SELECT * FROM retrieval_cache WHERE query_hash = ?",
            (query_hash,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        cols = [desc[0] for desc in cur.description]
        d = dict(zip(cols, row))
        return CacheEntry(
            query=d["query"],
            query_hash=d["query_hash"],
            query_embedding=json.loads(d["query_embedding"]),
            answer=d["answer"],
            sources_json=d["sources_json"],
            content_fingerprint=d["content_fingerprint"],
            top_k=d["top_k"],
            url_filter=d["url_filter"],
            created_at=d["created_at"],
            hit_count=d["hit_count"],
            last_hit_at=d["last_hit_at"],
        )

    def _is_valid(self, entry: CacheEntry, current_fingerprint: str, now: float) -> bool:
        """Check if a cache entry is still valid (not expired, content unchanged)."""
        # TTL check
        if now - entry.created_at > self._ttl:
            logger.debug("Cache entry expired (TTL): %s", entry.query[:60])
            return False
        # Content change check
        if entry.content_fingerprint != current_fingerprint:
            logger.debug("Cache entry invalidated (content changed): %s", entry.query[:60])
            return False
        return True

    def _record_hit(self, query_hash: str, now: float) -> None:
        self._conn.execute(
            "UPDATE retrieval_cache SET hit_count = hit_count + 1, last_hit_at = ? WHERE query_hash = ?",
            (now, query_hash),
        )
        self._conn.commit()

    def _evict_if_needed(self) -> None:
        """Remove oldest entries if cache exceeds max_entries."""
        cur = self._conn.execute("SELECT COUNT(*) FROM retrieval_cache")
        count = cur.fetchone()[0]
        if count <= self._max_entries:
            return

        to_remove = count - self._max_entries
        cur = self._conn.execute(
            "SELECT query_hash FROM retrieval_cache ORDER BY last_hit_at ASC, created_at ASC LIMIT ?",
            (to_remove,),
        )
        hashes = [row[0] for row in cur.fetchall()]
        if hashes:
            placeholders = ",".join("?" * len(hashes))
            self._conn.execute(
                f"DELETE FROM retrieval_cache WHERE query_hash IN ({placeholders})",
                hashes,
            )
            self._conn.commit()
            # Update in-memory index
            remove_set = set(hashes)
            self._embedding_index = [
                (h, e) for h, e in self._embedding_index if h not in remove_set
            ]
            logger.info("Evicted %d oldest cache entries", len(hashes))

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
