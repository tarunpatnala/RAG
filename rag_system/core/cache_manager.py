"""Content caching with hash-based change detection.

Stores a content hash per URL so re-ingestion is skipped when
the upstream page has not changed.  Uses SQLite for metadata
and the filesystem for raw content snapshots.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import xxhash

from rag_system.config import CacheConfig


class CacheManager:
    """Manages URL content caching with change detection."""

    def __init__(self, config: CacheConfig) -> None:
        self._dir = Path(config.cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._db_path = config.metadata_db
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    # ------------------------------------------------------------------
    # Database setup
    # ------------------------------------------------------------------
    def _init_db(self) -> None:
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS url_cache (
                url            TEXT PRIMARY KEY,
                content_hash   TEXT NOT NULL,
                etag           TEXT,
                last_modified  TEXT,
                last_fetched   REAL NOT NULL,
                content_file   TEXT NOT NULL,
                metadata       TEXT
            )
        """)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def content_hash(self, content: str) -> str:
        """Return a fast 128-bit xxHash hex digest of *content*."""
        return xxhash.xxh128_hexdigest(content.encode("utf-8"))

    def has_changed(self, url: str, new_content: str) -> bool:
        """Return True if *new_content* differs from the cached version."""
        row = self._get_row(url)
        if row is None:
            return True
        return row["content_hash"] != self.content_hash(new_content)

    def get_cached_content(self, url: str) -> Optional[str]:
        """Return previously cached raw content for *url*, or None."""
        row = self._get_row(url)
        if row is None:
            return None
        content_path = Path(row["content_file"])
        if not content_path.exists():
            return None
        return content_path.read_text(encoding="utf-8")

    def get_cached_metadata(self, url: str) -> Optional[dict]:
        """Return cached HTTP metadata (etag, last-modified) for conditional requests."""
        row = self._get_row(url)
        if row is None:
            return None
        return {
            "etag": row["etag"],
            "last_modified": row["last_modified"],
        }

    def update(
        self,
        url: str,
        content: str,
        etag: Optional[str] = None,
        last_modified: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        """Write content to cache and update the metadata DB.  Returns the content hash."""
        h = self.content_hash(content)
        content_file = self._content_path(url)
        content_file.parent.mkdir(parents=True, exist_ok=True)
        content_file.write_text(content, encoding="utf-8")

        self._conn.execute(
            """
            INSERT INTO url_cache (url, content_hash, etag, last_modified, last_fetched, content_file, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(url) DO UPDATE SET
                content_hash  = excluded.content_hash,
                etag          = excluded.etag,
                last_modified = excluded.last_modified,
                last_fetched  = excluded.last_fetched,
                content_file  = excluded.content_file,
                metadata      = excluded.metadata
            """,
            (
                url,
                h,
                etag,
                last_modified,
                time.time(),
                str(content_file),
                json.dumps(metadata) if metadata else None,
            ),
        )
        self._conn.commit()
        return h

    def is_cached(self, url: str) -> bool:
        return self._get_row(url) is not None

    def get_all_cached_urls(self) -> list[str]:
        """Return all URLs currently in the cache."""
        cur = self._conn.execute("SELECT url FROM url_cache ORDER BY last_fetched DESC")
        return [row[0] for row in cur.fetchall()]

    def clear(self, url: Optional[str] = None) -> None:
        """Clear cache for a single URL or the entire cache."""
        if url:
            row = self._get_row(url)
            if row:
                Path(row["content_file"]).unlink(missing_ok=True)
            self._conn.execute("DELETE FROM url_cache WHERE url = ?", (url,))
        else:
            self._conn.execute("DELETE FROM url_cache")
            # Remove all cached content files
            for p in self._dir.glob("content_*"):
                p.unlink(missing_ok=True)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _content_path(self, url: str) -> Path:
        safe_name = quote(url, safe="")[:200]
        return self._dir / f"content_{safe_name}.txt"

    def _get_row(self, url: str) -> Optional[dict]:
        cur = self._conn.execute(
            "SELECT * FROM url_cache WHERE url = ?", (url,)
        )
        row = cur.fetchone()
        if row is None:
            return None
        cols = [desc[0] for desc in cur.description]
        return dict(zip(cols, row))

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
