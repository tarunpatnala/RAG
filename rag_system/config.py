"""Centralized configuration for the RAG system."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


@dataclass
class ScraperConfig:
    """Web scraping settings."""
    user_agent: str = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    )
    request_timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    respect_robots_txt: bool = True


@dataclass
class ChunkingConfig:
    """Text chunking settings."""
    chunk_size: int = 512          # tokens
    chunk_overlap: int = 64        # tokens (~12%)
    min_chunk_size: int = 50       # tokens – discard tiny fragments
    separators: list[str] = field(default_factory=lambda: [
        "\n## ", "\n### ", "\n#### ",   # markdown headings
        "\n\n", "\n", ". ", " ",
    ])
    include_section_context: bool = True  # prepend heading hierarchy


@dataclass
class EmbeddingConfig:
    """Embedding model settings."""
    model_name: str = "all-MiniLM-L6-v2"
    dimension: int = 384
    batch_size: int = 64
    normalize: bool = True
    device: str = "cpu"  # "cuda" if GPU available


@dataclass
class VectorStoreConfig:
    """ChromaDB vector store settings."""
    collection_name: str = "rag_documents"
    persist_directory: str = str(_project_root() / "data" / "chroma_db")
    distance_metric: str = "cosine"  # cosine | l2 | ip
    search_top_k: int = 10


@dataclass
class CacheConfig:
    """Content caching settings."""
    cache_dir: str = str(_project_root() / "data" / "cache")
    metadata_db: str = str(_project_root() / "data" / "cache" / "cache_meta.db")


@dataclass
class PageIndexConfig:
    """PageIndex hierarchical tree index settings."""
    index_dir: str = str(_project_root() / "data" / "page_index")
    llm_model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 4096


@dataclass
class RetrievalCacheConfig:
    """Semantic retrieval cache settings."""
    db_path: str = str(_project_root() / "data" / "cache" / "retrieval_cache.db")
    similarity_threshold: float = 0.92   # cosine similarity for semantic match
    ttl_seconds: float = 86_400          # 24 hours
    max_entries: int = 1_000


@dataclass
class BM25Config:
    """BM25 keyword retrieval settings."""
    index_dir: str = str(_project_root() / "data" / "bm25_index")
    k1: float = 1.5       # term frequency saturation
    b: float = 0.75       # document length normalization (0=none, 1=full)
    enabled: bool = True   # master toggle


@dataclass
class RAGConfig:
    """Top-level configuration aggregating all sub-configs."""
    scraper: ScraperConfig = field(default_factory=ScraperConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    page_index: PageIndexConfig = field(default_factory=PageIndexConfig)
    retrieval_cache: RetrievalCacheConfig = field(default_factory=RetrievalCacheConfig)
    bm25: BM25Config = field(default_factory=BM25Config)

    # LLM key – read from env
    anthropic_api_key: Optional[str] = field(
        default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY")
    )

    documents_dir: str = str(_project_root() / "data" / "documents")

    @classmethod
    def load(cls) -> "RAGConfig":
        """Load configuration (from env / defaults)."""
        return cls()
