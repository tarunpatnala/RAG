"""Ingestion pipeline — orchestrates scraping, processing, chunking, embedding, and indexing.

This is the main entry point for the data ingestion side of the RAG system.
It coordinates all components and handles caching logic.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from rag_system.config import RAGConfig
from rag_system.core.cache_manager import CacheManager
from rag_system.core.chunker import Chunk, SemanticChunker
from rag_system.core.document_processor import DocumentProcessor, StructuredDocument
from rag_system.core.embeddings import EmbeddingGenerator
from rag_system.core.page_index import PageIndexBuilder, PageTree
from rag_system.core.scraper import WebScraper
from rag_system.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class IngestionStats:
    """Statistics for a single ingestion run."""
    urls_processed: int = 0
    urls_skipped_cached: int = 0
    urls_failed: int = 0
    urls_removed: int = 0
    removed_urls: list[str] = field(default_factory=list)
    total_chunks: int = 0
    total_tokens: int = 0
    duration_seconds: float = 0.0
    details: list[dict] = field(default_factory=list)


class IngestionPipeline:
    """End-to-end URL ingestion pipeline."""

    def __init__(self, config: RAGConfig) -> None:
        self._cfg = config
        self._scraper = WebScraper(config.scraper)
        self._cache = CacheManager(config.cache)
        self._processor = DocumentProcessor(config.documents_dir)
        self._chunker = SemanticChunker(config.chunking)
        self._embedder = EmbeddingGenerator(config.embedding)
        self._store = VectorStore(config.vector_store)
        self._page_index_builder = PageIndexBuilder(
            config.page_index, api_key=config.anthropic_api_key
        )

    def ingest(self, urls: list[str], force: bool = False) -> IngestionStats:
        """Ingest a list of URLs.

        Args:
            urls: URLs to process.
            force: If True, bypass cache and re-process everything.

        Returns:
            IngestionStats with details of the run.
        """
        stats = IngestionStats()
        t_start = time.perf_counter()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Ingesting URLs...", total=len(urls))

            for url in urls:
                progress.update(task, description=f"Processing: {url}")
                result = self._ingest_url(url, force=force)
                stats.details.append(result)

                if result["status"] == "processed":
                    stats.urls_processed += 1
                    stats.total_chunks += result["chunks"]
                    stats.total_tokens += result["tokens"]
                elif result["status"] == "cached":
                    stats.urls_skipped_cached += 1
                elif result["status"] == "failed":
                    stats.urls_failed += 1

                progress.advance(task)

        # ── Orphan cleanup: remove data for URLs no longer in the list ──
        self._cleanup_removed_urls(urls, stats)

        stats.duration_seconds = time.perf_counter() - t_start
        return stats

    def _cleanup_removed_urls(
        self, current_urls: list[str], stats: IngestionStats
    ) -> None:
        """Remove chunks, cache, and PageIndex for URLs that are no longer in the active list."""
        current_set = set(current_urls)

        # Gather all previously known URLs from both vector store and cache
        previously_known: set[str] = set()
        previously_known.update(self._store.get_indexed_urls())
        previously_known.update(self._cache.get_all_cached_urls())

        orphaned = previously_known - current_set
        if not orphaned:
            return

        console.print(
            f"\n[yellow]Cleaning up {len(orphaned)} removed URL(s)...[/yellow]"
        )
        for url in sorted(orphaned):
            console.print(f"  [red]Removing:[/red] {url}")

            # 1. Delete chunks from vector store
            self._store.delete_by_url(url)

            # 2. Delete PageIndex tree
            self._page_index_builder.delete(url)

            # 3. Delete cache entry
            self._cache.clear(url)

            stats.urls_removed += 1
            stats.removed_urls.append(url)

        console.print(
            f"[yellow]Removed {len(orphaned)} orphaned URL(s) from all stores.[/yellow]"
        )

    def _ingest_url(self, url: str, force: bool = False) -> dict:
        """Process a single URL through the full pipeline."""
        detail = {"url": url, "status": "unknown", "chunks": 0, "tokens": 0}

        try:
            # ── Step 1: Fetch (with conditional GET) ──────────────
            cached_meta = self._cache.get_cached_metadata(url) if not force else None
            etag = cached_meta["etag"] if cached_meta else None
            last_mod = cached_meta["last_modified"] if cached_meta else None

            page = self._scraper.fetch(url, etag=etag, last_modified=last_mod)

            if page is None and not force:
                # 304 Not Modified
                console.print(f"  [dim]Skipped (not modified): {url}[/dim]")
                detail["status"] = "cached"
                return detail

            if page is None:
                # Fetch failed
                console.print(f"  [red]Failed to fetch: {url}[/red]")
                detail["status"] = "failed"
                return detail

            # ── Step 2: Check content hash ─────────────────────────
            content_hash = self._cache.content_hash(page.main_content)
            if not force and not self._cache.has_changed(url, page.main_content):
                console.print(f"  [dim]Skipped (content unchanged): {url}[/dim]")
                detail["status"] = "cached"
                # Still update the cache timestamp
                self._cache.update(
                    url, page.main_content,
                    etag=self._scraper.last_etag,
                    last_modified=self._scraper.last_modified_header,
                )
                return detail

            # ── Step 3: Process into structured document ──────────
            console.print(f"  [cyan]Processing:[/cyan] {page.title}")
            doc = self._processor.process(page, content_hash=content_hash)

            # ── Step 4: Chunk ─────────────────────────────────────
            chunks = self._chunker.chunk_document(doc)
            if not chunks:
                console.print(f"  [yellow]No chunks generated for: {url}[/yellow]")
                detail["status"] = "failed"
                return detail

            # ── Step 5: Embed ─────────────────────────────────────
            console.print(f"  [cyan]Embedding {len(chunks)} chunks...[/cyan]")
            embeddings = self._embedder.embed_chunks(chunks)

            # ── Step 6: Delete old chunks for this URL, then upsert ─
            self._store.delete_by_url(url)
            self._store.upsert_chunks(chunks, embeddings)

            # ── Step 7: Build PageIndex ───────────────────────────
            console.print(f"  [cyan]Building PageIndex...[/cyan]")
            self._page_index_builder.build(doc, chunks)

            # ── Step 8: Update cache ──────────────────────────────
            self._cache.update(
                url, page.main_content,
                etag=self._scraper.last_etag,
                last_modified=self._scraper.last_modified_header,
                metadata={"title": doc.title, "chunk_count": len(chunks)},
            )

            total_tokens = sum(c.token_count for c in chunks)
            console.print(
                f"  [green]Done:[/green] {len(chunks)} chunks, "
                f"{total_tokens} tokens"
            )

            detail["status"] = "processed"
            detail["chunks"] = len(chunks)
            detail["tokens"] = total_tokens
            return detail

        except Exception as e:
            logger.exception("Error processing %s: %s", url, e)
            console.print(f"  [red]Error: {e}[/red]")
            detail["status"] = "failed"
            detail["error"] = str(e)
            return detail

    # ------------------------------------------------------------------
    # Accessors for the query engine
    # ------------------------------------------------------------------
    @property
    def embedder(self) -> EmbeddingGenerator:
        return self._embedder

    @property
    def vector_store(self) -> VectorStore:
        return self._store

    def load_page_trees(self) -> list[PageTree]:
        return self._page_index_builder.load_all()

    def close(self) -> None:
        self._cache.close()
