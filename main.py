#!/usr/bin/env python3
"""CLI entry point for the Production RAG System.

Commands:
    ingest   — Scrape, process, chunk, embed, and index URLs
    query    — Ask questions against the indexed content
    status   — Show system status (collection size, cached URLs, etc.)
    clear    — Clear the vector store and/or cache
"""

from __future__ import annotations

import logging
import sys

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

from rag_system.config import RAGConfig
from rag_system.core.rag_engine import RAGEngine
from rag_system.pipeline import IngestionPipeline

console = Console()

# Default CommBank URLs
DEFAULT_URLS = [
    "https://www.commbank.com.au/home-loans.html",
    "https://www.commbank.com.au/personal-loans.html",
    "https://www.commbank.com.au/credit-cards.html",
    "https://www.commbank.com.au/commbank-yello.html",
    "https://www.commbank.com.au/business.html",
]


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )
    # Quiet noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("trafilatura").setLevel(logging.WARNING)


# ======================================================================
# CLI Group
# ======================================================================
@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging.")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """Production RAG System with PageIndex, Vector Search & Caching."""
    _setup_logging(verbose)
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose


# ======================================================================
# INGEST command
# ======================================================================
@cli.command()
@click.option(
    "--urls", "-u",
    multiple=True,
    help="URLs to ingest. If omitted, uses default CommBank URLs.",
)
@click.option("--force", "-f", is_flag=True, help="Force re-processing (ignore cache).")
def ingest(urls: tuple[str, ...], force: bool) -> None:
    """Scrape, process, chunk, embed, and index URLs."""
    target_urls = list(urls) if urls else DEFAULT_URLS

    console.print(Panel(
        f"[bold]Ingesting {len(target_urls)} URLs[/bold]\n"
        + "\n".join(f"  - {u}" for u in target_urls),
        title="RAG Ingestion Pipeline",
        border_style="blue",
    ))

    config = RAGConfig.load()
    pipeline = IngestionPipeline(config)

    try:
        stats = pipeline.ingest(target_urls, force=force)
    finally:
        pipeline.close()

    # ── Summary ───────────────────────────────────────────────────
    table = Table(title="Ingestion Summary", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("URLs processed", str(stats.urls_processed))
    table.add_row("URLs skipped (cached)", str(stats.urls_skipped_cached))
    table.add_row("URLs failed", str(stats.urls_failed))
    table.add_row("URLs removed (orphan cleanup)", str(stats.urls_removed))
    table.add_row("Total chunks", str(stats.total_chunks))
    table.add_row("Total tokens", str(stats.total_tokens))
    table.add_row("Duration", f"{stats.duration_seconds:.1f}s")
    console.print(table)

    # Show removed URLs if any
    if stats.removed_urls:
        removed_table = Table(title="Removed URLs (Orphan Cleanup)", show_header=True)
        removed_table.add_column("URL", style="red")
        for url in stats.removed_urls:
            removed_table.add_row(url)
        console.print(removed_table)

    # Per-URL details
    detail_table = Table(title="Per-URL Details", show_header=True)
    detail_table.add_column("URL", style="dim", max_width=60)
    detail_table.add_column("Status")
    detail_table.add_column("Chunks", justify="right")
    detail_table.add_column("Tokens", justify="right")
    for d in stats.details:
        status_style = {
            "processed": "green",
            "cached": "yellow",
            "failed": "red",
        }.get(d["status"], "white")
        detail_table.add_row(
            d["url"],
            f"[{status_style}]{d['status']}[/{status_style}]",
            str(d.get("chunks", 0)),
            str(d.get("tokens", 0)),
        )
    console.print(detail_table)


# ======================================================================
# QUERY command
# ======================================================================
@cli.command()
@click.argument("question", required=False)
@click.option("--top-k", "-k", default=8, help="Number of chunks to retrieve.")
@click.option("--url-filter", help="Restrict search to a specific source URL.")
@click.option("--no-page-index", is_flag=True, help="Disable PageIndex reasoning.")
@click.option("--no-generate", is_flag=True, help="Return raw context without LLM generation.")
@click.option("--no-cache", is_flag=True, help="Bypass retrieval cache (force fresh retrieval).")
@click.option("--json", "output_json", is_flag=True, help="Output results in search schema JSON.")
@click.option("--interactive", "-i", is_flag=True, help="Enter interactive query mode.")
def query(
    question: str | None,
    top_k: int,
    url_filter: str | None,
    no_page_index: bool,
    no_generate: bool,
    no_cache: bool,
    output_json: bool,
    interactive: bool,
) -> None:
    """Ask a question against the indexed content."""
    config = RAGConfig.load()
    pipeline = IngestionPipeline(config)

    page_trees = pipeline.load_page_trees()
    engine = RAGEngine(
        config=config,
        embedder=pipeline.embedder,
        vector_store=pipeline.vector_store,
        page_trees=page_trees,
    )

    store_count = pipeline.vector_store.count
    if store_count == 0:
        console.print("[yellow]Vector store is empty. Run 'ingest' first.[/yellow]")
        pipeline.close()
        return

    cache_stats = engine.retrieval_cache.stats()
    console.print(
        f"[dim]Vector store: {store_count} chunks | "
        f"PageIndex: {len(page_trees)} documents | "
        f"Query cache: {cache_stats['total_entries']} entries, "
        f"{cache_stats['total_hits']} hits[/dim]"
    )

    def _run_query(q: str) -> None:
        console.print(f"\n[bold cyan]Q:[/bold cyan] {q}\n")
        response = engine.query(
            question=q,
            top_k=top_k,
            url_filter=url_filter,
            use_page_index=not no_page_index,
            generate_answer=not no_generate,
            use_cache=not no_cache,
        )

        # ── JSON schema output ─────────────────────────────────────
        if output_json:
            import json as _json
            schema = response.to_search_schema()
            console.print_json(_json.dumps(schema, indent=2, ensure_ascii=False))
            return

        # ── Rich table output (default) ────────────────────────────
        # Show cache hit indicator
        if response.cache_hit:
            console.print(
                f"[bold magenta]CACHE HIT ({response.cache_hit_type})[/bold magenta] "
                f"— served from retrieval cache\n"
            )

        console.print(Panel(response.answer, title="Answer", border_style="green"))

        # Source table
        src_table = Table(title="Sources", show_header=True)
        src_table.add_column("#", justify="right", width=3)
        src_table.add_column("Score", justify="right", width=6)
        src_table.add_column("Reranked", justify="right", width=8)
        src_table.add_column("Source", width=8)
        src_table.add_column("URL", max_width=50)
        src_table.add_column("Section", max_width=30)
        for i, s in enumerate(response.sources[:8], 1):
            src_table.add_row(
                str(i),
                f"{s.score:.3f}",
                f"{s.reranker_score:.3f}",
                s.source,
                s.metadata.get("source_url", ""),
                s.metadata.get("section_heading", ""),
            )
        console.print(src_table)

        timing_parts = [
            f"Retrieval: {response.retrieval_time_ms:.0f}ms",
            f"Generation: {response.generation_time_ms:.0f}ms",
            f"Total: {response.total_time_ms:.0f}ms",
        ]
        if response.cache_hit:
            timing_parts.append(f"Cache: {response.cache_hit_type}")
        console.print(f"[dim]{' | '.join(timing_parts)}[/dim]")

    if interactive:
        console.print("[bold]Interactive mode. Type 'quit' to exit.[/bold]\n")
        while True:
            try:
                q = console.input("[bold cyan]Ask: [/bold cyan]").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q.lower() in ("quit", "exit", "q"):
                break
            if q:
                _run_query(q)
    elif question:
        _run_query(question)
    else:
        console.print("[yellow]Provide a question or use --interactive / -i mode.[/yellow]")

    pipeline.close()


# ======================================================================
# STATUS command
# ======================================================================
@cli.command()
def status() -> None:
    """Show system status."""
    config = RAGConfig.load()
    pipeline = IngestionPipeline(config)

    table = Table(title="RAG System Status", show_header=True)
    table.add_column("Component", style="cyan")
    table.add_column("Detail", style="green")

    table.add_row("Vector store chunks", str(pipeline.vector_store.count))
    table.add_row("PageIndex documents", str(len(pipeline.load_page_trees())))
    table.add_row("Embedding model", config.embedding.model_name)
    table.add_row("Embedding dimension", str(config.embedding.dimension))
    table.add_row("Chunk size (tokens)", str(config.chunking.chunk_size))
    table.add_row("Chunk overlap (tokens)", str(config.chunking.chunk_overlap))
    table.add_row("Distance metric", config.vector_store.distance_metric)
    table.add_row("LLM model", config.page_index.llm_model)
    table.add_row("API key set", "Yes" if config.anthropic_api_key else "No")

    # Retrieval cache stats
    from rag_system.core.retrieval_cache import RetrievalCache
    rc = RetrievalCache(
        db_path=config.retrieval_cache.db_path,
        similarity_threshold=config.retrieval_cache.similarity_threshold,
        ttl_seconds=config.retrieval_cache.ttl_seconds,
        max_entries=config.retrieval_cache.max_entries,
    )
    rc_stats = rc.stats()
    table.add_row("Query cache entries", str(rc_stats["total_entries"]))
    table.add_row("Query cache hits (total)", str(rc_stats["total_hits"]))
    table.add_row("Semantic threshold", f"{rc_stats['similarity_threshold']:.2f}")
    table.add_row("Cache TTL", f"{rc_stats['ttl_seconds'] / 3600:.0f}h")
    rc.close()

    console.print(table)
    pipeline.close()


# ======================================================================
# CLEAR command
# ======================================================================
@cli.command()
@click.option("--vectors", is_flag=True, help="Clear vector store.")
@click.option("--cache", is_flag=True, help="Clear content cache.")
@click.option("--query-cache", is_flag=True, help="Clear retrieval/query cache.")
@click.option("--all", "clear_all", is_flag=True, help="Clear everything.")
@click.confirmation_option(prompt="Are you sure you want to clear data?")
def clear(vectors: bool, cache: bool, query_cache: bool, clear_all: bool) -> None:
    """Clear stored data."""
    config = RAGConfig.load()
    pipeline = IngestionPipeline(config)

    if clear_all or vectors:
        pipeline.vector_store.delete_all()
        console.print("[green]Vector store cleared.[/green]")

    if clear_all or cache:
        from rag_system.core.cache_manager import CacheManager
        cm = CacheManager(config.cache)
        cm.clear()
        cm.close()
        console.print("[green]Content cache cleared.[/green]")

    if clear_all or query_cache:
        from rag_system.core.retrieval_cache import RetrievalCache
        rc = RetrievalCache(
            db_path=config.retrieval_cache.db_path,
            similarity_threshold=config.retrieval_cache.similarity_threshold,
            ttl_seconds=config.retrieval_cache.ttl_seconds,
            max_entries=config.retrieval_cache.max_entries,
        )
        rc.invalidate_all()
        rc.close()
        console.print("[green]Retrieval/query cache cleared.[/green]")

    pipeline.close()


# ======================================================================
# Entry point
# ======================================================================
if __name__ == "__main__":
    cli()
