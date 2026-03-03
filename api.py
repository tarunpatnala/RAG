#!/usr/bin/env python3
"""FastAPI wrapper for the Production RAG System.

Provides a REST API for all RAG operations: query, ingest, status, and clear.

Run with:
    python api.py                                         # Direct run (port 8000)
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload  # Dev mode with hot reload
    uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1  # Production (single worker)

Swagger UI available at: http://localhost:8000/docs
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from rag_system.config import RAGConfig
from rag_system.core.cache_manager import CacheManager
from rag_system.core.rag_engine import RAGEngine
from rag_system.pipeline import IngestionPipeline, IngestionStats

logger = logging.getLogger(__name__)

# ======================================================================
# Default CommBank URLs (same as main.py)
# ======================================================================
DEFAULT_URLS = [
    "https://www.commbank.com.au/home-loans.html",
    "https://www.commbank.com.au/personal-loans.html",
    "https://www.commbank.com.au/credit-cards.html",
    "https://www.commbank.com.au/commbank-yello.html",
    "https://www.commbank.com.au/business.html",
]


# ======================================================================
# Pydantic Models — Request
# ======================================================================
class QueryRequest(BaseModel):
    """Request body for POST /query."""

    question: str = Field(
        ..., min_length=1, max_length=2000, description="The question to ask"
    )
    top_k: int = Field(
        default=8, ge=1, le=50, description="Number of chunks to retrieve"
    )
    url_filter: Optional[str] = Field(
        default=None, description="Restrict search to a specific source URL"
    )
    use_page_index: bool = Field(
        default=True, description="Enable PageIndex reasoning"
    )
    use_bm25: bool = Field(
        default=True, description="Enable BM25 keyword retrieval"
    )
    generate_answer: bool = Field(
        default=True,
        description="Generate LLM answer (False = raw context only)",
    )
    use_cache: bool = Field(default=True, description="Use retrieval cache")


class IngestRequest(BaseModel):
    """Request body for POST /ingest."""

    urls: Optional[list[str]] = Field(
        default=None,
        description="URLs to ingest. If omitted, uses default CommBank URLs.",
    )
    force: bool = Field(
        default=False, description="Force re-processing (ignore cache)"
    )


class ClearTarget(str, Enum):
    """What to clear."""

    vectors = "vectors"
    cache = "cache"
    query_cache = "query_cache"
    all = "all"


class ClearRequest(BaseModel):
    """Request body for POST /clear."""

    targets: list[ClearTarget] = Field(
        default=[ClearTarget.all],
        description="What to clear: vectors, cache, query_cache, or all",
    )


# ======================================================================
# Pydantic Models — Response
# ======================================================================
class SourceResult(BaseModel):
    """A single retrieved source chunk."""

    chunk_id: str
    text: str
    score: float = Field(description="Raw vector similarity score")
    reranker_score: float = Field(
        description="Combined score after hybrid reranking"
    )
    source: str = Field(description="Retrieval source(s): vector, bm25, page_index, or combinations like vector+bm25+page_index")
    source_url: str
    section_heading: str


class QueryResponse(BaseModel):
    """Response from /query endpoints."""

    answer: str
    sources: list[SourceResult]
    query: str
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float
    model: str
    cache_hit: bool
    cache_hit_type: str
    search_schema: dict = Field(
        description="Azure AI Search-style JSON response schema"
    )


class IngestionDetail(BaseModel):
    """Per-URL ingestion detail."""

    url: str
    status: str = Field(description="processed | cached | failed")
    chunks: int = 0
    tokens: int = 0
    error: Optional[str] = None


class IngestionResponse(BaseModel):
    """Response from /ingest/status."""

    urls_processed: int
    urls_skipped_cached: int
    urls_failed: int
    urls_removed: int
    removed_urls: list[str]
    total_chunks: int
    total_tokens: int
    duration_seconds: float
    details: list[IngestionDetail]
    status: str = Field(description="completed | in_progress | no_data")


class CacheStats(BaseModel):
    """Retrieval cache statistics."""

    total_entries: int
    total_hits: int
    similarity_threshold: float
    ttl_seconds: float


class SystemStatus(BaseModel):
    """Response from /status."""

    vector_store_chunks: int
    bm25_index_chunks: int
    page_index_documents: int
    indexed_urls: list[str]
    embedding_model: str
    embedding_dimension: int
    chunk_size: int
    chunk_overlap: int
    distance_metric: str
    llm_model: str
    api_key_set: bool
    cache_stats: CacheStats
    uptime_seconds: float


class HealthResponse(BaseModel):
    """Response from /health."""

    status: str = "ok"
    timestamp: str
    vector_store_chunks: int
    uptime_seconds: float


class ClearResponse(BaseModel):
    """Response from /clear."""

    cleared: list[str]
    message: str


# ======================================================================
# Application State — Singleton holder
# ======================================================================
class AppState:
    """Holds the singleton RAG components initialised at startup."""

    def __init__(self) -> None:
        self.config: Optional[RAGConfig] = None
        self.pipeline: Optional[IngestionPipeline] = None
        self.engine: Optional[RAGEngine] = None
        self.started_at: Optional[datetime] = None
        # Track last ingestion
        self.last_ingestion_stats: Optional[IngestionStats] = None
        self.ingestion_in_progress: bool = False
        self._lock: asyncio.Lock = asyncio.Lock()

    def rebuild_engine(self) -> None:
        """Rebuild the RAG engine (e.g. after ingestion adds new page trees)."""
        page_trees = self.pipeline.load_page_trees()
        self.engine = RAGEngine(
            config=self.config,
            embedder=self.pipeline.embedder,
            vector_store=self.pipeline.vector_store,
            page_trees=page_trees,
            bm25_index=self.pipeline.bm25_index,
        )


state = AppState()


# ======================================================================
# Lifespan — Startup / Shutdown
# ======================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise heavy components once at startup; clean up on shutdown."""
    # ── Startup ──────────────────────────────────────────────────────
    logger.info("Initialising RAG system...")
    state.config = RAGConfig.load()
    state.pipeline = IngestionPipeline(state.config)
    state.rebuild_engine()
    state.started_at = datetime.now(timezone.utc)
    logger.info(
        "RAG system ready: %d chunks, %d page trees",
        state.pipeline.vector_store.count,
        len(state.pipeline.load_page_trees()),
    )

    yield  # ── App is running ────────────────────────────────────────

    # ── Shutdown ─────────────────────────────────────────────────────
    logger.info("Shutting down RAG system...")
    if state.pipeline:
        state.pipeline.close()


# ======================================================================
# FastAPI App
# ======================================================================
app = FastAPI(
    title="RAG System API",
    description=(
        "Production RAG System with PageIndex, Vector Search, "
        "Retrieval Caching & Azure AI Search-style JSON output."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ======================================================================
# Global exception handler
# ======================================================================
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


# ======================================================================
# GET /health
# ======================================================================
@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health():
    """Simple health check."""
    now = datetime.now(timezone.utc)
    uptime = (now - state.started_at).total_seconds() if state.started_at else 0
    return HealthResponse(
        status="ok",
        timestamp=now.isoformat(),
        vector_store_chunks=state.pipeline.vector_store.count,
        uptime_seconds=round(uptime, 1),
    )


# ======================================================================
# GET /status
# ======================================================================
@app.get("/status", response_model=SystemStatus, tags=["system"])
async def system_status():
    """Comprehensive system status — mirrors the CLI ``status`` command."""
    now = datetime.now(timezone.utc)
    uptime = (now - state.started_at).total_seconds() if state.started_at else 0

    rc_stats = state.engine.retrieval_cache.stats()
    indexed_urls = sorted(state.pipeline.vector_store.get_indexed_urls())

    return SystemStatus(
        vector_store_chunks=state.pipeline.vector_store.count,
        bm25_index_chunks=state.pipeline.bm25_index.count,
        page_index_documents=len(state.pipeline.load_page_trees()),
        indexed_urls=indexed_urls,
        embedding_model=state.config.embedding.model_name,
        embedding_dimension=state.config.embedding.dimension,
        chunk_size=state.config.chunking.chunk_size,
        chunk_overlap=state.config.chunking.chunk_overlap,
        distance_metric=state.config.vector_store.distance_metric,
        llm_model=state.config.page_index.llm_model,
        api_key_set=bool(state.config.anthropic_api_key),
        cache_stats=CacheStats(
            total_entries=rc_stats["total_entries"],
            total_hits=rc_stats["total_hits"],
            similarity_threshold=rc_stats["similarity_threshold"],
            ttl_seconds=rc_stats["ttl_seconds"],
        ),
        uptime_seconds=round(uptime, 1),
    )


# ======================================================================
# Query helpers + endpoints
# ======================================================================
async def _execute_query(
    question: str,
    top_k: int = 8,
    url_filter: Optional[str] = None,
    use_page_index: bool = True,
    use_bm25: bool = True,
    generate_answer: bool = True,
    use_cache: bool = True,
) -> QueryResponse:
    """Shared query logic for both GET and POST endpoints."""
    if state.engine is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialised")

    if state.pipeline.vector_store.count == 0:
        raise HTTPException(
            status_code=404,
            detail="Vector store is empty. Run ingestion first via POST /ingest",
        )

    # Run synchronous engine.query in a thread to keep the event loop free
    response = await asyncio.to_thread(
        state.engine.query,
        question=question,
        top_k=top_k,
        url_filter=url_filter,
        use_page_index=use_page_index,
        use_bm25=use_bm25,
        generate_answer=generate_answer,
        use_cache=use_cache,
    )

    # Map RAGResponse -> QueryResponse
    sources = [
        SourceResult(
            chunk_id=s.chunk_id,
            text=s.text,
            score=round(s.score, 4),
            reranker_score=round(s.reranker_score, 4),
            source=s.source,
            source_url=s.metadata.get("source_url", ""),
            section_heading=s.metadata.get("section_heading", ""),
        )
        for s in response.sources
    ]

    return QueryResponse(
        answer=response.answer,
        sources=sources,
        query=response.query,
        retrieval_time_ms=round(response.retrieval_time_ms, 1),
        generation_time_ms=round(response.generation_time_ms, 1),
        total_time_ms=round(response.total_time_ms, 1),
        model=response.model,
        cache_hit=response.cache_hit,
        cache_hit_type=response.cache_hit_type,
        search_schema=response.to_search_schema(),
    )


@app.post("/query", response_model=QueryResponse, tags=["query"])
async def query_post(request: QueryRequest):
    """Full RAG query via POST with JSON body.

    Returns the generated answer, ranked sources, timing, and
    the Azure AI Search-style ``search_schema`` for integration.
    """
    return await _execute_query(
        question=request.question,
        top_k=request.top_k,
        url_filter=request.url_filter,
        use_page_index=request.use_page_index,
        use_bm25=request.use_bm25,
        generate_answer=request.generate_answer,
        use_cache=request.use_cache,
    )


@app.get("/query", response_model=QueryResponse, tags=["query"])
async def query_get(
    question: str = Query(
        ..., min_length=1, max_length=2000, description="The question to ask"
    ),
    top_k: int = Query(default=8, ge=1, le=50),
    url_filter: Optional[str] = Query(default=None),
    use_page_index: bool = Query(default=True),
    use_bm25: bool = Query(default=True),
    generate_answer: bool = Query(default=True),
    use_cache: bool = Query(default=True),
):
    """Full RAG query via GET with query parameters (simpler for curl/browser)."""
    return await _execute_query(
        question=question,
        top_k=top_k,
        url_filter=url_filter,
        use_page_index=use_page_index,
        use_bm25=use_bm25,
        generate_answer=generate_answer,
        use_cache=use_cache,
    )


# ======================================================================
# Ingestion endpoints
# ======================================================================
async def _run_ingestion(urls: list[str], force: bool) -> None:
    """Run ingestion in the background, updating state when done."""
    async with state._lock:
        state.ingestion_in_progress = True
        try:
            stats = await asyncio.to_thread(state.pipeline.ingest, urls, force)
            state.last_ingestion_stats = stats
            # Rebuild engine to pick up new/changed page trees
            state.rebuild_engine()
            logger.info(
                "Ingestion complete: %d processed, %d cached, %d failed",
                stats.urls_processed,
                stats.urls_skipped_cached,
                stats.urls_failed,
            )
        except Exception as e:
            logger.exception("Ingestion failed: %s", e)
        finally:
            state.ingestion_in_progress = False


@app.post("/ingest", status_code=202, tags=["ingestion"])
async def ingest(request: IngestRequest):
    """Trigger ingestion (runs in background).

    Returns **202 Accepted** immediately. Poll ``GET /ingest/status`` for results.
    """
    if state.ingestion_in_progress:
        raise HTTPException(status_code=409, detail="Ingestion already in progress")

    target_urls = request.urls if request.urls else DEFAULT_URLS

    # Fire and forget — ingestion runs in background
    asyncio.create_task(_run_ingestion(target_urls, request.force))

    return {
        "status": "accepted",
        "message": f"Ingestion started for {len(target_urls)} URLs",
        "urls": target_urls,
        "force": request.force,
    }


@app.get("/ingest/status", response_model=IngestionResponse, tags=["ingestion"])
async def ingest_status():
    """Returns the result of the last ingestion run (or in-progress indicator)."""
    if state.ingestion_in_progress:
        return IngestionResponse(
            urls_processed=0,
            urls_skipped_cached=0,
            urls_failed=0,
            urls_removed=0,
            removed_urls=[],
            total_chunks=0,
            total_tokens=0,
            duration_seconds=0,
            details=[],
            status="in_progress",
        )

    stats = state.last_ingestion_stats
    if stats is None:
        return IngestionResponse(
            urls_processed=0,
            urls_skipped_cached=0,
            urls_failed=0,
            urls_removed=0,
            removed_urls=[],
            total_chunks=0,
            total_tokens=0,
            duration_seconds=0,
            details=[],
            status="no_data",
        )

    details = [
        IngestionDetail(
            url=d["url"],
            status=d["status"],
            chunks=d.get("chunks", 0),
            tokens=d.get("tokens", 0),
            error=d.get("error"),
        )
        for d in stats.details
    ]

    return IngestionResponse(
        urls_processed=stats.urls_processed,
        urls_skipped_cached=stats.urls_skipped_cached,
        urls_failed=stats.urls_failed,
        urls_removed=stats.urls_removed,
        removed_urls=stats.removed_urls,
        total_chunks=stats.total_chunks,
        total_tokens=stats.total_tokens,
        duration_seconds=round(stats.duration_seconds, 2),
        details=details,
        status="completed",
    )


# ======================================================================
# POST /clear
# ======================================================================
@app.post("/clear", response_model=ClearResponse, tags=["system"])
async def clear(request: ClearRequest):
    """Clear vector store, content cache, and/or query cache."""
    cleared: list[str] = []
    targets = request.targets
    is_all = ClearTarget.all in targets

    if is_all or ClearTarget.vectors in targets:
        await asyncio.to_thread(state.pipeline.vector_store.delete_all)
        cleared.append("vectors")

    if is_all or ClearTarget.cache in targets:
        cm = CacheManager(state.config.cache)
        cm.clear()
        cm.close()
        cleared.append("content_cache")

    if is_all or ClearTarget.query_cache in targets:
        state.engine.retrieval_cache.invalidate_all()
        cleared.append("query_cache")

    # Rebuild engine after clearing vectors (page trees may have changed)
    if "vectors" in cleared:
        state.rebuild_engine()

    return ClearResponse(
        cleared=cleared,
        message=f"Cleared: {', '.join(cleared)}",
    )


# ======================================================================
# Entry point
# ======================================================================
if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Quiet noisy libraries
    for lib in ("httpx", "chromadb", "sentence_transformers", "urllib3", "trafilatura"):
        logging.getLogger(lib).setLevel(logging.WARNING)

    uvicorn.run(app, host="0.0.0.0", port=8000)
