"""RAG Query Engine — hybrid retrieval combining vector search + BM25 + PageIndex.

Pipeline:
  1. Check retrieval cache (exact match, then semantic similarity)
  2. On cache miss:
     a. Embed the user query
     b. Vector search (semantic similarity via ChromaDB)
     c. BM25 keyword search (term-frequency scoring)
     d. PageIndex reasoning (structured tree navigation)
     e. Three-way merge & re-rank results
     f. Send to LLM with context
  3. Store result in retrieval cache
  4. Return answer with sources and cache-hit metadata
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import anthropic

from rag_system.config import RAGConfig
from rag_system.core.bm25_index import BM25Index
from rag_system.core.embeddings import EmbeddingGenerator
from rag_system.core.page_index import PageIndexReasoner, PageTree
from rag_system.core.retrieval_cache import RetrievalCache
from rag_system.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """A single retrieved chunk with scoring metadata."""
    chunk_id: str
    text: str
    score: float
    source: str  # "vector" | "page_index" | "both"
    metadata: dict = field(default_factory=dict)
    reranker_score: float = 0.0  # score after PageIndex reranking


@dataclass
class RAGResponse:
    """Complete RAG response with provenance."""
    answer: str
    sources: list[RetrievalResult]
    query: str
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float
    model: str
    cache_hit: bool = False          # True if served from retrieval cache
    cache_hit_type: str = ""         # "exact" | "semantic" | ""

    def to_search_schema(self) -> dict:
        """Serialise response in Azure AI Search-style JSON schema.

        Format:
        {
          "@data.context": "<query>",
          "@data.count": <int>,
          "@search.answers": [{ "text": "...", "score": ... }],
          "value": [
            {
              "@search.score": <float>,
              "@search.rerankerScore": <float>,
              "Content": "...",
              "sourcefile": "...",
              "doc_url": "..."
            }
          ]
        }
        """
        answers: list[dict] = []
        if self.answer and not self.answer.startswith("Retrieved context (no LLM"):
            answers.append({
                "text": self.answer,
                "score": self.sources[0].score if self.sources else 0.0,
            })

        value: list[dict] = []
        for src in self.sources:
            doc_url = src.metadata.get("source_url", "")
            # Derive a sourcefile label from the URL path
            sourcefile = doc_url.rstrip("/").rsplit("/", 1)[-1] if doc_url else ""
            value.append({
                "@search.score": round(src.score, 4),
                "@search.rerankerScore": round(src.reranker_score, 4),
                "Content": src.text,
                "sourcefile": sourcefile,
                "doc_url": doc_url,
            })

        return {
            "@data.context": self.query,
            "@data.count": len(self.sources),
            "@search.answers": answers,
            "value": value,
        }


class RAGEngine:
    """Hybrid retrieval-augmented generation engine."""

    def __init__(
        self,
        config: RAGConfig,
        embedder: EmbeddingGenerator,
        vector_store: VectorStore,
        page_trees: list[PageTree],
        bm25_index: Optional[BM25Index] = None,
    ) -> None:
        self._cfg = config
        self._embedder = embedder
        self._store = vector_store
        self._trees = page_trees
        self._bm25 = bm25_index

        self._reasoner: Optional[PageIndexReasoner] = None
        if page_trees:
            self._reasoner = PageIndexReasoner(
                config.page_index, api_key=config.anthropic_api_key
            )

        self._llm: Optional[anthropic.Anthropic] = None
        if config.anthropic_api_key:
            self._llm = anthropic.Anthropic(api_key=config.anthropic_api_key)

        # ── Retrieval cache ───────────────────────────────────────────
        self._retrieval_cache = RetrievalCache(
            db_path=config.retrieval_cache.db_path,
            similarity_threshold=config.retrieval_cache.similarity_threshold,
            ttl_seconds=config.retrieval_cache.ttl_seconds,
            max_entries=config.retrieval_cache.max_entries,
        )

    def query(
        self,
        question: str,
        top_k: int = 10,
        url_filter: Optional[str] = None,
        use_page_index: bool = True,
        use_bm25: bool = True,
        generate_answer: bool = True,
        use_cache: bool = True,
    ) -> RAGResponse:
        """Run the full RAG pipeline for *question*."""
        t_start = time.perf_counter()

        # ── Content fingerprint (for cache invalidation) ──────────────
        content_fp = self._get_content_fingerprint()

        # ── Embed the query (needed for both cache lookup and retrieval)
        query_embedding = self._embedder.embed_query(question)

        # ── 0. Check retrieval cache ──────────────────────────────────
        if use_cache:
            cached = self._retrieval_cache.lookup(
                query=question,
                query_embedding=query_embedding,
                content_fingerprint=content_fp,
                top_k=top_k,
                url_filter=url_filter or "",
            )
            if cached:
                t_done = time.perf_counter()
                sources = self._deserialise_sources(cached.sources_json)
                hit_type = (
                    "exact"
                    if cached.query.strip().lower() == question.strip().lower()
                    else "semantic"
                )
                return RAGResponse(
                    answer=cached.answer,
                    sources=sources,
                    query=question,
                    retrieval_time_ms=(t_done - t_start) * 1000,
                    generation_time_ms=0.0,
                    total_time_ms=(t_done - t_start) * 1000,
                    model=self._cfg.page_index.llm_model if self._llm else "none",
                    cache_hit=True,
                    cache_hit_type=hit_type,
                )

        # ── 1. Retrieve ──────────────────────────────────────────────
        retrieval_results = self._retrieve(
            question, query_embedding=query_embedding,
            top_k=top_k, url_filter=url_filter,
            use_page_index=use_page_index, use_bm25=use_bm25,
        )
        t_retrieval = time.perf_counter()

        # ── 2. Generate ──────────────────────────────────────────────
        if generate_answer and self._llm:
            answer = self._generate(question, retrieval_results)
        else:
            answer = self._format_context_only(retrieval_results)
        t_done = time.perf_counter()

        response = RAGResponse(
            answer=answer,
            sources=retrieval_results,
            query=question,
            retrieval_time_ms=(t_retrieval - t_start) * 1000,
            generation_time_ms=(t_done - t_retrieval) * 1000,
            total_time_ms=(t_done - t_start) * 1000,
            model=self._cfg.page_index.llm_model if self._llm else "none",
        )

        # ── 3. Store in retrieval cache ──────────────────────────────
        if use_cache:
            self._retrieval_cache.store(
                query=question,
                query_embedding=query_embedding,
                answer=answer,
                sources_json=self._serialise_sources(retrieval_results),
                content_fingerprint=content_fp,
                top_k=top_k,
                url_filter=url_filter or "",
            )

        return response

    # ------------------------------------------------------------------
    # Retrieval: merge vector + BM25 + PageIndex
    # ------------------------------------------------------------------
    def _retrieve(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int,
        url_filter: Optional[str],
        use_page_index: bool,
        use_bm25: bool,
    ) -> list[RetrievalResult]:
        # ── Vector search ─────────────────────────────────────────────
        if url_filter:
            vector_hits = self._store.search_by_url(query_embedding, url_filter, top_k=top_k)
        else:
            vector_hits = self._store.search(query_embedding, top_k=top_k)

        vector_map: dict[str, RetrievalResult] = {}
        for hit in vector_hits:
            # Convert distance to similarity score (cosine: 1 - distance)
            score = max(0.0, 1.0 - hit["distance"])
            vector_map[hit["id"]] = RetrievalResult(
                chunk_id=hit["id"],
                text=hit["document"],
                score=score,
                source="vector",
                metadata=hit["metadata"],
            )

        # ── BM25 keyword search ───────────────────────────────────────
        bm25_map: dict[str, float] = {}
        if use_bm25 and self._bm25 and self._bm25.is_ready:
            bm25_hits = self._bm25.search(query, top_k=top_k, url_filter=url_filter)
            if bm25_hits:
                # Max-normalise BM25 scores to [0, 1]
                max_score = max(h["score"] for h in bm25_hits)
                for hit in bm25_hits:
                    norm = hit["score"] / max_score if max_score > 0 else 0.0
                    bm25_map[hit["id"]] = norm

        # ── PageIndex reasoning ───────────────────────────────────────
        pi_map: dict[str, float] = {}
        if use_page_index and self._reasoner and self._trees:
            pi_chunk_ids = self._reasoner.find_relevant_chunks(
                query, self._trees, max_chunks=top_k
            )
            # Assign decreasing scores based on rank from the reasoner
            for rank, cid in enumerate(pi_chunk_ids):
                pi_map[cid] = 1.0 - (rank * 0.05)

        # ── Three-way merge & re-rank ─────────────────────────────────
        merged = self._merge_results(vector_map, bm25_map, pi_map, top_k)
        return merged

    def _merge_results(
        self,
        vector_map: dict[str, RetrievalResult],
        bm25_scores: dict[str, float],
        pi_scores: dict[str, float],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Three-way weighted merge of vector, BM25, and PageIndex results.

        Base weights: 0.40 vector + 0.35 BM25 + 0.25 PageIndex.
        Weights are dynamically re-normalised when a retriever is disabled
        (empty score map → weight 0, remaining weights sum to 1.0).
        """
        all_ids = set(vector_map.keys()) | set(bm25_scores.keys()) | set(pi_scores.keys())
        scored: list[RetrievalResult] = []

        # ── Dynamic weight calculation ────────────────────────────────
        w_vec = 0.40
        w_bm25 = 0.35 if bm25_scores else 0.0
        w_pi = 0.25 if pi_scores else 0.0
        total_w = w_vec + w_bm25 + w_pi
        if total_w > 0:
            w_vec /= total_w
            w_bm25 /= total_w
            w_pi /= total_w

        # ── Fetch texts for chunks only found via BM25 or PageIndex ───
        need_fetch = [cid for cid in all_ids if cid not in vector_map]
        fetch_docs: dict[str, dict] = {}
        if need_fetch:
            for item in self._store.search_by_ids(need_fetch):
                fetch_docs[item["id"]] = item

        for cid in all_ids:
            v_score = vector_map[cid].score if cid in vector_map else 0.0
            b_score = bm25_scores.get(cid, 0.0)
            p_score = pi_scores.get(cid, 0.0)

            combined = w_vec * v_score + w_bm25 * b_score + w_pi * p_score

            # ── Source label ("+"-joined) ─────────────────────────────
            sources: list[str] = []
            if cid in vector_map:
                sources.append("vector")
            if cid in bm25_scores:
                sources.append("bm25")
            if cid in pi_scores:
                sources.append("page_index")
            source_label = "+".join(sources)

            if cid in vector_map:
                result = vector_map[cid]
                result.reranker_score = combined
                result.score = v_score
                result.source = source_label
            elif cid in fetch_docs:
                result = RetrievalResult(
                    chunk_id=cid,
                    text=fetch_docs[cid]["document"],
                    score=0.0,
                    reranker_score=combined,
                    source=source_label,
                    metadata=fetch_docs[cid].get("metadata", {}),
                )
            else:
                continue

            scored.append(result)

        scored.sort(key=lambda r: r.reranker_score, reverse=True)
        return scored[:top_k]

    # ------------------------------------------------------------------
    # LLM answer generation
    # ------------------------------------------------------------------
    def _generate(self, question: str, results: list[RetrievalResult]) -> str:
        """Generate an answer using Claude with retrieved context."""
        context_parts: list[str] = []
        for i, r in enumerate(results, 1):
            source_url = r.metadata.get("source_url", "unknown")
            section = r.metadata.get("section_heading", "")
            header = f"[Source {i}: {source_url}"
            if section:
                header += f" > {section}"
            header += f" | relevance: {r.reranker_score:.2f}]"
            context_parts.append(f"{header}\n{r.text}")

        context = "\n\n---\n\n".join(context_parts)

        system_prompt = """You are a knowledgeable financial products assistant specialising in Commonwealth Bank (CommBank) products. Answer the user's question based ONLY on the provided context.

Rules:
1. Only use information present in the context below
2. If the context doesn't contain enough information, say so clearly
3. Cite the source numbers [Source N] when referencing specific information
4. Provide structured, clear answers
5. For product comparisons, use tables when appropriate
6. Include specific rates, fees, or conditions when available in the context"""

        user_prompt = f"""Context:
{context}

Question: {question}

Provide a comprehensive answer based on the context above."""

        try:
            response = self._llm.messages.create(
                model=self._cfg.page_index.llm_model,
                max_tokens=2048,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return response.content[0].text
        except Exception as e:
            logger.error("LLM generation failed: %s", e)
            return self._format_context_only(results)

    @staticmethod
    def _format_context_only(results: list[RetrievalResult]) -> str:
        """Fallback: return raw context without LLM generation."""
        parts = ["Retrieved context (no LLM generation):\n"]
        for i, r in enumerate(results, 1):
            url = r.metadata.get("source_url", "")
            parts.append(f"--- Result {i} (score: {r.reranker_score:.3f}, source: {r.source}) ---")
            parts.append(f"URL: {url}")
            parts.append(r.text)
            parts.append("")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Content fingerprinting & serialisation helpers
    # ------------------------------------------------------------------
    def _get_content_fingerprint(self) -> str:
        """Compute a fingerprint of all indexed content for cache invalidation."""
        all_hashes: list[str] = []
        total = self._store.count
        if total > 0:
            result = self._store._collection.get(
                limit=total,
                include=["metadatas"],
            )
            if result["metadatas"]:
                for meta in result["metadatas"]:
                    if meta and "content_hash" in meta:
                        h = meta["content_hash"]
                        if h not in all_hashes:
                            all_hashes.append(h)
        return RetrievalCache.compute_content_fingerprint(all_hashes)

    @staticmethod
    def _serialise_sources(results: list[RetrievalResult]) -> str:
        """Serialise RetrievalResult list to JSON for cache storage."""
        return json.dumps([
            {
                "chunk_id": r.chunk_id,
                "text": r.text,
                "score": r.score,
                "reranker_score": r.reranker_score,
                "source": r.source,
                "metadata": r.metadata,
            }
            for r in results
        ])

    @staticmethod
    def _deserialise_sources(sources_json: str) -> list[RetrievalResult]:
        """Deserialise cached sources back to RetrievalResult objects."""
        data = json.loads(sources_json)
        return [
            RetrievalResult(
                chunk_id=d["chunk_id"],
                text=d["text"],
                score=d["score"],
                reranker_score=d.get("reranker_score", d["score"]),
                source=d["source"],
                metadata=d.get("metadata", {}),
            )
            for d in data
        ]

    @property
    def retrieval_cache(self) -> RetrievalCache:
        """Expose the retrieval cache for stats / management."""
        return self._retrieval_cache
