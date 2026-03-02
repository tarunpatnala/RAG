"""Smart text chunking with section-context preservation.

Implements recursive character splitting with:
  - Token-aware sizing (via tiktoken)
  - Configurable overlap
  - Section heading context prepended to each chunk
  - Rich metadata per chunk (source URL, section path, position)
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Optional

import tiktoken

from rag_system.config import ChunkingConfig
from rag_system.core.document_processor import StructuredDocument

logger = logging.getLogger(__name__)

_ENC = tiktoken.get_encoding("cl100k_base")


def _token_len(text: str) -> int:
    return len(_ENC.encode(text, disallowed_special=()))


@dataclass
class Chunk:
    """A single chunk of text ready for embedding."""
    chunk_id: str
    text: str
    token_count: int
    metadata: dict = field(default_factory=dict)


class SemanticChunker:
    """Splits a ``StructuredDocument`` into contextualised chunks."""

    def __init__(self, config: ChunkingConfig) -> None:
        self._cfg = config

    def chunk_document(self, doc: StructuredDocument) -> list[Chunk]:
        """Produce a list of ``Chunk`` objects from *doc*."""
        self._global_counter = 0  # reset per document for unique IDs
        chunks: list[Chunk] = []

        # Strategy: chunk each section individually to preserve heading context,
        # then chunk any remaining main_content not covered by sections.
        if doc.sections:
            chunks.extend(self._chunk_sections(doc))
        else:
            chunks.extend(self._chunk_plain(doc.plain_text, doc))

        # Also chunk tables as standalone chunks
        for i, table_md in enumerate(doc.tables_markdown):
            tlen = _token_len(table_md)
            if tlen < self._cfg.min_chunk_size:
                continue
            cid = self._make_id(doc.url, f"table-{i}", self._next_counter())
            chunks.append(Chunk(
                chunk_id=cid,
                text=f"[Table from {doc.title}]\n\n{table_md}",
                token_count=tlen,
                metadata={
                    "source_url": doc.url,
                    "title": doc.title,
                    "chunk_type": "table",
                    "table_index": i,
                    "content_hash": doc.content_hash,
                },
            ))

        # Deduplicate any remaining ID collisions (safety net)
        seen_ids: set[str] = set()
        deduped: list[Chunk] = []
        for c in chunks:
            if c.chunk_id in seen_ids:
                c.chunk_id = c.chunk_id + hashlib.md5(c.text.encode()).hexdigest()[:8]
            seen_ids.add(c.chunk_id)
            deduped.append(c)
        chunks = deduped

        logger.info(
            "Chunked %s into %d chunks (avg %d tokens)",
            doc.url,
            len(chunks),
            sum(c.token_count for c in chunks) // max(len(chunks), 1),
        )
        return chunks

    # ------------------------------------------------------------------
    # Section-aware chunking
    # ------------------------------------------------------------------
    def _chunk_sections(self, doc: StructuredDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        for sec in doc.sections:
            if not sec["content"].strip():
                continue
            context_prefix = ""
            if self._cfg.include_section_context:
                context_prefix = f"[{doc.title}] [{sec['path']}]\n\n"

            section_text = context_prefix + sec["content"]
            section_chunks = self._split_text(section_text)

            for idx, text in enumerate(section_chunks):
                cid = self._make_id(doc.url, sec["heading"], self._next_counter())
                chunks.append(Chunk(
                    chunk_id=cid,
                    text=text,
                    token_count=_token_len(text),
                    metadata={
                        "source_url": doc.url,
                        "title": doc.title,
                        "section_heading": sec["heading"],
                        "section_path": sec["path"],
                        "section_level": sec["level"],
                        "chunk_index": idx,
                        "chunk_type": "section",
                        "content_hash": doc.content_hash,
                    },
                ))

        # If sections didn't cover all content, chunk the main plaintext too
        total_section_text = " ".join(s["content"] for s in doc.sections if s["content"])
        if doc.plain_text and len(doc.plain_text) > len(total_section_text) * 1.3:
            chunks.extend(self._chunk_plain(doc.plain_text, doc))

        return chunks

    def _chunk_plain(self, text: str, doc: StructuredDocument) -> list[Chunk]:
        """Chunk raw plaintext without section context."""
        chunks: list[Chunk] = []
        parts = self._split_text(text)
        for idx, part in enumerate(parts):
            cid = self._make_id(doc.url, "plain", self._next_counter())
            chunks.append(Chunk(
                chunk_id=cid,
                text=f"[{doc.title}]\n\n{part}",
                token_count=_token_len(part),
                metadata={
                    "source_url": doc.url,
                    "title": doc.title,
                    "chunk_index": idx,
                    "chunk_type": "plain",
                    "content_hash": doc.content_hash,
                },
            ))
        return chunks

    # ------------------------------------------------------------------
    # Recursive text splitting
    # ------------------------------------------------------------------
    def _split_text(self, text: str) -> list[str]:
        """Recursively split *text* into chunks respecting token limits."""
        if _token_len(text) <= self._cfg.chunk_size:
            if _token_len(text) >= self._cfg.min_chunk_size:
                return [text.strip()]
            return []

        return self._recursive_split(text, self._cfg.separators)

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        if not separators:
            return self._hard_split(text)

        sep = separators[0]
        remaining_seps = separators[1:]
        parts = text.split(sep)

        result: list[str] = []
        current = ""
        for part in parts:
            candidate = (current + sep + part) if current else part
            if _token_len(candidate) <= self._cfg.chunk_size:
                current = candidate
            else:
                if current:
                    if _token_len(current) >= self._cfg.min_chunk_size:
                        result.append(current.strip())
                    elif result:
                        # Merge short tail into previous chunk
                        result[-1] = result[-1] + sep + current.strip()

                # If this single part exceeds chunk_size, split deeper
                if _token_len(part) > self._cfg.chunk_size:
                    result.extend(self._recursive_split(part, remaining_seps))
                    current = ""
                else:
                    current = part

        if current.strip():
            tlen = _token_len(current)
            if tlen >= self._cfg.min_chunk_size:
                result.append(current.strip())
            elif result:
                result[-1] = result[-1] + sep + current.strip()

        # Apply overlap
        if self._cfg.chunk_overlap > 0 and len(result) > 1:
            result = self._add_overlap(result)

        return result

    def _hard_split(self, text: str) -> list[str]:
        """Token-level hard split as a last resort."""
        tokens = _ENC.encode(text, disallowed_special=())
        chunks: list[str] = []
        step = self._cfg.chunk_size - self._cfg.chunk_overlap
        for i in range(0, len(tokens), step):
            chunk_tokens = tokens[i : i + self._cfg.chunk_size]
            decoded = _ENC.decode(chunk_tokens).strip()
            if _token_len(decoded) >= self._cfg.min_chunk_size:
                chunks.append(decoded)
        return chunks

    def _add_overlap(self, chunks: list[str]) -> list[str]:
        """Add trailing overlap from the previous chunk."""
        overlap_tokens = self._cfg.chunk_overlap
        result = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tokens = _ENC.encode(chunks[i - 1], disallowed_special=())
            overlap_text = _ENC.decode(prev_tokens[-overlap_tokens:]).strip()
            result.append(overlap_text + " " + chunks[i])
        return result

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _next_counter(self) -> int:
        """Return a monotonically increasing counter for unique chunk IDs."""
        val = getattr(self, "_global_counter", 0)
        self._global_counter = val + 1
        return val

    @staticmethod
    def _make_id(url: str, section: str, index: int) -> str:
        raw = f"{url}::{section}::{index}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
