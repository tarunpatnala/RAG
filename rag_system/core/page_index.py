"""PageIndex — hierarchical tree index for reasoning-based RAG.

Inspired by VectifyAI's PageIndex concept: instead of relying solely
on vector similarity, build a structured Table-of-Contents tree for
each document and use LLM reasoning to navigate the tree at query time.

The index is complementary to vector search — it enables the RAG engine
to perform *structured retrieval* when a query maps naturally to a
known section, and *vector retrieval* for open-ended semantic queries.

Tree structure per document:
{
  "url": "...",
  "title": "...",
  "summary": "...",
  "nodes": [
    {
      "node_id": "0001",
      "name": "Section heading",
      "summary": "Brief description of this section's content",
      "content_preview": "First ~200 chars",
      "chunk_ids": ["abc123", ...],
      "nodes": [ ... ]        # child sections
    }
  ]
}
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import anthropic

from rag_system.config import PageIndexConfig
from rag_system.core.chunker import Chunk
from rag_system.core.document_processor import StructuredDocument

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Tree node
# ------------------------------------------------------------------

class PageNode:
    """A node in the hierarchical page index tree."""

    def __init__(
        self,
        node_id: str,
        name: str,
        summary: str,
        content_preview: str = "",
        level: int = 0,
        chunk_ids: Optional[list[str]] = None,
        children: Optional[list["PageNode"]] = None,
    ) -> None:
        self.node_id = node_id
        self.name = name
        self.summary = summary
        self.content_preview = content_preview
        self.level = level
        self.chunk_ids = chunk_ids or []
        self.children = children or []

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "name": self.name,
            "summary": self.summary,
            "content_preview": self.content_preview,
            "level": self.level,
            "chunk_ids": self.chunk_ids,
            "nodes": [c.to_dict() for c in self.children],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PageNode":
        return cls(
            node_id=d["node_id"],
            name=d["name"],
            summary=d["summary"],
            content_preview=d.get("content_preview", ""),
            level=d.get("level", 0),
            chunk_ids=d.get("chunk_ids", []),
            children=[cls.from_dict(c) for c in d.get("nodes", [])],
        )


# ------------------------------------------------------------------
# Page tree (one per URL)
# ------------------------------------------------------------------

class PageTree:
    """Full hierarchical index for a single page / document."""

    def __init__(
        self,
        url: str,
        title: str,
        summary: str,
        nodes: Optional[list[PageNode]] = None,
    ) -> None:
        self.url = url
        self.title = title
        self.summary = summary
        self.nodes = nodes or []

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "title": self.title,
            "summary": self.summary,
            "nodes": [n.to_dict() for n in self.nodes],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PageTree":
        return cls(
            url=d["url"],
            title=d["title"],
            summary=d["summary"],
            nodes=[PageNode.from_dict(n) for n in d.get("nodes", [])],
        )

    def to_toc_string(self, indent: int = 0) -> str:
        """Render the tree as a readable table-of-contents string."""
        lines: list[str] = []
        lines.append(f"Document: {self.title}")
        lines.append(f"URL: {self.url}")
        lines.append(f"Summary: {self.summary}")
        lines.append("")
        for node in self.nodes:
            lines.extend(self._node_toc(node, indent))
        return "\n".join(lines)

    def _node_toc(self, node: PageNode, indent: int) -> list[str]:
        prefix = "  " * indent
        lines = [f"{prefix}- [{node.node_id}] {node.name}: {node.summary}"]
        for child in node.children:
            lines.extend(self._node_toc(child, indent + 1))
        return lines

    def find_node(self, node_id: str) -> Optional[PageNode]:
        """Find a node by its ID in the tree."""
        return self._search(self.nodes, node_id)

    def _search(self, nodes: list[PageNode], target: str) -> Optional[PageNode]:
        for n in nodes:
            if n.node_id == target:
                return n
            found = self._search(n.children, target)
            if found:
                return found
        return None

    def get_all_chunk_ids(self, node_id: Optional[str] = None) -> list[str]:
        """Return chunk IDs from a subtree (or the entire tree)."""
        if node_id:
            node = self.find_node(node_id)
            if not node:
                return []
            return self._collect_chunk_ids(node)
        ids: list[str] = []
        for n in self.nodes:
            ids.extend(self._collect_chunk_ids(n))
        return ids

    def _collect_chunk_ids(self, node: PageNode) -> list[str]:
        ids = list(node.chunk_ids)
        for child in node.children:
            ids.extend(self._collect_chunk_ids(child))
        return ids


# ------------------------------------------------------------------
# PageIndex builder & reasoner
# ------------------------------------------------------------------

class PageIndexBuilder:
    """Builds and persists PageTree indexes.

    Uses an LLM to generate rich summaries for the tree nodes when
    an API key is available; falls back to a heuristic builder otherwise.
    """

    def __init__(self, config: PageIndexConfig, api_key: Optional[str] = None) -> None:
        self._cfg = config
        self._index_dir = Path(config.index_dir)
        self._index_dir.mkdir(parents=True, exist_ok=True)
        self._client: Optional[anthropic.Anthropic] = None
        if api_key:
            self._client = anthropic.Anthropic(api_key=api_key)

    # ------------------------------------------------------------------
    # Build index
    # ------------------------------------------------------------------
    def build(self, doc: StructuredDocument, chunks: list[Chunk]) -> PageTree:
        """Build a PageTree for *doc*, linking to *chunks*."""
        if self._client:
            tree = self._build_with_llm(doc, chunks)
        else:
            tree = self._build_heuristic(doc, chunks)

        self._save(tree)
        return tree

    def load(self, url: str) -> Optional[PageTree]:
        """Load a previously built tree from disk."""
        path = self._tree_path(url)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return PageTree.from_dict(data)

    def load_all(self) -> list[PageTree]:
        """Load all saved page trees."""
        trees: list[PageTree] = []
        for p in self._index_dir.glob("*.json"):
            data = json.loads(p.read_text(encoding="utf-8"))
            trees.append(PageTree.from_dict(data))
        return trees

    # ------------------------------------------------------------------
    # LLM-powered index builder
    # ------------------------------------------------------------------
    def _build_with_llm(self, doc: StructuredDocument, chunks: list[Chunk]) -> PageTree:
        """Use Claude to generate a rich hierarchical index with summaries."""
        # Build a condensed representation for the LLM
        sections_repr = ""
        for sec in doc.sections:
            indent = "  " * (sec["level"] - 1)
            preview = sec["content"][:300] if sec["content"] else ""
            sections_repr += f"{indent}- [H{sec['level']}] {sec['heading']}: {preview}\n"

        chunk_map = "\n".join(
            f"  - chunk_id={c.chunk_id}  section={c.metadata.get('section_heading', 'general')}  "
            f"tokens={c.token_count}"
            for c in chunks
        )

        prompt = f"""Analyze this web page and create a hierarchical tree index (like a detailed table of contents).

PAGE TITLE: {doc.title}
PAGE URL: {doc.url}
PAGE DESCRIPTION: {doc.description}

SECTIONS FOUND:
{sections_repr}

CHUNKS AVAILABLE:
{chunk_map}

Create a JSON object with this structure:
{{
  "summary": "A 2-3 sentence summary of the entire page",
  "nodes": [
    {{
      "node_id": "0001",
      "name": "Section name",
      "summary": "What this section covers and key information",
      "chunk_ids": ["relevant_chunk_ids_here"],
      "level": 1,
      "nodes": [...]
    }}
  ]
}}

Rules:
1. Each node should have a clear, descriptive summary that helps decide if this section is relevant to a query
2. Map chunk_ids to the most relevant section nodes
3. Use hierarchical nesting matching the heading structure
4. node_ids should be sequential: "0001", "0002", etc.
5. Every chunk should appear in at least one node's chunk_ids
6. Include the key topics, products, or facts covered in each section summary

Return ONLY the JSON object, no other text."""

        try:
            response = self._client.messages.create(
                model=self._cfg.llm_model,
                max_tokens=self._cfg.max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )

            raw = response.content[0].text.strip()
            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1]
                if raw.endswith("```"):
                    raw = raw.rsplit("```", 1)[0]
                raw = raw.strip()

            data = json.loads(raw)

            nodes = [PageNode.from_dict(n) for n in data.get("nodes", [])]
            tree = PageTree(
                url=doc.url,
                title=doc.title,
                summary=data.get("summary", doc.description),
                nodes=nodes,
            )
            logger.info("Built LLM-powered PageIndex for %s (%d nodes)", doc.url, len(nodes))
            return tree

        except Exception as e:
            logger.warning("LLM PageIndex failed for %s: %s — falling back to heuristic", doc.url, e)
            return self._build_heuristic(doc, chunks)

    # ------------------------------------------------------------------
    # Heuristic (no-LLM) index builder
    # ------------------------------------------------------------------
    def _build_heuristic(self, doc: StructuredDocument, chunks: list[Chunk]) -> PageTree:
        """Build a tree using heading structure and simple text overlap."""
        # Create a chunk lookup by section heading
        chunk_by_section: dict[str, list[str]] = {}
        for c in chunks:
            sec = c.metadata.get("section_heading", "__plain__")
            chunk_by_section.setdefault(sec, []).append(c.chunk_id)

        # Build nodes from flat sections
        nodes: list[PageNode] = []
        counter = 1
        node_stack: list[PageNode] = []

        for sec in doc.sections:
            node = PageNode(
                node_id=f"{counter:04d}",
                name=sec["heading"],
                summary=sec["content"][:200] if sec["content"] else sec["heading"],
                content_preview=sec["content"][:200] if sec["content"] else "",
                level=sec["level"],
                chunk_ids=chunk_by_section.get(sec["heading"], []),
            )
            counter += 1

            # Nest under correct parent based on level
            while node_stack and node_stack[-1].level >= sec["level"]:
                node_stack.pop()

            if node_stack:
                node_stack[-1].children.append(node)
            else:
                nodes.append(node)

            node_stack.append(node)

        # Assign any unmatched chunks to the root
        assigned = set()
        for n in nodes:
            assigned.update(self._collect_all_ids(n))

        unmatched = [c.chunk_id for c in chunks if c.chunk_id not in assigned]
        if unmatched and nodes:
            nodes[0].chunk_ids.extend(unmatched)
        elif unmatched:
            nodes.append(PageNode(
                node_id=f"{counter:04d}",
                name="General Content",
                summary=doc.description or doc.title,
                chunk_ids=unmatched,
            ))

        tree = PageTree(
            url=doc.url,
            title=doc.title,
            summary=doc.description or f"Content from {doc.title}",
            nodes=nodes,
        )
        logger.info("Built heuristic PageIndex for %s (%d nodes)", doc.url, len(nodes))
        return tree

    def _collect_all_ids(self, node: PageNode) -> set[str]:
        ids = set(node.chunk_ids)
        for child in node.children:
            ids.update(self._collect_all_ids(child))
        return ids

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _save(self, tree: PageTree) -> None:
        path = self._tree_path(tree.url)
        path.write_text(
            json.dumps(tree.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Saved PageIndex: %s", path)

    def delete(self, url: str) -> bool:
        """Delete the PageIndex tree for a specific URL.  Returns True if deleted."""
        path = self._tree_path(url)
        if path.exists():
            path.unlink()
            logger.info("Deleted PageIndex for URL: %s", url)
            return True
        return False

    def _tree_path(self, url: str) -> Path:
        safe = quote(url, safe="")[:180]
        return self._index_dir / f"{safe}.json"


# ------------------------------------------------------------------
# PageIndex Reasoner — uses the tree at query time
# ------------------------------------------------------------------

class PageIndexReasoner:
    """Navigate page trees at query time to find relevant sections.

    Uses LLM reasoning when available to pick the best tree nodes
    for a given query; otherwise uses keyword matching.
    """

    def __init__(self, config: PageIndexConfig, api_key: Optional[str] = None) -> None:
        self._cfg = config
        self._client: Optional[anthropic.Anthropic] = None
        if api_key:
            self._client = anthropic.Anthropic(api_key=api_key)

    def find_relevant_chunks(
        self,
        query: str,
        trees: list[PageTree],
        max_chunks: int = 15,
    ) -> list[str]:
        """Return chunk IDs relevant to *query* by reasoning over the trees."""
        if self._client:
            return self._reason_with_llm(query, trees, max_chunks)
        return self._reason_heuristic(query, trees, max_chunks)

    def _reason_with_llm(
        self, query: str, trees: list[PageTree], max_chunks: int
    ) -> list[str]:
        """Use Claude to reason about which tree nodes are relevant."""
        toc = "\n\n".join(t.to_toc_string() for t in trees)

        prompt = f"""Given the following document indexes (table of contents), identify the most relevant sections for the user's query.

DOCUMENT INDEXES:
{toc}

USER QUERY: {query}

Return a JSON array of objects with the node_ids of the most relevant sections, ordered by relevance.
Format: [{{"url": "...", "node_ids": ["0001", "0003"]}}]

Rules:
1. Select sections that would contain information to answer the query
2. Include parent sections if their children are relevant
3. Be selective — only include truly relevant sections
4. Consider the section summaries to make your decision

Return ONLY the JSON array."""

        try:
            response = self._client.messages.create(
                model=self._cfg.llm_model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )

            raw = response.content[0].text.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1]
                if raw.endswith("```"):
                    raw = raw.rsplit("```", 1)[0]
                raw = raw.strip()

            selections = json.loads(raw)
            chunk_ids: list[str] = []
            for sel in selections:
                url = sel.get("url", "")
                node_ids = sel.get("node_ids", [])
                tree = next((t for t in trees if t.url == url), None)
                if tree:
                    for nid in node_ids:
                        chunk_ids.extend(tree.get_all_chunk_ids(nid))

            # Deduplicate while preserving order
            seen: set[str] = set()
            unique: list[str] = []
            for cid in chunk_ids:
                if cid not in seen:
                    seen.add(cid)
                    unique.append(cid)

            return unique[:max_chunks]

        except Exception as e:
            logger.warning("LLM reasoning failed: %s — falling back to heuristic", e)
            return self._reason_heuristic(query, trees, max_chunks)

    def _reason_heuristic(
        self, query: str, trees: list[PageTree], max_chunks: int
    ) -> list[str]:
        """Keyword-based fallback for tree navigation."""
        query_lower = query.lower()
        query_terms = set(query_lower.split())

        scored: list[tuple[float, str]] = []  # (score, chunk_id)
        for tree in trees:
            self._score_nodes(tree.nodes, query_terms, scored)

        scored.sort(key=lambda x: x[0], reverse=True)

        seen: set[str] = set()
        result: list[str] = []
        for _, cid in scored:
            if cid not in seen:
                seen.add(cid)
                result.append(cid)
            if len(result) >= max_chunks:
                break
        return result

    def _score_nodes(
        self,
        nodes: list[PageNode],
        query_terms: set[str],
        out: list[tuple[float, str]],
    ) -> None:
        for node in nodes:
            text = f"{node.name} {node.summary}".lower()
            node_terms = set(text.split())
            overlap = len(query_terms & node_terms)
            if overlap > 0:
                score = overlap / max(len(query_terms), 1)
                for cid in node.chunk_ids:
                    out.append((score, cid))
            self._score_nodes(node.children, query_terms, out)
