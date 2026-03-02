"""Document processor — converts scraped pages into structured Markdown documents.

Each URL is rendered as a well-structured Markdown file with:
  - YAML-style front-matter metadata
  - Hierarchical headings mirroring the original page
  - Cleaned body text
  - Extracted tables (as Markdown tables)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import quote

from rag_system.core.scraper import ScrapedPage, Section

logger = logging.getLogger(__name__)


@dataclass
class StructuredDocument:
    """An enriched, serializable document produced from a scraped page."""
    url: str
    title: str
    description: str
    content_markdown: str      # full Markdown representation
    plain_text: str            # clean plaintext for chunking
    sections: list[dict]       # flat list: [{heading, level, content, path}]
    tables_markdown: list[str]
    metadata: dict
    content_hash: str = ""
    created_at: float = field(default_factory=time.time)


class DocumentProcessor:
    """Transforms ``ScrapedPage`` objects into ``StructuredDocument``s."""

    def __init__(self, output_dir: str) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def process(self, page: ScrapedPage, content_hash: str = "") -> StructuredDocument:
        """Convert a *ScrapedPage* into a *StructuredDocument*."""
        # Flatten sections with full heading paths for contextual chunking
        flat_sections = self._flatten_sections(page.sections)

        # Build markdown
        md_parts: list[str] = []
        md_parts.append(self._front_matter(page, content_hash))
        md_parts.append(f"# {page.title}\n")
        if page.description:
            md_parts.append(f"> {page.description}\n")

        # Section content
        for sec in flat_sections:
            prefix = "#" * min(sec["level"] + 1, 6)
            md_parts.append(f"{prefix} {sec['heading']}\n")
            if sec["content"]:
                md_parts.append(sec["content"] + "\n")

        # If no sections extracted, use main_content
        if not flat_sections and page.main_content:
            md_parts.append(page.main_content + "\n")

        # Tables
        tables_md = self._tables_to_markdown(page.tables)
        if tables_md:
            md_parts.append("\n## Data Tables\n")
            for tmd in tables_md:
                md_parts.append(tmd + "\n")

        content_markdown = "\n".join(md_parts)

        # Plain text (for embedding / chunking) — use trafilatura output
        plain_text = page.main_content if page.main_content else self._sections_to_text(flat_sections)

        doc = StructuredDocument(
            url=page.url,
            title=page.title,
            description=page.description,
            content_markdown=content_markdown,
            plain_text=plain_text,
            sections=flat_sections,
            tables_markdown=tables_md,
            metadata={
                "fetched_at": page.fetched_at,
                "link_count": len(page.links),
                "table_count": len(page.tables),
                "section_count": len(flat_sections),
                **page.meta,
            },
            content_hash=content_hash,
        )

        # Persist to disk
        self._save(doc)
        return doc

    # ------------------------------------------------------------------
    # Markdown helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _front_matter(page: ScrapedPage, content_hash: str) -> str:
        lines = [
            "---",
            f"url: {page.url}",
            f"title: \"{page.title}\"",
            f"content_hash: {content_hash}",
            f"fetched_at: {page.fetched_at}",
            "---\n",
        ]
        return "\n".join(lines)

    @staticmethod
    def _flatten_sections(
        sections: list[Section], path: Optional[list[str]] = None
    ) -> list[dict]:
        """Recursively flatten the section tree, recording the heading path."""
        if path is None:
            path = []
        result: list[dict] = []
        for sec in sections:
            current_path = path + [sec.heading]
            result.append({
                "heading": sec.heading,
                "level": sec.level,
                "content": sec.content,
                "path": " > ".join(current_path),
            })
            if sec.subsections:
                result.extend(
                    DocumentProcessor._flatten_sections(sec.subsections, current_path)
                )
        return result

    @staticmethod
    def _tables_to_markdown(tables: list[list[list[str]]]) -> list[str]:
        """Convert raw table data to Markdown table strings."""
        md_tables: list[str] = []
        for table in tables:
            if not table:
                continue
            header = table[0]
            col_count = len(header)
            lines = ["| " + " | ".join(header) + " |"]
            lines.append("| " + " | ".join(["---"] * col_count) + " |")
            for row in table[1:]:
                # Pad / trim to match header width
                padded = (row + [""] * col_count)[:col_count]
                lines.append("| " + " | ".join(padded) + " |")
            md_tables.append("\n".join(lines))
        return md_tables

    @staticmethod
    def _sections_to_text(sections: list[dict]) -> str:
        parts: list[str] = []
        for sec in sections:
            parts.append(sec["heading"])
            if sec["content"]:
                parts.append(sec["content"])
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _save(self, doc: StructuredDocument) -> Path:
        safe = quote(doc.url, safe="")[:180]
        path = self._output_dir / f"{safe}.md"
        path.write_text(doc.content_markdown, encoding="utf-8")

        # Also save metadata JSON
        meta_path = self._output_dir / f"{safe}.meta.json"
        meta_path.write_text(
            json.dumps(doc.metadata, indent=2, default=str),
            encoding="utf-8",
        )
        logger.info("Saved document: %s -> %s", doc.url, path)
        return path
