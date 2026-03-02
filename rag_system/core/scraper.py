"""Web scraper with structured content extraction.

Uses *trafilatura* for high-quality main-content extraction and
*BeautifulSoup* for structural analysis (headings, sections, tables,
links).  Supports conditional HTTP requests via ETag / Last-Modified
to minimise bandwidth when content hasn't changed.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
import trafilatura
from bs4 import BeautifulSoup, Tag

from rag_system.config import ScraperConfig

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Data models
# ------------------------------------------------------------------
@dataclass
class Section:
    """A logical section of a web page."""
    heading: str
    level: int  # 1–6 corresponding to h1–h6
    content: str
    subsections: list["Section"] = field(default_factory=list)


@dataclass
class ScrapedPage:
    """Structured representation of a scraped web page."""
    url: str
    title: str
    description: str
    main_content: str          # clean full-text (trafilatura)
    raw_html: str
    sections: list[Section]    # hierarchical heading tree
    tables: list[list[list[str]]]  # list of tables, each a list of rows
    links: list[dict]          # [{text, href}]
    meta: dict                 # HTTP metadata – etag, last-modified, status
    fetched_at: float


# ------------------------------------------------------------------
# Scraper
# ------------------------------------------------------------------
class WebScraper:
    """Fetch and structurally parse web pages."""

    def __init__(self, config: ScraperConfig) -> None:
        self._cfg = config
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": config.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        })

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    def fetch(
        self,
        url: str,
        etag: Optional[str] = None,
        last_modified: Optional[str] = None,
    ) -> Optional[ScrapedPage]:
        """Fetch *url* and return a structured ``ScrapedPage``.

        If *etag* / *last_modified* are supplied, a conditional GET is
        issued.  Returns ``None`` when the server replies 304 (not modified).
        """
        html = self._download(url, etag, last_modified)
        if html is None:
            return None
        return self._parse(url, html)

    # ------------------------------------------------------------------
    # Download with retries & conditional GET
    # ------------------------------------------------------------------
    def _download(
        self,
        url: str,
        etag: Optional[str] = None,
        last_modified: Optional[str] = None,
    ) -> Optional[str]:
        headers: dict[str, str] = {}
        if etag:
            headers["If-None-Match"] = etag
        if last_modified:
            headers["If-Modified-Since"] = last_modified

        for attempt in range(1, self._cfg.max_retries + 1):
            try:
                resp = self._session.get(
                    url,
                    headers=headers,
                    timeout=self._cfg.request_timeout,
                    allow_redirects=True,
                )
                if resp.status_code == 304:
                    logger.info("Content not modified (304): %s", url)
                    return None
                resp.raise_for_status()

                # Store HTTP caching headers on the session for the caller
                self._last_etag = resp.headers.get("ETag")
                self._last_modified = resp.headers.get("Last-Modified")
                self._last_status = resp.status_code

                return resp.text

            except requests.RequestException as exc:
                logger.warning(
                    "Attempt %d/%d failed for %s: %s",
                    attempt, self._cfg.max_retries, url, exc,
                )
                if attempt < self._cfg.max_retries:
                    time.sleep(self._cfg.retry_delay * attempt)
        logger.error("All %d attempts failed for %s", self._cfg.max_retries, url)
        return None

    # ------------------------------------------------------------------
    # Parse HTML into structured ScrapedPage
    # ------------------------------------------------------------------
    def _parse(self, url: str, html: str) -> ScrapedPage:
        soup = BeautifulSoup(html, "lxml")

        # -- Title --
        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()

        # -- Meta description --
        desc_tag = soup.find("meta", attrs={"name": "description"})
        description = ""
        if desc_tag and desc_tag.get("content"):
            description = desc_tag["content"].strip()

        # -- Main content via trafilatura --
        main_content = trafilatura.extract(
            html,
            include_links=False,
            include_tables=True,
            include_comments=False,
            output_format="txt",
            favor_recall=True,
        ) or ""

        # -- Sections (heading hierarchy) --
        sections = self._extract_sections(soup)

        # -- Tables --
        tables = self._extract_tables(soup)

        # -- Links --
        links = self._extract_links(soup, url)

        return ScrapedPage(
            url=url,
            title=title,
            description=description,
            main_content=main_content,
            raw_html=html,
            sections=sections,
            tables=tables,
            links=links,
            meta={
                "etag": getattr(self, "_last_etag", None),
                "last_modified": getattr(self, "_last_modified", None),
                "status": getattr(self, "_last_status", None),
            },
            fetched_at=time.time(),
        )

    # ------------------------------------------------------------------
    # Structural extraction helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_sections(soup: BeautifulSoup) -> list[Section]:
        """Build a hierarchical list of sections from heading tags."""
        headings = soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
        if not headings:
            return []

        sections: list[Section] = []
        stack: list[Section] = []

        for h in headings:
            level = int(h.name[1])
            text = h.get_text(separator=" ", strip=True)

            # Collect text between this heading and the next heading
            content_parts: list[str] = []
            for sibling in h.next_siblings:
                if isinstance(sibling, Tag) and sibling.name in (
                    "h1", "h2", "h3", "h4", "h5", "h6"
                ):
                    break
                if isinstance(sibling, Tag):
                    t = sibling.get_text(separator=" ", strip=True)
                    if t:
                        content_parts.append(t)

            section = Section(
                heading=text,
                level=level,
                content="\n".join(content_parts),
            )

            # Nest under the correct parent
            while stack and stack[-1].level >= level:
                stack.pop()

            if stack:
                stack[-1].subsections.append(section)
            else:
                sections.append(section)

            stack.append(section)

        return sections

    @staticmethod
    def _extract_tables(soup: BeautifulSoup) -> list[list[list[str]]]:
        """Extract all HTML tables as lists of rows."""
        tables: list[list[list[str]]] = []
        for table in soup.find_all("table"):
            rows: list[list[str]] = []
            for tr in table.find_all("tr"):
                cells = [
                    td.get_text(separator=" ", strip=True)
                    for td in tr.find_all(["th", "td"])
                ]
                if any(cells):
                    rows.append(cells)
            if rows:
                tables.append(rows)
        return tables

    @staticmethod
    def _extract_links(soup: BeautifulSoup, base_url: str) -> list[dict]:
        """Extract unique links with text."""
        seen: set[str] = set()
        links: list[dict] = []
        for a in soup.find_all("a", href=True):
            href = urljoin(base_url, a["href"])
            if href in seen:
                continue
            seen.add(href)
            text = a.get_text(separator=" ", strip=True)
            if text and urlparse(href).scheme in ("http", "https"):
                links.append({"text": text, "href": href})
        return links

    @property
    def last_etag(self) -> Optional[str]:
        return getattr(self, "_last_etag", None)

    @property
    def last_modified_header(self) -> Optional[str]:
        return getattr(self, "_last_modified", None)
