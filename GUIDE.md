# Production RAG System - A Beginner's Guide

## What This Document Covers

This guide explains every part of a production-grade **Retrieval-Augmented Generation (RAG)** system built in Python. It's written for someone who is new to RAG. By the end, you'll understand what RAG is, why each component exists, and how they all fit together.

---

## Table of Contents

1. [What Is RAG?](#1-what-is-rag)
2. [Why Do We Need RAG?](#2-why-do-we-need-rag)
3. [System Architecture Overview](#3-system-architecture-overview)
4. [Project Structure](#4-project-structure)
5. [The Data Pipeline (Ingestion)](#5-the-data-pipeline-ingestion)
   - 5.1 [Web Scraping](#51-web-scraping-scraperpycore)
   - 5.2 [Document Processing](#52-document-processing-document_processorpycore)
   - 5.3 [Chunking](#53-chunking-chunkerpycore)
   - 5.4 [Embeddings](#54-embeddings-embeddingspycore)
   - 5.5 [Vector Store](#55-vector-store-vector_storepystorage)
6. [Content Caching](#6-content-caching-cache_managerpycore)
7. [Orphan Cleanup](#7-orphan-cleanup)
8. [PageIndex - Hierarchical Tree Indexing](#8-pageindex---hierarchical-tree-indexing-page_indexpycore)
9. [BM25 Keyword Retrieval](#9-bm25-keyword-retrieval-bm25_indexpycore)
10. [The Query Pipeline (Retrieval + Generation)](#10-the-query-pipeline-rag_enginepycore)
11. [Retrieval Caching](#11-retrieval-caching-retrieval_cachepycore)
12. [Search Schema JSON Output](#12-search-schema-json-output)
13. [The Orchestrator](#13-the-orchestrator-pipelinepy)
14. [The CLI Interface](#14-the-cli-interface-mainpy)
15. [The REST API](#15-the-rest-api-apipy)
16. [Configuration](#16-configuration-configpy)
17. [Key Concepts Glossary](#17-key-concepts-glossary)
18. [How to Run the System](#18-how-to-run-the-system)

---

## 1. What Is RAG?

**Retrieval-Augmented Generation** is a technique that makes Large Language Models (LLMs) like Claude smarter about **specific data** they weren't trained on.

Here's the problem RAG solves:

- An LLM like Claude knows general knowledge, but it doesn't know the current contents of, say, the CommBank home loans page.
- If you ask "What's CommBank's current home loan interest rate?", the LLM will either make something up (hallucinate) or say it doesn't know.

RAG fixes this by adding a **retrieval step** before asking the LLM:

```
Without RAG:
  User Question --> LLM --> Answer (may hallucinate)

With RAG:
  User Question --> Search your data --> Find relevant passages --> LLM + passages --> Accurate answer
```

The LLM now answers based on **actual data** you provide, not just its training. This dramatically reduces hallucination and lets the LLM work with up-to-date, domain-specific information.

---

## 2. Why Do We Need RAG?

| Problem | RAG Solution |
|---|---|
| LLMs have knowledge cutoff dates | RAG feeds them fresh, scraped data |
| LLMs hallucinate facts | RAG grounds answers in real source documents |
| LLMs can't access private/proprietary data | RAG indexes your specific content |
| Long documents exceed LLM context limits | RAG retrieves only the relevant chunks |

---

## 3. System Architecture Overview

This system has two main phases:

### Phase 1: Ingestion (prepare the data)

```
URL --> Scrape HTML --> Extract structured content --> Split into chunks
    --> Generate embeddings --> Store in vector database
    --> Build PageIndex tree
```

### Phase 2: Query (answer questions)

```
User question --> Check retrieval cache (exact hash, then semantic similarity)
              --> On cache HIT: return cached answer instantly (~15ms)
              --> On cache MISS:
                  --> Embed the question
                  --> Vector search (find similar chunks)
                  --> BM25 keyword search (find term-matching chunks)
                  --> PageIndex reasoning (find relevant sections)
                  --> Three-way merge & re-rank (vector + BM25 + PageIndex)
                  --> Send to LLM with context
                  --> Store result in retrieval cache
                  --> Return answer with sources (optionally as JSON search schema)
```

Here's how all the components connect:

```
                        +------------------+
                        |    main.py       |  <-- CLI: ingest / query / status / clear
                        |   (Click CLI)    |
                        +--------+---------+
                                 |
                +----------------+----------------+
                |                                 |
       +--------v---------+             +--------v---------+
       |   pipeline.py    |             |  rag_engine.py   |
       |  (Ingestion +    |             |  (Query: retrieve|
       |   orphan cleanup |             |  + cache + gen)  |
       |   + BM25 rebuild)|             |  (3-way merge)   |
       +--------+---------+             +--------+---------+
                |                                |
  +------+-----+----+-----+------+       +------+------+------+-------+
  |      |     |    |     |      |       |      |      |      |       |
+-v---+ +v--+ +v--+ +v-+ +v--+ +v---+ +v----+ +v--+ +v---+ +v-----+ +v--------+
|scrap| |doc| |chu| |em| |vec| |page| |bm25 | |vec| |bm25| |page  | |retrieval|
|er   | |pro| |nk | |be| |sto| |idx | |index| |sto| |idx | |index | |cache    |
|.py  | |.py| |.py| |d | |re | |.py | |.py  | |re | |.py | |.py   | |.py      |
+-----+ +---+ +---+ |.p| |.py| +----+ +-----+ |.py| +----+ +------+ +---------+
                     +--+ +---+                 +---+
```

---

## 4. Project Structure

```
RAG/
├── main.py                          # CLI entry point - the commands you run
├── api.py                           # REST API entry point (FastAPI + Uvicorn)
├── requirements.txt                 # Python packages needed
├── GUIDE.md                         # This guide
│
├── rag_system/                      # Main Python package
│   ├── __init__.py                  # Package marker (defines version)
│   ├── config.py                    # All settings in one place
│   ├── pipeline.py                  # Orchestrates the full ingestion flow + orphan cleanup
│   │
│   ├── core/                        # Core processing modules
│   │   ├── scraper.py               # Fetches and parses web pages
│   │   ├── document_processor.py    # Turns raw HTML into structured documents
│   │   ├── chunker.py               # Splits documents into small pieces
│   │   ├── embeddings.py            # Converts text into number vectors
│   │   ├── page_index.py            # Builds hierarchical document trees
│   │   ├── bm25_index.py            # BM25 keyword retrieval index
│   │   ├── rag_engine.py            # The query engine - retrieval + generation
│   │   ├── retrieval_cache.py       # Two-tier semantic query cache
│   │   └── cache_manager.py         # Avoids re-processing unchanged content
│   │
│   └── storage/                     # Data persistence
│       └── vector_store.py          # ChromaDB vector database wrapper
│
└── data/                            # All generated data lives here
    ├── cache/                       # Cached web content + retrieval cache SQLite DBs
    ├── documents/                   # Structured Markdown documents
    ├── chroma_db/                   # ChromaDB persistent vector storage
    ├── page_index/                  # Hierarchical tree indexes (JSON)
    └── bm25_index/                  # BM25 keyword index (pickle)
```

---

## 5. The Data Pipeline (Ingestion)

Ingestion is the process of turning raw web pages into searchable data. Let's walk through each step.

### 5.1 Web Scraping (`scraper.py`/core)

**What it does:** Downloads web pages and extracts their structure.

**Why it's needed:** We need the actual content from URLs before we can do anything with it.

**How it works:**

1. **HTTP Download** - Uses the `requests` library to fetch the HTML from a URL. It includes retry logic (tries up to 3 times with increasing delays) in case the server is slow or temporarily unavailable.

2. **Conditional GET** - This is a bandwidth-saving technique. After the first fetch, we store HTTP headers called `ETag` and `Last-Modified`. On subsequent fetches, we send these back to the server. If the content hasn't changed, the server responds with HTTP status `304 Not Modified` instead of sending the entire page again.

3. **Content Extraction** - Uses two tools:
   - **trafilatura**: A library specifically designed to extract the main content from a web page, stripping away navigation bars, footers, ads, and boilerplate. It gives us clean text.
   - **BeautifulSoup**: A general HTML parser that lets us extract the structural elements: headings (h1-h6), tables, and links.

4. **Output** - A `ScrapedPage` object containing:
   - `title`: The page title
   - `description`: The meta description
   - `main_content`: Clean plaintext (from trafilatura)
   - `sections`: A hierarchical list of headings and their content
   - `tables`: All HTML tables as structured data
   - `links`: All hyperlinks found on the page
   - `meta`: HTTP metadata (ETag, Last-Modified)

**Key code pattern - Section extraction:**
The scraper builds a tree of sections by reading headings (h1 through h6) in order. When it encounters an h2 after an h1, it nests the h2 under the h1. This preserves the document's logical structure.

```
h1: Home Loans               <-- Top level
  h2: Why Choose CommBank?    <-- Nested under h1
    h3: First Home Buyers     <-- Nested under h2
  h2: Interest Rates          <-- Sibling of first h2
```

---

### 5.2 Document Processing (`document_processor.py`/core)

**What it does:** Converts a scraped page into a clean, structured Markdown document.

**Why it's needed:** Raw HTML is messy. We need a clean, standardised format that's easy to chunk and search.

**How it works:**

1. **Flatten sections** - The nested heading tree gets flattened into a list, but each section remembers its full path. For example:
   ```
   heading: "First Home Buyers"
   path: "Home Loans > Why Choose CommBank? > First Home Buyers"
   level: 3
   content: "Awarded Canstar's Bank of the Year..."
   ```
   This path is important later because it gives each chunk *context* about where it came from in the document.

2. **Generate Markdown** - Builds a complete Markdown file with:
   - YAML front-matter (URL, title, content hash, timestamp)
   - Hierarchical headings matching the original page
   - Body text
   - Tables converted to Markdown format

3. **Save to disk** - Each URL gets two files:
   - `{url}.md` - The Markdown document
   - `{url}.meta.json` - Metadata (fetch time, link count, etc.)

**Output:** A `StructuredDocument` with both `content_markdown` (formatted) and `plain_text` (for chunking).

---

### 5.3 Chunking (`chunker.py`/core)

**What it does:** Splits documents into small pieces (chunks) that fit within the embedding model's and LLM's context limits.

**Why it's needed:** You can't embed or search an entire web page at once. Search works better with small, focused pieces of text. Think of it like an index at the back of a textbook - each entry points to a specific paragraph, not an entire chapter.

**How it works:**

The chunker uses a **recursive character splitting** strategy with these settings:
- **Chunk size:** 512 tokens (about 380 words)
- **Chunk overlap:** 64 tokens (about 48 words)
- **Minimum chunk size:** 50 tokens (discard tiny fragments)

#### What are tokens?
Tokens are the units LLMs process text in. A token is roughly 3/4 of a word. "Commonwealth Bank home loans" is about 5 tokens. The system uses `tiktoken` (OpenAI's tokenizer) to count tokens accurately.

#### The splitting algorithm:

```
Step 1: Is the text small enough? (< 512 tokens)
        Yes --> Return it as a single chunk
        No  --> Go to Step 2

Step 2: Try splitting on the first separator in this priority list:
        "\n## "    (Markdown h2 headings)
        "\n### "   (Markdown h3 headings)
        "\n#### "  (Markdown h4 headings)
        "\n\n"     (Double newline / paragraph break)
        "\n"       (Single newline)
        ". "       (Sentence boundary)
        " "        (Word boundary)

Step 3: Combine the parts back together until they fill a chunk.
        If a part is still too big, recurse with the next separator.

Step 4: As a last resort, do a hard split at the token level.
```

The idea is to split at **natural boundaries** (heading breaks > paragraph breaks > sentence breaks > word breaks) rather than cutting text mid-sentence.

#### Section context preservation:

This is a key technique. Each chunk gets a prefix that tells you *where* in the document it came from:

```
[Home loans, interest rates, calculators & offers | CommBank] [Home loans > Why Choose CommBank? > First Home Buyers]

Awarded Canstar's Bank of the Year for First Home Buyers...
```

Without this context, a chunk like "Awarded Canstar's Bank of the Year" is ambiguous - awarded for what? By whom? The section path makes each chunk self-contained.

#### Overlap:

The last 64 tokens of each chunk are prepended to the next chunk. This ensures that if a sentence is split across two chunks, both chunks contain the full sentence.

```
Chunk 1: "...flexible home loan features to suit your needs."
Chunk 2: "...to suit your needs. Awarded Canstar's Bank of the Year..."
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                              This is the overlap from Chunk 1
```

#### Output:

Each chunk is a `Chunk` object with:
- `chunk_id`: A unique SHA-256 hash
- `text`: The chunk content (with section context prefix)
- `token_count`: How many tokens it contains
- `metadata`: Source URL, title, section heading, section path, chunk type, content hash

---

### 5.4 Embeddings (`embeddings.py`/core)

**What it does:** Converts text into numerical vectors (lists of numbers) that capture semantic meaning.

**Why it's needed:** Computers can't search by meaning using raw text. By converting text to vectors, we can find "similar" text using mathematical distance.

**The key insight:** Texts with similar meaning end up as vectors that are close together in a high-dimensional space:

```
"home loan interest rates"  -->  [0.23, -0.45, 0.12, ...]   \
"mortgage rate comparison"  -->  [0.21, -0.42, 0.15, ...]    |-- Close together!
                                                              /
"credit card annual fee"    -->  [0.67, 0.31, -0.55, ...]   <-- Far away
```

**How it works:**

1. **Model:** Uses `sentence-transformers` with the `all-MiniLM-L6-v2` model. This is a lightweight model (80MB) that runs locally on your CPU - no API calls needed.

2. **Dimension:** Each piece of text becomes a vector of **384 numbers** (384 dimensions). This is the model's fixed output size.

3. **Normalisation:** Vectors are normalised (scaled to unit length) so that cosine similarity can be computed efficiently.

4. **Batch processing:** Multiple texts are embedded in a single batch for efficiency rather than one at a time.

5. **Caching:** An in-memory cache avoids re-computing embeddings for identical texts within a session.

**Output:** For each chunk, a list of 384 floating-point numbers representing its semantic meaning.

---

### 5.5 Vector Store (`vector_store.py`/storage)

**What it does:** Stores chunk embeddings in a searchable database and performs similarity search.

**Why it's needed:** We need a fast way to find the most relevant chunks for a user's question. A vector database is optimised for this "nearest neighbor" search.

**Technology:** [ChromaDB](https://www.trychroma.com/) - an open-source, lightweight vector database that runs locally with persistent storage.

**How it works:**

1. **Storing (Upsert):**
   Each chunk is stored with three things:
   - Its **ID** (the unique hash from the chunker)
   - Its **embedding** (the 384-dimensional vector)
   - Its **document text** (the actual chunk content)
   - Its **metadata** (source URL, section heading, etc.)

2. **Searching:**
   When you ask a question:
   - Your question gets embedded into the same 384-dimensional space
   - ChromaDB finds the chunks whose embeddings are **closest** to your question's embedding
   - "Closest" is measured by **cosine similarity** (how much two vectors point in the same direction)

   ```
   Your question: "What are CommBank home loan rates?"
   Embedding:     [0.25, -0.43, 0.11, ...]

   ChromaDB finds:
     Chunk A (distance: 0.164) -- "Interest rate 5.59% PA..."       <-- Most similar
     Chunk B (distance: 0.577) -- "Apply for a home loan..."        <-- Somewhat similar
     Chunk C (distance: 0.599) -- "Personal loan features..."       <-- Less similar
   ```

3. **Metadata filtering:**
   You can filter search results by metadata. For example, search only chunks from a specific URL:
   ```python
   store.search_by_url(query_embedding, "https://www.commbank.com.au/home-loans.html")
   ```

4. **Persistence:**
   ChromaDB stores everything in the `data/chroma_db/` directory. Data survives program restarts.

---

## 6. Content Caching (`cache_manager.py`/core)

**What it does:** Avoids re-scraping and re-processing URLs whose content hasn't changed.

**Why it's needed:** Scraping, chunking, and embedding are expensive operations. If a web page hasn't changed since last time, we should skip all that work.

**How it works - Two layers of caching:**

### Layer 1: HTTP Conditional Requests

When we first scrape a URL, the web server sends headers like:
```
ETag: "abc123"
Last-Modified: Sat, 01 Mar 2026 10:00:00 GMT
```

On the next scrape, we send these back:
```
If-None-Match: "abc123"
If-Modified-Since: Sat, 01 Mar 2026 10:00:00 GMT
```

If nothing changed, the server responds with **HTTP 304 Not Modified** (no body) instead of sending the entire page again. This saves bandwidth and time.

### Layer 2: Content Hash Comparison

Even if the server doesn't support conditional requests, we compute a **hash** (digital fingerprint) of the page content using xxHash128:

```
Content: "Home loans, interest rates..."  -->  Hash: "a7b3c9d2e1f4..."
```

If the hash matches what we stored last time, the content is identical and we skip reprocessing.

### Storage:

Caching metadata is stored in a **SQLite database** (`cache_meta.db`):

| url | content_hash | etag | last_modified | last_fetched | content_file |
|-----|-------------|------|---------------|--------------|-------------|
| https://commbank.com.au/home-loans.html | a7b3c9... | "xyz" | Sat, 01... | 1709312000.0 | /path/to/file.txt |

The actual content is stored as text files on disk.

### Result:

In our test, the second ingestion run completed in **0.4 seconds** (vs 3.1 seconds for the first run) because all 5 URLs returned HTTP 304.

---

## 7. Orphan Cleanup

**What it does:** Automatically removes all data (chunks, cache entries, PageIndex trees) for URLs that are no longer in the active ingestion list.

**Why it's needed:** If you initially ingest 5 URLs and later drop 2 of them, the old chunks, cache files, and PageIndex trees for those 2 URLs would remain in the system forever, polluting search results with stale data. Orphan cleanup ensures the system only contains data for the URLs you actively manage.

**How it works:**

At the end of every ingestion run, the pipeline compares the **current URL list** against all **previously known URLs** (gathered from both the vector store and the content cache):

```
Previously known URLs:    {A, B, C, D, E}
Current ingestion list:   {A, B, C}
Orphaned URLs:            {D, E}   <-- these get cleaned up
```

For each orphaned URL, three things are deleted:
1. **Vector store chunks** — all chunks whose `source_url` metadata matches the orphaned URL
2. **PageIndex tree** — the JSON tree file for that URL
3. **Content cache** — the cached content and SQLite metadata row

**Where it lives:**

The cleanup logic is in `pipeline.py` in the `_cleanup_removed_urls()` method. It runs automatically at the end of every `ingest` command — no manual intervention needed.

**Example:**

```
# First run: ingest all 5 URLs
python main.py ingest                         # 109 chunks, 5 PageIndex docs

# Second run: ingest only 3 URLs
python main.py ingest -u url1 -u url2 -u url3 # Orphan cleanup removes url4, url5

# Status confirms cleanup
python main.py status                         # 58 chunks, 3 PageIndex docs
```

The CLI displays which URLs were removed in a dedicated "Removed URLs (Orphan Cleanup)" table after ingestion.

---

## 8. PageIndex - Hierarchical Tree Indexing (`page_index.py`/core)

**What it does:** Builds a structured "table of contents" tree for each document and uses it to find relevant sections at query time.

**Why it's needed:** Vector search has a fundamental limitation: it matches by **text similarity**, not by **understanding**. PageIndex adds a complementary retrieval path that reasons about document structure.

### The Problem with Vector-Only Search

Consider this scenario:
- A chunk says: "See our home loan comparison table in the rates section."
- Your question is: "What are the home loan rates?"
- The **rates table** chunk would be the best answer, but the **reference chunk** about the rates might score higher in vector search because it contains more of your query's keywords.

PageIndex solves this by building a map of the document's structure, allowing the system to **navigate** to the right section rather than just matching text.

### How PageIndex Works

#### Building the tree (indexing phase):

For each document, a tree is built that looks like this:

```json
{
  "url": "https://www.commbank.com.au/home-loans.html",
  "title": "Home loans, interest rates, calculators & offers | CommBank",
  "summary": "CommBank home loans page covering loan types, rates, ...",
  "nodes": [
    {
      "node_id": "0001",
      "name": "Why Choose CommBank?",
      "summary": "Benefits of CommBank home loans including awards...",
      "chunk_ids": ["abc123", "def456"],
      "nodes": [
        {
          "node_id": "0002",
          "name": "First Home Buyers",
          "summary": "Support for first-time home buyers...",
          "chunk_ids": ["ghi789"]
        }
      ]
    }
  ]
}
```

Each node has:
- A **name** (the section heading)
- A **summary** (what the section contains)
- **chunk_ids** (which chunks belong to this section)
- **children nodes** (sub-sections)

#### Two building modes:

1. **LLM-powered** (when `ANTHROPIC_API_KEY` is set): Claude analyses the document and generates rich, detailed summaries for each tree node. This produces a higher-quality index.

2. **Heuristic** (no API key): Uses the heading structure and text overlap to build the tree. Summaries are just the first 200 characters of each section.

#### Querying the tree (retrieval phase):

When a user asks a question:

1. **LLM reasoning** (with API key): Claude reads all the tree summaries and decides which sections are relevant, returning their node IDs. This is like asking a librarian "Where would I find information about X?"

2. **Keyword matching** (without API key): Compares query terms against node names and summaries, scoring each by term overlap.

The result is a list of **chunk IDs** from the relevant sections, which are then fetched from the vector store.

### Why This Complements Vector Search

| Aspect | Vector Search | PageIndex |
|---|---|---|
| How it finds results | Text similarity (cosine distance) | Structural reasoning (tree navigation) |
| Strengths | Great for open-ended queries | Great for structured/navigational queries |
| Weaknesses | Can miss structurally relevant content | Requires good document structure |
| Example query it excels at | "flexible home loan features" | "What are the home loan types?" |

The RAG engine uses **both** and merges results (see next section).

---

## 9. BM25 Keyword Retrieval (`bm25_index.py`/core)

**What it does:** Provides a term-frequency-based keyword search as a third retrieval path alongside vector search and PageIndex reasoning.

**Why it's needed:** Dense vector embeddings capture semantic meaning but can miss exact keyword matches. Product names, acronyms, specific codes, and technical terms often score poorly in vector search because the embedding model doesn't recognise them. BM25 excels at these exact-match scenarios because it scores documents based on how many query terms they contain and how rare those terms are across the corpus.

### How BM25 Works

BM25 (Best Match 25) is a classic information retrieval algorithm that scores how well a document matches a query based on three factors:

1. **Term Frequency (TF)** — How many times the query term appears in the document. More occurrences = higher score, but with diminishing returns (controlled by `k1=1.5`).

2. **Inverse Document Frequency (IDF)** — How rare the term is across all documents. Rare terms (like "conveyancing") contribute more than common terms (like "loan").

3. **Document Length Normalization** — Longer documents are penalised slightly so they don't dominate just because they contain more words (controlled by `b=0.75`).

```
Example:

Query: "home loan redraw facility"

Document A: "What is a home loan redraw facility?"
  → High BM25 score (all query terms present, short document)

Document B: "Apply for a home loan online..."
  → Lower BM25 score (only "home" and "loan" present, missing "redraw" and "facility")

Document C: "Credit card annual fee comparison"
  → Score = 0 (no query terms present)
```

### Why This Complements Vector Search

| Aspect | Vector Search | BM25 |
|---|---|---|
| How it matches | Semantic similarity (meaning) | Exact term presence (keywords) |
| Strengths | "mortgage" matches "home loan" | "LVR" matches "LVR" exactly |
| Weaknesses | May miss exact terms/acronyms | Misses synonyms and paraphrasing |
| Score range | 0.0 - 1.0 (cosine similarity) | 0.0 - unbounded (normalised to 0-1 before merging) |

Together, vector search catches semantic intent while BM25 catches exact terminology — a combination known as **hybrid retrieval**.

### Implementation Details

#### Tokenization

The system uses a lightweight custom tokenizer (no external NLP libraries):
- Converts text to lowercase
- Strips punctuation
- Removes common English stop words (130+ words like "the", "is", "and")
- Discards single-character tokens

This is sufficient for the financial products domain and avoids heavy dependencies like nltk or spacy.

#### Index Building (during ingestion)

After all URLs are processed and orphans cleaned up, the pipeline rebuilds a **global BM25 index** from all chunks in the vector store:

```
[per-URL ingestion loop]
  Fetch → Process → Chunk → Embed → Store → PageIndex
[orphan cleanup]
[BM25 index rebuild]  ← Reads ALL chunks, tokenizes, builds BM25Okapi
```

The index is global (not per-URL) because BM25's IDF calculation needs the full corpus to determine which terms are genuinely rare. Per-URL indexes would have too few documents (~10-30 chunks) for statistically meaningful IDF values.

#### Persistence

The BM25 index is serialised to disk via Python's `pickle` at `data/bm25_index/bm25_index.pkl`. It loads automatically on startup and is rebuilt during each ingestion that processes or removes URLs.

#### BM25 Score Normalization

Raw BM25 scores are unbounded (they depend on corpus size and term frequencies). Before merging with other retrievers, scores are **max-normalised** to [0, 1]:

```
normalised_score = raw_score / max_score_in_result_set
```

This means the top BM25 result always gets a score of 1.0, making it comparable with vector similarity scores (also 0-1) and PageIndex rank scores.

### Configuration

| Setting | Default | What it controls |
|---|---|---|
| `bm25.k1` | 1.5 | Term frequency saturation. Higher = more credit for repeated terms |
| `bm25.b` | 0.75 | Document length normalization. 0 = no penalty for long docs, 1 = full penalty |
| `bm25.enabled` | True | Master toggle for BM25 retrieval |

---

## 10. The Query Pipeline (`rag_engine.py`/core)

**What it does:** Takes a user's question, retrieves relevant information from three sources (vector search, BM25 keyword search, and PageIndex reasoning), merges the results with a weighted three-way formula, and generates an answer. It also integrates a retrieval cache to skip expensive work for repeated or semantically similar queries.

**This is the heart of the RAG system.**

### Step-by-step flow:

```
Question: "What home loan options does CommBank offer?"
    |
    v
[0. Check retrieval cache]
    |  Exact hash match? --> HIT --> Return cached response (15ms)
    |  Semantic match (>= 0.92 cosine)? --> HIT --> Return cached response
    |  Miss --> continue
    v
[1. Embed the question]
    |  --> [0.25, -0.43, 0.11, ...]  (384-dimensional vector)
    v
[2. Vector Search]        [3. BM25 Search]          [4. PageIndex Reasoning]
    |  ChromaDB cosine        |  Term-frequency           |  Navigate tree
    |  search                 |  scoring against           |  structure
    |  Returns top-K          |  all chunks               |  Returns relevant
    |  similar chunks         |  Returns keyword           |  node chunk IDs
    v                         v  matches                   v
[5. Three-Way Merge & Re-rank]
    |  Vector (40%) + BM25 (35%) + PageIndex (25%)
    |  Weights auto-redistribute when any retriever is disabled
    |  Both raw and reranked scores are preserved
    v
[6. Build context]
    |  Format top results as numbered sources
    v
[7. Generate answer]
    |  Send question + context to Claude
    v
[8. Store in retrieval cache]
    |  Save response keyed by query hash + embedding
    v
[Final Answer with source citations]
```

### Three-Way Scoring

Every retrieval result carries **two scores**:

| Score | Field | What it measures |
|---|---|---|
| **Vector score** | `score` / `@search.score` | Raw cosine similarity from ChromaDB (1 - distance). 0.0 for non-vector results. |
| **Reranker score** | `reranker_score` / `@search.rerankerScore` | Combined score after three-way merge. This is what results are sorted by. |

The reranker formula:
```
reranker_score = 0.40 * vector_score + 0.35 * bm25_score + 0.25 * pageindex_score
```

- **Vector score** is derived from cosine similarity (1 - distance)
- **BM25 score** is max-normalised to [0, 1] (top BM25 hit = 1.0)
- **PageIndex score** is based on the rank position from the reasoner (1.0 for rank 1, decreasing by 0.05 per rank)

#### Dynamic Weight Redistribution

When a retriever is disabled (via `--no-bm25` or `--no-page-index`), its weight is automatically redistributed so the remaining weights sum to 1.0:

| Active Retrievers | Vector | BM25 | PageIndex |
|---|---|---|---|
| All three | 0.40 | 0.35 | 0.25 |
| Vector + BM25 (no PageIndex) | 0.533 | 0.467 | 0.0 |
| Vector + PageIndex (no BM25) | 0.615 | 0.0 | 0.385 |
| Vector only | 1.0 | 0.0 | 0.0 |

Each result is tagged with its source(s) as a `+`-joined label:
- `"vector"` — found only by vector search
- `"bm25"` — found only by BM25
- `"page_index"` — found only by PageIndex
- `"vector+bm25"` — found by vector and BM25
- `"vector+bm25+page_index"` — found by all three methods (usually the highest quality)

Results found by multiple retrievers tend to have the highest reranker scores because they receive contributions from multiple scoring paths.

### Answer Generation

If an Anthropic API key is configured, the merged context is sent to Claude with a system prompt that instructs it to:
- Only use information from the provided context
- Cite source numbers `[Source N]`
- Use tables for comparisons
- Be honest when the context doesn't contain enough information

Without an API key, the raw retrieved context is returned as-is.

---

## 11. Retrieval Caching (`retrieval_cache.py`/core)

**What it does:** Caches query results so that repeated or semantically similar questions return instantly without re-running vector search, PageIndex reasoning, or LLM generation.

**Why it's needed:** Content caching (Section 6) avoids re-scraping unchanged web pages. But what about the **query side**? If a user asks "What are the home loan rates?" and then asks the same question again 5 minutes later, the system would repeat the full retrieval pipeline. Retrieval caching avoids this by storing and serving previously computed results.

### The difference between Content Caching and Retrieval Caching

| Aspect | Content Caching | Retrieval Caching |
|---|---|---|
| **Phase** | Ingestion (data pipeline) | Query (retrieval pipeline) |
| **What it caches** | Raw scraped web page content | Query results (answer + sources) |
| **Cache key** | URL + content hash | Query text + query embedding |
| **Saves** | Scraping, chunking, embedding time | Retrieval, reranking, LLM generation time |

### Two-Tier Matching Strategy

The retrieval cache uses two levels of lookup:

#### Tier 1: Exact Match (O(1) lookup)
The query string is lowercased, combined with `top_k` and `url_filter`, then hashed with SHA-256. If this hash matches a stored entry, it's an instant hit.

```
Query: "What are the home loan rates?"
Hash:  sha256("what are the home loan rates?||8||") → "a3b7c9..."
       ↓
Lookup in SQLite → FOUND → return cached answer
```

#### Tier 2: Semantic Match (cosine similarity)
If no exact match is found, the query's embedding is compared against all cached query embeddings using cosine similarity. If the best match exceeds a threshold (default **0.92**), it's treated as a hit.

```
New query:      "What home loan rates does CommBank offer?"
Cached query:   "What are the home loan rates?"
Cosine similarity: 0.94 (>= 0.92 threshold) → SEMANTIC HIT
```

The 0.92 threshold is deliberately conservative. In testing:
- "CommBank" vs "Commonwealth Bank" = 0.73 similarity (correctly rejected)
- Different phrasing of same intent = 0.85 similarity (correctly rejected)
- Near-identical phrasing = 0.94+ (correctly accepted)

This prevents returning stale results when the query intent differs, even slightly.

### Cache Invalidation

Cache entries become invalid in three ways:

1. **Content fingerprint change** — When any document's content changes during re-ingestion, a new fingerprint is computed from all content hashes in the vector store. Old cache entries that were generated against a different fingerprint are automatically rejected.

2. **TTL expiry** — Each entry has a time-to-live of 24 hours (configurable). After that, a fresh retrieval is forced.

3. **LRU eviction** — When the cache exceeds 1,000 entries (configurable), the least-recently-used entries are deleted.

### Storage

The cache is persisted in SQLite at `data/cache/retrieval_cache.db`. Each row stores:
- The query text and its SHA-256 hash
- The query embedding (as JSON array)
- The complete answer and serialised sources
- A content fingerprint for invalidation
- Hit count and timestamps for LRU eviction

An **in-memory embedding index** (a list of numpy arrays) is loaded at startup for fast cosine similarity computation during Tier 2 lookups.

### Performance Impact

| Scenario | Without Cache | With Cache |
|---|---|---|
| First query | ~130ms | ~130ms (cache miss, stores result) |
| Same query again | ~130ms | **~15ms** (exact cache hit) |
| Semantically similar query | ~130ms | **~15ms** (semantic cache hit) |

### Usage

```bash
# Normal query (cache enabled by default)
python main.py query "What are the home loan rates?"

# Bypass cache (force fresh retrieval)
python main.py query "What are the home loan rates?" --no-cache

# Clear just the retrieval cache
python main.py clear --query-cache --yes
```

---

## 12. Search Schema JSON Output

**What it does:** The `--json` flag on the `query` command outputs retrieval results in a structured JSON format modelled after Azure AI Search's response schema.

**Why it's needed:** If you want to integrate this RAG system with a frontend application, API gateway, or another service, you need a standardised response format. The search schema provides a well-defined JSON structure with separate fields for vector scores, reranker scores, content, and source metadata.

### The Schema

```json
{
  "@data.context": "What are the home loan options?",
  "@data.count": 8,
  "@search.answers": [
    {
      "text": "CommBank offers several home loan options...",
      "score": 0.5807
    }
  ],
  "value": [
    {
      "@search.score": 0.5807,
      "@search.rerankerScore": 0.7084,
      "Content": "Awarded Canstar's Bank of the Year...",
      "sourcefile": "home-loans.html",
      "doc_url": "https://www.commbank.com.au/home-loans.html"
    }
  ]
}
```

### Field Reference

| Field | Description |
|---|---|
| `@data.context` | The original query string |
| `@data.count` | Number of results returned |
| `@search.answers` | LLM-generated answer (empty if `--no-generate` is used) |
| `@search.answers[].text` | The generated answer text |
| `@search.answers[].score` | Score of the top source used for generation |
| `value` | Array of retrieved chunks, ordered by reranker score |
| `@search.score` | Raw vector similarity score (0.0 for non-vector results) |
| `@search.rerankerScore` | Combined score after three-way hybrid reranking (vector 40% + BM25 35% + PageIndex 25%) |
| `Content` | The chunk text, including section context prefix |
| `sourcefile` | Filename derived from the source URL (e.g., `home-loans.html`) |
| `doc_url` | Full source URL |

### Usage

```bash
# JSON output with retrieval only
python main.py query "What are the home loan options?" --json --no-generate

# JSON output with LLM generation (populates @search.answers)
python main.py query "What are the home loan options?" --json
```

---

## 13. The Orchestrator (`pipeline.py`)

**What it does:** Coordinates all the components in the correct order for the ingestion pipeline, including orphan cleanup at the end.

**Why it's needed:** Individual components (scraper, processor, chunker, etc.) don't know about each other. The pipeline wires them together and handles the flow.

### The `_ingest_url()` method processes each URL through 8 steps:

```
Step 1: Fetch          --> Scrape the URL (with conditional GET for caching)
Step 2: Hash check     --> Compare content hash to detect changes
Step 3: Process        --> Convert to structured Markdown document
Step 4: Chunk          --> Split into token-aware chunks with metadata
Step 5: Embed          --> Generate 384-dimensional vectors for each chunk
Step 6: Store          --> Delete old chunks for this URL, then upsert new ones
Step 7: PageIndex      --> Build hierarchical tree index
Step 8: Cache update   --> Save content hash and HTTP metadata
```

If the URL hasn't changed (detected at Step 1 or Step 2), it short-circuits and skips steps 3-8.

After all URLs are processed, the pipeline runs two global steps:
1. **Orphan cleanup** (see Section 7) — removes data for URLs no longer in the active list
2. **BM25 index rebuild** (see Section 9) — reads all chunks from the vector store and builds a fresh global BM25 keyword index. This only runs when URLs were actually processed or removed.

The pipeline uses **Rich** for progress bars and formatted console output, so you can see what's happening in real-time.

---

## 14. The CLI Interface (`main.py`)

**What it does:** Provides the user-facing commands to operate the system.

Built with [Click](https://click.palletsprojects.com/) (a Python library for building CLIs) and [Rich](https://rich.readthedocs.io/) (for beautiful terminal output).

### Commands:

#### `ingest` - Load data into the system
```bash
# Ingest the default 5 CommBank URLs
python main.py ingest

# Ingest custom URLs
python main.py ingest -u "https://example.com/page1" -u "https://example.com/page2"

# Force re-processing (ignore cache)
python main.py ingest --force
```

After ingestion, the CLI shows a summary table with per-URL stats and an orphan cleanup report if any URLs were removed.

#### `query` - Ask questions
```bash
# Single question
python main.py query "What are CommBank's home loan rates?"

# Without LLM answer generation (shows raw context)
python main.py query "Compare credit cards" --no-generate

# Interactive mode (ask multiple questions)
python main.py query -i

# Limit results
python main.py query "loan features" -k 5

# Search only one URL
python main.py query "rates" --url-filter "https://www.commbank.com.au/home-loans.html"

# Disable BM25 keyword retrieval (vector + PageIndex only)
python main.py query "loan features" --no-bm25

# Bypass retrieval cache (force fresh retrieval)
python main.py query "loan features" --no-cache

# Output in search schema JSON format
python main.py query "What are the home loan options?" --json

# JSON output without LLM generation (retrieval only)
python main.py query "What are the home loan options?" --json --no-generate
```

The default table view shows both **Score** (raw vector similarity) and **Reranked** (combined after three-way merge) columns. When using `--json`, the output follows the search schema described in Section 12.

#### `status` - Check system state
```bash
python main.py status
```

Shows: vector store chunk count, BM25 index chunk count, PageIndex document count, embedding model details, LLM model, API key status, retrieval cache entries/hits, semantic threshold, and cache TTL.

#### `clear` - Remove data
```bash
python main.py clear --vectors --yes       # Clear vector store only
python main.py clear --cache --yes         # Clear content cache only
python main.py clear --query-cache --yes   # Clear retrieval/query cache only
python main.py clear --all --yes           # Clear everything
```

---

## 15. The REST API (`api.py`)

**What it does:** Provides an HTTP API for all RAG operations, allowing frontends, services, or tools like `curl` to interact with the system over the network.

**Technology:** [FastAPI](https://fastapi.tiangolo.com/) with [Uvicorn](https://www.uvicorn.org/) as the ASGI server.

### Starting the Server

```bash
# Direct run (simplest)
python api.py

# With hot reload (development)
uvicorn api:app --host 0.0.0.0 --port 8000 --reload

# Production (single worker — required for singleton state)
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1
```

Once running, the **Swagger UI** is available at `http://localhost:8000/docs` for interactive exploration.

### How It Differs from the CLI

| Aspect | CLI (`main.py`) | API (`api.py`) |
|---|---|---|
| Initialisation | Creates components per command, then destroys them | Singleton — creates once at startup, reuses for all requests |
| Ingestion | Synchronous (blocks until done) | Asynchronous (returns 202 immediately, runs in background) |
| Output | Rich terminal tables | JSON (Pydantic-validated) |
| Query format | Command-line arguments | JSON body (POST) or query parameters (GET) |
| Concurrency | Single-threaded | Event loop + thread pool via `asyncio.to_thread` |

### Singleton Lifecycle

The RAG system's heavy components (embedding model ~90MB, ChromaDB persistent client) are initialised **once** at startup via FastAPI's `lifespan` context manager:

```
Startup:
  RAGConfig.load() → IngestionPipeline(config) → RAGEngine(config, embedder, store, trees)

Running:
  All requests share the same pipeline and engine instances

Shutdown:
  pipeline.close()  → release SQLite + ChromaDB connections
```

After ingestion, `rebuild_engine()` re-creates the `RAGEngine` to pick up new PageIndex trees while keeping the existing pipeline (and its heavy embedding model) alive.

### Endpoints

#### `GET /health` — Health Check
```bash
curl http://localhost:8000/health
# {"status":"ok","timestamp":"...","vector_store_chunks":109,"uptime_seconds":42.1}
```

#### `GET /status` — Full System Status
```bash
curl http://localhost:8000/status
# Returns: chunks, PageIndex docs, indexed URLs, config, cache stats, uptime
```

#### `POST /query` — RAG Query (JSON body)
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What are the home loan rates?","top_k":5,"generate_answer":false}'
```

#### `GET /query` — RAG Query (query parameters)
```bash
curl "http://localhost:8000/query?question=What+credit+cards+are+available&top_k=3"
```

Both query endpoints accept a `use_bm25` parameter (default `true`) to enable or disable BM25 keyword retrieval. They return the same response: the answer, ranked sources with three-way scores, timing, cache hit status, and the full `search_schema` (Azure AI Search format).

#### `POST /ingest` — Trigger Ingestion (Background)
```bash
# Default URLs
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" -d '{}'

# Custom URLs with force re-processing
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"urls":["https://example.com/page1"],"force":true}'
```

Returns `202 Accepted` immediately. Returns `409 Conflict` if ingestion is already running.

#### `GET /ingest/status` — Check Ingestion Progress
```bash
curl http://localhost:8000/ingest/status
# {"status":"in_progress",...} or {"status":"completed","urls_processed":5,...}
```

#### `POST /clear` — Clear Data Stores
```bash
# Clear just the query cache
curl -X POST http://localhost:8000/clear \
  -H "Content-Type: application/json" -d '{"targets":["query_cache"]}'

# Clear everything
curl -X POST http://localhost:8000/clear \
  -H "Content-Type: application/json" -d '{"targets":["all"]}'
```

Valid targets: `vectors`, `cache`, `query_cache`, `all`.

### Thread Safety

All synchronous RAG operations (embedding, ChromaDB search, LLM calls) run in a thread pool via `asyncio.to_thread`, keeping the FastAPI event loop responsive. SQLite connections use `check_same_thread=False` to safely cross thread boundaries. An `asyncio.Lock` prevents concurrent ingestion.

---

## 16. Configuration (`config.py`)

**What it does:** Centralises all settings using Python dataclasses with sensible defaults.

**Why it's needed:** Having settings scattered across files makes a system hard to maintain. All tunable parameters live in one file.

### Key settings and what they mean:

#### Chunking Settings (`ChunkingConfig`)

| Setting | Default | What it controls |
|---|---|---|
| `chunk_size` | 512 tokens | How large each chunk is. Larger = more context per chunk but less precise search |
| `chunk_overlap` | 64 tokens | How much chunks overlap. Prevents losing info at chunk boundaries |
| `min_chunk_size` | 50 tokens | Minimum size to keep. Prevents tiny useless fragments |

#### Embedding Settings (`EmbeddingConfig`)

| Setting | Default | What it controls |
|---|---|---|
| `model_name` | `all-MiniLM-L6-v2` | The embedding model. Lightweight, runs on CPU |
| `dimension` | 384 | Vector size. Must match the embedding model's output |

#### Vector Store Settings (`VectorStoreConfig`)

| Setting | Default | What it controls |
|---|---|---|
| `distance_metric` | `cosine` | How similarity is measured. Cosine works best for text |
| `search_top_k` | 10 | How many results to return from vector search |

#### PageIndex Settings (`PageIndexConfig`)

| Setting | Default | What it controls |
|---|---|---|
| `llm_model` | `claude-sonnet-4-20250514` | Which Claude model to use for PageIndex and answer generation |

#### Retrieval Cache Settings (`RetrievalCacheConfig`)

| Setting | Default | What it controls |
|---|---|---|
| `similarity_threshold` | 0.92 | Cosine similarity cutoff for semantic cache hits. Higher = stricter matching |
| `ttl_seconds` | 86,400 (24h) | How long cache entries remain valid before expiring |
| `max_entries` | 1,000 | Maximum number of cached queries. Oldest entries evicted when exceeded |

#### BM25 Settings (`BM25Config`)

| Setting | Default | What it controls |
|---|---|---|
| `k1` | 1.5 | Term frequency saturation. Higher values give more weight to repeated query terms |
| `b` | 0.75 | Document length normalization. 0 = no penalty for long documents, 1 = full penalty |
| `enabled` | True | Master toggle for BM25 keyword retrieval |

### Environment variables:

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Enables LLM-powered PageIndex building, tree reasoning, and answer generation |

---

## 17. Key Concepts Glossary

| Term | Definition |
|---|---|
| **RAG** | Retrieval-Augmented Generation. A pattern where you search your data first, then give the results to an LLM to generate an answer. |
| **Embedding** | A numerical vector representation of text that captures semantic meaning. Similar texts produce similar vectors. |
| **Vector** | A list of numbers (e.g., 384 floats). In this context, the numerical representation of a text chunk. |
| **Vector Database** | A database optimised for storing vectors and finding the nearest ones to a query vector. ChromaDB in this project. |
| **Cosine Similarity** | A measure of how similar two vectors are based on the angle between them. 1.0 = identical, 0.0 = unrelated. |
| **Chunk** | A small piece of a document (typically 512 tokens) that can be individually embedded and searched. |
| **Token** | The basic unit an LLM processes. Roughly 3/4 of a word. "Commonwealth Bank" = ~3 tokens. |
| **PageIndex** | A hierarchical tree structure (like a table of contents) that enables structured navigation of documents. Inspired by VectifyAI's PageIndex framework. |
| **BM25** | Best Match 25. A term-frequency-based retrieval algorithm that scores documents by how many query terms they contain and how rare those terms are (IDF). Complements vector search for exact keyword matching. |
| **Hybrid Retrieval** | Using multiple retrieval methods (vector search + BM25 + PageIndex) and merging their results for better accuracy. This system uses a three-way hybrid. |
| **Reranking** | A second scoring pass that combines scores from all three retrievers to produce a final ranking. The `reranker_score` uses a weighted formula: 40% vector + 35% BM25 + 25% PageIndex. |
| **Max-Normalization** | Scaling raw scores so the highest score becomes 1.0. Used to bring BM25's unbounded scores into the [0, 1] range for merging with other retrievers. |
| **TF-IDF** | Term Frequency-Inverse Document Frequency. The core principle behind BM25: terms that appear often in a document but rarely across the corpus are the most informative. |
| **Conditional GET** | An HTTP technique where the server only sends content if it has changed, saving bandwidth. Uses ETag and Last-Modified headers. |
| **Content Hash** | A digital fingerprint of content. If two pages produce the same hash, their content is identical. Uses xxHash128. |
| **Content Fingerprint** | A store-level hash derived from all content hashes in the vector store. Used by the retrieval cache to detect when underlying data has changed. |
| **Retrieval Cache** | A query-side cache that stores previously computed retrieval results. Supports exact match (hash lookup) and semantic match (cosine similarity). |
| **TTL (Time-to-Live)** | How long a cached entry remains valid. After expiry, a fresh retrieval is forced. Default: 24 hours. |
| **LRU Eviction** | Least Recently Used eviction. When the cache is full, the entries that haven't been accessed for the longest time are removed first. |
| **Orphan Cleanup** | Automatic removal of chunks, cache entries, and PageIndex trees for URLs that are no longer in the active ingestion list. |
| **Search Schema** | A standardised JSON response format (modelled after Azure AI Search) with `@search.score`, `@search.rerankerScore`, and structured metadata. |
| **FastAPI** | A modern Python web framework for building REST APIs. Provides automatic request validation, OpenAPI docs, and async support. |
| **Lifespan** | FastAPI's startup/shutdown hook. Used to initialise heavy singletons (embedding model, ChromaDB) once at server start and clean up on exit. |
| **Upsert** | Insert-or-update. If the ID exists, update it; otherwise, insert a new record. |
| **trafilatura** | A Python library that extracts the main content from web pages, removing boilerplate (nav bars, footers, ads). |
| **sentence-transformers** | A Python library for generating text embeddings using pre-trained transformer models. |
| **tiktoken** | A tokenizer library that counts tokens the same way LLMs do, ensuring accurate chunk sizing. |

---

## 18. How to Run the System

### Prerequisites

- Python 3.12+
- pip (Python package manager)

### Setup

```bash
# Navigate to the project
cd "/path/to/RAG"

# Install dependencies
pip install -r requirements.txt

# (Optional) Set API key for LLM features
export ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### Quick Start — CLI

```bash
# Step 1: Ingest the CommBank URLs
python main.py ingest

# Step 2: Check status
python main.py status

# Step 3: Ask questions (rich table output)
python main.py query "What home loan options are available?"

# Step 4: Ask questions (JSON search schema output)
python main.py query "What home loan options are available?" --json

# Step 5: Interactive mode for multiple questions
python main.py query -i
```

### Quick Start — REST API

```bash
# Step 1: Start the server
python api.py

# Step 2: Trigger ingestion (in another terminal)
curl -X POST http://localhost:8000/ingest -H "Content-Type: application/json" -d '{}'

# Step 3: Check status
curl http://localhost:8000/status

# Step 4: Ask a question
curl "http://localhost:8000/query?question=What+home+loan+options+are+available"

# Step 5: Browse the interactive Swagger UI
open http://localhost:8000/docs
```

### What happens without an API key?

Everything works except:
- PageIndex uses heuristic (heading-based) building instead of LLM-generated summaries
- PageIndex query reasoning uses keyword matching instead of LLM reasoning
- Queries return raw retrieved context instead of a generated answer

The vector search, BM25 keyword retrieval, content caching, retrieval caching, orphan cleanup, chunking, embedding, and JSON schema output all work fully without an API key.

---

## Summary

This RAG system follows a clean, layered architecture:

1. **Scrape** web pages with structural extraction
2. **Process** into clean, structured Markdown documents
3. **Chunk** with token-aware splitting and section context preservation
4. **Embed** chunks into 384-dimensional vectors
5. **Store** in ChromaDB for persistent, searchable vector storage
6. **Index** with PageIndex hierarchical trees for structured navigation
7. **Index** with BM25 keyword index for exact term matching
8. **Cache content** hashes and HTTP metadata to skip unchanged URLs
9. **Clean up orphans** automatically when URLs are removed from the ingestion list
10. **Query** with three-way hybrid retrieval (vector + BM25 + PageIndex), weighted scoring (40/35/25), and LLM generation
11. **Cache queries** with two-tier retrieval cache (exact hash + semantic similarity)
12. **Output** in standardised search schema JSON format for API integration
13. **Serve** via FastAPI REST API with Swagger UI, background ingestion, and thread-safe concurrency

Each component is modular, testable, and replaceable. The system is designed to be production-ready with proper error handling, progress tracking, configurable settings, and two entry points: a CLI for terminal use and a REST API for service integration.
