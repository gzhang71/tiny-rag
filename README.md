# tiny_rag

A minimal RAG (Retrieval-Augmented Generation) system built with FAISS/Chroma, sentence-transformers, and Claude.

## Installation

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-...
```

## Usage

```bash
# Ask a question over inline text
python main.py --text "Alice loves cats. Bob loves dogs." "Who loves cats?"

# Ask a question over a single file
python main.py --file path/to/doc.txt "What is the main topic?"

# Ask a question over a directory of .txt files
python main.py --dir path/to/docs/ "Summarize the key points"
```

### Persistent vector DB (Chroma)

With `--store chroma`, embeddings are stored in a local Chroma database (default `./chroma_db`), so you ingest once and query as many times as you like:

```bash
# ingest once (re-running upserts — no duplicates)
python main.py --store chroma --dir path/to/docs/ "Summarize the key points"

# later: query without re-ingesting
python main.py --store chroma "What are the key risks mentioned?"
```

To run Chroma as a standalone server on your Mac instead of embedded mode:

```bash
chroma run --path ./chroma_data          # in one terminal
python main.py --store chroma --chroma-host localhost "Your question"
```

### Retrieval channels

Retrieval runs through several parallel channels whose ranked lists are merged with Reciprocal Rank Fusion (RRF):

| Channel | Kind | What it catches |
|---|---|---|
| `dense` | semantic | paraphrases, related concepts (vector similarity) |
| `bm25` | sparse lexical | keyword relevance (Okapi BM25) |
| `lexical` | exact match | verbatim phrases from the query |
| `entity` | NER | shared named entities: ticket ids, codes, dates, names, URLs |

All four are on by default; pick a subset with `--channels`. Two optional post-fusion stages:

- `--mmr [LAMBDA]` — MMR diversity selection: avoids filling the top-k with near-duplicate chunks; lambda in [0,1] trades relevance (1.0) against diversity (default 0.7).
- `--rerank` — re-score candidates with a local cross-encoder (most accurate, a bit slower).

```bash
python main.py --channels dense,bm25 --mmr --rerank --file doc.txt "What changed in v2.3?"
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--top-k` | 5 | Number of chunks to retrieve |
| `--chunk-size` | 512 | Characters per chunk |
| `--overlap` | 64 | Overlap between chunks |
| `--store` | `faiss` | Vector store: `faiss` (in-memory) or `chroma` (persistent DB) |
| `--persist-dir` | `./chroma_db` | Chroma DB directory (embedded mode) |
| `--chroma-host` / `--chroma-port` | — / 8000 | Connect to a running `chroma run` server |
| `--channels` | all four | Comma-separated retrieval channels to fuse |
| `--mmr [LAMBDA]` | off | MMR diversity selection after fusion (lambda default 0.7) |
| `--rerank` | off | Cross-encoder rerank stage after fusion/MMR |

## Python API

```python
from rag import RAGPipeline, IndexType

# Default: exact flat index, good for small corpora
pipeline = RAGPipeline()
pipeline.ingest_file("doc.txt")
print(pipeline.query("What is this about?"))

# HNSW: approximate index, sub-linear search for large corpora (10k+ chunks)
pipeline = RAGPipeline(index_type=IndexType.HNSW)
pipeline.ingest_directory("./docs/")
print(pipeline.query("What are the key themes?"))

# Chroma: persistent vector DB — survives restarts, ingest once / query forever
from rag import StoreBackend
pipeline = RAGPipeline(backend=StoreBackend.CHROMA, persist_dir="./chroma_db")
pipeline.ingest_directory("./docs/")   # only needed the first time
print(pipeline.query("What are the key themes?"))

# choose retrieval channels and add cross-encoder reranking
from rag import Channel
pipeline = RAGPipeline(channels=(Channel.DENSE, Channel.BM25), rerank=True)
```

## Architecture

```
Ingest:   load → chunk → embed
Retrieve: query → [dense | bm25 | lexical | entity] → RRF fusion → (MMR) → (rerank) → top-k chunks
Generate: chunks + query → Claude → answer
```

**Vector store** — two backends behind a common interface (`BaseVectorStore`):
- `faiss` (default) — in-process, in-memory. `FLAT` gives exact cosine search; switch to `HNSW` for large corpora to get sub-linear query time at the cost of approximate results.
- `chroma` — a real vector DB that runs entirely on your machine: embedded and persisted to a local directory by default, or as a standalone server via `chroma run`. Uses cosine space and stable `source:chunk_index` ids, so re-ingesting a document updates it in place.

Both backends consume the same L2-normalised embeddings and return cosine-similarity scores, so results are comparable across backends.

**Retrieval** — multi-tunnel: dense vectors for meaning, BM25 for keywords, exact phrase match for verbatim quotes, and entity overlap (rule-based NER) for ids/names/dates. Ranked lists are merged with Reciprocal Rank Fusion, which is rank-based so the tunnels' different score scales never need calibrating. Optional Maximal Marginal Relevance selection then picks a top-k that covers different aspects instead of near-duplicates, and an optional cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`, local) re-scores for the final ordering.

**Embeddings** — `all-MiniLM-L6-v2` via sentence-transformers (runs locally, no API key needed).

**Generation** — `claude-opus-4-8` with adaptive thinking via the Anthropic SDK.
