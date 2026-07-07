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

### Options

| Flag | Default | Description |
|---|---|---|
| `--top-k` | 5 | Number of chunks to retrieve |
| `--chunk-size` | 512 | Characters per chunk |
| `--overlap` | 64 | Overlap between chunks |
| `--store` | `faiss` | Vector store: `faiss` (in-memory) or `chroma` (persistent DB) |
| `--persist-dir` | `./chroma_db` | Chroma DB directory (embedded mode) |
| `--chroma-host` / `--chroma-port` | — / 8000 | Connect to a running `chroma run` server |

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
```

## Architecture

```
Ingest:   load → chunk → embed
Retrieve: embed query → ANN search → top-k chunks
Generate: chunks + query → Claude → answer
```

**Vector store** — two backends behind a common interface (`BaseVectorStore`):
- `faiss` (default) — in-process, in-memory. `FLAT` gives exact cosine search; switch to `HNSW` for large corpora to get sub-linear query time at the cost of approximate results.
- `chroma` — a real vector DB that runs entirely on your machine: embedded and persisted to a local directory by default, or as a standalone server via `chroma run`. Uses cosine space and stable `source:chunk_index` ids, so re-ingesting a document updates it in place.

Both backends consume the same L2-normalised embeddings and return cosine-similarity scores, so results are comparable across backends.

**Embeddings** — `all-MiniLM-L6-v2` via sentence-transformers (runs locally, no API key needed).

**Generation** — `claude-opus-4-8` with adaptive thinking via the Anthropic SDK.
