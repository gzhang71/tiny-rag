# tiny_rag

A minimal RAG (Retrieval-Augmented Generation) system built with FAISS, sentence-transformers, and Claude.

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

### Options

| Flag | Default | Description |
|---|---|---|
| `--top-k` | 5 | Number of chunks to retrieve |
| `--chunk-size` | 512 | Characters per chunk |
| `--overlap` | 64 | Overlap between chunks |

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
```

## Architecture

```
Ingest:   load → chunk → embed
Retrieve: embed query → ANN search → top-k chunks
Generate: chunks + query → Claude → answer
```

**Vector index** — defaults to exact cosine search (`FLAT`). Switch to `HNSW` for large corpora to get sub-linear query time at the cost of approximate results. Both use L2-normalised vectors so inner product equals cosine similarity.

**Embeddings** — `all-MiniLM-L6-v2` via sentence-transformers (runs locally, no API key needed).

**Generation** — `claude-opus-4-8` with adaptive thinking via the Anthropic SDK.
