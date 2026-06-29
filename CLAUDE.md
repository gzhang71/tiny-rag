# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
pip3 install -r requirements.txt
export ANTHROPIC_API_KEY=sk-...
```

## Running

```bash
python3 main.py --text "Alice loves cats." "Who loves cats?"
python3 main.py --file path/to/doc.txt "What is the main topic?"
python3 main.py --dir path/to/docs/ "Summarize the key points"
```

## Architecture

`RAGPipeline` (in `rag/pipeline.py`) is the single public entry point. It wires together four independent stages:

1. **Ingest** (`rag/ingest/`) — `loader.py` reads files/directories; `chunker.py` splits text into fixed-size overlapping chunks (`Chunk` dataclass from `rag/store/document.py`); `embedder.py` wraps `sentence-transformers` (`all-MiniLM-L6-v2`) to produce L2-normalised float32 vectors.

2. **Store** (`rag/store/`) — `VectorStore` wraps a FAISS index. Two index types selectable via `IndexType` enum: `FLAT` (exact cosine, `IndexFlatIP`, default) and `HNSW` (approximate, `IndexHNSWFlat` with inner-product metric). Both rely on L2-normalised vectors so inner product == cosine similarity. Switch to `HNSW` at ~10k+ chunks.

3. **Retrieve** (`rag/retrieve/retriever.py`) — embeds the query, calls `VectorStore.search`, returns `list[tuple[Chunk, float]]` (chunk + cosine score).

4. **Generate** (`rag/generate/generator.py`) — passes retrieved chunks as context to Claude (`claude-opus-4-8`) via the Anthropic SDK with adaptive thinking and streaming.

All imports are absolute from the `rag` package root (e.g. `from rag.store.document import Chunk`). The project root must be on `sys.path` (running `python3 main.py` from `tiny_rag/` handles this automatically).

## Key knobs

| Parameter | Where | Default |
|---|---|---|
| `index_type` | `RAGPipeline(index_type=IndexType.HNSW)` | `FLAT` |
| `hnsw_m` | `VectorStore(hnsw_m=32)` | 32 |
| `hnsw_ef_construction` | `VectorStore(hnsw_ef_construction=200)` | 200 |
| `hnsw_ef_search` | `VectorStore(hnsw_ef_search=64)` or `store.set_ef_search(n)` | 64 |
| `chunk_size` / `overlap` | `RAGPipeline(chunk_size=512, overlap=64)` | 512 / 64 |
| `top_k` | `RAGPipeline(top_k=5)` | 5 |
