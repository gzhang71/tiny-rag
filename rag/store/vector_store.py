from enum import Enum

import faiss
import numpy as np

from rag.store.base import BaseVectorStore
from rag.store.document import Chunk


class IndexType(str, Enum):
    FLAT = "flat"    # exact cosine — best for < ~10k chunks
    HNSW = "hnsw"   # approximate, sub-linear search — best for > ~10k chunks


class VectorStore(BaseVectorStore):
    def __init__(
        self,
        dim: int,
        index_type: IndexType = IndexType.FLAT,
        # HNSW-specific knobs
        hnsw_m: int = 32,              # edges per node per layer; higher = better recall, more RAM
        hnsw_ef_construction: int = 200,  # beam width during build; higher = better recall, slower build
        hnsw_ef_search: int = 64,      # beam width during query; can be tuned after build
    ):
        self.index_type = index_type
        self._chunks: list[Chunk] = []

        if index_type == IndexType.FLAT:
            # IndexFlatIP: exact inner-product on L2-normalised vectors == cosine similarity
            self.index = faiss.IndexFlatIP(dim)
        else:
            # IndexHNSWFlat with inner-product metric; vectors must be L2-normalised before add/search
            self.index = faiss.IndexHNSWFlat(dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efConstruction = hnsw_ef_construction
            self.index.hnsw.efSearch = hnsw_ef_search

    def set_ef_search(self, ef_search: int) -> None:
        """Tune recall vs. latency at query time (HNSW only)."""
        if self.index_type == IndexType.HNSW:
            self.index.hnsw.efSearch = ef_search

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        self.index.add(embeddings.astype(np.float32))
        self._chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[tuple[Chunk, float]]:
        q = query_embedding.astype(np.float32).reshape(1, -1)
        scores, indices = self.index.search(q, min(top_k, len(self._chunks)))
        return [
            (self._chunks[i], float(scores[0][rank]))
            for rank, i in enumerate(indices[0])
            if i >= 0
        ]

    def __len__(self) -> int:
        return len(self._chunks)
