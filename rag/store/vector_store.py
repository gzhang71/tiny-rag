import numpy as np
import faiss

from .document import Chunk


class VectorStore:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors = cosine
        self._chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        self.index.add(embeddings.astype(np.float32))
        self._chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[tuple[Chunk, float]]:
        q = query_embedding.astype(np.float32).reshape(1, -1)
        scores, indices = self.index.search(q, min(top_k, len(self._chunks)))
        return [(self._chunks[i], float(scores[0][rank])) for rank, i in enumerate(indices[0]) if i >= 0]

    def __len__(self) -> int:
        return len(self._chunks)
