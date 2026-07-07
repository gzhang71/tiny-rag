from abc import ABC, abstractmethod
from enum import Enum

import numpy as np

from rag.store.document import Chunk


class StoreBackend(str, Enum):
    FAISS = "faiss"    # in-process, in-memory — fastest, but vectors die with the process
    CHROMA = "chroma"  # local vector DB — persists to disk, or connects to a `chroma run` server


class BaseVectorStore(ABC):
    @abstractmethod
    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None: ...

    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[tuple[Chunk, float]]: ...

    @abstractmethod
    def chunks(self) -> list[Chunk]:
        """All stored chunks — used to build lexical (BM25) indexes over the corpus."""

    @abstractmethod
    def __len__(self) -> int: ...
