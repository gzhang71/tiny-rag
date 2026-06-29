from rag.ingest.embedder import Embedder
from rag.store.document import Chunk
from rag.store.vector_store import VectorStore


class Retriever:
    def __init__(self, embedder: Embedder, store: VectorStore):
        self.embedder = embedder
        self.store = store

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        q_emb = self.embedder.embed_one(query)
        return self.store.search(q_emb, top_k=top_k)
