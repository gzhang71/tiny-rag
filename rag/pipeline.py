from rag.ingest.chunker import chunk_text
from rag.ingest.embedder import Embedder
from rag.ingest.loader import load_file, load_directory
from rag.retrieve.rerank import CrossEncoderReranker
from rag.retrieve.retriever import Channel, DEFAULT_CHANNELS, Retriever
from rag.generate.generator import Generator
from rag.store.base import BaseVectorStore, StoreBackend
from rag.store.vector_store import IndexType, VectorStore


class RAGPipeline:
    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 64,
        top_k: int = 5,
        backend: StoreBackend = StoreBackend.FAISS,
        # retrieval channels
        channels: tuple[Channel, ...] = DEFAULT_CHANNELS,
        mmr_lambda: float | None = None,
        rerank: bool = False,
        rrf_k: int = 60,
        # FAISS-specific
        index_type: IndexType = IndexType.FLAT,
        # Chroma-specific
        persist_dir: str | None = "./chroma_db",
        collection: str = "tiny_rag",
        chroma_host: str | None = None,
        chroma_port: int = 8000,
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.top_k = top_k

        self.embedder = Embedder()
        self.store: BaseVectorStore
        if backend == StoreBackend.CHROMA:
            # imported lazily so the faiss backend works without chromadb installed
            from rag.store.chroma_store import ChromaStore

            self.store = ChromaStore(
                persist_dir=persist_dir,
                collection=collection,
                host=chroma_host,
                port=chroma_port,
            )
        else:
            self.store = VectorStore(dim=self.embedder.dim, index_type=index_type)
        self.retriever = Retriever(
            self.embedder,
            self.store,
            channels=channels,
            rrf_k=rrf_k,
            mmr_lambda=mmr_lambda,
            reranker=CrossEncoderReranker() if rerank else None,
        )
        self.generator = Generator()

    def ingest_text(self, text: str, source: str = "inline") -> int:
        chunks = chunk_text(text, source=source, chunk_size=self.chunk_size, overlap=self.overlap)
        embeddings = self.embedder.embed([c.text for c in chunks])
        self.store.add(chunks, embeddings)
        return len(chunks)

    def ingest_file(self, path: str) -> int:
        return self.ingest_text(load_file(path), source=path)

    def ingest_directory(self, directory: str, glob: str = "**/*.txt") -> int:
        total = 0
        for source, text in load_directory(directory, glob=glob).items():
            total += self.ingest_text(text, source=source)
        return total

    def query(self, question: str) -> str:
        if len(self.store) == 0:
            raise RuntimeError("No documents ingested. Call ingest_text/ingest_file first.")
        chunks = self.retriever.retrieve(question, top_k=self.top_k)
        return self.generator.generate(question, chunks)
