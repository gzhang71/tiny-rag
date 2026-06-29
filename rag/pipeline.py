from .ingest.chunker import chunk_text
from .ingest.embedder import Embedder
from .ingest.loader import load_file, load_directory
from .retrieve.retriever import Retriever
from .generate.generator import Generator
from .store.vector_store import VectorStore


class RAGPipeline:
    def __init__(self, chunk_size: int = 512, overlap: int = 64, top_k: int = 5):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.top_k = top_k

        self.embedder = Embedder()
        self.store = VectorStore(dim=self.embedder.dim)
        self.retriever = Retriever(self.embedder, self.store)
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
