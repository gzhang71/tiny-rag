from rag.store.document import Chunk


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[Chunk]:
    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(Chunk(text=text[start:end], source=source, chunk_index=idx))
        if end == len(text):
            break
        start += chunk_size - overlap
        idx += 1
    return chunks
