import anthropic

from rag.store.document import Chunk

_MODEL = "claude-opus-4-8"

_SYSTEM = (
    "You are a helpful assistant. Answer the user's question using only the provided context. "
    "If the answer is not in the context, say so."
)


def _build_context(chunks: list[tuple[Chunk, float]]) -> str:
    parts = []
    for chunk, score in chunks:
        parts.append(f"[Source: {chunk.source}, chunk {chunk.chunk_index}, score={score:.3f}]\n{chunk.text}")
    return "\n\n---\n\n".join(parts)


class Generator:
    def __init__(self, model: str = _MODEL):
        self.client = anthropic.Anthropic()
        self.model = model

    def generate(self, query: str, chunks: list[tuple[Chunk, float]]) -> str:
        context = _build_context(chunks)
        user_message = f"Context:\n{context}\n\nQuestion: {query}"

        with self.client.messages.stream(
            model=self.model,
            max_tokens=1024,
            thinking={"type": "adaptive"},
            system=_SYSTEM,
            messages=[{"role": "user", "content": user_message}],
        ) as stream:
            return stream.get_final_message().content[-1].text
