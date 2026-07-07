"""Maximal Marginal Relevance — diversity-aware selection after fusion.

Similarity ranking alone tends to fill the top-k with near-paraphrases of the
same passage (overlapping chunks make this worse). MMR greedily picks the
candidate maximising

    lambda * relevance(query, c)  -  (1 - lambda) * max_similarity(c, selected)

so each pick must add relevant information the already-selected chunks don't
carry. lambda=1 is pure relevance; lower values trade relevance for coverage.
"""
import numpy as np

from rag.store.document import Chunk


def mmr_select(
    query_embedding: np.ndarray,
    candidate_embeddings: np.ndarray,
    candidates: list[Chunk],
    top_k: int,
    lambda_: float = 0.7,
) -> list[tuple[Chunk, float]]:
    """Embeddings must be L2-normalised (dot product == cosine similarity).
    Returned scores are the MMR objective at selection time, so they decrease
    monotonically and reflect both relevance and novelty."""
    if not candidates:
        return []
    relevance = candidate_embeddings @ query_embedding
    selected: list[tuple[int, float]] = []
    remaining = set(range(len(candidates)))

    while remaining and len(selected) < top_k:
        if not selected:
            best = max(remaining, key=lambda i: relevance[i])
            score = float(relevance[best])
        else:
            chosen = candidate_embeddings[[i for i, _ in selected]]
            redundancy = candidate_embeddings @ chosen.T  # sim of every candidate to each pick
            def mmr(i: int) -> float:
                return lambda_ * relevance[i] - (1.0 - lambda_) * float(redundancy[i].max())
            best = max(remaining, key=mmr)
            score = mmr(best)
        selected.append((best, score))
        remaining.remove(best)

    return [(candidates[i], score) for i, score in selected]
