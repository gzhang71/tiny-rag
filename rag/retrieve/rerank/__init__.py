"""Post-fusion rerank stages, applied in this order when enabled:

1. MMR (`mmr.py`) — diversity-aware selection of the final top-k
2. Cross-encoder (`cross_encoder.py`) — joint (query, chunk) relevance scoring
"""
from rag.retrieve.rerank.cross_encoder import Reranker
from rag.retrieve.rerank.mmr import mmr_select

__all__ = ["Reranker", "mmr_select"]
