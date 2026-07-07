"""Post-fusion rerank stages. All stages implement the `RerankStage` ABC
(`base.py`) and are applied in this order when enabled:

1. MMR (`mmr.py`, `MMRReranker`) — diversity-aware selection of the final top-k
2. Cross-encoder (`cross_encoder.py`, `CrossEncoderReranker`) — joint
   (query, chunk) relevance scoring
"""
from rag.retrieve.rerank.base import RerankStage
from rag.retrieve.rerank.cross_encoder import CrossEncoderReranker
from rag.retrieve.rerank.mmr import MMRReranker, mmr_select

__all__ = ["RerankStage", "CrossEncoderReranker", "MMRReranker", "mmr_select"]
