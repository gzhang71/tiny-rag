"""Retrieval tunnels — the parallel channels a query runs through.

Every tunnel inherits `RetrieveTunnel` (`base.py`): `search(query, top_k)`
returning ranked (chunk, score) pairs, and `__len__` reporting how many chunks
it has indexed. `DenseTunnel` searches the vector store; the others are
corpus indexes built from the store's chunks.
"""
from rag.retrieve.retrieve_tunnel.base import RetrieveTunnel
from rag.retrieve.retrieve_tunnel.bm25 import BM25Tunnel
from rag.retrieve.retrieve_tunnel.dense import DenseTunnel
from rag.retrieve.retrieve_tunnel.lexical import LexicalTunnel
from rag.retrieve.retrieve_tunnel.ner import EntityTunnel

__all__ = ["RetrieveTunnel", "DenseTunnel", "BM25Tunnel", "LexicalTunnel", "EntityTunnel"]
