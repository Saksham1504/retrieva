"""Hybrid retriever: Reciprocal Rank Fusion (RRF) over several retrievers.

Each retriever returns its own ranking. A chunk's fused score is the sum, over the
retrievers that returned it, of 1 / (rrf_k + rank). Only ranks are used, so scores from
different retrievers (BM25 scores vs cosine similarities) never need to be compared.

    retrieve(query, k) -> [(chunk_id, rrf_score), ...]   best first

Note: the returned scores are RRF scores (small numbers), not similarities, so they are
not suitable for deciding when to abstain.
"""
from collections import defaultdict

from retrieval.base import Retriever


class HybridRetriever(Retriever):
    def __init__(self, retrievers, rrf_k=60, pool=20):
        """retrievers: retrievers to fuse.  rrf_k: damping constant (60 is the usual default).
        pool: how many results to take from each retriever before fusing (deeper than k)."""
        self.retrievers = list(retrievers)
        self.rrf_k = rrf_k
        self.pool = pool

    def retrieve(self, query, k=5):
        fused = defaultdict(float)
        for retriever in self.retrievers:
            for rank, (chunk_id, _score) in enumerate(retriever.retrieve(query, k=self.pool), start=1):
                fused[chunk_id] += 1.0 / (self.rrf_k + rank)
        ranked = sorted(fused.items(), key=lambda item: (-item[1], item[0]))   # ties: stable by chunk id
        return ranked[:k]
