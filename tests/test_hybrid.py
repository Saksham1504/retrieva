"""Tests for retrieval/hybrid.py using fake retrievers with fixed rankings.

Run from the rag_doc root:  python -m pytest tests/test_hybrid.py
"""
import pytest

from retrieval.base import Retriever
from retrieval.hybrid import HybridRetriever


class Fixed(Retriever):
    """Always returns the same ranking (scores are made up and ignored by RRF)."""

    def __init__(self, ids):
        self.ids = ids

    def retrieve(self, query, k=5):
        return [(cid, 100.0 - i) for i, cid in enumerate(self.ids[:k])]


def test_chunk_ranked_well_by_both_beats_chunk_ranked_first_by_one():
    bm25 = Fixed(["a", "b", "c"])
    dense = Fixed(["x", "b", "y"])
    top = HybridRetriever([bm25, dense]).retrieve("q", k=3)
    assert top[0][0] == "b"                       # rank 2 in both beats rank 1 in only one


def test_scores_are_the_sum_of_reciprocal_ranks():
    hybrid = HybridRetriever([Fixed(["a", "b"]), Fixed(["b", "a"])], rrf_k=60)
    scores = dict(hybrid.retrieve("q", k=2))
    expected = 1 / 61 + 1 / 62
    assert scores["a"] == pytest.approx(expected)
    assert scores["b"] == pytest.approx(expected)


def test_returns_at_most_k_and_sorted_best_first():
    hybrid = HybridRetriever([Fixed(["a", "b", "c", "d"]), Fixed(["d", "c", "b", "a"])])
    result = hybrid.retrieve("q", k=2)
    assert len(result) == 2
    assert [s for _, s in result] == sorted((s for _, s in result), reverse=True)


def test_pool_limits_how_deep_each_retriever_is_read():
    deep = Fixed(["a", "b", "c", "d", "e", "f"])
    hybrid = HybridRetriever([deep], pool=2)
    assert [cid for cid, _ in hybrid.retrieve("q", k=5)] == ["a", "b"]


def test_chunk_found_by_only_one_retriever_is_still_returned():
    hybrid = HybridRetriever([Fixed(["a"]), Fixed(["z"])])
    assert {cid for cid, _ in hybrid.retrieve("q", k=5)} == {"a", "z"}
