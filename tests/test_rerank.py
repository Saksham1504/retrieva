"""Tests for retrieval/rerank.py with a fake first stage and a fake cross-encoder (no download).

Run from the rag_doc root:  python -m pytest tests/test_rerank.py
"""
import re

from retrieval.base import Retriever
from retrieval.rerank import RerankRetriever

CHUNKS = [
    {"chunk_id": "a", "text": "a weak entity is drawn as a double rectangle"},
    {"chunk_id": "b", "text": "a clustered index sets the physical order of rows"},
    {"chunk_id": "c", "text": "deadlock happens when transactions wait for each other"},
    {"chunk_id": "d", "text": "the double rectangle symbol marks a weak entity set"},
]


class FixedFirstStage(Retriever):
    """Returns a fixed (and deliberately poor) ordering and records the k it was asked for."""

    def __init__(self, ids):
        self.ids = ids
        self.asked_k = None

    def retrieve(self, query, k=5):
        self.asked_k = k
        return [(cid, 1.0) for cid in self.ids[:k]]


class FakeCrossEncoder:
    """Scores a (query, text) pair by how many words they share."""

    def predict(self, pairs, batch_size=32, show_progress_bar=False):
        def words(s):
            return set(re.findall(r"[a-z]+", s.lower()))
        return [float(len(words(q) & words(t))) for q, t in pairs]


def make(first_stage, candidates=20):
    return RerankRetriever(first_stage, CHUNKS, candidates=candidates, model=FakeCrossEncoder())


def test_reorders_the_shortlist_by_reranker_score():
    first = FixedFirstStage(["b", "c", "d", "a"])            # first stage puts the answers last
    result = make(first).retrieve("which symbol marks a weak entity", k=2)
    assert [cid for cid, _ in result] == ["d", "a"]


def test_returns_at_most_k_with_float_scores_best_first():
    result = make(FixedFirstStage(["a", "b", "c", "d"])).retrieve("weak entity double rectangle", k=3)
    assert len(result) == 3
    scores = [s for _, s in result]
    assert all(isinstance(s, float) for s in scores)
    assert scores == sorted(scores, reverse=True)


def test_asks_the_first_stage_for_the_full_candidate_pool():
    first = FixedFirstStage(["a", "b", "c", "d"])
    make(first, candidates=4).retrieve("anything", k=2)
    assert first.asked_k == 4                                # not just k


def test_empty_shortlist_returns_empty_list():
    assert make(FixedFirstStage([])).retrieve("anything", k=5) == []
