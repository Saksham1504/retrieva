"""Tests for retrieval/dense.py using a tiny fake encoder, so nothing is downloaded.

Run from the rag_doc root:  python -m pytest tests/test_dense.py
"""
import re
import zlib

import numpy as np

from retrieval.dense import DenseRetriever, cache_path


class FakeModel:
    """Hashes words into a small vector: texts that share words get similar vectors."""

    def __init__(self, dim=64):
        self.dim = dim
        self.calls = 0

    def encode(self, texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False):
        self.calls += 1
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                out[i, zlib.crc32(word.encode()) % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-9)


CHUNKS = [
    {"chunk_id": "a", "text": "a weak entity is drawn as a double rectangle in an er diagram"},
    {"chunk_id": "b", "text": "a clustered index establishes the physical sorting order of rows"},
    {"chunk_id": "c", "text": "deadlock occurs when transactions wait for each other to release locks"},
]


def make(tmp_path, chunks=CHUNKS, model=None):
    model = model or FakeModel()
    return DenseRetriever(chunks, model_name="fake-model", cache_dir=tmp_path,
                          model=model, query_prefix=""), model


def test_most_similar_chunk_is_first(tmp_path):
    retriever, _ = make(tmp_path)
    results = retriever.retrieve("which shape draws a weak entity", k=3)
    assert results[0][0] == "a"
    assert all(isinstance(cid, str) and isinstance(score, float) for cid, score in results)
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)


def test_k_larger_than_corpus_returns_everything(tmp_path):
    retriever, _ = make(tmp_path)
    assert len(retriever.retrieve("anything", k=10)) == 3


def test_vectors_are_cached_and_reused(tmp_path):
    _, first_model = make(tmp_path)
    assert first_model.calls == 1                      # embedded all chunks once
    assert cache_path(CHUNKS, "fake-model", tmp_path).exists()

    _, second_model = make(tmp_path)
    assert second_model.calls == 0                     # loaded from disk, no re-embedding


def test_cache_is_not_reused_when_a_chunk_changes(tmp_path):
    make(tmp_path)
    changed = [dict(c) for c in CHUNKS]
    changed[1]["text"] = "completely different text about hashing buckets"
    _, model = make(tmp_path, chunks=changed)
    assert model.calls == 1                            # re-embedded because the text changed
