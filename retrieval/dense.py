"""Dense (embedding) retriever.

Embeds every chunk once with a sentence-transformers model, caches the vectors on disk,
and ranks chunks by cosine similarity to the query. Same interface as BM25Retriever:
    retrieve(query, k) -> [(chunk_id, score), ...]   best first
"""
import hashlib
import re
from pathlib import Path

import numpy as np

from retrieval.base import Retriever

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
# bge models work best when a short query carries this instruction; passages get none.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def cache_path(chunks, model_name, cache_dir):
    """The cache file name changes whenever the model or any chunk text changes,
    so stale vectors are never reused."""
    h = hashlib.sha256(model_name.encode("utf-8"))
    for c in chunks:
        h.update(c["chunk_id"].encode("utf-8"))
        h.update(c["text"].encode("utf-8"))
    slug = re.sub(r"[^A-Za-z0-9]+", "-", model_name).strip("-")
    return Path(cache_dir) / f"embeddings_{slug}_{h.hexdigest()[:12]}.npy"


class DenseRetriever(Retriever):
    def __init__(self, chunks, model_name=DEFAULT_MODEL, cache_dir="data/processed",
                 model=None, query_prefix=QUERY_PREFIX):
        self.ids = [c["chunk_id"] for c in chunks]
        self.model_name = model_name
        self.query_prefix = query_prefix
        if model is None:
            from sentence_transformers import SentenceTransformer   # slow import, so only when needed
            model = SentenceTransformer(model_name)
        self.model = model
        self.embeddings = self._load_or_embed(chunks, cache_dir)

    def _load_or_embed(self, chunks, cache_dir):
        path = cache_path(chunks, self.model_name, cache_dir)
        if path.exists():
            cached = np.load(path)
            if cached.shape[0] == len(chunks):
                return cached
        texts = [c["text"] for c in chunks]
        vectors = self.model.encode(texts, batch_size=32, normalize_embeddings=True,
                                    show_progress_bar=True)
        vectors = np.asarray(vectors, dtype=np.float32)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, vectors)
        return vectors

    def retrieve(self, query, k=5):
        q = self.model.encode([self.query_prefix + query], normalize_embeddings=True,
                              show_progress_bar=False)
        scores = self.embeddings @ np.asarray(q, dtype=np.float32)[0]   # cosine: vectors are unit length
        top = np.argsort(-scores)[:k]
        return [(self.ids[i], float(scores[i])) for i in top]
