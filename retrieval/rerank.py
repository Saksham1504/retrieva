"""Cross-encoder reranker.

A first-stage retriever (BM25, dense or hybrid) produces a shortlist of candidate chunks.
The cross-encoder then reads the query and each candidate TOGETHER and scores how well
the chunk answers the query, which is slower but more accurate than comparing separate
embeddings. The shortlist is reordered by that score and the top k are returned.

    retrieve(query, k) -> [(chunk_id, reranker_score), ...]   best first

The returned scores are cross-encoder scores, which is what the abstention gate will use.
"""
from retrieval.base import Retriever

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class RerankRetriever(Retriever):
    def __init__(self, first_stage, chunks, model_name=DEFAULT_MODEL, candidates=20, model=None):
        """first_stage: any retriever.  chunks: the chunk dicts (the reranker needs their text).
        candidates: how many first-stage results to rerank (should be >= the k you ask for)."""
        self.first_stage = first_stage
        self.text = {c["chunk_id"]: c["text"] for c in chunks}
        self.model_name = model_name
        self.candidates = candidates
        if model is None:
            from sentence_transformers import CrossEncoder   # slow import, so only when needed
            model = CrossEncoder(model_name)
        self.model = model

    def retrieve(self, query, k=5):
        shortlist = self.first_stage.retrieve(query, k=max(self.candidates, k))
        if not shortlist:
            return []
        ids = [chunk_id for chunk_id, _ in shortlist]
        pairs = [(query, self.text[chunk_id]) for chunk_id in ids]
        scores = self.model.predict(pairs, batch_size=32, show_progress_bar=False)
        ranked = sorted(zip(ids, (float(s) for s in scores)), key=lambda item: -item[1])
        return ranked[:k]
