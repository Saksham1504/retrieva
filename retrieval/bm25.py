import re
from rank_bm25 import BM25Okapi
from retrieval.base import Retriever


def tokenize(text):
    return re.findall(r"\w+", text.lower())


class BM25Retriever(Retriever):
    def __init__(self, chunks):
        self.ids = [c["chunk_id"] for c in chunks]
        self.index = BM25Okapi([tokenize(c["text"]) for c in chunks])

    def retrieve(self, query, k=5):
        scores = self.index.get_scores(tokenize(query))
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.ids[i], float(scores[i])) for i in top]