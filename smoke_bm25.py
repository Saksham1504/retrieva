from retrieval.base import load_chunks
from retrieval.bm25 import BM25Retriever

chunks = load_chunks()
by_id = {c["chunk_id"]: c for c in chunks}
r = BM25Retriever(chunks)
for cid, score in r.retrieve("your test question here", k=5):
    print(f"{score:6.2f}  {cid}  {by_id[cid]['text'][:100]!r}")