import random
from collections import defaultdict
from retrieval.base import load_chunks

random.seed(0)
by_doc = defaultdict(list)
for c in load_chunks():
    by_doc[c["doc_id"]].append(c)
print(f"{len(by_doc)} documents, {sum(len(v) for v in by_doc.values())} chunks\n")
picked = [c for cs in by_doc.values() for c in random.sample(cs, min(2, len(cs)))][:10]
for c in picked:
    print(f"[{c['chunk_id']}]\n{c['text']}\n")