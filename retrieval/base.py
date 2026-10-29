import json

DEFAULT_CHUNKS_PATH = "data/processed/chunks.jsonl"


def load_chunks(path=DEFAULT_CHUNKS_PATH):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class Retriever:
    def retrieve(self, query, k=5):
        """Return a list of (chunk_id, score) pairs, best first."""
        raise NotImplementedError


    