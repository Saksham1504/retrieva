import json
from pathlib import Path
from rag.ingest.loaders import load_all

CHUNK_SIZE = 1200   # max characters per chunk (roughly 300 tokens)
OVERLAP = 150       # characters shared between neighbouring chunks


def find_break(text, start, hard_end):
    """Choose where to end a chunk: prefer a paragraph break, then a sentence
    end, then a line break, then a space. Never cut it below half size."""
    if hard_end >= len(text):
        return len(text)
    earliest = start + CHUNK_SIZE // 2
    for sep in ("\n\n", ". ", "\n", " "):
        i = text.rfind(sep, earliest, hard_end)
        if i != -1:
            return i + len(sep)
    return hard_end   # no good break found: cut hard


def chunk_text(text):
    """Return a list of (start, end) positions. Chunk = text[start:end]."""
    spans = []
    start = 0
    while start < len(text):
        end = find_break(text, start, start + CHUNK_SIZE)
        if text[start:end].strip():
            spans.append((start, end))
        if end >= len(text):
            break
        next_start = end - OVERLAP
        space = text.find(" ", next_start, end)   # start the next chunk at a word boundary
        if space != -1:
            next_start = space + 1
        start = max(next_start, start + 1)        # always move forward
    return spans


def build_chunks(docs):
    rows = []
    for d in docs:
        for i, (s, e) in enumerate(chunk_text(d["text"])):
            rows.append({
                "chunk_id": f"{d['doc_id']}_c{i:04d}",
                "doc_id": d["doc_id"],
                "source": d["source"],
                "text": d["text"][s:e],
                "char_start": s,
                "char_end": e,
            })
    ids = [r["chunk_id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate chunk IDs (two files with the same name?)"
    return rows


if __name__ == "__main__":
    rows = build_chunks(load_all("data/raw"))
    out = Path("data/processed/chunks.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if rows:
        lengths = sorted(len(r["text"]) for r in rows)
        print(f"{len(rows)} chunks written to {out}")
        print(f"length min/median/max: {lengths[0]} / {lengths[len(lengths)//2]} / {lengths[-1]}")
    else:
        print("No chunks. Is data/raw empty?")