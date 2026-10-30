"""  it checks if the pdf is scanned or not 
    if the pdf contains photos it will mention photos:1 
    and tell if its scanned or not
"""

"""Corpus sanity checks. Run from the rag_doc root:

    python check_pdf.py

1. For every PDF in data/raw: does it have extractable text, or is it scanned?
2. In data/processed/chunks.jsonl: which lines repeat most (running headers/footers)?
"""
from collections import Counter
from pathlib import Path

import pymupdf

from retrieval.base import load_chunks


def check_pdfs(folder="data/raw"):
    print("== PDF text check ==")
    for path in sorted(Path(folder).rglob("*.pdf")):
        with pymupdf.open(path) as doc:
            n_pages = len(doc)
            empty = sum(1 for page in doc if len(page.get_text().strip()) < 20)
        status = "OK" if empty < n_pages * 0.1 else "MOSTLY EMPTY (scanned?)"
        print(f"{path.name}: {n_pages} pages, {empty} without text -> {status}")


def check_repeats(top=15):
    print("\n== Most repeated lines in chunks (possible headers/footers) ==")
    chunks = load_chunks()
    lines = Counter(
        line.strip()
        for c in chunks
        for line in c["text"].splitlines()
        if line.strip()
    )
    for line, n in lines.most_common(top):
        print(f"{n:4d}  {line[:80]!r}")


if __name__ == "__main__":
    check_pdfs()
    check_repeats()