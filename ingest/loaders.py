from pathlib import Path
import re


def clean(text):
    text = re.sub(r"-\n(?=[a-z])", "", text)   # fix "inter-\nnational" -> "international"
    text = re.sub(r"[ \t]+", " ", text)        # many spaces/tabs -> one space
    text = re.sub(r"\n{3,}", "\n\n", text)     # 3+ blank lines -> one blank line
    return text.strip()


def load_file(path):
    path = Path(path)
    if path.suffix == ".pdf":
        import pymupdf
        pages = [page.get_text() for page in pymupdf.open(path)]
        return clean("\n\n".join(pages))
    elif path.suffix in (".md", ".txt"):
        return clean(path.read_text(encoding="utf-8"))
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")


def load_all(folder):
    docs = []
    for path in sorted(Path(folder).rglob("*")):
        if path.suffix in (".pdf", ".md", ".txt"):
            docs.append({
                "doc_id": path.stem,
                "source": str(path),
                "text": load_file(path),
            })
    return docs


if __name__ == "__main__":
    docs = load_all("data/raw")
    print(f"{len(docs)} documents loaded")
    for d in docs[:3]:
        print(f"\n--- {d['doc_id']} ({len(d['text'])} characters)")
        print(d["text"][:500])