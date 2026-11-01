"""Grounded answer generation with an abstention gate.

    question -> retriever (top-k chunks with scores)
             -> gate: refuse if the top-1 score is below `abstain_below`
             -> LLM answers using ONLY the numbered sources and cites them like [1]
             -> if the LLM says the sources are insufficient, refuse too

The LLM is any function llm(instructions, user_input) -> text, so the pipeline can be
tested without an API key (see tests/test_generation.py).
"""
import re

ABSTAIN_TEXT = "I couldn't find a confident answer to that in the documents."
NO_ANSWER_TOKEN = "INSUFFICIENT_CONTEXT"

INSTRUCTIONS = f"""You answer questions using ONLY the numbered sources provided.
Rules:
- Use only facts stated in the sources. Do not use outside knowledge.
- After each sentence that states a fact, cite the source numbers like [1] or [2][3].
- If the sources do not contain the answer, reply with exactly {NO_ANSWER_TOKEN} and nothing else.
- Be concise: at most 5 sentences."""


def format_sources(chunks):
    """Number the chunks [1], [2], ... so the model can cite them."""
    return "\n\n".join(f"[{i}] {c['text'].strip()}" for i, c in enumerate(chunks, start=1))


def build_input(question, chunks):
    return f"Sources:\n{format_sources(chunks)}\n\nQuestion: {question}"


def extract_citations(answer, n_sources):
    """Return (valid_numbers, invalid_numbers) for citations like [2] found in the answer."""
    numbers = [int(m) for m in re.findall(r"\[(\d+)\]", answer)]
    valid = sorted({n for n in numbers if 1 <= n <= n_sources})
    invalid = sorted({n for n in numbers if not 1 <= n <= n_sources})
    return valid, invalid


class RagPipeline:
    def __init__(self, retriever, chunks, llm, abstain_below, k=5):
        """retriever: must return scores where higher = more confident (use a reranker config).
        abstain_below: refuse when the top-1 score is below this value."""
        self.retriever = retriever
        self.by_id = {c["chunk_id"]: c for c in chunks}
        self.llm = llm
        self.abstain_below = abstain_below
        self.k = k

    def answer(self, question):
        results = self.retriever.retrieve(question, k=self.k)
        top1 = float(results[0][1]) if results else float("-inf")
        base = {
            "question": question,
            "top1_score": top1,
            "sources": [{"chunk_id": cid, "score": float(score)} for cid, score in results],
        }

        if not results or top1 < self.abstain_below:
            return {**base, "answer": ABSTAIN_TEXT, "abstained": True, "abstained_by": "gate",
                    "cited": [], "invalid_citations": []}

        chunks = [self.by_id[cid] for cid, _ in results]
        raw = self.llm(INSTRUCTIONS, build_input(question, chunks)).strip()

        if NO_ANSWER_TOKEN in raw:
            return {**base, "answer": ABSTAIN_TEXT, "abstained": True, "abstained_by": "model",
                    "cited": [], "invalid_citations": []}

        valid, invalid = extract_citations(raw, len(chunks))
        return {**base, "answer": raw, "abstained": False, "abstained_by": None,
                "cited": [results[n - 1][0] for n in valid], "invalid_citations": invalid}
