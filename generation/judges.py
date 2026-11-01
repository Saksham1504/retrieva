"""LLM judges for generated answers.

faithfulness: split the answer into factual claims and check each against the retrieved
              sources. Score = supported claims / total claims. This is hallucination control.
correctness:  compare the answer with the golden set's reference answer.
              1.0 = correct, 0.5 = partial, 0.0 = incorrect. This catches answers that are
              faithful but do not actually answer the question.

Each judge takes llm(instructions, user_input) -> text, like the pipeline does.
"""
import json
import re

from generation.pipeline import format_sources

FAITHFULNESS_INSTRUCTIONS = """You check whether an answer is supported by its numbered sources.
Split the answer into atomic factual claims (ignore citation markers like [1]).
For each claim, decide whether the sources explicitly support it. A claim that needs outside
knowledge, or that the sources contradict, is NOT supported.
Return ONLY JSON in exactly this form, with no other text:
{"claims": [{"claim": "...", "supported": true}]}"""

CORRECTNESS_INSTRUCTIONS = """You grade an answer against a reference answer.
Verdict "correct": the answer contains the key information of the reference answer.
Verdict "partial": it contains some of it but misses important parts.
Verdict "incorrect": the key information is missing or contradicts the reference.
Return ONLY JSON in exactly this form, with no other text:
{"verdict": "correct"}"""

VERDICT_SCORES = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}


def parse_json(text):
    """Pull the JSON object out of a judge reply, tolerating code fences and stray text."""
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"no JSON object in judge reply: {text[:80]!r}")
    return json.loads(text[start:end + 1])


def judge_faithfulness(llm, answer, chunks):
    """chunks: the chunk dicts the answer was generated from, in the order they were numbered."""
    reply = llm(FAITHFULNESS_INSTRUCTIONS, f"Sources:\n{format_sources(chunks)}\n\nAnswer:\n{answer}")
    claims = parse_json(reply)["claims"]
    supported = sum(bool(c.get("supported")) for c in claims)
    return {
        "n_claims": len(claims),
        "n_supported": supported,
        "faithfulness": supported / len(claims) if claims else None,
        "claims": claims,
    }


def judge_correctness(llm, question, answer, reference):
    reply = llm(CORRECTNESS_INSTRUCTIONS,
                f"Question: {question}\n\nReference answer:\n{reference}\n\nAnswer to grade:\n{answer}")
    verdict = str(parse_json(reply)["verdict"]).strip().lower()
    if verdict not in VERDICT_SCORES:
        raise ValueError(f"unexpected verdict {verdict!r}")
    return {"verdict": verdict, "score": VERDICT_SCORES[verdict]}
