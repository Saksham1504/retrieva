"""Tests for the generation package using fake LLMs, so no API key or network is needed.

Run from the rag_doc root:  python -m pytest tests/test_generation.py
"""
import json

import pytest

from generation.judges import (judge_correctness, judge_faithfulness, parse_json)
from generation.openai_llm import OpenAILLM
from generation.pipeline import (ABSTAIN_TEXT, NO_ANSWER_TOKEN, RagPipeline, build_input,
                                 extract_citations)
from retrieval.base import Retriever

CHUNKS = [
    {"chunk_id": "a", "text": "A weak entity is drawn as a double rectangle."},
    {"chunk_id": "b", "text": "A clustered index sets the physical order of rows."},
]


class FixedRetriever(Retriever):
    def __init__(self, results):
        self.results = results

    def retrieve(self, query, k=5):
        return self.results[:k]


class RecordingLLM:
    """Returns a fixed reply and remembers what it was asked."""

    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def __call__(self, instructions, user_input):
        self.calls.append((instructions, user_input))
        return self.reply


# ------------------------------------------------------------------- pipeline
def test_gate_refuses_without_calling_the_llm():
    llm = RecordingLLM("should never be used")
    pipeline = RagPipeline(FixedRetriever([("a", 0.5)]), CHUNKS, llm, abstain_below=1.0)
    result = pipeline.answer("anything")
    assert result["abstained"] and result["abstained_by"] == "gate"
    assert result["answer"] == ABSTAIN_TEXT
    assert llm.calls == []


def test_confident_retrieval_produces_an_answer_with_mapped_citations():
    llm = RecordingLLM("A weak entity is a double rectangle [1]. Indexes order rows [2][9].")
    pipeline = RagPipeline(FixedRetriever([("a", 5.0), ("b", 3.0)]), CHUNKS, llm, abstain_below=1.0)
    result = pipeline.answer("how is a weak entity drawn?")
    assert not result["abstained"]
    assert result["cited"] == ["a", "b"]              # [1] -> chunk a, [2] -> chunk b
    assert result["invalid_citations"] == [9]         # the model cited a source that does not exist


def test_model_can_decline_even_when_the_gate_passes():
    llm = RecordingLLM(NO_ANSWER_TOKEN)
    pipeline = RagPipeline(FixedRetriever([("a", 5.0)]), CHUNKS, llm, abstain_below=1.0)
    result = pipeline.answer("something unrelated")
    assert result["abstained"] and result["abstained_by"] == "model"
    assert result["answer"] == ABSTAIN_TEXT


def test_prompt_has_numbered_sources_and_the_question():
    text = build_input("How is a weak entity drawn?", CHUNKS)
    assert "[1] A weak entity is drawn as a double rectangle." in text
    assert "[2] A clustered index" in text
    assert text.rstrip().endswith("Question: How is a weak entity drawn?")


def test_extract_citations_ignores_out_of_range_numbers():
    assert extract_citations("fact [1][3]. more [2].", n_sources=3) == ([1, 2, 3], [])
    assert extract_citations("fact [0] and [7]", n_sources=3) == ([], [0, 7])


# --------------------------------------------------------------------- judges
def test_parse_json_handles_code_fences_and_stray_text():
    assert parse_json('```json\n{"verdict": "correct"}\n```') == {"verdict": "correct"}
    assert parse_json('Sure! {"verdict": "partial"} Hope that helps.') == {"verdict": "partial"}
    with pytest.raises(ValueError):
        parse_json("no json here")


def test_faithfulness_is_supported_claims_over_total_claims():
    reply = json.dumps({"claims": [
        {"claim": "A weak entity is a double rectangle", "supported": True},
        {"claim": "Weak entities are always optional", "supported": False},
        {"claim": "Indexes order rows", "supported": True},
        {"claim": "It uses a diamond", "supported": True},
    ]})
    result = judge_faithfulness(RecordingLLM(reply), "some answer [1]", CHUNKS)
    assert result["n_claims"] == 4 and result["n_supported"] == 3
    assert result["faithfulness"] == pytest.approx(0.75)


def test_faithfulness_with_no_claims_is_none():
    assert judge_faithfulness(RecordingLLM('{"claims": []}'), "ok", CHUNKS)["faithfulness"] is None


def test_correctness_maps_verdicts_to_scores():
    for verdict, score in [("correct", 1.0), ("partial", 0.5), ("incorrect", 0.0)]:
        result = judge_correctness(RecordingLLM(json.dumps({"verdict": verdict})), "q", "answer", "reference")
        assert result["score"] == score
    with pytest.raises(ValueError):
        judge_correctness(RecordingLLM('{"verdict": "maybe"}'), "q", "answer", "reference")


# ------------------------------------------------------------------ OpenAILLM
class FakeResponse:
    def __init__(self, text, status="completed"):
        self.output_text = text
        self.status = status


class FakeClient:
    """Mimics client.responses.create(...) and counts the calls."""

    def __init__(self, text="hello"):
        self.text = text
        self.calls = 0
        outer = self

        class Responses:
            def create(self, **kwargs):
                outer.calls += 1
                outer.last_kwargs = kwargs
                return FakeResponse(outer.text)

        self.responses = Responses()


def test_openai_llm_sends_instructions_input_and_token_limit(tmp_path):
    client = FakeClient("the answer")
    llm = OpenAILLM(model="test-model", cache_path=tmp_path / "cache.json", client=client)
    assert llm("be brief", "what is x?") == "the answer"
    assert client.last_kwargs["model"] == "test-model"
    assert client.last_kwargs["instructions"] == "be brief"
    assert client.last_kwargs["input"] == "what is x?"
    assert client.last_kwargs["max_output_tokens"] >= 1000


def test_identical_calls_are_served_from_the_cache_even_in_a_new_instance(tmp_path):
    path = tmp_path / "cache.json"
    first_client = FakeClient("cached text")
    OpenAILLM(model="m", cache_path=path, client=first_client)("i", "q")
    second_client = FakeClient("different text")
    llm = OpenAILLM(model="m", cache_path=path, client=second_client)
    assert llm("i", "q") == "cached text"
    assert second_client.calls == 0 and llm.calls == 0


def test_two_instances_sharing_a_cache_file_do_not_overwrite_each_other(tmp_path):
    path = tmp_path / "cache.json"
    generator = OpenAILLM(model="m", cache_path=path, client=FakeClient("generated"))
    judge = OpenAILLM(model="m", cache_path=path, client=FakeClient("judged"))
    generator("i", "question one")
    judge("i", "question two")                          # written after the generator, from its own copy
    fresh = OpenAILLM(model="m", cache_path=path, client=FakeClient("never used"))
    assert fresh("i", "question one") == "generated"
    assert fresh("i", "question two") == "judged"


def test_empty_model_output_raises_a_helpful_error(tmp_path):
    llm = OpenAILLM(model="m", cache_path=tmp_path / "cache.json", client=FakeClient(""))
    with pytest.raises(RuntimeError, match="max_output_tokens"):
        llm("i", "q")
