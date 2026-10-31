"""Unit tests for eval/run_eval.py's scoring function.

Run from the rag_doc root:  python -m pytest tests/test_eval_metrics.py
"""
import pytest

from eval.run_eval import evaluate_question

# Two answer groups: group 1 has two equivalent copies (a1, a2); group 2 has one chunk (b1).
ENTRY = {
    "relevant_chunks": ["a1", "a2", "b1"],
    "answer_groups": [["a1", "a2"], ["b1"]],
}


def test_all_groups_covered():
    r = evaluate_question(ENTRY, ["x", "a2", "b1", "y", "z"], k=5)
    assert r["recall"] == 1.0
    assert r["full"] == 1.0
    assert r["precision"] == pytest.approx(2 / 5)
    assert r["mrr"] == pytest.approx(1 / 2)


def test_partial_coverage():
    r = evaluate_question(ENTRY, ["a1", "x", "y", "z", "w"], k=5)
    assert r["recall"] == 0.5
    assert r["full"] == 0.0
    assert r["mrr"] == 1.0


def test_copies_of_the_same_passage_count_once():
    # Returning both equivalent copies must not push recall above 1 or count as two groups.
    r = evaluate_question(ENTRY, ["a1", "a2", "b1", "x", "y"], k=5)
    assert r["recall"] == 1.0
    assert r["precision"] == pytest.approx(3 / 5)


def test_nothing_relevant_found():
    r = evaluate_question(ENTRY, ["x", "y", "z", "w", "v"], k=5)
    assert r["recall"] == 0.0
    assert r["full"] == 0.0
    assert r["precision"] == 0.0
    assert r["mrr"] == 0.0


def test_only_top_k_counts():
    r = evaluate_question(ENTRY, ["x", "y", "z", "w", "v", "a1", "b1"], k=5)
    assert r["recall"] == 0.0
    assert r["retrieved"] == ["x", "y", "z", "w", "v"]
