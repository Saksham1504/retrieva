"""Unit tests for eval/run_eval.py's scoring function.

Run from the rag_doc root:  python -m pytest tests/test_eval_metrics.py
"""
import pytest

from eval.run_eval import abstention_stats, best_threshold, evaluate_question, sweep_thresholds

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


# ---------------------------------------------------------------- abstention gate
def test_abstention_stats_counts_correct_and_wrong_refusals():
    answerable = [5.0, 3.0, -1.0]
    unanswerable = [-4.0, 2.0]
    s = abstention_stats(answerable, unanswerable, threshold=0.0)    # refuse when score < 0
    assert s["correct_refusals"] == 1                                # -4 is refused, 2 is answered
    assert s["wrong_refusals"] == 1                                  # -1 is refused by mistake
    assert s["balanced_accuracy"] == pytest.approx(0.5 * (1 / 2 + (1 - 1 / 3)))


def test_sweep_finds_a_threshold_that_separates_perfectly_separable_scores():
    answerable = [3.0, 4.0, 5.0]
    unanswerable = [-2.0, -1.0, 0.0]
    best = best_threshold(sweep_thresholds(answerable, unanswerable))
    assert 0.0 < best["threshold"] < 3.0
    assert best["correct_refusals"] == 3
    assert best["wrong_refusals"] == 0
    assert best["balanced_accuracy"] == 1.0


def test_threshold_sits_midway_in_the_gap_between_the_two_groups():
    answerable = [10.0, 11.0]
    unanswerable = [-1.0, 0.0]
    best = best_threshold(sweep_thresholds(answerable, unanswerable))
    assert best["balanced_accuracy"] == 1.0
    assert best["threshold"] == pytest.approx(5.0)                   # halfway between 0 and 10


def test_sweep_with_overlapping_scores_reports_the_trade_off():
    answerable = [1.0, 6.0, 7.0, 8.0]
    unanswerable = [-3.0, 0.0, 2.0, 5.0]                             # 2.0 and 5.0 look like answerable scores
    best = best_threshold(sweep_thresholds(answerable, unanswerable))
    assert best["correct_refusals"] + best["wrong_refusals"] > 0
    assert best["balanced_accuracy"] < 1.0
