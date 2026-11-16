import numpy as np
from src.metrics import binarize, choose_threshold, evaluate
from src.data import encode_labels


def test_all_classes_preserved():
    result = encode_labels([[0, 2], [2], []], n_labels=4)
    assert result.shape == (3, 4)
    assert result[0].tolist() == [1, 0, 1, 0]


def test_multilabel_threshold():
    p = np.array([[0.8, 0.2], [0.1, 0.9]])
    assert binarize(p, 0.5).tolist() == [[1, 0], [0, 1]]


def test_perfect_scores():
    y = np.array([[1, 0], [0, 1]])
    p = np.array([[0.9, 0.1], [0.2, 0.8]])
    result = evaluate(y, p, 0.5)
    assert result["micro_f1"] == 1.0
    assert result["macro_f1"] == 1.0
    assert result["exact_match_accuracy"] == 1.0


def test_threshold_selection_uses_passed_validation_only():
    y = np.array([[1, 0], [0, 1]])
    p = np.array([[0.7, 0.2], [0.3, 0.8]])
    chosen, scores = choose_threshold(y, p, candidates=[0.25, 0.5, 0.75])
    assert chosen == 0.5
    assert len(scores) == 3
