"""Multi-label evaluation and validation-only threshold selection."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score,
    hamming_loss, precision_score, recall_score,
)


def binarize(probabilities: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (np.asarray(probabilities) >= threshold).astype(int)


def evaluate(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    pred = binarize(probabilities, threshold)
    out = {
        "micro_f1": float(f1_score(y_true, pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "samples_f1": float(f1_score(y_true, pred, average="samples", zero_division=0)),
        "micro_precision": float(precision_score(y_true, pred, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(y_true, pred, average="micro", zero_division=0)),
        "exact_match_accuracy": float(accuracy_score(y_true, pred)),
        "hamming_loss": float(hamming_loss(y_true, pred)),
    }
    try:
        out["micro_average_precision"] = float(
            average_precision_score(y_true, probabilities, average="micro")
        )
    except ValueError:
        out["micro_average_precision"] = None
    return out


def choose_threshold(y_val: np.ndarray, probabilities: np.ndarray,
                     candidates=None) -> tuple[float, list[dict]]:
    """Optimize MICRO F1 on the validation split only; never touch the test split."""
    if candidates is None:
        candidates = np.arange(0.10, 0.76, 0.05)
    scores = [
        {"threshold": round(float(t), 3),
         "micro_f1": float(f1_score(y_val, binarize(probabilities, t),
                                   average="micro", zero_division=0))}
        for t in candidates
    ]
    best = max(scores, key=lambda row: (row["micro_f1"], -abs(row["threshold"] - 0.5)))
    return best["threshold"], scores
