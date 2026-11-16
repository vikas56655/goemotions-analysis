"""Train a TF-IDF + one-vs-rest Logistic Regression baseline.

Run: python -m src.train --max-train 12000 --max-val 2000 --max-test 2000
Use --max-train 0 --max-val 0 --max-test 0 for all examples.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

from src.data import extract_xy, load_goemotions, take_split
from src.metrics import choose_threshold, evaluate
from src.visualize import (save_cooccurrence, save_f1_per_label,
                           save_label_frequency, save_threshold_curve)


def train(max_train=12000, max_val=2000, max_test=2000, seed=42,
          output_dir="models", report_dir="reports"):
    output = Path(output_dir)
    report = Path(report_dir)
    output.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)

    dataset, names = load_goemotions()
    splits = {
        "train": take_split(dataset["train"], max_train, seed),
        "validation": take_split(dataset["validation"], max_val, seed + 1),
        "test": take_split(dataset["test"], max_test, seed + 2),
    }
    x_train, y_train = extract_xy(splits["train"], len(names))
    x_val, y_val = extract_xy(splits["validation"], len(names))
    x_test, y_test = extract_xy(splits["test"], len(names))

    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                                   max_features=50000, sublinear_tf=True)),
        ("classifier", OneVsRestClassifier(
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
        )),
    ])
    print(f"Training on {len(x_train):,} messages; {len(names)} labels")
    model.fit(x_train, y_train)
    val_probs = model.predict_proba(x_val)
    threshold, curve = choose_threshold(y_val, val_probs)
    test_probs = model.predict_proba(x_test)
    results = {
        "model": "tfidf_ovr_logistic_regression",
        "dataset": "google-research-datasets/go_emotions / simplified",
        "seed": seed,
        "sizes": {name: len(split) for name, split in splits.items()},
        "threshold": threshold,
        "validation": evaluate(y_val, val_probs, threshold),
        "test": evaluate(y_test, test_probs, threshold),
        "label_names": names,
        "note": "Test evaluated once after tuning on validation.",
    }
    joblib.dump(model, output / "baseline.joblib")
    (output / "metadata.json").write_text(
        json.dumps({"label_names": names, "threshold": threshold}, indent=2),
        encoding="utf-8",
    )
    (report / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    save_label_frequency(y_train, names, report / "label_frequency.png")
    save_cooccurrence(y_train, names, report / "cooccurrence.png")
    save_f1_per_label(y_test, test_probs, threshold, names, report / "per_label_f1.png")
    save_threshold_curve(curve, threshold, report / "threshold_curve.png")
    print(json.dumps(results["test"], indent=2))
    print(f"Saved model to {output.resolve()} and plots to {report.resolve()}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-train", type=int, default=12000)
    parser.add_argument("--max-val", type=int, default=2000)
    parser.add_argument("--max-test", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--report-dir", default="reports")
    train(**vars(parser.parse_args()))
