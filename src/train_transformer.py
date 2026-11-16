"""Optional transformer fine-tuning, not required for baseline or the app.

Install requirements-transformer.txt before running.
Example: python -m src.train_transformer --max-train 4000 --max-val 800
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)

from src.data import load_goemotions, take_split
from src.metrics import choose_threshold, evaluate


def main(model_name: str, max_train: int, max_val: int, max_test: int,
         epochs: float, batch_size: int, seed: int, output_dir: str):
    from datasets import Dataset

    dataset, names = load_goemotions()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    selected = {
        "train": take_split(dataset["train"], max_train, seed),
        "validation": take_split(dataset["validation"], max_val, seed + 1),
        "test": take_split(dataset["test"], max_test, seed + 2),
    }

    def prepare(split):
        texts = list(split["text"])
        multi_hot = np.zeros((len(texts), len(names)), dtype=np.float32)
        for row, indices in enumerate(split["labels"]):
            multi_hot[row, list(indices)] = 1.0
        tokens = tokenizer(texts, truncation=True, max_length=160, padding="max_length")
        tokens["labels"] = multi_hot.tolist()
        return Dataset.from_dict(tokens)

    prepared = {key: prepare(value) for key, value in selected.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(names),
        problem_type="multi_label_classification",
        id2label={i: name for i, name in enumerate(names)},
        label2id={name: i for i, name in enumerate(names)},
    )
    args = TrainingArguments(
        output_dir=output_dir, num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch", save_strategy="epoch",
        save_total_limit=1, learning_rate=2e-5, weight_decay=0.01,
        seed=seed, report_to="none", load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )
    trainer = Trainer(model=model, args=args,
                      train_dataset=prepared["train"],
                      eval_dataset=prepared["validation"])
    trainer.train()

    def probabilities(split_name):
        logits = trainer.predict(prepared[split_name]).predictions
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -80, 80)))

    val_probs = probabilities("validation")
    val_truth = np.array(prepared["validation"]["labels"], dtype=int)
    threshold, _ = choose_threshold(val_truth, val_probs)
    test_probs = probabilities("test")
    test_truth = np.array(prepared["test"]["labels"], dtype=int)
    results = {
        "model": model_name,
        "threshold": threshold,
        "sizes": {k: len(v) for k, v in selected.items()},
        "validation": evaluate(val_truth, val_probs, threshold),
        "test": evaluate(test_truth, test_probs, threshold),
    }
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(target / "best_model"))
    tokenizer.save_pretrained(str(target / "best_model"))
    (target / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="distilbert/distilbert-base-uncased")
    parser.add_argument("--max-train", type=int, default=4000)
    parser.add_argument("--max-val", type=int, default=800)
    parser.add_argument("--max-test", type=int, default=800)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="models/transformer")
    main(**vars(parser.parse_args()))
