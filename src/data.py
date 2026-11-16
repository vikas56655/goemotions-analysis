"""Load the official simplified GoEmotions split and convert labels to multi-hot arrays."""
from __future__ import annotations

import numpy as np
from datasets import load_dataset
from sklearn.preprocessing import MultiLabelBinarizer

DATASET_ID = "google-research-datasets/go_emotions"
CONFIG = "simplified"


def load_goemotions():
    dataset = load_dataset(DATASET_ID, CONFIG)
    label_names = dataset["train"].features["labels"].feature.names
    return dataset, label_names


def encode_labels(label_lists: list[list[int]], n_labels: int) -> np.ndarray:
    """Always keep all label columns, including classes absent in a sampled split."""
    mlb = MultiLabelBinarizer(classes=list(range(n_labels)))
    return mlb.fit_transform(label_lists).astype(np.int8)


def take_split(split, limit: int | None = None, seed: int = 42):
    """Random reproducible subset; None (or 0) means the complete split."""
    if limit is None or limit <= 0 or limit >= len(split):
        return split
    return split.shuffle(seed=seed).select(range(limit))


def extract_xy(split, n_labels: int):
    return list(split["text"]), encode_labels(list(split["labels"]), n_labels)
