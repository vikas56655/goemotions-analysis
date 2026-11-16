"""Publication-ready, data-derived plots. Never hardcode evaluation scores."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score

sns.set_theme(style="whitegrid", context="notebook")


def save_label_frequency(y, names, output):
    counts = np.asarray(y).sum(axis=0)
    order = np.argsort(counts)
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.barh(np.asarray(names)[order], counts[order], color="#427bd1")
    ax.set(xlabel="Number of examples", title="GoEmotions · training label frequencies")
    fig.tight_layout()
    fig.savefig(output, dpi=175, bbox_inches="tight")
    plt.close(fig)


def save_cooccurrence(y, names, output):
    counts = np.asarray(y, dtype=float).T @ np.asarray(y, dtype=float)
    sizes = np.diag(counts)
    denom = np.sqrt(np.outer(sizes, sizes))
    matrix = np.divide(counts, denom, out=np.zeros_like(counts), where=denom > 0)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(matrix, xticklabels=names, yticklabels=names, cmap="Blues",
                vmin=0, vmax=1, ax=ax, square=True)
    ax.set_title("Emotion co-occurrence · cosine-normalized counts")
    fig.tight_layout()
    fig.savefig(output, dpi=175, bbox_inches="tight")
    plt.close(fig)


def save_f1_per_label(y, probs, threshold, names, output):
    pred = (np.asarray(probs) >= threshold).astype(int)
    values = f1_score(y, pred, average=None, zero_division=0)
    order = np.argsort(values)
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.barh(np.asarray(names)[order], values[order], color="#20a18f")
    ax.set(xlim=(0, 1), xlabel="F1 score", title="Test F1 by emotion")
    fig.tight_layout()
    fig.savefig(output, dpi=175, bbox_inches="tight")
    plt.close(fig)


def save_threshold_curve(scores, selected, output):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot([row["threshold"] for row in scores],
            [row["micro_f1"] for row in scores], marker="o", color="#427bd1")
    ax.axvline(selected, ls="--", color="#d07437", label=f"Selected: {selected:.2f}")
    ax.set(xlabel="Decision threshold", ylabel="Validation micro-F1",
           title="Threshold tuning · validation only")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=175, bbox_inches="tight")
    plt.close(fig)
