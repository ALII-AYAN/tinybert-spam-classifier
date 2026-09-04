"""All matplotlib / seaborn plotting, isolated from the training logic.

Every function saves to disk (headless-safe) and returns the output path.
Nothing here calls ``plt.show()`` so the code runs on a server or in CI.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # must be set before pyplot is imported

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402

sns.set_theme(style="whitegrid")


def plot_class_distribution(labels, output_path):
    output_path = Path(output_path)
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=2)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.barplot(x=["ham", "spam"], y=counts, hue=["ham", "spam"], palette="coolwarm", legend=False, ax=ax)
    ax.set_title("Class distribution")
    ax.set_ylabel("count")
    for i, value in enumerate(counts):
        ax.text(i, value, str(value), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_training_history(history: dict, output_path):
    """history: {'train_loss': [...], 'val_f1': [...], ...}"""
    output_path = Path(output_path)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, history["train_loss"], marker="o", color="#e74c3c")
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_xticks(list(epochs))

    for key, color in (
        ("val_accuracy", "#2ecc71"),
        ("val_precision", "#3498db"),
        ("val_recall", "#9b59b6"),
        ("val_f1", "#f39c12"),
    ):
        if key in history:
            axes[1].plot(epochs, history[key], marker="o", label=key.replace("val_", ""))
    axes[1].set_title("Validation metrics")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xticks(list(epochs))
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_confusion_matrix(y_true, y_pred, output_path, labels=("ham", "spam")):
    output_path = Path(output_path)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_title("Confusion matrix")
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_report_heatmap(report_df, output_path):
    output_path = Path(output_path)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(report_df, annot=True, cmap="YlGnBu", fmt=".2f", ax=ax)
    ax.set_title("Classification report")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
