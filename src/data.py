"""Dataset loading, tokenisation and DataLoader construction."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from .config import normalise_label


class SpamDataset(Dataset):
    """Tokenises texts once, up-front, and serves tensors to the DataLoader."""

    def __init__(self, texts, labels, tokenizer, max_len: int = 128):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(int(self.labels[idx]), dtype=torch.long),
        }


def load_spam_csv(
    csv_path: str | Path,
    text_column: str | None = None,
    label_column: str | None = None,
    encoding: str = "latin-1",
) -> pd.DataFrame:
    """Read a 2-column SMS-spam style CSV and return a clean [label, text] frame.

    Auto-detects columns when they are not supplied, drops empty rows and
    normalises labels to 0 (ham) / 1 (spam).
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}\n"
            "Download the UCI SMS Spam Collection and save it as data/spam.csv "
            "(see data/README.md)."
        )

    df = pd.read_csv(csv_path, encoding=encoding)

    if text_column is None or label_column is None:
        text_column, label_column = _detect_columns(df)

    df = df[[label_column, text_column]].rename(
        columns={label_column: "label", text_column: "text"}
    )
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].ne("") & df["text"].ne("nan")]

    df["label"] = df["label"].map(normalise_label)
    return df[["label", "text"]].reset_index(drop=True)


def _detect_columns(df: pd.DataFrame) -> tuple[str, str]:
    """Find the label / text columns by name, falling back to position."""
    lowered = {str(c).strip().lower(): c for c in df.columns}
    for cand in ("label", "labels", "class", "category", "target", "v1"):
        if cand in lowered:
            label_col = lowered[cand]
            break
    else:
        label_col = df.columns[0]

    for cand in ("text", "message", "sms", "email", "content", "body", "v2"):
        if cand in lowered and lowered[cand] != label_col:
            text_col = lowered[cand]
            break
    else:
        text_col = next((c for c in df.columns if c != label_col), df.columns[1])

    return text_col, label_col


def build_dataloaders(
    df: pd.DataFrame,
    tokenizer,
    max_len: int = 128,
    batch_size: int = 16,
    test_size: float = 0.2,
    seed: int = 42,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Stratified train/val split -> two DataLoaders."""
    x_train, x_val, y_train, y_val = train_test_split(
        df["text"].tolist(),
        df["label"].tolist(),
        test_size=test_size,
        random_state=seed,
        stratify=df["label"].tolist(),
    )

    train_ds = SpamDataset(x_train, y_train, tokenizer, max_len)
    val_ds = SpamDataset(x_val, y_val, tokenizer, max_len)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader
