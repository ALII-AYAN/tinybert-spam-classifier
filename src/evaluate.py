"""Evaluate a saved checkpoint on a held-out split.

Usage
-----
    python -m src.evaluate --data data/spam.csv --checkpoint models/tinybert_spam_classifier.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .config import LABEL_NAMES, TrainConfig
from .data import SpamDataset, load_spam_csv
from .engine import evaluate
from .model import load_model
from .plots import plot_confusion_matrix
from .utils import get_device, set_seed


def parse_args() -> argparse.Namespace:
    cfg = TrainConfig()
    parser = argparse.ArgumentParser(
        description="Evaluate a trained spam classifier checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", type=Path, default=cfg.data_path)
    parser.add_argument("--checkpoint", type=Path, default=cfg.output_dir / "tinybert_spam_classifier.pt")
    parser.add_argument("--model-name", default=cfg.model_name)
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--max-len", type=int, default=cfg.max_len)
    parser.add_argument("--test-size", type=float, default=cfg.test_size)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--device", default=cfg.device)
    parser.add_argument("--split", choices=["val", "all"], default="val")
    parser.add_argument("--save-report", type=Path, default=None, help="Optional JSON report path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    df = load_split(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    dataset = SpamDataset(df["text"].tolist(), df["label"].tolist(), tokenizer, args.max_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = load_model(args.checkpoint, model_name=args.model_name, device=device)
    metrics, y_true, y_pred = evaluate(model, loader, device)

    print("\n" + classification_report(y_true, y_pred, target_names=LABEL_NAMES, digits=4))
    print("Summary:", json.dumps({k: round(v, 4) for k, v in metrics.items()}, indent=2))

    if args.save_report:
        args.save_report.parent.mkdir(parents=True, exist_ok=True)
        args.save_report.write_text(json.dumps(metrics, indent=2))
        print(f"Report saved to {args.save_report}")

    plot_confusion_matrix(y_true, y_pred, Path("outputs") / "confusion_matrix_eval.png")
    print("Confusion matrix saved to outputs/confusion_matrix_eval.png")


def load_split(args) -> pd.DataFrame:
    """Rebuild the same val split used during training (or use the full file)."""
    df = load_spam_csv(args.data)
    if args.split == "all":
        return df

    _, x_val, _, y_val = train_test_split(
        df["text"].tolist(),
        df["label"].tolist(),
        test_size=args.test_size,
        random_state=args.seed,
        stratify=df["label"].tolist(),
    )
    return pd.DataFrame({"label": y_val, "text": x_val})


if __name__ == "__main__":
    main()
