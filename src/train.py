"""Train the TinyBERT spam classifier.

Usage
-----
    python -m src.train --data data/spam.csv --epochs 3
    python -m src.train --model-name prajjwal1/bert-mini --batch-size 32 --lr 3e-5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .config import LABEL_NAMES, TrainConfig
from .data import build_dataloaders, load_spam_csv
from .engine import evaluate, train_one_epoch
from .model import TinyBERTSpamClassifier, save_model
from .plots import (
    plot_class_distribution,
    plot_confusion_matrix,
    plot_report_heatmap,
    plot_training_history,
)
from .utils import count_parameters, get_device, set_seed


def parse_args() -> argparse.Namespace:
    cfg = TrainConfig()
    parser = argparse.ArgumentParser(
        description="Train a TinyBERT spam/ham classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", type=Path, default=cfg.data_path, help="Path to the CSV dataset")
    parser.add_argument("--model-name", default=cfg.model_name, help="Hugging Face model id")
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--lr", type=float, default=cfg.lr)
    parser.add_argument("--max-len", type=int, default=cfg.max_len)
    parser.add_argument("--dropout", type=float, default=cfg.dropout)
    parser.add_argument("--weight-decay", type=float, default=cfg.weight_decay)
    parser.add_argument("--warmup-ratio", type=float, default=cfg.warmup_ratio)
    parser.add_argument("--test-size", type=float, default=cfg.test_size)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--device", default=cfg.device, choices=["auto", "cuda", "cpu", "mps"])
    parser.add_argument("--num-workers", type=int, default=cfg.num_workers)
    parser.add_argument("--output-dir", type=Path, default=cfg.output_dir, help="Where to save checkpoints")
    parser.add_argument("--plot-dir", type=Path, default=cfg.plot_dir, help="Where to save charts")
    parser.add_argument("--no-plots", action="store_true", help="Skip chart generation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    print(f"Device: {device}")
    df = load_spam_csv(args.data)
    print(f"Loaded {len(df)} rows -> ham={(df['label'] == 0).sum()}, spam={(df['label'] == 1).sum()}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_loader, val_loader = build_dataloaders(
        df,
        tokenizer,
        max_len=args.max_len,
        batch_size=args.batch_size,
        test_size=args.test_size,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    model = TinyBERTSpamClassifier(args.model_name, dropout=args.dropout).to(device)
    print(f"Trainable parameters: {count_parameters(model):,}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_accuracy": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
    }
    best_f1, best_state = -1.0, None

    for epoch in range(1, args.epochs + 1):
        print(f"\n===== Epoch {epoch}/{args.epochs} =====")
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, epoch)
        metrics, y_true, y_pred = evaluate(model, val_loader, device)

        history["train_loss"].append(train_loss)
        for key in ("accuracy", "precision", "recall", "f1"):
            history[f"val_{key}"].append(metrics[key])

        print(
            f"  train_loss={train_loss:.4f}  "
            f"acc={metrics['accuracy']:.4f}  prec={metrics['precision']:.4f}  "
            f"rec={metrics['recall']:.4f}  f1={metrics['f1']:.4f}"
        )

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # ---------------- persist artefacts ----------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "tinybert_spam_classifier.pt"

    save_model(
        model,
        checkpoint_path,
        extra={
            "max_len": args.max_len,
            "model_name": args.model_name,
            "best_val_f1": best_f1,
            "label_names": LABEL_NAMES,
        },
    )
    tokenizer.save_pretrained(output_dir / "tokenizer")

    summary = {
        "best_val_f1": best_f1,
        "history": history,
        "hyperparameters": {
            "model_name": args.model_name,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "max_len": args.max_len,
            "seed": args.seed,
        },
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + classification_report(y_true, y_pred, target_names=LABEL_NAMES, digits=4))
    print(f"Best validation F1: {best_f1:.4f}")
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Tokenizer  : {output_dir / 'tokenizer'}")
    print(f"Summary    : {output_dir / 'training_summary.json'}")

    if not args.no_plots:
        plot_dir = Path(args.plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_class_distribution(df["label"], plot_dir / "class_distribution.png")
        plot_training_history(history, plot_dir / "training_history.png")
        plot_confusion_matrix(y_true, y_pred, plot_dir / "confusion_matrix.png")

        report_df = pd.DataFrame(
            classification_report(y_true, y_pred, target_names=LABEL_NAMES, output_dict=True)
        ).T.iloc[:-1, :-1]
        plot_report_heatmap(report_df, plot_dir / "classification_report.png")
        print(f"Charts     : {plot_dir}")


if __name__ == "__main__":
    main()
