"""Classify one message (or a file of messages) from the command line.

Usage
-----
    python -m src.predict --text "WINNER!! Claim your free prize now"
    python -m src.predict --file messages.txt
    echo "Hey, are we still on for 6pm?" | python -m src.predict --stdin
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from transformers import AutoTokenizer

from .config import TrainConfig
from .engine import predict_proba
from .model import load_model
from .utils import get_device


def parse_args() -> argparse.Namespace:
    cfg = TrainConfig()
    parser = argparse.ArgumentParser(description="Classify messages as ham or spam.")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--text", help="A single message to classify")
    group.add_argument("--file", type=Path, help="Text file, one message per line")
    group.add_argument("--stdin", action="store_true", help="Read messages from stdin")

    parser.add_argument("--checkpoint", type=Path, default=cfg.output_dir / "tinybert_spam_classifier.pt")
    parser.add_argument("--model-name", default=cfg.model_name)
    parser.add_argument("--max-len", type=int, default=cfg.max_len)
    parser.add_argument("--device", default=cfg.device)
    return parser.parse_args()


def collect_inputs(args) -> list[str]:
    if args.text:
        return [args.text]
    if args.file:
        return [line.strip() for line in Path(args.file).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.stdin:
        return [line.strip() for line in sys.stdin.read().splitlines() if line.strip()]
    # No flag: fall back to stdin only when it is piped, otherwise prompt once.
    if not sys.stdin.isatty():
        return [line.strip() for line in sys.stdin.read().splitlines() if line.strip()]
    return [input("Message: ").strip()]


def main() -> None:
    args = parse_args()
    texts = collect_inputs(args)
    if not texts:
        print("No message provided.", file=sys.stderr)
        sys.exit(1)

    device = get_device(args.device)
    tokenizer_dir = args.checkpoint.parent / "tokenizer"
    tokenizer = (
        AutoTokenizer.from_pretrained(tokenizer_dir)
        if tokenizer_dir.exists()
        else AutoTokenizer.from_pretrained(args.model_name)
    )
    model = load_model(args.checkpoint, model_name=args.model_name, device=device)

    for text, result in zip(texts, predict_proba(model, tokenizer, texts, device, args.max_len)):
        tag = "SPAM" if result["label_id"] == 1 else "HAM"
        print(f"[{tag}] p_spam={result['p_spam']:.4f} conf={result['confidence']:.4f} :: {text[:90]}")


if __name__ == "__main__":
    main()
