"""Convert a legacy checkpoint (raw ``state_dict``) into the self-describing format.

The original single-file trainer saved checkpoints like this:

    torch.save(model.state_dict(), "...tinybert_spam_classifier_withNN.pth")

That file stores the weights only - no model id, no label names. ``load_model``
can still read it, but converting once makes the checkpoint self-describing so
the weights can never be loaded into a mismatched architecture.

Usage
-----
    python scripts/convert_legacy_checkpoint.py models/old.pth models/new.pt
    python scripts/convert_legacy_checkpoint.py models/old.pth --model-name prajjwal1/bert-mini
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.config import LABEL_NAMES  # noqa: E402
from src.model import TinyBERTSpamClassifier, save_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Upgrade a raw state_dict checkpoint.")
    parser.add_argument("source", type=Path, help="legacy .pth / .pt file")
    parser.add_argument("target", type=Path, nargs="?", help="output path (default: alongside source)")
    parser.add_argument("--model-name", default="prajjwal1/bert-mini")
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()

    if not args.source.exists():
        raise SystemExit(f"Source checkpoint not found: {args.source}")

    target = args.target or args.source.with_name("tinybert_spam_classifier.pt")

    state_dict = torch.load(args.source, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        print("Source is already in the new format; nothing to convert.")
        if target != args.source:
            save_model(
                _rebuild(args.model_name, state_dict["state_dict"], args.dropout),
                target,
                extra={"model_name": args.model_name, "label_names": LABEL_NAMES,
                       **state_dict.get("config", {})},
            )
            print(f"Copied to {target}")
        return

    model = _rebuild(args.model_name, state_dict, args.dropout)
    save_model(
        model,
        target,
        extra={"model_name": args.model_name, "label_names": LABEL_NAMES,
               "source": args.source.name},
    )
    print(f"Converted {args.source} -> {target}")


def _rebuild(model_name: str, state_dict, dropout: float) -> TinyBERTSpamClassifier:
    model = TinyBERTSpamClassifier(model_name, dropout=dropout)
    model.load_state_dict(state_dict)  # raises loudly on an architecture mismatch
    return model


if __name__ == "__main__":
    main()
