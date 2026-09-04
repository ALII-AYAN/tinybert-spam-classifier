"""Create a small, stratified sample of the full dataset for the repo.

The full UCI SMS Spam Collection is ~500 KB and is not committed. A 200-row
sample keeps the repository self-contained: a reviewer can clone and run
``python -m src.train --data data/sample.csv`` to verify the pipeline works
end-to-end without downloading anything.

Usage
-----
    python scripts/make_sample.py                      # data/spam.csv -> data/sample.csv
    python scripts/make_sample.py --n 300 --seed 7
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.data import load_spam_csv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a stratified repo sample.")
    parser.add_argument("--source", type=Path, default=Path("data/spam.csv"))
    parser.add_argument("--target", type=Path, default=Path("data/sample.csv"))
    parser.add_argument("--n", type=int, default=200, help="Total rows in the sample")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = load_spam_csv(args.source)
    # Keep the real ham/spam ratio so the sample behaves like the full set.
    per_class = max(1, args.n // df["label"].nunique())
    sample = (
        df.groupby("label", group_keys=False)
        .apply(lambda g: g.sample(n=min(per_class, len(g)), random_state=args.seed))
        .sample(frac=1.0, random_state=args.seed)  # shuffle so classes interleave
        .reset_index(drop=True)
    )

    args.target.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(args.target, index=False)

    counts = sample["label"].value_counts().sort_index()
    print(f"Wrote {len(sample)} rows -> {args.target}")
    print(f"  ham : {counts.get(0, 0)}")
    print(f"  spam: {counts.get(1, 0)}")
    print("Note: sample.csv is for smoke-testing the pipeline only.")


if __name__ == "__main__":
    main()
