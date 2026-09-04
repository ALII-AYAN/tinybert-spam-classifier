"""Central configuration: constants, default paths and hyper-parameters.

Keep every magic number in one place so the training / evaluation / inference
scripts stay short and reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------
# Project layout
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

DEFAULT_DATASET = DATA_DIR / "spam.csv"
DEFAULT_CHECKPOINT = MODEL_DIR / "tinybert_spam_classifier.pt"

# --------------------------------------------------------------------------
# Label mapping (single source of truth for train / eval / predict / GUI)
# --------------------------------------------------------------------------
LABEL_NAMES = ["ham", "spam"]  # index 0 -> ham, index 1 -> spam
LABEL2ID = {"ham": 0, "spam": 1}
ID2LABEL = {0: "ham", 1: "spam"}

# Raw labels found in the wild, normalised to lower-case strings.
_SPAM_ALIASES = {"spam", "1", "true"}
_HAM_ALIASES = {"ham", "not spam", "0", "false", "legit"}


def normalise_label(value) -> int:
    """Map a raw dataset label ('Spam', 'ham', 1, 0, ...) to 0 (ham) / 1 (spam)."""
    if isinstance(value, (int, float, bool)) and not isinstance(value, str):
        return int(bool(value))

    text = str(value).strip().lower()
    if text in _SPAM_ALIASES:
        return 1
    if text in _HAM_ALIASES:
        return 0
    raise ValueError(f"Unknown label {value!r}. Expected one of: ham, spam, 0, 1.")


@dataclass
class TrainConfig:
    """Hyper-parameters used by ``src/train.py``."""

    # data
    data_path: Path = DEFAULT_DATASET
    text_column: str | None = None   # None -> auto-detect
    label_column: str | None = None  # None -> auto-detect
    test_size: float = 0.2
    seed: int = 42

    # model
    model_name: str = "prajjwal1/bert-mini"
    max_len: int = 128
    dropout: float = 0.1

    # optimisation
    epochs: int = 3
    batch_size: int = 16
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.0

    # runtime
    device: str = "auto"  # 'auto' | 'cuda' | 'cpu'
    num_workers: int = 0
    output_dir: Path = MODEL_DIR
    save_plots: bool = True
    plot_dir: Path = OUTPUT_DIR

    def to_dict(self) -> dict:
        """Plain JSON-serialisable view, used for the run summary."""
        return {k: (str(v) if isinstance(v, Path) else v) for k, v in self.__dict__.items()}
