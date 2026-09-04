"""Small shared helpers: reproducibility, device selection, logging."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Make runs reproducible across Python / NumPy / PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # CuDNN non-determinism is a real source of 'same code, different score'.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(preference: str = "auto") -> torch.device:
    """Resolve the requested device string into a torch.device."""
    preference = (preference or "auto").lower()
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if preference == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device(preference)


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    params = model.parameters()
    if trainable_only:
        return sum(p.numel() for p in params if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def format_metrics(metrics: dict) -> str:
    return " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items() if isinstance(v, float))
