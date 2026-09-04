"""Model definition plus checkpoint save / load helpers.

Checkpoint format (a plain dict) is intentionally explicit:
    {
        "state_dict": ...,
        "model_name": "prajjwal1/bert-mini",
        "label_names": ["ham", "spam"],
        "config": {...}
    }
``load_model`` also accepts a raw ``state_dict`` so legacy ``.pth`` files saved
by the original single-file script keep working.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from .config import LABEL_NAMES


class TinyBERTSpamClassifier(nn.Module):
    """Pretrained BERT encoder + one hidden layer + a 2-way softmax head."""

    def __init__(
        self,
        model_name: str = "prajjwal1/bert-mini",
        num_labels: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.model_name = model_name
        self.bert = AutoModel.from_pretrained(model_name)

        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels),
        )

    @property
    def hidden_size(self) -> int:
        return self.bert.config.hidden_size

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # [CLS] token representation of the last hidden state.
        pooled_output = outputs.last_hidden_state[:, 0]
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        return logits, loss


def save_model(
    model: TinyBERTSpamClassifier,
    path: str | Path,
    extra: dict | None = None,
) -> Path:
    """Persist weights + everything needed to rebuild the architecture."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "state_dict": model.state_dict(),
        "model_name": getattr(model, "model_name", "prajjwal1/bert-mini"),
        "label_names": LABEL_NAMES,
    }
    if extra:
        payload["config"] = extra

    torch.save(payload, path)
    return path


def load_model(
    path: str | Path,
    model_name: str | None = None,
    device: torch.device | str = "cpu",
    dropout: float = 0.1,
) -> TinyBERTSpamClassifier:
    """Rebuild the model from a checkpoint (new dict format or legacy state_dict)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\nTrain first:  python -m src.train"
        )

    checkpoint = torch.load(path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model_name = model_name or checkpoint.get("model_name", "prajjwal1/bert-mini")
        state_dict = checkpoint["state_dict"]
    else:  # legacy: the file *is* the state_dict
        state_dict = checkpoint
        model_name = model_name or "prajjwal1/bert-mini"

    AutoConfig.from_pretrained(model_name)  # fail fast on a bad model id
    model = TinyBERTSpamClassifier(model_name, dropout=dropout)
    model.load_state_dict(state_dict)
    return model.to(device)
