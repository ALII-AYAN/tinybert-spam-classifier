"""Training / evaluation primitives shared by the CLI scripts."""

from __future__ import annotations

import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm


def train_one_epoch(model, loader, optimizer, scheduler, device, epoch: int = 1):
    """Run a full gradient-descent pass; return the mean training loss."""
    model.train()
    total_loss, seen = 0.0, 0
    progress = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False)

    for batch in progress:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        _, loss = model(input_ids, attention_mask, labels=labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        seen += batch_size
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(seen, 1)


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[dict, list[int], list[int]]:
    """Return (metrics, ground-truth labels, predictions)."""
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    total_loss, seen = 0.0, 0

    for batch in tqdm(loader, desc="[eval]", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits, loss = model(input_ids, attention_mask, labels=labels)

        if loss is not None:
            total_loss += loss.item() * labels.size(0)
            seen += labels.size(0)

        all_preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    metrics = {
        "loss": total_loss / max(seen, 1),
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
    }
    return metrics, all_labels, all_preds


@torch.no_grad()
def predict_proba(model, tokenizer, texts, device, max_len: int = 128, batch_size: int = 32):
    """Batched inference over raw strings -> list of {label, confidence}."""
    model.eval()
    results = []

    for start in range(0, len(texts), batch_size):
        chunk = [str(t) for t in texts[start : start + batch_size]]
        encoding = tokenizer(
            chunk,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )
        logits, _ = model(encoding["input_ids"].to(device), encoding["attention_mask"].to(device))
        probs = torch.softmax(logits, dim=1).cpu()

        for prob in probs:
            spam_p = float(prob[1])
            results.append(
                {
                    "label": "spam" if spam_p >= 0.5 else "ham",
                    "label_id": int(spam_p >= 0.5),
                    "confidence": spam_p if spam_p >= 0.5 else float(prob[0]),
                    "p_spam": spam_p,
                }
            )

    return results
