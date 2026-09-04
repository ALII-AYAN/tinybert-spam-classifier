"""Tkinter desktop demo for the TinyBERT spam classifier (no animations)."""

from __future__ import annotations

import argparse
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import torch
from transformers import AutoTokenizer

from src.config import TrainConfig
from src.engine import predict_proba
from src.model import load_model
from src.utils import get_device

BG = "#f5f7fb"
CARD = "#ffffff"
SPAM_BG = "#ffe3e3"
HAM_BG = "#dcfce7"


def parse_args() -> argparse.Namespace:
    cfg = TrainConfig()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--checkpoint", type=Path, default=cfg.output_dir / "tinybert_spam_classifier.pt")
    parser.add_argument("--model-name", default=cfg.model_name)
    parser.add_argument("--max-len", type=int, default=cfg.max_len)
    parser.add_argument("--device", default=cfg.device)
    return parser.parse_args()


def build_gui(args):
    device = get_device(args.device)
    tokenizer_dir = args.checkpoint.parent / "tokenizer"
    tokenizer = (
        AutoTokenizer.from_pretrained(tokenizer_dir)
        if tokenizer_dir.exists()
        else AutoTokenizer.from_pretrained(args.model_name)
    )
    model = load_model(args.checkpoint, model_name=args.model_name, device=device)

    root = tk.Tk()
    root.title("TinyBERT Spam Classifier")
    root.geometry("560x460")
    root.minsize(480, 420)
    root.configure(bg=BG)

    card = tk.Frame(root, bg=CARD, padx=20, pady=18, highlightthickness=1, highlightbackground="#e3e8ef")
    card.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.9, relheight=0.9)

    tk.Label(card, text="TinyBERT Spam Classifier", font=("Segoe UI", 16, "bold"), bg=CARD, fg="#1f2937").pack(pady=(0, 4))
    tk.Label(card, text="Type an SMS or email below and classify it.", font=("Segoe UI", 10), bg=CARD, fg="#6b7280").pack()

    text_frame = tk.Frame(card, bg=CARD)
    text_frame.pack(fill="both", expand=True, pady=12)

    input_box = tk.Text(
        text_frame,
        height=8,
        wrap="word",
        font=("Segoe UI", 11),
        bg="#f8fafc",
        relief="solid",
        bd=1,
        padx=8,
        pady=8,
    )
    scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=input_box.yview)
    input_box.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side="right", fill="y")
    input_box.pack(side="left", fill="both", expand=True)

    result_box = tk.Frame(card, bg="#f3f4f6", padx=12, pady=10)
    result_box.pack(fill="x")

    result_label = tk.Label(result_box, text="Waiting for input", font=("Segoe UI", 14, "bold"), bg="#f3f4f6", fg="#374151")
    result_label.pack()
    confidence_label = tk.Label(result_box, text="", font=("Segoe UI", 9), bg="#f3f4f6", fg="#6b7280")
    confidence_label.pack()

    def classify() -> None:
        message = input_box.get("1.0", "end").strip()
        if not message:
            messagebox.showwarning("Empty input", "Please enter a message first.")
            return

        result = predict_proba(model, tokenizer, [message], device, args.max_len)[0]
        is_spam = result["label_id"] == 1

        result_label.config(
            text="SPAM" if is_spam else "HAM (safe)",
            fg="#b91c1c" if is_spam else "#15803d",
        )
        confidence_label.config(text=f"p(spam) = {result['p_spam']:.3f}   confidence = {result['confidence']:.3f}")
        result_box.config(bg=SPAM_BG if is_spam else HAM_BG)
        result_label.config(bg=SPAM_BG if is_spam else HAM_BG)
        confidence_label.config(bg=SPAM_BG if is_spam else HAM_BG)

    def clear() -> None:
        input_box.delete("1.0", "end")
        result_label.config(text="Waiting for input", fg="#374151")
        confidence_label.config(text="")
        result_box.config(bg="#f3f4f6")
        result_label.config(bg="#f3f4f6")
        confidence_label.config(bg="#f3f4f6")

    button_row = tk.Frame(card, bg=CARD)
    button_row.pack(fill="x", pady=(12, 0))

    ttk.Button(button_row, text="Classify", command=classify).pack(side="left", expand=True, fill="x", padx=(0, 6))
    ttk.Button(button_row, text="Clear", command=clear).pack(side="left", expand=True, fill="x", padx=(6, 0))

    # Ctrl+Enter shortcut
    root.bind_all("<Control-Return>", lambda _event: classify())
    return root


def main() -> None:
    args = parse_args()
    if not Path(args.checkpoint).exists():
        messagebox.showerror("Checkpoint missing", f"{args.checkpoint}\n\nTrain the model first:\n  python -m src.train")
        return
    root = build_gui(args)
    root.mainloop()
    del root
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
