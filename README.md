# TinyBERT Spam Classifier

Lightweight SMS spam detection built on a four-layer **BERT-mini** encoder with a hierarchical classification head, trained with mixed precision and served through a Tkinter desktop app.

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Overview

Spam filtering has to reconcile two goals that pull in opposite directions: catch as much spam as possible, and stay cheap enough to run. Full-sized BERT reaches strong accuracy but carries roughly 110M parameters, which is hard to justify for a filter that has to score every incoming message.

This project builds the classifier around BERT-mini — four transformer layers, 256 hidden units, four attention heads — and adds a head that progressively compresses the `[CLS]` representation before classifying it. On the SMS Spam Collection the model reaches an **F1-score of 0.9329** with an inference latency of about **47 ms per message**, at a fraction of the compute a full transformer would need.

The system has three components:

- **Data processing** — label mapping, BERT-mini tokenisation, truncation/padding to 128 tokens, and an 80/20 stratified split
- **Model training** — the encoder plus custom head, with AMP mixed precision, AdamW optimisation, and evaluation on standard classification metrics
- **GUI application** — an interactive interface for classifying a message and inspecting the confidence score

---

## Demo

```
$ python -m src.predict --text "WINNER!! You have won a £1000 cash prize. Reply NOW to claim."

[SPAM] p_spam=0.9987 conf=0.9987 :: WINNER!! You have won a £1000 cash prize. Reply NOW to claim.

$ python -m src.predict --text "Hey, are we still on for 6pm tomorrow?"

[HAM] p_spam=0.0012 conf=0.9988 :: Hey, are we still on for 6pm tomorrow?
```

Or launch the desktop app:

```bash
python -m app.gui
```

Type a message, press **Classify** (or `Ctrl+Enter`), and the result panel shows the verdict together with the spam probability.

---

## Architecture

The model is a four-stage pipeline: contextual encoding, feature representation, hierarchical encoding, and binary classification.

```
Input text
    │
    ▼
BERT-mini tokenizer          truncation + padding to 128 tokens
    │
    ▼
BERT-mini encoder            4 layers, hidden 256, 4 attention heads
    │                        → batch × 128 × 256
    ▼
[CLS] representation         256-d
    │
    ▼
Representation layer         Linear(256 → 512) → ReLU → Dropout(0.2) → LayerNorm
    │
    ▼
Hierarchical encoding        Linear(512 → 256) → ReLU → Dropout(0.2)
                             Linear(256 → 128) → ReLU → Dropout(0.2)
                             Linear(128 →  64) → ReLU
    ▼
Output layer                 Linear(64 → 2) → ham / spam logits
```

Two ideas drive this design. First, the head **expands before it compresses**: running the 256-d `[CLS]` vector up to 512 dimensions gives the task-specific layers room to separate the classes before the 512 → 256 → 128 → 64 pyramid squeezes out redundancy. Second, the compression itself is a regulariser — by the time the representation reaches 64 dimensions it has to be discriminative, because there is no room left to memorise noise.

Baseline details:

| Property | Value |
|---|---|
| Encoder | `prajjwal1/bert-mini` — 4 layers, 256 hidden, 4 heads |
| Input length | 128 tokens (truncate / pad) |
| Head dimensions | 256 → 512 → 256 → 128 → 64 → 2 |
| Dropout | 0.2 |
| Total parameters | ~11.5M (~11.2M encoder + ~0.3M head) |
| For comparison | BERT-base ≈ 110M parameters |

---

## Setup

```bash
git clone https://github.com/your-username/tinybert-spam-classifier.git
cd tinybert-spam-classifier

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

**Tkinter** (needed only for `app.gui`) ships with CPython on Windows and macOS. On Debian/Ubuntu:

```bash
sudo apt-get install python3-tk
```

---

## Dataset

The [SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) — 5,572 messages, split 4,825 ham and 747 spam.

Download `spam.csv` and place it in `data/`:

```bash
mv ~/Downloads/spam.csv data/spam.csv
```

That is the only setup step. The Kaggle file uses `v1`/`v2` column headers plus three empty filler columns; the loader detects the real columns and ignores the rest, so the file works exactly as downloaded. It also handles `label,text` and `label,message` layouts, and normalises label spellings (`Spam`, `not spam`, `1`, `0`, any casing).

---

## Training

```bash
python -m src.train --data data/spam.csv --epochs 3
```

The experiment uses a stratified 80/20 split, giving 4,457 training messages and 1,115 validation messages.

```bash
python -m src.train \
  --data data/spam.csv \
  --model-name prajjwal1/bert-mini \
  --epochs 3 \
  --batch-size 32 \
  --lr 2e-5 \
  --max-len 128 \
  --dropout 0.2 \
  --seed 42
```

| Flag | Default | Notes |
|---|---|---|
| `--model-name` | `prajjwal1/bert-mini` | Any HF encoder works — `bert-base-uncased`, `distilbert-base-uncased`, ... |
| `--epochs` | `3` | Three is the sweet spot; a fourth overfits |
| `--batch-size` | `32` | Drop to 16 if you run out of memory |
| `--lr` | `2e-5` | Standard fine-tuning LR for BERT |
| `--max-len` | `128` | Covers the vast majority of messages |
| `--dropout` | `0.2` | Applied in the representation and encoding layers |
| `--device` | `auto` | `auto` / `cuda` / `cpu` / `mps` |
| `--no-amp` | off | Disable mixed precision (AMP is CUDA-only) |
| `--no-plots` | off | Skip chart generation |

### Mixed precision

Training runs under automatic mixed precision: forward passes execute in fp16 under `torch.autocast`, and a `GradScaler` scales gradients to keep reduced-precision updates numerically stable. This cuts memory use and lets a larger batch fit on the same GPU. It activates only on CUDA — on CPU the flag is ignored rather than silently slowing things down.

After training:

```
models/
├── tinybert_spam_classifier.pt   # best-validation-F1 weights
├── tokenizer/                    # so inference never hits the network
└── training_summary.json         # per-epoch metrics + hyperparameters
outputs/
├── class_distribution.png
├── training_history.png
├── confusion_matrix.png
└── classification_report.png
```

---

## Results

Evaluated on the held-out 20% of the SMS Spam Collection (1,115 messages):

| Metric | Value |
|---|---|
| **F1-score** | **0.9329** |
| **Inference latency** | **~47 ms / message** |
| Training epochs | 3 |
| Sequence length | 128 tokens |
| Parameters | ~11.5M |

Training loss decreased across all three epochs, with the largest reduction in the first epoch and progressively smaller improvements after — the expected pattern for fine-tuning a pretrained encoder on a small dataset, where most of the task adaptation happens immediately.

**Why F1 is the headline number.** Spam is only 13.4% of the corpus, so accuracy is a misleading metric here: a classifier that predicted "ham" for every message would still score 86.6%. F1 is reported as the primary result because it balances precision and recall on the minority class, which is the one that matters.

**Reproducing the latency figure:**

```bash
python scripts/benchmark_latency.py --checkpoint models/tinybert_spam_classifier.pt
```

This times a single-message forward pass over 300 iterations after a 20-iteration warm-up, and reports mean, median and p95 latency. Timings are hardware-dependent — the 47 ms figure comes from the evaluation machine, so run the benchmark on your own hardware to compare.

---

## Inference

```bash
# single message
python -m src.predict --text "Free entry to win a car! Text CAR to 87121"

# a file, one message per line
python -m src.predict --file messages.txt

# pipe it
cat messages.txt | python -m src.predict --stdin
```

Evaluate a checkpoint:

```bash
python -m src.evaluate --data data/spam.csv --checkpoint models/tinybert_spam_classifier.pt
```

### From Python

```python
import torch
from transformers import AutoTokenizer
from src.model import load_model
from src.engine import predict_proba
from src.utils import get_device

device = get_device()
tokenizer = AutoTokenizer.from_pretrained("models/tokenizer")
model = load_model("models/tinybert_spam_classifier.pt", device=device)

result = predict_proba(model, tokenizer, ["Free entry to win a car!"], device)[0]
print(result)
# {'label': 'spam', 'label_id': 1, 'confidence': 0.997, 'p_spam': 0.997}
```

---

## Using your own checkpoint

Drop it in `models/` under the name `tinybert_spam_classifier.pt` and every command above works with no extra flags. To keep a different filename:

```bash
python -m src.predict --checkpoint models/my_model.pth --text "Free entry to win a car!"
python -m app.gui      --checkpoint models/my_model.pth
```

Checkpoints are saved as a dict holding the weights plus the model id and config, so weights cannot be silently loaded into a mismatched architecture. A plain `state_dict` from `torch.save(model.state_dict(), path)` still loads fine, and older checkpoints can be upgraded to the self-describing format:

```bash
python scripts/convert_legacy_checkpoint.py models/old_model.pth models/tinybert_spam_classifier.pt
```

Note that `models/` is gitignored — weights are build artifacts, so publish them with Git LFS or as a release asset rather than committing binaries.

---

## Project layout

```
tinybert-spam-classifier/
├── src/
│   ├── config.py            # hyperparameters, paths, label mapping
│   ├── data.py              # CSV loading, tokenisation, DataLoaders
│   ├── model.py             # BERT-mini + hierarchical head, save/load
│   ├── engine.py            # train_one_epoch / evaluate / predict_proba (AMP)
│   ├── plots.py             # charts (Agg backend, headless-safe)
│   ├── train.py             # CLI: train and export a checkpoint
│   ├── evaluate.py          # CLI: score a checkpoint
│   ├── predict.py           # CLI: classify a message or a file
│   └── utils.py             # seeding, device selection, AMP helpers
├── app/
│   └── gui.py               # Tkinter desktop app
├── scripts/
│   ├── benchmark_latency.py
│   ├── convert_legacy_checkpoint.py
│   └── make_sample.py
├── data/                    # dataset (gitignored)
├── models/                  # checkpoints (gitignored)
├── outputs/                 # generated charts and reports
├── CLEANUP_NOTES.md         # notes on refactoring the original prototype
└── requirements.txt
```

Each module handles one job, so changing the encoder, the head dimensions or the training loop means touching exactly one file.

---

## Implementation notes

Decisions that made a measurable difference:

- **Stratified splitting.** With 13.4% spam, a plain random split can give the validation set a noticeably different class balance than the training set. Stratifying holds both at the corpus ratio.
- **Best-F1 checkpointing.** The final epoch is not automatically the best one. Training tracks validation F1 and saves the weights from the best epoch.
- **Layer normalisation in the representation layer.** Placing `LayerNorm` after the 256 → 512 expansion stabilises activations through the compression pyramid that follows.
- **Gradient scaling** with AMP, without which fp16 gradients underflow and the loss fails to converge.
- **Seeding.** Python, NumPy, PyTorch and CuDNN are all seeded and CuDNN's benchmark autotuner is disabled — without that, identical code gives slightly different scores between runs.
- **Label normalisation.** Mapping `{'spam': 1, 'ham': 0}` turns any other spelling into `NaN`, which then fails inside the loss function. Unknown labels raise immediately instead.
- **Charts are saved, never shown.** `plt.show()` hangs a headless run, so `src/plots.py` uses the `Agg` backend and only writes PNGs.

---

## Citation

```bibtex
@article{ali2026tinybert,
  title   = {4-Layer BERT-mini Spam Classifier: Design, Implementation, and Evaluation},
  author  = {Ali, Ayan and Majeed, Muhammad Rashid and Shah, Farhan},
  journal = {School of Artificial Intelligence,
             Nanjing University of Information Science and Technology},
  year    = {2026}
}
```

---

## License

MIT — see [LICENSE](LICENSE).

The dataset is a re-upload of the [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection); refer to UCI's terms for how it may be used.
