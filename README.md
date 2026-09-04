# TinyBERT Spam Classifier

SMS and email spam detection built on **TinyBERT** (`prajjwal1/bert-mini`) with PyTorch. Fine-tunes a 4.4M-parameter encoder with a small classification head, and ships with a Tkinter desktop app for trying it out.

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Demo

```
$ python -m src.predict --text "WINNER!! You have won a £1000 cash prize. Reply NOW to claim."

[SPAM] p_spam=0.9987 conf=0.9987 :: WINNER!! You have won a £1000 cash prize. Reply NOW to claim.

$ python -m src.predict --text "Hey, are we still on for 6pm tomorrow?"

[HAM] p_spam=0.0012 conf=0.9988 :: Hey, are we still on for 6pm tomorrow?
```

Or run the desktop app:

```bash
python -m app.gui
```

Type a message, hit **Classify** (or `Ctrl+Enter`), and the result panel shows the verdict with the spam probability.

---

## Why TinyBERT

BERT-base gets marginally better scores on this dataset, but it's 110M parameters and needs a GPU to be pleasant to work with. `bert-mini` is 4.4M parameters — about 17 MB — and trains in a few minutes on a laptop CPU while landing within a point of the bigger models. For something you'd actually deploy as a filter, that tradeoff is worth it.

The head is deliberately small: `Linear -> ReLU -> Dropout -> Linear(2)` on top of the `[CLS]` token. Anything wider overfits 5.5k examples.

---

## Setup

```bash
git clone https://github.com/your-username/tinybert-spam-classifier.git
cd tinybert-spam-classifier

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

**Tkinter** (only needed for `app.gui`) ships with CPython on Windows and macOS. On Debian/Ubuntu:

```bash
sudo apt-get install python3-tk
```

---

## Dataset

The [SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) — 5,572 messages, 4,825 ham and 747 spam.

Download `spam.csv` and drop it in `data/`:

```bash
mv ~/Downloads/spam.csv data/spam.csv
```

That's the only setup step. The loader auto-detects the `v1`/`v2` columns the Kaggle file ships with and ignores the three empty filler columns, so the file works as downloaded. It also handles the `label,text` and `label,message` layouts, and normalises label spellings (`Spam`, `not spam`, `1`, `0`, any casing).

Spam is only **13.4%** of the corpus, which shapes how the results should be read — see [Results](#results).

---

## Training

```bash
python -m src.train --data data/spam.csv --epochs 3
```

Runs in roughly 3–4 minutes on a mid-range GPU, or about 15 minutes on a CPU. Everything is driven by flags, so there are no interactive prompts:

```bash
python -m src.train \
  --data data/spam.csv \
  --model-name prajjwal1/bert-mini \
  --epochs 3 \
  --batch-size 16 \
  --lr 2e-5 \
  --max-len 128 \
  --seed 42
```

| Flag | Default | Notes |
|---|---|---|
| `--model-name` | `prajjwal1/bert-mini` | Any HF encoder works — `bert-base-uncased`, `distilbert-base-uncased`, ... |
| `--epochs` | `3` | Three is the sweet spot; a fourth overfits |
| `--batch-size` | `16` | Drop to 8 if you run out of memory |
| `--lr` | `2e-5` | Standard fine-tuning LR for BERT |
| `--max-len` | `128` | Covers the vast majority of messages |
| `--device` | `auto` | `auto` / `cuda` / `cpu` / `mps` |
| `--no-plots` | off | Skip chart generation |

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

Three epochs, `bert-mini`, batch size 16, lr 2e-5, seed 42. Stratified 80/20 split — **1,115** held-out messages.

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| ham | 0.99 | 0.99 | 0.99 | 966 |
| spam | 0.97 | 0.93 | 0.95 | 149 |
| **accuracy** | | | **0.987** | 1115 |
| macro avg | 0.98 | 0.96 | 0.97 | 1115 |

Confusion matrix on the validation split:

|  | predicted ham | predicted spam |
|---|---|---|
| **actual ham** | 961 | 5 |
| **actual spam** | 10 | 139 |

Accuracy alone is a misleading headline here: a model that answered "ham" to everything would still score 86.6%, because ham is 87% of the data. The number that matters is **spam recall — 0.93**, i.e. 10 spam messages out of 149 slipped through. Precision is higher than recall, so the model errs toward the safe side: it would rather let a spam through than flag a real message.

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

Drop it in `models/` and name it `tinybert_spam_classifier.pt` and every command above works with no extra flags. To keep a different filename:

```bash
python -m src.predict --checkpoint models/my_model.pth --text "Free entry to win a car!"
python -m app.gui      --checkpoint models/my_model.pth
```

Checkpoints are saved as a dict containing the weights plus the model id and config, so the weights can't be silently loaded into the wrong architecture. A plain `state_dict` saved with `torch.save(model.state_dict(), path)` loads fine too — and if you want to upgrade an older checkpoint to the self-describing format:

```bash
python scripts/convert_legacy_checkpoint.py models/old_model.pth models/tinybert_spam_classifier.pt
```

Note that `models/` is gitignored. Weights are build artifacts; publish them with Git LFS or as a release asset rather than committing binaries.

---

## Project layout

```
tinybert-spam-classifier/
├── src/
│   ├── config.py            # hyperparameters, paths, label mapping
│   ├── data.py              # CSV loading, tokenisation, DataLoaders
│   ├── model.py             # TinyBERT + classification head, save/load
│   ├── engine.py            # train_one_epoch / evaluate / predict_proba
│   ├── plots.py             # charts (Agg backend, headless-safe)
│   ├── train.py             # CLI: train and export a checkpoint
│   ├── evaluate.py          # CLI: score a checkpoint
│   ├── predict.py           # CLI: classify a message or a file
│   └── utils.py             # seeding, device selection, helpers
├── app/
│   └── gui.py               # Tkinter desktop app
├── scripts/
│   └── convert_legacy_checkpoint.py
├── data/                    # dataset (gitignored)
├── models/                  # checkpoints (gitignored)
├── outputs/                 # generated charts and reports
├── CLEANUP_NOTES.md         # notes on refactoring the original prototype
└── requirements.txt
```

Each module does one job, so swapping the encoder, changing the pooling strategy or adding a metric means touching exactly one file.

---

## Implementation notes

A few decisions that made a measurable difference while building this:

- **Stratified split.** With 13.4% spam, a plain random split can hand the validation set a noticeably different class balance than the training set. Stratifying keeps both at the corpus ratio.
- **Best-F1 checkpointing.** The last epoch isn't automatically the best one. Training tracks validation F1 and saves the weights from the best epoch, not the final one.
- **Seeding.** Python, NumPy, PyTorch and CuDNN are all seeded, and CuDNN's benchmark autotuner is disabled — without that, identical code gives slightly different scores run to run.
- **Label normalisation.** Mapping `{'spam': 1, 'ham': 0}` silently turns any other spelling into `NaN`, which then blows up inside the loss function. Unknown labels raise immediately instead.
- **Gradient clipping** at `max_norm=1.0`, which smooths out the occasional loss spike on long messages.
- **Charts are saved, never shown.** `plt.show()` hangs a headless run, so `src/plots.py` uses the `Agg` backend and only writes PNGs.

---

## License

MIT — see [LICENSE](LICENSE).

The dataset is a re-upload of the [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection); refer to UCI's terms for how it may be used.
