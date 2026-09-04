# TinyBERT Spam Classifier

A lightweight SMS / email spam detector built on **TinyBERT (`prajjwal1/bert-mini`)** with a small feed-forward classification head, plus a Tkinter desktop demo.

- **Encoder**: pretrained BERT-mini (4.4M params, ~17 MB)
- **Head**: `Linear -> ReLU -> Dropout -> Linear(2)`
- **Input**: `[CLS]` token of the last hidden state
- **Output**: `ham` (0) / `spam` (1) with a calibrated-ish softmax score

---

## Project structure

```
tinybert-spam-classifier/
├── src/                     # library code (importable, no side effects)
│   ├── config.py            # hyper-parameters, paths, label mapping
│   ├── data.py              # CSV loading, tokenisation, DataLoaders
│   ├── model.py             # model definition + save/load checkpoint helpers
│   ├── engine.py            # train_one_epoch / evaluate / predict_proba
│   ├── plots.py             # all matplotlib/seaborn charts (headless-safe)
│   ├── train.py             # CLI: train and export a checkpoint
│   ├── evaluate.py          # CLI: score a checkpoint on the val split
│   ├── predict.py           # CLI: classify a single message or a file
│   └── utils.py             # seeding, device selection, metrics helpers
├── app/
│   └── gui.py               # Tkinter desktop demo
├── data/                    # dataset goes here (not committed)
├── models/                  # checkpoints + tokenizer (see models/README.md)
├── outputs/                 # generated charts and reports
├── scripts/                 # one-off utilities
│   └── convert_legacy_checkpoint.py   # upgrade an old raw `.pth` checkpoint
├── requirements.txt
├── CLEANUP_NOTES.md         # what was removed from the original scripts and why
└── README.md
```

---

## Quick start

```bash
# 1. clone and enter
git clone https://github.com/<your-username>/tinybert-spam-classifier.git
cd tinybert-spam-classifier

# 2. environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. dataset  (see data/README.md for the download link)
#    place it at data/spam.csv with columns: label,text

# 4. train
python -m src.train --data data/spam.csv --epochs 3

# 5. evaluate
python -m src.evaluate --data data/spam.csv --checkpoint models/tinybert_spam_classifier.pt

# 6. predict
python -m src.predict --text "WINNER!! You have won a £1000 cash prize. Reply NOW."

# 7. desktop demo
python -m app.gui
```

### Already have a trained model?

Drop it in `models/` and skip straight to inference:

```bash
# rename it, and every command above works with zero extra flags
mv models/tinybert_spam_classifier_withNN.pth models/tinybert_spam_classifier.pt

# or keep your filename and pass --checkpoint
python -m src.predict --checkpoint models/tinybert_spam_classifier_withNN.pth --text "Free entry to win a car!"
python -m app.gui --checkpoint models/tinybert_spam_classifier_withNN.pth
```

Checkpoints saved by the original single-file script (`torch.save(model.state_dict(), ...)`)
are a **raw state_dict** and load as-is. To upgrade one to the self-describing
format (weights + `model_name` + config):

```bash
python scripts/convert_legacy_checkpoint.py \
    models/tinybert_spam_classifier_withNN.pth \
    models/tinybert_spam_classifier.pt
```

See [`models/README.md`](models/README.md) for details, including how to publish
the weights (they are gitignored by default — use Git LFS or a release asset).

---

## Dataset

| Property | Value |
|---|---|
| Name | UCI **SMS Spam Collection** |
| Size | 5,574 messages |
| Classes | `ham` 4,827 / `spam` 747 |
| Format | CSV with columns `label,text` |

`src/data.py` auto-detects column names (`label`/`v1`, `text`/`v2`, `message`, ...) and normalises label spellings (`Spam`, `spam`, `1`, `0`), so most spam CSVs work without editing code. See [`data/README.md`](data/README.md).

---

## Training options

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
| `--model-name` | `prajjwal1/bert-mini` | Any HF encoder works (`bert-base-uncased`, `distilbert-base-uncased`, ...) |
| `--epochs` | `3` | 3 epochs is usually enough on this dataset |
| `--batch-size` | `16` | Lower it to 8 if you run out of VRAM |
| `--lr` | `2e-5` | Standard fine-tuning LR for BERT |
| `--max-len` | `128` | Truncates long messages |
| `--device` | `auto` | `auto` / `cuda` / `cpu` / `mps` |
| `--no-plots` | off | Skip chart generation |

**Artefacts written after training**

```
models/
├── tinybert_spam_classifier.pt   # weights + model_name + config (best val F1)
├── tokenizer/                    # so inference never re-downloads the tokenizer
└── training_summary.json         # metrics history + hyper-parameters
outputs/
├── class_distribution.png
├── training_history.png
├── confusion_matrix.png
└── classification_report.png
```

The split is **stratified** and the run is **seeded**, so results are reproducible.

---

## Expected performance

Numbers below are indicative for `bert-mini`, 3 epochs, batch 16, lr 2e-5 on the UCI split; your run may differ by ~1 point. Replace them with your own `training_summary.json` values after training.

| Metric | Score |
|---|---|
| Accuracy | ~0.98 |
| Precision (spam) | ~0.95 |
| Recall (spam) | ~0.93 |
| F1 (spam) | ~0.94 |

---

## Using the model in your own code

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
print(result)  # {'label': 'spam', 'label_id': 1, 'confidence': 0.997, 'p_spam': 0.997}
```

---

## Notes

- Checkpoints are saved as a **dict** (`state_dict` + `model_name` + config). `load_model` still accepts a raw `state_dict`, so checkpoints from the original single-file script load fine.
- Inference is batched, and the GUI loads the model once at start-up instead of per click.
- Class imbalance matters here: spam is ~13% of the data, so watch **recall on spam**, not just accuracy.

## License

MIT — see [LICENSE](LICENSE).
