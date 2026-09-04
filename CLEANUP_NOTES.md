# ----------------------------------------------------------------------------
# What was stripped out of the original two scripts, and why.
# Nothing here is required for the model to train, evaluate or predict.
# ----------------------------------------------------------------------------

## A. Kept (core logic - the project actually depends on this)

| Original code | Where it lives now |
|---|---|
| `SpamDataset` | `src/data.py` |
| `TinyBERTSpamClassifier` (BERT + Linear/ReLU/Dropout/Linear head) | `src/model.py` |
| `train_epoch` / `eval_model` | `src/engine.py` |
| `AutoTokenizer`, `AdamW`, `get_linear_schedule_with_warmup` | `src/train.py` |
| Accuracy / precision / recall / F1, confusion matrix, classification report | `src/engine.py`, `src/evaluate.py`, `src/plots.py` |
| `torch.save(model.state_dict())` | `src/model.py::save_model` (now also stores `model_name` + config) |
| Tkinter window, input box, classify button, result label | `app/gui.py` |

## B. Removed (decoration / local-machine-only / harmful on GitHub)

### 1. GUI animations - pure decoration, ~25 lines

```python
def fade_in(window, alpha=0.0): ...      # window.attributes("-alpha", ...)
def bounce_animation(label): ...         # font size flip-flop
def hover_in(event): ...                 # button colour on mouse enter
def hover_out(event): ...
```

- Adds 4 callbacks that do nothing for classification accuracy.
- `fade_in` re-enters `window.after()` every 20 ms; on slow machines it can
  swallow the first user click.

### 2. Hand-drawn gradient background - ~10 lines + 450 canvas lines

```python
gradient = tk.Canvas(root, width=550, height=450)
for i in range(450):
    color = f"#{int(240 - i/10):02x}{int(248 - i/15):02x}{255:02x}"
    gradient.create_line(0, i, 550, i, fill=color)
```

- 450 canvas objects drawn at start-up to fake a gradient.
- Replaced with a plain background colour + a bordered card.

### 3. Hard-coded absolute Windows path - breaks for everyone else

```python
save_path = r"C:\Users\ALI\Desktop\old windows data 33232\NLP Project\working\model\tinybert_spam_classifier_withNN.pth"
```

- Contains a personal username, only works on one machine, and cannot exist on
  Linux/macOS/CI. Replaced with `models/tinybert_spam_classifier.pt`,
  overridable via `--output-dir`.

### 4. `input("Enter dataset path: ")` inside `main()`

- Blocks automation and CI. Replaced with `argparse` (`--data`, with a default).

### 5. `plt.show()` after every figure

- Freezes headless runs; on a server it raises or hangs.
- `src/plots.py` sets `matplotlib.use("Agg")` and only saves PNGs.

### 6. Duplicate / redundant charts

- Six separate figures were created (`class_distribution`, `train_loss_curve`,
  `metrics_per_epoch`, `final_metrics_bar`, `confusion_matrix`,
  `classification_report_heatmap`), many restating the same numbers.
- Trimmed to four: class distribution, training history (loss + 4 metrics in
  one figure), confusion matrix, classification report.

### 7. Emoji identifiers and emoji-heavy console output

- `desc="🚀 Training"`, `print("✅ Model saved ...")`, `📊`, `🧮`, ...
- Cute locally, but they break on terminals without emoji fonts, pollute logs,
  and make diffs noisy. Text-only now.

## C. Bugs fixed while refactoring

| Issue in the original | Fix |
|---|---|
| `train_test_split(..., random_state=42)` without `stratify` | Added `stratify=labels` - keeps the 13% spam ratio in both splits |
| `df['label'].map({'spam': 1, 'ham': 0})` - any other spelling becomes `NaN` | `normalise_label()` handles case, whitespace, `0`/`1`, `not spam`; unknown values raise instead of silently becoming NaN |
| Model was saved from the **last** epoch, not the best | Tracks best validation F1 and restores those weights before saving |
| No `clip_grad_norm_` | Added `max_norm=1.0` |
| `precision_score(...)` without `zero_division` | Added `zero_division=0` (removes the sklearn warning) |
| No seeding of Python/NumPy/CuDNN | `set_seed()` in `src/utils.py` |
| Empty/NULL text rows pass straight through | Dropped in `load_spam_csv` |
| GUI reloads nothing but rebuilds the model per process with no error handling | `load_model` raises a clear message when the checkpoint is missing |
| `model.state_dict()` saved without the model id | Checkpoint now stores `model_name` + config, so `load_model` cannot silently mismatch architectures |
