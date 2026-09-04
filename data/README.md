# Data directory

Put your dataset here as **`data/spam.csv`**.

The dataset itself is **not** tracked by git — see [Why not commit it](#why-not-commit-it) below.

## Download

**SMS Spam Collection** (UCI / Kaggle)

- Kaggle: <https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset>
- UCI (canonical): <https://archive.ics.uci.edu/dataset/228/sms+spam+collection>

Download `spam.csv` and drop it in this folder:

```bash
mv ~/Downloads/spam.csv data/spam.csv
```

## Your file: verified layout

The Kaggle `spam.csv` ships with five columns — two real ones and three empty
filler columns:

```
v1,v2,,,
ham,"Go until jurong point, crazy.. Available only in bugis..."
spam,"Free entry in 2 a wkly comp to win FA Cup final tkts..."
```

`src/data.py` auto-detects `v1` (label) and `v2` (text) and ignores the rest.
Confirmed on the real file:

| Property | Value |
|---|---|
| Rows | 5,572 |
| `ham` | 4,825 |
| `spam` | 747 |
| Spam ratio | **13.41%** |
| Empty texts | 0 |
| Avg. length | 80 chars |
| Longest | 910 chars |

You do **not** need to edit the file or pass `--text-column` / `--label-column`.

## Other formats that work unmodified

Column names are matched case-insensitively, and when nothing matches the
loader falls back to position (first column = label, second = text):

- `v1,v2` — Kaggle / original UCI layout
- `label,text` — the canonical form used throughout this README
- `label,message`
- `class,text`
- `category,email`

Label values are normalised: `ham` / `spam` / `0` / `1` / `not spam`, any casing
or surrounding whitespace. Anything else raises a clear error instead of
silently becoming `NaN`.

## Encoding note

`load_spam_csv` defaults to `encoding="latin-1"`, which is what the UCI file
uses and, unlike UTF-8, never raises on an unexpected byte.

The Kaggle re-upload has **mixed encoding**: most `£` signs are correct latin-1
(`0xA3`), but some rows were double-encoded and show up as garbled characters
such as `澹900 prize` instead of `£900`. This is cosmetic — it affects a handful
of spam messages, and the model treats them as rare tokens. It does not affect
training or metrics in any measurable way.

## Why not commit it

`.gitignore` excludes `data/*.csv`, `data/*.tsv`, `data/*.zip` deliberately:

1. **Licensing** — UCI distributes the corpus for research/educational use.
   Redistributing it inside your repo is a grey area. Linking is not.
2. **It is input, not source** — reviewers clone code, not data. Half a megabyte
   of CSV in git history is permanent bloat.
3. **It is reproducible** — the download link above is a stable, permanent URL.

Keep the file locally so you can train; let the README point everyone else at
the source.

## Optional: ship a tiny sample

If you want reviewers to run the training script **without** downloading
anything, commit a few hundred rows:

```bash
python - <<'PY'
import pandas as pd
from src.data import load_spam_csv

df = load_spam_csv("data/spam.csv")
sample = pd.concat([
    df[df["label"] == 0].sample(n=150, random_state=42),   # ham
    df[df["label"] == 1].sample(n=150, random_state=42),   # spam
]).sample(frac=1, random_state=42).reset_index(drop=True)
sample.to_csv("data/sample.csv", index=False)
print(len(sample), "rows ->  data/sample.csv")
PY

# then allow just this one file
echo '!data/sample.csv' >> .gitignore
```

Train on it with `python -m src.train --data data/sample.csv`.
Expect lower scores — 300 rows is a smoke test, not a benchmark.
