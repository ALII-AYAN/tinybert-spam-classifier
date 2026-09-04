# Model checkpoints

Put your trained weights here.

## Where exactly

```
models/
└── tinybert_spam_classifier.pt    <- default name every script looks for
```

If your file is named `tinybert_spam_classifier_withNN.pth` (the name used by the
original single-file script), either rename it or tell the scripts where it is:

```bash
# option A - rename (then no flags needed anywhere)
mv models/tinybert_spam_classifier_withNN.pth models/tinybert_spam_classifier.pt

# option B - keep the name and point at it explicitly
python -m src.evaluate --checkpoint models/tinybert_spam_classifier_withNN.pth
python -m src.predict  --checkpoint models/tinybert_spam_classifier_withNN.pth \
                       --text "WINNER!! Claim your free prize now"
python -m app.gui      --checkpoint models/tinybert_spam_classifier_withNN.pth
```

## Legacy `.pth` files load fine

Your file was saved with `torch.save(model.state_dict())`, i.e. a **raw
state_dict**. `src/model.py::load_model` detects that and rebuilds the
architecture around it, so no conversion is required.

If you want the self-describing format instead (it stores `model_name` + config
so the weights can never be loaded into the wrong architecture):

```bash
python scripts/convert_legacy_checkpoint.py \
    models/tinybert_spam_classifier_withNN.pth \
    models/tinybert_spam_classifier.pt
```

## Optional but recommended: save the tokenizer next to it

Without it, inference downloads the tokenizer from Hugging Face on first run
(fails on an offline machine).

```bash
python -c "from transformers import AutoTokenizer; \
AutoTokenizer.from_pretrained('prajjwal1/bert-mini').save_pretrained('models/tokenizer')"
```

`src/predict.py` and `app/gui.py` automatically prefer `models/tokenizer/` when
it exists.

## Resulting layout

```
models/
├── tinybert_spam_classifier.pt
└── tokenizer/                <- optional
    ├── tokenizer.json
    ├── vocab.txt
    └── tokenizer_config.json
```

## Important: these files are gitignored

`.gitignore` excludes `models/**`, `*.pt` and `*.pth` on purpose — binaries do not
belong in git history. To publish your weights, pick one:

```bash
# 1. Git LFS (keeps them in the repo, downloadable on clone)
git lfs install
git lfs track "models/*.pt"
git add .gitattributes models/tinybert_spam_classifier.pt

# 2. GitHub Release asset (cleanest for a ~17 MB file)
#    GitHub web UI -> Releases -> attach the .pt file,
#    then link to it from the README.

# 3. Force-add (small files only)
git add -f models/tinybert_spam_classifier.pt
```
