# Overall Process: Training and Testing BERT Models

This document summarizes the end-to-end workflow used in:

- `tm_research/EmoModel_Advanced_PhoBERT_v2_Test.ipynb`
- `tm_research/EmoModel_Advanced_CafeBert_Test.ipynb`
- `tm_research/EmoModel_Advanced_VisoBERT_Test.ipynb`

The three notebooks follow the same high-level pattern: optional quick-start inference, configuration, data loading and preprocessing, tokenization, model training with Hugging Face `Trainer`, evaluation on test data, and model export to Google Drive.

## 1) Execution Modes

Each notebook supports two modes via `SKIP_TRAINING`:

- `False` (default): run full training + evaluation.
- `True`: load a previously trained model from Google Drive for quick inference/testing.

Quick-start loading includes:
- Loading `AutoModelForSequenceClassification` and tokenizer from saved model path.
- Loading `label_mappings.json` (`label2id`, `id2label`).
- Moving model to `cuda` (if available) or `cpu`.

## 2) Configuration

Common configuration pattern:

- `DATA_DIR = '/content/drive/MyDrive/thesis/data'`
- Batch and optimization settings:
  - `PER_DEVICE_TRAIN_BATCH_SIZE = 64`
  - `PER_DEVICE_EVAL_BATCH_SIZE = 128`
  - `GRADIENT_ACCUMULATION_STEPS = 1`
  - `NUM_EPOCHS = 20`
  - `LEARNING_RATE = 2e-5`
  - `WARMUP_RATIO = 0.1`
  - `WEIGHT_DECAY = 0.01`
- Sequence length:
  - PhoBERT-v2 and ViSoBERT define `MAX_LENGTH = 256`.
  - CafeBERT tokenization also uses `max_length=256`.
- Optional Vietnamese segmentation:
  - `USE_UNDERTHESEA_TOKENIZER = False` by default.
  - If enabled and package is available, text is segmented with `underthesea.word_tokenize(..., format='text')`.

Model-specific identifiers:

- PhoBERT-v2: `MODEL_NAME = 'vinai/phobert-base-v2'`
- CafeBERT: `MODEL_NAME = 'uitnlp/CafeBERT'`
- ViSoBERT: `MODEL_NAME = 'uitnlp/visobert'`

## 3) Environment and Tracking

All notebooks:

- Install runtime dependencies in Colab (`transformers`, `datasets`, `torch`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `openpyxl`, `underthesea`, `wandb`).
- Initialize Weights & Biases (`wandb.init(...)`) and log metrics.
- Enable mixed precision (`fp16`) when CUDA is available.

## 4) Data Loading and Preparation

### PhoBERT-v2 and ViSoBERT

Data files:

- Train: `processed/train_1500_para_final.csv`
- Validation: `processed/val_processed.csv`
- Test: `processed/test_gen_final.csv`

Processing:

- Rename columns into standard schema: `text`, `label`.
- Clean rows (`dropna`) and strip string fields.
- Optional Underthesea tokenization on `text`.
- Build class mappings from training labels:
  - `label2id = {label: idx}`
  - `id2label = {idx: label}`

### CafeBERT

Data files:

- Train: `processed/train_10000_final.csv`
- Validation: `processed/val_processed.csv`
- Test: `processed/test_processed.csv`

Processing:

- Reshape columns to `text`, `label` (train drops extra columns first).
- Clean rows (`dropna`) and strip string fields.
- Optional Underthesea tokenization.
- Build mappings from unique labels and add numeric label column `labels` to each split.

## 5) Dataset Conversion and Tokenization

All notebooks:

- Convert pandas DataFrames to Hugging Face `Dataset`.
- Load tokenizer + sequence classification model with:
  - `num_labels`
  - `id2label`
  - `label2id`
- Tokenize text using:
  - `padding='max_length'`
  - `truncation=True`
  - `max_length=256`

Label field differs by notebook:

- PhoBERT-v2 / ViSoBERT use `label`.
- CafeBERT uses `labels`.

## 6) Training Setup

Core training behavior in all notebooks:

- Train using Hugging Face `Trainer`.
- Evaluate each epoch (`eval_strategy='epoch'`).
- Save each epoch (`save_strategy='epoch'`).
- Keep best checkpoint (`load_best_model_at_end=True`).
- Use early stopping callback.
- Apply cosine LR scheduling.

Metric functions:

- PhoBERT-v2 / ViSoBERT:
  - Return `accuracy`, weighted `precision`, weighted `recall`, weighted `f1`.
  - Best-model metric: `f1`.
- CafeBERT:
  - Return `accuracy`, `f1_macro`, `f1_weighted`.
  - Best-model metric: `f1_macro`.

Early stopping:

- PhoBERT-v2 / ViSoBERT: patience = 5.
- CafeBERT: `EARLY_STOPPING_PATIENCE = 7`.

## 7) Training Execution

### PhoBERT-v2 and ViSoBERT

- Single-run test configuration (`RANDOM_SEEDS = [42]`, `NUM_RUNS = 1`).
- Build one `Trainer` and execute `trainer.train()`.

### CafeBERT

- Loop over `SEEDS` (configured as `[42]` in test mode, but designed for multi-seed runs).
- For each seed:
  - Reset random seeds.
  - Reinitialize model weights.
  - Train and evaluate.
  - Store run metrics in `all_results`.
- Keep best model in memory (`best_model_ref`) using weighted F1 on test results as selection criterion.

## 8) Test Evaluation

All notebooks evaluate on held-out test set:

- Run `trainer.predict(...)` (or prediction trainer for best in-memory model in CafeBERT).
- Compute aggregate metrics (`accuracy`, F1 variants, precision/recall where defined).
- Print full `classification_report`.
- Build and plot confusion matrix with Seaborn heatmap.
- Log test metrics to W&B.

Additional outputs in PhoBERT-v2 and ViSoBERT:

- Per-class probability extraction from logits.
- Export prediction probabilities with source sentences to CSV.

## 9) Model Saving and Versioning

All notebooks save model artifacts to Google Drive with versioned naming:

- Model weights (`save_pretrained` / `trainer.save_model`)
- Tokenizer
- `label_mappings.json`
- `model_metadata.json` (training params and run metadata)

Version naming:

- PhoBERT-v2: `PhoBERT_v2_Advanced_Test_seed{seed}_{timestamp}`
- ViSoBERT: `ViSoBERT_Advanced_Test_seed{seed}_{timestamp}`
- CafeBERT: `CafeBert_Advanced_Test_{data_file}_seed{best_seed}_{timestamp}`

## 10) Outputs and Deliverables

Each run produces:

- Best trained checkpoint and tokenizer on Google Drive.
- Label mapping and metadata JSON files for reproducible inference.
- Test-set metrics and confusion matrix.
- W&B run history (training/evaluation logs).
- (PhoBERT-v2 and ViSoBERT) prediction probability CSV export.

---

## Quick Comparison Table

| Aspect | PhoBERT-v2 | CafeBERT | ViSoBERT |
|---|---|---|---|
| Base model | `vinai/phobert-base-v2` | `uitnlp/CafeBERT` | `uitnlp/visobert` |
| Train file | `train_1500_para_final.csv` | `train_10000_final.csv` | `train_1500_para_final.csv` |
| Test file | `test_gen_final.csv` | `test_processed.csv` | `test_gen_final.csv` |
| Label field in HF Dataset | `label` | `labels` | `label` |
| Best metric for checkpoint | `f1` (weighted) | `f1_macro` | `f1` (weighted) |
| Early stopping patience | 5 | 7 | 5 |
| Multi-seed loop support | Not emphasized | Yes (best seed/model selection) | Not emphasized |
| Extra probability CSV export | Yes | No | Yes |

