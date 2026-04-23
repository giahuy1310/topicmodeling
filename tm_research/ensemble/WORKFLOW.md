# Ensemble Framework Workflow

Vietnamese emotion classification using a **stacked ensemble**: five heterogeneous base models → accuracy-normalized weighted average → QLoRA-fine-tuned Gemma-2-9B-it meta-model.

---

## Architecture Overview

```
                 ┌──────────────────────────────────────────────────┐
                 │              DATA SPLITS                         │
                 │  train_1500_final.csv / val_final.csv /          │
                 │  test_final.csv  →  LabelMap (alphabetical)      │
                 └──────────────────┬───────────────────────────────┘
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          │           BASE MODEL TIER (notebooks 01–04)        │
          │                                                     │
          │  PhoBERT   CafeBERT   ViBERT   LogReg   LinearSVC  │
          │     ↓          ↓        ↓        ↓         ↓       │
          │  5-fold OOF probabilities  (N_train × C)           │
          │  full-train refit val/test probs (N × C)           │
          └─────────────────────────┬───────────────────────────┘
                                    │
          ┌─────────────────────────▼───────────────────────────┐
          │        STACKING LAYER (notebook 05)                  │
          │                                                      │
          │  Compute weights  w_i = val_acc_i / Σ val_acc_j     │
          │  Build structured-token prompt (text + stack block   │
          │  + weighted average) → JSONL for train/val/test      │
          └─────────────────────────┬───────────────────────────┘
                                    │
          ┌─────────────────────────▼───────────────────────────┐
          │        META-MODEL (notebook 06)                      │
          │                                                      │
          │  QLoRA fine-tune Gemma-2-9B-it on structured JSONL  │
          │  Completion-only loss → learns to output             │
          │  <label>EMOTION</label>                              │
          └─────────────────────────┬───────────────────────────┘
                                    │
          ┌─────────────────────────▼───────────────────────────┐
          │        EVALUATION (notebook 07)                      │
          │                                                      │
          │  LoRA meta-model vs. baselines on test set           │
          │  (weighted-avg argmax, best single model, zero-shot) │
          └─────────────────────────────────────────────────────┘
```

---

## Stage-by-Stage Workflow

### Stage 1 — Data Loading (`utils_io.py`)

`load_splits()` reads the three CSV files, normalizes column names to `text` / `label`, and builds a stable `LabelMap` (labels sorted alphabetically so integer IDs are consistent across all notebooks). The label map is persisted to `artifacts/label_map.json`.

**Artifact produced:** `artifacts/label_map.json`

---

### Stage 2 — BERT Base Models (notebooks 01, 02, 03)

Each notebook calls `run_bert_oof(cfg, train_df, val_df, test_df, label_map)` with a different `model_name` / `output_name`. The three models are:

| Notebook | Model | `output_name` |
|----------|-------|---------------|
| 01 | `vinai/phobert-base-v2` | `phobert` |
| 02 | `uitnlp/CafeBERT` | `cafebert` |
| 03 | `FPTAI/vibert-base-cased` | `vibert` |

**OOF loop** (`kfold_oof_probs` in `utils_stacking.py`):

1. Split `train_df` into `n_folds` stratified folds.
2. For each fold: fine-tune on the in-fold rows, predict soft probabilities on the out-of-fold rows.
3. Assemble a full-train `(N_train, C)` OOF probability matrix.

**Full-train refit**: After OOF, the model is retrained on all of `train_df` and used to predict `val_df` and `test_df`.

**Artifacts produced per model:**
- `artifacts/probs/{name}_oof.npy` — OOF soft probs (N_train × C)
- `artifacts/probs/{name}_val.npy` — val soft probs (N_val × C)
- `artifacts/probs/{name}_test.npy` — test soft probs (N_test × C)
- `artifacts/metrics/{name}.json` — OOF / val / test metrics + tuning snapshot

---

### Stage 3 — Sklearn Base Models (notebook 04)

Two classical models (LogReg, LinearSVC) share the same TF-IDF pipeline and run through the same `kfold_oof_probs` helper. LinearSVC is wrapped in `CalibratedClassifierCV` to produce calibrated probabilities.

**Artifacts produced:** same pattern — `logreg_*.npy`, `svc_*.npy`, and their `.json` metrics.

---

### Stage 4 — Weights and Meta-Dataset (notebook 05)

1. Loads all five `*_val.npy` probability arrays.
2. Computes per-model validation accuracy and derives normalized weights:
   ```
   w_i = val_acc_i / Σ_j val_acc_j
   ```
3. Formats a structured-token prompt for every example in train / val / test:
   ```
   [TEXT] … [/TEXT]
   [STACK]
   <phobert w=0.205>{joy=0.910, sadness=0.040, …}
   <cafebert w=0.198>…
   …
   [/STACK]
   [WEIGHTED_AVG]{joy=0.870, …}[/WEIGHTED_AVG]
   ```
   - **Train** rows use **OOF probs** (never seen by the model that produced them).
   - **Val / Test** rows use full-train-refit probs.
   - Train and val rows include the gold `<label>EMOTION</label>` completion for SFT; test rows store the gold in a separate `gold` field.

**Artifacts produced:**
- `artifacts/weights.json`
- `artifacts/meta_jsonl/train.jsonl`
- `artifacts/meta_jsonl/val.jsonl`
- `artifacts/meta_jsonl/test.jsonl`

---

### Stage 5 — LoRA Meta-Model Training (notebook 06)

Gemma-2-9B-it is loaded in **4-bit NF4** quantization and fitted with a LoRA adapter using `SFTTrainer` (TRL). The loss is restricted to the assistant span only (`completion_only_loss=True`), so the model learns only to produce `<label>EMOTION</label>`.

**Artifacts produced:**
- `artifacts/lora_adapter/` — LoRA adapter weights + tokenizer
- `artifacts/lora_train_meta.json` — training metadata

---

### Stage 6 — Evaluation (notebook 07)

The test set is evaluated against four systems:

| System | Description |
|--------|-------------|
| `baseline_weighted_avg_argmax` | argmax of the 5-model weighted average |
| `baseline_best_single_{name}` | argmax of the single best base model (by val acc) |
| `zero_shot_gemma` | Gemma-2-9B-it with the same prompts but no LoRA |
| `lora_gemma_meta` | LoRA-adapted Gemma-2-9B-it (the full system) |

Metrics: accuracy, macro recall, macro F1, weighted F1, per-class F1, confusion matrix.

**Artifacts produced:**
- `artifacts/metrics/ensemble_summary.json`
- `Confusion_matrix/ensemble_llm_meta.png`

---

## Hyperparameter Reference

### BERT Base Models — `BertOOFConfig`

| Parameter | Default (notebooks 01–03) | Meaning |
|-----------|---------------------------|---------|
| `model_name` | varies per notebook | HuggingFace checkpoint to fine-tune |
| `output_name` | `phobert` / `cafebert` / `vibert` | Prefix for all saved artifact files |
| `seed` | random (drawn fresh each run) | Controls fold splits, weight init, data shuffling |
| `n_folds` | `5` | Number of stratified K-fold splits for OOF |
| `num_epochs` | `20` | Maximum fine-tuning epochs per fold and full-train refit |
| `learning_rate` | `2e-5` | AdamW peak learning rate |
| `lr_scheduler_type` | `"cosine"` | LR decay schedule after warmup |
| `warmup_ratio` | `0.1` | Fraction of total steps used for linear warmup |
| `weight_decay` | `0.01` | L2 regularization applied to all non-bias parameters |
| `train_batch_size` | `32` | Per-device training batch size |
| `eval_batch_size` | `64` | Per-device evaluation / inference batch size |
| `grad_accum_steps` | `2` | Gradient accumulation steps (effective batch = 32 × 2 = 64) |
| `max_length` | `256` | Tokenizer truncation / padding length |
| `early_stopping_patience` | `5` | Stop if monitored metric does not improve for this many epochs |
| `metric_for_best_model` | `"f1"` | Metric used by `EarlyStoppingCallback` and checkpoint selection |
| `max_grad_norm` | `1.0` | Gradient clipping threshold |
| `adam_beta1` | `0.9` | AdamW first-moment decay |
| `adam_beta2` | `0.999` | AdamW second-moment decay |
| `adam_epsilon` | `1e-8` | AdamW numerical stability constant |
| `fp16` / `bf16` | auto-detected | Mixed-precision: `bf16` on Ampere+ GPUs, `fp16` otherwise |
| `work_dir` | `"./_bert_oof_work"` | Temporary directory for HuggingFace trainer checkpoints |
| `extra_tokenizer_kwargs` | `{}` | Passed verbatim to `AutoTokenizer.from_pretrained` |
| `extra_model_kwargs` | `{}` | Passed verbatim to `AutoModelForSequenceClassification.from_pretrained` |

---

### Sklearn Base Models (notebook 04)

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `ngram_range` | `(1, 2)` | TF-IDF uses unigrams and bigrams |
| `min_df` | `2` | Ignore terms appearing in fewer than 2 documents |
| `sublinear_tf` | `True` | Apply `1 + log(tf)` instead of raw term frequency |
| **LogReg** `C` | `1.0` | Inverse regularization strength (L2) |
| **LogReg** `solver` | `"lbfgs"` | Optimization algorithm |
| **LogReg** `max_iter` | `1000` | Maximum solver iterations |
| **LinearSVC** `C` | `1.0` | Regularization parameter |
| **CalibratedClassifierCV** `method` | `"sigmoid"` | Platt scaling for probability calibration |
| **CalibratedClassifierCV** `cv` | `3` | Inner CV folds used to fit the calibrator |
| `SEED` | `123` | Fixed random seed for sklearn models |
| `N_FOLDS` | `5` | Stratified K-fold splits |

---

### Weight Computation

Three strategies are implemented in `utils_stacking.py`; notebook 05 uses `compute_weights` (linear):

| Function | Formula | When to use |
|----------|---------|-------------|
| `compute_weights` | `w_i = acc_i / Σ acc_j` | Default — proportional to validation accuracy |
| `compute_weights_softmax` | `w_i = softmax(acc_i / T)` | `temperature < 1` sharpens toward the best model |
| `compute_weights_lsq` | NNLS minimizing log-loss on val | Optimal weights when base models are miscalibrated |

---

### LoRA Meta-Model — Gemma-2-9B-it (notebook 06)

**4-bit quantization (BitsAndBytes):**

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `load_in_4bit` | `True` | Quantize weights to 4-bit NF4 at load time |
| `bnb_4bit_quant_type` | `"nf4"` | Normal Float 4 — better distribution than int4 |
| `bnb_4bit_compute_dtype` | `bfloat16` | Upcast to bf16 for matrix multiplications |
| `bnb_4bit_use_double_quant` | `True` | Quantize the quantization constants (saves ~0.4 bits/param) |

**LoRA adapter:**

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `r` | `16` | Rank of the low-rank decomposition matrices |
| `lora_alpha` | `32` | Scaling factor (`alpha / r = 2.0` effective LR multiplier) |
| `lora_dropout` | `0.05` | Dropout applied inside LoRA layers |
| `bias` | `"none"` | Bias terms are not adapted |
| `target_modules` | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` | Attention + MLP projections are adapted |
| `task_type` | `"CAUSAL_LM"` | Causal language modelling head |

**SFT training:**

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `num_train_epochs` | `2` | Short fine-tune to avoid over-fitting the small JSONL dataset |
| `per_device_train_batch_size` | `16` | Sequences per GPU per step |
| `gradient_accumulation_steps` | `2` | Effective batch size = 32 |
| `learning_rate` | `1e-4` | Higher than BERT fine-tuning because only LoRA params are trained |
| `lr_scheduler_type` | `"cosine"` | Cosine decay |
| `warmup_ratio` | `0.1` | Linear warmup for first 10 % of steps |
| `weight_decay` | `0.0` | No L2 on LoRA params |
| `bf16` | `True` | bfloat16 training |
| `packing` | `False` | Each sample is padded independently (avoids cross-contamination) |
| `completion_only_loss` | `True` | Cross-entropy loss applied only to the `<label>…</label>` span |
| `MAX_SEQ_LEN` | `1024` | Maximum token length per example |

**Inference (notebook 07):**

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `max_new_tokens` | `12` | Budget for the `<label>EMOTION</label>` token span |
| `do_sample` | `False` | Greedy decoding |
| `batch_size` | `4` | Examples processed per forward pass during evaluation |

---

## Artifact Directory Layout

```
artifacts/
├── label_map.json                  # label2id / id2label
├── weights.json                    # per-model val accuracy + normalized weights
├── lora_train_meta.json            # base model name, system prompt, paths
├── probs/
│   ├── phobert_oof.npy             # (N_train, C)
│   ├── phobert_val.npy             # (N_val, C)
│   ├── phobert_test.npy            # (N_test, C)
│   ├── cafebert_{oof,val,test}.npy
│   ├── vibert_{oof,val,test}.npy
│   ├── logreg_{oof,val,test}.npy
│   └── svc_{oof,val,test}.npy
├── metrics/
│   ├── phobert.json
│   ├── cafebert.json
│   ├── vibert.json
│   ├── logreg.json
│   ├── svc.json
│   └── ensemble_summary.json
├── meta_jsonl/
│   ├── train.jsonl                 # OOF probs + gold completions
│   ├── val.jsonl                   # full-train probs + gold completions
│   └── test.jsonl                  # full-train probs, gold in separate field
└── lora_adapter/                   # LoRA weights + tokenizer
```
