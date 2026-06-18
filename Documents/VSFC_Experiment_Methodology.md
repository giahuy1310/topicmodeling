# VSFC Experiment — Reproducibility Methodology

This section documents the UIT-VSFC stacked-ensemble experiment (`tm_research/VSFC_ensemble/`, notebooks 01–07). It covers only what is required to reproduce the run: label space, base-model training policy, meta-model rebuild, and data splits with evaluation metrics.

---

## 1. Label mapping and label-space adjustment

**Task.** Native **3-class sentiment** classification on UIT-VSFC. There is **no transfer from VSMEC**: no 7-class emotion labels, no emotion-to-polarity collapse, and no reuse of VSMEC checkpoints or artifacts.

**Source labels.** The corpus is loaded from Hugging Face as `uitnlp/vietnamese_students_feedback` (`revision="refs/convert/parquet"`). Each row has an integer `sentiment` field mapped to strings as follows (`tm_research/eval/vsfc_labels.py`):

| Integer (`sentiment`) | String label |
|----------------------|--------------|
| 0 | `negative` |
| 1 | `neutral` |
| 2 | `positive` |

**Model-facing label IDs.** After preprocessing, integer IDs used by all classifiers are assigned by **alphabetical sort** of the string labels:

| ID | Label |
|----|-------|
| 0 | `negative` |
| 1 | `neutral` |
| 2 | `positive` |

This map is written to `data/processed/vsfc/label_map.json` during export and re-derived consistently in `load_vsfc_splits()`.

**Meta-model prompt encoding.** The stacked LLM uses schema `compact_v1`:

- Label abbreviations in probability blocks: `neg`, `neu`, `pos`
- Model abbreviations: `p` (PhoBERT), `c` (CafeBERT), `v` (ViBERT), `l` (LogReg), `s` (LinearSVC)
- Gold / predicted completion format: `<label>positive</label>` (full string label, not abbreviation)

Probabilities are rounded to **2 decimal places** in prompts.

---

## 2. Base models: retrained or evaluated only

**All five base models are retrained on VSFC training data.** Pre-trained Hugging Face weights are used only as initialization for fine-tuning; no VSFC/VSMEC ensemble checkpoint is loaded.

| # | Model | HF checkpoint | Training procedure |
|---|-------|---------------|-------------------|
| 01 | PhoBERT | `vinai/phobert-base-v2` | 3-fold stratified OOF on train → full-train refit → predict val/test |
| 02 | CafeBERT | `uitnlp/CafeBERT` | same |
| 03 | ViBERT | `FPTAI/vibert-base-cased` | same |
| 04 | LogReg | TF-IDF + `LogisticRegression` | same OOF + refit pattern |
| 04 | LinearSVC | TF-IDF + `CalibratedClassifierCV(LinearSVC)` | same OOF + refit pattern |

**Out-of-fold (OOF) protocol** (`kfold_oof_probs`, `StratifiedKFold`, `n_splits=3`):

1. Split the **training set** into 3 stratified folds.
2. For each fold: fit on in-fold rows, output soft probabilities on out-of-fold rows.
3. Concatenate to form `(N_train, 3)` OOF probability matrices saved as `artifacts/probs/{name}_oof.npy`.

**Full-train refit** (after OOF):

1. Retrain on **all** training rows.
2. Predict soft probabilities on validation and test → `{name}_val.npy`, `{name}_test.npy`.

**Key hyperparameters (BERT, notebooks 01–03):**

- `num_epochs=20`, `learning_rate=2e-5`, `max_length=256`, `n_folds=3`
- `train_batch_size=32`, `grad_accum_steps=2`, `early_stopping_patience=5`
- `metric_for_best_model="f1_macro"` (checkpoint selection during fine-tuning)
- `seed`: drawn fresh per run (`random.SystemRandom().randint(1, 1_000_000)`) and logged in `artifacts/metrics/{name}.json`

**Key hyperparameters (sklearn, notebook 04):**

- TF-IDF: unigrams + bigrams (`ngram_range=(1,2)`), `min_df=2`, `sublinear_tf=True`
- LogReg: `C=1.0`, `solver="lbfgs"`, `max_iter=1000`
- LinearSVC: `C=1.0`, wrapped in `CalibratedClassifierCV(method="sigmoid", cv=3)` for probabilities
- Fixed `SEED=123`, `N_FOLDS=3`

**Text preprocessing** (`VSFC_DataPreprocessing.ipynb` → `preprocess_vietnamese_text`): same Vietnamese cleaning pipeline as VSMEC (emoji normalization, URL/email removal, punctuation handling). **Stopwords are not removed** (negation words such as *không* are kept for BERT).

---

## 3. Rebuilding the stacked / meta-model for VSFC

The VSFC ensemble is built **end-to-end on VSFC data**; the VSMEC meta-model is not adapted or fine-tuned further.

**Pipeline order:**

1. `VSFC_DataPreprocessing.ipynb` → CSVs under `data/processed/vsfc/`
2. Notebooks **01–04** → base-model probability artifacts
3. Notebook **05** → ensemble weights + meta JSONL
4. Notebook **06** → QLoRA SFT of Gemma-2-9B-it
5. Notebook **07** → held-out test evaluation

### 3.1 Ensemble weights (notebook 05)

Per-model **validation accuracy** is computed from `{name}_val.npy` argmax predictions. Weights are accuracy-normalized:

```
w_i = val_acc_i / Σ_j val_acc_j
```

Saved to `artifacts/weights.json` together with the system prompt (`build_meta_system_prompt`).

### 3.2 Meta-dataset construction (notebook 05)

Structured prompts are rendered for every example (`format_prompt`, schema `compact_v1`):

```
[TEXT] … [/TEXT]
[WEIGHTED_AVG]{neg=0.12,neu=0.45,pos=0.43}[/WEIGHTED_AVG]
[STACK]
p:{neg=0.10,neu=0.50,pos=0.40}
c:{…}
…
[/STACK]
```

| Split | Probability source | Gold completion in JSONL |
|-------|-------------------|--------------------------|
| **Train** | OOF probabilities (leakage-safe) | Yes — `<label>…</label>` |
| **Validation** | Full-train-refit probabilities | Yes |
| **Test** | Full-train-refit probabilities | No — gold stored in separate `gold` field |

Outputs: `artifacts/meta_jsonl/{train,val,test}.jsonl`.

### 3.3 Meta-model training (notebook 06)

A **new** LoRA adapter is trained from the VSFC meta JSONL; the VSMEC LoRA weights are not used.

| Setting | Value |
|---------|-------|
| Base LLM | `google/gemma-2-9b-it` (4-bit NF4 QLoRA) |
| LoRA | `r=16`, `alpha=32`, `dropout=0.05`; targets attention + MLP projections |
| SFT | `num_train_epochs=2`, `lr=1e-4`, effective batch 32 |
| `max_seq_len` | 800 |
| Loss | Completion-only (`completion_only_loss=True`) — model learns to emit `<label>…</label>` |
| Train / eval data | `meta_jsonl/train.jsonl` / `meta_jsonl/val.jsonl` |
| Test | **Not used** during meta training (held for notebook 07) |

Artifact: `artifacts/lora_adapter/` + `artifacts/lora_train_meta.json`.

---

## 4. Train / validation / test split and metric choice

### 4.1 Data splits

**Partition source.** The official UIT-VSFC Hugging Face splits are preserved without re-partitioning:

| Split | HF key | Official size |
|-------|--------|---------------|
| Train | `train` | 11,426 |
| Validation | `validation` | 1,583 |
| Test | `test` | 3,166 |

The corpus authors used a **proportional (stratified) partition** so each split preserves the corpus-wide class distribution. This pipeline keeps that property end-to-end.

**Preprocessing adjustments (within each split only):**

- Deduplicate on cleaned `text` (`keep="first"`) — never across splits
- Shuffle **train only** with `random_state=42`
- Val and test keep original order

Final CSVs: `data/processed/vsfc/vsfc_{train,val,test}_final.csv`.

**Cross-validation inside train:**

- OOF folds: `StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)`
- BERT notebooks: fold seed equals the run seed (logged in metrics JSON)
- Sklearn notebook: `random_state=123`

**Leakage control:**

- Test set is untouched until notebook **07**
- Meta-model SFT uses train JSONL (OOF probs) + val JSONL for monitoring; test JSONL is inference-only in evaluation

### 4.2 Metrics

**Base-model checkpoint selection (BERT):** validation **macro-F1** (`metric_for_best_model="f1_macro"`) with early stopping.

**Base-model reporting** (saved per model in `artifacts/metrics/{name}.json`): accuracy, macro recall, macro F1, weighted F1 on OOF / val / test.

**Ensemble weighting:** validation **accuracy** per base model.

**Final test evaluation (notebook 07)** — four systems compared on the held-out test set:

| System | Description |
|--------|-------------|
| `baseline_weighted_avg_argmax` | Argmax of the 5-model weighted average |
| `baseline_best_single_{name}` | Argmax of the single base model with highest val accuracy |
| `zero_shot_gemma` | Gemma-2-9B-it, same prompts, no LoRA |
| `lora_gemma_meta` | LoRA-adapted Gemma-2-9B-it (full stacked system) |

**Reported test metrics:** accuracy, **macro-F1** (primary comparison metric alongside accuracy), micro-F1, per-class precision/recall/F1 (`classification_report`), and confusion matrix.

Results are aggregated in `artifacts/metrics/ensemble_summary.json` and a per-example table in `artifacts/test_predictions.csv`.

---

## Reproduction checklist

1. Run `VSFC_DataPreprocessing.ipynb`.
2. Run `VSFC_ensemble/01` → `07` in order.
3. Record BERT run seeds from each `artifacts/metrics/{phobert,cafebert,vibert}.json` when reporting results.
4. Set `HF_TOKEN` with Gemma-2 license accepted before notebooks 06–07.
5. On Colab, use `tmp_root='/content/vsfc_ensemble_tmp'` so VSFC artifacts stay isolated from the VSMEC `ensemble/` run.
