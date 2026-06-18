# VSFC Ensemble Framework Workflow

Native **3-class sentiment** classification on UIT-VSFC using the same stacked-ensemble design as `tm_research/ensemble/` (VSMEC emotions): five base models → accuracy-normalized weighted average → QLoRA Gemma-2-9B-it meta-model.

**Not included:** VSMEC transfer, 7→3 emotion collapse, or reusing VSMEC checkpoints. All models train on VSFC train/val/test splits from `VSFC_DataPreprocessing.ipynb`.

---

## Prerequisites

1. Run [VSFC_DataPreprocessing.ipynb](../VSFC_DataPreprocessing.ipynb) → `data/processed/vsfc/vsfc_{train,val,test}_final.csv`
2. Run notebooks **01 → 07** in order (Colab: use `tmp_root='/content/vsfc_ensemble_tmp'` so artifacts do not clash with `ensemble/`)
3. Optional: notebook **08** — batch demo inference on the test split (or a custom CSV)

---

## Architecture

| Stage | VSMEC (`ensemble/`) | VSFC (`VSFC_ensemble/`) |
|-------|---------------------|-------------------------|
| Labels | 7 emotions | `negative`, `neutral`, `positive` |
| Data | `data/processed/*_final.csv` | `data/processed/vsfc/vsfc_*_final.csv` |
| Prob dims | `(N, 7)` | `(N, 3)` |
| Prompt abbrev | A, D, J, … | `neg`, `neu`, `pos` |
| Meta completion | `<label>Enjoyment</label>` | `<label>positive</label>` |

Notebook order:

| # | Notebook | Output |
|---|----------|--------|
| 01–03 | PhoBERT, CafeBERT, ViBERT OOF | `artifacts/probs/{name}_{oof,val,test}.npy` |
| 04 | LogReg + SVC OOF | same |
| 05 | Weights + meta JSONL | `weights.json`, `meta_jsonl/*.jsonl` |
| 06 | QLoRA Gemma-2-9B-it | `lora_adapter/` |
| 07 | Evaluation + test export | `metrics/ensemble_summary.json`, `test_predictions.csv` |
| 08 | Demo batch predictions (optional) | `demo_predictions.csv` |

---

## Artifact layout

```
VSFC_ensemble/artifacts/
├── label_map.json
├── weights.json              # includes system_prompt, prompt_schema
├── probs/
│   ├── phobert_{oof,val,test}.npy
│   └── … (cafebert, vibert, logreg, svc)
├── meta_jsonl/
│   ├── train.jsonl           # OOF probs + gold completions
│   ├── val.jsonl
│   └── test.jsonl
├── metrics/
│   └── ensemble_summary.json
├── test_predictions.csv        # per-example gold vs weighted vs LoRA (notebook 07)
├── demo_predictions.csv        # batch demo output: weighted + LoRA preds (notebook 08)
└── lora_adapter/
```

Colab env vars: `TM_VSFC_ENSEMBLE_ARTIFACTS_DIR`, `TM_VSFC_ENSEMBLE_PERSISTENT_ARTIFACTS_DIR`.

---

## Evaluation systems (notebook 07)

1. `baseline_weighted_avg_argmax`
2. `baseline_best_single_{name}`
3. `zero_shot_gemma`
4. `lora_gemma_meta`

Hyperparameters match [ensemble/WORKFLOW.md](../ensemble/WORKFLOW.md) unless noted; `MAX_SEQ_LEN=800` for LoRA is sufficient for 3-class compact_v1 prompts.
