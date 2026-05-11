# Ensemble Learning for Vietnamese Emotion Recognition

This repository contains research code for Vietnamese fine-grained emotion recognition, with **ensemble learning** as the main direction.

The current core pipeline is a **stacked ensemble** in `tm_research/ensemble`, where multiple base classifiers are combined and refined by a meta-model. Other components (data augmentation notebooks, topic modeling CLI, and earlier experiments) are kept as supporting/experimental work.

## Project Focus

- **Main topic**: Ensemble learning for Vietnamese emotion classification.
- **Primary objective**: Improve macro-level and minority-class performance through model combination and meta-learning.
- **Supporting experiments**:
  - LLM-based data augmentation and standalone single-model studies.
  - Topic modeling research CLI (kept for experimentation, not the main thesis track).

## Main Pipeline (Ensemble)

The ensemble workflow is documented in `tm_research/ensemble/WORKFLOW.md`.

High-level flow:

1. Train heterogeneous base models and generate out-of-fold (OOF) probabilities.
2. Build a stacking/meta dataset from model probabilities and weighted aggregates.
3. Fine-tune a meta-model (Gemma + LoRA/QLoRA setup) on structured ensemble signals.
4. Evaluate ensemble systems against single-model and weighted-average baselines.

## Dataset

- **Corpus**: UIT-VSMEC (Vietnamese social media emotion corpus).
- **Typical labels**: `Enjoyment, Sadness, Fear, Anger, Disgust, Surprise, Other`.
- **Working format**:
  - `text`: Vietnamese social media sentence.
  - `label`: emotion category.

## Ensemble Components

The `tm_research/ensemble` directory includes staged notebooks and utilities:

- `01_base_phobert_oof.ipynb`
- `02_base_cafebert_oof.ipynb`
- `03_base_vibert_oof.ipynb`
- `04_base_sklearn_oof.ipynb`
- `05_weights_and_meta_dataset.ipynb`
- `06_train_meta_lora_gemma.ipynb`
- `07_evaluate_ensemble.ipynb`
- `utils_io.py` and related helpers

## Experimental results (notebook / thesis)

The numbers below are the **reported results** from the thesis write-up (`Documents/Thesis.md`, Chapter 4: Experiments and Results), which corresponds to the ensemble and augmentation experiments implemented in this repository’s notebooks.

### Baseline BERT models (original UIT-VSMEC, averaged over five seeds)


| Model            | Accuracy | Weighted F1 |
| ---------------- | -------- | ----------- |
| PhoBERT v2-Large | 0.6075   | 0.602       |
| ViBERT           | 0.6106   | 0.6101      |
| CafeBERT         | 0.6344   | 0.628       |
| XLM-RoBERTa      | 0.5691   | 0.5574      |


CafeBERT is used as the primary model for augmentation experiments.

### Minority-class F1 (Fear, Anger, Surprise) with CafeBERT and Gemini-2.5-Flash augmentation


| Setting     | Fear | Anger | Surprise |
| ----------- | ---- | ----- | -------- |
| Original    | 67%  | 38%   | 60%      |
| Paraphrased | 67%  | 42%   | 61%      |
| Generated   | 68%  | 46%   | 61%      |


### CafeBERT robustness (original vs. augmented training)


| Training data         | Accuracy | Weighted F1 |
| --------------------- | -------- | ----------- |
| Original              | 0.6344   | 0.628       |
| Paraphrased augmented | 0.6676   | 0.6682      |
| Generated augmented   | 0.6703   | 0.671       |


### BERT models trained on augmented data (test set)


| Model    | Test accuracy | Test F1 (weighted) |
| -------- | ------------- | ------------------ |
| CafeBERT | 0.6703        | 0.671              |
| PhoBERT  | 0.6378        | 0.633              |
| ViBERT   | 0.6387        | 0.6375             |
| XLM-R    | 0.6214        | 0.6227             |


### Ensemble vs. baselines (UIT-VSMEC; thesis Table 12)


| Method                                                               | Accuracy   | Weighted F1 |
| -------------------------------------------------------------------- | ---------- | ----------- |
| **Stacked ensemble + Gemma meta-model (thesis: “Genma”) (OUR WORK)** | **0.7023** | **0.7022**  |
| CafeBERT + augmented data                                            | 0.6676     | 0.6682      |
| Gemma zero-shot prompting                                            | 0.6864     | 0.6809      |
| MLR + preprocessing + key-clause extraction                          | 0.6436     | 0.6440      |
| CafeBERT (prior work, thesis ref.)                                   | —          | 0.6612      |
| VisoBERT (prior work, thesis ref.)                                   | 0.6810     | 0.6837      |
| XLM-R-Large (prior work, thesis ref.)                                | 0.6137     | 0.6020      |


Narrative discussion, non-transformer baselines (Tables 9–10), and evaluation metrics setup are in `Documents/Thesis.md` (Chapter 4–5 and abstract).

## Supporting / Experimental Work

These are still useful but are **not the main project direction**:

- `tm_research/Data_Augmentation_LLM.ipynb`
- `tm_research/Data_Augmentation_LLM.md`
- `tm_research/EmoModel_BiLSTM.ipynb`
- `tm_research/EmoModel_Vectorization_Comparison.ipynb`
- `tm_research/Data_eval_comprehensive.ipynb`
- `tm_research/EmoModel_Local.ipynb`

## Topic Modeling CLI (Experimental Component)

The repository also keeps an existing topic modeling research CLI under `tm_research`. This is now considered an auxiliary experiment track.

### Quickstart

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source ./.venv/bin/activate
```

1. Install dependencies:

```bash
pip install -r requirements.txt
```

1. Configure `.env` (for Gemini/LangChain workflows if needed).
2. Run a sample pipeline:

```bash
python -m tm_research.cli run --query "quantum computing benchmarks" --num-results 10 --num-topics 8
```

### CLI Commands

```bash
python -m tm_research.cli ingest --query "<your query>" --num-results 10 --out data/raw.jsonl
python -m tm_research.cli model --input data/raw.jsonl --num-topics 8 --out data/topics.json
python -m tm_research.cli summarize --input data/topics.json --out data/summary.md
python -m tm_research.cli run --query "<your query>" --num-results 10 --num-topics 8
```

Use classic TF-IDF + NMF:

```bash
python -m tm_research.cli model --input data/raw.jsonl --num-topics 8 --method classic --out data/topics.json
python -m tm_research.cli run --query "<your query>" --num-results 10 --num-topics 8 --method classic
```

## Project Structure (Current)

```text
tm_research/
  ensemble/
  cli.py
  modeling.py
  research.py
  ingestion.py
  config.py
data/
  (created at runtime)
env.example
requirements.txt
```

## Notes

- Ensemble learning is the canonical direction for this project.
- Other notebooks/modules are preserved as experiments, baselines, or legacy references.

