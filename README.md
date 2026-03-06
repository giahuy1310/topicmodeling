# An LLM-Augmented Approach for Emotion Recognition in Vietnamese Online Social Platforms

This repository contains the code and experiments for the thesis **“An LLM-Augmented Approach for Emotion Recognition in Vietnamese Online Social Platforms”** by **Nguyễn Gia Huy**.  
The work focuses on improving fine-grained emotion recognition (ER) for Vietnamese social media text using **Large Language Model (LLM)–based data augmentation** and **Vietnamese monolingual BERT models**.

## Overview

**Problem context**
- **Low-resource language**: Vietnamese NLP suffers from limited, imbalanced datasets, especially for fine-grained emotions.
- **Task**: Fine-grained **Emotion Recognition** on Vietnamese social media text (e.g., Facebook, TikTok, Zalo), with labels such as **Enjoyment, Sadness, Fear, Anger, Disgust, Surprise, Other**.
- **Main challenges**:
  - Data scarcity and strong class imbalance in the UIT-VSMEC corpus.
  - Degraded performance of baseline models on minority emotions (Fear, Anger, Surprise).

**Proposed approach**
- Use **Gemini-2.5-Flash** as an LLM to augment UIT-VSMEC with:
  - **Synthetic generation** of new sentences for minority emotions.
  - **Paraphrasing** of existing examples while preserving emotion and meaning.
- Fine-tune **Vietnamese monolingual BERT models** (especially **CafeBERT**) on the augmented datasets.
- Evaluate whether LLM-augmented data improves:
  - Overall performance (Accuracy, Macro F1, Weighted F1).
  - Robustness and performance on minority emotion classes.

## Dataset

- **Corpus**: **UIT-VSMEC** (Vietnamese Social Media Emotion Corpus).
- **Size**: 6,927 human-annotated sentences from Vietnamese social media.
- **Labels**: `Enjoyment, Sadness, Fear, Anger, Disgust, Surprise, Other`.
- **Split**: train, validation, test (as defined by the original UIT-VSMEC work).
- **Imbalance**: severe skew toward majority labels (e.g., Enjoyment) and very few examples for Surprise, Anger, Fear.

The working format used in this repository is a simple table with:

- **Column 1**: `text` – Vietnamese sentence from social media.  
- **Column 2**: `label` – emotion category following UIT-VSMEC guidelines.

## Methodology

**1. Baseline models**
- Train several BERT-based Vietnamese models on the original UIT-VSMEC data, including:
  - **PhoBERT v2 Large**
  - **VisoBERT**
  - **CafeBERT**
  - **XLM-Roberta**
- Evaluate using:
  - **Accuracy**
  - **Macro F1** (focus metric due to class imbalance)
  - **Weighted F1**

**2. LLM-based data augmentation with Gemini-2.5-Flash**
- **Goal**: enrich low-frequency emotions (Fear, Anger, Surprise).
- **Strategies**:
  - **Synthetic generation**: 10-shot prompting to generate new, emotionally consistent examples for minority classes.
  - **Paraphrasing**: rewrite existing sentences while strictly preserving semantic meaning and emotion label.
- Both strategies are defined by detailed prompt templates (see thesis appendices) and are designed for Vietnamese social media style.

**3. Training setups**
- Train CafeBERT and other models on:
  - Original dataset.
  - Original + paraphrased data.
  - Original + newly generated synthetic data.
- Evaluate on:
  - Original test set.
  - A newly generated LLM-based test set to test robustness and generalization.

**4. Planned ensemble framework (future work)**
- Design an **LLM–BERT ensemble** where:
  - Multiple downstream models (CafeBERT, CNN-BERT, CNN-BiLSTM, etc.) produce candidate predictions.
  - **Gemini-2.5-Flash** acts as a high-level selector/voter over these predictions to improve reliability, especially for ambiguous or minority classes.

## Key Results (Summary)

- **Best-performing model**: **CafeBERT** trained on **newly generated synthetic data**.
- **Performance on generated test set**:
  - **Accuracy**: 73%
  - **Macro F1**: 74%
  - **Weighted F1**: 73%
- **Improvement over original CafeBERT (no augmentation)**:
  - From 68% accuracy and 67% macro F1 (original data only)
  - To 73% accuracy and 74% macro F1 (with LLM-generated data)
- **Improvement over previous state of the art** on UIT-VSMEC:
  - Previous best (CNN + word2vec): **59.74% weighted F1**
  - This work: **73% weighted F1**
- **Minority emotions** (Fear, Anger, Surprise) show consistent F1-score gains, especially for **Anger**, confirming the effectiveness of targeted LLM-based augmentation.

## Emotion Recognition Notebook (`EmoModel.ipynb`)

The main experimental pipeline for emotion recognition is implemented in the notebook `EmoModel.ipynb` inside the `tm_research` directory.

### Running the notebook on Google Colab

1. Upload `EmoModel.ipynb` to [Google Colab](https://colab.research.google.com/).
2. In Colab, set accelerator: `Runtime` → `Change runtime type` → select **GPU** or **TPU** (if available).
3. Upload or mount the UIT-VSMEC data in the expected format (two columns: `text`, `label`).
4. Follow the cells in the notebook to:
   - Load and preprocess the dataset (including cleaning and normalization for Vietnamese social media text).
   - Perform LLM-based data augmentation (synthetic generation and paraphrasing with Gemini-2.5-Flash).
   - Train baseline and augmented CafeBERT (and other BERT) models.
   - Evaluate with Accuracy, Macro F1, and Weighted F1 and export results.

### Running the notebook locally

1. Navigate to the notebook:

```bash
cd tm_research
jupyter notebook EmoModel.ipynb
```

2. Prepare the UIT-VSMEC-based dataset locally with two columns (`text`, `label`).  
3. Execute the notebook step by step as in the Colab workflow.

## Topic Modeling Research CLI (Existing Component)

This repository also includes a general research pipeline for topic modeling using LLMs and classic methods under the `tm_research` package. The CLI described below comes from this component and can still be used as in the original project.

### Quickstart

1. **Create and activate a virtual environment**

```bash
python3 -m venv .venv
source ./.venv/bin/activate
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Configure environment**

Create a `.env` file (or copy `env.example` to `.env`) and set your API keys as needed.

4. **Run a sample research pipeline (LLM default: Gemini)**

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

To use classic TF-IDF + NMF instead of the default LLM-based method:

```bash
python -m tm_research.cli model --input data/raw.jsonl --num-topics 8 --method classic --out data/topics.json
python -m tm_research.cli run --query "<your query>" --num-results 10 --num-topics 8 --method classic
```

## Project Structure

```text
tm_research/
  __init__.py
  config.py
  ingestion.py
  modeling.py
  research.py
  cli.py
data/
  (created at runtime)
env.example
requirements.txt
```

### Gemini setup

- Set `GOOGLE_API_KEY` in `.env` to enable Gemini via LangChain.
- The CLI defaults to Gemini-based topic modeling; use `--method classic` for TF-IDF + NMF.
