# Large Language Model-based Topic Modeling with LangChain for Vietnamese Educational Online Communities

## Vietnamese Emotion Classification Model

This repository includes a comprehensive Vietnamese emotion classification system using state-of-the-art transformer models.

### Emotion Classification Features

- **Multiple Model Comparison**: PhoBERT, XLM-RoBERTa, and mBERT
- **Google Colab Compatible**: Ready to run on Google Colab with **TPU** or GPU support
- **Hardware Acceleration**: 
  - TPU (v2-8): ~30-40 minutes ⚡ (FASTEST)
  - GPU (T4): ~1-1.5 hours 🚀
  - CPU: 5-10 hours (not recommended)
- **Dataset**: Vietnamese emotion dataset (Excel files)
  - Pre-split into train/validation/test sets
  - Format: First column (emotion label), Second column (Vietnamese sentence)
- **Comprehensive Evaluation**: Accuracy, F1-scores, confusion matrices, per-class metrics
- **Production Ready**: Trained model saved for inference
- **Automatic Hardware Detection**: Automatically uses best available hardware (TPU > GPU > CPU)

### Dataset Preparation

Ensure your dataset is in the `data/` folder with the following structure:
```
data/
├── train_nor_811.xlsx   # Training set
├── valid_nor_811.xlsx   # Validation set
└── test_nor_811.xlsx    # Test set
```

Each Excel file should have 2 columns:
1. **First column**: Emotion label (e.g., "joy", "sadness", "anger")
2. **Second column**: Vietnamese sentence/text

### Running the Emotion Classification Notebook

#### Option 1: Google Colab (Recommended)

1. Upload the notebook to [Google Colab](https://colab.research.google.com/)
2. Enable accelerator: `Runtime` → `Change runtime type` → Select **TPU** or **GPU**
   - **TPU (v2-8)**: Fastest option (~30-40 minutes) ⚡
   - **GPU (T4)**: Fast option (~1-1.5 hours) 🚀
3. Upload your data files to Google Drive
4. Follow the instructions in the notebook or see `TPU_GUIDE.md` for details

**Benefits**: Free TPU/GPU access, no local setup required, very fast training with TPU

#### Option 2: Local Jupyter Notebook

1. Navigate to the notebook:
```bash
cd tm_research
jupyter notebook EmoModel.ipynb
```

2. The notebook includes:
   - Data loading from Excel files (local or Google Drive)
   - Exploratory data analysis with visualizations
   - Training 3 Vietnamese NLP models (PhoBERT, XLM-RoBERTa, mBERT)
   - Model comparison and selection based on F1-score
   - Full training with best model on complete dataset
   - Comprehensive evaluation and metrics on test set
   - Model saving for future use
   - Custom text prediction examples

3. Results are saved to:
   - **Colab**: `/content/drive/MyDrive/emotion_classifier_model/` (persists in Google Drive)
   - **Local**: `./results_final/` and `./emotion_classifier_model/`

## Quickstart

1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source ./.venv/bin/activate
```

1. Install dependencies

```bash
pip install -r requirements.txt
```

1. Configure environment

Create a `.env` file (or copy `env.example` to `.env`) and set your API keys as needed.

1. Run a sample research pipeline (LLM default: Gemini)

```bash
python -m tm_research.cli run --query "quantum computing benchmarks" --num-results 10 --num-topics 8
```

## Commands

```bash
python -m tm_research.cli ingest --query "<your query>" --num-results 10 --out data/raw.jsonl
python -m tm_research.cli model --input data/raw.jsonl --num-topics 8 --out data/topics.json
python -m tm_research.cli summarize --input data/topics.json --out data/summary.md
python -m tm_research.cli run --query "<your query>" --num-results 10 --num-topics 8
```

To use classic TF-IDF+NMF instead of LLM:

```bash
python -m tm_research.cli model --input data/raw.jsonl --num-topics 8 --method classic --out data/topics.json
python -m tm_research.cli run --query "<your query>" --num-results 10 --num-topics 8 --method classic
```

## Project structure

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
- The CLI defaults to Gemini-based topic modeling; use `--method classic` for TF-IDF+NMF.
