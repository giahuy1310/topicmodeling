# Data Preprocessing Pipeline

This document describes the end-to-end data preprocessing pipeline used for preparing Vietnamese text data for emotion classification. The pipeline is implemented in `DataPreprocessing.ipynb`.

## Overview

The pipeline transforms raw Vietnamese social-media text into a clean, model-ready dataset through the following stages:

1. **Load & Inspect** – Load raw data, display schema, check for missing values.
2. **Text Cleaning** – Replace emojis, text emoticons, and abbreviations; strip URLs, HTML, and special characters.
3. **Deduplication** – Drop duplicate sentences based on cleaned text.
4. **Validation & Visualisation** – Emotion distribution and text-length statistics.
5. **Export** – Shuffle and save the cleaned dataset.

## Input / Output

| Item | Path |
|------|------|
| Input file | `data/processed/train_1500_para_clean.csv` |
| Output file | `data/processed/train_1500_para_final.csv` |

The raw dataset contains **10,558 records** across 5 columns (`Emotion`, `Sentence`, `emotion_vn`, `source_example_idx`, `source_example`). Only the three essential columns are retained for preprocessing: `Sentence`, `Emotion`, and `emotion_vn`.

## Emotion Labels

The dataset is approximately balanced across **7 emotion classes**:

| Emotion   | Count | Percentage |
|-----------|------:|------------|
| Enjoyment | 1,556 | 14.76%     |
| Sadness   | 1,500 | 14.23%     |
| Other     | 1,499 | 14.22%     |
| Surprise  | 1,499 | 14.22%     |
| Disgust   | 1,498 | 14.21%     |
| Anger     | 1,496 | 14.19%     |
| Fear      | 1,493 | 14.16%     |
| **Total** | **10,541** | **100%** |

(Counts shown after deduplication.)

## Text Cleaning Steps

Each sentence passes through a sequential cleaning pipeline implemented in `preprocess_vietnamese_text()`:

### 1. Emoji Replacement

A hand-crafted dictionary of **116 emoji-to-Vietnamese mappings** converts Unicode emoji characters into descriptive Vietnamese words. For example:

- 😀 → `vui vẻ` (happy)
- 😢 → `buồn bã` (sad)
- 😠 → `tức giận` (angry)
- ❤️ → `yêu thương` (love)

Any remaining emojis not in the dictionary are stripped by a comprehensive Unicode regex pattern covering emoticons, symbols, pictographs, and other emoji blocks.

### 2. Text Emoticon Replacement

**39 text emoticon patterns** are converted to Vietnamese descriptors, sorted by length (longest first) to avoid partial matches. Examples:

- `:))` / `:)))` / `=))` → `cười lớn` (laughing)
- `:(` / `:-(` → `buồn bã` (sad)
- `<3` → `yêu thương` (love)
- `T_T` → `khóc` (crying)

### 3. Abbreviation Expansion

**95 Vietnamese internet abbreviations** are expanded to their full forms via exact word-level matching. Examples:

- `ko` / `k` / `hok` → `không` (no/not)
- `dc` / `đc` → `được` (can/ok)
- `ns` → `nói` (say)
- `bth` → `bình thường` (normal)

### 4. Special Character Cleaning

A series of regex operations handles remaining noise:

- **URLs** (`http…`, `www.…`) are removed.
- **HTML tags** (`<…>`) are stripped.
- **Non-Vietnamese characters** are replaced with spaces, preserving Vietnamese diacritics and basic punctuation (`.,!?`).
- **Repeated punctuation** is collapsed (e.g., `!!!` → `!`).
- **Excess whitespace** is normalized and the result is lowercased.

### Design Decision: No Stopword Removal

Stopwords are intentionally **not** removed. The downstream model is BERT-based (CafeBERT), which relies on self-attention over **all** tokens. Function words like *không* (negation) and *rất* (very) carry critical sentiment and emotion signals that the transformer architecture is designed to leverage.

## Deduplication

Duplicate detection is performed on the `Sentence_clean` column. Only the first occurrence of each unique cleaned sentence is kept.

- Records before dedup: **10,558**
- Duplicates found: **17**
- Records after dedup: **10,541**

## Shuffle & Export

After all cleaning and deduplication, the dataset is shuffled with a fixed random seed (`42`) for reproducibility and exported to CSV.

The final output contains 4 columns: `Sentence`, `Emotion`, `emotion_vn`, and `Sentence_clean`, with **10,541 records** ready for model training.
