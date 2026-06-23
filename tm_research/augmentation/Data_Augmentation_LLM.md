# Data Augmentation Framework (LLM) - Technical Overview

This document explains the augmentation framework implemented in [`augmentation/Data_Augmentation_LLM.ipynb`](Data_Augmentation_LLM.ipynb).

## 1) Goal

The framework augments Vietnamese social-media emotion data to reduce class imbalance for a 7-class emotion classification task.

Core objectives:

- Increase low-resource emotion classes with synthetic examples.
- Preserve target emotion consistency in generated samples.
- Support batch, async generation for faster throughput.
- Merge generated data with original training data and export cleaned outputs.

## 2) Input Dataset Assumptions

The notebook expects a training dataframe with at least:

- `Sentence`
- `Emotion`
- `emotion_vn`

Observed emotion labels:

- `Enjoyment`, `Disgust`, `Other`, `Sadness`, `Anger`, `Fear`, `Surprise`

The notebook visualizes distribution and computes imbalance ratio before augmentation.

## 3) High-Level Architecture

Main components:

1. **Environment and setup**
   - Installs and imports LangChain + Gemini client + pandas + plotting libs.
   - Loads `GOOGLE_API_KEY` from Colab Secret or local `.env`.

2. **`BatchEmotionAugmenterLangChain` class**
   - Handles prompt construction, few-shot sampling, async batch generation, and parsing.

3. **`run_balanced_augmentation(...)` function**
   - Computes per-class deficits relative to a target count.
   - Calls augmenter selectively for classes that need synthetic samples.
   - Trims generated outputs to exact required size.

4. **Post-processing and export**
   - Combines original + augmented rows.
   - Fixes minor label normalization.
   - Exports processed CSV files.

## 4) Augmenter Class Design

Class: `BatchEmotionAugmenterLangChain`

### 4.1 Emotion Mapping

`EMOTIONS` maps English label to Vietnamese description:

- `Enjoyment -> vui ve`
- `Disgust -> kho chiu`
- `Other -> khac`
- `Sadness -> buon`
- `Anger -> gian du`
- `Fear -> so hai`
- `Surprise -> ngac nhien`

### 4.2 LLM Configuration

Constructor arguments (main ones):

- `model_name` (default observed: `gemini-2.5-flash`)
- `temperature` (example default in active class: `0.7`)
- `max_concurrency` (parallel request cap)
- `verbose` (logging)

LLM backend:

- `ChatGoogleGenerativeAI(...)`

### 4.3 Prompting Strategy

Prompt structure:

- **System message**: enforce Vietnamese social-media style and emotion-appropriate generation constraints.
- **Human message**:

  - Requires exact number of outputs (`num_variations`).
  - Provides emotion label + Vietnamese description.
  - Supplies 10 few-shot examples sampled from same class.
  - Enforces strict output format:

    - `[number]. [sentence] (example [source_number])`

### 4.4 Few-shot Example Sampling

Method: `_get_few_shot_examples(emotion, num_examples=10)`

- Filters rows by `Emotion`.
- Randomly samples `num_examples`.
- Uses replacement when class size is smaller than requested examples.

This increases diversity across repeated calls.

### 4.5 Batch Input Preparation

Method: `prepare_batch_inputs(...)`

- Repeats generation tasks per emotion:

  - `num_calls_per_emotion` times
  - each call asks for `num_variations_per_call` outputs

- Builds one input dict per call including:

  - emotion metadata
  - 10 examples
  - `num_variations`

Expected raw generation volume:

- `len(inputs) * num_variations_per_call`

### 4.6 Async Batch Execution

Method: `_run_batch_async(inputs, delay)`

- Splits jobs into chunks of size `max_concurrency`.
- Executes each chunk via `chain.abatch(...)`.
- Applies delay between chunks to reduce API pressure.
- On failure:

  - logs warning
  - inserts empty fallback result for failed chunk items.

### 4.7 Output Parsing and Structuring

Method: `_parse_generated_text(...)`

- Parses each returned line.
- Extracts optional source pointer from `(example X)`.
- Removes numeric prefix.
- Returns normalized records:

  - `sentence`
  - `source_example_idx`
  - `source_example` (if recoverable)

Method: `run_batch_augmentation(...)`

- Orchestrates prepare + async run + parse.
- Builds dataframe columns:

  - `Emotion`, `Sentence`, `emotion_vn`
  - `augmented=True`
  - `source='langchain_batch'`
  - source trace fields

Method: `combine_with_original(augmented_df)`

- Copies original dataframe and tags:

  - `augmented=False`
  - `source='original'`

- Concatenates original and generated rows.

## 5) Balanced Augmentation Logic

Function: `run_balanced_augmentation(batch_augmenter, df_train, ...)`

### 5.1 Planning Stage

For each emotion:

- Compute current count.
- Compare with `target_per_emotion`.
- Skip cases:

  - already at/above target
  - deficit below `min_samples_to_augment`

For classes needing augmentation:

- `needed = target_per_emotion - current_count`
- `num_calls = ceil(needed / variations_per_call)`

### 5.2 Generation Stage

For each planned class:

- Temporarily override augmenter `max_concurrency`.
- Call `run_batch_augmentation(...)` for that class only.
- Keep only `head(needed)` to enforce exact deficit fill.

### 5.3 Final Aggregation

- Concatenate all per-class generated subsets.
- Return single augmentation dataframe.
- If no class requires augmentation, return empty dataframe.

## 6) Execution Flow in Notebook

Observed flow:

1. Initialize augmenter instance.
2. Run `run_balanced_augmentation(...)` with selected target and generation params.
3. Merge with original via `combine_with_original(...)`.
4. Save to `tm_research/data/processed/train_balanced_optimized.csv`.
5. Post-process, validate distribution, export cleaned file.

## 7) Key Tunable Parameters

Most impactful controls:

- `target_per_emotion`: target class size.
- `variations_per_call`: outputs requested from each prompt call.
- `num_calls_per_emotion`: generation rounds per class (inside batch augment method).
- `max_concurrency`: parallel request throughput.
- `delay_between_batches`: pacing between chunks.
- `min_samples_to_augment`: ignore tiny deficits.
- `temperature`: creativity vs consistency tradeoff.

## 8) Data Lineage and Traceability

Framework includes provenance fields:

- `augmented` boolean
- `source`
- `source_example_idx`
- `source_example`

These fields support auditing generated samples and downstream filtering.

## 9) Practical Notes

- The notebook contains a stricter prompt draft (commented) and a simpler active prompt class.
- Current active class focuses on emotion-consistent generation, not guaranteed semantic paraphrase of a specific sentence.
- If strict semantic preservation is required, adapt the active prompt to the stricter variant and keep parser format constraints aligned.

## 10) Output Artifacts

Primary outputs used in notebook:

- `train_balanced_optimized.csv` (combined original + augmented, all columns)
- `train_1500_gen_eval.csv` (same rows after `emotion_vn` fix, with trace columns for fidelity eval)
- `train_1500_gen_clean.csv` (training-ready: `Emotion`, `Sentence`, `emotion_vn`; trace columns dropped)
- `test_gen_clean.csv` (cleaned export with selected columns)
