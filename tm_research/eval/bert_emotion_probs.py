"""Load a fine-tuned 7-class emotion classifier and batch-predict softmax probs.

Conventions match EmoModel_Advanced_*_Test.ipynb notebooks:
  - Model saved with AutoModelForSequenceClassification + label_mappings.json
  - max_length = 256, tokenizer padding + truncation
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


def load_emotion_classifier(
    model_dir: str | os.PathLike,
) -> Tuple[object, object, List[str]]:
    """Load a saved HuggingFace 7-class emotion classifier.

    Parameters
    ----------
    model_dir : path to directory containing:
        - HuggingFace model weights (config.json, pytorch_model.bin / safetensors)
        - ``label_mappings.json`` with ``id2label`` and ``label2id``

    Returns
    -------
    (model, tokenizer, class_names)
      class_names is ordered by integer id (0, 1, 2, …) from id2label
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    label_map_path = model_dir / "label_mappings.json"
    if not label_map_path.exists():
        raise FileNotFoundError(f"label_mappings.json not found in {model_dir}")

    with open(label_map_path, "r", encoding="utf-8") as f:
        label_maps = json.load(f)

    id2label = {int(k): v for k, v in label_maps["id2label"].items()}
    num_classes = len(id2label)
    class_names: List[str] = [id2label[i] for i in range(num_classes)]

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir),
        num_labels=num_classes,
        id2label=id2label,
        label2id={v: k for k, v in id2label.items()},
        ignore_mismatched_sizes=False,
    )
    model.eval()

    return model, tokenizer, class_names


def predict_probs(
    model,
    tokenizer,
    texts: List[str],
    max_length: int = 256,
    batch_size: int = 64,
    device: Optional[str] = None,
) -> np.ndarray:
    """Predict softmax probabilities over 7 emotion classes.

    Parameters
    ----------
    model, tokenizer : from load_emotion_classifier
    texts : list of raw Vietnamese text strings
    max_length : sequence truncation (256 matches training notebooks)
    batch_size : GPU batch size
    device : "cuda", "cpu", or None (auto-detect)

    Returns
    -------
    (N, 7) float32 numpy array of softmax probabilities
    """
    import torch
    import torch.nn.functional as F

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()

    all_probs: List[np.ndarray] = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            outputs = model(**enc)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1).cpu().numpy().astype(np.float32)

        all_probs.append(probs)

    return np.vstack(all_probs)
