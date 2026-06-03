"""Collapse 7-class UIT-VSMEC emotion probabilities to 3-class sentiment.

Mapping (NRC-inspired, adapted to UIT-VSMEC labels):
  positive : Enjoyment
  negative : Anger, Fear, Sadness, Disgust
  neutral  : Other, Surprise   (Surprise is ambiguous → neutral by default)

The mapping is a module-level constant so ablations can override it before
calling collapse_batch without touching function signatures.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Polarity constants (match UIT-VSFC gold: 0=negative, 1=neutral, 2=positive)
# ---------------------------------------------------------------------------
POLARITY_NEGATIVE = "negative"
POLARITY_NEUTRAL  = "neutral"
POLARITY_POSITIVE = "positive"

# Order used internally for the 3-class vector (matches VSFC integer codes).
POLARITY_CLASSES: List[str] = [POLARITY_NEGATIVE, POLARITY_NEUTRAL, POLARITY_POSITIVE]
POLARITY_TO_IDX: Dict[str, int] = {p: i for i, p in enumerate(POLARITY_CLASSES)}

# UIT-VSFC integer → string (from HuggingFace dataset card)
VSFC_INT_TO_LABEL: Dict[int, str] = {0: POLARITY_NEGATIVE, 1: POLARITY_NEUTRAL, 2: POLARITY_POSITIVE}

# ---------------------------------------------------------------------------
# 7 → 3 mapping
# Surprise is placed under "neutral" (ambiguous emotions → neutral).
# To ablate: set EMOTION_TO_POLARITY["Surprise"] = None and re-run.
# ---------------------------------------------------------------------------
EMOTION_TO_POLARITY: Dict[str, Optional[str]] = {
    "Enjoyment": POLARITY_POSITIVE,
    "Anger":     POLARITY_NEGATIVE,
    "Fear":      POLARITY_NEGATIVE,
    "Sadness":   POLARITY_NEGATIVE,
    "Disgust":   POLARITY_NEGATIVE,
    "Other":     POLARITY_NEUTRAL,
    "Surprise":  POLARITY_NEUTRAL,   # change to None to exclude from sums
}


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def collapse_probs(probs: np.ndarray, class_names: List[str]) -> np.ndarray:
    """Collapse a single 7-class probability vector to 3-class sentiment.

    Parameters
    ----------
    probs : (7,) float array — softmax over UIT-VSMEC emotion classes
    class_names : list of emotion label strings in the same order as ``probs``

    Returns
    -------
    (3,) float array — [negative, neutral, positive] summed probabilities
    """
    probs = np.asarray(probs, dtype=np.float32)
    out = np.zeros(3, dtype=np.float32)
    for idx, name in enumerate(class_names):
        polarity = EMOTION_TO_POLARITY.get(name)
        if polarity is not None:
            out[POLARITY_TO_IDX[polarity]] += probs[idx]
    return out


def collapse_batch(probs: np.ndarray, class_names: List[str]) -> np.ndarray:
    """Collapse a batch of 7-class probability vectors to 3-class sentiment.

    Parameters
    ----------
    probs : (N, 7) float array
    class_names : list of emotion label strings matching column order

    Returns
    -------
    (N, 3) float array — [negative, neutral, positive]
    """
    probs = np.asarray(probs, dtype=np.float32)
    N = probs.shape[0]
    out = np.zeros((N, 3), dtype=np.float32)
    for idx, name in enumerate(class_names):
        polarity = EMOTION_TO_POLARITY.get(name)
        if polarity is not None:
            col = POLARITY_TO_IDX[polarity]
            out[:, col] += probs[:, idx]
    return out


def probs_to_pred_labels(collapsed: np.ndarray) -> List[str]:
    """Convert (N, 3) collapsed probs to a list of predicted polarity strings."""
    indices = collapsed.argmax(axis=1)
    return [POLARITY_CLASSES[i] for i in indices]


def evaluate_sentiment(
    y_true: List[str],
    y_pred: List[str],
    labels: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Compute macro-F1, weighted F1, and MCC for 3-class sentiment predictions.

    Parameters
    ----------
    y_true, y_pred : lists of polarity strings ("negative"/"neutral"/"positive")
    labels : optional subset of labels to use; defaults to POLARITY_CLASSES

    Returns
    -------
    dict with keys: macro_f1, weighted_f1, mcc, and per-class F1s
    """
    from sklearn.metrics import f1_score, matthews_corrcoef

    if labels is None:
        labels = POLARITY_CLASSES

    macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0))
    mcc = float(matthews_corrcoef(y_true, y_pred))
    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

    result: Dict[str, float] = {"macro_f1": macro_f1, "weighted_f1": weighted_f1, "mcc": mcc}
    for label, score in zip(labels, per_class):
        result[f"f1_{label}"] = float(score)
    return result
