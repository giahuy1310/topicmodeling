"""Stacking helpers: KFold OOF, weight computation, and prompt formatting."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold

from .utils_io import LabelMap

BASE_MODEL_NAMES: Tuple[str, ...] = (
    "phobert",
    "cafebert",
    "vibert",
    "logreg",
    "svc",
)


@dataclass
class StackedRow:
    text: str
    probs_per_model: Dict[str, np.ndarray]
    weights: Dict[str, float]
    label: Optional[str] = None


def kfold_oof_probs(
    train_fit_predict_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    X: Sequence,
    y: np.ndarray,
    num_classes: int,
    n_splits: int = 5,
    seed: int = 42,
) -> Tuple[np.ndarray, List[float]]:
    """Compute OOF probabilities via stratified K-fold.

    ``train_fit_predict_fn(X_tr, y_tr, X_va) -> probs`` must return a
    ``(len(X_va), num_classes)`` matrix.
    Returns the full ``(N, C)`` OOF matrix and per-fold accuracies.
    """

    x_arr = np.asarray(X, dtype=object)
    y = np.asarray(y)
    n = len(x_arr)
    oof = np.zeros((n, num_classes), dtype=np.float32)
    fold_accs: List[float] = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(n), y), start=1):
        probs = train_fit_predict_fn(x_arr[tr_idx], y[tr_idx], x_arr[va_idx])
        probs = np.asarray(probs, dtype=np.float32)
        if probs.shape != (len(va_idx), num_classes):
            raise ValueError(
                f"fold {fold_idx}: expected probs shape {(len(va_idx), num_classes)}, got {probs.shape}"
            )
        oof[va_idx] = probs
        fold_acc = float((probs.argmax(axis=1) == y[va_idx]).mean())
        fold_accs.append(fold_acc)
        print(f"  fold {fold_idx}/{n_splits} acc={fold_acc:.4f}")
    return oof, fold_accs


def compute_weights(acc_dict: Dict[str, float]) -> Dict[str, float]:
    """Default: ``w_i = acc_i / sum_j acc_j`` (normalized accuracy)."""

    total = sum(max(a, 0.0) for a in acc_dict.values())
    if total <= 0:
        n = len(acc_dict)
        return {k: 1.0 / n for k in acc_dict}
    return {k: max(a, 0.0) / total for k, a in acc_dict.items()}


def compute_weights_softmax(acc_dict: Dict[str, float], temperature: float = 1.0) -> Dict[str, float]:
    """Softmax over accuracies (sharper than linear normalization)."""

    keys = list(acc_dict.keys())
    arr = np.array([acc_dict[k] for k in keys], dtype=np.float64) / max(temperature, 1e-8)
    arr = arr - arr.max()
    e = np.exp(arr)
    w = e / e.sum()
    return {k: float(v) for k, v in zip(keys, w)}


def compute_weights_lsq(
    val_probs_per_model: Dict[str, np.ndarray],
    y_val: np.ndarray,
    num_classes: int,
) -> Dict[str, float]:
    """Constrained NNLS weights minimizing log-loss on validation.

    Optional alternative to accuracy-normalized weights.
    """

    from scipy.optimize import minimize

    keys = list(val_probs_per_model.keys())
    P = np.stack([val_probs_per_model[k] for k in keys], axis=0)
    eye = np.eye(num_classes, dtype=np.float32)
    one_hot = eye[y_val]

    def neg_log_loss(w_raw: np.ndarray) -> float:
        w = np.maximum(w_raw, 0)
        s = w.sum()
        if s <= 0:
            return 1e9
        w = w / s
        mix = np.tensordot(w, P, axes=([0], [0]))
        mix = np.clip(mix, 1e-9, 1.0)
        return float(-np.mean(np.sum(one_hot * np.log(mix), axis=1)))

    x0 = np.full(len(keys), 1.0 / len(keys), dtype=np.float64)
    res = minimize(
        neg_log_loss,
        x0,
        method="L-BFGS-B",
        bounds=[(0.0, 1.0)] * len(keys),
    )
    w = np.maximum(res.x, 0)
    if w.sum() <= 0:
        w = np.full_like(w, 1.0 / len(w))
    else:
        w = w / w.sum()
    return {k: float(v) for k, v in zip(keys, w)}


def weighted_average(
    probs_per_model: Dict[str, np.ndarray],
    weights: Dict[str, float],
) -> np.ndarray:
    """Compute weighted-average probability rows.

    ``probs_per_model[m]`` is ``(N, C)``; returns ``(N, C)``.
    """

    keys = [k for k in probs_per_model if k in weights]
    if not keys:
        raise ValueError("No overlapping models between probs_per_model and weights.")
    P = np.stack([probs_per_model[k] for k in keys], axis=0)
    w = np.array([weights[k] for k in keys], dtype=np.float32)
    s = w.sum()
    if s <= 0:
        raise ValueError("weights must sum > 0")
    w = w / s
    return np.tensordot(w, P, axes=([0], [0]))


def _format_prob_dict(probs: np.ndarray, label_map: LabelMap, decimals: int = 3) -> str:
    parts = [
        f"{label_map.id2label[i]}={probs[i]:.{decimals}f}"
        for i in range(label_map.num_classes)
    ]
    return "{" + ", ".join(parts) + "}"


def format_prompt(
    text: str,
    probs_per_model: Dict[str, np.ndarray],
    weights: Dict[str, float],
    label_map: LabelMap,
    label: Optional[str] = None,
    model_order: Iterable[str] = BASE_MODEL_NAMES,
) -> Dict[str, str]:
    """Render one stacked example as ``{prompt, completion}``.

    Prompt schema (structured tokens, same for train/val/test):

        [TEXT] ... [/TEXT]
        [STACK]
        <phobert w=0.205>{joy=0.910, ...}
        ...
        [/STACK]
        [WEIGHTED_AVG]{joy=0.870, ...}[/WEIGHTED_AVG]

    The completion (only present when ``label`` is given) is
    ``<label>EMOTION</label>``.
    """

    used_models = [m for m in model_order if m in probs_per_model and m in weights]
    lines = [f"[TEXT] {text} [/TEXT]", "[STACK]"]
    for m in used_models:
        w = weights[m]
        lines.append(f"<{m} w={w:.3f}>{_format_prob_dict(probs_per_model[m], label_map)}")
    lines.append("[/STACK]")
    avg = weighted_average(
        {m: probs_per_model[m][None, :] for m in used_models},
        {m: weights[m] for m in used_models},
    )[0]
    lines.append(f"[WEIGHTED_AVG]{_format_prob_dict(avg, label_map)}[/WEIGHTED_AVG]")
    prompt = "\n".join(lines)
    completion = f"<label>{label}</label>" if label is not None else ""
    return {"prompt": prompt, "completion": completion}


def write_jsonl(rows: Iterable[dict], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return p


def parse_label_from_completion(text: str, label_map: LabelMap) -> Optional[str]:
    """Extract ``<label>...</label>`` from a model completion.

    Falls back to first label keyword found anywhere in ``text``; returns
    ``None`` when no class name is recognized.
    """

    import re

    m = re.search(r"<label>\s*([^<\n]+?)\s*</label>", text, flags=re.IGNORECASE)
    if m:
        cand = m.group(1).strip()
        if cand in label_map.label2id:
            return cand
        for lab in label_map.label2id:
            if lab.lower() == cand.lower():
                return lab
    lower = text.lower()
    for lab in sorted(label_map.label2id, key=len, reverse=True):
        if lab.lower() in lower:
            return lab
    return None
