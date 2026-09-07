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

# ---------------------------------------------------------------------------
# Compact prompt schema (compact_v1)
# ---------------------------------------------------------------------------

PROMPT_SCHEMA_VERSION = "compact_v1"
PROMPT_PROB_DECIMALS = 2

# Long model name → single-char abbreviation used in stack lines.
MODEL_ABBR: Dict[str, str] = {
    "phobert": "p",
    "cafebert": "c",
    "vibert": "v",
    "logreg": "l",
    "svc": "s",
}

# Emotion label → short abbreviation used inside prob dicts.
# Full names are kept in completions (<label>Anger</label>) and system legends.
LABEL_ABBR: Dict[str, str] = {
    "Anger": "A",
    "Disgust": "D",
    "Enjoyment": "J",
    "Fear": "F",
    "Other": "O",
    "Sadness": "Sa",
    "Surprise": "Su",
}

# Reverse mapping (abbr → full name).
ABBR_TO_LABEL: Dict[str, str] = {v: k for k, v in LABEL_ABBR.items()}


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
    n_splits: int = 3,
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


def stack_prob_features(
    probs_per_model: Dict[str, np.ndarray],
    model_order: Iterable[str] = BASE_MODEL_NAMES,
) -> np.ndarray:
    """Concatenate per-model ``(N, C)`` matrices into ``(N, M*C)`` in ``model_order``."""

    keys = list(model_order)
    missing = [k for k in keys if k not in probs_per_model]
    if missing:
        raise ValueError(f"Missing model(s) in probs_per_model: {', '.join(missing)}")
    arrays = [np.asarray(probs_per_model[k], dtype=np.float32) for k in keys]
    n0, _c0 = arrays[0].shape
    for k, arr in zip(keys, arrays):
        if arr.ndim != 2:
            raise ValueError(f"{k}: expected 2D (N, C), got shape {arr.shape}")
        if arr.shape[0] != n0:
            raise ValueError(
                f"row count mismatch: {keys[0]} has {n0} rows, {k} has {arr.shape[0]}"
            )
        if arr.shape[1] != _c0:
            raise ValueError(
                f"class-count mismatch: {keys[0]} has {_c0} classes, {k} has {arr.shape[1]}"
            )
    return np.concatenate(arrays, axis=1)


def fit_logreg_meta(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    C_grid: Sequence[float] = (0.25, 1.0, 4.0),
    seed: int = 123,
) -> Tuple[object, Dict[str, float]]:
    """Fit a multinomial LogReg stacking head; select ``C`` by val macro-F1.

    Trains on OOF concatenated probability features. Returns
    ``(fitted_estimator, {"C": ..., "val_macro_f1": ...})``.
    """

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score

    if not C_grid:
        raise ValueError("C_grid must be non-empty")

    best_clf = None
    best_c = None
    best_f1 = -1.0
    for C in C_grid:
        clf = LogisticRegression(
            C=float(C),
            solver="lbfgs",
            max_iter=1000,
            random_state=seed,
        )
        clf.fit(X_train, y_train)
        pred = clf.predict(X_val)
        f1 = float(f1_score(y_val, pred, average="macro", zero_division=0))
        if f1 > best_f1:
            best_f1 = f1
            best_c = float(C)
            best_clf = clf
    assert best_clf is not None and best_c is not None
    return best_clf, {"C": best_c, "val_macro_f1": best_f1}


def _format_prob_dict(probs: np.ndarray, label_map: LabelMap, decimals: int = 3) -> str:
    parts = [
        f"{label_map.id2label[i]}={probs[i]:.{decimals}f}"
        for i in range(label_map.num_classes)
    ]
    return "{" + ", ".join(parts) + "}"


def _format_prob_dict_compact(
    probs: np.ndarray,
    label_map: LabelMap,
    decimals: int = PROMPT_PROB_DECIMALS,
) -> str:
    """Compact prob dict using abbreviated label names, no spaces.

    Example: ``{A=0.66,D=0.30,J=0.01,F=0.01,O=0.01,Sa=0.01,Su=0.00}``
    Falls back to full label name if no abbreviation defined.
    """
    parts = [
        f"{LABEL_ABBR.get(label_map.id2label[i], label_map.id2label[i])}={probs[i]:.{decimals}f}"
        for i in range(label_map.num_classes)
    ]
    return "{" + ",".join(parts) + "}"


def build_meta_system_prompt(label_map: LabelMap, weights: Dict[str, float]) -> str:
    """Build the system prompt for the compact_v1 schema.

    Includes label legend, model legend, and per-model weights so this
    information does not need to be repeated on every user turn.
    """
    full_names = label_map.class_names
    label_legend = ", ".join(
        f"{LABEL_ABBR.get(n, n)}={n}" for n in full_names
    )
    model_legend = ", ".join(
        f"{MODEL_ABBR.get(m, m)}={m}" for m in BASE_MODEL_NAMES if m in MODEL_ABBR
    )
    weight_str = ", ".join(
        f"{MODEL_ABBR.get(m, m)}={weights[m]:.3f}"
        for m in BASE_MODEL_NAMES
        if m in weights
    )
    class_list = ", ".join(full_names)
    return (
        "You are an emotion classifier for Vietnamese social-media text. "
        f"Pick exactly one label from: {class_list}. "
        "You receive: [TEXT] the input sentence [/TEXT], "
        "then [WEIGHTED_AVG] the weighted-average probability over all base models [/WEIGHTED_AVG], "
        "then [STACK] per-model probabilities [/STACK]. "
        f"Label abbreviations: {label_legend}. "
        f"Model abbreviations: {model_legend}. "
        f"Model weights (higher = more reliable on validation): {weight_str}. "
        "Output ONLY the final answer in the exact format: <label>LABEL</label>."
    )


def format_prompt(
    text: str,
    probs_per_model: Dict[str, np.ndarray],
    weights: Dict[str, float],
    label_map: LabelMap,
    label: Optional[str] = None,
    model_order: Iterable[str] = BASE_MODEL_NAMES,
) -> Dict[str, str]:
    """Render one stacked example as ``{prompt, completion}`` (compact_v1 schema).

    Prompt schema (train/val/test):

        [TEXT] ... [/TEXT]
        [WEIGHTED_AVG]{A=0.66,D=0.30,...}[/WEIGHTED_AVG]
        [STACK]
        p:{A=0.85,D=0.14,...}
        c:{...}
        v:{...}
        l:{...}
        s:{...}
        [/STACK]

    [WEIGHTED_AVG] is emitted before [STACK] so it is preserved even when a
    low ``max_length`` cap truncates the tail of the sequence.

    Stack lines use abbreviated model keys (p/c/v/l/s) with no per-line
    weight annotations; weights are stated once in the system prompt via
    ``build_meta_system_prompt``.

    Prob values use 2-decimal abbreviated label keys (A/D/J/F/O/Sa/Su).

    The completion (only present when ``label`` is given) is
    ``<label>EMOTION</label>`` with the full label name unchanged.
    """

    used_models = [m for m in model_order if m in probs_per_model and m in weights]
    avg = weighted_average(
        {m: probs_per_model[m][None, :] for m in used_models},
        {m: weights[m] for m in used_models},
    )[0]

    lines = [
        f"[TEXT] {text} [/TEXT]",
        f"[WEIGHTED_AVG]{_format_prob_dict_compact(avg, label_map)}[/WEIGHTED_AVG]",
        "[STACK]",
    ]
    for m in used_models:
        abbr = MODEL_ABBR.get(m, m)
        lines.append(f"{abbr}:{_format_prob_dict_compact(probs_per_model[m], label_map)}")
    lines.append("[/STACK]")

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
