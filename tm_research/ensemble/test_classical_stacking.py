"""Unit tests for classical stacking ablations (soft voting + LogReg meta).

Run with:  python -m pytest tm_research/ensemble/test_classical_stacking.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import f1_score

from tm_research.ensemble.utils_stacking import (
    BASE_MODEL_NAMES,
    fit_logreg_meta,
    stack_prob_features,
    weighted_average,
)


def _one_hot_rows(class_ids: np.ndarray, num_classes: int) -> np.ndarray:
    eye = np.eye(num_classes, dtype=np.float32)
    return eye[class_ids]


def test_stack_prob_features_shape_and_order():
    n, c = 4, 3
    probs = {
        "phobert": np.full((n, c), 0.1, dtype=np.float32),
        "cafebert": np.full((n, c), 0.2, dtype=np.float32),
        "vibert": np.full((n, c), 0.3, dtype=np.float32),
        "logreg": np.full((n, c), 0.4, dtype=np.float32),
        "svc": np.full((n, c), 0.5, dtype=np.float32),
    }
    X = stack_prob_features(probs)
    assert X.shape == (n, len(BASE_MODEL_NAMES) * c)
    # Concatenation follows BASE_MODEL_NAMES, so first C cols are phobert (0.1)
    # and the last C cols are svc (0.5).
    np.testing.assert_allclose(X[:, :c], 0.1)
    np.testing.assert_allclose(X[:, -c:], 0.5)
    np.testing.assert_allclose(X[:, c : 2 * c], 0.2)


def test_stack_prob_features_missing_model_raises():
    n, c = 2, 3
    probs = {m: np.ones((n, c), dtype=np.float32) for m in BASE_MODEL_NAMES if m != "vibert"}
    with pytest.raises(ValueError, match="vibert"):
        stack_prob_features(probs)


def test_stack_prob_features_row_count_mismatch_raises():
    n, c = 3, 2
    probs = {m: np.ones((n, c), dtype=np.float32) for m in BASE_MODEL_NAMES}
    probs["svc"] = np.ones((n - 1, c), dtype=np.float32)
    with pytest.raises(ValueError, match="row"):
        stack_prob_features(probs)


def test_weighted_soft_vote_argmax_matches_hand_mix():
    # Two models, two classes. Model A is certain on class 0; B on class 1.
    # Equal weights → mix [0.5, 0.5]; heavier weight on A → class 0 wins.
    probs = {
        "a": np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        "b": np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32),
    }
    mix = weighted_average(probs, {"a": 0.75, "b": 0.25})
    expected = 0.75 * probs["a"] + 0.25 * probs["b"]
    np.testing.assert_allclose(mix, expected, rtol=1e-5)
    assert np.array_equal(mix.argmax(1), np.array([0, 0]))


def test_fit_logreg_meta_learns_separable_problem_and_selects_c():
    rng = np.random.default_rng(0)
    n_train, n_val, n_classes = 80, 40, 3
    y_train = rng.integers(0, n_classes, size=n_train)
    y_val = rng.integers(0, n_classes, size=n_val)

    # Each "base model" is a slightly noisy one-hot of the true label.
    def noisy_one_hot(y, noise=0.05):
        p = _one_hot_rows(y, n_classes)
        p = p + noise * rng.random(p.shape).astype(np.float32)
        p = p / p.sum(axis=1, keepdims=True)
        return p

    names = BASE_MODEL_NAMES
    train_probs = {m: noisy_one_hot(y_train) for m in names}
    val_probs = {m: noisy_one_hot(y_val) for m in names}
    X_train = stack_prob_features(train_probs)
    X_val = stack_prob_features(val_probs)

    C_grid = (0.25, 1.0, 4.0)
    clf, meta = fit_logreg_meta(X_train, y_train, X_val, y_val, C_grid=C_grid, seed=123)
    assert meta["C"] in C_grid
    assert 0.0 <= meta["val_macro_f1"] <= 1.0

    pred_val = clf.predict(X_val)
    val_f1 = float(f1_score(y_val, pred_val, average="macro", zero_division=0))
    assert val_f1 == pytest.approx(meta["val_macro_f1"], abs=1e-9)
    assert val_f1 > 0.7
