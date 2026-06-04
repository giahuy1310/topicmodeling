"""Smoke tests for VSFC compact_v1 prompt formatting."""

from __future__ import annotations

import numpy as np
import pytest

from tm_research.VSFC_ensemble.utils_io import LabelMap
from tm_research.VSFC_ensemble.utils_stacking import (
    LABEL_ABBR,
    build_meta_system_prompt,
    compute_weights,
    format_prompt,
    parse_label_from_completion,
    weighted_average,
)


@pytest.fixture
def label_map() -> LabelMap:
    label2id = {"negative": 0, "neutral": 1, "positive": 2}
    id2label = {i: k for k, i in label2id.items()}
    return LabelMap(label2id=label2id, id2label=id2label)


@pytest.fixture
def weights() -> dict:
    return compute_weights({"phobert": 0.5, "logreg": 0.4, "svc": 0.3})


def test_label_abbr_three_classes():
    assert set(LABEL_ABBR.keys()) == {"negative", "neutral", "positive"}


def test_format_prompt_weighted_avg_before_stack(label_map, weights):
    probs = {m: np.array([0.7, 0.2, 0.1], dtype=np.float32) for m in weights}
    out = format_prompt("xin chào", probs, weights, label_map, label="positive")
    assert "[WEIGHTED_AVG]" in out["prompt"]
    assert out["prompt"].index("[WEIGHTED_AVG]") < out["prompt"].index("[STACK]")
    assert "neg=" in out["prompt"] and "neu=" in out["prompt"] and "pos=" in out["prompt"]
    assert out["completion"] == "<label>positive</label>"


def test_system_prompt_sentiment_wording(label_map, weights):
    sp = build_meta_system_prompt(label_map, weights)
    assert "sentiment" in sp.lower()
    assert "negative" in sp and "positive" in sp


def test_weighted_average_shape(label_map, weights):
    n = 4
    probs = {m: np.random.dirichlet([1, 1, 1], size=n).astype(np.float32) for m in weights}
    avg = weighted_average(probs, weights)
    assert avg.shape == (n, 3)


def test_parse_label_from_completion(label_map):
    assert parse_label_from_completion("<label>neutral</label>", label_map) == "neutral"
    assert parse_label_from_completion("guess: positive", label_map) == "positive"
