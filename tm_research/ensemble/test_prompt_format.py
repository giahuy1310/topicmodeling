"""Unit tests for the compact_v1 prompt schema in utils_stacking.

Run with:  python -m pytest tm_research/ensemble/test_prompt_format.py -v
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from tm_research.ensemble.utils_io import LabelMap
from tm_research.ensemble.utils_stacking import (
    ABBR_TO_LABEL,
    BASE_MODEL_NAMES,
    LABEL_ABBR,
    MODEL_ABBR,
    PROMPT_PROB_DECIMALS,
    PROMPT_SCHEMA_VERSION,
    build_meta_system_prompt,
    format_prompt,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

LABELS = ["Anger", "Disgust", "Enjoyment", "Fear", "Other", "Sadness", "Surprise"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}


@pytest.fixture()
def label_map() -> LabelMap:
    return LabelMap(label2id=LABEL2ID, id2label=ID2LABEL)


@pytest.fixture()
def weights() -> dict[str, float]:
    raw = {"phobert": 0.206, "cafebert": 0.210, "vibert": 0.202, "logreg": 0.191, "svc": 0.191}
    total = sum(raw.values())
    return {k: v / total for k, v in raw.items()}


@pytest.fixture()
def probs_per_model() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    out = {}
    for m in BASE_MODEL_NAMES:
        p = rng.dirichlet(np.ones(len(LABELS))).astype(np.float32)
        out[m] = p
    return out


@pytest.fixture()
def prompt_and_completion(label_map, weights, probs_per_model):
    result = format_prompt(
        text="test text",
        probs_per_model=probs_per_model,
        weights=weights,
        label_map=label_map,
        label="Anger",
    )
    return result


# ---------------------------------------------------------------------------
# Schema version
# ---------------------------------------------------------------------------

def test_schema_version():
    assert PROMPT_SCHEMA_VERSION == "compact_v1"


# ---------------------------------------------------------------------------
# format_prompt output order
# ---------------------------------------------------------------------------

def test_weighted_avg_before_stack(prompt_and_completion):
    prompt = prompt_and_completion["prompt"]
    wa_pos = prompt.index("[WEIGHTED_AVG]")
    stack_pos = prompt.index("[STACK]")
    assert wa_pos < stack_pos, "[WEIGHTED_AVG] must appear before [STACK]"


def test_all_structural_markers_present(prompt_and_completion):
    prompt = prompt_and_completion["prompt"]
    for marker in ("[TEXT]", "[/TEXT]", "[WEIGHTED_AVG]", "[/WEIGHTED_AVG]", "[STACK]", "[/STACK]"):
        assert marker in prompt, f"Missing marker: {marker}"


# ---------------------------------------------------------------------------
# No per-line weights in stack lines
# ---------------------------------------------------------------------------

def test_no_per_line_weight_annotation(prompt_and_completion):
    prompt = prompt_and_completion["prompt"]
    stack_section = prompt.split("[STACK]", 1)[-1].split("[/STACK]", 1)[0]
    assert "w=" not in stack_section, "Stack lines must not contain per-line w= weight annotations"


# ---------------------------------------------------------------------------
# Abbreviated model keys in stack lines
# ---------------------------------------------------------------------------

def test_abbreviated_model_keys_in_stack(prompt_and_completion):
    prompt = prompt_and_completion["prompt"]
    stack_section = prompt.split("[STACK]", 1)[-1].split("[/STACK]", 1)[0]
    for full_name in BASE_MODEL_NAMES:
        assert full_name not in stack_section, (
            f"Full model name '{full_name}' must not appear in stack lines; use abbreviation"
        )
    for abbr in MODEL_ABBR.values():
        assert f"\n{abbr}:" in stack_section or stack_section.strip().startswith(f"{abbr}:"), (
            f"Expected abbreviated model key '{abbr}:' in stack lines"
        )


# ---------------------------------------------------------------------------
# Abbreviated label keys in prob dicts
# ---------------------------------------------------------------------------

def test_abbreviated_label_keys_in_probs(prompt_and_completion):
    prompt = prompt_and_completion["prompt"]
    for full_label in LABELS:
        abbr = LABEL_ABBR.get(full_label, full_label)
        # Full names must not appear inside prob dicts; abbreviated forms must.
        # Prob dicts are inside {...} — check by pattern.
        pattern_full = re.compile(rf"\b{re.escape(full_label)}=\d")
        assert not pattern_full.search(prompt), (
            f"Full label '{full_label}=' must not appear in prob dicts (use '{abbr}=')"
        )
    for abbr in LABEL_ABBR.values():
        assert f"{abbr}=" in prompt, f"Abbreviated label '{abbr}=' not found in prompt"


def test_prob_decimals(prompt_and_completion):
    prompt = prompt_and_completion["prompt"]
    # Extract a prob value and verify it has PROMPT_PROB_DECIMALS decimal places.
    match = re.search(r"A=(\d+\.\d+)", prompt)
    assert match, "Could not find 'A=<float>' in prompt"
    decimal_part = match.group(1).split(".")[-1]
    assert len(decimal_part) == PROMPT_PROB_DECIMALS, (
        f"Expected {PROMPT_PROB_DECIMALS} decimal places, got {len(decimal_part)}"
    )


# ---------------------------------------------------------------------------
# Completion uses full label name
# ---------------------------------------------------------------------------

def test_completion_uses_full_label_name(prompt_and_completion):
    completion = prompt_and_completion["completion"]
    assert completion == "<label>Anger</label>", (
        f"Completion must use full label name; got: {completion!r}"
    )


def test_completion_empty_when_no_label(label_map, weights, probs_per_model):
    result = format_prompt(
        text="test",
        probs_per_model=probs_per_model,
        weights=weights,
        label_map=label_map,
        label=None,
    )
    assert result["completion"] == ""


# ---------------------------------------------------------------------------
# build_meta_system_prompt
# ---------------------------------------------------------------------------

def test_system_prompt_contains_all_weight_keys(label_map, weights):
    sp = build_meta_system_prompt(label_map, weights)
    for m in BASE_MODEL_NAMES:
        abbr = MODEL_ABBR[m]
        assert abbr in sp, f"System prompt missing weight for model abbreviation '{abbr}'"


def test_system_prompt_contains_all_label_keys(label_map, weights):
    sp = build_meta_system_prompt(label_map, weights)
    for full_label, abbr in LABEL_ABBR.items():
        assert abbr in sp, f"System prompt missing label abbreviation '{abbr}'"
        assert full_label in sp, f"System prompt missing full label name '{full_label}'"


def test_system_prompt_output_format_instruction(label_map, weights):
    sp = build_meta_system_prompt(label_map, weights)
    assert "<label>LABEL</label>" in sp, "System prompt must specify the output format"


# ---------------------------------------------------------------------------
# ABBR_TO_LABEL is inverse of LABEL_ABBR
# ---------------------------------------------------------------------------

def test_abbr_to_label_is_inverse():
    for full, abbr in LABEL_ABBR.items():
        assert ABBR_TO_LABEL[abbr] == full, (
            f"ABBR_TO_LABEL[{abbr!r}] = {ABBR_TO_LABEL.get(abbr)!r}, expected {full!r}"
        )
